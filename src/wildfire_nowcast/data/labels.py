"""Build the ``fire_state`` label channel (C1 channel 0) from GOFER perimeters.

Output goes to ``data/interim/{fire_id}/fire_state.zarr`` - **never** to the C1
path. Per ADR-003(b) nothing lands at ``data/fires/{fire_id}/tensor.zarr`` until
all 14 channels exist; the guard that enforces that lives in
:mod:`wildfire_nowcast.data.assemble`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from wildfire_nowcast.common.contract import CELL_SIZE_M, CRS_STRING
from wildfire_nowcast.common.grid import BBox, Grid
from wildfire_nowcast.common.paths import interim_dir
from wildfire_nowcast.common.states import FIRELINE_V2, StateRule, apply_state_rule
from wildfire_nowcast.data.gofer import DEFAULT_CFIRE_CONF, GoferArchive
from wildfire_nowcast.data.qa import fire_qa_report
from wildfire_nowcast.data.rasterize import (
    COVER_THRESHOLD,
    DEFAULT_OVERSAMPLE,
    line_mask,
    polygon_mask,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BUFFER_M",
    "LabelBuild",
    "fire_domain_grid",
    "build_fire_state",
    "write_interim_fire_state",
]

#: C1.2 - the per-fire domain is the final-perimeter bbox buffered outward by
#: this much before snapping to the continental lattice. 10 km ~ 3 h of a fast
#: wind-driven run at 3 km/h, i.e. somewhere for the model to spread *into*.
DEFAULT_BUFFER_M = 10_000.0


def fire_domain_grid(
    bounds: BBox,
    *,
    buffer_m: float = DEFAULT_BUFFER_M,
    res_m: float = CELL_SIZE_M,
) -> Grid:
    """The C1.2 domain for a fire: bbox + ``buffer_m``, snapped to the lattice.

    Geometry itself is :class:`wildfire_nowcast.common.grid.Grid` (C0); this
    function only states the *convention* (how much buffer, snap outward) that
    C1.2 fixes and that the manifest records.
    """
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    return Grid.from_bounds(
        (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m),
        cell_size_m=res_m,
        snap=True,
    )


@dataclass
class LabelBuild:
    """Everything produced for one fire's label channel."""

    fire_id: str
    grid: Grid
    times: pd.DatetimeIndex
    state: np.ndarray  # (T, H, W) uint8
    qa: dict[str, Any]
    provenance: dict[str, Any]

    def to_dataset(self) -> xr.Dataset:
        """C1-compatible interim store.

        Written as a named variable rather than a channel-indexed 4-D array
        because C1's "float32 except fire_state uint8" cannot be expressed in a
        single zarr array; see the layout PROPOSAL in
        docs/decisions.md. ``channel_index: 0`` records where this
        variable belongs once the full tensor is assembled.
        """
        da = xr.DataArray(
            self.state,
            dims=("time", "y", "x"),
            coords={
                "time": self.times.to_numpy(),
                "y": self.grid.y_coords,
                "x": self.grid.x_coords,
            },
            name="fire_state",
            attrs={
                "long_name": "fire state",
                "flag_values": [0, 1, 2],
                "flag_meanings": "unburned burning burned_out",
                "channel_index": 0,
                "units": "1",
            },
        )
        ds = xr.Dataset({"fire_state": da})
        ds["x"].attrs.update(units="m", standard_name="projection_x_coordinate")
        ds["y"].attrs.update(units="m", standard_name="projection_y_coordinate")
        ds.attrs.update(
            {
                "fire_id": self.fire_id,
                "crs": self.grid.crs,
                # common.contract reads cell_size_m; resolution_m kept as an alias.
                "cell_size_m": float(self.grid.cell_size_m),
                "resolution_m": self.grid.cell_size_m,
                "bbox_5070": json.dumps(list(self.grid.bounds)),
                "time_convention": "end_of_hour",
                "time_units": "hourly UTC, naive; value is the END of the hour it describes",
                "time_start_utc": self.times[0].strftime("%Y-%m-%dT%H:%M:%S"),
                "time_end_utc": self.times[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                "n_hours": int(len(self.times)),
                "contract": "INTERFACES v1 C1 channel 0 only — PARTIAL, not a C1 tensor",
                "provenance": json.dumps(self.provenance),
                "qa": json.dumps(self.qa),
            }
        )
        return ds


def build_fire_state(
    fire_id: str,
    *,
    archive: GoferArchive | None = None,
    rule: StateRule = FIRELINE_V2,
    buffer_m: float = DEFAULT_BUFFER_M,
    res_m: float = CELL_SIZE_M,
    oversample: int = DEFAULT_OVERSAMPLE,
    cover_threshold: float = COVER_THRESHOLD,
    cfire_conf: float = DEFAULT_CFIRE_CONF,
    with_east_west_noise: bool = True,
) -> LabelBuild:
    """Rasterise one GOFER fire onto the C1 grid and apply the state rule."""
    arch = archive or GoferArchive()
    perims = arch.perimeters(fire_id, to_crs=CRS_STRING)
    times = pd.DatetimeIndex(pd.to_datetime(perims["tUTC"]))

    grid = fire_domain_grid(tuple(perims.total_bounds), buffer_m=buffer_m, res_m=res_m)

    perim_masks = np.stack(
        [
            polygon_mask(g, grid, factor=oversample, threshold=cover_threshold)
            for g in perims.geometry
        ]
    )

    line_masks = None
    lines_used: dict[str, Any] = {}
    if rule == FIRELINE_V2:
        lines = arch.fire_lines(fire_id, conf=cfire_conf, to_crs=CRS_STRING)
        by_ts = dict(zip(lines["timestep"], lines.geometry, strict=True))
        line_masks = np.stack([line_mask(by_ts.get(ts), grid) for ts in perims["timestep"]])
        missing = [ts for ts in perims["timestep"] if ts not in by_ts]
        lines_used = {
            "cfire_conf": cfire_conf,
            "cfireline_timesteps_missing": len(missing),
        }

    state = apply_state_rule(perim_masks, rule=rule, fire_line_masks=line_masks)

    east = west = None
    if with_east_west_noise:
        try:
            east = GoferArchive(arch.root, "east").perimeters(fire_id)
            west = GoferArchive(arch.root, "west").perimeters(fire_id)
        except (FileNotFoundError, KeyError) as exc:
            # ADR-103 (4). This fallback used to leave no trace, and what it
            # removes is not cosmetic: without east and west,
            # `fire_qa_report` simply omits `label_noise_east_west`, so the
            # per-fire measurement behind R6 (GOFER-East systematically larger
            # than West, 0.63 km mean centroid offset) VANISHES FROM THE QA
            # REPORT with no key saying it was ever attempted. An absent key and
            # a measured zero are not the same fact, and the model's observation
            # noise is calibrated off exactly this number.
            logger.warning(
                "%s: the east/west GOFER variants under %s did not open (%s: %s), so "
                "label_noise_east_west will be ABSENT from this fire's QA report "
                "rather than measured",
                fire_id,
                arch.root,
                type(exc).__name__,
                exc,
            )
            east = west = None

    meta = arch.lookup(fire_id)
    provenance: dict[str, Any] = {
        **arch.provenance(),
        "gofer_pull_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "state_rule": rule,
        "state_rule_status": "RATIFIED — INTERFACES C1.1 (ADR-006 P1)",
        "state_rule_impl": (
            "wildfire_nowcast.common.states.apply_state_rule (C0: one implementation)"
        ),
        "grid_buffer_m": buffer_m,
        "polygon_rasterization": {
            "method": "area_fraction",
            "oversample_factor": oversample,
            "cover_threshold": cover_threshold,
        },
        "gofer_fname": str(meta.fname),
        "gofer_fyear": int(meta.fyear),
        "acres_official": float(meta.acres_official),
        "goes_ignition_utc": str(meta.GOESIg_UTC),
        **lines_used,
    }

    qa = fire_qa_report(
        fire_id=fire_id,
        perims=perims,
        state=state,
        grid=grid,
        east=east,
        west=west,
        extra={"state_rule": rule},
    )
    return LabelBuild(
        fire_id=fire_id, grid=grid, times=times, state=state, qa=qa, provenance=provenance
    )


def write_interim_fire_state(
    build: LabelBuild, root: Path | None = None, *, name: str = "fire_state"
) -> Path:
    """Write ``data/interim/{fire_id}/{name}.zarr`` (+ a sidecar QA json).

    ``name`` exists so competing state rules can sit side by side while R3 is
    unresolved. This function refuses to write anywhere near the C1 path.
    """
    base = Path(root) if root is not None else interim_dir()
    if "fires" in base.parts:
        raise ValueError(
            f"refusing to write a partial (label-only) product under {base}; "
            "ADR-003(b) reserves data/fires/ for complete 14-channel tensors"
        )
    out = base / build.fire_id / f"{name}.zarr"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = build.to_dataset()
    ds.to_zarr(out, mode="w", consolidated=True)
    (out.parent / f"{name}.qa.json").write_text(
        json.dumps({"provenance": build.provenance, "qa": build.qa}, indent=2) + "\n"
    )
    return out
