"""1 km -> 2 km coarsening of the built corpus (ADR-054 R1-prep). ADDITIVE.

Why this exists
---------------
The label product is **~2 km effective on a 1 km grid**. ADR-054 (3) names four
symptoms that one resolution choice could explain at once, and pre-registers
R1-P1/R1-P2 to test it. This module builds the corpus that makes the test
possible: the SAME 21 fires, the SAME split, one variable changed.

It writes to ``data/fires_2km/`` and **nothing else**. It never opens a store
under ``data/fires/`` for writing, never touches ``data/norm_stats.json``, and
:func:`two_km_dir` refuses to resolve to the 1 km corpus root.

THE COARSENING RULE, AND WHY IT IS NOT ONE RULE
----------------------------------------------
``sim/coarsen.py`` already owns the adjudicated burned-set rule (fractional-area
occupancy at 0.5, identical in form and threshold to ``data/rasterize``'s
``polygon_mask``, which defines TRUTH). It is imported, not re-derived - a
producer and a verifier computing geometry through different code is exactly how
a tensor passes its check and is still wrong (C0, docs/interfaces.md).

But a corpus is not one field. Three data types need three rules, and getting
this wrong SILENTLY is the failure mode:

``mean``       INTENSIVE CONTINUOUS fields (weather, elevation, slope, canopy,
               moisture, the aspect unit-vector components). The block mean is
               the definition of the field's value over a larger cell. Convex,
               so ``canopy_cover`` stays inside [0, 100] and C1.7 survives by
               construction rather than by luck.
``occupancy``  EXTENSIVE BINARY area fields (the burned set, the burn scar, the
               barrier mask). A mask is NOT a mean: the coarse cell is set iff
               at least half its area is set. ``>=`` not ``>``, so a 2-of-4 tie
               RETAINS the feature. This is truth's own rule and threshold.
``modal``      CATEGORICAL fields (``fuel_model_id``). A class id cannot be
               averaged - the mean of FBFM40 classes 98 and 142 is 120, a
               different fuel that happens to exist. Majority vote over the four
               sub-cells; ties broken by the fire's own 1 km class histogram
               (commonest class wins), then by ascending id. The tie RATE is
               measured per fire and recorded, so the tiebreak's weight is a
               number rather than an assumption.

``fire_state`` is none of these and is not aggregated directly. It is a
three-valued absorbing state whose meaning is fixed by C1.1, so it is REBUILT:
``ever`` and the active-fireline indicator are each coarsened by the occupancy
rule, and ``common.states.fireline_v2`` - the single implementation of C1.1 - is
re-applied to them. Every C1.1 guarantee then holds by construction rather than
by inspection. ``line_dilation=0`` because the 1 km field was already dilated
once; dilating again would be a second cell of growth applied at 2 km.

LATTICE AND PADDING
-------------------
C1.2 snaps a domain outward to a single continental lattice. At 2 km that is the
2 km lattice, which can sit up to one 1 km cell outside the 1 km domain on each
edge. The 1 km arrays are therefore PADDED outward, never cropped: cropping
would shrink the C1.2 buffer, and the buffer is the clause with a failure mode.

* ``fire_state`` pads with UNBURNED. Verified per fire, not assumed: the outer
  ring of the 1 km domain is unburned at every timestep (C1.2 puts >= 10 cells
  of buffer around the final footprint), so the pad continues an all-unburned
  boundary rather than inventing one.
* ``features`` pad by EDGE REPLICATION. The padded band is a 1 km strip at the
  outer edge of a 10 km buffer; the count of affected coarse cells is recorded
  per fire so the reader can size it.

NESTED-SNAP IDENTITY (used to derive 2 km domains without re-reading source)
---------------------------------------------------------------------------
For integer-nested lattices, ``floor(floor(u/1000)*1000/2000)*2000 ==
floor(u/2000)*2000`` and likewise for ``ceil``. So re-snapping an existing 1 km
domain to 2 km gives EXACTLY the domain that snapping the raw buffered
perimeter bbox to 2 km would give. That is why block assignment can be recomputed
at 2 km from the manifests alone, and why the answer is not an approximation.
:func:`resnap_bounds` is that identity; ``coarsen_2km_selftest`` asserts it
against a direct computation rather than trusting the algebra.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from wildfire_nowcast.common.contract import (
    BURNING,
    CHANNELS,
    STATIC_CHANNELS,
    UNBURNED,
)
from wildfire_nowcast.common.grid import BBox, Grid
from wildfire_nowcast.common.paths import data_dir, fires_dir
from wildfire_nowcast.common.states import fireline_v2
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.data.assemble import ChannelBundle, write_fire_tensor

# The adjudicated binary rule and the nesting decomposition it is built on.
# Imported rather than re-implemented (C0). ``_blocks`` is private and `sim`
# exposes no public float block-mean, so this module reaches for the private
# name deliberately: a second copy of the decomposition is the one thing C0
# exists to prevent, and importing across the fence is the smaller cost.
from wildfire_nowcast.sim.coarsen import (
    OCCUPANCY_THRESHOLD,
    _blocks,
    coarsen_occupancy,
)

__all__ = [
    "REFINE",
    "TARGET_CELL_SIZE_M",
    "AGGREGATION",
    "AGGREGATION_RATIONALE",
    "two_km_dir",
    "resnap_bounds",
    "target_grid",
    "Padding",
    "padding_for",
    "block_mean",
    "block_occupancy",
    "modal_class",
    "CoarsenedFire",
    "coarsen_fire",
    "write_two_km_fire",
    "corpus_content_fingerprint",
    "build_corpus",
]

#: 1 km -> 2 km. Exactly nested: each coarse cell is 2 x 2 = 4 sub-cells.
REFINE = 2
TARGET_CELL_SIZE_M = 1000.0 * REFINE

#: Per-channel aggregation rule. Every C1 channel appears exactly once; a channel
#: added to C1 without a rule here raises at import of :func:`coarsen_fire`
#: rather than being silently averaged.
AGGREGATION: dict[str, str] = {
    "fire_state": "state_rule",
    "wind_u10": "mean",
    "wind_v10": "mean",
    "temp_2m": "mean",
    "rh_2m": "mean",
    "elevation": "mean",
    "slope": "mean",
    "aspect_sin": "mean",
    "aspect_cos": "mean",
    "fuel_model_id": "modal",
    "canopy_cover": "mean",
    "fuel_moisture_proxy": "mean",
    "water_barrier_mask": "occupancy",
    "recent_burn_scar": "occupancy",
}

AGGREGATION_RATIONALE: dict[str, str] = {
    "state_rule": (
        "NOT aggregated. ever(t) and the active-fireline indicator are each "
        "coarsened by the occupancy rule, then C1.1 fireline_v2 is re-applied "
        "through common/states.py with line_dilation=0. Absorbing, monotone, no "
        "0->2 skip and one contiguous burning run hold by construction."
    ),
    "mean": (
        "intensive continuous field: the block mean IS the value over the larger "
        "cell. Convex, so a bounded channel stays inside its C1.7 range."
    ),
    "occupancy": (
        "extensive binary area field: set iff covered fraction >= 0.5. Truth's "
        "own rule and threshold (data/rasterize COVER_THRESHOLD, "
        "sim/coarsen OCCUPANCY_THRESHOLD). A mask is not a mean."
    ),
    "modal": (
        "categorical enumeration: majority of the 4 sub-cells; ties to the "
        "fire's commonest 1 km class, then ascending id. Averaging a class id "
        "invents a fuel model that exists and is wrong."
    ),
}

_MEAN = "mean"
_OCCUPANCY = "occupancy"
_MODAL = "modal"


def two_km_dir() -> Path:
    """``data/fires_2km/``. Refuses to resolve onto the 1 km corpus root."""
    out = data_dir() / "fires_2km"
    if out.resolve() == Path(fires_dir()).resolve():  # pragma: no cover - guard
        raise RuntimeError("the 2 km corpus root must never resolve to the 1 km corpus root")
    return out


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def resnap_bounds(bounds: BBox, cell_size_m: float = TARGET_CELL_SIZE_M) -> BBox:
    """Re-snap outer edges outward to a coarser, integer-nested lattice.

    Exact, not approximate: see the nested-snap identity in the module doc.
    """
    x0, y0, x1, y1 = (float(v) for v in bounds)
    return (
        float(np.floor(x0 / cell_size_m) * cell_size_m),
        float(np.floor(y0 / cell_size_m) * cell_size_m),
        float(np.ceil(x1 / cell_size_m) * cell_size_m),
        float(np.ceil(y1 / cell_size_m) * cell_size_m),
    )


def target_grid(fine: Grid, refine: int = REFINE) -> Grid:
    """The coarse grid covering ``fine``, snapped outward to the coarse lattice."""
    cell = fine.cell_size_m * refine
    return Grid.from_bounds(
        resnap_bounds(fine.bounds, cell), cell_size_m=cell, crs=fine.crs, snap=False
    )


@dataclass(frozen=True)
class Padding:
    """1 km cells of outward pad needed to align a fine grid to a coarse one."""

    top: int
    bottom: int
    left: int
    right: int

    @property
    def total_cells(self) -> int:
        return self.top + self.bottom + self.left + self.right

    def as_dict(self) -> dict[str, int]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
        }


def padding_for(fine: Grid, coarse: Grid) -> Padding:
    """Outward pad, in FINE cells, that makes ``fine`` exactly tile ``coarse``."""
    step = fine.cell_size_m
    pad = Padding(
        top=int(round((coarse.y_max - fine.y_max) / step)),
        bottom=int(round((fine.y_min - coarse.y_min) / step)),
        left=int(round((fine.x_min - coarse.x_min) / step)),
        right=int(round((coarse.x_max - fine.x_max) / step)),
    )
    if min(pad.top, pad.bottom, pad.left, pad.right) < 0:
        raise ValueError(f"coarse grid does not contain the fine grid: {pad}")
    ny = fine.ny + pad.top + pad.bottom
    nx = fine.nx + pad.left + pad.right
    refine = int(round(coarse.cell_size_m / step))
    if (ny, nx) != (coarse.ny * refine, coarse.nx * refine):
        raise ValueError(
            f"padded fine shape {(ny, nx)} does not tile the coarse grid "
            f"{(coarse.ny, coarse.nx)} at refine={refine}"
        )
    return pad


def _pad_edge(arr: np.ndarray, pad: Padding) -> np.ndarray:
    """Edge-replicate the last two axes."""
    widths = [(0, 0)] * (arr.ndim - 2) + [(pad.top, pad.bottom), (pad.left, pad.right)]
    return np.pad(arr, widths, mode="edge")


def _pad_constant(arr: np.ndarray, pad: Padding, value: int) -> np.ndarray:
    widths = [(0, 0)] * (arr.ndim - 2) + [(pad.top, pad.bottom), (pad.left, pad.right)]
    return np.pad(arr, widths, mode="constant", constant_values=value)


# --------------------------------------------------------------------------
# the three aggregations
# --------------------------------------------------------------------------


def block_mean(arr: np.ndarray, refine: int = REFINE) -> np.ndarray:
    """Arithmetic mean over each exactly-nested ``refine x refine`` block.

    Accumulated in float64 and returned float32. The block decomposition is
    ``sim.coarsen``'s, so the continuous rule and the binary rule cannot disagree
    about which sub-cells belong to which coarse cell.
    """
    src = np.asarray(arr, dtype=np.float64)
    if src.ndim == 2:
        return _blocks(src, refine).mean(axis=(1, 3)).astype(np.float32)
    out = np.stack([_blocks(src[i], refine).mean(axis=(1, 3)) for i in range(src.shape[0])])
    return out.astype(np.float32)


def _occupancy_2d(arr: np.ndarray, refine: int = REFINE) -> np.ndarray:
    return coarsen_occupancy(np.asarray(arr) > 0, refine, OCCUPANCY_THRESHOLD)


def block_occupancy(arr: np.ndarray, refine: int = REFINE) -> np.ndarray:
    """Truth's rule: set iff the covered fraction is ``>= 0.5``. Float32 0/1."""
    src = np.asarray(arr)
    if src.ndim == 2:
        return _occupancy_2d(src, refine).astype(np.float32)
    return np.stack([_occupancy_2d(src[i], refine) for i in range(src.shape[0])]).astype(np.float32)


def modal_class(
    arr: np.ndarray, refine: int = REFINE, *, priority: Sequence[float] | None = None
) -> tuple[np.ndarray, int]:
    """Majority class per block, plus the number of blocks whose mode was tied.

    ``priority`` is the class order used to break ties - the first listed class
    wins. Defaults to descending frequency over the whole input, then ascending
    class id, which is the fire's own histogram rather than an arbitrary
    preference for low ids (low FBFM40 ids are the NON-BURNABLE classes, so
    ascending-id alone would systematically grow non-burnable ground).
    """
    src = np.asarray(arr, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError(f"modal_class expects a 2-D field, got shape {src.shape}")
    values, counts = np.unique(src, return_counts=True)
    if priority is None:
        order = np.lexsort((values, -counts))
        classes = values[order]
    else:
        classes = np.asarray(priority, dtype=np.float64)
        missing = set(np.unique(src).tolist()) - set(classes.tolist())
        if missing:
            raise ValueError(
                f"priority order omits classes present in the field: {sorted(missing)}"
            )
    blocks = _blocks(src, refine)  # (h, R, w, R)
    h, w = blocks.shape[0], blocks.shape[2]
    flat = blocks.transpose(0, 2, 1, 3).reshape(h, w, refine * refine)
    tally = np.stack([(flat == c).sum(axis=2) for c in classes], axis=2)
    best = tally.argmax(axis=2)
    top = tally.max(axis=2)
    ties = int(((tally == top[:, :, None]).sum(axis=2) > 1).sum())
    return classes[best].astype(np.float32), ties


# --------------------------------------------------------------------------
# one fire
# --------------------------------------------------------------------------


@dataclass
class CoarsenedFire:
    """The 2 km channels for one fire, plus everything measured while building."""

    fire_id: str
    grid: Grid
    times: pd.DatetimeIndex
    channels: dict[str, np.ndarray]
    qa: dict[str, Any]


#: FBFM40 non-burnable classes (NB1-NB9, ids 91-99). Used only to REPORT how the
#: burnable fraction moves under the modal rule; nothing branches on it.
_NON_BURNABLE_FBFM40 = frozenset(range(91, 100))


def _burnable_fraction(fuel: np.ndarray) -> float:
    arr = np.asarray(fuel)
    nb = np.isin(np.rint(arr).astype(np.int64), sorted(_NON_BURNABLE_FBFM40))
    return float(np.count_nonzero(~nb)) / float(arr.size)


def _degenerate_report(name: str, fine: np.ndarray, coarse: np.ndarray) -> dict[str, Any]:
    """Constancy before and after. A degenerate channel is REPORTED, never hidden."""
    fu = np.unique(np.asarray(fine))
    cu = np.unique(np.asarray(coarse))
    return {
        "channel": name,
        "n_distinct_1km": int(fu.size),
        "n_distinct_2km": int(cu.size),
        "constant_1km": bool(fu.size == 1),
        "constant_2km": bool(cu.size == 1),
        "constant_value_2km": float(cu[0]) if cu.size == 1 else None,
        "became_constant": bool(fu.size > 1 and cu.size == 1),
    }


def coarsen_fire(ds: xr.Dataset, *, refine: int = REFINE) -> CoarsenedFire:
    """Coarsen one open C1 tensor. Pure: reads ``ds``, writes nothing."""
    missing = [c for c in CHANNELS if c not in AGGREGATION]
    if missing:  # pragma: no cover - structural guard
        raise KeyError(f"no aggregation rule declared for C1 channels {missing}")

    fine = Grid.from_dataset(ds)
    coarse = target_grid(fine, refine)
    pad = padding_for(fine, coarse)
    fire_id = str(ds.attrs.get("fire_id", ""))
    times = pd.DatetimeIndex(pd.to_datetime(np.asarray(ds["time"].values)))

    state1 = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    edge_ring_burned = int(
        np.count_nonzero(state1[:, 0, :])
        + np.count_nonzero(state1[:, -1, :])
        + np.count_nonzero(state1[:, :, 0])
        + np.count_nonzero(state1[:, :, -1])
    )
    state1p = _pad_constant(state1, pad, UNBURNED)
    n_t = int(state1p.shape[0])
    ever2 = np.stack([_occupancy_2d(state1p[i] >= BURNING, refine) for i in range(n_t)])
    act2 = np.stack([_occupancy_2d(state1p[i] == BURNING, refine) for i in range(n_t)])
    state2 = fireline_v2(ever2, act2, line_dilation=0, validate=True)

    channels: dict[str, np.ndarray] = {"fire_state": state2}
    per_channel: list[dict[str, Any]] = []
    fuel_ties = 0

    for name in CHANNELS:
        if name == "fire_state":
            continue
        rule = AGGREGATION[name]
        raw = np.asarray(ds["features"].isel(channel=CHANNELS.index(name) - 1).values)
        src = raw[0] if name in STATIC_CHANNELS else raw
        padded = _pad_edge(src, pad)
        if rule == _MEAN:
            out = block_mean(padded, refine)
        elif rule == _OCCUPANCY:
            out = block_occupancy(padded, refine)
        elif rule == _MODAL:
            out, fuel_ties = modal_class(padded, refine)
        else:  # pragma: no cover - structural guard
            raise KeyError(f"unknown aggregation rule {rule!r} for channel {name!r}")
        channels[name] = out
        row = _degenerate_report(name, src, out)
        row["rule"] = rule
        if rule == _OCCUPANCY:
            frac_fine = float(np.count_nonzero(src > 0)) / float(np.asarray(src).size)
            frac_coarse = float(np.count_nonzero(out > 0)) / float(out.size)
            any_touch = _blocks(np.asarray(padded) > 0, refine).any(axis=(1, 3))
            row["cell_fraction_1km"] = round(frac_fine, 6)
            row["cell_fraction_2km"] = round(frac_coarse, 6)
            row["cell_fraction_2km_if_any_touch"] = round(
                float(np.count_nonzero(any_touch)) / float(any_touch.size), 6
            )
            row["area_km2_1km"] = float(np.count_nonzero(src > 0))
            row["area_km2_2km"] = float(np.count_nonzero(out > 0)) * (refine**2)
        if rule == _MODAL:
            # A tie rate that is not small makes the tiebreak load-bearing, so
            # its WEIGHT is measured rather than argued: the same field under the
            # alternative (ascending class id) tiebreak, and how often the two
            # disagree. Reported either way.
            alt, _ = modal_class(padded, refine, priority=np.unique(padded).tolist())
            row["tied_blocks"] = int(fuel_ties)
            row["tied_block_fraction"] = round(float(fuel_ties) / float(out.size), 6)
            row["tiebreak_alt_disagreement_fraction"] = round(
                float(np.count_nonzero(alt != out)) / float(out.size), 6
            )
            row["burnable_fraction_1km"] = round(_burnable_fraction(src), 6)
            row["burnable_fraction_2km"] = round(_burnable_fraction(out), 6)
        per_channel.append(row)

    qa: dict[str, Any] = {
        "fire_id": fire_id,
        "refine": int(refine),
        "cell_size_m_1km": float(fine.cell_size_m),
        "cell_size_m_2km": float(coarse.cell_size_m),
        "shape_1km": [int(fine.ny), int(fine.nx)],
        "shape_2km": [int(coarse.ny), int(coarse.nx)],
        "padding_1km_cells": pad.as_dict(),
        "coarse_cells_touching_pad": int(_pad_touched_cells(fine, coarse, pad, refine)),
        "outer_ring_burned_cell_hours_1km": edge_ring_burned,
        "aggregation": {c: AGGREGATION[c] for c in CHANNELS},
        "per_channel": per_channel,
        "degenerate_channels": [r["channel"] for r in per_channel if r["constant_2km"]],
        "channels_that_became_constant": [
            r["channel"] for r in per_channel if r["became_constant"]
        ],
    }
    qa.update(_state_qa(state1, state2, refine))
    return CoarsenedFire(fire_id, coarse, times, channels, qa)


def _pad_touched_cells(fine: Grid, coarse: Grid, pad: Padding, refine: int) -> int:
    """Number of coarse cells containing at least one padded fine sub-cell."""
    flag = np.zeros((fine.ny + pad.top + pad.bottom, fine.nx + pad.left + pad.right), dtype=bool)
    flag[:, :] = True
    flag[pad.top : pad.top + fine.ny, pad.left : pad.left + fine.nx] = False
    return int(np.count_nonzero(_blocks(flag, refine).any(axis=(1, 3))))


def _state_qa(state1: np.ndarray, state2: np.ndarray, refine: int) -> dict[str, Any]:
    """Label QA at both resolutions, including the AREA-CONSERVATION residual."""
    cell1, cell2 = 1.0, float(refine) ** 2
    ever1 = (state1 >= BURNING).sum(axis=(1, 2)).astype(np.float64) * cell1
    ever2 = (state2 >= BURNING).sum(axis=(1, 2)).astype(np.float64) * cell2
    resid = ever2 - ever1
    final1, final2 = float(ever1[-1]), float(ever2[-1])
    burning_frames_1 = int(np.count_nonzero((state1 == BURNING).any(axis=(1, 2))))
    burning_frames_2 = int(np.count_nonzero((state2 == BURNING).any(axis=(1, 2))))
    n = int(state1.shape[0])
    return {
        "final_burned_area_km2_1km": final1,
        "final_burned_area_km2_2km": final2,
        "final_area_residual_km2": final2 - final1,
        "final_area_relative_residual": (final2 - final1) / final1 if final1 > 0 else None,
        "hourly_area_residual_km2_max_abs": float(np.max(np.abs(resid))),
        "hourly_area_relative_residual_max_abs": float(
            np.max(np.abs(resid[ever1 > 0] / ever1[ever1 > 0])) if np.any(ever1 > 0) else 0.0
        ),
        "monotone_1km": bool(np.all(np.diff(ever1) >= 0)),
        "monotone_2km": bool(np.all(np.diff(ever2) >= 0)),
        "frames_with_no_burning_1km": n - burning_frames_1,
        "frames_with_no_burning_2km": n - burning_frames_2,
        "frames_with_no_burning_fraction_1km": round((n - burning_frames_1) / n, 4) if n else None,
        "frames_with_no_burning_fraction_2km": round((n - burning_frames_2) / n, 4) if n else None,
        "n_hours": n,
    }


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

_PROVENANCE_STRUCTURED = ("qa", "build_timings_s", "polygon_rasterization", "ignition_components")


def write_two_km_fire(
    fire_id: str,
    *,
    source_root: Path | None = None,
    out_root: Path | None = None,
    norm_stats: Path | None = None,
    refine: int = REFINE,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build ``{out_root}/{fire_id}/`` from ``{source_root}/{fire_id}/``.

    The C2 manifest is the 1 km manifest with the geometry keys replaced and the
    coarsening provenance added, so every ratified C2 key survives verbatim and
    ``cv_fold`` / ``spatial_block_id`` / ``n_hours`` are carried, not recomputed.
    """
    src_root = Path(source_root) if source_root else Path(fires_dir())
    dst_root = Path(out_root) if out_root else two_km_dir()
    man = json.loads((src_root / fire_id / "manifest.json").read_text())
    ds = open_tensor(src_root / fire_id / "tensor.zarr")
    try:
        built = coarsen_fire(ds, refine=refine)
    finally:
        ds.close()

    bundle = ChannelBundle(
        fire_id=fire_id,
        grid=built.grid,
        times=built.times,
        cv_fold=int(man["cv_fold"]),
        spatial_block_id=int(man["spatial_block_id"]),
        fuel_vintage_lag_years=int(man["fuel_vintage_lag_years"]),
        n_ignition_components=int(man["n_ignition_components"]),
        ignition_time_utc=str(man["ignition_time_utc"]),
        gofer_version=str(man["gofer_version"]),
    )
    for name, values in built.channels.items():
        bundle.add(name, values)

    prov: dict[str, Any] = dict(man.get("provenance") or {})
    prov.update(
        {
            "derived_from": f"{src_root.name}/{fire_id}/tensor.zarr",
            "derived_from_cell_size_m": "1000.0",
            "coarsening_adr": "ADR-054",
            "coarsening_refine": str(refine),
            "coarsening_rule_binary": (
                "fractional-area occupancy >= "
                f"{OCCUPANCY_THRESHOLD} (sim/coarsen.coarsen_occupancy)"
            ),
            "coarsening_rule_continuous": "arithmetic block mean over the 2x2 sub-cell block",
            "coarsening_rule_categorical": (
                "majority of 4 sub-cells; ties to the fire's commonest 1 km class, "
                "then ascending class id"
            ),
            "coarsening_rule_fire_state": AGGREGATION_RATIONALE["state_rule"],
            "coarsening_padding": "outward to the 2 km lattice; features edge-replicated, "
            "fire_state constant UNBURNED",
            "contract_status": (
                "NOT C1-CONFORMANT BY DESIGN: C1 fixes 1000 m cells, this store is "
                "2000 m. Every other C1/C2/C3 invariant is asserted. See the R1-prep "
                "entry in the data status log."
            ),
        }
    )
    bundle.provenance = prov
    bundle.qa = built.qa

    stats_path = str(norm_stats if norm_stats else dst_root / "norm_stats.json")
    tensor_path, manifest_path = write_fire_tensor(
        bundle,
        norm_stats_path=stats_path,
        tensor_path=dst_root / fire_id / "tensor.zarr",
        manifest_path=dst_root / fire_id / "manifest.json",
    )
    patched = json.loads(manifest_path.read_text())
    patched["label_source"] = man.get("label_source")
    patched["cell_size_m"] = float(built.grid.cell_size_m)
    manifest_path.write_text(json.dumps(patched, indent=2) + "\n")
    return tensor_path, manifest_path, built.qa


def corpus_content_fingerprint(root: Path, fire_ids: Sequence[str]) -> str:
    """A CONTENT fingerprint of a corpus: sha256 over per-fire array digests.

    Deliberately separate from C8's ``split_fingerprint``, which hashes
    ``(fire_id, cv_fold, spatial_block_id, n_hours)`` and ``train_folds`` and is
    therefore INVARIANT under a resolution change. This one is not: it hashes the
    grid and the bytes.
    """
    import hashlib

    h = hashlib.sha256()
    for fire_id in sorted(fire_ids):
        ds = open_tensor(Path(root) / fire_id / "tensor.zarr")
        try:
            grid = Grid.from_dataset(ds)
            h.update(fire_id.encode())
            h.update(f"{grid.cell_size_m}:{grid.ny}:{grid.nx}:{grid.x_min}:{grid.y_max}".encode())
            h.update(np.asarray(ds["fire_state"].values, dtype=np.uint8).tobytes())
            h.update(np.asarray(ds["features"].values, dtype=np.float32).tobytes())
        finally:
            ds.close()
    return h.hexdigest()[:16]


def build_corpus(
    fire_ids: Sequence[str] | None = None,
    *,
    source_root: Path | None = None,
    out_root: Path | None = None,
    refine: int = REFINE,
) -> dict[str, Any]:
    """Build every fire and return the per-fire QA reports."""
    src_root = Path(source_root) if source_root else Path(fires_dir())
    dst_root = Path(out_root) if out_root else two_km_dir()
    ids = (
        list(fire_ids)
        if fire_ids
        else sorted(p.name for p in src_root.iterdir() if (p / "tensor.zarr").exists())
    )
    reports: dict[str, Any] = {}
    for fire_id in ids:
        _, _, qa = write_two_km_fire(
            fire_id, source_root=src_root, out_root=dst_root, refine=refine
        )
        reports[fire_id] = qa
    return {"out_root": str(dst_root), "refine": int(refine), "fires": reports}


def _summary_rows(reports: Mapping[str, Any]) -> list[str]:
    rows = []
    for fire_id, qa in sorted(reports.items()):
        rows.append(
            f"[2km] {fire_id:<32} {qa['shape_1km'][0]}x{qa['shape_1km'][1]} -> "
            f"{qa['shape_2km'][0]}x{qa['shape_2km'][1]}  "
            f"area {qa['final_burned_area_km2_1km']:.0f} -> "
            f"{qa['final_burned_area_km2_2km']:.0f} km2 "
            f"({qa['final_area_relative_residual']:+.4f})"
        )
    return rows


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.data.coarsen_2km", allow_abbrev=False
    )
    ap.add_argument("--fire", action="append", default=None)
    ap.add_argument("--refine", type=int, default=REFINE)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    out = build_corpus(
        args.fire,
        source_root=Path(args.source_root) if args.source_root else None,
        out_root=Path(args.out_root) if args.out_root else None,
        refine=args.refine,
    )
    for line in _summary_rows(out["fires"]):
        print(line)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"[2km] report -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
