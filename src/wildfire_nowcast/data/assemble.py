"""C1/C2 assembly - and the guard that keeps a partial tensor off the C1 path.

ADR-003(b): nothing lands at ``data/fires/{fire_id}/tensor.zarr`` until all 14
channels exist. That is enforced here, once, by :func:`write_fire_tensor`, which
raises rather than writing a short tensor. Everything upstream (labels, GEE
exports) writes into ``data/interim/`` and this module is the only door out.

**The store itself is built by :mod:`wildfire_nowcast.common.zarr_io`** (C0,
ADR-007): channel order, dtypes, coordinate conventions and the C1.3
``time_convention`` attr are the contract's business, and the producer must not
compute them through different code than the verifier. This module stages the
channels, checks completeness, and delegates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from wildfire_nowcast.common import paths as _paths
from wildfire_nowcast.common.contract import CHANNELS, STATIC_CHANNELS
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import fire_manifest_path, fire_tensor_path
from wildfire_nowcast.common.zarr_io import build_manifest as _build_manifest
from wildfire_nowcast.common.zarr_io import build_tensor_dataset
from wildfire_nowcast.common.zarr_io import write_fire as _write_fire

__all__ = [
    "C1_CHANNELS",
    "N_CHANNELS",
    "STATIC_CHANNELS",
    "ChannelBundle",
    "MissingChannelsError",
    "write_fire_tensor",
    "build_manifest",
]

#: C1 channel order, index = position. Owned by ``common.contract`` (C0); this
#: alias exists only so data-side call sites read in C1 terms.
C1_CHANNELS: tuple[str, ...] = CHANNELS
N_CHANNELS = len(C1_CHANNELS)

#: Provenance entries whose value is structured (a dict), not a scalar. C2's
#: ``provenance`` is nominally {source: pull-date}; these two are deliberately
#: nested because flattening the QA report to a string would destroy it.
_STRUCTURED_PROVENANCE = (
    "qa",
    "build_timings_s",
    "polygon_rasterization",
    "ignition_components",
)


class MissingChannelsError(RuntimeError):
    """Raised instead of writing an incomplete tensor to the C1 path."""


@dataclass
class ChannelBundle:
    """Channels collected so far for one fire, plus everything C2 needs."""

    fire_id: str
    grid: Grid
    times: pd.DatetimeIndex
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)
    cv_fold: int | None = None
    #: [v2] C2 key; connected-component id over overlapping fire domains (C3.1).
    spatial_block_id: int | None = None
    #: [v2.7] C2 keys (ADR-014). Both are ``None`` until set, never defaulted -
    #: see :func:`build_manifest`.
    fuel_vintage_lag_years: int | None = None
    n_ignition_components: int | None = None
    ignition_time_utc: str | None = None
    gofer_version: str | None = None

    def add(self, name: str, values: np.ndarray) -> None:
        if name not in C1_CHANNELS:
            raise KeyError(f"{name!r} is not a C1 channel; C1 order is fixed")
        arr = np.asarray(values)
        t, h, w = len(self.times), self.grid.ny, self.grid.nx
        expected_static, expected_dynamic = (h, w), (t, h, w)
        if name in STATIC_CHANNELS:
            if arr.shape not in (expected_static, expected_dynamic):
                raise ValueError(f"{name}: expected {expected_static}, got {arr.shape}")
        elif arr.shape != expected_dynamic:
            raise ValueError(f"{name}: expected {expected_dynamic}, got {arr.shape}")
        self.channels[name] = arr

    @property
    def missing(self) -> list[str]:
        return [c for c in C1_CHANNELS if c not in self.channels]

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dataset(self) -> xr.Dataset:
        """The C1 v2 store, built by the canonical writer (C0)."""
        if not self.complete:
            raise MissingChannelsError(f"missing {self.missing}")
        return build_tensor_dataset(
            self.channels,
            self.grid,
            self.times.to_numpy(),
            attrs={
                "fire_id": self.fire_id,
                "bbox_5070": list(self.grid.bounds),
                "n_hours": int(len(self.times)),
            },
        )


def build_manifest(bundle: ChannelBundle, norm_stats_path: str) -> dict[str, Any]:
    """The C2 manifest dict. ``provenance.qa`` carries the per-fire QA report."""
    if bundle.cv_fold is None:
        raise ValueError("cv_fold must be assigned before writing a manifest (C2)")
    if bundle.spatial_block_id is None:
        raise ValueError(
            "spatial_block_id must be assigned before writing a manifest "
            "(C2 v2 / C3.1: folds are spatially blocked, effective n = 11)"
        )
    # [v2.7] Both are REQUIRED with no default, matching the canonical builder.
    # A default of 1 component is silently wrong on exactly the fires the clause
    # exists for, and a wrong number a reader trusts is worse than a raise.
    if bundle.fuel_vintage_lag_years is None:
        raise ValueError(
            "fuel_vintage_lag_years must be assigned before writing a manifest "
            "(C2 v2.7 / ADR-014 §3: fuels are 3-6 years stale and the lag varies "
            "by fire, so no consumer may be left to infer it)"
        )
    if bundle.n_ignition_components is None:
        raise ValueError(
            "n_ignition_components must be assigned before writing a manifest "
            "(C2 v2.7 / ADR-014 §7: GOFER files separate lightning ignitions "
            "under one fire id; derive it with data.ignitions."
            "count_ignition_components, do not default it to 1)"
        )
    flat: Mapping[str, Any] = {
        k: v for k, v in bundle.provenance.items() if k not in _STRUCTURED_PROVENANCE
    }
    man = _build_manifest(
        fire_id=bundle.fire_id,
        gofer_version=bundle.gofer_version or "",
        bbox_5070=list(bundle.grid.bounds),
        ignition_time_utc=bundle.ignition_time_utc,
        n_hours=int(len(bundle.times)),
        cv_fold=int(bundle.cv_fold),
        spatial_block_id=int(bundle.spatial_block_id),
        fuel_vintage_lag_years=int(bundle.fuel_vintage_lag_years),
        n_ignition_components=int(bundle.n_ignition_components),
        provenance=flat,
        norm_stats_path=norm_stats_path,
    )
    # Re-attach the structured entries the canonical builder stringifies. They
    # are JSON objects on purpose: a QA report flattened to str(dict) is not a
    # QA report, and the per-fire QA report is the point of C2 provenance.
    for key in _STRUCTURED_PROVENANCE:
        if key in bundle.provenance:
            man["provenance"][key] = bundle.provenance[key]
    if bundle.qa:
        man["provenance"]["qa"] = bundle.qa
    return man


def write_fire_tensor(
    bundle: ChannelBundle,
    *,
    norm_stats_path: str | None = None,
    tensor_path: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write the C1 tensor and C2 manifest - **only** if all 14 channels exist."""
    if not bundle.complete:
        raise MissingChannelsError(
            f"refusing to write {bundle.fire_id} to the C1 path: missing channels "
            f"{bundle.missing}. ADR-003(b) — a partial product goes to data/interim/, "
            "never to data/fires/{fire_id}/tensor.zarr."
        )
    stats_path = norm_stats_path or str(_paths.norm_stats_path())
    tpath = Path(tensor_path) if tensor_path else fire_tensor_path(bundle.fire_id)
    mpath = Path(manifest_path) if manifest_path else fire_manifest_path(bundle.fire_id)
    manifest = build_manifest(bundle, stats_path)
    written_tensor, written_manifest = _write_fire(
        bundle.to_dataset(), manifest, tpath.parent, tensor_name=tpath.name
    )
    if written_manifest != mpath:  # pragma: no cover - only when paths are overridden
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(written_manifest.read_text())
        written_manifest.unlink()
        written_manifest = mpath
    return written_tensor, written_manifest
