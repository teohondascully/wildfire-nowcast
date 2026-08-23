"""Physical-plausibility audit - the data-side complement to C1.7.

**Why this exists (R11, ADR-010).** The contract checks STRUCTURE, not
plausibility. C1.7 closed exactly one instance of that class - ``canopy_cover``
and ``fuel_model_id`` - and it did so deliberately narrowly: ADR-010 admits only
ranges outside which *no* legitimate value exists, so a hard fail carries no
false-positive mode. That is the right call for a contract. It leaves eleven
channels whose sentinels would still sail through: a ``-9999`` in ``elevation``
is finite, it is static, it is not a mask, and it is not channel 9 or 10.

So this module is the other half, and it is deliberately NOT a contract:

* It never blocks a build and never exits non-zero. Severity is
  ``ok`` / ``suspect``, and a ``suspect`` is a prompt to look, not a verdict.
* Its ranges are *plausibility* ranges, not definitional ones. They are
  documented with their justification and they live here, in ``data/``, rather
  than in ``common/`` - precisely because an unratified clause in the contract
  is the same mistake as a tolerated sentinel, pointed the other way.
* Where the contract DOES adjudicate a range it is imported, never restated
  (C0). ``canopy_cover`` and ``fuel_model_id`` come from
  :mod:`wildfire_nowcast.common.contract` and are reported here only so one
  table covers all fourteen channels.

The sentinel scan is the part that generalises. It looks for *exact* equality
against the values that GIS and remote-sensing products actually use for NoData,
in every channel, whether or not that channel has a plausible range. CZU's
``-9999`` was inside no declared range and violated no structural clause; it was
caught by a human reading a mean. This makes that reading mechanical.

**Non-finite is always a failure, never a pass** (ADR-012): every verdict ladder
here tests finiteness first and reports ``suspect`` on NaN/inf rather than
letting a comparison against NaN evaluate False and fall through to ``ok``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from wildfire_nowcast.common.contract import (
    BINARY_CHANNELS,
    CHANNELS,
    FBFM40_CLASSES,
    FEATURES,
    FIRE_STATE,
    FIRE_STATE_VALUES,
    PHYSICAL_RANGES,
    STATIC_CHANNELS,
)

__all__ = [
    "SENTINEL_VALUES",
    "PLAUSIBLE_RANGES",
    "CATEGORICAL_DOMAINS",
    "SCAR_SELF_OVERLAP_SUSPECT",
    "burn_scar_leak_report",
    "channel_audit",
    "audit_channels",
    "audit_dataset",
    "audit_norm_stats",
    "audit_built_fires",
]

#: NoData / fill values that real GIS and remote-sensing products emit. Tested by
#: EXACT equality, so a legitimate ``-9999.0`` would have to be a genuine
#: measurement of exactly that value - impossible in every channel we carry.
#:
#: Provenance of each entry, so this list is not folklore:
#: ``-9999`` USGS LFPS / LANDFIRE (the CZU case, ADR-010) and most ESRI rasters;
#: ``-32768`` / ``32767`` int16 rasters (SRTM, many NED derivatives);
#: ``-32767`` common int16 variant; ``255`` / ``65535`` unsigned byte / uint16
#: masks (MODIS, Landsat QA); ``-999`` / ``9999`` NOAA and NWS text products;
#: ``-3.4028235e38`` float32 GeoTIFF NoData (GDAL's default);
#: ``1e20`` GRIB/NetCDF missing_value convention (RTMA's own upstream format).
SENTINEL_VALUES: tuple[float, ...] = (
    -9999.0,
    -32768.0,
    -32767.0,
    32767.0,
    255.0,
    65535.0,
    -999.0,
    9999.0,
    -3.4028234663852886e38,
    1e20,
)

#: channel -> (low, high), INCLUSIVE. PLAUSIBILITY, not definition - see module
#: docstring. Each bound is justified, because an unjustified bound is a future
#: false positive that someone will silence rather than investigate.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    # RTMA 10 m analysis winds. The strongest CONUS surface analysis values are
    # well under 60 m/s (the highest reliably measured non-tornadic surface gust
    # is ~113 m/s at Mt Washington); a 2.5 km ANALYSIS cannot legitimately hold
    # a value near this bound, so it is loose on purpose.
    "wind_u10": (-60.0, 60.0),
    "wind_v10": (-60.0, 60.0),
    # 200-340 K = -73 to +67 C. CONUS records are -63 C (Rogers Pass MT) and
    # +57 C (Furnace Creek CA); the bound is wider than the record on both sides.
    "temp_2m": (200.0, 340.0),
    "rh_2m": (0.0, 100.0),  # definitional, but not contract-adjudicated
    # CONUS elevation: Badwater Basin -86 m to Mt Whitney 4421 m. 3DEP is a
    # bare-earth DEM, so no structure heights.
    "elevation": (-150.0, 4600.0),
    "slope": (0.0, 90.0),  # definitional
    "aspect_sin": (-1.0, 1.0),  # definitional
    "aspect_cos": (-1.0, 1.0),  # definitional
    # `common.derive.dead_fuel_moisture_simard` clips to [1, 60] %. A value
    # outside that means the clip was bypassed, i.e. the channel is not what its
    # documented formula says it is.
    "fuel_moisture_proxy": (1.0, 60.0),
}

#: channel -> the complete set of legal values. Enumerations, not ranges.
CATEGORICAL_DOMAINS: dict[str, frozenset[int]] = {
    FIRE_STATE: frozenset(FIRE_STATE_VALUES),
    "fuel_model_id": FBFM40_CLASSES,
    **{c: frozenset({0, 1}) for c in BINARY_CHANNELS},
}

#: Channels whose range the CONTRACT adjudicates (C1.7). Reported here for a
#: complete table; the authority is `common.contract`, never this module.
_CONTRACT_ADJUDICATED = frozenset(PHYSICAL_RANGES) | frozenset(CATEGORICAL_DOMAINS)


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    """Min/max/mean over finite cells, plus an explicit non-finite count.

    Returns ``None`` statistics when nothing is finite rather than emitting NaN,
    because a NaN that flows into a comparison silently evaluates False.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    n_nonfinite = int(arr.size - finite.sum())
    if not finite.any():
        return {
            "n": int(arr.size),
            "n_nonfinite": n_nonfinite,
            "min": None,
            "max": None,
            "mean": None,
        }
    good = arr[finite]
    return {
        "n": int(arr.size),
        "n_nonfinite": n_nonfinite,
        "min": round(float(good.min()), 6),
        "max": round(float(good.max()), 6),
        "mean": round(float(good.mean()), 6),
    }


def _sentinel_hits(values: np.ndarray) -> dict[str, int]:
    """Exact-equality count for every known NoData sentinel present."""
    arr = np.asarray(values, dtype=np.float64)
    hits: dict[str, int] = {}
    for s in SENTINEL_VALUES:
        n = int(np.count_nonzero(arr == s))
        if n:
            hits[repr(s)] = n
    return hits


def channel_audit(name: str, values: np.ndarray) -> dict[str, Any]:
    """Audit one channel. Pure: array in, report out. Never raises on data."""
    stats = _finite_summary(values)
    report: dict[str, Any] = {
        "channel": name,
        "index": CHANNELS.index(name) if name in CHANNELS else None,
        "static": name in STATIC_CHANNELS,
        "contract_adjudicated_range": name in _CONTRACT_ADJUDICATED,
        "stats": stats,
        "findings": [],
    }
    findings: list[str] = report["findings"]

    if stats["n_nonfinite"]:
        findings.append(
            f"{stats['n_nonfinite']} of {stats['n']} cells are NaN/inf — treated as a "
            "finding, never as pass-by-default (ADR-012)"
        )

    hits = _sentinel_hits(values)
    if hits:
        report["sentinel_hits"] = hits
        for value, n in hits.items():
            findings.append(
                f"{n} cells equal the known NoData sentinel {value} "
                f"({100.0 * n / max(stats['n'], 1):.2f}% of the channel)"
            )

    lo_hi = PHYSICAL_RANGES.get(name) or PLAUSIBLE_RANGES.get(name)
    if lo_hi is not None:
        lo, hi = lo_hi
        report["range"] = [lo, hi]
        report["range_authority"] = (
            "contract C1.7 (definitional)"
            if name in PHYSICAL_RANGES
            else "data-side plausibility (advisory)"
        )
        if stats["min"] is None:
            findings.append("no finite cell — range is unverifiable, which is a finding")
        else:
            n_out = int(np.count_nonzero(np.isfinite(values) & ((values < lo) | (values > hi))))
            if n_out:
                findings.append(
                    f"{n_out} cells outside [{lo}, {hi}] "
                    f"(observed {stats['min']} .. {stats['max']})"
                )

    domain = CATEGORICAL_DOMAINS.get(name)
    if domain is not None:
        arr = np.asarray(values)
        finite = np.isfinite(arr.astype(np.float64))
        present = {int(v) for v in np.unique(arr[finite])} if finite.any() else set()
        report["values_present"] = sorted(present)
        illegal = sorted(present - {int(v) for v in domain})
        if illegal:
            findings.append(f"values outside the legal class set: {illegal}")
        nonintegral = bool(np.any(arr[finite] != np.rint(arr[finite]))) if finite.any() else False
        if nonintegral:
            findings.append(
                "non-integral values in an enumerated channel — a class raster was "
                "resampled by interpolation, i.e. classes that do not exist were invented"
            )

    if stats["min"] is not None and stats["max"] == stats["min"] and not report["static"]:
        findings.append(
            f"dynamic channel is constant at {stats['min']} — std = 0 NaNs every "
            "normalisation that divides by it (C3)"
        )

    report["verdict"] = "suspect" if findings else "ok"
    return report


def audit_channels(channels: dict[str, np.ndarray], *, fire_id: str = "") -> dict[str, Any]:
    """Audit a ``{channel_name: array}`` mapping - the single implementation.

    Called both from the build (before the store exists) and from
    :func:`audit_dataset` (after it does), so a fire audited at build time and
    the same fire re-audited from disk cannot disagree.
    """
    reports = {name: channel_audit(name, arr) for name, arr in channels.items()}
    suspect = sorted(n for n, r in reports.items() if r["verdict"] == "suspect")
    return {
        "fire_id": fire_id,
        "n_channels": len(reports),
        "channels": reports,
        "suspect_channels": suspect,
        "verdict": "suspect" if suspect else "ok",
        "policy": (
            "PLAUSIBILITY audit, advisory only — never blocks a build, never exits "
            "non-zero. The contract (C1.5/C1.7) is the authority on structure and on "
            "the two definitional ranges; this is the R11 complement covering the "
            "twelve channels C1.7 deliberately does not police."
        ),
    }


def audit_dataset(ds: xr.Dataset) -> dict[str, Any]:
    """Audit all 14 C1 channels of one fire's store."""
    channels: dict[str, np.ndarray] = {FIRE_STATE: ds[FIRE_STATE].values}
    feat = ds[FEATURES]
    for i, name in enumerate(str(c) for c in feat["channel"].values):
        channels[name] = feat.values[:, i]
    return audit_channels(channels, fire_id=str(ds.attrs.get("fire_id", "")))


#: A fire's own scar, stamped into channel 13, is a LABEL in the feature stack.
#: On Kincade it put **86.5 % of the burned cells** in as "already burned" and no
#: structural clause could see it: the channel was binary, static and finite.
#: Above this fraction the channel is asserting that most of what the fire burned
#: had already burned before it started, which is a claim to check rather than a
#: threshold to trust - hence ``suspect``, never a failure.
SCAR_SELF_OVERLAP_SUSPECT = 0.50


def burn_scar_leak_report(ds: xr.Dataset) -> dict[str, Any]:
    """Is channel 13 telling the model where this fire is going to burn?

    Measures the SAME estimand the Kincade defect was found on - *what fraction
    of the cells this fire ever burns are already flagged as recently burned* -
    rather than an AUC surrogate. Direct measurement beats a proxy here for the
    reason ADR-016's `find -newermt` note gives: prefer a check that reads the
    value over one that proves a set is empty.

    ``lift`` is that fraction over the domain-wide base rate. A prior scar that
    is genuinely near the fire will sit slightly above 1; the Kincade leak sat at
    86.5 % against a base rate that made it enormous. Deliberately NOT a contract
    clause: a legitimately scar-heavy fire exists (C1.6's own deferral reasoning),
    so this reports and never blocks.
    """
    state = np.asarray(ds[FIRE_STATE].values)
    feat = ds[FEATURES]
    names = [str(c) for c in feat["channel"].values]
    scar = np.asarray(feat.values[0, names.index("recent_burn_scar")]) >= 0.5
    ever = state[-1] > 0

    n_cells = int(ever.size)
    n_burned = int(ever.sum())
    n_scar = int(scar.sum())
    base_rate = (n_scar / n_cells) if n_cells else 0.0
    overlap = int(np.logical_and(scar, ever).sum())
    frac_of_burned = (overlap / n_burned) if n_burned else 0.0
    findings: list[str] = []
    if frac_of_burned > SCAR_SELF_OVERLAP_SUSPECT:
        findings.append(
            f"{frac_of_burned:.1%} of this fire's burned cells are already flagged in "
            "channel 13. Kincade's self-scar leak looked exactly like this (86.5%): check "
            "the C1 guard window and the incident-name exclusion before using this fire"
        )
    return {
        "n_cells": n_cells,
        "n_burned_cells": n_burned,
        "n_scar_cells": n_scar,
        "scar_base_rate": round(base_rate, 6),
        "scar_cells_inside_final_footprint": overlap,
        "fraction_of_burned_cells_prescarred": round(frac_of_burned, 6),
        "lift_over_base_rate": (round(frac_of_burned / base_rate, 3) if base_rate > 0 else None),
        "findings": findings,
        "verdict": "suspect" if findings else "ok",
        "estimand": (
            "fraction of cells with fire_state[-1] > 0 whose static recent_burn_scar == 1"
        ),
    }


def audit_norm_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """C3.4 - norm-stats-level sanity, a SEPARATE check from per-fire QA.

    A per-fire defect propagates globally through shared normalisation: CZU's
    ``-9999`` would have moved the TRAIN mean canopy to -492% while every
    individual held-out fire stayed clean. So the aggregate is audited on its
    own terms, not inferred from the per-fire reports.
    """
    order: list[str] = list(stats.get("channel_order", []))
    mean: dict[str, float] = stats.get("mean", {})
    std: dict[str, float] = stats.get("std", {})
    findings: list[str] = []
    per_channel: dict[str, Any] = {}

    for name in order:
        mu, sd = mean.get(name), std.get(name)
        entry: dict[str, Any] = {"mean": mu, "std": sd, "findings": []}
        if mu is None or sd is None:
            entry["findings"].append("missing mean or std")
        else:
            if not (np.isfinite(mu) and np.isfinite(sd)):
                entry["findings"].append("non-finite mean or std")
            elif sd <= 0:
                entry["findings"].append(f"std = {sd} <= 0 — C3 requires std > 0")
            lo_hi = PHYSICAL_RANGES.get(name) or PLAUSIBLE_RANGES.get(name)
            # Categorical channels carry the identity transform (mean 0, std 1)
            # by C3.2, so their "mean" is not a physical quantity.
            categorical = name in CATEGORICAL_DOMAINS
            if lo_hi and not categorical and np.isfinite(mu):
                lo, hi = lo_hi
                if not (lo <= mu <= hi):
                    entry["findings"].append(
                        f"TRAIN mean {mu} is outside [{lo}, {hi}] — one fire's fill "
                        "value can do this while every fire passes its own QA (C3.4)"
                    )
        per_channel[name] = entry
        findings.extend(f"{name}: {f}" for f in entry["findings"])

    n_blocks = stats.get("n_train_blocks")
    if not isinstance(n_blocks, int) or n_blocks < 1:
        findings.append("n_train_blocks missing or not a positive int (C3.3)")
    elif n_blocks < 2:
        findings.append(f"n_train_blocks = {n_blocks} — bootstrap only, not reportable (C3.3)")

    return {
        "n_train_blocks": n_blocks,
        "train_folds": stats.get("train_folds"),
        "train_fire_ids": stats.get("train_fire_ids"),
        "channels": per_channel,
        "findings": findings,
        "verdict": "suspect" if findings else "ok",
    }


def audit_built_fires(
    fire_ids: list[str] | None = None, *, patch_manifests: bool = True
) -> dict[str, Any]:
    """Audit every built fire plus the shared norm stats; optionally patch C2.

    Writes the audit into ``manifest['provenance']['qa']['physical_audit']`` so
    the per-fire QA report the charter asks for actually contains it, and returns
    the dataset-level roll-up. This function does not write that roll-up
    anywhere: the ``audit`` command in ``data/cli.py`` is what serialises it, and
    the destination is named there once rather than claimed here twice.
    """
    from wildfire_nowcast.common.paths import fires_dir, norm_stats_path  # noqa: PLC0415
    from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: PLC0415

    root = fires_dir()
    ids = fire_ids or sorted(p.name for p in root.iterdir() if (p / "tensor.zarr").exists())
    per_fire: dict[str, Any] = {}
    for fid in ids:
        ds = open_tensor(root / fid / "tensor.zarr")
        try:
            rep = audit_dataset(ds)
            rep["burn_scar_leak"] = burn_scar_leak_report(ds)
        finally:
            ds.close()
        rep["fire_id"] = fid
        if rep["burn_scar_leak"]["verdict"] == "suspect":
            rep["verdict"] = "suspect"
        per_fire[fid] = rep
        mpath = root / fid / "manifest.json"
        if patch_manifests and mpath.is_file():
            man = json.loads(mpath.read_text())
            man.setdefault("provenance", {}).setdefault("qa", {})["physical_audit"] = rep
            mpath.write_text(json.dumps(man, indent=2) + "\n")

    ns_path = Path(norm_stats_path())
    ns_report = audit_norm_stats(json.loads(ns_path.read_text())) if ns_path.is_file() else None
    suspects = sorted(f for f, r in per_fire.items() if r["verdict"] == "suspect")
    return {
        "n_fires": len(per_fire),
        "fires": per_fire,
        "norm_stats": ns_report,
        "suspect_fires": suspects,
        "verdict": "suspect"
        if suspects or (ns_report and ns_report["verdict"] == "suspect")
        else "ok",
    }
