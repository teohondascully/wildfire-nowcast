"""D11 - the per-channel C1.6 leakage DISTRIBUTION over the 21-fire corpus.

**REPORT ONLY. This module proposes no threshold and contains no pass/fail bar.**
ADR-040 (4) P5 ruled that the C1.6 bar is set from the observed distribution, by
the maintainer, after seeing it - so a number chosen here, beside the data it was
derived from, would be a bar fitted to a result. Nothing in this file compares a
measured value against a constant, and `write_report` has no verdict field.

**The estimand, stated in its own units (ADR-041 (4)).** For one STATIC channel
of one fire, over one cell mask:

    leakage = |2 * AUC - 1|

where ``AUC = P(x_burned > x_unburned) + 0.5 * P(x_burned == x_unburned)`` is the
tie-corrected Mann-Whitney statistic of that channel's values against the fire's
FINAL ever-burned footprint. The unit of aggregation is the (fire, channel, mask)
triple; there is one value per triple and they are NOT independent draws - the
eight channels of one fire share one landscape, and fires sharing a spatial block
share terrain. Every summary below therefore names its unit and its set.

``|2*AUC-1|`` is the rank-biserial correlation magnitude: 0 = the channel cannot
order burned above unburned at all, 1 = perfect separation. It is invariant to
any monotone transform of the channel, so it does not care about units, and it is
symmetric, so a channel that predicts UNBURNED scores the same as one predicting
BURNED. See docs/interfaces.md, clause C1.6.

**Two masks, both reported, neither privileged.**
``all_cells``  - the clause exactly as written in docs/interfaces.md.
``burnable``   - cells whose FBFM40 class is burnable. This was the proposed
                 remedy for coastal fires, where canopy and elevation separate
                 ocean from land definitionally rather than leakily. It is
                 reported because it was proposed, NOT because it works: it
                 degenerates two of the eight channels by construction (see
                 ``DEGENERATE_*``), which is itself part of the shape.

**Known confounds, carried per row rather than averaged away.**
``label_source``          gofer (12 fires) vs gofer_ext (9, our reimplementation).
``n_ignition_components`` a multi-ignition fire's "final footprint" is several
                          fires in different terrain, so a static channel
                          separating them is expected rather than leaky.
``spatial_block_id``      concentration is this project's recurring failure mode.

**Controls (non-negotiable; a measure never observed to fire is not evidence).**
POSITIVE, all three routed through the SAME ``score_field`` the corpus uses:
  1. a channel set to the fire's own footprint -> analytic 1.0 exactly;
  2. a binary channel present on 96% of burned and 4% of unburned cells ->
     analytic ``a - b`` = 0.92, chosen to match the magnitude of the real
     channel-13 defect C1.6 was written for (docs/interfaces.md records it at
     ~0.92);
  3. a continuous channel = footprint + N(0,1) -> analytic ``2*Phi(1/sqrt2)-1``
     = 0.5205, which exercises the tie-free path the terrain channels use.
NEGATIVE, expected to read zero:
  4. the REAL channel values permuted across the mask. Marginal distribution and
     tie structure are preserved exactly, so anything above sampling error is an
     estimator defect;
  5. an iid uniform field.
DIAGNOSTIC - **classified as a diagnostic in this docstring BEFORE it was run,**
because ADR-047 records a control being reclassified after it failed:
  6. the SPATIAL NULL. A smooth random field, independent of the fire by
     construction, scored against a compact blob footprint. Its expectation is 0
     by symmetry but its spread is NOT small, because a smooth field over a
     landscape has few effective degrees of freedom. This is a property of the
     estimand and the geometry, not of the estimator, so it can neither validate
     nor invalidate the measure - which is exactly why it is not a pass/fail
     control. It is here because it is the single most decision-relevant number
     for reading the distribution's tail.

The permutation null SD is reported per row for scale ONLY. It is the WRONG null
for a spatially autocorrelated channel and must not be read as a significance
test; control 6 is why.

READ-ONLY with respect to every existing artifact. Writes exactly one new file
under ``data/leakage/``. No tensor, manifest, norm_stats.json, qa_audit.json,
split or crossings.json is opened for writing anywhere in this module.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from wildfire_nowcast.common.contract import (
    CHANNEL_INDEX_OFFSET,
    CONTRACT_VERSION,
    FBFM40_NONBURNABLE,
    STATIC_CHANNELS,
)
from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.common.paths import data_dir, fires_dir

__all__ = [
    "LeakageScore",
    "FireFields",
    "MASKS",
    "STATIC_ORDER",
    "avg_ranks",
    "score_field",
    "load_fire",
    "measure_corpus",
    "positive_controls",
    "negative_controls",
    "spatial_null_diagnostic",
    "build_report",
    "write_report",
    "leakage_path",
    "main",
]

logger = logging.getLogger(__name__)

#: The eight C1 static channels, in C1 channel order. Reading the order from the
#: contract rather than spelling it keeps this table from drifting (C0).
STATIC_ORDER: tuple[str, ...] = tuple(
    name
    for name in (
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "fuel_model_id",
        "canopy_cover",
        "water_barrier_mask",
        "recent_burn_scar",
    )
)

if set(STATIC_ORDER) != set(STATIC_CHANNELS):  # pragma: no cover - import-time guard
    raise RuntimeError(
        "STATIC_ORDER has drifted from contract.STATIC_CHANNELS: "
        f"{sorted(set(STATIC_ORDER) ^ set(STATIC_CHANNELS))}. C1.6 is a per-channel "
        "clause; a channel silently missing from this table is a row that never "
        "appears in the distribution, which is the emptiness-is-invisible defect."
    )

MASKS: tuple[str, ...] = ("all_cells", "burnable")

#: Degeneracy reasons. A degenerate triple is EXCLUDED from every distribution
#: and COUNTED, never allowed to average in as a zero - a constant channel does
#: not "score 0 leakage", it has no leakage statistic at all.
DEGENERATE_CONSTANT = "constant_over_mask"
DEGENERATE_NO_BURNED = "no_burned_cells_in_mask"
DEGENERATE_NO_UNBURNED = "no_unburned_cells_in_mask"
DEGENERATE_EMPTY = "mask_empty"

#: Chosen to BRACKET the measured autocorrelation of every real static channel
#: (1-11 cells e-fold), so no channel has to be compared to an extrapolated row.
_SPATIAL_NULL_SIGMAS_CELLS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)
_SPATIAL_NULL_DRAWS = 200


def leakage_path() -> Path:
    """``data/leakage/c1_6_channel_leakage.json`` - the one file D11 writes.

    A DESTINATION, not a citation: ``write_report`` below produces it from a
    built corpus and no clone carries it.
    """
    return data_dir() / "leakage" / "c1_6_channel_leakage.json"


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------


def avg_ranks(x: np.ndarray) -> np.ndarray:
    """Tie-averaged 1-based ranks of ``x``.

    Ties must take the AVERAGE rank, not an arbitrary order-dependent one:
    ``fuel_model_id`` and the two {0,1} masks are almost entirely ties, and a
    tie-broken rank would let the sort order of equal values manufacture
    separation out of nothing.
    """
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts + 1) / 2.0)[inv]


@dataclass(frozen=True)
class LeakageScore:
    """One (channel, mask) measurement, with everything needed to audit it."""

    leakage: float | None
    auc: float | None
    n_cells: int
    n_burned: int
    n_unburned: int
    n_distinct_values: int
    tie_fraction: float
    null_sd: float | None
    degenerate: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "leakage": self.leakage,
            "auc": self.auc,
            "n_cells": self.n_cells,
            "n_burned": self.n_burned,
            "n_unburned": self.n_unburned,
            "n_distinct_values": self.n_distinct_values,
            "tie_fraction": self.tie_fraction,
            "permutation_null_sd": self.null_sd,
            "degenerate": self.degenerate,
        }


def score_field(
    values: np.ndarray,
    burned: np.ndarray,
    mask: np.ndarray | None = None,
) -> LeakageScore:
    """``|2*AUC-1|`` of ``values`` against ``burned``, restricted to ``mask``.

    Every number in this report - corpus rows, positive controls, negative
    controls and the spatial null - comes through this one function. A control
    that exercises a different code path from the measurement it validates is
    not a control.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    y = np.asarray(burned, dtype=bool).ravel()
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        v, y = v[m], y[m]
    n = int(v.size)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n == 0:
        return LeakageScore(None, None, 0, 0, 0, 0, 0.0, None, DEGENERATE_EMPTY)
    uniq, counts = np.unique(v, return_counts=True)
    n_distinct = int(uniq.size)
    tie_frac = float((counts[counts > 1]).sum() / n) if n else 0.0
    if n_pos == 0:
        return LeakageScore(
            None, None, n, 0, n_neg, n_distinct, tie_frac, None, DEGENERATE_NO_BURNED
        )
    if n_neg == 0:
        return LeakageScore(
            None, None, n, n_pos, 0, n_distinct, tie_frac, None, DEGENERATE_NO_UNBURNED
        )
    if n_distinct <= 1:
        return LeakageScore(
            None, None, n, n_pos, n_neg, n_distinct, tie_frac, None, DEGENERATE_CONSTANT
        )
    r = avg_ranks(v)
    auc = float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    # Mann-Whitney U null SD WITH the tie correction, mapped into 2*AUC-1 units.
    tie_term = float(((counts.astype(np.float64) ** 3) - counts).sum())
    var_u = (n_pos * n_neg / 12.0) * ((n + 1) - tie_term / (n * (n - 1.0)))
    null_sd = 2.0 * math.sqrt(max(var_u, 0.0)) / (n_pos * n_neg)
    return LeakageScore(
        leakage=abs(2.0 * auc - 1.0),
        auc=auc,
        n_cells=n,
        n_burned=n_pos,
        n_unburned=n_neg,
        n_distinct_values=n_distinct,
        tie_fraction=tie_frac,
        null_sd=null_sd,
        degenerate=None,
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# corpus loading - READ ONLY
# --------------------------------------------------------------------------


@dataclass
class FireFields:
    """One fire's static fields, footprint and masks. Nothing here is written."""

    fire_id: str
    label_source: str
    spatial_block_id: int
    cv_fold: int
    gofer_version: str
    n_ignition_components: int
    fuel_vintage_lag_years: int
    n_hours: int
    shape: tuple[int, int]
    burned: np.ndarray
    statics: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    integrity: dict[str, Any] = field(default_factory=dict)


def load_fire(fire_dir: Path) -> FireFields:
    """Read one fire's tensor + manifest. Opened read-only, never written."""
    manifest = json.loads((fire_dir / "manifest.json").read_text())
    ds = xr.open_zarr(fire_dir / "tensor.zarr")
    try:
        state = np.asarray(ds["fire_state"].values)
        names = [str(c) for c in ds["channel"].values]
        feats = np.asarray(ds["features"].values, dtype=np.float64)
    finally:
        ds.close()

    # Footprint. C1.4 enforces fire_state non-decreasing, so the final frame IS
    # the ever-burned set - but READ THE VALUE rather than trusting the clause:
    # the two definitions are computed separately and their disagreement is
    # counted into the artifact.
    final = state[-1] > 0
    ever = (state > 0).any(axis=0)
    burned = ever
    n_disagree = int((final != ever).sum())

    statics: dict[str, np.ndarray] = {}
    nonstatic: dict[str, int] = {}
    for name in STATIC_ORDER:
        if name not in names:
            continue
        ci = names.index(name)
        block = feats[:, ci, :, :]
        spread = block.max(axis=0) - block.min(axis=0)
        nonstatic[name] = int((spread != 0.0).sum())
        statics[name] = block[0]

    fuel = statics.get("fuel_model_id")
    if fuel is None:
        raise RuntimeError(f"{fire_dir.name}: no fuel_model_id channel; cannot mask")
    nonburnable = np.isin(np.rint(fuel).astype(int), sorted(FBFM40_NONBURNABLE))
    masks = {
        "all_cells": np.ones(burned.shape, dtype=bool),
        "burnable": ~nonburnable,
    }

    return FireFields(
        fire_id=str(manifest["fire_id"]),
        label_source=str(manifest.get("label_source", "unknown")),
        spatial_block_id=int(manifest["spatial_block_id"]),
        cv_fold=int(manifest["cv_fold"]),
        gofer_version=str(manifest.get("gofer_version", "unknown")),
        n_ignition_components=int(manifest.get("n_ignition_components", 1)),
        fuel_vintage_lag_years=int(manifest.get("fuel_vintage_lag_years", -1)),
        n_hours=int(manifest.get("n_hours", state.shape[0])),
        shape=(int(burned.shape[0]), int(burned.shape[1])),
        burned=burned,
        statics=statics,
        masks=masks,
        integrity={
            "footprint_final_vs_ever_disagreeing_cells": n_disagree,
            "nonstatic_cells_per_channel": nonstatic,
            "channel_index_offset": CHANNEL_INDEX_OFFSET,
            "n_nonburnable_cells": int(nonburnable.sum()),
            "nonburnable_fraction": float(nonburnable.mean()),
            "burned_fraction_all_cells": float(burned.mean()),
        },
    )


def load_corpus(root: Path | None = None) -> list[FireFields]:
    base = Path(root) if root is not None else fires_dir()
    dirs = sorted(p for p in base.iterdir() if (p / "manifest.json").is_file())
    if not dirs:
        raise RuntimeError("no fires on disk: refusing to report an empty scan")
    return [load_fire(p) for p in dirs]


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------


def measure_corpus(fires: list[FireFields]) -> list[dict[str, Any]]:
    """One row per (fire, channel, mask). This is the whole distribution."""
    rows: list[dict[str, Any]] = []
    for f in fires:
        for name in STATIC_ORDER:
            present = name in f.statics
            for mask_name in MASKS:
                if not present:
                    rows.append(
                        {
                            "fire_id": f.fire_id,
                            "label_source": f.label_source,
                            "spatial_block_id": f.spatial_block_id,
                            "cv_fold": f.cv_fold,
                            "n_ignition_components": f.n_ignition_components,
                            "channel": name,
                            "mask": mask_name,
                            "present": False,
                            "leakage": None,
                            "degenerate": "channel_absent",
                        }
                    )
                    continue
                s = score_field(f.statics[name], f.burned, f.masks[mask_name])
                row = {
                    "fire_id": f.fire_id,
                    "label_source": f.label_source,
                    "spatial_block_id": f.spatial_block_id,
                    "cv_fold": f.cv_fold,
                    "n_ignition_components": f.n_ignition_components,
                    "channel": name,
                    "mask": mask_name,
                    "present": True,
                }
                row.update(s.as_dict())
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------


def positive_controls(fires: list[FireFields], *, seed: int = 11011) -> dict[str, Any]:
    """Three planted leaks of KNOWN magnitude, all through ``score_field``.

    Each names its analytic target before it is measured. "Non-zero" is the
    weakest possible bar for a positive control, so these assert the MAGNITUDE:
    a measure that fires at the wrong number is as broken as one that never
    fires, and only a magnitude check can see the difference.
    """
    rng = np.random.default_rng(seed)
    perfect: list[float] = []
    graded: list[dict[str, float]] = []
    gaussian: list[float] = []
    for f in fires:
        burned = f.burned
        # (1) the historical channel-13 defect in its pure form: the channel IS
        #     the fire's own footprint. Analytic |2*AUC-1| = 1 exactly.
        perfect.append(float(score_field(burned.astype(np.float64), burned).leakage or 0.0))

        # (2) the same defect degraded to the ~0.92 magnitude docs/interfaces.md
        #     records for the real leak. For a BINARY channel present on a
        #     fraction a of burned and b of unburned cells, 2*AUC-1 == a - b
        #     EXACTLY (the tie terms cancel), so the realised fractions give an
        #     exact target with no sampling slack at all.
        chan = np.zeros(burned.shape, dtype=np.float64)
        pos_idx = np.flatnonzero(burned.ravel())
        neg_idx = np.flatnonzero(~burned.ravel())
        take_p = rng.random(pos_idx.size) < 0.96
        take_n = rng.random(neg_idx.size) < 0.04
        flat = chan.ravel()
        flat[pos_idx[take_p]] = 1.0
        flat[neg_idx[take_n]] = 1.0
        a_hat = float(take_p.mean())
        b_hat = float(take_n.mean())
        got = score_field(chan, burned).leakage
        graded.append(
            {
                "fire_id": f.fire_id,  # type: ignore[dict-item]
                "a_hat": a_hat,
                "b_hat": b_hat,
                "analytic": abs(a_hat - b_hat),
                "measured": float(got or 0.0),
                "abs_error": abs(float(got or 0.0) - abs(a_hat - b_hat)),
            }
        )

        # (3) a CONTINUOUS planted leak: footprint + N(0,1). Two unit-variance
        #     normals separated by 1 give AUC = Phi(1/sqrt2), so the target is
        #     0.52050. This is the tie-free path elevation and slope take.
        cont = burned.astype(np.float64) + rng.standard_normal(burned.shape)
        gaussian.append(float(score_field(cont, burned).leakage or 0.0))

    target_gauss = 2.0 * _normal_cdf(1.0 / math.sqrt(2.0)) - 1.0
    max_graded_err = max(g["abs_error"] for g in graded)
    gauss_err = abs(float(np.mean(gaussian)) - target_gauss)
    return {
        "perfect_footprint_copy": {
            "analytic": 1.0,
            "measured_min": float(np.min(perfect)),
            "measured_max": float(np.max(perfect)),
            "n_fires": len(perfect),
            "exact_on_all_fires": bool(np.allclose(perfect, 1.0, atol=1e-12)),
        },
        "degraded_scar_a096_b004": {
            "analytic_rule": "2*AUC-1 == a - b exactly for a binary channel",
            "nominal_target": 0.92,
            "measured_mean": float(np.mean([g["measured"] for g in graded])),
            "max_abs_error_vs_analytic": max_graded_err,
            "per_fire": graded,
        },
        "continuous_shifted_gaussian": {
            "analytic": target_gauss,
            "measured_mean": float(np.mean(gaussian)),
            "measured_min": float(np.min(gaussian)),
            "measured_max": float(np.max(gaussian)),
            "abs_error_of_mean": gauss_err,
        },
        "fired": bool(
            np.allclose(perfect, 1.0, atol=1e-12) and max_graded_err < 1e-9 and gauss_err < 0.02
        ),
    }


def negative_controls(
    fires: list[FireFields], *, seed: int = 909, n_permutations: int = 20
) -> dict[str, Any]:
    """Two nulls that must read zero: value permutation, and an iid field."""
    rng = np.random.default_rng(seed)
    perm_z: list[float] = []
    perm_vals: list[float] = []
    iid_vals: list[float] = []
    for f in fires:
        for mask_name in MASKS:
            mask = f.masks[mask_name]
            for values in f.statics.values():
                base = score_field(values, f.burned, mask)
                if base.degenerate is not None or base.null_sd in (None, 0.0):
                    continue
                pool = np.asarray(values, dtype=np.float64).ravel()[mask.ravel()]
                y = f.burned.ravel()[mask.ravel()]
                for _ in range(n_permutations):
                    s = score_field(rng.permutation(pool), y)
                    if s.leakage is None:
                        continue
                    perm_vals.append(s.leakage)
                    perm_z.append(s.leakage / float(base.null_sd))
        for _ in range(n_permutations):
            noise = rng.random(f.burned.shape)
            s = score_field(noise, f.burned)
            if s.leakage is not None:
                iid_vals.append(s.leakage)
    perm_arr = np.asarray(perm_vals)
    iid_arr = np.asarray(iid_vals)
    max_z = float(np.max(np.abs(perm_z))) if perm_z else float("nan")
    return {
        "value_permutation_within_mask": {
            "n_draws": int(perm_arr.size),
            "mean": float(perm_arr.mean()),
            "p50": float(np.percentile(perm_arr, 50)),
            "p95": float(np.percentile(perm_arr, 95)),
            "max": float(perm_arr.max()),
            "max_abs_z_vs_permutation_null_sd": max_z,
        },
        "iid_uniform_field": {
            "n_draws": int(iid_arr.size),
            "mean": float(iid_arr.mean()),
            "p95": float(np.percentile(iid_arr, 95)),
            "max": float(iid_arr.max()),
        },
        # Flat means: consistent with zero at the estimator's own sampling scale.
        # 4 SD is a sampling statement about ~8k independent draws, not a bar on
        # any measured channel.
        "passed": bool(max_z < 4.0 and iid_arr.max() < 0.2),
    }


def _smooth_random_field(
    shape: tuple[int, int], sigma_cells: float, rng: np.random.Generator
) -> np.ndarray:
    """White noise Gaussian-smoothed in the Fourier domain. Independent of the
    fire by construction: it is generated from a seed and never sees ``burned``."""
    ny, nx = shape
    noise = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    filt = np.exp(-2.0 * (math.pi**2) * (sigma_cells**2) * (ky**2 + kx**2))
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))


def _acf_efold_cells(values: np.ndarray, mask: np.ndarray) -> float | None:
    """Crude e-folding length of the field's spatial autocorrelation, in cells.

    Averaged over the row and column directions. Used only to say WHICH spatial
    null row is comparable to a given real channel; it decides nothing.
    """
    v = np.asarray(values, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool)
    if m.sum() < 10 or float(np.nanstd(v[m])) == 0.0:
        return None
    v = v - float(v[m].mean())
    v[~m] = 0.0
    denom = float((v[m] ** 2).sum())
    if denom <= 0:
        return None
    max_lag = int(min(v.shape) // 2)
    for lag in range(1, max_lag + 1):
        c_row = float((v[:, :-lag] * v[:, lag:]).sum())
        c_col = float((v[:-lag, :] * v[lag:, :]).sum())
        if (c_row + c_col) / (2.0 * denom) < math.exp(-1.0):
            return float(lag)
    return float(max_lag)


def spatial_null_diagnostic(
    fires: list[FireFields], *, seed: int = 24601, n_draws: int = _SPATIAL_NULL_DRAWS
) -> dict[str, Any]:
    """What an INDEPENDENT smooth field scores against a compact footprint.

    Declared a DIAGNOSTIC in this module's docstring before it was run. It is
    not a pass/fail control: its expectation is 0 by symmetry, but its spread is
    a property of the estimand and of blob-versus-smooth-field geometry, so it
    can neither validate nor invalidate the estimator.
    """
    rng = np.random.default_rng(seed)
    by_sigma: dict[str, Any] = {}
    for sigma in _SPATIAL_NULL_SIGMAS_CELLS:
        pooled: list[float] = []
        per_fire: list[dict[str, Any]] = []
        efolds: list[float] = []
        for f in fires:
            vals: list[float] = []
            for i in range(n_draws):
                fld = _smooth_random_field(f.shape, sigma, rng)
                if i < 20:
                    # Measure the null field's OWN autocorrelation with the same
                    # estimator used on the real channels, so a reader can match
                    # a channel to the comparable null row by measurement rather
                    # than by an assumed sigma-to-e-fold conversion.
                    e = _acf_efold_cells(fld, f.masks["all_cells"])
                    if e is not None:
                        efolds.append(e)
                s = score_field(fld, f.burned)
                if s.leakage is not None:
                    vals.append(s.leakage)
            arr = np.asarray(vals)
            pooled.extend(vals)
            per_fire.append(
                {
                    "fire_id": f.fire_id,
                    "mean": float(arr.mean()),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "max": float(arr.max()),
                }
            )
        p = np.asarray(pooled)
        by_sigma[f"sigma_{sigma:g}_cells"] = {
            "measured_acf_efold_cells_p50": float(np.percentile(efolds, 50)),
            "n_draws": int(p.size),
            "mean": float(p.mean()),
            "p50": float(np.percentile(p, 50)),
            "p90": float(np.percentile(p, 90)),
            "p95": float(np.percentile(p, 95)),
            "p99": float(np.percentile(p, 99)),
            "max": float(p.max()),
            "per_fire": per_fire,
        }
    return {
        "what_this_is": (
            "|2*AUC-1| of a smooth random field, independent of the fire, against "
            "that fire's footprint. Expectation 0; spread is geometry, not error."
        ),
        "classified_as": "diagnostic, declared before the run (not pass/fail)",
        "by_correlation_length": by_sigma,
    }


# --------------------------------------------------------------------------
# distribution summaries
# --------------------------------------------------------------------------

_QUANTILES = (0, 5, 10, 25, 50, 75, 90, 95, 100)


def _quantile_block(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    out: dict[str, Any] = {"n": int(a.size), "mean": float(a.mean())}
    for q in _QUANTILES:
        out[f"p{q}"] = float(np.percentile(a, q))
    return out


def _group(rows: list[dict[str, Any]], key: str, mask: str) -> dict[str, Any]:
    buckets: dict[Any, list[float]] = defaultdict(list)
    for r in rows:
        if r["mask"] != mask or r.get("leakage") is None:
            continue
        buckets[r[key]].append(float(r["leakage"]))
    return {str(k): _quantile_block(v) for k, v in sorted(buckets.items(), key=str)}


def _cross_group(rows: list[dict[str, Any]], mask: str) -> dict[str, Any]:
    """channel x label_source. The marginal split of the two label sources is
    confounded by channel composition only if the channels differ between them -
    they do not - but a crossed table lets a reader see a per-channel gofer vs
    gofer_ext difference directly instead of inferring it from two margins."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["mask"] != mask or r.get("leakage") is None:
            continue
        buckets[(str(r["channel"]), str(r["label_source"]))].append(float(r["leakage"]))
    out: dict[str, Any] = {}
    for ch in STATIC_ORDER:
        row: dict[str, Any] = {}
        for src in ("gofer", "gofer_ext"):
            vals = buckets.get((ch, src), [])
            row[src] = (
                {"n": len(vals), "p50": float(np.percentile(vals, 50)), "max": float(max(vals))}
                if vals
                else {"n": 0}
            )
        a, b = row["gofer"], row["gofer_ext"]
        row["p50_difference_gofer_minus_ext"] = (
            round(a["p50"] - b["p50"], 6) if a["n"] and b["n"] else None
        )
        out[ch] = row
    return out


def _tail_composition(rows: list[dict[str, Any]], mask: str, frac: float) -> dict[str, Any]:
    """Composition of the top ``frac`` of rows BY RANK. No value threshold is
    used or implied: the tail is defined as a fixed share of the sample, so this
    describes the sample's shape and cannot be read as a proposed bar."""
    live = [r for r in rows if r["mask"] == mask and r.get("leakage") is not None]
    live.sort(key=lambda r: -float(r["leakage"]))
    k = max(1, int(round(frac * len(live))))
    top = live[:k]

    def share(key: str) -> list[list[Any]]:
        c = Counter(str(r[key]) for r in top)
        return [[name, n, round(n / k, 4)] for name, n in c.most_common()]

    return {
        "definition": f"top {frac:.0%} of {len(live)} non-degenerate rows, by rank",
        "k": k,
        "cut_value_at_rank_k": float(top[-1]["leakage"]),
        "by_fire": share("fire_id"),
        "by_channel": share("channel"),
        "by_spatial_block_id": share("spatial_block_id"),
        "by_label_source": share("label_source"),
    }


def _matched_null_table(
    rows: list[dict[str, Any]],
    mask: str,
    acf_median: dict[str, float],
    null: dict[str, Any],
) -> dict[str, Any]:
    """Each channel's observed values beside the spatial-null row of the SAME
    measured smoothness.

    This is a SIDE-BY-SIDE, not a test. Three limits, stated because they bound
    what the comparison can mean: (a) the null is a Gaussian random field matched
    on one summary of autocorrelation, not on a channel's real structure - terrain
    has ridges and a Gaussian field does not; (b) ``fuel_model_id`` is categorical,
    so a continuous-field analogue is rough at best; (c) the e-fold estimator
    cannot resolve below ~2 cells, so the shortest-range channels are matched to a
    null row that is if anything too WIDE, which understates their excess.
    """
    live: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["mask"] == mask and r.get("leakage") is not None:
            live[str(r["channel"])].append(float(r["leakage"]))
    out: dict[str, Any] = {}
    for ch, vals in sorted(live.items()):
        efold = acf_median.get(ch)
        if efold is None:
            continue
        key, row = min(
            null.items(),
            key=lambda kv: abs(float(kv[1]["measured_acf_efold_cells_p50"]) - efold),
        )
        out[ch] = {
            "channel_acf_efold_cells_p50": efold,
            "matched_null_row": key,
            "matched_null_acf_efold_cells_p50": row["measured_acf_efold_cells_p50"],
            "observed_p50": float(np.percentile(vals, 50)),
            "observed_max": float(max(vals)),
            "null_p50": row["p50"],
            "null_p90": row["p90"],
            "null_p95": row["p95"],
            "null_max": row["max"],
        }
    return out


def build_report() -> dict[str, Any]:
    fires = load_corpus()
    logger.info("loaded %d fires", len(fires))
    rows = measure_corpus(fires)

    # per-channel autocorrelation length, so a reader can pick the comparable
    # spatial-null row for each channel rather than guessing.
    acf: dict[str, list[float]] = defaultdict(list)
    for f in fires:
        for name, values in f.statics.items():
            e = _acf_efold_cells(values, f.masks["all_cells"])
            if e is not None:
                acf[name].append(e)

    acf_median = {k: float(np.percentile(v, 50)) for k, v in sorted(acf.items())}
    controls = {
        "positive": positive_controls(fires),
        "negative": negative_controls(fires),
        "spatial_null_diagnostic": spatial_null_diagnostic(fires),
    }
    null_rows = controls["spatial_null_diagnostic"]["by_correlation_length"]

    summaries: dict[str, Any] = {}
    for mask in MASKS:
        live = [
            float(r["leakage"]) for r in rows if r["mask"] == mask and r.get("leakage") is not None
        ]
        deg = [r for r in rows if r["mask"] == mask and r.get("leakage") is None]
        per_fire_max = sorted(
            (
                {
                    "fire_id": fid,
                    "max_leakage": max(v for _, v in items),
                    "argmax_channel": max(items, key=lambda t: t[1])[0],
                    "n_live_channels": len(items),
                }
                for fid, items in _by_fire(rows, mask).items()
            ),
            key=lambda d: -float(d["max_leakage"]),
        )
        per_block_max = sorted(
            (
                {
                    "spatial_block_id": bid,
                    "max_leakage": max(v for _, _, v in items),
                    "argmax": max(items, key=lambda t: t[2])[:2],
                    "n_fires": len({fid for fid, _, _ in items}),
                    "n_live_rows": len(items),
                    "median_leakage": float(np.percentile([v for _, _, v in items], 50)),
                }
                for bid, items in _by_block(rows, mask).items()
            ),
            key=lambda d: -float(d["max_leakage"]),
        )
        summaries[mask] = {
            "overall": _quantile_block(live),
            "by_channel": _group(rows, "channel", mask),
            "by_label_source": _group(rows, "label_source", mask),
            "by_channel_x_label_source": _cross_group(rows, mask),
            "by_channel_vs_matched_spatial_null": _matched_null_table(
                rows, mask, acf_median, null_rows
            ),
            "by_spatial_block_id": _group(rows, "spatial_block_id", mask),
            "by_n_ignition_components": _group(rows, "n_ignition_components", mask),
            "per_fire_max": per_fire_max,
            "per_block_max": per_block_max,
            "tail_top_10pct": _tail_composition(rows, mask, 0.10),
            "tail_top_25pct": _tail_composition(rows, mask, 0.25),
            "degenerate_rows": [
                {
                    "fire_id": r["fire_id"],
                    "channel": r["channel"],
                    "reason": r["degenerate"],
                    "label_source": r["label_source"],
                }
                for r in deg
            ],
            "exceedance_curve": _exceedance(live),
        }

    return {
        "task": "D11",
        "what": "C1.6 per-channel leakage DISTRIBUTION over the 21-fire corpus",
        "report_only": True,
        "proposes_no_threshold": True,
        "estimand": "|2*AUC-1| of a static channel vs the fire's final ever-burned footprint",
        "unit_of_aggregation": "(fire, channel, mask) triple",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
        "n_fires": len(fires),
        "n_channels": len(STATIC_ORDER),
        "masks": list(MASKS),
        "channel_autocorrelation_efold_cells_median": acf_median,
        "controls": controls,
        # Covariates that differ between the two label sources and could drive a
        # difference in the statistic without any difference in leakage: the
        # estimator's null spread depends on domain size and on class balance.
        "label_source_covariates": {
            src: {
                "n_fires": len(grp),
                "domain_cells_p50": float(
                    np.percentile([f.shape[0] * f.shape[1] for f in grp], 50)
                ),
                "burned_fraction_p50": float(
                    np.percentile([f.integrity["burned_fraction_all_cells"] for f in grp], 50)
                ),
                "nonburnable_fraction_p50": float(
                    np.percentile([f.integrity["nonburnable_fraction"] for f in grp], 50)
                ),
                "n_hours_p50": float(np.percentile([f.n_hours for f in grp], 50)),
                "fires": [f.fire_id for f in grp],
            }
            for src, grp in _by_source(fires).items()
        },
        "integrity": {f.fire_id: f.integrity for f in fires},
        "summaries": summaries,
        "rows": rows,
    }


def _by_source(fires: list[FireFields]) -> dict[str, list[FireFields]]:
    out: dict[str, list[FireFields]] = defaultdict(list)
    for f in fires:
        out[f.label_source].append(f)
    return dict(sorted(out.items()))


def _by_fire(rows: list[dict[str, Any]], mask: str) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        if r["mask"] == mask and r.get("leakage") is not None:
            out[str(r["fire_id"])].append((str(r["channel"]), float(r["leakage"])))
    return out


def _by_block(rows: list[dict[str, Any]], mask: str) -> dict[int, list[tuple[str, str, float]]]:
    out: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    for r in rows:
        if r["mask"] == mask and r.get("leakage") is not None:
            out[int(r["spatial_block_id"])].append(
                (str(r["fire_id"]), str(r["channel"]), float(r["leakage"]))
            )
    return out


def _exceedance(values: list[float]) -> list[list[float]]:
    """The empirical survival function, on a fixed 0.05 grid. This is the
    distribution itself expressed as counts; it selects nothing."""
    a = np.asarray(values, dtype=np.float64)
    grid = np.arange(0.0, 1.0001, 0.05)
    return [[round(float(g), 2), int((a > g).sum())] for g in grid]


def write_report(path: Path | None = None) -> Path:
    """Write the D11 artifact. ADDITIVE ONLY; refuses the frozen corpus.

    The controls BLOCK the write: an artifact whose measure was never observed
    to fire at a known magnitude is not evidence, and shipping one would put a
    number in front of a reader that nothing had validated.
    """
    out = Path(path) if path is not None else leakage_path()
    if Path(fires_dir()) in out.parents or out.name in {
        "norm_stats.json",
        "qa_audit.json",
        "crossings.json",
    }:
        raise RuntimeError("D11 is additive: refusing to write into the frozen corpus")
    rep = build_report()
    if not rep["controls"]["positive"]["fired"]:
        raise RuntimeError("positive control did NOT fire at its analytic magnitude")
    if not rep["controls"]["negative"]["passed"]:
        raise RuntimeError("a negative control was NOT flat; the estimator is suspect")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Write the artifact and print its path. Progress narration is a DIAGNOSTIC.

    Was a bare ``print(write_report())`` over a ``verbose=True`` wired in one
    line above it, which fused the path (this program's answer) to the corpus
    count (a fact about the run). ADR-103 separates them and puts the level here,
    in the only place allowed to set one.
    """
    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.data.leakage", description=__doc__
    )
    parser.add_argument("--out", default=None, help="artifact path (default: the canonical one)")
    add_logging_arguments(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_from_args(args, default_verbosity=1)
    print(write_report(Path(args.out) if args.out else None))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
