"""[M9] The CONDITIONAL RESPONSE: how truth's spread rate moves with covariates,
and how the model's does.

ADR-037 (2) established that no GLOBAL lever reaches the growth mis-calibration -
a 30x change in the latent's mean bought 0.3%, a 2x2 factorial killed both
candidate mechanisms, and the ONE-PARAMETER wind ellipse loses the same 16%
train->held-out that we do. What no arm sweep has ever measured is the thing
those three results point at: **the SLOPE of the predicted rate against the
covariates, beside the slope of the realised rate against the same covariates.**

THE ESTIMAND
------------
Growth is very nearly ``frontier_length x rate x horizon``, so the rate - not the
growth - is the quantity a covariate acts on::

    y = log( new burned cells in the growth band / frontier cells at t0 )

fitted against standardised window-level covariates, ONCE FOR TRUTH and ONCE FOR
THE MODEL'S ENSEMBLE MEAN, on the SAME windows. Under a compressed conditional
response the model's slopes are shrunk toward zero relative to truth's, and the
compression ratio ``beta_model / beta_truth`` is the number that says so.

WHY GROWTH WINDOWS ONLY, AND WHY THAT IS NOT A CHOICE
------------------------------------------------------
``log(0)`` is undefined and truth is exactly 0 on ~2/3 of windows (51-91%
dataset-wide; see :mod:`wildfire_nowcast.eval.masks`). Adding an epsilon would
make every slope a function of the epsilon. So the regression is fitted on the
GROWTH stratum, which ADR-026 already makes the reporting standard, and the
dormant stratum is reported separately in absolute cells because a rate is not
defined there. **This means the slope comparison CANNOT speak to the missing OFF
state** - that is a
different measurement on a different stratum, and conflating them is what made
"we over-predict 2.66-3.06x" survive for weeks as a statement about spread rate.

UNCERTAINTY IS ACROSS BLOCKS, NOT ACROSS WINDOWS
-------------------------------------------------
Windows overlap in time and within a fire, so a window-level standard error is a
fiction with a plausible-looking magnitude. Every slope is therefore fitted
INDEPENDENTLY PER BLOCK and reported as mean +/- SD over blocks - the same
equal-block convention ADR-021 (4) adopted for every other criterion, for the
same reason (Creek alone is 47% of the window-pooled held-out mass). The pooled
fit is emitted beside it and is never the headline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from wildfire_nowcast.model.inputs import (
    ForecastWindow,
    contagion_seed,
    static_index,
    weather_index,
)
from wildfire_nowcast.model.spread import fuel_group_fields

__all__ = [
    "COVARIATES",
    "window_covariates",
    "window_row",
    "fit_response",
    "compare_responses",
    "explain_block_deficit",
]

#: The covariates, in the order they are reported. Each is ONE scalar per window,
#: measured over the GROWTH BAND (weather averaged over the horizon), because
#: that is the region the model is allowed to grow into and a domain mean is
#: dominated by far-field cells no forecast was uncertain about.
#:
#: ``log_frontier_cells`` is in the list ON PURPOSE even though the response is
#: already per frontier cell: its slope is a DIRECT TEST of the proportionality
#: the estimand assumes. A slope of 0 means growth really is proportional to
#: frontier length; anything else says the geometry term is mis-specified, and
#: that would be a finding about my estimand rather than about the model.
COVARIATES: tuple[str, ...] = (
    "wind_speed",
    "rh_2m",
    "temp_2m",
    "fuel_moisture_proxy",
    "slope",
    "canopy_cover",
    "burnable_fraction",
    "barrier_fraction",
    "log_frontier_cells",
)

_EVENT_UNBURNED = 0


def _band(x0: np.ndarray, radius: int) -> np.ndarray:
    """Unburned cells within ``radius`` of the t0 frontier - the same mask C6 scores on.

    Imported behaviour, not re-derived: ``eval.masks.growth_band`` is the single
    definition and this calls it. A second spelling of the scoring region is how
    a diagnostic ends up describing a different experiment from the one it
    explains.
    """
    from wildfire_nowcast.eval.masks import growth_band

    return growth_band(x0, radius)


def window_covariates(window: ForecastWindow, *, band_radius: int) -> dict[str, float]:
    """One scalar per covariate for one window, in raw physical units."""
    x0 = np.asarray(window.x0)
    band = _band(x0, band_radius)
    frontier = contagion_seed(x0)
    n_frontier = int(frontier.sum())
    static = np.asarray(window.static, dtype=np.float64)
    weather = np.asarray(window.weather, dtype=np.float64)

    def _band_mean(field: np.ndarray) -> float:
        return float(field[band].mean()) if band.any() else float("nan")

    u = weather[:, weather_index("wind_u10")]
    v = weather[:, weather_index("wind_v10")]
    speed = np.hypot(u, v).mean(axis=0)

    fuel = static[static_index("fuel_model_id")]
    _, _, burnable = fuel_group_fields(fuel)

    return {
        "wind_speed": _band_mean(speed),
        "rh_2m": _band_mean(weather[:, weather_index("rh_2m")].mean(axis=0)),
        "temp_2m": _band_mean(weather[:, weather_index("temp_2m")].mean(axis=0)),
        "fuel_moisture_proxy": _band_mean(
            weather[:, weather_index("fuel_moisture_proxy")].mean(axis=0)
        ),
        "slope": _band_mean(static[static_index("slope")]),
        "canopy_cover": _band_mean(static[static_index("canopy_cover")]),
        "burnable_fraction": _band_mean(burnable.astype(np.float64)),
        "barrier_fraction": _band_mean(static[static_index("water_barrier_mask")]),
        "log_frontier_cells": math.log(max(n_frontier, 1)),
        "_n_frontier_cells": float(n_frontier),
        "_n_band_cells": float(band.sum()),
    }


def _band_growth(state: np.ndarray, x0: np.ndarray, band: np.ndarray) -> float:
    """New burned cells inside the band at the LAST lead. ``state`` is ``[L,H,W]``."""
    del x0
    return float((np.asarray(state)[-1] > _EVENT_UNBURNED)[band].sum())


def window_row(
    window: ForecastWindow,
    samples: np.ndarray | None,
    *,
    band_radius: int,
    fire_id: str,
    spatial_block_id: int,
) -> dict[str, Any]:
    """Covariates + realised growth + (optionally) the ensemble-mean growth.

    ``samples`` is ``uint8[M, L, H, W]`` straight from :meth:`predict`, i.e. the
    same array C6 scores, so the model-side response is measured on the same
    quantity the gate is. ``None`` gives the truth-side row alone, which needs no
    model at all - that half of this measurement is a property of the DATA.
    """
    x0 = np.asarray(window.x0)
    band = _band(x0, band_radius)
    cov = window_covariates(window, band_radius=band_radius)
    truth_growth = _band_growth(window.truth, x0, band)
    row: dict[str, Any] = {
        "fire_id": fire_id,
        "spatial_block_id": int(spatial_block_id),
        "t0": int(getattr(window, "t0", -1)),
        "truth_growth": truth_growth,
        "truth_growth_cells_domain": int(window.truth_growth_cells()),
        "model_growth": None,
        **cov,
    }
    if samples is not None:
        member = np.asarray(samples) > _EVENT_UNBURNED  # [M, L, H, W]
        row["model_growth"] = float(member[:, -1][:, band].sum(axis=1).mean())
    return row


def _design(rows: Sequence[Mapping[str, Any]], covariates: Sequence[str]) -> np.ndarray:
    return np.column_stack([[float(r[c]) for r in rows] for c in covariates])


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mean) / sd, mean, sd


def fit_response(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    *,
    covariates: Sequence[str] = COVARIATES,
) -> dict[str, Any]:
    """OLS of ``log(target / frontier cells)`` on STANDARDISED covariates.

    Slopes are per one standard deviation OF THIS SAMPLE's covariate, so a truth
    slope and a model slope fitted on the SAME rows are directly comparable - the
    standardisation is shared, which is the only reason the ratio between them
    means anything.

    Univariate slopes are returned beside the joint ones. The covariates are
    strongly collinear (RH and fuel moisture especially), so a joint coefficient
    can be small because another column ate the signal, and reporting only joint
    slopes would let "the model does not respond to wind" be an artifact of the
    design matrix rather than a property of the model.
    """
    usable = [
        r
        for r in rows
        if r.get(target) is not None
        and float(r[target]) > 0
        and float(r["_n_frontier_cells"]) > 0
        and all(np.isfinite(float(r[c])) for c in covariates)
    ]
    if len(usable) <= len(covariates) + 1:
        return {"target": target, "n": len(usable), "insufficient": True}

    y = np.array([math.log(float(r[target]) / float(r["_n_frontier_cells"])) for r in usable])
    x_raw = _design(usable, covariates)
    x, mean, sd = _standardise(x_raw)
    design = np.column_stack([np.ones(len(usable)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(len(usable) - design.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0.0, None))

    univariate: dict[str, dict[str, float]] = {}
    for k, name in enumerate(covariates):
        d = np.column_stack([np.ones(len(usable)), x[:, k]])
        b, *_ = np.linalg.lstsq(d, y, rcond=None)
        r = y - d @ b
        s2 = float(r @ r) / max(len(usable) - 2, 1)
        s = math.sqrt(max(float(np.linalg.pinv(d.T @ d)[1, 1]) * s2, 0.0))
        univariate[name] = {"slope": float(b[1]), "se": s}

    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "target": target,
        "n": len(usable),
        "intercept": float(beta[0]),
        "mean_log_rate": float(y.mean()),
        "sd_log_rate": float(y.std(ddof=1)),
        "r2": (1.0 - float(resid @ resid) / ss_tot) if ss_tot > 0 else None,
        "joint": {
            name: {"slope": float(beta[k + 1]), "se": float(se[k + 1])}
            for k, name in enumerate(covariates)
        },
        "univariate": univariate,
        "covariate_mean": {n: float(m) for n, m in zip(covariates, mean, strict=True)},
        "covariate_sd": {n: float(s) for n, s in zip(covariates, sd, strict=True)},
        "insufficient": False,
    }


def compare_responses(
    rows: Sequence[Mapping[str, Any]],
    *,
    covariates: Sequence[str] = COVARIATES,
    truth_key: str = "truth_growth",
    model_key: str = "model_growth",
) -> dict[str, Any]:
    """Truth's slopes, the model's slopes, and the compression between them.

    **Fitted on the SAME rows for both**: a row is used only if BOTH targets are
    positive, so the two regressions share a design matrix and a standardisation
    and the ratio of their slopes is a comparison rather than a coincidence. The
    row counts are printed to make that auditable.
    """
    shared = [
        r
        for r in rows
        if r.get(truth_key) is not None
        and r.get(model_key) is not None
        and float(r[truth_key]) > 0
        and float(r[model_key]) > 0
        and float(r["_n_frontier_cells"]) > 0
        and all(np.isfinite(float(r[c])) for c in covariates)
    ]
    truth = fit_response(shared, truth_key, covariates=covariates)
    model = fit_response(shared, model_key, covariates=covariates)
    if truth.get("insufficient") or model.get("insufficient"):
        return {"n": len(shared), "insufficient": True, "truth": truth, "model": model}

    compression: dict[str, Any] = {}
    for kind in ("joint", "univariate"):
        compression[kind] = {}
        for name in covariates:
            bt = truth[kind][name]["slope"]
            bm = model[kind][name]["slope"]
            st = truth[kind][name]["se"]
            sm = model[kind][name]["se"]
            compression[kind][name] = {
                "truth_slope": bt,
                "truth_se": st,
                "model_slope": bm,
                "model_se": sm,
                "difference": bm - bt,
                # THE UNITS ARE IN THE KEY. [maintainer ruling, 2026-08-09]
                # ADR-033 (7)'s separation denominator was ambiguous between SD
                # ACROSS SEEDS and SD ACROSS HELD-OUT BLOCKS and the ambiguity was
                # worth 3.7-6.6x; the ruling is BLOCK SD, and the fix that works is
                # naming the basis in the key rather than in a docstring.
                # **This one is NEITHER.** It is the within-sample OLS standard
                # error over WINDOWS, which overstates precision because windows
                # overlap in time and within a fire. The BLOCK-SD version of this
                # comparison is `runs/_m9_scaling.py`'s `z_paired_block_sd`, and
                # that is the one to quote. No quantity anywhere in M9 uses a SEED
                # SD denominator.
                "difference_se_window_ols": math.hypot(st, sm),
                "z_window_ols": (
                    (bm - bt) / math.hypot(st, sm) if math.hypot(st, sm) > 0 else None
                ),
                "uncertainty_basis": "within-sample OLS SE over WINDOWS (not blocks, not seeds)",
                # `None` rather than a huge number when truth's own slope is
                # inside its own noise: a compression RATIO against a slope that
                # is not distinguishable from zero is not a measurement, and
                # printing 12.4 there would be the compound-statistic error
                # ADR-035 (1) is a post-mortem of.
                "compression": (bm / bt) if abs(bt) > 2 * st and abs(bt) > 1e-9 else None,
                "truth_slope_resolved": bool(abs(bt) > 2 * st),
            }
    return {
        "n": len(shared),
        "n_rows_offered": len(rows),
        "insufficient": False,
        "truth": truth,
        "model": model,
        "compression": compression,
        "note": (
            "compression = model_slope / truth_slope, reported ONLY where truth's own slope "
            "exceeds 2 of its standard errors. Values near 0 mean the model does not respond; "
            "1 means it responds as truth does; negative means it responds the wrong way."
        ),
    }


def explain_block_deficit(
    rows: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    *,
    covariates: Sequence[str] = COVARIATES,
    truth_key: str = "truth_growth",
    model_key: str = "model_growth",
    kind: str = "joint",
) -> dict[str, Any]:
    """Does the SLOPE DIFFERENCE account for each block's log-rate deficit?

    This is the CZU question made arithmetic. If the model's response is
    compressed, its per-block log-rate deficit is, to first order::

        d_b  ==  (beta_model - beta_truth) . (xbar_b - xbar_all)   +   constant

    So the OBSERVED deficit per block is compared against the deficit PREDICTED
    from the fitted slope difference and that block's own covariate means. Blocks
    the compression explains land on the line; blocks it does not are the
    residual, and naming the residual is the point.

    A Mahalanobis distance would answer a different question and has already been
    asked and REFUTED (ADR-035 (3): blocks 5 and 6 are equally far from train at
    1.23/1.08/4.17/4.07 yet 2.2x apart in spread). This asks whether the fitted
    RESPONSE, not the distance, accounts for the block.
    """
    shared = [
        r
        for r in rows
        if r.get(truth_key) is not None
        and r.get(model_key) is not None
        and float(r[truth_key]) > 0
        and float(r[model_key]) > 0
        and float(r["_n_frontier_cells"]) > 0
        and all(np.isfinite(float(r[c])) for c in covariates)
    ]
    if comparison.get("insufficient") or not shared:
        return {"insufficient": True}

    sd = comparison["truth"]["covariate_sd"]
    grand = comparison["truth"]["covariate_mean"]
    delta = np.array(
        [
            comparison["compression"][kind][name]["model_slope"]
            - comparison["compression"][kind][name]["truth_slope"]
            for name in covariates
        ]
    )

    blocks = sorted({int(r["spatial_block_id"]) for r in shared})
    out: list[dict[str, Any]] = []
    for block in blocks:
        sub = [r for r in shared if int(r["spatial_block_id"]) == block]
        z = np.array(
            [
                (float(np.mean([float(r[name]) for r in sub])) - grand[name]) / sd[name]
                for name in covariates
            ]
        )
        log_truth = float(
            np.mean([math.log(float(r[truth_key]) / float(r["_n_frontier_cells"])) for r in sub])
        )
        log_model = float(
            np.mean([math.log(float(r[model_key]) / float(r["_n_frontier_cells"])) for r in sub])
        )
        out.append(
            {
                "spatial_block_id": block,
                "n_windows": len(sub),
                "mean_log_rate_truth": log_truth,
                "mean_log_rate_model": log_model,
                "observed_deficit": log_model - log_truth,
                "predicted_deficit_from_slopes": float(delta @ z),
                "covariate_z": {n: float(v) for n, v in zip(covariates, z, strict=True)},
            }
        )

    # The constant is not free: it is the sample-wide mean deficit, which the
    # slope difference cannot produce by construction (z is centred). Reporting
    # the residual WITHOUT removing it would credit the slopes with the level.
    level = float(np.mean([r["observed_deficit"] for r in out]))
    pred_level = float(np.mean([r["predicted_deficit_from_slopes"] for r in out]))
    for r in out:
        r["observed_deviation_from_level"] = r["observed_deficit"] - level
        r["predicted_deviation_from_level"] = r["predicted_deficit_from_slopes"] - pred_level
        r["residual"] = r["observed_deviation_from_level"] - r["predicted_deviation_from_level"]
        denom = abs(r["observed_deviation_from_level"])
        r["explained_fraction"] = None if denom < 1e-9 else 1.0 - abs(r["residual"]) / denom
    ss_obs = float(sum(r["observed_deviation_from_level"] ** 2 for r in out))
    ss_res = float(sum(r["residual"] ** 2 for r in out))
    return {
        "insufficient": False,
        "kind": kind,
        "mean_deficit_level": level,
        "blocks": out,
        "variance_explained_across_blocks": (None if ss_obs <= 0 else 1.0 - ss_res / ss_obs),
        "note": (
            "The LEVEL of the deficit is removed from both sides before the residual is "
            "taken: standardised covariate deviations are centred by construction, so a "
            "slope difference can only ever explain the SPREAD between blocks, never the "
            "overall level. Crediting it with the level would be the same compound-statistic "
            "error ADR-035 (1) is a post-mortem of."
        ),
    }
