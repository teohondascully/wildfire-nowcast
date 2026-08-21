"""Scoring a probability field directly — the M -> infinity limit of a C6 score.

C6's ``evaluate`` takes ``samples``, which is right: it is model-agnostic and it
is the only form ELMFIRE and the baselines have in common. But every M-member
ensemble pays a Monte-Carlo penalty on a quadratic score,

    E[Brier_M] = Brier_inf + E[p(1-p)] / M

and at M = 24 in the growth band that is a real fraction of the number being
compared (measured: it is ~0.0004 against a band Brier of ~0.012, i.e. ~3%).
The penalty is symmetric between models AT EQUAL M, so the C6 comparison stays
fair — but it is not zero, and a model whose forecast is a closed-form
probability should be able to say what its sampler cost it. That is all this
module is for. **It never replaces a C6 number in a gate**; it explains one.

Nothing here is a new metric definition: the band mask and the Brier primitive
are imported from :mod:`wildfire_nowcast.eval.masks` and
:mod:`wildfire_nowcast.eval.metrics`, so a mean-field score and a C6 score are
computed on the same cells with the same estimator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from wildfire_nowcast.common.contract import UNBURNED
from wildfire_nowcast.eval.masks import default_band_radius, growth_band
from wildfire_nowcast.eval.metrics import _reliability_summary as reliability_summary
from wildfire_nowcast.eval.metrics import brier, reliability

__all__ = ["mean_field_scores"]

_PROB_EPS = 1e-7


def mean_field_scores(
    model: Any,
    windows: Sequence[Any],
    *,
    horizon_h: int = 3,
    band_radius_cells: int | None = None,
) -> dict[str, Any]:
    """Band Brier / NLL / growth ratio of ``model.predict_proba`` over ``windows``.

    ``model`` must expose ``predict_proba(x0, static, weather, horizon_h) ->
    float[L,H,W]``. Baselines do not, and that is fine — this is a diagnostic
    for closed-form models, not a comparison table.
    """
    radius = (
        int(band_radius_cells) if band_radius_cells is not None else default_band_radius(horizon_h)
    )
    sse = nll_sum = n = 0.0
    pred_growth = obs_growth = 0.0
    bins: list[dict[str, float]] = []
    per_fire: dict[str, dict[str, float]] = {}
    for w in windows:
        prob = np.asarray(model.predict_proba(w.x0, w.static, w.weather, horizon_h))
        truth = np.asarray(w.truth[:horizon_h]) > UNBURNED
        mask = growth_band(w.x0, radius)
        if not mask.any():
            continue
        p = np.clip(prob[:, mask], _PROB_EPS, 1.0 - _PROB_EPS)
        y = truth[:, mask].astype(np.float64)
        s, count = brier(p, y)
        sse += s
        n += count
        nll_sum += float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).sum())
        raw = reliability(p, y)
        bins = (
            raw
            if not bins
            else [
                {
                    **b,
                    "n": b["n"] + r["n"],
                    "sum_p": b["sum_p"] + r["sum_p"],
                    "sum_y": b["sum_y"] + r["sum_y"],
                }
                for b, r in zip(bins, raw, strict=True)
            ]
        )
        unburned0 = np.asarray(w.x0) == UNBURNED
        pg = float(prob[-1][unburned0].sum())
        og = float(truth[-1][unburned0].sum())
        pred_growth += pg
        obs_growth += og
        entry = per_fire.setdefault(
            w.fire_id, {"predicted_new_cells": 0.0, "observed_new_cells": 0.0, "n_windows": 0.0}
        )
        entry["predicted_new_cells"] += pg
        entry["observed_new_cells"] += og
        entry["n_windows"] += 1
    for entry in per_fire.values():
        entry["growth_ratio"] = (
            entry["predicted_new_cells"] / entry["observed_new_cells"]
            if entry["observed_new_cells"] > 0
            else float("nan")
        )
    return {
        "estimator": "mean field (M -> infinity); NOT a C6 gate number",
        "band_brier": (sse / n) if n else None,
        "band_nll": (nll_sum / n) if n else None,
        "band_cells": n,
        "band_reliability": reliability_summary(bins) if bins else None,
        "predicted_new_cells": pred_growth,
        "observed_new_cells": obs_growth,
        "growth_ratio": (pred_growth / obs_growth) if obs_growth > 0 else None,
        "per_fire": per_fire,
        "n_windows": len(windows),
    }
