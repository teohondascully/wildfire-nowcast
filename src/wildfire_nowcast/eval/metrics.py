"""C6 — the metrics API. Model-agnostic: samples + truth, never model internals.

INTERFACES C6::

    evaluate(samples, truth: uint8[T,H,W]) -> dict with keys:
      brier_{1,2,3}h, reliability_bins (list), arrival_crps,
      dispersion_ratio, best_member_iou

Those six keys are always present and always mean the DOMAIN mask, fixed and
non-configurable, so no argument choice can change what a headline number means.
Everything else is additive and lives under ``by_mask`` / ``diagnostics``.

Three decisions in here are scientific, not clerical, and are argued in place
below because they will decide G2 and G3:

1. **The scoring mask.** See :mod:`wildfire_nowcast.eval.masks`. A domain-wide
   score on hourly GOFER steps is ~79% "nothing happened"; it is reported
   because it is incorruptible, not because it is informative.
2. **The CRPS estimator is FAIR (unbiased) by default.** The ordinary empirical
   CRPS estimator is biased in favour of UNDER-dispersed ensembles, which is
   precisely the failure mode G3 exists to detect. Using it would let an
   ensemble score better by collapsing.
3. **``dispersion_ratio`` on a binary field is a CALIBRATION statistic, not an
   independence detector, and it cannot see ensemble collapse.** This is
   algebraic, not empirical — see :func:`dispersion` — so C6 additionally
   reports ``area_dispersion_ratio`` and ``member_diversity``, which can.
   Anyone running the G3 independent-noise ablation must read those two, or the
   ablation will appear to pass.

Pooling
-------
``evaluate`` scores ONE window. Windows are pooled with :func:`aggregate`, which
combines sufficient statistics (sums and counts) rather than averaging per-window
scores — averaging Brier scores across windows with different pixel counts is a
different and wrong quantity, and it is the standard way a leave-fire-out table
ends up subtly incorrect.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from wildfire_nowcast.common.calibration import (
    GATE_CRITERION_KEY as CALIBRATION_GATE_KEY,
)
from wildfire_nowcast.common.calibration import (
    GATE_MASK as CALIBRATION_GATE_MASK,
)
from wildfire_nowcast.common.calibration import (
    frontier_rings,
    pool_strata,
    ring_strata,
    terms_from_strata,
)
from wildfire_nowcast.common.calibration import (
    terms_to_metric_dict as calibration_to_metric_dict,
)
from wildfire_nowcast.common.dispersion import growth_calibration
from wildfire_nowcast.common.iou_terms import (
    GATE_CRITERION_KEY,
    decompose_by_horizon,
    terms_to_metric_dict,
    truth_empty_by_lead,
)
from wildfire_nowcast.eval.masks import (
    DEFAULT_EVENT,
    default_band_radius,
    event_field,
    scoring_masks,
)

__all__ = [
    "evaluate",
    "aggregate",
    "brier",
    "reliability",
    "arrival_times",
    "crps_ensemble",
    "dispersion",
    "fuzzy_iou",
    "DEFAULT_LEADS",
    "DEFAULT_N_BINS",
]

DEFAULT_LEADS: tuple[int, ...] = (1, 2, 3)
DEFAULT_N_BINS = 10

_EPS = 1e-12


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def brier(p: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """``(mean squared error, n)`` — returns the SUFFICIENT STATISTICS, pooled form."""
    diff = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    n = int(diff.size)
    return (float(np.sum(diff * diff)), n)


def reliability(
    p: np.ndarray, y: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> list[dict[str, float]]:
    """Per-bin ``(n, sum_p, sum_y)`` for a reliability diagram.

    Sums rather than means, so bins from many windows and many fires pool
    exactly. With ``M`` members the forecast takes only ``M+1`` distinct values,
    so bin occupancy is lumpy by construction; ``n`` is carried on every bin so
    a diagram can show it and an empty bin is visibly empty rather than plotted
    as a point on the diagonal.
    """
    probs = np.asarray(p, dtype=np.float64).ravel()
    obs = np.asarray(y, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, int(n_bins) - 1)
    out: list[dict[str, float]] = []
    for b in range(int(n_bins)):
        sel = idx == b
        out.append(
            {
                "bin_index": b,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": int(np.count_nonzero(sel)),
                "sum_p": float(probs[sel].sum()),
                "sum_y": float(obs[sel].sum()),
            }
        )
    return out


def _finalise_bins(bins: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn pooled ``(n, sum_p, sum_y)`` into plottable means."""
    out = []
    for b in bins:
        n = int(b["n"])
        entry = dict(b)
        entry["mean_forecast"] = float(b["sum_p"] / n) if n else None
        entry["observed_frequency"] = float(b["sum_y"] / n) if n else None
        out.append(entry)
    return out


def _reliability_summary(bins: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Murphy's Brier decomposition ``BS = REL - RES + UNC``, plus ECE.

    ``REL`` (reliability, lower is better) is the calibration term G3 is about;
    ``RES`` (resolution) is the discrimination term. Reporting only Brier hides
    the trade: a model can improve Brier by becoming less sharp.
    """
    total = sum(int(b["n"]) for b in bins)
    if not total:
        return {"reliability": None, "resolution": None, "uncertainty": None, "ece": None}
    base = sum(float(b["sum_y"]) for b in bins) / total
    rel = res = ece = 0.0
    for b in bins:
        n = int(b["n"])
        if not n:
            continue
        p_bar = float(b["sum_p"]) / n
        y_bar = float(b["sum_y"]) / n
        rel += n * (p_bar - y_bar) ** 2
        res += n * (y_bar - base) ** 2
        ece += n * abs(p_bar - y_bar)
    return {
        "reliability": rel / total,
        "resolution": res / total,
        "uncertainty": base * (1.0 - base),
        "ece": ece / total,
        "base_rate": base,
    }


def arrival_times(event: np.ndarray, cap: int | None = None) -> np.ndarray:
    """First lead (1-based) at which ``event`` is true; ``cap`` if never.

    ``event`` is ``[..., L, H, W]``. Censoring at ``L + 1`` is not a convenience
    — insights/data item 3 (addendum) measures that GOFER's East and West
    variants disagree about when a fire ENDS by tens of hours (CZU: 68 steps vs
    174). An uncapped arrival-time target inherits that, and the tail would
    dominate CRPS at a magnitude far larger than the 1-3 h nowcast horizon this
    project is scored on. The cap makes the metric answer "did it arrive within
    the window", which is the question the labels can actually support.
    """
    arr = np.asarray(event, dtype=bool)
    n_lead = arr.shape[-3]
    ceiling = int(cap) if cap is not None else n_lead + 1
    any_hit = arr.any(axis=-3)
    first = np.argmax(arr, axis=-3) + 1
    return np.where(any_hit, first, ceiling).astype(np.float64)


def crps_ensemble(
    members: np.ndarray, observed: np.ndarray, *, fair: bool = True
) -> tuple[float, int]:
    """``(sum CRPS, n)`` for an ensemble of scalars per cell.

    ``members`` is ``[M, N]``, ``observed`` is ``[N]``. Energy form::

        CRPS = mean_i |x_i - y| - c * sum_ij |x_i - x_j|

    with ``c = 1/(2 M (M-1))`` for the FAIR estimator and ``1/(2 M^2)`` for the
    ordinary one. The ordinary estimator is biased low for an under-dispersed
    ensemble, i.e. it pays a model for collapsing — unacceptable when the same
    number is used to adjudicate G3. Fair is the default; ``fair=False`` exists
    only so the two can be compared.

    A deterministic predictor (M = 1, or all members identical) has a zero
    spread term either way, so this choice never moves a baseline — it only ever
    removes a reward the learned ensemble would otherwise get for collapsing.
    """
    x = np.asarray(members, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    n_members, n_cells = x.shape
    if n_cells == 0:
        return (0.0, 0)
    mae = np.abs(x - y[None, :]).mean(axis=0)
    if n_members < 2:
        return (float(mae.sum()), int(n_cells))
    # sum_ij |xi - xj| = 2 * sum_i (2i - M + 1) x_(i)  for x sorted ascending.
    ordered = np.sort(x, axis=0)
    weights = (2 * np.arange(n_members) - n_members + 1).astype(np.float64)[:, None]
    pair_sum = 2.0 * (weights * ordered).sum(axis=0)
    denom = 2.0 * n_members * (n_members - 1) if fair else 2.0 * n_members * n_members
    return (float(np.sum(mae - pair_sum / denom)), int(n_cells))


def dispersion(p: np.ndarray, y: np.ndarray, n_members: int) -> tuple[float, float, int]:
    """``(sum ensemble variance, sum squared error, n)`` for the spread-skill ratio.

    For a perfect ensemble of ``M`` members drawn from the predictive
    distribution, ``E[(mean - y)^2] = (1 + 1/M) sigma^2`` while the unbiased
    sample variance estimates ``sigma^2``, so the calibrated ratio is::

        dispersion_ratio = sqrt( var * (M+1)/M / mse ) = 1

    **The trap.** For BINARY members the sample variance is exactly
    ``p(1-p) M/(M-1)`` — mechanically determined by the ensemble mean. And
    ``E[(p-y)^2 | p] = p(1-p)`` exactly when the forecast is calibrated. So on a
    binary field this ratio is ALGEBRAICALLY a calibration statistic and equals
    1 for any calibrated forecast, however the members are correlated. It
    therefore CANNOT detect ensemble collapse.

    That matters because the G3 ablation — independent per-pixel noise — fails
    by producing members that are individually well-calibrated per pixel and
    nearly identical in every aggregate (independent noise averages out over
    thousands of pixels, so every member burns almost the same total area and
    the ensemble has no scenario spread at all). Read ``area_dispersion_ratio``
    and ``member_diversity`` for that. Recorded here rather than in a status
    file because whoever reads this number next needs to know it.
    """
    prob = np.asarray(p, dtype=np.float64)
    obs = np.asarray(y, dtype=np.float64)
    m = int(n_members)
    var = prob * (1.0 - prob) * (m / (m - 1.0)) if m > 1 else np.zeros_like(prob)
    err = (prob - obs) ** 2
    return (float(var.sum()), float(err.sum()), int(prob.size))


def fuzzy_iou(a: np.ndarray, b: np.ndarray, tolerance_cells: int = 0) -> float:
    """IoU, optionally counting a hit within ``tolerance_cells`` as a hit.

    At ``tolerance_cells = 0`` this is the ordinary Jaccard index. Above 0 it is
    the symmetric fuzzy form ``(|A & dil(B)| + |B & dil(A)|) / (2 |A | B|)``,
    which is bounded by 1 and reduces exactly to Jaccard at 0.

    A tolerance of 1 cell is the honest scale for these labels, not leniency:
    insights/data item 4 measures GOFER-East vs GOFER-West on the same fire at a
    mean symmetric difference of 78 cells on the 1 km grid and a mean centroid
    offset of 0.63 km. A metric that resolves single cells is measuring GOES
    viewing geometry. Both tolerances are reported so the gap between them is
    visible.
    """
    from wildfire_nowcast.common.states import dilate

    set_a = np.asarray(a, dtype=bool)
    set_b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(set_a | set_b))
    if union == 0:
        return 1.0  # both empty: perfect agreement, and 0/0 is not 0 here
    tol = int(tolerance_cells)
    if tol <= 0:
        return float(np.count_nonzero(set_a & set_b)) / union
    hit_a = np.count_nonzero(set_a & dilate(set_b, tol))
    hit_b = np.count_nonzero(set_b & dilate(set_a, tol))
    return float(hit_a + hit_b) / (2.0 * union)


# --------------------------------------------------------------------------
# C6 entry point
# --------------------------------------------------------------------------


def evaluate(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    event: str = DEFAULT_EVENT,
    leads: Sequence[int] = DEFAULT_LEADS,
    band_radius_cells: int | None = None,
    n_bins: int = DEFAULT_N_BINS,
    tolerance_cells: int = 1,
    crps_fair: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one window. ``samples`` ``[M,L,H,W]``, ``truth`` ``[T>=L,H,W]``.

    ``x0`` is optional and is DATA, not a model internal — supplying it unlocks
    the ``growth_band`` mask and the growth-restricted IoU, both of which are
    what a G2/G5 verdict should actually cite. Without it the dict says so
    instead of quietly reporting domain numbers under a band name.
    """
    pred = np.asarray(samples)
    if pred.ndim != 4:
        raise ValueError(f"samples must be [n_members, horizon, H, W], got {pred.shape}")
    n_members, n_lead, height, width = (int(v) for v in pred.shape)

    obs_all = np.asarray(truth)
    if obs_all.ndim != 3:
        raise ValueError(f"truth must be [T, H, W], got {obs_all.shape}")
    if obs_all.shape[0] < n_lead:
        raise ValueError(
            f"truth covers {obs_all.shape[0]} steps but samples cover {n_lead}; "
            "truth[k] must be the label for samples[:, k] (C1.3 end-of-hour phase)"
        )
    if obs_all.shape[1:] != (height, width):
        raise ValueError(f"truth grid {obs_all.shape[1:]} != samples grid {(height, width)}")

    member_event = event_field(pred, event)  # [M, L, H, W]
    truth_event = event_field(obs_all[:n_lead], event)  # [L, H, W]
    prob = member_event.mean(axis=0, dtype=np.float64)  # [L, H, W]

    masks = scoring_masks(x0, (height, width), n_lead, band_radius_cells=band_radius_cells)
    kept_leads = [int(k) for k in leads if 1 <= int(k) <= n_lead]
    ring_radius = (
        int(band_radius_cells) if band_radius_cells is not None else default_band_radius(n_lead)
    )

    by_mask: dict[str, Any] = {}
    pool: dict[str, Any] = {}
    for name, mask in masks.items():
        by_mask[name], pool[name] = _score_mask(
            member_event=member_event,
            truth_event=truth_event,
            prob=prob,
            mask=mask,
            x0=x0,
            n_members=n_members,
            n_lead=n_lead,
            leads=kept_leads,
            n_bins=n_bins,
            tolerance_cells=tolerance_cells,
            crps_fair=crps_fair,
            ring_radius=ring_radius,
        )

    primary = by_mask["domain"]
    result: dict[str, Any] = {
        # --- the six C6 keys. Always the DOMAIN mask; never reconfigurable. ---
        "brier_1h": primary["brier_by_lead"].get(1),
        "brier_2h": primary["brier_by_lead"].get(2),
        "brier_3h": primary["brier_by_lead"].get(3),
        "reliability_bins": primary["reliability_bins"],
        "arrival_crps": primary["arrival_crps"],
        "dispersion_ratio": primary["dispersion_ratio"],
        "best_member_iou": primary["best_member_iou"],
        # --- [v2.10] C6.4: the decomposition. REPORTED value above is unchanged.
        "best_member_iou_silence": primary["best_member_iou_silence"],
        "best_member_iou_shape": primary["best_member_iou_shape"],
        GATE_CRITERION_KEY: primary[GATE_CRITERION_KEY],
        "best_member_iou_silent_floor": primary["best_member_iou_silent_floor"],
        "best_member_iou_gate_criterion": GATE_CRITERION_KEY,
        # --- [ADR-020] G3's calibration criterion. Per lead, inside
        # `by_mask[<mask>]["reliability_summary"][<lead>]`; these two keys name
        # WHICH key and WHICH mask a gate reads, so a table cannot quote the
        # diluted domain value as the G3 number by accident.
        "calibration_gate_criterion": CALIBRATION_GATE_KEY,
        "calibration_gate_mask": CALIBRATION_GATE_MASK,
        # --- context ------------------------------------------------------
        "event": event,
        "n_members": n_members,
        "horizon_h": n_lead,
        "grid_shape": [height, width],
        "leads_scored": kept_leads,
        "primary_mask": "domain",
        "crps_estimator": "fair" if crps_fair else "biased",
        "tolerance_cells": int(tolerance_cells),
        "band_radius_cells": (
            int(band_radius_cells)
            if band_radius_cells is not None
            else default_band_radius(n_lead)
        ),
        "by_mask": by_mask,
        "diagnostics": _diagnostics(member_event, truth_event, x0, n_members),
        "notes": _notes(x0, n_members),
        "_pool": {
            "n_windows": 1,
            "event": event,
            "n_members": n_members,
            "leads": kept_leads,
            "by_mask": pool,
        },
    }
    if meta:
        result["meta"] = dict(meta)
    return result


def _score_mask(
    *,
    member_event: np.ndarray,
    truth_event: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    x0: np.ndarray | None,
    n_members: int,
    n_lead: int,
    leads: Sequence[int],
    n_bins: int,
    tolerance_cells: int,
    crps_fair: bool,
    ring_radius: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cells = int(np.count_nonzero(mask))
    empty = cells == 0

    # [ADR-020] G3's calibration criterion needs a partition of the scored set
    # that the FORECAST did not choose, or climatology satisfies it trivially.
    # The rings come from `x0` alone — same provenance as the growth_band mask.
    rings = None if x0 is None else frontier_rings(x0, ring_radius)
    n_rings = int(ring_radius) + 2

    brier_by_lead: dict[int, float | None] = {}
    brier_pool: dict[str, dict[str, float]] = {}
    bins_out: list[dict[str, Any]] = []
    bins_pool: dict[str, list[dict[str, float]]] = {}
    rings_pool: dict[str, list[dict[str, float]]] = {}
    rel_summary: dict[str, Any] = {}
    cal_block = terms_from_strata(None, None).block_dict()
    for lead in leads:
        p_lead = prob[lead - 1][mask]
        y_lead = truth_event[lead - 1][mask].astype(np.float64)
        sse, n = brier(p_lead, y_lead)
        brier_by_lead[lead] = (sse / n) if n else None
        brier_pool[str(lead)] = {"sse": sse, "n": n}
        raw_bins = reliability(p_lead, y_lead, n_bins)
        bins_pool[str(lead)] = raw_bins
        raw_rings = (
            None
            if rings is None
            else ring_strata(prob[lead - 1], truth_event[lead - 1], mask, rings, n_rings)
        )
        rings_pool[str(lead)] = [s.as_dict() for s in (raw_rings or [])]
        cal = terms_from_strata(raw_bins, raw_rings)
        cal_block = cal.block_dict()
        rel_summary[str(lead)] = {
            **_reliability_summary(raw_bins),
            **calibration_to_metric_dict(cal),
        }
        for entry in _finalise_bins(raw_bins):
            bins_out.append({"lead_h": lead, **entry})

    # --- arrival-time CRPS -------------------------------------------------
    flat = mask.ravel()
    truth_arrival = arrival_times(truth_event).ravel()[flat]
    member_arrival = arrival_times(member_event).reshape(n_members, -1)[:, flat]
    crps_sum, crps_n = crps_ensemble(member_arrival, truth_arrival, fair=crps_fair)
    # "Active" = cells where the truth OR any member arrived inside the window.
    # Cells nobody claims contribute exactly 0 to the sum, so the domain-mean
    # CRPS scales with domain size and is comparable only between models on the
    # SAME grid; this one is comparable across fires.
    ceiling = float(n_lead + 1)
    active = (truth_arrival < ceiling) | (member_arrival < ceiling).any(axis=0)
    crps_active_sum, crps_active_n = crps_ensemble(
        member_arrival[:, active], truth_arrival[active], fair=crps_fair
    )

    # --- dispersion --------------------------------------------------------
    disp_var, disp_err, disp_n = dispersion(prob[:, mask], truth_event[:, mask], n_members)
    area_var, area_err, area_n, area_signed = _area_dispersion(
        member_event, truth_event, mask, n_members
    )
    # [M9] SEPARATE FROM `_area_dispersion` ON PURPOSE, and it is not a style
    # choice. `tests/test_playthrough_dispersion.py` (infra's) plants
    # mutations by REPLACING `_area_dispersion` wholesale, so widening its return
    # arity silently breaks somebody else's mutation coverage — which it did, and
    # that suite caught it in the same session. A new quantity gets a new
    # function; an existing mutation surface stays exactly the shape its owner
    # declared it to be.
    area_truth = _truth_area_sum(truth_event, mask)

    # --- mode capture ------------------------------------------------------
    iou = _iou_block(member_event, truth_event, mask, x0, tolerance_cells)

    factor = (n_members + 1.0) / n_members
    metrics: dict[str, Any] = {
        "n_cells": cells,
        "brier_by_lead": brier_by_lead,
        "reliability_bins": bins_out,
        "reliability_summary": rel_summary,
        "arrival_crps": (crps_sum / crps_n) if crps_n else None,
        "arrival_crps_active": (crps_active_sum / crps_active_n) if crps_active_n else None,
        "arrival_crps_active_cells": crps_active_n,
        "arrival_crps_censor_cap_h": n_lead + 1,
        "dispersion_ratio": _ratio(disp_var * factor, disp_err),
        "area_dispersion_ratio": _ratio(area_var * factor, area_err),
        **_first_moment_block(sum_pred=area_truth + area_signed, sum_truth=area_truth, n=area_n),
        **cal_block,
        **iou,
    }
    pool = {
        "n_cells": cells,
        "brier": brier_pool,
        "reliability": bins_pool,
        # [ADR-020] frontier-ring sufficient statistics. Pooled the same way the
        # reliability bins are: sum (n, sum_p, sum_y) across windows FIRST, then
        # take the deviation once. The other order would average a per-window
        # |mean p - mean y| whose small strata are noise-dominated even for a
        # perfect forecast, i.e. it would report an error floor that is an
        # artifact of the window size.
        "calibration_rings": rings_pool,
        "crps": {"sum": crps_sum, "n": crps_n},
        "crps_active": {"sum": crps_active_sum, "n": crps_active_n},
        "dispersion": {"sum_var": disp_var, "sum_sq_err": disp_err, "n": disp_n},
        # `sum_signed` is ADDITIVE and DIAGNOSTIC (M5). The denominator of
        # `area_dispersion_ratio` is the squared error of the ensemble-mean area,
        # which a systematically over-predicting forecast fills with BIAS. Without
        # the signed sum a reader cannot tell "the ensemble is too narrow" from
        # "the ensemble is in the wrong place", and those have opposite remedies.
        "area_dispersion": {
            "sum_var": area_var,
            "sum_sq_err": area_err,
            "n": area_n,
            "sum_signed": area_signed,
            # [M9] The FIRST MOMENT's own denominator. `sum_signed` already carries
            # `sum(mean_area) - sum(truth_area)`, so one more additive sum makes
            # `growth_calibration` poolable exactly, at every level, rather than
            # being reconstructable only by recomputing truth from the tensors —
            # which is how it was obtained until now (`sim/s5_report.py`), OUTSIDE
            # the artifact the gate is read from. ADR-039 (5) makes this quantity a
            # GATE CONDITION, and a gate condition that is not in `results.json`
            # is the `_headline` allow-list defect for the third time.
            "sum_truth": area_truth,
        },
        "iou": {
            "best_member_sum": _nan_to_zero(iou["best_member_iou"]),
            "mean_member_sum": _nan_to_zero(iou["mean_member_iou"]),
            "best_member_tol_sum": _nan_to_zero(iou["best_member_iou_tolerant"]),
            "best_member_growth_sum": _nan_to_zero(iou["best_member_iou_growth"]),
            "best_member_by_horizon_sum": list(iou["best_member_iou_by_horizon"]),
            "n_growth": 0 if iou["best_member_iou_growth"] is None else 1,
            "n": 1,
            # [v2.10] C6.4. silence/shape pool with the SAME unweighted-mean-over-
            # windows convention as best_member_iou, so the pooled numbers still
            # satisfy silence + shape == best_member_iou exactly. The gate
            # criterion pools only over windows where it is DEFINED, so its
            # denominator is carried separately and per horizon — a window with no
            # growth contributes nothing rather than contributing a zero.
            "silence_by_horizon_sum": list(iou["best_member_iou_silence_by_horizon"]),
            "shape_by_horizon_sum": list(iou["best_member_iou_shape_by_horizon"]),
            "silent_floor_by_horizon_sum": list(
                iou["best_member_iou_silent_floor_by_horizon"]
            ),
            "shape_masked_by_horizon_sum": [
                0.0 if v is None else float(v)
                for v in iou[f"{GATE_CRITERION_KEY}_by_horizon"]
            ],
            "n_shape_masked_by_horizon": [
                0 if v is None else 1 for v in iou[f"{GATE_CRITERION_KEY}_by_horizon"]
            ],
        },
        "empty": empty,
    }
    return metrics, pool


def _area_error_decomposition(
    *, sum_var: float, sum_sq_err: float, sum_signed: float, n: int
) -> dict[str, Any]:
    """[M5] Split ``area_dispersion_ratio``'s DENOMINATOR into bias and scatter.

    **NONE OF THESE KEYS IS A GATE CRITERION AND NONE MAY BE SUBSTITUTED FOR
    ONE.** G3's dispersion half is adjudicated on ``area_dispersion_ratio``
    (C6.1 / ADR-011), full stop. What this adds is an explanation of a failure,
    never a replacement for the verdict — and it is written down here because it
    is emitted in the same milestone that scores that gate, which is exactly when
    a flattering variant would be easiest to slide in.

    The denominator ``sum (mean_area - truth_area)^2`` decomposes as
    ``n * bias^2 + n * scatter^2``. A forecaster that over-predicts growth by 3x
    fills it with ``bias``, and its dispersion ratio is then low for a reason
    that has nothing to do with how wide its ensemble is. Our own kernel
    over-predicts held-out growth 2.66-3.06x (ADR-021 (3b)), so this is the
    measurement that says whether a low ratio means "too narrow" or "in the wrong
    place" — which have opposite remedies.
    """
    note = (
        "DIAGNOSTIC ONLY. `area_dispersion_ratio` is the G3 criterion (C6.1/ADR-011); "
        "`ratio_debiased` divides by the error a PERFECTLY UNBIASED forecaster of the same "
        "sharpness would have, and exists to attribute a failure, never to replace the "
        "verdict. IT IS IN THE SAME (SD) UNITS AS THE CRITERION — see the M8 fix below; "
        "it was in VARIANCE units until 2026-08-09 and every value printed before then is "
        "the SQUARE of the right one."
    )
    if not n or sum_sq_err <= 0:
        inner: dict[str, Any] = {
            "bias": None,
            "scatter": None,
            "bias_fraction": None,
            "ratio_debiased": None,
            "note": "undefined: no scored leads or zero error",
        }
    else:
        bias = sum_signed / n
        bias_ss = n * bias * bias
        scatter_ss = max(sum_sq_err - bias_ss, 0.0)
        inner = {
            "bias": bias,
            "scatter": float(np.sqrt(scatter_ss / n)),
            "bias_fraction": bias_ss / sum_sq_err,
            # [M8 FIX] **THE SQUARE ROOT WAS MISSING AND IT REVERSED CONCLUSIONS.**
            # `area_dispersion_ratio` goes through `_ratio`, which takes a sqrt, so
            # the CRITERION is in SD units. This companion divided two sums of
            # squares and was therefore in VARIANCE units — the SQUARE of the
            # quantity it sits next to. For every value below 1 (which is all of
            # ours) squaring moves it DOWN, so a debiased ratio that is genuinely
            # ABOVE the raw one was printed BELOW it and the attribution read
            # backwards. Measured on `m6_fair_brier0_s1`: the record said
            # 0.5799 -> 0.5081 where the truth is 0.5799 -> **0.7128**, and
            # sqrt(0.5081) = 0.7128 exactly. Raised by sim (S5) against my
            # file, on 155 of 170 cells reversing direction.
            # `sum_var` already carries the finite-ensemble `factor` at both call
            # sites, so this is the sqrt and NOTHING else.
            "ratio_debiased": (
                float(np.sqrt(max(sum_var, 0.0) / scatter_ss)) if scatter_ss > 0 else None
            ),
            "note": note,
        }
    # NESTED under one non-numeric key ON PURPOSE. `common/null_check._flatten`
    # takes every numeric value in a `by_mask` block as a METRIC, and
    # `C6_METRICS` refuses to skip one it does not know — an unregistered metric
    # is an unchecked metric (C-2, one level down). That registry lives in
    # `common/`, which C-4 FREEZES while a lead is running, and these four
    # numbers are a decomposition of an existing criterion rather than four new
    # criteria. Nesting them says exactly that and keeps the null check's
    # namespace honest. If they are ever to be RANKED they must be registered
    # first, and that is an infra edit and an maintainer ruling, not mine.
    return {"area_error_decomposition": inner}


def _first_moment_block(*, sum_pred: float, sum_truth: float, n: int) -> dict[str, Any]:
    """[M9] G3's FIRST-MOMENT condition input: ``growth_calibration``.

    ADR-039 (5) makes ``|log(growth_calibration)|`` a G3 CONDITION, measured
    against the wind ellipse's on the same held-out blocks. Until this function
    the quantity existed nowhere in ``eval/`` — only in ``sim/s5_report.py``,
    where simviz recomputes truth areas from the C1 tensors. A gate condition
    that cannot be read out of ``results.json`` is the ``_headline`` allow-list
    defect a third time (C6.4's shape term, ADR-020's ``calibration_error``), so
    it is emitted here, from the same sufficient statistics the dispersion
    criterion already pools.

    The arithmetic is deliberately the RATIO OF SUMS, not the mean of per-window
    ratios: a per-window ratio is undefined on every dormant window (truth 0) and
    the mean of the defined ones would silently be a growth-window-only estimate
    wearing an all-window label. That is the stratum confusion this whole
    milestone is about.

    **NESTED UNDER ONE NON-NUMERIC KEY ON PURPOSE**, exactly as
    :func:`_area_error_decomposition` is. ``common/null_check._flatten`` treats
    every numeric value in a ``by_mask`` block as a metric and ``C6_METRICS``
    refuses to skip one it does not know; that registry lives in ``common/``,
    which C-4 freezes to me. Nesting keeps the null check's namespace honest
    without a cross-boundary write. Ranking this key would require registering it
    first, which is an infra edit and an maintainer ruling, not mine.
    """
    return {
        "first_moment": {
            "growth_calibration": growth_calibration(sum_pred, sum_truth),
            "pred_area_sum": float(sum_pred),
            "truth_area_sum": float(sum_truth),
            "n_scored_leads": int(n),
            "note": (
                "growth_calibration = sum(ensemble-mean event area) / sum(truth event area) "
                "over scored leads. 1.0 is perfect; >1 over-predicts. RATIO OF SUMS, so it is "
                "defined on a stratum containing dormant windows (per-window ratios are not). "
                "UNDEFINED (null) when truth grew nothing — the correct answer for a dormant "
                "stratum is that a calibration ratio cannot be measured, not 0 and not inf. "
                "Implementation is common.dispersion.growth_calibration (C0), never a local copy."
            ),
        }
    }


def _truth_area_sum(truth_event: np.ndarray, mask: np.ndarray) -> float:
    """[M9] Total TRUTH event area over the scored leads — the first moment's denominator.

    The one quantity in this module that depends on NO model, which is why it is
    the honest denominator for a calibration ratio and why it has its own
    function rather than riding along with the dispersion statistics.
    """
    return float(truth_event[:, mask].sum())


def _area_dispersion(
    member_event: np.ndarray,
    truth_event: np.ndarray,
    mask: np.ndarray,
    n_members: int,
) -> tuple[float, float, int, float]:
    """Spread-skill on TOTAL BURNED AREA — the collapse detector.

    Independent per-pixel noise averages out: thousands of independent Bernoulli
    pixels give a total area whose spread is O(sqrt(N)) and vanishes relative to
    the scenario spread a real fire has. So an ensemble can be perfectly
    calibrated per pixel and have essentially zero area spread. A shared latent
    ``z_t`` is what puts the spread back. This ratio is where the difference
    shows up, and it is the number the G3 ablation must be judged on.
    """
    areas = member_event[:, :, mask].sum(axis=2).astype(np.float64)  # [M, L]
    truth_area = truth_event[:, mask].sum(axis=1).astype(np.float64)  # [L]
    mean_area = areas.mean(axis=0)
    var = areas.var(axis=0, ddof=1) if n_members > 1 else np.zeros_like(mean_area)
    err = (mean_area - truth_area) ** 2
    signed = mean_area - truth_area
    return float(var.sum()), float(err.sum()), int(mean_area.size), float(signed.sum())


def _iou_block(
    member_event: np.ndarray,
    truth_event: np.ndarray,
    mask: np.ndarray,
    x0: np.ndarray | None,
    tolerance_cells: int,
) -> dict[str, Any]:
    """Best-member mode-capture IoU, plus the versions that are not inflated.

    ``best_member_iou`` is ``max_m mean_k IoU(member m at lead k, truth at lead
    k)`` — the best whole TRAJECTORY, which is what "mode capture" means. The
    per-lead maximum (also reported) lets a different member win at every lead
    and is therefore optimistic; both are here so nobody has to guess which was
    computed.

    ``best_member_iou_growth`` restricts both sets to cells UNBURNED at ``t0``.
    Fire is absorbing, so the full-set IoU is dominated by the already-burned
    region every model reproduces for free: a fire that does not grow at all
    scores ~1.0 on the unrestricted IoU.

    **This docstring used to call ``best_member_iou_growth`` "the number that
    means something". THAT SENTENCE IS RETIRED (ADR-022 (4), my own proposal
    against my own words).** It was wrong twice over and both are measured, not
    argued. infra's null check scores it at 0.3333 for a do-nothing
    forecast against 0.3124 for genuine skill; and ADR-023's ZERO-CAPTURE AXIOM
    then showed WHY, for the whole best-of-M family at once — a
    higher-is-better metric must pay a forecast that claims nothing the MINIMUM
    of its range, and this one pays 1/3. The restriction to unburned cells
    removes the absorbing-region inflation but NOT the empty-vs-empty
    convention, so it inherits the pathology whole. **``best_member_iou``,
    ``best_member_iou_growth``, ``best_member_iou_tolerant`` and
    ``mean_member_iou`` may be REPORTED and may never be quoted as capability.**
    Anything adjudicating a gate uses ``best_member_iou_shape_masked``, whose
    null floor is exactly 0.

    **[v2.10 / C6.4] ``best_member_iou`` IS REPORTED AND NEVER A GATE CRITERION**
    (ADR-017). It ranks doing nothing above doing something: empty-vs-empty is
    IoU 1.0, so a member that predicts nothing banks a full point on every lead
    where truth did not grow. The decomposition below is computed by
    ``common.iou_terms`` — the single, contract-adjudicated implementation (C0) —
    and the gate criterion is ``common.iou_terms.GATE_CRITERION_KEY``. The
    undecomposed value is left BIT-IDENTICAL and still emitted, because hiding it
    would hide the pathology.
    """
    n_members, n_lead = member_event.shape[0], member_event.shape[1]
    per_member_lead = np.empty((n_members, n_lead), dtype=np.float64)
    per_member_lead_tol = np.empty_like(per_member_lead)
    for m in range(n_members):
        for k in range(n_lead):
            pred = member_event[m, k] & mask
            obs = truth_event[k] & mask
            per_member_lead[m, k] = fuzzy_iou(pred, obs, 0)
            per_member_lead_tol[m, k] = fuzzy_iou(pred, obs, tolerance_cells)

    trajectory = per_member_lead.mean(axis=1)
    # Best-member mode capture AT EACH HORIZON, not only at the last lead.
    # ADR-015 (3) requires the model to be adjudicated against the ellipse's
    # own best-calibrated form AT THAT HORIZON, so the metric has to exist at
    # every horizon too. Entry H-1 is `max_m mean_{k<=H} IoU`, which is exactly
    # what scoring a length-H window would have produced — one pass instead of
    # three, and identical by construction rather than by hope. The last entry
    # equals `best_member_iou`, which is asserted in the self-test.
    cumulative = np.cumsum(per_member_lead, axis=1) / np.arange(1, n_lead + 1)[None, :]
    cumulative_best = cumulative.max(axis=0)
    growth: float | None = None
    if x0 is not None:
        unburned0 = (np.asarray(x0) == 0) & mask
        scores = np.empty((n_members, n_lead), dtype=np.float64)
        for m in range(n_members):
            for k in range(n_lead):
                scores[m, k] = fuzzy_iou(
                    member_event[m, k] & unburned0, truth_event[k] & unburned0, 0
                )
        growth = float(scores.mean(axis=1).max())

    # [v2.10] C6.4 — shape / silence split, in common/ (C0), model-blind.
    truth_empty = truth_empty_by_lead(truth_event, mask)
    terms = decompose_by_horizon(per_member_lead, truth_empty)

    return {
        "best_member_iou": float(trajectory.max()),
        "mean_member_iou": float(trajectory.mean()),
        "best_member_iou_by_lead": [float(v) for v in per_member_lead.max(axis=0)],
        "best_member_iou_by_horizon": [float(v) for v in cumulative_best],
        "best_member_iou_tolerant": float(per_member_lead_tol.mean(axis=1).max()),
        "best_member_iou_growth": growth,
        **terms_to_metric_dict(terms),
    }


def _diagnostics(
    member_event: np.ndarray,
    truth_event: np.ndarray,
    x0: np.ndarray | None,
    n_members: int,
) -> dict[str, Any]:
    """Numbers that explain a score rather than being one."""
    n_lead = member_event.shape[1]
    truth_final = truth_event[-1]
    burned0 = (np.asarray(x0) > 0) if x0 is not None else None
    truth_growth = (
        int(np.count_nonzero(truth_final & ~burned0)) if burned0 is not None else None
    )
    member_growth = (
        [int(v) for v in np.count_nonzero(member_event[:, -1] & ~burned0[None], axis=(1, 2))]
        if burned0 is not None
        else None
    )

    # Member diversity: mean pairwise IoU at the final lead. 1.0 means every
    # member is the same fire, i.e. a collapsed ensemble wearing M hats.
    diversity: float | None = None
    if 2 <= n_members <= 64:
        pairs = []
        for i in range(n_members):
            for j in range(i + 1, n_members):
                pairs.append(fuzzy_iou(member_event[i, -1], member_event[j, -1], 0))
        diversity = float(np.mean(pairs)) if pairs else None

    distinct = len({member_event[m].tobytes() for m in range(n_members)})
    return {
        "truth_growth_cells": truth_growth,
        "member_growth_cells": member_growth,
        "is_zero_growth_window": None if truth_growth is None else truth_growth == 0,
        "mean_pairwise_member_iou": diversity,
        "n_distinct_members": distinct,
        "truth_burned_cells_final": int(np.count_nonzero(truth_final)),
        "horizon_h": n_lead,
    }


def _notes(x0: np.ndarray | None, n_members: int) -> list[str]:
    notes = [
        "Headline keys are the DOMAIN mask: mostly far-field cells no model was ever "
        "uncertain about. Cite by_mask['growth_band'] alongside them, never instead.",
        "dispersion_ratio on a binary field is algebraically a CALIBRATION statistic and "
        "cannot detect ensemble collapse; read area_dispersion_ratio and "
        "diagnostics.mean_pairwise_member_iou for that.",
        "[v2.10 / C6.4] best_member_iou is REPORTED and NEVER a gate criterion (ADR-017): "
        "empty-vs-empty is IoU 1.0, so predicting nothing is bankable. Gate on "
        f"{GATE_CRITERION_KEY}. best_member_iou_silent_floor is what a forecast that "
        "predicts NOTHING scores on this window — read it beside every IoU number.",
        "The IoU decomposition is ~inert under the DOMAIN mask (truth is essentially never "
        "empty there) and carries the whole pathology under by_mask['growth_band']. Quote "
        "the band terms.",
    ]
    if x0 is None:
        notes.append(
            "x0 was not supplied: growth_band mask, growth-restricted IoU and the "
            "zero-growth flag are UNAVAILABLE, not zero."
        )
    if n_members < 2:
        notes.append(
            "n_members < 2: dispersion and CRPS spread terms are identically 0. This is a "
            "deterministic forecast scored with probabilistic tools, not a calibrated one."
        )
    return notes


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= _EPS:
        return None
    return float(np.sqrt(max(numerator, 0.0) / denominator))


def _nan_to_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)


# --------------------------------------------------------------------------
# pooling across windows / fires
# --------------------------------------------------------------------------


def aggregate(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool per-window :func:`evaluate` dicts into one score set.

    Combines SUFFICIENT STATISTICS, not per-window scores. Averaging per-window
    Brier scores weights a window with 12 disputed cells the same as one with
    1,200, which is a different quantity from the pooled Brier and is the usual
    way a leave-fire-out table ends up quietly wrong.

    Note on what this may and may not be used for: pooling across FIRES is only
    meaningful within one fold, and CV spread must be quoted against **11
    effective blocks, not 28 fires** (C3.1 / ADR-006 P4) — buffered domains
    overlap, so two fires in one block are one sample. Quoting n=28 overstates
    precision by ~sqrt(28/11).
    """
    items = [dict(r) for r in results]
    if not items:
        raise ValueError("aggregate needs at least one evaluate() result")
    pools = [r["_pool"] for r in items]
    events = {p["event"] for p in pools}
    if len(events) > 1:
        raise ValueError(f"cannot pool results scored on different events: {sorted(events)}")
    member_counts = sorted({int(p["n_members"]) for p in pools})
    n_members = member_counts[0]

    mask_names = sorted({name for p in pools for name in p["by_mask"]})
    by_mask: dict[str, Any] = {}
    for name in mask_names:
        blocks = [p["by_mask"][name] for p in pools if name in p["by_mask"]]
        by_mask[name] = _pool_mask(blocks, n_members)

    primary = by_mask.get("domain", next(iter(by_mask.values())))
    out: dict[str, Any] = {
        "brier_1h": primary["brier_by_lead"].get(1),
        "brier_2h": primary["brier_by_lead"].get(2),
        "brier_3h": primary["brier_by_lead"].get(3),
        "reliability_bins": primary["reliability_bins"],
        "arrival_crps": primary["arrival_crps"],
        "dispersion_ratio": primary["dispersion_ratio"],
        "best_member_iou": primary["best_member_iou"],
        # --- [v2.10] C6.4 --------------------------------------------------
        "best_member_iou_silence": primary.get("best_member_iou_silence"),
        "best_member_iou_shape": primary.get("best_member_iou_shape"),
        GATE_CRITERION_KEY: primary.get(GATE_CRITERION_KEY),
        "best_member_iou_silent_floor": primary.get("best_member_iou_silent_floor"),
        "best_member_iou_gate_criterion": GATE_CRITERION_KEY,
        # --- [ADR-020] G3's calibration criterion; see evaluate(). -----------
        "calibration_gate_criterion": CALIBRATION_GATE_KEY,
        "calibration_gate_mask": CALIBRATION_GATE_MASK,
        "event": sorted(events)[0],
        "n_windows": len(items),
        "n_members": member_counts if len(member_counts) > 1 else n_members,
        "primary_mask": "domain",
        "by_mask": by_mask,
        "diagnostics": _pool_diagnostics(items),
        "notes": items[0].get("notes", []),
    }
    return out


def _pool_mask(blocks: Sequence[Mapping[str, Any]], n_members: int) -> dict[str, Any]:
    leads = sorted({int(k) for b in blocks for k in b["brier"]})
    brier_by_lead: dict[int, float | None] = {}
    bins_out: list[dict[str, Any]] = []
    rel_summary: dict[str, Any] = {}
    cal_block = terms_from_strata(None, None).block_dict()
    for lead in leads:
        sse = sum(float(b["brier"][str(lead)]["sse"]) for b in blocks if str(lead) in b["brier"])
        n = sum(int(b["brier"][str(lead)]["n"]) for b in blocks if str(lead) in b["brier"])
        brier_by_lead[lead] = (sse / n) if n else None
        merged = _merge_bins(
            [b["reliability"][str(lead)] for b in blocks if str(lead) in b["reliability"]]
        )
        merged_rings = pool_strata(
            [
                b.get("calibration_rings", {}).get(str(lead), [])
                for b in blocks
                if b.get("calibration_rings", {}).get(str(lead))
            ]
        )
        pooled_cal = terms_from_strata(merged, merged_rings or None)
        cal_block = pooled_cal.block_dict()
        rel_summary[str(lead)] = {
            **_reliability_summary(merged),
            **calibration_to_metric_dict(pooled_cal),
        }
        for entry in _finalise_bins(merged):
            bins_out.append({"lead_h": lead, **entry})

    def _sum(path: str, key: str) -> float:
        return float(sum(float(b[path][key]) for b in blocks))

    crps_n = _sum("crps", "n")
    crps_active_n = _sum("crps_active", "n")
    iou_n = float(sum(int(b["iou"]["n"]) for b in blocks))
    growth_n = float(sum(int(b["iou"]["n_growth"]) for b in blocks))
    factor = (n_members + 1.0) / n_members
    return {
        "n_cells": int(sum(int(b["n_cells"]) for b in blocks)),
        "brier_by_lead": brier_by_lead,
        "reliability_bins": bins_out,
        "reliability_summary": rel_summary,
        "arrival_crps": (_sum("crps", "sum") / crps_n) if crps_n else None,
        "arrival_crps_active": (
            (_sum("crps_active", "sum") / crps_active_n) if crps_active_n else None
        ),
        "arrival_crps_active_cells": int(crps_active_n),
        "dispersion_ratio": _ratio(
            _sum("dispersion", "sum_var") * factor, _sum("dispersion", "sum_sq_err")
        ),
        "area_dispersion_ratio": _ratio(
            _sum("area_dispersion", "sum_var") * factor, _sum("area_dispersion", "sum_sq_err")
        ),
        **_area_error_decomposition(
            sum_var=_sum("area_dispersion", "sum_var") * factor,
            sum_sq_err=_sum("area_dispersion", "sum_sq_err"),
            sum_signed=_sum("area_dispersion", "sum_signed"),
            n=int(sum(int(b["area_dispersion"]["n"]) for b in blocks)),
        ),
        # [M9] Poolable EXACTLY, because both terms are sums. A pooled
        # growth_calibration is therefore the same number a single call to
        # `evaluate` over the same windows would give — verified by the
        # first-moment playthrough, which is the property a "pooled" ratio
        # usually does NOT have.
        **_first_moment_block(
            sum_pred=_sum("area_dispersion", "sum_truth")
            + _sum("area_dispersion", "sum_signed"),
            sum_truth=_sum("area_dispersion", "sum_truth"),
            n=int(sum(int(b["area_dispersion"]["n"]) for b in blocks)),
        ),
        # IoU is a per-window max; it has no pooled sufficient statistic, so this
        # is an unweighted mean over windows and is labelled as such.
        "best_member_iou": (
            float(sum(float(b["iou"]["best_member_sum"]) for b in blocks) / iou_n)
            if iou_n
            else None
        ),
        "mean_member_iou": (
            float(sum(float(b["iou"]["mean_member_sum"]) for b in blocks) / iou_n)
            if iou_n
            else None
        ),
        "best_member_iou_tolerant": (
            float(sum(float(b["iou"]["best_member_tol_sum"]) for b in blocks) / iou_n)
            if iou_n
            else None
        ),
        "best_member_iou_growth": (
            float(sum(float(b["iou"]["best_member_growth_sum"]) for b in blocks) / growth_n)
            if growth_n
            else None
        ),
        "best_member_iou_by_horizon": (
            [
                float(v / iou_n)
                for v in np.sum(
                    [np.asarray(b["iou"]["best_member_by_horizon_sum"], float) for b in blocks],
                    axis=0,
                )
            ]
            if iou_n
            else None
        ),
        "iou_pooling": "unweighted mean over windows (a per-window max has no pooled form)",
        "n_windows": int(len(blocks)),
        **cal_block,
        **_pool_iou_terms(blocks, iou_n),
    }


def _pool_iou_terms(blocks: Sequence[Mapping[str, Any]], iou_n: float) -> dict[str, Any]:
    """[v2.10] C6.4 — pool the shape/silence split across windows.

    Two different denominators, deliberately:

    * ``silence`` / ``shape`` use the window count, exactly like
      ``best_member_iou``, so ``silence + shape == best_member_iou`` survives
      pooling and the REPORTED value stays auditable against its own parts.
    * the gate criterion uses the count of windows where it is DEFINED. Windows
      with no truth growth have no shape to capture; averaging a 0 in for them
      would reintroduce a label statistic into the score, which is the whole bug.
      The denominator is published as ``..._n_windows_by_horizon`` so a thin
      horizon is visible rather than silently averaged.
    """
    if not iou_n:
        return {}

    def _stack(key: str) -> np.ndarray:
        return np.sum([np.asarray(b["iou"][key], dtype=np.float64) for b in blocks], axis=0)

    silence = _stack("silence_by_horizon_sum") / iou_n
    shape = _stack("shape_by_horizon_sum") / iou_n
    floor = _stack("silent_floor_by_horizon_sum") / iou_n
    masked_sum = _stack("shape_masked_by_horizon_sum")
    masked_n = _stack("n_shape_masked_by_horizon")
    masked = [
        float(s / n) if n else None for s, n in zip(masked_sum, masked_n, strict=True)
    ]
    return {
        "best_member_iou_silence": float(silence[-1]),
        "best_member_iou_shape": float(shape[-1]),
        GATE_CRITERION_KEY: masked[-1],
        "best_member_iou_silent_floor": float(floor[-1]),
        "best_member_iou_silence_by_horizon": [float(v) for v in silence],
        "best_member_iou_shape_by_horizon": [float(v) for v in shape],
        f"{GATE_CRITERION_KEY}_by_horizon": masked,
        "best_member_iou_silent_floor_by_horizon": [float(v) for v in floor],
        f"{GATE_CRITERION_KEY}_n_windows_by_horizon": [int(v) for v in masked_n],
        "best_member_iou_gate_criterion": GATE_CRITERION_KEY,
    }


def _merge_bins(bin_lists: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, float]]:
    if not bin_lists:
        return []
    merged = [dict(b) for b in bin_lists[0]]
    for other in bin_lists[1:]:
        for i, b in enumerate(other):
            merged[i]["n"] = int(merged[i]["n"]) + int(b["n"])
            merged[i]["sum_p"] = float(merged[i]["sum_p"]) + float(b["sum_p"])
            merged[i]["sum_y"] = float(merged[i]["sum_y"]) + float(b["sum_y"])
    return merged


def _pool_diagnostics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled context — above all, how much of the sample was zero-growth.

    This is the first thing to read on any pooled result. If 79% of windows had
    no growth (insights/data item 1), a headline score is mostly a report on how
    well the model predicted that nothing happened, and persistence is the thing
    to compare it against.
    """
    flags = [r["diagnostics"].get("is_zero_growth_window") for r in items]
    known = [f for f in flags if f is not None]
    diversity = [
        r["diagnostics"].get("mean_pairwise_member_iou")
        for r in items
        if r["diagnostics"].get("mean_pairwise_member_iou") is not None
    ]
    growth = [
        r["diagnostics"].get("truth_growth_cells")
        for r in items
        if r["diagnostics"].get("truth_growth_cells") is not None
    ]
    return {
        "n_windows": len(items),
        "n_zero_growth_windows": int(sum(1 for f in known if f)),
        "zero_growth_fraction": (float(sum(1 for f in known if f)) / len(known)) if known else None,
        "truth_growth_cells_total": int(sum(growth)) if growth else None,
        "mean_pairwise_member_iou": float(np.mean(diversity)) if diversity else None,
    }
