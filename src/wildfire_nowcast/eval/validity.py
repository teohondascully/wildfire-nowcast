"""C6.2 - baseline validity. A degenerate baseline VOIDS its gate.

INTERFACES C6.2 (ADR-011), binding:

    if a baseline ignites zero cells on the held-out set it is not a distinct
    baseline, and any gate resting on beating it is **VOID, not passed**.

This module is that clause as code, so a gate verdict cannot be written without
it having been evaluated. It is model-agnostic in exactly the C6 sense: it reads
``samples`` and ``truth`` and never a model internal, so it applies unchanged to
the ellipse, to the learned kernel, and to ELMFIRE at G5.

What it measures, and why these three and not one
-------------------------------------------------
``n_new_cells_predicted``
    Ensemble-mean newly-burned cells over the horizon, summed over windows.
    Zero is the hard VOID condition. It is a *count*, not a score, because the
    failure it catches is invisible to every score: the Brier-fitted ellipse
    posted a BETTER Brier than persistence while igniting nothing at all.
``growth_ratio``
    Predicted / observed new cells. A baseline can ignite a non-zero but absurd
    number of cells - 1 cell against 782 is not zero and is not a baseline
    either - so "not exactly zero" is a necessary condition, never a sufficient
    one. The verdict ladder therefore has a DEGENERATE band between VOID and OK.
``n_windows_with_any_ignition``
    Localisation of the previous two. One window igniting 800 cells and 300
    igniting none is a different pathology from a uniform under-prediction, and
    the two need different fixes.

Every verdict here treats a non-finite value as FAIL, never as pass-by-default
(project policy, ADR-012): a diagnostic that fails to ``ok`` is worse than no
diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from wildfire_nowcast.eval.masks import DEFAULT_EVENT, event_field

__all__ = [
    "window_ignition_counts",
    "baseline_validity",
    "VOID",
    "DEGENERATE",
    "OK",
    "UNDEFINED",
    "NULL_MODEL",
]

VOID = "VOID"
DEGENERATE = "DEGENERATE"
OK = "OK"
#: The scored set contains NO truth growth at all, so ``growth_ratio`` is 0/0.
#: The check has not passed - it has not run. Added after this ladder was found
#: crashing on ``f"{None:.3g}"`` in its own OK branch: the None case fell
#: through to the verdict that says "usable as a gate floor", so with one fewer
#: format specifier it would have reported a PASS on a sample that contained
#: nothing to validate against. Same family as the NaN-through-a-verdict-ladder
#: (simviz P1) and the unlearnable-zero assertion (M3 B3): a diagnostic whose
#: undefined case lands in `ok` is worse than no diagnostic.
UNDEFINED = "UNDEFINED"
#: Persistence ignites zero cells BY DEFINITION, so the VOID clause is a
#: category error against it: C6.2 targets a baseline that is supposed to
#: spread and has stopped. The caller must DECLARE a null model
#: (``null_model=True``); the check never infers it, because inferring it from
#: a zero count is exactly how the clause would be talked out of firing on the
#: baseline it was written for. Measured and worth keeping: the Brier-fitted
#: ellipse landed 0.005x of truth's growth - i.e. it had converged to within a
#: rounding error of the null model, which is the finding behind ADR-011.
NULL_MODEL = "NULL_MODEL"

#: Predicted/observed growth outside this band is reported as DEGENERATE: not a
#: hard VOID (only zero is), but not a baseline anyone should be scored against
#: either. An order of magnitude either way is deliberately generous - the
#: measured failure was a factor of infinity, and a tight band here would turn a
#: validity check into a second calibration rule, which is not its job.
DEGENERATE_RATIO_BAND: tuple[float, float] = (0.1, 10.0)


def window_ignition_counts(
    samples: np.ndarray,
    truth: np.ndarray,
    x0: np.ndarray,
    *,
    event: str = DEFAULT_EVENT,
) -> dict[str, Any]:
    """New-cell counts for ONE window: ``(mean over members, max, truth)``.

    "New" is relative to ``x0``, so already-burned cells - which every model
    reproduces for free because fire is absorbing - cannot inflate the count.
    """
    pred = np.asarray(samples)
    if pred.ndim != 4:
        raise ValueError(f"samples must be [n_members, horizon, H, W], got {pred.shape}")
    burned0 = np.asarray(x0) > 0
    member_final = event_field(pred[:, -1], event) & ~burned0[None]
    truth_final = event_field(np.asarray(truth)[pred.shape[1] - 1], event) & ~burned0
    per_member = np.count_nonzero(member_final, axis=(1, 2)).astype(np.float64)
    return {
        "mean_new_cells": float(per_member.mean()),
        "max_new_cells": float(per_member.max()),
        "min_new_cells": float(per_member.min()),
        "truth_new_cells": float(np.count_nonzero(truth_final)),
        "any_member_ignited": bool(per_member.max() > 0),
    }


def _off_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """[M5] Split the growth ratio by DORMANT vs GROWTH windows. The OFF STATE.

    sim's S3 decomposed our 2.66-3.06x held-out over-prediction and found
    it is **not a spread-rate error**: on the 446 windows where truth grew, the
    trained kernel scores 0.984 / 0.874 - better calibrated than the calibrated
    ellipse's 0.845, which UNDER-predicts. The whole excess is bought in the 953
    windows where truth did nothing, and the kernel ignites in **953 of 953** of
    them while its own untrained initialisation manages 863 and the ellipse 885.
    Training made the model strictly worse at predicting NOTHING.

    That is a different defect with a different remedy, and the distinction is
    load-bearing: dividing the rate by ~3 would drive the growth-window ratio
    from 0.98 to ~0.33, and ADR-020 (4)(b) fixed G3's calibration criterion on
    the GROWTH-MASKED subset - decided blind, before this was measured. A global
    scale would break the gate it appears to help.

    Computed here, inside C6.2's own instrument and from the same per-window
    counts the clause already uses, so the decomposition and the ratio C6.2
    reports cannot drift apart the way two implementations of one quantity do.
    """
    dormant = [r for r in rows if float(r["truth_new_cells"]) <= 0]
    growing = [r for r in rows if float(r["truth_new_cells"]) > 0]
    pred_dormant = float(sum(float(r["mean_new_cells"]) for r in dormant))
    pred_growth = float(sum(float(r["mean_new_cells"]) for r in growing))
    truth_growth = float(sum(float(r["truth_new_cells"]) for r in growing))
    total_excess = (pred_dormant + pred_growth) - truth_growth
    silent = [r for r in dormant if float(r["mean_new_cells"]) <= 0.0]
    all_members_silent = [r for r in dormant if not bool(r["any_member_ignited"])]
    per_dormant = sorted(float(r["mean_new_cells"]) for r in dormant)
    return {
        "n_windows": len(rows),
        "n_dormant_windows": len(dormant),
        "n_growth_windows": len(growing),
        "growth_stratum_ratio": (pred_growth / truth_growth) if truth_growth > 0 else None,
        "all_window_ratio": (
            ((pred_dormant + pred_growth) / truth_growth) if truth_growth > 0 else None
        ),
        "predicted_cells_in_dormant_windows": pred_dormant,
        "dormant_share_of_excess": (
            (pred_dormant / total_excess) if abs(total_excess) > 1e-9 else None
        ),
        "n_dormant_windows_with_zero_expected_ignition": len(silent),
        "n_dormant_windows_where_no_member_ignited": len(all_members_silent),
        "dormant_off_rate": (len(silent) / len(dormant)) if dormant else None,
        "median_expected_cells_in_dormant_windows": (
            per_dormant[len(per_dormant) // 2] if per_dormant else None
        ),
        "min_expected_cells_in_dormant_windows": (per_dormant[0] if per_dormant else None),
        "note": (
            "`dormant_off_rate` is the fraction of zero-growth windows in which the model "
            "expects to ignite NOTHING. A model with no OFF state scores 0.0 here however "
            "well calibrated its spread rate is, and `growth_stratum_ratio` is where the "
            "spread rate is actually visible. ADR-020 (4)(b) puts G3's calibration criterion "
            "on the growth-masked subset, so these two must be read separately."
        ),
    }


#: [M6] Bars for :func:`off_state_verdict`. They are the bars of a PLAYTHROUGH
#: (ADR-030), not of a gate: the scenario's answer is known by construction, so
#: these separate "the model represents dormancy" from "the model is switched off
#: at random", and nothing here adjudicates G3.
OFF_STATE_MIN_DORMANT_OFF_RATE = 0.50
OFF_STATE_MAX_FALSE_OFF_RATE = 0.20


def off_state_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_dormant_off_rate: float = OFF_STATE_MIN_DORMANT_OFF_RATE,
    max_false_off_rate: float = OFF_STATE_MAX_FALSE_OFF_RATE,
) -> dict[str, Any]:
    """[M6] Can this ensemble say NOTHING WILL HAPPEN, and only when it is true?

    Scoring function for the OFF-STATE PLAYTHROUGH (ADR-030). ``rows`` are
    :func:`window_ignition_counts` dicts. It returns a pass/fail verdict from TWO
    rates, and carrying both is the whole point:

    ``dormant_off_rate``
        fraction of ZERO-GROWTH windows where the ensemble expects nothing -
        CAPACITY. Every kernel this project has trained scores **0.0000** here.
    ``false_off_rate``
        fraction of GROWING windows where the ensemble expects nothing -
        CONDITIONING. A model that is switched off at random hours buys a high
        ``dormant_off_rate`` and pays for it here.

    **A one-rate test would PASS the exact model M6 already measured as not
    solving this.** The activity gate raised the train member-area variance ratio
    1.19 -> 10.04 at no NLL cost - the ensemble CONTAINS dormant members - while
    ``dormant_off_rate`` stayed at 0.0000, because a mixture whose mixing weight
    does not depend on the hour never drives the ensemble MEAN to zero. Capacity
    is not knowledge, and this verdict is built so the two cannot be confused.
    """
    dormant = [r for r in rows if float(r["truth_new_cells"]) <= 0]
    growing = [r for r in rows if float(r["truth_new_cells"]) > 0]
    off_dormant = [r for r in dormant if float(r["mean_new_cells"]) <= 0.0]
    off_growing = [r for r in growing if float(r["mean_new_cells"]) <= 0.0]
    dormant_off = (len(off_dormant) / len(dormant)) if dormant else None
    false_off = (len(off_growing) / len(growing)) if growing else None
    reasons: list[str] = []
    if not dormant or not growing:
        reasons.append(
            "UNDEFINED: the scenario must contain BOTH dormant and growing windows, "
            f"got {len(dormant)} dormant and {len(growing)} growing"
        )
    else:
        if dormant_off is not None and dormant_off < min_dormant_off_rate:
            reasons.append(
                f"cannot represent an OFF state: dormant_off_rate={dormant_off:.4f} "
                f"< {min_dormant_off_rate}"
            )
        if false_off is not None and false_off > max_false_off_rate:
            reasons.append(
                f"OFF in the WRONG hours: false_off_rate={false_off:.4f} "
                f"> {max_false_off_rate} — capacity without conditioning"
            )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "dormant_off_rate": dormant_off,
        "false_off_rate": false_off,
        "n_dormant_windows": len(dormant),
        "n_growth_windows": len(growing),
        "min_dormant_off_rate": float(min_dormant_off_rate),
        "max_false_off_rate": float(max_false_off_rate),
    }


def baseline_validity(
    counts: Sequence[Mapping[str, Any]],
    *,
    name: str = "",
    scope: str = "held-out",
    null_model: bool = False,
) -> dict[str, Any]:
    """Pool :func:`window_ignition_counts` into the C6.2 verdict.

    Returns a dict whose ``verdict`` is one of :data:`VOID`, :data:`DEGENERATE`,
    :data:`OK`, :data:`NULL_MODEL`, together with the sentence a report must
    print next to any gate that rests on this baseline.

    ``null_model`` must be declared by the caller for a baseline whose zero
    ignition is definitional (persistence). It is never inferred - see
    :data:`NULL_MODEL`.
    """
    rows = list(counts)
    if not rows:
        raise ValueError("baseline_validity needs at least one window")
    pred_total = float(sum(float(r["mean_new_cells"]) for r in rows))
    truth_total = float(sum(float(r["truth_new_cells"]) for r in rows))
    n_ignited = int(sum(1 for r in rows if bool(r["any_member_ignited"])))
    n_truth_growth = int(sum(1 for r in rows if float(r["truth_new_cells"]) > 0))
    ratio = (pred_total / truth_total) if truth_total > 0 else None

    if null_model:
        verdict, why = (
            NULL_MODEL,
            f"declared null model: ignites {pred_total:.0f} cells by construction while truth "
            f"grows {truth_total:.0f}. C6.2's VOID clause does not apply, because it targets a "
            "baseline that is SUPPOSED to spread and has stopped",
        )
    elif not np.isfinite(pred_total) or (ratio is not None and not np.isfinite(ratio)):
        verdict, why = VOID, "non-finite ignition count — treated as FAIL, never as pass"
    elif pred_total <= 0.0:
        verdict, why = (
            VOID,
            f"ignites ZERO cells on the {scope} set while truth grows {truth_total:.0f}",
        )
    elif ratio is None:
        verdict, why = (
            UNDEFINED,
            f"truth grows ZERO cells across all {len(rows)} windows of the {scope} set, so "
            f"the growth ratio is 0/0. The baseline ignited {pred_total:.0f} cells; whether "
            "that is calibrated is UNMEASURED, not satisfied",
        )
    elif not (DEGENERATE_RATIO_BAND[0] <= ratio <= DEGENERATE_RATIO_BAND[1]):
        verdict, why = (
            DEGENERATE,
            f"ignites {pred_total:.0f} cells against truth's {truth_total:.0f} "
            f"(ratio {ratio:.3g}) — non-zero, but outside "
            f"[{DEGENERATE_RATIO_BAND[0]}, {DEGENERATE_RATIO_BAND[1]}]x",
        )
    else:
        verdict, why = (
            OK,
            f"ignites {pred_total:.0f} cells against truth's {truth_total:.0f} "
            f"(ratio {ratio:.3g}) in {n_ignited} of {len(rows)} windows",
        )

    consequence = {
        VOID: "Any gate resting on beating this baseline is VOID, NOT PASSED.",
        DEGENERATE: "Report this ratio next to any gate verdict that rests on this baseline.",
        OK: "Distinct from persistence; usable as a gate floor.",
        UNDEFINED: (
            "NOT a pass. C6.2 could not be evaluated on this sample; treat it exactly as "
            "you would a missing check, and re-run on a set that contains growth."
        ),
        NULL_MODEL: (
            "Not a gate floor on its own: a model that beats only this has beaten "
            "'nothing happens', which is right in most windows for observational reasons."
        ),
    }[verdict]
    statement = f"C6.2 baseline validity [{name or 'model'}]: {verdict} — {why}. {consequence}"
    return {
        "clause": "C6.2 (ADR-011)",
        "off_state": _off_state(rows),
        "name": name,
        "scope": scope,
        "verdict": verdict,
        "reason": why,
        "statement": statement,
        "n_windows": len(rows),
        "n_windows_with_any_ignition": n_ignited,
        "n_windows_with_truth_growth": n_truth_growth,
        "n_new_cells_predicted": pred_total,
        "n_new_cells_truth": truth_total,
        "growth_ratio": ratio,
        "mean_new_cells_per_window": pred_total / len(rows),
        "gate_voided": verdict == VOID,
        "is_null_model": bool(null_model),
    }
