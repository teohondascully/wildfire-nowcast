"""[U0] ``stage_decay`` - does a fire's growth fall as the fire AGES?

WHY THIS MODULE EXISTS (ADR-058 (1), C0)
----------------------------------------
The most-cited finding in this project - *real fires decelerate and ours does
not* - was computed by two UNTRACKED scratch scripts, ``runs/_m9_response.py``
and ``runs/_m9_scaling.py``. They were unfingerprinted, untested, and outside
every guard, so the headline number had no single tracked implementation. This
module is the promotion: the two published estimators are reproduced here off the
same artifacts, and ONE estimand is defined for the experiment to be scored on.

THE ESTIMAND, AND WHY IT IS THIS ONE
------------------------------------
``stage_decay`` for one spatial block is::

    order that block's held-out windows by age;
    drop the middle window if the count is odd;
    stage_decay = log(mean growth over the LATE half)
                - log(mean growth over the EARLY half)

Negative = the fire decelerates as it ages. Positive = it accelerates. It is
computed identically for TRUTH and for a forecast's ensemble-mean growth on the
SAME windows, so the two are differenced pairwise.

Six properties, each of which is the reason a rival form was rejected:

1. **It isolates STAGE, not a between-fire scaling law.** It is computed WITHIN a
   block, and held-out fold 3 is one fire per block, so a pooled comparison
   between a big fire and a small one cannot enter it. That confound is not
   hypothetical: the pooled elasticity is dominated by Creek (785 of 1372
   windows, owning the long-frontier bins outright) and the within-block truth
   elasticities span 35x (-5.93 to -0.17, ADR-058 (3)).
2. **It is a per-BLOCK scalar**, so the separation denominator can be the SD
   ACROSS HELD-OUT BLOCKS and never across seeds (ADR-042).
3. **It makes the fewest modelling choices of any form we have.** No regression,
   no logarithm of a single window, no conditioning on ``growth > 0``, no
   covariate control, no bin count, and no division by a noisy regressor. The
   only choice is WHERE to split, and it is the median, which is read off the
   sample rather than chosen.
4. **It depends on age only through RANK ORDER.** Hours-since-ignition, the
   ``t0`` window index and any monotone reparameterisation of age give the
   IDENTICAL value. The published table does not have this property: its bin
   count is itself a function of the window count
   (``min(8, max(3, n // 20))``), so CZU is read on 3 bins and Creek on 8.
5. **It uses every window.** The published statistic reads the first and last of
   3-8 bins, i.e. 12.5%-66% of the sample, and the discarded middle is where the
   non-monotonicity lives - on 3 of 5 held-out fires the maximum bin sits
   BETWEEN the two endpoints being compared (:func:`published_stage_bins` prints
   this).
6. **It is exactly antisymmetric under time reversal** (with distinct ages), so
   "decay" and "acceleration" are the same measurement with opposite sign rather
   than two conventions. Equal halves are what buy that identity, which is why
   the middle window of an odd sample is dropped rather than assigned.

Growth, not growth-per-frontier-cell, deliberately. **The support this choice
originally carried was ADR-058 (4), which ADR-060 (1) retracted IN FULL. It has
been RE-DERIVED under the estimand above rather than deleted**, because a
justification whose evidence is void is worth less than no justification: it
reads as settled. Same rows, same held-out fold 3 (blocks 4/5/6/7/12, 1372
windows, one fire per block), same corpus fingerprint ``b3e5dadad01eaef9``.
Recorded at ``runs/m12_frontier.json``, whose control reproduces the five
published truth values of ``runs/u0.json`` at a maximum absolute error of
exactly 0.0 and was checked against a perturbed row so that it could have
failed.

* **Frontier length RISES on 5 of 5 held-out blocks** -- +0.277 Creek, +1.036
  CZU, +0.678 Dolan, +0.617 July, +0.598 Borel -- while growth falls on 4 of 5.
  Frontier length is not tracking stage on this fold.
* **The per-frontier-cell RATE agrees in SIGN with growth on 5 of 5 and sits
  BELOW it on 5 of 5**: -3.277 vs -2.175 Creek, -1.907 vs -1.184 CZU, -1.332 vs
  -0.040 July, -2.451 vs -1.818 Borel, +0.546 vs +1.456 Dolan. It is computed
  DIRECTLY on the rows, never as ``stage_decay(growth) - stage_decay(frontier)``:
  the mean of a ratio is not the ratio of means. No window is dropped -- the
  frontier is non-empty on all 1372.

So dividing by frontier length does not remove the effect. It DEEPENS it on
every block where truth decelerates, and it moves no block's direction, which is
what makes the choice safe rather than lucky: growth is preferred on property 3's
grounds alone, because the ratio reintroduces frontier length as a denominator
and frontier length is the regressor whose treatment made two of our own
estimators disagree (ADR-058 (2)). **No elasticity appears anywhere above:
ADR-058 (2) rules that form estimator-dependent and unquotable, and the retracted
paragraph's error was to quote a second estimator-dependent statistic while
believing that the absence of a regression made it robust.**

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not adjudicate anything. :func:`licence` ASKS the C6 registry whether
``stage_decay`` may decide a gate, and today the registry says NO
(``NonAdjudicatingMetricError``), so every scoring path here returns
``not_a_verdict``. Licensing is a contract change and is the maintainer's
(ADR-058 (5)); an instrument that certifies itself certifies nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from wildfire_nowcast.common.null_check import (
    NonAdjudicatingMetricError,
    assert_may_adjudicate,
)
from wildfire_nowcast.common.separation import BlockPair, Separation, separation

__all__ = [
    "STAGE_DECAY_KEY",
    "MIN_WINDOWS_PER_BLOCK",
    "OUTCOME_OK",
    "OUTCOME_TOO_FEW_WINDOWS",
    "OUTCOME_ZERO_HALF_MEAN",
    "OUTCOME_NOT_FINITE",
    "StageDecay",
    "stage_decay",
    "stage_decay_by_block",
    "paired_stage_gap",
    "distance_to_truth_by_block",
    "separation_of_blocks",
    "proportional_closure_separation",
    "apply_stage_slope_to_rows",
    "inject_stage_slope",
    "injection_severity",
    "SEVERITY_UNIT",
    "synthetic_exponential_growth",
    "expected_stage_decay_of_exponential",
    "known_beta_recovery",
    "endpoint_log_ratio",
    "published_stage_bins",
    "published_frontier_bins",
    "published_growth_elasticities",
    "published_within_block_growth_elasticities",
    "weighted_loglog_slope",
    "as_rate_elasticity",
    "RATE_MINUS_GROWTH_ELASTICITY",
    "REPRODUCTION_TOL",
    "reproduction_error",
    "licence",
    "sign_test",
    "estimand_digest",
    "D3_LICENSED_ESTIMAND_SHA256",
    "ESTIMAND_FUNCTIONS",
]

#: The channel name, spelled ONCE. Every caller and every registry question uses
#: this constant so a table cannot ask about a channel it is not reporting.
STAGE_DECAY_KEY: Final = "stage_decay"

#: Fewest windows a block may contribute. Four per half is already thin; below
#: eight the two half-means are dominated by single windows and the estimand is
#: reporting one hour rather than one stage. Not fitted on anything - it is a
#: refusal threshold, and it returns UNDEFINED rather than a number.
MIN_WINDOWS_PER_BLOCK: Final = 8

OUTCOME_OK: Final = "OK"
OUTCOME_TOO_FEW_WINDOWS: Final = "UNDEFINED_too_few_windows"
OUTCOME_ZERO_HALF_MEAN: Final = "UNDEFINED_zero_half_mean"
OUTCOME_NOT_FINITE: Final = "UNDEFINED_non_finite_or_negative_growth"

#: A rate elasticity is a GROWTH elasticity minus one, because
#: ``log(growth / frontier) = log(growth) - log(frontier)``. This constant exists
#: because ADR-058 (2) quotes RATE elasticities (-1.7668 / +0.1960 / -0.3893)
#: while ``runs/m9_scaling.json`` stores GROWTH elasticities (-0.7668 / +1.1960 /
#: +0.6107) under the unqualified key ``elasticities_from_bin_means``. The
#: conversion is correct and the artifact does not state it, which is the
#: units-in-the-key failure this project has now paid for three times.
RATE_MINUS_GROWTH_ELASTICITY: Final = -1.0

#: The unit of the MDE ladder's severity axis, spelled once so an artifact cannot
#: report a severity without saying what it is. ``eval/power`` requires the
#: severity to be declared in a unit the channel under test does not compute, and
#: this one is raw displaced cell count: no logarithm, no half-means, no rank.
SEVERITY_UNIT: Final = (
    "equal-block mean of sum|growth' - growth| / sum(growth), i.e. the fraction of the block's "
    "total forecast growth the perturbation MOVES. Cell counts only."
)

#: Absolute tolerance at which a recomputation counts as reproducing a published
#: number. Both published estimators are pure functions of stored rows, so the
#: honest expectation is bitwise; 1e-9 leaves room for a platform ULP and nothing
#: else. A tolerance wide enough to hide a real disagreement is not a check.
REPRODUCTION_TOL: Final = 1e-9


# --------------------------------------------------------------------------
# the estimand
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageDecay:
    """One block's ``stage_decay``, with the OUTCOME beside the number.

    ``outcome`` is first and is a STRING, not a bool: ADR-057 (1) is a
    post-mortem of reading bare numbers out of an artifact that carried an
    ``outcome`` field saying they were undefined. ``value`` is ``None`` on every
    outcome except :data:`OUTCOME_OK`, so an undefined block cannot be read as a
    small one.
    """

    outcome: str
    value: float | None
    n_windows: int
    n_per_half: int
    n_dropped_middle: int
    early_mean: float | None
    late_mean: float | None

    @property
    def defined(self) -> bool:
        return self.outcome == OUTCOME_OK and self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            STAGE_DECAY_KEY: self.value,
            "n_windows": self.n_windows,
            "n_per_half": self.n_per_half,
            "n_dropped_middle": self.n_dropped_middle,
            "early_mean_growth": self.early_mean,
            "late_mean_growth": self.late_mean,
        }


def stage_decay(
    growth: Sequence[float] | np.ndarray,
    age: Sequence[float] | np.ndarray,
    *,
    min_windows: int = MIN_WINDOWS_PER_BLOCK,
) -> StageDecay:
    """``log(mean late-half growth) - log(mean early-half growth)`` for one block.

    ``age`` may be hours since ignition, a ``t0`` index, or anything monotone in
    time: only the induced ORDER is used, so the value cannot depend on the age
    scale. Ties are broken stably, i.e. by input order.

    Returns a three-valued result. Zero growth in either half is UNDEFINED and is
    never a pass — a fire that burned nothing in half its life has no measurable
    stage response, and reporting ``-inf`` or a clipped number would let a
    degenerate block set the direction of an equal-block mean.
    """
    g = np.asarray(growth, dtype=np.float64).reshape(-1)
    a = np.asarray(age, dtype=np.float64).reshape(-1)
    if g.size != a.size:
        raise ValueError(f"growth has {g.size} entries and age has {a.size}; they must pair")
    n = int(g.size)
    if n < int(min_windows):
        return StageDecay(OUTCOME_TOO_FEW_WINDOWS, None, n, 0, 0, None, None)
    if not (np.isfinite(g).all() and np.isfinite(a).all()) or bool((g < 0).any()):
        return StageDecay(OUTCOME_NOT_FINITE, None, n, 0, 0, None, None)

    order = np.argsort(a, kind="stable")
    half = n // 2
    early = float(g[order[:half]].mean())
    late = float(g[order[n - half :]].mean())
    if early <= 0.0 or late <= 0.0:
        return StageDecay(OUTCOME_ZERO_HALF_MEAN, None, n, half, n - 2 * half, early, late)
    return StageDecay(
        OUTCOME_OK,
        math.log(late) - math.log(early),
        n,
        half,
        n - 2 * half,
        early,
        late,
    )


def stage_decay_by_block(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    age_key: str = "t0",
    block_key: str = "spatial_block_id",
    min_windows: int = MIN_WINDOWS_PER_BLOCK,
) -> dict[int, StageDecay]:
    """:func:`stage_decay` per spatial block, from window rows.

    ``rows`` are the window records ``eval.response.window_row`` emits, so truth
    and any forecast are read from the SAME rows and therefore the same windows.
    """
    by_block: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        value = row.get(target)
        if value is None:
            continue
        by_block.setdefault(int(row[block_key]), []).append((float(value), float(row[age_key])))
    return {
        block: stage_decay([g for g, _ in pairs], [t for _, t in pairs], min_windows=min_windows)
        for block, pairs in sorted(by_block.items())
    }


def paired_stage_gap(
    rows: Sequence[Mapping[str, Any]],
    *,
    truth_key: str = "truth_growth",
    model_key: str = "model_growth",
    age_key: str = "t0",
    block_key: str = "spatial_block_id",
    min_windows: int = MIN_WINDOWS_PER_BLOCK,
) -> dict[str, Any]:
    """Per-block ``model - truth`` gap in ``stage_decay``, plus its equal-block mean.

    The gap is PAIRED within a block, so a fire that is simply harder than
    another cannot contribute: what is averaged over blocks is the difference,
    never two separately pooled levels. ``sd`` is the SD ACROSS BLOCKS, which is
    the only denominator this project accepts (ADR-042).
    """
    truth = stage_decay_by_block(
        rows, target=truth_key, age_key=age_key, block_key=block_key, min_windows=min_windows
    )
    model = stage_decay_by_block(
        rows, target=model_key, age_key=age_key, block_key=block_key, min_windows=min_windows
    )
    per_block: dict[str, Any] = {}
    gaps: list[float] = []
    for block in sorted(set(truth) | set(model)):
        t, m = truth.get(block), model.get(block)
        # Narrowed on `.value is not None` rather than on the `defined` property:
        # a property cannot narrow a type, and `float - None` reaching a verdict
        # path is the exact class of defect mypy found three times here already.
        gap: float | None = None
        if t is not None and m is not None and t.defined and m.defined:
            if t.value is None or m.value is None:  # pragma: no cover - `defined` forbids it
                raise AssertionError("StageDecay.defined is True with a None value")
            gap = m.value - t.value
        if gap is not None:
            gaps.append(gap)
        per_block[str(block)] = {
            "truth": t.as_dict() if t is not None else None,
            "model": m.as_dict() if m is not None else None,
            "gap_model_minus_truth": gap,
            "outcome": OUTCOME_OK if gap is not None else "UNDEFINED_block_not_scored",
        }
    n = len(gaps)
    return {
        "metric": STAGE_DECAY_KEY,
        "per_block": per_block,
        "n_blocks": n,
        "outcome": OUTCOME_OK if n >= 2 else "UNDEFINED_fewer_than_two_blocks",
        "equal_block_mean_gap": float(np.mean(gaps)) if n else None,
        "sd_across_blocks": float(np.std(gaps, ddof=1)) if n > 1 else None,
        "n_blocks_model_above_truth": int(sum(1 for g in gaps if g > 0)),
        "unanimous_sign": bool(n > 0 and (all(g > 0 for g in gaps) or all(g < 0 for g in gaps))),
        "uncertainty_basis": "SD ACROSS HELD-OUT BLOCKS (ADR-042). Never seeds, never windows.",
    }


def distance_to_truth_by_block(
    rows: Sequence[Mapping[str, Any]],
    *,
    truth_key: str = "truth_growth",
    model_key: str = "model_growth",
    age_key: str = "t0",
    block_key: str = "spatial_block_id",
    min_windows: int = MIN_WINDOWS_PER_BLOCK,
) -> dict[int, float]:
    """``|stage_decay(model) - stage_decay(truth)|`` per block. Lower is better.

    This is the quantity an architecture arm claims to reduce, so it is the
    quantity two arms are separated on. Blocks where either side is UNDEFINED are
    ABSENT rather than zero.
    """
    gap = paired_stage_gap(
        rows,
        truth_key=truth_key,
        model_key=model_key,
        age_key=age_key,
        block_key=block_key,
        min_windows=min_windows,
    )
    return {
        int(block): abs(float(entry["gap_model_minus_truth"]))
        for block, entry in gap["per_block"].items()
        if entry["gap_model_minus_truth"] is not None
    }


def separation_of_blocks(
    candidate: Mapping[int, float],
    reference: Mapping[int, float],
    *,
    lower_is_better: bool = True,
) -> Separation:
    """Paired equal-block separation of two per-block scalars.

    Delegates to :func:`common.separation.separation` - the same function G2/G3
    and M11's power analysis use - so a ``stage_decay`` separation is
    commensurable with every other separation in this project by construction
    rather than by claim.
    """
    pairs = [
        BlockPair(
            block_id=int(block),
            candidate=float(candidate[block]),
            reference=float(reference[block]),
            label=f"block {block}",
        )
        for block in sorted(set(candidate) & set(reference))
    ]
    return separation(pairs, lower_is_better=lower_is_better)


def proportional_closure_separation(distances: Mapping[int, float]) -> dict[str, Any]:
    """The separation reached by ANY arm that closes a common FRACTION of the gap.

    If a candidate reduces every block's distance-to-truth by the same fraction
    ``f``, its per-block margin is ``f * d_b``, so the separation is::

        mean(f * d) / sd(f * d)  ==  mean(d) / sd(d)

    - **exactly independent of f**. Closing 5% of the gap and closing 100% of it
    produce the SAME number. That is an identity, not an estimate, and it is what
    turns "what improvement is detectable" into "is this channel capable of
    detecting an improvement at all at this block count".

    An arm can only beat this ceiling by being MORE homogeneous across blocks
    than the distances themselves, i.e. by improving the blocks that were already
    close proportionally more. ``required_cv`` states how homogeneous: reaching a
    separation of ``s`` needs a margin coefficient of variation of at most
    ``1 / s``.
    """
    values = np.array([float(v) for v in distances.values()], dtype=np.float64)
    n = int(values.size)
    if n < 2:
        return {
            "outcome": "UNDEFINED_fewer_than_two_blocks",
            "separation_sd": None,
            "n_blocks": n,
        }
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd <= 0.0:
        return {
            "outcome": "UNDEFINED_zero_block_sd",
            "separation_sd": None,
            "n_blocks": n,
            "mean_distance": mean,
            "sd_distance": sd,
        }
    return {
        "outcome": OUTCOME_OK,
        "separation_sd": mean / sd,
        "n_blocks": n,
        "mean_distance": mean,
        "sd_distance": sd,
        "distance_cv": sd / mean if mean > 0 else None,
        "invariant_to_the_fraction_closed": True,
        "note": (
            "Separation of a proportional-closure arm from the reference. Independent of the "
            "fraction closed BY CONSTRUCTION, so it is a ceiling for that whole family of arms."
        ),
    }


def apply_stage_slope_to_rows(
    rows: Sequence[Mapping[str, Any]],
    delta: float,
    *,
    target: str = "model_growth",
    age_key: str = "t0",
    block_key: str = "spatial_block_id",
) -> list[dict[str, Any]]:
    """One degradation rung: :func:`inject_stage_slope` applied WITHIN each block.

    Within each block, because the injection is about a fire's own age axis. A
    pooled injection would tilt fires against each other and measure the
    between-fire scaling law this estimand exists to exclude.
    """
    by_block: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_block.setdefault(int(row[block_key]), []).append(i)
    out = [dict(row) for row in rows]
    for indices in by_block.values():
        growth = [float(rows[i][target]) for i in indices]
        age = [float(rows[i][age_key]) for i in indices]
        for i, value in zip(indices, inject_stage_slope(growth, age, delta), strict=True):
            out[i][target] = float(value)
    return out


def injection_severity(
    base: Sequence[Mapping[str, Any]],
    perturbed: Sequence[Mapping[str, Any]],
    *,
    target: str = "model_growth",
    block_key: str = "spatial_block_id",
) -> float:
    """How much forecast a perturbation MOVED, in units of :data:`SEVERITY_UNIT`.

    The severity axis of an MDE ladder must not be computed by the channel under
    test, or the ladder measures the channel against itself and any channel looks
    sensitive (``eval/power``'s module docstring, and M11's ``severity_source``
    note). This one is a plain L1 displacement of predicted cells and
    ``stage_decay`` is BLIND to part of it by construction: rearranging growth
    within a half changes this number and leaves ``stage_decay`` bit-identical,
    which is asserted in ``eval.selftest`` rather than argued here.

    Equal-block mean, matching the estimand it is a severity axis for, so one
    long fire cannot set the ladder's x-coordinate for the fold.
    """
    moved: dict[int, float] = {}
    total: dict[int, float] = {}
    for old, new in zip(base, perturbed, strict=True):
        block = int(old[block_key])
        moved[block] = moved.get(block, 0.0) + abs(float(new[target]) - float(old[target]))
        total[block] = total.get(block, 0.0) + float(old[target])
    per_block = [moved[b] / total[b] for b in sorted(moved) if total[b] > 0.0]
    return float(np.mean(per_block)) if per_block else 0.0


# --------------------------------------------------------------------------
# synthesis with a KNOWN stage decay (ADR-058 (10) item 3)
# --------------------------------------------------------------------------


def synthetic_exponential_growth(beta: float, n: int, *, amplitude: float = 1.0) -> np.ndarray:
    """``g_i = amplitude * exp(beta * i / n)`` for ``i = 0 .. n-1``.

    The age coordinate is ``i / n`` rather than ``i / (n - 1)`` ON PURPOSE: it
    makes :func:`expected_stage_decay_of_exponential` exactly ``beta / 2`` for
    EVERY even ``n``, so the known-beta control asserts a MAGNITUDE against a
    closed form that does not depend on the sample size (ADR-051's standard,
    which asked for an analytic identity rather than a non-zero reading).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.arange(int(n), dtype=np.float64)
    return float(amplitude) * np.exp(float(beta) * i / float(n))


def expected_stage_decay_of_exponential(beta: float) -> float:
    """Closed form: ``beta / 2``.

    With ``g_i = A exp(beta i / n)`` and even ``n``, the late half is the early
    half multiplied by ``exp(beta (n/2) / n)`` term by term, so the ratio of the
    two means is exactly ``exp(beta / 2)`` - no approximation and no dependence
    on ``A`` or ``n``. That is what makes it a recovery target rather than a
    fitted expectation.
    """
    return float(beta) / 2.0


def inject_stage_slope(
    growth: Sequence[float] | np.ndarray,
    age: Sequence[float] | np.ndarray,
    delta: float,
) -> np.ndarray:
    """Multiply each window's growth by ``exp(delta * (u - 0.5))``, ``u`` = age rank / n.

    A degradation rung for the MDE ladder. On a FLAT growth series it shifts
    ``stage_decay`` by exactly ``delta / 2``; on a real series the realised shift
    is whatever it is, and the ladder MEASURES it rather than assuming it - the
    severity axis must not be computed by the channel under test (``eval/power``
    module docstring), and here it is a property of the construction.

    Returns an array in the INPUT order, so it can be written straight back into
    the rows it came from.
    """
    g = np.asarray(growth, dtype=np.float64).reshape(-1)
    a = np.asarray(age, dtype=np.float64).reshape(-1)
    if g.size != a.size:
        raise ValueError(f"growth has {g.size} entries and age has {a.size}; they must pair")
    n = int(g.size)
    if n == 0:
        return g.copy()
    rank = np.empty(n, dtype=np.float64)
    rank[np.argsort(a, kind="stable")] = np.arange(n, dtype=np.float64)
    return g * np.exp(float(delta) * (rank / float(n) - 0.5))


def _piecewise_rows(
    levels: Sequence[float], *, per_level: int, block: int = 0
) -> list[dict[str, Any]]:
    """``per_level`` windows at each level, ages ``0, 1, 2, ...``.

    A hand-built fixture whose bin means and half means are both exact rationals,
    so both statistics have closed forms and nothing is asserted against its own
    output.
    """
    rows: list[dict[str, Any]] = []
    for level in levels:
        for _ in range(int(per_level)):
            rows.append(
                {
                    "spatial_block_id": block,
                    "fire_id": f"synthetic_block_{block}",
                    "t0": len(rows),
                    "growth": float(level),
                    "_n_frontier_cells": 1.0,
                }
            )
    return rows


def known_beta_recovery() -> dict[str, Any]:
    """The known-beta control for ``stage_decay``: recovery, AGREEMENT, REVERSAL.

    ADR-058 (10) item 3, on ADR-057 (5)'s standard: *a divergence test that can
    only ever show divergence is broken in the other direction*. Four clauses,
    each able to fail on its own:

    1. **RECOVERY, against a closed form.** Forecasts synthesised with a known
       ``beta`` must read back ``beta / 2`` exactly (:func:`
       expected_stage_decay_of_exponential`), across sign, magnitude, sample size
       and amplitude. ``beta = 0`` must read exactly ``0.0``.
    2. **AGREEMENT - the disqualifying clause.** Two forecasts built to have the
       SAME stage decay while differing in everything else (amplitude 1 vs
       987.65, 40 windows vs 400) must be reported as identical; a bit-identical
       pair must differ by EXACTLY 0.0; and five blocks of a candidate that IS
       the reference must separate at EXACTLY 0.0 with 0 blocks favouring. An
       estimator that manufactures a difference here is disqualified whatever it
       does on the divergence clause.
    3. **ORDER REVERSAL against the published statistic.** Two cases where
       ``stage_decay`` says X decays MORE than Y while the published
       first-bin-to-last-bin ratio says the opposite, both in closed form. That
       proves the two are different estimands rather than two spellings of one,
       which is the whole reason ADR-058 (2)'s factor of 2.6 was possible.
    4. **THE REVERSAL'S OWN CONTROL.** A second pair on which the two statistics
       AGREE, so clause 3 is a property of the cases and not of the comparison.
    """
    recovery: list[dict[str, Any]] = []
    for beta in (-3.0, -2.0, -0.7, 0.0, 0.5, 3.0):
        for n, amplitude in ((40, 1.0), (400, 987.65), (4000, 1e-3)):
            growth = synthetic_exponential_growth(beta, n, amplitude=amplitude)
            measured = stage_decay(growth, np.arange(n, dtype=np.float64))
            expected = expected_stage_decay_of_exponential(beta)
            recovery.append(
                {
                    "beta": beta,
                    "n": n,
                    "amplitude": amplitude,
                    "expected": expected,
                    "measured": measured.value,
                    "abs_error": (
                        None if measured.value is None else abs(measured.value - expected)
                    ),
                    "outcome": measured.outcome,
                }
            )
    recovery_exact = all(
        r["abs_error"] is not None and float(r["abs_error"]) <= 1e-12 for r in recovery
    )
    zero_is_exactly_zero = all(r["measured"] == 0.0 for r in recovery if r["beta"] == 0.0)

    # --- clause 2: AGREEMENT -------------------------------------------------
    shared_beta = -1.3
    a_small = stage_decay(
        synthetic_exponential_growth(shared_beta, 40, amplitude=1.0),
        np.arange(40, dtype=np.float64),
    )
    a_large = stage_decay(
        synthetic_exponential_growth(shared_beta, 400, amplitude=987.65),
        np.arange(400, dtype=np.float64),
    )
    same_beta_difference = (
        None
        if a_small.value is None or a_large.value is None
        else abs(a_large.value - a_small.value)
    )
    identical = synthetic_exponential_growth(0.42, 120, amplitude=3.5)
    one = stage_decay(identical, np.arange(120, dtype=np.float64))
    two = stage_decay(identical.copy(), np.arange(120, dtype=np.float64))
    identical_difference = (
        None if one.value is None or two.value is None else abs(one.value - two.value)
    )
    blocks = {b: 0.4 + 0.3 * b for b in range(5)}
    self_separation = separation_of_blocks(blocks, dict(blocks), lower_is_better=True)
    agrees_at_identity = bool(
        identical_difference == 0.0
        and same_beta_difference is not None
        and same_beta_difference <= 1e-12
        and self_separation.separation_sd == 0.0
        and self_separation.blocks_favouring == 0
    )

    # --- clauses 3 and 4: ORDER REVERSAL, and a pair that AGREES -------------
    per_level = 20
    cases = {
        "x_late_recovery": [10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        "y_late_collapse": [10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0],
        "z_monotone_decline": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        "w_flat": [10.0] * 8,
    }
    closed_form = {
        "x_late_recovery": (math.log(8.0 / 13.0), math.log(5.0 / 10.0)),
        "y_late_collapse": (math.log(61.0 / 70.0), math.log(1.0 / 10.0)),
        "z_monotone_decline": (math.log(9.0 / 17.0), math.log(3.0 / 10.0)),
        "w_flat": (0.0, 0.0),
    }
    measured_cases: dict[str, dict[str, Any]] = {}
    for name, levels in cases.items():
        rows = _piecewise_rows(levels, per_level=per_level)
        sd_value = stage_decay([float(r["growth"]) for r in rows], [float(r["t0"]) for r in rows])
        bins = published_stage_bins(rows, targets=("growth",))["0"]["bins"]
        endpoint = endpoint_log_ratio(bins, "growth")
        want_sd, want_ep = closed_form[name]
        measured_cases[name] = {
            "stage_decay": sd_value.value,
            "stage_decay_closed_form": want_sd,
            "stage_decay_abs_error": (
                None if sd_value.value is None else abs(sd_value.value - want_sd)
            ),
            "endpoint_log_ratio": endpoint,
            "endpoint_closed_form": want_ep,
            "endpoint_abs_error": None if endpoint is None else abs(endpoint - want_ep),
            "n_bins": len(bins),
        }
    magnitudes_exact = all(
        c["stage_decay_abs_error"] is not None
        and float(c["stage_decay_abs_error"]) <= 1e-12
        and c["endpoint_abs_error"] is not None
        and float(c["endpoint_abs_error"]) <= 1e-12
        for c in measured_cases.values()
    )
    x, y = measured_cases["x_late_recovery"], measured_cases["y_late_collapse"]
    z, w = measured_cases["z_monotone_decline"], measured_cases["w_flat"]
    order_reverses = bool(
        x["stage_decay"] < y["stage_decay"] and x["endpoint_log_ratio"] > y["endpoint_log_ratio"]
    )
    statistics_agree_on_the_control_pair = bool(
        z["stage_decay"] < w["stage_decay"] and z["endpoint_log_ratio"] < w["endpoint_log_ratio"]
    )

    # --- the refusals, which are also outcomes -------------------------------
    dead_early = stage_decay([0.0] * 10 + [1.0] * 10, np.arange(20, dtype=np.float64))
    refuses_zero_half = bool(
        dead_early.outcome == OUTCOME_ZERO_HALF_MEAN and dead_early.value is None
    )
    ages = np.arange(60, dtype=np.float64)
    series = synthetic_exponential_growth(-1.1, 60, amplitude=2.0)
    forward = stage_decay(series, ages)
    reversed_ = stage_decay(series[::-1].copy(), ages)
    antisymmetric = bool(
        forward.value is not None
        and reversed_.value is not None
        and abs(forward.value + reversed_.value) <= 1e-12
    )
    rescaled = stage_decay(series, np.exp(ages / 17.0) * 3.0 - 11.0)
    age_scale_invariant = bool(
        forward.value is not None and rescaled.value is not None and forward.value == rescaled.value
    )

    return {
        "outcome": (
            OUTCOME_OK
            if (
                recovery_exact
                and zero_is_exactly_zero
                and agrees_at_identity
                and magnitudes_exact
                and order_reverses
                and statistics_agree_on_the_control_pair
                and refuses_zero_half
                and antisymmetric
                and age_scale_invariant
            )
            else "FAILED"
        ),
        "recovery": recovery,
        "recovery_exact": recovery_exact,
        "zero_beta_is_exactly_zero": zero_is_exactly_zero,
        "max_recovery_abs_error": max(
            float(r["abs_error"]) for r in recovery if r["abs_error"] is not None
        ),
        "agreement": {
            "shared_beta": shared_beta,
            "same_beta_different_amplitude_and_n": same_beta_difference,
            "bit_identical_pair_difference": identical_difference,
            "self_separation_sd": self_separation.separation_sd,
            "self_blocks_favouring": self_separation.blocks_favouring,
        },
        "agrees_at_identity": agrees_at_identity,
        "cases": measured_cases,
        "magnitudes_exact": magnitudes_exact,
        "order_reverses": order_reverses,
        "statistics_agree_on_the_control_pair": statistics_agree_on_the_control_pair,
        "refuses_zero_half": refuses_zero_half,
        "antisymmetric_under_time_reversal": antisymmetric,
        "age_scale_invariant": age_scale_invariant,
        "not_a_verdict": (
            "Known-answer verification on SYNTHETIC and hand-built series. It licenses the "
            "ESTIMATOR, never an arm, and it says nothing about any held-out fire."
        ),
    }


# --------------------------------------------------------------------------
# the PUBLISHED estimators, reproduced (ADR-058 (10) item 1)
# --------------------------------------------------------------------------


def endpoint_log_ratio(bins: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """``log(last bin / first bin)`` -- the statistic ADR-058 (4)'s table reads.

    Reproduced here so it can be COMPARED with :func:`stage_decay` rather than
    argued about. It is not the estimand: it reads two of 3-8 bins and discards
    the interior, and on this fold the interior contains the maximum on 3 of 5
    fires. **ADR-060 (1) retracted ADR-058 (4) in full**, so this function
    reproduces a retracted statistic ON PURPOSE, and nothing it returns may be
    quoted as a result.
    """
    if len(bins) < 2:
        return None
    first, last = float(bins[0][key]), float(bins[-1][key])
    if first <= 0.0 or last <= 0.0:
        return None
    return math.log(last) - math.log(first)


def _published_bin_count(n: int, n_bins: int) -> int:
    """``min(n_bins, max(3, n // 20))`` - verbatim from ``runs/_m9_scaling.py``.

    Reproduced rather than improved. Note what it does: the number of bins is a
    function of the SAMPLE SIZE, so the published per-fire statistic is read on 3
    bins for CZU and 8 for Creek and the two are not the same estimator.
    """
    return int(min(n_bins, max(3, n // 20)))


def published_stage_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str] = ("truth_growth", "model_growth"),
    n_bins: int = 8,
    min_windows: int = 40,
    age_key: str = "t0",
    block_key: str = "spatial_block_id",
) -> dict[str, Any]:
    """Reproduce ``runs/m9_scaling.json``'s ``fire_stage_by_t0``.

    Windows of a block sorted by ``t0`` and split with ``numpy.array_split`` into
    :func:`_published_bin_count` groups; each bin reports the mean of every
    target. Blocks with fewer than ``min_windows`` windows are OMITTED, exactly
    as the script omits them.
    """
    by_block: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(int(row[block_key]), []).append(row)

    out: dict[str, Any] = {}
    for block, block_rows in sorted(by_block.items()):
        ordered = sorted(block_rows, key=lambda r: r[age_key])
        if len(ordered) < min_windows:
            continue
        groups = np.array_split(np.arange(len(ordered)), _published_bin_count(len(ordered), n_bins))
        bins: list[dict[str, Any]] = []
        for idx in groups:
            entry: dict[str, Any] = {
                "t0_mean": float(np.mean([float(ordered[i][age_key]) for i in idx])),
                "n": int(idx.size),
                "frontier_mean": float(
                    np.mean([float(ordered[i]["_n_frontier_cells"]) for i in idx])
                ),
            }
            for target in targets:
                entry[target] = float(np.mean([float(ordered[i][target]) for i in idx]))
            bins.append(entry)
        out[str(block)] = {
            "fire_id": str(ordered[0].get("fire_id", "")),
            "bins": bins,
            "n_bins": len(bins),
            "interior_exceeds_both_endpoints": {
                target: bool(
                    len(bins) > 2
                    and max(float(b[target]) for b in bins[1:-1])
                    > max(float(bins[0][target]), float(bins[-1][target]))
                )
                for target in targets
            },
            "endpoint_log_ratio": {target: endpoint_log_ratio(bins, target) for target in targets},
        }
    return out


def weighted_loglog_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float | int]:
    """Weighted OLS of ``log y`` on ``log x`` over BINS - ported from ``_m9_scaling.py``.

    Ported VERBATIM, including its standard-error formula (a weighted residual
    variance rescaled by ``n / dof``), because the point of this function is to
    reproduce a published number. Improving it here would answer a different
    question from the one the artifact answers, and the disagreement would be
    unattributable.

    The slope is the elasticity OF THE BINNED TARGET, i.e. of GROWTH when the
    target is growth. See :func:`as_rate_elasticity` before comparing it with a
    rate elasticity.
    """
    keep = (y > 0) & (x > 0)
    lx, ly, lw = np.log(x[keep]), np.log(y[keep]), w[keep]
    if lx.size < 3:
        return {"elasticity": float("nan"), "se": float("nan"), "n_bins": int(lx.size)}
    design = np.column_stack([np.ones(lx.size), lx])
    sw = np.sqrt(lw)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], ly * sw, rcond=None)
    resid = ly - design @ beta
    dof = max(lx.size - 2, 1)
    s2 = float((lw * resid**2).sum() / lw.sum()) * lx.size / dof
    cov = np.linalg.pinv((design * lw[:, None]).T @ design) * s2
    return {
        "elasticity": float(beta[1]),
        "intercept": float(beta[0]),
        "se": float(math.sqrt(max(cov[1, 1], 0.0))),
        "n_bins": int(lx.size),
    }


def as_rate_elasticity(growth_elasticity: float) -> float:
    """Convert a GROWTH elasticity to a RATE elasticity: subtract exactly 1.

    ``log(growth / frontier) = log(growth) - log(frontier)``, so regressing
    either against ``log(frontier)`` gives slopes differing by exactly one. This
    is the conversion ADR-058 (2) applied silently: ``runs/m9_scaling.json``
    stores -0.7668 / +1.1960 / +0.6107 and the ADR quotes -1.7668 / +0.1960 /
    -0.3893. The conversion is right; the artifact key does not say which unit it
    is in, which is why it is a named function here instead of a literal ``- 1``.
    """
    return float(growth_elasticity) + RATE_MINUS_GROWTH_ELASTICITY


def published_frontier_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str],
    n_bins: int = 8,
) -> list[dict[str, Any]]:
    """Reproduce ``runs/m9_scaling.json``'s ``bins``: ALL windows binned by frontier."""
    frontier = np.array([float(r["_n_frontier_cells"]) for r in rows], dtype=np.float64)
    groups = np.array_split(np.argsort(frontier), n_bins)
    table: list[dict[str, Any]] = []
    for idx in groups:
        entry: dict[str, Any] = {
            "n_windows": int(idx.size),
            "frontier_mean": float(frontier[idx].mean()),
            "frontier_min": float(frontier[idx].min()),
            "frontier_max": float(frontier[idx].max()),
        }
        for target in targets:
            values = np.array([float(rows[i][target]) for i in idx], dtype=np.float64)
            entry[target] = float(values.mean())
        table.append(entry)
    return table


def published_growth_elasticities(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str],
    n_bins: int = 8,
) -> dict[str, dict[str, float | int]]:
    """Reproduce ``elasticities_from_bin_means`` - GROWTH elasticities, pooled."""
    table = published_frontier_bins(rows, targets=targets, n_bins=n_bins)
    counts = np.array([float(e["n_windows"]) for e in table], dtype=np.float64)
    frontier = np.array([float(e["frontier_mean"]) for e in table], dtype=np.float64)
    return {
        target: weighted_loglog_slope(
            frontier, np.array([float(e[target]) for e in table], dtype=np.float64), counts
        )
        for target in targets
    }


def published_within_block_growth_elasticities(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str],
    n_bins: int = 8,
    min_windows: int = 40,
    block_key: str = "spatial_block_id",
) -> dict[str, Any]:
    """Reproduce ``within_block_elasticities`` - the table ADR-058 (3) quotes."""
    by_block: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(int(row[block_key]), []).append(row)

    out: dict[str, Any] = {}
    for block, block_rows in sorted(by_block.items()):
        if len(block_rows) < min_windows:
            out[str(block)] = {"n": len(block_rows), "skipped": "fewer than 40 windows"}
            continue
        frontier = np.array([float(r["_n_frontier_cells"]) for r in block_rows], dtype=np.float64)
        groups = np.array_split(np.argsort(frontier), _published_bin_count(len(block_rows), n_bins))
        bin_frontier = np.array([frontier[i].mean() for i in groups], dtype=np.float64)
        bin_counts = np.array([float(i.size) for i in groups], dtype=np.float64)
        entry: dict[str, Any] = {
            "fire_id": str(block_rows[0].get("fire_id", "")),
            "n": len(block_rows),
            "n_bins": len(groups),
        }
        for target in targets:
            means = np.array(
                [np.mean([float(block_rows[i][target]) for i in idx]) for idx in groups],
                dtype=np.float64,
            )
            entry[target] = weighted_loglog_slope(bin_frontier, means, bin_counts)
        out[str(block)] = entry
    return out


def reproduction_error(
    recomputed: float | None,
    published: float | None,
    *,
    tol: float = REPRODUCTION_TOL,
) -> dict[str, Any]:
    """One reproduction cell: both values, the absolute error, and the OUTCOME.

    The outcome STRING is the field a reader should take the answer from
    (ADR-057 (1)). ``NaN`` on both sides is reproduction, not failure: the
    published fit declares NaN where a bin set was too small, and a
    reimplementation that turned that into a number would not be reproducing it.
    """
    if recomputed is None or published is None:
        return {
            "recomputed": recomputed,
            "published": published,
            "abs_error": None,
            "outcome": "NOT_REPRODUCED_missing_value",
        }
    a, b = float(recomputed), float(published)
    if math.isnan(a) and math.isnan(b):
        return {"recomputed": a, "published": b, "abs_error": 0.0, "outcome": "REPRODUCED_nan"}
    err = abs(a - b)
    return {
        "recomputed": a,
        "published": b,
        "abs_error": err,
        "outcome": "REPRODUCED" if err <= tol else "NOT_REPRODUCED",
    }


# --------------------------------------------------------------------------
# the criterion that CAN benefit from more blocks (ADR-060 (7) item 2)
# --------------------------------------------------------------------------


def sign_test(n_favourable: int, n_blocks: int) -> dict[str, Any]:
    """One-sided exact binomial tail at p = 0.5: ``P(X >= n_favourable)``.

    ADR-060 (7) item 2: an effect size cannot benefit from more blocks, because
    adding blocks raises the SD as fast as the mean; a paired SIGN criterion can,
    because the tail shrinks geometrically. 4/5 is p = 0.1875 and 11/14 is
    p = 0.0287 on the SAME per-block direction. G2 already stands on unanimity
    plus a sign test rather than an SD (ADR-053 (5)).

    Exact, via :func:`math.comb` - no normal approximation, which at n = 14 would
    be wrong in the third decimal and is where a borderline call would land.

    Reports the OUTCOME string beside the number (ADR-057 (1)); ties must be
    resolved by the CALLER before it gets here, since a block whose value is
    exactly zero is not evidence in either direction and silently counting it as
    unfavourable would be a choice made inside a p-value.
    """
    n = int(n_blocks)
    k = int(n_favourable)
    if n <= 0:
        return {"outcome": "UNDEFINED_no_blocks", "p_one_sided": None, "k": k, "n": n}
    if not 0 <= k <= n:
        raise ValueError(f"n_favourable={k} is not in [0, {n}]")
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return {
        "outcome": OUTCOME_OK,
        "p_one_sided": tail / float(2**n),
        "k": k,
        "n": n,
        "fraction": k / float(n),
        "tail_numerator": tail,
        "denominator": 2**n,
        "note": (
            "One-sided exact binomial tail at p=0.5, i.e. the chance of seeing at least this "
            "many blocks pointing one way if each block were a coin flip. It tests DIRECTION "
            "and says nothing about magnitude."
        ),
    }


# --------------------------------------------------------------------------
# provenance: is this the estimand ADR-060 licensed, or a re-tuned one?
# --------------------------------------------------------------------------

#: The functions that ARE the estimand. Everything else in this module is a
#: published-statistic reproduction, a control, or plumbing.
ESTIMAND_FUNCTIONS: Final = (
    "StageDecay",
    "stage_decay",
    "stage_decay_by_block",
    "paired_stage_gap",
)

#: SHA-256 of the source of :data:`ESTIMAND_FUNCTIONS`, as they stood when
#: ``known_beta_recovery`` licensed them (U0 deliverable 3, ruled in ADR-060).
#: Pinned so that a run record can CLAIM "the same estimator" and be checked
#: rather than believed. Changing the estimand means updating this constant in
#: the SAME commit and re-running D3 - that friction is the point, because a
#: silently re-tuned estimand invalidates every result that cites this one.
D3_LICENSED_ESTIMAND_SHA256: Final = (
    "a78c175a278d6041be769b8d998c4811030f8c3aa82af46143a25920ad03a903"
)


def estimand_digest() -> dict[str, Any]:
    """Hash the LIVE source of the estimand and compare it with the pinned digest.

    Reads this module's own file and slices out :data:`ESTIMAND_FUNCTIONS` by AST,
    so reformatting elsewhere in the file, new helpers and new docstrings cannot
    move it, and an edit to the estimand itself cannot fail to.
    """
    import ast
    import hashlib
    from pathlib import Path

    text = Path(__file__).read_text()
    tree = ast.parse(text)
    segments = [
        ast.get_source_segment(text, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in ESTIMAND_FUNCTIONS
    ]
    found = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in ESTIMAND_FUNCTIONS
    )
    digest = hashlib.sha256("\n".join(segments).encode()).hexdigest()
    missing = [name for name in ESTIMAND_FUNCTIONS if name not in found]
    return {
        "outcome": (
            "UNCHANGED_SINCE_D3"
            if digest == D3_LICENSED_ESTIMAND_SHA256 and not missing
            else "CHANGED_SINCE_D3"
        ),
        "sha256": digest,
        "pinned": D3_LICENSED_ESTIMAND_SHA256,
        "functions": list(found),
        "missing": missing,
    }


# --------------------------------------------------------------------------
# the guard: ask the registry, never remember (ADR-059 (5))
# --------------------------------------------------------------------------


def licence(*, gate: str = "the stage experiment") -> dict[str, Any]:
    """Ask C6 whether :data:`STAGE_DECAY_KEY` may decide ``gate``. Never assume.

    ADR-059 (5) found that ``eval/`` had no call site for
    :func:`common.null_check.assert_may_adjudicate`: the registry was correct and
    nothing forced a scoring path through it, which is the difference between a
    guard existing and a guard being wired. This is the call site.

    It CATCHES the refusal and returns it rather than propagating, because the
    refusal is the expected answer today and a scoring run must still produce its
    numbers - what it must not do is produce a verdict. Every caller reads
    ``may_adjudicate`` and reports ``not_a_verdict`` when it is False.
    """
    try:
        spec = assert_may_adjudicate(STAGE_DECAY_KEY, gate=gate)
    except NonAdjudicatingMetricError as exc:
        return {
            "metric": STAGE_DECAY_KEY,
            "gate": gate,
            "may_adjudicate": False,
            "outcome": "NOT_LICENSED",
            "registry_refusal": str(exc),
            "checked_by": "common.null_check.assert_may_adjudicate",
        }
    return {
        "metric": STAGE_DECAY_KEY,
        "gate": gate,
        "may_adjudicate": True,
        "outcome": "LICENSED",
        "direction": spec.direction,
        "note": spec.note,
        "checked_by": "common.null_check.assert_may_adjudicate",
    }
