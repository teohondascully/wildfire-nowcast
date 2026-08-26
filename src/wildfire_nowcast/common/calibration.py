"""G3's calibration criterion - STRATIFIED calibration error (ADR-020).

WHY THE OLD ONE HAD TO GO
-------------------------
G3's bar was "reliability within ±10 pts". The metric behind it, ``REL`` (the
Murphy calibration term), is measured (A11, ADR-020 §3) to rank a DO-NOTHING null
ABOVE a forecaster with genuine skill: null 0.0050 vs skill 0.0099 on the growth
band, stable across disjoint seed sets and both masks. That is C6.0's failure
condition exactly, and the cause is structural rather than incidental.

Two independent mechanisms, and BOTH have to be closed:

1. **REL is a mean SQUARE, and the bar is stated in POINTS.** A silent forecast
   occupies one bin, at deviation ``base_rate``; squaring turns 7.1 points into
   0.0050, while a forecast that ventures a confident probability and is
   sometimes wrong contributes ``n_b · dev_b²`` from its confident bins. Under a
   squared aggregation, never venturing an opinion is cheap. Under the LINEAR
   aggregation the bar is actually written in - mean ``|forecast − observed|``,
   i.e. ECE - the same measurement reverses: null 0.071 vs skill 0.034. The
   metric did not even have the units of the bar it was adjudicating.
2. **Calibration against the forecast's own bins is satisfied exactly by
   CLIMATOLOGY.** This is not a defect of any particular estimator; it is what
   calibration means. A forecast that says ``p = base_rate`` on every reachable
   cell is perfectly calibrated, carries zero information, and drives ECE and REL
   to zero - it issues only one probability, so its whole reliability diagram is
   one point sitting on the diagonal. At a 1-7% base rate the do-nothing null is
   a crude approximation to that forecast, which is why it wins in the first
   place. Measured, growth band at 3 h, ``null_climatology`` vs ``skillful``::

       members      8       32      128     512
       ECE clim     0.0795  0.0256  0.0110  0.0007
       ECE skill    0.0347  0.0482  0.0484  0.0484
       REL clim     0.0084  0.0012  0.0003  0.00002
       REL skill    0.0105  0.0163  0.0158  0.0154

   Note the trap in the first column: at 8 members the ensemble MEAN is a noisy
   probability estimate, and that noise - uncorrelated with the truth - inflates
   climatology's ECE enough to hide the whole problem. **A cheap ensemble makes a
   calibration statistic look sound.** By 32 members a zero-information forecast
   beats genuine skill on ECE, and by 512 it is indistinguishable from the
   ORACLE. So: no statistic that partitions only by the forecast's own
   probability can survive a null check containing climatology, and
   ``common/null_check/`` now contains one.

THE ESTIMAND, AND WHAT IT FORCES
--------------------------------
G3 asks: *when the ensemble says 30%, does it happen 30% of the time - on the
cells where the outcome was in doubt?* Mechanism 1 says: measure the deviation in
points. Mechanism 2 says: it is not enough to be calibrated against the
forecast's own bins, because a forecast can choose bins so coarse that the claim
is vacuous. The standard strengthening is **calibration within every subgroup of
a declared family** (multicalibration): a climatological forecast is calibrated
marginally and miscalibrated inside any subgroup that carries signal.

So the criterion here is the WORST occupancy-weighted mean absolute deviation
over a small, declared family of subgroup partitions of the scored set:

``calibration_error_bins``
    the classical reliability diagram: strata are the forecast's own probability
    bins. Identical to ECE. **On the path C6 actually uses** - where C6 bins the
    forecast once for its diagram and hands the sufficient statistics to
    :func:`terms_from_strata` - this is asserted BITWISE equal to
    ``eval.metrics._reliability_summary``'s ``ece``, not assumed. The standalone
    :func:`bin_strata` convenience path sums by ``bincount`` where C6 sums in a
    loop, so it agrees to floating point (measured 6.9e-17) and NOT bitwise; that
    weaker guarantee is the one its own test asserts. [A12 correction: this
    paragraph previously claimed bit-identity for both paths, which was true of
    neither statement as written and false of one of them. C0 is satisfied
    because the number that GATES comes from a single binning.]
``calibration_error_frontier``
    strata are distance-from-the-``t0``-frontier rings, in cells. This partition
    is computed from ``x0`` ALONE - the same provenance as the ``growth_band``
    mask itself, never from the outcome - and it is the dominant covariate of
    burn probability, so a forecast with the wrong radial profile is
    miscalibrated in the way that matters for a spread nowcast.
``calibration_error``  (**THE GATE CRITERION**, :data:`GATE_CRITERION_KEY`)
    ``max`` of the two. "Within ±10 pts" should hold on the diagram AND inside
    the decision-relevant strata, not merely on average over a partition the
    forecaster chose for itself.

Measured on the growth band at 3 h, 32 members, 5 seeds (``make null-check``):
oracle **0.0000** exactly, genuine skill **0.0484**, do-nothing null **0.0712**
(= the base rate of the scored set, exactly), climatology **0.1097**. Lower is
better, and the value is a probability in the bar's own units. Every one of those
numbers is FLAT in the member count (climatology moves 0.1094 -> 0.1098 between 8
and 512 members) - which is the property a gate criterion needs and which the
forecast-bin term does not have.

WHY "GROWTH-MASKED" IS THE BAND AND NOT THE OUTCOME
---------------------------------------------------
C6.4 rescued ``best_member_iou`` by dropping leads where truth did not grow. That
move does NOT transfer here, and the reason is about the estimand rather than
about convenience: IoU is genuinely UNDEFINED on empty truth, so dropping those
leads removes nothing. A calibration statistic is perfectly well defined on a
lead where nothing burned - the observed frequency there is 0 - and selecting
leads BY THE OUTCOME biases the comparison it makes. Keep only the leads where
the event happened and every well-calibrated forecast looks under-confident, by
construction, because you selected the realisations that came in above
expectation. **A well-calibrated ensemble would be penalised for the property G3
exists to certify.** So the mask is the ``growth_band`` - unburned cells within
reach of the ``t0`` frontier, computed from ``x0`` alone - and the strata are
likewise from ``x0`` alone. Nothing here conditions on the answer.

THE DECLARED BLIND SPOT
-----------------------
A calibration statistic can always be satisfied by the conditional-mean forecast
on its own family: a model predicting ``p = observed rate of ring r`` inside ring
``r`` scores 0 here while knowing nothing about a particular fire. That is not a
bug to be patched, it is the limit of what "calibrated" can mean, and it is the
same shape as A7's "a range clause cannot catch a units error inside the range".
It is why G3 is a CONJUNCTION: the dispersion half
(``area_dispersion_ratio`` inside :data:`~wildfire_nowcast.common.dispersion.BAR_INTERVAL`)
and the collapse ablation reject exactly the forecasts this term cannot see. That
bar is NAMED rather than spelled, and the reason is this very sentence: it read
``[0.8, 1.2]`` from before ADR-039 replaced the bar until I33 found it, in the
one package whose whole purpose is to be the single implementation of what the
contract adjudicates. Do not read this number alone as capability.

MODEL-AGNOSTIC BY CONSTRUCTION
------------------------------
Everything here consumes a probability field, a label field, a mask and ``x0``.
Nothing in this module can see which model produced the probabilities, whether it
was trained, or what it scored. There is no import of ``model/``, no checkpoint
load and no read of ``runs/`` in this file or in ``tests/test_calibration.py`` -
that is checkable by grep, which is the point. It was written and validated
against constructed cases with known answers before being pointed at anything.

C0: this is the ONE implementation. ``eval/metrics.py`` supplies the fields and
calls in here; it does not restate the arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.common.states import dilate

__all__ = [
    "GATE_CRITERION_KEY",
    "GATE_MASK",
    "SUBGROUP_FAMILIES",
    "Stratum",
    "CalibrationTerms",
    "strata_stats",
    "weighted_abs_deviation",
    "frontier_rings",
    "pool_strata",
    "bin_strata",
    "ring_strata",
    "terms_from_strata",
    "calibration_terms",
    "terms_to_metric_dict",
]

#: The key G3's calibration half is adjudicated on (ADR-020). Written down once,
#: in code, so a downstream table cannot pick a different one by accident - the
#: same guard C6.4 put on ``best_member_iou_shape_masked``.
GATE_CRITERION_KEY = "calibration_error"

#: ...and the mask it must be read under. The ``domain`` value is emitted too,
#: for audit, and is DILUTED by cells nobody was ever uncertain about (masks.py).
#: Quoting the domain number as the G3 number is the mistake this constant exists
#: to make impossible to commit silently.
GATE_MASK = "growth_band"

#: The declared subgroup family. A calibration claim is only as strong as the
#: partitions it is checked on, so the family is NAMED rather than implicit, and
#: enlarging it is a contract change, not an implementation detail.
SUBGROUP_FAMILIES: tuple[str, ...] = ("bins", "frontier")

_ATOL = 1e-9


@dataclass(frozen=True)
class Stratum:
    """Poolable sufficient statistics for one subgroup at one lead.

    Sums rather than means, so strata from many windows and many fires pool
    EXACTLY - and so the deviation is computed once, after pooling. That order
    matters: ``|mean p − mean y|`` inside a stratum of 3 cells is dominated by
    sampling noise even for a perfect forecast, so a per-window number is noisy
    upward while the pooled number is not. Pool first, then deviate.
    """

    key: int
    n: int
    sum_p: float
    sum_y: float

    @property
    def mean_forecast(self) -> float | None:
        return (self.sum_p / self.n) if self.n else None

    @property
    def observed_frequency(self) -> float | None:
        return (self.sum_y / self.n) if self.n else None

    def as_dict(self) -> dict[str, float]:
        return {
            "key": int(self.key),
            "n": int(self.n),
            "sum_p": float(self.sum_p),
            "sum_y": float(self.sum_y),
        }


def strata_stats(index: np.ndarray, p: np.ndarray, y: np.ndarray, n_strata: int) -> list[Stratum]:
    """Sufficient statistics per stratum, for a 1-D integer stratum index.

    ``index`` selects the subgroup of each scored cell; ``p`` is the forecast
    probability and ``y`` the observed 0/1 outcome for the same cell. Empty
    strata are RETAINED with ``n = 0`` so a pooled list has a fixed length and an
    unoccupied subgroup is visibly unoccupied rather than absent.
    """
    idx = np.asarray(index).ravel().astype(np.int64)
    probs = np.asarray(p, dtype=np.float64).ravel()
    obs = np.asarray(y, dtype=np.float64).ravel()
    if not (idx.shape == probs.shape == obs.shape):
        raise ValueError(
            f"index/p/y must be the same length, got {idx.shape}/{probs.shape}/{obs.shape}"
        )
    total = int(n_strata)
    if total < 1:
        raise ValueError(f"n_strata must be >= 1, got {n_strata}")
    if idx.size and (idx.min() < 0 or idx.max() >= total):
        raise ValueError(f"stratum index outside 0..{total - 1}: [{idx.min()}, {idx.max()}]")
    counts = np.bincount(idx, minlength=total).astype(np.int64)
    sum_p = np.bincount(idx, weights=probs, minlength=total)
    sum_y = np.bincount(idx, weights=obs, minlength=total)
    return [
        Stratum(key=k, n=int(counts[k]), sum_p=float(sum_p[k]), sum_y=float(sum_y[k]))
        for k in range(total)
    ]


def weighted_abs_deviation(strata: Sequence[Stratum | Mapping[str, Any]]) -> float | None:
    """``sum_k n_k |mean p_k − mean y_k| / sum_k n_k`` - the deviation, in POINTS.

    ``None`` when nothing was scored: an empty set has no calibration, and
    returning 0.0 there would score an unevaluated forecast as perfect (C-1's
    "unverifiable is a failure, not a pass").

    LINEAR in the deviation, deliberately. The squared form (``REL``) is what
    made silence win: concentrating all of the error into one modest deviation
    beats spreading a smaller mean error across bins that include a few confident
    mistakes. Linear aggregation is also the only form in the units of G3's own
    "±10 pts" bar.
    """
    items = [s if isinstance(s, Stratum) else Stratum(**_stratum_fields(s)) for s in strata]
    total = sum(int(s.n) for s in items)
    if total <= 0:
        return None
    acc = 0.0
    for s in items:
        if s.n:
            acc += s.n * abs((s.sum_p / s.n) - (s.sum_y / s.n))
    return float(acc / total)


def _stratum_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": int(raw.get("key", raw.get("bin_index", 0))),
        "n": int(raw["n"]),
        "sum_p": float(raw["sum_p"]),
        "sum_y": float(raw["sum_y"]),
    }


def frontier_rings(x0: np.ndarray, max_radius: int) -> np.ndarray:
    """Ring index per cell: 0 = burned at ``t0``, ``r`` = ``r`` cells out, cap + 1 = beyond.

    Chebyshev rings grown by successive 8-connected dilations, which is the SAME
    stencil ``eval.masks.growth_band`` uses to build the band - so ring ``r`` for
    ``r`` in ``1..max_radius`` partitions the band exactly, and the "beyond"
    bucket is exactly its complement. Depends on ``x0`` only.
    """
    burned = np.asarray(x0) > 0
    cap = int(max_radius)
    if cap < 1:
        raise ValueError(f"max_radius must be >= 1, got {max_radius}")
    rings = np.full(burned.shape, cap + 1, dtype=np.int32)
    rings[burned] = 0
    reach = burned.copy()
    for r in range(1, cap + 1):
        nxt = dilate(reach, 1)
        rings[nxt & ~reach] = r
        reach = nxt
    return rings


def pool_strata(blocks: Sequence[Sequence[Stratum | Mapping[str, Any]]]) -> list[Stratum]:
    """Sum sufficient statistics across windows/fires, stratum by stratum."""
    acc: dict[int, list[float]] = {}
    for block in blocks:
        for raw in block:
            s = raw if isinstance(raw, Stratum) else Stratum(**_stratum_fields(raw))
            slot = acc.setdefault(s.key, [0.0, 0.0, 0.0])
            slot[0] += s.n
            slot[1] += s.sum_p
            slot[2] += s.sum_y
    return [Stratum(key=k, n=int(acc[k][0]), sum_p=acc[k][1], sum_y=acc[k][2]) for k in sorted(acc)]


@dataclass(frozen=True)
class CalibrationTerms:
    """One lead, one mask: the criterion, its two terms, and its null floor."""

    error: float | None
    bins: float | None
    frontier: float | None
    silent_floor: float | None
    n_scored: int
    n_occupied_bins: int
    n_occupied_rings: int
    unavailable_reason: str = ""

    def check(self) -> CalibrationTerms:
        """The criterion is ``max`` of its terms, and every term is a probability.

        Raises rather than warns, on the ``IouTerms.check`` standard: if this
        stops holding, the metric has changed shape and every caller must be
        re-derived rather than quietly reading a different quantity.
        """
        for name in ("error", "bins", "frontier", "silent_floor"):
            value = getattr(self, name)
            if value is None:
                continue
            if not np.isfinite(value) or not (-_ATOL <= value <= 1.0 + _ATOL):
                raise AssertionError(
                    f"calibration term {name}={value} is not a probability in [0, 1]. "
                    "Every term here is a mean absolute deviation between two "
                    "probabilities, so this cannot happen without a shape error."
                )
        present = [v for v in (self.bins, self.frontier) if v is not None]
        expected = max(present) if present else None
        if expected is None:
            if self.error is not None:
                raise AssertionError(
                    f"calibration_error={self.error} with no subgroup family scored"
                )
        elif self.error is None or not np.isclose(self.error, expected, atol=_ATOL, rtol=0.0):
            raise AssertionError(
                f"calibration_error={self.error} != max(bins={self.bins}, "
                f"frontier={self.frontier})={expected}. The criterion is the WORST "
                "subgroup family by definition (ADR-020); a mismatch means a caller "
                "is reading a different quantity than the one the gate names."
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        """The PER-LEAD keys. Numeric only, deliberately.

        Anything non-numeric (which family gates, which mask, why a term is
        missing) belongs at the block level, not per lead: it is the same string
        three times over, and a string sitting where a per-lead score belongs is
        picked up by ``null_check._flatten`` as an unscoreable metric.
        """
        return {
            GATE_CRITERION_KEY: self.error,
            f"{GATE_CRITERION_KEY}_bins": self.bins,
            f"{GATE_CRITERION_KEY}_frontier": self.frontier,
            f"{GATE_CRITERION_KEY}_silent_floor": self.silent_floor,
            "calibration_n_scored": self.n_scored,
            "calibration_n_occupied_bins": self.n_occupied_bins,
            "calibration_n_occupied_rings": self.n_occupied_rings,
        }

    def block_dict(self) -> dict[str, Any]:
        """The BLOCK-level keys: what gates, under which mask, and what is missing."""
        return {
            "calibration_gate_criterion": GATE_CRITERION_KEY,
            "calibration_gate_mask": GATE_MASK,
            "calibration_unavailable_reason": self.unavailable_reason,
        }


def terms_from_strata(
    bin_strata: Sequence[Stratum | Mapping[str, Any]] | None,
    ring_strata: Sequence[Stratum | Mapping[str, Any]] | None,
    *,
    unavailable_reason: str = "",
) -> CalibrationTerms:
    """Assemble the criterion from POOLED sufficient statistics.

    This is the entry point both the per-window path and the pooling path use, so
    a pooled number is computed by the same code as a single-window one and the
    two cannot drift (C0, one implementation).
    """
    bins = list(bin_strata or [])
    rings = list(ring_strata or [])
    bin_items = [s if isinstance(s, Stratum) else Stratum(**_stratum_fields(s)) for s in bins]
    ring_items = [s if isinstance(s, Stratum) else Stratum(**_stratum_fields(s)) for s in rings]

    dev_bins = weighted_abs_deviation(bin_items) if bin_items else None
    dev_rings = weighted_abs_deviation(ring_items) if ring_items else None
    present = [v for v in (dev_bins, dev_rings) if v is not None]
    error = max(present) if present else None

    n_scored = sum(int(s.n) for s in bin_items) or sum(int(s.n) for s in ring_items)
    total_y = sum(float(s.sum_y) for s in (bin_items or ring_items))
    floor = (total_y / n_scored) if n_scored else None

    reason = unavailable_reason
    if dev_rings is None and not reason:
        reason = (
            "the frontier-stratified term needs x0; without it only the "
            "forecast's own bins are checked, which climatology satisfies trivially"
        )
    return CalibrationTerms(
        error=error,
        bins=dev_bins,
        frontier=dev_rings,
        silent_floor=floor,
        n_scored=int(n_scored),
        n_occupied_bins=sum(1 for s in bin_items if s.n),
        n_occupied_rings=sum(1 for s in ring_items if s.n),
        unavailable_reason=reason,
    ).check()


def bin_strata(
    prob: np.ndarray, truth_event: np.ndarray, mask: np.ndarray, n_bins: int = 10
) -> list[Stratum]:
    """Family A: strata are the forecast's own probability bins (ECE's partition).

    C6 already bins the forecast for its reliability diagram, so the C6 path
    passes those sufficient statistics straight in and this is used only by
    callers that have not. Same edges and same right-open convention as
    ``eval.metrics.reliability`` - asserted by a test that puts values exactly ON
    the bin edges and requires identical occupancy, which is what an off-by-one
    in ``digitize`` would break. The resulting deviation agrees with C6's ``ece``
    to floating point, not bitwise: see the module docstring.
    """
    sel = np.asarray(mask, dtype=bool)
    p_vec = np.asarray(prob, dtype=np.float64)[sel]
    y_vec = np.asarray(truth_event)[sel].astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p_vec, edges[1:-1], right=False), 0, int(n_bins) - 1)
    return strata_stats(idx, p_vec, y_vec, int(n_bins))


def ring_strata(
    prob: np.ndarray, truth_event: np.ndarray, mask: np.ndarray, rings: np.ndarray, n_rings: int
) -> list[Stratum]:
    """Family B: strata are distance-from-frontier rings, from ``x0`` alone."""
    sel = np.asarray(mask, dtype=bool)
    p_vec = np.asarray(prob, dtype=np.float64)[sel]
    y_vec = np.asarray(truth_event)[sel].astype(np.float64)
    return strata_stats(np.asarray(rings)[sel], p_vec, y_vec, int(n_rings))


def calibration_terms(
    prob: np.ndarray,
    truth_event: np.ndarray,
    mask: np.ndarray,
    *,
    rings: np.ndarray | None,
    n_rings: int,
    bins: Sequence[Stratum | Mapping[str, Any]] | None = None,
    n_bins: int = 10,
) -> CalibrationTerms:
    """The criterion for ONE lead, from a probability field and its labels.

    ``prob``/``truth_event``/``mask``/``rings`` are all ``[H, W]``. ``bins`` may be
    supplied by a caller that already binned the forecast (C6 does), so the
    reliability diagram is computed once and this module reads its sufficient
    statistics rather than re-deriving them.
    """
    family_a = bin_strata(prob, truth_event, mask, n_bins) if bins is None else bins
    family_b = None if rings is None else ring_strata(prob, truth_event, mask, rings, int(n_rings))
    return terms_from_strata(family_a, family_b)


def terms_to_metric_dict(terms: CalibrationTerms) -> dict[str, Any]:
    """C6-shaped PER-LEAD metric keys. See :meth:`CalibrationTerms.block_dict`."""
    return terms.as_dict()
