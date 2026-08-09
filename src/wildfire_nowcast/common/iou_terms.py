"""C6.4 — the ``best_member_iou`` SHAPE / SILENCE decomposition (ADR-017).

WHY THIS EXISTS
---------------
``best_member_iou`` is ``max_m mean_k IoU(member m at lead k, truth at lead k)``.
It is therefore a mean over LEADS, and the leads are not alike. Inside the C6
``growth_band`` a large fraction of leads have **zero new truth cells**, and the
IoU of two empty sets is 1.0 by convention (0/0 is not 0 here, and that
convention is defensible in isolation). So a member that predicts NOTHING banks
a full point on every such lead, and a member that ignites one wrong cell there
scores zero. Silence is bankable, and the measured consequence (ADR-017) is that
a model igniting zero cells outranks every model that predicts anything.

This module splits the score so that the bankable part is visible and separable.
It computes THREE quantities from the same per-member-per-lead IoU matrix:

``silence``  (arithmetic term)
    The contribution of empty-truth leads to the reported score.
``shape``    (arithmetic term)
    The contribution of leads where truth actually grew.
    ``silence + shape == best_member_iou`` EXACTLY — this is arithmetic, not a
    redefinition, and :meth:`IouTerms.check` raises if it ever stops holding.
``shape_masked``  (**THE GATE CRITERION**, :data:`GATE_CRITERION_KEY`)
    Empty-truth leads are DROPPED, and the best member is selected on the
    surviving leads. ``None`` when no lead grew — undefined, never 0 and never 1.

WHY THE MASKED VARIANT GATES, AND THE ARITHMETIC TERMS ONLY REPORT
------------------------------------------------------------------
C6.4 permits either. The masked variant is chosen for three reasons, all of
which are properties of the ESTIMAND and none of which reference any model:

1. **Its null floor is exactly 0, which is the minimum of its range.** A model
   that predicts nothing scores 0 on every window that has any growth at all.
   That is the C6.0 property we want from a gate criterion: doing nothing must
   score worst, not merely worse. The arithmetic ``shape`` term also gives the
   null 0, but its own MAXIMUM is ``n_growing / horizon`` rather than 1, so its
   scale is set by the label statistics of the window rather than by the
   forecast — two horizons, or two fires, are not on the same scale.
2. **Member selection is not contaminated.** The arithmetic terms decompose the
   score of the member chosen by the FULL trajectory, i.e. chosen partly on the
   silence bonus. A silent member can win that argmax and drag the shape term to
   0 even when another member captured the shape well. "Best-member mode
   capture" should select on the quantity being reported.
3. **It is a genuine IoU in [0, 1]**, so a reader can interpret it without
   knowing the zero-growth rate of the sample it was computed on.

The arithmetic terms are still emitted, always, because they reconstruct the
REPORTED value exactly and therefore make the pathology auditable: the size of
``silence`` IS the size of the problem, per fire and per horizon.

MODEL-AGNOSTIC BY CONSTRUCTION
------------------------------
Everything here consumes an IoU matrix and a per-lead boolean, both derived from
samples and labels alone. Nothing in this module can see which model produced
the samples, how many parameters it has, or whether it was trained. It was
written and validated against constructed cases with known answers
(``tests/test_iou_terms.py``) before being pointed at any run.

C0: this is the ONE implementation of the decomposition. ``eval/metrics.py``
supplies the IoU matrix it already computes and calls in here;
``sim/replay.py`` computed the same split independently and agrees (asserted in
``tests/test_iou_terms.py::test_agrees_with_simviz_replay_decomposition``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GATE_CRITERION_KEY",
    "REPORTED_ONLY_KEY",
    "IouTerms",
    "jaccard",
    "truth_empty_by_lead",
    "silent_floor",
    "decompose_best_member_iou",
    "decompose_by_horizon",
    "terms_to_metric_dict",
]

#: The key a gate may be adjudicated on (C6.4). Written down once, in code, so a
#: downstream table cannot pick a different one by accident.
GATE_CRITERION_KEY = "best_member_iou_shape_masked"

#: The key that is REPORTED and never a gate criterion (C6.4).
REPORTED_ONLY_KEY = "best_member_iou"

_ATOL = 1e-9


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean fields, with C6's empty-vs-empty convention (1.0).

    Kept here so the convention that CAUSED the pathology has one definition
    that the decomposition is written against. It is bit-identical to
    ``eval.metrics.fuzzy_iou(a, b, 0)`` — asserted by a test, not assumed, on the
    same standard A10 used when re-homing ``split_fingerprint``.
    """
    set_a = np.asarray(a, dtype=bool)
    set_b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(set_a | set_b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(set_a & set_b)) / union


def truth_empty_by_lead(truth_event: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``[L]`` bool: does truth have NO cell at this lead, inside ``mask``?

    This is the label statistic the whole pathology rests on, and it depends on
    labels and the mask only — never on a forecast.
    """
    obs = np.asarray(truth_event, dtype=bool)
    if obs.ndim < 1:
        raise ValueError(f"truth_event must be [L, ...], got shape {obs.shape}")
    if mask is not None:
        sel = np.asarray(mask, dtype=bool)
        return np.array([not np.any(obs[k] & sel) for k in range(obs.shape[0])], dtype=bool)
    return np.array([not np.any(obs[k]) for k in range(obs.shape[0])], dtype=bool)


def silent_floor(truth_empty: np.ndarray, horizon: int | None = None) -> float:
    """What a forecast that predicts NOTHING scores on this window.

    Depends only on the labels, so it is the same number for every model — which
    is the point. A metric whose null value is not zero must publish that value
    beside every number it produces, or the numbers read as skill.
    """
    empty = np.asarray(truth_empty, dtype=bool)
    n_lead = int(empty.size) if horizon is None else int(horizon)
    if not 1 <= n_lead <= int(empty.size):
        raise ValueError(f"horizon {horizon} outside 1..{empty.size}")
    return float(np.count_nonzero(empty[:n_lead]) / n_lead)


@dataclass(frozen=True)
class IouTerms:
    """One window (or one pooled block), one mask, one horizon."""

    horizon: int
    undecomposed: float
    silence: float
    shape: float
    shape_masked: float | None
    silent_floor: float
    best_member: int
    best_member_masked: int | None
    n_empty_leads: int
    n_growing_leads: int
    n_members: int

    def check(self) -> IouTerms:
        """The split is ARITHMETIC. If it stops reconstructing, say so loudly."""
        total = self.silence + self.shape
        if not np.isclose(total, self.undecomposed, atol=_ATOL, rtol=0.0):
            raise AssertionError(
                f"decomposition does not reconstruct best_member_iou: "
                f"{self.silence} + {self.shape} = {total} != {self.undecomposed}. "
                "The split is arithmetic; a mismatch means the metric changed shape "
                "and every caller of this module must be re-derived (C6.4)."
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_h": self.horizon,
            "best_member_iou": self.undecomposed,
            "best_member_iou_silence": self.silence,
            "best_member_iou_shape": self.shape,
            GATE_CRITERION_KEY: self.shape_masked,
            "best_member_iou_silent_floor": self.silent_floor,
            "best_member": self.best_member,
            "best_member_masked": self.best_member_masked,
            "n_empty_leads": self.n_empty_leads,
            "n_growing_leads": self.n_growing_leads,
            "n_members": self.n_members,
        }


def decompose_best_member_iou(
    per_member_lead: np.ndarray,
    truth_empty: np.ndarray,
    *,
    horizon: int | None = None,
) -> IouTerms:
    """Decompose C6's best-member IoU for one window at one horizon.

    ``per_member_lead`` is ``[M, L]``: ``IoU(member m at lead k, truth at lead
    k)`` inside whatever mask the caller scored. ``truth_empty`` is ``[L]`` from
    :func:`truth_empty_by_lead` under the SAME mask — passing a mismatched pair
    is the one way to get a wrong answer here, so the shapes are checked.

    ``horizon`` scores leads ``0..horizon-1``; the default is all of them. This
    is exactly what scoring a shorter window would have produced, which is what
    C6.2's per-horizon adjudication rule needs (ADR-015).
    """
    per = np.asarray(per_member_lead, dtype=np.float64)
    if per.ndim != 2:
        raise ValueError(f"per_member_lead must be [n_members, n_lead], got {per.shape}")
    n_members, n_lead = (int(v) for v in per.shape)
    if n_members < 1 or n_lead < 1:
        raise ValueError(f"per_member_lead must be non-empty, got {per.shape}")
    empty = np.asarray(truth_empty, dtype=bool)
    if empty.shape != (n_lead,):
        raise ValueError(
            f"truth_empty must be [n_lead]={n_lead}, got {empty.shape}. It must be computed "
            "under the SAME mask as the IoU matrix, or the split is meaningless."
        )
    span = n_lead if horizon is None else int(horizon)
    if not 1 <= span <= n_lead:
        raise ValueError(f"horizon {horizon} outside 1..{n_lead}")

    per_h = per[:, :span]
    empty_h = empty[:span]
    grew_h = ~empty_h

    trajectory = per_h.mean(axis=1)
    best = int(np.argmax(trajectory))
    row = per_h[best]

    if grew_h.any():
        masked_trajectory = per_h[:, grew_h].mean(axis=1)
        best_masked: int | None = int(np.argmax(masked_trajectory))
        shape_masked: float | None = float(masked_trajectory[best_masked])
    else:
        # Every lead is empty-vs-something. There is no shape to capture, so the
        # gate criterion is UNDEFINED. Returning 0.0 would punish every model for
        # a property of the labels; returning 1.0 is the bug this clause exists
        # to remove. None propagates and forces the pooler to declare the count.
        best_masked = None
        shape_masked = None

    return IouTerms(
        horizon=span,
        undecomposed=float(trajectory[best]),
        silence=float(row[empty_h].sum() / span),
        shape=float(row[grew_h].sum() / span),
        shape_masked=shape_masked,
        silent_floor=float(np.count_nonzero(empty_h) / span),
        best_member=best,
        best_member_masked=best_masked,
        n_empty_leads=int(np.count_nonzero(empty_h)),
        n_growing_leads=int(np.count_nonzero(grew_h)),
        n_members=n_members,
    ).check()


def decompose_by_horizon(
    per_member_lead: np.ndarray, truth_empty: np.ndarray
) -> list[IouTerms]:
    """:func:`decompose_best_member_iou` at every horizon ``1..L``.

    Entry ``H-1`` is what a length-``H`` window would have scored, so a 1/2/3 h
    table comes out of one pass and is identical BY CONSTRUCTION rather than by
    hope. The last entry equals the full-window decomposition.
    """
    n_lead = int(np.asarray(per_member_lead).shape[1])
    return [
        decompose_best_member_iou(per_member_lead, truth_empty, horizon=h)
        for h in range(1, n_lead + 1)
    ]


def terms_to_metric_dict(by_horizon: list[IouTerms]) -> dict[str, Any]:
    """Flatten a per-horizon decomposition into C6-shaped metric keys."""
    if not by_horizon:
        raise ValueError("terms_to_metric_dict needs at least one horizon")
    last = by_horizon[-1]
    return {
        "best_member_iou_silence": last.silence,
        "best_member_iou_shape": last.shape,
        GATE_CRITERION_KEY: last.shape_masked,
        "best_member_iou_silent_floor": last.silent_floor,
        "best_member_iou_n_empty_leads": last.n_empty_leads,
        "best_member_iou_n_growing_leads": last.n_growing_leads,
        "best_member_iou_silence_by_horizon": [t.silence for t in by_horizon],
        "best_member_iou_shape_by_horizon": [t.shape for t in by_horizon],
        f"{GATE_CRITERION_KEY}_by_horizon": [t.shape_masked for t in by_horizon],
        "best_member_iou_silent_floor_by_horizon": [t.silent_floor for t in by_horizon],
        "best_member_iou_gate_criterion": GATE_CRITERION_KEY,
    }
