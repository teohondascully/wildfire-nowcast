"""C6.0 — EVERY METRIC MUST BEAT A DO-NOTHING NULL (ADR-017).

    .venv/bin/python -m wildfire_nowcast.common.null_check
    make null-check

THE RULE
--------
Before any metric adjudicates any gate, score a model that predicts NOTHING.
**If the null wins, the metric is broken — not the model.**

Three pathologies of exactly this shape have already been found, and each was
found LATE, by hand, after it had already influenced a decision:

* Brier-fitting drove the wind-ellipse to optimal SILENCE — it ignited zero
  cells while truth grew 782 (ADR-011).
* ``dispersion_ratio`` scores a COLLAPSED ensemble at 1.000 and a healthy one at
  1.051, so the original G3 bar would have PASSED collapse (ADR-011).
* ``best_member_iou`` banks empty-vs-empty as IoU 1.0, so a zero-ignition model
  outranks a trained one at every horizon (ADR-017).

Our scoring rules systematically reward not predicting. That is R14's persistence
attractor reappearing in the INSTRUMENTS rather than in the model. This module
makes the check mechanical and runnable, so a fourth instance is found by CI
rather than by an argument about a gate.

HOW IT IS MODEL-BLIND
---------------------
Every forecaster here is synthesised from LABELS ALONE. There is no checkpoint,
no import of ``model/``, no run directory. The harness cannot see which model
"ours" is, because none of them is. The scenario is generated, not sampled from
a fire we chose, and the whole file was written and validated against
constructed cases with known answers before it was pointed at any number.

TWO VERDICTS, NOT ONE [A12, ADR-022 (1)]
----------------------------------------
Every metric here gets **two independent verdicts**, because C6.0 was asking two
different questions through one answer and that is what made the instrument flip.

``verdict`` — THE COMPARISON. *Can any forecaster with genuine skill beat a
    degenerate one under this metric, on this sample?* Answered by ranking
    against a reference model (:data:`SKILL_REFERENCES`, the ORACLE, the collapse
    ablation). Tiers ``BROKEN`` / ``BLIND`` / ``SILENCE_FAVOURING`` / ``ok``.

``capture_verdict`` — THE ZERO-CAPTURE AXIOM. *Does this metric pay a strictly
    positive score to a forecast that claims nothing?* Answered with no reference
    model and no threshold. Tiers ``PAYS_FOR_NOTHING`` / ``ok``.

Measured on ``best_member_iou`` in the growth band, both are TRUE and they
disagree: the comparison says ``ok`` (``skillful_calibrated`` 0.40321 beats the
null's 0.33333) while the axiom says ``PAYS_FOR_NOTHING`` (the null is paid
0.33333 for an empty forecast). ADR-017's finding lives in the SECOND one, which
is why collapsing them into a single string silenced it. A metric may be
beatable-in-principle and still carry a term any model can bank without
predicting anything; those are not the same property and one string cannot hold
both.

Only the axiom and the hard comparison tiers can void a gate — see
:attr:`MetricVerdict.is_failure`.

THE COMPARISON TIERS
--------------------
``BROKEN``  (hard)
    A degenerate model ranks at least as well as the best forecast the metric
    admits — the ORACLE for a skill metric, the healthy ensemble for a spread
    metric. No defensible score can do this. There is no reading of the data
    under which not predicting is as good as being right.
``BLIND``  (hard, spread metrics only)
    The metric cannot SEPARATE a collapsed ensemble from a healthy one: the gap
    it opens between them is inside its own seed-to-seed noise. Which one it
    then prefers is a coin flip, and measured here it does flip with the seed —
    ``dispersion_ratio`` prefers COLLAPSE on 4 of 5 seeds and health on the 5th,
    all within 1%. ADR-011's own word for it is "a blind instrument"; this is
    that word made testable, and it is the stable finding where the direction of
    a 1% preference is not.
``SILENCE_FAVOURING``  (reporting)
    A degenerate model ranks above ``skillful`` — genuine but imperfect skill,
    partial recall plus realistic false alarms — while still ranking below the
    best admissible forecast. This is NOT automatically a bug: a proper score at
    a 1% base rate legitimately prefers silence to a sub-coin-flip predictor
    (R14, C6.2). It is a C-1 reporting gap: printed unconditionally, non-zero
    under ``--strict``, never a silent pass. Making it hard would be a false
    positive that teaches people to disable the check, which is the argument
    that has kept C1.6 off the hard tier twice.
THE ZERO-CAPTURE AXIOM, AND WHY A COMPARISON COULD NOT SETTLE THIS [A12]
------------------------------------------------------------------------
Every comparison tier above rests on a REFERENCE MODEL, and A12 established that
a comparison cannot adjudicate a metric carrying an ADDITIVE DEGENERATE TERM.
Measured here (growth band, 37 windows, seeds 0/1/2, means):

===================  =======  =======  =======  =======  =======  =======
model                    M=4      M=8     M=16     M=32     M=64    M=128
===================  =======  =======  =======  =======  =======  =======
null_zero_ignition   0.33333  0.33333  0.33333  0.33333  0.33333  0.33333
null_empty           0.33333  0.33333  0.33333  0.33333  0.33333  0.33333
skillful             0.28143  0.30977  0.32646  0.33985  0.35878  0.39181
skillful_calibrated  0.34760  0.37003  0.38947  0.40321  0.42099  0.44116
===================  =======  =======  =======  =======  =======  =======

The null's score is INVARIANT — it is the zero-growth lead fraction (1/3 here),
a property of the LABELS, identical to the last digit at every M and every seed.
The reference's score is MONOTONE IN THE MEMBER COUNT, because best-of-M is a
maximum over members. So "does the null beat genuine skill" flips from YES to NO
by turning a knob that has nothing to do with the metric: ``skillful`` crosses the
null between M=16 and M=32, and ``skillful_calibrated`` is already above it at
M=4. That is exactly how the verdict moved between A11 (M=8, one reference) and
A12 (M=32, a second and stronger reference). **Neither verdict was evidence about
the metric.** ``mean_member_iou`` is a MEAN rather than a maximum, is therefore
flat in M (0.2126 -> 0.2200), and never went quiet — which is why precisely the
best-of-M family fell silent and nothing else did.

What IS evidence about the metric needs no reference and no threshold: a forecast
that claims nothing captures nothing, so a capture metric must pay it the minimum
of its range. Anything above that minimum is credit paid for an empty forecast,
and **it is bankable by any model** — which is the mechanism simviz measured
directly on one real window (kernel 0.020 vs ellipse 0.833 because 9/24 members
predicted nothing) and the reason the do-nothing floor sat at 0.464/0.326/0.219
above every kernel. Measured here, ``null_empty``, every seed and M in {4, 32}:

=========================  ===========  ===========
metric                          domain  growth_band
=========================  ===========  ===========
best_member_iou              0.0 exact      0.33333
best_member_iou_growth          0.33333      0.33333
best_member_iou_tolerant     0.0 exact      0.33333
mean_member_iou              0.0 exact      0.33333
best_member_iou_shape        0.0 exact    0.0 exact
**shape_masked** (gate)      0.0 exact    0.0 exact
resolution_{1,2,3}h          0.0 exact    0.0 exact
=========================  ===========  ===========

Three properties of that table are why the axiom is the right instrument.
(a) The comparison exonerated ``best_member_iou_tolerant`` at A11; the axiom does
not. (b) It flags ``best_member_iou_growth`` on the DOMAIN mask, where
``best_member_iou`` is clean — a case no growth-band comparison looks at.
(c) The line is EXACT equality with zero: every sound metric lands on exactly
``0.0``, so no tolerance has to be justified and C-3 has no constant to bind to.

``ZERO_CLAIM`` is ``null_empty`` and NOT persistence, deliberately: on the domain
mask persistence scores ``best_member_iou`` 0.857 by reproducing the already-
burned region, which is capture it EARNED. Using it as the axiom's subject would
manufacture a false positive on every metric — the failure mode that teaches
people to disable a check.

The axiom is confined to ``higher_is_better`` metrics because for an ERROR metric
a silent forecast may legitimately score well at a 1% base rate — that is R14,
not a defect, and is what the reporting tier is for.

WHY THIS IS NOT "COMPARE AGAINST THE MODELS UNDER TEST" [A12]
--------------------------------------------------------------
ADR-022 (1) frames the second question as *does the metric rank a null above the
MODELS UNDER TEST?*. That framing is right about the substance and cannot be
implemented literally here, for two reasons that are the same reason:

* It would require this harness to score a checkpoint, destroying the
  model-blindness that licenses its verdicts and that lets a gate's instrument be
  fixed BEFORE the result exists.
* A verdict computed that way would move as the model trains: the same metric
  would be "broken" on Monday and "fine" on Friday with no change to the metric.
  That is the M-knob defect again, wearing different clothes.

The axiom is the model-blind form of the same claim, and it is strictly stronger:
it does not ask whether OUR model happens to sit below the floor today, it
measures the size of the floor any model can bank. 0.33333 of bankable credit is
a fact about the metric; whether a given kernel is above or below it is a fact
about the kernel.

WHAT COUNTS AS A NULL — WIDENED [ADR-020]

A gate-eligible metric must avoid the hard tier. A metric the contract has
already quarantined (``dispersion_ratio`` by C6.1, ``best_member_iou`` by C6.4)
is EXPECTED to fail and is reported, not fatal — those are the POSITIVE CONTROLS
that prove this harness has teeth. If they ever come back clean, suspect the
harness before believing the good news.

-----------------------------------------
C6.0's text names a model that predicts nothing, and the two zero-ignition nulls
are exactly that. But the property those nulls exploit is not silence, it is
carrying NO INFORMATION ABOUT THE SITUATION; at a 1-7% base rate, silence is
merely a *bad approximation* to the climatological forecast. :class:`Climatology`
is the sharp form — it knows where the fire is and what fraction of reachable
cells burn on average, and nothing else — and it is the decisive test of any
CALIBRATION statistic, because it is perfectly calibrated by construction.
Adding it found that ``ece_*`` and ``reliability_*`` are both beaten by a model
with zero information, which no do-nothing null could show.

Two consequences worth reading before trusting any verdict here:

* **Member count changes verdicts.** See :data:`DEFAULT_MEMBERS`. At 8 members
  the ensemble mean's own sampling noise inflates a no-information forecast's
  ECE by ~0.08 and hides the pathology entirely. The masking direction is the
  flattering one, which is why the default is not the cheap one.
* **The null floor of a metric is a LABEL STATISTIC.** Metrics marked
  :data:`LABEL_STATISTIC` must be identical for every forecaster and seed, and
  the harness raises a PROBLEM if they are not — that is how a model-dependent
  quantity leaking into a supposedly model-free floor gets caught.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.calibration import GATE_CRITERION_KEY as CALIBRATION_METRIC
from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY as GATE_METRIC
from wildfire_nowcast.common.iou_terms import REPORTED_ONLY_KEY

__all__ = [
    "GATE_METRIC",
    "CALIBRATION_METRIC",
    "Climatology",
    "fit_climatology",
    "CalibratedSkillful",
    "fit_calibrated_skillful",
    "forecasters_for",
    "CLIMATOLOGY",
    "SKILLFUL_CALIBRATED",
    "SKILL_REFERENCES",
    "MetricSpec",
    "C6_METRICS",
    "Window",
    "Forecaster",
    "FORECASTERS",
    "DEGENERATE",
    "SKILLFUL",
    "ORACLE",
    "synthetic_windows",
    "windows_from_tensor",
    "MetricVerdict",
    "NullCheckReport",
    "run_null_check",
    "format_report",
    "degenerates_for",
    "strongest_reference",
    "NULLS",
    "COLLAPSE",
    "DETERMINISTIC",
    "ZERO_CLAIM",
    "VERDICT_PAYS_FOR_NOTHING",
    "CAPTURE_NOT_APPLICABLE",
    "compare",
    "NOISE_FLOOR_SD",
    "DEFAULT_SEEDS",
    "DEFAULT_MEMBERS",
    "main",
]

VERDICT_OK = "ok"
VERDICT_BROKEN = "BROKEN"
VERDICT_BLIND = "BLIND"
VERDICT_SILENCE_FAVOURING = "SILENCE_FAVOURING"
VERDICT_PAYS_FOR_NOTHING = "PAYS_FOR_NOTHING"
VERDICT_UNDECIDABLE = "undecidable"

#: The zero-capture axiom does not apply to this metric's orientation. Reported
#: as its own token rather than as ``ok``, so "the axiom passed" and "the axiom
#: was never evaluated" can never be read as the same result — the absent-clause-
#: reads-as-passing-clause failure this project has now hit four times.
CAPTURE_NOT_APPLICABLE = "n/a"

#: Metric directions. ``target`` means "closer to ``target`` is better" — the
#: shape that hid the ``dispersion_ratio`` pathology, because a collapsed
#: ensemble sat exactly ON the target.
HIGHER = "higher_is_better"
LOWER = "lower_is_better"
TARGET = "closer_to_target_is_better"
DIAGNOSTIC = "diagnostic"  #: reported, never ranked
LABEL_STATISTIC = "label_statistic"  #: must be IDENTICAL for every model


#: What KIND of degeneracy a metric can be fooled by, which decides which
#: degenerate models it is tested against.
#:
#: This scoping is a real judgement and is argued here rather than buried. A
#: COLLAPSED ensemble legitimately beats an over-dispersed one on a proper score
#: — that is a sharpness trade, not a pathology — so testing Brier against
#: collapse produces a verdict about dispersion tuning and says nothing about
#: silence. But a SPREAD metric that prefers collapse is measuring the opposite
#: of what it is named after, which is exactly ADR-011's `dispersion_ratio`
#: finding. So: every metric is tested against the SILENCE nulls; only spread
#: metrics are additionally tested against collapse.
FAMILY_SKILL = "skill"
FAMILY_SPREAD = "spread"


@dataclass(frozen=True)
class MetricSpec:
    """How one C6 metric is read, and whether it may decide anything.

    ``gate_eligible`` records the CONTRACT's current ruling, not a judgement made
    here: C6.1 rules ``dispersion_ratio`` out, C6.4 rules ``best_member_iou``
    out, and G2/G3 name the rest. Recording it in code means a table cannot
    quietly adjudicate on a quarantined metric. Changing an entry is an
    maintainer ruling, not an infra edit.
    """

    direction: str
    gate_eligible: bool
    target: float | None = None
    family: str = FAMILY_SKILL
    quarantined_by: str = ""
    note: str = ""


#: Every numeric metric C6 emits, with its orientation. **A metric absent from
#: this table cannot be null-checked, and the harness says so rather than
#: skipping it** — an unregistered metric is an unchecked metric, which is the
#: C-2 lesson one level down.
C6_METRICS: dict[str, MetricSpec] = {
    # --- named by G2 / ADR-018 as adjudicating criteria --------------------
    "brier_1h": MetricSpec(LOWER, True),
    "brier_2h": MetricSpec(LOWER, True),
    "brier_3h": MetricSpec(LOWER, True),
    "arrival_crps": MetricSpec(LOWER, True),
    GATE_METRIC: MetricSpec(
        HIGHER, True, note="C6.4 gate criterion: empty-truth leads dropped; null floor is 0."
    ),
    # --- named by G3 / ADR-011 ---------------------------------------------
    "area_dispersion_ratio": MetricSpec(
        TARGET,
        True,
        target=1.0,
        family=FAMILY_SPREAD,
        note="C6.1 makes this the G3 criterion instead of dispersion_ratio.",
    ),
    # --- named by G3 / ADR-020 as the CALIBRATION criterion -----------------
    # Per lead, because G3 reports reliability at 1/2/3 h and C6.2 [v2.8]
    # adjudicates each horizon separately.
    **{
        f"{CALIBRATION_METRIC}_{h}h": MetricSpec(
            LOWER,
            True,
            note="ADR-020 G3 calibration criterion: max(forecast-bin deviation, "
            "frontier-ring deviation) on the growth-masked decision set, in "
            "probability POINTS. Null floor = the base rate exactly; oracle = 0.",
        )
        for h in (1, 2, 3)
    },
    **{
        f"{CALIBRATION_METRIC}_bins_{h}h": MetricSpec(
            LOWER, False, note="term A of the criterion; identical to ece_*h."
        )
        for h in (1, 2, 3)
    },
    **{
        f"{CALIBRATION_METRIC}_frontier_{h}h": MetricSpec(
            LOWER,
            False,
            note="term B: calibration inside distance-from-frontier strata, the "
            "partition the forecast did NOT choose. This is the term climatology "
            "fails and every forecast-bin statistic passes.",
        )
        for h in (1, 2, 3)
    },
    **{
        f"{CALIBRATION_METRIC}_silent_floor_{h}h": MetricSpec(
            LABEL_STATISTIC,
            False,
            note="what a forecast that predicts nothing scores here = the base rate "
            "of the scored set. A property of the LABELS; the harness asserts it is "
            "identical for every forecaster and seed.",
        )
        for h in (1, 2, 3)
    },
    # --- G3's FORMER calibration criterion, retired by ADR-020 -------------
    "reliability_1h": MetricSpec(
        LOWER,
        False,
        quarantined_by="ADR-020 (G3 rewritten)",
        note="REL is a mean SQUARE while the bar is stated in POINTS, and it is "
        "satisfied exactly by climatology. G3's calibration half is now "
        f"{CALIBRATION_METRIC}_*h. REPORTED, never a gate criterion.",
    ),
    "reliability_3h": MetricSpec(
        LOWER, False, quarantined_by="ADR-020 (G3 rewritten)", note="see reliability_1h."
    ),
    # --- quarantined by contract: expected to fail, kept as positive controls
    "dispersion_ratio": MetricSpec(
        TARGET,
        False,
        target=1.0,
        family=FAMILY_SPREAD,
        quarantined_by="C6.1 (ADR-011)",
        note="algebraically a calibration statistic on a binary field; scores COLLAPSED at "
        "1.000 and healthy at 1.051, i.e. anti-correlated with the thing G3 measures.",
    ),
    REPORTED_ONLY_KEY: MetricSpec(
        HIGHER,
        False,
        quarantined_by="C6.4 (ADR-017)",
        note="empty-vs-empty is IoU 1.0, so silence is bankable. REPORTED, never a gate.",
    ),
    # --- reported diagnostics; no gate names them --------------------------
    "arrival_crps_active": MetricSpec(LOWER, False, note="reported beside arrival_crps."),
    "mean_member_iou": MetricSpec(
        HIGHER, False, quarantined_by="C6.4 (ADR-017)", note="same empty-vs-empty convention."
    ),
    "best_member_iou_tolerant": MetricSpec(
        HIGHER, False, quarantined_by="C6.4 (ADR-017)", note="same empty-vs-empty convention."
    ),
    "best_member_iou_growth": MetricSpec(
        HIGHER, False, quarantined_by="C6.4 (ADR-017)", note="same empty-vs-empty convention."
    ),
    "best_member_iou_shape": MetricSpec(
        HIGHER,
        False,
        note="the arithmetic shape term. Comparable between models on identical windows, but "
        "its maximum is n_growing/horizon rather than 1, so it is reported, not gated.",
    ),
    "best_member_iou_silence": MetricSpec(
        DIAGNOSTIC, False, note="how much of the reported score came from saying nothing."
    ),
    "best_member_iou_silent_floor": MetricSpec(
        LABEL_STATISTIC,
        False,
        note="what a predicts-nothing forecast scores on this sample. A property of the "
        "LABELS: it must be identical for every model, and the harness asserts it.",
    ),
    "ece_1h": MetricSpec(
        LOWER,
        False,
        quarantined_by="ADR-020 (climatology)",
        note="the LINEAR calibration statistic. It beats the silence nulls, and it is "
        "beaten by climatology, which scores ~0 on it by construction — that is the "
        "whole reason the gate criterion needs a second subgroup family.",
    ),
    "ece_2h": MetricSpec(LOWER, False, quarantined_by="ADR-020 (climatology)"),
    "ece_3h": MetricSpec(LOWER, False, quarantined_by="ADR-020 (climatology)"),
    "reliability_2h": MetricSpec(LOWER, False, quarantined_by="ADR-020 (G3 rewritten)"),
    "resolution_1h": MetricSpec(HIGHER, False, note="Murphy decomposition term; reported."),
    "resolution_2h": MetricSpec(HIGHER, False),
    "resolution_3h": MetricSpec(HIGHER, False),
}


# --------------------------------------------------------------------------
# windows — labels only
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One evaluation window: the state at ``t0`` and the ``L`` labels after it."""

    t0: int
    x0: np.ndarray  # uint8 [H, W]
    truth: np.ndarray  # uint8 [L, H, W]


def _ellipse(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2 <= 1.0


def synthetic_windows(
    *,
    n_hours: int = 40,
    horizon_h: int = 3,
    shape: tuple[int, int] = (40, 40),
    p_grow: float = 0.55,
    seed: int = 20260808,
) -> tuple[list[Window], dict[str, Any]]:
    """A label sequence with a KNOWN zero-growth rate, and windows over it.

    Deliberately synthetic rather than a real fire: the pathology's magnitude is
    a function of the zero-growth rate, so that rate must be a declared input,
    not something inherited from whichever fire happened to be on disk. The
    generated statistic is returned alongside so the report states it.

    Growth is an anisotropic ellipse with a drifting centre, made absorbing by
    construction (C1.1's guarantee, reproduced here so the fixture is legal
    state data rather than an arbitrary boolean field).
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    grow = rng.random(n_hours) < float(p_grow)
    steps = np.cumsum(grow.astype(np.float64))
    # Radius AND centre advance only on a growth hour, so a dormant hour adds
    # bitwise zero cells. Drifting the centre every hour instead would make every
    # hour a growth hour and quietly destroy the property being tested — the
    # zero-growth rate is the whole point of this fixture.
    radius = 3.0 + steps * 1.10
    cy = height / 2.0 - steps * 0.32
    cx = width / 2.0 + steps * 0.46

    ever = np.zeros((n_hours, height, width), dtype=bool)
    prev = np.zeros((height, width), dtype=bool)
    for t in range(n_hours):
        prev = prev | _ellipse(shape, cy[t], cx[t], radius[t], radius[t] * 0.72)
        ever[t] = prev

    state = np.zeros((n_hours, height, width), dtype=np.uint8)
    for t in range(n_hours):
        older = ever[t - 1] if t else np.zeros_like(ever[t])
        state[t] = np.where(ever[t] & ~older, 1, np.where(ever[t], 2, 0)).astype(np.uint8)

    windows = [
        Window(t0=t0, x0=state[t0], truth=state[t0 + 1 : t0 + 1 + horizon_h])
        for t0 in range(n_hours - horizon_h)
        if state[t0].any()
    ]
    total_leads = sum(int(w.truth.shape[0]) for w in windows)
    zero_growth = sum(
        1
        for w in windows
        for k in range(w.truth.shape[0])
        if not np.any((w.truth[k] > 0) & ~(w.x0 > 0))
    )
    stats = {
        "source": "synthetic_windows",
        "n_windows": len(windows),
        "n_leads": total_leads,
        "grid_shape": list(shape),
        "horizon_h": horizon_h,
        "zero_growth_lead_fraction": (zero_growth / total_leads) if total_leads else None,
        "seed": seed,
    }
    return windows, stats


def windows_from_tensor(
    tensor_path: str | Path, *, horizon_h: int = 3, stride: int = 1, max_windows: int | None = None
) -> tuple[list[Window], dict[str, Any]]:
    """Windows read from any C1 store — the C4 synthetic fire or a real fire."""
    from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: PLC0415

    ds = open_tensor(Path(tensor_path))
    state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    n_t = int(state.shape[0])
    windows: list[Window] = []
    for t0 in range(0, n_t - horizon_h, max(1, int(stride))):
        if not state[t0].any():
            continue
        windows.append(Window(t0=t0, x0=state[t0], truth=state[t0 + 1 : t0 + 1 + horizon_h]))
        if max_windows is not None and len(windows) >= max_windows:
            break
    total_leads = sum(int(w.truth.shape[0]) for w in windows)
    zero_growth = sum(
        1
        for w in windows
        for k in range(w.truth.shape[0])
        if not np.any((w.truth[k] > 0) & ~(w.x0 > 0))
    )
    stats = {
        "source": str(tensor_path),
        "n_windows": len(windows),
        "n_leads": total_leads,
        "grid_shape": list(state.shape[1:]),
        "horizon_h": horizon_h,
        "zero_growth_lead_fraction": (zero_growth / total_leads) if total_leads else None,
    }
    return windows, stats


# --------------------------------------------------------------------------
# forecasters — synthesised from labels, never from a model
# --------------------------------------------------------------------------

Forecaster = Callable[[Window, int, np.random.Generator], np.ndarray]


def _as_samples(member_burned: np.ndarray) -> np.ndarray:
    """Boolean ``[M, L, H, W]`` -> C5 ``uint8`` fire_state (1 = burned)."""
    return np.where(np.asarray(member_burned, dtype=bool), 1, 0).astype(np.uint8)


def null_zero_ignition(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """PERSISTENCE. Ignites exactly zero new cells. ADR-017's null of record."""
    burned = window.x0 > 0
    n_lead = int(window.truth.shape[0])
    return _as_samples(np.broadcast_to(burned, (n_members, n_lead, *burned.shape)))


def null_empty(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Predicts NOTHING AT ALL — not even the fire that is already burning."""
    n_lead = int(window.truth.shape[0])
    return np.zeros((n_members, n_lead, *window.x0.shape), dtype=np.uint8)


def oracle(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Predicts the truth exactly. Nothing may outrank this on any metric."""
    truth = np.asarray(window.truth) > 0
    return _as_samples(np.broadcast_to(truth, (n_members, *truth.shape)))


def _growth_candidates(window: Window) -> tuple[np.ndarray, np.ndarray]:
    """``(true new cells at the final lead, plausible-but-wrong cells)``."""
    from wildfire_nowcast.common.states import dilate  # noqa: PLC0415

    burned = window.x0 > 0
    truth_final = (np.asarray(window.truth[-1]) > 0) & ~burned
    reach = dilate(burned, 3) & ~burned
    return truth_final, reach & ~truth_final


def _skillful_members(
    window: Window,
    n_members: int,
    rng: np.random.Generator,
    *,
    recall: float,
    false_alarm: float,
    shared_latent: bool,
) -> np.ndarray:
    """Genuine but imperfect skill: partial recall plus realistic false alarms.

    ``shared_latent`` is the CLAUDE.md correlated-innovation structure: one
    per-member multiplier scales the whole increment, so members are different
    SCENARIOS. With it off, members differ only by independent per-pixel noise —
    individually calibrated, no scenario spread — which is the G3 ablation and
    the degenerate case ``dispersion_ratio`` cannot see.
    """
    burned = window.x0 > 0
    hits, misses = _growth_candidates(window)
    n_lead = int(window.truth.shape[0])
    out = np.zeros((n_members, n_lead, *burned.shape), dtype=bool)
    for m in range(n_members):
        latent = float(rng.uniform(0.45, 1.55)) if shared_latent else 1.0
        take_hit = rng.random(burned.shape) < min(1.0, recall * latent)
        take_miss = rng.random(burned.shape) < min(1.0, false_alarm * latent)
        claimed = (hits & take_hit) | (misses & take_miss)
        for k in range(n_lead):
            frac = (k + 1) / n_lead
            phase = rng.random(burned.shape) < frac
            out[m, k] = burned | (claimed & phase)
        # absorbing in time, per C1.1
        for k in range(1, n_lead):
            out[m, k] |= out[m, k - 1]
    return _as_samples(out)


def skillful(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Informative and imperfect: recall 0.55, false-alarm 0.10, shared latent.

    **Read this before trusting a verdict that rests on it.** Those two constants
    were chosen against a generated fixture whose growth-band base rate is ~2%.
    On a real fire the base rate is far lower — Kincade's growth band is 0.24% at
    1 h — so a flat 10% false-alarm rate over a 3-cell reach makes this forecaster
    over-predict by a factor of tens. It is informative but grossly OVER-CONFIDENT
    off-fixture, and a proper score is supposed to rank an over-confident forecast
    below silence. See :func:`fit_calibrated_skillful`.
    """
    return _skillful_members(
        window, n_members, rng, recall=0.55, false_alarm=0.10, shared_latent=True
    )


@dataclass(frozen=True)
class CalibratedSkillful:
    """Genuine skill whose TOTAL predicted growth matches the observed growth.

    **This exists because the harness was not holding its own reference model to
    the project's own rule.** C6.2 [v2.8] says a baseline's scale must be
    CALIBRATED TO REPRODUCE OBSERVED MEAN GROWTH, never left free — an
    uncalibrated baseline is not a distinct baseline, it is a broken one. The
    null check's ``skillful`` reference was never held to that, and off its
    fixture it over-predicts by tens of times. The consequence is a FALSE ALARM
    on every proper score: measured on CZU (60% zero-growth leads, growth-band
    base rate 0.06%), ``brier_1h`` flags SILENCE_FAVOURING with null 0.0005
    against ``skillful`` 0.0007 — which is Brier working correctly on a forecast
    that claims tens of times too many cells, not Brier failing.

    A check whose reference model is unrealistic reports the reference's defects
    as the metric's. So this variant keeps the DISCRIMINATION of ``skillful``
    (the same recall on true growth cells, the same shared latent, so members are
    still scenarios) and fits ONE constant — the false-alarm rate — so that the
    expected number of claimed cells equals the observed number of new cells,
    summed over all scored windows. It is fitted on the labels of the windows
    being scored, exactly as :class:`Climatology` is, and for the same reason: a
    reference must be fair on THIS sample or it is not a reference.
    """

    recall: float
    #: One false-alarm rate PER LEAD, not one overall. C6.2 [v2.8] ratified
    #: exactly this for the ellipse baseline after ADR-015 measured that the
    #: calibration horizon is worth ~4.7x in over-prediction ratio: growth is not
    #: linear in the horizon, so a model calibrated in TOTAL is miscalibrated at
    #: every individual lead. Measured here on CZU, the truth's own band rate runs
    #: 0.00055 / 0.00184 / 0.00540 at 1/2/3 h — a 10x ratio where linear phasing
    #: would give 3x, so a total-calibrated reference over-predicts ~3x at 1 h and
    #: a per-horizon calibration criterion correctly says so.
    false_alarm_by_lead: tuple[float, ...]
    n_windows_fitted: int
    #: What it was fitted to, per lead: (true new cells, plausible-but-wrong).
    n_hits_by_lead: tuple[int, ...]
    n_misses_by_lead: tuple[int, ...]

    def __call__(
        self, window: Window, n_members: int, rng: np.random.Generator
    ) -> np.ndarray:
        from wildfire_nowcast.common.states import dilate  # noqa: PLC0415

        burned = window.x0 > 0
        reach = dilate(burned, 3) & ~burned
        n_lead = int(window.truth.shape[0])
        out = np.zeros((n_members, n_lead, *burned.shape), dtype=bool)
        for m in range(n_members):
            latent = float(rng.uniform(0.45, 1.55))
            u_hit = rng.random(burned.shape)
            u_miss = rng.random(burned.shape)
            claimed = np.zeros(burned.shape, dtype=bool)
            for k in range(n_lead):
                hits_k = (np.asarray(window.truth[k]) > 0) & ~burned
                fa = self.false_alarm_by_lead[min(k, len(self.false_alarm_by_lead) - 1)]
                claimed |= (hits_k & (u_hit < min(1.0, self.recall * latent))) | (
                    reach & ~hits_k & (u_miss < min(1.0, fa * latent))
                )
                out[m, k] = burned | claimed  # absorbing in time, per C1.1
        return _as_samples(out)


def fit_calibrated_skillful(
    windows: Sequence[Window], *, recall: float = 0.55
) -> CalibratedSkillful:
    """Fit :class:`CalibratedSkillful`'s per-lead constants from labels alone."""
    from wildfire_nowcast.common.states import dilate  # noqa: PLC0415

    if not windows:
        raise ValueError("fit_calibrated_skillful needs at least one window")
    horizon = max(int(w.truth.shape[0]) for w in windows)
    hits = np.zeros(horizon, dtype=np.int64)
    misses = np.zeros(horizon, dtype=np.int64)
    for w in windows:
        burned = w.x0 > 0
        reach = dilate(burned, 3) & ~burned
        for k in range(int(w.truth.shape[0])):
            hit_k = (np.asarray(w.truth[k]) > 0) & ~burned
            hits[k] += int(np.count_nonzero(hit_k))
            misses[k] += int(np.count_nonzero(reach & ~hit_k))
    # recall*H_k + fa_k*M_k == H_k  =>  fa_k = (1 - recall) * H_k / M_k. A lead
    # with no plausible-but-wrong cells gets 0, which is correct rather than a
    # division by zero.
    fa = [
        0.0 if m <= 0 else float(min(1.0, max(0.0, (1.0 - recall) * h / m)))
        for h, m in zip(hits.tolist(), misses.tolist(), strict=True)
    ]
    return CalibratedSkillful(
        recall=float(recall),
        false_alarm_by_lead=tuple(fa),
        n_windows_fitted=len(windows),
        n_hits_by_lead=tuple(int(v) for v in hits),
        n_misses_by_lead=tuple(int(v) for v in misses),
    )


def collapse_indep_noise(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """The G3 ablation shape: same marginals, NO scenario spread.

    Members are individually calibrated per pixel and nearly identical in every
    aggregate, because independent noise averages out over thousands of cells.
    This is the ensemble G3 exists to reject, and ``dispersion_ratio`` scores it
    as perfect — which is why it is in the DEGENERATE set here.
    """
    return _skillful_members(
        window, n_members, rng, recall=0.55, false_alarm=0.10, shared_latent=False
    )


@dataclass(frozen=True)
class Climatology:
    """Persistence outside the reachable band, the AVERAGE growth rate inside it.

    **Why this belongs in the degenerate set, even though it "predicts
    something".** C6.0's literal text names a model that predicts nothing, and the
    two zero-ignition nulls are that. But the property those nulls actually
    exploit is not silence — it is carrying NO INFORMATION about the situation,
    and at a 1-7% base rate a silent forecast is very nearly the climatological
    one. Climatology is the exact, sharpened form: it knows where the fire is and
    what fraction of reachable cells burn on average, and nothing else. No wind,
    no fuel, no direction, no shape of the front. It cannot distinguish any cell
    of the band from any other.

    It is the sharpest possible test of a CALIBRATION statistic, because it is
    perfectly calibrated marginally BY CONSTRUCTION, and any statistic it beats
    is a statistic that cannot tell information from its absence. The rate is
    fitted on the labels of the very windows being scored — a free parameter the
    degenerate is GIVEN. That is deliberate and is the conservative direction for
    a safety check, the same reasoning as :data:`NOISE_FLOOR_SD` being 2 and not
    1: a check like this should flag more, not fewer.

    Members are independent per pixel, so this is also collapsed. That is a
    property of climatology, not a modelling choice — there is no scenario to
    vary.
    """

    #: Marginal P(a band cell is burned by lead k), fitted over ALL windows. One
    #: constant per lead, never per window: a rate re-fitted per window would see
    #: that window's answer, which is a different (and much stronger) model.
    rate_by_lead: tuple[float, ...]
    band_radius: int
    n_windows_fitted: int

    def __call__(
        self, window: Window, n_members: int, rng: np.random.Generator
    ) -> np.ndarray:
        from wildfire_nowcast.eval.masks import growth_band  # noqa: PLC0415

        burned = window.x0 > 0
        band = growth_band(window.x0, self.band_radius)
        n_lead = int(window.truth.shape[0])
        out = np.zeros((n_members, n_lead, *burned.shape), dtype=bool)
        for m in range(n_members):
            # One uniform per cell, thresholded by the CUMULATIVE rate, so a
            # member is absorbing in time (C1.1) and its marginals are exactly
            # `rate_by_lead` at every lead.
            u = rng.random(burned.shape)
            for k in range(n_lead):
                rate = self.rate_by_lead[min(k, len(self.rate_by_lead) - 1)]
                out[m, k] = burned | (band & (u < rate))
        return _as_samples(out)


def fit_climatology(
    windows: Sequence[Window], *, band_radius: int | None = None
) -> Climatology:
    """Fit the one constant :class:`Climatology` has, from labels alone."""
    from wildfire_nowcast.eval.masks import default_band_radius, growth_band  # noqa: PLC0415

    if not windows:
        raise ValueError("fit_climatology needs at least one window")
    horizon = max(int(w.truth.shape[0]) for w in windows)
    radius = int(band_radius) if band_radius is not None else default_band_radius(horizon)
    burned_by_lead = np.zeros(horizon, dtype=np.float64)
    cells_by_lead = np.zeros(horizon, dtype=np.float64)
    for w in windows:
        band = growth_band(w.x0, radius)
        n_band = int(np.count_nonzero(band))
        for k in range(int(w.truth.shape[0])):
            burned_by_lead[k] += int(np.count_nonzero((w.truth[k] > 0) & band))
            cells_by_lead[k] += n_band
    rates = np.divide(
        burned_by_lead, cells_by_lead, out=np.zeros_like(burned_by_lead), where=cells_by_lead > 0
    )
    return Climatology(
        rate_by_lead=tuple(float(r) for r in rates),
        band_radius=radius,
        n_windows_fitted=len(windows),
    )


FORECASTERS: dict[str, Forecaster] = {
    "null_zero_ignition": null_zero_ignition,
    "null_empty": null_empty,
    "collapse_indep_noise": collapse_indep_noise,
    "skillful": skillful,
    "oracle": oracle,
}

#: The name the fitted :class:`Climatology` is registered under.
CLIMATOLOGY = "null_climatology"

#: ...and the growth-calibrated skill reference.
SKILLFUL_CALIBRATED = "skillful_calibrated"


def forecasters_for(windows: Sequence[Window]) -> dict[str, Forecaster]:
    """:data:`FORECASTERS` plus the two models that must be FITTED on ``windows``.

    Separate from the static dict because these two have a parameter, and a
    parameter fitted on the wrong sample would be a different model. Callers that
    want the static set (constructed-case tests) still get it.
    """
    return {
        **FORECASTERS,
        CLIMATOLOGY: fit_climatology(windows),
        SKILLFUL_CALIBRATED: fit_calibrated_skillful(windows),
    }


#: The information-free models of C6.0. Tested against every metric.
#:
#: ``null_zero_ignition`` / ``null_empty`` are C6.0's literal do-nothing nulls;
#: ``null_climatology`` is the same idea sharpened — see :class:`Climatology`.
NULLS: frozenset[str] = frozenset({"null_zero_ignition", "null_empty", CLIMATOLOGY})

#: The collapse ablation. Tested against SPREAD metrics only — see FAMILY_SKILL.
COLLAPSE = "collapse_indep_noise"

#: Forecasters that never draw from the rng, so their score is IDENTICAL at every
#: seed. Declared rather than detected, and verified by a test that runs each one
#: under two different generators and asserts bit-equality — a mis-declaration
#: must break the build, not silently cache a stochastic model at one seed.
#:
#: :func:`run_null_check` scores these ONCE and replicates, which is bitwise the
#: same numbers (they were already identical) for a third less work: measured,
#: they are 2.86 s of the 7.7 s each report costs, and the null-check fixture is
#: over half the test suite's wall clock. The scoring itself is C6's, so this is
#: the only saving available here that does not touch another lead's module or
#: change a verdict.
DETERMINISTIC: frozenset[str] = frozenset({"null_zero_ignition", "null_empty", "oracle"})

#: THE SUBJECT OF THE ZERO-CAPTURE AXIOM: a forecast that claims nothing anywhere
#: at any lead, for every member.
#:
#: It is ``null_empty`` and NOT ``null_zero_ignition``, and that choice is
#: load-bearing rather than cosmetic. Persistence reproduces the already-burned
#: region, so on the DOMAIN mask it legitimately scores ``best_member_iou``
#: 0.857 — capture it earned by making a correct claim. An axiom pointed at it
#: would flag every capture metric in the table, which is a false-positive safety
#: check, which is how safety checks get switched off (the argument that has kept
#: C1.6 off the hard tier twice).
ZERO_CLAIM = "null_empty"

#: Everything that must never win, for reporting.
DEGENERATE: frozenset[str] = NULLS | {COLLAPSE}
SKILLFUL = "skillful"
ORACLE = "oracle"

#: Every forecaster that has GENUINE SKILL. A degenerate must be measurably worse
#: than the BEST of these, not than an arbitrary one of them.
#:
#: Two, not one, and the reason is a defect this harness had: a single reference
#: that is over-confident off-fixture makes every proper score look
#: silence-favouring, and the check then reports the REFERENCE's defect as the
#: METRIC's. Requiring "the degenerate beats every skilful forecaster we can
#: build" is both the conservative reading of C6.0 and the one that does not
#: manufacture false alarms — and a false-positive safety check is the thing that
#: teaches people to disable safety checks (the argument that kept C1.6 off the
#: hard tier twice). The BLIND comparison deliberately does NOT use this: collapse
#: is compared against ``skillful`` specifically, because ``collapse_indep_noise``
#: is that exact model with the shared latent removed, and a controlled ablation
#: must vary one thing.
SKILL_REFERENCES: frozenset[str] = frozenset({SKILLFUL, SKILLFUL_CALIBRATED})


def degenerates_for(spec: MetricSpec) -> frozenset[str]:
    return NULLS | {COLLAPSE} if spec.family == FAMILY_SPREAD else NULLS


def strongest_reference(spec: MetricSpec) -> str:
    """The model a degenerate may never match.

    For a skill metric that is the ORACLE: nothing may be as good as being
    exactly right. For a SPREAD metric it is ``skillful``, the healthy
    shared-latent ensemble — a perfect forecast has zero error, so its
    spread-skill ratio is undefined, and that is a property of the estimand
    rather than a defect of the metric.
    """
    return SKILLFUL if spec.family == FAMILY_SPREAD else ORACLE


# --------------------------------------------------------------------------
# scoring + verdicts
# --------------------------------------------------------------------------


def _c6_scorer() -> Callable[[np.ndarray, np.ndarray, np.ndarray], Mapping[str, Any]]:
    """The default score function: C6 ``evaluate`` + ``aggregate``.

    Imported lazily and injected, so ``common/`` does not depend on ``eval/`` at
    module scope and the harness stays usable against ANY scoring function. C6
    is modelling's module; this file only calls its documented entry point.
    """
    from wildfire_nowcast.eval.metrics import aggregate, evaluate  # noqa: PLC0415

    def score(samples: np.ndarray, truth: np.ndarray, x0: np.ndarray) -> Mapping[str, Any]:
        return evaluate(samples, truth, x0=x0)

    score.aggregate = aggregate  # type: ignore[attr-defined]
    return score


#: Scalars a ``by_mask`` block carries that are BOOKKEEPING, not scores — counts,
#: denominators and censoring caps. Everything else numeric is treated as a
#: metric and must be registered in :data:`C6_METRICS`, so a metric added
#: tomorrow breaks this check loudly instead of passing by omission. That is the
#: C-2 lesson one level down: the burden belongs on the thing being added.
_STRUCTURAL_KEYS: frozenset[str] = frozenset(
    {
        "n_cells",
        "n_windows",
        "arrival_crps_active_cells",
        "arrival_crps_censor_cap_h",
        "best_member_iou_n_empty_leads",
        "best_member_iou_n_growing_leads",
        "base_rate",
        "uncertainty",
        "calibration_n_scored",
        "calibration_n_occupied_bins",
        "calibration_n_occupied_rings",
    }
)


def _flatten(block: Mapping[str, Any]) -> dict[str, float | None]:
    """Every numeric score in one ``by_mask`` block, registered or not."""
    out: dict[str, float | None] = {}
    for lead, value in (block.get("brier_by_lead") or {}).items():
        out[f"brier_{int(lead)}h"] = value
    for lead, summary in (block.get("reliability_summary") or {}).items():
        for stat, value in (summary or {}).items():
            if stat in _STRUCTURAL_KEYS:
                continue
            out[f"{stat}_{int(lead)}h"] = value
    for key, value in block.items():
        if key in _STRUCTURAL_KEYS or key.startswith("n_") or key.endswith("_by_horizon"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = float(value)
    return out


#: How a comparison came out, once seed noise is accounted for.
#: How many seed SDs a paired difference must exceed to count as real. A noise
#: floor measured from this run's own seeds, not a constant fitted to any sample
#: of fires — so C-3's "state the fitting sample" has nothing to bind to here,
#: and that is deliberate: the alternative is a magic separation threshold.
NOISE_FLOOR_SD = 2.0

CMP_BETTER = "better"
CMP_WORSE = "worse"
CMP_INDISTINGUISHABLE = "indistinguishable"


def _advantage(a: float, b: float, spec: MetricSpec) -> float:
    """Signed advantage of ``a`` over ``b``: positive means ``a`` is better."""
    if spec.direction == HIGHER:
        return a - b
    if spec.direction == LOWER:
        return b - a
    if spec.direction == TARGET:
        target = float(spec.target if spec.target is not None else 1.0)
        return abs(b - target) - abs(a - target)
    raise ValueError(f"{spec.direction} is not rankable")


def compare(
    a_by_seed: Sequence[float], b_by_seed: Sequence[float], spec: MetricSpec
) -> tuple[str, float, float]:
    """PAIRED comparison across seeds: ``(verdict, mean advantage, sd)``.

    Paired, because every forecaster is scored on identical windows with the same
    seed, so the difference has far less variance than either score. And
    seed-aware, because the alternative is a threshold — and a threshold here
    would be a constant deciding a pass/fail, which C-3 makes expensive for good
    reason. The noise floor is MEASURED from the same experiment: a difference
    smaller than its own seed-to-seed standard deviation is not a finding.

    That distinction is not academic. Measured on this fixture,
    ``dispersion_ratio`` prefers the COLLAPSED ensemble on 4 of 5 seeds and the
    healthy one on the 5th, all within 1%: reporting the 4/5 as a direction would
    be reporting a coin flip. ADR-018 quotes 100-159 seed SD for a real effect —
    this is the same discipline applied to the instruments.

    :data:`NOISE_FLOOR_SD` is a NOISE FLOOR, not a fitted threshold: the scale it
    is measured against is estimated from this run's own seeds, and it states no
    claim about any sample of fires (C-3). It is set to 2 rather than 1 because a
    5-seed SD carries ~35% relative error, and because the conservative direction
    for a SAFETY check is to flag more, not fewer — an undecided comparison here
    resolves to "the metric cannot tell these apart", which is a finding.
    """
    diffs = [
        _advantage(float(a), float(b), spec)
        for a, b in zip(a_by_seed, b_by_seed, strict=True)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
    ]
    if not diffs:
        return CMP_INDISTINGUISHABLE, float("nan"), float("nan")
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    floor = NOISE_FLOOR_SD * sd
    if mean > floor:
        return CMP_BETTER, mean, sd
    if mean < -floor:
        return CMP_WORSE, mean, sd
    return CMP_INDISTINGUISHABLE, mean, sd


@dataclass
class MetricVerdict:
    """One metric under one mask, carrying BOTH of C6.0's answers.

    ``verdict`` is the COMPARISON (beatable in principle by genuine skill?) and
    ``capture_verdict`` is the ZERO-CAPTURE AXIOM (does it pay for an empty
    forecast?). They are separate fields because they are separate questions and
    they measurably disagree — see the module docstring. ``is_flagged`` is the
    disjunction, for callers that only need "did anything fire".
    """

    metric: str
    mask: str
    verdict: str
    scores: dict[str, float | None]
    gate_eligible: bool
    quarantined_by: str = ""
    detail: str = ""
    capture_verdict: str = CAPTURE_NOT_APPLICABLE
    capture_detail: str = ""

    @property
    def is_failure(self) -> bool:
        """C-1 ``fail``, from EITHER verdict.

        Hard from the comparison when a degenerate ranks with the best forecast
        the metric admits (``BROKEN``) or when the metric cannot see collapse at
        all (``BLIND``). Hard from the axiom when the metric pays a strictly
        positive score for an empty forecast: that credit is bankable by any
        model, so a gate resting on it is measuring partly silence.
        """
        if not self.gate_eligible:
            return False
        return (
            self.verdict in (VERDICT_BROKEN, VERDICT_BLIND)
            or self.capture_verdict == VERDICT_PAYS_FOR_NOTHING
        )

    @property
    def is_reporting_gap(self) -> bool:
        """C-1 ``reporting``: gate-eligible and silence-favouring.

        Printed unconditionally and non-zero under ``--strict``, but not a hard
        failure, because a proper score at a 1% base rate legitimately prefers
        silence to a sub-coin-flip predictor (R14). Making that a hard fail would
        be a false positive that teaches people to disable the check — the exact
        argument that kept C1.6 off the hard tier.
        """
        return self.gate_eligible and self.verdict == VERDICT_SILENCE_FAVOURING

    @property
    def is_flagged(self) -> bool:
        """Either verdict says something other than ``ok``/not-applicable."""
        return self.verdict != VERDICT_OK or self.capture_verdict == VERDICT_PAYS_FOR_NOTHING


@dataclass
class NullCheckReport:
    scenario: dict[str, Any]
    verdicts: list[MetricVerdict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not any(v.is_failure for v in self.verdicts)

    @property
    def reporting_ok(self) -> bool:
        return self.ok and not any(v.is_reporting_gap for v in self.verdicts)

    def failures(self) -> list[MetricVerdict]:
        return [v for v in self.verdicts if v.is_failure]

    def reporting_gaps(self) -> list[MetricVerdict]:
        return [v for v in self.verdicts if v.is_reporting_gap]

    def pays_for_nothing(self) -> list[MetricVerdict]:
        """Metrics that pay a strictly positive score for an empty forecast."""
        return [v for v in self.verdicts if v.capture_verdict == VERDICT_PAYS_FOR_NOTHING]

    def quarantined_confirmed(self) -> list[MetricVerdict]:
        """Known-broken metrics this run reproduced — the positive controls.

        Reads BOTH verdicts (``is_flagged``). Before A12 it read only the
        comparison, so when the refit raised the reference above the null floor
        the controls emptied and the harness looked healthy while the pathology
        was untouched. A positive control that can be silenced by improving the
        reference model was never testing the metric.
        """
        return [v for v in self.verdicts if v.quarantined_by and v.is_flagged]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause": "C6.0",
            "ok": self.ok,
            "scenario": self.scenario,
            "problems": self.problems,
            "verdicts": [
                {
                    "metric": v.metric,
                    "mask": v.mask,
                    "verdict": v.verdict,
                    "capture_verdict": v.capture_verdict,
                    "gate_eligible": v.gate_eligible,
                    "quarantined_by": v.quarantined_by,
                    "scores": v.scores,
                    "detail": v.detail,
                    "capture_detail": v.capture_detail,
                }
                for v in self.verdicts
            ],
        }


def _best_skill_reference(means: Mapping[str, float | None], spec: MetricSpec) -> str:
    """Which of :data:`SKILL_REFERENCES` scores best on this metric."""
    scored = [n for n in sorted(SKILL_REFERENCES) if means.get(n) is not None]
    if not scored:
        return SKILLFUL
    best = scored[0]
    for name in scored[1:]:
        if _advantage(float(means[name]), float(means[best]), spec) > 0:
            best = name
    return best


def _capture_verdict(
    spec: MetricSpec, by_seed: Mapping[str, list[float | None]]
) -> tuple[str, str]:
    """THE ZERO-CAPTURE AXIOM — no reference model, no threshold, no member count.

    A ``higher_is_better`` capture metric must pay :data:`ZERO_CLAIM` the minimum
    of its range. Every such metric registered here has minimum 0 and every sound
    one lands on EXACTLY ``0.0``, so the line is strict positivity and there is no
    constant for C-3 to bind. Evaluated per seed rather than on the mean: "pays
    for nothing" is a claim about the metric, so one seed paying is enough.
    """
    if spec.direction != HIGHER:
        return CAPTURE_NOT_APPLICABLE, ""
    if ZERO_CLAIM not in by_seed:
        return (
            VERDICT_UNDECIDABLE,
            f"{ZERO_CLAIM!r} is not among the scored forecasters, so the zero-capture axiom "
            "could not be evaluated. Reported, never silently passed.",
        )
    paid = [
        float(v)
        for v in by_seed[ZERO_CLAIM]
        if v is not None and np.isfinite(v) and float(v) > 0.0
    ]
    if not paid:
        return VERDICT_OK, ""
    worst = max(paid)
    return (
        VERDICT_PAYS_FOR_NOTHING,
        f"a forecast that claims NOTHING anywhere at any lead is paid {worst:.5f} on "
        f"{len(paid)} of {len(by_seed[ZERO_CLAIM])} seeds, where the minimum of this metric's "
        "range is 0. That credit is bankable by any model without predicting anything, so this "
        "metric is a joint test of capture and of saying nothing and cannot isolate the first. "
        "Needs no reference model and no member count, which is exactly why it is the verdict "
        "that survives changing either (ADR-022 (1)).",
    )


def _verdict_for(
    metric: str,
    mask: str,
    spec: MetricSpec,
    by_seed: Mapping[str, list[float | None]],
) -> MetricVerdict:
    means = {
        name: (float(np.mean([v for v in vals if v is not None and np.isfinite(v)])) if any(
            v is not None and np.isfinite(v) for v in vals
        ) else None)
        for name, vals in by_seed.items()
    }
    capture, capture_detail = _capture_verdict(spec, by_seed)
    base = MetricVerdict(
        metric=metric,
        mask=mask,
        verdict=VERDICT_UNDECIDABLE,
        scores=means,
        gate_eligible=spec.gate_eligible,
        quarantined_by=spec.quarantined_by,
        capture_verdict=capture,
        capture_detail=capture_detail,
    )
    reference = strongest_reference(spec)
    if means.get(reference) is None or means.get(SKILLFUL) is None:
        base.detail = (
            f"reference model {reference!r} scores None/non-finite here, so there is nothing "
            "to rank against. Undecidable is reported, never silently passed."
        )
        return base
    degenerate = sorted(k for k in degenerates_for(spec) if means.get(k) is not None)
    if not degenerate:
        base.detail = "no degenerate model produced a score"
        return base

    if spec.family == FAMILY_SPREAD and means.get(COLLAPSE) is not None:
        outcome, mean, sd = compare(by_seed[COLLAPSE], by_seed[SKILLFUL], spec)
        if outcome == CMP_INDISTINGUISHABLE:
            base.verdict = VERDICT_BLIND
            base.detail = (
                f"collapsed {means[COLLAPSE]:.4f} vs healthy {means[SKILLFUL]:.4f}: paired "
                f"advantage {mean:+.4f} +- {sd:.4f} over seeds, i.e. INSIDE its own seed "
                "noise. This metric cannot separate a COLLAPSED ensemble from a healthy one, "
                "so which one it prefers is a coin flip — and ensemble collapse is this "
                "project's central claim (ADR-011)."
            )
            return base

    broken = {}
    for name in degenerate:
        outcome, mean, sd = compare(by_seed[name], by_seed[reference], spec)
        if outcome in (CMP_BETTER, CMP_INDISTINGUISHABLE):
            broken[name] = (outcome, mean, sd)
    if broken:
        base.verdict = VERDICT_BROKEN
        base.detail = (
            f"{sorted(broken)} are not measurably worse than {reference} "
            f"({means[reference]:.4f}): "
            + "; ".join(
                f"{k} {v[1]:+.4f} +- {v[2]:.4f} ({v[0]})" for k, v in sorted(broken.items())
            )
            + ". A degenerate model cannot be as good as the best forecast this metric admits; "
            "the metric is measuring something other than capability."
        )
        return base

    # A degenerate must be worse than the BEST skilful forecaster, not than an
    # arbitrary one — see SKILL_REFERENCES for why this is two models and not one.
    skill_ref = _best_skill_reference(means, spec)
    favoured = {}
    for name in degenerate:
        outcome, mean, sd = compare(by_seed[name], by_seed[skill_ref], spec)
        if outcome in (CMP_BETTER, CMP_INDISTINGUISHABLE):
            favoured[name] = (outcome, mean, sd)
    if favoured:
        base.verdict = VERDICT_SILENCE_FAVOURING
        base.detail = (
            f"{sorted(favoured)} are not measurably worse than the best forecaster with genuine "
            f"skill ({skill_ref} {means[skill_ref]:.4f}): "
            + "; ".join(
                f"{k} {v[1]:+.4f} +- {v[2]:.4f} ({v[0]})" for k, v in sorted(favoured.items())
            )
            + ". Not automatically a bug at a low base rate (R14/C6.2), but this metric cannot "
            "be read as capability without stating it, and it must not gate."
        )
        return base

    base.verdict = VERDICT_OK
    base.detail = "every degenerate model ranks below genuine skill, beyond seed noise"
    return base


#: Seeds the harness re-runs every forecaster under. More than one is not
#: optional: with a single seed, ``dispersion_ratio`` reports a DIRECTION for a
#: difference that is a coin flip, and the harness would publish it.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Members per forecast. **Not a performance knob — it changes verdicts**, and it
#: changes them in the flattering direction when it is small.
#:
#: Measured on this fixture (growth band, 3 h, ``null_climatology`` vs
#: ``skillful``), ECE at increasing member counts::
#:
#:     M      8      32     128     512
#:     clim   0.0795 0.0256 0.0110  0.0007
#:     skill  0.0347 0.0482 0.0484  0.0484
#:
#: The ensemble MEAN of ``M`` Bernoulli draws is itself noisy, and that noise is
#: uncorrelated with the truth, so at ``M = 8`` it inflates a no-information
#: forecast's ECE by ~0.08 and HIDES the fact that climatology drives ECE to
#: zero. Raising ``M`` does not create the pathology, it stops masking it. 32 is
#: the smallest power of two at which the masking is gone here (climatology
#: already beats genuine skill), and it costs ~12 s. Terms that pool many cells
#: per stratum — the frontier term, Brier, CRPS — are flat in ``M`` (0.1094 ->
#: 0.1098), which is precisely the property a gate criterion needs: a bar that
#: moves with the member count is not a bar.
DEFAULT_MEMBERS = 32


def run_null_check(
    windows: Sequence[Window],
    scenario: Mapping[str, Any] | None = None,
    *,
    n_members: int = DEFAULT_MEMBERS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    forecasters: Mapping[str, Forecaster] | None = None,
    score_fn: Callable[..., Any] | None = None,
) -> NullCheckReport:
    """Score every forecaster on identical windows and adjudicate each metric.

    Identical windows, an identical member count and identical seeds for every
    forecaster: the only thing that varies is the forecast, which is what makes
    each verdict a property of the METRIC rather than of the sample. Comparisons
    are PAIRED across seeds — see :func:`compare`.
    """
    if not windows:
        raise ValueError("run_null_check needs at least one window")
    if not seeds:
        raise ValueError("run_null_check needs at least one seed")
    models = dict(forecasters or forecasters_for(windows))
    scorer = score_fn or _c6_scorer()
    pool = getattr(scorer, "aggregate", None)
    if pool is None:  # pragma: no cover - only for an injected scorer
        from wildfire_nowcast.eval.metrics import aggregate as pool  # noqa: PLC0415

    report = NullCheckReport(scenario=dict(scenario or {}))
    report.scenario.setdefault("n_windows", len(windows))
    report.scenario["n_members"] = int(n_members)
    report.scenario["seeds"] = [int(s) for s in seeds]
    report.scenario["forecasters"] = sorted(models)
    report.scenario["degenerate"] = sorted(DEGENERATE)
    report.scenario["zero_claim"] = ZERO_CLAIM
    if ZERO_CLAIM not in models:
        report.problems.append(
            f"the zero-capture axiom's subject {ZERO_CLAIM!r} is absent from the forecaster set, "
            "so no capture metric can be adjudicated against an empty forecast. A missing "
            "control is a PROBLEM, not a skip."
        )

    # pooled[model][seed_index] = the aggregated score dict
    pooled: dict[str, list[Mapping[str, Any]]] = {}
    for name, fn in models.items():
        runs = []
        for seed in seeds:
            # A forecaster that never draws produces the same score at every
            # seed, so scoring it once and replicating is bitwise identical, not
            # an approximation. See DETERMINISTIC — declared, and asserted by a
            # test that runs each one under two generators.
            if runs and name in DETERMINISTIC:
                runs.append(runs[0])
                continue
            rng = np.random.default_rng(int(seed))
            runs.append(
                pool(
                    [
                        scorer(fn(w, n_members, rng), np.asarray(w.truth), np.asarray(w.x0))
                        for w in windows
                    ]
                )
            )
        pooled[name] = runs

    masks = sorted({m for runs in pooled.values() for m in runs[0].get("by_mask", {})})
    for mask in masks:
        flat = {
            name: [_flatten(run["by_mask"][mask]) for run in runs] for name, runs in pooled.items()
        }
        seen = sorted({k for runs in flat.values() for run in runs for k in run})
        unregistered = [k for k in seen if k not in C6_METRICS]
        if unregistered:
            report.problems.append(
                f"[{mask}] C6 emits {unregistered} but common/null_check.C6_METRICS does not "
                "declare their orientation, so they cannot be null-checked. An unregistered "
                "metric is an unchecked metric (C-2, one level down)."
            )
        for metric in seen:
            spec = C6_METRICS.get(metric)
            if spec is None:
                continue
            by_seed = {name: [run.get(metric) for run in runs] for name, runs in flat.items()}
            if spec.direction == LABEL_STATISTIC:
                values = {v for runs in by_seed.values() for v in runs if v is not None}
                if len(values) > 1:
                    report.problems.append(
                        f"[{mask}] {metric} is a property of the LABELS and must be identical "
                        f"for every forecaster and seed, but got {sorted(values)}. Something "
                        "model-dependent has leaked into it."
                    )
                continue
            if spec.direction == DIAGNOSTIC:
                continue
            report.verdicts.append(_verdict_for(metric, mask, spec, by_seed))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_report(report: NullCheckReport) -> str:
    lines = ["C6.0 — DO-NOTHING NULL CHECK (ADR-017)", ""]
    scenario = report.scenario
    lines.append(
        f"scenario: {scenario.get('source', 'n/a')}  windows={scenario.get('n_windows')} "
        f"members={scenario.get('n_members')} grid={scenario.get('grid_shape')} "
        f"seeds={scenario.get('seeds')} zero-growth leads="
        f"{_pct(scenario.get('zero_growth_lead_fraction'))}"
    )
    lines.append(f"degenerate models (must never win): {', '.join(sorted(DEGENERATE))}")
    lines.append(
        f"two verdicts per metric: [CMP] the comparison against a reference model, "
        f"[CAP] the zero-capture axiom against {ZERO_CLAIM!r}. They answer different "
        "questions and they disagree (ADR-022 (1))."
    )
    lines.append("")
    by_mask: dict[str, list[MetricVerdict]] = {}
    for v in report.verdicts:
        by_mask.setdefault(v.mask, []).append(v)
    for mask, verdicts in sorted(by_mask.items()):
        lines.append(f"--- mask: {mask} " + "-" * max(0, 60 - len(mask)))
        for v in verdicts:
            tag = "GATE" if v.gate_eligible else f"reported ({v.quarantined_by or 'n/a'})"
            flag = {
                VERDICT_OK: "  ok  ",
                VERDICT_BROKEN: "BROKEN",
                VERDICT_BLIND: "BLIND ",
                VERDICT_SILENCE_FAVOURING: "SILENT",
                VERDICT_UNDECIDABLE: " n/a  ",
            }[v.verdict]
            cap = {
                VERDICT_OK: " ok ",
                VERDICT_PAYS_FOR_NOTHING: "PAYS",
                VERDICT_UNDECIDABLE: "n/a ",
                CAPTURE_NOT_APPLICABLE: "  - ",
            }[v.capture_verdict]
            scores = " ".join(
                f"{_abbrev(k)}={_num(v.scores.get(k))}" for k in sorted(v.scores)
            )
            lines.append(f"  [{flag}|{cap}] {v.metric:<32} {tag:<22} {scores}")
            if v.verdict in (VERDICT_BROKEN, VERDICT_BLIND, VERDICT_SILENCE_FAVOURING):
                lines.append(f"       CMP -> {v.detail}")
            if v.capture_verdict in (VERDICT_PAYS_FOR_NOTHING, VERDICT_UNDECIDABLE):
                lines.append(f"       CAP -> {v.capture_detail}")
        lines.append("")
    for problem in report.problems:
        lines.append(f"  [PROBLEM] {problem}")
    confirmed = report.quarantined_confirmed()
    if confirmed:
        lines.append(
            "positive controls reproduced (already quarantined by contract): "
            f"{sorted({v.metric for v in confirmed})} — dispersion_ratio (C6.1) and "
            "best_member_iou (C6.4) are KNOWN-broken, so if they ever come back clean, "
            "suspect this harness before believing the good news."
        )
    else:
        lines.append(
            "NO POSITIVE CONTROL REPRODUCED. dispersion_ratio (C6.1) and best_member_iou "
            "(C6.4) are quarantined by contract and MUST come back flagged. An empty control "
            "set means this harness stopped detecting pathologies we have confirmed by other "
            "methods — suspect the harness, not the metrics."
        )
    paid = report.pays_for_nothing()
    if paid:
        lines.append(
            "zero-capture axiom: these metrics pay a strictly positive score to a forecast "
            f"that claims nothing — {sorted({v.metric for v in paid})}. Needs no reference "
            "model, so no change to a reference model can silence it."
        )
    lines.append("")
    for gap in report.reporting_gaps():
        lines.append(
            f"[REPORTING] {gap.metric} ({gap.mask}) is gate-eligible and SILENCE_FAVOURING. "
            "Not a hard failure (R14: a proper score at a 1% base rate legitimately prefers "
            "silence to a sub-coin-flip predictor), but it must not be quoted as capability "
            "without stating the base rate. Non-zero under --strict."
        )
    if report.failures():
        lines.append(
            f"FAIL — {len(report.failures())} GATE-ELIGIBLE metric verdict(s) either rank a "
            f"DEGENERATE model first or pay for an empty forecast: "
            f"{sorted({v.metric for v in report.failures()})}. C6.0: if the null wins, the "
            "metric is broken, not the model."
        )
    else:
        lines.append(
            "OK — no gate-eligible metric ranks a degenerate model above the best forecast it "
            "admits, and none pays a positive score for an empty forecast."
        )
    return "\n".join(lines)


#: Short, DISTINCT column labels for the score table. The previous rule
#: (``name.split("_")[0][:4]``) printed ``null=`` for all THREE nulls and
#: ``skil=`` for both skill references, so six columns carried two labels and a
#: reader could not tell which model scored what. A report that cannot be read
#: correctly is not a report.
_LABELS: dict[str, str] = {
    "null_zero_ignition": "persist",
    "null_empty": "empty",
    "null_climatology": "clim",
    "collapse_indep_noise": "collapse",
    "skillful": "skill",
    "skillful_calibrated": "skill_cal",
    "oracle": "oracle",
}


def _abbrev(name: str) -> str:
    return _LABELS.get(name, name[:9])


def _num(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:6.3f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.common.null_check",
        description="C6.0 — score a do-nothing null against every C6 metric.",
    )
    parser.add_argument(
        "--tensor",
        default=None,
        help="C1 store to draw windows from (default: a generated label sequence with a "
        "declared zero-growth rate, so the check needs no data on disk)",
    )
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument(
        "--members",
        type=int,
        default=DEFAULT_MEMBERS,
        help="members per forecast. NOT a performance knob: too few members mask a "
        "calibration pathology behind the ensemble mean's own sampling noise "
        "(see DEFAULT_MEMBERS)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="comparisons are PAIRED across these seeds; one seed cannot "
        "distinguish a real difference from a coin flip",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--json", dest="json_out", default=None, help="also write the report here")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="C-1: promote SILENCE_FAVOURING reporting gaps on gate-eligible metrics to "
        "hard failures. Run this before any number from these metrics enters a gate.",
    )
    args = parser.parse_args(argv)

    if args.tensor:
        windows, scenario = windows_from_tensor(
            args.tensor, horizon_h=args.horizon, max_windows=args.max_windows
        )
    else:
        windows, scenario = synthetic_windows(horizon_h=args.horizon)
        if args.max_windows:
            windows = windows[: args.max_windows]
    report = run_null_check(
        windows, scenario, n_members=args.members, seeds=tuple(args.seeds)
    )
    print(format_report(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2))
    return 0 if (report.reporting_ok if args.strict else report.ok) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
