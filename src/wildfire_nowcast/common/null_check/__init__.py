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

# [A15] This module was a single 1,636-line file. It is now a package split by
# responsibility — the metric table, the windows, the forecasters, the
# adjudication, the report — and this file re-exports the whole of the previous
# public surface, so every existing import path resolves unchanged. Nothing
# below is new API; if a name is here it was importable from
# `wildfire_nowcast.common.null_check` before the split.
#
# [A17] `CALIBRATION_METRIC` / `GATE_METRIC` are imported from the modules that
# DEFINE them rather than from `registry`, which only re-aliases them. Same
# objects, and it keeps C0's single implementation one import away, not two.
from wildfire_nowcast.common.calibration import GATE_CRITERION_KEY as CALIBRATION_METRIC
from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY as GATE_METRIC
from wildfire_nowcast.common.null_check.cli import (
    format_report,
    main,
)
from wildfire_nowcast.common.null_check.forecasters import (
    CLIMATOLOGY,
    COLLAPSE,
    DEGENERATE,
    DETERMINISTIC,
    FORECASTERS,
    NULLS,
    ORACLE,
    SKILL_REFERENCES,
    SKILLFUL,
    SKILLFUL_CALIBRATED,
    ZERO_CLAIM,
    CalibratedSkillful,
    Climatology,
    Forecaster,
    collapse_indep_noise,
    degenerates_for,
    fit_calibrated_skillful,
    fit_climatology,
    forecasters_for,
    null_empty,
    null_zero_ignition,
    oracle,
    skillful,
    strongest_reference,
)
from wildfire_nowcast.common.null_check.registry import (
    C6_METRICS,
    CAPTURE_NOT_APPLICABLE,
    DIAGNOSTIC,
    FAMILY_SKILL,
    FAMILY_SPREAD,
    HIGHER,
    LABEL_STATISTIC,
    LOWER,
    TARGET,
    VERDICT_BLIND,
    VERDICT_BROKEN,
    VERDICT_OK,
    VERDICT_PAYS_FOR_NOTHING,
    VERDICT_SILENCE_FAVOURING,
    VERDICT_UNDECIDABLE,
    MetricSpec,
)
from wildfire_nowcast.common.null_check.verdicts import (
    CMP_BETTER,
    CMP_INDISTINGUISHABLE,
    CMP_WORSE,
    DEFAULT_MEMBERS,
    DEFAULT_SEEDS,
    NOISE_FLOOR_SD,
    MetricVerdict,
    NullCheckReport,
    compare,
    run_null_check,
)
from wildfire_nowcast.common.null_check.verdicts import (
    _capture_verdict as _capture_verdict,
)
from wildfire_nowcast.common.null_check.verdicts import (
    _flatten as _flatten,
)
from wildfire_nowcast.common.null_check.windows import (
    Window,
    synthetic_windows,
    windows_from_tensor,
)

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

#: [A15] Names that were importable from the flat module and are NOT in the list
#: above. They were public in practice — tests, `sim/quarantine.py` and the
#: clause registry all reach for them — so the split must not quietly narrow the
#: surface. Listed separately rather than merged, so the pre-split `__all__`
#: stays readable as the thing it was.
__all__ += [
    "CAPTURE_NOT_APPLICABLE",
    "CMP_BETTER",
    "CMP_INDISTINGUISHABLE",
    "CMP_WORSE",
    "DIAGNOSTIC",
    "FAMILY_SKILL",
    "FAMILY_SPREAD",
    "HIGHER",
    "LABEL_STATISTIC",
    "LOWER",
    "TARGET",
    "VERDICT_BLIND",
    "VERDICT_BROKEN",
    "VERDICT_OK",
    "VERDICT_PAYS_FOR_NOTHING",
    "VERDICT_SILENCE_FAVOURING",
    "VERDICT_UNDECIDABLE",
    "collapse_indep_noise",
    "null_empty",
    "null_zero_ignition",
    "oracle",
    "skillful",
]
