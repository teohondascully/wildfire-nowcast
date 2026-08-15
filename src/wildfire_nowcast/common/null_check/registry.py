"""WHICH C6 METRICS EXIST, how each is read, and which may decide a gate.

The table is the harness's contract with C6: a metric absent from
:data:`C6_METRICS` cannot be null-checked, and :func:`..verdicts.run_null_check`
raises a PROBLEM rather than skipping it, so a metric added tomorrow breaks the
check loudly instead of passing by omission.

``gate_eligible`` records the CONTRACT's ruling, never a judgement made here.
Changing an entry is a maintainer ruling; see C6.1, C6.4 and C6.6.

**[v2.15] C6.6 — FOUR METRICS BECAME NON-ADJUDICATING (ADR-053 (1)(2)).** Brier,
arrival-time CRPS, ``calibration_error`` and reliability have NEGATIVE Spearman
against ``|log(area error)|`` on M11's degradation ladder, so on those channels a
forecast that is 40x TOO SMALL beats a 3%-correct one by 17% on Brier. They MAY
BE REPORTED and they MAY NOT DECIDE A GATE. Three channels may adjudicate today
and :func:`adjudicating_metrics` derives that set from the flags rather than
restating it: ``area_dispersion_ratio``, ``growth_calibration`` and
``best_member_iou_shape_masked``.

**Ask the registry, do not remember.** :func:`assert_may_adjudicate` RAISES on a
flagged channel. It raises rather than warning because the failure this repairs
is a human reading a number and forgetting a ruling made two weeks earlier — a
warning is read by the same person who forgot.
"""

from __future__ import annotations

from dataclasses import dataclass

from wildfire_nowcast.common.calibration import GATE_CRITERION_KEY as CALIBRATION_METRIC
from wildfire_nowcast.common.dispersion import FIRST_MOMENT_KEY
from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY as GATE_METRIC
from wildfire_nowcast.common.iou_terms import REPORTED_ONLY_KEY

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
#: [v2.15] C6.6 — the citation carried on every channel the ladder disqualified.
#: Stated once so four entries cannot drift apart, and so the number a reader
#: needs (the Spearman, and what produced it) travels with the ruling.
_ANTI_MONOTONE = (
    "C6.6 [v2.15] NON-ADJUDICATING (ADR-053 (1)(2)). M11's degradation ladder, "
    "0.053x-8.0x area error at n=5 held-out blocks, measured Spearman {rho} against "
    "|log(area error)| — the WRONG SIGN — and NO minimum detectable effect anywhere on "
    "the ladder (plateau 0.66-1.58). A forecast 40x TOO SMALL beats a 3%-correct one by "
    "17% on Brier. REPORTED, never a gate criterion."
)

C6_METRICS: dict[str, MetricSpec] = {
    # --- NAMED BY G2 / ADR-018, DISQUALIFIED AT v2.15 BY C6.6 --------------
    **{
        f"brier_{h}h": MetricSpec(
            LOWER,
            False,
            quarantined_by="C6.6 (ADR-053 (1)(2))",
            note=_ANTI_MONOTONE.format(rho="-0.45"),
        )
        for h in (1, 2, 3)
    },
    "arrival_crps": MetricSpec(
        LOWER,
        False,
        quarantined_by="C6.6 (ADR-053 (1)(2))",
        note=_ANTI_MONOTONE.format(rho="-0.34"),
    ),
    GATE_METRIC: MetricSpec(
        HIGHER, True, note="C6.4 gate criterion: empty-truth leads dropped; null floor is 0."
    ),
    # --- named by G3 / ADR-011 ---------------------------------------------
    "area_dispersion_ratio": MetricSpec(
        TARGET,
        True,
        target=1.0,
        family=FAMILY_SPREAD,
        note="C6.1 makes this the G3 criterion instead of dispersion_ratio. One of the three "
        "channels C6.6 leaves able to adjudicate; MDE 1.021x at the 2.0 bar, n=5 (ADR-053).",
    ),
    # --- named by G3 / C6.5 as the FIRST-MOMENT condition -------------------
    FIRST_MOMENT_KEY: MetricSpec(
        TARGET,
        True,
        target=1.0,
        note="C6.5's first-moment condition, defined against a REFERENCE rather than an "
        "absolute bar. Registered at v2.15 so C6.6 can be asked about it: it is one of the "
        "three channels that may still adjudicate. MDE 1.056x, and DEGENERATE on an area "
        "ladder because it IS the perturbed quantity there (ADR-053). Emitted by "
        "`eval/baseline_run.py` as `band_growth_calibration`.",
    ),
    # --- G3's ADR-020 CALIBRATION criterion, DISQUALIFIED AT v2.15 by C6.6 --
    # Per lead, because G3 reports reliability at 1/2/3 h and C6.2 [v2.8]
    # adjudicates each horizon separately.
    **{
        f"{CALIBRATION_METRIC}_{h}h": MetricSpec(
            LOWER,
            False,
            quarantined_by="C6.6 (ADR-053 (1)(2))",
            note="ADR-020 G3 calibration criterion: max(forecast-bin deviation, "
            "frontier-ring deviation) on the growth-masked decision set, in "
            "probability POINTS. Null floor = the base rate exactly; oracle = 0. "
            + _ANTI_MONOTONE.format(rho="-0.14"),
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
    # Already non-gating since v2.10; C6.6 adds a SECOND, independent reason at
    # v2.15 and it is the stronger one — reliability's Spearman is -0.80, the
    # worst of the four. Both citations are kept: a clause retired for one reason
    # can be revisited on that reason, and the reader must see the other.
    "reliability_1h": MetricSpec(
        LOWER,
        False,
        quarantined_by="ADR-020 (G3 rewritten); C6.6 (ADR-053 (1)(2))",
        note="REL is a mean SQUARE while the bar is stated in POINTS, and it is "
        "satisfied exactly by climatology. G3's calibration half is now "
        f"{CALIBRATION_METRIC}_*h. " + _ANTI_MONOTONE.format(rho="-0.80"),
    ),
    "reliability_3h": MetricSpec(
        LOWER,
        False,
        quarantined_by="ADR-020 (G3 rewritten); C6.6 (ADR-053 (1)(2))",
        note="see reliability_1h. " + _ANTI_MONOTONE.format(rho="-0.80"),
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
    "reliability_2h": MetricSpec(
        LOWER,
        False,
        quarantined_by="ADR-020 (G3 rewritten); C6.6 (ADR-053 (1)(2))",
        note="see reliability_1h. " + _ANTI_MONOTONE.format(rho="-0.80"),
    ),
    "resolution_1h": MetricSpec(HIGHER, False, note="Murphy decomposition term; reported."),
    "resolution_2h": MetricSpec(HIGHER, False),
    "resolution_3h": MetricSpec(HIGHER, False),
}


# --------------------------------------------------------------------------
# [v2.15] C6.6 — ask the registry whether a channel may decide anything
# --------------------------------------------------------------------------


class NonAdjudicatingMetricError(RuntimeError):
    """Raised when a gate is about to be decided on a channel the contract bars."""


def adjudicating_metrics() -> frozenset[str]:
    """Every channel the contract currently lets decide a gate.

    DERIVED from the flags, never restated: a second list would be a second
    source of truth, and this project has now paid three times for one of those
    going stale (ADR-057). The trio itself is pinned by
    ``tests/test_null_check.py``, which is where a change should be visible.
    """
    return frozenset(name for name, spec in C6_METRICS.items() if spec.gate_eligible)


def may_adjudicate(metric: str) -> bool:
    """Is ``metric`` allowed to decide a gate? Unregistered means NO."""
    spec = C6_METRICS.get(metric)
    return spec is not None and spec.gate_eligible


def assert_may_adjudicate(metric: str, *, gate: str = "a gate") -> MetricSpec:
    """RAISE unless ``metric`` may decide ``gate``. Returns its spec if it may.

    **Raises rather than warns, and that is the whole point (C6.6).** The failure
    being repaired is a human quoting a number while forgetting a ruling made
    two weeks ago; a warning is read by exactly the person who already forgot.
    C-1 says an unverifiable guard is a hard failure, and "may this decide"
    answered by memory is unverifiable.

    An UNREGISTERED metric raises too. An unknown channel is not a permitted one:
    that is C-2 one level down, and it is why ``run_null_check`` already refuses
    to skip a metric it does not know.
    """
    spec = C6_METRICS.get(metric)
    if spec is None:
        raise NonAdjudicatingMetricError(
            f"{metric!r} is not in C6_METRICS, so nothing has ruled on whether it may decide "
            f"{gate}. An unregistered metric is an unchecked metric; register it with a "
            "direction and a gate_eligible flag before letting it decide anything."
        )
    if not spec.gate_eligible:
        raise NonAdjudicatingMetricError(
            f"{metric!r} MAY NOT decide {gate}: {spec.quarantined_by or 'reported-only'}. "
            f"{spec.note} Channels that may adjudicate today: "
            f"{sorted(adjudicating_metrics())}. Report this number if it is informative; "
            "do not let it pass or fail anything."
        )
    return spec
