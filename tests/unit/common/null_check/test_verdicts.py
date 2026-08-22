"""``common/null_check/verdicts.py`` - C6.0's two answers per metric, and the replicates.

A null check that runs each forecaster once reports a DIRECTION for a difference
that is a coin flip. The seed tuple is what stops that, and it stops it by being
a set of distinct seeds rather than by having five entries.
"""

from __future__ import annotations

from wildfire_nowcast.common.null_check.registry import (
    CAPTURE_NOT_APPLICABLE,
    VERDICT_BLIND,
    VERDICT_BROKEN,
    VERDICT_OK,
    VERDICT_PAYS_FOR_NOTHING,
    VERDICT_SILENCE_FAVOURING,
)
from wildfire_nowcast.common.null_check.verdicts import (
    DEFAULT_MEMBERS,
    DEFAULT_SEEDS,
    MetricVerdict,
    NullCheckReport,
)


def test_the_default_seeds_are_distinct_because_a_repeat_is_not_a_replicate() -> None:
    """Five entries holding four values is a four-seed harness that reports five.

    Nothing downstream would notice: the tuple still has length five, every run
    still completes, and the spread across "seeds" simply comes out narrower than
    it should, which makes a coin flip look like a direction. That is the failure
    this constant was introduced to prevent, so the property to hold is
    distinctness, not length.
    """
    assert len(set(DEFAULT_SEEDS)) == len(DEFAULT_SEEDS), (
        f"DEFAULT_SEEDS={DEFAULT_SEEDS} repeats a value, so the harness has fewer "
        "independent replicates than it reports"
    )
    assert len(DEFAULT_SEEDS) >= 3, "one or two replicates cannot separate a direction from noise"
    assert all(isinstance(seed, int) for seed in DEFAULT_SEEDS)
    assert DEFAULT_MEMBERS >= 32, (
        "the member count is not a performance knob: below 32 the ensemble mean's own noise "
        "inflates a no-information forecast's ECE and hides the pathology"
    )


def test_a_quarantined_metric_never_fails_the_gate_and_is_still_reported() -> None:
    """C6.6: a metric that may not adjudicate cannot be the reason a gate closes."""
    broken = MetricVerdict(
        metric="brier_1h",
        mask="growth",
        verdict=VERDICT_BROKEN,
        scores={},
        gate_eligible=False,
        quarantined_by="C6.6",
    )
    assert not broken.is_failure and not broken.is_reporting_gap
    assert broken.is_flagged, "a quarantined metric still reports what it saw"

    gating = MetricVerdict(
        metric="iou_shape_masked",
        mask="growth",
        verdict=VERDICT_BROKEN,
        scores={},
        gate_eligible=True,
    )
    assert gating.is_failure


def test_the_two_answers_are_separate_and_either_one_can_fail_alone() -> None:
    """The comparison and the zero-capture axiom disagree measurably (ADR-022 (1))."""
    pays = MetricVerdict(
        metric="reliability",
        mask="all",
        verdict=VERDICT_OK,
        scores={},
        gate_eligible=True,
        capture_verdict=VERDICT_PAYS_FOR_NOTHING,
    )
    assert pays.is_failure, "a metric that pays for an empty forecast passed on the comparison"

    blind = MetricVerdict(
        metric="dispersion_ratio",
        mask="all",
        verdict=VERDICT_BLIND,
        scores={},
        gate_eligible=True,
        capture_verdict=CAPTURE_NOT_APPLICABLE,
    )
    assert blind.is_failure

    silent = MetricVerdict(
        metric="brier_3h",
        mask="growth",
        verdict=VERDICT_SILENCE_FAVOURING,
        scores={},
        gate_eligible=True,
    )
    assert silent.is_reporting_gap and not silent.is_failure

    report = NullCheckReport(scenario={}, verdicts=[silent])
    assert report.ok and not report.reporting_ok
    assert report.reporting_gaps() == [silent] and report.failures() == []

    report.problems.append("the harness could not score anything")
    assert not report.ok, "a problem was not enough to fail the report"
