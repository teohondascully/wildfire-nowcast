"""``common/null_check/cli.py`` - what ``make null-check`` actually prints.

The report is the artifact: C6.0's verdicts reach a human through this text and
nowhere else. Two things in it are load-bearing rather than cosmetic. A metric
that is reported but may not gate has to say WHICH clause quarantined it, or the
reader cannot tell a metric excluded by contract from one excluded by accident.
And the absent positive control has to be stated loudly, because an empty control
set is what a harness that stopped working looks like.
"""

from __future__ import annotations

from wildfire_nowcast.common.null_check.cli import format_report
from wildfire_nowcast.common.null_check.registry import (
    VERDICT_BROKEN,
    VERDICT_OK,
    VERDICT_SILENCE_FAVOURING,
)
from wildfire_nowcast.common.null_check.verdicts import MetricVerdict, NullCheckReport


def _report(*verdicts: MetricVerdict) -> NullCheckReport:
    return NullCheckReport(
        scenario={"source": "unit", "n_windows": 2, "grid_shape": [8, 8]},
        verdicts=list(verdicts),
    )


def test_a_quarantined_metric_is_printed_with_the_clause_that_quarantined_it() -> None:
    """A tag reading ``reported (C6.6)`` and one reading ``reported (n/a)`` differ.

    The first says a contract clause took this metric out of the gate on the
    record; the second says nobody knows why it is not gating. Collapsing them
    loses the only trace of C6.6 that reaches a reader of the report.
    """
    text = format_report(
        _report(
            MetricVerdict(
                metric="brier_1h",
                mask="growth",
                verdict=VERDICT_BROKEN,
                scores={"null": 0.1, "skill": 0.2},
                gate_eligible=False,
                quarantined_by="C6.6",
                detail="the null beat genuine skill",
            )
        )
    )
    assert "reported (C6.6)" in text, (
        f"the quarantining clause is not in the report, so a reader cannot tell why the "
        f"metric is not gating:\n{text}"
    )
    assert "n/a" not in text.split("brier_1h")[1].split("\n")[0]
    assert "the null beat genuine skill" in text, "a BROKEN verdict printed no reason"


def test_a_gate_eligible_metric_is_marked_GATE_and_carries_no_quarantine_note() -> None:
    """The control on the test above: the two tags are produced by one expression."""
    text = format_report(
        _report(
            MetricVerdict(
                metric="iou_shape_masked",
                mask="growth",
                verdict=VERDICT_OK,
                scores={},
                gate_eligible=True,
            )
        )
    )
    row = next(line for line in text.splitlines() if "iou_shape_masked" in line)
    assert "GATE" in row and "reported" not in row, row


def test_an_empty_positive_control_set_says_so_rather_than_reading_clean() -> None:
    """A harness that stopped detecting confirmed pathologies must not look healthy."""
    silent = format_report(
        _report(
            MetricVerdict(
                metric="brier_3h",
                mask="growth",
                verdict=VERDICT_SILENCE_FAVOURING,
                scores={},
                gate_eligible=True,
            )
        )
    )
    assert "NO POSITIVE CONTROL REPRODUCED" in silent

    confirmed = format_report(
        _report(
            MetricVerdict(
                metric="dispersion_ratio",
                mask="all",
                verdict=VERDICT_BROKEN,
                scores={},
                gate_eligible=False,
                quarantined_by="C6.1",
            )
        )
    )
    assert "positive controls reproduced" in confirmed
    assert "NO POSITIVE CONTROL REPRODUCED" not in confirmed
