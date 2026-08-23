"""THE TEXT REPORT AND THE COMMAND LINE.

    .venv/bin/python -m wildfire_nowcast.common.null_check
    make null-check

The report prints every verdict, both of them, whether or not anything fired -
including the positive controls, so an empty control set is visible as a
sentence rather than as an absence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.common.null_check.forecasters import DEGENERATE, ZERO_CLAIM
from wildfire_nowcast.common.null_check.registry import (
    CAPTURE_NOT_APPLICABLE,
    VERDICT_BLIND,
    VERDICT_BROKEN,
    VERDICT_OK,
    VERDICT_PAYS_FOR_NOTHING,
    VERDICT_SILENCE_FAVOURING,
    VERDICT_UNDECIDABLE,
)
from wildfire_nowcast.common.null_check.verdicts import (
    DEFAULT_MEMBERS,
    DEFAULT_SEEDS,
    MetricVerdict,
    NullCheckReport,
    run_null_check,
)
from wildfire_nowcast.common.null_check.windows import synthetic_windows, windows_from_tensor


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
            scores = " ".join(f"{_abbrev(k)}={_num(v.scores.get(k))}" for k in sorted(v.scores))
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
    add_logging_arguments(parser)
    args = parser.parse_args(argv)
    # ADR-103: the ONE place this program is allowed to configure logging.
    configure_from_args(args)

    if args.tensor:
        windows, scenario = windows_from_tensor(
            args.tensor, horizon_h=args.horizon, max_windows=args.max_windows
        )
    else:
        windows, scenario = synthetic_windows(horizon_h=args.horizon)
        if args.max_windows:
            windows = windows[: args.max_windows]
    report = run_null_check(windows, scenario, n_members=args.members, seeds=tuple(args.seeds))
    print(format_report(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2))
    return 0 if (report.reporting_ok if args.strict else report.ok) else 1
