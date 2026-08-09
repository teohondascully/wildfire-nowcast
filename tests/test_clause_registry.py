"""C-2 made mechanical: no clause may exist only on paper.

C1.5's ``features must be finite`` was ratified at v2.3 and never implemented.
It sat in INTERFACES.md for two contract versions while an all-NaN channel
passed 56 checks and ``+inf`` was actively blessed by two green clauses. It was
found by hand, late, and only because someone went looking.

This module makes that audit permanent and automatic:

1. Every numbered clause in ``docs/interfaces.md`` appears in
   :data:`CLAUSE_IMPLEMENTATIONS` with a status. **A new clause fails this
   suite until someone classifies it** — which is the entire point: ratifying a
   clause now breaks the build until it is implemented, deferred on the record,
   or declared unenforceable with a reason.
2. Registry entries cannot lie. Every check id an entry claims must actually
   appear in a report generated from conformant artifacts, and every dotted
   symbol must import.
3. Every deferred clause is in ``DEFERRED_CLAUSES`` and therefore printed on
   every contract report.
4. Every pass/fail constant states the sample it was fitted on (C-3).

The parser reads INTERFACES.md rather than a hand-kept list, so the check
cannot drift from the contract it audits.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from wildfire_nowcast.common import contract as C
from wildfire_nowcast.common import splits as S
from wildfire_nowcast.common.paths import repo_root
from wildfire_nowcast.common.synthetic import SyntheticFire

_INTERFACES = "docs/interfaces.md"

#: Clause ids in a heading: ``## C1.``, ``### [v2.3] C1.5 ...``, ``## C-2.``
_CLAUSE_RE = re.compile(r"^#{2,3}\s+(?:\[[^\]]+\]\s+)?(C-?\d+(?:\.\d+)?)\b")

#: ...and in a BOLD LINE: ``**C-4.1 OWNERSHIP OF eval/ IS SPLIT...**``
#:
#: WHY THIS SECOND FORM EXISTS [A12]. C-4.1 and C-4.2 were ratified at v2.11 as
#: bold paragraphs INSIDE C-4's section rather than as their own headings, so the
#: heading parser saw only ``C-4`` and **both sub-clauses were invisible to the
#: C-2 audit**. That is the C1.5 failure with the auditor as its subject: a
#: clause the audit cannot see can stay fiction indefinitely without ever
#: breaking the build, and the build would look green while it did.
#:
#: The fix is here rather than in INTERFACES.md deliberately — infra does
#: not edit that file, and more importantly **an audit whose coverage depends on
#: another lead's markdown style is not an audit.** Matching only at line start
#: keeps it from harvesting the dozens of in-prose references ("C-1's corollary",
#: "(C-3)"): measured against v2.11, this pattern matches exactly C-4.1 and
#: C-4.2 and nothing else. Widening can only ever make the build stricter.
_CLAUSE_BOLD_RE = re.compile(r"^\*\*(C-?\d+(?:\.\d+)?)\b")


def interfaces_path() -> Path:
    return repo_root() / _INTERFACES


def clauses_in_interfaces() -> set[str]:
    text = interfaces_path().read_text().splitlines()
    return {
        m.group(1)
        for line in text
        for pattern in (_CLAUSE_RE, _CLAUSE_BOLD_RE)
        if (m := pattern.match(line))
    }


@pytest.fixture(scope="module")
def emitted_check_ids(default_synthetic: SyntheticFire) -> set[str]:
    """Every check id the checker emits on CONFORMANT artifacts.

    Conformant on purpose: a clause that only ever appears in a failure branch
    is a clause nobody sees pass, and this is the set a registry claim is
    checked against.
    """
    ids = {c.check_id for c in C.check_all(default_synthetic.tensor_path).checks}
    ids |= {c.check_id for c in S.check_split_assignment().checks}
    ids |= {
        c.check_id
        for c in S.check_run_split(
            {
                "split_before": {"fingerprint": "x"},
                "split_after": {"fingerprint": "x"},
                # [v2.11] C-4.2's clauses only emit when a code fingerprint is
                # present, so a conformant payload must carry one at BOTH ends —
                # otherwise the registry would claim two check ids that this
                # fixture can never make appear.
                "common_code_before": {"fingerprint": "code-x"},
                "common_code_after": {"fingerprint": "code-x"},
                # [v2.12] C-4.3's clauses only emit when an ENVIRONMENT
                # fingerprint is present, and for the same reason as above a
                # conformant payload must carry one at BOTH ends. This suite
                # caught the omission the moment C-4.3 was registered, which is
                # the fourth time the C-2 audit has caught its own maintainer.
                "environment_before": {"fingerprint": "env-x"},
                "environment_after": {"fingerprint": "env-x"},
            },
            current={"fingerprint": "x"},
        ).checks
    }
    return ids


def test_interfaces_parses_into_the_clauses_we_expect() -> None:
    """Guard the parser itself: if it silently matched nothing, everything passes."""
    found = clauses_in_interfaces()
    assert {"C-1", "C-2", "C-3", "C0", "C1", "C1.1", "C1.7", "C2", "C3", "C8"} <= found
    assert len(found) >= 25, found


def test_bold_declared_subclauses_are_discovered_too() -> None:
    """C-4.1/C-4.2 are declared in BOLD, not in a heading — pin that they are seen.

    Without this, a future contract edit that reformats a heading into a bold
    line silently removes a clause from the audit, and the build stays green
    while a ratified clause becomes unenforceable. The audit's own blind spot is
    the thing worth regression-testing: it is the only failure here that hides
    every other failure.
    """
    found = clauses_in_interfaces()
    assert {"C-4", "C-4.1", "C-4.2"} <= found, sorted(found)


def test_every_interfaces_clause_is_classified() -> None:
    """THE audit. A ratified clause that nobody has classified fails here.

    This is the check that would have caught C1.5 in 2026-08 at v2.3 instead of
    at v2.5, and it is the reason a future C9 cannot quietly become fiction.
    """
    declared = clauses_in_interfaces()
    registered = set(C.CLAUSE_IMPLEMENTATIONS)
    unclassified = sorted(declared - registered)
    assert not unclassified, (
        f"INTERFACES.md ratifies {unclassified} but contract.CLAUSE_IMPLEMENTATIONS does not "
        "classify them. C-2: ratification is not implementation. Add each with a status of "
        "enforced / external / process / deferred — a clause living only in INTERFACES.md is "
        "worse than no clause, because everyone downstream believes they are protected."
    )
    stale = sorted(registered - declared)
    assert not stale, f"registry claims clauses INTERFACES.md does not define: {stale}"


def test_registry_statuses_are_legal() -> None:
    legal = {C.CLAUSE_ENFORCED, C.CLAUSE_EXTERNAL, C.CLAUSE_PROCESS, C.CLAUSE_DEFERRED}
    bad = {k: v.status for k, v in C.CLAUSE_IMPLEMENTATIONS.items() if v.status not in legal}
    assert not bad, bad


def test_claimed_check_ids_really_run(emitted_check_ids: set[str]) -> None:
    """A registry entry cannot claim a check that does not exist.

    Without this, the registry is just a second document that can rot — the
    exact failure mode it exists to prevent, one level up.
    """
    missing: dict[str, list[str]] = {}
    for clause, impl in sorted(C.CLAUSE_IMPLEMENTATIONS.items()):
        absent = [cid for cid in impl.checks if cid not in emitted_check_ids]
        if absent:
            missing[clause] = absent
    assert not missing, (
        f"these clauses claim check ids that no report emits: {missing}. Either the check was "
        "renamed or the claim was aspirational."
    )


def _resolve(dotted: str) -> object:
    """Import ``pkg.mod.Class.attr``: longest importable prefix, then getattr."""
    parts = dotted.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:split_at]))
        except ImportError:
            continue
        for part in parts[split_at:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(f"no importable module prefix in {dotted!r}")


def test_claimed_symbols_import() -> None:
    unresolved: dict[str, str] = {}
    for clause, impl in sorted(C.CLAUSE_IMPLEMENTATIONS.items()):
        for dotted in impl.where:
            try:
                _resolve(dotted)
            except Exception as exc:  # noqa: BLE001 - the message is the assertion
                unresolved[f"{clause}:{dotted}"] = f"{type(exc).__name__}: {exc}"
    assert not unresolved, unresolved


def test_enforced_clauses_actually_point_at_something() -> None:
    """``enforced`` with neither a check nor a symbol is a claim with no referent."""
    empty = [
        clause
        for clause, impl in C.CLAUSE_IMPLEMENTATIONS.items()
        if impl.status in (C.CLAUSE_ENFORCED, C.CLAUSE_EXTERNAL) and not (impl.checks or impl.where)
    ]
    assert not empty, empty


def test_process_and_deferred_clauses_explain_themselves() -> None:
    silent = [
        clause
        for clause, impl in C.CLAUSE_IMPLEMENTATIONS.items()
        if impl.status in (C.CLAUSE_PROCESS, C.CLAUSE_DEFERRED) and not impl.note.strip()
    ]
    assert not silent, silent


def test_deferred_clauses_are_declared_on_every_report() -> None:
    """A deferred clause the report does not name is an undeclared gap (C-1)."""
    deferred = {
        clause
        for clause, impl in C.CLAUSE_IMPLEMENTATIONS.items()
        if impl.status == C.CLAUSE_DEFERRED
    }
    assert deferred == set(C.DEFERRED_CLAUSES), (deferred, set(C.DEFERRED_CLAUSES))


def test_c1_5_is_enforced_not_merely_ratified() -> None:
    """The specific regression this whole module generalises."""
    impl = C.CLAUSE_IMPLEMENTATIONS["C1.5"]
    assert impl.status == C.CLAUSE_ENFORCED
    assert "features_finite" in impl.checks


def test_every_threshold_states_its_fitting_sample() -> None:
    """C-3: no constant decides a pass/fail without naming the sample behind it."""
    constants = {
        "PHYSICAL_RANGES.canopy_cover",
        "FBFM40_CLASSES",
        "MIN_TRAIN_BLOCKS_FOR_REPORTING",
        "MIN_BUFFER_MARGIN_CELLS",
        "_COORD_TOL_M",
    }
    missing = sorted(constants - set(C.THRESHOLD_PROVENANCE))
    assert not missing, (
        f"{missing} decide pass/fail but state no fitting sample. C-3: any constant that "
        "decides a pass/fail must state the sample it was fitted on, and that sample must "
        "span >= 2 spatial blocks."
    )
    for name, why in C.THRESHOLD_PROVENANCE.items():
        assert len(why.strip()) > 40, name


def test_physical_ranges_only_hold_definitional_bounds() -> None:
    """C1.7 is DEFINITIONAL. A plausibility bound smuggled in here becomes a hard
    fail with a false-positive mode, which is the C1.6 mistake pointed the other way."""
    assert set(C.PHYSICAL_RANGES) == {"canopy_cover"}, (
        "adding a range here makes it a HARD FAIL. Rows 3/5/1/2 of the A7 table are "
        "plausibility, not definition, and need an ADR at `reporting` severity first."
    )


def test_contract_version_matches_interfaces() -> None:
    header = interfaces_path().read_text().splitlines()[0]
    assert C.CONTRACT_VERSION in header, (header, C.CONTRACT_VERSION)
