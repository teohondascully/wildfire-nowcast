"""The evidence tier: a CLASSIFICATION that charges rent, not a dispensation.

I33. ``cited_paths`` and ``cited_runs`` decide which tracked files carry citation
obligations. Until now that decision was ``rel.startswith("runs/")`` - a question
about a file, answered by asking where it lives. When the G6 evidence was
force-added under ``reports/figures/``, those artifacts stopped being outputs and
became CITERS, and their provenance strings - pointing at the rest of the
evidence, still untracked - turned seven tests red.

The alternative on the table was ``DEBT``/``EXEMPT``: declare each unresolvable
provenance string acceptable, one at a time, forever. That is a loosening. This
module exists to hold the replacement to a higher bar than "the tests went
green": the tier must be UNABLE to hide source, and every exclusion it grants
must cost something that is checked.

Every obligation below is exercised by a PLANT that is observed failing, not by
reading the implementation and agreeing with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cited_paths as CP  # noqa: E402
import cited_runs as CR  # noqa: E402
import evidence as E  # noqa: E402
from wildfire_nowcast.common.paths import repo_root  # noqa: E402

#: The declared membership, pinned as a SET and not a count (ADR-154). A count
#: cannot see a swap: one artifact leaving and another arriving keeps it still
#: while changing entirely what a clone can recompute.
EXPECTED_EVIDENCE_FILES = frozenset(
    {
        "reports/figures/s14g5/creek_cost_projection.json",
        "reports/figures/s16/creek_episode.json",
    }
)  # these two ARE tracked, so they resolve and may be spelled


#: Specimen paths, ASSEMBLED AT RUN TIME. This file is scanned by the very check
#: it tests: written literally, every plant below would BE a citation of a file
#: no cloner can open, and this module would fail `cited_paths` on its own
#: specimens. It did, on the first run - the fifth time a scanner in this
#: repository has been its own first offender. The tempting fix, declaring this
#: path, would make the one file guaranteed to contain invented paths the one
#: file that can never report one.
#: Split at every SLASH, not merely at a leading letter. The first attempt split
#: one letter off the front, which leaves a remainder that still carries a
#: separator and a file suffix - still a path, so the scan resolved it and
#: reported it, correctly. A fragment is safe only when it cannot be a path on
#: its own, which means it must contain no separator at all.
#:
#: The SECOND attempt failed too, and more instructively: the comment written to
#: explain the first failure SPELLED the offending remainder, so the sentence
#: describing the trap was itself an instance of it. Describe; never spell.
def _join(*parts: str) -> str:
    return "/".join(parts)


_REP = _join("rep" + "orts", "figures")
_RUNS = "ru" + "ns"
_SRC = _join("src", "a.py")
_GHOST = _join(_REP, "ghost.json")
_ORPHAN = _join(_REP, "orphan.json")
_GOOD = _join(_REP, "s16", "good_record.json")

#: Placed AFTER the fragments it is built from, which the first version was not.
EXPECTED_PREFIXES = (_RUNS + "/",)


def test_the_membership_is_the_set_it_is_declared_to_be() -> None:
    """Fails when an entry arrives AND when one leaves."""
    assert set(E.EVIDENCE_FILES) == EXPECTED_EVIDENCE_FILES
    assert E.EVIDENCE_PREFIXES == EXPECTED_PREFIXES


def test_every_entry_states_a_reason_a_reader_could_act_on() -> None:
    for rel, reason in E.EVIDENCE_FILES.items():
        assert len(reason) > 80, f"{rel}: a one-line reason is an assertion, not a reason"
        assert "recomput" in reason or "check" in reason or "quote" in reason, (
            f"{rel}: the reason must say what a READER gets, not what we wanted"
        )


# --------------------------------------------------------------------------
# THE THREE OBLIGATIONS, EACH PLANTED AND OBSERVED FIRING
# --------------------------------------------------------------------------


def test_an_untracked_declaration_is_a_phantom_and_is_reported(monkeypatch) -> None:
    """Obligation 1. A declaration is a claim that a reader can open the file."""
    monkeypatch.setattr(E, "EVIDENCE_FILES", {_GHOST: "x" * 90})
    problems = E.evidence_problems({_SRC}, {_GHOST: [_SRC]})
    assert len(problems) == 1, problems
    assert "NOT TRACKED" in problems[0]
    assert f"git add -f {_GHOST}" in problems[0], (
        "the message must name the command: the declaration and the add are one change"
    )


def test_source_can_never_be_classified_as_evidence(monkeypatch) -> None:
    """Obligation 2, and it is the clause that makes 'nothing is loosened' TRUE.

    Without it the tier is a universal escape hatch: any module could be made
    invisible to every citation scan by listing it here. Planted on a path that
    is TRACKED and CITED, so the only thing that can fire is the extension.
    """
    fired = []
    for suffix in (".py", ".md", ".yaml", ".html", ".sh"):
        rel = f"{_REP}/pretend{suffix}"
        monkeypatch.setattr(E, "EVIDENCE_FILES", {rel: "x" * 90})
        problems = E.evidence_problems({rel, _SRC}, {rel: [_SRC]})
        assert len(problems) == 1, (suffix, problems)
        assert "its extension is SOURCE" in problems[0], (suffix, problems)
        fired.append(suffix)
    assert fired == [".py", ".md", ".yaml", ".html", ".sh"]


def test_an_entry_nothing_cites_is_not_evidence(monkeypatch) -> None:
    """Obligation 3. Evidence exists to be read from the published surface."""
    rel = _ORPHAN
    monkeypatch.setattr(E, "EVIDENCE_FILES", {rel: "x" * 90})
    problems = E.evidence_problems({rel}, {})
    assert len(problems) == 1 and "cited by NO non-evidence file" in problems[0], problems

    # ...and a citation from OTHER EVIDENCE does not rescue it, or the tier could
    # bootstrap itself: two artifacts citing each other would justify both.
    monkeypatch.setattr(E, "EVIDENCE_FILES", {rel: "x" * 90})
    problems = E.evidence_problems({rel}, {rel: [_join(_RUNS, "m1.json")]})
    assert len(problems) == 1, problems

    # The POSITIVE control: one real reader clears it, so the assertions above
    # are about the citation and not about the plumbing.
    monkeypatch.setattr(E, "EVIDENCE_FILES", {rel: "x" * 90})
    assert E.evidence_problems({rel}, {rel: ["src/wildfire_nowcast/sim/g6_report.py"]}) == []


def test_a_healthy_entry_produces_no_problem(monkeypatch) -> None:
    """The negative control. If ``evidence_problems`` always fired, every plant
    above would pass while proving nothing about the conditions they name.

    One entry, satisfying all three obligations at once: a non-source extension,
    present in the index, and cited by a file that is not itself evidence.
    """
    rel = _GOOD
    monkeypatch.setattr(E, "EVIDENCE_FILES", {rel: "x" * 90})
    assert E.evidence_problems({rel, _SRC}, {rel: [_SRC]}) == []

    # ...and it is genuinely load-bearing: break ONE condition at a time and the
    # same call fires, so the empty list above is a verdict rather than a default.
    assert E.evidence_problems(set(), {rel: [_SRC]})  # not tracked
    assert E.evidence_problems({rel}, {rel: [_join(_RUNS, "x.json")]})  # only evidence cites it


# --------------------------------------------------------------------------
# THE TIER CANNOT DRIFT BETWEEN THE TWO SCANNERS
# --------------------------------------------------------------------------


def test_both_scanners_answer_the_citer_question_with_the_SAME_function() -> None:
    """Not 'both agree today' - the same object, so they cannot come to differ.

    Two scanners with two copies of this rule is how ``reports/`` evidence came
    to be scanned as source while ``runs/`` evidence was not.
    """
    assert CP.is_evidence is E.is_evidence
    assert CR.is_evidence is E.is_evidence

    # The third scanner. `tests/test_hygiene.py` excludes tracked evidence from
    # the internal-tooling tell scan for the same reason and, until I33, with its
    # own private copy of the prefix.
    import test_hygiene as HY

    assert HY.evidence is E, "the tell scan holds a SECOND module object for one file"
    assert HY.ARTIFACT_PREFIX == E.EVIDENCE_PREFIXES[0]


def test_the_prefix_behaviour_is_preserved_exactly() -> None:
    """I33 changed the KEY, not the verdict, for everything that already worked."""
    for rel in (
        _join(_RUNS, "m33_summary.json"),
        _join(_RUNS, "_m24_ladder.py"),
        _join(_RUNS, "deep", "er", "x.json"),
    ):
        assert E.is_evidence(rel)
    for rel in ("src/wildfire_nowcast/sim/g6_report.py", "README.md", "tests/test_hygiene.py"):
        assert not E.is_evidence(rel)


def test_a_tracked_artifact_that_is_NOT_declared_is_still_scanned_as_a_citer() -> None:
    """The tier cannot widen silently: exclusion requires a declaration.

    ``reports/figures/s14g5/g5_four_blocks.json`` is tracked evidence in every
    ordinary sense and is deliberately NOT in the tier, because it needed no
    exclusion. If membership ever became "anything under reports/figures", this
    goes green while the guarantee is gone.
    """
    assert not E.is_evidence("reports/figures/s14g5/g5_four_blocks.json")
    assert not E.is_evidence("reports/figures/s14g5/measured_cost_4blocks.json")


# --------------------------------------------------------------------------
# AGAINST THE REAL TREE
# --------------------------------------------------------------------------


def test_the_declared_evidence_is_excluded_from_the_citer_walk_and_nothing_else_is() -> None:
    """Measured on this repository, not on a fixture."""
    root = repo_root()
    tracked = set(CP.tracked_files(root))
    excluded = {rel for rel in tracked if E.is_evidence(rel)}
    from_prefix = {rel for rel in tracked if rel.startswith(EXPECTED_PREFIXES)}
    declared_and_tracked = EXPECTED_EVIDENCE_FILES & tracked
    assert excluded == from_prefix | declared_and_tracked, sorted(
        excluded ^ (from_prefix | declared_and_tracked)
    )


def test_the_tier_is_reported_on_every_run_not_only_when_it_fails() -> None:
    """A boundary that only speaks on failure is one a reader never learns about."""
    text = CP.report(CP.enumerate_references(repo_root()))
    assert "ARTIFACT TIER" in text
    assert "individually declared file(s)" in text
    for rel in EXPECTED_EVIDENCE_FILES:
        assert rel in text, f"{rel} is excluded and is not named in the report"


@pytest.mark.parametrize("rel", sorted(EXPECTED_EVIDENCE_FILES))
def test_each_declared_file_is_present_on_disk_even_if_not_yet_tracked(rel: str) -> None:
    """Separates the two failure modes the tier must never confuse.

    ABSENT means the declaration is fiction. UNTRACKED means the ``git add -f``
    has not landed yet and the entry is one command from true. ``cited_paths``
    reports the second; this asserts it is never silently the first.
    """
    assert (repo_root() / rel).is_file(), (
        f"{rel} is declared as evidence and does not exist on disk at all. That is "
        "not a pending add, it is a fiction, and no command fixes it."
    )
