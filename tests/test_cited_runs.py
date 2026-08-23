"""The ``runs/`` citation class: enumerated by a walk, not by a typed pattern.

The class was first enumerated with ``runs/[A-Za-z0-9_]*\\.json`` and came back
as twelve. It was twenty-one then and is twenty-four now, the three additions
being M19's sweep artifacts. The two misses were STRUCTURAL rather than unlucky:
the character class excludes ``/`` and ``-``, so no subdirectory can ever match,
and the literal suffix excludes every extension but one. The five files that
second miss hid are the analysis scripts that produced published numbers.

These tests hold the replacement to a higher bar than "it found more": they
measure both patterns on the SAME corpus, they demonstrate the superseded
pattern's two blind spots on strings small enough to read, and they plant a
citation in a throwaway repository to prove the walk reports one it has never
seen before.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import cited_runs as C  # noqa: E402

from wildfire_nowcast.common.paths import repo_root  # noqa: E402

#: What the tree is expected to hold. Both halves are pinned: a citation that
#: stops being tracked fails, and a NEW citation that nobody accounted for fails.
#:
#: 17 -> 20 when M19's three sweep artifacts were tracked, so that the finding
#: that the documented collapse ablation fires 97 times in 200 can be CHECKED by
#: a clone rather than only re-run. Both numbers here were OBSERVED FAILING
#: FIRST (``assert 20 == 17`` and, below, ``assert 15 == 12``); neither was
#: derived from the change and then confirmed by the change.
EXPECTED_TRACKED = 20
EXPECTED_EXEMPT = 4
EXPECTED_CLASS = EXPECTED_TRACKED + EXPECTED_EXEMPT

#: The internal coordination role, spelled in halves for the same reason
#: ``tests/test_hygiene.py`` does it: this file is scanned by that check too, and
#: a scan that cannot see one file is the failure this repository keeps paying
#: for.
_ROLE = re.compile("orchestr" + "ator", re.IGNORECASE)

#: Specimen tokens, spelled in HALVES for the same reason and with the same
#: consequence. This file is read by the very scan it tests: written literally,
#: three invented paths would BE citations of files that do not exist, and this
#: module would fail its own check. The tempting fix - exempt this path - would
#: make the one file guaranteed to contain unresolvable citations the one file
#: that can never report one. That is the shape of a check that cannot fail.
_RUNS = "ru" + "ns/"
_SPECIMEN_DEEP = _RUNS + "deep/er/still-here_v2.json"
_SPECIMEN_SCRIPT = _RUNS + "tool.py"
_SPECIMEN_NESTED = _RUNS + "kernel-nll_only-20260808-044220/model.json"


def _enum() -> C.Enumeration:
    return C.enumerate_citations(repo_root())


# --------------------------------------------------------------------------
# THE CLASS ITSELF
# --------------------------------------------------------------------------


def test_every_cited_runs_file_is_tracked_or_declared() -> None:
    """A path in published source that a public reader cannot open is a defect."""
    enum = _enum()
    assert enum.problems == [], "\n".join(enum.problems)


def test_the_class_is_the_size_it_is_declared_to_be() -> None:
    enum = _enum()
    tracked = enum.of_kind("tracked")
    exempt = enum.of_kind("exempt")
    assert len(tracked) == EXPECTED_TRACKED, [c.token for c in tracked]
    assert len(exempt) == EXPECTED_EXEMPT, [c.token for c in exempt]
    assert len(tracked) + len(exempt) == EXPECTED_CLASS


def test_the_five_analysis_scripts_are_in_the_tree() -> None:
    """The half the superseded pattern could not see, named individually."""
    tracked = {c.token for c in _enum().of_kind("tracked")}
    for name in (
        "runs/_m9_response.py",
        "runs/_m9_scaling.py",
        "runs/_m10_bitidentity.py",
        "runs/_s1_bitidentity.py",
        "runs/_s1_score.py",
    ):
        assert name in tracked, f"{name} is cited by tracked source and is not tracked"


# --------------------------------------------------------------------------
# THE CONTROL: BOTH PATTERNS, ONE CORPUS
# --------------------------------------------------------------------------


def _source_file_tokens(pattern: re.Pattern[str]) -> set[str]:
    """File-shaped ``runs/`` tokens cited by tracked NON-artifact files."""
    base = repo_root()
    out: set[str] = set()
    for rel in C.tracked_files(base):
        if rel.startswith("runs/"):
            continue
        path = base / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        out |= {t for t in C.tokens_in(text, pattern) if t.endswith(C.FILE_SUFFIXES)}
    return out


def test_the_replacement_finds_the_whole_class_where_the_superseded_pattern_found_twelve() -> None:
    """Both measured here, now, on one corpus. Neither number is remembered."""
    superseded = _source_file_tokens(C.SUPERSEDED_PATTERN)
    replacement = _source_file_tokens(C.RUNS_TOKEN)

    # 12 -> 15: the three M19 artifacts are ``runs/<word>.json``, which is the one
    # shape the superseded pattern could always see. The number that carries the
    # finding is ``missed`` below, and it is UNCHANGED at 9.
    assert len(superseded) == 15, sorted(superseded)
    assert len(replacement) == EXPECTED_CLASS, sorted(replacement)
    assert superseded < replacement, "the replacement must be a strict superset"

    missed = replacement - superseded
    assert len(missed) == 9, sorted(missed)
    assert sum(1 for t in missed if t.endswith(".py")) == 5, sorted(missed)
    assert sum(1 for t in missed if "/results.json" in t) == 4, sorted(missed)


@pytest.mark.parametrize(
    ("token", "why"),
    [
        ("runs/baselines-20260808-095003/results.json", "a subdirectory: `/` is not in the class"),
        ("runs/_s1_score.py", "a non-json extension: the suffix is a literal"),
        (_SPECIMEN_NESTED, "a hyphen AND a subdirectory"),
    ],
)
def test_the_superseded_pattern_is_blind_by_construction(token: str, why: str) -> None:
    """The blind spots are demonstrated on three strings, not inferred from a count."""
    assert not C.SUPERSEDED_PATTERN.fullmatch(token), why
    assert C.RUNS_TOKEN.fullmatch(token), f"the replacement must match: {token}"


def test_the_superseded_pattern_still_matches_what_it_was_built_for() -> None:
    """A negative control. If it matched nothing, the comparison would be empty."""
    assert C.SUPERSEDED_PATTERN.fullmatch("runs/s1.json")
    assert C.SUPERSEDED_PATTERN.fullmatch("runs/m6_innovation_autocorr.json")


# --------------------------------------------------------------------------
# THE POSITIVE CONTROL: A CITATION THIS SCAN HAS NEVER SEEN
# --------------------------------------------------------------------------


def _throwaway_repo(tmp_path: Path, body: str) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "module.py").write_text(body)
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    return tmp_path


def test_a_planted_citation_is_reported_by_the_real_walk(tmp_path: Path) -> None:
    """Plant one, in a repository the scan has never read, and read the verdict."""
    root = _throwaway_repo(
        tmp_path,
        f'"""Reads ``{_SPECIMEN_DEEP}`` and ``{_SPECIMEN_SCRIPT}``."""\n',
    )
    enum = C.enumerate_citations(root)
    problems = "\n".join(enum.problems)
    assert _SPECIMEN_DEEP in problems, problems
    assert _SPECIMEN_SCRIPT in problems, problems


def test_a_citation_of_a_tracked_file_is_not_reported(tmp_path: Path) -> None:
    """The negative half: tracking the cited file clears it. Same walk, same repo."""
    root = _throwaway_repo(tmp_path, f'"""Reads ``{_SPECIMEN_SCRIPT}``."""\n')
    (root / "runs").mkdir()
    (root / _SPECIMEN_SCRIPT).write_text("x = 1\n")
    subprocess.run(["git", "add", "-f", _SPECIMEN_SCRIPT], cwd=root, check=True)
    enum = C.enumerate_citations(root)
    assert enum.problems == [], "\n".join(enum.problems)


def test_a_directory_citation_is_reported_and_never_enforced(tmp_path: Path) -> None:
    """A run DIRECTORY is not an obligation: tracking one means tracking checkpoints."""
    root = _throwaway_repo(tmp_path, '"""    make replay RUN=runs/baselines-20260808-052918"""\n')
    enum = C.enumerate_citations(root)
    assert enum.problems == []
    assert [c.token for c in enum.of_kind("dir_or_prefix")] == ["runs/baselines-20260808-052918"]


# --------------------------------------------------------------------------
# THE EXEMPTION CARRIES ITS OWN REASON, AND THE REASON IS MEASURED
# --------------------------------------------------------------------------


def test_every_exemption_states_a_reason_and_a_number() -> None:
    assert C.EXEMPT, "an empty exemption list would make the class test vacuous"
    for token, (reason, count) in C.EXEMPT.items():
        assert reason.strip(), token
        assert count > 0, f"{token}: an exemption whose measurement is 0 has no reason left"


def test_the_exemption_reason_is_still_true_where_the_file_is_present() -> None:
    """Fails in BOTH directions: a count that rises and a count that falls.

    Skipped in a clone, where the untracked evidence does not exist. That is
    stated rather than hidden: this assertion is a local one, and the assertion
    that gates CI is ``test_every_cited_runs_file_is_tracked_or_declared``, which
    is derived from git and therefore identical everywhere.
    """
    base = repo_root()
    checked = 0
    for token, (_reason, expected) in C.EXEMPT.items():
        path = base / token
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        actual = len(_ROLE.findall(text))
        assert actual == expected, (
            f"{token}: the exemption is justified by {expected} internal-role occurrences "
            f"and the file now carries {actual}. If it is 0 the file is publishable and the "
            "exemption must go; if it moved, the reason must be restated with the new number."
        )
        checked += 1
    if checked == 0:
        pytest.skip("no exempt artifact present: this is a clone, not the evidence machine")


def test_this_module_is_scanned_by_the_check_it_tests_and_is_not_exempt() -> None:
    """No self-exemption, asserted rather than trusted.

    The specimens above are joined at runtime so this file contains no literal
    unresolvable citation. That property is what buys the absence of an
    exemption, so it is checked here rather than assumed.
    """
    rel = "tests/test_cited_runs.py"
    assert rel in C.tracked_files(repo_root()), "not tracked, so the scan never reads it"
    source = (repo_root() / rel).read_bytes().decode()
    unresolved = {
        t
        for t in C.tokens_in(source)
        if t.endswith(C.FILE_SUFFIXES) and t not in C.EXEMPT and t not in set(C.tracked_files())
    }
    assert not unresolved, f"this file now contains a LITERAL unresolvable citation: {unresolved}"
    # ...and the fragments really are here, so the halves are halves.
    assert "ru" in source and _RUNS in _SPECIMEN_DEEP
