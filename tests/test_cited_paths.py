"""Reachability of every path a tracked file names, and the checks on the checker.

WHAT THIS ASSERTS, IN ONE SENTENCE: a reader who clones this repository can open
every path its tracked files name, except the ones enumerated with a reason.

WHAT IT DOES NOT ASSERT: that the file at the other end SUPPORTS the sentence
citing it. Reachability and support are different properties (ADR-105 (3)) and
conflating them is how four citations of an 87-line prose file once satisfied a
provenance rule while backing nothing.

WHY THE CHECKS BELOW ARE MOSTLY ABOUT THE INSTRUMENT. The live tree passes, and a
passing scan is worth exactly as much as the evidence that it could still fail.
So the population assertion is one test, and the rest put the classifier in front
of inputs whose answer is known: a planted citation in a repository the walk has
never read, a tracked file that must NOT be reported, the three structural skips,
and the four directions the debt pin fails in. None of them depend on how much
debt is outstanding, so none of them dies the day the debt reaches zero, which is
how this repository lost its last anti-vacuity control (ADR-110).

SPECIMENS ARE JOINED AT RUNTIME. This file is read by the scan it tests, so a
specimen written literally would be a citation of a path that does not exist, and
this module would fail its own check. That is not hypothetical: it happened twice
in the session that wrote this file, once here and once in
``tests/test_large_file_guard.py``, and both were caught by the control run
rather than by review.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import cited_paths as C  # noqa: E402

from wildfire_nowcast.common.paths import repo_root  # noqa: E402


def _invented(stem: str, suffix: str) -> str:
    """A path that does not exist, assembled so this file holds no literal citation.

    The join is at the EXTENSION, which is where it has to be: splitting a path
    anywhere else leaves a fragment that is still path-shaped, and this scanner
    reads fragments. That was learned by writing it the other way first, in this
    file and in `tests/test_large_file_guard.py`, and being told so by the check.
    """
    return stem + suffix


#: Specimens, all assembled. None of these paths exists and none of them is
#: written literally anywhere in this file.
_SPECIMEN_DEEP = _invented("quite/deep/invented_v2", ".json")
_SPECIMEN_SHALLOW = _invented("invented/thing", ".py")
_SPECIMEN_INNER = _invented("inner/thing", ".py")
_SPECIMEN_INNER_GONE = _invented("inner/gone", ".py")
_SPECIMEN_INNER_DEEP = _invented("src/pkg/inner/thing", ".py")
_SPECIMEN_CITER = _invented("tests/nowhere", ".py")
_SPECIMEN_TARGET = _invented("invented/path", ".json")
_SPECIMEN_NOT_EXEMPT = _invented("runs/not_exempt/results", ".json")
_SPECIMEN_NEW_CITER = _invented("src/wildfire_nowcast/sim/brand_new", ".py")


def _enum() -> C.Enumeration:
    return C.enumerate_references(repo_root())


def _throwaway_repo(tmp_path: Path, body: str, name: str = "module.py") -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / name).write_text(body)
    subprocess.run(["git", "add", name], cwd=tmp_path, check=True)
    return tmp_path


# --------------------------------------------------------------------------
# the live tree
# --------------------------------------------------------------------------


def test_every_path_named_by_a_tracked_file_resolves_or_is_declared() -> None:
    """The gate. Everything else in this file exists to make this one mean something."""
    enum = _enum()
    assert enum.problems == [], "\n".join(enum.problems)


def test_the_corpus_is_not_empty_and_the_resolution_kinds_are_populated() -> None:
    """Anti-vacuity on the WALK: a scan over nothing satisfies the gate above."""
    enum = _enum()
    assert len(enum.references) > 300, f"only {len(enum.references)} references found"
    assert enum.counts.get("tracked", 0) > 100, enum.counts
    assert enum.counts.get("unresolved", 0) == sum(C.DEBT.values()), enum.counts


def test_the_debt_is_owned_by_other_leads_and_infra_has_none() -> None:
    """Infra fixes its own the day it finds them; a burn-down it could clear is a to-do list."""
    mine = [
        rel
        for rel in C.DEBT
        if rel.startswith(("tools/", "tests/", "configs/", "src/wildfire_nowcast/common/"))
    ]
    assert mine == [], f"infra declared debt on its own surface instead of fixing it: {mine}"
    assert all(count > 0 for count in C.DEBT.values()), "a zero entry is STALE, remove it"


# --------------------------------------------------------------------------
# the capability: a citation this walk has never seen
# --------------------------------------------------------------------------


def test_a_planted_citation_is_reported_by_the_real_walk(tmp_path: Path) -> None:
    """Plant one in a repository the scan has never read, and read the verdict."""
    root = _throwaway_repo(tmp_path, f'"""Reads ``{_SPECIMEN_DEEP}``."""\n')
    enum = C.enumerate_references(root, declared={}, debt={})
    problems = "\n".join(enum.problems)
    assert "module.py" in problems, problems
    assert [r.resolution for r in enum.references] == ["unresolved"], enum.references


def test_a_citation_of_a_tracked_file_is_not_reported(tmp_path: Path) -> None:
    """The negative half, on the same walk and the same repository shape.

    Without it, a classifier that answered UNRESOLVED to everything would pass
    the test above and mean nothing.
    """
    root = _throwaway_repo(tmp_path, f'"""Reads ``{_SPECIMEN_SHALLOW}``."""\n')
    target = root / _SPECIMEN_SHALLOW
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n")
    subprocess.run(["git", "add", _SPECIMEN_SHALLOW], cwd=root, check=True)
    enum = C.enumerate_references(root, declared={}, debt={})
    assert enum.problems == [], "\n".join(enum.problems)
    assert [r.resolution for r in enum.references] == ["tracked"], enum.references


def test_a_path_relative_to_a_subtree_resolves_by_SUFFIX(tmp_path: Path) -> None:
    """``common/contract.py`` names a real file even though the repo path is longer.

    A reader can find it, so it is reachable. This is the one resolution rule
    that is a judgement rather than an identity, and it is put in front of a case
    that must resolve and a case that must not.
    """
    body = f'"""Reads ``{_SPECIMEN_INNER}`` and ``{_SPECIMEN_INNER_GONE}``."""\n'
    root = _throwaway_repo(tmp_path, body)
    (root / _SPECIMEN_INNER_DEEP).parent.mkdir(parents=True)
    (root / _SPECIMEN_INNER_DEEP).write_text("x = 1\n")
    subprocess.run(["git", "add", _SPECIMEN_INNER_DEEP], cwd=root, check=True)
    enum = C.enumerate_references(root, declared={}, debt={})
    by_token = {r.token: r.resolution for r in enum.references}
    assert by_token[_SPECIMEN_INNER] == "suffix", by_token
    assert by_token[_SPECIMEN_INNER_GONE] == "unresolved", by_token


# --------------------------------------------------------------------------
# the three structural skips, demonstrated on strings small enough to read
# --------------------------------------------------------------------------


def test_a_variable_expansion_is_not_a_repository_path() -> None:
    """``$$d/tensor.zarr`` in the Makefile is a shell variable, not a citation."""
    assert list(C.tokens_in("$$d/tensor.zarr")) == []
    assert list(C.tokens_in('exec "$root/tools/push_guard.py"')) == []


def test_an_absolute_path_is_not_a_repository_path() -> None:
    """The tail of an absolute path is not a repository-relative citation.

    Absolute-path literals are a different defect with its own check in
    ``tests/test_hygiene.py``; what must not happen is this scanner reporting the
    tail of one as a broken relative citation. The specimen is assembled for two
    reasons at once: that other check reads this file for volume roots, and this
    one reads it for path-shaped fragments.
    """
    absolute = "/Vol" + "umes/scratch2/fires/tensor" + ".zarr"
    assert list(C.tokens_in(absolute)) == []


def test_a_run_of_extensions_is_not_a_path() -> None:
    """``.md/.rst/.txt`` in a help string is three suffixes with slashes between them."""
    assert list(C.tokens_in("Python modules only, no .md/.rst/.txt")) == []


def test_the_skips_have_not_swallowed_the_ordinary_case() -> None:
    """The control for the three above. A skip list that skips everything is silent."""
    found = [token for token, _line in C.tokens_in("see docs/interfaces.md for the clause")]
    assert found == ["docs/interfaces.md"], found
    lines = [line for _token, line in C.tokens_in("a\nb\nsee tools/cited_runs.py here")]
    assert lines == [3], lines


# --------------------------------------------------------------------------
# the four directions the pin fails in, on synthetic input
# --------------------------------------------------------------------------


def test_an_undeclared_citer_fails() -> None:
    problems = C._audit_debt({_SPECIMEN_NEW_CITER: 1})
    assert any(p.startswith("UNRESOLVABLE") for p in problems), problems


def test_a_risen_count_fails() -> None:
    rel = next(iter(C.DEBT))
    problems = C._audit_debt(dict.fromkeys([rel], C.DEBT[rel] + 1))
    assert any(p.startswith("RISEN") for p in problems), problems


def test_a_fallen_count_fails_so_a_sweep_is_recorded() -> None:
    rel = next(rel for rel, count in C.DEBT.items() if count > 1)
    problems = C._audit_debt({rel: C.DEBT[rel] - 1})
    assert any(p.startswith("FALLEN") for p in problems), problems


def test_a_cleared_file_is_STALE_until_its_entry_is_removed() -> None:
    problems = C._audit_debt({})
    assert len([p for p in problems if p.startswith("STALE")]) == len(C.DEBT), problems


def test_a_declaration_that_describes_nothing_is_stale() -> None:
    pairs = {(_SPECIMEN_CITER, _SPECIMEN_TARGET): "specimen"}
    problems = C._audit_declarations(pairs, seen=set())
    assert problems and problems[0].startswith("STALE DECLARATION"), problems
    assert C._audit_declarations(pairs, seen={(_SPECIMEN_CITER, _SPECIMEN_TARGET)}) == []


def test_an_evidence_declaration_must_be_backed_by_the_module_that_owns_the_reason() -> None:
    """The ``evidence`` category borrows ``cited_runs.EXEMPT``. It is held to it."""
    fake = {(_SPECIMEN_CITER, _SPECIMEN_NOT_EXEMPT): "evidence"}
    problems = C._audit_declarations(fake, seen=set(fake))
    assert any(p.startswith("UNBACKED EVIDENCE") for p in problems), problems


# --------------------------------------------------------------------------
# the checker is scanned by itself
# --------------------------------------------------------------------------


def test_the_declaration_files_hide_nothing() -> None:
    """Both files carrying declarations are scanned, with no path-shaped exemption.

    A token inside them resolves ONLY if it is a path this module declares
    somewhere. Anything else is an ordinary unresolved citation, so a new broken
    reference cannot be smuggled into the one place a reader would not look for
    it.
    """
    enum = _enum()
    declared_tokens = {token for _citer, token in C._declared_pairs()}
    for ref in enum.references:
        if ref.citer in C.DECLARATION_FILES and ref.resolution == "declaration":
            assert ref.token in declared_tokens, ref
    scanned = {ref.citer for ref in enum.references}
    assert "tools/cited_paths.py" in scanned, "the checker is not being scanned by itself"


def test_both_declaration_files_are_tracked_so_the_scan_reads_them() -> None:
    files = set(C.tracked_files(repo_root()))
    for rel in C.DECLARATION_FILES:
        assert rel in files, f"{rel} is not tracked, so the scan never reads it"


def test_every_declared_category_states_a_reason() -> None:
    assert C.DECLARED, "an empty declaration table makes the category tests vacuous"
    for category, (reason, pairs) in C.DECLARED.items():
        assert len(reason.split()) > 15, f"{category}: a one-line reason is not a reason"
        assert pairs, f"{category}: a category with no members is stale"
