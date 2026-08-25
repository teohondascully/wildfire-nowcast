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

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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
_SPECIMEN_BARE_IN_A_DIRECTORY = _invented("data/fires/x/manifest", ".json")


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
    """On a SYNTHETIC entry, because the live table stopped being able to supply one.

    This selected a real ``DEBT`` entry carrying more than one citation and
    decremented it. When ``data/isotropy.py``'s 3 were discharged at D15 no entry
    with a count above 1 was left anywhere, and the generator expression raised
    ``StopIteration`` - so paying the debt down broke the test that proves the
    pin notices a payment. A capability parameterised off the live burn-down
    loses its power exactly when the burn-down is worked, which is the one moment
    it has to keep it. ``_audit_debt`` takes its table as an argument for this
    reason and the three neighbouring directions already use synthetic input.
    """
    rel = _SPECIMEN_NEW_CITER
    problems = C._audit_debt({rel: 2}, debt={rel: 3})
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


# --------------------------------------------------------------------------
# the delegations, EXECUTED rather than printed
# --------------------------------------------------------------------------
#
# A delegation is a claim about ANOTHER checker, and it is the one kind of claim
# a tool makes that its own tests cannot reach by accident. This module printed
# on every run that the bare-filename tier was "left to tests/test_hygiene.py,
# whose pattern set covers that class" and that pattern set covered exactly one
# token of it (ADR-116). Neither module was wrong about itself; nothing owned the
# hand-off. So each entry in `DELEGATED` names a module, a reader inside it and a
# PROBE, and the probe is run through the reader here, in both directions.


def _load(path: Path) -> ModuleType:
    """Import a module BY PATH, under a private name and outside ``sys.modules``.

    By path, because that is what the delegation names and it is what a reader
    following the delegation would open. Under a private name, because pytest has
    its own entry for the same file and a delegation check must not decide which
    of the two is authoritative.
    """
    spec = importlib.util.spec_from_file_location(f"_delegated_{path.stem}", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _delegation_problems(
    klass: str, entry: tuple[str, str, str], *, root: Path, tracked: set[str]
) -> list[str]:
    """Everything wrong with one delegation, as a list so a plant can be read."""
    module_rel, reader_name, probe = entry
    problems: list[str] = []
    if module_rel not in tracked:
        return [f"{klass}: {module_rel} is not tracked, so a reader cannot open it"]
    reader = getattr(_load(root / module_rel), reader_name, None)
    if not callable(reader):
        return [f"{klass}: {module_rel} has no callable named {reader_name}"]
    if not reader(probe):
        problems.append(
            f"{klass}: {module_rel}::{reader_name} does not catch its own probe {probe!r}, "
            "so this module is printing a hand-off the other side does not perform"
        )
    if reader("an ordinary sentence that names nothing at all"):
        problems.append(
            f"{klass}: {module_rel}::{reader_name} answers YES to ordinary prose, so "
            "catching the probe demonstrated nothing"
        )
    if [token for token, _line in C.bare_tokens_in(probe)] != [probe]:
        problems.append(
            f"{klass}: the probe {probe!r} is not in the tier this module excludes, so it "
            "cannot show that the delegated class is covered"
        )
    return problems


def test_every_delegation_names_a_reader_that_actually_performs_it() -> None:
    """The check. Each printed hand-off is executed against the check it names."""
    assert C.DELEGATED, "an empty delegation table makes the plants below vacuous"
    tracked = set(C.tracked_files(repo_root()))
    problems = [
        problem
        for klass, entry in C.DELEGATED.items()
        for problem in _delegation_problems(klass, entry, root=repo_root(), tracked=tracked)
    ]
    assert not problems, "\n  ".join(["a printed delegation is not performed:", *problems])


def test_the_delegation_check_catches_a_hand_off_nobody_performs(tmp_path: Path) -> None:
    """C3.5, on a synthetic receiver, so the plants do not depend on the live table.

    Four ways a delegation can be false and one way it can be true, and the true
    one is asserted first: a checker that reported a problem on every input would
    satisfy the four plants and mean nothing.
    """
    receiver = tmp_path / "receiver.py"
    receiver.write_text(
        "def catches(text):\n"
        "    return [t for t in text.split() if t == 'probe.md']\n"
        "def catches_nothing(text):\n"
        "    return []\n"
        "def catches_everything(text):\n"
        "    return ['yes']\n"
    )
    rel = "receiver.py"
    tracked = {rel}
    kwargs = {"root": tmp_path, "tracked": tracked}

    assert _delegation_problems("control", (rel, "catches", "probe.md"), **kwargs) == []

    blind = _delegation_problems("plant", (rel, "catches_nothing", "probe.md"), **kwargs)
    assert any("does not catch its own probe" in p for p in blind), blind

    loud = _delegation_problems("plant", (rel, "catches_everything", "probe.md"), **kwargs)
    assert any("answers YES to ordinary prose" in p for p in loud), loud

    missing = _delegation_problems("plant", (rel, "no_such_reader", "probe.md"), **kwargs)
    assert any("has no callable named" in p for p in missing), missing

    untracked = _delegation_problems("plant", ("gone" + ".py", "catches", "probe.md"), **kwargs)
    assert any("is not tracked" in p for p in untracked), untracked

    outside = _delegation_problems("plant", (rel, "catches", "not a token"), **kwargs)
    assert any("is not in the tier" in p for p in outside), outside


def test_the_bare_tier_is_counted_and_is_not_silently_empty() -> None:
    """The excluded tier is a MEASUREMENT on every run, not a sentence about itself.

    A caveat that quantifies itself in prose rots exactly like any other
    enumeration; this one is computed by the same walk that produces the verdict,
    so it cannot disagree with the tree it was measured on.
    """
    enum = _enum()
    assert enum.bare_tier > 100, f"the bare-filename tier reads {enum.bare_tier}"
    assert f"BARE FILENAME TIER: {enum.bare_tier} " in C.report(enum)
    for klass, (module_rel, reader, _probe) in C.DELEGATED.items():
        assert f"{module_rel} -> {reader}()" in C.report(enum), klass
    assert list(C.bare_tokens_in("see manifest.json and np.log here")) == [("manifest.json", 1)]
    assert list(C.bare_tokens_in(f"see {_SPECIMEN_BARE_IN_A_DIRECTORY}")) == []


def test_every_declared_category_states_a_reason() -> None:
    assert C.DECLARED, "an empty declaration table makes the category tests vacuous"
    for category, (reason, pairs) in C.DECLARED.items():
        assert len(reason.split()) > 15, f"{category}: a one-line reason is not a reason"
        assert pairs, f"{category}: a category with no members is stale"


# --------------------------------------------------------------------------
# I30: the message names the LINE and the REMEDY
#
# Twice on 2026-08-25 a lead cited a module that was written but not yet
# tracked - once in `eval/`, once in `sim/` - and both times the failure named
# only the CITING FILE. Diagnosing which token, on which line, and that the fix
# was one `git add`, took a `git diff | grep` on each occasion. A checker that
# identifies a problem it could have LOCATED is spending someone else's minutes
# to save its own.
# --------------------------------------------------------------------------


def test_the_failure_NAMES_THE_LINE_and_says_git_add_when_the_file_is_merely_untracked(
    tmp_path: Path,
) -> None:
    """The exact live shape: a PACKAGE-RELATIVE citation of an untracked module.

    The token is cited as `pkg/<module>` while the file sits at
    `src/pkg/<module>`, which is how every citation in this repository is
    actually written. An exact-path check reports nothing for that case - it was
    tried first and printed the generic sentence on the very failure it was built
    for - so the remedy is matched by SUFFIX, the same way `_suffix_index`
    resolves tracked tokens.
    """
    token = _invented("pkg/latecomer", ".py")
    root = _throwaway_repo(tmp_path, f'"""Builds ``{token}``."""\n')
    real = root / "src" / token
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("# written, not yet added\n")

    problems = "\n".join(C.enumerate_references(root, declared={}, debt={}).problems)

    assert "module.py:1" in problems, (
        f"the citing LINE is not named, so the reader still has to grep for it:\n{problems}"
    )
    assert f"cites `{token}`" in problems, (
        f"the offending TOKEN is not named, only the file that carries it:\n{problems}"
    )
    assert f"`git add src/{token}`" in problems, (
        "the remedy does not name the one command that fixes it, and it is the command "
        f"that fixed this exact failure twice in one day:\n{problems}"
    )


def test_a_citation_of_something_that_does_NOT_exist_keeps_the_general_advice(
    tmp_path: Path,
) -> None:
    """The control. `git add` is only correct when there is something to add.

    Without this half the previous test would pass just as well against a message
    that says `git add` unconditionally - which would be wrong advice on every
    genuine typo and every reference to a thing that was never written.
    """
    token = _invented("pkg/never_written", ".py")
    root = _throwaway_repo(tmp_path, f'"""Builds ``{token}``."""\n')

    problems = "\n".join(C.enumerate_references(root, declared={}, debt={}).problems)

    assert f"cites `{token}`" in problems, problems
    assert "git add" not in problems, (
        "the message offers `git add` for a path that does not exist anywhere. The remedy "
        f"must distinguish UNTRACKED from ABSENT:\n{problems}"
    )
    assert "name it inline, cite something tracked" in problems, problems


def test_the_remedy_reads_the_disk_but_the_VERDICT_still_comes_from_the_index(
    tmp_path: Path,
) -> None:
    """The invariant this module is built on, re-asserted where it was put at risk.

    Resolution is decided by `git ls-files` and never by what is on this disk;
    otherwise the check passes for the author and fails for the cloner, which is
    the whole defect it exists to catch. I30 reads the filesystem to choose a
    remedy SENTENCE, so the risk is that an untracked file starts counting as
    resolved. It does not: the token below is present on disk and still
    unresolved.
    """
    token = _invented("pkg/present_but_unadded", ".py")
    root = _throwaway_repo(tmp_path, f'"""Builds ``{token}``."""\n')
    real = root / "src" / token
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("# on disk, absent from the index\n")

    enum = C.enumerate_references(root, declared={}, debt={})
    assert [r.resolution for r in enum.references] == ["unresolved"], (
        "a file that exists on disk but not in the index was treated as resolved. The "
        "remedy lookup has leaked into the verdict and the check now passes for whoever "
        "has the file and fails for everyone who clones."
    )
    assert enum.problems, "it resolved to nothing and reported no problem either"
