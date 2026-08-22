"""The code-fingerprint boundary either side of the tree-wide format sweep.

**A PUBLISHED NUMBER WHOSE FINGERPRINT RESOLVES TO NOTHING IS A NUMBER NOBODY
CAN CHECK.** Every S1 and E1 artifact on disk stamps
``scoring_code_fingerprint`` at both ends, and every one of them stamps
``85ecff60dff714b8``. The sweep moved that value. Left in a commit message, the
old hash becomes an orphan the moment the message scrolls out of anyone's view;
pinned here it stays resolvable, on the ``FINGERPRINT_PRE_D6`` precedent in
``tests/test_splits.py`` - where the superseded split fingerprint is kept as a
named constant precisely so the archive boundary is readable in the test rather
than lost in a diff.

**WHAT THE SWEEP WAS, AND WHY THE TRANSITION IS SAFE TO CROSS.** One command,
``ruff format src tests tools``, no hand edit anywhere in the 151 files it
touched. 88 files were reformatted. The claim that this is a REFORMAT and not a
REWRITE is not taken on the tool's word and cannot be taken from a diffstat - a
formatter that moved a call out of a loop produces the same line counts. It was
verified structurally, per file, by ``ast.dump(..., include_attributes=False)``:
**151 files checked, 88 bytes moved, 0 ASTs moved.** That check is reproduced
here as :func:`test_the_transition_is_formatting_only`, so the claim is
re-derivable from the repository rather than believed.

**THE ONE THING THE FORMATTER DID THAT WAS NOT WHITESPACE**, recorded because it
is the reason a separate commit exists at all: two docstrings in ``tests/``
opened on four quote characters, so their CONTENT began with ``"``, and ruff
inserts a space there. That changes a string's VALUE. Both were normalised by
hand in the commit BEFORE the sweep - visible in a two-line diff instead of
buried in an 88-file bulk reformat - after which the sweep measured 0 of 151.

**AND THE RESOLUTION LIMIT OF THE OTHER HALF OF THE EVIDENCE, WHICH BELONGS
BESIDE THE NUMBER PERMANENTLY.** The sweep was also checked numerically: one S1
fold row (fold 1 / arm A, 575 windows) re-scored through the swept code and
compared to its archived artifact. Two levels were built, and their sensitivity
is NOT the same:

* the raw 575 window rows, compared as bytes of JSON - a planted **1 ULP** in
  one row is detected;
* the pooled 14-block ``stage_decay`` over all 5346 rows, compared by ``repr``
  - a planted **1 ULP PASSED**. Measured ladder rather than assumed: a
  single-row relative nudge of 1e-16 and 1e-12 moves 0 of 14 blocks, 1e-9 moves
  1, and nudging ALL 575 rows by 1e-16 still moves nothing.

So the pooled level's resolution is roughly **1e-9 relative on one row of
5346**, and the raw-rows level is the one carrying the words "bit-identical".
An aggregate that cannot resolve a perturbation reports AGREEMENT, not
BLINDNESS, which is this project's most repeated instrument failure (C6.1's
blind ``dispersion_ratio``; M11's plateaued Brier). It is written down here
because an instrument quoted without its resolution is the thing we keep
finding, and this one was found against our own check.

**A BLIND SPOT IN THE FINGERPRINTS THEMSELVES, FOUND BY PLANTING AGAINST THIS
FILE RATHER THAN BY WATCHING IT GO GREEN.** Substituting the WRONG commit below
-- ``481722d``, the transition commit's grandparent, instead of the real one --
**passes both fingerprint assertions.** Both trees hash to ``85ecff60dff714b8``
and ``0ebd997542cb02af`` exactly, because they differ only in ``tests/``, and
**neither fingerprint covers** ``tests/``. The plant is caught by
:func:`test_the_transition_is_formatting_only` alone, on the two docstrings --
i.e. by the AST check and by nothing else.

So: **the instrument that binds every published number to its code cannot
distinguish those two commits.** That is arguably correct by design -- tests
produce no numbers, and the declared scope of ``SCORING_SUBTREES`` and
``COMMON_SUBTREES`` is the code a number is computed THROUGH. But **the
difference between "correct by design" and "nobody checked" is only whether it
is written down**, so it is written down here, where the next person reading
these pins will meet it. As a rule: **a code fingerprint answers "which covered
code", and "which commit" is a strictly larger question.** If you need the
second, use :data:`PRE_FORMAT_SWEEP_COMMIT` and
:data:`POST_FORMAT_SWEEP_COMMIT`, which are shas and answer it exactly.

**BOTH NEW VALUES WERE PREDICTED BEFORE THEY WERE OBSERVED.** The sweep was
first run in a throwaway copy of the tree, and ``bacc6ae6e2f87d13`` and
``d7825a66c3ddb47a`` were read off that copy and written down BEFORE anything
was committed here -- with that sandbox mechanism itself controlled first, by
requiring it to reproduce the two KNOWN pre-sweep values. The live tree then
produced both predicted values. **A number predicted before the observation is
evidence; the same number read off afterwards is only a record**, and this is
the evidence that the sweep did what its commit message claims.

**WHY THESE TESTS DO NOT PIN THE LIVE TREE.** An obvious design - assert that
``scoring_code_fingerprint()`` equals the of-record value - was rejected
deliberately. That fingerprint covers ``eval/`` and ``model/``, two other leads'
working directories, so such a test would go red on their every legitimate edit
and force them to change a file in ``tests/``, which they do not own. A guard
that makes a correct change look like a failure gets deleted, and it would tax
the wrong people. Everything below is pinned to a FIXED COMMIT instead, so it
answers "what was the code" permanently and stays quiet about what the code
becomes next.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from wildfire_nowcast.common.codefingerprint import (
    COMMON_SUBTREES,
    DIGEST_CHARS,
    SCORING_SUBTREES,
    code_fingerprint,
)
from wildfire_nowcast.common.paths import repo_root

# --------------------------------------------------------------------------
# THE PINS.
# --------------------------------------------------------------------------

#: The commit whose tree hashes to the OF_RECORD pair: the sweep itself, whose
#: entire content is ``ruff format src tests tools`` applied to
#: :data:`PRE_FORMAT_SWEEP_COMMIT`. Verified byte-for-byte at the time: extract
#: the parent, run the one command, and the result equals this commit's tree.
POST_FORMAT_SWEEP_COMMIT = "25ad15697ac5d5e977070b5fcf3eaf8170202c03"

#: The commit whose tree hashes to the PRE pair below: the two-character
#: docstring normalisation, i.e. the last commit before the sweep. Pinned as a
#: FULL sha because an abbreviation is not a name - it is a prefix that a
#: growing repository can make ambiguous.
PRE_FORMAT_SWEEP_COMMIT = "07997b28f4b60ea4985b710e973de013c4dcb23f"

#: ``eval/`` + ``model/`` - the code a reported number is computed IN.
#: **BOUND TO THIS VALUE:** every artifact of S1 (``runs/s1.json`` and all 15
#: matrix cells' ``s1_rows_*.json``) and of E1 (``runs/e1_rows_*.json``), at
#: BOTH ends, per C-4.2. Also every run record written between ADR-057 and the
#: sweep. If you are holding one of those and it stamps this value, it was
#: produced by :data:`PRE_FORMAT_SWEEP_COMMIT`'s tree.
SCORING_FINGERPRINT_PRE_FORMAT_SWEEP = "85ecff60dff714b8"

#: The same subtrees after the sweep. Identical code, different bytes.
SCORING_FINGERPRINT_OF_RECORD = "bacc6ae6e2f87d13"

#: ``common/`` - C-4's frozen shared surface. Moves too: 10 of its 28 files were
#: non-conformant. Recorded because the S1 artifacts stamp this one as well, in
#: ``common_code_before`` / ``common_code_after``.
COMMON_FINGERPRINT_PRE_FORMAT_SWEEP = "0ebd997542cb02af"

#: The same, after the sweep.
COMMON_FINGERPRINT_OF_RECORD = "d7825a66c3ddb47a"

_PINS = {
    "SCORING_FINGERPRINT_PRE_FORMAT_SWEEP": SCORING_FINGERPRINT_PRE_FORMAT_SWEEP,
    "SCORING_FINGERPRINT_OF_RECORD": SCORING_FINGERPRINT_OF_RECORD,
    "COMMON_FINGERPRINT_PRE_FORMAT_SWEEP": COMMON_FINGERPRINT_PRE_FORMAT_SWEEP,
    "COMMON_FINGERPRINT_OF_RECORD": COMMON_FINGERPRINT_OF_RECORD,
}

#: The scope of the sweep, as it was actually invoked.
SWEPT_DIRS = ("src", "tests", "tools")


# --------------------------------------------------------------------------
# Helpers. Each one refuses rather than skips when it cannot see its subject:
# a check that cannot run is not a check that passed (ADR-069 (4)).
# --------------------------------------------------------------------------


def _git(*args: str, cwd: Path | None = None) -> str:
    root = repo_root() if cwd is None else cwd
    if not (repo_root() / ".git").exists():
        pytest.fail(
            "no .git in this checkout, so the pinned commit cannot be resolved and the "
            "fingerprint boundary cannot be verified. That is an UNVERIFIABLE tree, which "
            "is not a green one — failing rather than skipping, on test_hygiene.py's "
            "precedent (ADR-069 (4))."
        )
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        pytest.fail(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def _materialise(commit: str, destination: Path) -> Path:
    """Extract one pinned commit's ``src``/``tests``/``tools`` into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", commit, *SWEPT_DIRS],
        cwd=repo_root(),
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        pytest.fail(
            f"cannot extract {commit}: "
            f"{archive.stderr.decode(errors='replace').strip()}. The pinned commit is the "
            "only thing that makes these fingerprints resolvable; if it is unreachable, "
            "say so loudly."
        )
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive.stdout, check=True)
    package = destination / "src" / "wildfire_nowcast"
    if not package.is_dir():
        pytest.fail(f"extracted {commit} but {package} is absent")
    return package


def _ruff() -> Path:
    ruff = repo_root() / ".venv" / "bin" / "ruff"
    if not ruff.is_file():
        pytest.fail(f"{ruff} is missing; run `make install` (ADR-001, C-4.3)")
    return ruff


def _python_files(root: Path) -> list[Path]:
    return sorted(
        p
        for directory in SWEPT_DIRS
        for p in (root / directory).rglob("*.py")
        if "__pycache__" not in p.parts and "egg-info" not in str(p)
    )


# --------------------------------------------------------------------------
# The tests.
# --------------------------------------------------------------------------


def test_the_four_pins_are_well_formed_and_none_of_them_collide() -> None:
    """Cheap, and it catches the copy-paste that makes a boundary meaningless.

    A pair whose two halves are equal records no transition at all, and two
    pairs that share a value record the wrong one. Both are one keystroke away
    and neither would fail any other test in this file.
    """
    for name, value in _PINS.items():
        assert len(value) == DIGEST_CHARS, f"{name} is not {DIGEST_CHARS} hex chars"
        assert set(value) <= set("0123456789abcdef"), f"{name} is not lowercase hex"
    assert SCORING_FINGERPRINT_PRE_FORMAT_SWEEP != SCORING_FINGERPRINT_OF_RECORD
    assert COMMON_FINGERPRINT_PRE_FORMAT_SWEEP != COMMON_FINGERPRINT_OF_RECORD
    assert len(set(_PINS.values())) == 4, "two pins share a value; one of them is wrong"
    for name, sha in (
        ("PRE_FORMAT_SWEEP_COMMIT", PRE_FORMAT_SWEEP_COMMIT),
        ("POST_FORMAT_SWEEP_COMMIT", POST_FORMAT_SWEEP_COMMIT),
    ):
        assert len(sha) == 40, f"{name}: pin the full sha; a prefix can go ambiguous"
    assert PRE_FORMAT_SWEEP_COMMIT != POST_FORMAT_SWEEP_COMMIT


def test_the_pre_sweep_commit_reproduces_both_pre_sweep_fingerprints() -> None:
    """The pin that makes every archived S1 and E1 artifact RESOLVABLE.

    Reads a FIXED commit, never the working tree, so it answers "which code
    produced ``85ecff60dff714b8``" permanently and does not go stale when
    anyone edits ``model/`` tomorrow.
    """
    with tempfile.TemporaryDirectory() as tmp:
        package = _materialise(PRE_FORMAT_SWEEP_COMMIT, Path(tmp))
        scoring = code_fingerprint(SCORING_SUBTREES, status="pin", root=package)
        common = code_fingerprint(COMMON_SUBTREES, status="pin", root=package)
    assert scoring["fingerprint"] == SCORING_FINGERPRINT_PRE_FORMAT_SWEEP, (
        f"{PRE_FORMAT_SWEEP_COMMIT[:7]} hashes to {scoring['fingerprint']}, not the "
        f"{SCORING_FINGERPRINT_PRE_FORMAT_SWEEP} that every S1 and E1 artifact stamps. "
        "Either the pinned commit is wrong or the artifacts are unresolvable; both are "
        "worth stopping for."
    )
    assert common["fingerprint"] == COMMON_FINGERPRINT_PRE_FORMAT_SWEEP
    assert len(scoring["per_file"]) == 30
    assert len(common["per_file"]) == 28


def test_the_sweep_commit_reproduces_both_of_record_fingerprints() -> None:
    """The other half of the boundary: OF_RECORD resolves to a SHA, not only to a re-run.

    :func:`test_the_transition_is_formatting_only` derives the of-record values
    by re-running the formatter, which proves the RELATIONSHIP. This proves the
    IDENTITY: there is a specific commit whose tree hashes to them, so a reader
    holding a future artifact stamped ``bacc6ae6e2f87d13`` can name the code
    without running anything.

    **This is also the test that the blind spot in the module docstring applies
    to.** It would pass just as happily on any later commit that changes only
    ``tests/``, because neither fingerprint covers ``tests/``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        package = _materialise(POST_FORMAT_SWEEP_COMMIT, Path(tmp))
        scoring = code_fingerprint(SCORING_SUBTREES, status="pin", root=package)
        common = code_fingerprint(COMMON_SUBTREES, status="pin", root=package)
    assert scoring["fingerprint"] == SCORING_FINGERPRINT_OF_RECORD, (
        f"{POST_FORMAT_SWEEP_COMMIT[:7]} hashes to {scoring['fingerprint']}, not the "
        f"{SCORING_FINGERPRINT_OF_RECORD} recorded as of record."
    )
    assert common["fingerprint"] == COMMON_FINGERPRINT_OF_RECORD
    assert len(scoring["per_file"]) == 30
    assert len(common["per_file"]) == 28


def test_the_transition_is_formatting_only() -> None:
    """**THE LOAD-BEARING TEST. Reformatted, not rewritten - re-derived, not believed.**

    Takes the pre-sweep commit's tree, applies the one command the sweep applied,
    and asserts two things about the result:

    1. every one of the 151 files is AST-IDENTICAL to its pre-sweep self, so no
       structure and no literal moved. This is the check a diffstat cannot do;
    2. the reformatted tree hashes to the OF_RECORD pair, which is what licenses
       reading an artifact stamped with a PRE value against today's code.

    Stable by construction: it depends on a fixed commit and on the ruff pinned
    by ``test_format_hook.py`` to agree across the hook, the lock file and
    ``.venv``. It says nothing about the working tree, so it will not fire on
    another lead's legitimate edit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        before = Path(tmp) / "before"
        after = Path(tmp) / "after"
        _materialise(PRE_FORMAT_SWEEP_COMMIT, before)
        shutil.copytree(before, after)
        shutil.copy2(repo_root() / "pyproject.toml", after / "pyproject.toml")
        formatted = subprocess.run(
            [str(_ruff()), "format", *SWEPT_DIRS],
            cwd=after,
            capture_output=True,
            text=True,
            check=False,
        )
        assert formatted.returncode == 0, formatted.stderr

        old_files = _python_files(before)
        assert len(old_files) == 151, (
            f"expected 151 files in the sweep's scope, saw {len(old_files)}"
        )
        moved_bytes, moved_ast = 0, []
        for old in old_files:
            new = after / old.relative_to(before)
            assert new.is_file(), f"{old} vanished under the formatter"
            if old.read_bytes() != new.read_bytes():
                moved_bytes += 1
            dump_old = ast.dump(ast.parse(old.read_text()), include_attributes=False)
            dump_new = ast.dump(ast.parse(new.read_text()), include_attributes=False)
            if dump_old != dump_new:
                moved_ast.append(str(old.relative_to(before)))

        assert moved_ast == [], (
            f"the format sweep is NOT structure-preserving on {moved_ast}. A file whose AST "
            "moves is a rewrite, not a reformat, and the numerical record either side of the "
            "transition cannot be read as the same program."
        )
        assert moved_bytes == 88, (
            f"{moved_bytes} files reformatted, expected 88. The pinned ruff has changed its "
            "output; the boundary below was measured against a different formatter."
        )

        package = after / "src" / "wildfire_nowcast"
        assert (
            code_fingerprint(SCORING_SUBTREES, status="pin", root=package)["fingerprint"]
            == SCORING_FINGERPRINT_OF_RECORD
        )
        assert (
            code_fingerprint(COMMON_SUBTREES, status="pin", root=package)["fingerprint"]
            == COMMON_FINGERPRINT_OF_RECORD
        )


def test_the_ast_check_can_actually_fail() -> None:
    """A check that cannot fail is not a check (C3.5). Plant one; it must be caught.

    The plant is a comparison operator inside a file the sweep DOES reformat,
    which is the exact scenario the AST check exists for: a logic change hiding
    under a legitimate reformat, invisible to a diffstat and to a byte count.
    A pure-whitespace change is planted alongside and must NOT be reported, or
    the check is only saying "bytes moved" in an expensive costume.
    """
    with tempfile.TemporaryDirectory() as tmp:
        before = Path(tmp) / "before"
        after = Path(tmp) / "after"
        _materialise(PRE_FORMAT_SWEEP_COMMIT, before)
        shutil.copytree(before, after)

        logic = after / "src/wildfire_nowcast/common/dispersion.py"
        text = logic.read_text()
        target = "    inside = d <= bar\n"
        assert target in text, (
            "the plant site moved, so this control would have planted NOTHING and reported "
            "green. A plant without an assertion that it landed is a control that reports "
            "on an empty intervention."
        )
        logic.write_text(text.replace(target, "    inside = d < bar\n", 1))

        whitespace = after / "src/wildfire_nowcast/common/grid.py"
        blank = whitespace.read_text()
        assert "\n\n\n" in blank, "the whitespace control's site moved"
        whitespace.write_text(blank.replace("\n\n\n", "\n\n\n\n", 1))

        def ast_moved(rel: str) -> bool:
            old = ast.dump(ast.parse((before / rel).read_text()), include_attributes=False)
            new = ast.dump(ast.parse((after / rel).read_text()), include_attributes=False)
            return old != new

        assert ast_moved("src/wildfire_nowcast/common/dispersion.py"), (
            "the AST check did not see the G3 dispersion bar's comparison flip. It is "
            "therefore unable to see the only thing it was built to see."
        )
        assert not ast_moved("src/wildfire_nowcast/common/grid.py"), (
            "the AST check reported a pure-whitespace edit, so it is not distinguishing "
            "structure from layout and its 0-of-151 result means nothing."
        )
