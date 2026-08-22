"""C7 and C0 hygiene rules that were true only by discipline until A10.

Both clauses were ratified and satisfied on disk, and neither was ENFORCED -
the same shape as C1.5 (ratified v2.3, implemented v2.5, all-NaN passing 56
checks in between). They pass today, which is exactly when to wire them: a
hygiene rule is cheap to keep and expensive to restore.

* C7 - *no hardcoded paths in src/*, *no hardcoded GCP project id*, and the
  house rule that *notebooks are never imported by src*.
* C0 - anything the contract adjudicates has ONE implementation, in ``common/``.
  The retired ``data/`` duplicates must stay deleted, and the second copy of the
  split fingerprint must agree with the first.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
import yaml

import commit_guard  # tools/commit_guard.py, via `pythonpath` in pyproject.toml
from wildfire_nowcast.common.paths import repo_root

SRC = repo_root() / "src" / "wildfire_nowcast"


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_there_is_source_to_check() -> None:
    """A scan over an empty file list passes vacuously - check the scan first."""
    assert len(_py_files()) > 20


# --------------------------------------------------------------------------
# C7
# --------------------------------------------------------------------------

#: Absolute-path literals. ``/tmp`` is excluded: it is a documented scratch
#: location, not a repo-specific path, and banning it would push people to
#: hardcode something worse.
_ABS_PATH_RE = re.compile(r"""['"](?:/Users/|/home/|/Volumes/|[A-Za-z]:\\\\)""")


def test_no_absolute_path_literals_in_src() -> None:
    """C7: *No hardcoded paths in src/*. Everything resolves via ``common.paths``."""
    offenders: list[str] = []
    for path in _py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _ABS_PATH_RE.search(line):
                offenders.append(f"{path.relative_to(repo_root())}:{lineno}: {line.strip()[:100]}")
    assert not offenders, (
        "C7 forbids hardcoded paths in src/. Use wildfire_nowcast.common.paths (every default "
        "is overridable by an env var, which is what lets tests and other leads redirect "
        "output):\n" + "\n".join(offenders)
    )


def test_no_hardcoded_gcp_project_id_in_src() -> None:
    """C7 [v2]: the project id is read from ``$WILDFIRE_GEE_PROJECT`` (ADR-003).

    Checked structurally rather than by matching the id itself - writing the
    real project id into a test file is the very thing the clause forbids.
    """
    offenders: list[str] = []
    for path in _py_files():
        source = path.read_text()
        for match in re.finditer(r"""(?<![\w.])project\s*=\s*["']([^"']+)["']""", source):
            line = source[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(repo_root())}:{line}: project={match.group(1)!r}")
    assert not offenders, (
        "C7 [v2]: no hardcoded GCP project id. Read $WILDFIRE_GEE_PROJECT "
        "(wildfire_nowcast.data.sources.gee.ENV_PROJECT):\n" + "\n".join(offenders)
    )

    gee = SRC / "data" / "sources" / "gee.py"
    if gee.is_file():
        assert "WILDFIRE_GEE_PROJECT" in gee.read_text()


def test_src_never_imports_a_notebook() -> None:
    """House rule: *notebooks are never imported by src.*

    A package that imports a notebook is not installable, and a notebook that is
    a dependency is a notebook nobody may reorganise.
    """
    offenders: list[str] = []
    for path in _py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                head = name.split(".")[0]
                if head in {"nbformat", "nbconvert", "papermill", "notebooks", "import_ipynb"}:
                    offenders.append(f"{path.relative_to(repo_root())}: imports {name}")
        if ".ipynb" in path.read_text():
            offenders.append(f"{path.relative_to(repo_root())}: references a .ipynb path")
    assert not offenders, offenders


def test_every_run_dir_records_config_git_sha_and_split(tmp_path: Path) -> None:
    """C7 + C8 in one artifact: resolved config, git SHA, dirty flag, fingerprint."""
    from wildfire_nowcast.common.config import load_yaml
    from wildfire_nowcast.common.runs import create_run_dir

    run = create_run_dir({"experiment": "x", "lr": 0.1}, run_id="r", runs_root=tmp_path)
    payload = load_yaml(run.config_path)
    meta = payload["_run"]
    assert payload["lr"] == 0.1
    for key in ("git_sha", "git_dirty", "created_utc", "split_fingerprint"):
        assert key in meta, key


def test_git_sha_short_is_not_the_string_unknown() -> None:
    """Regression: ``git rev-parse --short`` without a ref returns nothing, so
    every run in ``runs/`` recorded ``git_sha_short: "unknown"`` beside a
    perfectly good full SHA. A provenance field that degrades to a placeholder
    looks recorded and is not."""
    from wildfire_nowcast.common.runs import git_sha

    full, short = git_sha(), git_sha(short=True)
    if full is None:
        pytest.skip("not a git work tree")
    assert short and short != "unknown"
    assert full.startswith(short)


# --------------------------------------------------------------------------
# C0
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["grid.py", "states.py", "norm_stats.py", "moisture.py"])
def test_retired_data_duplicates_stay_deleted(name: str) -> None:
    """C0: ``data/`` imports ``common/``; it does not re-implement it.

    These four modules were deleted at A5 after the byte-identical check (R9).
    The failure mode C0 exists to prevent is the producer and the verifier
    computing geometry through different code - a tensor that passes its check
    and is still wrong.
    """
    duplicate = SRC / "data" / name
    assert not duplicate.exists(), (
        f"{duplicate} is back. C0: the single implementation lives in common/ and data/ imports it."
    )


def test_data_assemble_delegates_to_the_common_manifest_builder() -> None:
    from wildfire_nowcast.data import assemble

    assert assemble._build_manifest.__module__ == "wildfire_nowcast.common.zarr_io"


def test_the_state_rule_has_exactly_one_implementation() -> None:
    from wildfire_nowcast.common import states

    hits = [
        f"{p.relative_to(repo_root())}"
        for p in _py_files()
        if p.name not in {"states.py", "contract.py"} and "def apply_state_rule" in p.read_text()
    ]
    assert not hits, f"C1.1's rule is re-implemented outside common/states.py: {hits}"
    assert hasattr(states, "apply_state_rule")


# --------------------------------------------------------------------------
# [A14] ONE source of truth for the contract version (ADR-033 (1), ADR-036 (5))
# --------------------------------------------------------------------------

#: A MODULE DOCSTRING claiming a contract version, e.g.
#: ``"""Executable form of INTERFACES.md contracts ... - **v2.5**."""``
#:
#: Deliberately narrow. In-body citations like ``[v2.7] ADD fuel_vintage_lag_years``
#: or ``C8 (INTERFACES v2.8)`` are CORRECT and must survive: they say when a clause
#: was ratified, which does not change when the contract version moves. What must
#: not survive is a module announcing which version it implements, because that is
#: a claim about the present that nothing updates. Both observed instances
#: (`common/contract.py`, `common/zarr_io.py`) were of exactly this shape and both
#: sat SEVEN versions stale in the files that adjudicate and write the artifacts.
#: Scoped to the SUMMARY LINE of a module docstring, and excluding the bracketed
#: ``[v2.7]`` citation form. That narrowness is deliberate and was arrived at by
#: running the wide version first: it flagged `eval/reporting.py` ("enforced at
#: v2.3"), `sim/components.py` ("[v2.7]") and `data/ignitions.py` ("C2 [v2.7]"),
#: all of which are CORRECT - they record when a clause was ratified, which does
#: not change when the contract moves. A hygiene rule with false positives on
#: three other leads' files gets disabled within a day (the C1.6 lesson), and it
#: would have had me "fixing" correct docstrings across two ownership boundaries.
_VERSION_CLAIM_RE = re.compile(r"(?<!\[)\bv\d+\.\d+\b")


def test_no_module_docstring_SUMMARY_LINE_claims_a_contract_version() -> None:
    """The version has ONE home: INTERFACES.md line 1, read by ``CONTRACT_VERSION``.

    This is the mechanism half of ADR-033 (1). Deriving the constant removes the
    duplicate that FORCED a lead to edit the maintainer's file; this test stops a
    new duplicate being written into a docstring, which is how the last two got in.
    Both known instances were the module SUMMARY LINE announcing which version the
    module implements - a claim about the present that nothing ever updates:

        ``Executable form of INTERFACES.md contracts C1, C2 and C3 - **v2.5**.``
        ``Reading and writing C1/C2/C3 artifacts - **INTERFACES v2.5**.``
    """
    offenders: list[str] = []
    for path in _py_files():
        try:
            doc = ast.get_docstring(ast.parse(path.read_text()))
        except SyntaxError:  # pragma: no cover - a file we cannot parse
            continue
        if not doc:
            continue
        summary = doc.splitlines()[0]
        if _VERSION_CLAIM_RE.search(summary):
            offenders.append(f"{path.relative_to(repo_root())}: {summary.strip()[:90]}")
    assert not offenders, (
        "these module summary lines state a contract version. There is exactly one home for it "
        "— INTERFACES.md line 1, parsed by contract.CONTRACT_VERSION. A second copy is a second "
        "thing to forget, and both known instances sat seven versions stale:\n"
        + "\n".join(offenders)
    )


def test_the_docstring_scan_can_actually_fire() -> None:
    """The scan's own positive control - a pattern that matches nothing passes.

    Four of this project's confident false negatives came from a query that
    silently matched nothing (a relative ``find -newermt``, a future timestamp, an
    unquoted glob, a wrong norm-stats key). The narrowing above is exactly the
    kind of edit that could quietly empty this check, so the pattern is exercised
    against both known offenders and against the citations it must NOT flag.
    """
    assert _VERSION_CLAIM_RE.search("Executable form of INTERFACES contracts — **v2.5**.")
    assert _VERSION_CLAIM_RE.search("Reading and writing C1/C2/C3 artifacts — INTERFACES v2.5.")
    assert not _VERSION_CLAIM_RE.search("C2 [v2.7] ``n_ignition_components`` — fires in a fire id.")
    assert not _VERSION_CLAIM_RE.search("C6.4 — the best_member_iou decomposition (ADR-017).")


def test_the_contract_version_is_DERIVED_and_has_NO_literal_fallback() -> None:
    """PLANTED DEFECT: INTERFACES.md line 1 becomes unreadable.

    The requirement is not merely "it currently agrees" - it is that a failure to
    read the file **raises** rather than falling back. A fallback is how drift
    hides: the constant would keep its last value and the checker would print a
    version it had not read, which is the stale-checker hazard wearing a current
    label.
    """
    from wildfire_nowcast.common import contract as C

    assert C.CONTRACT_VERSION == C._read_contract_version()

    real = C.repo_root
    try:
        C.repo_root = lambda: Path("/nonexistent-repo-root-for-this-test")
        with pytest.raises(RuntimeError, match="cannot read the contract version"):
            C._read_contract_version()
    finally:
        C.repo_root = real

    # ...and an unparseable first line is equally loud.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp)
        (fake / "docs").mkdir()
        (fake / "docs" / "interfaces.md").write_text("# a header with no version\n")
        try:
            C.repo_root = lambda: fake
            with pytest.raises(RuntimeError, match="cannot parse a contract version"):
                C._read_contract_version()
        finally:
            C.repo_root = real

    assert C._read_contract_version() == C.CONTRACT_VERSION, "not restored"


def test_the_version_parser_reads_the_REAL_file_and_agrees_with_it() -> None:
    """Reads a VALUE out of INTERFACES.md rather than proving a mismatch is absent."""
    from wildfire_nowcast.common import contract as C

    line = (repo_root() / C.INTERFACES_RELATIVE_PATH).read_text().splitlines()[0]
    assert C.CONTRACT_VERSION in line, (C.CONTRACT_VERSION, line)
    assert C.CONTRACT_VERSION.startswith("v") and "." in C.CONTRACT_VERSION


def test_no_config_carries_its_own_interfaces_version_literal() -> None:
    """C7/C0: the resolved config is STAMPED from the contract, never asserted by yaml.

    A yaml literal was the third copy of this one fact, and because a test pinned
    it equal to the constant, an maintainer bump made the build red until
    infra edited ``configs/``. That is the mechanically-forced edit ADR-033
    ruled must be fixed at the mechanism.
    """
    from wildfire_nowcast.common.config import INTERFACES_VERSION_KEY, load_config
    from wildfire_nowcast.common.contract import CONTRACT_VERSION
    from wildfire_nowcast.common.paths import configs_dir

    offenders = [
        f"{p.name}:{n}"
        for p in sorted(configs_dir().glob("*.yaml"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.match(rf"^\s*{INTERFACES_VERSION_KEY}\s*:", line)
    ]
    assert not offenders, offenders
    # ...and the stamp still lands, so removing the literal did not remove the key.
    assert load_config("base.yaml")[INTERFACES_VERSION_KEY] == CONTRACT_VERSION


def test_a_config_declaring_a_DIFFERENT_version_is_an_error(tmp_path: Path) -> None:
    """PLANTED DEFECT: an experiment claims conformance to a version not in force."""
    from wildfire_nowcast.common.config import load_config

    (tmp_path / "stale.yaml").write_text("interfaces_version: v2.5\nseed: 0\n")
    with pytest.raises(ValueError, match="but the contract in force is"):
        load_config(tmp_path / "stale.yaml", configs_root=tmp_path)


# --------------------------------------------------------------------------
# [A16] The PUBLIC tree cites nothing a public reader cannot open
# --------------------------------------------------------------------------
#
# This repo is public. Four manual sweeps failed to finish this job and three of
# them reported clean while being wrong, so the deliverable is not a sweep - it is
# this check.
#
# WHAT IS BANNED IS THE UNRESOLVABLE REFERENCE, NOT THE CONTENT. The scientific
# reasoning in those comments is the best documentation in the repo and deleting
# it to make a grep clean would be a net loss. What must not survive is a pointer
# to something outside the repo that a reader is invited to go and read: the
# private agent-instruction file, and the internal coordination role names. State
# the constraint inline, or cite ``README.md`` / ``docs/decisions.md``, which carry
# the same content publicly.
#
# ``ADR-NNN`` citations are DELIBERATELY NOT BANNED. 66 tracked files use them,
# README's last paragraph tells the reader plainly that they will not resolve and
# why they are kept, and a convention that is disclosed is not a leak.

#: The tells, spelled in HALVES and joined at runtime.
#:
#: This module is scanned by its own check, with NO self-exemption - see
#: ``test_the_scan_reads_its_own_source_and_is_not_self_exempt``. An exemption
#: would be the exact failure this check exists to prevent: a scan narrowed until
#: it cannot see something. Splitting the literals is what buys that, and it costs
#: one line of ugliness in one file. The alternative (skip this path) makes the
#: one file most likely to acquire a tell the one file that can never report one.
_TELL_PATTERNS: dict[str, str] = {
    # The agent-instruction file. It is not in the repo, it names the tooling, and
    # "see <it>" is an instruction a reader cannot follow.
    "agent-instruction file": "CLAUD" + "E" + r"\.md",
    # The coordination role that adjudicates gates. Internal process vocabulary.
    "coordination role": "orchestr" + "ator",
    # The internal agent role names, e.g. "<area>-lead's finding".
    "agent role name": r"\b(?:infra|data|model|sim|simviz)[- ]" + "le" + "ad",
}

_TELL_RE = re.compile("|".join(f"(?:{p})" for p in _TELL_PATTERNS.values()), re.IGNORECASE)

#: BURN-DOWN LIST, NOT AN EXEMPTION. ``model/`` and ``eval/`` were fenced at A16
#: because modelling held a running experiment out of them (C-4 freezes a running
#: lead's surface). These 31 are recorded so the fence does not make the suite red,
#: and are to be RETIRED - file by file, to zero - once that work lands. The check
#: fails if any count goes UP (a new tell) and equally if any count comes DOWN
#: without the entry being removed (a burn-down list that never shrinks is an
#: excuse, and a stale entry is how one starts). Deleting an entry once its file is
#: clean is the intended edit; there is no path that keeps the list alive quietly.
FENCED_BURN_DOWN: dict[str, int] = {
    "src/wildfire_nowcast/eval/selftest.py": 2,
    "src/wildfire_nowcast/model/__init__.py": 1,
    "src/wildfire_nowcast/model/api.py": 1,
    "src/wildfire_nowcast/model/baselines/__init__.py": 1,
    "src/wildfire_nowcast/model/baselines/ellipse.py": 4,
    "src/wildfire_nowcast/model/kernel.py": 6,
    "src/wildfire_nowcast/model/labelnoise.py": 1,
    "src/wildfire_nowcast/model/latent.py": 6,
    "src/wildfire_nowcast/model/spread.py": 3,
    "src/wildfire_nowcast/model/train.py": 6,
}


def _tracked_files() -> list[str]:
    """The PUBLIC tree, which is exactly the TRACKED tree.

    A tell only matters where a stranger can read it. Working-memory directories
    are ignored per-clone rather than deleted, so a filesystem walk would scan
    files that were never published and miss the point in both directions.
    """
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(p for p in out.stdout.splitlines() if p)


def scan_tracked_tree() -> dict[str, int]:
    """``{relative path: number of tells}``, for every tracked file with at least one."""
    counts: dict[str, int] = {}
    for rel in _tracked_files():
        path = repo_root() / rel
        if not path.is_file():  # a submodule or a broken link
            continue
        # decode with replacement rather than skipping: a file this scan cannot
        # read is a hole in the scan, and holes are what the four sweeps were.
        text = path.read_bytes().decode("utf-8", errors="replace")
        n = len(_TELL_RE.findall(text))
        if n:
            counts[rel] = n
    return counts


def new_or_grown_tells(counts: Mapping[str, int]) -> list[str]:
    """Files carrying a tell that the burn-down list does not already account for.

    Pure, so the planted-defect tests can feed it a hypothetical tree instead of
    having to dirty the repo to prove the check bites.
    """
    offenders = []
    for rel, n in sorted(counts.items()):
        allowed = FENCED_BURN_DOWN.get(rel, 0)
        if n > allowed:
            offenders.append(f"{rel}: {n} tell(s), {allowed} accounted for")
    return offenders


def stale_burn_down_entries(counts: Mapping[str, int]) -> list[str]:
    """Burn-down entries whose file no longer carries that many tells."""
    stale = []
    for rel, expected in sorted(FENCED_BURN_DOWN.items()):
        actual = counts.get(rel, 0)
        if actual < expected:
            stale.append(f"{rel}: burn-down says {expected}, file now has {actual}")
    return stale


def test_the_public_tree_has_files_to_scan() -> None:
    """Corpus positive control. A scan over an empty list is a green light for nothing."""
    files = _tracked_files()
    assert len(files) > 100, f"only {len(files)} tracked files — the scan lost its corpus"
    assert "README.md" in files and "docs/interfaces.md" in files


def test_the_tell_scan_can_actually_fire() -> None:
    """PATTERN positive control - the half this project has got wrong four times.

    Three of the four failed sweeps reported clean from a scan that matched
    nothing. This asserts the compiled pattern still matches realistic sentences,
    and - the harder half - that it still does NOT match the citations and the
    ordinary English that must survive. A pattern narrowed until it is silent and a
    pattern widened until it is ignored both end with the check switched off.
    """
    tell_file = "CLAUD" + "E" + ".md"
    role = "data" + "-" + "le" + "ad"
    must_fire = [
        f"# Frozen scientific ground truth ({tell_file}); mirrored here so runs record it.",
        f"the 1-2 {tell_file} assumed (ADR-014 3). This is staleness, not leakage",
        f"WHERE THIS CLAUSE CAME FROM, AND IT IS {role.upper()}'S",
        "Classified by the " + "ORCHESTR" + "ATOR, not by the implementer",
        "Model" + "-" + "le" + "ad's package: the C5 prediction API",
    ]
    for line in must_fire:
        assert _TELL_RE.search(line), f"the scan went blind to: {line!r}"

    must_not_fire = [
        "C6.4 - the best_member_iou decomposition (ADR-017).",
        "ADR-047 (7): a fingerprinted file that is not there is never benign.",
        "denominator is the model's own mean-area error, exactly 0 there",
        "the lead author of the GOFER paper",  # 'lead' alone is ordinary English
        "modelling owns eval/; infra may make additive edits to eval/metrics.py",
        "See README.md and docs/decisions.md for the ground truth.",
    ]
    for line in must_not_fire:
        assert not _TELL_RE.search(line), f"false positive on: {line!r}"

    # SECOND control, live only while the burn-down list is non-empty: the fenced
    # population is real text in real files, so the pattern is exercised against
    # the tree and not only against fixtures. Guarded explicitly rather than left
    # to become `>= 0` on the day the list is retired - a control that quietly
    # turns vacuous is worse than one that was never written, because it still
    # reads as coverage. When the list empties, the specimens above are the
    # control, and they do not depend on the tree's state at all.
    if FENCED_BURN_DOWN:
        assert sum(scan_tracked_tree().values()) >= sum(FENCED_BURN_DOWN.values())


def test_a_tell_planted_in_a_file_is_found_by_the_REAL_reader(tmp_path: Path) -> None:
    """PLANTED DEFECT, end to end through the same reader the scan uses."""
    clean = tmp_path / "clean.py"
    clean.write_text('"""A module that cites README.md and ADR-047. Nothing to see."""\n')
    assert not _TELL_RE.findall(clean.read_bytes().decode("utf-8", errors="replace"))

    planted = tmp_path / "planted.py"
    planted.write_text('"""Ground truth is frozen; see ' + "CLAUD" + "E" + '.md."""\n')
    assert len(_TELL_RE.findall(planted.read_bytes().decode("utf-8", errors="replace"))) == 1


def test_no_unresolvable_internal_tooling_reference_in_the_public_tree() -> None:
    """The check. A public reader is never told to go and read something private."""
    offenders = new_or_grown_tells(scan_tracked_tree())
    assert not offenders, (
        "these tracked files cite internal tooling a public reader cannot open. KEEP THE "
        "CONTENT — the reasoning in those comments is worth more than a clean grep — and "
        "replace only the REFERENCE: state the constraint inline, or cite README.md / "
        "docs/decisions.md, which carry it publicly. ADR-NNN citations are fine and are "
        "deliberately not matched here:\n  " + "\n  ".join(offenders)
    )


def test_the_burn_down_list_has_not_gone_stale() -> None:
    """A burn-down list that never shrinks is an excuse, so shrinking it is enforced."""
    stale = stale_burn_down_entries(scan_tracked_tree())
    assert not stale, (
        "GOOD NEWS, and it needs one edit: these files have been cleaned but their "
        "burn-down entries still claim the old count. Lower the number, or delete the "
        "entry once the file is at zero. An entry nobody has to touch is how a temporary "
        "fence becomes a permanent exemption:\n  " + "\n  ".join(stale)
    )


def test_the_burn_down_list_only_covers_the_surface_it_was_granted() -> None:
    """The fence was ``model/`` and ``eval/``. It may not quietly grow a third arm."""
    for rel in FENCED_BURN_DOWN:
        assert rel.startswith(("src/wildfire_nowcast/model/", "src/wildfire_nowcast/eval/")), rel
        assert (repo_root() / rel).is_file(), f"{rel} is listed but does not exist"


def test_the_planted_defects_the_burn_down_check_must_catch() -> None:
    """C3.5: every clause ships with the defect it catches. Three, on a fake tree.

    Pure functions, so this runs on every machine in milliseconds and does not
    depend on the repo being dirtied and restored.
    """
    live = scan_tracked_tree()

    # 1. A NEW tell in a file that is clean today.
    grown = dict(live)
    grown["src/wildfire_nowcast/common/paths.py"] = 1
    assert any("common/paths.py" in o for o in new_or_grown_tells(grown))

    # 2. An allowlisted count going UP - the fence must not absorb new debt.
    bumped = dict(live)
    bumped["src/wildfire_nowcast/model/train.py"] = (
        FENCED_BURN_DOWN["src/wildfire_nowcast/model/train.py"] + 1
    )
    assert any("model/train.py" in o for o in new_or_grown_tells(bumped))

    # 3. An allowlisted count going DOWN without the entry being retired.
    burned = dict(live)
    burned["src/wildfire_nowcast/model/train.py"] = 1
    assert any("model/train.py" in s for s in stale_burn_down_entries(burned))
    burned.pop("src/wildfire_nowcast/eval/selftest.py", None)
    assert any("eval/selftest.py" in s for s in stale_burn_down_entries(burned))

    # ...and the live tree passes all three, which is the assertion that makes the
    # three above mean something rather than being satisfied by a broken scanner.
    assert not new_or_grown_tells(live) and not stale_burn_down_entries(live)


def test_the_scan_reads_its_own_source_and_is_not_self_exempt() -> None:
    """This file holds the tell strings as PATTERNS, and is scanned anyway.

    The patterns are spelled in halves and joined at runtime, so the source
    contains no literal tell and needs no exemption. Asserted rather than trusted,
    because the tempting fix - skipping this path - would make the one file most
    likely to acquire a tell the one file that could never report one. That is the
    same shape as the check that could not fail.
    """
    rel = "tests/test_hygiene.py"
    assert rel in _tracked_files(), "this file is not tracked, so the scan never sees it"
    assert rel not in FENCED_BURN_DOWN
    assert rel not in scan_tracked_tree(), "this file now contains a LITERAL tell"

    # the halves really are halves: no source line matches on its own.
    source = (repo_root() / rel).read_text()
    assert not _TELL_RE.findall(source)
    assert "CLAUD" in source and "orchestr" in source  # ...but the fragments are here


# --------------------------------------------------------------------------
# [I11] No TRACKED file publishes an operator's home directory
# --------------------------------------------------------------------------
#
# `test_no_absolute_path_literals_in_src` above is C7 and reads `src/*.py` only.
# That scope was exactly wide enough to miss what happened: 64 absolute paths of
# the form `<home>/<user>/Projects/wildfire-nowcast/...` were committed inside
# twelve `runs/` RESULT ARTIFACTS, publishing a local account name and a
# directory layout on a portfolio repository. Not one of them was in `src/`, so
# not one of them was ever in front of a check.
#
# The scan below therefore reads the TRACKED TREE, which is the definition of
# the public surface, and not a directory anyone chose. It is deliberately
# narrower in what it matches than C7's regex: C7 bans a hardcoded path in CODE,
# where `/tmp` and a bare `/Volumes` are still worth arguing about; this bans a
# path that NAMES A PERSON, anywhere, which is a leak rather than a style
# question. The two overlap and neither subsumes the other.
#
# Producer-side, `common.paths.repo_relative` is what an artifact writes; this is
# what refuses to publish the ones that did not use it.

#: A user home directory: the root, then a NAME, then a separator. The trailing
#: separator is load-bearing - it is what makes this pattern skip the pattern
#: DEFINITIONS in this file, which end at the root, while still catching every
#: real path. Spelled in halves for the same reason the tell patterns are.
_HOME_ROOTS = ("/U" + "sers", "/ho" + "me", "/Vol" + "umes")
_HOME_PATH_RE = re.compile(
    # `\\+` and not `\\`: a Windows path inside a Python or JSON string literal
    # is written with its backslashes ESCAPED, so the bytes on disk carry two of
    # them. A pattern that insists on one would read the source of the very leak
    # it is looking for and see nothing.
    "(?:" + "|".join(_HOME_ROOTS) + r")/[A-Za-z0-9._-]+/" + r"|[A-Za-z]:\\+U" + r"sers"
)


def scan_tracked_tree_for_home_paths() -> dict[str, int]:
    """``{relative path: number of home-directory paths}``, tracked files only."""
    counts: dict[str, int] = {}
    for rel in _tracked_files():
        path = repo_root() / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        n = len(_HOME_PATH_RE.findall(text))
        if n:
            counts[rel] = n
    return counts


def test_no_tracked_file_publishes_a_home_directory_path() -> None:
    """The whole public surface, not `src/`. This is what the 64 escaped through."""
    offenders = scan_tracked_tree_for_home_paths()
    assert not offenders, (
        "these tracked files publish an operator's home directory. Write paths into "
        "artifacts with `wildfire_nowcast.common.paths.repo_relative`, which renders a path "
        "inside the repository relative to it and leaves anything outside alone:\n"
        + "\n".join(f"  {rel}: {n}" for rel, n in sorted(offenders.items()))
    )


def test_the_home_path_scan_can_actually_fire() -> None:
    """Five shapes it must catch and three it must not, read as values.

    Every specimen is BUILT at runtime rather than written out, for the reason
    the test below asserts: this file is scanned by the check it is testing, so a
    literal specimen would make the file its own first offender.
    """
    bs = chr(92)
    must_catch = [
        '"checkpoint": "/U' + 'sers/someone/Projects/wildfire-nowcast/runs/x"',
        "path = '/ho" + "me/ci-runner/work/repo/data'",
        "/Vol" + "umes/scratch2/fires/tensor.zarr",
        # a Windows path as it looks, and as a source file has to escape it
        "C:" + bs + "U" + "sers" + bs + "someone",
        "C:" + bs * 2 + "U" + "sers" + bs * 2 + "someone",
    ]
    for specimen in must_catch:
        assert _HOME_PATH_RE.search(specimen), f"NOT CAUGHT: {specimen}"

    must_not_catch = [
        "runs/s1.json",
        "data/fires",
        # the pattern definitions in this very file: a root with no name after it
        "/U" + "sers/",
    ]
    for specimen in must_not_catch:
        assert not _HOME_PATH_RE.search(specimen), f"FALSE POSITIVE: {specimen}"


def test_the_home_path_scan_reads_this_file_and_is_not_self_exempt() -> None:
    """Same rule as the tell scan: the file holding the pattern is scanned by it."""
    rel = "tests/test_hygiene.py"
    assert rel in _tracked_files()
    assert rel not in scan_tracked_tree_for_home_paths()


# --------------------------------------------------------------------------
# [I4] No commit in this repository's history carries attribution
# --------------------------------------------------------------------------
#
# THIS IS LAYER (b). Layer (a) is the `commit-msg` hook in
# `.pre-commit-config.yaml`, which arrives with `make install`. A single layer is
# already KNOWN to be insufficient here: the maintainer used `git commit
# --no-verify` during the repair that motivated the guard, and `--no-verify`
# walks straight past a hook. This layer scans the COMMITTED history, so it sees
# exactly what the hook let through.
#
# IT IS NOT SCOPED TO A WINDOW. The audit that missed the original defect looked
# at the eight commits it expected to be dirty. Scoping a scan to where you
# expect the problem is the allow-list defect wearing a different hat, and it is
# the fifth member of that family in this project. Every commit reachable from
# every ref, or the scan does not run.
#
# The rules live in `tools/commit_guard.py` and are DERIVED: git's own trailer
# grammar, git's own identity, git's own remote hosts, and a Unicode category.
# One rule (attribution constructions in prose) could not be derived; it is
# declared as the maintained surface and its keys are pinned below, so widening
# it is a visible edit rather than a quiet one.

#: The maintained surface, pinned. Growing rule 5 means changing this line too.
EXPECTED_CONSTRUCTIONS = {
    "assistance-of",
    "co-authorship",
    "credited-to-a-link",
    "credited-to-an-agent",
    "on-behalf-of",
    "sign-off",
}


def _guard_inputs() -> tuple[Path, frozenset[str], frozenset[str]]:
    root = repo_root()
    return root, commit_guard.own_emails(root), commit_guard.own_remote_hosts(root)


def _scan(messages: Mapping[str, str]) -> dict[str, list[commit_guard.Finding]]:
    root, emails, hosts = _guard_inputs()
    return commit_guard.scan_corpus(messages, repo=root, allowed_emails=emails, allowed_hosts=hosts)


#: Shape-accurate reconstructions of what a tool actually emits, with the vendor
#: strings replaced by placeholders. That substitution is legitimate here and NOT
#: a weakened fixture, because every rule these trip is a rule about SHAPE (a
#: trailer block, an address that is not ours, a host we do not push to, a
#: pictograph). `test_the_specimens_are_caught_by_shape_and_not_by_name` proves
#: it by swapping the placeholder for a different one and getting the same
#: verdict. Keeping real product names out of the public tree is the same
#: hygiene rule the tell scan above enforces.
TRAILER_FORM = (
    "readme: correct two public over-claims\n"
    "\n"
    "a body paragraph that is entirely ordinary.\n"
    "\n"
    "Co-Authored-By: Some Assistant (1M context) <noreply@vendor.example>\n"
    "Tool-Session: https://vendor.example/code/session_00000000000000000\n"
)
BADGE_FORM = (
    "readme: correct two public over-claims\n"
    "\n"
    "a body paragraph that is entirely ordinary.\n"
    "\n"
    "\N{ROBOT FACE} Generated with [Some Tool](https://vendor.example/tool)\n"
)

#: Rule 6's form, and it carries NO attribution at all. That is the point: it is
#: rejected for the dash alone, so a hook that had quietly stopped running rule 6
#: could not be rescued by one of the other five. Spelled as an escape because
#: this file is scanned for the literal character.
DASH_FORM = (
    "readme: correct two public over-claims \N{EM DASH} the deceleration paragraph\n"
    "\n"
    "a body paragraph that is entirely ordinary.\n"
)


def test_the_history_corpus_is_whole_and_is_not_a_window() -> None:
    """CORPUS POSITIVE CONTROL. A scan over one commit is a green light for nothing.

    Three separate ways this scan could report clean while seeing almost
    nothing, all checked by reading a value rather than by proving a set empty:
    a shallow clone (CI's default checkout is depth 1), a corpus that does not
    match git's own count, and an empty message.
    """
    root = repo_root()
    assert not commit_guard.is_shallow(root), (
        "this is a SHALLOW clone, so a full-history scan cannot be performed and must not "
        "report clean. Run `git fetch --unshallow`. In CI this means `actions/checkout` lost "
        "its `fetch-depth: 0`."
    )
    messages = commit_guard.commit_messages(root)
    declared = int(
        subprocess.run(
            ["git", "rev-list", "--all", "--count"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    assert len(messages) == declared, (
        f"the scan read {len(messages)} messages but git reports {declared} commits reachable "
        "from all refs"
    )
    assert declared >= 50, f"only {declared} commits reachable: the scan lost its corpus"
    assert all(m.strip() for m in messages.values()), "a commit message came back empty"

    # every ref tip is inside the corpus, so no branch is invisible to the scan.
    tips = subprocess.run(
        ["git", "for-each-ref", "--format=%(objectname)"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    unseen = [t for t in tips if t not in messages]
    assert not unseen, f"these ref tips are not in the scanned corpus: {unseen}"


def test_every_attribution_rule_fires_on_its_own_specimen() -> None:
    """PATTERN POSITIVE CONTROL, per rule. A silent pattern is an invisible pass.

    Each of the five rules gets a specimen it must catch and the specimen names
    the rule, so a rule that goes blind fails HERE with its own name rather than
    disappearing into an aggregate zero.
    """
    root, emails, hosts = _guard_inputs()

    def rules(message: str) -> set[str]:
        return {
            f.rule
            for f in commit_guard.scan_message(
                message, repo=root, allowed_emails=emails, allowed_hosts=hosts
            )
        }

    specimens = {
        "trailer": "subject\n\nbody.\n\nReviewed-By: Someone Else <x@vendor.example>\n",
        "foreign-identity": "subject\n\nreported by x@vendor.example in the tracker.\n",
        "foreign-url": "subject\n\nsee https://vendor.example/thing for the rationale.\n",
        "badge-character": "subject\n\nbody \N{ROBOT FACE} done.\n",
        "co-authorship": "subject\n\nthis change was co-authored.\n",
        "credited-to-an-agent": "subject\n\nthe patch was written by an assistant.\n",
        "assistance-of": "subject\n\nwritten with the help of somebody.\n",
        "on-behalf-of": "subject\n\ncommitted on behalf of somebody.\n",
        "credited-to-a-link": "subject\n\nGenerated with [Some Tool](https://x.example/t)\n",
        "sign-off": "subject\n\nthis was signed-off-by a reviewer.\n",
        # Rule 6, and the reason its specimen is spelled as an escape: this file
        # is itself scanned for the character, so writing it literally would put
        # the tell in the tree in order to test for it.
        "em-dash": "subject\n\na body \N{EM DASH} with a typographic dash.\n",
        # [I11] Rule 6 widened from the dash to the whole non-ASCII punctuation
        # class. One specimen per CATEGORY that has ever appeared or plausibly
        # could: Po (the 5 already in this history), Pi/Pf (curly quotes), Pd is
        # above. Each is spelled as an escape for the same reason the dash is.
        "typographic-punctuation": "subject\n\na body \N{MIDDLE DOT} dot.\n",
    }
    for extra in (
        "subject\n\nquotes: \N{LEFT SINGLE QUOTATION MARK}x\N{RIGHT SINGLE QUOTATION MARK}\n",
        "subject\n\na body with an ellipsis\N{HORIZONTAL ELLIPSIS}\n",
        "subject\n\na body with an en dash \N{EN DASH} in it.\n",
    ):
        assert rules(extra) & {"typographic-punctuation", "em-dash"}, (
            f"rule 6 missed a member of its own class: {extra!r}"
        )
    # Rule 4's Zs half is PROSPECTIVE - 0 in 157 messages - so it is controlled
    # by a plant and never by a find.
    assert "badge-character" in rules("subject\n\na body\N{NO-BREAK SPACE}with nbsp.\n")
    for expected_rule, message in specimens.items():
        assert expected_rule in rules(message), (
            f"rule {expected_rule!r} went blind: it no longer fires on its own specimen"
        )

    # ...and the harder half. A pattern widened until it is ignored is switched
    # off exactly as thoroughly as one narrowed until it is silent. These are
    # real sentences from this repository's own commit messages and prose.
    must_not_fire = [
        "eval: make MISSING loud rather than silent (ADR-047)",
        "the model accelerates on 5 of 5 held-out blocks under both estimands",
        # This line USED to be rule 4's negative control, on the stated grounds
        # that dash punctuation is deliberately not a badge character. Rule 6
        # reversed that decision, so the line moved into `specimens` above and
        # what stands here in its place is the ASCII form the rule asks for. A
        # retired negative control is deleted, not left to pass for a new reason.
        "docs: interfaces v2.16 - a CV matrix may span splits",
        "readme: retract two public over-claims about deceleration (owed repair 9)",
        "common: derive the scoring-code module list instead of maintaining it",
        "the figure was generated by sim/movie.py from the run of record",
        "data: note the lead author of the GOFER paper in the provenance dict",
    ]
    for line in must_not_fire:
        assert not rules(line), f"false positive on ordinary prose: {line!r} -> {rules(line)}"


def test_the_two_forms_that_actually_ship_are_each_caught_by_TWO_rules() -> None:
    """The forms a harness emits, and the reason redundancy is deliberate here.

    Layer (a) can be bypassed and layer (b) is what catches the bypass, so a
    single rule carrying a whole form is a single point of failure inside the
    layer that is supposed to be the backstop. Both real forms trip at least two
    INDEPENDENT rules, so retiring or breaking any one rule still leaves them
    caught.
    """
    root, emails, hosts = _guard_inputs()
    for name, form in (("trailer form", TRAILER_FORM), ("badge form", BADGE_FORM)):
        found = commit_guard.scan_message(
            form, repo=root, allowed_emails=emails, allowed_hosts=hosts
        )
        distinct = {f.rule for f in found}
        assert len(distinct) >= 2, f"{name} is carried by a single rule {distinct}: no redundancy"


def test_the_specimens_are_caught_by_shape_and_not_by_name() -> None:
    """The fixtures use placeholder vendors. This proves that costs nothing.

    If any rule were a vendor list, swapping the placeholder would change the
    verdict. It does not: the same message with a different made-up vendor
    produces the same findings, which is what "derived rather than maintained"
    has to mean operationally.
    """
    root, emails, hosts = _guard_inputs()

    def count(message: str) -> int:
        return len(
            commit_guard.scan_message(
                message, repo=root, allowed_emails=emails, allowed_hosts=hosts
            )
        )

    for form in (TRAILER_FORM, BADGE_FORM):
        renamed = form.replace("vendor.example", "some-other-vendor.example").replace(
            "Some Assistant", "A Completely Different Product"
        )
        assert count(form) == count(renamed) >= 2, (
            "the verdict moved when only the vendor's NAME changed, so some rule is a name list"
        )


#: THE ONE DEBT THIS SCAN CARRIES, ENUMERATED BY SHA, AND WHY IT IS NOT AN
#: EXEMPTION. Rule 6 (no typographic dash) was adopted after this history was
#: written, and a commit message cannot be edited without rewriting history.
#: Rewriting was measured and rejected: it would orphan the published commits
#: while leaving each of them fetchable by sha, so the remedy would multiply the
#: exposure it was meant to remove. These messages are therefore PERMANENT.
#:
#: The list is keyed by full sha and by COUNT, so it fails in both directions:
#: a new offending commit is not in it, and an existing entry whose count no
#: longer matches means the history moved under the scan. It cannot grow
#: quietly, because every entry names an object that already exists and no
#: future commit can be added to it after the fact.
#:
#: Four of the seven dash commits are on `origin/main` and are the public ones;
#: three are reachable only from the local pre-public archive ref, which is never
#: pushed. The middle-dot commit is on `origin/main`.
#:
#: [I11] KEYED BY RULE AS WELL AS BY SHA. Rule 6 was widened from the dash to the
#: whole non-ASCII punctuation class, so the debt has to distinguish which tell
#: it is forgiving on which commit. A single count per sha would let a dash
#: allowance absorb a middle dot that appeared on the same commit, which is the
#: allow-list defect in miniature. Both directions still fail, now per rule.
PUBLISHED_PUNCTUATION_DEBT: dict[str, dict[str, int]] = {
    "25ad15697ac5d5e977070b5fcf3eaf8170202c03": {"typographic-punctuation": 5},
    "481722de2e442a7b5f311196817eaa94f06dd3e2": {"em-dash": 2},
    "56bf4e7bed0a99fdc92f6770971c2795ee11dd0e": {"em-dash": 2},
    "7d0e10d72a7ec2b302f8b0f5c6c9418ad5f4fe9e": {"em-dash": 2},
    "83a7cfb817fda54946c78bc0d0f02d25a2d7a657": {"em-dash": 1},
    "8639910f89a5af17e73d4506cfb8c7c2c81764c9": {"em-dash": 1},
    "ae3f26dccfb46389a6f09518b2f7344e65abf860": {"em-dash": 1},
    "f99f4870e21101bd28077198752f841fe2b60c83": {"em-dash": 1},
}

#: The rules whose findings the debt list may account for. Nothing else is ever
#: subtracted: an attribution finding on a debt-listed commit still fails.
_DEBT_RULES = ("em-dash", "typographic-punctuation")

#: [I11] THE DEBT SHAS THAT A CLONE CANNOT SEE, AND WHY THIS DISTINCTION IS NOT
#: A LOOPHOLE. Three of the eight are reachable only from
#: `refs/archive/pre-public-backup`, a LOCAL-ONLY ref that is never pushed. On
#: the maintainer's disk `git log --all` reaches 163 commits; in a clone it
#: reaches 111. A staleness check that reads "the commit now carries 0" cannot
#: tell "the message was rewritten" from "the commit is not in this corpus", and
#: it read the second as the first: the list was exact locally and turned CI red
#: on its first real run, which is this repository's "a local green is not
#: evidence" rule catching its own guard.
#:
#: So the membership is DECLARED and then MEASURED against
#: `origin/main` in both directions, rather than being inferred at run time from
#: the absence that caused the problem. An entry that is absent and NOT declared
#: local-only is still stale, loudly.
LOCAL_ONLY_DEBT_SHAS = frozenset(
    {
        "56bf4e7bed0a99fdc92f6770971c2795ee11dd0e",
        "8639910f89a5af17e73d4506cfb8c7c2c81764c9",
        "f99f4870e21101bd28077198752f841fe2b60c83",
    }
)


def unaccounted(
    offenders: Mapping[str, list[commit_guard.Finding]],
) -> dict[str, list[commit_guard.Finding]]:
    """Findings the debt list does not already account for, keyed by sha.

    Pure, so the planted-defect tests can feed it a hypothetical corpus. Only
    :data:`_DEBT_RULES` findings are ever subtracted, PER RULE, and only up to the
    declared count: a commit that acquires a SECOND kind of tell reports it even
    if it is on the list, and a dash allowance never absorbs a middle dot.
    """
    out: dict[str, list[commit_guard.Finding]] = {}
    for sha, findings in offenders.items():
        declared = PUBLISHED_PUNCTUATION_DEBT.get(sha, {})
        remainder = [f for f in findings if f.rule not in _DEBT_RULES]
        for rule in _DEBT_RULES:
            accountable = [f for f in findings if f.rule == rule]
            remainder += accountable[declared.get(rule, 0) :]
        if remainder:
            out[sha] = remainder
    return out


def stale_debt_entries(
    offenders: Mapping[str, list[commit_guard.Finding]],
    *,
    corpus: Iterable[str] | None = None,
) -> list[str]:
    """Debt entries whose commit no longer carries that many of that tell.

    A burn-down that only ever fails upward is an excuse. This history is
    immutable, so the only ways an entry can go stale are a rewrite or a typo in
    the list, and both are things the scan must say out loud rather than absorb.

    ``corpus`` is every sha that was SCANNED, which is not the same set as the
    shas that produced findings: a debt entry whose message was repaired would
    vanish from ``offenders`` and has to be distinguished from one whose commit
    was never in this corpus. Defaults to the keys of ``offenders``, which is the
    conservative reading (absent means repaired) and is why it should be passed.
    """
    known = set(offenders) if corpus is None else set(corpus)
    stale = []
    for sha, declared in sorted(PUBLISHED_PUNCTUATION_DEBT.items()):
        if sha not in known:
            if sha not in LOCAL_ONLY_DEBT_SHAS:
                stale.append(
                    f"{sha}: the debt list names it and this corpus does not contain it at all. "
                    "Either the history was rewritten or the entry is a typo"
                )
            continue
        for rule, expected in sorted(declared.items()):
            actual = len([f for f in offenders.get(sha, []) if f.rule == rule])
            if actual != expected:
                stale.append(
                    f"{sha}: the debt list says {expected} {rule}, the commit now carries {actual}"
                )
    return stale


def test_no_commit_reachable_from_any_ref_carries_attribution() -> None:
    """THE CHECK. Every commit, every ref, no window, and one enumerated debt."""
    offenders = unaccounted(_scan(commit_guard.commit_messages(repo_root())))
    assert not offenders, (
        "these commits carry attribution in their message. This repository is single author "
        "and its history carries none. Rewriting published history is expensive, so read "
        "`git log` for the shas below before deciding:\n  "
        + "\n  ".join(f"{sha}: {[str(f) for f in findings]}" for sha, findings in offenders.items())
    )


def test_the_punctuation_debt_is_exactly_what_it_declares() -> None:
    """The debt list is measured against the real history in both directions.

    Without this, `unaccounted` would silently forgive a sha whose message was
    never scanned at all, which is the allow-list defect with a nicer name.
    """
    messages = commit_guard.commit_messages(repo_root())
    offenders = _scan(messages)
    stale = stale_debt_entries(offenders, corpus=messages)
    assert not stale, "the punctuation debt list no longer describes this history:\n  " + (
        "\n  ".join(stale)
    )

    for rule in _DEBT_RULES:
        found = sum(len([f for f in findings if f.rule == rule]) for findings in offenders.values())
        declared = sum(
            d.get(rule, 0) for sha, d in PUBLISHED_PUNCTUATION_DEBT.items() if sha in messages
        )
        assert found == declared, (
            f"the scan found {found} {rule} findings across all refs but the debt list declares "
            f"{declared}"
        )
        # POSITIVE CONTROL, PER RULE. A debt list that accounted for everything,
        # or a rule that had gone blind, would make the assertion above pass
        # vacuously. Both counts are non-zero in this history: 10 dashes in 7
        # commits, 5 middle dots in 1.
        assert found > 0, f"rule 6 found no {rule} in a history known to contain some"
        planted = dict(offenders)
        planted[f"planted-{rule}"] = [commit_guard.Finding(rule, "planted")]
        assert set(unaccounted(planted)) == {f"planted-{rule}"}, (
            f"a {rule} finding on a commit the debt list does not name was forgiven"
        )


def test_the_debt_holds_on_THE_PUBLISHED_CORPUS_ALONE() -> None:
    """[I11] The corpus CI reads, reproduced here. This is the test that was missing.

    A clone reaches 111 commits from `origin/main`; this machine reaches 163 from
    `--all`, because 52 live on a local-only archive ref. The debt list was exact
    against the second and stale against the first, so it passed locally and
    turned CI red on its first real run. Measuring the gating corpus is not
    optional just because a bigger corpus is available.
    """
    published = commit_guard.commit_messages(repo_root(), refs=("origin/main",))
    assert len(published) > 50, "the published corpus scan found almost nothing"
    assert len(published) < len(commit_guard.commit_messages(repo_root())), (
        "the two corpora are the same size, so this test is measuring nothing new"
    )

    offenders = _scan(published)
    assert not stale_debt_entries(offenders, corpus=published), (
        "the debt list does not describe the corpus a clone sees:\n  "
        + "\n  ".join(stale_debt_entries(offenders, corpus=published))
    )
    assert not unaccounted(offenders)


def test_the_local_only_debt_shas_are_exactly_the_unpublished_ones() -> None:
    """Declared, then measured against git in BOTH directions.

    Inferring this set at run time from "the commit is missing" would forgive the
    exact defect the staleness check exists to catch, so it is written down and
    then checked against what `origin/main` actually reaches.
    """
    published = set(commit_guard.commit_messages(repo_root(), refs=("origin/main",)))
    assert LOCAL_ONLY_DEBT_SHAS <= set(PUBLISHED_PUNCTUATION_DEBT), (
        "a local-only entry that is not in the debt list is a stale declaration"
    )
    measured = {sha for sha in PUBLISHED_PUNCTUATION_DEBT if sha not in published}
    assert measured == LOCAL_ONLY_DEBT_SHAS, (
        f"declared local-only {sorted(LOCAL_ONLY_DEBT_SHAS)} but git says {sorted(measured)}. "
        "If one of these has since been published its entry must be asserted, not skipped."
    )


def test_a_debt_sha_that_is_absent_WITHOUT_being_declared_local_only_is_stale() -> None:
    """The control for the skip. Absence must not become a free pass."""
    present = {"481722de2e442a7b5f311196817eaa94f06dd3e2"}
    stale = stale_debt_entries({}, corpus=present)
    # the declared local-only three are skipped...
    for sha in LOCAL_ONLY_DEBT_SHAS:
        assert not any(sha in s for s in stale), sha
    # ...and every other absent entry is reported.
    for sha in PUBLISHED_PUNCTUATION_DEBT:
        if sha in LOCAL_ONLY_DEBT_SHAS or sha in present:
            continue
        assert any(sha in s for s in stale), f"{sha} went absent and nothing said so"
    # the one commit declared present carries no findings here, so its counts
    # are stale in the ordinary direction rather than skipped.
    assert any("481722de" in s and "carries 0" in s for s in stale), stale


def test_a_dash_allowance_does_not_forgive_a_middle_dot_on_the_same_commit() -> None:
    """The reason the debt is keyed by RULE. Planted on a real debt-listed sha."""
    sha = "481722de2e442a7b5f311196817eaa94f06dd3e2"
    assert PUBLISHED_PUNCTUATION_DEBT[sha] == {"em-dash": 2}, "fixture sha changed"

    within_allowance = {sha: [commit_guard.Finding("em-dash", "x")] * 2}
    assert unaccounted(within_allowance) == {}, "its own declared dashes must be forgiven"

    other_tell = {
        sha: [commit_guard.Finding("em-dash", "x")] * 2
        + [commit_guard.Finding("typographic-punctuation", "planted middle dot")]
    }
    left = unaccounted(other_tell)
    assert list(left) == [sha], left
    assert [f.rule for f in left[sha]] == ["typographic-punctuation"], left


def test_the_history_scan_catches_a_PLANTED_commit_message() -> None:
    """PLANTED DEFECT, C3.5. The scan must find a defect placed in a real corpus.

    Planted into the live corpus rather than into a fixture, so what is exercised
    is the same dictionary the check above reads, at the same size. The final
    assertion is the one that makes the first two mean anything: the UNPLANTED
    corpus is clean, so this test can distinguish a working scanner from one that
    reports everything.
    """
    live = dict(commit_guard.commit_messages(repo_root()))
    assert not unaccounted(_scan(live)), (
        "the live corpus is dirty beyond its declared em dash debt, so this planted defect "
        "proves nothing"
    )

    for label, form in (("planted0", TRAILER_FORM), ("planted1", BADGE_FORM)):
        planted = dict(live)
        planted[label] = form
        caught = unaccounted(_scan(planted))
        assert set(caught) == {label}, f"the planted {label} was not the only thing found: {caught}"


def test_author_and_committer_identity_is_single_across_the_whole_history() -> None:
    """Identity, checked separately from the message. Both halves were needed.

    The commit that shipped attribution had a CLEAN author and committer: the
    tell was in the message body only. Checking identity alone would have passed
    it, and checking the message alone would miss a genuine second contributor
    being added. These are two facts and they get two assertions.
    """
    rows = commit_guard.committer_identities(repo_root())
    assert rows, "no identities read: the scan lost its corpus"
    extra = commit_guard.single_identity_violations(rows)
    assert not extra, f"this repository is declared single author but the history carries: {extra}"

    # PLANTED DEFECT for this rule, on a hypothetical history: a second identity
    # must be reported. Without this the assertion above passes on a repo with
    # one commit just as happily as on one with a thousand.
    salted = set(rows) | {
        ("Someone Else", "else@vendor.example", "Someone Else", "else@vendor.example")
    }
    assert commit_guard.single_identity_violations(salted), "the identity check cannot fire"


# --------------------------------------------------------------------------
# [I4] Layer (a): the hook exists, is wired, and actually rejects
# --------------------------------------------------------------------------


def test_the_commit_msg_hook_arrives_with_make_install_rather_than_by_instruction() -> None:
    """A guard that only works if someone remembers a command is not a guard.

    Three separate things have to be true and each fails silently on its own:
    `pre-commit install` must wire the commit-msg hook type (it wires only
    `pre-commit` by default), the hook must be registered at the commit-msg
    stage, and `make install` must run the installer.
    """
    config = yaml.safe_load((repo_root() / ".pre-commit-config.yaml").read_text())
    assert "commit-msg" in config.get("default_install_hook_types", []), (
        "`pre-commit install` will not wire the commit-msg hook, so the guard would arrive as "
        "an instruction; set `default_install_hook_types`"
    )
    local = [r for r in config["repos"] if r.get("repo") == "local"]
    hooks = [h for r in local for h in r["hooks"] if h["id"] == "commit-guard"]
    assert len(hooks) == 1, "the commit-guard hook is not registered exactly once"
    assert hooks[0]["stages"] == ["commit-msg"], hooks[0]
    entry = Path(hooks[0]["entry"])
    script = repo_root() / entry
    assert script.is_file(), f"the hook points at {entry}, which does not exist"
    assert script.stat().st_mode & 0o111, f"{entry} is not executable, so `language: script` fails"
    assert script.read_text().startswith("#!"), f"{entry} has no shebang"

    makefile = (repo_root() / "Makefile").read_text()
    install_body = re.search(r"^install:.*?(?=^\S)", makefile, flags=re.MULTILINE | re.DOTALL)
    assert install_body and "hooks" in install_body.group(0), (
        "`make install` no longer installs the hooks, so a fresh clone would have none"
    )


def test_the_hook_REJECTS_and_ACCEPTS_end_to_end_through_its_real_entry_point(
    tmp_path: Path,
) -> None:
    """PLANTED DEFECT for layer (a), run as a subprocess through the shebang.

    Not by importing the function: the hook is a script invoked by git, and the
    ways this breaks in practice (a lost executable bit, a broken shebang, a
    non-zero exit that is not read) are invisible to an in-process call.
    """
    script = repo_root() / "tools" / "commit_guard.py"

    def run(message: str) -> subprocess.CompletedProcess[str]:
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(message)
        return subprocess.run(
            [str(script), str(path), "--repo", str(repo_root())],
            capture_output=True,
            text=True,
            check=False,
        )

    for form in (TRAILER_FORM, BADGE_FORM, DASH_FORM):
        rejected = run(form)
        assert rejected.returncode == 1, f"the hook ACCEPTED an attribution trailer: {rejected}"
        assert "commit-guard" in rejected.stderr

    clean = run("tests: add the commit-message attribution guard\n\nan ordinary body.\n")
    assert clean.returncode == 0, f"the hook REJECTED a clean message: {clean.stderr}"

    # The dash control is the same sentence with the ASCII form, so what the two
    # runs differ by is one character and nothing else. Without it, DASH_FORM
    # being rejected would be consistent with the hook rejecting everything.
    ascii_dash = run(DASH_FORM.replace("\N{EM DASH}", "-"))
    assert ascii_dash.returncode == 0, f"the hook REJECTED an ASCII dash: {ascii_dash.stderr}"

    # git's own comment lines are dropped before scanning, because a guard that
    # rejects text git is about to discard is a guard someone turns off.
    commented = run(
        "tests: add the guard\n\n# Co-Authored-By: Someone <x@vendor.example>\n# please ignore\n"
    )
    assert commented.returncode == 0, "the hook read a comment git would have stripped"


def test_the_maintained_surface_is_pinned_and_is_the_only_one() -> None:
    """Rule 5 is the only list, and rules 1-4 must not quietly grow one.

    The pin makes widening rule 5 a visible edit. The second half is the part
    that matters more: if `tools/commit_guard.py` ever contained this
    repository's own address as a literal, rule 2 would have stopped being
    derived and nothing else would have said so.
    """
    assert set(commit_guard.ATTRIBUTION_CONSTRUCTIONS) == EXPECTED_CONSTRUCTIONS, (
        "rule 5's pattern set has moved. That is allowed, and it is meant to be a diff someone "
        "argues for: update EXPECTED_CONSTRUCTIONS in the same commit."
    )
    assert len(EXPECTED_CONSTRUCTIONS) <= 8, "the maintained surface is growing into a tell list"

    source = (repo_root() / "tools" / "commit_guard.py").read_text()
    assert not commit_guard._EMAIL_RE.findall(source), (
        "the guard now contains a literal e-mail address, so rule 2 is no longer derived from "
        "git's own identity"
    )
    hosts = {commit_guard._url_host(u) for u in commit_guard._URL_RE.findall(source)}
    assert not (hosts - {""}), (
        f"the guard now hardcodes a host {hosts}, so rule 3 is no longer derived from the repo's "
        "own remotes"
    )
