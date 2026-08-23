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
import inspect
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from unittest import mock

import pytest
import yaml

import commit_guard  # tools/commit_guard.py, via `pythonpath` in pyproject.toml
import isolated_suite  # tools/isolated_suite.py, same route
import mutation  # tools/mutation.py, same route
import prose_scan  # tools/prose_scan.py, same route
from wildfire_nowcast.common.paths import repo_root
from wildfire_nowcast.eval import stage
from wildfire_nowcast.eval.stage import estimand_digest

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
    # The SAME internal vocabulary in its other spelling. This pattern was added
    # after a measurement, not on principle: 18 occurrences across four tracked
    # files, in three different leads' hand, none of which the two patterns above
    # could see because none of them says "lead" or names the instruction file.
    # To a public reader it reads as a handle for a user that does not exist. The
    # negative lookahead is load-bearing: `@dataclass` and `@dataclasses.field`
    # are ordinary Python and must never match.
    "agent handle": "@" + r"(?:infra|data|model|sim|simviz|maintainer)(?![A-Za-z])",
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


#: Tracked EVIDENCE, excluded from the tell scan and counted rather than ignored.
#:
#: The twenty artifacts under ``runs/`` are tracked so that published numbers can
#: be CHECKED by a clone. They are machine-written records of runs that happened,
#: and they may not be edited to look better, which is the same reason the four
#: ``baselines-*/results.json`` are declared out of the tree entirely in
#: `.gitignore` rather than cleaned. A tell inside one of them cannot be repaired
#: without falsifying evidence, so an entry in the burn-down list above would be
#: an exemption wearing a burn-down list's clothes: nobody could ever retire it.
#: The exclusion is therefore explicit, narrow, and MEASURED by
#: `test_the_artifact_exclusion_is_narrow_and_is_not_hiding_a_growing_population`.
ARTIFACT_PREFIX = "runs/"


def scan_tracked_artifacts() -> dict[str, int]:
    """``{artifact: number of tells}`` for the tracked evidence the scan excludes."""
    counts: dict[str, int] = {}
    for rel in _tracked_files():
        if not rel.startswith(ARTIFACT_PREFIX):
            continue
        path = repo_root() / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        n = len(_TELL_RE.findall(text))
        if n:
            counts[rel] = n
    return counts


def scan_tracked_tree() -> dict[str, int]:
    """``{relative path: number of tells}``, for every tracked file with at least one."""
    counts: dict[str, int] = {}
    for rel in _tracked_files():
        if rel.startswith(ARTIFACT_PREFIX):
            continue  # evidence, not a reading surface: see ARTIFACT_PREFIX
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


def test_the_agent_handle_pattern_answers_both_ways() -> None:
    """CAPABILITY, on strings small enough to read, independent of the tree.

    The lookahead is the whole risk in this pattern: matched too widely it hits
    ``@dataclass`` on 40 modules and the family gets switched off; matched too
    narrowly it sees nothing and reads as coverage. Both directions are asserted
    here rather than inferred from a count of what the tree happens to contain.
    """
    handle = "@" + "simviz"
    for fires in (f"proposed by {handle}, ruled on here", "@" + "model" + " owns it"):
        assert _TELL_RE.search(fires), f"the handle pattern stopped matching: {fires!r}"
    for clean in (
        "@dataclass(frozen=True)",
        "@dataclasses.dataclass",
        "@modelling_helper",
        "the data/ package owns it",
        "email someone@example.com about it",
    ):
        assert not _TELL_RE.search(clean), f"false positive on: {clean!r}"


def test_the_artifact_exclusion_is_narrow_and_is_not_hiding_a_growing_population() -> None:
    """What the tell scan does NOT read, measured rather than trusted.

    Tracked evidence is excluded because a tell inside a run record cannot be
    repaired without falsifying the record. An exclusion nobody measures is how a
    scan goes quiet, so the population inside it is pinned exactly: a NEW tell in
    tracked evidence fails here even though it is excluded from the gate, and a
    tell that disappears fails too, because evidence is not supposed to change.
    """
    assert ARTIFACT_PREFIX == "ru" + "ns/", "the exclusion has moved off tracked evidence"
    excluded = scan_tracked_artifacts()
    assert excluded == {"runs/u0b.json": 1}, (
        "the tells inside tracked evidence have changed. This population is FROZEN: "
        f"the records may not be edited to look better. Measured {excluded}."
    )
    # ...and the exclusion really is what is keeping it out of the gate.
    assert not any(rel.startswith(ARTIFACT_PREFIX) for rel in scan_tracked_tree())
    assert new_or_grown_tells(scan_tracked_tree()) == []


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
    assert set(published) <= set(commit_guard.commit_messages(repo_root())), (
        "the published corpus is not a subset of every ref, so one of the two scans is wrong"
    )

    # NON-VACUITY THAT HOLDS IN A CLONE TOO, and the reason this is not phrased
    # as "the two corpora differ": in a clone they do NOT differ, because the
    # extra 52 commits live on a ref a clone never receives. An assertion that
    # is true only on the maintainer's disk is the same defect this test exists
    # to repair, one level up, and it turned CI red once before being fixed.
    expected_here = {
        sha: d for sha, d in PUBLISHED_PUNCTUATION_DEBT.items() if sha not in LOCAL_ONLY_DEBT_SHAS
    }
    assert set(expected_here) <= set(published), (
        "a debt sha declared PUBLISHED is missing from the published corpus"
    )

    offenders = _scan(published)
    assert sum(len(v) for v in offenders.values()) == sum(
        sum(d.values()) for d in expected_here.values()
    ), "the published corpus scan does not find the tells the debt list declares for it"
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


# --------------------------------------------------------------------------
# THE SOURCE-PROSE PIN
#
# The sweep of 2026-08-21 removed 1,732 typographic characters from prose, and
# the guard that followed it covers commit MESSAGES. Nothing covered SOURCE: an
# em dash planted in `common/paths.py` afterwards left this suite at 44 passed,
# exit 0, and the tracked count drifted 407 -> 415 during the sweep itself. A
# one-time cleanup with no mechanism is a snapshot; this is the mechanism.
#
# Scope is decided STRUCTURALLY, by `tools/prose_scan.py`, never by a list of
# file names. Docstrings and comments are prose and are bounded; live string
# literals are 467 occurrences of JSON field names, contract violation messages
# and expected values a test compares against, where removing a character is a
# behaviour change, and they are deliberately NOT bounded.
# --------------------------------------------------------------------------

#: MEASURED at `cc82876`, by `.venv/bin/python tools/prose_scan.py --repo .`:
#: 14 in docstrings + 2 in comments = 16, of which exactly 1 is exempt. THIS
#: NUMBER MAY FALL AND MAY NOT RISE, and it fails in both directions: a pin
#: larger than the debt forgives the next regression, which is the failure mode
#: seven allow-lists in this repository have already had.
#: The 15 are 10 SECTION SIGN, 3 MIDDLE DOT and 2 more section signs across
#: `common/`, `data/`, `sim/` and `tests/` - the DASH sweep did not widen to the
#: class, exactly as the commit-message rule had to be widened at I11.
PINNED_PROSE_OCCURRENCES = 15

#: The one permanent exception, ADR-080 (2). Pinned as a COUNT so that it
#: expires with its reason: see the test below for the two conditions that must
#: both hold for it to be granted at all.
PINNED_EXEMPT_OCCURRENCES = 1

#: Tracked NON-Python text: everything except machine output and binaries, so
#: `Makefile`, `LICENSE` and `.github/workflows/ci.yml` are IN. Separate number
#: from the one above because these files have no literals to distinguish and a
#: different set of owners.
#: The deny-list shape earned itself immediately: an allow-list of `.md .rst
#: .txt .cfg .ini` read 2, and inverting it to "not machine output" read 7 - the
#: five it could not see were MIDDLE DOTs in the comment block of the CI
#: workflow, the most public file in the repository. Swept here, comment-only,
#: with the parsed YAML document compared before and after. The 2 that remain
#: are in `docs/interfaces.md`, which is the contract document and is the
#: maintainer's to edit, not infra's.
PINNED_PROSE_FILE_OCCURRENCES = 2

#: NOT a bound, and deliberately not pinned to equality: a lead adding a contract
#: violation message moves it, and a pin that another lead's legitimate edit turns
#: red is a pin that gets deleted. Recorded so that the boundary of the pin above
#: is a measurement rather than a sentence. Measured at `cc82876`.
MEASURED_LIVE_LITERALS = 467


def _prose_scan() -> list[prose_scan.Occurrence]:
    return prose_scan.scan_repository(repo_root())


def _licensed_exempt_spans() -> list[prose_scan.Span]:
    """The spans the published estimand digest hashes, or an assertion failure.

    THIS is the exemption, and it is granted by a live measurement rather than by
    a file name. `eval/stage.py:estimand_digest` hashes the source of four
    top-level definitions INCLUDING their docstrings, and pins the hash to the
    value that licensed the D3 estimand. An em dash inside those spans may not be
    swept, because moving it would move `D3_LICENSED_ESTIMAND_SHA256` and break
    the only link between published numbers and the code that computed them.

    Two conditions, both checked here. If the digest stops hashing these spans -
    a function leaving `ESTIMAND_FUNCTIONS`, a rename, a move out of module scope
    - the spans derived here stop reproducing the digest and this fails. If the
    digest is re-pinned or reads CHANGED, the reason for the exemption is gone and
    this fails too. Neither can be satisfied by editing a list of paths.
    """
    spans, derived = prose_scan.estimand_hashed_spans(repo_root())
    live = estimand_digest()
    assert derived == live["sha256"], (
        "the spans this exemption is derived from no longer reproduce the digest that "
        f"licenses it (derived {derived}, live {live['sha256']}). The exemption is granted to "
        "the BYTES eval/stage.estimand_digest hashes, not to a file. If the estimand moved, "
        "sweep the dash and delete PINNED_EXEMPT_OCCURRENCES; do not re-derive around it."
    )
    assert live["outcome"] == "UNCHANGED_SINCE_D3", (
        f"estimand_digest reads {live['outcome']}. The exemption exists because those bytes are "
        "pinned to published numbers; once they are not, the reason is gone and the exemption "
        "goes with it."
    )
    return spans


def test_typographic_punctuation_in_prose_is_pinned_and_only_falls() -> None:
    """The durability half. Bounded in BOTH directions, like the mypy burn-down."""
    prose = [o for o in _prose_scan() if o.region in prose_scan.PROSE_REGIONS]
    exempt, in_scope = prose_scan.partition_exempt(prose, _licensed_exempt_spans())

    assert len(exempt) == PINNED_EXEMPT_OCCURRENCES, (
        f"the licensed exemption now covers {len(exempt)} occurrence(s), not "
        f"{PINNED_EXEMPT_OCCURRENCES}. If it is 0 the reason has evaporated and the constant "
        "must go; if it grew, someone put new typography inside a hashed estimand."
    )
    assert len(in_scope) <= PINNED_PROSE_OCCURRENCES, (
        f"typographic punctuation in docstrings and comments has GROWN to {len(in_scope)}, "
        f"over the pin of {PINNED_PROSE_OCCURRENCES}. Prose is free to fix - the character is "
        "a tell and nothing reads it - so fix it rather than raising the pin:\n  "
        + "\n  ".join(str(o) for o in sorted(in_scope, key=lambda o: (o.path, o.line)))
    )
    assert len(in_scope) == PINNED_PROSE_OCCURRENCES, (
        f"GOOD NEWS, and it needs one edit: the debt is down to {len(in_scope)}. Lower "
        f"PINNED_PROSE_OCCURRENCES to {len(in_scope)} in this commit. A pin larger than the "
        "debt is an allowance for the next one."
    )


def test_the_exemption_expires_with_its_reason_and_not_with_its_file_name() -> None:
    """The plant for the exemption: take away its reason and it must stop applying.

    Rebinding `ESTIMAND_FUNCTIONS` is the same manoeuvre as dropping a function
    from it. The exempt occurrence must become IN SCOPE, and the derivation must
    stop reproducing the published digest - so the burn-down would go red rather
    than carry an exemption whose justification no longer exists.
    """
    prose = [o for o in _prose_scan() if o.region in prose_scan.PROSE_REGIONS]
    licensed = prose_scan.estimand_hashed_spans(repo_root())
    exempt_now, in_scope_now = prose_scan.partition_exempt(prose, licensed[0])
    assert exempt_now, "nothing is exempt today, so this plant would prove nothing"

    with mock.patch.object(stage, "ESTIMAND_FUNCTIONS", ("paired_stage_gap",)):
        narrowed, digest = prose_scan.estimand_hashed_spans(repo_root())
    assert digest != estimand_digest()["sha256"], (
        "narrowing the estimand did not move the derived digest, so the derivation is not "
        "actually reading ESTIMAND_FUNCTIONS and the exemption is pinned to nothing"
    )
    exempt_after, in_scope_after = prose_scan.partition_exempt(prose, narrowed)
    assert not exempt_after, "the exemption survived the removal of the span that licensed it"
    assert len(in_scope_after) == len(exempt_now) + len(in_scope_now), (
        "the occurrence did not come back into scope, so the exemption is not a partition. "
        "Measured against the LIVE split rather than against the pin, so that lowering the "
        "pin cannot make this test say something it did not measure."
    )


def test_typographic_punctuation_in_tracked_prose_files_is_pinned_too() -> None:
    """Markdown and friends. Whole-file prose, so there is no literal to exclude."""
    found = [o for o in _prose_scan() if o.region == prose_scan.REGION_PROSE]
    assert len(found) == PINNED_PROSE_FILE_OCCURRENCES, (
        f"tracked prose files carry {len(found)} typographic characters against a pin of "
        f"{PINNED_PROSE_FILE_OCCURRENCES}. Raising this pin is not a repair:\n  "
        + "\n  ".join(str(o) for o in sorted(found, key=lambda o: (o.path, o.line)))
    )


def test_the_prose_corpus_is_a_REFUSAL_and_not_a_list_of_extensions() -> None:
    """An allow-list of suffixes cannot see the files that have none.

    This is the same defect twice on the record: a citation pattern rooted at
    `/runs/` could not see `data/fires`, and a json-only pattern could not
    see a subdirectory. Here it would have been `Makefile`, `LICENSE` and the CI
    workflow. The rule is therefore stated as what is EXCLUDED - machine output
    and binaries - so that a file nobody enumerated is in scope by default.
    """
    for rel in ("Makefile", "LICENSE", ".gitignore", ".github/workflows/ci.yml", "README.md"):
        assert prose_scan.is_prose_file(rel), rel
    for rel in ("runs/s1.json", "requirements.lock", "reports/figures/x.png"):
        assert not prose_scan.is_prose_file(rel), rel
    assert not prose_scan.is_prose_file("src/wildfire_nowcast/common/paths.py")

    tracked = prose_scan.tracked_files(repo_root())
    corpus = [rel for rel in tracked if prose_scan.is_prose_file(rel)]
    assert "Makefile" in corpus and ".github/workflows/ci.yml" in corpus, (
        "the extensionless and workflow files are not in the live corpus, so the rule above "
        "is aspirational rather than what the pin actually measures"
    )
    assert not [rel for rel in corpus if rel.endswith(".json")], corpus[:5]


def test_the_pin_is_not_bounding_the_live_literals() -> None:
    """The control that proves the pin CAN pass, and is not passing by scoping to nothing.

    467 typographic characters sit in live literals today. If the classifier ever
    folded them into prose the pin could not be met at all, and if it ever folded
    prose into literals the pin would pass vacuously. Measuring both halves on the
    SAME corpus is what makes either number readable.
    """
    scanned = _prose_scan()
    literals = [o for o in scanned if o.region == prose_scan.REGION_LITERAL]
    assert len(literals) > 10 * PINNED_PROSE_OCCURRENCES, (
        f"live literals are down to {len(literals)} against {MEASURED_LIVE_LITERALS} measured "
        f"at cc82876, which puts them within an order of the {PINNED_PROSE_OCCURRENCES} the pin "
        "bounds. Either a sweep has started rewriting literals - which changes behaviour and "
        "artifact bytes - or the classifier has begun counting them as prose."
    )
    assert not [o for o in scanned if o.region == prose_scan.REGION_CODE], (
        "a typographic character was found outside every comment, docstring and literal. "
        "That should be a syntax error; the classifier is more likely to be wrong than Python"
    )


_SPECIMEN = '''"""A module docstring with an em dash — in it."""

# A comment with a middle dot · in it.

MESSAGE = "a live literal with an em dash — that may not be swept"


def f() -> None:
    """A docstring whose dash is ESCAPED \\u2014 and is invisible to a byte scan."""
    return None
'''


def test_the_scanner_tells_prose_from_a_live_literal_structurally() -> None:
    """C3.5: the classifier ships with the four cases it has to separate.

    One specimen, four characters, four different verdicts - and no file name is
    consulted to reach any of them.
    """
    found = prose_scan.scan_python_source("specimen.py", _SPECIMEN)
    by_region: dict[str, list[prose_scan.Occurrence]] = {}
    for occ in found:
        by_region.setdefault(occ.region, []).append(occ)

    assert len(by_region[prose_scan.REGION_DOCSTRING]) == 2, by_region
    assert len(by_region[prose_scan.REGION_COMMENT]) == 1, by_region
    assert len(by_region[prose_scan.REGION_LITERAL]) == 1, by_region
    assert prose_scan.REGION_CODE not in by_region, by_region

    escaped = [o for o in by_region[prose_scan.REGION_DOCSTRING] if o.line == 9]
    assert escaped, (
        "the ESCAPED dash was missed. A byte-level scan reads `\\u2014` as six ASCII "
        "characters while every reader of help() sees an em dash; 31 such escapes are "
        "already sitting in tracked artifacts because a scan claimed 0 non-ASCII bytes"
    )
    assert "—" not in _SPECIMEN[_SPECIMEN.index("ESCAPED") :], (
        "the escaped case is only a test if the specimen really has no literal dash there"
    )


def test_the_plant_this_suite_missed_is_now_caught_and_the_control_passes() -> None:
    """The maintainer's own plant, replayed: an em dash in `common/paths.py`.

    It went into a docstring, the hygiene suite returned 44 passed, exit 0, and
    that is the whole reason this pin exists. RED with the plant, GREEN without,
    on the real file's real text.
    """
    text = (SRC / "common" / "paths.py").read_text()
    clean = prose_scan.scan_python_source("common/paths.py", text)
    assert not [o for o in clean if o.region in prose_scan.PROSE_REGIONS], (
        "the control failed: common/paths.py is not clean, so a red result below would "
        "not be attributable to the plant"
    )

    head, sep, tail = text.partition('"""')
    assert sep, "common/paths.py has no docstring to plant in"
    planted = head + sep + "planted — dash. " + tail
    found = [
        o
        for o in prose_scan.scan_python_source("common/paths.py", planted)
        if o.region in prose_scan.PROSE_REGIONS
    ]
    assert len(found) == 1 and found[0].char == "—", found


# --------------------------------------------------------------------------
# THE MUTATION GATE (plan Task 5.6)
#
# `make mutation` is the slow half and runs on demand; these are the assertions
# that must hold on every commit, so that the gate cannot rot between sweeps.
# The sweep itself is not run here: it takes about forty minutes and builds git
# worktrees, and a suite that does that is a suite people stop running.
# --------------------------------------------------------------------------

_MUTATION_SNIPPET = """
def f(a, b, flag):
    if a >= b and not flag:
        return a * 2 + 1
    return len(b)
"""


def test_the_mutation_gate_is_a_make_target_and_is_deliberately_not_in_ci() -> None:
    """The gate exists, it names the tool, and it stays out of the three-minute path."""
    makefile = (repo_root() / "Makefile").read_text()
    assert re.search(r"^mutation:", makefile, re.M), "make mutation has gone"
    assert "tools/mutation.py" in makefile, "the target no longer invokes the sweep"
    gate = re.search(r"^ci: (.*)$", makefile, re.M)
    check = re.search(r"^check: (.*)$", makefile, re.M)
    assert gate and check
    assert "mutation" not in gate.group(1).split(), (
        "make ci now runs the mutation sweep. That is a forty-minute gate on a path whose "
        "job is a verdict in three, and a gate people wait out is a gate they route around. "
        "If it belongs in CI it belongs in a scheduled job, which is a different edit."
    )
    assert "mutation" not in check.group(1).split(), check.group(1)


def test_the_mutation_catalogue_finds_exactly_the_tokens_it_declares() -> None:
    """The enumerator, with the control: a token outside the catalogue is not a site.

    `len` and `return` are NAME tokens and `(` is an OP token, and none of the
    three is mutable. Without that half, an enumerator that reported every token
    would pass the first assertion and would make every survivor count meaningless.
    """
    found = {(s.old, s.new) for s in mutation.mutable_sites(_MUTATION_SNIPPET)}
    assert (">=", ">") in found
    assert ("and", "or") in found
    assert ("not", "") in found
    assert ("*", "/") in found
    assert ("+", "-") in found
    assert ("2", "3") in found and ("1", "2") in found
    assert not {pair for pair in found if pair[0] in ("len", "return", "(", ")", ":", "f", "a")}, (
        f"the enumerator reported an unmutable token: {sorted(found)}"
    )
    assert all(s.new != s.old for s in mutation.mutable_sites(_MUTATION_SNIPPET))


def test_the_mutant_selection_is_deterministic_and_lands_inside_the_list() -> None:
    """A sweep whose sample moved between runs could not carry a pin at all."""
    sites = mutation.mutable_sites(_MUTATION_SNIPPET)
    for fraction in mutation.SAMPLE_FRACTIONS:
        first = mutation.select_site(sites, fraction)
        assert first in sites
        assert first == mutation.select_site(sites, fraction)
    chosen = {mutation.select_site(sites, f) for f in mutation.SAMPLE_FRACTIONS}
    assert len(chosen) == len(mutation.SAMPLE_FRACTIONS), (
        f"the fractions collapse onto {len(chosen)} distinct site(s) on this snippet, so the "
        "sample is smaller than it claims"
    )


def test_the_survivor_budget_is_a_number_with_its_method_attached() -> None:
    """A pin without a reproduction recipe is a number someone has to believe."""
    assert isinstance(mutation.SURVIVOR_BUDGET, int) and mutation.SURVIVOR_BUDGET >= 0
    assert "mutants" in mutation.MEASURED_AT and "common/" in mutation.MEASURED_AT
    assert mutation.TARGET_PACKAGES == ("common", "eval")


def test_the_swept_corpus_is_not_empty_and_is_the_two_declared_packages() -> None:
    """The vacuity check: a sweep over no module reports 0 survivors and passes."""
    modules = mutation.target_modules(repo_root())
    assert len(modules) > 30, modules
    assert all(m.endswith(".py") for m in modules)
    for package in mutation.TARGET_PACKAGES:
        assert any(f"/{package}/" in m for m in modules), package
    assert not [m for m in modules if "/model/" in m or "/sim/" in m or "/data/" in m], (
        "the sweep has grown outside common/ and eval/, which changes what the budget counts"
    )


def test_the_sweep_reads_pytests_own_summary_and_not_the_progress_output() -> None:
    """`failing_tests` decides what gets deselected, so it is pinned in both directions."""
    output = (
        "tests/test_a.py .F                             [ 50%]\n"
        "FAILED tests/test_a.py::test_one - AssertionError: boom\n"
        "ERROR tests/test_b.py::test_two\n"
        "FAILED tests/test_a.py::test_one - AssertionError: boom\n"
        "1 failed, 2 passed in 1.00s\n"
    )
    assert mutation.failing_tests(output) == [
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
    ], "the deselect list must be de-duplicated, sorted, and must carry ERROR as well as FAILED"
    assert mutation.failing_tests("2 passed in 0.1s\n") == []
    assert mutation.failing_tests("a line mentioning FAILED in the middle\n") == []


# --------------------------------------------------------------------------
# THE STALE-BYTECODE HAZARD, MADE STRUCTURALLY IMPOSSIBLE
#
# CPython invalidates a `.pyc` on (source mtime in WHOLE SECONDS, source size).
# A same-length edit made and reverted inside one wall-clock second is therefore
# invisible and the stale bytecode runs instead: a file that verifiably reads
# `max(a, b)` executes `min(a, b)`. Reproduced here, in this file, below.
#
# It breaks a whole class of by-hand verification. A mutant that never executes
# leaves the suite green and is recorded as a SURVIVOR it never was; a mutant
# left in the cache after a revert leaves the suite red against a tree that is
# byte-identical to HEAD. Both directions are silent.
#
# A protocol that depends on a person remembering to clear a cache is not a
# protocol, so the setting lives on the repository's own test path.
# --------------------------------------------------------------------------


def test_the_repo_test_path_disables_bytecode_writing() -> None:
    """`make test`, `make test-all` and therefore `make ci` all reach pytest through
    ONE variable, so the setting travels with them rather than with a habit."""
    makefile = (repo_root() / "Makefile").read_text()
    pytest_var = re.search(r"^PYTEST := (.*)$", makefile, re.M)
    assert pytest_var, "the PYTEST variable has gone, so this test is checking nothing"
    assert "PYTHONDONTWRITEBYTECODE=1" in pytest_var.group(1), (
        "the test path no longer disables bytecode writing. A same-length source edit and "
        "revert inside one second then runs stale bytecode, and every by-hand mutation check "
        "in this repository silently stops measuring what it says it measures."
    )
    for target in ("test:", "test-all:", "playthrough:"):
        body = re.search(rf"^{re.escape(target)}.*\n((?:\t.*\n)+)", makefile, re.M)
        assert body and "$(PYTEST)" in body.group(1), (
            f"{target} now invokes pytest directly and routes around the setting: {body}"
        )

    # The other half, and it is a different half. Disabling the WRITE does not
    # stop a `.pyc` already on disk from being READ, and eleven such directories
    # were live in this tree when the defect was found.
    assert re.search(r"^purge-bytecode:\n\t.*__pycache__", makefile, re.M), (
        "the purge target has gone or stopped naming __pycache__"
    )
    for target in ("test", "test-all"):
        assert re.search(rf"^{re.escape(target)}: purge-bytecode\b", makefile, re.M), (
            f"`make {target}` no longer clears stale bytecode before it runs, so a cache "
            "written before the setting existed can still decide what executes"
        )


def test_the_purge_removes_a_real_cache_and_leaves_the_source_alone(tmp_path: Path) -> None:
    """The purge is measured by running THE MAKEFILE'S OWN recipe on a cache it made.

    Deleting directories by pattern is exactly the kind of line that reads correctly
    and behaves otherwise, so it gets an execution rather than a reading. The
    command is lifted out of the Makefile rather than copied into this file: a copy
    would keep passing after the real one was changed.
    """
    makefile = (repo_root() / "Makefile").read_text()
    recipe = re.search(r"^purge-bytecode:\n\t@?(.*)$", makefile, re.M)
    assert recipe, "the purge recipe is gone"
    command = recipe.group(1)

    for name in ("src", "tests", "tools"):
        (tmp_path / name).mkdir()
    (tmp_path / "src" / "m.py").write_text("VALUE = 1\n")
    keep = tmp_path / "src" / "keep.txt"
    keep.write_text("not bytecode\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import m", "x"],
        check=True,
        capture_output=True,
        cwd=tmp_path / "src",
        env={**env, "PYTHONPATH": str(tmp_path / "src")},
    )
    cache = tmp_path / "src" / "__pycache__"
    assert cache.is_dir(), "the fixture never produced a cache, so the purge would prove nothing"

    subprocess.run(command, shell=True, cwd=tmp_path, check=True)
    assert not cache.exists(), f"`{command}` did not remove the cache"
    assert (tmp_path / "src" / "m.py").read_text() == "VALUE = 1\n", "the purge edited a source"
    assert keep.is_file(), "the purge deleted a file that is not bytecode"
    subprocess.run(command, shell=True, cwd=tmp_path, check=True)


def test_the_hazard_is_real_and_the_probe_the_sweep_uses_detects_it(tmp_path: Path) -> None:
    """The positive control. Reproduce the defect, then show the guard reports it.

    Written as an end-to-end reproduction rather than as a claim about CPython:
    the module is imported, edited to a SAME-LENGTH variant, re-imported, and
    returns the OLD answer. If a future CPython changes its invalidation rule this
    fails, and that is the correct outcome, because the guard downstream would then
    be defending against nothing.

    THE TIMESTAMPS ARE SET, NOT WAITED FOR. The first version of this test simply
    edited the file quickly and hoped both writes landed in the same wall-clock
    second; it passed alone and went RED under load, when the two writes straddled
    a second boundary and the hazard did not reproduce. A test for a timing hazard
    that is itself timing-dependent reports the load on the machine. So the mtime is
    pinned to a fixed whole second and then advanced by half of one: the mutated
    file is genuinely NEWER than the bytecode and the same size, which is the exact
    condition CPython cannot see, and it is now reproduced on every run rather than
    on a lucky one.
    """
    victim = tmp_path / "victim.py"
    victim.write_text("def pick(a, b):\n    return min(a, b)\n")
    probe = (
        "import importlib, importlib.util, marshal, os, pathlib, sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        f"p = pathlib.Path({str(victim)!r})\n"
        "STAMP = 1700000000.0\n"
        "os.utime(p, (STAMP, STAMP))\n"
        "import victim\n"
        "first = victim.pick(1, 2)\n"
        "p.write_text(p.read_text().replace('min(a, b)', 'max(a, b)'))\n"
        "os.utime(p, (STAMP + 0.5, STAMP + 0.5))\n"
        "del sys.modules['victim']\n"
        "importlib.invalidate_caches()\n"
        "import victim as again\n"
        "spec = importlib.util.find_spec('victim')\n"
        "loaded = spec.loader.get_code('victim')\n"
        "fresh = compile(pathlib.Path(spec.origin).read_text(), spec.origin, 'exec')\n"
        "print(first, again.pick(1, 2),\n"
        "      'MATCH' if marshal.dumps(loaded) == marshal.dumps(fresh) else 'STALE')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": ""},
    ).stdout.split()
    assert out[0] == "1", out
    assert out[1] == "1", (
        "the same-length edit DID take effect, so this interpreter no longer reproduces the "
        f"hazard and the STALE_BYTECODE verdict is defending against nothing: {out}"
    )
    assert out[2] == "STALE", (
        f"the probe the mutation sweep relies on did not notice the stale bytecode: {out}"
    )
    assert (tmp_path / "__pycache__").is_dir()
    assert mutation.purge_bytecode(tmp_path) == 1
    assert not (tmp_path / "__pycache__").exists()


def test_every_declared_equivalent_mutant_names_a_proof_that_exists() -> None:
    """An EQUIVALENT verdict is an exemption, so it expires with its proof.

    The registry names a test node id. If that test is renamed or deleted, the
    justification is gone and the mutant goes back to being ordinary debt.
    """
    assert mutation.EQUIVALENT_MUTANTS, "the registry is empty, so this checks nothing"
    for key, (reason, node) in mutation.EQUIVALENT_MUTANTS.items():
        rel, line_text, index, old, new = key
        assert (repo_root() / rel).is_file(), rel
        text = (repo_root() / rel).read_text()
        # The mutated form is accepted, and ONLY that one. Without it this very
        # assertion kills the mutant it exempts: the sweep has the pinned line
        # mutated while the suite runs, this goes red, and EQUIVALENT becomes an
        # unreachable state - which is exactly what the first corrected sweep
        # reported, `equivalent 0` with a proven-equivalent mutant in the corpus.
        mutated = mutation.equivalence_line_as_mutated(key)
        assert line_text in text or mutated in text, (
            f"{rel} contains neither {line_text!r} nor its declared mutant {mutated!r}, so the "
            "proof was about a line that has moved on. Re-prove it or delete the entry."
        )
        assert len(reason) > 80, f"{key} is exempted without an argument"
        path, _, name = node.partition("::")
        assert name and name in (repo_root() / path).read_text(), (
            f"{node} does not exist, so nothing proves {key} is equivalent"
        )
        assert index >= 0 and old and new


def test_the_equivalence_key_is_content_addressed_and_not_a_line_number() -> None:
    """A key that moved with the file would go stale on any edit above it."""
    source = "def f(a, b):\n    return max(a, 0) + min(b, 0)\n"
    sites = mutation.mutable_sites(source)
    zeros = [s for s in sites if s.old == "0"]
    assert len(zeros) == 2, sites
    first = mutation.equivalence_key("m.py", source, zeros[0])
    second = mutation.equivalence_key("m.py", source, zeros[1])
    assert first != second, "the two `0`s on one line share a key, so an exemption would cover both"
    padded = "# a new line above\n" + source
    moved = [s for s in mutation.mutable_sites(padded) if s.old == "0"]
    assert mutation.equivalence_key("m.py", padded, moved[0]) == first, (
        "inserting a line above the site changed its key, so every entry would go stale on an "
        "unrelated edit"
    )


def test_the_gates_verdict_is_the_right_way_round_in_all_four_states() -> None:
    """The decision itself, in a millisecond rather than through a 40-minute sweep.

    A gate whose comparison is only reachable by running the slow path is a gate
    nobody ever checks the direction of.
    """
    budget = mutation.SURVIVOR_BUDGET
    assert mutation.budget_verdict(budget, 0, 1)[0] == 0
    assert mutation.budget_verdict(budget + 1, 0, 1)[0] == 1
    assert mutation.budget_verdict(budget - 1, 0, 1)[0] == 1
    assert "never rises" in mutation.budget_verdict(budget + 1, 0, 0)[1]
    assert "Lower SURVIVOR_BUDGET" in mutation.budget_verdict(budget - 1, 0, 0)[1]

    # An unexecuted mutant outranks everything else: the sweep did not measure.
    code, message = mutation.budget_verdict(budget, 1, 0)
    assert code == 2 and "never executed" in message, message
    assert mutation.budget_verdict(budget - 5, 3, 0)[0] == 2, (
        "an unmeasured mutant was allowed to be reported as a debt reduction, which is the "
        "one way a broken sweep looks like progress"
    )


def test_a_mutant_that_does_not_parse_is_not_offered_as_a_mutant() -> None:
    """The fake kill, which is the false survivor's mirror and was in the first baseline.

    ``*`` is mutated to ``/`` by token, and the token rules cannot see grammar, so
    the unpack in ``run(["git", *args])`` becomes ``[..., /args]``. That file does
    not compile, every test fails on a collection error, and the mutant reads
    KILLED - crediting a test with noticing something it never saw. Three of the
    126 baseline mutants were this, all three counted as kills.
    """
    unpack = 'def f(*args):\n    return run(["git", *args])\n'
    assert not [s for s in mutation.mutable_sites(unpack) if s.old == "*"], (
        "an unpack `*` is still offered as a mutation site, so `/args` would be run "
        "and its collection error counted as a test doing its job"
    )

    # The control: an ordinary multiplication on the same corpus is still offered,
    # so the filter is removing what does not compile and not the operator itself.
    product = "def f(a, b):\n    return a * b\n"
    stars = [s for s in mutation.mutable_sites(product) if s.old == "*"]
    assert len(stars) == 1 and stars[0].new == "/", stars
    assert mutation.apply_site(product, stars[0]) == "def f(a, b):\n    return a / b\n"

    for source in (unpack, product, "x = 1\n", "y = not True\n"):
        for site in mutation.mutable_sites(source):
            ast.parse(mutation.apply_site(source, site))  # raises if the filter leaked


def test_the_three_fake_kills_the_first_baseline_contained_are_gone_from_the_real_corpus() -> None:
    """Named sites, on the live tree, because the general rule was found through them."""
    for rel, line in (
        ("src/wildfire_nowcast/common/runs.py", 59),
        ("src/wildfire_nowcast/common/runs.py", 80),
        ("src/wildfire_nowcast/common/seeds.py", 61),
    ):
        source = (repo_root() / rel).read_text()
        assert "*" in source.splitlines()[line - 1], f"{rel}:{line} no longer holds a star"
        offered = [s for s in mutation.mutable_sites(source) if s.line == line and s.old == "*"]
        assert not offered, f"{rel}:{line} is offered again and would be a fake kill: {offered}"


def test_a_probe_that_cannot_run_is_unmeasured_and_does_not_abort_the_sweep() -> None:
    """One unmeasurable mutant is one unmeasured verdict, not the loss of the sweep.

    The first version raised out of ``_read_back`` with ``check=True`` and killed a
    42-minute run in its 39th minute, discarding every measurement taken beside it.
    """
    assert issubclass(mutation.ProbeFailed, Exception)
    source = inspect.getsource(mutation._read_back)
    assert "check=False" in source, "a probe failure aborts the sweep again"

    sweep = mutation.Sweep(
        results=[
            mutation.Result("m.py", 0.2, "PROBE_FAILED", "d", 1),
            mutation.Result("m.py", 0.5, "SURVIVED", "d", 1),
            mutation.Result("m.py", 0.8, "KILLED", "d", 1),
        ]
    )
    assert [r.verdict for r in sweep.unmeasured] == ["PROBE_FAILED"]
    assert len(sweep.survivors) == 1 and len(sweep.killed) == 1
    assert mutation.budget_verdict(len(sweep.survivors), len(sweep.unmeasured), 0)[0] == 2, (
        "a sweep carrying an unmeasured mutant reported a verdict on the budget"
    )


def test_the_registry_tolerates_its_own_mutant_and_nothing_else() -> None:
    """The tolerance above is one line-form wide, not a hole in the check.

    A sweep applying the DECLARED mutation is the sweep working; any other edit to
    the pinned line is the proof going stale and must still break the key.

    NO FILE IS READ HERE, AND THAT IS THE POINT. The first version of this test
    asserted the pristine line was on disk, and the very next sweep killed the
    equivalent mutant with it - I had reintroduced the defect I was fixing, one
    function down. **A test that asserts on the text of a swept source file
    manufactures a kill for every mutant of that file**, because the sweep has
    that text mutated while the suite runs. So this is a property of two strings.
    """
    key = next(iter(mutation.EQUIVALENT_MUTANTS))
    rel, line_text, index, old, new = key
    mutated = mutation.equivalence_line_as_mutated(key)
    assert mutated != line_text and old in line_text and new in mutated
    assert len(mutation.mutable_sites(mutated)) == len(mutation.mutable_sites(line_text)), (
        "the declared mutation changed how many sites the line has, so the index in the key "
        "no longer means the same thing on the two forms"
    )
    assert rel.endswith(".py")

    # A DIFFERENT mutation of the same line is not tolerated: the key stops
    # resolving, which is what makes the entry expire rather than accumulate.
    for wrong in (index + 1, index - 1):
        if 0 <= wrong < len(mutation.mutable_sites(line_text)):
            with pytest.raises(ValueError, match="not"):
                mutation.equivalence_line_as_mutated((rel, line_text, wrong, old, new))
    with pytest.raises(ValueError, match="does not exist"):
        mutation.equivalence_line_as_mutated((rel, line_text, 99, old, new))


# --------------------------------------------------------------------------
# Plan Task 5.7 - A NAME-BASED ROUTE FROM A MODULE TO ITS TEST
#
# 29 test files against 119 source modules, and until this section existed only
# 9 modules had a same-named test. The consequence is not a coverage number: it
# is that a reader who wants to know what is asserted about `common/grid.py` has
# no way to find out except to grep the whole suite, and a lead adding a module
# has no place the next reader will look. The mutation survivors cluster in the
# packages with no such route, which is the argument for building one.
#
# `common/` is the worked example and the only package under this rule today.
# `model/`, `eval/` and `sim/` belong to other leads and are being edited in
# parallel; the convention they can adopt is written down in tests/README.md
# rather than imposed from here.
# --------------------------------------------------------------------------

#: The source package this rule covers, and the directory its tests mirror.
_MIRRORED_PACKAGE = "common"
_MIRROR_ROOT = "tests/unit/common"

#: Modules under `common/` with no mirroring unit test yet. A BURN-DOWN LIST in
#: the same two directions as the mypy exemptions and the survivor budget: a
#: module that is NOT listed and has no mirror turns this red, and a module that
#: IS listed and has gained one turns it red too. So an entry cannot outlive its
#: reason, and the list can only shrink without a visible edit.
#:
#: `__init__.py` files are excluded by rule rather than by entry: they re-export
#: and hold no behaviour, so a mirroring file for one would assert nothing.
_COMMON_MODULES_WITHOUT_A_MIRRORING_TEST: frozenset[str] = frozenset(
    {
        "components.py",
        "contract.py",
        "dispersion.py",
        "environment.py",
        "iou_terms.py",
        "null_check/registry.py",
        "playthrough.py",
        "seeds.py",
        "separation.py",
        "splits.py",
        "synthetic.py",
    }
)


def _common_modules() -> list[str]:
    """Every tracked module under ``common/``, package-relative, minus ``__init__``."""
    package = SRC / _MIRRORED_PACKAGE
    return sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    )


def mirror_test_path(module_relative: str) -> Path:
    """``null_check/windows.py`` -> ``tests/unit/common/null_check/test_windows.py``.

    Mechanical, with no special cases: the test file is ``test_`` prefixed onto
    the module's own file name, in a directory that mirrors the module's own
    directory. ``__main__.py`` therefore maps to ``test___main__.py``, which is
    ugly and is kept anyway, because a convention with an exception in it is a
    convention nobody can check.
    """
    source = Path(module_relative)
    return repo_root() / _MIRROR_ROOT / source.parent / f"test_{source.name}"


def test_every_common_module_either_has_a_mirroring_test_or_is_named_as_debt() -> None:
    """The route exists, or the absence of it is written down. Not neither."""
    modules = _common_modules()
    assert len(modules) > 15, f"only {len(modules)} modules found; the walk is wrong"

    missing_and_unlisted: list[str] = []
    listed_but_present: list[str] = []
    for module in modules:
        has_mirror = mirror_test_path(module).is_file()
        listed = module in _COMMON_MODULES_WITHOUT_A_MIRRORING_TEST
        if not has_mirror and not listed:
            missing_and_unlisted.append(module)
        if has_mirror and listed:
            listed_but_present.append(module)

    assert not missing_and_unlisted, (
        f"{missing_and_unlisted} have no {_MIRROR_ROOT}/... mirror and are not on the "
        "burn-down list. Add the test, or add the module to "
        "_COMMON_MODULES_WITHOUT_A_MIRRORING_TEST with the rest of the debt."
    )
    assert not listed_but_present, (
        f"{listed_but_present} now have a mirroring test and are still listed as debt. "
        "Remove them from _COMMON_MODULES_WITHOUT_A_MIRRORING_TEST: a burn-down entry that "
        "outlives its reason is an allow-list."
    )


def test_the_burn_down_list_names_only_modules_that_exist() -> None:
    """A stale entry would silently forgive a module that was renamed away."""
    modules = set(_common_modules())
    stale = sorted(_COMMON_MODULES_WITHOUT_A_MIRRORING_TEST - modules)
    assert not stale, f"{stale} are on the burn-down list and are not modules under common/"


def test_the_mirror_is_a_package_so_a_basename_can_repeat() -> None:
    """``tests/unit/common/test_states.py`` and ``tests/test_states.py`` coexist.

    Under pytest's default ``prepend`` import mode two test files with the same
    basename and no ``__init__.py`` between them collide at import and the whole
    suite errors. The ``__init__.py`` files are therefore load-bearing, not
    decoration, and this is what says so.
    """
    root = repo_root() / _MIRROR_ROOT
    assert root.is_dir()
    for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        if directory.name == "__pycache__":
            continue
        assert (directory / "__init__.py").is_file(), f"{directory} is not a package"
    assert (root.parent / "__init__.py").is_file(), "tests/unit is not a package"

    repeated = [
        path.name
        for path in root.rglob("test_*.py")
        if (repo_root() / "tests" / path.name).is_file()
    ]
    assert repeated, (
        "no mirrored file shares a basename with a top-level test file, so the packaging "
        "above is not currently being exercised by anything"
    )


def test_the_convention_is_written_down_where_another_lead_would_look() -> None:
    """A convention that lives only in a test is a convention only its author knows."""
    readme = repo_root() / "tests" / "README.md"
    assert readme.is_file(), "tests/README.md is missing"
    text = readme.read_text(encoding="utf-8")
    for expected in (_MIRROR_ROOT, "tests/contract/", "burn-down"):
        assert expected in text, f"tests/README.md does not mention {expected!r}"


def test_a_kill_carries_the_node_that_caught_it_including_an_ERROR_node() -> None:
    """Two of the I12 sweep's 58 kills named no test, and both were attributable.

    ``common/playthrough.py:240`` is killed by a COLLECTION ERROR in
    ``tests/test_playthrough_asymmetric_gate.py``, and
    ``common/null_check/forecasters.py:82`` by a fixture error in
    ``tests/test_null_check.py::test_best_member_iou_is_flagged_in_the_growth_band``,
    which raises the C1.3 phase guard ``truth covers 3 steps but samples cover 40``.
    Both print an ``ERROR`` line rather than a ``FAILED`` one. The deselect path
    already read both; the reporting path grepped for ``FAILED`` alone and threw
    the attribution away, so the sweep held the answer and printed nothing.
    """
    source = inspect.getsource(mutation.run_one)
    assert "failing_tests(output)" in source, (
        "run_one is back to grepping for FAILED alone, so a kill by a collection error or a "
        "fixture error is recorded with no node again"
    )
    assert 'startswith("FAILED ")' not in source

    error_only = (
        "tests/test_x.py E                              [100%]\n"
        "ERROR tests/test_x.py - ValueError: truth covers 3 steps but samples cover 40\n"
        "1 error in 1.24s\n"
    )
    assert mutation.failing_tests(error_only) == ["tests/test_x.py"], (
        "an ERROR-only run still reports no node, which is the case that produced the two "
        "unattributable kills"
    )


# --------------------------------------------------------------------------
# isolation: a suite run that a neighbour's plant cannot enter
# --------------------------------------------------------------------------


def test_the_isolated_runner_is_a_make_target_that_reaches_the_script() -> None:
    """Isolation has to be a command. It cannot be a thing people remember."""
    makefile = (repo_root() / "Makefile").read_text()
    body = re.search(r"^test-isolated:.*\n((?:\t.*\n)+)", makefile, re.M)
    assert body and "tools/isolated_suite.py" in body.group(1), body
    assert (repo_root() / "tools" / "isolated_suite.py").is_file()


def test_the_isolated_runner_carries_nothing_from_the_working_tree() -> None:
    """`pristine=True` is the whole point, so it is asserted rather than assumed.

    Carrying the working tree would overlay exactly the neighbour edit the run
    exists to exclude, and the result would look identical to a correct one.
    """
    source = inspect.getsource(isolated_suite.isolate)
    assert "pristine=True" in source, (
        "the isolated runner would overlay the shared working tree, which is the one "
        "thing it must not do"
    )


def test_the_isolation_check_REFUSES_when_the_import_escapes_the_workspace(
    tmp_path: Path,
) -> None:
    """The negative control, and it is the test that makes the rest mean anything.

    The editable install puts the real `src/` on `sys.path`, so a workspace whose
    own `src/` is missing silently imports the shared tree. Pointed at an empty
    directory, the check must refuse rather than proceed - otherwise a green
    isolated run could be a green shared run wearing its name.
    """
    python = repo_root() / ".venv" / "bin" / "python"
    if not python.exists():  # pragma: no cover - provisioned environments only
        pytest.skip("no .venv in this tree")
    with pytest.raises(SystemExit, match="REFUSING TO RUN"):
        isolated_suite.assert_self_contained(tmp_path, python)


def test_the_isolated_runner_returns_pytests_own_exit_code() -> None:
    """A wrapper that reports its own success is a wrapper that hides failures."""
    source = inspect.getsource(isolated_suite.main)
    assert "return proc.returncode" in source, (
        "the isolated runner no longer passes pytest's exit code through"
    )
    assert "check=False" in source, "a non-zero pytest exit would raise instead of reporting"


def test_the_sweep_records_the_sha_it_actually_measured() -> None:
    """A pinned number that cannot name its commit is not attributable.

    This session had to infer which commit a budget of 21 was measured at by
    comparing a process start time against commit timestamps five seconds apart.
    The sha is read out of the WORKSPACE, not the repo, because the repo can move
    during a 45-minute run and the workspace cannot.
    """
    assert "head" in mutation.Sweep().to_dict()
    # The module, not one named function: this assertion broke once already when the
    # code moved between `sweep` and its helper, which is a test tracking a location
    # rather than a behaviour. What must be true is that the sha comes from a
    # WORKSPACE path and never from the repo.
    source = inspect.getsource(mutation)
    assert 'str(spaces[0]), "rev-parse", "HEAD"' in source, (
        "the swept sha is no longer read from the workspace, so a sweep that outlives a "
        "commit would report the wrong one"
    )
    assert 'str(repo), "rev-parse", "HEAD"' not in source, (
        "the sha is being read from the repo, which moves during a 45-minute sweep"
    )
    printer = inspect.getsource(mutation.main)
    assert "result.head" in printer, "the sha is recorded but never shown"


def test_the_sweep_removes_its_worktrees_even_when_it_raises() -> None:
    """Cleanup on the failure path, because that is the path that repeats.

    The removal used to be reachable only on success, so a sweep that raised left
    three full worktrees behind. This session filled a disk exactly that way: a
    failure leaks copies, the next sweep fails for want of space and leaks three
    more. The failure mode compounds, which is why it is the one worth testing.

    THE BOUNDARY IS THE POINT AND THE FIRST FIX GOT IT WRONG. Wrapping only the
    measurement still leaked, because the workspaces are BUILT before it and the
    self-containment control raises there. So this asserts that `build_workspace`
    is inside the guarded region, not merely that a `finally` exists somewhere.
    """
    source = inspect.getsource(mutation.sweep)
    assert "try:" in source and "finally:" in source, "the sweep has no cleanup guard"
    guarded = source[source.index("try:") : source.index("finally:")]
    assert "build_workspace" in guarded, (
        "the workspaces are built OUTSIDE the try, so a failure while building them - "
        "which is exactly where the self-containment control raises - leaks a worktree"
    )
    assert "assert_workspace_is_self_contained" in guarded, (
        "the self-containment control is outside the guarded region"
    )
    remover = inspect.getsource(mutation._remove_worktrees)
    assert '"worktree", "remove", "--force"' in remover, (
        "the cleanup helper no longer removes worktrees"
    )
    assert "_remove_worktrees" in source, "the finally no longer reaches the cleanup helper"


# --------------------------------------------------------------------------
# what a gate LEAVES BEHIND, multiplied by how often it runs
# --------------------------------------------------------------------------


def test_temp_entries_actually_notices_a_new_entry(tmp_path: Path) -> None:
    """The leak detector, exercised rather than asserted about.

    A before/after set difference that never sees anything is indistinguishable
    from a tool that does not leak, which is the whole failure mode here.
    """
    with mock.patch.object(mutation.tempfile, "gettempdir", return_value=str(tmp_path)):
        before = mutation.temp_entries()
        assert before == set()
        (tmp_path / "wnc-selftest-abc").mkdir()
        after = mutation.temp_entries()
    assert after - before == {"wnc-selftest-abc"}


def test_a_sweep_that_leaks_cannot_report_success() -> None:
    """A leak is a failure of the run, not a footnote under it.

    One 4.7 MB temp directory per suite run is nothing. This gate runs the suite
    once per mutant, so the same directory became 70 GB and 15,462 entries in a
    day. Cost per invocation is not the number that matters; cost times
    invocations is - so the sweep measures its own residue and exits non-zero.
    """
    assert mutation.Sweep(leaked=["x"]).to_dict()["leaked_temp_entries"] == ["x"]
    source = inspect.getsource(mutation.main)
    assert "if result.leaked:\n        return 3" in source, (
        "a leaking sweep can exit 0 again, so the residue is advisory rather than a gate"
    )
    body = inspect.getsource(mutation.sweep)
    guarded = body[body.index("finally:") :]
    assert "temp_entries() - before" in guarded, (
        "the residue is not measured on the failure path, which is the path that repeats"
    )
    assert "shutil.rmtree(root" in guarded, "the sweep no longer removes its own workspace root"


def test_the_sweep_can_be_measured_before_it_is_run_in_full() -> None:
    """`--max-mutants` exists so the gate's cost is knowable in minutes, not hours."""
    source = inspect.getsource(mutation.sweep)
    assert "jobs = jobs[:max_mutants]" in source, "the cap no longer truncates the job list"
    parser_source = inspect.getsource(mutation.main)
    assert '"--max-mutants"' in parser_source


def test_the_isolated_runner_removes_its_own_temp_root() -> None:
    """It used to leave one empty directory per invocation.

    Individually trivial, which is exactly how the large leak accumulated: nobody
    refuses a cost of one directory, and nobody multiplies it by the run count.
    """
    source = inspect.getsource(isolated_suite.main)
    guarded = source[source.index("finally:") :]
    assert "shutil.rmtree(root" in guarded, (
        "the isolated runner leaves its mkdtemp root behind on every run"
    )


def test_a_child_launched_by_the_sweep_shares_the_parents_temp_directory() -> None:
    """The two line test that would have caught a detector blind twice over.

    ``_child_env`` is built from nothing rather than copied, so anything not
    listed is gone from the child. TMPDIR was not listed, so children fell back to
    ``/tmp`` while the leak detector compared before and after in the PARENT's
    temp directory. The difference was taken over a location the suite never wrote
    to, and it reported an empty leak list whatever the children left behind.
    """
    python = repo_root() / ".venv" / "bin" / "python"
    if not python.exists():  # pragma: no cover - provisioned environments only
        pytest.skip("no .venv in this tree")
    env = mutation._child_env(repo_root(), python)
    assert "TMPDIR" in env, "the child temp directory is unpinned again"
    child = subprocess.run(
        [str(python), "-c", "import tempfile; print(tempfile.gettempdir())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert child == mutation.tempfile.gettempdir(), (
        f"the sweep's children write to {child} while the detector watches "
        f"{mutation.tempfile.gettempdir()}"
    )


def test_the_leak_detector_FIRES_on_a_child_that_deliberately_leaks(tmp_path: Path) -> None:
    """The standing positive control. A detector never seen to fire is worth nothing.

    This one was green twice while blind, so the mechanism is exercised end to
    end: a private temp directory for the whole process tree, a child launched
    through the sweep's own ``_child_env`` that leaks on purpose, and the parent's
    before/after difference required to name it. The negative half runs on the
    same rig, because a difference that reports nothing for a clean child and
    nothing for a leaking one is not a detector.
    """
    python = repo_root() / ".venv" / "bin" / "python"
    if not python.exists():  # pragma: no cover - provisioned environments only
        pytest.skip("no .venv in this tree")

    def run(code: str) -> list[str]:
        with mock.patch.object(mutation.tempfile, "gettempdir", return_value=str(tmp_path)):
            env = mutation._child_env(repo_root(), python)
            before = mutation.temp_entries()
            subprocess.run([str(python), "-c", code], env=env, check=True, capture_output=True)
            return sorted(mutation.temp_entries() - before)

    leaked = run("import tempfile; tempfile.mkdtemp(prefix='deliberate-leak-')")
    assert leaked and leaked[0].startswith("deliberate-leak-"), (
        "a child that deliberately leaked was not seen by the detector, so the detector "
        f"is watching the wrong directory again: {leaked}"
    )

    clean = run("import tempfile;  tempfile.TemporaryDirectory().cleanup()")
    assert clean == [], f"a child that leaked nothing was reported as leaking: {clean}"


def test_the_detectors_ignore_list_stays_narrow(tmp_path: Path) -> None:
    """An exclusion is a hole. This one is one prefix wide and must stay that way.

    ``pytest-of-<user>`` is pytest's own tmp_path base: created once, garbage
    collected to the last three runs, bounded. Everything else that appears in the
    temp directory during a sweep is residue. The unplanted control fired on that
    single name, which is why the exclusion exists at all, and widening it is how a
    detector quietly stops detecting.
    """
    assert mutation.IGNORED_TEMP_PREFIXES == ("pytest-of-",), (
        "the ignore list grew. Every added prefix is a class of leak this gate can no "
        "longer see, and it was added by someone who found it inconvenient."
    )
    with mock.patch.object(mutation.tempfile, "gettempdir", return_value=str(tmp_path)):
        (tmp_path / "pytest-of-someone").mkdir()
        assert mutation.temp_entries() == set(), "the pytest base is not being excluded"
        (tmp_path / "wnc-selftest-abcd").mkdir()
        (tmp_path / "deliberate-leak-abcd").mkdir()
        assert mutation.temp_entries() == {"wnc-selftest-abcd", "deliberate-leak-abcd"}
