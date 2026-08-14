"""C7 and C0 hygiene rules that were true only by discipline until A10.

Both clauses were ratified and satisfied on disk, and neither was ENFORCED —
the same shape as C1.5 (ratified v2.3, implemented v2.5, all-NaN passing 56
checks in between). They pass today, which is exactly when to wire them: a
hygiene rule is cheap to keep and expensive to restore.

* C7 — *no hardcoded paths in src/*, *no hardcoded GCP project id*, and the
  house rule that *notebooks are never imported by src*.
* C0 — anything the contract adjudicates has ONE implementation, in ``common/``.
  The retired ``data/`` duplicates must stay deleted, and the second copy of the
  split fingerprint must agree with the first.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from wildfire_nowcast.common.paths import repo_root

SRC = repo_root() / "src" / "wildfire_nowcast"


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_there_is_source_to_check() -> None:
    """A scan over an empty file list passes vacuously — check the scan first."""
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

    Checked structurally rather than by matching the id itself — writing the
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
    computing geometry through different code — a tensor that passes its check
    and is still wrong.
    """
    duplicate = SRC / "data" / name
    assert not duplicate.exists(), (
        f"{duplicate} is back. C0: the single implementation lives in common/ and data/ imports "
        "it."
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
#: ``"""Executable form of INTERFACES.md contracts ... — **v2.5**."""``
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
#: all of which are CORRECT — they record when a clause was ratified, which does
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
    module implements — a claim about the present that nothing ever updates:

        ``Executable form of INTERFACES.md contracts C1, C2 and C3 — **v2.5**.``
        ``Reading and writing C1/C2/C3 artifacts — **INTERFACES v2.5**.``
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
    """The scan's own positive control — a pattern that matches nothing passes.

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

    The requirement is not merely "it currently agrees" — it is that a failure to
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
# them reported clean while being wrong, so the deliverable is not a sweep — it is
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
#: This module is scanned by its own check, with NO self-exemption — see
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
#: and are to be RETIRED — file by file, to zero — once that work lands. The check
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
    """PATTERN positive control — the half this project has got wrong four times.

    Three of the four failed sweeps reported clean from a scan that matched
    nothing. This asserts the compiled pattern still matches realistic sentences,
    and — the harder half — that it still does NOT match the citations and the
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
    # to become `>= 0` on the day the list is retired — a control that quietly
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

    # 2. An allowlisted count going UP — the fence must not absorb new debt.
    bumped = dict(live)
    bumped["src/wildfire_nowcast/model/train.py"] = FENCED_BURN_DOWN[
        "src/wildfire_nowcast/model/train.py"
    ] + 1
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
    because the tempting fix — skipping this path — would make the one file most
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
