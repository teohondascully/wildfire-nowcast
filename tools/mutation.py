"""Mutation sweep over ``common/`` and ``eval/``, with a survivor budget that only falls.

A green suite says the tests ran. It does not say they would have noticed. The
external audit measured that directly: a sweep over the best-covered 22 modules
killed 40 of 66 single-site mutants and left **26 survivors**, including a
4-connectivity table that could be made asymmetric, a dilation slice that could be
moved by one, an inverted ``not`` inside C-4.2's own clause, and a ``>`` that could
become ``>=`` so that ties count as advantages. Every one of those left 745 tests
passing.

This is that measurement made repeatable and turned into a gate. The budget starts
at the measured survivor count and may only decrease, which is the same
two-directional burn-down the mypy exemption list uses: it fails when the count
RISES, and it fails when the count FALLS without the pin being lowered in the same
commit, so a pin cannot quietly become an over-estimate that forgives a new gap.

**It runs in a git worktree, never in the working tree.** Two of this project's
recorded process failures were a lead editing a file another lead was running
against (C-4 breaches, ADR-052 (5), ADR-053). A mutation sweep edits `eval/` by
construction, so running it in place would breach that fence a hundred times per
invocation. The workspace is built from ``HEAD``, then every tracked file that
differs in the working tree is copied over it and the number of carried files is
PRINTED, so the sweep measures the tree the developer is looking at and says how
it got there.

**Five controls, because a sweep that reports zero survivors is exactly what a
broken sweep reports.**

1. The workspace must import ITS OWN copy of ``wildfire_nowcast``. The editable
   install points ``sys.path`` at the real ``src/``, so without this the mutants
   would never be loaded and all of them would read SURVIVED.
2. Every mutant is read back through the interpreter that is about to run the
   tests, from ``module.__file__``, and the run is abandoned if the mutated token
   is not there. A mutation that failed to apply is not a survivor.
3. THE BYTECODE IS CHECKED, NOT ONLY THE SOURCE. CPython invalidates a ``.pyc``
   on ``(source mtime in WHOLE SECONDS, source size)``, so a same-length edit made
   and reverted inside one second is invisible and the stale bytecode runs
   instead: the file verifiably says ``max`` and the program returns the ``min``
   answer. Reproduced here before this guard was written. Every run purges
   ``__pycache__``, sets ``PYTHONDONTWRITEBYTECODE``, and compares
   ``marshal.dumps`` of the code object the LOADER would hand the interpreter
   against a fresh ``compile`` of the file on disk. A mismatch is
   ``STALE_BYTECODE``: a refusal to measure, which is a different claim from
   "nothing caught it" and is never counted as one.
4. A MUTANT THAT DOES NOT PARSE IS NOT A MUTANT. The token rules are blind to
   grammar, so ``*`` becomes ``/`` inside ``run(["git", *args])`` too, and the
   result does not compile. Every test then fails on a collection error and the
   mutant reads KILLED: a test is credited with noticing something it never saw,
   which is the exact mirror of the false survivor. The first baseline contained
   three. Such sites are dropped at enumeration.
5. The unmutated workspace must exit 0 first. Any test that fails there is an
   environment artifact of the sandbox (no ``.venv`` of its own, no installed
   hooks), is deselected for the sweep, and is NAMED in the output rather than
   quietly dropped, because a suite that is already red kills every mutant.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

#: The packages under sweep. ``common/`` is infra's and ``eval/`` is the scoring
#: code every published number came out of; the plan names exactly these two.
TARGET_PACKAGES: Final = ("common", "eval")

#: Where in each module's site list the mutants are taken. Inherited unchanged
#: from the audit sweep so that this gate's numbers are comparable with the 26 of
#: 66 already on the record, rather than being a fresh number nothing can be read
#: against.
SAMPLE_FRACTIONS: Final = (0.2, 0.5, 0.8)

#: Single-token substitutions. A boundary becomes its neighbour, an inequality
#: flips, a conjunction weakens: the defects that survive review and change a
#: verdict, not the ones that raise on the first call.
OPERATOR_MUTATIONS: Final = {
    ">=": ">",
    "<=": "<",
    ">": ">=",
    "<": "<=",
    "==": "!=",
    "!=": "==",
    "+": "-",
    "-": "+",
    "*": "/",
}
KEYWORD_MUTATIONS: Final = {
    "and": "or",
    "or": "and",
    "True": "False",
    "False": "True",
    "not": "",
}

#: A NOT-KILLED MUTANT IS ONE OF THREE THINGS, AND LUMPING THEM MANUFACTURES THE
#: DEFECT THIS GATE EXISTS TO PREVENT.
#:
#: * ``SURVIVED`` - real debt. A test could kill it and none does. This is the
#:   only state the budget counts.
#: * ``EQUIVALENT`` - provably unkillable. No input distinguishes the mutant from
#:   the original, so demanding a test for it is demanding a test that asserts
#:   something false. Declared below, WITH the proof, and the declaration fails if
#:   the proof stops existing.
#: * ``STALE_BYTECODE`` / ``NOT_APPLIED`` - never executed. A refusal to measure,
#:   which is not the same claim as "nothing caught it" and must never be counted
#:   as one.
#:
#: A budget that lumped the second into the first would push a lead to write an
#: untestable test to reach zero. A budget that lumped the third into the first
#: would count a measurement that did not happen.
#:
#: Keyed by ``(module, stripped source line, index of the site on that line,
#: old, new)``. NOT by line number, which moves under any edit above it, and not
#: by file name, which would make this an allow-list. Reformat or edit the line
#: and the entry stops matching, which is correct: the proof was about that line.
EQUIVALENT_MUTANTS: Final = {
    (
        "src/wildfire_nowcast/common/states.py",
        "ys_dst = slice(max(dy, 0), h + min(dy, 0))",
        2,
        "0",
        "1",
    ): (
        "dy takes only -1, 0 and 1, so `h + min(dy, 1)` differs from `h + min(dy, 0)` at "
        "dy == 1 alone, where the end index h+1 CLIPS to h on a length-h axis. The two forms "
        "therefore agree on every input the offset table can produce.",
        "tests/test_states_geometry.py::"
        "test_the_dilation_slice_survivor_is_an_EQUIVALENT_MUTANT_and_this_is_the_proof",
    ),
}

#: THE BUDGET, and it counts ``SURVIVED`` ALONE. Measured, not chosen; see
#: ``MEASURED_AT`` for the command. It may be lowered by a commit that kills a
#: survivor; it may not be raised, and the gate fails in BOTH directions so that a
#: stale over-estimate is as loud as a regression.
SURVIVOR_BUDGET: Final = 62

#: How the number above was obtained, so that a reader can reproduce it rather
#: than believe it.
MEASURED_AT: Final = (
    "126 mutants (42 modules x 3 fractions) over common/ and eval/, by "
    "`python tools/mutation.py --pristine --workers 3`, suite `pytest -x -m 'not slow'`. "
    "Baseline at cc82876 before this gate existed: 126 mutants, 55 killed, 62 survived, "
    "0 unmeasured, no equivalence declared."
)

_PYTEST_ARGS: Final = ("-x", "-q", "-m", "not slow", "-p", "no:randomly", "-p", "no:cacheprovider")

#: Directories whose untracked files are carried into the workspace. A new test
#: file is the usual reason a sweep is being re-run, and a sweep that could not
#: see it would report the old number with total confidence.
_CARRIED_UNTRACKED: Final = ("src/", "tests/", "tools/", "configs/")


@dataclass(frozen=True)
class Site:
    """One mutable token: where it is, what it says, what it would say."""

    line: int
    col_start: int
    col_end: int
    old: str
    new: str


@dataclass
class Result:
    """The verdict on one mutant."""

    module: str
    fraction: float
    verdict: str
    descriptor: str
    n_sites: int
    exit_code: int | None = None
    seconds: float = 0.0
    detail: str = ""


@dataclass
class Sweep:
    """Everything one invocation measured."""

    results: list[Result] = field(default_factory=list)
    deselected: list[str] = field(default_factory=list)
    carried: int = 0
    seconds: float = 0.0

    @property
    def survivors(self) -> list[Result]:
        return [r for r in self.results if r.verdict == "SURVIVED"]

    @property
    def killed(self) -> list[Result]:
        return [r for r in self.results if r.verdict == "KILLED"]

    @property
    def equivalent(self) -> list[Result]:
        return [r for r in self.results if r.verdict == "EQUIVALENT"]

    @property
    def unmeasured(self) -> list[Result]:
        """Mutants that never executed. NOT survivors, and not quietly dropped."""
        return [
            r
            for r in self.results
            if r.verdict in ("STALE_BYTECODE", "NOT_APPLIED", "PROBE_FAILED")
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": SURVIVOR_BUDGET,
            "measured_at": MEASURED_AT,
            "n_mutants": len(self.results),
            "n_killed": len(self.killed),
            "n_survived": len(self.survivors),
            "n_equivalent": len(self.equivalent),
            "n_unmeasured": len(self.unmeasured),
            "deselected": self.deselected,
            "carried_working_tree_files": self.carried,
            "seconds": round(self.seconds, 1),
            "results": [asdict(r) for r in self.results],
        }


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------


def apply_site(source: str, site: Site) -> str:
    """``source`` with one token replaced. The only place a mutant is constructed."""
    lines = source.splitlines(keepends=True)
    line = lines[site.line - 1]
    lines[site.line - 1] = line[: site.col_start] + site.new + line[site.col_end :]
    return "".join(lines)


def mutable_sites(source: str) -> list[Site]:
    """Every single-token mutation site in ``source`` THAT STILL PARSES, in file order.

    Token-level rather than AST-level on purpose: the mutant must be byte-identical
    to the original everywhere else, and an AST round-trip reformats the file, which
    would make the diff unreadable and would move the code fingerprint for reasons
    that have nothing to do with the mutation.

    THE PARSE FILTER IS NOT TIDINESS; IT REMOVES FAKE KILLS. The token rules are
    blind to grammar, so ``*`` is mutated to ``/`` wherever it appears - including
    the unpack in ``subprocess.run(["git", *args])``, which yields ``[..., /args]``
    and does not compile. Every test then fails on a collection error, the mutant
    reads KILLED, and the sweep credits a test that noticed nothing. That is the
    exact mirror of the false survivor this tool was written to prevent, and the
    first baseline had **three** of them (``common/runs.py`` twice,
    ``common/seeds.py`` once), all three counted as kills.

    Stated as a general rule rather than a special case for ``*args``: a mutation
    that cannot be compiled is not a mutation, because a suite cannot fail to
    notice a file the parser rejects. Costs one parse per site, about 24 s over the
    whole corpus, against a sweep measured in tens of minutes.
    """
    out: list[Site] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.start[0] != token.end[0]:
            continue
        new: str | None = None
        if token.type == tokenize.OP:
            new = OPERATOR_MUTATIONS.get(token.string)
        elif token.type == tokenize.NAME:
            new = KEYWORD_MUTATIONS.get(token.string)
        elif token.type == tokenize.NUMBER:
            new = _mutate_number(token.string)
        if new is None:
            continue
        site = Site(token.start[0], token.start[1], token.end[1], token.string, new)
        try:
            ast.parse(apply_site(source, site))
        except SyntaxError:
            continue
        out.append(site)
    return out


def _mutate_number(literal: str) -> str | None:
    """Move a numeric literal off its value: a boundary is where off-by-one lives."""
    try:
        value = float(literal)
    except ValueError:
        return None
    if value == 0:
        return "1"
    if literal.isdigit():
        return str(int(literal) + 1)
    return repr(round(value * 1.5 + 0.001, 6))


def select_site(sites: Sequence[Site], fraction: float) -> Site:
    """The site at ``fraction`` of the way through the list. Deterministic."""
    return sites[int(len(sites) * fraction) % len(sites)]


def equivalence_key(rel: str, source: str, site: Site) -> tuple[str, str, int, str, str]:
    """The content-addressed identity of one mutant, for :data:`EQUIVALENT_MUTANTS`."""
    line = source.splitlines()[site.line - 1]
    on_this_line = [s for s in mutable_sites(source) if s.line == site.line]
    return (rel, line.strip(), on_this_line.index(site), site.old, site.new)


def equivalence_note(rel: str, source: str, site: Site) -> tuple[str, str] | None:
    """``(reason, proving test node id)`` if this mutant is declared unkillable."""
    return EQUIVALENT_MUTANTS.get(equivalence_key(rel, source, site))


def target_modules(repo: Path) -> list[str]:
    """Tracked modules under the swept packages, as repo-relative paths."""
    args = [
        "git",
        "-C",
        str(repo),
        "ls-files",
        *[f"src/wildfire_nowcast/{p}" for p in TARGET_PACKAGES],
    ]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    return sorted(line for line in out.splitlines() if line.endswith(".py"))


# --------------------------------------------------------------------------
# workspace
# --------------------------------------------------------------------------


def build_workspace(repo: Path, workspace: Path, *, pristine: bool = False) -> int:
    """A git worktree at HEAD, overlaid with the working tree. Returns files carried.

    ``pristine`` carries nothing, so the sweep measures the COMMIT and the number it
    reports can be quoted against a sha. That is the mode a reported figure must use:
    three leads share this tree, and a working-tree overlay silently pulls another
    lead's half-written test file into a number attributed to this one. It happened
    while this tool was being written - six untracked test files appeared under
    ``tests/`` mid-task - which is why the flag exists rather than the convention.
    """
    if workspace.exists():
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(workspace)],
            capture_output=True,
            text=True,
            check=False,
        )
        shutil.rmtree(workspace, ignore_errors=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(workspace), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    purge_bytecode(workspace)
    if pristine:
        return 0
    changed = subprocess.run(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    untracked = [
        line
        for line in subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if line.startswith(_CARRIED_UNTRACKED)
    ]
    carried = 0
    for rel in sorted(set(changed) | set(untracked)):
        source, destination = repo / rel, workspace / rel
        if not source.exists():
            destination.unlink(missing_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        carried += 1
    purge_bytecode(workspace)
    return carried


def _child_env(workspace: Path, python: Path) -> dict[str, str]:
    """A minimal, explicit environment. The workspace's own ``src`` comes FIRST."""
    return {
        "PATH": f"{python.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(workspace / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "_ZO_DOCTOR": "0",
    }


def _run_pytest(workspace: Path, python: Path, extra: Sequence[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [str(python), "-m", "pytest", *_PYTEST_ARGS, *extra],
        cwd=workspace,
        env=_child_env(workspace, python),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def assert_workspace_is_self_contained(workspace: Path, python: Path) -> None:
    """Control 1: the workspace must import its OWN package, not the editable one."""
    proc = subprocess.run(
        [str(python), "-c", "import wildfire_nowcast; print(wildfire_nowcast.__file__)"],
        cwd=workspace,
        env=_child_env(workspace, python),
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = Path(proc.stdout.strip()).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise RuntimeError(
            f"the workspace imports {resolved}, which is OUTSIDE {workspace}. Every mutant "
            "would read SURVIVED because none of them would ever be loaded."
        )


def failing_tests(output: str) -> list[str]:
    """Node ids pytest reported as FAILED or ERROR, from its own summary lines."""
    out: list[str] = []
    for line in output.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            out.append(line.split(" ", 1)[1].split(" ")[0])
    return sorted(set(out))


def baseline(workspace: Path, python: Path) -> list[str]:
    """Control 3: run unmutated, deselect whatever the sandbox itself breaks."""
    code, output = _run_pytest(workspace, python, ())
    if code == 0:
        return []
    broken = failing_tests(output)
    if not broken:
        raise RuntimeError(
            f"unmutated workspace exited {code} and named no test:\n{output[-3000:]}"
        )
    deselect = [arg for node in broken for arg in ("--deselect", node)]
    code, output = _run_pytest(workspace, python, deselect)
    if code != 0:
        raise RuntimeError(
            f"unmutated workspace still exits {code} after deselecting {broken}. A sweep against "
            f"a red suite kills every mutant and means nothing.\n{output[-3000:]}"
        )
    return broken


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def module_name(rel: str) -> str:
    """``src/wildfire_nowcast/common/states.py`` -> ``wildfire_nowcast.common.states``."""
    return rel.removeprefix("src/").removesuffix(".py").replace("/", ".")


#: Control 2, in two halves. The first reads the mutated LINE back through the
#: interpreter that is about to run the tests. The second is the one that matters
#: and was added once the hazard was demonstrated: it compares the BYTECODE the
#: loader would hand the interpreter against a fresh compile of the file on disk.
#:
#: CPython invalidates a ``.pyc`` on ``(source mtime in WHOLE SECONDS, source
#: size)``. A same-length edit made and reverted inside one second is therefore
#: invisible and the stale bytecode runs instead, so a mutant whose source is
#: verifiably present can fail to execute and read as a SURVIVOR. Reading the
#: source proves nothing about that; ``marshal.dumps`` of the loaded code object
#: against a fresh ``compile`` proves it exactly.
_PROBE = """
import importlib.util as u, marshal, pathlib, sys
spec = u.find_spec({name!r})
origin = pathlib.Path(spec.origin)
loaded = spec.loader.get_code({name!r})
fresh = compile(origin.read_text(), str(origin), "exec")
print(origin.read_text().splitlines()[{index}])
print("BYTECODE_MATCHES_SOURCE" if marshal.dumps(loaded) == marshal.dumps(fresh) else "STALE")
"""


class ProbeFailed(Exception):
    """The read-back could not be performed, so nothing about this mutant is known."""


def _read_back(workspace: Path, python: Path, rel: str, site: Site) -> tuple[str, bool]:
    """Return ``(the line the interpreter reads, whether its bytecode is that source)``.

    ``check=False`` DELIBERATELY. The first version raised, and a single probe
    failure aborted a 42-minute sweep at the 39th minute with a traceback and no
    partial result. One unmeasurable mutant is one unmeasured verdict, not the loss
    of every measurement taken beside it; the caller turns this into
    ``PROBE_FAILED``, which the budget counts as unmeasured and never as a pass.
    """
    proc = subprocess.run(
        [str(python), "-c", _PROBE.format(name=module_name(rel), index=site.line - 1)],
        cwd=workspace,
        env=_child_env(workspace, python),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = proc.stdout.rstrip("\n").splitlines()
    if proc.returncode != 0 or len(lines) < 2:
        raise ProbeFailed(
            f"exit {proc.returncode}: {(proc.stderr.strip().splitlines() or [''])[-1]}"
        )
    return lines[0], lines[-1] == "BYTECODE_MATCHES_SOURCE"


def purge_bytecode(root: Path) -> int:
    """Delete every ``__pycache__`` under ``root``. Returns how many were removed.

    ``PYTHONDONTWRITEBYTECODE`` stops a cache being WRITTEN; it does not stop one
    already on disk being READ. Both halves are needed, and neither is a substitute
    for the probe above, which is what actually reads a value.
    """
    removed = 0
    for cache in sorted(root.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    return removed


def run_one(
    workspace: Path, python: Path, rel: str, fraction: float, deselect: Sequence[str]
) -> Result:
    """Apply one mutant, run the suite, restore the file whatever happens."""
    path = workspace / rel
    original = path.read_text(encoding="utf-8")
    sites = mutable_sites(original)
    if not sites:
        return Result(rel, fraction, "NO_SITES", "", 0)
    site = select_site(sites, fraction)
    mutated = apply_site(original, site)
    expected = mutated.splitlines()[site.line - 1]
    descriptor = f"{rel.split('wildfire_nowcast/')[-1]}:{site.line} {site.old!r}->{site.new!r}"
    started = time.monotonic()
    try:
        path.write_text(mutated, encoding="utf-8")
        purge_bytecode(workspace / "src")
        try:
            seen, bytecode_is_fresh = _read_back(workspace, python, rel, site)
        except ProbeFailed as exc:
            return Result(
                rel,
                fraction,
                "PROBE_FAILED",
                descriptor,
                len(sites),
                detail=f"the read-back could not run, so this mutant was not measured: {exc}",
            )
        if seen != expected:
            return Result(
                rel,
                fraction,
                "NOT_APPLIED",
                descriptor,
                len(sites),
                detail=f"the interpreter reads {seen!r}, not the mutant",
            )
        if not bytecode_is_fresh:
            return Result(
                rel,
                fraction,
                "STALE_BYTECODE",
                descriptor,
                len(sites),
                detail=(
                    "the source is mutated and the loader would still run the OLD bytecode. "
                    "This is a REFUSAL to measure, never a survivor: a mutant that does not "
                    "execute cannot be said to have survived anything."
                ),
            )
        code, output = _run_pytest(workspace, python, deselect)
    finally:
        path.write_text(original, encoding="utf-8")
        purge_bytecode(workspace / "src")
    verdict = "SURVIVED" if code == 0 else "KILLED"
    if verdict == "SURVIVED":
        equivalent = equivalence_note(rel, original, site)
        if equivalent is not None:
            return Result(
                rel,
                fraction,
                "EQUIVALENT",
                descriptor,
                len(sites),
                exit_code=code,
                seconds=round(time.monotonic() - started, 1),
                detail=equivalent[0],
            )
    tail = [ln for ln in output.splitlines() if ln.startswith("FAILED ")][:1]
    return Result(
        rel,
        fraction,
        verdict,
        descriptor,
        len(sites),
        exit_code=code,
        seconds=round(time.monotonic() - started, 1),
        detail=tail[0] if tail else "",
    )


def sweep(
    repo: Path, python: Path, root: Path, *, workers: int, only: str = "", pristine: bool = False
) -> Sweep:
    """Build ``workers`` workspaces and run every mutant exactly once."""
    modules = [m for m in target_modules(repo) if only in m]
    jobs = [(m, f) for m in modules for f in SAMPLE_FRACTIONS]
    spaces = [root / f"ws{i}" for i in range(workers)]
    out = Sweep()
    started = time.monotonic()

    for space in spaces:
        out.carried = build_workspace(repo, space, pristine=pristine)
        assert_workspace_is_self_contained(space, python)
    out.deselected = baseline(spaces[0], python)
    deselect = [arg for node in out.deselected for arg in ("--deselect", node)]

    def confirm(space: Path) -> None:
        """Every workspace is proven green before it judges anything, not just ws0."""
        code, output = _run_pytest(space, python, deselect)
        if code != 0:
            raise RuntimeError(f"{space} is not green before any mutant:\n{output[-2000:]}")

    def work(index: int) -> list[Result]:
        return [
            run_one(spaces[index], python, module, fraction, deselect)
            for module, fraction in jobs[index::workers]
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(confirm, spaces[1:]))
        for batch in pool.map(work, range(workers)):
            out.results.extend(batch)
    out.results.sort(key=lambda r: (r.module, r.fraction))
    out.seconds = time.monotonic() - started
    for space in spaces:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(space)],
            capture_output=True,
            text=True,
            check=False,
        )
    return out


def budget_verdict(survived: int, unmeasured: int, equivalent: int) -> tuple[int, str]:
    """``(exit code, message)``. Pure, so the gate's decision is testable in a millisecond.

    Extracted from ``main`` deliberately: a 40-minute sweep is not a way to find out
    whether the comparison is the right way round, and a gate whose verdict logic is
    only reachable through the slow path is a gate nobody checks.
    """
    if unmeasured:
        return 2, (
            f"FAIL: {unmeasured} mutant(s) never executed, so this sweep did not measure what "
            "it claims. A refusal to measure is not a pass and not a survivor."
        )
    if survived > SURVIVOR_BUDGET:
        return 1, (
            f"FAIL: {survived} survivors against a budget of {SURVIVOR_BUDGET}. "
            "The budget never rises."
        )
    if survived < SURVIVOR_BUDGET:
        return 1, (
            f"FAIL: {survived} survivors against a budget of {SURVIVOR_BUDGET}. Lower "
            f"SURVIVOR_BUDGET to {survived} in this commit: a budget larger than the debt "
            "forgives the next regression."
        )
    return 0, (f"OK: {survived} survivors, exactly the budget; {equivalent} proved unkillable.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", default="", help="substring filter over module paths")
    parser.add_argument(
        "--pristine",
        action="store_true",
        help="measure HEAD exactly; carry no working-tree file. Use this for a quoted number.",
    )
    parser.add_argument(
        "--workspace-root",
        default="",
        help="where worktrees are built (default: a temp dir, NEVER inside the repo)",
    )
    parser.add_argument("--json", default="", help="write the full result set here")
    parser.add_argument(
        "--no-budget",
        action="store_true",
        help="measure and report without enforcing the budget (how a new pin is taken)",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    # OUTSIDE the repository on purpose: a worktree under `repo/` would be visible
    # to `git ls-files --others`, to ruff and to the hygiene scan, and a sweep that
    # changes what the hygiene suite is looking at is measuring itself.
    root = (
        Path(args.workspace_root).resolve()
        if args.workspace_root
        else Path(tempfile.gettempdir()) / "wildfire-nowcast-mutation"
    )
    root.mkdir(parents=True, exist_ok=True)
    result = sweep(
        repo,
        Path(sys.executable),
        root,
        workers=args.workers,
        only=args.only,
        pristine=args.pristine,
    )

    n_survived = len(result.survivors)
    print(
        f"mutants {len(result.results)}  killed {len(result.killed)}  survived {n_survived}  "
        f"equivalent {len(result.equivalent)}  unmeasured {len(result.unmeasured)}"
    )
    print(f"carried {result.carried} working-tree file(s); deselected {result.deselected}")
    for row in result.survivors:
        print(f"  SURVIVED    {row.descriptor}")
    for row in result.equivalent:
        print(f"  EQUIVALENT  {row.descriptor}")
    for row in result.unmeasured:
        print(f"  {row.verdict}  {row.descriptor}  {row.detail}")
    print(f"{result.seconds / 60:.1f} min")
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    if args.only or args.no_budget:
        return 0
    code, message = budget_verdict(n_survived, len(result.unmeasured), len(result.equivalent))
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
