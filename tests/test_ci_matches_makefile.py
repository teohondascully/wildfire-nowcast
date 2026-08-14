"""The CI workflow and ``make ci`` must run the same gate.

[A15] CI is new, and the failure mode it arrives with is well known here: two
descriptions of one fact drift, and the one nobody executes becomes fiction. The
README tells a reader to run ``make test``; the workflow file tells GitHub what
to run; ``make ci`` claims to be the second one. Three statements, one fact.

So this module reads BOTH files and asserts they agree, and — because a scan
that matches nothing passes vacuously, which has produced four confident false
negatives in this project — it checks its own parse first.

It is deliberately not a yaml parse. The workflow's shell steps are the thing
being audited, and matching ``make <target>`` in a ``run:`` block is what
actually says which targets execute.
"""

from __future__ import annotations

import re

from wildfire_nowcast.common.paths import repo_root

WORKFLOW = ".github/workflows/ci.yml"
MAKEFILE = "Makefile"

#: Bootstrap targets a workflow must run and a gate target cannot depend on.
#: ``install`` builds the interpreter the gate then runs through, so requiring
#: it as a prerequisite of ``ci`` would rebuild the venv on every local run.
BOOTSTRAP_TARGETS = frozenset({"install", "venv"})

_MAKE_CALL_RE = re.compile(r"\bmake\s+([a-z][a-z0-9-]*)")


def _workflow_text() -> str:
    return (repo_root() / WORKFLOW).read_text()


def _makefile_text() -> str:
    return (repo_root() / MAKEFILE).read_text()


def workflow_targets() -> set[str]:
    """Every ``make <target>`` the workflow's ``run:`` steps invoke."""
    targets: set[str] = set()
    in_run = False
    for line in _workflow_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # a target named only in a comment is documentation
        if re.match(r"^-?\s*(name|uses|with|env):", stripped):
            in_run = False
        if stripped.startswith("run:"):
            in_run = True
        if in_run:
            targets.update(_MAKE_CALL_RE.findall(stripped))
    return targets - BOOTSTRAP_TARGETS


def ci_target_prerequisites() -> list[str]:
    """The prerequisite list of the Makefile's ``ci`` target."""
    match = re.search(r"^ci:(.*)$", _makefile_text(), flags=re.MULTILINE)
    assert match, "the Makefile has no `ci:` target"
    return match.group(1).split()


def test_the_parses_are_not_silently_empty() -> None:
    """Positive control. Both scans must find something before either can pass."""
    assert (repo_root() / WORKFLOW).is_file(), "there is no CI workflow to audit"
    assert len(workflow_targets()) >= 4, workflow_targets()
    assert len(ci_target_prerequisites()) >= 4, ci_target_prerequisites()
    assert _MAKE_CALL_RE.findall("        run: make lint") == ["lint"]


def test_make_ci_runs_exactly_what_the_workflow_runs() -> None:
    """The whole point: ``make ci`` locally == the gate on GitHub."""
    in_workflow = workflow_targets()
    in_makefile = set(ci_target_prerequisites())
    assert in_workflow == in_makefile, (
        f"CI and `make ci` have drifted. Only in {WORKFLOW}: "
        f"{sorted(in_workflow - in_makefile)}; only in `make ci`: "
        f"{sorted(in_makefile - in_workflow)}. A gate described in two places is a gate that "
        "will be described wrongly in one of them."
    )


def test_every_target_ci_runs_exists_in_the_makefile() -> None:
    """A workflow step naming a target that no longer exists fails only on GitHub."""
    defined = set(re.findall(r"^([a-z][a-z0-9-]*):", _makefile_text(), flags=re.MULTILINE))
    missing = sorted(t for t in workflow_targets() | BOOTSTRAP_TARGETS if t not in defined)
    assert not missing, f"{WORKFLOW} calls make targets that do not exist: {missing}"


def test_the_gate_still_contains_the_checks_that_define_it() -> None:
    """Equality is not enough: deleting a step from BOTH files keeps them equal.

    [A17] ``test_make_ci_runs_exactly_what_the_workflow_runs`` compares two sets,
    so dropping ``typecheck`` from the Makefile and the workflow in one commit
    passes it. This names the gate's load-bearing steps so that removing one is
    an argument someone has to make in a diff, not a two-line deletion that
    stays green.
    """
    required = {"lint", "typecheck", "test-all"}
    in_both = workflow_targets() & set(ci_target_prerequisites())
    missing = sorted(required - in_both)
    assert not missing, (
        f"the CI gate no longer runs {missing}. These are the checks a clean checkout can run "
        "with no data and no credentials; a gate without them is a green light for a tree "
        "nobody has checked."
    )


def test_the_type_checker_is_pinned_and_runs_on_a_supported_interpreter() -> None:
    """An unpinned type checker turns CI red on a day nobody changed anything.

    Also pins the interpreter mypy ITSELF runs on. That is a different thing
    from the version it checks FOR, and getting it wrong is silent: uv picks
    the newest interpreter it can find, numpy's stubs then fail to parse, and
    mypy exits early having reported nine errors out of a tree of 109 files.
    """
    makefile = _makefile_text()
    version = re.search(r"^MYPY_VERSION \?= ([0-9][0-9a-z.]*)$", makefile, flags=re.MULTILINE)
    assert version, "MYPY_VERSION is no longer pinned in the Makefile"
    assert "mypy==$(MYPY_VERSION)" in makefile, (
        "MYPY_VERSION is declared but the invocation does not resolve it with `==`, so the "
        f"pin ({version.group(1)}) is decoration"
    )
    assert "--python $(PY_VERSION)" in makefile, (
        "the mypy invocation no longer pins the interpreter mypy runs on; on the default "
        "interpreter it stops early on numpy's stubs and a truncated run looks like a clean one"
    )


def test_the_workflow_pins_an_interpreter_the_project_supports() -> None:
    """ADR-001 pins CPython 3.12; the Makefile is where that pin lives.

    Checked by reading the value out of both files rather than by asserting a
    literal here, which would be a third copy of the same fact.
    """
    py_version = re.search(r"^PY_VERSION \?= ([0-9.]+)$", _makefile_text(), flags=re.MULTILINE)
    assert py_version, "the Makefile no longer pins PY_VERSION"
    pyproject = (repo_root() / "pyproject.toml").read_text()
    requires = re.search(r'requires-python = ">=([0-9.]+)"', pyproject)
    assert requires, "pyproject.toml no longer declares requires-python"
    assert tuple(int(p) for p in py_version.group(1).split(".")) >= tuple(
        int(p) for p in requires.group(1).split(".")
    ), (
        f"the pinned interpreter {py_version.group(1)} is older than pyproject's floor "
        f"{requires.group(1)}: CI would run a python the package declares unsupported"
    )
