"""Run the test suite in a detached worktree, so a neighbour's edit cannot enter it.

WHY THIS IS A TARGET AND NOT A HABIT. Four leads share one working tree and three
of them run plant-and-revert protocols, in which a file is DELIBERATELY corrupted
for a few seconds. Any full-suite run during that window measures the plant. It
has now happened to two leads independently: a baseline read 3 failed / 888 passed
at a sha known green, with `git status` clean by the time it was inspected,
because a neighbour was mid-revert in a package the reader does not own.

That failure mode cannot be fixed by care, because the reader has no way to know
the window was open. It can be fixed by not reading the shared tree at all: this
builds a worktree at HEAD, which is immutable while it runs, and runs there.

THREE THINGS IT DOES THAT A BARE `git worktree add` DOES NOT.

1. It symlinks `.venv`. Without one, two tests fail for reasons that are about
   the sandbox and not the code (`iso/.venv/bin/ruff is missing`, and a ruff
   version comparison that then resolves some other ruff). A harness that always
   reports two failures teaches people to ignore its failures.
2. It puts the worktree FIRST on `PYTHONPATH` and then CHECKS that it won, by
   reading `wildfire_nowcast.__file__` back. The editable install points
   `sys.path` at the real `src/`, so without this the suite would import the very
   tree it is trying not to read, pass, and prove nothing.
3. It purges bytecode and disables writing it, for the reason in the Makefile.

Exit code is pytest's own, passed through unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutation import build_workspace, purge_bytecode  # noqa: E402

# ADR-103: a logger, and NO handler configured at import. `main` configures.
logger = logging.getLogger(__name__)

CONTROL = (
    "import pathlib, sys, wildfire_nowcast\n"
    "print(pathlib.Path(wildfire_nowcast.__file__).resolve())\n"
)


def isolate(repo: Path, workspace: Path) -> None:
    """A worktree at HEAD with a usable environment. Carries no working-tree file."""
    build_workspace(repo, workspace, pristine=True)
    venv = workspace / ".venv"
    if not venv.exists():
        venv.symlink_to(repo / ".venv")
    purge_bytecode(workspace)


def child_env(workspace: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def assert_self_contained(workspace: Path, python: Path) -> None:
    """The control. A suite that imported the shared tree measured the shared tree."""
    seen = subprocess.run(
        [str(python), "-c", CONTROL],
        cwd=workspace,
        env=child_env(workspace),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not seen.startswith(str(workspace.resolve())):
        raise SystemExit(
            f"REFUSING TO RUN: the interpreter imports {seen}, which is outside {workspace}. "
            "The isolation did not take, so this run would measure the shared tree while "
            "claiming not to."
        )


def main(argv: list[str] | None = None) -> int:
    # ADR-103: imported HERE rather than at module scope so this stays runnable
    # against a tree where the package is not installed - it builds the workspace
    # the package is then installed into.
    from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args

    parser = argparse.ArgumentParser(description=__doc__)
    add_logging_arguments(parser)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--keep", action="store_true", help="do not remove the worktree")
    parser.add_argument("pytest_args", nargs="*")
    # parse_known_args, so `-q` and `-k foo` reach pytest instead of being
    # rejected here. This wrapper has opinions about the TREE, none about pytest.
    args, passthrough = parser.parse_known_args(argv)
    # ADR-103: the ONE place this program configures logging. INFO by default
    # rather than WARNING, because progress narration IS what this wrapper is
    # for: a 40 minute run that says nothing looks identical to a hung one.
    configure_from_args(args, default_verbosity=1)

    repo = Path(args.repo).resolve()
    python = repo / ".venv" / "bin" / "python"
    root = Path(tempfile.mkdtemp(prefix="wildfire-isolated-"))
    workspace = root / "ws"
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    try:
        isolate(repo, workspace)
        assert_self_contained(workspace, python)
        # PROGRESS, not output: it says what the tool is doing, and the only
        # line this program produces that a caller reads is pytest's own.
        logger.info("isolated at %s in %s", head, workspace)
        rest = [a for a in [*passthrough, *args.pytest_args] if a != "--"]
        proc = subprocess.run(
            [str(python), "-m", "pytest", *rest],
            cwd=workspace,
            env=child_env(workspace),
            check=False,
        )
        print(f"isolated suite at {head}: pytest exit {proc.returncode}")
        return proc.returncode
    finally:
        if not args.keep:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(workspace)],
                capture_output=True,
                text=True,
                check=False,
            )
            # The mkdtemp ROOT as well, not only the worktree inside it. Removing
            # the worktree left an empty directory per invocation - individually
            # trivial, which is exactly how 15,462 of them accumulated elsewhere in
            # this repository in a single day. What a tool leaves behind is part of
            # its cost, and it is multiplied by how often the tool is meant to run.
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
