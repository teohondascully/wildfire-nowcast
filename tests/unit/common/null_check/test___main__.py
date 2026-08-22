"""``common/null_check/__main__.py`` - the entry point ``make null-check`` runs.

One line long, and it is the line between the gate and nothing at all.
``python -m wildfire_nowcast.common.null_check`` with a guard that never fires
exits 0 having done no work, so C6.0 reports success by not running. That is the
shape of false green this project has recorded nine times, and it is invisible to
every test that imports the package instead of executing it.
"""

from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wildfire_nowcast.common.null_check", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def test_running_the_module_runs_the_cli_rather_than_exiting_silently() -> None:
    """Executed in a subprocess, because importing it is exactly what does not test it."""
    proc = _run("--help")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert "usage:" in proc.stdout.lower(), (
        "python -m wildfire_nowcast.common.null_check produced no usage text, so the module "
        f"ran to completion without reaching the CLI. stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )
    assert "null" in proc.stdout.lower()


def test_the_module_reports_a_bad_argument_instead_of_succeeding_at_nothing() -> None:
    """The control: a run that does nothing would exit 0 here too, and must not."""
    proc = _run("--not-a-real-option")
    assert proc.returncode != 0, (
        "an unknown option was accepted, which is what a module body that never reaches "
        "argparse looks like from the outside"
    )
