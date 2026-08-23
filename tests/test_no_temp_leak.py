"""The suite must not leave temporary directories behind.

WHY THIS FILE EXISTS
--------------------
On 2026-08-22 the maintainer's disk filled. ``/private/tmp`` held **15,462
directories across two prefixes, 70.6 GB**: 8,853 ``wnc-selftest-*`` and 6,605
``simviz-selftest-*``. Nothing was wrong with any single test. Two fixtures
called ``tempfile.mkdtemp()``, which never cleans up, and each wrote a ~4.7 MB
synthetic tensor.

The multiplier was the mutation gate. ``make mutation`` runs the entire suite
once per mutant, 117 mutants to a sweep, so one sweep leaked roughly half a
gigabyte and several sweeps ran in a day. **Every individual run was correct and
the aggregate was ruinous**, which is precisely the class of defect a per-test
assertion cannot see: no test leaked enough to notice, and nothing measured the
sum.

WHAT THIS ASSERTS
-----------------
That a real pytest process, given a private ``TMPDIR``, leaves it empty. It runs
a subprocess rather than inspecting the current one because the leak is created
by fixtures at import and call time and is only observable after interpreter
exit, when ``atexit`` handlers have run. Checking in-process would pass while
the directories still existed.

It carries its own positive control: a subprocess that deliberately leaks must
be caught, or this file is decoration.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Kept small on purpose. This runs a real interpreter; it is a guard, not a
#: second suite. The adopted self-tests are the exact modules that leaked.
_PROBE_TARGET = "tests/test_adopted_selftests.py"


def _run_with_private_tmpdir(tmp_path: Path, code: str | None, target: str | None) -> int:
    """Run pytest (or ``code``) with ``TMPDIR`` pointed at a private directory."""
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if code is not None:
        args = [sys.executable, "-c", code]
    else:
        args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(target)]
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False).returncode


def _leftovers(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir())


def test_the_adopted_selftests_leave_no_temporary_directory(tmp_path: Path) -> None:
    """The regression itself: 70.6 GB across 15,462 directories, in one day."""
    private = tmp_path / "tmpdir"
    private.mkdir()
    rc = _run_with_private_tmpdir(private, None, _PROBE_TARGET)
    left = _leftovers(private)
    assert rc == 0, f"probe suite did not pass, so a clean TMPDIR proves nothing (rc={rc})"
    assert left == [], (
        f"{len(left)} temporary entries survived the run: {left[:5]}. "
        "A fixture is using tempfile.mkdtemp without cleanup. Use a per-process "
        "scratch registered with atexit, never a per-call mkdtemp."
    )


def test_the_check_catches_a_process_that_actually_leaks() -> None:
    """POSITIVE CONTROL. Without this, the test above passes on a broken checker."""
    import tempfile

    with tempfile.TemporaryDirectory() as outer:
        private = Path(outer) / "tmpdir"
        private.mkdir()
        rc = _run_with_private_tmpdir(
            private,
            "import tempfile; tempfile.mkdtemp(prefix='deliberate-leak-')",
            None,
        )
        left = _leftovers(private)
        assert rc == 0, "the control subprocess should exit cleanly; it only leaks"
        assert len(left) == 1, f"the control did not leak, so the guard is unproven: {left}"
        assert left[0].startswith("deliberate-leak-")


@pytest.mark.parametrize("prefix", ["wnc-selftest-", "simviz-selftest-"])
def test_the_two_known_prefixes_are_not_created_per_call(prefix: str) -> None:
    """Pin the specific fixtures that caused it, by behaviour rather than by name.

    Calling the fixture twice in one process must not produce two directories.
    """
    code = (
        "import tempfile, glob, os, sys\n"
        "before = set(glob.glob(os.path.join(tempfile.gettempdir(), %r + '*')))\n"
        "from wildfire_nowcast.%s import selftest as m\n"
        "f = getattr(m, %r, None)\n"
        "if f is None: sys.exit(0)\n"
        "f(); f()\n"
        "after = set(glob.glob(os.path.join(tempfile.gettempdir(), %r + '*')))\n"
        "sys.exit(1 if len(after - before) > 1 else 0)\n"
    )
    pkg, fn = (
        ("eval", "_synthetic_dataset") if prefix.startswith("wnc") else ("sim", "_open_synthetic")
    )
    import tempfile

    with tempfile.TemporaryDirectory() as outer:
        private = Path(outer) / "tmpdir"
        private.mkdir()
        rc = _run_with_private_tmpdir(private, code % (prefix, pkg, fn, prefix), None)
        assert rc == 0, (
            f"calling {pkg}.selftest.{fn} twice created more than one {prefix}* directory. "
            "It must build its scratch once per process, not once per call."
        )
