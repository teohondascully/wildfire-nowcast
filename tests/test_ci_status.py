"""``make ci-status`` must never answer "I could not tell" with a zero.

The target exists because on 2026-08-21 five consecutive local gate runs exited
0 while the public badge had been red for seven days: every one of them answered
a question about a laptop, and nothing in the repository answered the question
about the repository. A probe that fixes that and then returns 0 when GitHub is
unreachable would be worse than no probe, because a zero gets quoted.

So the properties under test are the REFUSALS, not the happy path. Every one of
them is exercised without touching the network: the single function that shells
out to ``gh`` is replaced, which is also why it is a single function.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

import ci_status

REPO_ROOT = Path(ci_status.__file__).resolve().parents[1]


def _run(monkeypatch: pytest.MonkeyPatch, result: Any) -> int:
    """Run the CLI against a canned ``gh`` answer (or exception)."""

    def fake(sha: str, workflow: str) -> dict[str, str] | None:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ci_status, "run_for_sha", fake)
    return ci_status.main([])


# --------------------------------------------------------------------------
# 1. Unknown is not green
# --------------------------------------------------------------------------


def test_the_success_and_failure_codes_are_distinct_and_only_one_is_zero() -> None:
    assert ci_status.EXIT_OK == 0
    assert ci_status.EXIT_RED != 0
    assert ci_status.EXIT_UNKNOWN != 0
    assert ci_status.EXIT_RED != ci_status.EXIT_UNKNOWN, (
        "'red' and 'could not tell' must be distinguishable by a caller: one means fix the "
        "code, the other means go and look"
    )


def test_a_missing_gh_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ci_status.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="NOT a pass"):
        ci_status.run_for_sha("0" * 40, "ci.yml")
    code = ci_status.main([])
    assert code != 0 and code == ci_status.EXIT_UNKNOWN


def test_no_run_for_the_commit_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    code = _run(monkeypatch, None)
    assert code != 0 and code == ci_status.EXIT_UNKNOWN


def test_a_gh_error_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    code = _run(monkeypatch, RuntimeError("network down"))
    assert code != 0 and code == ci_status.EXIT_UNKNOWN


def test_a_run_still_in_progress_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"status": "in_progress", "conclusion": "", "url": "u"}
    code = _run(monkeypatch, row)
    assert code != 0 and code == ci_status.EXIT_UNKNOWN


def test_a_cancelled_run_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """This repo has a real cancelled run in its history; it is not a green one."""
    row = {"status": "completed", "conclusion": "cancelled", "url": "u"}
    code = _run(monkeypatch, row)
    assert code != 0 and code == ci_status.EXIT_RED


def test_a_failed_run_is_red(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"status": "completed", "conclusion": "failure", "url": "u"}
    code = _run(monkeypatch, row)
    assert code != 0 and code == ci_status.EXIT_RED


def test_a_completed_successful_run_is_the_only_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the tool must be capable of returning 0 at all."""
    row = {"status": "completed", "conclusion": "success", "url": "u"}
    assert _run(monkeypatch, row) == ci_status.EXIT_OK


# --------------------------------------------------------------------------
# 2. The subject of the claim is resolved, not assumed
# --------------------------------------------------------------------------


def test_a_short_sha_resolves_rather_than_reading_as_no_run() -> None:
    """`gh run list --commit` matches full shas only. Found by a control, not by reading.

    Asking about a known-green commit by its 7-character form returned "no run
    exists", which is a false negative wearing the tool's own refusal message.
    """
    full = ci_status.resolve_sha("HEAD")
    assert full is not None and len(full) == 40
    assert ci_status.resolve_sha(full[:7]) == full
    assert ci_status.resolve_sha("nosuchrevisionanywhere") is None


def test_an_unresolvable_revision_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ci_status, "resolve_sha", lambda _rev: None)
    code = ci_status.main([])
    assert code != 0 and code == ci_status.EXIT_UNKNOWN


# --------------------------------------------------------------------------
# 3. The target is wired, and is NOT part of `make ci`
# --------------------------------------------------------------------------


def test_the_makefile_exposes_it_and_keeps_it_out_of_the_gate() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert "tools/ci_status.py" in makefile, "`make ci-status` does not invoke the tool"

    ci_line = re.search(r"^ci:(.*)$", makefile, flags=re.MULTILINE)
    assert ci_line, "the Makefile has no `ci:` target"
    assert "ci-status" not in ci_line.group(1).split(), (
        "`ci-status` is a prerequisite of `make ci`. It must not be: it needs the network and "
        "the GitHub API, and a gate that cannot run offline stops being run"
    )
