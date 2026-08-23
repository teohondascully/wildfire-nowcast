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


# --------------------------------------------------------------------------
# 4. The subject of the claim is LABELLED, not assumed (ADR-125 (7), (9))
#
# The header used to print the commit named with `--sha` on a line labelled
# `HEAD`, above three lines computed against the real HEAD. Four consecutive
# lines under one heading, two different commits. The exit code was correct
# throughout, which is why nothing caught it: only the header lied, and the
# header is the part a reader quotes.
#
# The plant is that exact invocation. `88ebd935`, eleven commits behind
# `7c6ae04`, is the sha ADR-125 was written about: a real commit with a real
# green run, printed under the label `HEAD` while `working tree clean` beneath
# it described a commit eleven commits ahead of it.
# --------------------------------------------------------------------------

#: The plant of record. Named, not derived, so that the test says out loud which
#: observation it is: this is the sha from ADR-125 (7). `HEAD~1` is the fallback
#: for a checkout that does not contain it (a shallow clone, or a fork whose
#: history predates it); the property under test needs only "some commit that is
#: not HEAD", but the recorded instance is preferred when it is available.
PLANT_SHA_OF_RECORD = "88ebd9357860cb14472db2bdc4357aa7a20737da"

#: The labels whose value is computed against HEAD and against nothing else.
HEAD_RELATIVE_LABELS = ("working tree", "unpushed", "behind")


def _plant_sha() -> str:
    head = ci_status.resolve_sha("HEAD")
    assert head is not None, "no HEAD in this checkout; the whole tool is untestable here"
    for revision in (PLANT_SHA_OF_RECORD, "HEAD~1", "HEAD^"):
        candidate = ci_status.resolve_sha(revision)
        if candidate is not None and candidate != head:
            return candidate
    pytest.skip("this checkout has no commit other than HEAD, so the plant cannot be built")


def _rows(output: str) -> list[tuple[str, str]]:
    """The header's ``(label, value)`` pairs, split at its fixed 18-column gutter.

    Deliberately structural rather than a substring search: the defect was that a
    TRUE value sat under a FALSE label, so a test that only asked whether a sha
    appeared somewhere in the output would have passed on the broken tool.
    """
    rows = []
    for line in output.splitlines():
        if line.startswith(" ") or len(line) <= 18 or line[17] != " ":
            continue
        rows.append((line[:18].strip(), line[18:].strip()))
    return rows


def _run_argv(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], row: dict[str, str]
) -> tuple[int, list[str]]:
    """Run the CLI with a canned ``gh`` answer; return ``(exit code, shas asked about)``.

    The shas come back because the canned answer is the same whatever is asked:
    a test that only read the answer could not tell a bound probe from an
    unbound one, which is the other half of ADR-125.
    """
    asked: list[str] = []

    def fake(sha: str, workflow: str) -> dict[str, str] | None:
        asked.append(sha)
        return row

    monkeypatch.setattr(ci_status, "run_for_sha", fake)
    return ci_status.main(argv), asked


_GREEN_ROW = {"status": "completed", "conclusion": "success", "url": "u"}


def test_control_the_default_invocation_is_unchanged_and_fires_no_mismatch_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CONTROL: `--sha` omitted. The subject IS HEAD, so the header must not move.

    This half must pass both before and after the repair -- that is what makes it
    a control. If it ever starts failing, the repair has changed the output of the
    invocation that was never wrong.
    """
    head = ci_status.resolve_sha("HEAD")
    assert head is not None
    code, _asked = _run_argv(monkeypatch, [], _GREEN_ROW)
    out = capsys.readouterr().out
    rows = _rows(out)

    assert out.splitlines()[0] == f"HEAD              {head}", (
        "the default invocation's first line is the pre-existing HEAD row and must be "
        "byte-identical to it"
    )
    assert [label for label, _ in rows][:1] == ["HEAD"]
    assert "subject" not in out, "the mismatch path fired on a query that IS HEAD"
    assert "not the subject" not in out
    for label, value in rows:
        if label in HEAD_RELATIVE_LABELS:
            assert "HEAD" not in value, (
                f"the {label!r} row was tagged with HEAD's sha although the subject IS HEAD; "
                "the tag exists only to disambiguate, and there is nothing here to disambiguate"
            )
    assert code == ci_status.EXIT_OK, "exit codes are load-bearing and do not move"


def test_plant_a_sha_that_is_not_head_is_never_printed_under_the_label_head(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PLANT: `--sha 88ebd935`, eleven commits behind HEAD. ADR-125 (7)'s own case.

    Against the unrepaired tool this fails twice over, on two independent
    readings of the same header: the queried commit is printed under `HEAD`, and
    `working tree` is printed under it while describing a different commit.
    """
    head = ci_status.resolve_sha("HEAD")
    plant = _plant_sha()
    assert head is not None and plant != head

    code, asked = _run_argv(monkeypatch, ["--sha", plant], _GREEN_ROW)
    out = capsys.readouterr().out
    rows = _rows(out)

    # (a) The label `HEAD` may only ever carry HEAD.
    for label, value in rows:
        if label == "HEAD":
            assert value.split()[0] == head, (
                f"a line labelled HEAD carries {value.split()[0]!r}, which is not HEAD ({head}). "
                "This is ADR-125 (7): the queried commit wearing HEAD's label."
            )

    # (b) The queried commit is printed, and never under HEAD's label.
    carrying_plant = [(label, value) for label, value in rows if plant in value]
    assert carrying_plant, f"the queried commit {plant} does not appear in the header at all"
    assert all(label != "HEAD" for label, _ in carrying_plant)
    subject_rows = [value for label, value in carrying_plant if label != "HEAD"]
    assert any("NOT HEAD" in value for value in subject_rows), (
        "the header does not say that the queried commit differs from HEAD; a reader who "
        "quotes this block can still attribute it to the commit they are standing on"
    )

    # (c) Every HEAD-relative line names the commit it actually describes, or is
    #     not printed at all. Suppression and attribution are both acceptable;
    #     silent misattribution is the defect.
    for label, value in rows:
        if label in HEAD_RELATIVE_LABELS:
            assert head[:7] in value, (
                f"the {label!r} row is computed against HEAD ({head[:7]}) but names no commit, "
                f"and it is printed beneath the queried commit {plant[:7]}. Its subject is not "
                "recoverable from the line."
            )

    # (d) The probe was bound to the FULL queried sha (ADR-125 (3)), and the exit
    #     code did not move: this repair is about the header only.
    assert asked == [plant], "the gh probe was not bound to the full queried sha"
    assert code == ci_status.EXIT_OK


def test_restored_a_sha_equal_to_head_is_labelled_truthfully_and_carries_no_tag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """RESTORED: `--sha HEAD`. The same flag, pointed back at the commit it was on.

    The third observation, and it is not decoration: a repair that tagged every
    `--sha` invocation as foreign would pass the plant and be wrong here. The
    discriminator is the SHA, not the flag.
    """
    head = ci_status.resolve_sha("HEAD")
    assert head is not None
    code, _asked = _run_argv(monkeypatch, ["--sha", head], _GREEN_ROW)
    out = capsys.readouterr().out
    rows = _rows(out)

    assert any(head in value for _, value in rows)
    assert "NOT HEAD" not in out, "the subject IS HEAD here and the header claims otherwise"
    assert "not the subject" not in out
    for label, value in rows:
        if label == "HEAD":
            assert value.split()[0] == head
    assert code == ci_status.EXIT_OK
