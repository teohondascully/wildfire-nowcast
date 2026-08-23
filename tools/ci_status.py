#!/usr/bin/env python
"""Ask GitHub whether the published head is green. `make ci` cannot answer that.

WHY THIS EXISTS
---------------
On 2026-08-21 the maintainer ran ``make typecheck`` five times, got exit 0 every
time, and reported a green gate. The badge on the public README had been red for
seven days and thirteen commits. Nothing in that sequence was careless: every
command was correct, every exit code was real, and every one of them answered a
question about a laptop rather than about the repository.

Two different defects produced that, and they need two different fixes.

* **The environments differed.** ``requirements.lock`` was installed by nothing,
  so a clean clone re-resolved and landed 15 of 73 packages on other versions.
  Fixed structurally in the Makefile: ``make install`` now syncs the lock.
  With that, a local green and a CI green are claims about the same package set.

* **The SUBJECT differed, and no fix to the environment touches this one.**
  ``make ci`` runs against the working copy in front of you: uncommitted edits
  included, unpushed commits included, and the commit GitHub actually built
  excluded. Even a perfectly reproducible gate cannot tell you that the thing
  you verified is the thing you published. This target closes that gap by
  asking the only party that knows.

WHAT IT REFUSES TO DO
---------------------
It never returns 0 on an unknown. Not when ``gh`` is missing, not when the API
call fails, not when no run exists for the commit, not while a run is still in
progress. This project's most expensive recurring failure is a truncated check
that reads as a clean one -- an all-NaN channel that passed 56 clauses, a
disjointness check that intersected a partition, a fingerprint that recorded
MISSING silently, and mypy exiting early inside numpy's stubs while reporting
nine errors from a tree of a hundred files. A CI probe that answered "could not
tell" with a zero would be the sixth. Every non-success exit prints what it
could not establish and why.

It also reports, rather than hides, the three ways the local tree can fail to be
the published one: uncommitted changes, unpushed commits, and a remote branch
that has moved past you. Every one of the three is a property of HEAD and of
nothing else, so when the commit under test is not HEAD each of those lines
carries HEAD's sha on the line itself. They are printed even when the run is
green, because "the commit I verified is green" and "the branch is green" are
different sentences and it is the second one a badge displays.

THE SUBJECT IS LABELLED, NEVER ASSUMED (ADR-125)
------------------------------------------------
Under ``--sha`` this tool used to print the queried commit on a line labelled
``HEAD`` while the drift lines beneath it still described the real HEAD: four
consecutive lines under one heading, describing two different commits, one of
them eleven commits behind the other. Two was the count on the day; the header
can carry THREE subjects, because ``origin/<branch>`` is the remote tip and
coincides with HEAD only while HEAD is pushed. The exit code was right the whole
time and the header was not, so a reader doing exactly the right thing --
invoking the shipped probe and reading its output -- was told they were standing
on a commit they were not on. A printed caveat is a claim (ADR-116), and this
was the anti-drift caveat itself drifting.

The rule the code now keeps, and it is the one a test asserts: **the word HEAD
labels a line only when that line's sha IS HEAD.** A commit named with ``--sha``
is labelled ``subject``, states its distance from HEAD, and never borrows HEAD's
label; the HEAD-relative lines under it are tagged with HEAD's own sha so that a
single line pasted out of this block still names what it describes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Conclusions GitHub can report. Only one of them is a pass.
SUCCESS = "success"

#: Exit codes, distinct so a caller can tell "red" from "could not tell".
EXIT_OK = 0
EXIT_RED = 1
EXIT_UNKNOWN = 3


def _git(*args: str) -> tuple[int, str]:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return proc.returncode, proc.stdout.strip()


def resolve_sha(revision: str) -> str | None:
    """Full 40-character sha for any revision, or ``None``.

    Always resolved, never passed through. ``gh run list --commit`` matches on
    the FULL sha and returns an empty list for an abbreviated one - so a short
    sha would have been reported as "no run exists for this commit", which is
    the exact shape of false negative this tool was written to refuse. Found by
    a control that asked about a known-green commit by its short form and got
    exit 3.
    """
    code, out = _git("rev-parse", revision)
    return out if code == 0 and len(out) == 40 else None


def working_tree_is_clean() -> bool:
    code, out = _git("status", "--porcelain")
    return code == 0 and out == ""


def upstream_state(branch: str) -> tuple[str | None, int | None, int | None]:
    """``(remote sha, commits ahead, commits behind)`` for ``origin/<branch>``."""
    code, remote = _git("rev-parse", f"origin/{branch}")
    if code != 0 or not remote:
        return None, None, None
    code, counts = _git("rev-list", "--left-right", "--count", f"origin/{branch}...HEAD")
    if code != 0 or not counts:
        return remote, None, None
    parts = counts.split()
    if len(parts) != 2:
        return remote, None, None
    behind, ahead = int(parts[0]), int(parts[1])
    return remote, ahead, behind


def distance_from_head(sha: str) -> tuple[int | None, int | None]:
    """``(commits ahead of HEAD, commits behind HEAD)`` for ``sha``, or ``(None, None)``.

    Printed beside the subject because the distance is the whole of ADR-125: the
    stale row that nearly adjudicated the wrong commit was green, real, and
    eleven commits behind. "Not HEAD" and "not HEAD, and eleven behind it" are
    the same fact at two very different volumes.
    """
    code, counts = _git("rev-list", "--left-right", "--count", f"HEAD...{sha}")
    if code != 0 or not counts:
        return None, None
    parts = counts.split()
    if len(parts) != 2:
        return None, None
    # left = reachable from HEAD only, i.e. how far `sha` sits BEHIND HEAD.
    behind, ahead = int(parts[0]), int(parts[1])
    return ahead, behind


def describe_subject(sha: str, head: str | None) -> str:
    """The parenthetical that follows a ``--sha`` subject on its own line."""
    if head is None:
        return "(--sha; HEAD did not resolve, so the two are uncompared)"
    if sha == head:
        return "(--sha, and it IS HEAD)"
    ahead, behind = distance_from_head(sha)
    if ahead is None or behind is None:
        return "(--sha; NOT HEAD)"
    if ahead and behind:
        return f"(--sha; NOT HEAD: {ahead} ahead, {behind} behind)"
    if behind:
        return f"(--sha; NOT HEAD: {behind} commit(s) behind HEAD)"
    if ahead:
        return f"(--sha; NOT HEAD: {ahead} commit(s) ahead of HEAD)"
    return "(--sha; NOT HEAD)"


def run_for_sha(sha: str, workflow: str) -> dict[str, str] | None:
    """The most recent workflow run for exactly ``sha``, or ``None``.

    Raises ``RuntimeError`` when the question could not be asked at all, which
    is deliberately distinct from asking it and learning there is no run.
    """
    if shutil.which("gh") is None:
        raise RuntimeError(
            "the `gh` CLI is not on PATH, so the published head's status could not be read. "
            "Install it (`brew install gh`) or open the Actions tab. This is NOT a pass."
        )
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--commit",
            sha,
            "--limit",
            "1",
            "--json",
            "conclusion,status,databaseId,headSha,displayTitle,url",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`gh run list` exited {proc.returncode}: {proc.stderr.strip() or 'no message'}. "
            "The published head's status is UNKNOWN, which is not the same as green."
        )
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"could not parse `gh run list` output: {exc}") from exc
    if not rows:
        return None
    row = rows[0]
    return {str(k): str(v) for k, v in row.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="main", help="branch the badge tracks")
    parser.add_argument("--workflow", default="ci.yml", help="workflow file name")
    parser.add_argument(
        "--sha",
        default=None,
        help="commit to ask about (default: HEAD; use origin/<branch> to ask about the badge)",
    )
    args = parser.parse_args(argv)

    sha = resolve_sha(args.sha or "HEAD")
    if sha is None:
        print(
            f"could not resolve {args.sha or 'HEAD'!r} to a commit in this checkout.",
            file=sys.stderr,
        )
        return EXIT_UNKNOWN

    head = resolve_sha("HEAD")
    subject_is_head = head is not None and sha == head

    remote_sha, ahead, behind = upstream_state(args.branch)
    dirty = not working_tree_is_clean()

    # `working tree`, `unpushed` and `behind` are computed against HEAD, so HEAD
    # is named ON EACH OF THEM when the subject is something else. Per line, not
    # once per block: the way this output travels is a line at a time.
    drift = "" if subject_is_head else f"   [HEAD {(head or '?')[:7]}, not the subject]"

    if args.sha is None:
        print(f"HEAD              {sha}")
    else:
        print(f"subject           {sha}   {describe_subject(sha, head)}")
        if not subject_is_head:
            print(f"HEAD              {head or '(unresolved)'}")
    print(f"origin/{args.branch:<10} {remote_sha or '(unknown)'}")
    tree = "DIRTY - uncommitted changes are in no CI run" if dirty else "clean"
    print(f"working tree      {tree}{drift}")
    if ahead:
        print(
            f"unpushed          {ahead} commit(s) ahead of origin/{args.branch}: never built{drift}"
        )
    if behind:
        print(
            f"behind            {behind} commit(s): the badge shows a commit you do not have{drift}"
        )

    try:
        run = run_for_sha(sha, args.workflow)
    except RuntimeError as exc:
        print(f"\nCOULD NOT TELL: {exc}", file=sys.stderr)
        return EXIT_UNKNOWN

    if run is None:
        print(
            f"\nCOULD NOT TELL: no `{args.workflow}` run exists for {sha}. "
            "An unbuilt commit is not a green one.",
            file=sys.stderr,
        )
        return EXIT_UNKNOWN

    status, conclusion = run.get("status", ""), run.get("conclusion", "")
    print(f"\nrun               {run.get('url', '(no url)')}")
    print(f"status/conclusion {status} / {conclusion or '(none yet)'}")

    if status != "completed":
        print(
            f"\nCOULD NOT TELL: the run for {sha} is {status!r}, not completed. Wait for it.",
            file=sys.stderr,
        )
        return EXIT_UNKNOWN
    if conclusion != SUCCESS:
        # Same reason as the GREEN note below: a verdict line quoted on its own
        # must carry its own subject. A misread RED is cheaper than a misread
        # GREEN and it is still a misread.
        elsewhere = "" if subject_is_head else f" That is the --sha subject, not HEAD ({head})."
        print(
            f"\nRED: the published commit {sha} concluded {conclusion!r}.{elsewhere}",
            file=sys.stderr,
        )
        return EXIT_RED

    print(f"\nGREEN: {sha} concluded {SUCCESS}.")
    if not subject_is_head:
        # The line above is the one that gets quoted. ADR-125's near-miss was a
        # green line that was TRUE of its own sha and read as a verdict on HEAD.
        print(
            "  Note: that is the commit you named with --sha. HEAD is "
            f"{head or '(unresolved)'} and this says nothing about it."
        )
    if dirty or ahead:
        # Built as a list of `str` rather than with `x and "text"`, whose type is
        # `Literal[False] | str` - the two strict errors that kept `tools/` out
        # of the type checker. The rewrite is also the readable form: an idiom
        # that smuggles a bool into a list of strings is exactly what a reader
        # skims past.
        notes: list[str] = []
        if dirty:
            notes.append("uncommitted changes")
        if ahead:
            notes.append("unpushed commits")
        print(
            "  Note: that is a statement about the COMMIT, not about your working copy - "
            "you have " + " and ".join(notes) + "."
        )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
