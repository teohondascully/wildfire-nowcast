#!/usr/bin/env python3
"""Refuse to publish a ref this remote does not already carry.

WHY THIS EXISTS. This repository's public history was rebuilt from scratch before
it went public. The history it replaced is still here, as a local-only branch
whose commits are FULLY UNRELATED to ``main`` -- there is no merge base and both
roots are the same day. Its commit messages are clean; its FILES are not, because
they reference internal tooling by name in comments and docstrings. Publishing it
is one flag away: ``git push --all`` sends every branch, and nothing about the
default configuration prevents that. Low probability, high cost, and mechanical
to close.

THE RULE, AND WHY IT NAMES NO BRANCH.

RULE 1  A REF MAY ONLY BE PUSHED TO A DESTINATION THE REMOTE ALREADY HAS.
    The allowed set is READ from ``refs/remotes/<remote>/``, i.e. from git's own
    record of what that remote carries. Nothing is typed in and no branch is
    named anywhere in this file: "main is the only publishable ref" is a
    DESCRIPTION of this remote, not a constant of mine, and the moment it stops
    being true the derivation says so without an edit. Publishing a NEW ref
    requires deliberately going around the guard, which is the whole point --
    ``git push --all`` and ``git push --mirror`` both stop here.
    With no remote-tracking refs at all the allowed set is EMPTY and every push
    is refused. That is the safe direction for a guard, and it is the same
    direction ``tools/commit_guard.py`` rule 3 fails in.

RULE 2  THE COMMIT PUSHED TO AN ALLOWED DESTINATION MUST SHARE HISTORY WITH IT.
    Rule 1 adjudicates the DESTINATION, so on its own it would permit
    ``git push origin <unrelated-branch>:<allowed-branch>`` -- which publishes
    exactly the content rule 1 exists to keep back, under a name that passes.
    Rule 2 asks ``git merge-base`` whether the commit being pushed is related to
    what is already published there. An amended or rebased commit still shares
    ancestry and is allowed (the repair that motivated the commit-message guard
    was a force-push and must stay possible); a disjoint history does not and is
    not. Again nothing is enumerated: "unrelated" is git's own verdict.

FAIL CLOSED. If the guard cannot determine what is being pushed it REFUSES. That
covers a malformed ref line, a missing remote name, a commit that is not in the
object database, an empty allowed set, and being invoked in a slot where stdin is
not git's (see ``--git-hook`` below). An EMPTY ref list is different and is
allowed: git writes one line per ref it intends to send, so zero lines means zero
refs are being published. That was measured, not assumed -- an up-to-date
``git push`` really does invoke the hook with empty stdin, and refusing it would
be a false positive on a command that publishes nothing.

TWO LAYERS, BECAUSE THE OBVIOUS WIRING IS BLIND. Measured before building:

  * ``--git-hook`` is the AUTHORITATIVE layer. It reads git's own pre-push
    protocol on stdin and therefore sees EVERY ref in the push.
  * ``--pre-commit-stage`` is the layer that arrives through
    ``.pre-commit-config.yaml``. It CANNOT see the whole push and does not
    pretend to: ``pre-commit`` consumes stdin in its own hook wrapper and then
    exports the FIRST pushable ref only. Measured on pre-commit 4.6.1 with a
    dummy remote: ``git push --all`` published two branches while the hook saw
    ``PRE_COMMIT_LOCAL_BRANCH=refs/heads/main`` and ``stdin=b''``. A guard built
    on that environment would have reported "pushing main" and allowed the exact
    command this file exists to stop. So this layer's real job is to check that
    the authoritative layer RAN during this push (it leaves a breadcrumb), and to
    adjudicate the one ref it can see as a cheap independent second opinion.
    Remove or break the authoritative layer and every push is refused rather
    than quietly unguarded.

``--install`` puts the authoritative layer in ``<hooks>/pre-push.legacy``, which
is the slot ``pre-commit`` itself invokes FIRST, with the full stdin piped in and
its exit status ORed into the result. That is measured behaviour, not a reading
of the docs, and the end-to-end test in ``tests/test_push_guard.py`` performs a
real push against a throwaway remote so the chain is pinned by an observation
rather than by trust. ``make hooks`` runs the install, so the guard arrives with
``make install`` instead of as an instruction someone has to remember -- which is
the failure mode the commit-message guard was built to close.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BREADCRUMB_MAX_AGE_S",
    "PushRef",
    "Refusal",
    "adjudicate",
    "authoritative_layer_ran",
    "hooks_dir",
    "install",
    "parse_push_refs",
    "published_refs",
    "read_breadcrumb",
    "shares_history",
    "tracking_ref",
]

#: A SECOND, WEAKER CONDITION ON THE BREADCRUMB. The one that does the work is
#: the parent-process identity below; this only bounds how long a recycled pid
#: could be believed. The two layers run milliseconds apart inside one push.
BREADCRUMB_MAX_AGE_S = 120.0

#: `git rev-parse --git-path` resolves this against the real git directory, so it
#: is correct inside a linked worktree too. Nothing tracked is written.
_BREADCRUMB = "push-guard-ran"

_SHIM = """#!/bin/sh
# Installed by `make hooks` (tools/push_guard.py --install). The logic lives in
# tools/push_guard.py -- edit that, not this. If the script is missing or the
# repository root cannot be resolved this exits non-zero, which REFUSES the push.
root=$(git rev-parse --show-toplevel) || exit 1
exec "$root/tools/push_guard.py" --git-hook "$@"
"""


@dataclass(frozen=True)
class PushRef:
    """One line of git's pre-push protocol.

    ``<local ref> <local sha> <remote ref> <remote sha>``, exactly as git writes it.
    """

    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str

    @property
    def is_delete(self) -> bool:
        """A deletion arrives with an all-zero local sha (and ``(delete)`` as the local ref)."""
        return bool(self.local_sha) and set(self.local_sha) == {"0"}


@dataclass(frozen=True)
class Refusal:
    """One reason a push is refused."""

    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.detail}"


def _git(args: list[str], repo: Path) -> str:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout


def _git_ok(args: list[str], repo: Path) -> bool:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).returncode == 0


def parse_push_refs(text: str) -> list[PushRef]:
    """Parse git's pre-push stdin. Raises ``ValueError`` on a line it cannot read.

    Exactly four whitespace-separated fields, no more and no fewer. A ref name
    cannot contain a space -- ``git check-ref-format "refs/heads/with space"``
    exits 1, checked rather than assumed -- so there is no ambiguity to resolve
    and a line with a different field count is one this parser does not
    understand. It says so instead of guessing: mis-attributing a sha to a ref
    would produce a confident verdict about the wrong thing, and the caller turns
    a ``ValueError`` here into a refusal.
    """
    refs: list[PushRef] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"cannot read this pre-push line: {line!r}")
        refs.append(PushRef(parts[0], parts[1], parts[2], parts[3]))
    return refs


def published_refs(repo: Path, remote: str) -> frozenset[str]:
    """The refs this remote already carries, as git records them locally.

    Read from ``refs/remotes/<remote>/``; never typed in. Symbolic refs (that is,
    ``refs/remotes/<remote>/HEAD``) are skipped: HEAD is a pointer at a branch,
    not a branch, and no push has it as a destination.
    """
    prefix = f"refs/remotes/{remote}/"
    try:
        raw = _git(["for-each-ref", "--format=%(refname)%09%(symref)", prefix], repo)
    except subprocess.CalledProcessError:
        return frozenset()
    out: set[str] = set()
    for line in raw.splitlines():
        name, _, symref = line.partition("\t")
        if symref.strip() or not name.startswith(prefix):
            continue
        out.add(f"refs/heads/{name[len(prefix) :]}")
    return frozenset(out)


def tracking_ref(remote: str, remote_ref: str) -> str:
    """``refs/heads/x`` on ``origin`` is recorded locally as ``refs/remotes/origin/x``."""
    return f"refs/remotes/{remote}/{remote_ref.removeprefix('refs/heads/')}"


def shares_history(repo: Path, left: str, right: str) -> bool:
    """git's own verdict on whether two commits are related at all."""
    return _git_ok(["merge-base", left, right], repo)


def _commit_exists(repo: Path, sha: str) -> bool:
    return _git_ok(["cat-file", "-e", f"{sha}^{{commit}}"], repo)


def adjudicate(
    refs: Iterable[PushRef],
    *,
    published: Iterable[str],
    repo: Path,
    remote: str,
) -> list[Refusal]:
    """Every reason this push is refused. Empty list means allowed.

    Pure with respect to the push: the ref list and the allowed set are
    ARGUMENTS, so a test can adjudicate a hypothetical push -- including one this
    repository must never perform -- without performing it.
    """
    refs = list(refs)
    allowed = frozenset(published)
    if not refs:
        return []
    if not allowed:
        return [
            Refusal(
                "no-published-refs",
                f"git has no record of any ref on {remote!r} (refs/remotes/{remote}/ is empty), "
                "so there is nothing this push could be an update to. Fetch first. If you are "
                "publishing to a NEW remote on purpose, that is what --no-verify is for.",
            )
        ]

    out: list[Refusal] = []
    for ref in refs:
        if ref.remote_ref not in allowed:
            out.append(
                Refusal(
                    "unpublished-ref",
                    f"{ref.remote_ref!r} is not a ref {remote!r} already carries "
                    f"(it carries {sorted(allowed)}). This push would PUBLISH it. "
                    "If that is what you meant, say so with --no-verify.",
                )
            )
            continue
        if ref.is_delete or not ref.local_sha:
            continue
        if not _commit_exists(repo, ref.local_sha):
            out.append(
                Refusal(
                    "unreadable-object",
                    f"{ref.local_sha!r} is not a commit in this object database, so what "
                    f"{ref.remote_ref!r} would receive cannot be checked.",
                )
            )
            continue
        against = tracking_ref(remote, ref.remote_ref)
        if not shares_history(repo, ref.local_sha, against):
            out.append(
                Refusal(
                    "unrelated-history",
                    f"{ref.local_sha[:12]} shares NO history with {against} "
                    f"(git merge-base finds nothing), so pushing it to {ref.remote_ref!r} would "
                    "replace what is published with an unrelated history.",
                )
            )
    return out


# --------------------------------------------------------------------------
# Installation and the breadcrumb the second layer checks
# --------------------------------------------------------------------------


def hooks_dir(repo: Path) -> Path:
    """Where git looks for hooks, honouring ``core.hooksPath``."""
    try:
        configured = _git(["config", "--get", "core.hooksPath"], repo).strip()
    except subprocess.CalledProcessError:
        configured = ""
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (repo / path)
    return (repo / _git(["rev-parse", "--git-path", "hooks"], repo).strip()).resolve()


def _breadcrumb_path(repo: Path) -> Path:
    return (repo / _git(["rev-parse", "--git-path", _BREADCRUMB], repo).strip()).resolve()


def _write_breadcrumb(repo: Path) -> None:
    """Record that the authoritative layer RAN, and in WHICH push. Not that it approved.

    The parent pid is the discriminating field, and it was measured, not assumed:
    ``pre-commit`` runs the legacy hook and the config hooks as siblings of one
    ``hook-impl`` process, so both observe the same ``os.getppid()``. A timestamp
    alone is not enough -- an earlier push's breadcrumb is still young, and a
    live end-to-end run caught exactly that: after deleting the authoritative
    layer, a second push seconds later was ALLOWED by a time-window check while
    the pytest fixture, whose deletion happened before any push, reported the
    hole closed. The fixture could not see it; the real push could.
    """
    try:
        path = _breadcrumb_path(repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.3f} {os.getppid()}\n", encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - unwritable git dir
        pass


def read_breadcrumb(repo: Path) -> tuple[float, int] | None:
    """``(unix time, parent pid)`` of the authoritative layer's last run, or ``None``."""
    try:
        path = _breadcrumb_path(repo)
        stamp, _, pid = path.read_text(encoding="utf-8").strip().partition(" ")
        return float(stamp), int(pid)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def authoritative_layer_ran(repo: Path, *, ppid: int, now: float | None = None) -> bool:
    """Did the full-ref-list layer run in THIS push, as opposed to some earlier one?"""
    crumb = read_breadcrumb(repo)
    if crumb is None:
        return False
    stamp, recorded_ppid = crumb
    if recorded_ppid != ppid:
        return False
    return (now if now is not None else time.time()) - stamp <= BREADCRUMB_MAX_AGE_S


def install(repo: Path) -> Path:
    """Write the authoritative hook into the slot ``pre-commit`` chains to, and verify it.

    Idempotent. Returns the path written.
    """
    target = hooks_dir(repo) / "pre-push.legacy"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SHIM, encoding="utf-8")
    target.chmod(0o755)
    return target


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _report(refusals: Sequence[Refusal], *, layer: str) -> int:
    if not refusals:
        return 0
    print(
        f"push-guard ({layer}): REFUSED. This repository publishes exactly the refs its\n"
        "remote already carries, because a local-only branch here holds the pre-public\n"
        "history and `git push --all` would send it.\n",
        file=sys.stderr,
    )
    for refusal in refusals:
        print(f"  {refusal}", file=sys.stderr)
    return 1


def run_git_hook(args: Sequence[str], *, repo: Path) -> int:
    """The authoritative layer: git's own pre-push protocol on stdin."""
    _write_breadcrumb(repo)

    if os.environ.get("PRE_COMMIT") and not os.environ.get("PRE_COMMIT_RUNNING_LEGACY"):
        return _report(
            [
                Refusal(
                    "wrong-slot",
                    "this ran as a pre-commit-managed hook, where pre-commit has already "
                    "consumed git's stdin, so the ref list cannot be read here. Wire this "
                    "entry point with `tools/push_guard.py --install`.",
                )
            ],
            layer="git hook",
        )
    if not args:
        return _report(
            [Refusal("no-remote-name", "git did not pass a remote name, so nothing can be read")],
            layer="git hook",
        )
    remote = args[0]

    if sys.stdin.isatty():
        return _report(
            [Refusal("no-ref-list", "stdin is a terminal, so the ref list cannot be read")],
            layer="git hook",
        )
    try:
        refs = parse_push_refs(sys.stdin.read())
    except ValueError as exc:
        return _report([Refusal("malformed-ref-list", str(exc))], layer="git hook")

    return _report(
        adjudicate(refs, published=published_refs(repo, remote), repo=repo, remote=remote),
        layer="git hook",
    )


def run_pre_commit_stage(*, repo: Path) -> int:
    """The layer that arrives through `.pre-commit-config.yaml`. Partial BY MEASUREMENT."""
    refusals: list[Refusal] = []

    if not authoritative_layer_ran(repo, ppid=os.getppid()):
        refusals.append(
            Refusal(
                "authoritative-layer-did-not-run",
                "the full-ref-list layer left no record of running in THIS push, so the push "
                "is only partly checked -- and this layer cannot see more than one of its "
                "refs. Run `make hooks`.",
            )
        )

    remote = os.environ.get("PRE_COMMIT_REMOTE_NAME", "")
    destination = os.environ.get("PRE_COMMIT_REMOTE_BRANCH", "")
    if not remote or not destination:
        refusals.append(
            Refusal(
                "no-push-context",
                "pre-commit exported no remote/branch for this push, so the one ref this layer "
                "can see cannot be identified either.",
            )
        )
    else:
        seen = PushRef(
            local_ref=os.environ.get("PRE_COMMIT_LOCAL_BRANCH", ""),
            local_sha=os.environ.get("PRE_COMMIT_TO_REF", ""),
            remote_ref=destination,
            remote_sha=os.environ.get("PRE_COMMIT_FROM_REF", ""),
        )
        refusals.extend(
            adjudicate([seen], published=published_refs(repo, remote), repo=repo, remote=remote)
        )
    return _report(refusals, layer="pre-commit stage")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--git-hook", action="store_true", help="authoritative: read git's stdin")
    mode.add_argument("--pre-commit-stage", action="store_true", help="second layer, partial")
    mode.add_argument("--install", action="store_true", help="wire the authoritative hook")
    parser.add_argument("--repo", type=Path, default=None, help="repository root (default: cwd)")
    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)

    repo = (args.repo or Path.cwd()).resolve()
    if args.install:
        target = install(repo)
        print(f"push-guard installed at {target}")
        chained = hooks_dir(repo) / "pre-push"
        if not os.access(chained, os.X_OK):
            print(
                f"push-guard: {chained} does not exist, so nothing will invoke the guard.\n"
                "Run `pre-commit install` (or `make hooks`) so the chain is complete.",
                file=sys.stderr,
            )
            return 1
        return 0
    if args.pre_commit_stage:
        return run_pre_commit_stage(repo=repo)
    return run_git_hook(rest, repo=repo)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
