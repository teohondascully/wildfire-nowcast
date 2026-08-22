#!/usr/bin/env python3
"""Reject attribution in a commit message, by DERIVING the rules rather than listing tells.

WHY THIS EXISTS. This repo is single author. One commit shipped an attribution
trailer and a session URL to the public remote and was force-pushed out. The rule
against it was written down, in a file, three hundred lines from where it was
needed, and the fix that was reached for first was "remember to paste the rule
into every dispatch". That is a hand-maintained instrument, and this project has
now closed four separate defects of exactly that shape by deriving the check
instead of maintaining it. So: no list of vendors, no list of assistant names, no
list of trailer keys. Six rules, five of which have zero maintained entries.
Rule 6 is the odd one out and says so: it is a typographic rule, not an
attribution rule, and it is here because this is the file git already runs.

RULE 1  NO TRAILER BLOCK AT ALL.
    Adjudicated by ``git interpret-trailers --parse``, i.e. by git's own grammar,
    not by a regex of mine reimplementing it. A producer and a verifier computing
    one fact through two different implementations is how this repo's own
    contract checker went wrong once already. No commit reachable from any ref
    carries a parsed trailer (measured over the whole history before adoption),
    so "the trailer block is empty" is a description of this history, not an
    aspiration, and it is now enforced. Nothing is allow-listed: a co-authorship
    trailer, a sign-off trailer and a tool's session trailer all fail for the
    same reason, which is that they are trailers, and none of them is named
    here.

RULE 2  NO IDENTITY OTHER THAN THE COMMITTING ONE.
    Any e-mail address in the message that is not one of the identities git
    itself reports for this repository. The allowed set is READ from
    ``git config`` and from the history's own author and committer fields; it is
    never typed in. A tool's no-reply address fails because it is not an address
    this repository commits under, not because anyone wrote it down.

RULE 3  NO URL OUTSIDE THIS REPOSITORY'S OWN REMOTES.
    The allowed hosts are the hosts of ``git remote -v``. With no remote
    configured the allowed set is empty and every URL fails, which is the safe
    direction for a guard. A session URL fails because its host is not a host
    this repository pushes to.

RULE 4  NO PICTOGRAPHS OR INVISIBLE FORMATTING.
    Non-ASCII characters in the Unicode symbol and format categories. Prose uses
    letters, digits and punctuation; a robot pictograph in a commit message is a
    badge, not prose. Measured against the whole history before being adopted:
    the only non-ASCII characters in any message are dash punctuation (category
    Pd, rule 6 below) and the middle dot (category Po, 5 occurrences, not
    covered by any rule).

RULE 5  ATTRIBUTION CONSTRUCTIONS. THE ONE MAINTAINED SURFACE, DECLARED AS SUCH.
    Rules 1 to 4 catch every attribution that arrives with STRUCTURE, which is
    every form a tool actually emits: a trailer, an identity, a link, a badge.
    They do not catch a plain English sentence in the body that credits someone.
    I could not derive that one, and the honest answer is to say so and make the
    maintained surface as small and as slow-moving as possible: these patterns
    are ENGLISH ATTRIBUTION CONSTRUCTIONS plus generic ROLE NOUNS. There is no
    vendor name and no product name in this file, deliberately, because the set
    of vendors is open and grows without warning while the set of ways to say
    "someone else helped" is closed and centuries old. A vendor list would be
    the allow-list defect wearing its fifth costume.

RULE 6  NO TYPOGRAPHIC DASH.
    Any non-ASCII character in the Unicode dash category, ``Pd``. This is the
    one rule here that is not about attribution, and it is stated as a separate
    rule rather than folded into rule 4 so that nobody reads it as one.

    WHY IT EXISTS. The typographic dash is the largest single machine-writing
    signature on this repository's public surface, an order of magnitude more
    common than every other tell put together. Six of them are already in four
    published commit messages, two of those in subject lines. Published commit
    messages cannot be edited without rewriting history, and rewriting this
    history was measured and rejected: it would orphan the published commits
    while leaving every one of them fetchable by sha, which multiplies the
    exposure it is meant to remove. So those six are permanent. This rule cannot
    repair them and does not pretend to. It exists so that the seventh is
    impossible rather than merely discouraged.

    WHY A CATEGORY AND NOT A CHARACTER. The en dash and the horizontal bar are
    the same tell wearing a different code point, and pinning U+2014 alone would
    be the allow-list defect again at length one. ASCII hyphen-minus is category
    ``Pd`` as well, which is why the test is ``ord(char) > 127``: the rule is
    about the TYPOGRAPHIC dash, not about dashes. Write ``-`` instead.

    WHAT THIS RULE DELIBERATELY DOES NOT COVER, STATED SO IT IS NOT DISCOVERED.
    The history also carries 5 middle dots (U+00B7, category ``Po``). They are
    the same class of tell and they are not covered, because the requirement
    this rule was written for named the dash. The count is recorded here so
    that widening it later is a decision with a number attached rather than a
    rediscovery.

TWO LAYERS, BECAUSE ONE IS KNOWN TO BE INSUFFICIENT. This module is the engine of
both. Layer (a) is the ``commit-msg`` hook wired through ``.pre-commit-config.yaml``
so it arrives with ``make install``; it can be defeated by ``git commit
--no-verify``, and it was, by the maintainer, during the repair that motivated
this file. Layer (b) is ``tests/test_hygiene.py``, which scans every commit
reachable from every ref and therefore sees exactly what ``--no-verify`` let
through. Neither layer is scoped to a window: the audit that missed the original
defect was scoped to the commits it expected to be dirty, which is the same
defect as an allow-list.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ATTRIBUTION_CONSTRUCTIONS",
    "Finding",
    "commit_messages",
    "committer_identities",
    "is_shallow",
    "own_emails",
    "own_remote_hosts",
    "parsed_trailers",
    "scan_corpus",
    "scan_message",
    "single_identity_violations",
    "strip_git_comments",
]

# --------------------------------------------------------------------------
# Rule 5's patterns. THE ONLY MAINTAINED LIST IN THIS FILE.
# `tests/test_hygiene.py` pins these keys, so growing the list is a visible edit
# with a diff someone has to justify, rather than a quiet widening.
# --------------------------------------------------------------------------

#: Generic role nouns. Not vendors, not products: the CATEGORY of thing that a
#: harness credits. This set does not need an edit when a new tool appears.
_ROLE_NOUN = r"(?:ai|assistant|agent|bot|llm|chatbot|language\s+model)"

ATTRIBUTION_CONSTRUCTIONS: Mapping[str, str] = {
    # "Co-authored", "Co Authored By", "coauthored". Also matches the trailer
    # key, which is fine: two independent rules catching one tell is the design.
    "co-authorship": r"co[-\s]?authored",
    "sign-off": r"signed[-\s]?off[-\s]?by",
    # "generated by an assistant", "written with the help of a bot", and so on:
    # a production verb, an agent preposition, and a role noun close behind it.
    "credited-to-an-agent": (
        r"\b(?:generated|authored|written|created|produced|drafted|assisted)"
        r"\s+(?:with|by)\b[^\n]{0,32}?\b" + _ROLE_NOUN + r"\b"
    ),
    "assistance-of": r"\bwith\s+(?:the\s+)?(?:help|assistance|aid)\s+of\b",
    "on-behalf-of": r"\bon\s+behalf\s+of\b",
    # The branded footer form, without naming a brand: a production verb and an
    # agent preposition on the SAME LINE as a link. Ordinary prose in this repo
    # does not credit a hyperlink; a tool's signature line always does.
    "credited-to-a-link": (
        r"\b(?:generated|authored|written|created|produced|made)\s+(?:with|by)\b"
        r"[^\n]*(?:\]\(|[a-zA-Z][a-zA-Z0-9+.-]*://)"
    ),
}

_CONSTRUCTION_RE = {k: re.compile(v, re.IGNORECASE) for k, v in ATTRIBUTION_CONSTRUCTIONS.items()}

# --------------------------------------------------------------------------
# Derived rules. No entries to maintain.
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>()\[\]\"'`]+")

#: Unicode general categories that carry no prose at all: symbols and invisible
#: formatting. Rule 4.
_BADGE_CATEGORIES = frozenset({"So", "Sk", "Sm", "Cf"})

#: Rule 6, kept separate from the set above because it is a different claim. A
#: pictograph is not prose; a typographic dash IS prose, and is banned for a
#: reason that has nothing to do with attribution. Folding it into
#: ``_BADGE_CATEGORIES`` would make one rule carry two arguments and would make
#: the finding say "is a symbol, not prose", which is false of a dash.
_DASH_CATEGORY = "Pd"

_SCISSORS = "------------------------ >8 ------------------------"


@dataclass(frozen=True)
class Finding:
    """One reason a commit message is rejected."""

    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.detail}"


def strip_git_comments(message: str, comment_char: str = "#") -> str:
    """Drop what git itself drops: comment lines and everything below the scissors.

    A ``commit-msg`` hook is handed the file as the editor left it. Scanning text
    git is about to discard would reject messages that never existed, and a guard
    with false positives is a guard someone turns off.
    """
    head = message.split(_SCISSORS, 1)[0]
    kept = [ln for ln in head.splitlines() if not ln.startswith(comment_char)]
    return "\n".join(kept)


def _git(args: list[str], repo: Path, *, stdin: str | None = None) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def parsed_trailers(message: str, repo: Path) -> list[str]:
    """Git's OWN verdict on whether this message has a trailer block."""
    return [
        ln
        for ln in _git(["interpret-trailers", "--parse"], repo, stdin=message).splitlines()
        if ln.strip()
    ]


def own_emails(repo: Path) -> frozenset[str]:
    """The identities git reports for this repository, lowercased.

    Read, never typed: ``git config user.email`` plus every author and committer
    address already in the history. A repository with one author therefore has
    one allowed address and did not have to declare it anywhere.
    """
    found: set[str] = set()
    for key in ("user.email", "author.email", "committer.email"):
        try:
            value = _git(["config", "--get", key], repo).strip()
        except subprocess.CalledProcessError:
            continue
        if value:
            found.add(value.lower())
    try:
        log = _git(["log", "--all", "--format=%ae%n%ce"], repo)
    except subprocess.CalledProcessError:  # pragma: no cover - a repo with no commits
        log = ""
    found.update(line.strip().lower() for line in log.splitlines() if line.strip())
    return frozenset(found)


def own_remote_hosts(repo: Path) -> frozenset[str]:
    """Hosts this repository actually pushes to, from ``git remote -v``.

    With no remote the set is EMPTY and rule 3 rejects every URL. That is the
    direction a guard should fail in.
    """
    try:
        raw = _git(["remote", "-v"], repo)
    except subprocess.CalledProcessError:  # pragma: no cover - not a git repo
        return frozenset()
    hosts: set[str] = set()
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        url = parts[1]
        match = re.match(r"(?:[a-zA-Z][a-zA-Z0-9+.-]*://)?(?:[^@/]+@)?([^/:]+)", url)
        if match:
            hosts.add(match.group(1).lower())
    return frozenset(hosts)


def _url_host(url: str) -> str:
    match = re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://(?:[^@/]+@)?([^/:?#]+)", url)
    return match.group(1).lower() if match else ""


def scan_message(
    message: str,
    *,
    repo: Path,
    allowed_emails: Iterable[str] = (),
    allowed_hosts: Iterable[str] = (),
) -> list[Finding]:
    """Every reason this message is rejected. Empty list means clean.

    All five rules always run. The function does not stop at the first finding,
    because "the defect I targeted is gone" is not the same claim as "the message
    is clean", and repairing this class against only the first symptom has
    already produced one broken commit message in this repo.
    """
    text = strip_git_comments(message)
    emails = {e.lower() for e in allowed_emails}
    hosts = {h.lower() for h in allowed_hosts}
    findings: list[Finding] = []

    for trailer in parsed_trailers(text, repo):
        findings.append(
            Finding("trailer", f"the message ends in a trailer block: {trailer.strip()!r}")
        )

    for address in _EMAIL_RE.findall(text):
        if address.lower() not in emails:
            findings.append(
                Finding("foreign-identity", f"{address!r} is not an identity of this repository")
            )

    for url in _URL_RE.findall(text):
        if _url_host(url) not in hosts:
            findings.append(
                Finding("foreign-url", f"{url!r} points outside this repository's own remotes")
            )

    for char in text:
        if ord(char) <= 127:
            continue
        category = unicodedata.category(char)
        if category in _BADGE_CATEGORIES:
            findings.append(
                Finding(
                    "badge-character",
                    f"U+{ord(char):04X} ({category}) is a symbol, not prose",
                )
            )
        elif category == _DASH_CATEGORY:
            name = unicodedata.name(char, "an unnamed dash")
            findings.append(
                Finding(
                    "em-dash",
                    f"U+{ord(char):04X} ({name}) is a typographic dash. Write '-'",
                )
            )

    for name, pattern in _CONSTRUCTION_RE.items():
        for hit in pattern.findall(text):
            snippet = hit if isinstance(hit, str) else str(hit)
            findings.append(Finding(name, f"attribution construction: {snippet.strip()!r}"))

    return findings


def is_shallow(repo: Path) -> bool:
    """A shallow clone cannot support a full-history claim, and must say so."""
    return _git(["rev-parse", "--is-shallow-repository"], repo).strip() == "true"


def commit_messages(repo: Path) -> dict[str, str]:
    """``{sha: message}`` for EVERY commit reachable from EVERY ref.

    Not a window, not a branch, not "since the last tag". The audit that missed
    the original defect was scoped to the commits it expected to be dirty.
    """
    # `%x00` is expanded by git into a NUL in its OUTPUT. A printable separator
    # would be a string a commit message could contain, and the one record that
    # could hide inside another is the one an attacker (or a paste) would use.
    raw = _git(["log", "--all", "--reverse", "--format=%x00%H%n%B"], repo)
    out: dict[str, str] = {}
    for chunk in raw.split("\x00"):
        if not chunk.strip():
            continue
        sha, _, body = chunk.partition("\n")
        out[sha.strip()] = body
    return out


def scan_corpus(
    messages: Mapping[str, str],
    *,
    repo: Path,
    allowed_emails: Iterable[str] = (),
    allowed_hosts: Iterable[str] = (),
) -> dict[str, list[Finding]]:
    """``{sha: findings}`` for every message with at least one finding.

    Takes the corpus as an ARGUMENT rather than reading it, so a test can plant a
    defective message in a hypothetical history and prove the scanner reports it
    without having to dirty the repository and restore it.
    """
    emails = tuple(allowed_emails)
    hosts = tuple(allowed_hosts)
    out: dict[str, list[Finding]] = {}
    for sha, message in messages.items():
        findings = scan_message(message, repo=repo, allowed_emails=emails, allowed_hosts=hosts)
        if findings:
            out[sha] = findings
    return out


def single_identity_violations(rows: Iterable[tuple[str, str, str, str]]) -> list[str]:
    """Every author/committer identity beyond the first, for a single-author repo.

    Pure, for the same reason ``scan_corpus`` is: the planted defect must be
    provable without inventing a second contributor.
    """
    identities = sorted(
        {(name, mail) for name, mail, _, _ in rows} | {(name, mail) for _, _, name, mail in rows}
    )
    if len(identities) <= 1:
        return []
    return [f"{name} <{mail}>" for name, mail in identities[1:]]


def committer_identities(repo: Path) -> set[tuple[str, str, str, str]]:
    """``{(author name, author email, committer name, committer email)}`` over all refs."""
    raw = _git(["log", "--all", "--format=%an%x1f%ae%x1f%cn%x1f%ce"], repo)
    rows: set[tuple[str, str, str, str]] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) == 4:
            rows.add((parts[0], parts[1], parts[2], parts[3]))
    return rows


def main(argv: list[str] | None = None) -> int:
    """``commit-msg`` hook entry point. Argument is the message file git supplies."""
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("message_file", type=Path, nargs="?")
    parser.add_argument(
        "--repo", type=Path, default=Path.cwd(), help="repository root (default: cwd)"
    )
    args = parser.parse_args(argv)

    if args.message_file is None:
        print("commit-guard: no commit message file was supplied", file=sys.stderr)
        return 2
    repo = args.repo.resolve()
    message = args.message_file.read_text(encoding="utf-8", errors="replace")

    findings = scan_message(
        message,
        repo=repo,
        allowed_emails=own_emails(repo),
        allowed_hosts=own_remote_hosts(repo),
    )
    if not findings:
        return 0

    print(
        "commit-guard: this commit message is rejected. This repository is single\n"
        "author, its history carries no attribution on any commit, and its public\n"
        "surface carries no typographic dash.\n",
        file=sys.stderr,
    )
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    print(
        "\nRemove the offending lines and commit again. `--no-verify` will get past\n"
        "this hook and will NOT get past tests/test_hygiene.py, which scans every\n"
        "commit reachable from every ref.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
