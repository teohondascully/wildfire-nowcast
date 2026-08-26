#!/usr/bin/env python3
"""Enumerate every ``runs/`` path the PUBLISHED tree cites, and say which resolve.

WHY THIS IS A WALK AND NOT A PATTERN
------------------------------------
The first enumeration of this class was ``runs/[A-Za-z0-9_]*\\.json``. It found
twelve of twenty-one, and it could not have found more: the character class
excludes ``/`` and ``-`` so it cannot reach into a subdirectory, and the literal
``.json`` suffix excludes every other extension. Both misses are STRUCTURAL. The
four ``runs/baselines-*/results.json`` failed on the first, the five
``runs/_*.py`` analysis scripts failed on the second, and the five are the half
that matters: they are the code that produced published numbers, cited from
public source and absent from the repository.

So this module states no shape at all beyond the literal segment ``runs/``. It
takes the maximal path-like run of characters that follows, and CLASSIFIES the
result rather than filtering it. A citation with an unanticipated shape lands in
a bucket and is reported; it cannot fall out of the scan.

WHAT IS AN OBLIGATION AND WHAT IS ONLY REPORTED
-----------------------------------------------
An obligation is a citation a reader would try to OPEN: a token with a file
extension, cited by a tracked file that is not itself an artifact under
``runs/``. Everything else is reported and never enforced:

``dir_or_prefix``   a token with no extension. Every one of these today is a run
                    DIRECTORY inside a copy-pasteable CLI example, or a
                    deliberately fake name in a test. Tracking a run directory
                    means tracking checkpoints, which is the 92 MB this
                    repository exists without.
``artifact tier``   a citation whose only citers are themselves artifacts under
                    ``runs/``. This is the transitive closure of the evidence
                    chain, not the public reading surface, and it is reported
                    with its size so the decision to stop is a number.

CLASSIFICATION IS DERIVED FROM GIT, NOT FROM THE FILESYSTEM
-----------------------------------------------------------
Whether a citation resolves is answered by ``git ls-files``, not by
``Path.exists``. A clone has none of the untracked evidence, so a filesystem
answer would differ between this machine and CI, and the check would then be
enforcing something different in the place it actually gates. Disk presence is
reported as extra information where it is available and is never a verdict.

WHAT THIS WALK CANNOT SEE, PRINTED RATHER THAN REMEMBERED
---------------------------------------------------------
It reads SOURCE BYTES, not values. A path ASSEMBLED at run time is invisible to
it, and an assembled path is not a rare shape here: it is how every run
directory in this repository is named. There is no fix inside a regex - seeing
these requires evaluating the program - so the honest move is to say so at the
point of use. :func:`assembled_sites` counts them and :func:`_report` prints the
count on every run, next to the findings and never mixed into them. That
placement is the whole point: this blindness was found by a lead who went
looking, and the next lead should be TOLD.

Assembly degrades in two ways, and the second is worse:

* the token vanishes outright, when the interruption follows the separator;
* the token TRUNCATES, when a literal stem precedes the placeholder. A stem
  that truncates loses its extension with its tail, so it lands in
  ``dir_or_prefix`` and is never enforced - it is reported as a directory
  citation, which is a category this module deliberately does not check.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence import is_evidence  # noqa: E402

#: The only shape assumption in this module: the literal segment, then the
#: maximal run of characters a POSIX path may use. No extension, no depth, no
#: character class narrower than "what a path is made of".
RUNS_TOKEN = re.compile(r"runs/[A-Za-z0-9._][A-Za-z0-9._/-]*")

#: The pattern this module REPLACES, kept so the control can measure both on the
#: same corpus rather than quoting a remembered number.
#:
#: **FROZEN. NOT TO BE EDITED (ADR-181 (5b)).** It is a record of what the old
#: reader did, and editing it so a test passes falsifies the very history the
#: differential test exists to measure.
#:
#: **IT HAS A FOURTH FAILURE MODE AND IT IS WORSE THAN THE THREE BLIND SPOTS.**
#: On a subdirectory or a non-``.json`` extension it emits NOTHING - it is blind,
#: and blindness OMITS. On a gzipped record it emits a TRUNCATED name:
#:
#:     runs/<something>.json.gz   ->   runs/<something>.json
#:
#: That name has never existed. **A phantom ASSERTS**, and because it ends
#: ``.json`` it is indistinguishable from a true finding by anything downstream.
#: This is why the I31 invariant had to gain the clause *"every leftover resolves
#: to NOTHING"*: without it, restating the invariant would be indistinguishable
#: from moving a bar.
SUPERSEDED_PATTERN = re.compile(r"runs/[A-Za-z0-9_]*\.json")

#: Extensions that make a token a FILE citation, i.e. something a reader opens.
#: A token without one names a directory or a prefix and is reported only.
#:
#: ``.gz`` ADDED AT I31 (ADR-181 (5a)), AND IT IS THE SUBSTANTIVE HALF OF THAT
#: REPAIR. Without it a gzipped token has no recognised extension, so it is
#: classified ``dir_or_prefix`` and NEVER ENFORCED. Measured on the tree
#: that shipped the defect: **34 gzipped records under ``runs/``, of which 0 were
#: checked for existence** - including the four the G6 headline rests on. A
#: cloner sent to a missing gzipped artifact was told nothing, which is precisely
#: the failure this module exists to prevent.
#:
#: **IT DOES NOT, ON ITS OWN, RESTORE THE SUPERSET INVARIANT, AND THAT WAS THE
#: RECOMMENDED FIX.** :data:`SUPERSEDED_PATTERN` truncates by construction and
#: never reads this tuple - the suffix set governs only the filter applied
#: afterwards. Measured before and after in a detached worktree with a real
#: citation staged: ``superseded < replacement`` is ``False`` both times, and the
#: leftover is unchanged. Two defects were being treated as one. See the PHANTOM
#: note on :data:`SUPERSEDED_PATTERN`.
FILE_SUFFIXES = (".gz", ".json", ".py", ".csv", ".md", ".png", ".pt", ".txt", ".yaml", ".zarr")

#: A ``runs/`` string whose character run is INTERRUPTED by an interpolation or a
#: concatenation, i.e. a citation this walk cannot resolve because the path does
#: not exist until the program runs. Reported, never enforced: there is nothing
#: to enforce, since the tool cannot know what the assembled name will be.
#:
#: ``stem`` is the literal part between the separator and the interruption. It
#: decides WHICH of the two degradations applies, and it is captured rather than
#: recomputed for a reason learned the hard way at I31: the first version of the
#: printer classified fragments by testing them against a spelled-out marker,
#: which made this module match ITSELF and inflated its own count by one. A
#: detector that has to write down the thing it detects becomes its own first
#: finding. Reading the group costs nothing and cannot do that.
ASSEMBLED_CITATION = re.compile(r"""runs/(?P<stem>[A-Za-z0-9._/-]*)(?:\{|%\(|%[sdr]|["']\s*\+)""")


class AssemblySite(NamedTuple):
    """One place a ``runs/`` path is built at run time rather than written down."""

    path: str
    line: int
    fragment: str
    stem: str

    @property
    def truncates(self) -> bool:
        """True when a literal stem survives the interruption.

        The worse of the two degradations. A vanished token is absent from the
        report and absent from the class; a truncated one is PRESENT, shorn of
        its extension, and is therefore filed as a directory citation - a
        category this module deliberately never enforces. It looks handled.
        """
        return bool(self.stem)


def assembled_sites(root: Path | None = None) -> list[AssemblySite]:
    """Every place a ``runs/`` citation is assembled at run time.

    Artifacts under ``runs/`` are skipped for the same reason they are skipped
    everywhere else here: they are evidence, not the public reading surface.
    """
    base = root or Path(__file__).resolve().parents[1]
    out: list[AssemblySite] = []
    for rel in tracked_files(base):
        if is_evidence(rel):
            continue
        path = base / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in ASSEMBLED_CITATION.finditer(line):
                out.append(AssemblySite(rel, lineno, match.group(0), match.group("stem")))
    return out


#: Citations that are cited, are files, and stay OUT of the tree, each with the
#: reason and a measurement that makes the reason falsifiable. ``tell_count`` is
#: the number of internal coordination-role occurrences the file carries; it is
#: asserted only where the file is present, and the day one of these reaches 0
#: the exemption has lost its reason and the test says so.
EXEMPT: dict[str, tuple[str, int]] = {
    "runs/baselines-20260808-095003/results.json": (
        "120 internal-role occurrences across these four, frozen inside run-time "
        "`note` strings that may not be edited because they are evidence",
        5,
    ),
    "runs/baselines-20260809-035037/results.json": ("same class as the above", 25),
    "runs/baselines-20260809-073414/results.json": ("same class as the above", 39),
    "runs/baselines-20260809-102243/results.json": ("same class as the above", 51),
}


@dataclass(frozen=True)
class Citation:
    """One ``runs/`` token, with who cites it and what it is."""

    token: str
    citers: tuple[str, ...]
    tracked: bool
    is_file_token: bool
    on_disk: bool

    @property
    def from_source(self) -> bool:
        """Cited by at least one tracked file that is not itself EVIDENCE.

        [I33] Was ``not c.startswith("runs/")``. The reasoning was right and the
        KEY was wrong: whether a file carries citation obligations is a property
        of the file, not of the directory it happens to sit in. See
        :mod:`tools.evidence`.
        """
        return any(not is_evidence(c) for c in self.citers)

    @property
    def kind(self) -> str:
        if not self.is_file_token:
            return "dir_or_prefix"
        if self.tracked:
            return "tracked"
        if self.token in EXEMPT:
            return "exempt"
        return "UNRESOLVABLE"


@dataclass
class Enumeration:
    citations: tuple[Citation, ...] = ()
    tracked_files: tuple[str, ...] = ()
    problems: list[str] = field(default_factory=list)

    def of_kind(self, kind: str, *, source_only: bool = True) -> list[Citation]:
        return [
            c for c in self.citations if c.kind == kind and (c.from_source if source_only else True)
        ]


def tracked_files(root: Path | None = None) -> list[str]:
    """The published tree, which is exactly the tracked tree."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=root or Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(p for p in out.stdout.splitlines() if p)


def tokens_in(text: str, pattern: re.Pattern[str] = RUNS_TOKEN) -> set[str]:
    """Every ``runs/`` token in one blob, with trailing punctuation trimmed."""
    return {m.rstrip("./-") for m in pattern.findall(text)}


def enumerate_citations(root: Path | None = None) -> Enumeration:
    """Every ``runs/`` citation in the tracked tree, classified."""
    base = root or Path(__file__).resolve().parents[1]
    files = tracked_files(base)
    tracked = set(files)
    citers: dict[str, set[str]] = {}
    for rel in files:
        path = base / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        for token in tokens_in(text):
            citers.setdefault(token, set()).add(rel)

    found = []
    for token, who in sorted(citers.items()):
        found.append(
            Citation(
                token=token,
                citers=tuple(sorted(who)),
                tracked=token in tracked,
                is_file_token=token.endswith(FILE_SUFFIXES),
                on_disk=(base / token).exists(),
            )
        )
    result = Enumeration(citations=tuple(found), tracked_files=tuple(files))
    for c in result.citations:
        if c.kind == "UNRESOLVABLE" and c.from_source:
            result.problems.append(
                f"{c.token}: cited by {', '.join(c.citers)} and neither tracked nor "
                "listed in EXEMPT. A public reader cannot open it."
            )
    for token in EXEMPT:
        if token in tracked:
            result.problems.append(
                f"{token}: listed in EXEMPT but is now TRACKED. Remove the exemption; "
                "a stale exemption is how an allow-list starts."
            )
    return result


def _report(enum: Enumeration, root: Path | None = None) -> str:
    lines = []
    for kind in ("tracked", "exempt", "UNRESOLVABLE", "dir_or_prefix"):
        group = enum.of_kind(kind)
        lines.append(f"{kind}: {len(group)}")
        for c in group:
            mark = "on disk" if c.on_disk else "absent"
            lines.append(f"    {c.token}  [{mark}]  <- {c.citers[0]}")
    artifact_tier = [
        c for c in enum.citations if not c.from_source and c.is_file_token and not c.tracked
    ]
    lines.append(f"artifact tier (cited only by tracked runs/ artifacts): {len(artifact_tier)}")
    lines.append(_limits_block(root))
    return "\n".join(lines)


def _limits_block(root: Path | None = None) -> str:
    """What the walk cannot see, printed on every run beside what it can.

    Kept OUT of :attr:`Enumeration.problems` on purpose. These are not defects
    and they never set the exit status; they are the boundary of the claim, and
    a boundary written only in a docstring is one the next reader rediscovers.
    """
    sites = assembled_sites(root)
    files = sorted({s.path for s in sites})
    truncating = [s for s in sites if s.truncates]
    out = [
        "",
        "SCAN LIMITS (not findings; they do not affect the exit status)",
        f"    runtime-assembled citations: {len(sites)} sites in {len(files)} tracked files",
        "    This walk reads source bytes, not values, so a path built at run time is",
        "    INVISIBLE to it. Seeing these would need the program evaluated, not read.",
        f"    of those, {len(truncating)} truncate rather than vanish: a literal stem",
        "    survives, loses its extension along with the tail, and is then filed as a",
        "    directory citation - the one category here that is never enforced:",
    ]
    out.extend(f"        {s.path}:{s.line}  {s.fragment}" for s in truncating)
    out.append(f"    files with assembled citations: {', '.join(files) or 'none'}")
    return "\n".join(out)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    enum = enumerate_citations()
    if not args.quiet:
        print(_report(enum))
    if enum.problems:
        print("\nUNRESOLVABLE CITATIONS:", file=sys.stderr)
        for p in enum.problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
