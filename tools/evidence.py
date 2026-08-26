"""TRACKED EVIDENCE versus SOURCE THAT CITES: one rule, named once.

WHY THIS MODULE EXISTS
----------------------
``cited_paths`` and ``cited_runs`` both enforce the same promise: a path written
into published source must be a path a public reader can open. To do that they
walk **tracked files**, and until I33 they decided which tracked files carry that
obligation with ``rel.startswith("runs/")``.

The reasoning behind that line was right and is preserved verbatim below. What
was wrong was the KEY. "Is this file evidence?" is a property of the file, and it
was being answered by asking where the file happens to live. So the moment the
G6 evidence was force-added under ``reports/figures/``, those artifacts stopped
being outputs and silently became CITERS: every path-shaped string inside
them turned into a citation from published source, pointing at the *rest* of the
evidence, which was still untracked. **Tracking the leaves of a provenance graph
pulls their parents in behind them.** Seven tests, measured one artifact at a
time with control and restore, not inferred.

THE DISTINCTION, STATED ONCE
----------------------------
``SOURCE``    code and prose a human reads. A path inside it is a PROMISE the
              repository makes to a reader: open this. Enforced.
``EVIDENCE``  a machine-written artifact tracked so a reader can OPEN it and
              recompute a published number. A path inside it is a PROVENANCE
              RECORD of what some run read and wrote. It is a statement about the
              past, it may not be edited to look better, and it is not a promise.
              Reported, never enforced.

Evidence is still enforced as a TARGET: a citation *of* an evidence file must
resolve, which is the entire reason it was tracked. What changes is that it is
not scanned as a CITER.

THIS TIER ADDS OBLIGATIONS; IT REMOVES NONE
-------------------------------------------
The alternative on the table was to declare each unresolved provenance string in
``cited_paths.DEBT`` / ``cited_runs.EXEMPT``. That is a LOOSENING: it says an
unresolvable citation is acceptable, one string at a time, forever. This says
something narrower and checkable instead - the file is not a citer at all - and
then charges rent for the privilege. Every entry in :data:`EVIDENCE_FILES` must:

1. **be TRACKED.** An entry naming a file that is not in the index is a phantom
   declaration, and this repository has paid for phantoms twice (ADR-181).
2. **not be SOURCE.** No ``.py``, ``.md``, ``.yaml``... Without this clause the
   tier is a universal escape hatch: any module could be made invisible to every
   citation scan by listing it here. This is the clause that makes "nothing gets
   loosened" true rather than merely intended.
3. **be CITED by something that is not itself evidence.** Evidence exists to be
   read from the published surface. An entry nothing cites is not evidence, it
   is an artifact somebody committed, and the tier must not become a place to
   put files that failed a check.

All three are checked by :func:`evidence_problems` on every run, and the
membership itself is pinned as a SET rather than a count (ADR-154) in
``tests/test_evidence_tier.py``.

ONE ENFORCEMENT POINT, NOT TWO
------------------------------
``cited_paths`` calls :func:`evidence_problems`; ``cited_runs`` calls only
:func:`is_evidence`. That asymmetry is deliberate. Both scanners must AGREE about
what a citer is, so both import the predicate; but a rule checked in two places
is a rule that can pass in one and fail in the other, and the report a human
reads is ``cited_paths``. C0 applies to checks as much as to code.

ADDING AN ENTRY IS ONE CHANGE, NOT TWO
--------------------------------------
Obligation (1) means a declaration written before the ``git add -f`` lands leaves
the tree RED between the two. That is intended and it is the same shape as I30's
same-commit rule: the declaration is the CLAIM that a reader can open the file,
so it must not exist while the claim is false. The failure message names the
exact command, so the coupling explains itself to whoever hits it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

#: Whole directories that are evidence. ``runs/`` is here UNCHANGED, and the
#: original reasoning is kept because it is the reasoning for the whole tier:
#: "A tracked run record is EVIDENCE: the paths inside it are what that run read
#: and wrote, written by the code, and they may not be edited to look better. It
#: is the public READING surface that carries citation obligations, and a
#: machine-written record is not one."
EVIDENCE_PREFIXES: Final[tuple[str, ...]] = ("runs/",)

#: Individually declared evidence artifacts, path -> why a reader needs it.
#:
#: A SET, pinned in both directions by the test module. Adding one is a decision
#: with a reason; removing one silently is a regression in what a clone can
#: recompute.
EVIDENCE_FILES: Final[Mapping[str, str]] = {
    "reports/figures/s14g5/creek_cost_projection.json": (
        "S14/G5: the projected ELMFIRE cost for creek, which is the block that "
        "did not complete. The G6 page states that projection as a number, so a "
        "reader who cannot open this file cannot check the one block whose "
        "absence the ELMFIRE result has to argue around."
    ),
    "reports/figures/s16/creek_episode.json": (
        "S16: the creek episode record behind the late-growth over-prediction "
        "figures. It carries the per-window cell counts the page quotes, and it "
        "is the only artifact from which the 14x-30x range can be recomputed."
    ),
}

#: Extensions that may NEVER be classified as evidence. See obligation (2): this
#: is the clause that stops the tier being an escape hatch for source.
FORBIDDEN_AS_EVIDENCE: Final[tuple[str, ...]] = (
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".txt",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".bash",
    ".zsh",
    ".mk",
    ".html",
)


def is_evidence(rel: str, files: Mapping[str, str] | None = None) -> bool:
    """Is this tracked path evidence rather than source that cites?

    ``files`` is injectable for the same reason ``cited_paths.enumerate_references``
    injects ``declared`` and ``debt``, and I got this wrong first: reading the
    module table unconditionally made every SYNTHETIC corpus report THIS
    repository's evidence as missing. A check that answers a question about a
    tree it was not given is not a check, and that exact sentence was already
    written twenty lines above the code I copied.

    The ONLY place this question is answered. Both scanners call it, so the two
    cannot drift into disagreeing about what a citer is - which is exactly the
    drift that let ``reports/`` evidence be scanned as source while ``runs/``
    evidence was not.
    """
    table = EVIDENCE_FILES if files is None else files
    return rel in table or rel.startswith(EVIDENCE_PREFIXES)


def evidence_problems(
    tracked: Iterable[str],
    citers_of: Mapping[str, Iterable[str]],
    files: Mapping[str, str] | None = None,
) -> list[str]:
    """The three obligations, checked. Empty list means the tier is honest.

    ``citers_of`` maps a path to the tracked files that mention it, which the
    callers already compute. Passing it in rather than re-walking keeps this
    module free of git and of the filesystem, so it can be exercised on a
    synthetic corpus.
    """
    table = EVIDENCE_FILES if files is None else files
    index = set(tracked)
    out: list[str] = []
    for rel in sorted(table):
        if rel not in index:
            out.append(
                f"{rel}: declared as EVIDENCE and is NOT TRACKED. A declaration that "
                "names nothing is a phantom: it silences a scan for a file no reader "
                f"has. THE DECLARATION AND THE ADD ARE ONE CHANGE -> `git add -f {rel}`, "
                "in the same commit as this entry."
            )
            continue
        if rel.endswith(FORBIDDEN_AS_EVIDENCE):
            out.append(
                f"{rel}: declared as EVIDENCE but its extension is SOURCE. Source may "
                "never enter this tier - that is how a module would be made invisible "
                "to every citation scan at once."
            )
        readers = [c for c in citers_of.get(rel, ()) if not is_evidence(c, table)]
        if not readers:
            out.append(
                f"{rel}: declared as EVIDENCE and cited by NO non-evidence file. Evidence "
                "exists to be read from the published surface; an entry nothing cites is "
                "an artifact somebody committed, and this tier is not a place to put one."
            )
    return out
