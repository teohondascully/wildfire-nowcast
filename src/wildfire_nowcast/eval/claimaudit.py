"""[M12] Sweep the TRACKED surface for quantitative claims a decision has retracted.

WHY THIS MODULE EXISTS
----------------------
``eval/stage.py`` justified a design choice by citing a decision section that a
later decision had retracted IN FULL, quoting both of its magnitudes. The
retraction reached the README and never reached the source. Deleting those two
numbers is a one-line edit; the DEFECT IS A CLASS, and this project has paid
seven times for repairing a named instance of a class instead of the class:
an enumerated scoring-module list, a fingerprint that could not tell two corpora
apart, a public-tell allowlist, a cheap null rung, a remembered attribution rule,
a shallow checkout that made a full-history scan vacuous, and a push guard whose
own remedy removed the only input that could have exercised it.

The class is: **a quantitative claim that the decision log has RETRACTED or
SUPERSEDED, still asserted in a tracked file.** Tracked means public.

WHAT IS MECHANICAL HERE AND WHAT IS NOT
--------------------------------------
The ENUMERATION is mechanical and is the whole point. Nothing in this module
encodes which numbers to look for; the retraction record is PARSED out of the
decision log at run time, so a retraction written tomorrow is covered without
anyone remembering to add it here. A hand-written list of numbers to hunt would
be the same allowlist failure one level down.

The VERDICT is not mechanical and is not attempted. Every hit is reported for a
human to rule RETRACTED / SUPERSEDED / STILL VALID / UNVERIFIABLE. A tool that
guessed would either be trusted when wrong or ignored when right.

THE TWO DETECTORS ARE INDEPENDENT ON PURPOSE
--------------------------------------------
``by_citation`` finds prose that cites a retracted section. ``by_magnitude``
finds a retracted section's own numbers, whether or not the citation travels with
them -- which is the harder case, because a number copied without its provenance
is exactly the one no reader can check. Either detector alone has a blind spot
the other covers.

SELF-SCANNING, NO SELF-EXEMPTION
--------------------------------
This module is scanned by its own sweep. Its planted-instance fixture therefore
spells the retracted magnitudes in HALVES and joins them at run time, the same
device ``tests/test_hygiene.py`` uses for its tells and for the same reason:
exempting the scanner's own path would make the file most likely to acquire a
defect the one file that could never report one.

WHERE THIS LIVES, STATED RATHER THAN ASSUMED
--------------------------------------------
It is a repo-hygiene instrument, not a scoring instrument: it computes no
quantity any gate reads, and :func:`main` returns an exit status rather than a
number. It sits in ``eval/`` because that is the tracked directory its author
owns, and the consequence is that ``scoring_code_fingerprint`` now covers it.
That over-coverage is conservative in the only direction that matters -- it can
make a fingerprint move for a harmless reason, never hold still for a harmful
one -- but ``tools/`` is where the comparable guards live and re-homing it there
is proposed, not performed.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tokenize
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = [
    "RETRACTION_MARKERS",
    "SCOPE_PATTERNS",
    "AdrSection",
    "RetractionRecord",
    "Finding",
    "parse_decision_sections",
    "build_retraction_record",
    "tracked_files",
    "in_scope",
    "prose_lines",
    "numeric_tokens",
    "citations",
    "is_distinctive",
    "scan_text",
    "scan_files",
    "self_test",
    "main",
]

# --------------------------------------------------------------------------
# the retraction record, parsed rather than remembered
# --------------------------------------------------------------------------

#: A decision RETRACTS when it says so. These are the words this log actually
#: uses, read off it rather than imagined: every one of them appears in a
#: sentence that names the section it invalidates. The list is deliberately
#: WIDE -- a false hit costs one line of a verdict table, a miss costs a public
#: over-claim -- and ``FALSIFIED`` is included even though it is common,
#: because a falsified pre-registration is a claim that must not survive in
#: source.
RETRACTION_MARKERS: Final = (
    "RETRACT",
    "RETRACTION",
    "WITHDRAW",
    "SUPERSED",
    "RESTATE",
    "NOT QUOTABLE",
    "NEVER QUOTE",
    "DO NOT QUOTE",
    "MUST NOT BE QUOTED",
    "CORRECTION TO",
    "CORRECTS",
    "AMENDMENT TO",
    "AMENDS",
    "IS WRONG",
    "WAS WRONG",
    "WAS FALSE",
    "FALSIFIED",
    "ABANDONED",
    "IS VOID",
    "OVER-CLAIM",
    "NO LONGER",
)

_MARKER_RE: Final = re.compile("|".join(re.escape(m) for m in RETRACTION_MARKERS), re.IGNORECASE)

#: ``ADR-060 (7.1)`` / ``ADR-058 (4)`` / ``ADR-042``. The part is optional; a
#: retraction that names no part invalidates the whole decision.
_CITATION_RE: Final = re.compile(r"\bADR-(\d{3})\s*\((\d+(?:\.\d+)?)\)|\bADR-(\d{3})\b")

_ADR_HEADING_RE: Final = re.compile(r"^##\s+ADR-(\d{3})\b")
_PART_HEADING_RE: Final = re.compile(r"^###\s+\((\d+(?:\.\d+)?)\)")

#: How far either side of a retraction marker a named target may sit. The log is
#: hard-wrapped at ~80 columns, so a sentence routinely spans three lines and a
#: line-scoped window would miss most targets. Measured, not guessed: at 0 the
#: record finds 0 targets, at 200 it finds the ADR-060 (1) -> ADR-058 (4) edge
#: this module was written for.
MARKER_WINDOW_CHARS: Final = 200


@dataclass(frozen=True)
class AdrSection:
    """One ``### (K)`` part of one ``## ADR-NNN`` decision, or its preamble."""

    adr: str
    part: str | None
    text: str

    @property
    def label(self) -> str:
        return f"ADR-{self.adr}" if self.part is None else f"ADR-{self.adr} ({self.part})"


@dataclass(frozen=True)
class RetractionRecord:
    """What the decision log has invalidated, and the magnitudes that went with it."""

    #: ``retracted label -> the labels that retracted it``
    tainted: Mapping[str, tuple[str, ...]]
    #: ``retracted label -> the distinctive numeric tokens its own text carries``
    void_magnitudes: Mapping[str, frozenset[str]]
    #: every section, so a target's text can be quoted back
    sections: Mapping[str, AdrSection]
    #: bare ``ADR-NNN`` targets seen beside a marker and DELIBERATELY not
    #: tainted -- carried so the narrowing is countable rather than invisible
    whole_decision_targets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def magnitude_owners(self) -> dict[str, tuple[str, ...]]:
        """``numeric token -> the retracted labels that carry it``."""
        owners: dict[str, list[str]] = {}
        for label, tokens in self.void_magnitudes.items():
            for token in tokens:
                owners.setdefault(token, []).append(label)
        return {token: tuple(sorted(labels)) for token, labels in owners.items()}


def parse_decision_sections(text: str) -> list[AdrSection]:
    """Split a decision log into ``ADR-NNN`` / ``(K)`` sections.

    Pure so the tests can feed it a hypothetical log. A line that is not under
    any ``## ADR-NNN`` heading is dropped: the file's preamble is prose about the
    file, not a decision.
    """
    sections: list[AdrSection] = []
    adr: str | None = None
    part: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if adr is not None and buf:
            sections.append(AdrSection(adr, part, "\n".join(buf)))

    for line in text.splitlines():
        heading = _ADR_HEADING_RE.match(line)
        if heading:
            flush()
            adr, part, buf = heading.group(1), None, [line]
            continue
        sub = _PART_HEADING_RE.match(line)
        if sub and adr is not None:
            flush()
            part, buf = sub.group(1), [line]
            continue
        if adr is not None:
            buf.append(line)
    flush()
    return sections


def _citation_labels(text: str) -> list[str]:
    out: list[str] = []
    for match in _CITATION_RE.finditer(text):
        if match.group(3):
            out.append(f"ADR-{match.group(3)}")
        else:
            out.append(f"ADR-{match.group(1)} ({match.group(2)})")
    return out


def citations(text: str) -> list[str]:
    """Every decision citation in a piece of prose, in order of appearance."""
    return _citation_labels(text)


def build_retraction_record(
    sections: Sequence[AdrSection], *, window: int = MARKER_WINDOW_CHARS
) -> RetractionRecord:
    """Read the log's own retraction statements and the magnitudes they void.

    A target is TAINTED when a retraction marker and a citation naming it sit
    within ``window`` characters of each other in an UNWRAPPED section. A section
    citing itself is ignored: decisions restate their own reasoning constantly,
    and treating that as a retraction would taint the whole log.

    **A TARGET MUST NAME A PART.** ``ADR-060 (1) retracts ADR-058 (4)`` taints a
    section; a marker sitting near a bare ``ADR-015`` does not taint all of
    ADR-015. Measured reason: every actual retraction in this log names a part,
    while whole-decision taints voided every number those decisions contain --
    dates, sample sizes, a superseded fingerprint's first half -- and produced
    four times more findings than parted ones, all of them noise.
    **The exclusion is not silent.** Every bare target seen next to a marker is
    kept in :attr:`RetractionRecord.whole_decision_targets` and counted in the
    report, so widening the rule is a decision someone can take with the number
    in front of them rather than a possibility nobody knows about.
    """
    by_label = {s.label: s for s in sections}
    tainted: dict[str, set[str]] = {}
    dropped_whole_decision: dict[str, set[str]] = {}

    for section in sections:
        flat = " ".join(section.text.split())
        for marker in _MARKER_RE.finditer(flat):
            lo = max(0, marker.start() - window)
            hi = min(len(flat), marker.end() + window)
            for target in _citation_labels(flat[lo:hi]):
                if target.startswith(f"ADR-{section.adr}"):
                    continue  # self-reference is not a retraction
                bucket = tainted if "(" in target else dropped_whole_decision
                bucket.setdefault(target, set()).add(section.label)

    void: dict[str, frozenset[str]] = {}
    for target in tainted:
        text = by_label[target].text if target in by_label else ""
        tokens = {tok for tok in numeric_tokens(text) if is_distinctive(tok)}
        if tokens:
            void[target] = frozenset(tokens)

    return RetractionRecord(
        tainted={k: tuple(sorted(v)) for k, v in sorted(tainted.items())},
        void_magnitudes=void,
        sections=by_label,
        whole_decision_targets={
            k: tuple(sorted(v)) for k, v in sorted(dropped_whole_decision.items())
        },
    )


# --------------------------------------------------------------------------
# what counts as a quantitative claim
# --------------------------------------------------------------------------

#: A number as this project writes them: bare, with an ``x`` multiplier, as a
#: percentage, as a ``n/m`` count, or as ``2^-10``.
_NUMBER_RE: Final = re.compile(
    r"(?<![\w.])(?:"
    r"2\^-?\d+"
    r"|\d{1,3}/\d{1,3}"
    r"|[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?x?%?"
    r")"
)

#: Promoted to distinctive whatever their digit count, because the surrounding
#: word makes them a claim: ``p = 0.029``, ``p~0.03``, ``p≈0.0065``.
_P_VALUE_RE: Final = re.compile(r"\bp\s*[=~≈]\s*([\d.]+(?:e-?\d+)?|2\^-?\d+)", re.IGNORECASE)


def normalise_token(token: str) -> str:
    """Canonical form of a numeric token: sign, ``+``, ``x`` and ``%`` dropped.

    Sign is dropped so ``-6.50`` in a retracted table still matches ``6.50``
    quoted without it. Dropping information here can only ever ADD hits, and an
    extra line in a verdict table is the cheap failure.
    """
    out = token.strip().lstrip("+-")
    return out.rstrip("x%")


def _significant_digits(token: str) -> int:
    digits = normalise_token(token).replace(".", "").lstrip("0")
    return len(digits)


def is_distinctive(token: str) -> bool:
    """Is this token specific enough that seeing it twice means something?

    THREE RULES, and each was set by measuring what the loose version reported:

    * **A decimal point is required.** Without it the top tokens the sweep
      returned were ``2026``, ``2020``, ``100`` and ``4848`` -- a year, a year, a
      round number and the first half of a superseded fingerprint. A retracted
      TABLE contains its sample sizes and its dates; those are not the retracted
      magnitude, and matching them buries the ones that are.
    * **Three significant digits.** ``2.088``, ``119.3`` and ``6.50`` clear it;
      ``0.05`` and ``2.0`` do not.
    * **Counts and exponent forms are distinctive by shape**, because this log
      states its sign tests that way (``12/14``, ``2^-10``) and they are claims.
      A denominator below three is excluded: ``1/2`` and ``0/1`` are notation,
      not results.

    **This is a NARROWING and it is disclosed rather than silent.** The loose
    form is still reachable -- pass ``strict=False`` -- and
    :func:`self_test` checks that the strict form still catches the planted
    instance, which is the only property that makes a narrowing legitimate.
    """
    core = normalise_token(token)
    if "^" in core:
        return True
    if "/" in core:
        numerator, _, denominator = core.partition("/")
        return denominator.isdigit() and int(denominator) >= 3 and numerator.isdigit()
    if core.endswith("x"):
        core = core[:-1]
    return "." in core and _significant_digits(core) >= 3


def numeric_tokens(text: str) -> list[str]:
    """Every numeric token in a piece of prose, normalised, in order."""
    found = [normalise_token(m.group(0)) for m in _NUMBER_RE.finditer(text)]
    found += [normalise_token(m.group(1)) for m in _P_VALUE_RE.finditer(text)]
    return found


# --------------------------------------------------------------------------
# the tracked surface
# --------------------------------------------------------------------------

#: What a stranger reads. Everything tracked, minus the binary and data
#: artifacts where a numeric coincidence carries no claim.
SCOPE_PATTERNS: Final = (
    "src/**/*.py",
    "tests/**/*.py",
    "tools/**/*.py",
    "docs/**",
    "*.md",
    "configs/**/*.yaml",
    ".github/**/*.yml",
    "Makefile",
    "pyproject.toml",
)


def tracked_files(root: Path) -> list[str]:
    """Every tracked path, from git rather than from a filesystem walk.

    The public surface IS the tracked surface: working directories are ignored
    per clone rather than deleted, so a walk would scan files no stranger can
    read and report defects that are not public.
    """
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return sorted(p for p in out.stdout.splitlines() if p)


def _glob_to_re(pattern: str) -> re.Pattern[str]:
    """``**`` spans directories, ``*`` does not, ``?`` is one character.

    Written out rather than delegated to ``PurePath.match``, which does not
    anchor ``**`` on this interpreter, or to ``fnmatch``, whose ``*`` crosses
    directory separators. Both would silently widen the scope, and a scope that
    is wider than it says is how a scan ends up reporting a file nobody meant to
    publish -- or, worse, reading as thorough while matching by accident.
    """
    out, i = [], 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif char == "*":
            out.append("[^/]*")
            i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile("".join(out) + r"\Z")


def in_scope(rel: str, *, patterns: Sequence[str] = SCOPE_PATTERNS) -> bool:
    """Does this tracked path carry prose a claim could hide in?"""
    return any(_glob_to_re(pattern).match(rel) for pattern in patterns)


def prose_lines(rel: str, text: str) -> list[tuple[int, str]]:
    """``(1-based line number, prose)`` for the parts of a file a human reads.

    For Python that is comments and string literals ONLY, so an array of
    coefficients cannot be mistaken for an assertion about a fire. For everything
    else the whole file is prose. A Python file that does not parse falls back to
    whole-file prose rather than being skipped: a file this sweep cannot read is
    a hole in the sweep, and holes are what the earlier passes were.
    """
    if not rel.endswith(".py"):
        return list(enumerate(text.splitlines(), 1))
    out: list[tuple[int, str]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            for offset, line in enumerate(token.string.splitlines()):
                out.append((token.start[0] + offset, line))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return list(enumerate(text.splitlines(), 1))
    return out


# --------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One place a tracked file may be asserting something the log has voided."""

    path: str
    lineno: int
    detector: str  # "citation" | "magnitude"
    target: str  # the retracted label
    retracted_by: tuple[str, ...]
    evidence: str  # the citation, or the numeric token
    line: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "lineno": self.lineno,
            "detector": self.detector,
            "target": self.target,
            "retracted_by": list(self.retracted_by),
            "evidence": self.evidence,
            "line": self.line.strip()[:160],
        }


def scan_text(
    rel: str,
    text: str,
    record: RetractionRecord,
    *,
    detectors: Sequence[str] = ("citation", "magnitude"),
) -> list[Finding]:
    """Both detectors over one file's prose."""
    owners = record.magnitude_owners()
    findings: list[Finding] = []
    for lineno, line in prose_lines(rel, text):
        if "citation" in detectors:
            for label in citations(line):
                if label in record.tainted:
                    findings.append(
                        Finding(rel, lineno, "citation", label, record.tainted[label], label, line)
                    )
        if "magnitude" in detectors:
            for token in numeric_tokens(line):
                if not is_distinctive(token):
                    continue
                for label in owners.get(token, ()):
                    findings.append(
                        Finding(rel, lineno, "magnitude", label, record.tainted[label], token, line)
                    )
    return findings


def scan_files(
    root: Path, record: RetractionRecord, *, files: Iterable[str] | None = None
) -> list[Finding]:
    """Both detectors over the tracked surface (or an explicit file list)."""
    names = list(files) if files is not None else [f for f in tracked_files(root) if in_scope(f)]
    findings: list[Finding] = []
    for rel in names:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        findings.extend(scan_text(rel, text, record))
    return findings


# --------------------------------------------------------------------------
# the controls. A sweep reporting zero findings without these is worth nothing.
# --------------------------------------------------------------------------


#: The planted instance, spelled in halves and joined at run time so this
#: module's own source does not carry the retracted magnitudes as literals. It
#: is the real defect this sweep was written for: ``eval/stage.py`` cited a
#: section retracted in full and quoted both of its numbers.
def _plant() -> str:
    frontier = "2." + "088"
    growth = "119" + ".3"
    elasticity = "6." + "50"
    return (
        "# growth, not growth-per-frontier-cell: ADR-0" + "58 (4) measured that frontier "
        f"length saturates (Creek's frontier grows {frontier}x while its growth falls "
        f"{growth}x, needing an elasticity of -{elasticity})\n"
    )


#: The discrimination control. A currently VALID claim, in the same shape, in the
#: same file, must NOT be reported -- otherwise the sweep flags everything and
#: says nothing. ``12/14`` is the truth-only stage result the log upholds.
def _clean() -> str:
    return "# truth decelerates on 12/14 spatial blocks, and that result stands.\n"


@dataclass
class SelfTestResult:
    """Every control, with the number each returned. Prints as the evidence."""

    checks: list[dict[str, object]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: object) -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    @property
    def ok(self) -> bool:
        return all(bool(c["passed"]) for c in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": self.checks}


def self_test(root: Path, decisions: Path) -> SelfTestResult:
    """Prove the sweep can fail before believing it when it passes.

    Six controls, and the first three are the ones that matter:

    1. **CORPUS** -- the tracked surface must be large. A scan over an empty file
       list passes vacuously, which is how four earlier sweeps reported clean.
    2. **RECORD** -- the retraction record must be non-empty and must contain the
       edge this module was written for. A record that parses nothing makes every
       later check green for the wrong reason.
    3. **PLANTED INSTANCE** -- the real defect, injected into the corpus, must be
       reported by BOTH detectors. This is the positive control: it proves the
       sweep can return non-zero.
    4. **DISCRIMINATION** -- a currently valid claim of the same shape, in the
       same file, must NOT be reported.
    5. **NARROWING** -- a record with the edge removed must NOT report the plant,
       so the hit in 3 is attributable to the record and not to the pattern.
    6. **EXIT STATUS** -- findings must produce a non-zero status.
    """
    result = SelfTestResult()

    scoped = [f for f in tracked_files(root) if in_scope(f)]
    result.add("corpus_is_not_empty", len(scoped) > 100, {"n_files_in_scope": len(scoped)})

    record = build_retraction_record(parse_decision_sections(decisions.read_text()))
    edge = "ADR-058 (4)"
    result.add(
        "record_parses_and_carries_the_known_edge",
        len(record.tainted) > 5 and edge in record.tainted,
        {
            "n_tainted": len(record.tainted),
            "known_edge": edge,
            "retracted_by": list(record.tainted.get(edge, ())),
            "n_void_magnitudes": len(record.void_magnitudes.get(edge, frozenset())),
        },
    )

    planted = scan_text("src/wildfire_nowcast/__planted__.py", _plant(), record)
    detectors_hit = sorted({f.detector for f in planted if f.target == edge})
    result.add(
        "planted_instance_is_caught_by_both_detectors",
        detectors_hit == ["citation", "magnitude"],
        {
            "n_findings": len(planted),
            "detectors": detectors_hit,
            "magnitudes": sorted({f.evidence for f in planted if f.detector == "magnitude"}),
        },
    )

    clean = scan_text("src/wildfire_nowcast/__clean__.py", _clean(), record)
    result.add(
        "a_valid_claim_of_the_same_shape_is_not_reported",
        clean == [],
        {"n_findings": len(clean), "findings": [f.as_dict() for f in clean]},
    )

    narrowed = RetractionRecord(
        tainted={k: v for k, v in record.tainted.items() if k != edge},
        void_magnitudes={k: v for k, v in record.void_magnitudes.items() if k != edge},
        sections=record.sections,
    )
    narrowed_hits = [f for f in scan_text("x.py", _plant(), narrowed) if f.target == edge]
    result.add(
        "removing_the_edge_from_the_record_silences_the_plant",
        narrowed_hits == [],
        {"n_findings": len(narrowed_hits)},
    )

    result.add(
        "findings_produce_a_non_zero_status",
        _status(planted) != 0 and _status([]) == 0,
        {"with_findings": _status(planted), "without": _status([])},
    )
    return result


def _status(findings: Sequence[Finding]) -> int:
    return 1 if findings else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep. Exit 1 on any finding, 2 if a control failed.

    A control failure outranks a finding: a sweep whose own controls are red has
    not measured the tree, and reporting "0 findings" from it would be the exact
    false green this module exists to prevent.
    """
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--decisions",
        type=Path,
        default=None,
        help="the decision log to read the retraction record out of",
    )
    parser.add_argument("--json", action="store_true", help="emit the findings as JSON")
    parser.add_argument("--self-test", action="store_true", help="run the controls and stop")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    decisions = args.decisions or (root / "coordination" / "DECISIONS.md")
    if not decisions.is_file():
        print(
            f"REFUSING to sweep: no decision log at {decisions}. Without the retraction "
            "record every file is trivially clean, which is the false green this exists "
            "to prevent."
        )
        return 2

    controls = self_test(root, decisions)
    if args.self_test or not controls.ok:
        print(json.dumps(controls.as_dict(), indent=2))
        return 0 if controls.ok else 2

    record = build_retraction_record(parse_decision_sections(decisions.read_text()))
    findings = scan_files(root, record)
    payload = {
        "controls": controls.as_dict(),
        "n_tainted_sections": len(record.tainted),
        "n_void_magnitudes": len(record.magnitude_owners()),
        "n_files_scanned": len([f for f in tracked_files(root) if in_scope(f)]),
        "n_findings": len(findings),
        "findings": [f.as_dict() for f in findings],
        "verdict_required": (
            "Every finding needs a human verdict: RETRACTED / SUPERSEDED / STILL VALID / "
            "UNVERIFIABLE. This tool enumerates; it does not rule."
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"controls ok · {payload['n_files_scanned']} files · "
            f"{payload['n_tainted_sections']} retracted sections · {len(findings)} findings"
        )
        for finding in findings:
            print(
                f"  {finding.path}:{finding.lineno}  [{finding.detector}] {finding.evidence!r} "
                f"-> {finding.target} (retracted by {', '.join(finding.retracted_by)})"
            )
    return _status(findings)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
