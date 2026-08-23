"""Classify non-ASCII punctuation in the tracked tree by its STRUCTURAL region.

The sweep of 2026-08-21 removed 1,732 occurrences of typographic punctuation from
prose and left 406 inside live string literals on purpose (JSON field names,
contract violation messages, expected values a test compares against): rewriting a
dash inside a literal changes program behaviour or an artifact's bytes. That sweep
is a snapshot. An em dash planted in ``common/paths.py`` afterwards left the
hygiene suite at 44 passed, exit 0, and the tracked count drifted 407 -> 415
during the task that performed the sweep, so regrowth is measured behaviour rather
than a worry.

This module is the half that holds it: it separates PROSE (docstrings and
comments, where a typographic character is a tell and is free to remove) from
LIVE LITERALS (where removing one is a behaviour change), and it does so by
reading the parse tree, never by consulting a list of file names. An allow-list of
files is how the previous seven instances of this class survived; a structural
rule has nothing to go stale.

Two things it does that a byte-level grep cannot:

* **It reads a docstring's VALUE, not its source bytes.** ``"\\u2014"`` inside a
  docstring is an em dash to every reader of ``help()`` and is invisible to a
  scan of the file's bytes. Thirty-one such escapes are already present in
  tracked JSON artifacts, found only after a commit claimed "0 non-ASCII bytes
  and therefore 0 dashes".
* **It classifies rather than filters.** Every occurrence is returned with its
  region, so the same corpus answers "how much prose debt is there" and "how many
  literals are deliberately left", and the second number cannot be quietly
  reclassified as the first.

The character class is imported from ``commit_guard`` rather than restated, so the
message rule and the source rule cannot drift into two different definitions of
"typographic".
"""

from __future__ import annotations

import argparse
import ast
import io
import logging
import re
import subprocess
import tokenize
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from commit_guard import _PUNCTUATION_CATEGORY_PREFIX

# ADR-103: a logger, and NO handler configured at import. `main` configures.
logger = logging.getLogger(__name__)

#: Regions a character can occupy. The first two are PROSE and are what the pin
#: bounds; the third is a live literal and is deliberately unbounded; the fourth
#: cannot occur for a punctuation category and exists so that an unexpected hit is
#: reported rather than dropped on the floor.
REGION_DOCSTRING: Final = "docstring"
REGION_COMMENT: Final = "comment"
REGION_LITERAL: Final = "literal"
REGION_CODE: Final = "code"

#: Non-Python tracked text. The whole file is prose, so there is no literal to
#: distinguish and the region is reported separately from Python docstrings.
REGION_PROSE: Final = "prose"

#: **A LIVE LITERAL THAT REACHES A READER** (ADR-097). It is a literal, so the
#: exclusion above still applies to the OTHER literals; this is the half of that
#: exclusion that was silent. ``eval/reporting.py:213`` holds
#: ``status="PROPOSAL <dash> reported, not enforced."`` and that string is
#: PRINTED. An em dash in a docstring is invisible to the senior engineer this
#: repository is written for; an em dash in a report is exactly what they see.
#: This is the FAILING category.
REGION_OUTPUT: Final = "output"

#: A literal that reaches a reader only on an ERROR path: a ``raise`` message or
#: an ``assert`` message. DECLARED AND COUNTED, not failing. It is reader-facing
#: too, and widening the gate to it is a PROPOSAL rather than a decision infra
#: may take alone: contract violation messages are compared against by tests in
#: three other leads' packages, so rewriting one is a behaviour change on a
#: surface infra does not own.
REGION_ERROR: Final = "err-msg"

#: What each region MEANS, printed beside its count on every run. The general
#: rule ADR-097 (4) draws from three defects in one day: **a tool that excludes a
#: category prints what it excluded next to its verdict, every time.** A tally
#: without this legend is what let `literal 473` read as a neutral number while
#: 63 of those characters were being shown to a reader.
REGION_LEGEND: Final[dict[str, str]] = {
    REGION_DOCSTRING: "prose, BOUNDED by the pin in tests/test_hygiene.py",
    REGION_COMMENT: "prose, BOUNDED by the same pin",
    REGION_CODE: "impossible for a punctuation category; non-zero means the classifier is wrong",
    REGION_OUTPUT: "FAILING: a live literal a READER SEES (printed, or passed by keyword)",
    REGION_ERROR: "EXCLUDED and DECLARED: reaches a reader only on a raise/assert path",
    REGION_LITERAL: "EXCLUDED and DECLARED: internal; rewriting one changes behaviour or bytes",
    REGION_PROSE: "tracked non-Python text, BOUNDED by its own pin",
}

#: The scanner's OWN blind spot, printed on every run rather than left to be
#: discovered. A literal appended to a list that a formatter prints later is
#: reader-facing and is NOT seen here: the sinks are direct. One instance was
#: found by hand in `common/playthrough.py` and fixed; the class is a PROPOSAL
#: to widen, not a silent gap.
UNSEEN_BY_CONSTRUCTION: Final = (
    "a literal STORED and formatted elsewhere (appended to a failures list, returned "
    "from a message builder); a text-rendering call whose name is not in the drawn "
    "allow-list printed below; a page fragment carrying no HTML tag of its own. The "
    "first is structural. The other two are ALLOW-LISTS and the missing entry nobody "
    "thought of is their standing failure mode."
)

PROSE_REGIONS: Final = frozenset({REGION_DOCSTRING, REGION_COMMENT, REGION_CODE})

#: Every region a report prints, in the order it prints them. Enumerated once so
#: that a new region cannot be added and then omitted from the verdict, which is
#: the exact shape of the defect this module is being extended to repair.
ALL_REGIONS: Final[tuple[str, ...]] = (
    REGION_DOCSTRING,
    REGION_COMMENT,
    REGION_CODE,
    REGION_OUTPUT,
    REGION_ERROR,
    REGION_LITERAL,
    REGION_PROSE,
)

#: Suffixes NOT scanned as prose, and the reason for each kind. A DENY-LIST, not
#: an allow-list, because an allow-list of extensions cannot see ``Makefile``,
#: ``LICENSE`` or ``.gitignore``, which have no extension at all and are exactly
#: the sort of file a sweep forgets. Same defect as a pattern rooted at ``/runs/``
#: that could not see ``data/fires``: the enumeration decides the census.
#:
#: ``.json`` and ``.lock`` are excluded because they are evidence artifacts and
#: machine output. Editing one to look better is the one thing evidence may not
#: undergo, and 31 escaped dashes already sit inside tracked JSON where they were
#: WRITTEN by a run rather than typed by anyone.
NON_PROSE_SUFFIXES: Final = frozenset(
    {".json", ".lock", ".npz", ".npy", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz"}
)


def is_typographic_punctuation(char: str) -> bool:
    """A non-ASCII character in a Unicode PUNCTUATION category (``Pc Pd Pe Pf Pi Po Ps``).

    ``ord(char) > 127`` because ASCII hyphen-minus is ``Pd`` too; the rule is about
    typographic punctuation, not about dashes. A category PREFIX rather than a list
    of categories, for the reason ``commit_guard`` gives: enumerating them leaves
    whichever one nobody thought of as the next gap.
    """
    if ord(char) <= 127:
        return False
    return unicodedata.category(char).startswith(_PUNCTUATION_CATEGORY_PREFIX)


@dataclass(frozen=True)
class Occurrence:
    """One typographic character, with the structural region it sits in."""

    path: str
    line: int
    region: str
    char: str
    category: str
    name: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line} [{self.region}] "
            f"U+{ord(self.char):04X} {self.name} ({self.category})"
        )


#: Token types that carry string content. ``FSTRING_MIDDLE`` is the literal text
#: between the braces of an f-string on 3.12; the surrounding ``FSTRING_START`` /
#: ``FSTRING_END`` carry only the quotes.
_STRING_TOKENS: Final = frozenset(
    {tokenize.STRING, tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END}
)


@dataclass(frozen=True)
class _TokenRegion:
    """The half-open span of one comment or string token, and what it is."""

    start: tuple[int, int]
    end: tuple[int, int]
    region: str

    def contains(self, line: int, col: int) -> bool:
        return self.start <= (line, col) < self.end


@dataclass(frozen=True)
class Span:
    """A closed line range inside one file."""

    path: str
    start_line: int
    end_line: int

    def contains(self, path: str, line: int) -> bool:
        return path == self.path and self.start_line <= line <= self.end_line


def _offending(text: str) -> Iterator[tuple[int, str]]:
    """``(offset, char)`` for every typographic character in ``text``."""
    for offset, char in enumerate(text):
        if is_typographic_punctuation(char):
            yield offset, char


def _describe(char: str) -> tuple[str, str]:
    return unicodedata.category(char), unicodedata.name(char, "UNNAMED")


def _docstring_targets(tree: ast.AST) -> list[ast.Constant]:
    """Every node whose value is a docstring, by position in the parse tree.

    ``ast.get_docstring`` returns the text and loses the node, and the node is what
    carries the line number a report has to name.
    """
    out: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            out.append(first.value)
    return out


def _token_regions(text: str, doc_starts: frozenset[tuple[int, int]]) -> list[_TokenRegion]:
    """Every comment and string token, with the region it constitutes.

    A token that STARTS where a docstring node starts is that docstring. Everything
    else that tokenizes as a string is a live literal. This is the structural
    distinction the pin rests on, and it is read from the tokenizer and the parse
    tree rather than from a list of files.
    """
    regions: list[_TokenRegion] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            region = REGION_COMMENT
        elif token.type in _STRING_TOKENS:
            region = REGION_DOCSTRING if token.start in doc_starts else REGION_LITERAL
        else:
            continue
        regions.append(_TokenRegion(token.start, token.end, region))
    return regions


# --------------------------------------------------------------------------
# WHICH LITERALS REACH A READER (ADR-097)
# --------------------------------------------------------------------------

#: Functions whose arguments are, by definition, shown to a person. Matched on
#: the ATTRIBUTE or NAME, so ``print``, ``sys.stderr.write`` and
#: ``logger.warning`` are all caught without resolving imports.
_OUTPUT_FUNCTIONS: Final = frozenset({"print", "warn", "write"})

#: Logging emissions. A diagnostic is read by a person too.
_OUTPUT_LOG_METHODS: Final = frozenset(
    {"debug", "info", "warning", "error", "exception", "critical", "log"}
)

#: Receivers whose method call is a logging emission rather than something else
#: that happens to be spelled ``.info``.
_LOG_RECEIVERS: Final = frozenset({"logger", "logging", "LOGGER", "_logger", "log"})

#: TEXT-RENDERING CALLS: their POSITIONAL arguments are drawn on a figure.
#:
#: THIS IS AN ALLOW-LIST AND IT IS SAID SO WHERE THE VERDICT IS PRINTED. A call
#: name nobody thought of is a gap, and the honest mitigation is disclosure, not
#: a claim of completeness. It is admitted because the alternative measured
#: worse: the two existing sinks are an output CALL and any KEYWORD argument,
#: and every one of these is positional, so a title drawn on the reader's screen
#: was invisible while the same string in a `print` was a gate failure. The
#: proposal measured 138 characters `sim/` renders and this scanner could not see,
#: against
#: the 10 the pin held for that package: 7% of what a reader of these figures
#: actually reads.
_DRAWN_CALLS: Final = frozenset(
    {"set_title", "suptitle", "text", "annotate", "stamp", "set_xlabel", "set_ylabel"}
)

#: HTML element names, used to recognise a literal that is a fragment of a
#: rendered PAGE rather than an internal string.
#:
#: WHY A CONTENT RULE HERE AND A CALL RULE ABOVE, which is the one place this
#: implementation departs from the proposal it came from. `sim/review.py` builds
#: its page through ``A = body.append``: the sink is an ALIAS, so the call is
#: named ``A`` and no list of call names can ever reach it. Measured, not
#: assumed: a call-name sink alone moves 51 characters and leaves 74 in that one
#: file. What the fragments have in common is their CONTENT, so that is what is
#: matched, against the standard element vocabulary rather than a general
#: ``<word>`` shape. The general shape was tried first and classified
#: ``reliability_summary[<lead>]`` in `sim/rundash.py` as markup, which is a
#: placeholder in a diagnostic, not a page.
_HTML_ELEMENTS: Final[tuple[str, ...]] = tuple(
    (
        "html head body meta title style script "
        "h1 h2 h3 h4 h5 h6 p div span code pre "
        "b i em strong small sub sup br hr a img "
        "ul ol li table thead tbody tr td th "
        "section article header footer figure figcaption "
        "details summary nav main label canvas"
    ).split()
)

_HTML_TAG: Final = re.compile(
    r"</?(?:" + "|".join(_HTML_ELEMENTS) + r")(?:\s[^>]*)?/?>", re.IGNORECASE
)


def _carries_markup(node: ast.AST) -> bool:
    """True when a literal contains a tag from the HTML element vocabulary."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_HTML_TAG.search(node.value))
    if isinstance(node, ast.JoinedStr):
        joined = "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        return bool(_HTML_TAG.search(joined))
    return False


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_output_call(node: ast.Call) -> bool:
    """``print(...)``, ``warnings.warn(...)``, ``f.write(...)``, ``logger.info(...)``."""
    name = _call_name(node.func)
    if name in _OUTPUT_FUNCTIONS:
        return True
    if name in _OUTPUT_LOG_METHODS and isinstance(node.func, ast.Attribute):
        receiver = node.func.value
        return isinstance(receiver, ast.Name) and receiver.id in _LOG_RECEIVERS
    return False


def _string_spans(node: ast.AST) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """Every string-bearing node in a subtree, as a half-open ``(start, end)`` span.

    ``JoinedStr`` is yielded whole AND its parts individually: an f-string's
    constant segments are what a byte scan sees, and the enclosing node is what
    covers a 3.11 tokenizer that does not split them. This is why the count here
    is not the maintainer's floor of 28: it sees f-strings and concatenations,
    which a search for direct string constants cannot.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.JoinedStr) or (
            isinstance(child, ast.Constant) and isinstance(child.value, str)
        ):
            if child.end_lineno is None or child.end_col_offset is None:
                continue
            yield (child.lineno, child.col_offset), (child.end_lineno, child.end_col_offset)


def reader_facing_spans(tree: ast.AST) -> tuple[list[_TokenRegion], list[ast.Constant]]:
    """``(spans, output constants)`` for every literal that reaches a reader.

    Four sinks. The second is a REFUSAL rather than a list of keyword names; the
    last two are ALLOW-LISTS and are printed beside the verdict as such:

    * an argument anywhere inside an OUTPUT call (:func:`_is_output_call`);
    * the value of ANY keyword argument to any call. A string handed to a call
      under a NAME is a labelled value, and labelled values are what reports
      carry - ``status=``, ``note=``, ``label=``, ``help=``, ``provenance=``.
      Enumerating the names would leave whichever one nobody thought of as the
      next gap, which is how seven allow-lists in this repository failed;
    * a POSITIONAL argument to a text-rendering call (:data:`_DRAWN_CALLS`).
      A dash in a docstring is invisible to the reader this repository is
      written for; a dash in a figure title is exactly what they see;
    * a literal carrying HTML markup (:data:`_HTML_ELEMENTS`), which is a
      fragment of a rendered page. This one is matched on CONTENT because the
      page's sink is an alias (``A = body.append``) and no call-name list can
      reach it.

    ``raise`` and ``assert`` messages are collected separately: reader-facing on
    an error path, declared and counted, not failing. See :data:`REGION_ERROR`.
    """
    spans: list[_TokenRegion] = []
    constants: list[ast.Constant] = []

    def take(node: ast.AST, region: str) -> None:
        for start, end in _string_spans(node):
            spans.append(_TokenRegion(start, end, region))
        if region == REGION_OUTPUT:
            constants.extend(
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _is_output_call(node) or _call_name(node.func) in _DRAWN_CALLS:
                for arg in node.args:
                    take(arg, REGION_OUTPUT)
            for keyword in node.keywords:
                if keyword.arg is not None:
                    take(keyword.value, REGION_OUTPUT)
        elif _carries_markup(node):
            take(node, REGION_OUTPUT)
        elif isinstance(node, ast.Raise) and node.exc is not None:
            take(node.exc, REGION_ERROR)
        elif isinstance(node, ast.Assert) and node.msg is not None:
            take(node.msg, REGION_ERROR)

    # OUTPUT wins over ERROR where a literal is in both, because the failing
    # category must not be weakened by an overlap.
    spans.sort(key=lambda region: region.region != REGION_OUTPUT)
    return spans, constants


def prose_spans(text: str) -> list[_TokenRegion]:
    """Every COMMENT and DOCSTRING span in a Python source, as regions.

    Public because a second scanner needs the same distinction and C0 says there
    is one implementation of anything. `tests/test_shared_temp_paths.py` uses it
    to tell a fixed `/tmp` path that a program WRITES TO from a paragraph
    explaining why one may not be written. Without it, the documentation of a
    defect is indistinguishable from the defect.
    """
    tree = ast.parse(text)
    doc_starts = frozenset((n.lineno, n.col_offset) for n in _docstring_targets(tree))
    return [
        region
        for region in _token_regions(text, doc_starts)
        if region.region in (REGION_COMMENT, REGION_DOCSTRING)
    ]


def scan_python_source(path: str, text: str) -> list[Occurrence]:
    """Every typographic character in ``text``, tagged with its region.

    Each character is located EXACTLY, at ``(line, column)``, and assigned to the
    token whose span contains it. Locating by token start instead would put a dash
    on line 300 of a docstring that opened on line 240 at line 240, and would then
    fail to account for line 300 at all.

    On top of the source scan, each docstring's DECODED value is compared with its
    source text: a docstring carrying ``\\u2014`` holds an em dash for every reader
    of ``help()`` while holding none for a scan of the file's bytes, and the excess
    is reported at the docstring's first line.
    """
    tree = ast.parse(text, filename=path)
    out: list[Occurrence] = []

    doc_nodes = _docstring_targets(tree)
    doc_starts = frozenset((node.lineno, node.col_offset) for node in doc_nodes)
    regions = _token_regions(text, doc_starts)
    reader_spans, output_constants = reader_facing_spans(tree)

    for lineno, line in enumerate(text.splitlines(), start=1):
        for col, char in _offending(line):
            region = REGION_CODE
            for candidate in regions:
                if candidate.contains(lineno, col):
                    region = candidate.region
                    break
            if region == REGION_LITERAL:
                for reach in reader_spans:
                    if reach.contains(lineno, col):
                        region = reach.region
                        break
            category, name = _describe(char)
            out.append(Occurrence(path, lineno, region, char, category, name))

    for node in doc_nodes:
        out.extend(_escaped_excess(path, text, node, REGION_DOCSTRING))
    for node in output_constants:
        out.extend(_escaped_excess(path, text, node, REGION_OUTPUT))
    return out


def _escaped_excess(path: str, text: str, node: ast.Constant, region: str) -> list[Occurrence]:
    """Characters present in a literal's VALUE and absent from its source bytes.

    ``"\\u2014"`` is an em dash to every reader of the string and is invisible to a
    scan of the file. Applied to docstrings since ADR-080; applied to
    output-reaching literals here, because a report printing an escaped dash
    prints a dash.
    """
    segment = ast.get_source_segment(text, node) or ""
    raw = Counter(char for _, char in _offending(segment))
    decoded = Counter(char for _, char in _offending(str(node.value)))
    out: list[Occurrence] = []
    for char, extra in (decoded - raw).items():
        category, name = _describe(char)
        out.extend(
            Occurrence(path, node.lineno, region, char, category, name) for _ in range(extra)
        )
    return out


def scan_prose_file(path: str, text: str) -> list[Occurrence]:
    """Every typographic character in a non-Python text file. All of it is prose."""
    out: list[Occurrence] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for _, char in _offending(line):
            category, name = _describe(char)
            out.append(Occurrence(path, lineno, REGION_PROSE, char, category, name))
    return out


def tracked_files(repo_root: Path) -> list[str]:
    """``git ls-files``, so the corpus is the PUBLISHED tree and not this disk."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def is_prose_file(rel: str) -> bool:
    """Everything tracked that is not Python, not machine output and not binary.

    Stated as a refusal rather than as a list of extensions so that an
    extensionless file - ``Makefile``, ``LICENSE``, ``.gitignore`` - is IN by
    default. A file nobody thought to enumerate is precisely where the next
    occurrence lands.
    """
    return not rel.endswith(".py") and Path(rel).suffix not in NON_PROSE_SUFFIXES


def scan_repository(repo_root: Path, *, include_prose: bool = True) -> list[Occurrence]:
    """Scan every tracked Python module, and optionally every tracked prose file.

    A file that does not decode as UTF-8 is binary and is skipped, not guessed at.
    """
    out: list[Occurrence] = []
    for rel in tracked_files(repo_root):
        path = repo_root / rel
        if not path.is_file():
            continue
        if rel.endswith(".py"):
            out.extend(scan_python_source(rel, path.read_text(encoding="utf-8")))
        elif include_prose and is_prose_file(rel):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # ADR-103: a scan that NARROWS ITS OWN CORPUS in silence is this
                # session's recurring defect. Skipping is still right - the file
                # is binary - but the reader has to be told the census shrank.
                logger.warning("skipping %s: it is tracked text that is not UTF-8", rel)
                continue
            out.extend(scan_prose_file(rel, text))
    return out


# --------------------------------------------------------------------------
# the output-literal debt, and who owns each line of it
# --------------------------------------------------------------------------

#: Output-reaching typographic characters that belong to another lead, per file.
#: **EMPTY, AND EMPTY IS THE SUCCESS STATE.** 63 were measured at `738e7b7`;
#: infra swept its own 38 in `common/`, `tools/` and `tests/` the same day, and
#: the last 25 in `eval/`, `model/`, `runs/` and `sim/` went in one burst:
#: `sim/` at `8b62729` (S11), `data/` at `8d3c5c0` (D14), `model/` at `7c78d32`
#: (M19); the 125 the widened sinks then exposed in `sim/` went at S12. Every
#: one of those files is STILL TRACKED, checked against
#: `git ls-files` rather than inferred from a count of zero, so this dict reads
#: "swept" and not "deleted".
#:
#: A BURN-DOWN, and it fails in FOUR directions:
#:
#: * a file NOT listed that carries one is NEW DEBT and fails;
#: * a listed file whose count RISES fails;
#: * a listed file whose count FALLS fails, so a sweep is RECORDED when it
#:   happens. A tolerated fall leaves slack that a later commit can refill
#:   without tripping RISEN, which made this a ceiling wearing the name of a pin
#:   until ADR-106 (4);
#: * a listed file that reaches ZERO is STALE and fails, so an entry cannot
#:   outlive its reason and the list cannot decay into an allow-list.
#:
#: **AT ZERO THE DICT IS EMPTY AND THE GATE STILL BINDS**, through the first
#: direction: any output-reaching literal anywhere in the tracked tree is
#: UNDECLARED and fails. What an empty dict does NOT establish is that the
#: classifier can still SEE one. That is a property of the classifier and not of
#: anyone's cleanliness, so it is asserted directly, on a synthetic module, by
#: `tests/test_prose_output_literals.py::
#: test_the_classifier_answers_both_ways_on_a_planted_literal`. A control that
#: inferred classifier health from the SIZE of this dict could not survive the
#: dict being emptied, and did not: a floor of 20 was left guarding a debt of 14.
#:
#: RE-POPULATED BY THE SINK WIDENING AT `88ebd93`, AND SWEPT AGAIN AT S12.
#: The drawn-text and page-fragment sinks made 125 previously invisible
#: characters visible across 13 `sim/` files, pinned as that package's debt with
#: the sweep owed by its owner. All 125 are gone: 100 EM DASH, 24 MIDDLE DOT and
#: 1 DOUBLE VERTICAL LINE, rewritten in the reader-visible literal only, on the
#: exact lines this scanner named and no others. The control that makes that a
#: measurement rather than a claim is the OTHER regions: `docstring`, `comment`,
#: `err-msg`, `literal` and `prose` are unchanged across the sweep, so nothing
#: was reclassified into a quieter bin - only the failing region fell, by
#: exactly 125. The scanner's count was 125 where the proposal that asked for
#: the widening measured 138; the two are different definitions and the gate
#: reproduces this one.
OUTPUT_LITERAL_DEBT: Final[dict[str, int]] = {}


@dataclass(frozen=True)
class OutputAudit:
    """The verdict on the FAILING category, with every direction it can fail in."""

    undeclared: dict[str, int]
    risen: dict[str, tuple[int, int]]
    fallen: dict[str, tuple[int, int]]
    stale: tuple[str, ...]
    declared_total: int
    found_total: int

    @property
    def ok(self) -> bool:
        return not self.undeclared and not self.risen and not self.fallen and not self.stale

    def lines(self) -> list[str]:
        out: list[str] = []
        for path, count in sorted(self.undeclared.items()):
            out.append(f"NEW output-reaching literal(s): {path} carries {count}, undeclared")
        for path, (was, now) in sorted(self.risen.items()):
            out.append(f"RISEN: {path} carried {was}, now carries {now}")
        for path, (was, now) in sorted(self.fallen.items()):
            out.append(
                f"FALLEN: {path} carried {was}, now carries {now}; set it to {now} in "
                "OUTPUT_LITERAL_DEBT. A pin that tolerates a fall leaves slack a later "
                "commit can refill without tripping RISEN."
            )
        for path in self.stale:
            out.append(f"STALE: {path} is clean; remove it from OUTPUT_LITERAL_DEBT")
        return out


def audit_output_literals(
    occurrences: Sequence[Occurrence], debt: Mapping[str, int] | None = None
) -> OutputAudit:
    """Compare the output-reaching literals found against the declared debt."""
    declared = dict(OUTPUT_LITERAL_DEBT if debt is None else debt)
    found: dict[str, int] = {}
    for occ in occurrences:
        if occ.region == REGION_OUTPUT:
            found[occ.path] = found.get(occ.path, 0) + 1
    undeclared = {path: n for path, n in found.items() if path not in declared}
    risen = {
        path: (declared[path], n)
        for path, n in found.items()
        if path in declared and n > declared[path]
    }
    fallen = {
        path: (declared[path], n)
        for path, n in found.items()
        if path in declared and 0 < n < declared[path]
    }
    stale = tuple(sorted(path for path in declared if found.get(path, 0) == 0))
    return OutputAudit(
        undeclared=undeclared,
        risen=risen,
        fallen=fallen,
        stale=stale,
        declared_total=sum(declared.values()),
        found_total=sum(found.values()),
    )


# --------------------------------------------------------------------------
# the one exemption, pinned to its reason rather than to its file name
# --------------------------------------------------------------------------

#: The module whose estimand source is hashed into a published provenance pin.
ESTIMAND_MODULE: Final = "src/wildfire_nowcast/eval/stage.py"


def estimand_hashed_spans(repo_root: Path) -> tuple[list[Span], str]:
    """The line spans ``eval.stage.estimand_digest`` hashes, and the digest of them.

    Re-derived here by the SAME rule the digest uses (top-level defs named in
    ``ESTIMAND_FUNCTIONS``, sliced with ``ast.get_source_segment``) and then checked
    against the live digest by the caller. That check is what makes this an
    exemption pinned to a reason: if the digest stops hashing these spans - because
    a function leaves ``ESTIMAND_FUNCTIONS``, is renamed, or moves out of module
    scope - the spans computed here stop matching the digest, and the exemption
    fails instead of quietly outliving the thing it was granted for.
    """
    import hashlib

    from wildfire_nowcast.eval.stage import ESTIMAND_FUNCTIONS

    text = (repo_root / ESTIMAND_MODULE).read_text(encoding="utf-8")
    tree = ast.parse(text)
    spans: list[Span] = []
    segments: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in ESTIMAND_FUNCTIONS:
            end = node.end_lineno if node.end_lineno is not None else node.lineno
            spans.append(Span(ESTIMAND_MODULE, node.lineno, end))
            segments.append(ast.get_source_segment(text, node) or "")
    digest = hashlib.sha256("\n".join(segments).encode()).hexdigest()
    return spans, digest


def partition_exempt(
    occurrences: Sequence[Occurrence], spans: Sequence[Span]
) -> tuple[list[Occurrence], list[Occurrence]]:
    """Split into ``(exempt, in_scope)`` by whether the line sits in a hashed span."""
    exempt: list[Occurrence] = []
    in_scope: list[Occurrence] = []
    for occ in occurrences:
        if any(span.contains(occ.path, occ.line) for span in spans):
            exempt.append(occ)
        else:
            in_scope.append(occ)
    return exempt, in_scope


def main(argv: list[str] | None = None) -> int:
    from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root (default: cwd)")
    parser.add_argument("--region", default="", help="only report this region")
    parser.add_argument(
        "--no-prose", action="store_true", help="Python modules only, no .md/.rst/.txt"
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv)
    # ADR-103: the ONE place this program configures logging. Imported here, not
    # at module scope, so the scanner stays importable without the package.
    configure_from_args(args)

    repo_root = Path(args.repo).resolve()
    occurrences = scan_repository(repo_root, include_prose=not args.no_prose)
    counts: dict[str, int] = {}
    for occ in occurrences:
        counts[occ.region] = counts.get(occ.region, 0) + 1
    for region in ALL_REGIONS:
        print(f"{region:>10}  {counts.get(region, 0):>4}  {REGION_LEGEND[region]}")

    audit = audit_output_literals(occurrences)
    print(
        f"\noutput-reaching literals: found {audit.found_total}, declared "
        f"{audit.declared_total} across {len(OUTPUT_LITERAL_DEBT)} file(s) owned by "
        "other leads (ADR-097). The declared number is a PIN, not a ceiling: it fails "
        "when it rises AND when it falls."
    )
    if not OUTPUT_LITERAL_DEBT:
        print(
            "the burn-down is COMPLETE: no file carries declared output-reaching debt. "
            "ZERO IS THE SUCCESS STATE AND IS A PASSING STATE. The gate still binds "
            "through the UNDECLARED direction, and that this scanner can still see such "
            "a literal is proven on a synthetic module, never by this count."
        )
    print(f"NOT SEEN BY THIS SCANNER: {UNSEEN_BY_CONSTRUCTION}")
    print(
        "DRAWN-TEXT SINK (an allow-list, positional arguments only): "
        + ", ".join(sorted(_DRAWN_CALLS))
    )
    print(
        f"PAGE-FRAGMENT SINK: a literal carrying one of {len(_HTML_ELEMENTS)} HTML "
        "element tags. Matched on CONTENT because the page's own sink is an alias."
    )
    for line in audit.lines():
        print(f"  [FAIL] {line}")

    if args.region:
        for occ in sorted(occurrences, key=lambda o: (o.path, o.line)):
            if occ.region == args.region:
                print(occ)

    print(f"verdict: {'PASS' if audit.ok else 'FAIL'}")
    return 0 if audit.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
