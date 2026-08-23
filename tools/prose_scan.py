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
import subprocess
import tokenize
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
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

PROSE_REGIONS: Final = frozenset({REGION_DOCSTRING, REGION_COMMENT, REGION_CODE})

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

    for lineno, line in enumerate(text.splitlines(), start=1):
        for col, char in _offending(line):
            region = REGION_CODE
            for candidate in regions:
                if candidate.contains(lineno, col):
                    region = candidate.region
                    break
            category, name = _describe(char)
            out.append(Occurrence(path, lineno, region, char, category, name))

    for node in doc_nodes:
        raw = Counter(char for _, char in _offending(ast.get_source_segment(text, node) or ""))
        decoded = Counter(char for _, char in _offending(str(node.value)))
        for char, extra in (decoded - raw).items():
            category, name = _describe(char)
            out.extend(
                Occurrence(path, node.lineno, REGION_DOCSTRING, char, category, name)
                for _ in range(extra)
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
    for region in (REGION_DOCSTRING, REGION_COMMENT, REGION_CODE, REGION_LITERAL, REGION_PROSE):
        print(f"{region:>10}  {counts.get(region, 0)}")
    if args.region:
        for occ in sorted(occurrences, key=lambda o: (o.path, o.line)):
            if occ.region == args.region:
                print(occ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
