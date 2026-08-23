"""A category a tool's prose names must be a key its code implements, and vice versa.

WHY THIS EXISTS. This project made a rule that a tool excluding a category prints
what it excluded beside its verdict. The rule is good and it had an undrawn
consequence: **every printed limitation is itself an assertion, and none of them
were tested.** Those are the sentences a reader trusts most, because they appear
in the output of the thing whose job is being trustworthy.

The instance that produced this module is ADR-116. ``tools/cited_paths.py``
documented a ``DECLARED`` category for "a path in another system entirely" that
its code has never implemented, while ``proposal`` was implemented and appeared
in no prose. The prose over-promised in one place and under-promised in another,
which is the shape a surface with no test takes. Code that drifts from its
documentation is caught by the code's own tests; a DOCSTRING that drifts is
caught by nothing, and in a checker the docstring is where the guarantees live.

WHAT IT CHECKS, AND WHAT IT CANNOT
----------------------------------
A module opts in by writing a glossary block: a header naming the attribute,
then one line per category, each beginning with the category's key in double
backquotes. The block ends at the first blank or non-conforming line, and an
indented line continues the description above it. Both sides are then compared
as sets, so a category added to the prose and not to the code fails, and so does
a key added to the code and not to the prose.

IT CANNOT READ FREE ENGLISH, and that limit is the interesting one, because free
English is exactly the form the phantom category took: "a specimen invented by a
test, a destination the program WRITES, the untracked corpus, ... or a path in
another system entirely" names five categories and no keys, and no mechanical
comparison can be made against it. The convention this module enforces is
therefore also the repair: NAME THE KEY, THEN DESCRIBE IT. A glossary that named
its categories would have failed on the day the phantom was written.

DISCOVERY IS DERIVED, NOT ENUMERATED. Every tracked Python file is read and the
adopters are whichever ones carry the header, so a third registry is picked up
with no edit here. The honest gap is the reverse: a registry that never adopts
the convention is not checked, and nothing in this module can force it to, so
the corpus control below asserts only that adopters exist and the capability
tests below run on synthetic modules that do not depend on the tree at all.

THIS FILE IS READ BY ITS OWN SCAN. The header is joined at runtime and a test
asserts the joined form appears nowhere in this source, for the same reason the
tell patterns in ``tests/test_hygiene.py`` are spelled in halves: a specimen
written literally would make this file an adopter of a registry it does not have,
and the alternative - skipping this path - would make the one file guaranteed to
contain fake glossaries the only file that could never report one.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from wildfire_nowcast.common.paths import repo_root

#: The header, joined at runtime. See the last paragraph of the module docstring.
MARKER_HEAD = "CATEG" + "ORIES OF"

#: ``<HEAD> ``ATTRIBUTE``  ...``: the line that opens a glossary and names the
#: module attribute it must agree with. Anything may follow on the same line, so
#: the header can carry its own sentence about who checks it.
MARKER = re.compile(re.escape(MARKER_HEAD) + r" ``([A-Za-z_][A-Za-z0-9_]*)``")

#: One glossary entry: the category key, in double backquotes, at the start of
#: the line once any comment prefix is removed. The key may contain spaces and
#: hyphens because real category names do.
ENTRY = re.compile(r"^``([^`\n]+)``(?:\s|$)")

#: Blank lines tolerated between the header and the first entry. Bounded so that
#: a header with no glossary under it cannot silently adopt a block much further
#: down the file.
_MAX_GAP = 2


def _uncomment(line: str) -> str:
    """Strip a leading ``#`` or ``#:`` marker, PRESERVING the indentation after it.

    The indentation is what distinguishes a continuation line from the end of the
    block, so a strip that discarded it would collapse every multi-line
    description into a syntax error the parser reports as a missing category.
    """
    stripped = line.lstrip()
    if stripped.startswith("#:"):
        rest = stripped[2:]
    elif stripped.startswith("#"):
        rest = stripped[1:]
    else:
        return line
    return rest[1:] if rest.startswith(" ") else rest


def documented_categories(text: str) -> dict[str, list[str]]:
    """``{attribute: the category names its prose lists}``, in source order.

    Works on a docstring and on a comment block alike, because the two are the
    same thing to a reader and the drift this catches happens in both.
    """
    lines = text.splitlines()
    found: dict[str, list[str]] = {}
    for index, line in enumerate(lines):
        match = MARKER.search(line)
        if match is None:
            continue
        cursor = index + 1
        gap = 0
        while cursor < len(lines) and not _uncomment(lines[cursor]).strip() and gap < _MAX_GAP:
            cursor += 1
            gap += 1
        names: list[str] = []
        while cursor < len(lines):
            content = _uncomment(lines[cursor])
            if not content.strip():
                break
            entry = ENTRY.match(content)
            if entry is not None:
                names.append(entry.group(1).strip())
            elif not content[:1].isspace():
                break
            cursor += 1
        found[match.group(1)] = names
    return found


def implemented_categories(text: str, attribute: str) -> list[str] | None:
    """The string keys of a module-level dict literal, or ``None`` if not readable.

    ``None`` is a REFUSAL, not an absence, and every caller turns it into a
    failure. A registry assembled at run time cannot be compared with prose by
    reading the source, and the one thing this must not do is report the
    comparison as clean because it could not perform it.
    """
    module = ast.parse(text)
    for node in module.body:
        if isinstance(node, ast.AnnAssign):
            names = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            names = list(node.targets)
            value = node.value
        else:
            continue
        if not any(isinstance(n, ast.Name) and n.id == attribute for n in names):
            continue
        if not isinstance(value, ast.Dict):
            return None
        keys: list[str] = []
        for key in value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return None
            keys.append(key.value)
        return keys
    return None


def audit(text: str, where: str) -> list[str]:
    """Every disagreement between a glossary and the dict it names, both directions."""
    problems: list[str] = []
    for attribute, documented in documented_categories(text).items():
        if not documented:
            problems.append(
                f"{where}: the glossary for {attribute} has no entries under its header. "
                "An entry begins with its category key in double backquotes."
            )
            continue
        implemented = implemented_categories(text, attribute)
        if implemented is None:
            problems.append(
                f"{where}: the prose documents categories of {attribute}, and no "
                "module-level dict literal of that name can be read from the source. "
                "Rename the header, or build the dict where it can be read."
            )
            continue
        for name in documented:
            if name not in implemented:
                problems.append(
                    f"{where}: {attribute} DOCUMENTS the category {name!r} and the code "
                    "does not implement it. Implement it or strike it from the prose."
                )
        for name in implemented:
            if name not in documented:
                problems.append(
                    f"{where}: {attribute} IMPLEMENTS the category {name!r} and the prose "
                    "does not document it. Document it or remove it."
                )
        if len(set(documented)) != len(documented):
            problems.append(f"{where}: {attribute} documents a category twice: {documented}")
    return problems


def tracked_python_files() -> list[str]:
    """The published tree, which is the only surface a reader has."""
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(p for p in out.stdout.splitlines() if p)


def audit_tree() -> tuple[list[str], dict[str, list[str]]]:
    """``(problems, {file: attributes it documents})`` over the tracked tree."""
    problems: list[str] = []
    adopters: dict[str, list[str]] = {}
    for rel in tracked_python_files():
        path = repo_root() / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        documented = documented_categories(text)
        if documented:
            adopters[rel] = sorted(documented)
        problems.extend(audit(text, rel))
    return problems, adopters


# --------------------------------------------------------------------------
# the live tree
# --------------------------------------------------------------------------


def test_no_documented_category_disagrees_with_the_code() -> None:
    """The check."""
    problems, _adopters = audit_tree()
    assert not problems, "\n  ".join(["prose and code disagree about a category:", *problems])


def test_there_is_something_to_check_and_the_parser_reached_it() -> None:
    """CORPUS POSITIVE CONTROL. An audit over zero adopters is a green light for nothing."""
    _problems, adopters = audit_tree()
    assert adopters, (
        "no tracked file carries a category glossary, so the tree-wide check above "
        "passed by reading nothing"
    )
    counted = sum(len(documented_categories((repo_root() / rel).read_text())) for rel in adopters)
    assert counted >= len(adopters) >= 1, adopters


def test_this_file_is_scanned_and_holds_no_literal_header() -> None:
    """The specimens below are joined at runtime, so this file adopts nothing.

    Without this, a fixture written literally would make this module claim a
    registry it does not have, and the tree-wide audit would report it forever.
    """
    rel = "tests/test_documented_categories.py"
    assert rel in tracked_python_files(), "this file is not tracked, so the scan never reads it"
    source = Path(__file__).read_text(encoding="utf-8")
    assert MARKER_HEAD not in source, "this file now contains a LITERAL glossary header"
    assert documented_categories(source) == {}, "this file has become an adopter"
    assert "CATEG" in source, "...but the fragments are here, so the join is what hides it"


# --------------------------------------------------------------------------
# capability, on synthetic modules that do not depend on the tree
# --------------------------------------------------------------------------


def _module(header_extra: str, entries: str, keys: str) -> str:
    """A tiny module carrying a glossary and a dict, both supplied by the caller."""
    return (
        f'"""A specimen.\n\n{MARKER_HEAD} ``TABLE``{header_extra}:\n\n{entries}\n"""\n\n'
        f"TABLE = {{{keys}}}\n"
    )


def test_a_glossary_that_matches_its_dict_is_clean() -> None:
    """The NEGATIVE control, first. Without it every plant below is satisfied by a
    checker that reports a problem on all input."""
    text = _module("", "``alpha``  the first.\n``beta``   the second.", "'alpha': 1, 'beta': 2")
    assert documented_categories(text) == {"TABLE": ["alpha", "beta"]}
    assert implemented_categories(text, "TABLE") == ["alpha", "beta"]
    assert audit(text, "specimen") == []


def test_a_category_the_prose_names_and_the_code_lacks_is_reported() -> None:
    """ADR-116's phantom, in the form it would have taken under this convention."""
    text = _module("", "``alpha``  the first.\n``ghost``  named, never built.", "'alpha': 1")
    problems = audit(text, "specimen")
    assert any("DOCUMENTS the category 'ghost'" in p for p in problems), problems


def test_a_category_the_code_implements_and_the_prose_omits_is_reported() -> None:
    """The other direction, which is the one that had been open for two months."""
    text = _module("", "``alpha``  the first.", "'alpha': 1, 'proposal': 2")
    problems = audit(text, "specimen")
    assert any("IMPLEMENTS the category 'proposal'" in p for p in problems), problems


def test_a_multi_line_description_does_not_end_the_block() -> None:
    """Continuations are indented, and a real glossary needs them."""
    entries = (
        "``alpha``  the first, described at some\n"
        "           length over two lines.\n"
        "``beta``   the second."
    )
    text = _module("", entries, "'alpha': 1, 'beta': 2")
    assert documented_categories(text)["TABLE"] == ["alpha", "beta"]
    assert audit(text, "specimen") == []


def test_a_header_naming_an_attribute_that_does_not_exist_is_reported() -> None:
    text = _module("", "``alpha``  the first.", "'alpha': 1").replace("TABLE = {", "OTHER = {")
    problems = audit(text, "specimen")
    assert any("no module-level dict literal" in p for p in problems), problems


def test_a_registry_assembled_at_run_time_REFUSES_rather_than_passes() -> None:
    """The refusal is the point: a comparison that cannot be made is not a clean one."""
    text = _module("", "``alpha``  the first.", "'alpha': 1").replace(
        "TABLE = {'alpha': 1}", "TABLE = dict(alpha=1)"
    )
    assert implemented_categories(text, "TABLE") is None
    assert any("no module-level dict literal" in p for p in audit(text, "specimen")), text


def test_a_dict_with_a_computed_key_REFUSES_rather_than_passes() -> None:
    text = _module("", "``alpha``  the first.", "'alpha': 1").replace(
        "TABLE = {'alpha': 1}", "NAME = 'alpha'\nTABLE = {NAME: 1}"
    )
    assert implemented_categories(text, "TABLE") is None


def test_a_header_with_no_glossary_under_it_is_reported() -> None:
    header = f"{MARKER_HEAD} ``TABLE``, and then nothing at all."
    text = f'"""A specimen.\n\n{header}\n"""\n\nTABLE = {{}}\n'
    problems = audit(text, "specimen")
    assert any("no entries under its header" in p for p in problems), problems


def test_a_comment_block_is_read_the_same_way_a_docstring_is() -> None:
    """The two registries in this tree live in a docstring and in a comment."""
    text = (
        f"# {MARKER_HEAD} ``TABLE``, checked both ways:\n"
        "#\n"
        "# ``alpha``  the first, wrapped\n"
        "#            onto a second line.\n"
        "# ``beta``   the second.\n"
        "#\n"
        "# Ordinary prose after the block, which must not be read as an entry.\n"
        "TABLE = {'alpha': 1, 'beta': 2}\n"
    )
    assert documented_categories(text) == {"TABLE": ["alpha", "beta"]}
    assert audit(text, "specimen") == []


def test_a_file_with_no_header_adopts_nothing() -> None:
    """The other half of the corpus control: silence must mean silence."""
    text = '"""Ordinary module. It has a dict and says nothing about categories."""\n\nTABLE = {'
    text += "'alpha': 1}\n"
    assert documented_categories(text) == {}
    assert audit(text, "specimen") == []
