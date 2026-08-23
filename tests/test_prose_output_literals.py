"""ADR-097: the half of `prose_scan`'s exclusion that a reader can actually see.

`prose_scan` excluded live string literals BY CONSTRUCTION, which is right - a
pin that rewrites a literal changes program behaviour or an artifact's bytes -
and it reported `literal 473` as a neutral tally. The tells were in there.
`eval/reporting.py:213` held ``status="PROPOSAL <dash> reported, not enforced."``
and that string is PRINTED.

The exclusion stays. What changes is that it is now DECLARED AND COUNTED, split
into the part that reaches a reader (FAILING) and the part that does not
(excluded, printed beside the verdict every run).

The maintainer sized the class at 28 with a search for direct string constants
and said so was a floor. This classifier reads the parse tree, so it sees
f-strings, concatenations and escaped dashes as well: **63 at `738e7b7`**, of
which 38 were infra's and are swept here, and 25 belong to `eval/`, `model/`,
`sim/` and `runs/` and were a burn-down.

**THE BURN-DOWN IS COMPLETE AND THAT IS WHY THIS FILE CHANGED (I16).** @simviz,
@data and @model swept the last 25 in one burst, the declared debt reached zero,
and four assertions here broke on the success of the thing they guarded. The
worst of them asserted `len(found) >= 20` over the live tree and said so in its
own failure message: *"Either three leads cleared their debt at once, or the sink
analysis has stopped matching anything."* It named both hypotheses and could not
separate them, because it tested a POPULATION as a proxy for a CAPABILITY. A
burn-down whose anti-vacuity control needs the debt never to reach zero is a
burn-down that cannot be completed.

**EVERY ASSERTION BELOW HOLDS AT ZERO DEBT**, and the capability is asserted
directly: a synthetic module carrying a known output-reaching literal is handed
to the classifier and the region that comes back is named. That is independent of
real debt, works in a clone, and is stronger than a floor on live tree contents,
because it names the exact input it detects instead of inferring detection from a
count.
"""

from __future__ import annotations

import ast
import functools
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import prose_scan  # noqa: E402 - tools/ is not a package; same route test_hygiene uses
from wildfire_nowcast.common.paths import repo_root  # noqa: E402

#: The surfaces infra owns. Zero output-reaching characters is the standing rule
#: here, not a burn-down: infra can fix its own the day it finds them.
INFRA_SURFACES = ("src/wildfire_nowcast/common/", "tools/", "tests/")


@functools.lru_cache(maxsize=1)
def _scan() -> tuple[prose_scan.Occurrence, ...]:
    """Scanned ONCE. Six assertions over the same corpus is one measurement."""
    return tuple(prose_scan.scan_repository(repo_root()))


def _occ(path: str, line: int, region: str) -> prose_scan.Occurrence:
    return prose_scan.Occurrence(path, line, region, "—", "Pd", "EM DASH")


#: Two synthetic modules that differ in ONE thing: the sink. Same character, same
#: line number, same surrounding words, `print(...)` in the first and a plain
#: assignment in the second. A classifier that answers OUTPUT for one and LITERAL
#: for the other is discriminating on the sink and on nothing else, which a pair
#: that also differed in wording or position could not establish.
_PLANTED_OUTPUT: str = 'def render():\n    print("a planted dash — shown to a reader")\n'
_PLANTED_INTERNAL: str = 'def render():\n    held = "a planted dash — shown to a reader"\n'

#: The same construction for the error path, which is DECLARED and not failing.
_PLANTED_ERROR: str = 'def render():\n    raise ValueError("a planted dash — shown to a reader")\n'


def _debt_on_infra_surfaces(debt: Mapping[str, int]) -> tuple[str, ...]:
    """Declared entries on a surface infra owns. There may never be any."""
    return tuple(sorted(path for path in debt if path.startswith(INFRA_SURFACES)))


def _debt_entries_at_zero(debt: Mapping[str, int]) -> tuple[str, ...]:
    """Declared entries carrying 0. A cleared file is REMOVED, never zeroed."""
    return tuple(sorted(path for path, count in debt.items() if count <= 0))


# --------------------------------------------------------------------------
# the live tree
# --------------------------------------------------------------------------


def test_infra_owns_zero_output_reaching_literals() -> None:
    """The 38 that were infra's are gone, and they may not come back.

    An empty result here is a clean surface only if the scanner still works, and
    that is not assumed: it is established on a synthetic module by
    :func:`test_the_classifier_answers_both_ways_on_a_planted_literal`.
    """
    left = [
        occ
        for occ in _scan()
        if occ.region == prose_scan.REGION_OUTPUT and occ.path.startswith(INFRA_SURFACES)
    ]
    assert not left, (
        "a typographic character reaches a reader from a surface infra owns:\n  "
        + "\n  ".join(str(occ) for occ in left)
    )


def test_the_rest_is_exactly_the_declared_debt() -> None:
    """Four directions, all of them failures: undeclared, risen, fallen, stale."""
    audit = prose_scan.audit_output_literals(_scan())
    assert audit.ok, "\n".join(audit.lines())


def test_the_debt_belongs_to_other_leads_and_says_so() -> None:
    """A burn-down that infra could clear itself would be a to-do list, not a fence.

    The live debt was EMPTY when this was written, so both rules held over it
    vacuously. Each is therefore ALSO put to a mapping that VIOLATES it, on the
    same two functions, so the test is shown to answer both ways rather than
    reporting green for want of an entry to judge. That construction is kept now
    that the debt is 13 files again: a rule that only ever sees conforming input
    cannot be distinguished from a rule that accepts everything.
    """
    live = prose_scan.OUTPUT_LITERAL_DEBT
    assert _debt_on_infra_surfaces(live) == (), "infra fixes its own the day it finds them"
    assert _debt_entries_at_zero(live) == (), "a cleared file is REMOVED, not set to 0"

    assert _debt_on_infra_surfaces({"tools/x.py": 1, "src/wildfire_nowcast/eval/y.py": 1}) == (
        "tools/x.py",
    )
    assert _debt_entries_at_zero({"src/wildfire_nowcast/sim/y.py": 0}) == (
        "src/wildfire_nowcast/sim/y.py",
    )


def test_the_classifier_answers_both_ways_on_a_planted_literal() -> None:
    """THE ANTI-VACUITY CONTROL. It asserts the CAPABILITY, not the population.

    The claim wanted here is that the sink analysis can still detect an
    output-reaching literal. The claim the old control tested is that such
    literals exist in the tree, which is an incidental property of how much debt
    happens to be outstanding, and the proxy holds only while the debt is unpaid.
    The debt is now zero and the proxy is gone, so the capability is asserted on a
    known input: this module is constructed here, classified here, and the region
    is named here. No count of anyone else's files takes part.

    Both directions on the SAME rig. A control that only answered OUTPUT could be
    satisfied by a classifier that answered OUTPUT to everything, which is the
    mirror of the vacuity it replaces.
    """
    output = prose_scan.scan_python_source("planted.py", _PLANTED_OUTPUT)
    assert [(occ.line, occ.region) for occ in output] == [(2, prose_scan.REGION_OUTPUT)], (
        "the sink analysis no longer sees a dash inside a print(). This is the failure "
        f"the live-tree floor was trying and failing to detect: {output}"
    )
    assert output[0].char == "—" and output[0].name == "EM DASH", output

    internal = prose_scan.scan_python_source("planted.py", _PLANTED_INTERNAL)
    assert [(occ.line, occ.region) for occ in internal] == [(2, prose_scan.REGION_LITERAL)], (
        "an internal literal was classified as reader-facing. A classifier that says "
        f"OUTPUT to everything passes the assertion above and means nothing: {internal}"
    )


def test_the_internal_literals_are_still_the_large_majority_and_still_excluded() -> None:
    """The exclusion is the point. It must remain an exclusion, not become a sweep.

    RE-MEASURED IN THE COMMIT THAT WIDENED THE SINK, which is the condition the
    widening was adopted under. The count moved 377 -> 252 and NOT ONE CHARACTER
    WAS REWRITTEN: the same characters were RECLASSIFIED out of `literal` and
    into `output` by the drawn-text and page-fragment sinks. The total across all
    regions was 438 on both sides of that reclassification, which is what
    distinguishes it from the sweep this assertion exists to catch, and it is why
    the threshold could be lowered without weakening it.

    THE COMMITTED TREE READS 257, NOT 252, and the difference is this file: the
    two new capability controls carry five internal literals of their own. The
    252 was measured before they existed. Both numbers are recorded because a
    single one would be wrong for whichever tree the next reader is holding.

    The bar is 200 against a measured 257, ~22% of headroom. Left at 250 it would
    have sat 2 below the value measured mid-change and fired on the next ordinary
    edit, which is the failure mode of a threshold re-derived from the number it
    is measuring.
    """
    scanned = _scan()
    internal = [occ for occ in scanned if occ.region == prose_scan.REGION_LITERAL]
    assert len(internal) > 200, (
        f"internal literals are down to {len(internal)} from 257 at the sink widening "
        "(377 before it). A sweep has started rewriting literals, which changes "
        "behaviour and artifact bytes."
    )


#: THE SAME ONE-VARIABLE CONSTRUCTION FOR THE TWO SINKS ADOPTED FROM @simviz's
#: PROPOSAL. Each pair differs in the sink and in nothing else: same character,
#: same line, same words.
_PLANTED_DRAWN: str = 'def render(ax):\n    ax.set_title("a planted dash — drawn on a figure")\n'
_PLANTED_UNDRAWN: str = 'def render(ax):\n    ax.set_zorder("a planted dash — drawn on a figure")\n'
#: JOINED AT RUNTIME, and the reason is the rule itself: the page-fragment sink
#: matches on CONTENT, so a complete tag written literally here would make this
#: file a page fragment and put an em dash on an infra surface in the failing
#: category. It did, on the first run, and the assertion that infra owns zero
#: caught it. The classifier still receives a complete tag; this file no longer
#: contains one.
_PAGE_FRAGMENT: str = "<" + "p>" + "a planted dash — in the page" + "</" + "p>"
_PLANTED_PAGE: str = 'def render():\n    frag = "' + _PAGE_FRAGMENT + '"\n'
_PLANTED_UNMARKED: str = 'def render():\n    frag = "[p]a planted dash — in the page[/p]"\n'


def test_the_drawn_text_sink_answers_both_ways() -> None:
    """A title a reader SEES is output; the same string handed to a non-drawing call is not.

    The negative half is what makes this a control rather than a claim: if the
    classifier had started answering OUTPUT for every positional argument, the
    first assertion would still pass and the category would be meaningless.
    """
    drawn = prose_scan.scan_python_source("planted.py", _PLANTED_DRAWN)
    assert [(occ.line, occ.region) for occ in drawn] == [(2, prose_scan.REGION_OUTPUT)], drawn
    assert drawn[0].name == "EM DASH", drawn

    undrawn = prose_scan.scan_python_source("planted.py", _PLANTED_UNDRAWN)
    assert [(occ.line, occ.region) for occ in undrawn] == [(2, prose_scan.REGION_LITERAL)], (
        "every positional argument is being treated as drawn text, which is not the "
        f"rule and would make the category unbounded: {undrawn}"
    )


def test_the_page_fragment_sink_answers_both_ways() -> None:
    """A literal carrying HTML markup is a page fragment; a lookalike is not.

    Matched on CONTENT, because `sim/review.py` appends through an ALIAS and no
    call-name list can reach it. The negative specimen is the same sentence in
    square brackets, so what separates them is the markup vocabulary and nothing
    else.
    """
    page = prose_scan.scan_python_source("planted.py", _PLANTED_PAGE)
    assert [(occ.line, occ.region) for occ in page] == [(2, prose_scan.REGION_OUTPUT)], page

    unmarked = prose_scan.scan_python_source("planted.py", _PLANTED_UNMARKED)
    assert [(occ.line, occ.region) for occ in unmarked] == [(2, prose_scan.REGION_LITERAL)], (
        f"a literal with no HTML tag was classified as a page fragment: {unmarked}"
    )


def test_the_page_fragment_rule_does_not_fire_on_a_placeholder_in_prose() -> None:
    """The first draft did, and it is recorded here rather than remembered.

    A general ``<word>`` shape classified ``reliability_summary[<lead>]`` in a
    diagnostic string as markup. The vocabulary is the standard HTML element set
    for exactly this reason, so the specimen that broke the general rule is kept
    as a permanent negative.
    """
    placeholder = 'def render():\n    msg = "summary[<lead>] — not markup"\n'
    scanned = prose_scan.scan_python_source("planted.py", placeholder)
    assert [(occ.line, occ.region) for occ in scanned] == [(2, prose_scan.REGION_LITERAL)], scanned


def test_the_error_path_literals_are_counted_and_declared_and_do_NOT_fail() -> None:
    """Reader-facing on an error path. Declared, counted, and deliberately not a gate.

    Widening the gate to `raise` and `assert` messages is a PROPOSAL, not a
    decision infra may take alone: contract violation strings are compared against
    by tests in three other leads' packages.

    This carried `len(errors) >= 30` over the live tree with the message "the
    raise/assert sink has stopped matching". That is the same population-as-proxy
    defect the output floor died of, in a region that simply had not been burned
    down yet, so it is replaced by the same construction rather than left to fail
    later for the wrong reason.
    """
    planted = prose_scan.scan_python_source("planted.py", _PLANTED_ERROR)
    assert [(occ.line, occ.region) for occ in planted] == [(2, prose_scan.REGION_ERROR)], (
        "the raise/assert sink has stopped matching, or it has leaked into the "
        f"failing category: {planted}"
    )
    audit = prose_scan.audit_output_literals(_scan())
    assert audit.ok, "error-path literals must not enter the failing category by accident"


# --------------------------------------------------------------------------
# the three directions the burn-down fails in, on synthetic input
# --------------------------------------------------------------------------


def test_an_undeclared_file_fails() -> None:
    audit = prose_scan.audit_output_literals(
        [_occ("sim/new.py", 3, prose_scan.REGION_OUTPUT)], debt={"sim/old.py": 1}
    )
    assert not audit.ok
    assert audit.undeclared == {"sim/new.py": 1}
    assert any("NEW output-reaching" in line for line in audit.lines())


def test_a_risen_count_fails_and_names_both_numbers() -> None:
    audit = prose_scan.audit_output_literals(
        [_occ("sim/old.py", 3, prose_scan.REGION_OUTPUT) for _ in range(3)],
        debt={"sim/old.py": 1},
    )
    assert not audit.ok
    assert audit.risen == {"sim/old.py": (1, 3)}
    assert any("carried 1, now carries 3" in line for line in audit.lines())


def test_a_cleared_file_fails_as_STALE_so_the_entry_cannot_outlive_its_reason() -> None:
    audit = prose_scan.audit_output_literals([], debt={"sim/old.py": 1})
    assert not audit.ok
    assert audit.stale == ("sim/old.py",)
    assert any("STALE" in line for line in audit.lines())


def test_a_count_that_FALLS_fails_too_because_a_tolerated_fall_leaves_slack() -> None:
    """The fourth direction, and the one this pin shipped without.

    A fall was tolerated on the reasoning that three leads write in these files and
    a pin that goes red on their progress gets deleted. Planting in a fresh clone
    showed what that costs: at 2102917 the declared debt read 25 while the tree
    held 24, because @model had swept one in eval/reporting.py. The scanner printed
    the declared 25 and PASSed, so the slack was invisible AND refillable: a later
    commit could put a new literal back into that file and RISEN would not fire,
    because 2 is not greater than 2. A pin is a pin in both directions.
    """
    audit = prose_scan.audit_output_literals(
        [_occ("sim/old.py", 3, prose_scan.REGION_OUTPUT)], debt={"sim/old.py": 4}
    )
    assert not audit.ok
    assert audit.fallen == {"sim/old.py": (4, 1)}
    assert any("FALLEN" in line and "set it to 1" in line for line in audit.lines())


def test_the_refill_the_tolerated_fall_allowed_is_now_caught() -> None:
    """The hazard itself, not just the direction: sweep one, add one, unnoticed."""
    swept = prose_scan.audit_output_literals(
        [_occ("eval/r.py", 3, prose_scan.REGION_OUTPUT)], debt={"eval/r.py": 2}
    )
    refilled = prose_scan.audit_output_literals(
        [
            _occ("eval/r.py", 3, prose_scan.REGION_OUTPUT),
            _occ("eval/r.py", 9, prose_scan.REGION_OUTPUT),
        ],
        debt={"eval/r.py": 2},
    )
    assert not swept.ok, "the sweep must be recorded when it happens"
    assert refilled.ok, "once recorded at 2 the refill reads as no change; hence the above"


def test_the_declared_number_is_never_printed_without_the_found_number() -> None:
    """A tool that prints one of two numbers has chosen which one you compare against."""
    out = subprocess.run(
        [sys.executable, str(repo_root() / "tools" / "prose_scan.py"), "--repo", str(repo_root())],
        capture_output=True,
        text=True,
    ).stdout
    line = next(ln for ln in out.splitlines() if "declared" in ln)
    assert "found" in line, line
    assert "PIN, not a ceiling" in line, line


# --------------------------------------------------------------------------
# what the classifier sees that a search for string constants cannot
# --------------------------------------------------------------------------

_SPECIMEN = '''"""Docstring dash — prose, bounded."""

# comment dash — prose, bounded.

INTERNAL = "an internal literal — excluded by construction"
KEY_MAP = {"a — b": 1}


def show(rows, count):
    print(f"an f-string dash — with {count} rows")
    print("a concatenated" + " dash — here")
    print("an escaped dash \\u2014 invisible to a byte scan")
    report(status="a keyword dash — in a reporting kwarg")
    raise ValueError("an error-path dash — declared, not failing")


def report(status):
    assert status, "an assert dash — also error path"
'''


def _by_region(text: str) -> dict[str, list[prose_scan.Occurrence]]:
    out: dict[str, list[prose_scan.Occurrence]] = {}
    for occ in prose_scan.scan_python_source("specimen.py", text):
        out.setdefault(occ.region, []).append(occ)
    return out


def test_the_classifier_separates_all_five_regions_on_one_specimen() -> None:
    """C3.5. Five verdicts, one file, and no file name is consulted for any of them."""
    regions = _by_region(_SPECIMEN)
    assert len(regions.get(prose_scan.REGION_DOCSTRING, [])) == 1, regions
    assert len(regions.get(prose_scan.REGION_COMMENT, [])) == 1, regions
    assert len(regions.get(prose_scan.REGION_LITERAL, [])) == 2, regions
    assert prose_scan.REGION_CODE not in regions, regions

    output_lines = sorted(occ.line for occ in regions.get(prose_scan.REGION_OUTPUT, []))
    assert output_lines == [10, 11, 12, 13], regions.get(prose_scan.REGION_OUTPUT)

    error_lines = sorted(occ.line for occ in regions.get(prose_scan.REGION_ERROR, []))
    assert error_lines == [14, 18], regions.get(prose_scan.REGION_ERROR)


def test_the_fstring_and_the_concatenation_are_the_strengthening_over_the_floor() -> None:
    """The maintainer's 28 was a floor because it saw only direct string constants.

    Lines 10 and 11 are an f-string and a concatenation. A search for
    `print("...")` finds neither, and both are printed to a reader.
    """
    output = _by_region(_SPECIMEN)[prose_scan.REGION_OUTPUT]
    assert 10 in {occ.line for occ in output}, "the f-string segment was missed"
    assert 11 in {occ.line for occ in output}, "the concatenated segment was missed"


def test_an_ESCAPED_dash_in_a_printed_literal_is_counted() -> None:
    """`"\\u2014"` is an em dash to the reader and nothing to a scan of the bytes."""
    output = _by_region(_SPECIMEN)[prose_scan.REGION_OUTPUT]
    assert 12 in {occ.line for occ in output}, (
        "the escaped dash on line 12 was not counted. It is the case that made a commit "
        "claim '0 non-ASCII bytes and therefore 0 dashes' while 31 sat in tracked JSON."
    )


def test_a_keyword_argument_is_a_sink_and_the_rule_is_a_REFUSAL_not_a_name_list() -> None:
    """`status=` is not special. ANY keyword is, which is why nothing can be forgotten.

    An enumerated list of keyword names would leave whichever one nobody thought
    of as the next gap; that is how seven allow-lists here have already failed.
    """
    specimen = 'def f(x):\n    g(unheard_of_keyword="a dash — here")\n'
    regions = _by_region(specimen)
    assert len(regions.get(prose_scan.REGION_OUTPUT, [])) == 1, regions


def test_a_literal_that_is_NOT_a_sink_stays_excluded() -> None:
    """A module-level constant and a dict key are internal and must remain so."""
    specimen = 'A = "dash — one"\nB = {"dash — two": 3}\nC = A.replace("—", "-")\n'
    regions = _by_region(specimen)
    assert len(regions[prose_scan.REGION_LITERAL]) == 3, regions
    assert prose_scan.REGION_OUTPUT not in regions, regions


# --------------------------------------------------------------------------
# the general rule ADR-097 (4) draws: an exclusion is printed beside the verdict
# --------------------------------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/prose_scan.py", "--repo", ".", *args],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_every_region_is_printed_with_what_it_MEANS_on_every_run() -> None:
    """Three defects in one day were scope silently narrower than the claim."""
    proc = _cli()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for region in prose_scan.ALL_REGIONS:
        assert region in proc.stdout, f"{region} is missing from the verdict"
        assert prose_scan.REGION_LEGEND[region][:30] in proc.stdout, region
    assert "EXCLUDED and DECLARED" in proc.stdout
    assert "FAILING" in proc.stdout
    assert "NOT SEEN BY THIS SCANNER" in proc.stdout, (
        "the scanner no longer declares its own blind spot, which is the rule it was "
        "extended to obey"
    )
    assert proc.stdout.rstrip().endswith("verdict: PASS")


def test_the_legend_covers_every_region_so_a_new_one_cannot_be_added_silently() -> None:
    assert set(prose_scan.REGION_LEGEND) == set(prose_scan.ALL_REGIONS)


def test_the_scanner_reads_its_own_source_and_is_not_self_exempt() -> None:
    """The likeliest offender must not be the one file that can never report."""
    tracked = prose_scan.tracked_files(repo_root())
    assert "tools/prose_scan.py" in tracked
    assert "tests/test_prose_output_literals.py" in tracked


@pytest.mark.parametrize("specimen", [_SPECIMEN])
def test_the_specimen_parses_so_the_scan_is_never_silently_skipped(specimen: str) -> None:
    assert isinstance(ast.parse(specimen), ast.Module)
