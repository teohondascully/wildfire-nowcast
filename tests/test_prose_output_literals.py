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
`sim/` and `runs/` and are a burn-down.
"""

from __future__ import annotations

import ast
import functools
import subprocess
import sys
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


# --------------------------------------------------------------------------
# the live tree
# --------------------------------------------------------------------------


def test_infra_owns_zero_output_reaching_literals() -> None:
    """The 38 that were infra's are gone, and they may not come back."""
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
    """Three directions, all of them failures: undeclared, risen, stale."""
    audit = prose_scan.audit_output_literals(_scan())
    assert audit.ok, "\n".join(audit.lines())


def test_the_debt_belongs_to_other_leads_and_says_so() -> None:
    """A burn-down that infra could clear itself would be a to-do list, not a fence."""
    assert not any(path.startswith(INFRA_SURFACES) for path in prose_scan.OUTPUT_LITERAL_DEBT)
    assert all(count > 0 for count in prose_scan.OUTPUT_LITERAL_DEBT.values())


def test_the_output_category_is_not_passing_by_scoping_to_nothing() -> None:
    """The anti-vacuity control: the classifier still finds the class it bounds.

    If the sink analysis silently stopped matching, every file would read zero and
    both tests above would pass while the category measured nothing.
    """
    found = [occ for occ in _scan() if occ.region == prose_scan.REGION_OUTPUT]
    assert len(found) >= 20, (
        f"only {len(found)} output-reaching characters found against a declared debt of "
        f"{sum(prose_scan.OUTPUT_LITERAL_DEBT.values())}. Either three leads cleared their "
        "debt at once, or the sink analysis has stopped matching anything."
    )


def test_the_internal_literals_are_still_the_large_majority_and_still_excluded() -> None:
    """The exclusion is the point. It must remain an exclusion, not become a sweep."""
    scanned = _scan()
    internal = [occ for occ in scanned if occ.region == prose_scan.REGION_LITERAL]
    assert len(internal) > 250, (
        f"internal literals are down to {len(internal)} from 367 at 738e7b7. A sweep has "
        "started rewriting literals, which changes behaviour and artifact bytes."
    )


def test_the_error_path_literals_are_counted_and_declared_and_do_NOT_fail() -> None:
    """Reader-facing on an error path. Declared, counted, and deliberately not a gate.

    Widening the gate to `raise` and `assert` messages is a PROPOSAL, not a
    decision infra may take alone: contract violation strings are compared against
    by tests in three other leads' packages.
    """
    errors = [occ for occ in _scan() if occ.region == prose_scan.REGION_ERROR]
    assert len(errors) >= 30, f"only {len(errors)}; the raise/assert sink has stopped matching"
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


def test_a_count_that_merely_FALLS_is_fine() -> None:
    """Three leads write in these files. A pin red on their progress gets edited."""
    audit = prose_scan.audit_output_literals(
        [_occ("sim/old.py", 3, prose_scan.REGION_OUTPUT)], debt={"sim/old.py": 4}
    )
    assert audit.ok, audit.lines()


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
