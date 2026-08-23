"""Adopts the self-test modules other leads wrote but could not land here.

``tests/`` is infra's directory, so sim and modelling each built
their verification suite as an *importable module inside their own package*
rather than writing into a directory they do not own:

* ``wildfire_nowcast.sim.selftest``  - plain ``test_*`` functions, no fixtures.
* ``wildfire_nowcast.eval.selftest`` - known-answer ``Check`` objects.
* ``wildfire_nowcast.data.coarsen_2km_selftest`` - 14 checks.
* ``wildfire_nowcast.data.crossings_selftest``   - 25 checks.
* ``wildfire_nowcast.data.isotropy_selftest``    - 11 checks.
* ``wildfire_nowcast.data.leakage_selftest``     - 12 checks.

The four ``data`` modules were adopted later than the first two and had, at that
point, never been collected by pytest at all: ``testpaths = ["tests"]`` and they
live under ``src``. Their 134 assertions were reachable only through
``python -m wildfire_nowcast.data.<module>``, which nothing in ``make ci`` runs.

That was the right call, and the point of this file is that it does not leave
them orphaned. **Nothing here reimplements or edits their logic**; this module
only collects and runs it, so ownership stays where it belongs and their
standalone entry points (``python -m wildfire_nowcast.sim.selftest``) keep
working unchanged.

Collection is by INTROSPECTION, not from a hand-maintained list. If sim
adds a test to its module tomorrow, `make test` runs it that day without
touching this file - a wiring that needs infra's attention to stay
complete would silently rot the moment another lead is mid-flight. The
completeness tests below then check that each module's own runner sees the same
set, so the standalone path and the pytest path cannot drift apart.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from wildfire_nowcast.data import (
    coarsen_2km_selftest,
    crossings_selftest,
    isotropy_selftest,
    leakage_selftest,
)
from wildfire_nowcast.eval import selftest as eval_selftest
from wildfire_nowcast.sim import selftest as sim_selftest


def _zero_arg_tests(module: object) -> list[Callable[[], None]]:
    """Every ``test_*`` callable in ``module`` that pytest could call directly."""
    out = []
    for name, obj in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue  # imported into the namespace, not defined here
        if inspect.signature(obj).parameters:
            continue  # takes fixtures; not adoptable without a wrapper
        out.append(obj)
    return out


SIM_TESTS = _zero_arg_tests(sim_selftest)
EVAL_CHECKS = list(eval_selftest.CHECKS)


# --------------------------------------------------------------------------
# sim - wildfire_nowcast.sim.selftest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", SIM_TESTS, ids=lambda f: f.__name__)
def test_sim_selftest(case: Callable[[], None]) -> None:
    """Run one of sim's self-tests as a first-class pytest case.

    These target defects that render as plausible-but-wrong rather than as a
    crash - a mirrored fire, a confident arrival time the ensemble does not
    support, an ignition hour reported as a 24 km spot fire.
    """
    case()


def test_every_sim_selftest_is_collected() -> None:
    """Guard the wiring itself: a test nobody runs is a test nobody has."""
    assert SIM_TESTS, "no tests were collected from sim.selftest — the wiring is broken"
    declared = {n for n in sim_selftest.__all__ if n.startswith("test_")}
    collected = {f.__name__ for f in SIM_TESTS}
    assert declared <= collected, (
        f"declared in __all__ but not collected here: {sorted(declared - collected)}"
    )
    assert collected == declared, (
        "sim.selftest defines tests missing from its own __all__, so its standalone runner and "
        f"this suite disagree: {sorted(collected - declared)}"
    )


# --------------------------------------------------------------------------
# modelling - wildfire_nowcast.eval.selftest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("check", EVAL_CHECKS, ids=lambda f: f.__name__)
def test_eval_selftest(check: Callable[[], eval_selftest.Check]) -> None:
    """Run one of modelling's known-answer C5/C6 verifications.

    Every one has an answer known BEFORE the code runs - hand-computed or forced
    by an algebraic identity - because a metric verified only against its own
    output is verified against nothing. The load-bearing one is
    ``collapse_is_invisible_to_dispersion_ratio``: it pins the fact that
    ``dispersion_ratio`` scores a COLLAPSED ensemble at exactly 1.000, which is
    why ADR-011 moved G3 onto ``area_dispersion_ratio``. If that check ever
    stops being able to see the difference, G3 is being adjudicated with a blind
    instrument and this suite is the thing that says so.
    """
    result = check()
    assert result.passed, f"{result.name}: {result.detail}\n{result.values}"


def test_every_eval_check_is_registered() -> None:
    """A ``check_*`` written but never added to ``CHECKS`` runs nowhere."""
    assert EVAL_CHECKS, "no checks were collected from eval.selftest"
    defined = {
        name
        for name, obj in vars(eval_selftest).items()
        if name.startswith("check_")
        and callable(obj)
        and getattr(obj, "__module__", None) == eval_selftest.__name__
    }
    registered = {f.__name__ for f in EVAL_CHECKS}
    assert defined == registered, (
        f"defined but not in CHECKS: {sorted(defined - registered)}; "
        f"in CHECKS but not defined here: {sorted(registered - defined)}"
    )


def test_eval_selftest_runner_reports_a_raising_check_as_a_failure() -> None:
    """`run_all` must convert an exception into a failed Check, not a crash.

    A suite that aborts on the first exception reports "1 failure" when it may
    have twelve, which is the same punch-list property `common.contract` keeps.
    """

    def check_that_raises() -> eval_selftest.Check:
        raise RuntimeError("boom")

    results = eval_selftest.run_all([check_that_raises])
    assert len(results) == 1
    assert results[0].passed is False
    assert "boom" in results[0].detail


# --------------------------------------------------------------------------
# data - the four wildfire_nowcast.data.*_selftest modules
# --------------------------------------------------------------------------

#: Tag -> module, for readable parametrisation ids. Same introspection as sim:
#: nothing here names an individual check, so a check added to any of these
#: modules is run by `make test` the same day.
DATA_SELFTEST_MODULES: dict[str, object] = {
    "coarsen_2km": coarsen_2km_selftest,
    "crossings": crossings_selftest,
    "isotropy": isotropy_selftest,
    "leakage": leakage_selftest,
}

DATA_TESTS: list[tuple[str, Callable[[], None]]] = [
    (f"{tag}:{fn.__name__}", fn)
    for tag, module in sorted(DATA_SELFTEST_MODULES.items())
    for fn in _zero_arg_tests(module)
]

#: What the four modules held when they were adopted: 14 + 25 + 11 + 12. A
#: FLOOR, not a pin. Losing a check is the failure this guards; adding one is
#: not a failure, and a two-sided pin would punish the next person to write one.
#: The two-sided invariant is elsewhere: each module's own runner must see
#: exactly the set collected here, which self-updates and cannot rot.
DATA_TESTS_AT_ADOPTION = 62


@pytest.mark.parametrize(("case_id", "case"), DATA_TESTS, ids=[c[0] for c in DATA_TESTS])
def test_data_selftest(case_id: str, case: Callable[[], None]) -> None:
    """Run one of data's self-tests as a first-class pytest case.

    Three of these read ``data/events/crossings.json``, which is untracked, and
    return early when it is absent. Left alone they would report PASSED in a
    clone having examined nothing. They are skipped instead, so the report
    distinguishes "checked" from "could not check" - which is the entire
    difference between a green suite and a green-looking one.
    """
    if (
        getattr(case, "__module__", None) == crossings_selftest.__name__
        and case.__name__ in crossings_selftest.ARTIFACT_DEPENDENT
        and not crossings_selftest.artifact_present()
    ):
        pytest.skip(f"{case.__name__} reads an untracked artifact that is absent here")
    case()


def test_every_data_selftest_is_collected() -> None:
    """Each module's OWN runner must see exactly the set collected here.

    The four runners each scan their own ``globals()`` for ``test_*``. This
    reproduces that scan and compares it to what ``_zero_arg_tests`` adopted, so
    a check that grows a fixture argument (which the adoption silently drops and
    the standalone runner then crashes on) is reported rather than lost.
    """
    assert DATA_TESTS, "no tests were collected from the data self-test modules"
    for tag, module in sorted(DATA_SELFTEST_MODULES.items()):
        runner_sees = {
            name for name, obj in vars(module).items() if name.startswith("test_") and callable(obj)
        }
        adopted = {fn.__name__ for fn in _zero_arg_tests(module)}
        assert runner_sees, f"{tag}: its own runner would collect nothing"
        assert runner_sees == adopted, (
            f"{tag}: its standalone runner and this suite disagree. "
            f"runner-only={sorted(runner_sees - adopted)} "
            f"pytest-only={sorted(adopted - runner_sees)}"
        )
    assert len(DATA_TESTS) >= DATA_TESTS_AT_ADOPTION, (
        f"the data self-tests have SHRUNK: {len(DATA_TESTS)} collected, "
        f"{DATA_TESTS_AT_ADOPTION} present when they were adopted"
    )


def test_the_artifact_dependent_list_is_derived_from_the_source_not_trusted() -> None:
    """``ARTIFACT_DEPENDENT`` must name exactly the checks that can no-op.

    A hand-written list of "the tests that skip themselves" is a fact stored in
    two places, and the copy in the list is the one that goes stale. This
    re-derives it from the module's own AST: a ``test_*`` whose body contains a
    bare ``return`` inside an ``if`` is a check that can complete without
    reaching an assertion.

    Controlled in both directions: the derivation must find a non-empty set on
    ``crossings_selftest`` and an EMPTY one on ``leakage_selftest``, so a
    derivation that simply returns every test name cannot pass.
    """
    import ast
    import inspect as _inspect

    def _can_return_early(module: object) -> set[str]:
        tree = ast.parse(_inspect.getsource(module))
        out = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.If) and any(
                    isinstance(b, ast.Return) and b.value is None for b in inner.body
                ):
                    out.add(node.name)
        return out

    derived = _can_return_early(crossings_selftest)
    assert derived, "the derivation found nothing; it cannot be a control for anything"
    assert _can_return_early(leakage_selftest) == set(), (
        "the derivation flags tests in a module that has none, so it is not selective"
    )
    assert derived == set(crossings_selftest.ARTIFACT_DEPENDENT), (
        "crossings_selftest.ARTIFACT_DEPENDENT is stale. "
        f"can no-op but unlisted={sorted(derived - set(crossings_selftest.ARTIFACT_DEPENDENT))}; "
        f"listed but cannot no-op={sorted(set(crossings_selftest.ARTIFACT_DEPENDENT) - derived)}"
    )


def test_the_no_op_checks_really_do_pass_over_nothing() -> None:
    """Prove the skip above is protecting against a real vacuous pass.

    Points the module's own path resolver at a location that has never existed
    and asserts each listed check still returns cleanly. If one of them raised
    instead, the skip would be hiding a genuine failure and this file would be
    the thing doing the hiding.

    This test EXPIRES the day the early returns are replaced by something that
    reports. It is deliberately the same shape as the defect it records, so
    fixing the defect turns it red and the skip above has to go with it.
    """
    from pathlib import Path as _Path

    def _never() -> _Path:
        return _Path("/nonexistent/never/crossings.json")

    real = crossings_selftest.crossings_path
    try:
        crossings_selftest.crossings_path = _never
        assert crossings_selftest.artifact_present() is False
        for name in crossings_selftest.ARTIFACT_DEPENDENT:
            getattr(crossings_selftest, name)()
    finally:
        crossings_selftest.crossings_path = real
    assert crossings_selftest.artifact_present() == crossings_selftest.crossings_path().exists()


# --------------------------------------------------------------------------
# Cross-lead: the standalone entry points must keep working
# --------------------------------------------------------------------------


def test_standalone_runners_still_work() -> None:
    """Adoption must not break `python -m wildfire_nowcast.{sim,eval}.selftest`.

    Each lead debugs its own package through its own runner; if adopting these
    into pytest quietly broke that, the adoption would have cost more than it
    bought.
    """
    assert sim_selftest.run_all() == 0
    assert eval_selftest.main([]) == 0


def test_the_data_standalone_runners_still_work() -> None:
    """The same guarantee for the four data modules, with their own exit codes.

    They do not share a convention and the differences are load-bearing:
    ``coarsen_2km`` collects failures and returns 1, the other three raise. The
    ``crossings`` runner returns the NUMBER of checks it ran, so comparing it to
    the collected count ties the two execution paths together with a value
    rather than with a zero that any broken runner would also return.
    """
    assert coarsen_2km_selftest.main() == 0
    assert isotropy_selftest.run_all() == 0
    assert leakage_selftest.main() == 0
    n_crossings = len(_zero_arg_tests(crossings_selftest))
    assert crossings_selftest.run_all() == n_crossings
