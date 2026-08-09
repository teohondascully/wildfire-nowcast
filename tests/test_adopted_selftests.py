"""Adopts the self-test modules other leads wrote but could not land here.

``tests/`` is infra's directory, so sim and modelling each built
their verification suite as an *importable module inside their own package*
rather than writing into a directory they do not own:

* ``wildfire_nowcast.sim.selftest``  — plain ``test_*`` functions, no fixtures.
* ``wildfire_nowcast.eval.selftest`` — known-answer ``Check`` objects.

That was the right call, and the point of this file is that it does not leave
them orphaned. **Nothing here reimplements or edits their logic**; this module
only collects and runs it, so ownership stays where it belongs and their
standalone entry points (``python -m wildfire_nowcast.sim.selftest``) keep
working unchanged.

Collection is by INTROSPECTION, not from a hand-maintained list. If sim
adds a test to its module tomorrow, `make test` runs it that day without
touching this file — a wiring that needs infra's attention to stay
complete would silently rot the moment another lead is mid-flight. The
completeness tests below then check that each module's own runner sees the same
set, so the standalone path and the pytest path cannot drift apart.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

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
# sim — wildfire_nowcast.sim.selftest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", SIM_TESTS, ids=lambda f: f.__name__)
def test_sim_selftest(case: Callable[[], None]) -> None:
    """Run one of sim's self-tests as a first-class pytest case.

    These target defects that render as plausible-but-wrong rather than as a
    crash — a mirrored fire, a confident arrival time the ensemble does not
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
# modelling — wildfire_nowcast.eval.selftest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("check", EVAL_CHECKS, ids=lambda f: f.__name__)
def test_eval_selftest(check: Callable[[], eval_selftest.Check]) -> None:
    """Run one of modelling's known-answer C5/C6 verifications.

    Every one has an answer known BEFORE the code runs — hand-computed or forced
    by an algebraic identity — because a metric verified only against its own
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
