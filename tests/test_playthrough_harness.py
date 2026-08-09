"""The mutation-coverage harness, tested the way it asks everyone else to test.

``common/playthrough.py`` exists to make one sentence enforceable: *a playthrough
that cannot fail must not ship*. That makes the harness itself the highest-risk
object in the repo — **if the coverage machinery is vacuous, every playthrough it
blesses is vacuous too, and the whole policy becomes a green light.** This module
therefore does two separate things:

1. **UNIT-PINS THE PROTOCOL.** Each rule the harness enforces gets a case that
   makes it fire: an undetected defect, a blind spot that closed, a probe failing
   on the clean world, a mutation that leaked out of its context manager.
2. **RUNS THE HARNESS THROUGH ITSELF.** :data:`HARNESS_PLAYTHROUGH` is a
   playthrough whose SUBJECT is a playthrough, whose observation is a coverage
   report, and whose planted defects are the ways a coverage harness goes blind —
   a probe that always passes, a defect that mutates nothing, a probe removed
   from the set. If the harness cannot detect a blind harness, it cannot be
   trusted to detect anyone else's.

The inner scenario is deliberately tiny and arithmetic — the sample variance of
four integers, whose Bessel-corrected value is known exactly — because the
harness's correctness must not depend on any model, metric or fixture in this
repo. It is the same reason ``common/calibration.py`` was validated on
constructed cases before being pointed at anything: an instrument validated
against the thing it measures has been calibrated, not tested.
"""


from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.common.playthrough import (
    Defect,
    Playthrough,
    PlaythroughError,
    Probe,
    attribute_defect,
    data_defect,
    no_defect,
    run,
)

# --------------------------------------------------------------------------
# [A14] SELF-DECLARATION, read by tests/test_playthrough_registry.py.
# Registration is AUTOMATIC and these constants are how this module identifies
# itself. They live HERE, beside the playthrough, so that adding or changing a
# playthrough never requires editing another lead's file — the mechanism fix for
# three consecutive forced cross-boundary writes (ADR-039 (6)).
# --------------------------------------------------------------------------
PLAYTHROUGH_OWNER = 'infra'
PLAYTHROUGH_NOTE = (
    'the coverage harness run through its own protocol. If this cannot go red, nothing it '
    'blesses can be trusted to go red either.'
)

# --------------------------------------------------------------------------
# a tiny inner playthrough with an arithmetically known answer
# --------------------------------------------------------------------------

#: Four integers. Mean 100, Bessel-corrected variance exactly 1000/3, population
#: variance exactly 250. Chosen so that ``ddof`` is visible at 1e-12 and so the
#: two known answers differ by 33%, which no tolerance can straddle by accident.
SAMPLE = (80, 90, 110, 120)
WANT_SAMPLE_VAR = 1000.0 / 3.0
WANT_MEAN = 100.0


@dataclass(frozen=True)
class World:
    """The inner scenario: numbers in, plus the estimator that will read them."""

    values: tuple[int, ...] = SAMPLE


class Estimator:
    """A stand-in for ``eval/metrics.py`` — an INSTRUMENT that can be mutated."""

    ddof = 1

    @classmethod
    def variance(cls, values: tuple[int, ...]) -> float:
        return float(np.var(np.asarray(values, dtype=np.float64), ddof=cls.ddof))

    @classmethod
    def mean(cls, values: tuple[int, ...]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64)))


def observe(world: World) -> dict[str, float]:
    return {
        "variance": Estimator.variance(world.values),
        "mean": Estimator.mean(world.values),
        "n": float(len(world.values)),
    }


EXACT_VARIANCE = Probe(
    "variance_exact",
    lambda obs: PT.approximately(obs["variance"], WANT_SAMPLE_VAR, tol=1e-9),
    note="the Bessel-corrected variance to 1e-9. The only probe that can see ddof.",
)
MEAN_PROBE = Probe(
    "mean_exact",
    lambda obs: PT.approximately(obs["mean"], WANT_MEAN, tol=1e-9),
    note="the mean is untouched by a variance defect; it catches a data shift instead.",
)
SCENARIO_GUARD = Probe(
    "scenario_has_four_points",
    lambda obs: obs["n"] == 4.0,
    note="pins the SCENARIO, not the instrument.",
    guard=True,
)

DDOF_DEFECT = Defect(
    "population_variance_ddof0",
    attribute_defect((Estimator, "ddof", 0)),
    note="a one-character change (ddof=0) that moves the reading 25%. The same defect "
    "modelling planted in metrics.py, where it moved 3% and only an exact case caught it.",
)
SHIFT_DEFECT = Defect(
    "every_value_shifted",
    data_defect(lambda w: replace(w, values=tuple(v + 7 for v in w.values))),
    note="a DATA defect: the spread is right and the location is wrong. Only the mean probe "
    "sees it, which is why the two probes are not redundant.",
)
BLIND_DEFECT = Defect(
    "reordering_the_values",
    data_defect(lambda w: replace(w, values=tuple(reversed(w.values)))),
    note="a DOCUMENTED BLIND SPOT: variance and mean are permutation-invariant, so no probe "
    "here can see a reordering. Asserted in the opposite direction so the day a probe becomes "
    "order-sensitive, the build says so.",
    detected=False,
)
#: The same data defect under a second name, used to prove the world is rebuilt
#: between arms rather than mutated in place.
SHIFT_DEFECT_CLONE = replace(SHIFT_DEFECT, name="every_value_shifted_again")

INNER = Playthrough(
    name="sample_variance",
    build=World,
    observe=observe,
    probes=(EXACT_VARIANCE, MEAN_PROBE, SCENARIO_GUARD),
    defects=(DDOF_DEFECT, SHIFT_DEFECT, BLIND_DEFECT),
    note="the smallest playthrough with a known answer: used to test the harness itself.",
)


# --------------------------------------------------------------------------
# 1. the protocol does what it says
# --------------------------------------------------------------------------


def test_a_well_formed_playthrough_reaches_full_mutation_coverage() -> None:
    report = run(INNER).assert_ok()
    assert report.mutation_coverage == 1.0
    assert report.passed
    assert all(report.clean.values()), report.clean


def test_each_defect_is_caught_by_exactly_the_probe_that_should_catch_it() -> None:
    """Attribution, not just detection — the ADR-032 (4) finding, mechanised.

    ``ddof`` is invisible to the mean and the shift is invisible to the variance.
    A harness that reported "something failed" would let a probe be deleted with
    no consequence; this is the map that makes the deletion break the build.
    """
    outcomes = {o.name: o for o in run(INNER).outcomes}
    assert outcomes["population_variance_ddof0"].caught_by == ("variance_exact",)
    assert outcomes["every_value_shifted"].caught_by == ("mean_exact",)
    assert outcomes["reordering_the_values"].caught_by == ()


def test_the_sole_catcher_map_names_the_load_bearing_probes() -> None:
    """`only the RANGE catches a missing square root` — as data, not a comment."""
    report = run(INNER)
    assert report.sole_catchers == {
        "population_variance_ddof0": "variance_exact",
        "every_value_shifted": "mean_exact",
    }
    assert any("SOLE catcher" in line for line in report.reporting)


def test_a_scenario_guard_is_not_reported_as_a_dead_probe() -> None:
    assert run(INNER).dead_probes == ()


def test_a_probe_that_catches_nothing_is_reported_but_does_not_fail_the_build() -> None:
    """Reporting tier, C-1's two tiers applied to the harness itself."""
    idle = Probe("idle", lambda obs: True, note="catches nothing and claims to judge.")
    report = run(replace(INNER, probes=(*INNER.probes, idle)))
    assert report.passed, report.failures
    assert report.dead_probes == ("idle",)
    assert any("catching NO declared defect" in line for line in report.reporting)
    with pytest.raises(PlaythroughError, match="catching NO declared defect"):
        report.assert_ok(strict=True)


# --------------------------------------------------------------------------
# 2. THE PLANTED DEFECTS OF THE HARNESS ITSELF
# --------------------------------------------------------------------------


def test_AN_UNDETECTED_PLANTED_DEFECT_IS_A_HARD_FAILURE() -> None:
    """THE assertion this whole module exists for.

    A playthrough declaring a defect that no probe catches is exactly simviz's
    first coarsening playthrough (ADR-031 (5)), which agreed with its defect
    within 1% and could not have failed. Here the probe set is reduced to the
    mean, which cannot see ``ddof``: the harness must refuse it.
    """
    blind = replace(INNER, probes=(MEAN_PROBE, SCENARIO_GUARD), defects=(DDOF_DEFECT,))
    report = run(blind)
    assert not report.passed
    assert report.mutation_coverage == 0.0
    assert any("WAS NOT DETECTED" in f for f in report.failures), report.failures
    with pytest.raises(PlaythroughError, match="WAS NOT DETECTED"):
        report.assert_ok()


def test_a_defect_that_mutates_nothing_cannot_be_declared_detectable() -> None:
    """The control that proves the harness does not hallucinate catches."""
    empty = Defect("changes_nothing", no_defect(), note="the null mutation.")
    report = run(replace(INNER, defects=(empty,)))
    assert not report.passed
    assert any("changes_nothing" in f and "NOT DETECTED" in f for f in report.failures)


def test_a_blind_spot_that_CLOSES_is_also_a_failure() -> None:
    """`update the record, do not delete the test`, enforced rather than written.

    The reordering defect is declared undetectable. Add an order-sensitive probe
    and it becomes detectable — which is good news about the instrument and must
    still break the build, because the record now says something false.
    """
    # An order-sensitive observation: the harness only ever sees `observe`'s
    # output, so order sensitivity has to enter there.
    def observe_with_order(world: World) -> dict[str, float]:
        out = observe(world)
        out["first"] = float(world.values[0])
        return out

    ordered = Probe(
        "first_value_is_80",
        lambda obs: obs["first"] == 80.0,
        note="order-sensitive on purpose: it closes the declared blind spot.",
    )
    report = run(
        replace(
            INNER,
            observe=observe_with_order,
            probes=(*INNER.probes, ordered),
            defects=(DDOF_DEFECT, BLIND_DEFECT),
        )
    )
    assert not report.passed
    assert any("BLIND SPOT" in f and "UPDATE THE RECORD" in f for f in report.failures)


def test_a_probe_failing_on_the_clean_world_is_a_hard_failure() -> None:
    """`the scenario must pass BEFORE the defect is planted`, as a protocol rule.

    Both existing playthroughs wrote that sentence by hand. If the clean world
    already fails, every "catch" below is attributable to the scenario and the
    coverage map is a fiction.
    """
    wrong = Probe("wrong_answer", lambda obs: PT.approximately(obs["variance"], 250.0, tol=1e-9))
    report = run(replace(INNER, probes=(wrong, EXACT_VARIANCE, MEAN_PROBE)))
    assert not report.passed
    assert any("FAILS on the CLEAN world" in f for f in report.failures)


def test_a_probe_that_RAISES_counts_as_a_catch_and_the_message_is_kept() -> None:
    """A mutated instrument often makes a probe explode rather than return False.

    That is a legitimate detection, but an ambiguous one, so the exception text is
    recorded and printed instead of being silently scored as a pass or a fail.
    """
    def explodes(obs: dict[str, float]) -> bool:
        if obs["variance"] < WANT_SAMPLE_VAR:
            raise ValueError("variance shrank")
        return True

    report = run(replace(INNER, probes=(Probe("explodes", explodes), MEAN_PROBE)))
    outcome = {o.name: o for o in report.outcomes}["population_variance_ddof0"]
    assert "explodes" in outcome.caught_by
    assert "ValueError: variance shrank" in outcome.errors["explodes"]


def test_an_instrument_mutation_does_not_leak_out_of_its_context() -> None:
    """A leaked monkeypatch corrupts every arm after it and presents as a flake."""
    assert Estimator.ddof == 1
    run(INNER)
    assert Estimator.ddof == 1, "attribute_defect did not restore the instrument"


def test_the_world_is_rebuilt_for_every_defect() -> None:
    """Two data defects in sequence must not compose. Measured, not assumed."""
    twice = replace(INNER, defects=(SHIFT_DEFECT, SHIFT_DEFECT_CLONE, BLIND_DEFECT))
    outcomes = {o.name: o for o in run(twice).outcomes}
    # If the world were shared, the second shift would land on the first and the
    # blind spot (permutation) would start failing for an unrelated reason.
    assert outcomes["every_value_shifted"].caught_by == ("mean_exact",)
    assert outcomes["every_value_shifted_again"].caught_by == ("mean_exact",)
    assert outcomes["reordering_the_values"].caught_by == ()


# --------------------------------------------------------------------------
# 3. malformed playthroughs are refused at CONSTRUCTION
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"defects": ()}, "no defects declared"),
        ({"probes": ()}, "no probes"),
        ({"probes": (SCENARIO_GUARD,)}, "every probe is a scenario guard"),
        ({"defects": (BLIND_DEFECT,)}, "every declared defect is a blind spot"),
        (
            {"defects": (DDOF_DEFECT, replace(DDOF_DEFECT, note="  "))},
            "states no reason",
        ),
        (
            {"defects": (DDOF_DEFECT, replace(DDOF_DEFECT, plant=no_defect()))},
            "duplicate defect names",
        ),
    ],
)
def test_a_malformed_playthrough_is_refused_before_it_can_run(kwargs, match) -> None:
    """Refused at CONSTRUCTION, not at run time: a playthrough that never declares
    a defect must not be able to reach a green result by simply not being run."""
    with pytest.raises(PlaythroughError, match=match):
        replace(INNER, **kwargs)


# --------------------------------------------------------------------------
# 4. THE HARNESS RUN THROUGH ITSELF
# --------------------------------------------------------------------------


def _coverage(inner: Playthrough) -> dict[str, object]:
    report = run(inner)
    return {
        "passed": report.passed,
        "coverage": report.mutation_coverage,
        "n_dead": len(report.dead_probes),
        "n_sole": len(report.sole_catchers),
    }


HARNESS_PLAYTHROUGH = Playthrough(
    name="mutation_coverage_harness",
    build=lambda: INNER,
    observe=_coverage,
    probes=(
        Probe(
            "clean_harness_passes",
            lambda obs: bool(obs["passed"]),
            note="a correct playthrough must be blessed.",
        ),
        Probe(
            "coverage_is_total",
            lambda obs: obs["coverage"] == 1.0,
            note="every declared defect detected.",
        ),
        Probe(
            "load_bearing_probes_are_named",
            lambda obs: obs["n_sole"] == 2,
            note="the sole-catcher map is populated; a harness that named none would hide "
            "exactly the ADR-032 (4) finding.",
        ),
    ),
    defects=(
        Defect(
            "inner_probe_set_reduced_until_blind",
            data_defect(lambda pt: replace(pt, probes=(MEAN_PROBE, SCENARIO_GUARD))),
            note="THE defect this policy exists for: a playthrough whose probes cannot see its "
            "own planted defect. simviz's first coarsening playthrough, exactly.",
        ),
        Defect(
            "inner_probes_all_trivially_true",
            data_defect(
                lambda pt: replace(
                    pt,
                    probes=(
                        Probe("always", lambda obs: True, note="a stubbed assertion"),
                        MEAN_PROBE,
                    ),
                )
            ),
            note="the bluntest vacuous playthrough: assertions that cannot fail. It must not "
            "reach full coverage.",
        ),
        Defect(
            "inner_defect_mutates_nothing",
            data_defect(lambda pt: replace(pt, defects=(Defect("noop", no_defect(), note="x"),))),
            note="a declared defect that changes nothing — the harness must not credit itself "
            "with catching it.",
        ),
    ),
    note="the harness, run through its own protocol. If this cannot go red, nothing below it "
    "can be trusted to go red either.",
)

#: [A14] The registry's entry-point convention is a module-level ``PLAYTHROUGH``.
#: Aliased rather than renamed: `HARNESS_PLAYTHROUGH` is referenced by name in this
#: file's own tests, and a rename to satisfy a convention is how a convention starts
#: costing more than it saves.
PLAYTHROUGH = HARNESS_PLAYTHROUGH


def test_THE_HARNESS_PASSES_ITS_OWN_PROTOCOL() -> None:
    report = run(HARNESS_PLAYTHROUGH).assert_ok()
    assert report.mutation_coverage == 1.0
    print(PT.format_report(report))


def test_the_formatted_report_names_every_defect_and_the_verdict() -> None:
    """`make playthrough` prints this; a report that omitted a defect would let a
    dropped mutation pass unnoticed at the one moment a human is looking."""
    text = PT.format_report(run(INNER))
    for defect in INNER.defects:
        assert defect.name in text
    assert "mutation coverage 100%" in text
    assert "verdict: PASS" in text
    assert "BLIND SPOT (must NOT be caught)" in text


def test_adopting_a_foreign_caught_map_enforces_the_same_requirement() -> None:
    """`coverage_from_caught_map` is how a playthrough infra does not OWN gets
    the same coverage requirement without anyone editing another lead's file."""
    ok = PT.coverage_from_caught_map(
        "foreign", {"nearest": ["disc:connectivity"], "all": ["disc:area"]}, clean_passes=True
    )
    assert ok.passed and ok.mutation_coverage == 1.0

    blind = PT.coverage_from_caught_map("foreign", {"nearest": []}, clean_passes=True)
    assert not blind.passed
    assert any("WAS NOT DETECTED" in f for f in blind.failures)

    dirty = PT.coverage_from_caught_map("foreign", {"nearest": ["x"]}, clean_passes=False)
    assert not dirty.passed
    assert any("does not pass every scenario" in f for f in dirty.failures)

    none_declared = PT.coverage_from_caught_map("foreign", {}, clean_passes=True)
    assert not none_declared.passed
    assert any("NO planted defects at all" in f for f in none_declared.failures)


def test_approximately_refuses_none_and_non_finite() -> None:
    """C1.5's lesson at the choke point: ``None`` must not compare its way to a pass."""
    assert PT.approximately(1.0, 1.0, tol=0.0)
    assert not PT.approximately(None, 1.0, tol=1e9)
    assert not PT.approximately(float("nan"), 1.0, tol=1e9)
    assert not PT.approximately(float("inf"), 1.0, tol=1e9)
