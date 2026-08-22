"""PLAYTHROUGH (ADR-030) - does ``area_dispersion_ratio`` recover a KNOWN spread?

This file exists because of a gap ADR-030 makes fatal: **G3 is adjudicated on
``area_dispersion_ratio`` and that metric had no known-answer test at all.** Every
check we had on it was an internal-consistency check or a comparison against
another number it produced itself. ADR-030's standing requirement is that no gate
is adjudicated on a metric whose implementation lacks a playthrough that recovers
a known answer AND detects a planted defect.

WHAT IS KNOWN BY CONSTRUCTION
-----------------------------
For an ensemble of ``M`` members whose per-window burned AREA is drawn iid with
variance ``c*V``, scored against a truth area drawn independently with variance
``V``, the metric's two sums have exact expectations::

    numerator   = E[ s^2 * (M+1)/M ]      = c*V*(M+1)/M
    denominator = E[ (mean - truth)^2 ]   = c*V/M + V

and ``metrics._ratio`` returns the SQUARE ROOT of their quotient, so the pooled
reading converges to a CLOSED FORM depending on nothing but ``c`` and ``M``::

    ratio(c, M) = sqrt( c * (M + 1) / (c + M) )

    c = 0     -> 0       exactly (a collapsed ensemble)
    c = 1     -> 1       exactly (spread == error: a calibrated ensemble)
    c = 4     -> 1.8439  at M=16 (over-dispersed)
    c = 1/4   -> 0.5114  at M=16 (a HALF-WIDTH ensemble)

**THE SQUARE ROOT IS NOT A DETAIL AND WRITING THIS TEST IS HOW I FOUND IT.**
``area_dispersion_ratio`` is in STANDARD-DEVIATION units, not variance units. My
own insights item 44 read the M5 candidate's 0.226 as a variance ratio and
reported the ensemble as "too narrow by ~2.1x in SD (1/sqrt(0.226))". That is
wrong by a square root: 0.2147 IS the SD ratio, so the G3 candidate's ensemble is
**~4.7x too narrow in SD**, and ADR-027's "roughly four times too narrow" was
right for a reason its own supporting insight got backwards. Inverting the closed
form, 0.2147 corresponds to ``c = 0.0435`` - the ensemble carries **4.4% of the
variance it needs**. A gate criterion nobody had ever run a known answer through
was being quoted in the wrong units in the file that explains it.

Nothing here is fitted, and nothing here is read back out of ``metrics.py``: the
closed form is written from the DEFINITION of a spread-skill ratio, in this file,
and the metric has to come to it.

THE PLANTED DEFECTS, AND THEY MUST BE CAUGHT
--------------------------------------------
A playthrough that cannot fail is the exact thing ADR-030 exists to prevent, so
two real bugs are planted in a CORRECTLY-BUILT ensemble and the harness must
reject both:

1. ``silently_collapsed`` - every member replaced by member 0. This is the shape
   of a real bug (an ensemble loop that reuses one draw), and it is invisible to
   every accuracy metric, because the ensemble MEAN is still a legitimate
   forecast field.
2. ``area_biased`` - a correctly-dispersed ensemble whose areas are all shifted.
   The reading collapses for a reason that has NOTHING to do with width, and the
   harness must both reject it and ATTRIBUTE it, via
   ``area_error_decomposition.bias_fraction``. That distinction is the one M5's
   P19 turned on, so a playthrough that could not separate them would leave the
   project's main dispersion diagnosis untested.

AND ONE DEFECT THE METRIC CANNOT SEE, ASSERTED AS A LIMITATION
--------------------------------------------------------------
``half_duplicated`` - half the members are exact copies of the other half - moves
the reading only from 1.00 to ``sqrt(((M-2)/(M-1))((M+1)/M)/((M+2)/M))`` = 0.939
at M=16, because duplication barely changes the SAMPLE VARIANCE and only inflates
the ensemble mean's own error. It is asserted here as a known blind spot rather
than left undiscovered: a playthrough that quietly dropped the defect it failed
to catch would be exactly the green-but-vacuous shape ADR-030 was written about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest

from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.eval import metrics as M
from wildfire_nowcast.eval.metrics import aggregate, evaluate

# --------------------------------------------------------------------------
# [A14] SELF-DECLARATION, read by tests/test_playthrough_registry.py.
# Registration is AUTOMATIC and these constants are how this module identifies
# itself. They live HERE, beside the playthrough, so that adding or changing a
# playthrough never requires editing another lead's file - the mechanism fix for
# three consecutive forced cross-boundary writes (ADR-039 (6)).
# --------------------------------------------------------------------------
PLAYTHROUGH_OWNER = "infra (adopted from modelling)"
PLAYTHROUGH_NOTE = (
    "G3's dispersion half. Closed-form spread-skill identity; 7 defects, 3 of them planted in "
    "eval/metrics.py itself."
)

# Grid big enough to hold every planted area with room to spare, and small
# enough that 600 windows x 16 members is a fast test.
HEIGHT, WIDTH = 16, 32
N_CELLS = HEIGHT * WIDTH
N_MEMBERS = 16
N_WINDOWS = 600
BASE_AREA = 120
#: Areas are drawn from a DISCRETE UNIFORM on 7 points at spacing ``g``, whose
#: variance is exactly ``g^2 (7^2 - 1)/12 = 4 g^2``. Discrete and exact on
#: purpose: areas are cell counts, so a rounded Gaussian would have a variance
#: that is only approximately what the closed form assumes.
BASE_SPACING = 4
SUPPORT = np.arange(-3, 4)

#: Pooled-estimator tolerance. The ratio is a ratio of two sums over 600 windows,
#: so it carries real sampling error; this is stated rather than tuned until the
#: test passes, and the collapsed case is asserted EXACTLY because it has no
#: sampling error at all.
TOLERANCE = 0.10


def closed_form_ratio(c: float, n_members: int) -> float:
    """``sqrt(c (M+1) / (c + M))`` - the spread-skill identity, from the definition.

    The square root is part of the definition of `area_dispersion_ratio`, not a
    convenience: the criterion is in SD units. See the module docstring.
    """
    return float(np.sqrt(c * (n_members + 1.0) / (c + n_members)))


def _areas(rng: np.random.Generator, spacing: int, size: tuple[int, ...]) -> np.ndarray:
    return BASE_AREA + spacing * rng.choice(SUPPORT, size=size)


def build_scenario(
    c: float,
    *,
    seed: int = 20260808,
    n_members: int = N_MEMBERS,
    n_windows: int = N_WINDOWS,
) -> list[dict]:
    """One window per entry, member areas at variance ``c*V``, truth at ``V``.

    ``c = 0`` puts every member at the distribution MEAN, which is a collapsed
    ensemble whose numerator is exactly zero.
    """
    rng = np.random.default_rng(seed)
    spacing = int(round(BASE_SPACING * np.sqrt(c)))
    if c > 0 and spacing < 1:
        raise ValueError(f"c={c} is too small to express on an integer cell grid")
    out = []
    for _ in range(n_windows):
        truth_area = int(_areas(rng, BASE_SPACING, ()))
        if c == 0:
            member_areas = np.full(n_members, BASE_AREA, dtype=int)
        else:
            member_areas = _areas(rng, spacing, (n_members,))
        out.append(
            {
                "member_areas": np.asarray(member_areas, dtype=int),
                "truth_area": truth_area,
            }
        )
    return out


def _fields(member_areas: np.ndarray, truth_area: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn planted AREAS into a C5 ``samples``/``truth`` pair, horizon 1.

    Cells are filled in raster order, so the area - the only quantity this
    playthrough is about - is exactly the number planted.
    """
    n_members = int(member_areas.size)
    samples = np.zeros((n_members, 1, N_CELLS), dtype=np.uint8)
    for m, area in enumerate(member_areas):
        samples[m, 0, : int(area)] = 1
    truth = np.zeros((1, N_CELLS), dtype=np.uint8)
    truth[0, : int(truth_area)] = 1
    return (
        samples.reshape(n_members, 1, HEIGHT, WIDTH),
        truth.reshape(1, HEIGHT, WIDTH),
    )


def score_scenario(windows: list[dict]) -> dict:
    """Run the scenario end to end through C6 and return the pooled reading."""
    results = [evaluate(*_fields(w["member_areas"], w["truth_area"])) for w in windows]
    pooled = aggregate(results)["by_mask"]["domain"]
    return {
        "area_dispersion_ratio": pooled["area_dispersion_ratio"],
        "n_windows": len(windows),
    }


def verdict(windows: list[dict], expected_c: float, *, tolerance: float = TOLERANCE) -> dict:
    """PASS/FAIL: does C6 recover the ratio this scenario was BUILT to have?"""
    got = score_scenario(windows)["area_dispersion_ratio"]
    want = closed_form_ratio(expected_c, N_MEMBERS)
    ok = got is not None and abs(got - want) <= max(tolerance, tolerance * want)
    return {
        "passed": bool(ok),
        "observed": got,
        "expected": want,
        "expected_c": expected_c,
        "tolerance": tolerance,
    }


# --------------------------------------------------------------------------
# 1. the metric recovers a known spread ACROSS A RANGE
# --------------------------------------------------------------------------


def test_collapsed_ensemble_scores_exactly_zero() -> None:
    """A collapsed ensemble has zero numerator, so the ratio is 0.0 EXACTLY.

    Asserted exactly rather than approximately: there is no sampling error in
    zero, and a tolerance here would hide the case the whole metric exists for.
    """
    got = score_scenario(build_scenario(0.0))["area_dispersion_ratio"]
    assert got == 0.0, f"a collapsed ensemble must score exactly 0.0, got {got!r}"


def test_calibrated_ensemble_recovers_one() -> None:
    """Spread drawn from the SAME law as the error must score ~1.0.

    This is the load-bearing case: it is the only one that tests the ``(M+1)/M``
    finite-ensemble factor, which is what makes the ratio 1 for a calibrated
    ensemble rather than ``M/(M+1)``.
    """
    v = verdict(build_scenario(1.0), 1.0)
    assert v["passed"], v
    assert 0.90 <= v["observed"] <= 1.10, v


def test_over_dispersed_ensemble_is_above_the_g3_bar() -> None:
    v = verdict(build_scenario(4.0), 4.0)
    assert v["passed"], v
    assert v["observed"] > 1.2, f"4x spread must read as over-dispersed, got {v['observed']}"


def test_under_dispersed_ensemble_reads_as_half_width_in_sd_units() -> None:
    """c = 1/4 - a HALF-WIDTH ensemble - must read ~0.51, not ~0.26 and not ~1.

    This is the case that pins the UNITS, and the units are what ADR-027's
    "~4x too narrow" rests on. Our own G3 candidate read 0.2147, which is BELOW
    this: it is narrower than half width, at ~4.7x too narrow in SD.
    """
    v = verdict(build_scenario(0.25), 0.25)
    assert v["passed"], v
    assert 0.45 <= v["observed"] <= 0.60, v
    assert v["observed"] > 0.2147, (
        "a half-width ensemble must score ABOVE the M5 candidate's 0.2147; if it did "
        "not, the candidate would not be the worse of the two and ADR-027's reading "
        "of the magnitude would be wrong"
    )


@pytest.mark.parametrize("c", [0.25, 1.0, 4.0])
def test_the_reading_is_monotone_and_matches_the_closed_form(c: float) -> None:
    v = verdict(build_scenario(c), c)
    assert v["passed"], v


def test_one_window_matches_the_definition_to_machine_precision() -> None:
    """No Monte Carlo: one hand-built window, arithmetic done here, exact match.

    The statistical cases above can only ever agree within sampling error. This
    one pins the metric to its own definition at 1e-12, so a change to the
    estimator (ddof, the (M+1)/M factor, the pooling order) cannot hide inside a
    tolerance that was sized for sampling noise.
    """
    member_areas = np.array([90, 100, 110, 140], dtype=int)
    truth_area = 130
    samples, truth = _fields(member_areas, truth_area)
    got = evaluate(samples, truth)["by_mask"]["domain"]["area_dispersion_ratio"]

    m = float(member_areas.size)
    a = member_areas.astype(np.float64)
    want = np.sqrt((a.var(ddof=1) * (m + 1.0) / m) / (a.mean() - truth_area) ** 2)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12), (got, want)


# --------------------------------------------------------------------------
# 2. THE PLANTED DEFECTS. If these ever pass, the harness is vacuous.
# --------------------------------------------------------------------------


def plant_silent_collapse(windows: list[dict]) -> list[dict]:
    """Every member becomes member 0 - an ensemble loop that reused one draw."""
    return [
        {
            "member_areas": np.full_like(w["member_areas"], w["member_areas"][0]),
            "truth_area": w["truth_area"],
        }
        for w in windows
    ]


def plant_half_duplicated(windows: list[dict]) -> list[dict]:
    """Half the members are copies of the other half. Subtle, and real."""
    out = []
    for w in windows:
        a = w["member_areas"]
        half = a[: a.size // 2]
        out.append({"member_areas": np.concatenate([half, half]), "truth_area": w["truth_area"]})
    return out


def plant_area_bias(windows: list[dict], shift: int = 40) -> list[dict]:
    """A correctly-DISPERSED ensemble put in the wrong PLACE.

    Spread is untouched; every member's area is shifted by a constant. The
    reading must fall, and the harness must say WHY - otherwise "too narrow" and
    "systematically wrong" are indistinguishable, which is precisely the
    confusion M5's P19 was written to resolve.
    """
    return [
        {"member_areas": w["member_areas"] + shift, "truth_area": w["truth_area"]} for w in windows
    ]


def _bias_fraction(windows: list[dict]) -> float:
    results = [evaluate(*_fields(w["member_areas"], w["truth_area"])) for w in windows]
    pooled = aggregate(results)["by_mask"]["domain"]
    return float(pooled["area_error_decomposition"]["bias_fraction"])


def test_the_harness_catches_a_silently_collapsed_member_set() -> None:
    """The negative control ADR-030 requires. This assertion is the test.

    The defect is planted in an ensemble that PASSES before it is planted, so a
    failure here can only be the defect and never the scenario.
    """
    good = build_scenario(1.0)
    assert verdict(good, 1.0)["passed"], "the scenario must pass BEFORE the defect is planted"

    bad = verdict(plant_silent_collapse(good), 1.0)
    assert not bad["passed"], (
        "a silently collapsed ensemble was scored as correctly dispersed — "
        f"the playthrough cannot fail and is therefore vacuous: {bad}"
    )
    assert bad["observed"] == 0.0


def test_the_harness_catches_a_correctly_wide_ensemble_in_the_wrong_place() -> None:
    """Second planted defect: spread is right, POSITION is wrong. Caught AND attributed."""
    good = build_scenario(1.0)
    assert verdict(good, 1.0)["passed"], "the scenario must pass BEFORE the defect is planted"
    assert _bias_fraction(good) < 0.05, "the clean scenario must be essentially unbiased"

    biased = plant_area_bias(good)
    bad = verdict(biased, 1.0)
    assert not bad["passed"], (
        f"a systematically displaced ensemble read as correctly dispersed: {bad}"
    )
    # Attribution, not just detection: the denominator must be BIAS-dominated, so
    # a reader can tell this failure apart from a genuinely narrow ensemble.
    assert _bias_fraction(biased) > 0.9, _bias_fraction(biased)


def test_member_duplication_is_a_DOCUMENTED_BLIND_SPOT_of_this_metric() -> None:
    """A defect the metric CANNOT see, asserted rather than left undiscovered.

    Duplicating half the members leaves the sample variance almost unchanged and
    only inflates the ensemble mean's own error, so the reading moves from 1.00 to
    a closed-form 0.939 at M=16 - inside any tolerance anyone would choose.
    **`area_dispersion_ratio` is therefore near-blind to member duplication**, and
    that belongs in the record next to the gate it adjudicates.
    """
    m = float(N_MEMBERS)
    want = float(np.sqrt(((m - 2.0) / (m - 1.0)) * ((m + 1.0) / m) / ((m + 2.0) / m)))
    got = verdict(plant_half_duplicated(build_scenario(1.0)), 1.0)
    assert got["observed"] == pytest.approx(want, abs=0.05), (got, want)
    assert got["passed"], (
        "if this ever FAILS, the metric has become sensitive to duplication and this "
        "documented blind spot has closed — update the record, do not delete the test"
    )


def test_the_harness_rejects_a_wrongly_specified_expectation() -> None:
    """A correctly-built c=1 ensemble must FAIL a c=4 expectation.

    Guards the scorer itself: a `verdict` that passed everything would make every
    assertion above trivially true.
    """
    assert not verdict(build_scenario(1.0), 4.0)["passed"]


# --------------------------------------------------------------------------
# 3. MUTATION TESTS - defects planted in the INSTRUMENT, not in the data.
#
# The maintainer's challenge, and it is the right one: *a collapsed-member-set
# defect that the metric would catch anyway is not a test of the metric.* Section
# 2 plants defects in the FORECAST. These plant them in `metrics.py` itself and
# ask whether this file notices - which is the only way to answer "can this
# playthrough actually fail".
#
# It also shows exactly WHICH assertion earns its keep, and two of the answers
# are uncomfortable:
#   * dropping the (M+1)/M finite-ensemble factor moves the calibrated reading by
#     only 3%, INSIDE the sampling tolerance - the statistical cases cannot see
#     it and ONLY the 1e-12 single-window assertion catches it;
#   * dropping the square root leaves the CALIBRATED case at exactly 1.000 and is
#     invisible to it - only scoring a RANGE catches it.
# A playthrough with one regime and one tolerance would have missed both.
# --------------------------------------------------------------------------


def _mutate(monkeypatch, **kwargs) -> None:
    """Install a mutated `_area_dispersion` / `_ratio` into the live metrics module."""
    from wildfire_nowcast.eval import metrics as M

    if "ratio" in kwargs:
        monkeypatch.setattr(M, "_ratio", kwargs["ratio"])
    if "area" in kwargs:
        monkeypatch.setattr(M, "_area_dispersion", kwargs["area"])


def _no_sqrt(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(max(numerator, 0.0) / denominator)


def _biased_variance(member_event, truth_event, mask, n_members):
    """ddof=0 - the population variance. A one-character change, and a real bug."""
    areas = member_event[:, :, mask].sum(axis=2).astype(np.float64)
    truth_area = truth_event[:, mask].sum(axis=1).astype(np.float64)
    mean_area = areas.mean(axis=0)
    var = areas.var(axis=0, ddof=0)
    err = (mean_area - truth_area) ** 2
    signed = float((mean_area - truth_area).sum())
    return float(var.sum()), float(err.sum()), int(mean_area.size), signed


def test_MUTATION_dropping_the_square_root_is_caught_ONLY_by_the_range() -> None:
    """Planted in the instrument: `_ratio` returns a variance ratio, not an SD ratio."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        _mutate(mp, ratio=_no_sqrt)
        calibrated = verdict(build_scenario(1.0), 1.0)
        over = verdict(build_scenario(4.0), 4.0)
        under = verdict(build_scenario(0.25), 0.25)
    finally:
        mp.undo()

    # THE UNCOMFORTABLE HALF: the calibrated case cannot see this defect at all,
    # because 1.0 squared is 1.0. A single-regime playthrough would be GREEN here.
    assert calibrated["passed"], (
        "if this ever fails, the calibrated case has become able to see a missing "
        "square root and this comment is out of date"
    )
    # THE HALF THAT EARNS ITS KEEP: the off-one regimes catch it decisively.
    assert not over["passed"] and not under["passed"], (over, under)


def test_MUTATION_dropping_the_finite_ensemble_factor_is_caught_ONLY_exactly() -> None:
    """Planted: the ``(M+1)/M`` correction is removed. A 3% effect at M=16.

    3% is inside any tolerance sized for the sampling error of a 600-window
    estimate, so the statistical cases CANNOT see it and the exact single-window
    assertion is the only thing standing between us and a silently wrong
    calibrated reading. That is why both kinds of assertion are in this file.
    """
    from _pytest.monkeypatch import MonkeyPatch

    from wildfire_nowcast.eval import metrics as M

    real_ratio = M._ratio

    def no_factor(numerator: float, denominator: float) -> float | None:
        return real_ratio(numerator * N_MEMBERS / (N_MEMBERS + 1.0), denominator)

    mp = MonkeyPatch()
    try:
        _mutate(mp, ratio=no_factor)
        statistical = verdict(build_scenario(1.0), 1.0)
        member_areas = np.array([90, 100, 110, 140], dtype=int)
        samples, truth = _fields(member_areas, 130)
        exact = evaluate(samples, truth)["by_mask"]["domain"]["area_dispersion_ratio"]
    finally:
        mp.undo()

    assert statistical["passed"], "the statistical case is blind to a 3% factor — stated, not fixed"
    a = member_areas.astype(np.float64)
    want = float(np.sqrt((a.var(ddof=1) * 5.0 / 4.0) / (a.mean() - 130) ** 2))
    assert abs(exact - want) > 1e-6, (
        "the 1e-12 window assertion did not move when the finite-ensemble factor "
        "was removed, so it is not testing the factor and this playthrough is weaker "
        f"than it claims: exact={exact} want={want}"
    )


def test_MUTATION_a_biased_variance_estimator_is_caught_exactly() -> None:
    """Planted: ``ddof=0``. Also ~3%, also invisible statistically, also caught exactly."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        _mutate(mp, area=_biased_variance)
        member_areas = np.array([90, 100, 110, 140], dtype=int)
        samples, truth = _fields(member_areas, 130)
        exact = evaluate(samples, truth)["by_mask"]["domain"]["area_dispersion_ratio"]
    finally:
        mp.undo()

    a = member_areas.astype(np.float64)
    want = float(np.sqrt((a.var(ddof=1) * 5.0 / 4.0) / (a.mean() - 130) ** 2))
    assert abs(exact - want) > 1e-6, exact


def test_MUTATION_a_metric_that_always_returns_one_fails_this_file() -> None:
    """The bluntest instrument defect there is: a metric that always says 1.0.

    If a stubbed-out metric passed this playthrough, every green result above
    would mean nothing. This is the assertion that says the file has content.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        _mutate(mp, ratio=lambda numerator, denominator: 1.0)
        collapsed = score_scenario(build_scenario(0.0))["area_dispersion_ratio"]
        over = verdict(build_scenario(4.0), 4.0)
    finally:
        mp.undo()

    assert collapsed != 0.0, "a stub metric was still scoring a collapsed ensemble at 0"
    assert not over["passed"]


# --------------------------------------------------------------------------
# 4. MUTATION COVERAGE - the assertions above, declared to the shared harness
#    so that DELETING one of them breaks the build (A13, ADR-030).
#
# Everything in sections 1-3 is correct and none of it is changed here. What was
# missing is a guarantee that it STAYS correct: the comments say "only the RANGE
# catches a missing square root" and "only the exact case catches (M+1)/M", and
# those sentences are the most valuable thing in this file - but a sentence
# cannot fail. Removing `test_over_dispersed_...` today leaves a green suite and
# silently removes the square root from coverage.
#
# `common.playthrough` turns the same declarations into data: every defect below
# must be caught by at least one probe, the blind spot must NOT be caught, and
# the report NAMES the sole catcher of each defect. That is finding ADR-032 (4)
# ("a single-regime playthrough would have been GREEN for 2 of 4 mutations")
# expressed as a build failure instead of as a warning.
#
# SCENARIO SIZE, stated rather than tuned: the harness scores 8 worlds where a
# single test scores one, so its scenarios use 400 windows instead of 600. The
# accuracy cost is MEASURED, not assumed - over 6 seeds at M=16 the worst
# deviation from the closed form is 0.038 at c=1 (tolerance 0.10, 2.6x margin)
# and 0.089 at c=4 (relative tolerance 0.184, 2.1x margin). The full-precision
# statements remain the tests above; this section is about coverage.
# --------------------------------------------------------------------------

#: The UNMUTATED `_ratio`, captured at import so a mutation can be expressed as a
#: transformation OF the real estimator rather than as a re-implementation of it.
_REAL_RATIO = M._ratio


def _drop_finite_ensemble_factor(numerator: float, denominator: float) -> float | None:
    """`_ratio` with the (M+1)/M correction divided back out."""
    return _REAL_RATIO(numerator * N_MEMBERS / (N_MEMBERS + 1.0), denominator)


HARNESS_WINDOWS = 400
#: Measured worst |deviation| over 6 seeds at M=16, W=400: 0.038 (c=1), 0.089
#: (c=4). Both tolerances below keep >= 2x margin on that measurement.
HARNESS_TOL = 0.10


@dataclass(frozen=True)
class DispersionWorld:
    """The scenarios one observation is taken from, plus the planted transform."""

    transform: Callable[[list[dict]], list[dict]] = lambda w: w
    n_windows: int = HARNESS_WINDOWS


def _dispersion_observe(world: DispersionWorld) -> dict[str, float | None]:
    """Score the calibrated case, the over-dispersed case, and the exact window.

    Two REGIMES plus one EXACT case, because that combination is exactly what
    ADR-032 (4) showed is required: c=1 alone is blind to a missing square root,
    and any statistical case is blind to a 3% factor.
    """
    out: dict[str, float | None] = {}
    for label, c in (("calibrated", 1.0), ("over", 4.0)):
        windows = world.transform(build_scenario(c, n_windows=world.n_windows))
        out[label] = score_scenario(windows)["area_dispersion_ratio"]
        out[f"{label}_expected"] = closed_form_ratio(c, N_MEMBERS)
    member_areas = np.array([90, 100, 110, 140], dtype=int)
    samples, truth = _fields(member_areas, 130)
    out["exact"] = evaluate(samples, truth)["by_mask"]["domain"]["area_dispersion_ratio"]
    a = member_areas.astype(np.float64)
    out["exact_expected"] = float(np.sqrt((a.var(ddof=1) * 5.0 / 4.0) / (a.mean() - 130) ** 2))
    return out


def _near(obs: dict, key: str) -> bool:
    want = obs[f"{key}_expected"]
    return PT.approximately(obs[key], want, tol=max(HARNESS_TOL, HARNESS_TOL * want))


PLAYTHROUGH = PT.Playthrough(
    name="area_dispersion_ratio",
    build=DispersionWorld,
    observe=_dispersion_observe,
    probes=(
        PT.Probe(
            "calibrated_reads_one",
            lambda obs: _near(obs, "calibrated"),
            note="spread drawn from the same law as the error reads 1.0. The load-bearing case "
            "for (M+1)/M -- and, as the mutations show, blind to it at this tolerance.",
        ),
        PT.Probe(
            "over_dispersed_reads_the_closed_form",
            lambda obs: _near(obs, "over"),
            note="c=4 at M=16 reads 1.8439. THE probe that sees a missing square root, because "
            "1.0 squared is still 1.0 and the calibrated case cannot.",
        ),
        PT.Probe(
            "single_window_matches_the_definition_to_1e_12",
            lambda obs: PT.approximately(obs["exact"], obs["exact_expected"], tol=1e-12),
            note="no Monte Carlo. THE probe that sees ddof and the finite-ensemble factor, both "
            "of which move the statistical cases by ~3% and hide inside sampling tolerance.",
        ),
    ),
    defects=(
        PT.Defect(
            "silently_collapsed_members",
            PT.data_defect(lambda w: replace(w, transform=plant_silent_collapse)),
            note="an ensemble loop that reuses one draw. Invisible to every accuracy metric "
            "because the ensemble MEAN is still a legitimate forecast field.",
        ),
        PT.Defect(
            "correctly_wide_ensemble_in_the_wrong_place",
            PT.data_defect(lambda w: replace(w, transform=plant_area_bias)),
            note="spread right, position wrong. The distinction M5's P19 turned on: 'too narrow' "
            "and 'systematically displaced' must not be the same reading.",
        ),
        PT.Defect(
            "ratio_drops_the_square_root",
            PT.attribute_defect((M, "_ratio", _no_sqrt)),
            note="the criterion silently becomes a VARIANCE ratio. This is the defect that "
            "corrected our own understanding of G3's units, and the calibrated case cannot see "
            "it (ADR-032 (4)).",
        ),
        PT.Defect(
            "ratio_drops_the_finite_ensemble_factor",
            PT.attribute_defect((M, "_ratio", _drop_finite_ensemble_factor)),
            note="(M+1)/M removed: a calibrated ensemble would read M/(M+1) instead of 1. A 3% "
            "effect at M=16, inside any tolerance sized for sampling error.",
        ),
        PT.Defect(
            "biased_variance_estimator_ddof0",
            PT.attribute_defect((M, "_area_dispersion", _biased_variance)),
            note="a one-character change to the population variance. Also ~3%, also invisible "
            "statistically.",
        ),
        PT.Defect(
            "metric_stubbed_to_always_return_one",
            PT.attribute_defect((M, "_ratio", lambda numerator, denominator: 1.0)),
            note="the bluntest instrument defect there is. If a stub passed, every green result "
            "in this file would mean nothing.",
        ),
        PT.Defect(
            "half_the_members_duplicated",
            PT.data_defect(lambda w: replace(w, transform=plant_half_duplicated)),
            note="THE RECORD WAS TOO BROAD HERE AND THE HARNESS SAID SO ON ITS FIRST RUN. This "
            "was declared a blind spot outright; measured, the blindness is REGIME-SCOPED. "
            "Duplication multiplies the reading by a closed-form 0.939 at M=16 in every regime, "
            "which stays inside tolerance at c=1 (|dup-cf| 0.070-0.075 vs 0.100) and at c=0.25 "
            "(0.023-0.027 vs 0.100) but NOT at c=4 (0.239-0.248 vs 0.184), where the clean "
            "case's own sampling shortfall has already spent ~60% of the budget. The practical "
            "warning is unchanged and is now sharper: G3's bar is [0.8, 1.2], i.e. c ~ 1, which "
            "is EXACTLY the regime where this metric cannot see member duplication.",
        ),
    ),
    note="G3's dispersion half. Closed form sqrt(c(M+1)/(c+M)), written from the definition of a "
    "spread-skill ratio and never read back out of metrics.py.",
)


@pytest.fixture(scope="module")
def coverage_report(playthrough_report) -> PT.PlaythroughReport:
    """Scored ONCE PER SESSION, via the shared factory in conftest.

    The protocol costs (1 + n_defects) observations, and `test_playthrough_registry`
    wants the same report to enforce coverage repo-wide. Sharing it is the same
    saving A12 took on the null-check fixture, and for the same reason: identical
    numbers computed twice are not a second measurement.
    """
    return playthrough_report(PLAYTHROUGH)


def test_MUTATION_COVERAGE_every_planted_defect_in_this_file_is_detected(
    coverage_report: PT.PlaythroughReport,
) -> None:
    """THE enforcement (A13). Deleting a probe above now breaks the build.

    It also PRINTS the sole-catcher map, which is the ADR-032 (4) finding as data:
    a missing square root is invisible to the calibrated regime, and the two 3%
    estimator defects are invisible to EVERY statistical regime.
    """
    print(PT.format_report(coverage_report))
    coverage_report.assert_ok()
    assert coverage_report.mutation_coverage == 1.0


def test_the_sole_catchers_are_the_ones_the_record_claims(
    coverage_report: PT.PlaythroughReport,
) -> None:
    """Pins ADR-032 (4) to the actual measurement rather than to its retelling.

    If a future change makes the calibrated case able to see a missing square
    root, this fails and the record gets updated - which is the right outcome and
    the opposite of a comment quietly going stale.
    """
    caught = {o.name: set(o.caught_by) for o in coverage_report.outcomes}
    exact = "single_window_matches_the_definition_to_1e_12"
    # ADR-032 (4), stated precisely: among the STATISTICAL regimes only the
    # off-one case sees a missing square root. The exact window sees it too --
    # my first version of this assertion said "only the range" and the harness
    # corrected it, which is the correction being pinned here.
    assert caught["ratio_drops_the_square_root"] == {"over_dispersed_reads_the_closed_form", exact}
    assert "calibrated_reads_one" not in caught["ratio_drops_the_square_root"]
    # The two ~3% estimator defects are invisible to EVERY statistical regime.
    assert caught["ratio_drops_the_finite_ensemble_factor"] == {exact}
    assert caught["biased_variance_estimator_ddof0"] == {exact}
    # ...and the calibrated case, on its own, catches no instrument defect at all.
    instrument = {
        "ratio_drops_the_square_root",
        "ratio_drops_the_finite_ensemble_factor",
        "biased_variance_estimator_ddof0",
    }
    assert all("calibrated_reads_one" not in caught[name] for name in instrument)
