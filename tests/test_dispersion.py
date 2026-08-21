"""G3's GEOMETRIC dispersion bar and its FIRST-MOMENT condition (ADR-039 (4), (5)).

Every test here plants the defect the condition exists to catch and asserts it is
caught. Three of them are the specific measured cases:

* an ``adr`` that the OLD ``[0.8, 1.2]`` bar passed and the geometric bar rejects,
* a candidate whose dispersion is in-bar while its MEAN is worse than the
  ellipse's — the compensating-error pass ADR-035's identity makes possible,
* an ``adr`` of ``None`` at perfect mean calibration, which must never read as a
  pass.
"""

from __future__ import annotations

import math

import pytest

from wildfire_nowcast.common import dispersion as D

# --------------------------------------------------------------------------
# the bar's SHAPE — the defect is in the interval itself, not in any input
# --------------------------------------------------------------------------


def test_the_OLD_bar_was_asymmetric_IN_OUR_FAVOUR() -> None:
    """The motivating measurement, pinned as arithmetic rather than as prose.

    ``[0.8, 1.2]`` reaches ``|log 0.8| = 0.2231`` into under-dispersion and only
    ``|log 1.2| = 0.1823`` into over-dispersion — 22% more tolerance on the side
    every arm we have ever run fails on.
    """
    old_low, old_high = 0.8, 1.2
    assert 1.0 / old_low > old_high, "if this stops holding, the old bar was symmetric"
    under, over = abs(math.log(old_low)), abs(math.log(old_high))
    assert under > over
    assert under / over == pytest.approx(1.224, abs=0.001), (under, over)


def test_the_new_bar_is_symmetric_in_log_space() -> None:
    lo, hi = D.BAR_INTERVAL
    assert lo == pytest.approx(1.0 / D.BAR_RATIO)
    assert abs(math.log(lo)) == pytest.approx(abs(math.log(hi)))
    for ratio in (1.01, 1.1, 1.2, 1.5, 3.0, 7.0):
        assert D.dispersion_condition(ratio).outcome == D.dispersion_condition(1 / ratio).outcome
        assert D.log_distance(ratio) == pytest.approx(D.log_distance(1 / ratio))


def test_PLANTED_the_bar_got_HARDER_on_the_side_we_fail() -> None:
    """PLANTED DEFECT: an ``adr`` the old bar passed and the new one must reject.

    ``0.82`` is inside ``[0.8, 1.2]`` and outside ``[1/1.2, 1.2] = [0.8333, 1.2]``.
    If this ever passes, the bar has been loosened back to the asymmetric form and
    G3 can again be cleared from below by 4% of slack that over-dispersion never
    had. There is no matching case in the other direction, by construction: the
    upper endpoint is unchanged.
    """
    assert 0.8 <= 0.82 <= 1.2, "the premise: the OLD bar passed this"
    assert D.dispersion_condition(0.82).outcome == D.FAIL
    assert D.dispersion_condition(1.2).outcome == D.PASS, "the upper endpoint is UNCHANGED"
    assert D.dispersion_condition(1.0 / 1.2).outcome == D.PASS, "...and is now mirrored exactly"
    assert D.dispersion_condition(0.8333).outcome == D.FAIL, "just inside the reciprocal"


def test_the_measured_G3_arms_are_still_rejected() -> None:
    """The four G3 attempts, at their reported equal-block values. All fail, harder."""
    for adr in (0.2147, 0.6324, 0.5171, 0.2389, 0.5799, 0.2203):
        assert D.dispersion_condition(adr).outcome == D.FAIL, adr
    # ...and the two seeds that were "in bar" under the old form are re-checked
    # rather than assumed: 0.8669 and 0.9065 are inside BOTH bars.
    for adr in (0.8669, 0.9065):
        assert D.dispersion_condition(adr).outcome == D.PASS, adr


# --------------------------------------------------------------------------
# UNDEFINED is an outcome, never a pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize("adr", [None, 0.0, -0.5, float("nan"), float("inf"), "0.9", True])
def test_PLANTED_an_unscoreable_adr_is_UNDEFINED_and_never_a_pass(adr: object) -> None:
    """PLANTED DEFECT: the metric returns something that is not a measurement.

    ``None`` is the real case — ``area_dispersion_ratio``'s denominator is the
    model's own mean-area error, which is exactly 0 at perfect mean calibration,
    so the criterion goes undefined as the model gets the first moment RIGHT
    (sim, found by building a playthrough it could not score). The others
    are the ways a missing key, a sentinel or a stringified table cell arrive at
    the same call site indistinguishably.
    """
    result = D.dispersion_condition(adr)
    assert result.outcome == D.UNDEFINED
    assert result.passed is False


def test_PLANTED_a_condition_result_REFUSES_to_be_read_as_a_bool() -> None:
    """PLANTED DEFECT: the caller writes ``if condition:``.

    This is the whole mechanism. ``None`` is falsy and ``low <= None <= high``
    raises, so the three ways to arrive at an unmeasurable criterion behave three
    different ways at the call site — one of them silently. Raising on ``bool()``
    collapses that to a single loud failure.
    """
    undefined = D.dispersion_condition(None)
    with pytest.raises(D.UndefinedConditionError, match="refusing to coerce"):
        bool(undefined)
    with pytest.raises(D.UndefinedConditionError):
        if undefined:  # noqa: SIM103 - the point is that this line must not run
            pass
    # even a genuine PASS refuses, so the rule has no exception to learn
    with pytest.raises(D.UndefinedConditionError):
        bool(D.dispersion_condition(1.0))
    assert D.dispersion_condition(1.0).passed is True


def test_an_illegal_outcome_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="illegal outcome"):
        D.ConditionResult(name="x", outcome="probably")


# --------------------------------------------------------------------------
# the FIRST-MOMENT condition
# --------------------------------------------------------------------------


def test_PLANTED_compensating_errors_no_longer_pass_G3() -> None:
    """PLANTED DEFECT: dispersion in-bar, first moment worse than the ellipse.

    ADR-035's identity (residual 2.2e-16) makes ``growth_calibration`` the
    dominant factor in ``adr``, so a candidate can land inside a dispersion bar by
    being wrong about the mean and wrong about the spread in compensating
    directions. Before this condition existed, the arm below passed G3's only
    criterion. It must now fail the gate while its dispersion half still passes —
    and BOTH facts must be visible.
    """
    out = D.g3_conditions(adr=0.95, candidate_growth_calibration=2.66,
                          reference_growth_calibration=1.79)
    assert out["conditions"]["dispersion"]["outcome"] == D.PASS
    assert out["conditions"]["first_moment"]["outcome"] == D.FAIL
    assert out["outcome"] == D.FAIL, "BOTH must hold; one is not enough"


def test_the_currently_measured_arms_FAIL_the_first_moment_condition() -> None:
    """Stated in advance by ADR-039 (5): we over-predict 2.66-3.06x, ellipse 1.79x.

    Pinned so that the condition's known-failing state is a fact in the test suite
    and not only a sentence in an ADR. If someone later loosens the comparison,
    this goes green for the wrong reason and the loosening is visible here.
    """
    for ours in (2.66, 3.06):
        r = D.first_moment_condition(ours, 1.79)
        assert r.outcome == D.FAIL, ours
        assert r.extra["margin_log"] < 0
    # symmetric by construction: a 1/2.66 UNDER-prediction is exactly as bad
    assert D.first_moment_condition(1 / 2.66, 1.79).outcome == D.FAIL
    # ...and matching the ellipse exactly is a PASS ("no worse than")
    assert D.first_moment_condition(1.79, 1.79).outcome == D.PASS
    assert D.first_moment_condition(1 / 1.79, 1.79).outcome == D.PASS
    assert D.first_moment_condition(1.2, 1.79).outcome == D.PASS


def test_PLANTED_an_unscoreable_REFERENCE_is_not_a_free_pass() -> None:
    """PLANTED DEFECT: the ellipse could not be scored.

    "We could not score the opponent" must never read as "we beat the opponent".
    That is C6.2's VOID-not-passed rule one level up, and it is the failure shape
    that made a degenerate Brier-fitted ellipse look like a baseline.
    """
    for bad in (None, 0.0, float("nan")):
        r = D.first_moment_condition(1.0, bad)
        assert r.outcome == D.UNDEFINED, bad
        assert r.passed is False
    assert D.first_moment_condition(None, 1.79).outcome == D.UNDEFINED


def test_the_condition_contains_no_fitted_constant() -> None:
    """C-3: the bar is whatever the reference achieved, so it cannot be tuned.

    Asserted by MOVING the reference and watching the verdict follow it — a
    property of the code, not a claim in a docstring.
    """
    assert D.first_moment_condition(2.0, 1.79).outcome == D.FAIL
    assert D.first_moment_condition(2.0, 2.50).outcome == D.PASS, (
        "the same candidate, a weaker opponent: the condition is reference-based"
    )


# --------------------------------------------------------------------------
# equal-block pooling of the first moment
# --------------------------------------------------------------------------


def test_first_moment_from_blocks_pools_equal_block_and_reports_both_poolings() -> None:
    cand = {4: 2.0, 5: 4.0, 6: 3.0, 7: 2.0}
    ref = {4: 1.8, 5: 1.8, 6: 1.8, 7: 1.8}
    r = D.first_moment_condition_from_blocks(cand, ref)
    assert r.outcome == D.FAIL
    assert r.extra["n_blocks"] == 4
    assert r.value == pytest.approx(2.75), "arithmetic equal-block mean of the RATIO"
    assert r.extra["alt_pooling_log"]["would_pass"] is False, (
        "both poolings agree here; the alternative is REPORTED so the maintainer can see "
        "whether the choice is ever outcome-determinative"
    )


def test_PLANTED_a_reference_scored_on_DIFFERENT_blocks_is_UNDEFINED() -> None:
    """PLANTED DEFECT: the opponent is measured on a block we did not score.

    Creek alone was 47% of the window-pooled held-out mass (ADR-021 (4)), so a
    differing block set can move the comparison more than the models do. That is
    not a weaker comparison; it is not a comparison.
    """
    r = D.first_moment_condition_from_blocks({4: 1.0, 5: 1.0}, {4: 1.0, 5: 1.0, 6: 1.0})
    assert r.outcome == D.UNDEFINED
    assert "not scored on the same blocks" in r.detail
    assert D.first_moment_condition_from_blocks({}, {}).outcome == D.UNDEFINED


def test_PLANTED_an_undefined_block_does_not_silently_shrink_the_sample() -> None:
    """PLANTED DEFECT: one block's ratio is ``None``.

    Same principle as ``equal_block_mean``: a block that cannot be scored must not
    disappear into a mean that is then reported as complete.
    """
    r = D.first_moment_condition_from_blocks({4: 1.0, 5: None}, {4: 1.8, 5: 1.8})
    assert r.outcome == D.UNDEFINED
    assert r.extra["undefined_blocks"] == ["5"]


# --------------------------------------------------------------------------
# combination
# --------------------------------------------------------------------------


def test_UNDEFINED_dominates_and_a_gate_is_never_passed_on_one_condition() -> None:
    both_pass = D.g3_conditions(1.0, 1.0, 1.79)
    assert both_pass["outcome"] == D.PASS

    # dispersion undefined (perfect mean calibration) + first moment perfect:
    # NOT ADJUDICABLE, and specifically not a pass.
    edge = D.g3_conditions(None, 1.0, 1.79)
    assert edge["conditions"]["first_moment"]["outcome"] == D.PASS
    assert edge["outcome"] == D.UNDEFINED

    # a definite failure on one side with an unmeasurable other side is still
    # UNDEFINED: we cannot say the gate failed on evidence we do not have.
    assert D.g3_conditions(0.2, None, 1.79)["outcome"] == D.UNDEFINED
    assert D.g3_conditions(0.2, 1.0, 1.79)["outcome"] == D.FAIL


def test_both_conditions_are_always_reported_separately() -> None:
    """ "Report them separately and always together" — asserted structurally."""
    out = D.g3_conditions(0.2147, 3.06, 1.79)
    assert set(out["conditions"]) == {"dispersion", "first_moment"}
    for cond in out["conditions"].values():
        assert cond["outcome"] in (D.PASS, D.FAIL, D.UNDEFINED)
        assert cond["detail"], "a condition with no explanation is a number nobody can audit"
    assert "not_a_verdict" in out, "this module reports; the maintainer adjudicates"


# --------------------------------------------------------------------------
# growth_calibration itself
# --------------------------------------------------------------------------


def test_growth_calibration_is_undefined_rather_than_zero_when_truth_did_not_grow() -> None:
    assert D.growth_calibration(10.0, 5.0) == pytest.approx(2.0)
    assert D.growth_calibration(0.0, 5.0) == 0.0
    assert D.growth_calibration(5.0, 0.0) is None, "0/0-shaped: undefined, not 0 and not inf"
    assert D.growth_calibration(None, 5.0) is None
    assert D.growth_calibration(float("nan"), 5.0) is None
    # a 0.0 candidate is a real measurement (predicted nothing) but has no log,
    # so the CONDITION is undefined while the RATIO is 0.
    assert D.first_moment_condition(0.0, 1.79).outcome == D.UNDEFINED
