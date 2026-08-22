"""``common/derive.py`` - the derived C1 channels, checked against their sources.

These formulas are the single definition both the synthetic generator and the
real ingestion path call (C0), so a defect here is a defect in every tensor at
once and no per-fire QA can see it. Two of the three tests below exist because a
mutation sweep found the site unguarded: the relative-humidity clip and the sign
that turns a row gradient into a northward one.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.derive import (
    aspect_to_sin_cos,
    dead_fuel_moisture_simard,
    slope_aspect_from_elevation,
)

#: Simard (1968) equilibrium moisture content, evaluated by hand at 300 K
#: (80.33 deg F) on each of the three branches of the piecewise fit, before the
#: output clip. Literals rather than a re-implementation of the formula: a test
#: that recomputes the expression it is checking agrees with any coefficient the
#: source happens to hold.
_SIMARD_AT_300K = {
    5.0: 1.2055013,  # H < 10, the "low" branch
    30.0: 5.8431015,  # 10 <= H <= 50, the "mid" branch
    80.0: 15.7714396,  # H > 50, the "high" branch
    100.0: 25.5791492,  # saturation, the largest humidity that exists
}


@pytest.mark.parametrize(("rh_pct", "expected"), sorted(_SIMARD_AT_300K.items()))
def test_the_simard_fit_reproduces_its_published_value_on_every_branch(
    rh_pct: float, expected: float
) -> None:
    """One point per branch, so a coefficient cannot move without this failing.

    ``fuel_moisture_proxy`` is C1 channel 11 and feeds the kernel directly. A
    wrong constant in one branch is invisible to every structural contract check
    (the value stays finite, in range and static), which is exactly the class R11
    named: the contract checks structure, not physical plausibility.
    """
    got = float(dead_fuel_moisture_simard(300.0, rh_pct))
    assert got == pytest.approx(expected, abs=1e-5), (
        f"EMC at 300 K, RH {rh_pct}% is {got}, not the Simard value {expected}"
    )


def test_relative_humidity_above_saturation_is_clipped_and_does_not_reach_the_fit() -> None:
    """RH is a percentage, so 120% must score as 100%, not as 120%.

    RTMA can report slightly over-saturated cells, and the high branch is
    quadratic in ``H``: at 300 K an unclipped 120% gives 39.8% EMC against 25.6%
    at saturation, a 1.6x error on a channel the model reads every hour. The
    lower end is the same argument.
    """
    saturated = float(dead_fuel_moisture_simard(300.0, 100.0))
    assert float(dead_fuel_moisture_simard(300.0, 120.0)) == pytest.approx(saturated, abs=1e-5)
    assert float(dead_fuel_moisture_simard(300.0, 400.0)) == pytest.approx(saturated, abs=1e-5)

    dry = float(dead_fuel_moisture_simard(300.0, 0.0))
    assert float(dead_fuel_moisture_simard(300.0, -25.0)) == pytest.approx(dry, abs=1e-5)

    # The control: inside the range the function is NOT flat, so the two
    # assertions above are reporting a clip rather than a constant.
    assert float(dead_fuel_moisture_simard(300.0, 60.0)) != pytest.approx(saturated, abs=1e-3)


def test_aspect_faces_downhill_in_all_four_cardinal_directions() -> None:
    """Axis 0 runs north to south, so a row gradient is a SOUTHWARD gradient.

    The sign that converts one into the other is a single character, it has no
    effect on slope (which takes a hypotenuse), and it silently mirrors every
    aspect channel north for south. A wind-slope interaction fitted on mirrored
    terrain is wrong in a way no range check can see.
    """
    n, cell = 9, 1000.0
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float64)

    # Elevation rising with the ROW index rises toward the SOUTH, so downhill,
    # which is what aspect reports, points NORTH: bearing 0.
    _, aspect = slope_aspect_from_elevation(rows * 10.0, cell)
    assert np.allclose(aspect, 0.0), f"terrain rising southward must face north, got {aspect[4, 4]}"

    _, aspect = slope_aspect_from_elevation(-rows * 10.0, cell)
    assert np.allclose(aspect, 180.0), "terrain rising northward must face south"

    # And the east-west pair, which shares no sign with the one above: if both
    # axes were flipped the two assertions above would still pass together.
    _, aspect = slope_aspect_from_elevation(cols * 10.0, cell)
    assert np.allclose(aspect, 270.0), "terrain rising eastward must face west"

    _, aspect = slope_aspect_from_elevation(-cols * 10.0, cell)
    assert np.allclose(aspect, 90.0), "terrain rising westward must face east"


def test_slope_is_the_gradient_magnitude_and_a_flat_cell_has_no_aspect() -> None:
    """The `(0, 0)` sine/cosine code is the only point no bearing maps to."""
    n, cell = 5, 1000.0
    rows = np.mgrid[0:n, 0:n][0].astype(np.float64)

    slope, _ = slope_aspect_from_elevation(rows * cell, cell)
    assert np.allclose(slope, 45.0), "a one-cell rise per one-cell run is 45 degrees"

    slope, aspect = slope_aspect_from_elevation(np.zeros((n, n)), cell)
    assert np.allclose(slope, 0.0) and np.allclose(aspect, 0.0)
    sin_a, cos_a = aspect_to_sin_cos(aspect, slope)
    assert np.allclose(sin_a, 0.0) and np.allclose(cos_a, 0.0)
