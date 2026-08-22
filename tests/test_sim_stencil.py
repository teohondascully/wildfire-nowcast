"""Unit tests for :mod:`wildfire_nowcast.sim.stencil`.

WHAT THIS FILE PROTECTS
-----------------------
``wind_independent_offset()`` is the estimator behind the published claim that
the trained kernel carries a sub-cell directional preference that does NOT track
the wind: 0.183 cells toward SSW at z = 8.1 over a bearing sweep, and 0.244 cells
toward SW at z = 6.1 in still air, against 0.003 cells at z = 0.7 for the
untrained initialisation and exactly 0.000 for both ellipses. The reason that
finding carries weight is that modelling's own internal estimate of the same
quantity, computed from weights this module never reads, agrees on the direction.
Two methods sharing no code is the whole argument, and it is an argument about a
BEARING, so an orientation error here would not weaken the finding, it would
invert it and silently destroy the agreement that justified reporting it.

At the time this file was written ``sim/stencil.py`` measured 0 percent line
coverage: 195 statements, none of them executed by any test.

THE FIXTURE
-----------
A fake predictor with a stencil that is known in closed form: a lobe of weight
0.5 on the neighbour cell the wind points at, plus a constant lobe of weight 0.1
at a fixed offset of 2 cells west and 1 cell south. Nothing about the real model
is assumed. The eight unit offsets of the sweep sum to zero exactly, so the
sweep mean centroid is analytically ``0.1 * bias / 0.6``, and the still-air
stencil is analytically the bias itself. The test therefore knows the answer
before it runs, which is what makes a disagreement a defect rather than a drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from wildfire_nowcast.sim.stencil import (
    SWEEP_BEARINGS,
    ProbeConstants,
    build_probe,
    measure_stencil,
    wind_independent_offset,
)

#: 2 cells EAST-negative (west) and 1 cell NORTH-negative (south). Deliberately
#: NOT on the diagonal: a symmetric offset would read the same under a compass
#: bearing and under a mathematical angle, so a diagonal fixture cannot tell the
#: two conventions apart and the test would pass under either.
BIAS_EAST, BIAS_NORTH = -2, -1
P_WIND = 0.5
P_BIAS = 0.1

#: 243.43 deg, i.e. west-southwest. Clockwise from north, matching the compass
#: convention the C1 wind channels use.
EXPECTED_BEARING_DEG = float((np.degrees(np.arctan2(BIAS_EAST, BIAS_NORTH)) + 360.0) % 360.0)
#: What the SAME vector would read as if the bearing were computed as a
#: mathematical angle (counter-clockwise from east) instead.
MATH_ANGLE_DEG = float((np.degrees(np.arctan2(BIAS_NORTH, BIAS_EAST)) + 360.0) % 360.0)
#: What it would read as if the centroid were taken over ROW INDEX, which grows
#: southward, instead of over northing.
ROW_INDEX_BEARING_DEG = float((np.degrees(np.arctan2(BIAS_EAST, -BIAS_NORTH)) + 360.0) % 360.0)

CONST = ProbeConstants(
    elevation=500.0,
    slope=0.0,
    aspect_sin=0.0,
    aspect_cos=1.0,
    fuel_model_id=101.0,
    canopy_cover=30.0,
    water_barrier_mask=0.0,
    recent_burn_scar=0.0,
    temp_2m=300.0,
    rh_2m=20.0,
    fuel_moisture_proxy=0.1,
    source="constants fixed in tests/test_sim_stencil.py, not read from a store",
)


class _BiasedStencil:
    """C5 ``predict()`` with a stencil that is wind lobe plus a constant lobe."""

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        size = int(x0.shape[0])
        c = size // 2
        u = float(weather[0, 0, c, c])
        v = float(weather[0, 1, c, c])
        out = np.zeros((n_members, horizon_h, size, size), dtype=np.uint8)
        out[:, :, c, c] = 1

        def light(east: int, north: int, members: int) -> None:
            out[:members, :, c - north, c + east] = 1

        speed = float(np.hypot(u, v))
        if speed > 0.0:
            light(int(round(u / speed)), int(round(v / speed)), int(round(P_WIND * n_members)))
            light(BIAS_EAST, BIAS_NORTH, int(round(P_BIAS * n_members)))
        else:
            # Still air: the constant lobe alone, plus a symmetric pair around it
            # so the centroid is unmoved and the sampling error is non-zero. A
            # zero spread would make the z statistic undefined and the test would
            # be asserting on a NaN.
            light(BIAS_EAST, BIAS_NORTH, n_members)
            light(BIAS_EAST + 1, BIAS_NORTH + 1, n_members // 2)
            light(BIAS_EAST - 1, BIAS_NORTH - 1, n_members // 2)
        return out


class _NeverIgnites:
    """A predictor that returns the initial condition and nothing else."""

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        out = np.zeros((n_members, horizon_h, *x0.shape), dtype=np.uint8)
        out[:, :, x0 > 0] = 1
        return out


@pytest.fixture(scope="module")
def offset() -> dict[str, Any]:
    return wind_independent_offset(_BiasedStencil(), CONST, n_members=40, size=25, seed=3)


def test_the_probe_is_isotropic_apart_from_the_wind() -> None:
    """The estimator's premise, asserted rather than assumed.

    Averaging the centroid over bearings only cancels the wind term if nothing
    else in the domain distinguishes one bearing from another. That is a property
    of ``build_probe``, and if it ever stops holding, every number this module
    publishes becomes a measurement of the probe.
    """
    x0, static, weather = build_probe(
        CONST, size=25, wind_speed_ms=8.0, bearing_deg=0.0, horizon_h=1
    )
    assert int(x0.sum()) == 1 and bool(x0[12, 12]), "the probe is not a single central ignition"
    for i in range(static.shape[0]):
        assert np.ptp(static[i]) == 0.0, f"static channel {i} varies across the probe domain"
    for i in (2, 3, 4):  # temp, rh, fuel moisture: everything except the two wind components
        assert np.ptp(weather[0, i]) == 0.0, f"weather channel {i} varies across the probe domain"
    assert CONST.slope == 0.0 and (CONST.aspect_sin, CONST.aspect_cos) == (0.0, 1.0)
    assert len(SWEEP_BEARINGS) >= 4 and len(set(SWEEP_BEARINGS)) == len(SWEEP_BEARINGS)
    spacing = {
        round(SWEEP_BEARINGS[i + 1] - SWEEP_BEARINGS[i], 6) for i in range(len(SWEEP_BEARINGS) - 1)
    }
    assert len(spacing) == 1, f"the bearing sweep is no longer uniform: {sorted(SWEEP_BEARINGS)}"


def test_the_wind_lobe_cancels_and_the_sweep_recovers_the_constant_offset(
    offset: dict[str, Any],
) -> None:
    """The estimator must return the constant term and only the constant term.

    Failure condition, in one sentence: a stencil whose wind response is five
    times the constant offset, which is what this fixture is, would report the
    wind direction of whichever bearing survived the averaging instead of the
    0.333 west, 0.167 south residual that is analytically there.
    """
    expected = (
        P_BIAS * BIAS_EAST / (P_WIND + P_BIAS),
        P_BIAS * BIAS_NORTH / (P_WIND + P_BIAS),
    )
    assert offset["n_bearings_with_ignition"] == len(SWEEP_BEARINGS)
    assert offset["sweep_mean_centroid_cells"] == pytest.approx(list(expected), abs=1e-12)
    assert offset["sweep_mean_magnitude_cells"] == pytest.approx(float(np.hypot(*expected)))
    # The individual bearings must NOT already look like the answer, or the
    # cancellation this function exists to perform is doing no work.
    per_bearing = [s["centroid_cells"] for s in offset["sweep"]]
    assert max(abs(cx - expected[0]) for cx, _ in per_bearing) > 0.5, (
        "every bearing already reports the constant offset, so the sweep is not "
        "averaging away a wind response and the fixture has stopped testing it"
    )


def test_the_bearing_is_a_COMPASS_bearing_measured_from_a_NORTH_UP_centroid(
    offset: dict[str, Any],
) -> None:
    """Two orientation conventions, both plausible, both wrong here.

    Failure condition, in one sentence: an offset that is west and south, which
    under a row-index centroid reads as west and NORTH, and under a mathematical
    angle reads 63 deg away from the compass answer. The published claim is a
    direction, so either error inverts it rather than blurring it.
    """
    assert offset["sweep_mean_bearing_deg"] == pytest.approx(EXPECTED_BEARING_DEG, abs=1e-6)
    assert offset["sweep_mean_bearing_deg"] != pytest.approx(ROW_INDEX_BEARING_DEG, abs=1.0)
    assert offset["sweep_mean_bearing_deg"] != pytest.approx(MATH_ANGLE_DEG, abs=1.0)
    # The residual points west and south, which is what a bearing in the third
    # quadrant means. Asserted on the components too, so a bearing helper that
    # happens to be self-consistently wrong cannot satisfy this.
    east, north = offset["sweep_mean_centroid_cells"]
    assert east < 0.0 and north < 0.0
    assert 180.0 < offset["sweep_mean_bearing_deg"] < 270.0


def test_the_still_air_estimator_agrees_with_the_sweep_on_DIRECTION(
    offset: dict[str, Any],
) -> None:
    """The second, assumption-free estimate of the same quantity.

    Still air removes the wind term by construction rather than by averaging, so
    the two estimators share no cancellation argument. Agreement on the bearing
    is the check that the sweep is not reporting an artefact of the averaging.
    """
    assert offset["still_centroid_cells"] == pytest.approx([BIAS_EAST, BIAS_NORTH], abs=1e-12)
    assert offset["still_bearing_deg"] == pytest.approx(EXPECTED_BEARING_DEG, abs=1e-6)
    assert offset["still_bearing_deg"] == pytest.approx(offset["sweep_mean_bearing_deg"], abs=1e-6)
    assert offset["sweep_mean_z"] > 3.0 and offset["still_z"] > 3.0
    assert np.isfinite(offset["still_n_ignition_events"]) and offset["still_n_ignition_events"] > 0


def test_the_standard_error_is_scaled_by_IGNITION_EVENTS_not_by_member_count() -> None:
    """A still-air stencil that lights 0.02 cells per member is not a 512-sample estimate.

    Failure condition, in one sentence: any stencil whose expected new cells per
    member is not 1, because dividing the centroid variance by the MEMBER count
    instead of by ``members * expected new cells`` then understates the standard
    error by the square root of that factor, and the z it feeds is the whole
    reason the published 0.183 cell offset was reported as a bias rather than as
    Monte Carlo noise pointing somewhere.

    A ratio between two member counts cannot see that error, because both
    denominators are wrong by the same factor and the ratio is 2.0 either way.
    The assertion is therefore on the ABSOLUTE standard error against the closed
    form for this two-cell stencil.
    """
    at_45 = 1  # the wind lobe sits on the north-east neighbour, i.e. one cell east
    total = P_WIND + P_BIAS
    cx = (P_WIND * at_45 + P_BIAS * BIAS_EAST) / total
    var_x = (P_WIND * (at_45 - cx) ** 2 + P_BIAS * (BIAS_EAST - cx) ** 2) / total

    seen = []
    for members in (40, 160):
        s = measure_stencil(_BiasedStencil(), CONST, size=25, n_members=members, bearing_deg=45.0)
        assert s["expected_new_cells"] == pytest.approx(total)
        assert s["n_ignition_events"] == pytest.approx(members * total)
        assert s["centroid_cells"][0] == pytest.approx(cx)
        expected_se_x = float(np.sqrt(var_x / (members * total)))
        assert s["centroid_se_cells"][0] == pytest.approx(expected_se_x, rel=1e-12), (
            f"at {members} members the centroid standard error is "
            f"{s['centroid_se_cells'][0]}, not the {expected_se_x} that "
            f"{members} x {total} ignition events implies"
        )
        seen.append(float(np.hypot(*s["centroid_se_cells"])))
    assert seen[0] / seen[1] == pytest.approx(2.0, rel=1e-9)


def test_a_stencil_that_ignites_NOTHING_reports_nan_and_not_due_north() -> None:
    """The degenerate path, which this repository has now shipped four times.

    A model that never puts a cell on the ground has no centroid. Returning
    ``(0, 0)`` would give bearing 0.0, which reads as a confident measurement of
    due north, and would be pooled into a sweep mean as if it were data.
    """
    dead = wind_independent_offset(_NeverIgnites(), CONST, n_members=8, size=15, seed=1)
    assert dead["n_bearings_with_ignition"] == 0
    assert np.isnan(dead["sweep_mean_bearing_deg"])
    assert np.isnan(dead["sweep_mean_magnitude_cells"])
    assert np.isnan(dead["still_bearing_deg"])
    one = measure_stencil(_NeverIgnites(), CONST, size=15, n_members=8, bearing_deg=90.0)
    assert one["expected_new_cells"] == 0.0
    assert all(np.isnan(v) for v in one["centroid_cells"])
