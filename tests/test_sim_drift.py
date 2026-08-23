"""Unit tests for :mod:`wildfire_nowcast.sim.drift`.

WHAT THIS FILE PROTECTS
-----------------------
``drift.py`` is the outside-the-model check on two structural-defect claims that
modelling made from its own gradients. Its artifact is
``reports/figures/drift_and_barrier.{png,json}`` and its published table is the
earth-frame / wind-frame resultant pair (truth 0.143 / 0.126, kernel 0.232 /
0.120, calibrated ellipse 0.288 / 0.170, untrained init 0.254 / 0.191). The
module measured **0 percent coverage** (250 statements, 62 branches) when this
file was written.

The reading published from that table is deliberately weak -- the stencil probe
supersedes it -- but two of the estimators underneath it are not weak, and both
are the kind that fail silently:

* **The barrier ratio is DIRECT-STANDARDISED by ring.** The module docstring
  gives the reason: barrier cells are not randomly placed relative to a fire
  front, water sits in valleys and roads on ridges, so a raw on-class /
  off-class rate ratio conflates suppression with geometry. A test that only
  checked the ratio on a geometrically balanced fixture would pass under either
  estimator. The fixture here is built so the crude ratio and the standardised
  ratio point in OPPOSITE directions.
* **The drift discriminator is a frame comparison.** An earth-fixed preference
  survives earth-frame averaging and cancels in the wind frame; a wind-tracking
  one does the reverse. ``R_earth >> R_wind`` is the entire signature, so the
  two frames must not be able to agree by construction.

WHAT IS NOT TESTED HERE, AND WHY
--------------------------------
``drift_statistics``, ``render_drift`` and ``main`` are not covered. They call
C5 ``predict()`` over a real C1 tensor store for four fires and a trained
checkpoint; a test of them would be an integration run, not a unit test, and
stubbing the predictor down to nothing would leave an assertion about a mock.
The window loop's own contract -- that it hands ``displacement`` NORTHINGS and
not row indices -- is checked here at the function that consumes them.

A DEFECT FOUND WHILE WRITING THIS FILE, AND HOW IT WAS CLOSED
-------------------------------------------------------------
An earlier ``drift`` carried a dead bearing-to-octant helper and its lookup
table: nothing in ``src/``, ``tests/``, ``tools/`` or ``runs/`` referenced
either. It also carried a units sniffer, ``np.degrees(x) if x <= 2 * np.pi
else x``, which silently treats any bearing below 6.28 DEGREES as radians --
i.e. exactly the bearings closest to due north on the compass convention this
package uses everywhere else. No test was written for it, because a test would
have pinned the behaviour of code that stood behind no number. It was DELETED
instead, after the zero-reference claim was re-checked three ways (AST loads
over every tracked file, raw text over every tracked file of any type, and
every dynamic ``getattr`` in the tree).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wildfire_nowcast.sim.drift import (
    DriftAccumulator,
    RingCounts,
    barrier_response,
    displacement,
    ring_index,
    suppression,
)

# --------------------------------------------------------------------------
# ring_index
# --------------------------------------------------------------------------


def test_ring_index_is_CHEBYSHEV_distance_and_the_burned_region_is_ring_zero() -> None:
    """Rings are square shells, because the kernel's neighbourhood is 8-connected.

    A Euclidean ring would put the diagonal neighbour in ring 2 while the kernel
    reaches it in one step, so the distance control would not be matched to the
    process it controls for.

    WHAT WOULD MAKE THIS FAIL: a 4-connected dilation, which leaves the four
    diagonal cells of ring 1 at 0 and pushes them into ring 2.
    """
    burned = np.zeros((7, 7), dtype=bool)
    burned[3, 3] = True
    rings = ring_index(burned, 2)

    assert rings[3, 3] == 0
    assert rings[2, 2] == 1 and rings[2, 4] == 1 and rings[4, 2] == 1  # diagonals are ring 1
    assert rings[3, 1] == 2 and rings[1, 1] == 2
    assert rings[0, 0] == 0, "beyond max_ring stays 0 and is excluded by the band mask"
    assert set(np.unique(rings)) == {0, 1, 2}


# --------------------------------------------------------------------------
# RingCounts pooling
# --------------------------------------------------------------------------


def test_RingCounts_pools_SUMS_so_a_three_cell_window_cannot_outvote_a_300_cell_one() -> None:
    """The weighting error the dataclass docstring names.

    Two windows are accumulated: one with 1 cell at probability 1.0, one with 99
    cells at probability 0.0. The cell-weighted rate is 0.01. A mean of the two
    windows' rates would be 0.5 -- fifty times larger, and driven entirely by the
    window with one cell in it.

    WHAT WOULD MAKE THIS FAIL: storing per-window rates and averaging them,
    which is the same weighting error that makes a pooled Brier read as skill
    when most windows are motionless.
    """
    acc = RingCounts()
    small = np.zeros((10, 10), dtype=bool)
    small[0, 0] = True
    acc.add(small, np.ones((10, 10)))

    big = np.zeros((10, 10), dtype=bool)
    big[1:, :] = True
    big[0, 0] = False
    assert int(big.sum()) == 90
    acc.add(big, np.zeros((10, 10)))

    assert acc.n_cells == 91.0
    assert acc.prob_sum == 1.0
    assert acc.rate == pytest.approx(1.0 / 91.0)
    assert acc.rate != pytest.approx(0.5, rel=1e-3)


def test_an_empty_RingCounts_reports_NaN_and_not_a_zero_rate() -> None:
    """No eligible cell is not the same measurement as a zero ignition rate.

    Zero would enter the ring-standardised ratio as evidence of total
    suppression; NaN is excluded by ``suppression``'s finiteness guard.

    WHAT WOULD MAKE THIS FAIL: returning 0.0 for an empty population, which
    turns "we could not look here" into "nothing burns here".
    """
    assert math.isnan(RingCounts().rate)
    acc = RingCounts()
    acc.add(np.zeros((4, 4), dtype=bool), np.ones((4, 4)))
    assert math.isnan(acc.rate)


# --------------------------------------------------------------------------
# suppression: the ring-standardised ratio
# --------------------------------------------------------------------------


def _acc_from(
    rows: list[tuple[int, bool, float, float]],
) -> dict[tuple[str, int, bool], RingCounts]:
    """Build the accumulator directly from (ring, on_class, n_cells, rate)."""
    acc: dict[tuple[str, int, bool], RingCounts] = {}
    for ring, on, n_cells, rate in rows:
        rc = RingCounts()
        rc.n_cells = float(n_cells)
        rc.prob_sum = float(n_cells) * float(rate)
        acc[("barrier", ring, on)] = rc
    return acc


def test_the_barrier_ratio_is_RING_STANDARDISED_and_reverses_the_crude_pooled_answer() -> None:
    """The confound the whole measurement exists to remove, made decisive.

    In BOTH rings a barrier cell ignites at HALF the rate of a non-barrier cell
    at the same distance, so the honest answer is 0.5 and barriers suppress. But
    the barrier cells are concentrated in ring 1, close to the fire, where every
    cell ignites ten times more often -- the geometric arrangement the module
    docstring describes. Pooling the raw sums instead gives
    (1000*0.10 + 100*0.01) / (100*0.20 + 1000*0.02) = 101/40 = 2.525, i.e.
    barriers appear to ACCELERATE the fire by 2.5x.

    Standardising within rings and weighting by the ON-class count returns 0.5
    exactly. A crude estimator and a standardised one therefore disagree in
    DIRECTION on this fixture, which is what makes the assertion a test of the
    method rather than of the arithmetic.

    WHAT WOULD MAKE THIS FAIL: summing probabilities across rings before taking
    the ratio, or weighting the per-ring ratios by the OFF-class count, which on
    this fixture reads 0.5 only because the two happen to coincide when the
    per-ring ratio is constant -- so the fixture also varies the ratio in the
    second assertion below.
    """
    acc = _acc_from(
        [
            (1, True, 1000, 0.10),
            (1, False, 100, 0.20),
            (2, True, 100, 0.01),
            (2, False, 1000, 0.02),
        ]
    )
    out = suppression(acc, "barrier", max_ring=2)
    assert out["ratio"] == pytest.approx(0.5, rel=1e-12)
    assert out["n_on_cells"] == 1100.0
    assert sorted(out["per_ring"]) == [1, 2]
    assert out["per_ring"][1]["on_rate"] == pytest.approx(0.10)
    assert out["per_ring"][1]["n_off"] == 100.0

    crude = (1000 * 0.10 + 100 * 0.01) / (100 * 0.20 + 1000 * 0.02)
    assert crude == pytest.approx(2.525, rel=1e-9)
    assert out["ratio"] < 1.0 < crude


def test_the_per_ring_ratios_are_weighted_by_the_ON_class_count_not_the_OFF_class_count() -> None:
    """Direct standardisation to the ON-class population, stated as a number.

    Ring 1 holds 900 barrier cells with a ratio of 1/2; ring 2 holds 100 with a
    ratio of 2/1. Weighting by the ON count gives
    (900*0.1 + 100*0.4) / (900*0.2 + 100*0.2) = 130/200 = 0.65. Weighting by the
    OFF count instead gives (100*0.1 + 900*0.4)/(100*0.2 + 900*0.2) = 370/200 =
    1.85, which is on the other side of 1.0 and would report barriers as
    accelerating.

    WHAT WOULD MAKE THIS FAIL: substituting ``off.n_cells`` for ``on.n_cells`` in
    either the numerator or the denominator weight.
    """
    acc = _acc_from(
        [
            (1, True, 900, 0.10),
            (1, False, 100, 0.20),
            (2, True, 100, 0.40),
            (2, False, 900, 0.20),
        ]
    )
    out = suppression(acc, "barrier", max_ring=2)
    assert out["ratio"] == pytest.approx(0.65, rel=1e-12)
    assert out["ratio"] < 1.0


def test_a_ring_with_no_OFF_class_cell_is_DROPPED_rather_than_scored() -> None:
    """A ring with nothing to compare against contributes no evidence.

    An unmatched ring has no controlled comparison in it, so including it would
    smuggle an uncontrolled rate into a ratio whose whole claim is that it is
    controlled for distance.

    Three ways a ring can be unusable are present: ring 1 has no OFF entry at
    all, ring 3 has an OFF entry with zero cells (so its rate is NaN), and ring 4
    has an ON entry with zero cells. Only ring 2 is scorable. The third case is
    the one that matters for the arithmetic: an ON population of zero cells has a
    NaN rate and no finiteness guard stands in front of the numerator, so
    admitting it turns the whole ratio into NaN rather than biasing it.

    WHAT WOULD MAKE THIS FAIL: treating a missing OFF population as a zero rate,
    which sends the denominator toward zero and the ratio toward infinity; or
    dropping the ``not on.n_cells`` clause, which admits ring 4 and returns NaN
    for a measurement that is perfectly well defined on ring 2.
    """
    acc = _acc_from(
        [
            (1, True, 10, 0.5),
            (2, True, 10, 0.5),
            (2, False, 10, 0.25),
            (3, True, 10, 0.5),
            (3, False, 0, 0.0),
            (4, True, 0, 0.0),
            (4, False, 10, 0.25),
        ]
    )
    out = suppression(acc, "barrier", max_ring=4)
    assert sorted(out["per_ring"]) == [2]
    assert out["ratio"] == pytest.approx(2.0)
    assert out["n_on_cells"] == 10.0

    empty = suppression({}, "barrier", max_ring=3)
    assert math.isnan(empty["ratio"])
    assert empty["per_ring"] == {}


def test_barrier_response_partitions_each_ring_into_on_class_and_off_class_exactly() -> None:
    """Every banded cell in a ring lands in exactly one of the two populations.

    The two flags are accumulated independently, and the ON and OFF counts of one
    class must sum to the ring's banded cell count -- otherwise a cell is either
    double counted or silently dropped from the denominator.

    WHAT WOULD MAKE THIS FAIL: using the barrier mask in place of its complement
    for the OFF population, which double counts every barrier cell and drops
    every non-barrier one.
    """
    shape = (9, 9)
    burned = np.zeros(shape, dtype=bool)
    burned[4, 4] = True
    rings = ring_index(burned, 2)
    band = rings > 0
    barrier = np.zeros(shape, dtype=bool)
    barrier[:, 4] = True  # a vertical road through the middle
    nonburnable = np.zeros(shape, dtype=bool)
    prob = np.full(shape, 0.25)

    acc: dict[tuple[str, int, bool], RingCounts] = {}
    barrier_response(
        prob,
        band=band,
        rings=rings,
        barrier=barrier,
        nonburnable=nonburnable,
        acc=acc,
        max_ring=2,
    )
    for r in (1, 2):
        n_ring = int((band & (rings == r)).sum())
        on = acc[("barrier", r, True)].n_cells
        off = acc[("barrier", r, False)].n_cells
        assert on + off == n_ring
        assert on > 0 and off > 0
    # 8 cells in ring 1, of which 2 sit on the road column.
    assert acc[("barrier", 1, True)].n_cells == 2.0
    assert acc[("barrier", 1, False)].n_cells == 6.0


# --------------------------------------------------------------------------
# The frame discriminator
# --------------------------------------------------------------------------


def test_the_wind_frame_along_wind_component_is_identically_zero_by_construction() -> None:
    """The method caveat this lead published beside the artifact, asserted.

    The residual is the component of the unit displacement PERPENDICULAR to the
    wind, so its projection back onto the wind axis is exactly zero for every
    window. That makes ``wind_frame.bearing_deg`` meaningless -- it is always
    0.0 -- and ``wind_frame.r`` the only informative wind-frame number. Anyone
    reading a wind-frame bearing off the JSON is reading an artifact of the
    definition.

    WHAT WOULD MAKE THIS FAIL: accumulating the raw displacement instead of its
    perpendicular residual, after which the along-wind term is non-zero and the
    two frames stop being a discriminator at all.
    """
    rng = np.random.default_rng(0)
    acc = DriftAccumulator.empty()
    for _ in range(50):
        d = (float(rng.normal()), float(rng.normal()))
        w = (float(rng.normal()), float(rng.normal()))
        acc.add(d, w)
    assert len(acc.earth) == 50
    for x, _y in acc.wind:
        assert abs(x) < 1e-12
    assert abs(acc.summary()["wind_frame"]["x"]) < 1e-12
    assert acc.summary()["wind_frame"]["bearing_deg"] == pytest.approx(0.0)


def test_an_EARTH_FIXED_bias_survives_the_earth_frame_and_cancels_in_the_wind_frame() -> None:
    """``R_earth >> R_wind`` is the published signature, and this is what makes it one.

    Four windows are given the same fixed north-east displacement under four
    winds pointing N, E, S and W. In the earth frame the perpendicular residuals
    accumulate toward one compass direction; in the wind frame the SAME residuals
    are rotated by the wind each time and cancel. The reverse fixture -- a
    displacement that always leans the same way RELATIVE to the wind -- inverts
    the comparison, so a single test cannot pass by having both numbers come out
    the same.

    WHAT WOULD MAKE THIS FAIL: omitting the rotation into the wind frame, which
    makes ``R_wind`` equal ``R_earth`` and destroys the discriminator while
    leaving both numbers looking plausible.
    """
    earth_fixed = DriftAccumulator.empty()
    d = (1.0, 1.0)
    for w in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)):
        earth_fixed.add(d, w)
    s = earth_fixed.summary()
    assert s["n_windows"] == 4
    assert s["earth_frame"]["r"] > 10.0 * max(s["wind_frame"]["r"], 1e-12)

    wind_tracking = DriftAccumulator.empty()
    for wx, wy in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)):
        # always 45 degrees to the LEFT of the wind, whatever the wind does
        left = (-wy, wx)
        disp = ((wx + left[0]) / math.sqrt(2.0), (wy + left[1]) / math.sqrt(2.0))
        wind_tracking.add(disp, (wx, wy))
    t = wind_tracking.summary()
    assert t["wind_frame"]["r"] > 10.0 * max(t["earth_frame"]["r"], 1e-12)


def test_the_resultant_bearing_is_a_COMPASS_bearing_and_not_a_mathematical_angle() -> None:
    """0 degrees is toward north and 90 is toward east, as everywhere else in C1.

    The same convention error that ``sim/stencil.py``'s tests exist to catch:
    ``arctan2(x, y)`` is the compass form and ``arctan2(y, x)`` is the maths
    form, and on a due-east resultant the two read 90 and 0. A drift statistic
    reported under the wrong one does not weaken the finding, it rotates it.

    WHAT WOULD MAKE THIS FAIL: swapping the argument order in ``resultant``.
    """
    east = DriftAccumulator.empty()
    east.earth.append((1.0, 0.0))
    assert east.summary()["earth_frame"]["bearing_deg"] == pytest.approx(90.0)

    north = DriftAccumulator.empty()
    north.earth.append((0.0, 1.0))
    assert north.summary()["earth_frame"]["bearing_deg"] == pytest.approx(0.0)

    south_west = DriftAccumulator.empty()
    south_west.earth.append((-1.0, -1.0))
    assert south_west.summary()["earth_frame"]["bearing_deg"] == pytest.approx(225.0)


def test_a_degenerate_window_is_SKIPPED_and_an_empty_accumulator_summarises_to_NaN() -> None:
    """A zero-length displacement or a still wind has no direction to contribute.

    Normalising either would divide by zero. The window is dropped, so
    ``n_windows`` reports how many windows actually carried a direction rather
    than how many were offered.

    WHAT WOULD MAKE THIS FAIL: admitting the degenerate window with a zero unit
    vector, which pulls every resultant toward the origin and reads as an absence
    of drift.
    """
    acc = DriftAccumulator.empty()
    acc.add((0.0, 0.0), (1.0, 0.0))
    acc.add((1.0, 0.0), (0.0, 0.0))
    assert acc.summary()["n_windows"] == 0
    s = acc.summary()
    assert math.isnan(s["earth_frame"]["r"])
    assert math.isnan(s["mean_cos_wind"])
    assert math.isnan(s["mean_displacement_m"])


# --------------------------------------------------------------------------
# displacement
# --------------------------------------------------------------------------


def test_displacement_follows_the_NORTHINGS_it_is_given_and_not_the_row_index() -> None:
    """The sign convention that keeps every drift bearing from pointing due south.

    ``ys`` are northings from the C1 coordinate, which DECREASE as the row index
    increases on a north-up raster. New burn on a lower row index is therefore
    displacement to the NORTH and must come out positive. The function's own
    docstring says using row indices here would point every drift statistic due
    south; this fixture is the demonstration.

    WHAT WOULD MAKE THIS FAIL: computing the centroid over ``np.arange(nrows)``
    instead of over the supplied northings, which returns -1000 where this
    asserts +1000.
    """
    shape = (5, 5)
    burned = np.zeros(shape, dtype=bool)
    burned[2, 2] = True
    prob = np.zeros(shape)
    prob[1, 2] = 1.0  # one row NORTH of the burned cell

    xs = np.array([-2000.0, -1000.0, 0.0, 1000.0, 2000.0])
    ys = np.array([2000.0, 1000.0, 0.0, -1000.0, -2000.0])  # northings, descending

    dx, dy = displacement(prob, burned, xs, ys)  # type: ignore[misc]
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(1000.0)

    # No new burn at all: undefined, reported as None rather than as (0, 0).
    assert displacement(np.zeros(shape), burned, xs, ys) is None
