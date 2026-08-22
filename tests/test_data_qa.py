"""Per-fire QA rules: monotone burned area, teleporting, and the km-scale label
noise measurement.

WHAT STANDS BEHIND THIS. ``data/qa.py`` produces the dict that goes verbatim
into every C2 manifest under ``provenance.qa``. It is the only place the corpus
states, per fire, whether burned area is monotone, how many cell-steps made an
illegal state transition, how far the labels teleported, and how far apart the
two GOFER viewing geometries put the same fire. It had **zero** line coverage.

The module's own design rule is that QA MEASURES and does not fix. That makes a
silent QA defect worse than a loud pipeline defect: the pipeline repairs the
monotonicity with a cumulative OR, and the QA numbers are the only record of how
much repair was needed. A QA function that quietly reports zero turns a repaired
corpus into a clean-looking one.

THE THREE THINGS TESTED, and why each is the sharp version rather than the easy
one:

* ``raster_qa`` is checked on a state field that VIOLATES each rule, not on a
  clean one. A checker that returns "fine" on everything passes a clean-input
  test perfectly.
* ``_max_gap_km`` is checked against a distance computed by hand, because a
  teleport distance that is merely non-zero cannot distinguish a one-cell
  rasterisation wobble from a 47 km filing artifact, and those two get opposite
  treatment downstream.
* ``east_west_noise`` is checked on the case where the mean OFFSET and the mean
  DISPLACEMENT VECTOR disagree. Label noise here is documented as systematic
  (one viewing geometry runs consistently larger), and the noise model built on
  top of this has to be biased rather than a symmetric dilate and erode. A
  measurement that only reports magnitude cannot tell those apart, and the
  vector mean is the only field that can.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from wildfire_nowcast.common.contract import CRS_STRING
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.qa import (
    _longest_run,
    _max_gap_km,
    _verdict,
    east_west_noise,
    raster_qa,
)

CELL_M = 1000.0


def _grid(n: int = 12) -> Grid:
    return Grid(x_min=0.0, y_max=float(n) * CELL_M, nx=n, ny=n, cell_size_m=CELL_M)


def _clean_report(raster: dict) -> dict:
    """A report whose vector half is unremarkable, so a verdict is about the raster."""
    return {
        "vector": {
            "hours_missing": 0,
            "invalid_geometries": 0,
            "unburning_area_total_km2": 0.0,
            "zero_growth_hour_fraction": 0.0,
            "longest_zero_growth_run_h": 0,
        },
        "raster": raster,
    }


# --------------------------------------------------------------------------
# monotone burned area and illegal transitions
# --------------------------------------------------------------------------


def test_a_cell_that_leaves_the_burned_set_breaks_monotonicity_and_is_counted() -> None:
    """Fire is absorbing. A cell that un-burns is a label defect, never physics.

    FAILS WHEN: ``burned_area_monotone`` is computed with ``>= -1`` slack, or the
    revert count is taken over ``state == BURNING`` instead of ``state != 0``,
    either of which lets a shrinking perimeter into the corpus reported as clean.
    """
    state = np.zeros((2, 12, 12), dtype=np.uint8)
    state[0, 3, 3] = 1  # burns, then vanishes

    report = raster_qa(state, _grid())

    assert report["burned_area_monotone"] is False
    assert report["n_burned_cell_decreases"] == 1
    assert report["cells_reverting_from_burned"] == 1
    assert _verdict(_clean_report(report))["pass"] is False
    assert any("decreases" in f for f in _verdict(_clean_report(report))["failures"])


def test_a_cell_that_jumps_unburned_to_burned_out_is_counted_as_a_skip() -> None:
    """0 -> 2 without passing through 1 is legal-looking to any per-frame check.

    Every frame holds only legal state values and the burned area only grows, so
    a value-domain check and a monotonicity check both pass. Only the pairwise
    transition catches it, and it matters because the contagion kernel seeds on
    state 1: a cell that never occupies it is a cell the kernel never spread
    from.

    FAILS WHEN: the skip count is computed within a frame rather than across a
    pair of frames, in which case it reads 0 on this input.
    """
    state = np.zeros((2, 12, 12), dtype=np.uint8)
    state[1, 4, 4] = 2

    report = raster_qa(state, _grid())

    assert report["burned_area_monotone"] is True, "the premise: area only grows here"
    assert report["state_values_present"] == [0, 2]
    assert report["cells_skipping_burning_state"] == 1
    assert _verdict(_clean_report(report))["pass"] is False


def test_a_frame_with_an_active_fire_and_no_burning_cell_is_a_warning_not_a_failure() -> None:
    """GOFER genuinely does this, so it warns; but it must be visible, because a
    step with no state-1 cell gives the kernel nothing to spread from.

    FAILS WHEN: the counter drops the ``& active`` term, at which point every
    fire warns for its pre-ignition frames and the signal is buried, or drops
    the frame count entirely and a dormant stretch is reported as normal.
    """
    state = np.zeros((3, 12, 12), dtype=np.uint8)
    state[1, 5, 5] = 1
    state[2, 5, 5] = 2  # burned out, nothing burning, fire still active

    report = raster_qa(state, _grid())

    assert report["frames_with_no_burning_cell"] == 2, "frame 0 is pre-ignition, frame 2 is dormant"
    assert report["frames_with_no_burning_cell_while_active"] == 1, "frame 0 has no fire yet"
    verdict = _verdict(_clean_report(report))
    assert verdict["pass"] is True
    assert any("no cell in state 1" in w for w in verdict["warnings"])


def test_raster_qa_refuses_a_field_that_is_not_time_by_space() -> None:
    """FAILS WHEN: the rank check is dropped and a single frame is broadcast as a
    one-step series, which silently reports every count as 0."""
    with pytest.raises(ValueError, match=r"\(T, H, W\)"):
        raster_qa(np.zeros((12, 12), dtype=np.uint8), _grid())


# --------------------------------------------------------------------------
# teleporting
# --------------------------------------------------------------------------


def test_teleport_distance_is_the_real_gap_not_merely_non_zero() -> None:
    """The magnitude decides the downstream treatment, so it is checked by hand.

    A detached body 6.4 km from the front is spotting or a label jump; a body one
    cell away is rasterisation wobble. Both make ``teleport_steps`` non-zero, so
    only the distance separates them.

    FAILS WHEN: the gap is measured from the body to the ANCHOR chosen for some
    other purpose rather than to the nearest previously burned cell, or the
    cell-to-km conversion drops the grid cell size and every gap reads in cells.
    """
    state = np.zeros((3, 12, 12), dtype=np.uint8)
    state[0, 2, 2] = 1
    state[1, 2, 2] = 2
    state[1, 2, 3] = 1
    state[2, 2, 2] = 2
    state[2, 2, 3] = 2
    state[2, 7, 7] = 1  # detached: nearest prior burned cell is (2, 3)

    report = raster_qa(state, _grid())

    expected_km = float(np.hypot(7 - 2, 7 - 3))  # 6.403... at 1 km cells
    assert report["teleport_steps"] == 1
    assert report["teleport_cells_total"] == 1
    assert report["teleport_max_gap_km"] == pytest.approx(expected_km, abs=1e-5)
    assert any("detached" in w for w in _verdict(_clean_report(report))["warnings"])


def test_growth_touching_the_front_diagonally_is_not_a_teleport() -> None:
    """The neighbourhood is 8-connected. A diagonal step is ordinary spread.

    FAILS WHEN: the dilation becomes 4-connected, which reclassifies every
    diagonal growth step in the corpus as a teleport and floods the warning with
    noise until nobody reads it.
    """
    state = np.zeros((2, 12, 12), dtype=np.uint8)
    state[0, 5, 5] = 1
    state[1, 5, 5] = 2
    state[1, 6, 6] = 1

    report = raster_qa(state, _grid())
    assert report["teleport_steps"] == 0
    assert report["teleport_max_gap_km"] == 0.0


def test_max_gap_km_scales_with_the_grid_cell_size() -> None:
    """FAILS WHEN: the helper returns cells and the caller labels them km, which
    would make a 2 km corpus report half the true distance."""
    detached = np.zeros((12, 12), dtype=bool)
    detached[0, 4] = True
    prior = np.zeros((12, 12), dtype=bool)
    prior[0, 0] = True

    at_1km = _max_gap_km(detached, prior, _grid())
    at_2km = _max_gap_km(
        detached, prior, Grid(x_min=0.0, y_max=24000.0, nx=12, ny=12, cell_size_m=2000.0)
    )
    assert at_1km == pytest.approx(4.0)
    assert at_2km == pytest.approx(8.0)
    assert _max_gap_km(detached, np.zeros((12, 12), dtype=bool), _grid()) == 0.0


def test_longest_run_counts_the_longest_streak_not_the_total() -> None:
    """FAILS WHEN: the accumulator is never reset on a False, so a fire with many
    scattered dormant hours reports one enormous dormant run."""
    flags = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert _longest_run(flags) == 3
    assert _longest_run(np.zeros(5, dtype=bool)) == 0
    assert _longest_run(np.ones(5, dtype=bool)) == 5


# --------------------------------------------------------------------------
# label noise, in kilometres
# --------------------------------------------------------------------------


def _series(geoms: list, timesteps: list[int]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"timestep": timesteps}, geometry=geoms, crs=CRS_STRING)


def test_the_parallax_offset_is_reported_as_a_VECTOR_not_only_a_magnitude() -> None:
    """The distinction the observation-noise model is built on.

    Both series below have a mean centroid offset of exactly 1 km. In the first
    the offset always points the same way, which is a systematic viewing-geometry
    error; in the second it alternates, which is random scatter. Only the vector
    mean separates them, and the choice between a biased noise model and a
    symmetric one depends on which of the two the corpus actually shows.

    FAILS WHEN: ``centroid_offset_vector_km`` is computed as the mean of the
    per-step distances rather than the mean of the per-step displacements, in
    which case both cases report the same number and the systematic component
    becomes unmeasurable.
    """
    always_east = _series([box(0, 0, 2000, 2000), box(0, 0, 3000, 3000)], [1, 2])
    reference = _series([box(1000, 0, 3000, 2000), box(1000, 0, 4000, 3000)], [1, 2])
    systematic = east_west_noise(always_east, reference)

    assert systematic["n_common_timesteps"] == 2
    assert systematic["centroid_offset_km_mean"] == pytest.approx(1.0)
    assert systematic["centroid_offset_vector_km"] == [pytest.approx(-1.0), pytest.approx(0.0)]

    alternating = _series([box(0, 0, 2000, 2000), box(2000, 0, 4000, 2000)], [1, 2])
    same_reference = _series([box(1000, 0, 3000, 2000), box(1000, 0, 3000, 2000)], [1, 2])
    random_scatter = east_west_noise(alternating, same_reference)

    assert random_scatter["centroid_offset_km_mean"] == pytest.approx(1.0)
    assert random_scatter["centroid_offset_vector_km"] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ], "alternating offsets must cancel in the vector mean; that is the whole point of it"


def test_the_equivalent_radius_mismatch_records_WHICH_variant_is_larger() -> None:
    """One geostationary slot running consistently larger is a bias, and the
    fraction is the field that says so.

    FAILS WHEN: the mismatch is stored as an absolute value before
    ``east_larger_fraction`` is computed, which pins that fraction at 1.0 for
    every fire and destroys the sign the bias lives in.
    """
    bigger = _series([box(0, 0, 4000, 4000), box(0, 0, 4000, 4000)], [1, 2])
    smaller = _series([box(0, 0, 2000, 2000), box(0, 0, 2000, 2000)], [1, 2])

    out = east_west_noise(bigger, smaller)
    assert out["east_larger_fraction"] == pytest.approx(1.0)
    assert out["equiv_radius_mismatch_km_mean"] > 0.0

    flipped = east_west_noise(smaller, bigger)
    assert flipped["east_larger_fraction"] == pytest.approx(0.0)
    assert flipped["equiv_radius_mismatch_km_mean"] == pytest.approx(
        out["equiv_radius_mismatch_km_mean"]
    ), "the magnitude is symmetric even though the fraction is not"


def test_disjoint_timesteps_report_zero_common_steps_rather_than_a_number() -> None:
    """FAILS WHEN: the empty intersection falls through to the statistics block
    and emits NaN means, which serialise into a manifest as null and read as
    "no label noise" instead of "not measured"."""
    east = _series([box(0, 0, 1000, 1000)], [1])
    west = _series([box(0, 0, 1000, 1000)], [99])
    out = east_west_noise(east, west)
    assert out == {"n_common_timesteps": 0}


def test_the_grid_form_of_the_disagreement_is_reported_in_cells() -> None:
    """The km numbers are the physics; the cell number is what the model sees.

    FAILS WHEN: the symmetric difference is computed as the difference of the two
    cell COUNTS rather than the count of cells where the two masks disagree, in
    which case two equal-area perimeters in different places score 0.
    """
    grid = _grid(12)
    east = _series([box(0, 8000, 4000, 12000)], [1])
    west = _series([box(8000, 8000, 12000, 12000)], [1])

    out = east_west_noise(east, west, grid)
    assert out["symmetric_difference_cells_mean"] == pytest.approx(32.0)
    assert out["symmetric_difference_cells_max"] == 32
    assert out["iou_mean"] == pytest.approx(0.0), "disjoint perimeters, equal area"
