"""Grid alignment: how a vector perimeter becomes cells, and how hours line up.

WHAT STANDS BEHIND THIS. Three small functions decide the geometry and the phase
of every label in the corpus, and all three were at or near zero coverage:

* ``rasterize.polygon_mask`` turns each hourly GOFER perimeter into the boolean
  footprint that becomes channel 0. Its rule is area fraction with a 0.5
  threshold, not centroid-in-polygon and not ``all_touched``.
* ``labels.fire_domain_grid`` fixes the per-fire domain: final-perimeter bbox,
  buffered outward, snapped to the continental lattice. Everything else in the
  fire is indexed against that grid.
* ``pipeline.weather_hours`` applies the one-hour lag that puts RTMA in phase
  with an end-of-hour perimeter stamp.

WHY THE THRESHOLD CHOICE IS WORTH A TEST. ``all_touched=True`` dilates every
perimeter by a cell. On a ~2 km effective product mapped to a 1 km grid that is
a systematic positive area bias applied to every fire and every hour, and it
would be invisible in any per-fire check because the tensor stays perfectly
conformant. The area-fraction rule at 0.5 is unbiased for a straight boundary,
which is the property asserted below rather than assumed.

WHY THE LAG IS WORTH A TEST. It is one subtraction, applied in one place, and
the module says so: applying it anywhere else, or not at all, trains every fire
in the corpus an hour out of phase with its own weather. There is no artifact
that would show that; the tensor is conformant either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.labels import (
    DEFAULT_BUFFER_M,
    LabelBuild,
    fire_domain_grid,
    write_interim_fire_state,
)
from wildfire_nowcast.data.pipeline import WEATHER_LAG, weather_hours
from wildfire_nowcast.data.rasterize import (
    COVER_THRESHOLD,
    DEFAULT_OVERSAMPLE,
    line_mask,
    oversampled_transform,
    polygon_coverage,
    polygon_mask,
)

CELL_M = 1000.0


def _grid(n: int = 5) -> Grid:
    return Grid(x_min=0.0, y_max=float(n) * CELL_M, nx=n, ny=n, cell_size_m=CELL_M)


# --------------------------------------------------------------------------
# the area-fraction rule
# --------------------------------------------------------------------------


def test_coverage_is_the_true_area_fraction_of_each_cell() -> None:
    """A 400 m strip of a 1 km cell is 0.4 of it, to the oversampling quantum.

    FAILS WHEN: the fine lattice is block-summed instead of block-averaged (every
    fraction scales by the oversample factor), or the sub-cell transform is built
    from the grid's cell size rather than cell size over factor.
    """
    grid = _grid()
    strip = box(0.0, 4000.0, 400.0, 5000.0)  # top-left cell, 40% of its width

    coverage = polygon_coverage(strip, grid)
    assert coverage.shape == grid.shape
    assert coverage[0, 0] == pytest.approx(0.4, abs=1e-6)
    assert coverage.sum() == pytest.approx(0.4, abs=1e-6), "no other cell is touched"


def test_the_threshold_is_inclusive_at_exactly_half_a_cell() -> None:
    """The boundary case, pinned in both directions.

    0.5 is the area-preserving choice: on a straight boundary the cells it
    includes and the cells it excludes cancel. A strictly-greater comparison
    biases every perimeter in the corpus slightly small, and a 0.4 threshold
    biases them all large.

    FAILS WHEN: ``polygon_mask`` uses ``>`` instead of ``>=``, which is a
    same-length edit and therefore exactly the kind that a stale bytecode file
    can hide; it is asserted at the boundary rather than near it.
    """
    grid = _grid()
    assert COVER_THRESHOLD == 0.5

    # 100x oversampling puts the coverage quantum at 0.01 of a cell width, so
    # these three strips are the adjacent representable values around the bar.
    fine = {"factor": 100}
    just_under = polygon_coverage(box(0.0, 4000.0, 490.0, 5000.0), grid, **fine)[0, 0]
    exactly_half = polygon_coverage(box(0.0, 4000.0, 500.0, 5000.0), grid, **fine)[0, 0]
    just_over = polygon_coverage(box(0.0, 4000.0, 510.0, 5000.0), grid, **fine)[0, 0]
    assert (just_under, exactly_half, just_over) == pytest.approx((0.49, 0.50, 0.51))

    assert not polygon_mask(box(0.0, 4000.0, 490.0, 5000.0), grid, **fine)[0, 0]
    assert polygon_mask(box(0.0, 4000.0, 500.0, 5000.0), grid, **fine)[0, 0], (
        "a cell exactly half covered is inside the perimeter"
    )
    assert polygon_mask(box(0.0, 4000.0, 510.0, 5000.0), grid, **fine)[0, 0]


def test_the_polygon_rule_erases_a_line_and_the_line_rule_keeps_it() -> None:
    """The two rules are deliberately different, and this is the case that shows it.

    A fire line has no area, so the area-fraction rule scores it far below the
    threshold and returns nothing. That is correct for a perimeter and fatal for
    an active-fire line, which is why the module carries a second rule using
    ``all_touched``.

    FAILS WHEN: ``line_mask`` is switched to ``all_touched=False``, at which
    point the active fire line channel goes empty on many hours and the state
    rule loses the seed it distinguishes burning from burned-out with.
    """
    grid = _grid()
    horizontal = LineString([(0.0, 4500.0), (5000.0, 4500.0)])

    assert polygon_mask(horizontal, grid).sum() == 0
    touched = line_mask(horizontal, grid)
    assert touched[0].all(), "every cell along the line is touched"
    assert touched.sum() == 5


def test_an_absent_geometry_gives_an_empty_mask_of_the_right_shape() -> None:
    """FAILS WHEN: ``None`` propagates into the rasteriser and raises, which
    would abort a build on the ordinary case of an hour with no mapped fire
    line instead of recording an empty frame."""
    grid = _grid()
    for empty in (None, box(0, 0, 0, 0)):
        assert polygon_coverage(empty, grid).shape == grid.shape
        assert polygon_coverage(empty, grid).sum() == 0.0
        assert polygon_mask(empty, grid).sum() == 0
    assert line_mask(None, grid).shape == grid.shape
    assert line_mask(None, grid).sum() == 0


def test_the_oversampled_transform_is_the_same_grid_seen_finer() -> None:
    """It must introduce no second definition of the grid origin.

    FAILS WHEN: the transform is anchored at ``y_min`` instead of ``y_max``, which
    flips every perimeter vertically while leaving the tensor perfectly
    conformant, or the factor guard is removed and a factor of 0 divides by zero.
    """
    grid = _grid()
    transform = oversampled_transform(grid, DEFAULT_OVERSAMPLE)

    assert transform.a == pytest.approx(CELL_M / DEFAULT_OVERSAMPLE)
    assert transform.e == pytest.approx(-CELL_M / DEFAULT_OVERSAMPLE)
    assert transform.c == pytest.approx(grid.x_min)
    assert transform.f == pytest.approx(grid.y_max)

    with pytest.raises(ValueError, match="factor must be >= 1"):
        oversampled_transform(grid, 0)


def test_a_finer_oversample_refines_the_fraction_it_does_not_move_the_cell() -> None:
    """FAILS WHEN: the reshape axes are transposed, which mixes sub-cells from
    different rows and produces coverage numbers that look plausible and belong
    to the wrong cell."""
    grid = _grid()
    three_tenths = box(0.0, 4000.0, 300.0, 5000.0)
    at_10 = polygon_coverage(three_tenths, grid, factor=10)
    at_20 = polygon_coverage(three_tenths, grid, factor=20)
    at_50 = polygon_coverage(three_tenths, grid, factor=50)
    assert at_10[0, 0] == pytest.approx(0.3, abs=1e-6)
    assert at_20[0, 0] == pytest.approx(0.3, abs=1e-6)
    assert at_50[0, 0] == pytest.approx(0.3, abs=1e-6)
    assert at_10.sum() == pytest.approx(at_50.sum(), abs=1e-6)


# --------------------------------------------------------------------------
# the per-fire domain
# --------------------------------------------------------------------------


def test_the_domain_is_the_bbox_buffered_outward_and_snapped_outward() -> None:
    """Both directions of the snap, because rounding to nearest loses the buffer.

    FAILS WHEN: ``snap`` rounds to nearest rather than outward, which can shave a
    cell off the buffered domain and put the model's spread room inside the
    perimeter it was supposed to sit outside.
    """
    exact = fire_domain_grid((1000.0, 1000.0, 2000.0, 2000.0), buffer_m=0.0)
    assert tuple(exact.bounds) == (1000.0, 1000.0, 2000.0, 2000.0)

    off_lattice = fire_domain_grid((1500.0, 1500.0, 2500.0, 2500.0), buffer_m=0.0)
    assert tuple(off_lattice.bounds) == (1000.0, 1000.0, 3000.0, 3000.0)

    buffered = fire_domain_grid((0.0, 0.0, 1000.0, 1000.0))
    assert DEFAULT_BUFFER_M == 10_000.0
    assert tuple(buffered.bounds) == (-10_000.0, -10_000.0, 11_000.0, 11_000.0)
    assert buffered.nx == buffered.ny == 21


def test_the_domain_grid_is_at_the_contract_cell_size_by_default() -> None:
    """FAILS WHEN: the default resolution stops coming from the contract, so a
    fire is built on a lattice the contract checker will not recognise."""
    grid = fire_domain_grid((0.0, 0.0, 5000.0, 5000.0), buffer_m=0.0)
    assert grid.cell_size_m == CELL_M
    coarse = fire_domain_grid((0.0, 0.0, 5000.0, 5000.0), buffer_m=0.0, res_m=2000.0)
    assert coarse.cell_size_m == 2000.0
    assert coarse.nx == 3, "5 km snapped outward onto a 2 km lattice is 6 km"


# --------------------------------------------------------------------------
# phase
# --------------------------------------------------------------------------


def test_the_weather_hour_is_the_hour_BEFORE_an_end_of_hour_perimeter_stamp() -> None:
    """One subtraction, and the sign of it decides whether the corpus is in phase.

    A perimeter stamped 05:00 describes 04:00 to 05:00, so the weather that drove
    it is the analysis at 04:00.

    FAILS WHEN: the lag is added instead of subtracted, or set to zero. Both
    produce a tensor that passes every structural check while every fire is
    trained against weather from the wrong hour, in the second case by exactly
    the amount a one-hour nowcast is trying to resolve.
    """
    assert WEATHER_LAG == pd.Timedelta(hours=1)

    stamps = pd.DatetimeIndex(pd.to_datetime(["2020-09-05T05:00:00", "2020-09-05T06:00:00"]))
    hours = weather_hours(stamps)

    assert list(hours.strftime("%Y-%m-%dT%H:%M:%S")) == [
        "2020-09-05T04:00:00",
        "2020-09-05T05:00:00",
    ]
    assert (stamps - hours == pd.Timedelta(hours=1)).all()


def test_the_lag_crosses_a_day_boundary_correctly() -> None:
    """FAILS WHEN: the lag is applied by editing the hour field rather than by
    timedelta arithmetic, which turns midnight into hour -1 or wraps it inside
    the same day."""
    midnight = pd.DatetimeIndex(pd.to_datetime(["2020-09-05T00:00:00"]))
    assert weather_hours(midnight)[0] == pd.Timestamp("2020-09-04T23:00:00")


# --------------------------------------------------------------------------
# the interim door
# --------------------------------------------------------------------------


def test_a_label_only_product_refuses_to_be_written_near_the_C1_path(tmp_path: Path) -> None:
    """A one-channel store under the corpus root would read as a built fire.

    FAILS WHEN: the path guard is dropped or narrowed to an exact match on the
    corpus root, so a label-only store lands one directory deeper and is picked
    up by every glob that enumerates built fires.
    """
    grid = _grid(3)
    build = LabelBuild(
        fire_id="synthetic_fire",
        grid=grid,
        times=pd.date_range("2020-09-05T00:00:00", periods=2, freq="h"),
        state=np.zeros((2, 3, 3), dtype=np.uint8),
        qa={"verdict": {"pass": True}},
        provenance={"label_source": "gofer"},
    )

    with pytest.raises(ValueError, match="refusing to write a partial"):
        write_interim_fire_state(build, tmp_path / "data" / "fires")

    written = write_interim_fire_state(build, tmp_path / "interim")
    assert written.exists()
    assert (written.parent / "fire_state.qa.json").is_file(), "QA travels with the labels"
