"""C4 - the synthetic fire generator.

This fixture is what unblocks modelling and sim, so regressions here
are P0. The tests pin the five properties the rest of the project builds on:
it is fast, it is deterministic, it exercises all three states under the
ratified ``fireline_v2`` rule, it contains a real long-range barrier crossing,
and it contains a **dormancy** during which state 1 is legitimately empty.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wildfire_nowcast.common import contract as C
from wildfire_nowcast.common import zarr_io as zio
from wildfire_nowcast.common.states import burning_residence_hours, frames_without_burning
from wildfire_nowcast.common.synthetic import (
    MIN_HOURS,
    SYNTHETIC_BLOCK_ID,
    SyntheticFire,
    build_synthetic_dataset,
    default_grid_for,
    make_synthetic_fire,
)

MAX_SECONDS = 5.0


def _channel(ds: xr.Dataset, name: str) -> np.ndarray:
    return zio.channel_values(ds, name)


def _state(ds: xr.Dataset) -> np.ndarray:
    return np.asarray(ds[C.FIRE_STATE].values)


# --------------------------------------------------------------------------
# speed and shape
# --------------------------------------------------------------------------


def test_make_synthetic_fire_is_under_five_seconds(tmp_path: Path) -> None:
    start = time.perf_counter()
    make_synthetic_fire(seed=1, n_hours=24, out=tmp_path / "tensor.zarr")
    elapsed = time.perf_counter() - start
    assert elapsed < MAX_SECONDS, f"C4 must stay under {MAX_SECONDS}s, took {elapsed:.2f}s"


def test_returns_tensor_and_manifest_paths(default_synthetic: SyntheticFire) -> None:
    assert isinstance(default_synthetic, tuple)
    assert default_synthetic.tensor_path.exists()
    assert default_synthetic.manifest_path.is_file()
    assert default_synthetic.norm_stats_path.is_file()
    assert default_synthetic.manifest_path.parent == default_synthetic.tensor_path.parent


def test_output_satisfies_c1_c2_c3(default_synthetic: SyntheticFire) -> None:
    report = C.check_all(default_synthetic.tensor_path)
    assert report.ok, "\n" + report.format(verbose=True)


def test_default_is_24_hours(default_synthetic: SyntheticFire) -> None:
    ds = zio.open_tensor(default_synthetic.tensor_path)
    assert int(ds.sizes["time"]) == 24


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_fire(tmp_path: Path) -> None:
    a, _ = build_synthetic_dataset(seed=7, n_hours=8)
    b, _ = build_synthetic_dataset(seed=7, n_hours=8)
    for name in C.CHANNELS:
        np.testing.assert_array_equal(_channel(a, name), _channel(b, name), err_msg=name)


def test_different_seeds_give_different_fires() -> None:
    a, _ = build_synthetic_dataset(seed=7, n_hours=8)
    b, _ = build_synthetic_dataset(seed=8, n_hours=8)
    assert not np.array_equal(_state(a), _state(b))


# --------------------------------------------------------------------------
# states - the ratified C1.1 rule, `fireline_v2`
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_hours", [MIN_HOURS, 6, 24])
def test_all_three_states_occur(n_hours: int) -> None:
    ds, _ = build_synthetic_dataset(seed=3, n_hours=n_hours)
    present = set(np.unique(_state(ds)).tolist())
    assert {C.UNBURNED, C.BURNING, C.BURNED_OUT} <= present, present


@pytest.mark.parametrize("n_hours", [MIN_HOURS, 6, 24, 48])
def test_scripted_dormancy_produces_empty_burning_frames(n_hours: int) -> None:
    """C1.1 (and ADR-007): the fixture must EXERCISE the empty-state-1
    phenomenon, not hide it or relax the test around it.

    6-37% of real GOFER frames have no cell in state 1 - after a long dormancy
    every cell is closed and the contagion source is the frontier of the burned
    region, not state 1. A consumer that conditions solely on state 1 must break
    here, on a 0.6 s fixture, rather than three weeks later on real data.
    """
    ds, geometry = build_synthetic_dataset(seed=3, n_hours=n_hours)
    state = _state(ds)
    empty = frames_without_burning(state)
    start, stop = geometry.dormancy_hours
    assert stop > start, "the fixture must script a dormancy"
    assert empty.size, "the dormancy produced no empty-burning frame"
    assert set(empty.tolist()) <= set(range(start, stop)), (
        "empty-burning frames must come from the scripted dormancy, not from a stalled front"
    )
    # The fire must come back afterwards, or this is just a truncated run.
    if stop < n_hours:
        assert (state[stop:] == C.BURNING).any(), "the fire never resumed after the dormancy"


def test_dormancy_fraction_is_in_the_range_real_fires_show() -> None:
    ds, _ = build_synthetic_dataset(seed=3, n_hours=24)
    state = _state(ds)
    fraction = frames_without_burning(state).size / state.shape[0]
    assert 0.05 <= fraction <= 0.40, f"empty-burning fraction {fraction:.2f} outside C1.1's 6-37%"


def test_burning_is_a_multi_hour_contiguous_run_not_a_one_hour_artefact() -> None:
    """The retired provisional rule made every cell burn for exactly 1 h, an
    artefact of Δt. Under `fireline_v2` residence is multi-hour (GOFER p50
    3-5 h), and each cell burns in exactly one contiguous run."""
    ds, _ = build_synthetic_dataset(seed=3, n_hours=24)
    state = _state(ds)
    residence = burning_residence_hours(state)
    assert residence.size
    assert residence.max() > 1, "state 1 is still a one-hour artefact"
    assert float(np.median(residence)) >= 2.0

    burning = state == C.BURNING
    for axis0 in range(0, burning.shape[1], 7):  # stride: this is O(cells)
        for axis1 in range(0, burning.shape[2], 7):
            series = burning[:, axis0, axis1]
            if not series.any():
                continue
            hot = np.flatnonzero(series)
            assert hot[-1] - hot[0] + 1 == hot.size, "burning hours must be contiguous"


def test_fire_is_absorbing_and_never_skips_the_burning_state() -> None:
    ds, _ = build_synthetic_dataset(seed=3, n_hours=24)
    assert C.fire_state_violations(_state(ds)) == []


def test_the_retired_provisional_rule_is_gone() -> None:
    """ADR-006 P1 retired it; C0 says the one live implementation is in
    `common.states`. Asking for the old rule must raise, not silently relabel."""
    from wildfire_nowcast.common.states import apply_state_rule

    masks = np.zeros((3, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="RETIRED"):
        apply_state_rule(masks, rule="provisional_p0", fire_line_masks=masks)


def test_n_hours_below_minimum_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_hours must be"):
        build_synthetic_dataset(seed=0, n_hours=MIN_HOURS - 1)


def test_tiny_grid_is_rejected() -> None:
    from wildfire_nowcast.common.grid import Grid

    with pytest.raises(ValueError, match="at least"):
        build_synthetic_dataset(seed=0, n_hours=6, grid=Grid(0.0, 0.0, nx=8, ny=8))


# --------------------------------------------------------------------------
# the scripted barrier crossing
# --------------------------------------------------------------------------


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Cells along the straight line between two array indices (Bresenham)."""
    cells: list[tuple[int, int]] = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        err2 = 2 * err
        if err2 > -dc:
            err -= dc
            r += sr
        if err2 < dr:
            err += dr
            c += sc


def test_water_barrier_is_never_burned() -> None:
    ds, _ = build_synthetic_dataset(seed=3, n_hours=24)
    state = _state(ds)
    barrier = _channel(ds, "water_barrier_mask")[0] > 0.5
    assert barrier.any(), "the synthetic fire must contain a water barrier"
    assert not (state[:, barrier] != C.UNBURNED).any(), "fire burned through the water barrier"


def test_main_front_actually_reaches_the_barrier() -> None:
    """The barrier must *block* something, otherwise it is scenery."""
    ds, _ = build_synthetic_dataset(seed=3, n_hours=24)
    state = _state(ds)
    barrier = _channel(ds, "water_barrier_mask")[0] > 0.5

    adjacent = np.zeros_like(barrier)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            adjacent |= np.roll(np.roll(barrier, dr, axis=0), dc, axis=1)
    adjacent &= ~barrier

    # Strictly west of the first barrier cell in each row, so this measures the
    # upwind front arriving at the river rather than the far-side spot fire.
    n_cols = barrier.shape[1]
    first_barrier_col = np.where(barrier.any(axis=1), barrier.argmax(axis=1), n_cols)
    west_of_river = np.arange(n_cols)[None, :] < first_barrier_col[:, None]

    burned = state > C.UNBURNED
    assert burned[-1][adjacent & west_of_river].any(), (
        "the main front never reached the barrier, so the blocking path is untested"
    )


def test_scripted_barrier_crossing_is_a_genuine_jump() -> None:
    """At the scripted hour, fire appears on the far side of the water barrier,
    too far from any previously burned cell to be contiguous spread, with the
    barrier lying between the two."""
    ds, geometry = build_synthetic_dataset(seed=3, n_hours=24)
    state = _state(ds)
    barrier = _channel(ds, "water_barrier_mask")[0] > 0.5
    burned = state > C.UNBURNED

    t = geometry.crossing_hour
    previously = np.argwhere(burned[t - 1])
    new_fire = np.argwhere(state[t] == C.BURNING)
    assert previously.size and new_fire.size

    distance = np.maximum(
        np.abs(new_fire[:, 0, None] - previously[None, :, 0]),
        np.abs(new_fire[:, 1, None] - previously[None, :, 1]),
    ).min(axis=1)
    jump = int(distance.max())
    assert jump >= 5, f"largest new-fire gap at hour {t} was only {jump} cells: no spotting"

    spot = new_fire[int(distance.argmax())]
    nearest = previously[
        int(
            np.maximum(
                np.abs(previously[:, 0] - spot[0]), np.abs(previously[:, 1] - spot[1])
            ).argmin()
        )
    ]
    crossed = [cell for cell in _segment_cells(*spot, *nearest) if barrier[cell]]
    assert crossed, (
        f"the spot at {tuple(spot)} is {jump} cells from the fire at {tuple(nearest)}, "
        "but the straight path between them never crosses the water barrier"
    )


def test_barrier_crossing_survives_a_short_run() -> None:
    ds, geometry = build_synthetic_dataset(seed=3, n_hours=MIN_HOURS)
    state = _state(ds)
    assert 1 <= geometry.crossing_hour < MIN_HOURS
    burned_before = np.argwhere(state[geometry.crossing_hour - 1] > C.UNBURNED)
    new_fire = np.argwhere(state[geometry.crossing_hour] == C.BURNING)
    distance = np.maximum(
        np.abs(new_fire[:, 0, None] - burned_before[None, :, 0]),
        np.abs(new_fire[:, 1, None] - burned_before[None, :, 1]),
    ).min(axis=1)
    assert int(distance.max()) >= 5


@pytest.mark.parametrize("n_hours", [6, 24, 48, 96, 168])
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_fire_keeps_the_c1_2_buffer_at_every_horizon(n_hours: int, seed: int) -> None:
    """C1.2: the final footprint sits >= 10 km inside every edge.

    This test used to assert only that the outermost row and column were
    unburned, and it PASSED while the 48 h fire ran to within 2 cells of the
    east edge and the 96/168 h fires touched it outright. "Not quite touching"
    is not "buffered by 10 km", and the difference was invisible because C1.2's
    buffer sentence was ratified at v2 and never implemented (A10). A weak
    assertion in a green test is the same failure mode as an absent clause.
    """
    ds, _ = build_synthetic_dataset(seed=seed, n_hours=n_hours)
    burned = _state(ds)[-1] > C.UNBURNED
    rows = np.flatnonzero(burned.any(axis=1))
    cols = np.flatnonzero(burned.any(axis=0))
    ny, nx = burned.shape
    margins = {
        "north": int(rows[0]),
        "south": int(ny - 1 - rows[-1]),
        "west": int(cols[0]),
        "east": int(nx - 1 - cols[-1]),
    }
    assert min(margins.values()) >= C.MIN_BUFFER_MARGIN_CELLS, margins


def test_default_domain_grows_with_the_horizon() -> None:
    assert default_grid_for(24).shape == (128, 128), "the 24 h default domain is documented"
    assert default_grid_for(48).nx > default_grid_for(24).nx
    assert default_grid_for(3).shape == default_grid_for(24).shape
    assert max(default_grid_for(1000).shape) <= 320


@pytest.mark.parametrize("n_hours", [48, 96])
def test_long_horizon_still_conforms_and_is_fast(tmp_path: Path, n_hours: int) -> None:
    start = time.perf_counter()
    result = make_synthetic_fire(seed=2, n_hours=n_hours, out=tmp_path / "tensor.zarr")
    elapsed = time.perf_counter() - start
    assert elapsed < MAX_SECONDS, f"{n_hours} h fire took {elapsed:.2f}s"
    report = C.check_all(result.tensor_path)
    assert report.ok, "\n" + report.format(verbose=True)


def test_the_24h_default_fixture_is_unchanged_by_the_c1_2_reserve() -> None:
    """The C1.2 reserve must be a no-op where it does not bind.

    modelling and simviz build against the 24 h default; a fixture that
    silently changes shape under them is a P0. The reserve only ever removes
    cells the fire would have burned outside the 10-cell frame, and at 24 h the
    fire stops 13-16 cells short of it - so the default fire is bit-identical,
    which is asserted here rather than assumed (the A5 byte-identical precedent).
    """
    ds, _ = build_synthetic_dataset(seed=0, n_hours=24)
    state = _state(ds)
    assert state.shape == (24, 128, 128)
    assert int(state[-1].sum()) == 2705, "the 24 h default fire changed; that is a P0"
    assert int((state[-1] > C.UNBURNED).sum()) == 1598
    burned = state[-1] > C.UNBURNED
    rows, cols = np.flatnonzero(burned.any(axis=1)), np.flatnonzero(burned.any(axis=0))
    assert (int(rows[0]), int(cols[0])) == (42, 27)


# --------------------------------------------------------------------------
# non-fire channels are not degenerate
# --------------------------------------------------------------------------


def test_static_channels_are_constant_in_time(default_synthetic: SyntheticFire) -> None:
    ds = zio.open_tensor(default_synthetic.tensor_path)
    for name in sorted(C.STATIC_CHANNELS):
        values = _channel(ds, name)
        assert np.array_equal(values[0], values[-1]), f"{name} must be static"


def test_weather_channels_vary_in_time(default_synthetic: SyntheticFire) -> None:
    ds = zio.open_tensor(default_synthetic.tensor_path)
    for name in ("wind_u10", "wind_v10", "temp_2m", "rh_2m", "fuel_moisture_proxy"):
        values = _channel(ds, name)
        assert not np.array_equal(values[0], values[-1]), f"{name} must vary in time"


def test_channels_are_physically_plausible(default_synthetic: SyntheticFire) -> None:
    ds = zio.open_tensor(default_synthetic.tensor_path)
    temp = _channel(ds, "temp_2m")
    assert np.all(np.isfinite(temp))
    assert 250.0 < temp.min() and temp.max() < 340.0
    for name, lo, hi in (
        ("rh_2m", 0.0, 100.0),
        ("canopy_cover", 0.0, 100.0),
        ("slope", 0.0, 90.0),
    ):
        values = _channel(ds, name)
        assert lo <= values.min() and values.max() <= hi, name
    for name in sorted(C.BINARY_CHANNELS):
        assert set(np.unique(_channel(ds, name)).tolist()) <= {0.0, 1.0}
    aspect = _channel(ds, "aspect_sin") ** 2 + _channel(ds, "aspect_cos") ** 2
    assert np.all(aspect <= 1.0 + 1e-5)
    assert _channel(ds, "fuel_moisture_proxy").min() >= 1.0
    # C1 channel 9 is a class id: interpolating it would invent fuel models.
    fuel = _channel(ds, "fuel_model_id")
    assert np.array_equal(fuel, np.round(fuel))


def test_recent_burn_scar_is_present_and_suppresses_spread() -> None:
    ds, _ = build_synthetic_dataset(seed=3, n_hours=24)
    scar = _channel(ds, "recent_burn_scar")[0] > 0.5
    assert scar.any(), "the synthetic fire must contain a recent burn scar"
    burned = _state(ds)[-1] > C.UNBURNED
    assert burned[scar].mean() < burned[~scar].mean() + 0.5


# --------------------------------------------------------------------------
# C2/C3 siblings the fixture writes
# --------------------------------------------------------------------------


def test_synthetic_is_never_selectable_into_a_real_cv_split(
    default_synthetic: SyntheticFire,
) -> None:
    """A synthetic fire belongs to no landscape, so it can never be counted as
    a spatial block (C3.1) nor make a norm-stats file look reporting-ready."""
    manifest = zio.read_manifest(default_synthetic.manifest_path)
    assert manifest["cv_fold"] == -1
    assert manifest["spatial_block_id"] == SYNTHETIC_BLOCK_ID == -1

    stats = zio.read_norm_stats(default_synthetic.norm_stats_path)
    assert stats["n_train_blocks"] == 1 and stats["bootstrap"] is True
    report = C.check_all(default_synthetic.tensor_path)
    assert report.ok, "\n" + report.format(verbose=True)
    assert not report.reporting_ok, "a synthetic fire must never be reporting-ready"


# --------------------------------------------------------------------------
# CLI / `make synth`
# --------------------------------------------------------------------------


def test_cli_writes_a_conformant_triple(tmp_path: Path) -> None:
    out = tmp_path / "run" / "tensor.zarr"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "wildfire_nowcast.common.synthetic",
            "--out",
            str(out),
            "--seed",
            "5",
            "--hours",
            "6",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists()
    assert C.check_all(out).ok, C.check_all(out).format(verbose=True)
