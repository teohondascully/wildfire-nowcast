"""Shared utilities: grid/CRS helpers, derived formulas, config loading, run dirs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wildfire_nowcast.common import zarr_io as zio
from wildfire_nowcast.common.config import (
    apply_overrides,
    deep_merge,
    dump_yaml,
    get_in,
    load_config,
    parse_override,
)
from wildfire_nowcast.common.contract import (
    CATEGORICAL_CHANNELS,
    CELL_SIZE_M,
    CHANNEL_INDEX,
    CHANNEL_INDEX_OFFSET,
    CHANNELS,
    CRS_STRING,
    FEATURE_CHANNELS,
    NORM_STATS_CATEGORICAL_NOTE,
    is_on_lattice,
)
from wildfire_nowcast.common.derive import (
    aspect_to_sin_cos,
    dead_fuel_moisture_simard,
    kelvin_to_fahrenheit,
    slope_aspect_from_elevation,
    wind_direction_to,
    wind_speed,
)
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import configs_dir, fire_tensor_path, repo_root
from wildfire_nowcast.common.runs import create_run_dir, git_sha, new_run_id, read_run

# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------


def test_grid_coordinates_follow_the_raster_convention() -> None:
    grid = Grid(x_min=-2_000_000.0, y_max=2_000_000.0, nx=5, ny=4)
    assert grid.shape == (4, 5)
    assert np.all(np.diff(grid.x_coords) == CELL_SIZE_M)
    assert np.all(np.diff(grid.y_coords) == -CELL_SIZE_M)
    assert grid.x_coords[0] == -2_000_000.0 + 500.0
    assert grid.y_coords[0] == 2_000_000.0 - 500.0
    assert grid.crs == CRS_STRING


def test_grid_bounds_are_outer_edges() -> None:
    grid = Grid(x_min=-2_000_000.0, y_max=2_000_000.0, nx=5, ny=4)
    assert grid.bounds == (-2_000_000.0, 1_996_000.0, -1_995_000.0, 2_000_000.0)


def test_grid_from_bounds_snaps_outward() -> None:
    grid = Grid.from_bounds((-2_000_123.0, 1_999_100.0, -1_998_900.0, 2_000_400.0))
    assert grid.bounds[0] <= -2_000_123.0
    assert grid.bounds[2] >= -1_998_900.0
    for edge in grid.bounds:
        assert edge % CELL_SIZE_M == 0.0


def test_grid_from_bounds_lands_on_the_continental_lattice() -> None:
    """C1.2 - two fires whose buffered domains overlap must agree cell-for-cell,
    or the C3.1 spatial blocking is comparing different ground."""
    a = Grid.from_bounds((-2_000_123.0, 1_999_100.0, -1_998_900.0, 2_000_400.0))
    b = Grid.from_bounds((-2_001_777.0, 1_998_010.0, -1_999_010.0, 2_000_990.0))
    assert is_on_lattice(a.x_coords) and is_on_lattice(a.y_coords)
    assert is_on_lattice(b.x_coords) and is_on_lattice(b.y_coords)
    shared = set(np.round(a.x_coords, 6)) & set(np.round(b.x_coords, 6))
    assert shared, "overlapping domains must share cell centres exactly"


def test_unsnapped_grids_are_detected() -> None:
    assert not is_on_lattice(Grid(x_min=-2_000_300.0, y_max=2_000_000.0, nx=4, ny=4).x_coords)


def test_grid_rowcol_and_xy_round_trip() -> None:
    grid = Grid(x_min=-2_000_000.0, y_max=2_000_000.0, nx=9, ny=7)
    for row in range(grid.ny):
        for col in range(grid.nx):
            assert grid.rowcol(*grid.xy(row, col)) == (row, col)
    assert grid.contains(0, 0) and not grid.contains(-1, 0) and not grid.contains(0, 9)


def test_grid_transform_matches_bounds() -> None:
    grid = Grid(x_min=-2_000_000.0, y_max=2_000_000.0, nx=5, ny=4)
    a, b, c, d, e, f = grid.transform
    assert (a, b, c, d, e, f) == (1000.0, 0.0, -2_000_000.0, 0.0, -1000.0, 2_000_000.0)
    affine = grid.rasterio_transform()
    assert affine * (0, 0) == (-2_000_000.0, 2_000_000.0)


def test_grid_round_trips_through_a_dataset(synthetic_ds) -> None:  # noqa: ANN001
    grid = Grid.from_dataset(synthetic_ds)
    assert grid.cell_size_m == CELL_SIZE_M
    assert grid.shape == (int(synthetic_ds.sizes["y"]), int(synthetic_ds.sizes["x"]))
    np.testing.assert_allclose(grid.x_coords, synthetic_ds["x"].values)
    np.testing.assert_allclose(grid.y_coords, synthetic_ds["y"].values)


def test_grid_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        Grid(0.0, 0.0, nx=0, ny=4)
    with pytest.raises(ValueError):
        Grid.from_bounds((0.0, 0.0, 0.0, 10.0))


# --------------------------------------------------------------------------
# derived channels
# --------------------------------------------------------------------------


def test_kelvin_to_fahrenheit() -> None:
    assert kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)
    assert kelvin_to_fahrenheit(310.928) == pytest.approx(100.0, abs=1e-2)


def test_simard_moisture_is_monotone_in_humidity_and_temperature() -> None:
    temp = np.full(5, 300.0)
    rh = np.array([5.0, 20.0, 40.0, 60.0, 90.0])
    emc = dead_fuel_moisture_simard(temp, rh)
    assert np.all(np.diff(emc) > 0), "EMC must increase with relative humidity"

    hot = dead_fuel_moisture_simard(np.full(3, 315.0), np.full(3, 40.0))
    cool = dead_fuel_moisture_simard(np.full(3, 285.0), np.full(3, 40.0))
    assert np.all(hot < cool), "EMC must decrease with temperature"


def test_simard_moisture_is_bounded_and_float32() -> None:
    temp = np.linspace(250.0, 330.0, 50)
    rh = np.linspace(0.0, 100.0, 50)
    emc = dead_fuel_moisture_simard(temp[:, None], rh[None, :])
    assert emc.dtype == np.float32
    assert emc.min() >= 1.0 and emc.max() <= 60.0
    assert np.all(np.isfinite(emc))


def test_slope_and_aspect_on_a_known_plane() -> None:
    """A plane tilting up to the east faces west: aspect 270 degrees."""
    nx = ny = 9
    cell = 1000.0
    east_rise = np.tile(np.arange(nx, dtype=np.float64) * cell, (ny, 1))
    slope, aspect = slope_aspect_from_elevation(east_rise, cell)
    assert slope[4, 4] == pytest.approx(45.0, abs=1e-3)
    assert aspect[4, 4] == pytest.approx(270.0, abs=1e-3)

    sin_a, cos_a = aspect_to_sin_cos(aspect, slope)
    assert sin_a[4, 4] == pytest.approx(-1.0, abs=1e-5)
    assert cos_a[4, 4] == pytest.approx(0.0, abs=1e-5)


def test_flat_terrain_has_zero_slope_and_no_aspect() -> None:
    flat = np.full((6, 6), 1234.0)
    slope, aspect = slope_aspect_from_elevation(flat, 1000.0)
    sin_a, cos_a = aspect_to_sin_cos(aspect, slope)
    assert np.all(slope == 0.0)
    assert np.all(sin_a == 0.0) and np.all(cos_a == 0.0)


def test_slope_aspect_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        slope_aspect_from_elevation(np.zeros((2, 3, 4)), 1000.0)


def test_wind_helpers() -> None:
    assert wind_speed(3.0, 4.0) == pytest.approx(5.0)
    assert wind_direction_to(0.0, 1.0) == pytest.approx(0.0)  # toward north
    assert wind_direction_to(1.0, 0.0) == pytest.approx(90.0)  # toward east


# --------------------------------------------------------------------------
# zarr io
# --------------------------------------------------------------------------


def test_hourly_times_and_iso_helpers() -> None:
    times = zio.hourly_times("2019-10-23T18:00:00", 4)
    assert times.size == 4
    assert np.all(np.diff(times) == np.timedelta64(1, "h"))
    assert zio.iso_naive(times[0]) == "2019-10-23T18:00:00"
    assert "Z" not in zio.utc_now_iso()
    with pytest.raises(ValueError):
        zio.hourly_times("2019-10-23T18:00:00", 0)


def test_static_channels_may_be_supplied_as_2d(synthetic_ds) -> None:  # noqa: ANN001
    grid = Grid.from_dataset(synthetic_ds)
    n_t = 4
    times = zio.hourly_times("2020-01-01T00:00:00", n_t)
    channels = {c: np.zeros(grid.shape, dtype=np.float32) for c in CHANNELS}
    channels["fire_state"] = np.zeros((n_t, *grid.shape), dtype=np.uint8)
    ds = zio.build_tensor_dataset(channels, grid, times)
    assert ds["features"].shape == (n_t, 13, *grid.shape)
    assert zio.get_channel(ds, "elevation").shape == (n_t, *grid.shape)
    assert ds["features"].dtype == np.float32


def test_get_channel_applies_the_index_offset_so_nobody_hardcodes_an_integer(
    synthetic_ds,  # noqa: ANN001
) -> None:
    """v2 moved channels 1-13 into one array but renumbered nothing. Reaching a
    channel by NAME must land on the v1 index, whatever the storage layout."""
    features = np.asarray(synthetic_ds["features"].values)
    for name in FEATURE_CHANNELS:
        position = CHANNEL_INDEX[name] - CHANNEL_INDEX_OFFSET
        np.testing.assert_array_equal(zio.channel_values(synthetic_ds, name), features[:, position])
    np.testing.assert_array_equal(
        zio.channel_values(synthetic_ds, "fire_state"),
        np.asarray(synthetic_ds["fire_state"].values),
    )
    with pytest.raises(KeyError):
        zio.get_channel(synthetic_ds, "ndvi")


def test_to_channel_dataset_round_trips_the_v1_read_view(synthetic_ds) -> None:  # noqa: ANN001
    exploded = zio.to_channel_dataset(synthetic_ds)
    assert set(exploded.data_vars) == set(CHANNELS)
    assert exploded["fire_state"].dtype == np.uint8
    np.testing.assert_array_equal(
        exploded["temp_2m"].values, zio.channel_values(synthetic_ds, "temp_2m")
    )


def test_build_tensor_dataset_rejects_shape_and_name_errors(synthetic_ds) -> None:  # noqa: ANN001
    grid = Grid.from_dataset(synthetic_ds)
    times = zio.hourly_times("2020-01-01T00:00:00", 3)
    good = {c: np.zeros((3, *grid.shape), dtype=np.float32) for c in CHANNELS}
    good["fire_state"] = np.zeros((3, *grid.shape), dtype=np.uint8)

    bad_shape = dict(good, elevation=np.zeros((3, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        zio.build_tensor_dataset(bad_shape, grid, times)

    bad_name = dict(good, ndvi=np.zeros((3, *grid.shape), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown channels"):
        zio.build_tensor_dataset(bad_name, grid, times)


def test_stack_channels_uses_c1_index_order(synthetic_ds) -> None:  # noqa: ANN001
    stacked = zio.stack_channels(synthetic_ds)
    assert stacked.shape[1] == len(CHANNELS)
    np.testing.assert_array_equal(
        stacked[:, 0], np.asarray(synthetic_ds["fire_state"].values, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        stacked[:, 12], zio.channel_values(synthetic_ds, "water_barrier_mask")
    )


def test_stack_channels_reports_missing(synthetic_ds) -> None:  # noqa: ANN001
    with pytest.raises(KeyError):
        zio.stack_channels(synthetic_ds.drop_vars("features"))


def test_compute_norm_stats_matches_numpy(synthetic_ds) -> None:  # noqa: ANN001
    stats = zio.compute_norm_stats([synthetic_ds], train_folds=[0, 1])
    assert stats["train_folds"] == [0, 1]
    values = zio.channel_values(synthetic_ds, "temp_2m", dtype=np.float64)
    assert stats["mean"]["temp_2m"] == pytest.approx(float(values.mean()), rel=1e-6)
    assert stats["std"]["temp_2m"] == pytest.approx(float(values.std()), rel=1e-6)


def test_compute_norm_stats_counts_spatial_blocks(synthetic_ds) -> None:  # noqa: ANN001
    """C3.3 - `n_train_blocks` counts distinct landscapes, not fires. Two fires
    from one block are still one block, which is the whole trap ADR-008 closed."""
    one = zio.compute_norm_stats([synthetic_ds], train_folds=[0], spatial_block_ids=[7])
    assert one["n_train_blocks"] == 1 and one["bootstrap"] is True

    same_block = zio.compute_norm_stats(
        [synthetic_ds, synthetic_ds], train_folds=[0, 1], spatial_block_ids=[7, 7]
    )
    assert same_block["n_train_blocks"] == 1, "two fires in one block are one block"

    two = zio.compute_norm_stats(
        [synthetic_ds, synthetic_ds], train_folds=[0, 1], spatial_block_ids=[7, 9]
    )
    assert two["n_train_blocks"] == 2 and "bootstrap" not in two


def test_norm_stats_force_the_categorical_identity_transform(synthetic_ds) -> None:  # noqa: ANN001
    """C3.2 - a caller cannot accidentally standardise an FBFM40 class id, even
    by passing real statistics for it."""
    mean = dict.fromkeys(CHANNELS, 3.0)
    std = dict.fromkeys(CHANNELS, 2.0)
    stats = zio.build_norm_stats(mean, std, train_folds=[0], n_train_blocks=2)
    for name in CATEGORICAL_CHANNELS:
        assert stats["mean"][name] == 0.0 and stats["std"][name] == 1.0
    assert stats["mean"]["temp_2m"] == 3.0
    assert stats[NORM_STATS_CATEGORICAL_NOTE]
    with pytest.raises(ValueError, match="n_train_blocks"):
        zio.build_norm_stats(mean, std, train_folds=[0], n_train_blocks=0)


def test_json_writes_are_atomic_and_readable(tmp_path: Path) -> None:
    path = zio.write_manifest({"fire_id": "x"}, tmp_path / "nested" / "manifest.json")
    assert zio.read_manifest(path) == {"fire_id": "x"}
    assert not list(tmp_path.rglob("*.tmp"))


def test_open_tensor_reports_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        zio.open_tensor(tmp_path / "absent.zarr")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def test_repo_root_and_c1_paths() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert fire_tensor_path("kincade") == root / "data" / "fires" / "kincade" / "tensor.zarr"


def test_data_dir_is_overridable_by_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from wildfire_nowcast.common import paths

    monkeypatch.setenv("WILDFIRE_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == tmp_path.resolve()
    assert paths.fires_dir() == tmp_path.resolve() / "fires"


# --------------------------------------------------------------------------
# config (C7)
# --------------------------------------------------------------------------


def test_deep_merge_recurses_into_mappings_only() -> None:
    base = {"a": {"b": 1, "c": 2}, "list": [1, 2]}
    over = {"a": {"c": 3}, "list": [9]}
    assert deep_merge(base, over) == {"a": {"b": 1, "c": 3}, "list": [9]}
    assert base == {"a": {"b": 1, "c": 2}, "list": [1, 2]}, "inputs must not be mutated"


def test_parse_override_preserves_types() -> None:
    assert parse_override("model.lr=0.001") == (["model", "lr"], 0.001)
    assert parse_override("train.epochs=10") == (["train", "epochs"], 10)
    assert parse_override("data.ids=[a, b]") == (["data", "ids"], ["a", "b"])
    assert parse_override("run.debug=true") == (["run", "debug"], True)
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")


def test_apply_overrides_creates_missing_nodes() -> None:
    assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


def test_shipped_configs_agree_with_the_contract() -> None:
    """A config that disagrees with `common.contract` is a second source of
    truth, and the run that logs it would attest to the wrong interfaces."""
    from wildfire_nowcast.common.contract import (
        CONTRACT_VERSION,
        MIN_TRAIN_BLOCKS_FOR_REPORTING,
        TIME_CONVENTION,
    )

    config = load_config(configs_dir() / "base.yaml")
    assert config["interfaces_version"] == CONTRACT_VERSION
    assert config["grid"]["crs"] == CRS_STRING
    assert config["grid"]["cell_size_m"] == CELL_SIZE_M
    assert config["grid"]["time_convention"] == TIME_CONVENTION
    assert config["data"]["cv"]["scheme"] == "spatial_block", "leave-fire-out is landscape leakage"
    assert config["data"]["cv"]["n_blocks"] == 11, "effective n is 11, not 28 (ADR-006 P4)"
    assert (
        config["data"]["norm_stats"]["min_train_blocks_for_reporting"]
        == MIN_TRAIN_BLOCKS_FOR_REPORTING
    )


def test_shipped_configs_compose() -> None:
    config = load_config(configs_dir() / "synthetic_smoke.yaml")
    assert config["grid"]["crs"] == CRS_STRING
    assert config["grid"]["cell_size_m"] == 1000
    assert config["data"]["source"] == "synthetic"
    assert config["run"]["prefix"] == "synth-smoke"
    assert "defaults" not in config
    assert get_in(config, "data.synthetic.n_hours") == 24
    assert get_in(config, "nope.nothing", default=7) == 7


def test_load_config_applies_overrides(tmp_path: Path) -> None:
    from wildfire_nowcast.common.contract import CONTRACT_VERSION

    dump_yaml({"a": {"b": 1}}, tmp_path / "c.yaml")
    config = load_config(tmp_path / "c.yaml", ["a.b=5", "a.z=hello"])
    # [A14] EVERY resolved config is stamped with the contract version in force,
    # including an ad-hoc one: C7 says a run records its resolved config, and a
    # recorded config that does not say which contract it was written against
    # cannot be read back safely. This is the key that used to be a yaml literal.
    assert config == {"a": {"b": 5, "z": "hello"}, "interfaces_version": CONTRACT_VERSION}


def test_load_config_detects_include_cycles(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text("defaults: [two]\n")
    (tmp_path / "two.yaml").write_text("defaults: [one]\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "one.yaml", configs_root=tmp_path)


def test_load_config_reports_missing_include(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text("defaults: [absent]\n")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "one.yaml", configs_root=tmp_path)


# --------------------------------------------------------------------------
# run directories (C7)
# --------------------------------------------------------------------------


def test_create_run_dir_records_config_and_git_sha(tmp_path: Path) -> None:
    config = {"seed": 3, "model": {"kind": "kernel"}}
    run = create_run_dir(config, run_id="unit-test", runs_root=tmp_path)

    assert run.path == tmp_path / "unit-test"
    assert run.config_path.is_file() and run.meta_path.is_file()

    restored, meta = read_run(run.path)
    assert restored == config, "the resolved config must round-trip verbatim"
    assert meta["run_id"] == "unit-test"
    assert meta["git_sha"] == (git_sha() or "unknown")
    assert meta["git_sha"] != "unknown", "run provenance needs a real git SHA in this repo"
    assert isinstance(meta["git_dirty"], bool)
    assert set(meta["versions"]) >= {"numpy", "xarray", "zarr", "torch"}

    on_disk = json.loads(run.meta_path.read_text())
    assert on_disk["git_sha"] == meta["git_sha"]


def test_create_run_dir_refuses_to_clobber(tmp_path: Path) -> None:
    create_run_dir({}, run_id="dup", runs_root=tmp_path)
    with pytest.raises(FileExistsError):
        create_run_dir({}, run_id="dup", runs_root=tmp_path)
    create_run_dir({"a": 1}, run_id="dup", runs_root=tmp_path, exist_ok=True)


def test_run_ids_are_sortable_and_prefixed() -> None:
    assert new_run_id("train").startswith("train-")
    assert len(new_run_id()) == len("run-20260807-120000")


def test_run_sub_creates_directories(tmp_path: Path) -> None:
    run = create_run_dir({}, run_id="sub", runs_root=tmp_path)
    figures = run.sub("figures")
    assert figures.is_dir() and figures.parent == run.path
