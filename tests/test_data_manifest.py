"""C2 manifest construction and the guard that keeps a partial tensor off the
C1 path.

WHAT STANDS BEHIND THIS. Every fire in the corpus carries a manifest, and the
manifest is where the split lives: ``cv_fold``, ``spatial_block_id``,
``fuel_vintage_lag_years``, ``n_ignition_components``, and the whole per-fire QA
report under ``provenance.qa``. ``data/assemble.py`` was at 39% line coverage
and none of the four refusals below were exercised.

THE DESIGN DECISION THIS FILE PROTECTS, because it is the one that would be
undone first by someone trying to make a build succeed: **the four C2 keys have
no defaults.** A default of ``1`` for ``n_ignition_components`` is silently wrong
on exactly the fires the field exists for, and those are the fires where a
multi-ignition filing artifact would otherwise be fed into the spot-event
inventory as if a fire had thrown embers 47 km. The builder raises instead. Each
raise is asserted separately here, because four checks collapsed into one
``if any(... is None)`` would still pass a test that only ever omits one key.

THE SECOND DESIGN DECISION: **structured provenance survives as JSON objects.**
The canonical manifest builder stringifies provenance values, which is right for
``{source: pull-date}`` and destroys a QA report. ``assemble`` re-attaches the
four structured entries afterwards. If that re-attachment is lost, every
manifest in the corpus carries ``provenance.qa`` as the repr of a dict: still
present, still non-empty, unreadable by any consumer, and no contract clause
looks inside it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildfire_nowcast.common.contract import STATIC_CHANNELS
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.data.assemble import (
    C1_CHANNELS,
    ChannelBundle,
    MissingChannelsError,
    build_manifest,
    write_fire_tensor,
)
from wildfire_nowcast.data.backfill import _insert_after, fuel_vintage_lag_years

N_HOURS = 3
NY = NX = 4


def _grid() -> Grid:
    return Grid(x_min=0.0, y_max=float(NY) * 1000.0, nx=NX, ny=NY, cell_size_m=1000.0)


def _times() -> pd.DatetimeIndex:
    return pd.date_range("2020-09-05T00:00:00", periods=N_HOURS, freq="h")


def _bundle(**kwargs) -> ChannelBundle:
    return ChannelBundle(fire_id="synthetic_fire", grid=_grid(), times=_times(), **kwargs)


def _fill(bundle: ChannelBundle) -> ChannelBundle:
    for name in C1_CHANNELS:
        if name in bundle.channels:
            continue
        shape = (NY, NX) if name in STATIC_CHANNELS else (N_HOURS, NY, NX)
        bundle.add(name, np.zeros(shape, dtype=np.float32))
    return bundle


def _ready() -> ChannelBundle:
    return _fill(
        _bundle(
            cv_fold=1,
            spatial_block_id=2,
            fuel_vintage_lag_years=4,
            n_ignition_components=1,
            ignition_time_utc="2020-09-05T00:00:00",
            gofer_version="v1.0",
        )
    )


# --------------------------------------------------------------------------
# staging: shapes and channel names
# --------------------------------------------------------------------------


def test_a_channel_name_outside_the_C1_order_is_rejected_at_staging() -> None:
    """FAILS WHEN: ``add`` stops consulting the contract's channel tuple, at
    which point a typo produces a fourteen-channel bundle that is complete by
    count and missing the channel it meant to carry."""
    bundle = _bundle()
    with pytest.raises(KeyError, match="not a C1 channel"):
        bundle.add("wind_u_10", np.zeros((N_HOURS, NY, NX)))
    assert "wind_u_10" not in bundle.channels


def test_a_dynamic_channel_supplied_as_a_single_frame_is_rejected() -> None:
    """The one shape error that would otherwise broadcast silently.

    FAILS WHEN: the dynamic branch accepts the static shape as well, which lets
    a single RTMA hour stand in for a whole fire and produces a tensor whose
    weather never changes.
    """
    bundle = _bundle()
    with pytest.raises(ValueError, match=r"expected \(3, 4, 4\)"):
        bundle.add("wind_u10", np.zeros((NY, NX)))


def test_a_static_channel_is_accepted_in_either_the_two_or_three_D_form() -> None:
    """Both are legal by design, so the asymmetry with dynamic channels is pinned.

    FAILS WHEN: the static branch is tightened to 2-D only, which breaks every
    fire whose terrain was staged after being broadcast over time.
    """
    bundle = _bundle()
    bundle.add("elevation", np.zeros((NY, NX)))
    bundle.add("slope", np.zeros((N_HOURS, NY, NX)))
    assert {"elevation", "slope"} <= set(bundle.channels)
    with pytest.raises(ValueError):
        bundle.add("aspect_sin", np.zeros((NY + 1, NX)))


def test_completeness_is_measured_against_all_fourteen_channels() -> None:
    """FAILS WHEN: ``missing`` is computed from the channels present rather than
    from the contract order, so an empty bundle reports itself complete."""
    bundle = _bundle()
    assert len(bundle.missing) == len(C1_CHANNELS) == 14
    assert bundle.complete is False
    _fill(bundle)
    assert bundle.missing == []
    assert bundle.complete is True


# --------------------------------------------------------------------------
# the four keys that have no default
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("omitted", "message"),
    [
        ("cv_fold", "cv_fold must be assigned"),
        ("spatial_block_id", "spatial_block_id must be assigned"),
        ("fuel_vintage_lag_years", "fuel_vintage_lag_years must be assigned"),
        ("n_ignition_components", "n_ignition_components must be assigned"),
    ],
)
def test_each_required_C2_key_raises_on_its_own(omitted: str, message: str) -> None:
    """Four separate refusals, checked one at a time.

    Checked individually because a single combined guard would pass a test that
    only ever omits one key, and because the wrong one going missing has
    different consequences: a missing fold silently trains on the held-out fire,
    a missing component count silently promotes a filing artifact to a spot
    event.

    FAILS WHEN: any of the four acquires a default value, or the four raises are
    merged into one that names only the first missing key.
    """
    ready = {
        "cv_fold": 1,
        "spatial_block_id": 2,
        "fuel_vintage_lag_years": 4,
        "n_ignition_components": 1,
    }
    del ready[omitted]
    bundle = _fill(_bundle(**ready))
    with pytest.raises(ValueError, match=message):
        build_manifest(bundle, "norm_stats.json")


def test_a_fold_of_zero_is_a_real_value_and_is_not_treated_as_missing() -> None:
    """Fold 0 is a train fold in the split of record.

    FAILS WHEN: the guards test truthiness instead of ``is None``, which refuses
    fold 0, block 0 and a zero fuel lag, all of which are legal.
    """
    bundle = _fill(
        _bundle(
            cv_fold=0,
            spatial_block_id=0,
            fuel_vintage_lag_years=0,
            n_ignition_components=1,
        )
    )
    manifest = build_manifest(bundle, "norm_stats.json")
    assert manifest["cv_fold"] == 0
    assert manifest["spatial_block_id"] == 0
    assert manifest["fuel_vintage_lag_years"] == 0


# --------------------------------------------------------------------------
# structured provenance
# --------------------------------------------------------------------------


def test_the_qa_report_reaches_the_manifest_as_a_dict_not_as_its_repr() -> None:
    """The per-fire QA report is the point of C2 provenance.

    FAILS WHEN: the re-attachment loop over the structured keys is dropped. The
    manifest still validates, ``provenance.qa`` is still present and non-empty,
    and every number inside it has become a substring of a string.
    """
    bundle = _ready()
    bundle.qa = {"verdict": {"pass": True, "warnings": []}, "raster": {"teleport_steps": 2}}
    bundle.provenance = {
        "label_source": "gofer",
        "build_timings_s": {"labels": 1.5},
        "polygon_rasterization": {"method": "area_fraction"},
    }

    manifest = build_manifest(bundle, "norm_stats.json")
    provenance = manifest["provenance"]

    assert isinstance(provenance["qa"], dict)
    assert provenance["qa"]["raster"]["teleport_steps"] == 2
    assert isinstance(provenance["build_timings_s"], dict)
    assert provenance["build_timings_s"]["labels"] == 1.5
    assert isinstance(provenance["polygon_rasterization"], dict)
    assert isinstance(provenance["label_source"], str), "scalar provenance stays a string"

    round_tripped = json.loads(json.dumps(manifest))
    assert round_tripped["provenance"]["qa"]["verdict"]["pass"] is True


# --------------------------------------------------------------------------
# the ADR-003(b) door
# --------------------------------------------------------------------------


def test_a_partial_bundle_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    """The single door out of the interim area, and it must not leave debris.

    FAILS WHEN: the completeness check moves below the first write, so a refused
    fire leaves a truncated store on the C1 path that later reads as a built
    fire with missing channels.
    """
    bundle = _bundle(
        cv_fold=0, spatial_block_id=0, fuel_vintage_lag_years=1, n_ignition_components=1
    )
    bundle.add("fire_state", np.zeros((N_HOURS, NY, NX), dtype=np.float32))
    target = tmp_path / "partial" / "tensor.zarr"

    with pytest.raises(MissingChannelsError, match="refusing to write"):
        write_fire_tensor(
            bundle,
            norm_stats_path="norm_stats.json",
            tensor_path=target,
            manifest_path=tmp_path / "partial" / "manifest.json",
        )
    assert not target.exists()
    assert not (tmp_path / "partial" / "manifest.json").exists()


def test_a_complete_bundle_writes_a_store_that_declares_its_time_convention(
    tmp_path: Path,
) -> None:
    """End-of-hour is the phase convention the whole corpus is aligned on.

    FAILS WHEN: the writer stops stamping ``time_convention`` into the store
    attrs. The manifest still records it, so the fire looks documented, and a
    tensor handed to a model without its manifest carries no time base at all.
    """
    tensor_path, manifest_path = write_fire_tensor(
        _ready(),
        norm_stats_path="norm_stats.json",
        tensor_path=tmp_path / "fire" / "tensor.zarr",
        manifest_path=tmp_path / "fire" / "manifest.json",
    )
    assert tensor_path.exists() and manifest_path.exists()

    dataset = open_tensor(tensor_path)
    try:
        assert dataset.attrs["time_convention"] == "end_of_hour"
        assert set(dataset.data_vars) == {"features", "fire_state"}
    finally:
        dataset.close()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["fire_id"] == "synthetic_fire"
    assert manifest["n_hours"] == N_HOURS


# --------------------------------------------------------------------------
# the derived C2 key, and the direction of the leakage it guards
# --------------------------------------------------------------------------


def test_a_fuel_vintage_that_postdates_ignition_is_refused_as_label_leakage() -> None:
    """A LANDFIRE release taken after the fire has the fire's own scar in it.

    This is the sharpest of the three checks in the function, because a negative
    lag is not a data-quality nuisance: it means the fuels channel was told
    where the fire went.

    FAILS WHEN: the sign test is written as ``abs(lag)`` or dropped, at which
    point a post-fire vintage is recorded as a positive staleness and reads as
    conservative.
    """
    leaking = {
        "ignition_time_utc": "2014-08-16T00:00:00",
        "provenance": {"fuels_staleness_years": "-2", "fuels_vintage_year": 2016},
    }
    with pytest.raises(ValueError, match="postdates ignition"):
        fuel_vintage_lag_years(leaking)


def test_a_manifest_that_disagrees_with_itself_about_the_lag_is_refused() -> None:
    """Two numbers for one fact is worse than one missing number.

    FAILS WHEN: the cross-check against ``ignition year - vintage year`` is
    dropped and the declared staleness is trusted, which is how a stale-fuels
    correction gets sized against a lag nobody re-derived.
    """
    consistent = {
        "ignition_time_utc": "2020-08-16T00:00:00",
        "provenance": {"fuels_staleness_years": "4", "fuels_vintage_year": 2016},
    }
    assert fuel_vintage_lag_years(consistent) == 4

    contradictory = {
        "ignition_time_utc": "2020-08-16T00:00:00",
        "provenance": {"fuels_staleness_years": "3", "fuels_vintage_year": 2016},
    }
    with pytest.raises(ValueError, match="contradicts ignition year"):
        fuel_vintage_lag_years(contradictory)

    with pytest.raises(ValueError, match="no provenance dict"):
        fuel_vintage_lag_years({"ignition_time_utc": "2020-08-16T00:00:00"})
    with pytest.raises(ValueError, match="fuels_staleness_years is absent"):
        fuel_vintage_lag_years({"provenance": {}})


def test_backfilled_keys_land_in_their_declared_position_and_nothing_is_dropped() -> None:
    """Key order in a manifest is public surface; a diff of it should be readable.

    FAILS WHEN: the insertion loop drops keys that are not the anchor, or appends
    silently when the anchor is absent instead of keeping the addition at all.
    """
    manifest = {"fire_id": "x", "cv_fold": 1, "spatial_block_id": 2, "n_hours": 5}
    out = _insert_after(manifest, {"fuel_vintage_lag_years": 4})
    assert list(out) == [
        "fire_id",
        "cv_fold",
        "spatial_block_id",
        "fuel_vintage_lag_years",
        "n_hours",
    ]

    without_anchor = _insert_after({"fire_id": "x", "n_hours": 5}, {"n_ignition_components": 1})
    assert without_anchor == {"fire_id": "x", "n_hours": 5, "n_ignition_components": 1}
