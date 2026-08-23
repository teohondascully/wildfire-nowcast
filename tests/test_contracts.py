"""C1/C2/C3 contract tests at INTERFACES **v2.8**, runnable against ANY tensor.

Two halves:

1. **Conformance** - the store under test (``--tensor-path``, default: a fresh
   synthetic fire) must satisfy every clause of C1/C2/C3.
2. **Teeth** - deliberately corrupted datasets must make the checker *fail*, and
   fail on the specific clause that was broken. A contract test that cannot fail
   proves nothing, so these are as important as the conformance tests.

Point the whole suite at the real fire::

    .venv/bin/pytest tests/test_contracts.py \\
        --tensor-path data/fires/2019_kincade/tensor.zarr
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wildfire_nowcast.common import contract as C
from wildfire_nowcast.common import zarr_io as zio

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _failed_ids(report: C.ContractReport) -> set[str]:
    """Ids of clauses that failed *hard*."""
    return {c.check_id for c in report.failures}


def _reporting_ids(report: C.ContractReport) -> set[str]:
    """Ids of clauses that are reporting gates (plumbing-ok, gate-blocking)."""
    return {c.check_id for c in report.reporting_gaps}


def _with_channel(ds: xr.Dataset, name: str, values: np.ndarray) -> xr.Dataset:
    """Copy of ``ds`` with one feature channel overwritten."""
    arr = np.asarray(ds[C.FEATURES].values).copy()
    arr[:, C.feature_index(name)] = values
    out = ds.copy()
    out[C.FEATURES] = (C.FEATURES_DIMS, arr, dict(ds[C.FEATURES].attrs))
    return out


def _with_fire_state(ds: xr.Dataset, values: np.ndarray) -> xr.Dataset:
    out = ds.copy()
    out[C.FIRE_STATE] = (C.FIRE_STATE_DIMS, values.astype(np.uint8))
    return out


# --------------------------------------------------------------------------
# C1 - per-fire tensor store
# --------------------------------------------------------------------------


def test_c1_full_report_passes(tensor_path: Path, labels_only: bool) -> None:
    required = [C.FIRE_STATE] if labels_only else C.CHANNELS
    report = C.check_tensor(
        tensor_path, required_channels=required, require_channel_coord=not labels_only
    )
    assert report.ok, "\n" + report.format(verbose=True)


def test_c1_is_the_v2_two_variable_layout(tensor_ds: xr.Dataset, labels_only: bool) -> None:
    """One zarr array holds one dtype, so C1's uint8 label and float32 features
    cannot share an array (ADR-006 P2). fire_state is (time,y,x) uint8;
    features is (time,channel,y,x) float32 carrying channels 1-13."""
    assert tuple(tensor_ds[C.FIRE_STATE].dims) == ("time", "y", "x")
    assert tensor_ds[C.FIRE_STATE].dtype == np.uint8
    if labels_only:
        return
    assert tuple(tensor_ds[C.FEATURES].dims) == ("time", "channel", "y", "x")
    assert tensor_ds[C.FEATURES].dtype == np.float32
    assert int(tensor_ds.sizes["channel"]) == C.N_FEATURE_CHANNELS == 13
    assert set(tensor_ds.data_vars) - C.ALLOWED_EXTRA_VARS == set(C.DATA_VARS)

    stacked = zio.to_stacked_dataarray(tensor_ds)
    assert stacked.dims == C.TENSOR_DIMS
    assert stacked.shape[1] == C.N_CHANNELS == 14


def test_c1_channel_order_preserves_the_v1_indices(
    tensor_ds: xr.Dataset, labels_only: bool
) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores are not channel-complete by design")
    on_disk = [str(c) for c in tensor_ds["channel"].values]
    assert on_disk == list(C.FEATURE_CHANNELS)
    # position along `features` + channel_index_offset == the v1 channel index
    offset = C._attr(tensor_ds, C.ATTR_CHANNEL_INDEX_OFFSET, var=C.FEATURES)
    assert int(offset) == C.CHANNEL_INDEX_OFFSET == 1
    for position, name in enumerate(on_disk):
        assert position + int(offset) == C.CHANNEL_INDEX[name]
    assert C.CHANNEL_INDEX["fire_state"] == 0
    assert C.CHANNEL_INDEX["water_barrier_mask"] == 12
    assert C.feature_index("water_barrier_mask") == 11


def test_c1_channels_are_reachable_by_name_not_by_integer(
    tensor_ds: xr.Dataset, labels_only: bool
) -> None:
    """No consumer should ever hardcode a features index."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no features array")
    for name in C.FEATURE_CHANNELS:
        da = zio.get_channel(tensor_ds, name)
        assert tuple(da.dims) == ("time", "y", "x"), name
        assert da.dtype == np.float32, name
    assert zio.get_channel(tensor_ds, C.FIRE_STATE).dtype == np.uint8


def test_c1_crs_is_epsg_5070(tensor_ds: xr.Dataset) -> None:
    assert C._epsg_of(tensor_ds.attrs.get("crs")) == 5070


def test_c1_cells_are_1000_m_and_on_the_continental_lattice(tensor_ds: xr.Dataset) -> None:
    x = np.asarray(tensor_ds["x"].values, dtype=np.float64)
    y = np.asarray(tensor_ds["y"].values, dtype=np.float64)
    assert np.allclose(np.diff(x), 1000.0, atol=1e-6), "x must ascend in exact 1 km steps"
    assert np.allclose(np.diff(y), -1000.0, atol=1e-6), "y must descend in exact 1 km steps"
    assert float(tensor_ds.attrs["cell_size_m"]) == 1000.0
    # C1.2: cell (i, j) must denote the same ground in every fire.
    assert C.is_on_lattice(x) and C.is_on_lattice(y)


def test_c1_time_is_hourly_monotone_naive_utc(tensor_ds: xr.Dataset) -> None:
    t = np.asarray(tensor_ds["time"].values)
    assert np.issubdtype(t.dtype, np.datetime64)
    diffs = np.diff(t)
    assert np.all(diffs > np.timedelta64(0, "s")), "time must be strictly increasing"
    assert np.all(diffs == np.timedelta64(1, "h")), "every step must be exactly 1 h"
    for attr in (C.ATTR_TIME_START, C.ATTR_TIME_END):
        value = tensor_ds.attrs[attr]
        assert isinstance(value, str) and not value.endswith("Z") and "+" not in value


def test_c1_3_time_convention_is_recorded(tensor_ds: xr.Dataset, tensor_path: Path) -> None:
    """C1.3 - GOFER tUTC is end-of-hour and RTMA is lagged to match. Getting it
    wrong trains every fire an hour out of phase and presents as a mediocre
    model, not as a bug, so the convention must be written down somewhere
    machine-readable."""
    recorded = tensor_ds.attrs.get(C.ATTR_TIME_CONVENTION)
    if recorded is None:
        manifest = tensor_path.parent / "manifest.json"
        if manifest.is_file():
            provenance = json.loads(manifest.read_text()).get("provenance", {})
            recorded = provenance.get(C.ATTR_TIME_CONVENTION)
    assert recorded == C.TIME_CONVENTION


def test_c1_1_fire_state_guarantees(tensor_ds: xr.Dataset) -> None:
    """The C1.1 guarantees: domain, absorbing, no 0 -> 2 skip, and therefore one
    contiguous burning run per cell."""
    state = np.asarray(tensor_ds[C.FIRE_STATE].values)
    assert C.fire_state_violations(state) == []
    assert set(np.unique(state)).issubset({0, 1, 2})
    assert np.all(state[1:] >= state[:-1]), "fire is absorbing: state must never decrease"
    assert not ((state[:-1] == C.UNBURNED) & (state[1:] == C.BURNED_OUT)).any()


def test_c1_1_empty_burning_frames_are_not_a_violation(tensor_ds: xr.Dataset) -> None:
    """C1.1: state 1 is legitimately EMPTY in 6-37% of real frames, because
    after a long dormancy every cell is closed. A consumer must therefore treat
    the frontier of the burned region, not state 1 alone, as the contagion
    source - this test exists to stop anyone re-adding a "burning cells in every
    frame" assertion to the contract."""
    state = np.asarray(tensor_ds[C.FIRE_STATE].values)
    empty = ~(state == C.BURNING).any(axis=(1, 2))
    assert C.fire_state_violations(state) == [], (
        f"{int(empty.sum())}/{empty.size} frames have no cell in state 1, which C1.1 declares "
        "legal; nothing about them may fail the contract"
    )
    assert (state == C.BURNING).any(), "no frame anywhere has a burning cell"


def test_c1_static_channels_do_not_vary_in_time(tensor_ds: xr.Dataset, labels_only: bool) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no features array")
    for name in sorted(C.STATIC_CHANNELS):
        values = zio.channel_values(tensor_ds, name)
        assert np.array_equal(values[0], values[-1]), f"{name} is declared static by C1"


def test_c1_5_features_are_finite(tensor_ds: xr.Dataset, labels_only: bool) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no features array")
    values = np.asarray(tensor_ds[C.FEATURES].values)
    assert np.isfinite(values).all(), "C1.5: `features` must contain no NaN/inf"


def test_c1_7_physical_ranges_hold(tensor_ds: xr.Dataset, labels_only: bool) -> None:
    """C1.7 on the store under test, asserted directly rather than via the report."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no features array")
    for name, (low, high) in C.PHYSICAL_RANGES.items():
        values = zio.channel_values(tensor_ds, name)
        assert float(values.min()) >= low and float(values.max()) <= high, (
            f"C1.7: {name} must lie in [{low}, {high}]; observed "
            f"[{float(values.min())}, {float(values.max())}]"
        )
    fuels = np.unique(zio.channel_values(tensor_ds, "fuel_model_id"))
    illegal = sorted(float(v) for v in fuels if int(v) not in C.FBFM40_CLASSES)
    assert not illegal, f"C1.7: not FBFM40 classes: {illegal}"


# --------------------------------------------------------------------------
# C2 - per-fire manifest
# --------------------------------------------------------------------------


def test_c2_manifest(manifest_path: Path, tensor_ds: xr.Dataset, labels_only: bool) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    report = C.check_manifest(manifest_path, ds=tensor_ds)
    assert report.ok, "\n" + report.format(verbose=True)


def test_c2_keys_exactly(manifest_path: Path, labels_only: bool) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    for key in C.MANIFEST_KEYS:
        assert key in manifest, f"C2 requires key {key!r}"
    assert isinstance(manifest["spatial_block_id"], int)


def test_c1_2_buffer_margin_holds(tensor_ds: xr.Dataset) -> None:
    """C1.2's 10 km buffer, asserted on the store under test.

    Ratified at v2 and unimplemented until A10: the checker enforced the lattice
    snap and the cell size - the other two sentences of C1.2 - and never looked
    at the buffer.
    """
    rep = C.check_tensor(tensor_ds)
    assert "buffer_margin" not in _failed_ids(rep), "\n" + rep.format()


def test_rejects_a_domain_that_clips_the_fire(synthetic_ds: xr.Dataset) -> None:
    """A clipped domain passes every other clause.

    The fire simply stops growing at the boundary, which reads as a mediocre
    model rather than as truncated data - the C1.3 failure mode in a different
    dress.
    """
    state = np.asarray(synthetic_ds[C.FIRE_STATE].values).copy()
    state[:, :, -3:] = C.BURNING  # footprint now 0 cells from the east edge
    state = np.maximum.accumulate(state, axis=0)
    broken = _with_fire_state(synthetic_ds, state)
    rep = C.check_tensor(broken)
    assert "buffer_margin" in _failed_ids(rep)
    assert "10 km" in rep.format()


def test_buffer_margin_is_unverifiable_not_vacuous_when_nothing_burned(
    synthetic_ds: xr.Dataset,
) -> None:
    """C-1: no final perimeter means the clause cannot be evaluated, not that it passed."""
    empty = np.zeros_like(np.asarray(synthetic_ds[C.FIRE_STATE].values))
    assert "buffer_margin" in _failed_ids(C.check_tensor(_with_fire_state(synthetic_ds, empty)))


@pytest.mark.parametrize("key", [C.MANIFEST_VINTAGE_LAG_KEY, C.MANIFEST_IGNITION_COMPONENTS_KEY])
def test_c2_v27_keys_are_present_and_int(manifest_path: Path, labels_only: bool, key: str) -> None:
    """C2 [v2.7], ADR-014. ``n_ignition_components`` is the one that bites: GOFER
    files separate lightning ignitions under one fire id (July Complex 2, SCU 2 -
    ADR-019 corrected SCU from 3, and corrected the ESTIMAND, since the old count
    used final-footprint components which cannot see a merge), so an undeclared
    multi-ignition fire feeds a 46 km filing artifact into P3 crossings mining as
    if it were spotting."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    value = manifest.get(key, manifest.get("provenance", {}).get(key))
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"C2 [v2.7] requires {key} as an int (root or provenance); got {value!r}"
    )


def test_c2_ignition_components_reproduces_the_ratified_deriver(
    manifest_path: Path, tensor_ds: xr.Dataset, labels_only: bool
) -> None:
    """C2 [v2.7] says DERIVED, not defaulted - so re-derive it and compare.

    ENFORCEMENT OF AN EXISTING CLAUSE, NOT A NEW ONE. "DERIVED" was previously
    unchecked: a manifest carrying a hand-typed integer satisfied every check
    there was, because a stored int cannot be told from a computed one by
    reading it. This re-runs the RATIFIED rule
    (``data.ignitions.count_ignition_components``, ADR-019 - the same estimand
    ``sim/components.py`` delegates to since S13) against the ``fire_state``
    field of the store the manifest describes, and requires the two to agree.

    Verified before it was added, so it cannot surprise anyone: it reproduces
    the stored integer on **21 of 21 corpus fires** (I22, re-measured; D19
    measured the same 21/21 independently) and on the synthetic fixture, whose
    literal ``1`` it would have caught - the deriver reads 2 there, because the
    scripted spot never merges and lands ~30 km out (I22).
    """
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    from wildfire_nowcast.data.ignitions import count_ignition_components

    manifest = json.loads(manifest_path.read_text())
    stored = manifest.get(
        C.MANIFEST_IGNITION_COMPONENTS_KEY,
        manifest.get("provenance", {}).get(C.MANIFEST_IGNITION_COMPONENTS_KEY),
    )
    derived = count_ignition_components(
        zio.fire_state_of(tensor_ds), cell_size_m=float(tensor_ds.attrs[C.ATTR_CELL_SIZE])
    ).n_ignition_components
    assert stored == derived, (
        f"C2 [v2.7] {C.MANIFEST_IGNITION_COMPONENTS_KEY} = {stored!r} in the manifest but the "
        f"ratified deriver reads {derived} on this very tensor. C2 says DERIVED, not defaulted: "
        "a producer may not assert a number it did not compute. If the RULE is what is wrong "
        "here, that is data.ignitions' owner's call and belongs in a BLOCKER, not in a manifest"
    )


def _manifest_missing(manifest_path: Path, tmp_path: Path, key: str) -> Path:
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(key, None)
    if isinstance(manifest.get("provenance"), dict):
        manifest["provenance"].pop(key, None)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("key", [C.MANIFEST_VINTAGE_LAG_KEY, C.MANIFEST_IGNITION_COMPONENTS_KEY])
def test_rejects_a_manifest_without_the_v27_keys(
    manifest_path: Path, tmp_path: Path, labels_only: bool, key: str
) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    assert key in _failed_ids(C.check_manifest(_manifest_missing(manifest_path, tmp_path, key)))


def test_a_numeric_string_is_not_a_machine_readable_int(
    manifest_path: Path, tmp_path: Path, labels_only: bool
) -> None:
    """The manifests already carried ``fuels_staleness_years: "3"``. ADR-014
    asked for machine-readable precisely because a consumer otherwise has to
    guess the type, and the guess is where the next silent defect lives."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    manifest[C.MANIFEST_VINTAGE_LAG_KEY] = "3"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert C.MANIFEST_VINTAGE_LAG_KEY in _failed_ids(C.check_manifest(path))


def test_the_v27_keys_may_live_at_the_root_or_in_provenance(
    manifest_path: Path, tmp_path: Path, labels_only: bool
) -> None:
    """INTERFACES lists them under C2 "Keys"; ADR-014 says ``provenance``. The
    contract is ambiguous about LOCATION, so the checker enforces the invariant
    and not a spelling it never ratified (same precedent as
    ``channel_index_offset``, accepted on the root or on ``features``)."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    for key, value in ((C.MANIFEST_VINTAGE_LAG_KEY, 4), (C.MANIFEST_IGNITION_COMPONENTS_KEY, 2)):
        manifest.pop(key, None)
        manifest.setdefault("provenance", {})[key] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    report = C.check_manifest(path)
    assert C.MANIFEST_VINTAGE_LAG_KEY not in _failed_ids(report)
    assert C.MANIFEST_IGNITION_COMPONENTS_KEY not in _failed_ids(report)


def test_a_vintage_lag_that_contradicts_its_own_provenance_fails(
    manifest_path: Path, tmp_path: Path, labels_only: bool
) -> None:
    """Two numbers for one fact is worse than one missing number."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    manifest["ignition_time_utc"] = "2021-08-14T21:00:00"
    manifest.setdefault("provenance", {})["fuels_vintage_year"] = "2016"
    manifest[C.MANIFEST_VINTAGE_LAG_KEY] = 1  # should be 5
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert "fuel_vintage_lag_consistent" in _failed_ids(C.check_manifest(path))


@pytest.mark.parametrize("fact_keys", list(C.PROVENANCE_REQUIRED_FACTS.values()))
def test_rejects_provenance_missing_a_required_fact(
    manifest_path: Path, tmp_path: Path, labels_only: bool, fact_keys: tuple[str, ...]
) -> None:
    """C2 [v2]: provenance MUST record the LANDFIRE vintage, state rule and fconf.

    Ratified at v2, unimplemented until A10 - the checker asserted only that
    ``provenance`` was a non-empty dict. None of the three facts is recoverable
    from the tensor, and a tensor built under the RETIRED provisional state rule
    is indistinguishable from a conformant one by inspection.
    """
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    for key in fact_keys:
        manifest.get("provenance", {}).pop(key, None)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert "provenance_declares_sources" in _failed_ids(C.check_manifest(path))


def test_c2_permits_extra_keys(manifest_path: Path, tmp_path: Path, labels_only: bool) -> None:
    """v2.1 states the superset rule explicitly; synthetic and data both
    rely on it, and it should be tested rather than assumed."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    manifest["some_lead_specific_key"] = {"nested": True}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    report = C.check_manifest(path)
    assert report.ok, "\n" + report.format(verbose=True)


# --------------------------------------------------------------------------
# C3 - normalization stats
# --------------------------------------------------------------------------


def test_c3_norm_stats_shape(norm_stats_path: Path, labels_only: bool) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C3 stats")
    report = C.check_norm_stats(norm_stats_path)
    assert report.ok, "\n" + report.format(verbose=True)

    stats = json.loads(norm_stats_path.read_text())
    # C3.2: the canonical shape is TOP-LEVEL dicts, not a nested block.
    assert stats["channel_order"] == list(C.CHANNELS)
    assert set(stats["mean"]) == set(C.CHANNELS)
    assert set(stats["std"]) == set(C.CHANNELS)
    assert all(v > 0 for v in stats["std"].values())
    # C3.2: categorical channels take the identity transform.
    for name in C.CATEGORICAL_CHANNELS:
        assert stats["mean"][name] == 0.0 and stats["std"][name] == 1.0
    assert stats[C.NORM_STATS_CATEGORICAL_NOTE]
    # C3.3: the bootstrap guard must be evaluable at all.
    assert isinstance(stats["n_train_blocks"], int) and stats["n_train_blocks"] >= 1


def test_c3_4_rejects_a_train_mean_outside_a_definitional_range(
    norm_stats_path: Path, tmp_path: Path, labels_only: bool
) -> None:
    """C3.4 - the shared file is checked as its own artifact.

    Measured case (ADR-010): CZU's ``-9999`` is 33.1% of train cell-hours and
    would have moved the TRAIN mean ``canopy_cover`` to **-492.13%** from
    27.94%, corrupting the normalisation of two held-out fires containing no
    NoData at all. No per-fire report can see that, because the poisoned fires
    are clean.
    """
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C3 stats")
    stats = json.loads(norm_stats_path.read_text())
    stats["mean"]["canopy_cover"] = -492.13
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(stats))
    report = C.check_norm_stats(path)
    assert "mean_within_physical_range" in _failed_ids(report)
    assert "-492" in report.format()


def test_c3_4_passes_on_the_shipped_stats(norm_stats_path: Path, labels_only: bool) -> None:
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C3 stats")
    assert "mean_within_physical_range" not in _failed_ids(C.check_norm_stats(norm_stats_path))


# --------------------------------------------------------------------------
# The standalone entry point data uses
# --------------------------------------------------------------------------


def test_contract_cli_passes_on_conformant_store(tensor_path: Path, labels_only: bool) -> None:
    argv = [str(tensor_path)] + (["--labels-only"] if labels_only else [])
    assert C.main(argv) == 0


def test_contract_cli_is_runnable_as_a_module(tensor_path: Path, labels_only: bool) -> None:
    """`python -m wildfire_nowcast.common.contract PATH` must work standalone."""
    argv = [sys.executable, "-m", "wildfire_nowcast.common.contract", str(tensor_path), "--json"]
    if labels_only:
        argv.append("--labels-only")
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["n_checks"] > 0
    assert payload["enforces_interfaces_version"] == C.CONTRACT_VERSION
    assert "reporting_ready" in payload


def test_contract_cli_fails_on_missing_store(tmp_path: Path) -> None:
    assert C.main([str(tmp_path / "nope.zarr"), "--skip-manifest", "--skip-norm-stats"]) == 1


def test_contract_version_is_printed_on_every_report() -> None:
    """A checker one version behind the contract fails conformant data and
    passes stale data, so the version it enforces is on every report.

    **The equality itself is asserted ONCE, in
    ``tests/test_clause_registry.py::test_contract_version_matches_interfaces``,
    which PARSES INTERFACES.md.** A hardcoded literal used to live here too, and
    a second place to update is a second place to forget - which is how a bump
    to v2.9 left the code on v2.8. One derived check beats two pinned ones.
    """
    report = C.ContractReport(target="x")
    report.add("C1", "demo", True, "ok")
    assert C.CONTRACT_VERSION in report.format()
    assert report.to_dict()["enforces_interfaces_version"] == C.CONTRACT_VERSION


def test_the_checker_declares_the_clauses_it_does_not_enforce() -> None:
    """The version string alone can lie, so the gap must be machine-readable.

    A checker printing "enforcing v2.8" while skipping a v2.3 clause is the
    stale-checker hazard wearing a current label - worse than an honestly old
    checker, because nobody thinks to look. This is C-1's corollary turned on
    the checker: declaring a weakness is a gate, omitting it is a failure.
    """
    assert isinstance(C.DEFERRED_CLAUSES, dict)
    for clause, why in C.DEFERRED_CLAUSES.items():
        assert clause.startswith("C"), clause
        assert len(why) > 40, f"{clause} is deferred without a stated reason"

    report = C.ContractReport(target="x")
    report.add("C1", "demo", True, "ok")
    text = report.format()
    payload = report.to_dict()
    for clause in C.DEFERRED_CLAUSES:
        assert clause in text, "every report must print what it does NOT check"
        assert clause in payload["deferred_clauses"]


# --------------------------------------------------------------------------
# ADR-003 interim label stores
# --------------------------------------------------------------------------


def _plain_xarray_label_store(path: Path, *, hourly: bool = True) -> Path:
    """An ADR-003 interim label store, written WITHOUT our writer.

    The checker must judge a store by what is on disk, not by which code wrote
    it, so this deliberately uses bare xarray and skips consolidated metadata.
    """
    n_t, ny, nx = 6, 40, 50
    x = -2_100_000.0 + (np.arange(nx) + 0.5) * 1000.0
    y = 2_050_000.0 - (np.arange(ny) + 0.5) * 1000.0
    step = np.timedelta64(1 if hourly else 3, "h")
    time = np.datetime64("2019-10-24T02:00:00", "s") + np.arange(n_t) * step
    state = np.zeros((n_t, ny, nx), dtype=np.uint8)
    for i in range(n_t):
        state[i, 18 : 20 + i, 20 : 22 + i] = C.BURNING
        if i:
            state[i][state[i - 1] > 0] = C.BURNED_OUT
    ds = xr.Dataset(
        {"fire_state": (("time", "y", "x"), state)},
        coords={"time": time, "y": y, "x": x},
        attrs={
            "crs": "EPSG:5070",
            "cell_size_m": 1000.0,
            C.ATTR_TIME_CONVENTION: C.TIME_CONVENTION,
            C.ATTR_TIME_START: str(time[0]),
            C.ATTR_TIME_END: str(time[-1]),
        },
    )
    ds.to_zarr(path, mode="w", consolidated=False)
    return path


def test_labels_only_mode_accepts_a_conformant_interim_store(tmp_path: Path) -> None:
    """ADR-003 interim label stores are checkable without being C1-complete."""
    store = _plain_xarray_label_store(tmp_path / "fire_state.zarr")
    report = C.check_tensor(store, required_channels=[C.FIRE_STATE], require_channel_coord=False)
    assert report.ok, "\n" + report.format(verbose=True)
    assert C.main([str(store), "--labels-only"]) == 0


def test_labels_only_mode_still_enforces_the_grid_and_time_rules(tmp_path: Path) -> None:
    """Relaxing *completeness* must not relax anything else."""
    store = _plain_xarray_label_store(tmp_path / "three_hourly.zarr", hourly=False)
    report = C.check_tensor(store, required_channels=[C.FIRE_STATE], require_channel_coord=False)
    assert "time_hourly" in _failed_ids(report)
    assert C.main([str(store), "--labels-only"]) == 1


def test_labels_only_mode_rejects_an_incomplete_store_at_the_c1_path(tmp_path: Path) -> None:
    """The same store fails full C1: interim != C1 (ADR-003)."""
    store = _plain_xarray_label_store(tmp_path / "fire_state.zarr")
    assert "variables_present" in _failed_ids(C.check_tensor(store))


# --------------------------------------------------------------------------
# Teeth: corrupted stores must be rejected, on the right clause
# --------------------------------------------------------------------------


def test_rejects_wrong_features_dtype(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.copy()
    broken[C.FEATURES] = broken[C.FEATURES].astype(np.float64)
    assert "dtypes" in _failed_ids(C.check_tensor(broken))


def test_rejects_float_fire_state(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.copy()
    broken[C.FIRE_STATE] = broken[C.FIRE_STATE].astype(np.float32)
    assert "dtypes" in _failed_ids(C.check_tensor(broken))


def test_rejects_missing_features_array(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.drop_vars(C.FEATURES)
    assert "variables_present" in _failed_ids(C.check_tensor(broken))


def test_rejects_the_retired_v1_one_variable_per_channel_layout(synthetic_ds: xr.Dataset) -> None:
    """A v1-layout store must be rejected with a message that says so, rather
    than confusing a lead who is looking at correct-looking data."""
    v1 = zio.to_channel_dataset(synthetic_ds)
    report = C.check_tensor(v1)
    failed = _failed_ids(report)
    assert "variables_present" in failed and "no_unknown_variables" in failed
    message = report.format(verbose=False)
    assert "v1" in message and "features" in message


def test_rejects_dropped_channel(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.isel(channel=slice(0, C.N_FEATURE_CHANNELS - 1))
    assert "n_channels" in _failed_ids(C.check_tensor(broken))


def test_rejects_extra_data_variable(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.copy()
    broken["ndvi"] = broken[C.FIRE_STATE].astype(np.float32)
    assert "no_unknown_variables" in _failed_ids(C.check_tensor(broken))


def test_rejects_reordered_channel_coord(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.copy()
    shuffled = list(C.FEATURE_CHANNELS)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    broken = broken.assign_coords(channel=np.array(shuffled, dtype="<U24"))
    assert "channel_coord" in _failed_ids(C.check_tensor(broken))


def _drop_offset(ds: xr.Dataset, *, root: bool, on_features: bool) -> xr.Dataset:
    out = ds.copy()
    if root:
        out.attrs.pop(C.ATTR_CHANNEL_INDEX_OFFSET, None)
    if on_features:
        features = out[C.FEATURES].copy()
        features.attrs = {
            k: v for k, v in features.attrs.items() if k != C.ATTR_CHANNEL_INDEX_OFFSET
        }
        out[C.FEATURES] = features
    return out


def test_rejects_channel_coord_that_still_lists_fire_state(synthetic_ds: xr.Dataset) -> None:
    """The commonest v1 -> v2 migration slip: stacking all 14 channels into
    `features`. The message must name it instead of printing a bare list."""
    features = np.asarray(synthetic_ds[C.FEATURES].values)
    labels = np.asarray(synthetic_ds[C.FIRE_STATE].values, dtype=np.float32)[:, None]
    stacked = np.concatenate([labels, features], axis=1)
    broken = synthetic_ds.drop_dims("channel").assign(
        **{C.FEATURES: (C.FEATURES_DIMS, stacked, dict(synthetic_ds[C.FEATURES].attrs))}
    )
    broken = broken.assign_coords(channel=np.array(list(C.CHANNELS), dtype="<U24"))
    report = C.check_tensor(broken)
    assert "channel_coord" in _failed_ids(report)
    assert "fire_state is its own uint8 variable" in report.format()


def test_rejects_missing_or_wrong_channel_index_offset(synthetic_ds: xr.Dataset) -> None:
    for value in (None, 0, 2):
        broken = _drop_offset(synthetic_ds, root=True, on_features=True)
        if value is not None:
            features = broken[C.FEATURES].copy()
            features.attrs[C.ATTR_CHANNEL_INDEX_OFFSET] = value
            broken[C.FEATURES] = features
        assert "channel_index_offset" in _failed_ids(C.check_tensor(broken)), value


def test_accepts_channel_index_offset_on_either_the_root_or_features(
    synthetic_ds: xr.Dataset,
) -> None:
    """C1 says the STORE records it and does not legislate where; data hung
    it on `features`, which is at least as defensible as the root."""
    on_features = _drop_offset(synthetic_ds, root=True, on_features=False)
    assert "channel_index_offset" not in _failed_ids(C.check_tensor(on_features))

    on_root = _drop_offset(synthetic_ds, root=False, on_features=True)
    assert "channel_index_offset" not in _failed_ids(C.check_tensor(on_root))


def test_rejects_wrong_crs(synthetic_ds: xr.Dataset) -> None:
    for wrong in ("EPSG:4326", "EPSG:3857", None):
        broken = synthetic_ds.copy()
        broken.attrs["crs"] = wrong
        assert "crs" in _failed_ids(C.check_tensor(broken)), wrong


def test_rejects_non_1km_cells(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.copy()
    broken = broken.assign_coords(x=np.asarray(broken["x"].values) * 0.5)
    assert "x_coord" in _failed_ids(C.check_tensor(broken))


def test_rejects_a_grid_off_the_continental_lattice(synthetic_ds: xr.Dataset) -> None:
    """C1.2 - shift the whole grid by 300 m and cell (i, j) stops meaning the
    same ground in every fire, which silently breaks the C3.1 spatial blocking
    while every other check still passes."""
    broken = synthetic_ds.copy()
    broken = broken.assign_coords(x=np.asarray(broken["x"].values) + 300.0)
    failed = _failed_ids(C.check_tensor(broken))
    assert "lattice_snap" in failed
    assert "x_coord" not in failed, "spacing is still 1 km; only the offset is wrong"


def test_rejects_south_up_y_axis(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.isel(y=slice(None, None, -1))
    assert "y_coord" in _failed_ids(C.check_tensor(broken))


def test_rejects_non_hourly_time(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.isel(time=[0, 1, 3, 5])
    assert "time_hourly" in _failed_ids(C.check_tensor(broken))


def test_rejects_non_monotone_time(synthetic_ds: xr.Dataset) -> None:
    order = list(range(int(synthetic_ds.sizes["time"])))
    order[2], order[3] = order[3], order[2]
    broken = synthetic_ds.isel(time=order)
    assert "time_monotone" in _failed_ids(C.check_tensor(broken))


def test_rejects_timezone_aware_iso_attrs(synthetic_ds: xr.Dataset) -> None:
    broken = synthetic_ds.copy()
    broken.attrs[C.ATTR_TIME_START] = broken.attrs[C.ATTR_TIME_START] + "Z"
    assert f"attr_{C.ATTR_TIME_START}" in _failed_ids(C.check_tensor(broken))


def test_rejects_a_wrong_time_convention_hard(synthetic_ds: xr.Dataset) -> None:
    """C1.3 is the silently-catastrophic clause: a store claiming start-of-hour
    is an hour out of phase with its weather everywhere. Never a soft warning."""
    broken = synthetic_ds.copy()
    broken.attrs[C.ATTR_TIME_CONVENTION] = "start_of_hour"
    assert "time_convention" in _failed_ids(C.check_tensor(broken))


def test_rejects_a_missing_time_convention_when_nothing_records_it(
    synthetic_ds: xr.Dataset,
) -> None:
    broken = synthetic_ds.copy()
    broken.attrs.pop(C.ATTR_TIME_CONVENTION, None)
    assert "time_convention" in _failed_ids(C.check_tensor(broken))


def test_time_convention_in_the_manifest_only_is_a_reporting_gap(
    synthetic_ds: xr.Dataset, tmp_path: Path
) -> None:
    """The invariant is verified, only its location is non-canonical: the store
    is usable, but must self-describe before it backs a reported number."""
    store = tmp_path / "tensor.zarr"
    stripped = synthetic_ds.copy()
    stripped.attrs.pop(C.ATTR_TIME_CONVENTION, None)
    zio.write_tensor(stripped, store)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"provenance": {C.ATTR_TIME_CONVENTION: C.TIME_CONVENTION}})
    )
    report = C.check_tensor(store)
    assert "time_convention" not in _failed_ids(report)
    assert "time_convention" in _reporting_ids(report)
    assert report.ok and not report.reporting_ok


def test_rejects_out_of_domain_fire_state(synthetic_ds: xr.Dataset) -> None:
    values = np.asarray(synthetic_ds[C.FIRE_STATE].values).copy()
    values[0, 0, 0] = 7
    broken = _with_fire_state(synthetic_ds, values)
    assert "fire_state_domain" in _failed_ids(C.check_tensor(broken))


def test_rejects_resurrected_fire(synthetic_ds: xr.Dataset) -> None:
    """A cell going 2 -> 0 means the label pipeline lost the absorbing state."""
    values = np.asarray(synthetic_ds[C.FIRE_STATE].values).copy()
    row, col = np.argwhere(values[-1] == C.BURNED_OUT)[0]
    values[-1, row, col] = C.UNBURNED
    assert "fire_state_absorbing" in _failed_ids(
        C.check_tensor(_with_fire_state(synthetic_ds, values))
    )


def test_rejects_a_cell_that_skips_the_burning_state(synthetic_ds: xr.Dataset) -> None:
    """C1.1 guarantees 0 -> 1 -> 2; a 0 -> 2 jump means the fire line was lost.
    It is still non-decreasing, so the absorbing check alone cannot see it."""
    values = np.asarray(synthetic_ds[C.FIRE_STATE].values).copy()
    values[:, 0, 0] = C.UNBURNED
    values[-1, 0, 0] = C.BURNED_OUT
    failed = _failed_ids(C.check_tensor(_with_fire_state(synthetic_ds, values)))
    assert "fire_state_no_skip" in failed
    assert "fire_state_absorbing" not in failed, "non-decreasing, but still wrong"


def test_rejects_a_static_channel_that_varies_in_time(synthetic_ds: xr.Dataset) -> None:
    values = zio.channel_values(synthetic_ds, "elevation").copy()
    values[-1] += 1.0
    broken = _with_channel(synthetic_ds, "elevation", values)
    assert "static_channels_constant" in _failed_ids(C.check_tensor(broken))


def test_rejects_a_non_binary_mask_channel(synthetic_ds: xr.Dataset) -> None:
    values = zio.channel_values(synthetic_ds, "water_barrier_mask").copy()
    values[0, 0, 0] = 0.5
    broken = _with_channel(synthetic_ds, "water_barrier_mask", values)
    assert "mask_channels_binary" in _failed_ids(C.check_tensor(broken))


def test_rejects_interpolated_fuel_class_ids(synthetic_ds: xr.Dataset) -> None:
    """FBFM40 ids are labels. A non-integral value means somebody resampled a
    class raster with bilinear interpolation, which invents fuel models."""
    values = zio.channel_values(synthetic_ds, "fuel_model_id").copy()
    values[:, 0, 0] = 121.5
    broken = _with_channel(synthetic_ds, "fuel_model_id", values)
    assert "class_channels_integral" in _failed_ids(C.check_tensor(broken))


# --------------------------------------------------------------------------
# Teeth: C1.5 finite + C1.7 physical range - "structure is not plausibility"
#
# R11/ADR-010. Every case below PASSED the v2.2 checker with a clean
# `OK - 56 checks passed (reporting-ready)`. That is the whole point: each one
# satisfies every structural declaration C1.5 makes and is still nonsense.
# --------------------------------------------------------------------------


def _const_channel(ds: xr.Dataset, name: str, value: float) -> xr.Dataset:
    values = zio.channel_values(ds, name).copy()
    values[...] = value
    return _with_channel(ds, name, values)


def test_c1_7_rejects_the_lfps_nodata_sentinel_on_canopy(synthetic_ds: xr.Dataset) -> None:
    """The exact artifact ADR-010 was written about.

    data rebuilt a CZU tensor with LFPS's `-9999` restored and the v2.2
    checker returned "OK - 42 checks passed (reporting-ready)" on a tensor whose
    mean canopy cover was -3085%. `-9999` is finite, integral and static, so it
    satisfied every C1.5 declaration. This is that tensor in miniature.
    """
    broken = _const_channel(synthetic_ds, "canopy_cover", -9999.0)
    report = C.check_tensor(broken)
    assert "range_canopy_cover" in _failed_ids(report), (
        "a mean canopy cover of -9999% must be a HARD failure (C1.7), not a reporting gap"
    )
    assert "range_canopy_cover" not in _reporting_ids(report), "C1.7 is `fail` severity"
    # The message must name the sentinel and the fill policy, or the next lead
    # to hit a coastal fire has to go read an ADR to find out what to do.
    message = next(c.message for c in report.failures if c.check_id == "range_canopy_cover")
    assert "-9999" in message and "LFPS" in message and "fill policy" in message.lower()


def test_c1_7_rejects_the_sentinel_on_fuel_model_id(synthetic_ds: xr.Dataset) -> None:
    broken = _const_channel(synthetic_ds, "fuel_model_id", -9999.0)
    assert "range_fuel_model_id" in _failed_ids(C.check_tensor(broken))


@pytest.mark.parametrize("value", [-0.01, 100.01, 3000.0, -1.0])
def test_c1_7_canopy_cover_is_a_percentage(synthetic_ds: xr.Dataset, value: float) -> None:
    """No legitimate value lies outside [0, 100]; that is why it is a hard fail."""
    values = zio.channel_values(synthetic_ds, "canopy_cover").copy()
    values[:, 0, 0] = value
    broken = _with_channel(synthetic_ds, "canopy_cover", values)
    assert "range_canopy_cover" in _failed_ids(C.check_tensor(broken))


@pytest.mark.parametrize("value", [0.0, 4242.0, 100.0, 90.0, 205.0, 150.0])
def test_c1_7_fuel_model_id_must_be_an_fbfm40_class(synthetic_ds: xr.Dataset, value: float) -> None:
    """Integral is not enough - FBFM40 is an ENUMERATION, not a number line.

    Every value here is finite, integral and static, so it satisfies C1.5
    completely. `0` is the one that matters most in practice: it is a common
    LANDFIRE fill and it is not a fuel model.
    """
    values = zio.channel_values(synthetic_ds, "fuel_model_id").copy()
    values[:, 0, 0] = value
    broken = _with_channel(synthetic_ds, "fuel_model_id", values)
    assert "range_fuel_model_id" in _failed_ids(C.check_tensor(broken)), (
        f"{value:g} is not an FBFM40 class"
    )


@pytest.mark.parametrize("value", [91.0, 98.0, 104.0, 124.0, 149.0, 165.0, 189.0, 204.0])
def test_c1_7_accepts_every_legal_fbfm40_group(synthetic_ds: xr.Dataset, value: float) -> None:
    """The bound of a hard-fail clause must be verified from BOTH sides.

    A range check that rejects everything is as useless as one that rejects
    nothing, and it is the failure mode that gets a hard clause deleted.
    """
    values = zio.channel_values(synthetic_ds, "fuel_model_id").copy()
    values[:, 0, 0] = value
    broken = _with_channel(synthetic_ds, "fuel_model_id", values)
    assert "range_fuel_model_id" not in _failed_ids(C.check_tensor(broken))


def test_c1_7_accepts_the_documented_fill_policy(synthetic_ds: xr.Dataset) -> None:
    """ADR-010's fill of record - FBFM40 98 (NB8 Open Water) + canopy 0 - passes.

    A range clause that rejected the ratified remedy would leave data with
    no legal way to represent a coastal domain.
    """
    filled = _const_channel(synthetic_ds, "fuel_model_id", float(C.FBFM40_OPEN_WATER))
    filled = _const_channel(filled, "canopy_cover", 0.0)
    failed = _failed_ids(C.check_tensor(filled))
    assert "range_fuel_model_id" not in failed and "range_canopy_cover" not in failed


@pytest.mark.parametrize(
    "name", ["rh_2m", "temp_2m", "wind_u10", "elevation", "canopy_cover", "fuel_moisture_proxy"]
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_c1_5_rejects_non_finite_features(synthetic_ds: xr.Dataset, name: str, bad: float) -> None:
    """A single non-finite cell anywhere in `features` is a hard failure.

    Measured on the v2.2 checker: an all-NaN `rh_2m` passed all 56 checks
    because nothing looked. Worse, `+inf` was actively BLESSED by two existing
    clauses - `inf == round(inf)` satisfies `class_channels_integral`, and an
    all-inf slab is `array_equal` to itself so it satisfies
    `static_channels_constant`. One NaN NaNs that channel's C3 mean and hence
    every fire's normalisation of it (C3.4).
    """
    values = zio.channel_values(synthetic_ds, name).copy()
    values[0, 0, 0] = bad
    broken = _with_channel(synthetic_ds, name, values)
    assert "features_finite" in _failed_ids(C.check_tensor(broken))


def test_c1_5_names_the_offending_channel(synthetic_ds: xr.Dataset) -> None:
    values = zio.channel_values(synthetic_ds, "rh_2m").copy()
    values[3, 1, 1] = np.nan
    broken = _with_channel(synthetic_ds, "rh_2m", values)
    message = next(
        c.message for c in C.check_tensor(broken).failures if c.check_id == "features_finite"
    )
    assert "rh_2m" in message and "NaN" in message


def test_infinite_fuel_model_id_no_longer_satisfies_the_integral_clause(
    synthetic_ds: xr.Dataset,
) -> None:
    """Regression for the precise blessing described above."""
    broken = _const_channel(synthetic_ds, "fuel_model_id", np.inf)
    failed = _failed_ids(C.check_tensor(broken))
    assert "features_finite" in failed
    assert "range_fuel_model_id" in failed
    assert "class_channels_integral" not in failed, (
        "recorded for the record: `inf == round(inf)` is True, so the integral clause STILL "
        "passes an infinite channel. It is not the clause that catches this, and must not be "
        "relied on to; C1.5 features_finite is"
    )


# --------------------------------------------------------------------------
# Teeth: the verdict ladder itself (ADR-012)
#
# "A diagnostic that fails to `ok` is worse than no diagnostic." sim
# found a NaN falling through a `<`/`<=` ladder into the `ok` branch, printing
# `cos: +nan [ok]` on 2 of 5 fires - hiding a good result AND passing a weak
# one. This checker is the project's most load-bearing verdict ladder.
# --------------------------------------------------------------------------


def test_nan_is_never_a_passing_verdict() -> None:
    """The premise, then the guarantee.

    Python's truthiness and its comparisons disagree about NaN, and that
    disagreement is the entire defect: every comparison is False (so an
    if/elif ladder reaches the trailing `else`), while `bool()` is True (so a
    raw statistic passed on as a verdict reads as PASS).
    """
    nan = float("nan")
    assert bool(nan) is True, "premise: bool(NaN) is truthy in Python"
    assert not (nan < 1e-9) and not (nan <= 1.0) and not (nan > 0.0), (
        "premise: NaN is False under EVERY comparison"
    )

    report = C.ContractReport(target="verdict-ladder")
    for value in (nan, np.float64("nan"), np.inf, -np.inf, None):
        report.add("C1", "synthetic_clause", value, "a non-finite outcome must never pass")
    assert report.failures, "a non-finite verdict reached the report as a PASS"
    assert len(report.failures) == len(report.checks)
    assert report.ok is False


def test_verdict_passes_real_booleans_through_unchanged() -> None:
    """The guard must not make correctly-written clauses harder to write."""
    report = C.ContractReport(target="verdict-ladder")
    assert report.add("C1", "a", True, "") is True
    assert report.add("C1", "b", np.bool_(True), "") is True
    assert report.add("C1", "c", np.all(np.array([True, True])), "") is True
    assert report.add("C1", "d", False, "") is False
    assert report.add("C1", "e", [], "") is False  # "no violations collected" idiom
    assert report.add("C1", "f", ["a violation"], "") is True


def test_an_empty_comparison_is_unverifiable_not_vacuously_true() -> None:
    """`np.all([])` is True. Nothing was compared, so nothing was verified."""
    report = C.ContractReport(target="verdict-ladder")
    assert np.all(np.array([], dtype=bool)), "premise: numpy vacuous truth"
    assert report.add("C1", "empty", np.array([], dtype=bool), "") is False


def test_a_malformed_attribute_fails_its_clause_instead_of_crashing(
    synthetic_ds: xr.Dataset,
) -> None:
    """One bad attribute must not truncate the punch list.

    If it raises, every clause after it goes unreported and under `check_all`
    the C2/C3 halves never run at all - one unparseable value turns into an
    invisible tensor.
    """
    for bad in ("one thousand", None, float("nan"), [1000.0]):
        broken = synthetic_ds.copy()
        broken.attrs[C.ATTR_CELL_SIZE] = bad
        report = C.check_tensor(broken)  # must not raise
        assert "cell_size_attr" in _failed_ids(report), bad
        assert len(report.checks) > 20, "the rest of the punch list must still be produced"


def test_a_malformed_bbox_fails_its_clause_instead_of_crashing(
    synthetic_ds: xr.Dataset, manifest_path: Path, tmp_path: Path
) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest["bbox_5070"] = ["a", "b", "c", "d"]
    broken = tmp_path / "manifest.json"
    broken.write_text(json.dumps(manifest))
    failed = _failed_ids(C.check_manifest(broken, ds=synthetic_ds))  # must not raise
    assert "bbox_matches_tensor" in failed and "type_bbox_5070" in failed


# --------------------------------------------------------------------------
# Teeth: C2
# --------------------------------------------------------------------------


def test_rejects_manifest_disagreeing_with_tensor(
    synthetic_ds: xr.Dataset, manifest_path: Path, tmp_path: Path
) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest["n_hours"] = int(manifest["n_hours"]) + 5
    manifest["bbox_5070"] = [0.0, 0.0, 1000.0, 1000.0]
    broken = tmp_path / "manifest.json"
    broken.write_text(json.dumps(manifest))
    failed = _failed_ids(C.check_manifest(broken, ds=synthetic_ds))
    assert "n_hours_matches_tensor" in failed
    assert "bbox_matches_tensor" in failed


def test_rejects_incomplete_manifest(tmp_path: Path) -> None:
    broken = tmp_path / "manifest.json"
    broken.write_text(json.dumps({"fire_id": "x"}))
    assert "keys" in _failed_ids(C.check_manifest(broken))


def test_rejects_manifest_without_spatial_block_id(
    manifest_path: Path, tmp_path: Path, labels_only: bool
) -> None:
    """C3.1 - without a block id a fire cannot be assigned to a spatially
    isolated fold, and leave-one-fire-out across an overlapping pair is
    landscape leakage."""
    if labels_only:
        pytest.skip("--labels-only: interim stores carry no C2 manifest")
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("spatial_block_id")
    broken = tmp_path / "manifest.json"
    broken.write_text(json.dumps(manifest))
    failed = _failed_ids(C.check_manifest(broken))
    assert "keys" in failed and "type_spatial_block_id" in failed


# --------------------------------------------------------------------------
# Teeth: C3
# --------------------------------------------------------------------------


def _stats_file(tmp_path: Path, stats: dict, name: str = "norm_stats.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(stats))
    return path


def test_rejects_norm_stats_with_missing_channel(norm_stats_path: Path, tmp_path: Path) -> None:
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    stats = json.loads(norm_stats_path.read_text())
    stats["mean"].pop("slope")
    assert "mean_shape" in _failed_ids(C.check_norm_stats(_stats_file(tmp_path, stats)))


def test_rejects_zero_std(norm_stats_path: Path, tmp_path: Path) -> None:
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    stats = json.loads(norm_stats_path.read_text())
    stats["std"]["slope"] = 0.0
    assert "std_positive" in _failed_ids(C.check_norm_stats(_stats_file(tmp_path, stats)))


def test_rejects_standardised_categorical_channels(norm_stats_path: Path, tmp_path: Path) -> None:
    """C3.2 - standardising an FBFM40 class id is meaningless arithmetic on a
    label; the identity transform is the contract, not a convention."""
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    stats = json.loads(norm_stats_path.read_text())
    stats["mean"]["fuel_model_id"] = 142.7
    stats["std"]["fuel_model_id"] = 31.2
    assert "categorical_identity" in _failed_ids(C.check_norm_stats(_stats_file(tmp_path, stats)))


def test_rejects_missing_n_train_blocks(norm_stats_path: Path, tmp_path: Path) -> None:
    """C3.3 is enforced, not advisory: without the count the bootstrap guard
    cannot be evaluated at all, which is worse than declaring 1."""
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    stats = json.loads(norm_stats_path.read_text())
    stats.pop("n_train_blocks", None)
    failed = _failed_ids(C.check_norm_stats(_stats_file(tmp_path, stats)))
    assert "n_train_blocks" in failed and "keys" in failed


def test_bootstrap_stats_are_usable_but_never_reportable(
    norm_stats_path: Path, tmp_path: Path
) -> None:
    """The whole point of promoting the bootstrap warning out of the JSON: it
    must bite at the gate, and it must not be skippable by not reading a note."""
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    stats = json.loads(norm_stats_path.read_text())
    stats.pop(C.NORM_STATS_LEGACY_BLOCK, None)  # isolate the clause under test
    stats["n_train_blocks"] = 1
    stats["bootstrap"] = True
    report = C.check_norm_stats(_stats_file(tmp_path, stats))
    assert report.ok, "a bootstrap file is valid for plumbing"
    assert not report.reporting_ok, "a bootstrap file is never valid for a reported number"
    assert _reporting_ids(report) == {"bootstrap_guard"}
    assert "REPORTING GATE" in report.format()
    assert report.to_dict()["reporting_ready"] is False

    stats["n_train_blocks"] = 2
    stats.pop("bootstrap")
    promoted = C.check_norm_stats(_stats_file(tmp_path, stats, "ok.json"))
    assert promoted.reporting_ok, "\n" + promoted.format(verbose=True)


def test_bootstrap_file_must_say_so(norm_stats_path: Path, tmp_path: Path) -> None:
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    stats = json.loads(norm_stats_path.read_text())
    stats["n_train_blocks"] = 1
    stats.pop("bootstrap", None)
    assert "bootstrap_marked" in _reporting_ids(C.check_norm_stats(_stats_file(tmp_path, stats)))


def test_legacy_nested_channels_block_is_a_reporting_gap(
    norm_stats_path: Path, tmp_path: Path
) -> None:
    """C3.2 fixed the canonical shape as top-level dicts. A duplicate nested
    block that AGREES is a cleanup item (ADR-008, A5); one that DISAGREES is a
    hard failure, because two consumers would then normalise differently."""
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    stats = json.loads(norm_stats_path.read_text())
    stats["n_train_blocks"] = 2
    stats.pop("bootstrap", None)
    stats["channels"] = {
        name: {"mean": stats["mean"][name], "std": stats["std"][name]} for name in C.CHANNELS
    }
    agreeing = C.check_norm_stats(_stats_file(tmp_path, stats))
    assert agreeing.ok and not agreeing.reporting_ok
    assert "no_legacy_nested_block" in _reporting_ids(agreeing)

    stats["channels"]["slope"]["std"] = float(stats["std"]["slope"]) * 2.0
    conflicting = C.check_norm_stats(_stats_file(tmp_path, stats, "conflict.json"))
    assert "nested_block_agrees" in _failed_ids(conflicting)


def test_an_incomparable_nested_value_is_a_conflict_not_an_agreement(
    norm_stats_path: Path, tmp_path: Path
) -> None:
    """ADR-012 regression, found by auditing this module's own ladders.

    The old code skipped any nested entry that was not finite and fell through
    to "does not contradict" - an UNVERIFIABLE comparison landing in the pass
    branch, which is the shape the policy forbids. An explicit `null` really is
    agreement (that is how a categorical channel is written) and must stay
    passing; a NaN is not.
    """
    if not norm_stats_path.is_file():
        pytest.skip("no norm stats available for the target under test")
    base = json.loads(norm_stats_path.read_text())
    base["n_train_blocks"] = 2
    base.pop("bootstrap", None)

    incomparable = dict(base)
    incomparable["channels"] = {"slope": {"mean": float("nan"), "std": float("nan")}}
    report = C.check_norm_stats(_stats_file(tmp_path, incomparable, "nan.json"))
    assert "nested_block_agrees" in _failed_ids(report), (
        "a nested value that cannot be compared to the canonical one must FAIL, not pass"
    )

    explicit_null = dict(base)
    explicit_null["channels"] = {"fuel_model_id": {"mean": None, "std": None}}
    ok_report = C.check_norm_stats(_stats_file(tmp_path, explicit_null, "null.json"))
    assert "nested_block_agrees" not in _failed_ids(ok_report), (
        "`null` for a categorical channel is agreement; this clause must not become noise"
    )


def test_for_reporting_mode_promotes_gaps_to_failures(tensor_path: Path) -> None:
    """`make contract` and `make contract-reporting` must disagree exactly on
    the reporting-gate clauses, or the gate is decoration."""
    plumbing = C.main([str(tensor_path)])
    reporting = C.main([str(tensor_path), "--for-reporting"])
    report = C.check_all(tensor_path)
    assert plumbing == (0 if report.ok else 1)
    assert reporting == (0 if report.reporting_ok else 1)
    assert reporting >= plumbing, "reporting mode may never be more permissive"
    if report.reporting_gaps and report.ok:
        assert plumbing == 0 and reporting == 1, "the gate must bite in reporting mode only"


# --------------------------------------------------------------------------
# The writer cannot produce a non-conformant store
# --------------------------------------------------------------------------


def test_writer_refuses_partial_tensor_at_the_c1_path(synthetic_ds: xr.Dataset) -> None:
    """ADR-003: a partial tensor is never written to the C1 path."""
    from wildfire_nowcast.common.grid import Grid

    grid = Grid.from_dataset(synthetic_ds)
    with pytest.raises(ValueError, match="missing C1 channels"):
        zio.build_tensor_dataset(
            {C.FIRE_STATE: np.asarray(synthetic_ds[C.FIRE_STATE].values)},
            grid,
            np.asarray(synthetic_ds["time"].values),
        )


def test_writer_output_is_conformant_by_construction(synthetic_ds: xr.Dataset) -> None:
    """`zarr_io` exists so that conformance is structural, not vigilant: if a
    store it wrote ever fails the checker, the checker is right."""
    report = C.check_tensor(synthetic_ds)
    assert report.ok, "\n" + report.format(verbose=True)
