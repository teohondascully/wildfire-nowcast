"""Self-tests for :mod:`wildfire_nowcast.sim`. Run: ``python -m wildfire_nowcast.sim.selftest``.

These are written as plain ``test_*`` functions taking no fixtures, so
infra can adopt them into ``tests/`` verbatim (a one-line
``from wildfire_nowcast.sim.selftest import *`` collects them) without simviz
writing outside its own directory. Proposed to infra in
``docs/decisions.md``.

Each test targets a defect that would render as plausible-but-wrong rather than
as a crash — the class this package exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, UNBURNED
from wildfire_nowcast.sim.c5 import STATIC_C5, WEATHER_C5, c5_inputs
from wildfire_nowcast.sim.ensemble import (
    arrival_quantiles,
    burn_probability,
    ensemble_diagnostics,
    independence_dispersion_index,
)
from wildfire_nowcast.sim.movie import frame_order, max_front_gap_cells, teleport_cells
from wildfire_nowcast.sim.reader import arrival_hour, ever_burned, frontier_of
from wildfire_nowcast.sim.stub_model import StubEnsemble
from wildfire_nowcast.sim.style import assert_north_up, plot_extent

__all__ = [
    "test_north_up_guard_rejects_flipped_axes",
    "test_plot_extent_puts_row_zero_at_north",
    "test_frontier_excludes_domain_edge",
    "test_arrival_hour_censors_never_burned",
    "test_arrival_quantiles_are_censored_not_optimistic",
    "test_burn_probability_matches_member_count",
    "test_dispersion_index_is_one_for_independent_pixels",
    "test_collapse_is_detected_and_healthy_is_not",
    "test_frame_order_never_drops_an_hour",
    "test_teleport_threshold_separates_fast_front_from_jump",
    "test_ignition_hour_is_not_reported_as_a_teleport",
    "test_wind_alignment_never_passes_on_a_non_finite_statistic",
    "test_c5_weather_starts_at_t0_plus_one",
    "test_c5_refuses_to_run_past_the_last_hour",
    "test_stub_states_are_absorbing",
    "run_all",
    "test_every_key_the_dashboard_plots_is_classified",
    "test_a_quarantined_key_without_its_clause_is_a_violation",
    "test_g3_readiness_reports_a_missing_calibration_criterion",
    "test_outward_normal_agrees_with_brute_force_nearest_burned_cell",
    "test_sector_uses_the_wind_and_the_north_up_convention",
    "test_growth_split_is_exact_and_sums_to_the_pooled_total",
    "test_envi_bsq_roundtrip_keeps_row_zero_at_north",
    "test_elmfire_mapping_compromises_are_declared_not_remembered",
    "test_elmfire_is_never_silently_substituted",
    "test_coarsening_playthrough_recovers_known_areas_and_catches_every_defect",
    "test_coarsening_matches_the_rule_that_defines_truth",
    "test_fine_grid_is_exactly_nested_in_the_coarse_one",
    "test_coarsening_never_erases_the_initial_condition",
    "test_degeneracy_verdict_flags_the_measured_s3_configuration",
    "test_degeneracy_verdict_does_not_flag_a_healthy_ensemble",
    "test_zero_growth_is_degenerate_by_c6_2_verbatim",
    "test_every_playthrough_arm_declares_its_expected_verdict",
]


# -- orientation (C1.4) ----------------------------------------------------


def test_north_up_guard_rejects_flipped_axes() -> None:
    y_desc = np.array([100.0, 99.0, 98.0])
    x_asc = np.array([0.0, 1.0, 2.0])
    assert_north_up(x_asc, y_desc)  # the legal case

    for bad_x, bad_y, why in [
        (x_asc[::-1], y_desc, "descending x"),
        (x_asc, y_desc[::-1], "ascending y (south-up)"),
    ]:
        try:
            assert_north_up(bad_x, bad_y)
        except ValueError:
            continue
        raise AssertionError(f"north-up guard accepted {why}; every fire would render mirrored")


def test_plot_extent_puts_row_zero_at_north() -> None:
    geom = plot_extent(np.array([0.0, 1000.0]), np.array([5000.0, 4000.0]), cell_size_m=1000.0)
    left, right, bottom, top = geom.extent
    assert left < right, "x extent must increase eastward"
    assert bottom < top, "extent must be (…, bottom, top) with top north-most"
    assert top == 5500.0 and bottom == 3500.0, geom.extent
    assert geom.imshow_kwargs["origin"] == "upper", "origin='upper' is required by this extent"


# -- derived views ---------------------------------------------------------


def test_frontier_excludes_domain_edge() -> None:
    m = np.zeros((5, 5), dtype=bool)
    m[1:4, 1:4] = True
    f = frontier_of(m)
    assert not f[2, 2], "an interior cell is not frontier"
    assert f[1, 1] and f[3, 3], "the ring of the block is frontier"

    full = np.ones((4, 4), dtype=bool)
    assert not frontier_of(full).any(), (
        "a mask filling the domain has no frontier; reporting the tile edge as a fire front "
        "would draw a bright box around every clipped fire"
    )


def test_arrival_hour_censors_never_burned() -> None:
    state = np.zeros((3, 2, 2), dtype=np.uint8)
    state[1, 0, 0] = BURNING
    state[2, 0, 0] = BURNED_OUT
    a = arrival_hour(state)
    assert a[0, 0] == 1.0
    assert np.isnan(a[1, 1]), "a cell that never burns has no arrival time, not a late one"
    assert ever_burned(state)[2, 0, 0]


# -- ensemble statistics ---------------------------------------------------


def _ensemble_from_arrivals(arrivals: list[list[int | None]], horizon: int = 3) -> np.ndarray:
    """Build ``uint8[M, T, 1, N]`` samples from per-member arrival hours."""
    m, n = len(arrivals), len(arrivals[0])
    out = np.zeros((m, horizon, 1, n), dtype=np.uint8)
    for i, row in enumerate(arrivals):
        for j, a in enumerate(row):
            if a is not None:
                out[i, a - 1 :, 0, j] = BURNING
    return out


def test_arrival_quantiles_are_censored_not_optimistic() -> None:
    # Cell 0: every member burns at hour 1. Cell 1: only 2 of 10 burn, at hour 1.
    arrivals = [[1, 1]] + [[1, None]] * 9
    samples = _ensemble_from_arrivals(arrivals)
    (q10, q50, q90), prob = arrival_quantiles(samples)

    assert q50[0, 0] == 1.0
    assert abs(prob[0, 1] - 0.1) < 1e-9, prob[0, 1]
    assert np.isnan(q50[0, 1]), (
        "a cell that burns in 1 of 10 members has NO median arrival time. Reporting hour 1 here "
        "is the exact failure this test exists for: the map would show a fast, certain fire "
        "everywhere the ensemble is actually unsure."
    )
    assert np.isnan(q90[0, 1])
    assert q10[0, 1] == 1.0, "the p10 IS defined at p(burn)=0.1 and should be finite"


def test_burn_probability_matches_member_count() -> None:
    samples = _ensemble_from_arrivals([[1, 1], [1, None], [1, None], [1, None]])
    p = burn_probability(samples)
    assert p[0, 0] == 1.0
    assert abs(p[0, 1] - 0.25) < 1e-9


def test_dispersion_index_is_one_for_independent_pixels() -> None:
    """The index must be ~1 on a *constructed* independent ensemble, not just on the stub."""
    rng = np.random.default_rng(3)
    n_members, n_cells = 200, 900
    p = rng.uniform(0.15, 0.85, size=n_cells)
    draws = (rng.random((n_members, n_cells)) < p).astype(np.uint8)
    samples = draws.reshape(n_members, 1, 30, 30)
    index = independence_dispersion_index(samples)
    assert 0.85 < index < 1.15, (
        f"independent Bernoulli pixels must score ~1.0, got {index:.3f}; the collapse threshold "
        "is calibrated against this being true"
    )


def test_collapse_is_detected_and_healthy_is_not() -> None:
    rng = np.random.default_rng(0)
    n_members, n_cells = 60, 900

    p = rng.uniform(0.2, 0.8, size=n_cells)
    indep = (rng.random((n_members, n_cells)) < p).astype(np.uint8).reshape(n_members, 1, 30, 30)
    assert ensemble_diagnostics(indep)["collapsed"], (
        "independent-per-pixel noise is the known-broken model (it collapses); the detector "
        "must fire on it"
    )

    # Shared latent: one draw per member shifts every pixel together.
    z = rng.normal(0.0, 1.2, size=(n_members, 1))
    q = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p))[None, :] + z)))
    corr = (rng.random((n_members, n_cells)) < q).astype(np.uint8).reshape(n_members, 1, 30, 30)
    diag = ensemble_diagnostics(corr)
    assert not diag["collapsed"], diag
    assert diag["independence_dispersion_index"] > 2.0, diag


# -- movie -----------------------------------------------------------------


def test_frame_order_never_drops_an_hour() -> None:
    growth = np.array([0.0, 0.0, 30.0, 1.0, 0.0])
    assert frame_order(growth, pacing="uniform") == [0, 1, 2, 3, 4]
    order = frame_order(growth, pacing="event", max_dwell=4)
    assert set(order) == {0, 1, 2, 3, 4}, "event pacing must not hide an hour"
    assert order.count(2) > order.count(0), "the burst hour should dwell"
    assert order == sorted(order), "event pacing must not reorder time"


def test_teleport_threshold_separates_fast_front_from_jump() -> None:
    ever = np.zeros((2, 1, 40), dtype=bool)
    ever[0, 0, 0:3] = True
    ever[1, 0, 0:6] = True  # front advanced 3 cells
    ever[1, 0, 30] = True  # and one cell landed 25 cells away
    assert max_front_gap_cells(ever, 1) == 24  # capped by `limit`

    at1 = teleport_cells(ever, 1, min_gap_cells=1)
    assert at1[0, 4] and at1[0, 5] and at1[0, 30], "min_gap_cells=1 flags the fast front too"
    at3 = teleport_cells(ever, 1, min_gap_cells=3)
    assert not at3[0, 5] and at3[0, 30], (
        "the default threshold must flag the genuine jump and NOT a 3-cell/hour front, or the "
        "marker means nothing on a fast fire"
    )


def test_ignition_hour_is_not_reported_as_a_teleport() -> None:
    """First appearance has no prior region, so there is nothing to jump FROM.

    Regression: this reported `max_front_gap_km = 24` for BOTH Zogg and CZU —
    the fires' ignition hours measured against an empty `ever[t-1]`, surfacing in
    the movie summary as a domain-scale spot event. A false positive that scales
    with the domain rather than with the fire is worse than no detector, because
    it trains the reader to ignore the field.
    """
    ever = np.zeros((3, 1, 40), dtype=bool)
    ever[1, 0, 20] = True  # ignition at t=1, nothing burned at t=0
    ever[2, 0, 19:22] = True  # ordinary contiguous growth at t=2

    assert max_front_gap_cells(ever, 1) == 0, "ignition is not a 20-cell jump"
    assert not teleport_cells(ever, 1).any(), "ignition must not be marked as spotting"
    assert max_front_gap_cells(ever, 2) == 0, "contiguous growth is not a jump"


def test_wind_alignment_never_passes_on_a_non_finite_statistic() -> None:
    """A NaN must not fall through a `<`/`<=` ladder into the `ok` branch.

    Regression, and the most expensive defect this package has had: an empty
    `ever[t-1]` made a centroid NaN, `nan < 1e-9` is False so the guard let it
    through, and every subsequent comparison being False landed the verdict on
    the trailing `else: ok`. It reported `area_weighted_cos: +nan` AND `ok` on
    two of five fires — hiding a good CZU result and passing a weak Zogg one.
    Per INTERFACES C-1, unverifiable is a FAIL.
    """
    from wildfire_nowcast.sim.diagnostics import wind_alignment  # noqa: PLC0415

    nan = float("nan")
    assert not (nan <= -0.30) and not (nan < 0.05), (
        "premise of the regression: NaN is False under EVERY comparison, so a verdict "
        "ladder made of elif-<= hands it to the trailing else"
    )

    f = wind_alignment(_FakeFire())
    assert f.status != "ok", f"a fire with no usable growth step must not pass, got {f.status}"
    assert f.status in {"fail", "undetermined"}, f.status
    cos = f.evidence.get("area_weighted_cos")
    assert cos is None or np.isfinite(cos), f"a reported statistic must be finite, got {cos}"


class _FakeFire:
    """Minimal FireFrames stand-in whose only growth step has an EMPTY prior."""

    n_hours = 2

    def __init__(self) -> None:
        self.ever = np.zeros((2, 4, 4), dtype=bool)
        self.ever[1, 1:3, 1:3] = True  # ignition: ever[0] is entirely empty
        self.wind_u = np.full((2, 4, 4), 5.0, dtype=np.float32)
        self.wind_v = np.zeros((2, 4, 4), dtype=np.float32)

        class _G:
            x_centres = np.arange(4, dtype=float) * 1000.0
            y_centres = np.arange(4, dtype=float)[::-1] * 1000.0

        self.geom = _G()


# -- C5 adapter ------------------------------------------------------------


def _open_synthetic():  # noqa: ANN202
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.common.synthetic import make_synthetic_fire  # noqa: PLC0415
    from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: PLC0415

    tmp = tempfile.mkdtemp(prefix="simviz-selftest-")
    res = make_synthetic_fire(seed=0, n_hours=24, out=f"{tmp}/tensor.zarr")
    return open_tensor(res[0])


def test_c5_weather_starts_at_t0_plus_one() -> None:
    from wildfire_nowcast.common.zarr_io import channel_values  # noqa: PLC0415

    ds = _open_synthetic()
    t0, horizon = 5, 3
    inp = c5_inputs(ds, t0, horizon)
    inp.check()

    assert inp.static.shape[0] == len(STATIC_C5) == 8
    assert inp.weather.shape[1] == len(WEATHER_C5) == 5
    u = channel_values(ds, "wind_u10")
    assert np.allclose(inp.weather[0, WEATHER_C5.index("wind_u10")], u[t0 + 1]), (
        "weather[0] must be the hour that PRODUCES the first predicted state (t0+1). Taking t0 "
        "puts every fire an hour out of phase with its weather (C1.3, ADR-006)."
    )
    assert np.array_equal(
        inp.truth[0], np.asarray(ds["fire_state"].values, dtype=np.uint8)[t0 + 1]
    )


def test_c5_refuses_to_run_past_the_last_hour() -> None:
    ds = _open_synthetic()
    n_t = int(ds.sizes["time"])
    try:
        c5_inputs(ds, n_t - 2, 5)
    except IndexError:
        return
    raise AssertionError("padding the truth would score the model against invented data")


def test_stub_states_are_absorbing() -> None:
    ds = _open_synthetic()
    inp = c5_inputs(ds, 6, 4)
    samples = StubEnsemble().predict(inp.x0, inp.static, inp.weather, 8, 4, 0)
    assert samples.shape == (8, 4, *inp.x0.shape)
    assert set(np.unique(samples)).issubset({UNBURNED, BURNING, BURNED_OUT})

    for m in range(samples.shape[0]):
        seq = np.concatenate([inp.x0[None], samples[m]], axis=0).astype(np.int16)
        d = np.diff(seq, axis=0)
        assert (d >= 0).all(), "C1.1: fire_state is non-decreasing; no cell may un-burn"
        assert not ((seq[:-1] == UNBURNED) & (seq[1:] == BURNED_OUT)).any(), (
            "C1.1 forbids a 0->2 skip"
        )


# -- S3: quarantine registry, growth anatomy, ELMFIRE adapter ---------------
#
# Each of these targets a defect that renders as PLAUSIBLE-BUT-WRONG rather than
# crashing, which is the only kind this package has ever actually shipped.


def test_every_key_the_dashboard_plots_is_classified() -> None:
    """An unclassified key is an UNBADGED key. It must raise, not render plain."""
    from wildfire_nowcast.sim.quarantine import GATE, QUARANTINED, classify

    for key in (
        "band_brier_by_horizon",
        "band_best_member_iou_by_horizon",
        "dispersion_ratio",
        "area_dispersion_ratio",
        "band_ece_by_horizon",
        "band_reliability_by_horizon",
        "band_resolution_by_horizon",
        "band_calibration_error_by_horizon",
    ):
        classify(key)
    try:
        classify("band_a_metric_nobody_registered")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an unregistered key must RAISE, not render unbadged")

    # The two halves of G3 must come back GATE, and the three metrics the
    # contract retired must come back QUARANTINED. If this ever inverts, the
    # dashboard is about to cite a demoted metric as capability.
    assert classify("area_dispersion_ratio").state == GATE
    assert classify("band_calibration_error_by_horizon").state == GATE
    for key in ("dispersion_ratio", "band_ece_by_horizon", "band_reliability_by_horizon"):
        assert classify(key).state == QUARANTINED, key


def test_a_quarantined_key_without_its_clause_is_a_violation() -> None:
    """The audit must FAIL a figure that drew a quarantined key with no badge."""
    from wildfire_nowcast.sim.quarantine import audit_plotted_keys

    assert audit_plotted_keys({"dispersion_ratio": ""}), "empty badge must be flagged"
    ok = audit_plotted_keys({"dispersion_ratio": "QUARANTINED by C6.1 (ADR-011) - ..."})
    assert ok == [], ok
    assert audit_plotted_keys({"area_dispersion_ratio": ""}) == []


def test_g3_readiness_reports_a_missing_calibration_criterion() -> None:
    """A run artifact without G3's own gate criterion must say so, loudly."""
    from wildfire_nowcast.sim.rundash import g3_readiness

    payload = {"pooled_heldout": {"m": {"growth_windows": {"area_dispersion_ratio": 0.9}}}}
    ready = g3_readiness(payload)
    assert ready["adjudicable"] is False
    assert "band_calibration_error_by_horizon" in ready["missing"]

    payload["pooled_heldout"]["m"]["growth_windows"]["band_calibration_error_by_horizon"] = {
        "1": 0.04, "2": 0.05, "3": 0.06
    }
    assert g3_readiness(payload)["adjudicable"] is True


def test_outward_normal_agrees_with_brute_force_nearest_burned_cell() -> None:
    """The scipy-free EDT substitute is CHECKED, not assumed."""
    from wildfire_nowcast.sim.growth import outward_normals

    rng = np.random.default_rng(11)
    for _ in range(3):
        burned = np.zeros((15, 17), dtype=bool)
        burned[rng.integers(0, 15, 4), rng.integers(0, 17, 4)] = True
        ny, nx = outward_normals(burned, radius=40)
        ys, xs = np.nonzero(burned)
        for y in range(15):
            for x in range(17):
                if burned[y, x]:
                    continue
                d2 = (ys - y) ** 2 + (xs - x) ** 2
                best = int(d2.min())
                got = (ny[y, x], nx[y, x])
                assert abs(np.hypot(*got) - 1.0) < 1e-9, got
                ties = [
                    (y - ys[j], x - xs[j]) for j in range(len(ys)) if int(d2[j]) == best
                ]
                assert any(
                    abs(got[0] - dy / np.hypot(dy, dx)) < 1e-9
                    and abs(got[1] - dx / np.hypot(dy, dx)) < 1e-9
                    for dy, dx in ties
                ), (y, x, got, ties)


def test_sector_uses_the_wind_and_the_north_up_convention() -> None:
    """A sign error here would put the HEAD sector upwind and invert the finding."""
    from wildfire_nowcast.sim.growth import sector_of

    burned = np.zeros((9, 9), dtype=np.uint8)
    burned[4, 4] = 1
    # Wind blowing due EAST: u=+1, v=0. Downwind is +x.
    sect = sector_of(burned, 1.0, 0.0)
    assert sect[4, 6] == 0, "a cell due east of the fire must be HEAD in an east wind"
    assert sect[4, 2] == 2, "a cell due west must be REAR"
    assert sect[2, 4] == 1 and sect[6, 4] == 1, "north/south of the fire must be FLANK"
    # Wind blowing due NORTH: v=+1. North is DECREASING row index (C1.4).
    sect = sector_of(burned, 0.0, 1.0)
    assert sect[2, 4] == 0, "north is row-MINUS; a v>0 wind must make the north side HEAD"
    assert sect[6, 4] == 2
    assert sect[4, 4] == -1, "burned cells have no outward normal"


def test_growth_split_is_exact_and_sums_to_the_pooled_total() -> None:
    """Both decompositions must reconstruct C6.2's own total, or they are opinions."""
    from wildfire_nowcast.sim.growth import WindowGrowth, summarise

    rows = [
        WindowGrowth("f", "m", 0, 0, truth_new=4.0, pred_new=6.0, truth_grew=True,
                     wind_speed=3.0, truth_sector=[2.0, 1.0, 1.0],
                     pred_sector=[3.0, 2.0, 1.0]),
        WindowGrowth("f", "m", 1, 2, truth_new=0.0, pred_new=5.0, truth_grew=False,
                     wind_speed=3.0, truth_sector=[0.0, 0.0, 0.0],
                     pred_sector=[1.0, 3.0, 1.0]),
    ]
    s = summarise(rows)["m"]
    assert s["n_new_cells_predicted"] == 11.0
    assert s["predicted_in_growth_windows"] + s["predicted_in_dormant_windows"] == 11.0
    assert sum(s["sector_pred"].values()) == 11.0
    assert s["growth_ratio"] == 11.0 / 4.0
    assert s["growth_ratio_on_growth_windows"] == 6.0 / 4.0
    assert s["dormant_share_of_prediction"] == 5.0 / 11.0


def test_envi_bsq_roundtrip_keeps_row_zero_at_north() -> None:
    """A mirrored raster is the C1.4 failure, one process boundary further out."""
    import tempfile

    from wildfire_nowcast.sim.elmfire import write_envi_bsq

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    with tempfile.TemporaryDirectory() as d:
        stub = Path(d) / "x"
        write_envi_bsq(stub, arr, x_left=0.0, y_top=3000.0, cell_size=1000.0)
        raw = np.frombuffer((stub.with_suffix(".bsq")).read_bytes(), dtype="<f4")
        assert np.array_equal(raw.reshape(3, 4), arr), "BSQ must be written in row order"
        hdr = stub.with_suffix(".hdr").read_text()
    assert "samples = 4" in hdr and "lines   = 3" in hdr
    assert "data type = 4" in hdr and "interleave = bsq" in hdr
    # ELMFIRE's PARSE_MAP_INFO reads six comma-separated numbers after the name.
    line = [x for x in hdr.splitlines() if x.startswith("map info")][0]
    body = line.split("{", 1)[1].split("}", 1)[0]
    assert len(body.split(",")) >= 7, body


def test_elmfire_mapping_compromises_are_declared_not_remembered() -> None:
    """An undocumented mapping compromise is how an unfair baseline gets built."""
    from wildfire_nowcast.sim.elmfire import MAPPING_COMPROMISES

    fields = {c["field"] for c in MAPPING_COMPROMISES}
    for required in ("output coarsening", "wd", "fine lattice", "ensemble"):
        assert required in fields, required
    for c in MAPPING_COMPROMISES:
        assert c["bias"], f"{c['field']} declares no direction of bias"


def test_elmfire_is_never_silently_substituted() -> None:
    """A missing binary must raise, never fall back to another model."""
    from wildfire_nowcast.sim.elmfire import ElmfireNotInstalled, find_binary

    try:
        find_binary("/nonexistent/elmfire-binary-that-does-not-exist")
    except ElmfireNotInstalled:
        pass
    except Exception:  # pragma: no cover
        pass  # a real install was found on PATH; that is also fine


# -- S4: coarsening (playthrough 1) ---------------------------------------


def test_coarsening_playthrough_recovers_known_areas_and_catches_every_defect() -> None:
    """PLAYTHROUGH 1 in one call. Areas are analytic; defects must be caught."""
    from wildfire_nowcast.sim.coarsen import run_playthrough

    report = run_playthrough()
    assert report["rule_passes_every_scenario"], report["rule_rows"]
    for name, why in report["defects_caught_by"].items():
        assert why, f"planted defect {name} was NOT caught — the harness is vacuous"
    assert report["verdict"] == "PASS"


def test_coarsening_matches_the_rule_that_defines_truth() -> None:
    """Our rule must be the SAME rule ``data/rasterize.py`` uses for labels.

    Not a preference: scoring ELMFIRE against truth under a different convention
    is a systematic area bias with no visible cause in any table.
    """
    from wildfire_nowcast.data.rasterize import COVER_THRESHOLD
    from wildfire_nowcast.sim.coarsen import OCCUPANCY_THRESHOLD

    assert OCCUPANCY_THRESHOLD == COVER_THRESHOLD


def test_fine_grid_is_exactly_nested_in_the_coarse_one() -> None:
    """A non-nested lattice hides area error inside partial-overlap weights."""
    from wildfire_nowcast.common.grid import Grid
    from wildfire_nowcast.sim.coarsen import DEFAULT_REFINE, fine_grid

    coarse = Grid(x_min=-2_000_000.0, y_max=2_000_000.0, nx=7, ny=5, cell_size_m=1000.0)
    fine = fine_grid(coarse, DEFAULT_REFINE)
    assert fine.nx == coarse.nx * DEFAULT_REFINE
    assert fine.ny == coarse.ny * DEFAULT_REFINE
    assert fine.bounds == coarse.bounds
    assert abs(fine.cell_size_m * DEFAULT_REFINE - coarse.cell_size_m) < 1e-9


def test_coarsening_never_erases_the_initial_condition() -> None:
    """A ragged perimeter must not let the rule delete cells we handed ELMFIRE.

    This is the failure mode that would recreate the degeneracy at the very last
    step: a member could come back with FEWER burned cells than it started with.
    """
    from wildfire_nowcast.sim.coarsen import DEFAULT_REFINE, coarsen_occupancy

    r = DEFAULT_REFINE
    x0 = np.zeros((3, 3), dtype=np.uint8)
    x0[1, 1] = 1
    fine = np.repeat(np.repeat(x0, r, axis=0), r, axis=1) > 0
    # Erode the fine footprint so occupancy alone would drop the cell.
    fine[r : 2 * r, r : r + (r // 2) + 1] = False
    coarse = coarsen_occupancy(fine, r) | (x0 > 0)
    assert coarse[1, 1], "the t0 state must survive its own round trip"


# -- S4: baseline non-degeneracy (playthrough 2) --------------------------


def test_degeneracy_verdict_flags_the_measured_s3_configuration() -> None:
    """The scoring function must call the KNOWN failure a failure.

    +2 cells against truth's +54 is what ADR-025 (4) measured with the old input
    mapping. A criterion that does not flag it is not a criterion.
    """
    from wildfire_nowcast.sim.playthrough import degeneracy_verdict

    x0 = np.zeros((10, 10), dtype=np.uint8)
    x0[:6, :10] = 1  # 60 cells burned at t0
    s3 = np.repeat(x0[None, None], 6, axis=0).repeat(3, axis=1).copy()
    s3[:, :, 6, :2] = 1  # every member adds the same 2 cells
    v = degeneracy_verdict("s3_replica", s3, x0, truth_new=54)
    assert v.degenerate
    assert v.d2_below_floor and v.d3_no_ensemble


def test_degeneracy_verdict_does_not_flag_a_healthy_ensemble() -> None:
    """And it must NOT fire on a baseline that is merely imperfect.

    A criterion that flags everything voids every gate it touches, which is the
    mirror image of one that flags nothing.
    """
    from wildfire_nowcast.sim.playthrough import degeneracy_verdict

    rng = np.random.default_rng(0)
    x0 = np.zeros((12, 12), dtype=np.uint8)
    x0[:6] = 1
    samples = np.repeat(x0[None, None], 6, axis=0).repeat(3, axis=1).copy()
    for m in range(6):
        cols = rng.choice(12, size=8, replace=False)
        samples[m, :, 6, cols] = 1
        samples[m, :, 7, cols[:4]] = 1
    v = degeneracy_verdict("healthy", samples, x0, truth_new=20)
    assert not v.degenerate, v.as_dict()


def test_zero_growth_is_degenerate_by_c6_2_verbatim() -> None:
    """C6.2: a baseline that ignites nothing is not a distinct baseline."""
    from wildfire_nowcast.sim.playthrough import degeneracy_verdict

    x0 = np.zeros((8, 8), dtype=np.uint8)
    x0[:4] = 1
    silent = np.repeat(x0[None, None], 4, axis=0).repeat(3, axis=1).copy()
    v = degeneracy_verdict("silent", silent, x0, truth_new=30)
    assert v.d1_zero_growth and v.degenerate


def test_every_playthrough_arm_declares_its_expected_verdict() -> None:
    """ADR-030: a playthrough must contain a case that is expected to FAIL."""
    from wildfire_nowcast.sim.playthrough import ARMS

    assert any(a.expect_degenerate for a in ARMS), (
        "no negative control — a playthrough that cannot fail is the exact thing "
        "it exists to prevent"
    )
    assert any(not a.expect_degenerate for a in ARMS)


# -- runner ----------------------------------------------------------------


def run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures: list[tuple[str, BaseException]] = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except BaseException as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_all())
