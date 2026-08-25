"""Self-tests for :mod:`wildfire_nowcast.sim`. Run: ``python -m wildfire_nowcast.sim.selftest``.

These are written as plain ``test_*`` functions taking no fixtures, so
infra can adopt them into ``tests/`` verbatim (a one-line
``from wildfire_nowcast.sim.selftest import *`` collects them) without simviz
writing outside its own directory. Proposed to infra in
``docs/decisions.md``.

Each test targets a defect that would render as plausible-but-wrong rather than
as a crash - the class this package exists to prevent.
"""

from __future__ import annotations

import atexit as _atexit
import functools as _functools
import json
import logging
import os
import shutil as _shutil
import sys
import tempfile as _tempfile
from pathlib import Path

import numpy as np

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, UNBURNED
from wildfire_nowcast.sim.c5 import STATIC_C5, WEATHER_C5, c5_inputs
from wildfire_nowcast.sim.ensemble import (
    COLLAPSE_INDEX_THRESHOLD,
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
    "test_the_one_step_null_is_exact_at_every_lead_and_the_cumulative_one_is_not",
    "test_every_collapse_statement_carries_the_horizon_it_was_taken_at",
    "test_a_failed_instrument_control_withholds_the_verdict_rather_than_passing_it",
    "test_an_arm_that_is_the_model_it_ablates_is_refused_at_the_verdict_not_at_the_load",
    "test_the_power_profile_reports_at_1_2_3_h_and_keeps_refusals_out_of_its_rates",
    "test_the_analytic_index_is_an_identity_and_agrees_with_sampling",
    "test_frame_order_never_drops_an_hour",
    "test_teleport_threshold_separates_fast_front_from_jump",
    "test_ignition_hour_is_not_reported_as_a_teleport",
    "test_wind_alignment_never_passes_on_a_non_finite_statistic",
    "test_c5_weather_starts_at_t0_plus_one",
    "test_c5_refuses_to_run_past_the_last_hour",
    "test_stub_states_are_absorbing",
    "test_g5_window_key_binds_every_argument_it_claims_to_and_no_others",
    "test_g5_a_missing_ensemble_raises_instead_of_substituting_silence",
    "test_g5_the_store_round_trips_bit_identically_and_leaves_no_partial_file",
    "test_g5_the_commensurability_control_can_fail_and_refuses_to_pass_without_a_reference",
    "test_episode_seed_offsets_are_the_windows_position_and_differ_by_draw",
    "test_episode_the_cross_check_can_fail_and_a_missing_window_is_not_a_pass",
    "test_episode_the_zoom_box_is_never_empty_and_never_leaves_the_domain",
    "test_episode_a_rank_with_no_population_behind_it_refuses_to_print_a_number",
    "test_episode_a_commitment_is_band_only_and_read_at_the_final_lead",
    "test_g5_the_gate_columns_carry_the_criterion_and_label_the_barred_one",
    "test_a_reliability_page_keeps_the_empty_bins_the_curve_drops",
    "test_the_reliability_page_curve_control_agrees_here_and_fails_on_a_plant",
    "test_the_wilson_interval_contains_its_own_estimate_where_float_error_says_otherwise",
    "test_a_commitment_is_counted_off_the_bin_lower_edge_and_never_off_a_straddle",
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
    "test_e1_domain_slice_is_exact_and_a_one_cell_shift_is_not",
    "test_e1_refuses_a_mismatched_refine_rather_than_misregistering",
    "test_the_head_distance_field_agrees_with_a_brute_force_over_every_burned_cell",
    "test_the_head_reads_the_ceiling_when_the_front_runs_at_the_ceiling",
    "test_the_head_is_measured_from_the_t0_state_and_not_from_the_final_one",
    "test_a_supplied_stack_is_cut_to_the_window_and_left_alone_when_it_already_is_one",
    "test_a_stack_that_cannot_be_reconciled_is_refused_and_the_message_names_the_cure",
    "test_e1_reads_the_verdict_off_the_pre_registered_rule_in_both_directions",
    "test_e1_a_partial_run_is_self_identifying_and_names_its_fires",
    "test_e1_scores_every_block_at_the_same_ensemble_size",
    "test_e1_does_not_carry_a_second_copy_of_the_estimator",
    "test_e1_records_the_registry_refusing_stage_decay_a_gate",
    "test_e1_decides_only_what_the_missing_blocks_cannot_change",
    "test_e1_refuses_to_score_a_fire_that_did_not_finish",
    "test_e1_refuses_a_window_that_may_have_hit_elmfire_s_wall_clock_cap",
    "test_e1_page_renders_a_partial_run_as_partial",
    "test_a_zero_member_ensemble_is_an_absent_measurement_not_a_degenerate_one",
    "test_the_non_degeneracy_playthrough_refuses_a_harness_with_no_arms",
    "test_the_playthrough_cli_writes_no_artifact_when_nothing_was_measured",
    "test_the_collapse_detector_refuses_an_ensemble_with_no_members",
    "test_g3_readiness_cannot_pronounce_a_payload_with_no_models_adjudicable",
    "test_the_coarsening_playthrough_refuses_a_harness_with_no_planted_defects",
    "test_auditing_zero_plotted_keys_is_not_an_honest_figure",
    "test_a_fire_with_no_measurable_step_publishes_no_teleport_statistics",
    "test_the_burn_probability_panel_draws_nothing_where_no_member_burned",
    "test_replay_draws_the_ensemble_beside_the_best_member",
    "test_the_s4_one_pager_is_reachable_from_the_command_line",
    "test_a_broken_ffmpeg_makes_the_writer_fall_back_AND_SAY_SO",
    "test_no_module_in_sim_configures_logging_at_import",
    "test_the_c2_number_is_delegated_and_the_topology_count_is_not_named_like_it",
    "test_the_c2_verdict_names_the_deriver_and_refuses_what_it_cannot_measure",
    "test_manifest_check_passes_the_derived_c2_value_and_fails_the_topology_count",
    "test_the_c2_value_follows_the_genealogy_rule_and_not_this_page_s_merge_window",
    "test_g6_the_page_cannot_print_a_bar_the_code_has_moved_away_from",
    "test_g6_an_absent_artifact_is_refused_not_rendered_as_an_empty_section",
    "test_g6_the_page_cannot_quietly_depend_on_a_file_the_reader_does_not_have",
    "test_g6_no_colour_is_defined_only_in_the_dark_block_so_one_mode_goes_blank",
    "test_g6_the_source_cannot_drift_back_into_the_output_literal_sink",
    "test_g6_a_number_on_the_page_cannot_silently_disagree_with_its_artifact",
    "test_g6_the_page_cannot_send_a_reader_to_a_file_that_is_not_in_the_repository",
]

# ADR-103: a logger, and NOTHING else at import. The PASS/FAIL lines and the
# tally are this runner's OUTPUT and stay on stdout; a test that could not run
# here, or a run that was interrupted, is a diagnostic about the run.
logger = logging.getLogger(__name__)


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
    """The healthy bar is the ANALYTIC index of the construct, not a round number.

    RE-DERIVED AT ONE STEP (ADR-114 (4)). This asserted
    ``independence_dispersion_index > 2.0``, a bar carried over from a period when
    nothing recorded which horizon a reading came from. The construct below has
    an exact one-step index, computed by
    :func:`~wildfire_nowcast.sim.collapse.analytic_shared_latent_index` from the
    marginals and ``latent_sigma`` alone: **13.726**. So the old bar sat **6.9x
    below** the value it was policing and could not have failed for any defect
    short of the instrument returning nothing. Asserting a magnitude against an
    identity can fail in both directions; asserting non-triviality can fail in
    neither.

    The band is +/-30%, which is 4.4 sampling SD at this member count: over 200
    seeds the estimator reads 13.66 +/- 0.95 on this construct, range
    [11.14, 16.09] = [0.81x, 1.17x] of the identity.
    """
    from wildfire_nowcast.sim.collapse import analytic_shared_latent_index  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n_members, n_cells = 60, 900
    latent_sigma = 1.2

    p = rng.uniform(0.2, 0.8, size=n_cells)
    indep = (rng.random((n_members, n_cells)) < p).astype(np.uint8).reshape(n_members, 1, 30, 30)
    assert ensemble_diagnostics(indep)["collapsed"], (
        "independent-per-pixel noise is the known-broken model (it collapses); the detector "
        "must fire on it"
    )

    # Shared latent: one draw per member shifts every pixel together.
    z = rng.normal(0.0, latent_sigma, size=(n_members, 1))
    q = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p))[None, :] + z)))
    corr = (rng.random((n_members, n_cells)) < q).astype(np.uint8).reshape(n_members, 1, 30, 30)
    diag = ensemble_diagnostics(corr)
    assert not diag["collapsed"], diag

    exact = analytic_shared_latent_index(p, latent_sigma)
    assert exact > COLLAPSE_INDEX_THRESHOLD, (
        f"the construct's exact index is {exact:.4f}, under the {COLLAPSE_INDEX_THRESHOLD} bar, "
        "so it is not a healthy control at all and the test above is vacuous"
    )
    measured = diag["independence_dispersion_index"]
    assert 0.70 * exact < measured < 1.30 * exact, (
        f"the estimator reads {measured:.4f} where the identity says {exact:.4f}. The old "
        "assertion here was `> 2.0`, which this construct clears by 6.9x and which therefore "
        "policed nothing."
    )


def test_the_one_step_null_is_exact_at_every_lead_and_the_cumulative_one_is_not() -> None:
    """ADR-114 (a)(b): three verdicts, each with an exact null, on ONE scene.

    This is the whole repair in one measurement. The ablation is the fixture at
    ``latent_sigma=0``, i.e. the independent-per-pixel model this project treats
    as known-broken, and the estimand is the cells a member ADDS in one step from
    a state every member shares.

    The falsifier, stated because all three horizons passing is the outcome that
    deserves the most suspicion: if the one-step index drifted with lead the way
    the cumulative one does (1.00 -> 1.25 -> 1.50 with no latent anywhere), this
    assertion would fail at k=2 and k=3 and the honest report would be
    "demonstrated at 1 h, NOT DEMONSTRATED at 2-3 h". It does not drift, and the
    same instrument in the same run still separates the latent-on arm.
    """
    from wildfire_nowcast.sim.collapse import (  # noqa: PLC0415
        COLLAPSED,
        NOT_COLLAPSED,
        per_horizon_collapse,
    )

    inp = c5_inputs(_open_synthetic(), 12, 3)
    ablation = per_horizon_collapse(
        StubEnsemble(latent_sigma=0.0).predict, inp, n_members=32, seed=0, n_replicates=8
    )
    full = per_horizon_collapse(
        StubEnsemble(latent_sigma=0.9).predict, inp, n_members=32, seed=0, n_replicates=8
    )

    assert [v.lead_h for v in ablation.verdicts] == [1, 2, 3]
    for v in ablation.verdicts:
        assert v.verdict == COLLAPSED, (
            f"the no-latent ablation must demonstrate collapse at lead {v.lead_h}; it read "
            f"{v.index:.4f} against the {v.threshold} bar with control "
            f"{v.controls.independent_index:.4f}"
        )
        assert 0.6 < v.index < 1.4, (
            f"the one-step null is 1.0 EXACTLY at every lead by algebra; lead {v.lead_h} read "
            f"{v.index:.4f}, which is a drift the cumulative estimand has and this one must not"
        )
    for v in full.verdicts:
        assert v.verdict == NOT_COLLAPSED, (
            f"the latent-on arm must not be called collapsed at lead {v.lead_h}: a repair that "
            f"makes every arm pass has broken the instrument, not fixed it ({v.index:.4f})"
        )
    assert full.cumulative_index_description > ablation.cumulative_index_description


def test_every_collapse_statement_carries_the_horizon_it_was_taken_at() -> None:
    """ADR-114 (d). The one variable that decides the verdict was the one the record dropped."""
    from wildfire_nowcast.sim.collapse import (  # noqa: PLC0415
        CUMULATIVE_FROM_T0,
        NOT_A_VERDICT,
        ONE_STEP_INCREMENT,
        per_horizon_collapse,
    )

    rng = np.random.default_rng(4)
    p = rng.uniform(0.2, 0.8, size=900)
    for lead_steps, expected in ((1, ONE_STEP_INCREMENT), (3, CUMULATIVE_FROM_T0)):
        draws = (rng.random((40, lead_steps, 900)) < p).astype(np.uint8)
        cumulative = np.maximum.accumulate(draws, axis=1).reshape(40, lead_steps, 30, 30)
        diag = ensemble_diagnostics(cumulative)
        assert diag["collapse_index_lead_h"] == lead_steps, diag
        assert diag["collapse_index_estimand"] == expected, diag
        assert diag["collapsed_is_a_verdict"] is False, (
            "ensemble_diagnostics is TRIAGE: it runs no instrument control, so it may describe "
            "and may not pronounce"
        )
        assert diag["collapse_verdict"].startswith(NOT_A_VERDICT), diag["collapse_verdict"]

    inp = c5_inputs(_open_synthetic(), 12, 3)
    result = per_horizon_collapse(
        StubEnsemble(latent_sigma=0.0).predict, inp, n_members=16, seed=1, n_replicates=4
    )
    for record in result.to_dict()["per_horizon_verdicts"]:
        assert "lead_h" in record and "estimand" in record and "conditioning" in record, record
        assert record["controls"]["independent_index"] > 0.0, record
    assert result.to_dict()["cumulative_is_a_verdict"] is False


def test_a_failed_instrument_control_withholds_the_verdict_rather_than_passing_it() -> None:
    """ADR-114 (c). No control reading, no verdict - and never a False that reads as healthy.

    The scene is one uncertain cell. No amount of dependence can move the spread
    of a one-cell sum away from the independent case, so the maximally-dependent
    control cannot clear the bar and the instrument has no power here at all. The
    reading that must NOT come back is ``not_collapsed``.
    """
    from wildfire_nowcast.sim.collapse import (  # noqa: PLC0415
        NOT_A_VERDICT,
        instrument_controls,
        one_step_collapse_verdict,
    )

    given = np.zeros((6, 6), dtype=np.uint8)
    rng = np.random.default_rng(2)
    samples = np.zeros((24, 1, 6, 6), dtype=np.uint8)
    samples[:, 0, 0, 0] = 1  # certain in every member: p = 1, contributes nothing
    samples[:, 0, 1, 1] = (rng.random(24) < 0.5).astype(np.uint8)  # the ONE uncertain cell

    verdict = one_step_collapse_verdict(
        samples, given, lead_h=1, conditioning="truth", seed=0, n_replicates=8
    )
    assert verdict.controls.n_uncertain_cells == 1, verdict.controls.to_dict()
    assert not verdict.controls.comonotone_ok, verdict.controls.to_dict()
    assert verdict.verdict.startswith(NOT_A_VERDICT), verdict.verdict
    assert verdict.is_a_verdict is False

    healthy = instrument_controls(np.full(400, 0.5), 24, seed=0, n_replicates=8)
    assert healthy.ok and healthy.independent_ok and healthy.comonotone_ok, healthy.to_dict()
    assert healthy.reason == ""


def test_an_arm_that_is_the_model_it_ablates_is_refused_at_the_verdict_not_at_the_load() -> None:
    """C5 [v2.18]: a bit-identical arm may be LOADED and may not be SCORED.

    The hazard is specific and it arrives through the feature built to prevent
    one. A model configured with no shared latent has an ablation arm that is
    the same forecast under another name, so the collapse index off that pair is
    ``1.0`` by construction - a perfect null, three ``collapsed`` verdicts, and
    nothing whatever about ``z_t``. Loading it is legitimate; taking a verdict
    from it is not.

    The control is the fixture pair, which DOES differ, so this asserts both
    directions in one run: a demonstrative pair yields three verdicts and a
    degenerate one yields none. Both halves are needed - a refusal that also
    refuses the good pair is a broken instrument, not a strict one.
    """
    from wildfire_nowcast.sim.collapse import (  # noqa: PLC0415
        ArmNotDemonstrativeError,
        measure_arm_separation,
        per_horizon_collapse_on_arm,
        resolve_arm_pair,
    )

    inp = c5_inputs(_open_synthetic(), 12, 3)

    pair = resolve_arm_pair("stub")
    assert pair.treatment_address == "stub-nolatent" and pair.null_address == "stub"
    checked = measure_arm_separation(pair, inp, n_members=8, seed=0)
    assert checked.measured_identical is False, checked.to_dict()
    assert checked.demonstrative and checked.refusal == "", checked.to_dict()
    result = per_horizon_collapse_on_arm(pair, inp, n_members=16, seed=0, n_replicates=4)
    assert [v.lead_h for v in result.verdicts] == [1, 2, 3]

    degenerate = _replace_pair(pair, treatment=pair.null)
    twin = measure_arm_separation(degenerate, inp, n_members=8, seed=0)
    assert twin.measured_identical is True, twin.to_dict()
    assert twin.n_identical_members == 8, twin.to_dict()
    assert twin.demonstrative is False and "BIT-IDENTICAL" in twin.refusal, twin.to_dict()
    try:
        per_horizon_collapse_on_arm(degenerate, inp, n_members=8, seed=0, n_replicates=4)
    except ArmNotDemonstrativeError as exc:
        assert "NO VERDICT" in str(exc), str(exc)
    else:
        raise AssertionError(
            "an arm bit-identical to the model it ablates scored a verdict. That verdict is "
            "1.0 by construction and would read as the cleanest collapse result in the repo."
        )


def test_the_power_profile_reports_at_1_2_3_h_and_keeps_refusals_out_of_its_rates() -> None:
    """C6.7 [v2.18]: report AT 1/2/3 h with the power at each lead.

    Two properties, and the second is the one a plausible implementation gets
    wrong. Every lead in the forecast horizon appears, and a scene the controls
    REFUSED is in neither rate's denominator: counting a refusal as
    ``not_collapsed`` would let a scene with no power at all lower the false-fire
    rate and make the instrument look more discriminating than it is. The
    denominators are therefore fields, and admissible + refused must equal the
    draws made.
    """
    from wildfire_nowcast.sim.collapse import lead_power_profile, resolve_arm_pair  # noqa: PLC0415

    inp = c5_inputs(_open_synthetic(), 12, 3)
    profile = lead_power_profile(
        resolve_arm_pair("stub"), inp, n_members=12, n_seeds=4, n_replicates=4
    )
    assert [row["lead_h"] for row in profile["by_lead"]] == [1, 2, 3], profile
    for row in profile["by_lead"]:
        for arm in ("treatment", "null"):
            admissible = row[f"{arm}_collapsed"] + row[f"{arm}_not_collapsed"]
            assert admissible + row[f"{arm}_refused"] == row["n_seeds"], row
        assert row["power"] is None or 0.0 <= row["power"] <= 1.0, row
        assert row["false_fire"] is None or 0.0 <= row["false_fire"] <= 1.0, row
    assert "separation" in profile["by_lead"][0]


def _replace_pair(pair, **changes):  # noqa: ANN001, ANN003, ANN202
    """``dataclasses.replace`` on an :class:`ArmPair`, spelled once."""
    import dataclasses  # noqa: PLC0415

    return dataclasses.replace(pair, **changes)


def test_the_analytic_index_is_an_identity_and_agrees_with_sampling() -> None:
    """The control on the control: at ``latent_sigma=0`` the identity IS the null."""
    from wildfire_nowcast.sim.collapse import (  # noqa: PLC0415
        analytic_shared_latent_index,
        increment_dispersion_index,
    )

    rng = np.random.default_rng(7)
    p = rng.uniform(0.15, 0.85, size=600)
    assert analytic_shared_latent_index(p, 0.0) == 1.0

    sigma = 0.8
    exact = analytic_shared_latent_index(p, sigma)
    z = rng.normal(0.0, sigma, size=(600, 1))
    q = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p))[None, :] + z)))
    draws = rng.random((600, 600)) < q
    sampled = increment_dispersion_index(draws.reshape(600, 20, 30))
    assert abs(sampled / exact - 1.0) < 0.15, (
        f"the identity says {exact:.4f} and 600 members sampled {sampled:.4f}; an identity that "
        "does not reproduce its own sampling distribution is not an identity"
    )


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

    Regression: this reported `max_front_gap_km = 24` for BOTH Zogg and CZU -
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
    two of five fires - hiding a good CZU result and passing a weak Zogg one.
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


@_functools.lru_cache(maxsize=1)
def _synthetic_tensor_path() -> str:  # noqa: ANN202
    """Build the synthetic tensor ONCE per process and delete it on exit.

    NOT ``mkdtemp`` per call. ``mkdtemp`` never cleans up, this writes a ~4.7 MB
    tensor, four tests call it, and the mutation gate runs the whole suite once
    per mutant. That reached 6,605 directories and 30 GB under this prefix alone
    before anyone looked at a disk. The path is cached; each caller still gets a
    freshly opened dataset, so no test can observe another test's mutations.
    """
    from wildfire_nowcast.common.synthetic import make_synthetic_fire  # noqa: PLC0415

    tmp = _tempfile.mkdtemp(prefix="simviz-selftest-")
    _atexit.register(_shutil.rmtree, tmp, ignore_errors=True)
    res = make_synthetic_fire(seed=0, n_hours=24, out=f"{tmp}/tensor.zarr")
    return str(res[0])


def _open_synthetic():  # noqa: ANN202
    from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: PLC0415

    return open_tensor(_synthetic_tensor_path())


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
    assert np.array_equal(inp.truth[0], np.asarray(ds["fire_state"].values, dtype=np.uint8)[t0 + 1])


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

    # G3's dispersion half must come back GATE, and every metric the contract
    # has retired must come back QUARANTINED. If this ever inverts, the
    # dashboard is about to cite a demoted metric as capability.
    assert classify("area_dispersion_ratio").state == GATE
    # [v2.15, C6.6] `calibration_error` MOVED from GATE to QUARANTINED. It is
    # anti-monotone (Spearman -0.14 against |log(area error)| on M11's ladder,
    # ADR-053 (1)(2)) and may be REPORTED but may not decide a gate. The badge
    # follows `common/null_check.C6_METRICS`, so this line tracks the ruling
    # rather than restating it; asserted in the QUARANTINED direction so a quiet
    # reversal of the ruling is still caught here. It was edited alongside the
    # v2.15 contract bump that made the move, and it carries no behaviour of its
    # own beyond the assertion.
    assert classify("band_calibration_error_by_horizon").state == QUARANTINED
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
        "1": 0.04,
        "2": 0.05,
        "3": 0.06,
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
                ties = [(y - ys[j], x - xs[j]) for j in range(len(ys)) if int(d2[j]) == best]
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
        WindowGrowth(
            "f",
            "m",
            0,
            0,
            truth_new=4.0,
            pred_new=6.0,
            truth_grew=True,
            wind_speed=3.0,
            truth_sector=[2.0, 1.0, 1.0],
            pred_sector=[3.0, 2.0, 1.0],
        ),
        WindowGrowth(
            "f",
            "m",
            1,
            2,
            truth_new=0.0,
            pred_new=5.0,
            truth_grew=False,
            wind_speed=3.0,
            truth_sector=[0.0, 0.0, 0.0],
            pred_sector=[1.0, 3.0, 1.0],
        ),
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
    """A missing binary must raise, never fall back to another model.

    **This test could not fail before.** It called ``find_binary`` on a path that
    does not exist, caught ``ElmfireNotInstalled`` and passed, caught anything
    else and passed, and had no ``else`` - so a ``find_binary`` that RETURNED a
    substitute, which is the one behaviour the name of this test forbids, went
    straight through it. Same family as the plant that renamed ``--render`` to
    ``--render-DISABLED`` and was still a substring (ADR-104 (5)).

    It now asserts in both worlds, because which world we are in depends on
    whether a vendored ELMFIRE happens to be built on this disk:

    * nothing installed  -> the call MUST raise ``ElmfireNotInstalled``;
    * something installed -> the call may return that binary, and what is then
      forbidden is returning the path we asked for, which does not exist. A
      returned path must exist and be executable.

    The branch that could not be exercised here is LOGGED rather than left
    implicit (ADR-097 (3): a check that excludes a case says so).
    """
    from wildfire_nowcast.sim.elmfire import ElmfireNotInstalled, find_binary

    missing = "/nonexistent/elmfire-binary-that-does-not-exist"
    try:
        installed: Path | None = find_binary()
    except ElmfireNotInstalled:
        installed = None

    if installed is None:
        try:
            got = find_binary(missing)
        except ElmfireNotInstalled:
            return
        raise AssertionError(
            f"no ELMFIRE is installed, yet find_binary({missing!r}) returned {got}. "
            "A silently substituted baseline is how a gate gets decided against nothing."
        )

    logger.warning(
        "an ELMFIRE binary is installed at %s, so the RAISING half of this test cannot "
        "run on this machine; asserting the substitution rule instead",
        installed,
    )
    got = find_binary(missing)
    assert got != Path(missing), f"find_binary returned the missing path {missing} itself"
    assert got.exists() and os.access(got, os.X_OK), (
        f"find_binary fell back to {got}, which is not an executable file"
    )


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


# -- E1: ELMFIRE against stage_decay (ADR-064) -----------------------------


def _e1_rows(values: dict[int, tuple[list[float], list[float]]]) -> list[dict[str, object]]:
    """Synthetic window rows: ``{block: (truth_growth, model_growth)}``."""
    rows: list[dict[str, object]] = []
    for block, (truth, model) in values.items():
        for t0, (tg, mg) in enumerate(zip(truth, model, strict=True)):
            rows.append(
                {
                    "fire_id": f"fire_{block}",
                    "spatial_block_id": block,
                    "t0": t0,
                    "truth_growth": float(tg),
                    "model_growth": float(mg),
                    "model_growth_by_member_prefix": {"1": float(mg), "2": float(mg)},
                }
            )
    return rows


def _e1_write(tmp: Path, rows: list[dict[str, object]]) -> list[Path]:
    import json  # noqa: PLC0415
    from itertools import groupby  # noqa: PLC0415

    paths: list[Path] = []
    key = lambda r: (r["fire_id"], r["spatial_block_id"])  # noqa: E731
    for (fire_id, block), group in groupby(sorted(rows, key=key), key=key):
        chunk = list(group)
        path = tmp / f"e1_rows_{fire_id}.json"
        path.write_text(
            json.dumps(
                {
                    "fire_id": fire_id,
                    "spatial_block_id": block,
                    "n_rows": len(chunk),
                    "n_windows_expected": len(chunk),
                    "n_members": 2,
                    "elapsed_s": 0.0,
                    "rows": chunk,
                }
            )
        )
        paths.append(path)
    return paths


def test_e1_domain_slice_is_exact_and_a_one_cell_shift_is_not() -> None:
    """The whole-domain fetch must be EXACTLY the per-window fetch, and the
    comparison must be able to fail. Network-free: the arithmetic is what is
    under test, not LFPS."""
    from wildfire_nowcast.common.grid import Grid
    from wildfire_nowcast.sim.coarsen import fine_grid
    from wildfire_nowcast.sim.elmfire import window_grids
    from wildfire_nowcast.sim.elmfire_stage import DomainStack
    from wildfire_nowcast.sim.landfire import NATIVE_LAYERS

    refine = 4
    coarse = Grid(x_min=0.0, y_max=10_000.0, nx=10, ny=8, cell_size_m=1000.0, crs="EPSG:5070")
    fine = fine_grid(coarse, refine)
    rng = np.random.default_rng(0)
    layers = {
        lay.stub: rng.integers(0, 200, size=fine.shape).astype(np.int16) for lay in NATIVE_LAYERS
    }
    domain = DomainStack("synthetic", coarse, fine, refine, layers, {"source": "synthetic"})

    x0 = np.zeros(coarse.shape, dtype=np.uint8)
    x0[4, 5] = 1
    window = window_grids(coarse, x0, reach_cells=2, refine=refine)
    sliced = domain.slice_for(window)
    r0, c0 = window.row0 * refine, window.col0 * refine
    for stub, arr in sliced.layers.items():
        expected = layers[stub][r0 : r0 + window.fine.ny, c0 : c0 + window.fine.nx]
        assert np.array_equal(arr, expected), f"{stub}: slice is not the sub-grid"
        shifted = layers[stub][r0 + 1 : r0 + 1 + window.fine.ny, c0 : c0 + window.fine.nx]
        assert not np.array_equal(arr, shifted), (
            f"{stub}: a one-cell-shifted slice compares EQUAL, so the check cannot fail"
        )
    assert sliced.grid.x_min == window.fine.x_min
    assert sliced.grid.y_max == window.fine.y_max


def test_e1_refuses_a_mismatched_refine_rather_than_misregistering() -> None:
    """A refine mismatch must RAISE. Silently slicing at the wrong lattice
    misregisters every fuel cell and would look like a physics result."""
    import pytest  # noqa: PLC0415

    from wildfire_nowcast.common.grid import Grid
    from wildfire_nowcast.sim.coarsen import fine_grid
    from wildfire_nowcast.sim.elmfire import window_grids
    from wildfire_nowcast.sim.elmfire_stage import DomainStack

    coarse = Grid(x_min=0.0, y_max=6_000.0, nx=6, ny=6, cell_size_m=1000.0, crs="EPSG:5070")
    domain = DomainStack(
        "synthetic",
        coarse,
        fine_grid(coarse, 4),
        4,
        {"dem": np.zeros(fine_grid(coarse, 4).shape, dtype=np.int16)},
        {},
    )
    x0 = np.zeros(coarse.shape, dtype=np.uint8)
    x0[3, 3] = 1
    with pytest.raises(ValueError, match="refine"):
        domain.slice_for(window_grids(coarse, x0, reach_cells=1, refine=3))


def test_the_head_distance_field_agrees_with_a_brute_force_over_every_burned_cell() -> None:
    """The boundary restriction is an OPTIMISATION and must change no value.

    :func:`~wildfire_nowcast.sim.headrate.distance_to_burned_fine` scans only the
    blocks on the burned set's boundary. That is correct, and correctness by
    argument is what a plausible-but-wrong number looks like on the way in, so it
    is compared against a scan over every burned fine cell. The comparison is
    then shown to be CAPABLE of failing by dropping blocks from the boundary.
    """
    from wildfire_nowcast.sim.headrate import block_boundary, distance_to_burned_fine

    rng = np.random.default_rng(20260823)
    for refine in (1, 3, 5):
        for _ in range(3):
            coarse = (rng.random((7, 9)) < 0.3).astype(np.uint8)
            if not coarse.any():
                coarse[3, 4] = 1
            got = distance_to_burned_fine(coarse, refine)
            burned_fine = np.repeat(np.repeat(coarse > 0, refine, axis=0), refine, axis=1)
            by, bx = np.nonzero(burned_fine)
            qy, qx = np.nonzero(~burned_fine)
            brute = np.zeros(burned_fine.shape, dtype=np.float64)
            d2 = (qy[:, None] - by[None, :]) ** 2 + (qx[:, None] - bx[None, :]) ** 2
            brute[qy, qx] = np.sqrt(d2.min(axis=1))
            assert np.allclose(got, brute), (refine, float(np.abs(got - brute).max()))

    # PLANT: the same comparison against a distance that skips half the boundary
    # must FAIL, or the assertion above proves nothing.
    coarse = np.zeros((7, 9), dtype=np.uint8)
    coarse[2:5, 3:6] = 1
    refine = 3
    edge = block_boundary(coarse > 0)
    rows, cols = np.nonzero(edge)
    kept = slice(0, max(1, len(rows) // 2))
    burned_fine = np.repeat(np.repeat(coarse > 0, refine, axis=0), refine, axis=1)
    qy, qx = np.nonzero(~burned_fine)
    partial = np.zeros(burned_fine.shape, dtype=np.float64)
    r0 = rows[kept] * refine
    c0 = cols[kept] * refine
    dy = np.maximum(
        np.maximum(r0[None, :] - qy[:, None], qy[:, None] - (r0[None, :] + refine - 1)), 0
    )
    dx = np.maximum(
        np.maximum(c0[None, :] - qx[:, None], qx[:, None] - (c0[None, :] + refine - 1)), 0
    )
    partial[qy, qx] = np.sqrt((dy * dy + dx * dx).min(axis=1))
    assert not np.allclose(partial, distance_to_burned_fine(coarse, refine)), (
        "dropping boundary blocks did not change the field, so the agreement above "
        "was not evidence of anything"
    )


def test_the_head_reads_the_ceiling_when_the_front_runs_at_the_ceiling() -> None:
    """A measurement that only ever reads BELOW a limit has not been shown to work.

    An arrival raster is built by hand at exactly ELMFIRE's crown fire spread
    rate limit and at half of it, so the instrument is checked against a rate it
    is supposed to detect and against one it is supposed to not detect.
    """
    from wildfire_nowcast.sim.headrate import (
        CROWN_FIRE_SPREAD_RATE_LIMIT_KMH as cap,
    )
    from wildfire_nowcast.sim.headrate import (
        distance_to_burned_fine,
        head_advance,
    )

    refine, cell_m, horizon = 10, 100.0, 3
    coarse = np.zeros((3, 40), dtype=np.uint8)
    coarse[1, 0] = 1
    dist = distance_to_burned_fine(coarse, refine)
    burned = np.repeat(np.repeat(coarse > 0, refine, axis=0), refine, axis=1)
    for rate_kmh in (cap, cap / 2.0):
        # arrival = distance / rate, in seconds. Never-reached stays negative.
        km = dist * cell_m / 1000.0
        arrival = np.where(burned, -1.0, km / rate_kmh * 3600.0)
        arrival = np.where(arrival > horizon * 3600.0, -1.0, arrival)
        got = head_advance(arrival, dist, burned, cell_size_m=cell_m, horizon_h=horizon)
        assert abs(got["sustained_head_kmh"] - rate_kmh) < 0.05 * rate_kmh, got
        assert abs(got["max_radial_rate_kmh"] - rate_kmh) < 0.05 * rate_kmh, got
        at_cap = got["share_beyond_floor_at_90pct_of_cap"]
        if rate_kmh == cap:
            assert at_cap > 0.9, ("a front running at the limit must be seen there", got)
        else:
            assert at_cap == 0.0, ("a front at half the limit must not read at it", got)


def test_the_head_is_measured_from_the_t0_state_and_not_from_the_final_one() -> None:
    """The regression this instrument was born with, kept as a test.

    The first version took the burned mask from a variable the lead loop rebinds
    to "burned by this lead". Handed that mask, the reached set is empty and the
    head reads 0.0 km beside tens of thousands of new cells. Zero is a plausible
    number for a fire that did not move, which is why ``predict`` cross-checks
    the reached count against its own new-cell count instead of trusting it.
    """
    from wildfire_nowcast.sim.headrate import distance_to_burned_fine, head_advance

    refine, cell_m, horizon = 5, 200.0, 3
    coarse = np.zeros((3, 20), dtype=np.uint8)
    coarse[1, 0] = 1
    dist = distance_to_burned_fine(coarse, refine)
    initial = np.repeat(np.repeat(coarse > 0, refine, axis=0), refine, axis=1)
    km = dist * cell_m / 1000.0
    arrival = np.where(initial, -1.0, km / 1.0 * 3600.0)
    arrival = np.where(arrival > horizon * 3600.0, -1.0, arrival)
    right = head_advance(arrival, dist, initial, cell_size_m=cell_m, horizon_h=horizon)
    assert right["head_km_by_lead"][-1] > 0.0 and right["n_reached_fine_cells"] > 0

    final = initial | (arrival >= 0.0)
    wrong = head_advance(arrival, dist, final, cell_size_m=cell_m, horizon_h=horizon)
    assert wrong["head_km_by_lead"] == [0.0, 0.0, 0.0] and wrong["n_reached_fine_cells"] == 0, (
        "the defect no longer reproduces, so this test no longer guards anything"
    )


def test_a_supplied_stack_is_cut_to_the_window_and_left_alone_when_it_already_is_one() -> None:
    """``ElmfireConfig.stack`` is a WHOLE-DOMAIN stack and ``predict`` runs on a
    WINDOW. Slicing it is the repair; the case where the window is the whole
    domain must stay the identity, because that is the shipped playthrough and a
    landed result depends on it being unchanged."""
    from wildfire_nowcast.common.grid import Grid
    from wildfire_nowcast.sim.coarsen import fine_grid
    from wildfire_nowcast.sim.elmfire import slice_stack_to_window, window_grids
    from wildfire_nowcast.sim.landfire import NativeStack

    refine = 4
    domain = Grid(x_min=0.0, y_max=20_000.0, nx=20, ny=20, cell_size_m=1000.0, crs="EPSG:5070")
    fine = fine_grid(domain, refine)
    layers = {"dem": np.arange(fine.ny * fine.nx, dtype=np.int16).reshape(fine.shape)}
    stack = NativeStack(grid=fine, layers=layers, provenance={"scope": "whole domain"})

    x0 = np.zeros(domain.shape, dtype=np.uint8)
    x0[10, 10] = 1
    whole = window_grids(domain, x0, reach_cells=40, refine=refine)
    assert whole.coarse.shape == domain.shape
    assert slice_stack_to_window(stack, whole, domain) is stack, (
        "when the window is the whole domain the slice must be the identity OBJECT"
    )

    small = window_grids(domain, x0, reach_cells=2, refine=refine)
    assert small.coarse.shape != domain.shape
    cut = slice_stack_to_window(stack, small, domain)
    assert cut.grid.shape == small.fine.shape
    assert cut.grid.x_min == small.fine.x_min and cut.grid.y_max == small.fine.y_max
    r0, c0 = small.row0 * refine, small.col0 * refine
    expected = layers["dem"][r0 : r0 + small.fine.ny, c0 : c0 + small.fine.nx]
    assert np.array_equal(cut.layers["dem"], expected), (
        "the slice is at the wrong offset, which is a wrong georeference and would "
        "read as a physics result"
    )


def test_a_stack_that_cannot_be_reconciled_is_refused_and_the_message_names_the_cure() -> None:
    """The carried defect was a shape mismatch raised AFTER the simulator ran.

    Three unusable stacks, one control. Each refusal must name ``stack_provider``,
    because the whole cost of this defect was that the message did not say what
    to do about it.
    """
    import pytest  # noqa: PLC0415

    from wildfire_nowcast.common.grid import Grid
    from wildfire_nowcast.sim.coarsen import fine_grid
    from wildfire_nowcast.sim.elmfire import (
        StackWindowError,
        slice_stack_to_window,
        window_grids,
    )
    from wildfire_nowcast.sim.landfire import NativeStack

    refine = 4
    domain = Grid(x_min=0.0, y_max=20_000.0, nx=20, ny=20, cell_size_m=1000.0, crs="EPSG:5070")
    fine = fine_grid(domain, refine)
    x0 = np.zeros(domain.shape, dtype=np.uint8)
    x0[10, 10] = 1
    window = window_grids(domain, x0, reach_cells=2, refine=refine)

    def _stack(grid: Grid) -> NativeStack:
        return NativeStack(
            grid=grid,
            layers={"dem": np.zeros(grid.shape, dtype=np.int16)},
            provenance={"plant": "yes"},
        )

    # CONTROL: the shipped shape must be accepted, or the three refusals below
    # would only prove that the function refuses everything.
    slice_stack_to_window(_stack(fine), window, domain)
    slice_stack_to_window(_stack(domain), window, domain)

    too_small = _stack(
        fine_grid(Grid(x_min=0.0, y_max=4_000.0, nx=4, ny=4, cell_size_m=1000.0), refine)
    )
    out_of_register = _stack(
        fine_grid(Grid(x_min=125.0, y_max=20_000.0, nx=20, ny=20, cell_size_m=1000.0), refine)
    )
    wrong_lattice = _stack(Grid(x_min=0.0, y_max=20_000.0, nx=40, ny=40, cell_size_m=500.0))
    for plant in (too_small, out_of_register, wrong_lattice):
        with pytest.raises(StackWindowError, match="stack_provider"):
            slice_stack_to_window(plant, window, domain)


def test_e1_reads_the_verdict_off_the_pre_registered_rule_in_both_directions() -> None:
    """ADR-064 (4) fixes the rule BEFORE the run: >=4/5 positive -> the defect is
    the model class's; >=4/5 negative -> E-P1 refuted; 3/5 -> not_a_verdict. A
    rule that can only return one of those is not a rule."""
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.elmfire_stage import HELD_OUT_BLOCKS, score

    up = [1.0] * 6 + [4.0] * 6  # accelerating
    down = [4.0] * 6 + [1.0] * 6  # decelerating
    cases = {
        "E-P1_HELD_defect_belongs_to_the_model_class": [up] * 5,
        "E-P1_REFUTED_the_defect_is_ours": [down] * 5,
        "not_a_verdict": [up, up, up, down, down],
    }
    for expected, shapes in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            rows = _e1_rows(
                {b: (down, shape) for b, shape in zip(HELD_OUT_BLOCKS, shapes, strict=True)}
            )
            got = score(_e1_write(Path(tmp), rows))
            assert got["verdict"] == expected, (
                f"{expected} case returned {got['verdict']} ({got['n_blocks_positive']}/5 positive)"
            )
            assert got["complete"] is True


def test_e1_a_partial_run_is_self_identifying_and_names_its_fires() -> None:
    """The expensive failure mode is a half-run someone later reads as finished."""
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.elmfire_stage import score

    up = [1.0] * 6 + [4.0] * 6
    with tempfile.TemporaryDirectory() as tmp:
        rows = _e1_rows({4: (up, up), 5: (up, up)})
        got = score(_e1_write(Path(tmp), rows))
    assert got["verdict"] == "not_a_verdict"
    assert got["complete"] is False
    assert "not_a_verdict" in got
    assert got["blocks_scored"] == [4, 5]
    assert "fire_4" in got["not_a_verdict"] and "fire_5" in got["not_a_verdict"], (
        "a partial run must NAME the fires that exist, not merely count them"
    )


def test_e1_scores_every_block_at_the_same_ensemble_size() -> None:
    """Blocks run at different member counts must not be compared across them."""
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.elmfire_stage import score

    up = [1.0] * 6 + [4.0] * 6
    with tempfile.TemporaryDirectory() as tmp:
        paths = _e1_write(Path(tmp), _e1_rows({4: (up, up), 5: (up, up)}))
        rich = json.loads(paths[0].read_text())
        rich["n_members"] = 4
        for row in rich["rows"]:
            # member 4 would read a DIFFERENT growth; the headline must not use it
            row["model_growth"] = 99.0
            row["model_growth_by_member_prefix"]["4"] = 99.0
        paths[0].write_text(json.dumps(rich))
        got = score(paths)
    assert got["headline_member_prefix"] == 2
    assert got["n_members_as_run"] == [2, 4]
    assert got["per_block"]["4"]["elmfire"] == got["per_block"]["5"]["elmfire"], (
        "the 4-member fire was read at 4 members while the other was read at 2"
    )


def test_e1_does_not_carry_a_second_copy_of_the_estimator() -> None:
    """C0 / ADR-064 (6): ELMFIRE is scored with ``eval/stage.py`` UNCHANGED. A
    local re-implementation is how one comparison becomes two measurements."""
    import inspect  # noqa: PLC0415

    from wildfire_nowcast.eval import stage as stage_module
    from wildfire_nowcast.sim import elmfire_stage

    source = inspect.getsource(elmfire_stage)
    banned = "def " + "stage_decay"
    assert banned not in source, (
        "sim/elmfire_stage.py defines its own stage_decay — that is a second "
        "implementation of the estimand and a C0 breach"
    )
    assert elmfire_stage.stage_decay_by_block is stage_module.stage_decay_by_block
    assert "log(" not in source.replace("log(mean", ""), (
        "the estimand's arithmetic appears to be spelled out here rather than imported"
    )


def test_e1_records_the_registry_refusing_stage_decay_a_gate() -> None:
    """``stage_decay`` may not decide a gate, and the artifact must SAY SO."""
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.elmfire_stage import score

    up = [1.0] * 6 + [4.0] * 6
    with tempfile.TemporaryDirectory() as tmp:
        got = score(_e1_write(Path(tmp), _e1_rows({4: (up, up)})))
    assert got["licence"]["may_adjudicate"] is False
    assert got["licence"]["outcome"] == "NOT_LICENSED"
    assert "G5" in got["not_a_gate"]


def test_e1_decides_only_what_the_missing_blocks_cannot_change() -> None:
    """ADR-064 (4)'s rule is a COUNT with a threshold, so 4 positive of 4 scored
    already meets >=4/5 whatever the fifth does - and 3 of 4 does not. Both
    directions must be right, or the rule is either timid or over-claiming."""
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.elmfire_stage import score

    up = [1.0] * 6 + [4.0] * 6
    down = [4.0] * 6 + [1.0] * 6
    with tempfile.TemporaryDirectory() as tmp:
        four_up = score(
            _e1_write(
                Path(tmp), _e1_rows({4: (down, up), 5: (down, up), 6: (down, up), 7: (down, up)})
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        three_up = score(
            _e1_write(
                Path(tmp), _e1_rows({4: (down, up), 5: (down, up), 6: (down, up), 7: (down, down)})
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        four_down = score(
            _e1_write(
                Path(tmp),
                _e1_rows({4: (down, down), 5: (down, down), 6: (down, down), 7: (down, down)}),
            )
        )

    assert four_up["complete"] is False
    assert four_up["verdict_determined_without_the_missing_blocks"] is True
    assert four_up["verdict"] == "E-P1_HELD_defect_belongs_to_the_model_class"
    assert four_up["blocks_missing"] == [12]

    assert three_up["verdict"] == "not_a_verdict", (
        "3 positive of 4 is NOT determined — the fifth block decides it, and "
        "reading a verdict here would be over-claiming"
    )
    assert three_up["verdict_determined_without_the_missing_blocks"] is False

    assert four_down["verdict"] == "E-P1_REFUTED_the_defect_is_ours"


def test_e1_refuses_to_score_a_fire_that_did_not_finish() -> None:
    """A prefix of a fire's life is a DIFFERENT estimand, not a noisier one:
    stage_decay splits at the block's own median age. Scoring a truncated fire
    would report an early-vs-earlier contrast as if it were the whole life."""
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.elmfire_stage import score

    up = [1.0] * 6 + [4.0] * 6
    with tempfile.TemporaryDirectory() as tmp:
        paths = _e1_write(Path(tmp), _e1_rows({4: (up, up), 5: (up, up)}))
        truncated = json.loads(paths[0].read_text())
        truncated["n_windows_expected"] = truncated["n_rows"] + 40
        paths[0].write_text(json.dumps(truncated))
        got = score(paths)
    assert got["blocks_scored"] == [5], "a truncated fire was scored as if complete"
    assert got["per_fire"]["fire_4"]["scored"] is False
    assert "refused" in got["per_fire"]["fire_4"]


def test_e1_refuses_a_window_that_may_have_hit_elmfire_s_wall_clock_cap() -> None:
    """ELMFIRE's ``MAX_RUNTIME`` is WALL-CLOCK and its abort is SILENT
    (``elmfire_level_set.f90:1263`` sets ``T = TSTOP + 1``). A capped window
    under-reports growth on exactly the late, large windows, which moves
    ``stage_decay`` toward deceleration. It must be refused, not averaged in."""
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.elmfire_stage import TRUNCATION_FRACTION, score

    up = [1.0] * 6 + [4.0] * 6
    with tempfile.TemporaryDirectory() as tmp:
        paths = _e1_write(Path(tmp), _e1_rows({4: (up, up), 5: (up, up)}))
        for path in paths:
            blob = json.loads(path.read_text())
            blob["elmfire_max_runtime_s"] = 600.0
            for row in blob["rows"]:
                row["_elapsed_s"] = 100.0  # 50 s/member on 2 members: well clear
            path.write_text(json.dumps(blob))
        clean = score(paths)
        assert clean["blocks_scored"] == [4, 5], "a clearly-safe run was refused"
        assert clean["per_fire"]["fire_4"]["max_runtime_margin"] > 2.0

        capped = json.loads(paths[0].read_text())
        capped["rows"][-1]["_elapsed_s"] = 2 * TRUNCATION_FRACTION * 600.0 + 1.0
        paths[0].write_text(json.dumps(capped))
        got = score(paths)
    assert got["blocks_scored"] == [5], "a window at the wall-clock cap was scored"
    assert got["per_fire"]["fire_4"]["n_windows_near_max_runtime"] == 1
    assert "refused" in got["per_fire"]["fire_4"]


def test_e1_page_renders_a_partial_run_as_partial() -> None:
    """A page built from 2 of 5 blocks must SAY 2 of 5 in its own title. A figure
    that looks finished is how a half-run gets quoted as a result."""
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim.e1_report import render
    from wildfire_nowcast.sim.elmfire_stage import score

    up = [1.0] * 6 + [4.0] * 6
    with tempfile.TemporaryDirectory() as tmp:
        payload = score(_e1_write(Path(tmp), _e1_rows({4: (up, up), 5: (up, up)})))
        out = render(payload, Path(tmp) / "page.png")
        assert out.exists() and out.stat().st_size > 5_000
    assert payload["n_blocks_scored"] == 2
    assert payload["verdict"] == "not_a_verdict"


# -- S10: an empty input is an ABSENT MEASUREMENT, never a verdict ---------
#
# Every test below plants the SAME shape in a different instrument: hand it
# nothing and see whether it answers anyway. The one that matters is the first.
# `sim/playthrough.py`'s lobotomised arm is a POSITIVE CONTROL - the instrument
# whose only job is to prove the detector can fire - and over a zero-member
# ensemble all three degeneracy criteria were satisfied vacuously, so it agreed
# with its own declared expectation and the playthrough PASSED. An alarm that
# sounds on nothing cannot attribute anything, and everything downstream of a
# control inherits whatever the control certified.


def test_a_zero_member_ensemble_is_an_absent_measurement_not_a_degenerate_one() -> None:
    """The lobotomy control must not read as fired when nothing ran.

    Both halves are asserted, because either alone is worthless: the empty
    ensemble must be REFUSED, and the measured zero-growth ensemble the control
    actually targets must still come back ``degenerate=True``. A guard that
    silenced both would look identical on the first assertion.
    """
    from wildfire_nowcast.sim.absent import AbsentMeasurementError  # noqa: PLC0415
    from wildfire_nowcast.sim.playthrough import degeneracy_verdict  # noqa: PLC0415

    x0 = np.zeros((8, 8), dtype=np.uint8)
    x0[:4] = 1

    for why, empty in (
        ("no members", np.zeros((0, 3, 8, 8), dtype=np.uint8)),
        ("no lead steps", np.zeros((6, 0, 8, 8), dtype=np.uint8)),
    ):
        try:
            got = degeneracy_verdict("empty", empty, x0, absolute_floor=10.0)
        except AbsentMeasurementError as exc:
            assert "NOTHING EXAMINED" in str(exc), str(exc)
        else:
            raise AssertionError(
                f"an ensemble with {why} returned degenerate={got.degenerate}. The "
                "lobotomised arm declares expect_degenerate=True, so this value makes "
                "the positive control agree with its expectation having measured nothing."
            )

    silent = np.repeat(x0[None, None], 4, axis=0).repeat(3, axis=1).copy()
    v = degeneracy_verdict("silent", silent, x0, absolute_floor=10.0)
    assert v.degenerate and v.d1_zero_growth, v.as_dict()
    assert v.n_members == 4 and v.n_lead_steps == 3, (
        "the denominator must travel with the verdict into the artifact"
    )
    assert v.as_dict()["n_members"] == 4


def test_the_non_degeneracy_playthrough_refuses_a_harness_with_no_arms() -> None:
    """``all([])`` is True, so an emptied ARMS would PASS over zero experiments."""
    from wildfire_nowcast.sim import playthrough as PT  # noqa: PLC0415
    from wildfire_nowcast.sim.absent import AbsentMeasurementError  # noqa: PLC0415

    kept = PT.ARMS
    try:
        PT.ARMS = ()
        try:
            PT.run_playthrough()
        except AbsentMeasurementError as exc:
            assert "arms" in str(exc), str(exc)
        else:
            raise AssertionError("a playthrough with no arms reported a verdict")
    finally:
        PT.ARMS = kept
    assert PT.ARMS is kept and len(PT.ARMS) >= 2


def test_the_playthrough_cli_writes_no_artifact_when_nothing_was_measured() -> None:
    """Refusal means the file does not appear, not that it appears saying zero.

    ``reports/figures/playthrough_nondegeneracy.json`` asserts by its own name
    that non-degeneracy was tested. Exit code 3, not 1: a caller reading only the
    status must be able to tell "no check ran" from "a check disagreed".
    """
    from wildfire_nowcast.sim import playthrough as PT  # noqa: PLC0415

    out = Path(_synthetic_tensor_path()).parent / "s10_refused.json"
    if out.exists():
        out.unlink()
    kept = PT.ARMS
    try:
        PT.ARMS = ()
        code = PT.main(["--out", str(out)])
    finally:
        PT.ARMS = kept
    assert code == PT.EXIT_NOTHING_EXAMINED == 3, code
    assert not out.exists(), f"{out} was written for a run that measured nothing"


def test_the_collapse_detector_refuses_an_ensemble_with_no_members() -> None:
    """``collapsed`` is the verdict whose positive control is latent_sigma=0.

    It was loud on an empty ensemble only by line ORDER: a reshape forty lines
    below had already been preceded by a vacuous ``mean_pairwise_iou`` of 1.0.
    The stub at ``latent_sigma=0`` must still be caught, or the guard has simply
    turned the detector off.
    """
    from wildfire_nowcast.sim.absent import AbsentMeasurementError  # noqa: PLC0415

    for fn in (ensemble_diagnostics, independence_dispersion_index):
        for empty in (
            np.zeros((0, 3, 6, 6), dtype=np.uint8),
            np.zeros((5, 0, 6, 6), dtype=np.uint8),
        ):
            try:
                fn(empty)
            except AbsentMeasurementError:
                continue
            raise AssertionError(f"{fn.__name__} answered over an empty ensemble")

    # The same constructed independent-per-pixel ensemble
    # `test_collapse_is_detected_and_healthy_is_not` uses, re-asserted here so a
    # guard that silenced the detector instead of guarding it fails in the test
    # that added the guard, not two tests away.
    rng = np.random.default_rng(0)
    n_members, n_cells = 60, 900
    q = rng.uniform(0.2, 0.8, size=n_cells)
    indep = (rng.random((n_members, n_cells)) < q).astype(np.uint8).reshape(n_members, 1, 30, 30)
    diag = ensemble_diagnostics(indep)
    assert diag["collapsed"], (
        f"the independent-per-pixel positive control stopped being detected: {diag}"
    )
    assert diag["n_pairwise_comparisons"] == 400, (
        "the pair count is capped at max_pairs and must be published, because "
        "`ious = [...] or [1.0]` hides a single-member ensemble behind a perfect score"
    )


def test_g3_readiness_cannot_pronounce_a_payload_with_no_models_adjudicable() -> None:
    """It seeded its key map inside the model loop, so no models meant no gaps."""
    from wildfire_nowcast.sim.rundash import g3_readiness  # noqa: PLC0415

    for empty in ({}, {"pooled_heldout": {}}, {"pooled_heldout": None}):
        got = g3_readiness(empty)
        assert not got["adjudicable"], got
        assert got["n_models_examined"] == 0, got
        assert len(got["missing"]) == 2, got

    full = {
        "pooled_heldout": {
            "kernel": {
                "growth_windows": {
                    "area_dispersion_ratio": 0.58,
                    "band_calibration_error": {"1": 0.04},
                }
            }
        }
    }
    assert g3_readiness(full)["adjudicable"], "a complete artifact must still read adjudicable"


def test_the_coarsening_playthrough_refuses_a_harness_with_no_planted_defects() -> None:
    """A playthrough that planted nothing catches all zero of its defects."""
    from wildfire_nowcast.sim import coarsen as CO  # noqa: PLC0415
    from wildfire_nowcast.sim.absent import AbsentMeasurementError  # noqa: PLC0415

    kept = dict(CO.DEFECTIVE_COARSENERS)
    try:
        CO.DEFECTIVE_COARSENERS.clear()
        try:
            CO.run_playthrough()
        except AbsentMeasurementError as exc:
            assert "planted_defects" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a coarsening playthrough with no planted defects reported a verdict; "
                "every_planted_defect_caught is vacuously true over an empty dict"
            )
    finally:
        CO.DEFECTIVE_COARSENERS.update(kept)
    assert set(CO.DEFECTIVE_COARSENERS) == set(kept)


def test_auditing_zero_plotted_keys_is_not_an_honest_figure() -> None:
    """An empty violations list meant both "all badged" and "none checked"."""
    from wildfire_nowcast.sim.absent import AbsentMeasurementError  # noqa: PLC0415
    from wildfire_nowcast.sim.quarantine import audit_plotted_keys  # noqa: PLC0415

    try:
        audit_plotted_keys({})
    except AbsentMeasurementError:
        pass
    else:
        raise AssertionError("an audit of no keys returned the clean verdict")
    assert audit_plotted_keys({"dispersion_ratio": ""}), "the audit stopped detecting"
    assert audit_plotted_keys({"area_dispersion_ratio": ""}) == []


def test_a_fire_with_no_measurable_step_publishes_no_teleport_statistics() -> None:
    """Four zeros meant "nothing jumped" AND "no hour was examined"."""
    from wildfire_nowcast.sim.movie import gap_summary  # noqa: PLC0415

    keys = ("detached_steps", "teleport_steps", "max_front_gap_km", "median_front_advance_cells")

    nothing = gap_summary([], 1.0, 3)
    assert nothing["n_gap_steps"] == 0
    for k in keys:
        assert k not in nothing, f"{k} was published for a fire with no step to measure"

    clean = gap_summary([0, 0, 0, 0], 1.0, 3)
    assert clean["n_gap_steps"] == 4
    assert all(clean[k] == 0 for k in keys), clean

    jumped = gap_summary([0, 1, 5], 1.0, 3)
    assert jumped["teleport_steps"] == 1 and jumped["max_front_gap_km"] == 5.0, jumped


# -- S10: the replay tool can show a PROBABILITY, not only one track -------


def test_the_burn_probability_panel_draws_nothing_where_no_member_burned() -> None:
    """``p == 0`` must not render as the bottom of the colour scale.

    "the ensemble put no weight here" and "the ensemble put its lowest non-zero
    weight here" are different statements, and a dark cell reads as the second.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from wildfire_nowcast.sim.ensemble import draw_burn_probability  # noqa: PLC0415

    geom = plot_extent(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([3000.0, 2000.0, 1000.0]),
        cell_size_m=1000.0,
    )
    prob = np.array([[0.0, 0.25, 0.5], [0.0, 0.0, 1.0], [0.75, 0.0, 0.0]])
    fig, ax = plt.subplots()
    try:
        im = draw_burn_probability(ax, geom, prob)
        drawn = np.asarray(im.get_array())
        assert np.isnan(drawn[prob == 0]).all(), "a zero-probability cell was given a colour"
        assert np.allclose(drawn[prob > 0], prob[prob > 0])
        assert im.get_clim() == (0.0, 1.0), im.get_clim()
    finally:
        plt.close(fig)


def test_replay_draws_the_ensemble_beside_the_best_member() -> None:
    """C6's band IoU is a BEST-MEMBER statistic, so every map has one track in it.

    An IoU of 0.4 from a member the rest agreed with and an IoU of 0.4 from the
    one member that guessed right are opposite findings about a probabilistic
    forecast and the same picture. The layout is asserted rather than the pixels:
    two columns per model plus truth, and one column per model when the panel is
    turned off.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from wildfire_nowcast.sim import replay as RP  # noqa: PLC0415

    ds = _open_synthetic()
    picks = [t for t, _ in [(w.t0, i) for i, w in RP.iter_eval_windows(ds, 3)][:2]]
    gate = RP.GateModels(
        models={"stub": StubEnsemble(latent_sigma=0.6)},
        provenance={"source": "sim.stub_model, a visualisation fixture"},
    )
    out_dir = Path(_synthetic_tensor_path()).parent
    seen: list[int] = []
    real_subplots = plt.subplots

    def _spy(*args: object, **kw: object):  # noqa: ANN202
        if len(args) >= 2:
            seen.append(int(args[1]))  # type: ignore[arg-type]
        return real_subplots(*args, **kw)

    for flag, want_cols in ((True, 3), (False, 2)):
        seen.clear()
        plt.subplots = _spy  # type: ignore[assignment]
        try:
            png = RP.render_small_multiples(
                "synthetic",
                ds,
                gate,
                picks,
                models=["stub"],
                horizon_h=3,
                stride=1,
                n_members=6,
                seed=5,
                band_radius_cells=6,
                out=out_dir / f"s10_replay_{flag}.png",
                show_burn_probability=flag,
            )
        finally:
            plt.subplots = real_subplots  # type: ignore[assignment]
        assert seen and seen[0] == want_cols, (
            f"show_burn_probability={flag} laid out {seen} columns, expected {want_cols}"
        )
        assert png.exists() and png.stat().st_size > 5_000
        png.unlink()


def test_the_s4_one_pager_is_reachable_from_the_command_line() -> None:
    """``render_verdict`` drew the S4 page and nothing invoked it.

    A renderer no entry point can reach is a renderer nobody will run.

    The first version of this test searched ``--help`` output for the string
    ``--render``, and a plant that renamed the flag to ``--render-DISABLED``
    left it PASSING, because the old name is a substring of the new one. It now
    parses an argument list, which a rename breaks.
    """
    from wildfire_nowcast.sim import playthrough as PT  # noqa: PLC0415

    args = PT.build_parser().parse_args(["--render", "one_pager.png"])
    assert args.render == "one_pager.png", args
    assert args.render_playthrough1.endswith("playthrough_coarsening.json"), args
    assert PT.build_parser().parse_args([]).render is None, "rendering must stay opt-in"
    assert callable(PT.render_verdict)


# -- S11: a fallback that used to be silent now says so (ADR-103) ----------


class _Captured(logging.Handler):
    """A handler that keeps records, for asserting that a diagnostic was EMITTED.

    Attached to one named logger and removed in a ``finally``, and it sets no
    level anywhere: WARNING already reaches handlers under the default effective
    level, so this observes the convention rather than configuring around it.
    ADR-103 permits configuration only in ``main``; this configures nothing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture(logger_name: str) -> tuple[logging.Logger, _Captured]:
    captured = _Captured()
    target = logging.getLogger(logger_name)
    target.addHandler(captured)
    return target, captured


def test_a_broken_ffmpeg_makes_the_writer_fall_back_AND_SAY_SO() -> None:
    """The fallback was a `print` from library code; the probe beside it said nothing.

    Two separate defects in ten lines: ``_writer`` printed to stderr from inside a
    library, so a caller could not turn it off or route it, and ``ffmpeg_usable``
    swallowed the exception from its own probe and returned ``False``, which reads
    as "no ffmpeg" when what happened was "ffmpeg is installed and broken". This
    machine's Homebrew ffmpeg is exactly the second case.

    What is asserted is the OBSERVABLE consequence - a GIF path and a WARNING that
    names the file - not the text of a message.
    """
    from wildfire_nowcast.sim import movie as MV  # noqa: PLC0415

    target, captured = _capture("wildfire_nowcast.sim.movie")
    real = MV.ffmpeg_usable
    try:
        MV.ffmpeg_usable = lambda: False  # type: ignore[assignment]
        writer, out = MV._writer(Path("nowhere/fire.mp4"), 4)
    finally:
        MV.ffmpeg_usable = real  # type: ignore[assignment]
        target.removeHandler(captured)

    assert out.suffix == ".gif", out
    assert type(writer).__name__ == "PillowWriter", type(writer).__name__
    warnings = [r for r in captured.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in captured.records]
    assert "fire.gif" in warnings[0].getMessage(), warnings[0].getMessage()
    assert warnings[0].name == "wildfire_nowcast.sim.movie", warnings[0].name


def test_no_module_in_sim_configures_logging_at_import() -> None:
    """The library half of ADR-103, asserted where simviz can see it fail.

    ``tests/test_logging_convention.py`` scans the whole shipped tree for this and
    is the enforcer; this is the same property measured through BEHAVIOUR rather
    than through the parse tree, and it runs in the standalone
    ``python -m wildfire_nowcast.sim.selftest`` where the pytest suite does not.
    Importing every module in this package must leave the root logger's handler
    list exactly as it was.
    """
    import importlib  # noqa: PLC0415
    import pkgutil  # noqa: PLC0415

    from wildfire_nowcast import sim as PKG  # noqa: PLC0415
    from wildfire_nowcast.common.logs import installed_handler  # noqa: PLC0415

    before = list(logging.getLogger().handlers)
    names = [m.name for m in pkgutil.iter_modules(PKG.__path__)]
    assert len(names) >= 20, names
    for name in names:
        importlib.import_module(f"{PKG.__name__}.{name}")
    assert list(logging.getLogger().handlers) == before, (
        "importing sim/ changed the root logger's handlers. A library that installs "
        "a handler decides the format for a program it is not."
    )
    assert installed_handler() is None or installed_handler() in before


# -- C2 vs this package's own component count (S13) -------------------------
#
# The defect these three target is the MIRROR of the usual one: not a check that
# cannot fail, but a check that could not PASS. `--manifest-check` asserted C2's
# `n_ignition_components` against this package's raw detached-topology count and
# so printed a CONTRACT failure on 9 of 21 corpus fires whose stored values
# ADR-019 ratified. Two rules, both defensible, one estimand named C2.


def test_the_c2_number_is_delegated_and_the_topology_count_is_not_named_like_it() -> None:
    """The record may not carry a second quantity under C2's name.

    The synthetic fire is the fixture BECAUSE the two rules disagree on it: its
    generated spot lands ~30 km out and never merges, so the ratified deriver
    counts 2 while the topology walk finds 4 unconnected births. A fixture where
    they agree could not tell a delegated number from a re-implemented one.
    """
    from wildfire_nowcast.data.ignitions import count_ignition_components  # noqa: PLC0415
    from wildfire_nowcast.sim import components as CP  # noqa: PLC0415
    from wildfire_nowcast.sim.reader import load_fire  # noqa: PLC0415

    fire = load_fire(_synthetic_tensor_path())
    result = CP.ignition_components(fire)

    assert "n_ignition_components" not in result, (
        "a key named `n_ignition_components` is back in sim's record. That name belongs to "
        "C2's estimand, which `data.ignitions` owns; exporting a 12 h-window topology count "
        "under it is what made --manifest-check fail on 9 of 21 correct fires (S13)."
    )
    independent = count_ignition_components(fire.state, cell_size_m=fire.geom.cell_size_m)
    assert result["c2_n_ignition_components_derived"] == independent.n_ignition_components, (
        "the C2 field in sim's record disagrees with the ratified deriver called directly. "
        "It is supposed to BE the ratified deriver, not agree with it."
    )
    assert result["n_components_detected"] != independent.n_ignition_components, (
        "this fixture no longer separates the two rules, so this test cannot tell a delegated "
        "C2 value from a re-implemented one. Point it at a fire where they disagree before "
        "trusting a green here."
    )
    assert result["c2_derivation"] == CP.C2_DERIVATION
    assert "data.ignitions" in result["c2_derivation"]


def test_the_c2_verdict_names_the_deriver_and_refuses_what_it_cannot_measure() -> None:
    """Verified / WRONG / NOT CHECKABLE - and the third is never silence."""
    from wildfire_nowcast.sim.components import C2_DERIVATION, c2_manifest_verdict  # noqa: PLC0415

    assert c2_manifest_verdict("f", 2, 2) is None

    wrong = c2_manifest_verdict("f", 3, 2)
    assert wrong is not None and "3" in wrong and "2" in wrong and C2_DERIVATION in wrong, wrong

    missing = c2_manifest_verdict("f", None, 2)
    assert missing is not None and "NO n_ignition_components" in missing, missing

    unmeasurable = c2_manifest_verdict("f", 1, None)
    assert unmeasurable is not None and "NOT CHECKED" in unmeasurable, (
        "a store the deriver cannot count must REFUSE. Returning None here would report "
        "'no problem found' for a comparison that never happened."
    )


def test_manifest_check_passes_the_derived_c2_value_and_fails_the_topology_count() -> None:
    """The wiring, end to end, with the S13 defect itself as the plant.

    CONTROL: a manifest carrying the value the ratified deriver produces exits 0.
    PLANT:   the same manifest carrying this module's raw topology count - the
             number `--manifest-check` used to assert C2 against - exits 1.
    If the comparison is ever re-pointed at the topology count, BOTH halves break,
    which is the property that keeps this defect from coming back quietly.
    """
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim import components as CP  # noqa: PLC0415
    from wildfire_nowcast.sim.reader import load_fire  # noqa: PLC0415

    src = Path(_synthetic_tensor_path())
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "fire"
        _shutil.copytree(src.parent, store)  # never touch data/ or the shared fixture
        tensor = store / src.name
        manifest = tensor.parent / "manifest.json"
        payload = json.loads(manifest.read_text())

        fire = load_fire(str(tensor))
        derived = CP.c2_ignition_components(fire)
        topology = CP.ignition_components(fire)["n_components_detected"]
        assert derived != topology, "the plant and the control would be the same number"

        argv = ["--tensor", str(tensor), "--outdir", str(Path(tmp) / "fig"), "--manifest-check"]

        payload["n_ignition_components"] = derived
        manifest.write_text(json.dumps(payload))
        assert CP.main(argv) == 0, (
            "the check refuses a manifest carrying exactly what the ratified deriver derives. "
            "That is the S13 defect: a contract check that cannot pass on correct data."
        )

        payload["n_ignition_components"] = topology
        manifest.write_text(json.dumps(payload))
        assert CP.main(argv) == 1, (
            "a manifest whose C2 integer is genuinely wrong passed. A check that cannot fail "
            "is worth less than no check, because everyone downstream believes it ran."
        )


def _two_rule_disagreement_fire():  # noqa: ANN202
    """CZU in miniature, in memory: a body 21 km out that merges 19 hours later.

    Built rather than loaded because the disagreement has to be GUARANTEED. The
    real CZU is the one fire in the 21-fire corpus where the genealogy rule and
    this package's 12 h window part company, its store is untracked, and a
    fixture that only happens to disagree today is a control that expires
    silently. Here the geometry forces it: the second body is far enough to be a
    ``separate_ignition`` on this page's rule and it merges, so the genealogy
    rule refuses to count it.
    """
    from wildfire_nowcast.sim.reader import FireFrames  # noqa: PLC0415
    from wildfire_nowcast.sim.style import plot_extent  # noqa: PLC0415

    h, w, n_t = 7, 30, 24
    state = np.zeros((n_t, h, w), dtype=np.uint8)
    for t in range(n_t):
        state[t, 2:5, : t + 1] = BURNING  # main body, one column per hour
        if t >= 2:
            state[t, 2:5, 22:24] = BURNING  # detached body, born h2, absorbed h21
    times = np.datetime64("2020-08-16T00", "ns") + np.arange(n_t) * np.timedelta64(1, "h")
    zeros = np.zeros((n_t, h, w), dtype=np.float32)
    return FireFrames(
        fire_id="two_rule_disagreement",
        state=state,
        times=times,
        wind_u=zeros,
        wind_v=zeros,
        barrier=np.zeros((h, w), dtype=bool),
        elevation=np.zeros((h, w), dtype=np.float32),
        geom=plot_extent(np.arange(w) * 1000.0, np.arange(h)[::-1] * 1000.0, cell_size_m=1000.0),
        source="in-memory (sim.selftest)",
        attrs={},
    )


def test_the_c2_value_follows_the_genealogy_rule_and_not_this_page_s_merge_window() -> None:
    """The C2 field must move with ``data/``'s rule even where this page disagrees.

    This test exists because a plant caught the previous one being blind. Swapping
    the delegated value for this module's OWN ignition estimate
    (``candidate_separate_ignitions``) left the synthetic-store test GREEN, because
    on that fire the two happen to read 2. Numeric agreement on one fixture is not
    delegation. Here they are 1 and 2 by construction, so the substitution fails.
    """
    from wildfire_nowcast.data.ignitions import count_ignition_components  # noqa: PLC0415
    from wildfire_nowcast.sim import components as CP  # noqa: PLC0415

    fire = _two_rule_disagreement_fire()
    result = CP.ignition_components(fire)
    independent = count_ignition_components(fire.state, cell_size_m=fire.geom.cell_size_m)

    assert independent.n_ignition_components == 1, (
        "the fixture stopped separating the rules: the genealogy rule no longer reads 1 on a "
        "body that merges, so a green here would prove nothing."
    )
    assert result["candidate_separate_ignitions"] == 2, (
        "the fixture stopped separating the rules: this page no longer reads 2 on a body that "
        "stays detached past its merge window."
    )
    assert result["c2_n_ignition_components_derived"] == 1, (
        "the C2 field followed THIS module's merge window instead of the genealogy rule "
        f"({CP.C2_DERIVATION}). On CZU that substitution reads 2 against a stored, ratified 1 "
        "and puts --manifest-check back where S13 found it."
    )
    assert (
        CP.c2_manifest_verdict(fire.fire_id, 1, result["c2_n_ignition_components_derived"]) is None
    )
    assert CP.c2_manifest_verdict(fire.fire_id, 2, result["c2_n_ignition_components_derived"])


# -- G5: the ELMFIRE cache, and the controls that make its table quotable ---
#
# `wildfire_nowcast.sim.g5` shipped ruff-clean and mypy-clean with NO suite behind
# it. That is a weaker state than it looks: the tree-wide scans that would have
# caught its two unresolvable citations read the GIT INDEX, so an UNTRACKED
# module is invisible to them and a green suite says nothing whatever about it.
# These tests are what that module needs before it can be described as tested.
#
# Every one of them targets a REFUSAL. The module's value is not that it can
# replay an ensemble - it is that it declines to replay the wrong one, and a
# refusal that has never been observed refusing is a comment.


def _g5_window(rng: np.random.Generator, h: int = 6, w: int = 7) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.zeros((h, w), dtype=np.uint8)
    x0[h // 2, w // 2] = BURNING
    weather = rng.normal(size=(3, 5, h, w)).astype(np.float32)
    return x0, weather


def test_g5_window_key_binds_every_argument_it_claims_to_and_no_others() -> None:
    """The cache key is the whole safety argument, so each binding is asserted.

    A hit is only evidence that the stored array belongs to this window if the
    key moves when the window moves. Each of the five bound arguments is
    perturbed by the smallest amount that is still a different call.
    """
    from wildfire_nowcast.sim.g5 import window_key  # noqa: PLC0415

    rng = np.random.default_rng(0)
    x0, weather = _g5_window(rng)
    base = window_key(x0, weather, 4, 3, 11)
    assert window_key(x0.copy(), weather.copy(), 4, 3, 11) == base, (
        "the key is not a pure function of its arguments; a cache keyed on it cannot be "
        "evidence of anything"
    )

    moved_x0 = x0.copy()
    moved_x0[0, 0] = BURNED_OUT
    moved_weather = weather.copy()
    moved_weather[0, 0, 0, 0] += np.float32(1e-3)
    perturbations = {
        "x0": window_key(moved_x0, weather, 4, 3, 11),
        "weather": window_key(x0, moved_weather, 4, 3, 11),
        "n_members": window_key(x0, weather, 5, 3, 11),
        "horizon_h": window_key(x0, weather, 4, 2, 11),
        "seed": window_key(x0, weather, 4, 3, 12),
    }
    collisions = [name for name, key in perturbations.items() if key == base]
    assert not collisions, (
        f"the key does not bind {collisions}; a stored ensemble would be replayed for a "
        "window it was not produced for and nothing downstream could tell"
    )


def test_g5_a_missing_ensemble_raises_instead_of_substituting_silence() -> None:
    """C6.2's failure mode, made unreachable: a miss RAISES and a wrong shape RAISES.

    Both directions matter and only one of them is obvious. A miss returning
    zeros is a baseline that predicts nothing and scores like a careful one. A
    HIT of the wrong shape is worse, because it is a real array from a real
    ELMFIRE run and it belongs to a different grid.
    """
    from wildfire_nowcast.sim.g5 import (  # noqa: PLC0415
        CachedEnsembleModel,
        EnsembleStore,
        MissingEnsemble,
        window_key,
    )

    rng = np.random.default_rng(1)
    x0, weather = _g5_window(rng)
    static = np.zeros((len(STATIC_C5), *x0.shape), dtype=np.float32)
    with _tempfile.TemporaryDirectory(prefix="simviz-g5-") as tmp:
        store = EnsembleStore(Path(tmp))
        model = CachedEnsembleModel(store)
        try:
            model.predict(x0, static, weather, 4, 3, 11)
        except MissingEnsemble as exc:
            assert "Refusing to substitute" in str(exc)
        else:
            raise AssertionError(
                "a window with no stored ensemble was answered rather than refused; "
                "a silently degraded baseline is exactly what C6.2 exists to catch"
            )

        key = window_key(x0, weather, 4, 3, 11)
        store.put(key, np.zeros((4, 3, x0.shape[0], x0.shape[1] + 1), dtype=np.uint8), {})
        try:
            model.predict(x0, static, weather, 4, 3, 11)
        except MissingEnsemble as exc:
            assert "expected" in str(exc)
        else:
            raise AssertionError(
                "a stored ensemble of the WRONG SHAPE was returned; it is a real ELMFIRE "
                "array for a different grid and it would score as this window's forecast"
            )
        assert model.hits == [], "a refused window must not be recorded as a hit"


def test_g5_the_store_round_trips_bit_identically_and_leaves_no_partial_file() -> None:
    """The replay claim is bitwise, and the `.npz` suffix trap is pinned.

    ``np.savez_compressed`` APPENDS ``.npz`` to a path that lacks it, so a temp
    name of ``key.npz.part`` is written as ``key.npz.part.npz`` and the rename
    then fails on a file that does not exist. That defect is recorded in the
    module and it is the kind that only shows up after a long run, so the
    absence of a stray part file is asserted rather than trusted.
    """
    from wildfire_nowcast.sim.g5 import EnsembleStore  # noqa: PLC0415

    rng = np.random.default_rng(2)
    samples = rng.integers(0, 3, size=(4, 3, 6, 7)).astype(np.uint8)
    with _tempfile.TemporaryDirectory(prefix="simviz-g5-") as tmp:
        store = EnsembleStore(Path(tmp))
        path = store.put("deadbeef", samples, {"fire_id": "x", "t0": 7})
        assert path.name == "deadbeef.npz"
        assert np.array_equal(store.get("deadbeef"), samples), "the replay is not bitwise"
        assert store.get("deadbeef").dtype == np.uint8
        assert store.meta("deadbeef") == {"fire_id": "x", "t0": 7}
        assert store.n_stored() == 1 and store.bytes_on_disk() > 0
        leftovers = sorted(p.name for p in Path(tmp).iterdir() if p.name != "deadbeef.npz")
        assert not leftovers, f"the atomic write left {leftovers} behind"

        # Idempotent overwrite: a re-run must replace, not accumulate.
        store.put("deadbeef", samples, {"fire_id": "x", "t0": 7})
        assert store.n_stored() == 1


def test_g5_the_commensurability_control_can_fail_and_refuses_to_pass_without_a_reference() -> None:
    """PLANT. The control that licenses the whole table is put in front of a difference.

    ``_commensurability_control`` is what allows an ELMFIRE column assembled
    outside ``run_baselines`` to be printed beside one produced by it. A control
    that has only ever been run on matching inputs is a control nobody has seen
    work, and this project has shipped four of those.

    The third case is the one that is easy to get wrong: with NO reference the
    function must report that it did not run. Absence of a failure is not a
    pass, and ``ran: False`` is how that is said in a machine-readable way.
    """
    from wildfire_nowcast.sim.g5 import _commensurability_control  # noqa: PLC0415

    class _Cal:
        def to_dict(self) -> dict[str, float]:
            return {"scale": 1.25}

    per_fire = {
        "2020_dolan": {
            "models": {
                "persistence": {"growth_windows": {"band_brier_by_horizon": {"3": 0.5}}},
                "ellipse": {"growth_windows": {"band_brier_by_horizon": {"3": 0.25}}},
                "elmfire": {"growth_windows": {"band_brier_by_horizon": {"3": 0.75}}},
            }
        }
    }
    reference = {
        "kind": "baselines",
        "stride": 16,
        "ellipse_calibration": {"rule_of_record": {"scale": 1.25}},
        "per_fire": {"2020_dolan": {"models": _copy_models(per_fire["2020_dolan"]["models"])}},
    }

    control = _commensurability_control(per_fire, reference, _Cal())
    assert control["ran"] and control["n_not_identical"] == 0
    assert control["calibration_identical"] is True
    assert control["verdict"].startswith("COMMENSURABLE"), control["verdict"]
    assert control["n_checked"] == 2, "only persistence and the ellipse are shared columns"

    # PLANT: one shared number moves in the last decimal place. ELMFIRE's own
    # column is untouched, which is the point - a difference in a column both
    # tables were supposed to reproduce condemns the column that only one of
    # them has.
    planted = json.loads(json.dumps(reference))
    planted["per_fire"]["2020_dolan"]["models"]["ellipse"]["growth_windows"][
        "band_brier_by_horizon"
    ]["3"] = 0.2500000001
    caught = _commensurability_control(per_fire, planted, _Cal())
    assert caught["n_not_identical"] == 1, (
        "the control did not see a shared column move; it cannot license the ELMFIRE column"
    )
    assert caught["verdict"].startswith("NOT COMMENSURABLE"), caught["verdict"]

    absent = _commensurability_control(per_fire, None, _Cal())
    assert absent["ran"] is False and "verdict" not in absent, (
        "a control with no reference reported a verdict; not running must never read as a pass"
    )


def _copy_models(models: dict[str, object]) -> dict[str, object]:
    """Deep copy through JSON, so the reference cannot share a mutable node with the subject."""
    copied: dict[str, object] = json.loads(json.dumps(models))
    return copied


def test_g5_the_gate_columns_carry_the_criterion_and_label_the_barred_one() -> None:
    """G3's columns must name the adjudicating key and mark the quarantined one.

    ``dispersion_ratio`` is carried on purpose - the sentence that states G3 uses
    that word and C6.1 bars that key from adjudicating - so the label beside it
    is doing the whole job of keeping a reader from quoting it.
    """
    from wildfire_nowcast.sim.g5 import G3_COLUMNS, g5_table  # noqa: PLC0415

    labels = dict(G3_COLUMNS)
    assert "band_area_dispersion_ratio" in labels
    assert "CRITERION" in labels["band_area_dispersion_ratio"]
    assert "REPORTED ONLY" in labels["dispersion_ratio"], (
        "the barred dispersion key is printed without saying it is barred"
    )
    assert "REPORTED ONLY" in labels["arrival_crps"]

    payload = {
        "stride": 16,
        "n_members": 4,
        "seed": 20260807,
        "split_fingerprint": "b3e5dadad01eaef9",
        "per_fire": {
            "2020_dolan": {
                "models": {
                    "elmfire": {
                        "growth_windows": {
                            "band_area_dispersion_ratio": 0.0844,
                            "band_brier_by_horizon": {"1": 0.1, "2": 0.2, "3": 0.3},
                        }
                    }
                }
            }
        },
    }
    text = g5_table(payload)
    assert "elmfire" in text and "0.0844" in text
    assert "0.1000/0.2000/0.3000" in text, "the per-horizon column collapsed to one number"
    assert "--" in text, (
        "a per-fire entry with no spatial_block_id must render as `--`; a diagnostic table "
        "that raises on a malformed payload fails exactly when it is wanted"
    )

    with_block = json.loads(json.dumps(payload))
    with_block["per_fire"]["2020_dolan"]["spatial_block_id"] = 6
    assert "    6 2020_dolan" in g5_table(with_block)


# -- the reliability page: a diagram that cannot hide its own sample size ---


def _rel_bins(lead_h: int, counts: list[int], burned: list[int]) -> list[dict[str, object]]:
    """Ten C6-shaped bins with the counts and burn totals a test wants to see drawn."""
    out: list[dict[str, object]] = []
    for i, (n, y) in enumerate(zip(counts, burned, strict=True)):
        lo = i / 10.0
        out.append(
            {
                "lead_h": lead_h,
                "bin_index": i,
                "bin_lower": lo,
                "bin_upper": lo + 0.1,
                "n": n,
                "sum_p": n * (lo + 0.05),
                "sum_y": y,
                "mean_forecast": (lo + 0.05) if n else None,
                "observed_frequency": (y / n) if n else None,
            }
        )
    return out


def test_a_reliability_page_keeps_the_empty_bins_the_curve_drops() -> None:
    """The occupancy axis exists to show an unused bin, so an unused bin must survive.

    ``dashboard.reliability_curve`` drops empty bins and is right to: plotting
    one at (0, 0) draws a perfectly-calibrated point resting on nothing. The
    occupancy row makes the opposite demand of the same data, and a page that
    silently inherited the drop would show nine bins where the forecast used
    seven and never say which two were unused.
    """
    from wildfire_nowcast.sim import reliability as R  # noqa: PLC0415

    counts = [1_000_000, 500, 0, 0, 40, 30, 0, 20, 9, 0]
    burned = [400, 40, 0, 0, 5, 0, 0, 0, 0, 0]
    bins = _rel_bins(3, counts, burned)

    rows = R.bin_rows(bins, 3)
    assert len(rows) == 10, "an empty bin was dropped from the occupancy table"
    assert [r.n for r in rows] == counts
    assert [r.occupied for r in rows] == [n > 0 for n in counts]
    assert rows[2].mean_forecast is None and rows[2].observed_frequency is None, (
        "an empty bin was given a forecast of 0.0; that is a measurement where there is none"
    )

    conc = R.concentration(bins, 3)
    assert conc["n_cells_total"] == sum(counts)
    assert conc["n_occupied_bins"] == 6 and conc["n_bins"] == 10
    assert abs(conc["lowest_bin_share"] - counts[0] / sum(counts)) < 1e-12

    com = R.commitment(bins, 3, 0.5)
    assert com["n_cells"] == 30 + 20 + 9 and com["n_burned"] == 0
    assert com["observed_frequency"] == 0.0
    bound = com["exact_one_sided_upper95"]
    assert bound is not None and 0.0 < bound < 1.0
    assert bound < 3.0 / com["n_cells"], (
        "the exact zero-success bound is not tighter than the rule-of-three approximation "
        "it replaces, so one of the two is wrong"
    )


def test_the_reliability_page_curve_control_agrees_here_and_fails_on_a_plant() -> None:
    """CONTROL PLUS PLANT on the check that says this page holds no second curve."""
    from wildfire_nowcast.sim import reliability as R  # noqa: PLC0415

    bins = _rel_bins(3, [1000, 500, 0, 0, 40, 30, 0, 20, 9, 0], [4, 40, 0, 0, 5, 0, 0, 0, 0, 0])
    agree = R.curve_agreement(bins, 3)
    assert agree["identical"] is True, agree
    assert agree["n_points_here"] == 6 == agree["n_points_dashboard"]
    assert agree["n_points_here"] < len(bins), (
        "every bin is occupied in this fixture, so the agreement check is comparing a curve "
        "that never had anything to drop and proves nothing"
    )

    planted = [dict(b) for b in bins]
    planted[1]["observed_frequency"] = 0.5
    disagree = R.curve_agreement(bins, 3, reference_bins=planted)
    assert disagree["identical"] is False, (
        "the curve control passed against a curve that had been moved; it cannot detect a "
        "second implementation drifting from the first"
    )


def test_the_wilson_interval_contains_its_own_estimate_where_float_error_says_otherwise() -> None:
    """The 1e-17 that matplotlib found and inspection did not.

    At ``k = 0`` the Wilson lower limit is analytically zero: ``centre`` and
    ``half`` are the same quantity and cancel. In floating point the
    cancellation leaves a residual of order 1e-17 with the WRONG SIGN, so the
    interval failed to contain its own point estimate on exactly the zero-burn
    bins this page is about. The raw formula is recomputed here so the guard is
    shown to be load-bearing rather than decorative.
    """
    from wildfire_nowcast.sim.reliability import wilson_interval  # noqa: PLC0415

    z = 1.959963984540054
    residuals = []
    for n in (9, 12, 18, 24, 25, 49, 162):
        denom = 1.0 + z * z / n
        centre = (0.0 + z * z / (2 * n)) / denom
        half = (z / denom) * ((z * z / (4 * n * n)) ** 0.5)
        residuals.append(centre - half)
        lo, hi = wilson_interval(0, n)
        assert lo == 0.0, f"the lower limit at k=0, n={n} is {lo!r}, above its own estimate"
        assert 0.0 < hi < 1.0
    assert any(r > 0 for r in residuals), (
        "the raw formula no longer produces a positive residual at k=0, so this test has "
        "stopped demonstrating why the guard exists"
    )

    lo, hi = wilson_interval(7, 7)
    assert hi == 1.0 and lo < 1.0
    lo, hi = wilson_interval(1, 4)
    assert lo < 0.25 < hi
    assert wilson_interval(0, 0) != wilson_interval(0, 1)


def test_a_commitment_is_counted_off_the_bin_lower_edge_and_never_off_a_straddle() -> None:
    """A cell is never counted as a commitment on the strength of a bin that straddles."""
    from wildfire_nowcast.sim import reliability as R  # noqa: PLC0415

    counts = [10, 10, 10, 10, 111, 7, 0, 0, 0, 0]
    bins = _rel_bins(2, counts, [0, 0, 0, 0, 5, 2, 0, 0, 0, 0])
    at_half = R.commitment(bins, 2, 0.5)
    assert at_half["n_cells"] == 7 and at_half["n_burned"] == 2, (
        "the [0.4, 0.5) bin was counted at a 0.5 threshold; its cells sit below the line"
    )
    assert at_half["bins_used"][0] == [0.5, 0.6], (
        "the lowest bin above the threshold is not the one at the threshold"
    )
    assert len(at_half["bins_used"]) == 5, (
        "the empty bins above the threshold were dropped from bins_used; a reader could not "
        "tell an arm that never used them from an arm for which they do not exist"
    )

    at_four = R.commitment(bins, 2, 0.4)
    assert at_four["n_cells"] == 118 and at_four["n_burned"] == 7
    assert R.commitment(bins, 2, 0.7)["n_cells"] == 0
    assert R.commitment(bins, 2, 0.7)["observed_frequency"] is None, (
        "an arm that never commits scored a frequency; 0/0 is not 0.0"
    )


# -- the episode replay module ---------------------------------------------


def test_episode_seed_offsets_are_the_windows_position_and_differ_by_draw() -> None:
    """The seed index, which decides WHICH ensemble a figure is of.

    Member seeds are ``base + offset``. Draw A's offset is the window's position
    in the evaluable list, draw B's is its position within the growth / dormant
    stratum. Get either wrong and every panel is a valid picture of a
    neighbouring experiment: no crash, no warning, different cells.
    """
    from wildfire_nowcast.sim.episode import seed_offsets_for_fire, window_positions

    # 8 hours, 4x4. Nothing is burned until t=2, so windows 0 and 1 do not exist.
    state = np.zeros((8, 4, 4), dtype=np.uint8)
    state[2:, 0, 0] = BURNING
    # One new cell at t=4. Windows at t0=2 and t0=3 see it arrive; the window at
    # t0=4 already has it in its own x0, so it is a DORMANT window and draw B
    # counts it in the other stratum.
    state[4:, 0, 1] = BURNING
    t0s, pos = window_positions(state, 3)
    assert t0s == [2, 3, 4], f"the evaluable windows moved: {t0s}"
    assert pos == {2: 0, 3: 1, 4: 2}, "draw A's offset is not the position in the filtered list"

    _, a = seed_offsets_for_fire(state, 3, "A")
    _, b = seed_offsets_for_fire(state, 3, "B")
    assert a == pos
    # windows 2 and 3 reach t=5 and grow; window 4 reaches t=7 and does not.
    assert b == {2: 0, 3: 1, 4: 0}, f"draw B did not count within the stratum: {b}"
    assert a != b, "the two draws produced the same seeds; one of them is not being applied"

    try:
        seed_offsets_for_fire(state, 3, "C")
    except ValueError:
        pass
    else:  # pragma: no cover - the refusal is the assertion
        raise AssertionError("an unknown draw was accepted and silently treated as one of ours")


def test_episode_the_cross_check_can_fail_and_a_missing_window_is_not_a_pass() -> None:
    """PLANT. The control that licenses "my numbers match the run's".

    This check is the whole reason a rendered episode is evidence rather than an
    illustration: it asserts that the cells drawn here are the SAME CELLS the
    run recorded, by set equality rather than by count. A control that has only
    ever been shown agreeing is a control nobody has seen work.
    """
    import gzip as _gzip  # noqa: PLC0415

    from wildfire_nowcast.sim.episode import cross_check  # noqa: PLC0415

    page = {
        "windows": [
            {
                "fire_id": "2020_creek",
                "t0": 1234,
                "confident_cells_drawA": [
                    {"row": 15, "col": 46},
                    {"row": 16, "col": 46},
                ],
            }
        ]
    }
    tmp = Path(_tempfile.mkdtemp(prefix="simviz-selftest-episode-"))
    _atexit.register(_shutil.rmtree, tmp, True)
    good = tmp / "cells.json.gz"
    with _gzip.open(good, "wt") as fh:
        json.dump(
            {
                "confident_cells": [
                    {"fire_id": "2020_creek", "t0": 1234, "row": 15, "col": 46},
                    {"fire_id": "2020_creek", "t0": 1234, "row": 16, "col": 46},
                ]
            },
            fh,
        )
    agreed = cross_check(page, good, "A")
    assert agreed["identical_on_every_window"] is True
    assert agreed["windows"][0]["n_mine"] == agreed["windows"][0]["n_artifact"] == 2

    # PLANT 1: same COUNT, one cell moved by one row. Every printable summary
    # agrees and the finding is about a different place.
    moved = tmp / "moved.json.gz"
    with _gzip.open(moved, "wt") as fh:
        json.dump(
            {
                "confident_cells": [
                    {"fire_id": "2020_creek", "t0": 1234, "row": 15, "col": 46},
                    {"fire_id": "2020_creek", "t0": 1234, "row": 17, "col": 46},
                ]
            },
            fh,
        )
    caught = cross_check(page, moved, "A")
    assert caught["identical_on_every_window"] is False, (
        "a moved cell was not seen; the control cannot license a rendered episode"
    )
    assert caught["windows"][0]["n_mine"] == caught["windows"][0]["n_artifact"] == 2, (
        "the plant was supposed to keep the counts equal"
    )
    assert caught["windows"][0]["only_mine"] and caught["windows"][0]["only_artifact"]

    # PLANT 2: the artifact does not carry this window at all. An empty
    # intersection must not read as agreement.
    empty = tmp / "empty.json.gz"
    with _gzip.open(empty, "wt") as fh:
        json.dump({"confident_cells": []}, fh)
    absent = cross_check(page, empty, "A")
    assert absent["identical_on_every_window"] is False, (
        "an artifact with no cells at all agreed with a page that has two"
    )
    assert absent["windows"][0]["n_artifact"] == 0


def test_episode_the_zoom_box_is_never_empty_and_never_leaves_the_domain() -> None:
    """A crop is a claim about where to look, and an empty one is a crash."""
    from wildfire_nowcast.sim.episode import zoom_box  # noqa: PLC0415

    shape = (40, 30)
    nothing = np.zeros(shape, dtype=bool)
    assert zoom_box([nothing], shape) == (0, 40, 0, 30), (
        "with nothing marked the box must fall back to the whole domain rather than collapse"
    )

    one = np.zeros(shape, dtype=bool)
    one[20, 15] = True
    r0, r1, c0, c1 = zoom_box([one], shape, minimum=18)
    assert (r1 - r0) >= 18 and (c1 - c0) >= 18, "a single cell produced a box smaller than asked"
    assert 0 <= r0 < r1 <= 40 and 0 <= c0 < c1 <= 30

    corner = np.zeros(shape, dtype=bool)
    corner[0, 0] = True
    r0, r1, c0, c1 = zoom_box([corner], shape)
    assert r0 >= 0 and c0 >= 0 and r1 <= 40 and c1 <= 30, "the box ran off the domain"
    assert r1 > r0 and c1 > c0


def test_episode_a_rank_with_no_population_behind_it_refuses_to_print_a_number() -> None:
    """``rank 0 of 0`` looks like a measurement. It is the absence of one."""
    from wildfire_nowcast.sim.episode import WindowFacts, _rank  # noqa: PLC0415

    facts = WindowFacts(
        fire_id="f",
        t0=1,
        position=0,
        time_t0="t",
        time_valid="t",
        burned_cells=10,
        band_cells=10,
        truth_growth_cells=1,
        max_wind_ms=1.0,
        mean_wind_ms=1.0,
        min_rh_pct=1.0,
        mean_wind_u=1.0,
        mean_wind_v=0.0,
    )
    assert "NOT COMPUTED" in _rank(0, facts)
    facts.n_growth_windows = 816
    assert _rank(13, facts) == "(rank 13 of 816)"


def test_episode_a_commitment_is_band_only_and_read_at_the_final_lead() -> None:
    """The commitment set is a statement about SCORED cells at the SCORED lead."""
    from wildfire_nowcast.sim.episode import confident_mask  # noqa: PLC0415

    prob = np.zeros((3, 4, 4))
    prob[0, 0, 0] = 1.0  # certain at lead 1, gone by lead 3
    prob[2, 1, 1] = 0.9  # inside the band
    prob[2, 3, 3] = 0.9  # outside the band
    band = np.zeros((4, 4), dtype=bool)
    band[1, 1] = True
    hit = confident_mask(prob, band)
    assert hit[1, 1] and not hit[3, 3], "a cell outside the scored band was counted"
    assert not hit[0, 0], "the mask read a lead other than the final one"
    assert int(hit.sum()) == 1


# -- runner ----------------------------------------------------------------


# -- the G6 report page (S18) ----------------------------------------------
#
# Adopted from `sim/g6_report.py --selftest`, which runs these same six
# predicates against the REAL artifacts and the REAL 1 MiB page. Neither half
# is sufficient. `--selftest` proves the SHIPPED page passes; it can only run
# where `runs/` exists, which is this laptop and nowhere else - not a clone,
# not a detached worktree, not CI. What follows proves each control CAN REFUSE,
# everywhere, with no artifact on disk. **A control that has only ever been
# handed a passing input has not been seen working**, which is the finding the
# report itself leads its last section with, so shipping the page's controls
# without that observation would have been the same defect one level up.
#
# Each is named for the way THE PAGE COULD LIE, not for the function it calls.


def _g6_bar_refusal(fn, *args, **kwargs) -> str:  # noqa: ANN001, ANN002, ANN003
    """Call something that must raise; return the message. Fail loudly if silent."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any refusal is the observation
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError(
        f"the plant did NOT fire: {getattr(fn, '__name__', fn)} accepted "
        f"{args!r} {kwargs!r}. A control observed only passing is not a control."
    )


def test_g6_the_page_cannot_print_a_bar_the_code_has_moved_away_from() -> None:
    """THE LIE: the page quotes a G3 bar that `common/` no longer enforces.

    Our prose has carried ``[0.8, 1.2]`` while the code carried
    ``[0.8333, 1.2]``, and the report says so on the page - so a page that
    retyped either number would reproduce the defect it is reporting. The
    renderer instead compares the interval an artifact STORED for itself
    against ``common.dispersion.BAR_INTERVAL`` read out of the code today, and
    refuses to render on disagreement.

    Planted in four directions, and the fourth is the one that matters. Moving
    the ARTIFACT's endpoint and moving the CODE's constant must both fire - if
    only the first did, the check would be comparing the artifact with itself.
    And the disagreement is planted in **draw B**, through ``load_facts``:
    until S18 the refusal read draw A's stored bar and nothing had ever looked
    at draw B's, while ADR-142 (e) puts draw B's numbers on the page against
    that same bar. A cross-check covering one of two sources has a blind side.
    """
    import dataclasses  # noqa: PLC0415
    import json as _json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.common.dispersion import BAR_INTERVAL  # noqa: PLC0415
    from wildfire_nowcast.sim import g6_report as g6  # noqa: PLC0415

    real = (float(BAR_INTERVAL[0]), float(BAR_INTERVAL[1]))

    # CONTROL: agreement is silent. If this raised, every plant below would
    # "fire" for a reason that has nothing to do with the bar.
    g6.assert_stored_bar_agrees(real, real, where="the control")

    for moved in ((0.8, 1.2), (real[0], 1.25), (0.5, 1.25)):
        msg = _g6_bar_refusal(g6.assert_stored_bar_agrees, moved, real, where="planted artifact")
        assert "BAR_INTERVAL" in msg and "planted artifact" in msg, msg
    # the same disagreement arriving from the CODE side, not the artifact
    msg = _g6_bar_refusal(g6.assert_stored_bar_agrees, real, (0.5, 1.25), where="planted code")
    assert "BAR_INTERVAL" in msg, msg

    def _stub(path: Path, low: float, high: float) -> None:
        path.write_text(
            _json.dumps(
                {
                    "scope": {},
                    "g3": {
                        "bar": [
                            {
                                "key": "band_area_dispersion_ratio",
                                "low": low,
                                "high": high,
                                "source": "a fixture, not a run",
                            }
                        ],
                        "models": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    with tempfile.TemporaryDirectory(prefix="g6_bar_") as raw:
        tmp = Path(raw)
        a, b = tmp / "m30_g3_drawA.json", tmp / "m30_g3_drawB.json"

        def _load(bar_b: tuple[float, float]) -> None:
            _stub(a, *real)
            _stub(b, *bar_b)
            g6.load_facts(
                dataclasses.replace(g6.ArtifactSet.default(tmp), g3_draw_a=a, g3_draw_b=b)
            )

        # CONTROL: both draws agreeing must not produce a BAR refusal. The
        # fixture is deliberately too thin to render, so `load_facts` still
        # fails - on a missing model, which is what we assert it fails on.
        control = _g6_bar_refusal(_load, real)
        assert "BAR_INTERVAL" not in control, (
            f"the fixture itself trips the bar check, so the plant below would "
            f"prove nothing: {control}"
        )
        planted = _g6_bar_refusal(_load, (0.5, 1.25))
        assert "BAR_INTERVAL" in planted, planted
        assert "draw B" in planted and "m30_g3_drawB" in planted, (
            f"the refusal does not name WHICH draw disagreed, so a reader "
            f"cannot find the file: {planted}"
        )


def test_g6_an_absent_artifact_is_refused_not_rendered_as_an_empty_section() -> None:
    """THE LIE: a table quietly vanishes and the page still looks complete.

    Every number on the G6 page is supposed to trace to a named file. The
    failure mode is not a crash - it is a section that renders empty, or a
    heading with nothing under it, on a page that otherwise looks finished. The
    reader has no way to tell a measurement that came out empty from an input
    that was never there. So absence raises, and the message says why.
    """
    import json as _json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from wildfire_nowcast.sim import g6_report as g6  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="g6_absent_") as raw:
        tmp = Path(raw)
        present = tmp / "present.json"
        present.write_text(_json.dumps({"k": 1}), encoding="utf-8")
        assert g6._read_json(present) == {"k": 1}  # CONTROL: a real file reads

        for missing in (tmp / "gone.json", tmp / "gone.json.gz"):
            msg = _g6_bar_refusal(g6._read_json, missing)
            assert msg.startswith("FileNotFoundError"), msg
            assert "traceable to a named artifact" in msg, (
                f"the refusal does not say WHY absence is fatal, so it reads as "
                f"an ordinary missing-file error: {msg}"
            )


def test_g6_the_page_cannot_quietly_depend_on_a_file_the_reader_does_not_have() -> None:
    """THE LIE: the page renders perfectly for its author and degrades for everyone.

    A ``<script>``, a ``<link>``, an ``@import`` or an ``<img>`` pointing at a
    sibling file all resolve on the machine that wrote the page, out of the
    working tree it was written in. Handed to anyone else - or opened after
    ``reports/figures/`` moves - they silently produce a page with no figures
    and no styling. **The defect is invisible to its author by construction**,
    which is precisely why looking at the page is not a check on it.
    """
    from wildfire_nowcast.sim import g6_report as g6  # noqa: PLC0415

    page = (
        "<!doctype html><html><head><style>"
        + g6.CSS
        + "</style></head><body><h1>x</h1>"
        + '<img src="data:image/png;base64,AAAA" alt="a figure">'
        + '<a href="#section-1">jump</a></body></html>'
    )
    g6.assert_page_is_self_contained(page)  # CONTROL: the real stylesheet passes

    plants = {
        "a script tag": page.replace("<body>", "<body><script>1</script>"),
        "a stylesheet link": page.replace("<body>", '<body><link rel="stylesheet" href="a.css">'),
        "a CSS import": page.replace("<style>", '<style>@import url("a.css");'),
        # Both of these are ASSEMBLED at run time. Not style: `tools/cited_paths.py`
        # resolves every path-shaped literal in TRACKED source against the git
        # index, and a plant is a path that must NOT resolve, so spelling one
        # here turns that reader red for being exactly what it is meant to be.
        # The tool names this form itself, under UNSEEN_BY_CONSTRUCTION.
        "a sibling image": page.replace(
            "<body>", '<body><img src="' + "/".join(("figures", "x.png")) + '">'
        ),
        "an outward link": page.replace(
            "<body>", '<body><a href="' + "/".join(("runs", "m30.json.gz")) + '">a</a>'
        ),
    }
    for label, planted in plants.items():
        msg = _g6_bar_refusal(g6.assert_page_is_self_contained, planted)
        assert msg.startswith("SelfCheckFailure"), f"{label}: {msg}"


def test_g6_no_colour_is_defined_only_in_the_dark_block_so_one_mode_goes_blank() -> None:
    """THE LIE: the page is legible in whichever mode its author happened to use.

    An undefined CSS custom property resolves to nothing, so a token defined
    only inside ``@media (prefers-color-scheme: dark)`` or only inside
    ``[data-theme="dark"]`` gives ink-on-nothing to a reader whose machine is
    set the other way. This runs on the module's OWN stylesheet, not on a
    fixture, so it is a live statement about the shipped page's palette - and
    it stays true in a clone, where the page itself cannot be rendered.
    """
    import re as _re  # noqa: PLC0415

    from wildfire_nowcast.sim import g6_report as g6  # noqa: PLC0415

    doc = f"<style>{g6.CSS}</style>"
    g6.assert_colour_tokens_have_a_bare_root_definition(doc)  # CONTROL: the real palette

    # Non-vacuity: if the stylesheet used no var() at all the control above
    # would pass over nothing, which is the exact shape of defect it exists for.
    used = set(_re.findall(r"var\((--[a-z0-9-]+)\)", g6.CSS))
    assert len(used) >= 10, f"the stylesheet uses {len(used)} tokens; too few to be checking"

    msg = _g6_bar_refusal(
        g6.assert_colour_tokens_have_a_bare_root_definition,
        doc.replace("  --ink: #1b1a17;\n", "", 1),
    )
    assert "--ink" in msg, msg
    assert "no stylesheet" in _g6_bar_refusal(
        g6.assert_colour_tokens_have_a_bare_root_definition, "<p>a page with no style block</p>"
    )
    assert "no bare :root block" in _g6_bar_refusal(
        g6.assert_colour_tokens_have_a_bare_root_definition,
        doc.replace(":root {", ':root[data-theme="light"] {', 1),
    )


def test_g6_the_source_cannot_drift_back_into_the_output_literal_sink() -> None:
    """THE LIE: a hand-taken clean verdict stays quoted after it stopped being true.

    ``sim/g6_report.py`` renders every glyph on the page from HTML entities and
    matplotlib mathtext so that the SOURCE stays ASCII. That is not typography
    pedantry: the module's prose and citation verdicts were taken BY HAND
    against its text, because an untracked file is invisible to scanners that
    walk the git index. One character typed directly puts it back into the
    output-literal sink, and the hand reading that said ``0`` is still on the
    record saying ``0``.
    """
    from wildfire_nowcast.sim import g6_report as g6  # noqa: PLC0415

    source = Path(g6.__file__).read_text(encoding="utf-8")
    assert len(source) > 50_000, "the source read back too short to be the module"
    g6.assert_source_is_ascii(source)  # CONTROL: the real file

    for label, ch in (
        ("an em dash", 0x2014),
        ("a non-breaking space", 0x00A0),
        ("a curly quote", 0x2019),
    ):
        msg = _g6_bar_refusal(g6.assert_source_is_ascii, source + "\n# " + chr(ch) + "\n")
        assert "non-ASCII in source" in msg, f"{label}: {msg}"


def test_g6_a_number_on_the_page_cannot_silently_disagree_with_its_artifact() -> None:
    """THE LIE: the page is a second opinion about the artifact rather than a view of it.

    The load-bearing one. A rendered value that drifted from the file it cites
    is undetectable by reading either, and it is the failure the whole report
    would be worthless under. Ten decimal places, not four: the third plant
    below prints the value correct to 4 dp and must still be refused, because a
    number that agrees to the printed precision is exactly what a value from
    the wrong source looks like.
    """
    from wildfire_nowcast.sim import g6_report as g6  # noqa: PLC0415

    v = 0.24652899779252743  # the S17 headline, draw A, equal-block
    g6.assert_page_states_value(f"our kernel reads {v:.10f} against the bar", v)  # CONTROL

    for label, text, value in (
        ("off by 1e-9", f"{v:.10f}", v + 1e-9),
        ("rounded to 4 dp", f"{v:.10f}", round(v, 4) + 1e-9),
        ("the page prints only 4 dp", f"our kernel reads {v:.4f}", v),
        ("the value is absent entirely", "a page with no numbers on it", v),
    ):
        msg = _g6_bar_refusal(g6.assert_page_states_value, text, value)
        assert msg.startswith("SelfCheckFailure"), f"{label}: {msg}"


def test_g6_the_page_cannot_send_a_reader_to_a_file_that_is_not_in_the_repository() -> None:
    """THE LIE: the page prints a path a cloner does not have.

    A path in prose is an instruction. This repository is public and
    ``coordination/`` is not in it, so a page naming a coordination document
    tells a stranger to open something they were never given - in the voice of
    a document whose entire argument is that a claim should rest on what can be
    checked. **Nothing we own could see it.** The literals were joined at run
    time so the tracked-source citation readers were blind by construction, and
    the files are present on the machine that renders the page so a look at the
    output finds them. **That is why this control scans the RENDERED OUTPUT and
    resolves against the git index rather than the filesystem** - ``exists()``
    would have called the page clean.

    Everything below is constructed. No ``runs/`` directory, no git, no real
    page: this half proves the control CAN REFUSE, anywhere; ``--selftest``
    proves the shipped page passes it here.
    """
    from wildfire_nowcast.sim import g6_report as g6  # noqa: PLC0415

    # Assembled at run time, all of them. `tools/cited_paths.py` resolves every
    # path-shaped literal in TRACKED source against the git index, and a plant
    # is by definition a path that must NOT resolve, so spelling one here would
    # turn that reader red for the plant being exactly what it is meant to be.
    tracked = frozenset(
        {
            "/".join(("src", "wildfire_nowcast", "common", "dispersion.py")),
            "/".join(("docs", "interfaces.md")),
        }
    )
    evidence = ["/".join(("runs", "m30_g3_drawA.json.gz"))]
    cited = "/".join(("common", "dispersion.py"))  # a TRAILING FRAGMENT of a tracked file

    page = (
        "<!doctype html><html><head><style>:root { --ink: #000; }</style></head>"
        '<body><p>the bar comes from <span class="mono">' + cited + "</span>.</p>"
        '<img src="data:image/png;base64,AAAA" alt="read from ' + evidence[0] + '">'
        "<table><tr><td>" + evidence[0] + "</td></tr></table>"
        "<caption>" + g6.EVIDENCE_DISCLOSURE + "</caption></body></html>"
    )

    # CONTROL. Silent - and note WHAT it is silent about: a tracked path spelled
    # the way the page spells it, and a declared artifact beside its disclosure.
    # A control that contained no path at all would pass a scanner that had
    # stopped looking, so the control carries one of each.
    g6.assert_page_cites_nothing_a_cloner_cannot_open(page, tracked=tracked, evidence=evidence)

    # NEGATIVE CONTROL on the scanner itself: it must SEE things, or every
    # verdict above is "found nothing" rather than "found nothing wrong".
    seen = g6.path_tokens_in_rendered(page)
    assert cited in seen, f"the scanner missed a path inside a span: {sorted(seen)}"
    assert evidence[0] in seen, f"the scanner missed a path in a table cell: {sorted(seen)}"
    assert g6._expand_braces("a{x,y}.md") == ["ax.md", "ay.md"], "the brace shorthand must expand"
    assert g6._expand_braces(cited) == [cited], "a token with no brace must come back unchanged"
    assert g6.resolves_against_the_index(cited, tracked), "a trailing fragment must resolve"
    assert not g6.resolves_against_the_index("/".join(("dispersion.py", "x.py")), tracked), (
        "the resolver accepted a path no tracked file ends with. A resolver that "
        "says yes to everything makes every plant below silent, so this line is "
        "checked before the plants rather than after them."
    )

    plants = {
        # the real defect: a coordination document, named in prose
        "a coordination file": page.replace(
            "<body>", "<body><p>see " + "/".join(("coordination", "STATE.md")) + "</p>"
        ),
        # the same class with a line number appended, which is how it appeared
        "a planning document with a line number": page.replace(
            "<body>", "<body><p>see " + "/".join(("docs", "play" + "book.md")) + ":510</p>"
        ),
        # the bare tier: strip the directory and the token is still an instruction
        "a BARE untracked filename": page.replace(
            "<body>", "<body><p>see " + "play" + "book.md" + "</p>"
        ),
        # THE ONE THAT PROVES THE EXEMPTION CANNOT BE GROWN BY PROSE: an
        # artifact-shaped path that is not in the DECLARED input set. If this
        # passed, "anything under runs/" would be exempt and the control would
        # be an open door with a directory name on it.
        "an artifact path that is not a declared input": page.replace(
            "<body>", "<body><p>see " + "/".join(("runs", "not_an_input.json")) + "</p>"
        ),
        # THE BRACE SHORTHAND. The page writes one token for two draw files, so
        # a scanner that stopped at `{` would have made a two-file shorthand the
        # ONE spelling nothing could catch - a fresh evasion in the shape of the
        # one being closed. Expanded, both halves are resolved.
        "two untracked files behind one brace": page.replace(
            "<body>",
            "<body><p>see " + "/".join(("coordination", "{STATE," + "BLOCKERS}.md")) + "</p>",
        ),
        # the disclosure deleted while the evidence row stays: an undisclosed
        # dead link is the defect; a disclosed origin is not.
        "the disclosure removed, the evidence row kept": page.replace(g6.EVIDENCE_DISCLOSURE, ""),
    }
    for label, planted in plants.items():
        msg = _g6_bar_refusal(
            g6.assert_page_cites_nothing_a_cloner_cannot_open,
            planted,
            tracked=tracked,
            evidence=evidence,
        )
        assert msg.startswith("SelfCheckFailure"), f"{label}: {msg}"

    # RESTORE, after the plants and not before them. Each plant built a copy, so
    # re-running the untouched control is the only evidence none of them left
    # state behind.
    g6.assert_page_cites_nothing_a_cloner_cannot_open(page, tracked=tracked, evidence=evidence)


def run_all() -> int:
    """Run every ``test_*`` in this module and print the tally.

    ``except BaseException`` used to sit here alone, so **Ctrl-C was recorded as a
    test failure and the runner carried on to the next test**: the one signal that
    means "stop" was swallowed and mislabelled, and the tally at the end counted a
    run nobody completed. An interruption is now logged and re-raised; a test
    failure is still caught, because catching those is what a runner is for.
    """
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures: list[tuple[str, BaseException]] = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except (KeyboardInterrupt, SystemExit):
            logger.warning(
                "interrupted during %s; %d of %d tests had run and the tally below is NOT a result",
                fn.__name__,
                tests.index(fn),
                len(tests),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_all())
