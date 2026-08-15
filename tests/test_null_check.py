"""C6.0 — the do-nothing null check, validated against KNOWN answers.

A safety check nobody has tried to fool is a safety check with unknown power.
This file establishes three things, in order of importance:

1. **It has teeth.** The two pathologies the contract already quarantined —
   ``dispersion_ratio`` (C6.1) and ``best_member_iou`` (C6.4) — must come back
   flagged. If they ever come back clean, the harness broke, not the metrics.
2. **It does not over-fire.** A metric known to be sound (the C6.4 gate
   criterion, ``area_dispersion_ratio``, Brier against a silence null) must come
   back ``ok``, so a real finding is not lost in noise.
3. **It is decidable, not lucky.** Verdicts are stable across DISJOINT seed sets,
   because a verdict that moves with the seed is a coin flip being reported as a
   finding — which is the exact failure the ``BLIND`` tier exists to name.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common import null_check as N

# One scenario, built once: 37 windows x 3 leads on a 40x40 grid is ~0.6 s per
# seed through the full C6 stack, and every test below wants the same labels.
SEEDS_A = (0, 1, 2)
SEEDS_B = (7, 8, 9)


@pytest.fixture(scope="module")
def scenario() -> tuple[list[N.Window], dict]:
    return N.synthetic_windows()


@pytest.fixture(scope="module")
def report_a(scenario) -> N.NullCheckReport:
    windows, stats = scenario
    return N.run_null_check(windows, stats, seeds=SEEDS_A)


@pytest.fixture(scope="module")
def report_b(scenario) -> N.NullCheckReport:
    windows, stats = scenario
    return N.run_null_check(windows, stats, seeds=SEEDS_B)


def _verdicts(report: N.NullCheckReport) -> dict[tuple[str, str], N.MetricVerdict]:
    return {(v.metric, v.mask): v for v in report.verdicts}


# --------------------------------------------------------------------------
# the fixture itself — a check on a fixture with the wrong statistics is a
# check on nothing
# --------------------------------------------------------------------------


def test_the_scenario_actually_contains_zero_growth_leads(scenario) -> None:
    """The pathology's SIZE is set by this rate, so it must be a real number.

    ADR-017 measured 21.9% of leads in the growth stratum with zero truth growth
    and a 46.4% null floor at 1 h. A fixture that always grows would let every
    silence-favouring metric pass, and the check would be decorative.
    """
    _, stats = scenario
    rate = stats["zero_growth_lead_fraction"]
    assert 0.15 < rate < 0.60, rate
    assert stats["n_windows"] >= 20


def test_the_scenario_is_legal_state_data(scenario) -> None:
    """Absorbing, in-domain: it must be the thing C1.1 describes, not any field."""
    windows, _ = scenario
    for w in windows:
        seq = np.concatenate([w.x0[None], w.truth])
        burned = seq > 0
        assert set(np.unique(seq)) <= {0, 1, 2}
        assert np.all(burned[1:] >= burned[:-1]), "fire must be absorbing (C1.1)"


# --------------------------------------------------------------------------
# 1. teeth — the known-broken metrics must come back flagged
# --------------------------------------------------------------------------


def test_best_member_iou_is_flagged_in_the_growth_band(report_a, report_b) -> None:
    """ADR-017's finding, reproduced by a harness that never saw a model.

    **This assertion was rewritten at A12 and the rewrite is the finding.** It
    used to read ``verdict != ok``, i.e. it asked the COMPARISON question, and it
    went green when the refit gave the harness a stronger reference model
    (``skillful_calibrated`` 0.40321 vs the null's 0.33333). That is a true
    measurement of a different thing. ADR-017's actual claim — that the metric
    pays for silence — lives in the CAPTURE verdict, which no reference model can
    move. The comparison is asserted alongside it precisely because it now says
    ``ok``: both answers are true, and pinning both is what stops either from
    being quietly read as the other (ADR-022 (1)).
    """
    for report in (report_a, report_b):
        v = _verdicts(report)[("best_member_iou", "growth_band")]
        assert v.capture_verdict == N.VERDICT_PAYS_FOR_NOTHING, v.capture_detail
        assert v.is_flagged
        assert not v.gate_eligible, "C6.4 already forbids this metric from gating"

        # The size of the pathology, exactly: a forecast claiming NOTHING is paid
        # the zero-growth lead fraction. Not "about a third" — 1/3, because the
        # fixture declares 33.3% and the metric hands it over whole.
        assert v.scores["null_empty"] == pytest.approx(1.0 / 3.0)
        assert v.scores["null_zero_ignition"] == pytest.approx(1.0 / 3.0)

        # ...and the comparison genuinely disagrees. If this ever flips back the
        # two questions have re-merged and the split has been undone.
        assert v.verdict == N.VERDICT_OK, v.detail
        assert v.scores["skillful_calibrated"] > v.scores["null_empty"]


def test_the_two_verdicts_disagree_and_that_is_the_point(report_a) -> None:
    """The C6.0 split, asserted as a property of the report rather than of one row.

    At least one metric must have a clean comparison AND a failing axiom.
    Otherwise the two fields are carrying the same information and the split has
    bought nothing — which is the state the harness was in before A12.
    """
    split_is_load_bearing = [
        v
        for v in report_a.verdicts
        if v.verdict == N.VERDICT_OK and v.capture_verdict == N.VERDICT_PAYS_FOR_NOTHING
    ]
    assert split_is_load_bearing, (
        "no metric passes the comparison while failing the axiom, so the two verdicts are "
        "redundant here and one of them is not doing any work"
    )
    assert {v.metric for v in split_is_load_bearing} >= {"best_member_iou"}


def test_the_axiom_cannot_be_silenced_by_a_better_reference_model() -> None:
    """Why the axiom exists: it never looks at a reference model at all.

    Constructed, so it needs no fixture: hand the capture check a table in which
    every skilful forecaster is PERFECT and the oracle is perfect, and it must
    still flag a metric that pays a zero-claim forecast. The comparison verdict
    would call this metric flawless.
    """
    spec = N.MetricSpec(N.HIGHER, True)
    by_seed = {
        "null_empty": [0.25, 0.25, 0.25],
        "null_zero_ignition": [0.25, 0.25, 0.25],
        "skillful": [1.0, 1.0, 1.0],
        "skillful_calibrated": [1.0, 1.0, 1.0],
        "oracle": [1.0, 1.0, 1.0],
    }
    verdict, detail = N._capture_verdict(spec, by_seed)
    assert verdict == N.VERDICT_PAYS_FOR_NOTHING, detail
    assert "0.25" in detail

    # Raising every reference to perfection changes nothing, because none of them
    # is read. Only ZERO_CLAIM's own score matters.
    by_seed_clean = {**by_seed, "null_empty": [0.0, 0.0, 0.0]}
    assert N._capture_verdict(spec, by_seed_clean)[0] == N.VERDICT_OK


def test_the_axiom_does_not_fire_on_error_metrics_or_spread_metrics() -> None:
    """R14: a silent forecast may legitimately score well on a LOWER metric.

    Brier at a 1% base rate is the standing example, and treating that as a
    defect would make the axiom a false-alarm generator on exactly the metrics
    G2 rests on. The axiom is confined to ``higher_is_better`` by construction,
    and this pins that scope rather than leaving it to the reader.
    """
    by_seed = {"null_empty": [0.02, 0.02, 0.02], "skillful": [0.05, 0.05, 0.05]}
    for direction in (N.LOWER, N.TARGET, N.DIAGNOSTIC, N.LABEL_STATISTIC):
        spec = N.MetricSpec(direction, True, target=1.0)
        assert N._capture_verdict(spec, by_seed)[0] == N.CAPTURE_NOT_APPLICABLE


def test_a_missing_zero_claim_is_undecidable_not_a_pass() -> None:
    """An axiom that cannot be evaluated must say so — C-2's verdict choke point."""
    spec = N.MetricSpec(N.HIGHER, True)
    verdict, detail = N._capture_verdict(spec, {"skillful": [0.5, 0.5]})
    assert verdict == N.VERDICT_UNDECIDABLE
    assert N.ZERO_CLAIM in detail


def test_a_gate_eligible_metric_that_pays_for_nothing_is_a_HARD_failure() -> None:
    """The build-breaking half of the split, on a constructed verdict.

    Only the axiom may void a gate on the silence question; a SILENCE_FAVOURING
    comparison stays a reporting gap (C-1, R14). Both directions are asserted so
    neither can drift into the other.
    """
    paying = N.MetricVerdict(
        metric="m", mask="growth_band", verdict=N.VERDICT_OK, scores={},
        gate_eligible=True, capture_verdict=N.VERDICT_PAYS_FOR_NOTHING,
    )
    assert paying.is_failure and paying.is_flagged

    favouring = N.MetricVerdict(
        metric="m", mask="growth_band", verdict=N.VERDICT_SILENCE_FAVOURING, scores={},
        gate_eligible=True, capture_verdict=N.VERDICT_OK,
    )
    assert not favouring.is_failure
    assert favouring.is_reporting_gap and favouring.is_flagged

    # A quarantined metric is expected to fail and must never break the build.
    assert not N.MetricVerdict(
        metric="m", mask="growth_band", verdict=N.VERDICT_BROKEN, scores={},
        gate_eligible=False, capture_verdict=N.VERDICT_PAYS_FOR_NOTHING,
    ).is_failure


def test_zero_claim_is_the_empty_forecast_and_not_persistence(report_a) -> None:
    """Pointing the axiom at persistence would make it a false-alarm generator.

    On the DOMAIN mask persistence scores ``best_member_iou`` ~0.86 by correctly
    reproducing the already-burned region — capture it EARNED. The empty forecast
    scores exactly 0 there, which is why it is the axiom's subject and why
    ``best_member_iou`` is clean on ``domain`` and flagged on ``growth_band``.
    """
    assert N.ZERO_CLAIM == "null_empty"
    dom = _verdicts(report_a)[("best_member_iou", "domain")]
    assert dom.scores["null_zero_ignition"] > 0.8
    assert dom.scores["null_empty"] == pytest.approx(0.0, abs=1e-12)
    assert dom.capture_verdict == N.VERDICT_OK

    band = _verdicts(report_a)[("best_member_iou", "growth_band")]
    assert band.capture_verdict == N.VERDICT_PAYS_FOR_NOTHING


def test_the_axiom_also_catches_growth_iou_on_the_domain_mask(report_a) -> None:
    """A case the growth-band comparison never looks at.

    ``best_member_iou_growth`` restricts to cells unburned at t0, so it inherits
    the empty-vs-empty convention on BOTH masks — while plain ``best_member_iou``
    is clean on ``domain``. The comparison tier has never examined this cell;
    the axiom flags it on the first run.
    """
    v = _verdicts(report_a)[("best_member_iou_growth", "domain")]
    assert v.capture_verdict == N.VERDICT_PAYS_FOR_NOTHING, v.capture_detail
    assert v.scores["null_empty"] == pytest.approx(1.0 / 3.0)


def test_the_declared_deterministic_forecasters_really_ignore_the_rng(scenario) -> None:
    """``DETERMINISTIC`` is a performance claim; a wrong entry would cache a lie.

    ``run_null_check`` scores these once and replicates across seeds. That is
    bitwise correct only while they really never draw, so the declaration is
    verified rather than trusted — under two generators seeded far apart.
    """
    windows, _ = scenario
    models = N.forecasters_for(windows)
    assert N.DETERMINISTIC <= set(models)
    for name in sorted(N.DETERMINISTIC):
        fn = models[name]
        for w in windows[:6]:
            a = fn(w, 8, np.random.default_rng(0))
            b = fn(w, 8, np.random.default_rng(987654321))
            assert np.array_equal(a, b), f"{name} is not deterministic; DETERMINISTIC lies"

    # ...and the ones NOT declared must actually vary, or the declaration is
    # merely incomplete in the safe direction and this test proves nothing.
    varying = [n for n in models if n not in N.DETERMINISTIC]
    w = windows[len(windows) // 2]
    assert any(
        not np.array_equal(
            models[n](w, 8, np.random.default_rng(0)),
            models[n](w, 8, np.random.default_rng(987654321)),
        )
        for n in varying
    )


def test_dispersion_ratio_cannot_separate_collapse_from_health(report_a, report_b) -> None:
    """ADR-011's finding. The STABLE form of it is blindness, not a direction.

    Measured here: the collapsed and healthy ensembles land within ~1% of each
    other and which one wins flips with the seed. `area_dispersion_ratio`, the
    metric C6.1 promoted in its place, separates them by ~3x — asserted in the
    same breath, because "this instrument is blind" only means something beside
    an instrument that is not.
    """
    for report in (report_a, report_b):
        v = _verdicts(report)[("dispersion_ratio", "growth_band")]
        assert v.verdict == N.VERDICT_BLIND, v.detail
        gap = abs(v.scores[N.COLLAPSE] - v.scores[N.SKILLFUL])
        assert gap < 0.05, gap

        area = _verdicts(report)[("area_dispersion_ratio", "growth_band")]
        assert area.verdict == N.VERDICT_OK, area.detail
        separation = abs(area.scores[N.COLLAPSE] - area.scores[N.SKILLFUL])
        assert separation > 0.2, separation


def test_the_positive_controls_are_not_empty(report_a) -> None:
    """The harness's own validation: the two contract-quarantined metrics reappear.

    ``quarantined_confirmed`` reads BOTH verdicts as of A12. Reading only the
    comparison is what emptied this set when the reference model improved — a
    positive control that a better reference can switch off was never testing the
    metric, only the reference.
    """
    confirmed = {v.metric for v in report_a.quarantined_confirmed()}
    assert {"best_member_iou", "dispersion_ratio"} <= confirmed, confirmed
    # The A11 comparison exonerated the tolerant variant; the axiom does not.
    assert {"best_member_iou_growth", "best_member_iou_tolerant"} <= confirmed, confirmed


# --------------------------------------------------------------------------
# 2. no over-firing — the sound metrics must come back clean
# --------------------------------------------------------------------------


def test_the_c6_4_gate_criterion_passes_its_own_null_check(report_a, report_b) -> None:
    """The point of A11: the corrected criterion ranks the null LAST, at 0.

    ``0.0`` is not "low", it is the minimum of the range, and a null scoring the
    minimum is what makes a criterion fit to gate. Asserted on both masks.
    """
    for report in (report_a, report_b):
        for mask in ("domain", "growth_band"):
            v = _verdicts(report)[(N.GATE_METRIC, mask)]
            assert v.gate_eligible
            assert v.verdict == N.VERDICT_OK, v.detail
            assert v.scores["skillful"] > 0.1
            assert v.scores["oracle"] == pytest.approx(1.0)
            assert v.scores["skillful"] < v.scores["oracle"]

        band = _verdicts(report)[(N.GATE_METRIC, "growth_band")]
        # In the band, a zero-ignition forecast predicts NOTHING, and the gate
        # criterion pays exactly 0 for it — the minimum of the range, not merely
        # a low score. This one number is what makes the criterion fit to gate.
        assert band.scores["null_zero_ignition"] == pytest.approx(0.0, abs=1e-9)
        assert band.scores["null_empty"] == pytest.approx(0.0, abs=1e-9)

        # On the DOMAIN mask persistence reproduces the already-burned region, so
        # its score is legitimately non-zero. It must still rank last.
        dom = _verdicts(report)[(N.GATE_METRIC, "domain")]
        assert 0.0 < dom.scores["null_zero_ignition"] < dom.scores["skillful"]


def test_brier_and_crps_rank_the_null_below_genuine_skill(report_a) -> None:
    """Sanity floor: if these fired, the harness would be crying wolf everywhere."""
    v = _verdicts(report_a)
    for metric in ("brier_1h", "brier_2h", "brier_3h", "arrival_crps"):
        assert v[(metric, "growth_band")].verdict == N.VERDICT_OK, v[(metric, "growth_band")].detail


def test_no_gate_eligible_metric_is_in_the_hard_tier(report_a, report_b) -> None:
    """The build-breaking condition, stated once.

    Reporting gaps (SILENCE_FAVOURING) are deliberately NOT included: C-1's two
    tiers, and R14 says a proper score at a 1% base rate legitimately prefers
    silence to a sub-coin-flip predictor. Those are asserted separately below so
    they are visible rather than tolerated.
    """
    for report in (report_a, report_b):
        assert report.ok, [(v.metric, v.mask, v.detail) for v in report.failures()]
        assert not report.problems, report.problems


# --------------------------------------------------------------------------
# 3. decidability — verdicts must not move with the seed
# --------------------------------------------------------------------------


def test_verdicts_are_stable_across_disjoint_seed_sets(report_a, report_b) -> None:
    """A verdict that flips between seed sets is a coin flip, not a finding."""
    a, b = _verdicts(report_a), _verdicts(report_b)
    unstable = {
        key: (a[key].verdict, b[key].verdict)
        for key in sorted(set(a) & set(b))
        if a[key].verdict != b[key].verdict
    }
    # Metrics sitting exactly on the noise floor are allowed to move; the ones
    # this task turns on are not.
    load_bearing = {
        ("best_member_iou", "growth_band"),
        ("dispersion_ratio", "growth_band"),
        ("area_dispersion_ratio", "growth_band"),
        (N.GATE_METRIC, "growth_band"),
        (N.GATE_METRIC, "domain"),
        ("brier_1h", "growth_band"),
        ("brier_3h", "growth_band"),
    }
    moved = load_bearing & set(unstable)
    assert not moved, {k: unstable[k] for k in moved}


def test_the_silent_floor_is_identical_for_every_forecaster(report_a) -> None:
    """A label statistic that moved with the model would be a leak, and the
    harness raises it as a PROBLEM rather than ranking it."""
    assert not [p for p in report_a.problems if "silent_floor" in p]


# --------------------------------------------------------------------------
# the comparison primitive, on numbers whose answer is known
# --------------------------------------------------------------------------


def test_compare_is_paired_and_respects_orientation() -> None:
    higher = N.MetricSpec(N.HIGHER, True)
    lower = N.MetricSpec(N.LOWER, True)
    target = N.MetricSpec(N.TARGET, True, target=1.0)

    # A constant advantage of +0.10 with zero variance is unambiguous.
    assert N.compare([0.6, 0.6, 0.6], [0.5, 0.5, 0.5], higher)[0] == N.CMP_BETTER
    assert N.compare([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], higher)[0] == N.CMP_WORSE
    # Lower-is-better inverts it, and nothing else changes.
    assert N.compare([0.5, 0.5, 0.5], [0.6, 0.6, 0.6], lower)[0] == N.CMP_BETTER
    # Closer-to-1 wins for a target metric, on both sides of the target.
    assert N.compare([1.02, 1.02, 1.02], [1.30, 1.30, 1.30], target)[0] == N.CMP_BETTER
    assert N.compare([0.98, 0.98, 0.98], [0.60, 0.60, 0.60], target)[0] == N.CMP_BETTER


def test_compare_refuses_to_call_a_difference_inside_its_own_noise() -> None:
    """The dispersion_ratio situation, in miniature: a 4-of-5 preference at 1%.

    Mean advantage +0.004 against a seed SD of ~0.02 is not a direction, and the
    harness must say ``indistinguishable`` rather than publish the sign.
    """
    a = [1.01, 0.97, 1.02, 0.99, 1.03]
    b = [1.00, 1.00, 1.00, 1.00, 1.00]
    outcome, mean, sd = N.compare(a, b, N.MetricSpec(N.HIGHER, True))
    assert outcome == N.CMP_INDISTINGUISHABLE
    assert abs(mean) < N.NOISE_FLOOR_SD * sd


# --------------------------------------------------------------------------
# the degenerate forecasters are what they claim to be
# --------------------------------------------------------------------------


def test_the_nulls_really_predict_nothing(scenario) -> None:
    """If ``null_zero_ignition`` ignited one cell, every verdict here is void."""
    windows, _ = scenario
    rng = np.random.default_rng(0)
    for w in windows[:8]:
        burned0 = w.x0 > 0
        persistence = N.null_zero_ignition(w, 4, rng) > 0
        assert np.array_equal(persistence, np.broadcast_to(burned0, persistence.shape))
        assert int((persistence & ~burned0[None, None]).sum()) == 0, "ignited a new cell"
        assert int((N.null_empty(w, 4, rng) > 0).sum()) == 0


def test_the_oracle_is_exactly_right_and_skillful_is_not(scenario) -> None:
    windows, _ = scenario
    rng = np.random.default_rng(0)
    w = windows[len(windows) // 2]
    truth_all = np.broadcast_to(w.truth > 0, (3, *w.truth.shape))
    assert np.array_equal(N.oracle(w, 3, rng) > 0, truth_all)
    skill = N.skillful(w, 8, rng) > 0
    truth = w.truth > 0
    assert not np.array_equal(skill[0], truth), "a 'skillful' model that is perfect proves nothing"
    # It must be informative, not noise: it recovers some real growth.
    burned0 = w.x0 > 0
    hits = int((skill[:, -1] & truth[-1] & ~burned0).sum())
    assert hits > 0


def test_an_unregistered_metric_is_reported_not_skipped(scenario) -> None:
    """C-2, one level down: a metric with no declared orientation is unchecked.

    Simulated by scoring through a wrapper that emits an extra key, because the
    real trigger is modelling adding a metric — which must break this check
    loudly rather than pass by omission.
    """
    from wildfire_nowcast.eval.metrics import aggregate, evaluate

    windows, stats = scenario

    def score(samples, truth, x0):
        out = dict(evaluate(samples, truth, x0=x0))
        for block in out["by_mask"].values():
            block["some_new_metric_nobody_declared"] = 0.5
        return out

    score.aggregate = lambda results: _inject(aggregate(results))

    def _inject(pooled):
        for block in pooled["by_mask"].values():
            block["some_new_metric_nobody_declared"] = 0.5
        return pooled

    report = N.run_null_check(windows[:5], stats, seeds=(0,), score_fn=score)
    assert any("some_new_metric_nobody_declared" in p for p in report.problems), report.problems
    assert not report.ok


# --------------------------------------------------------------------------
# C6.6 [v2.15] — which channels may decide a gate, asked rather than remembered
# --------------------------------------------------------------------------

#: The three channels the contract lets adjudicate at v2.15. PINNED HERE and
#: DERIVED in code, so the pin and the flags must agree: flip a flag and this is
#: red; add a name here without flipping a flag and this is red. A list that can
#: only fail in one direction is the defect ADR-057 catalogued.
ADJUDICATING_AT_V2_15 = {
    "area_dispersion_ratio",
    "growth_calibration",
    "best_member_iou_shape_masked",
}

#: Disqualified by C6.6, with the Spearman that did it (ADR-053 (1)(2)).
NON_ADJUDICATING_AT_V2_15 = {
    "brier_1h": "-0.45",
    "brier_2h": "-0.45",
    "brier_3h": "-0.45",
    "arrival_crps": "-0.34",
    "calibration_error_1h": "-0.14",
    "calibration_error_2h": "-0.14",
    "calibration_error_3h": "-0.14",
    "reliability_1h": "-0.80",
    "reliability_2h": "-0.80",
    "reliability_3h": "-0.80",
}


def test_exactly_three_channels_may_adjudicate() -> None:
    assert N.adjudicating_metrics() == ADJUDICATING_AT_V2_15


def test_the_four_anti_monotone_families_cannot_gate_anything() -> None:
    """C6.6's subject, channel by channel, and every one of them is registered.

    An UNregistered channel would sail past a membership test on
    ``adjudicating_metrics()`` while being just as un-rulable, so presence in the
    table is asserted first.
    """
    for metric in NON_ADJUDICATING_AT_V2_15:
        assert metric in N.C6_METRICS, f"{metric} is not registered, so nothing rules on it"
        assert not N.C6_METRICS[metric].gate_eligible, metric
        assert not N.may_adjudicate(metric), metric


def test_each_disqualified_channel_carries_the_number_that_disqualified_it() -> None:
    """A ruling with no measurement attached is folklore in two months."""
    for metric, rho in NON_ADJUDICATING_AT_V2_15.items():
        spec = N.C6_METRICS[metric]
        assert "C6.6" in spec.quarantined_by, (metric, spec.quarantined_by)
        assert rho in spec.note, (metric, rho, spec.note)
        assert "ADR-053" in spec.note, metric


def test_adjudicating_on_a_barred_channel_FAILS_rather_than_warns() -> None:
    """The whole point of C6.6 being code and not a note in a status file."""
    for metric in NON_ADJUDICATING_AT_V2_15:
        with pytest.raises(N.NonAdjudicatingMetricError, match="MAY NOT decide"):
            N.assert_may_adjudicate(metric, gate="G3")


def test_the_three_permitted_channels_are_allowed_through() -> None:
    """Both directions. A guard that refuses everything is not a guard either."""
    for metric in ADJUDICATING_AT_V2_15:
        spec = N.assert_may_adjudicate(metric, gate="G3")
        assert spec.gate_eligible


def test_an_unregistered_channel_may_not_adjudicate_either() -> None:
    with pytest.raises(N.NonAdjudicatingMetricError, match="not in C6_METRICS"):
        N.assert_may_adjudicate("elasticity_of_log_growth_rate", gate="a new gate")
    assert not N.may_adjudicate("elasticity_of_log_growth_rate")


def test_the_refusal_names_what_may_be_used_instead() -> None:
    """A guard that only says no teaches people to route around it."""
    with pytest.raises(N.NonAdjudicatingMetricError) as exc:
        N.assert_may_adjudicate("brier_3h", gate="G2")
    message = str(exc.value)
    assert "G2" in message
    for permitted in ADJUDICATING_AT_V2_15:
        assert permitted in message


def test_the_disqualified_channels_left_the_hard_tier(report_a) -> None:
    """The flag is live, not decorative: it decides C-1 severity in the harness.

    ``is_failure`` and ``is_reporting_gap`` both read ``gate_eligible``, so this
    is the observable consequence of the bump inside `make null-check`.
    """
    verdicts = _verdicts(report_a)
    for metric in NON_ADJUDICATING_AT_V2_15:
        for key, v in verdicts.items():
            if key[0] == metric:
                assert not v.is_failure, (key, v.detail)
                assert not v.is_reporting_gap, (key, v.detail)
