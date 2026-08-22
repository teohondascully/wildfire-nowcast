"""C6.4 - the shape/silence decomposition, validated on cases with KNOWN answers.

Written the way C1.7 was validated (A7): construct the artifact whose correct
score you can work out on paper, then assert the code produces it. Every case in
here has an answer derivable without running anything, including the two that
motivated ADR-017 - empty-vs-empty, and a model that predicts nothing.

**Model-blindness is a property this file has to establish, not claim.** Nothing
here imports ``model/``, loads a checkpoint or reads ``runs/``. The two
"forecasters" compared in
:func:`test_undecomposed_prefers_silence_while_the_gate_criterion_prefers_shape`
are hand-written arrays, and which of them is "ours" is not a question the
decomposition can answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.iou_terms import (
    GATE_CRITERION_KEY,
    decompose_best_member_iou,
    decompose_by_horizon,
    jaccard,
    silent_floor,
    terms_to_metric_dict,
    truth_empty_by_lead,
)
from wildfire_nowcast.eval.metrics import aggregate, evaluate, fuzzy_iou

# --------------------------------------------------------------------------
# the primitive and the convention that caused the pathology
# --------------------------------------------------------------------------


def test_empty_vs_empty_is_one_and_that_is_the_whole_problem() -> None:
    """The convention itself. Documented, defensible in isolation, and bankable."""
    empty = np.zeros((4, 4), dtype=bool)
    assert jaccard(empty, empty) == 1.0
    one = empty.copy()
    one[0, 0] = True
    # One wrong cell against an empty truth costs the FULL point, not a fraction.
    assert jaccard(one, empty) == 0.0


def test_jaccard_is_bit_identical_to_c6_fuzzy_iou() -> None:
    """C0: re-homing a convention is only safe if it reproduces the old value.

    Same standard A10 applied to ``split_fingerprint``. If these ever disagree,
    the decomposition is describing a different metric from the one C6 reports.
    """
    rng = np.random.default_rng(7)
    for _ in range(60):
        a = rng.random((6, 6)) < rng.uniform(0.0, 0.5)
        b = rng.random((6, 6)) < rng.uniform(0.0, 0.5)
        assert jaccard(a, b) == fuzzy_iou(a, b, 0)
    empty = np.zeros((6, 6), dtype=bool)
    assert jaccard(empty, empty) == fuzzy_iou(empty, empty, 0) == 1.0


def test_truth_empty_by_lead_reads_the_mask_not_the_forecast() -> None:
    truth = np.zeros((3, 5, 5), dtype=bool)
    truth[1, 0, 0] = True  # outside the mask
    truth[2, 4, 4] = True  # inside it
    mask = np.zeros((5, 5), dtype=bool)
    mask[3:, 3:] = True
    assert list(truth_empty_by_lead(truth, mask)) == [True, True, False]
    assert list(truth_empty_by_lead(truth)) == [True, False, False]


# --------------------------------------------------------------------------
# the four cases whose answers are known on paper
# --------------------------------------------------------------------------


def test_empty_truth_everywhere_leaves_the_gate_criterion_UNDEFINED() -> None:
    """Every lead is empty-vs-empty. The reported score is a perfect 1.0.

    This is ADR-017's mechanism in one assertion: the metric says the forecast
    was flawless, and the forecast said nothing. ``shape_masked`` must be None -
    NOT 0.0 (which would punish a model for a property of the labels) and NOT 1.0
    (which is the bug).
    """
    per = np.ones((3, 2))  # every member, every lead: empty vs empty
    empty = np.array([True, True])
    terms = decompose_best_member_iou(per, empty)
    assert terms.undecomposed == 1.0
    assert terms.silence == 1.0
    assert terms.shape == 0.0
    assert terms.shape_masked is None
    assert terms.silent_floor == 1.0
    assert terms.n_growing_leads == 0


def test_a_model_that_predicts_nothing_scores_the_floor_and_zero_shape() -> None:
    """A silent forecast: IoU 1 on every empty lead, 0 on every growing one.

    Known answer at 4 leads with 3 empty: reported 0.75, silence 0.75, shape 0.0,
    and the gate criterion EXACTLY 0 - the minimum of its range. That last number
    is the reason the masked variant is the gate criterion.
    """
    empty = np.array([True, True, False, True])
    per = np.tile(np.where(empty, 1.0, 0.0), (5, 1))
    terms = decompose_best_member_iou(per, empty)
    assert terms.undecomposed == pytest.approx(0.75)
    assert terms.silence == pytest.approx(0.75)
    assert terms.shape == pytest.approx(0.0)
    assert terms.shape_masked == pytest.approx(0.0)
    assert terms.silent_floor == pytest.approx(0.75)


def test_a_perfect_model_scores_one_everywhere() -> None:
    empty = np.array([True, False, False])
    per = np.ones((4, 3))
    terms = decompose_best_member_iou(per, empty)
    assert terms.undecomposed == pytest.approx(1.0)
    assert terms.silence == pytest.approx(1 / 3)
    assert terms.shape == pytest.approx(2 / 3)
    assert terms.shape_masked == pytest.approx(1.0)


def test_undecomposed_prefers_silence_while_the_gate_criterion_prefers_shape() -> None:
    """ADR-017's one-window proof, reconstructed from first principles.

    Two forecasters, 4 leads, 2 of them with no truth growth.

    * ``quiet``  says nothing anywhere: 1.0 on the empty leads, 0.0 elsewhere.
    * ``shapely`` captures 60% of the shape on the growing leads and puts one
      wrong cell on each empty lead.

    On paper: quiet = (1+1+0+0)/4 = 0.500, shapely = (0+0+0.6+0.6)/4 = 0.300.
    The metric ranks SAYING NOTHING above capturing 60% of the fire. Under the
    gate criterion the empty leads drop out: quiet = 0.0, shapely = 0.6.

    The ordering flips, and the flip is the point. Neither array is a model we
    own - the case is symmetric under swapping their names, and the assertions
    reference the arrays, not any run.
    """
    empty = np.array([True, True, False, False])
    quiet = np.array([[1.0, 1.0, 0.0, 0.0]])
    shapely = np.array([[0.0, 0.0, 0.6, 0.6]])

    q = decompose_best_member_iou(quiet, empty)
    s = decompose_best_member_iou(shapely, empty)

    assert q.undecomposed == pytest.approx(0.5)
    assert s.undecomposed == pytest.approx(0.3)
    assert q.undecomposed > s.undecomposed  # the pathology, reproduced

    assert q.shape_masked == pytest.approx(0.0)
    assert s.shape_masked == pytest.approx(0.6)
    assert s.shape_masked > q.shape_masked  # the correction

    # And the silence term localises the whole difference: quiet's entire score
    # is silence, shapely's is entirely shape.
    assert q.silence == pytest.approx(0.5)
    assert q.shape == pytest.approx(0.0)
    assert s.silence == pytest.approx(0.0)
    assert s.shape == pytest.approx(0.3)


def test_gate_criterion_selects_the_member_with_the_shape_not_the_silent_one() -> None:
    """Member selection must not be contaminated by the silence bonus.

    Member 0 is silent: 1.0 on the two empty leads, 0.0 on the growing one, so
    its trajectory mean is 2/3 and it WINS the undecomposed argmax. Member 1
    captures the shape perfectly on the growing lead but false-alarms on the
    empty ones, so its trajectory mean is 1/3 and it loses.

    Decomposing the winner would report shape 0.0 - a model with a perfect-shape
    member scoring zero on shape. The masked variant selects on the growing lead
    and reports 1.0, which is the true best-member mode capture.
    """
    empty = np.array([True, True, False])
    per = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    terms = decompose_best_member_iou(per, empty)
    assert terms.best_member == 0
    assert terms.shape == pytest.approx(0.0)
    assert terms.best_member_masked == 1
    assert terms.shape_masked == pytest.approx(1.0)


# --------------------------------------------------------------------------
# structural properties
# --------------------------------------------------------------------------


def test_the_split_is_arithmetic_on_random_inputs() -> None:
    """silence + shape == best_member_iou, always. It is a partition of leads."""
    rng = np.random.default_rng(11)
    for _ in range(300):
        n_members = int(rng.integers(1, 9))
        n_lead = int(rng.integers(1, 7))
        per = rng.random((n_members, n_lead))
        empty = rng.random(n_lead) < 0.4
        terms = decompose_best_member_iou(per, empty)  # .check() runs inside
        assert terms.silence + terms.shape == pytest.approx(terms.undecomposed, abs=1e-12)
        assert terms.undecomposed == pytest.approx(per.mean(axis=1).max())
        assert 0.0 <= terms.silence <= 1.0
        assert 0.0 <= terms.shape <= 1.0
        if terms.shape_masked is not None:
            assert 0.0 <= terms.shape_masked <= 1.0


def test_per_horizon_matches_scoring_a_shorter_window() -> None:
    """Entry H-1 must equal what a length-H window would have produced.

    C6.2's per-horizon rule (ADR-015) means 1/2/3 h are three separate
    adjudications, so "identical by construction" has to be asserted, not hoped.
    """
    rng = np.random.default_rng(3)
    per = rng.random((6, 4))
    empty = np.array([True, False, True, False])
    by_horizon = decompose_by_horizon(per, empty)
    assert len(by_horizon) == 4
    for h, terms in enumerate(by_horizon, start=1):
        direct = decompose_best_member_iou(per[:, :h], empty[:h])
        assert terms.undecomposed == pytest.approx(direct.undecomposed)
        assert terms.silence == pytest.approx(direct.silence)
        assert terms.shape == pytest.approx(direct.shape)
        assert terms.shape_masked == direct.shape_masked or terms.shape_masked == pytest.approx(
            direct.shape_masked
        )
    assert by_horizon[-1].undecomposed == pytest.approx(per.mean(axis=1).max())


def test_silent_floor_is_a_label_statistic() -> None:
    empty = np.array([True, False, True, True])
    assert silent_floor(empty) == pytest.approx(0.75)
    assert silent_floor(empty, 1) == 1.0
    assert silent_floor(empty, 2) == pytest.approx(0.5)


def test_mismatched_shapes_raise_rather_than_scoring_something_wrong() -> None:
    per = np.ones((2, 3))
    with pytest.raises(ValueError, match="SAME mask"):
        decompose_best_member_iou(per, np.array([True, False]))
    with pytest.raises(ValueError, match="horizon"):
        decompose_best_member_iou(per, np.array([True, False, True]), horizon=9)
    with pytest.raises(ValueError, match="n_members"):
        decompose_best_member_iou(np.ones(3), np.array([True, False, True]))


def test_metric_dict_names_the_gate_criterion_explicitly() -> None:
    """A downstream table must not have to GUESS which term gates."""
    per = np.array([[0.4, 0.8]])
    out = terms_to_metric_dict(decompose_by_horizon(per, np.array([True, False])))
    assert out["best_member_iou_gate_criterion"] == GATE_CRITERION_KEY
    assert GATE_CRITERION_KEY in out
    assert out[f"{GATE_CRITERION_KEY}_by_horizon"] == [None, pytest.approx(0.8)]


# --------------------------------------------------------------------------
# the wiring into C6 - the value modelling will actually read
# --------------------------------------------------------------------------


def _window(height: int = 24, width: int = 24) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.zeros((height, width), dtype=np.uint8)
    x0[10:14, 10:14] = 1
    truth = np.stack([x0.copy() for _ in range(3)])
    truth[1, 9:15, 9:15] = 1
    truth[2, 8:16, 8:16] = 1
    return x0, truth


def test_evaluate_emits_the_decomposition_and_it_reconstructs() -> None:
    x0, truth = _window()
    samples = np.stack([truth.copy() for _ in range(4)]).astype(np.uint8)
    result = evaluate(samples, truth, x0=x0)
    assert result["best_member_iou_gate_criterion"] == GATE_CRITERION_KEY
    for block in result["by_mask"].values():
        assert block["best_member_iou_silence"] + block["best_member_iou_shape"] == pytest.approx(
            block["best_member_iou"], abs=1e-9
        )
        by_h = block["best_member_iou_by_horizon"]
        sil = block["best_member_iou_silence_by_horizon"]
        shp = block["best_member_iou_shape_by_horizon"]
        for total, s, p in zip(by_h, sil, shp, strict=True):
            assert s + p == pytest.approx(total, abs=1e-9)


def test_evaluate_leaves_the_reported_value_bit_identical() -> None:
    """C6.4 keeps ``best_member_iou`` REPORTED. Adding terms must not move it."""
    x0, truth = _window()
    rng = np.random.default_rng(2)
    samples = (rng.random((5, 3, 24, 24)) < 0.3).astype(np.uint8)
    result = evaluate(samples, truth, x0=x0)
    block = result["by_mask"]["growth_band"]
    # Recompute the undecomposed value straight from the definition.
    from wildfire_nowcast.eval.masks import default_band_radius, growth_band

    mask = growth_band(x0, default_band_radius(3))
    per = np.array(
        [
            [fuzzy_iou((samples[m, k] > 0) & mask, (truth[k] > 0) & mask, 0) for k in range(3)]
            for m in range(5)
        ]
    )
    assert block["best_member_iou"] == pytest.approx(float(per.mean(axis=1).max()))


def test_a_null_forecast_scores_zero_on_the_gate_criterion_through_c6() -> None:
    """End to end: persistence in the growth band scores the floor, and 0 on shape."""
    x0, truth = _window()
    persistence = np.stack([np.stack([x0.copy() for _ in range(3)]) for _ in range(4)])
    result = evaluate(persistence.astype(np.uint8), truth, x0=x0)
    band = result["by_mask"]["growth_band"]
    assert band["best_member_iou_shape"] == pytest.approx(0.0)
    assert band[GATE_CRITERION_KEY] == pytest.approx(0.0)
    # Its entire reported score is silence, and it equals the label-only floor.
    assert band["best_member_iou"] == pytest.approx(band["best_member_iou_silence"])
    assert band["best_member_iou"] == pytest.approx(band["best_member_iou_silent_floor"])


def test_pooling_preserves_the_identity_and_declares_its_denominator() -> None:
    x0, truth = _window()
    rng = np.random.default_rng(5)
    results = [
        evaluate((rng.random((4, 3, 24, 24)) < 0.25).astype(np.uint8), truth, x0=x0)
        for _ in range(6)
    ]
    pooled = aggregate(results)
    for block in pooled["by_mask"].values():
        assert block["best_member_iou_silence"] + block["best_member_iou_shape"] == pytest.approx(
            block["best_member_iou"], abs=1e-9
        )
        # The gate criterion pools over DEFINED windows only, and says how many.
        counts = block[f"{GATE_CRITERION_KEY}_n_windows_by_horizon"]
        assert len(counts) == 3
        assert all(0 <= c <= len(results) for c in counts)


def test_domain_mask_decomposition_is_inert_and_that_is_correct() -> None:
    """Truth is essentially never empty domain-wide, so the split does nothing there.

    Asserted rather than left as a footnote: a reader who quotes the DOMAIN shape
    term and thinks the pathology is handled has quoted the undecomposed value
    under a new name. The band is where the correction lives.
    """
    x0, truth = _window()
    rng = np.random.default_rng(9)
    result = evaluate((rng.random((4, 3, 24, 24)) < 0.2).astype(np.uint8), truth, x0=x0)
    domain = result["by_mask"]["domain"]
    assert domain["best_member_iou_silence"] == pytest.approx(0.0)
    assert domain["best_member_iou_shape"] == pytest.approx(domain["best_member_iou"])
    band = result["by_mask"]["growth_band"]
    assert band["best_member_iou_silent_floor"] > 0.0


# --------------------------------------------------------------------------
# C0 - one implementation, and simviz's independent one agrees with it
# --------------------------------------------------------------------------


def test_agrees_with_simviz_replay_decomposition() -> None:
    """``sim.replay`` derived the same split independently. It must still agree.

    simviz reproduced C6 with 0 mismatches over 2,230 windows, which is why C0
    says WIRE rather than reinvent. This asserts the agreement instead of
    assuming it - and it is a skip rather than a hard failure because coupling
    CI to another lead's live working file bit us once already (A10 PROPOSAL 3).
    """
    replay = pytest.importorskip(
        "wildfire_nowcast.sim.replay",
        reason="sim/ is sim's directory; if it does not import, that is their "
        "signal to act on, not a reason to fail infra's suite",
    )
    rng = np.random.default_rng(4)
    height = width = 18
    x0 = np.zeros((height, width), dtype=np.uint8)
    x0[7:11, 7:11] = 1
    truth = np.stack([x0.copy() for _ in range(3)])
    truth[1, 6:12, 6:12] = 1
    truth[2, 6:12, 6:12] = 1  # lead 3 does not grow -> an empty-truth lead in the band
    band = np.ones((height, width), dtype=bool)
    band[x0 > 0] = False

    for _ in range(25):
        samples = (rng.random((6, 3, height, width)) < 0.15).astype(np.uint8)
        window = replay.C5Inputs(
            x0=x0,
            static=np.zeros((8, height, width), dtype=np.float32),
            weather=np.zeros((3, 5, height, width), dtype=np.float32),
            truth=truth,
            t0=0,
            horizon_h=3,
            times=np.zeros(3, dtype="datetime64[h]"),
        )
        theirs = replay.score_window(samples, window, fire_id="t", model="t", band=band)

        member_event = samples > 0
        truth_event = truth > 0
        per = np.array(
            [
                [jaccard(member_event[m, k] & band, truth_event[k] & band) for k in range(3)]
                for m in range(6)
            ]
        )
        mine = decompose_best_member_iou(per, truth_empty_by_lead(truth_event, band))

        assert mine.undecomposed == pytest.approx(theirs.band_best_member_iou)
        assert mine.silence == pytest.approx(theirs.silence_term)
        assert mine.shape == pytest.approx(theirs.shape_term)
        assert mine.n_empty_leads == theirs.n_empty_leads
