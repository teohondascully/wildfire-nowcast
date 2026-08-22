"""The one estimator behind every number in the channel-leakage artifact.

WHAT STANDS BEHIND THIS. ``data/leakage/c1_6_channel_leakage.json`` reports, per
fire and per static channel, how well that channel alone separates burned from
unburned cells. Its published quantiles over 164 live pairs (median 0.0954, p90
0.3460, max 0.7022) were the evidence for ruling that the burnable-masked
measurement is the one of record. ``data/leakage.py`` is 343 statements at
**zero** line coverage, and ``score_field`` is the single function every one of
those numbers, plus every control that validates them, passes through. The
module says so itself: a control that exercises a different code path from the
measurement it validates is not a control. That makes this function the highest
single-point-of-failure in the file.

THE TEST THAT MATTERS MOST IS THE DIFFERENTIAL. ``score_field`` computes the AUC
through tie-averaged Mann-Whitney ranks, which is O(n log n) and completely
opaque. The definition it implements is a mean over all burned/unburned pairs
with a half credit for ties, which is O(n^2) and completely transparent. Below,
the two are computed independently on tie-heavy integer fields and required to
agree exactly. That is the only assertion here that could catch a wrong
tie-handling constant, and tie handling is not a corner case in this corpus:
``fuel_model_id`` is categorical and the two barrier channels are {0,1}, so most
of the artifact's rows are almost entirely ties.

THE DEGENERATE VERDICTS ARE THE SECOND HALF. A channel that is constant over the
mask, a mask with no burned cell, a mask with no unburned cell and an empty mask
are four different reasons for "no number", and they must not collapse into one
another or into a numeric 0.0. A 0.0 in that artifact reads as "this channel
carries no leakage", which is a scientific claim; ``None`` with a named reason
reads as "not measurable here", which is a different one.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.data.leakage import (
    DEGENERATE_CONSTANT,
    DEGENERATE_EMPTY,
    DEGENERATE_NO_BURNED,
    DEGENERATE_NO_UNBURNED,
    _exceedance,
    _group,
    _quantile_block,
    _tail_composition,
    avg_ranks,
    score_field,
)


def _brute_force_auc(values: np.ndarray, burned: np.ndarray) -> float:
    """The definition, written out: mean over pairs, half credit for a tie.

    Deliberately quadratic and deliberately not sharing a line with the
    implementation under test.
    """
    positives = values[burned]
    negatives = values[~burned]
    total = 0.0
    for a in positives:
        for b in negatives:
            total += 1.0 if a > b else (0.5 if a == b else 0.0)
    return total / (positives.size * negatives.size)


# --------------------------------------------------------------------------
# ranks
# --------------------------------------------------------------------------


def test_tied_values_take_the_average_rank_and_not_an_arbitrary_order() -> None:
    """Order-dependent tie breaking would manufacture separation out of nothing.

    FAILS WHEN: the ranks come from ``argsort`` positions instead of the
    tie-averaged formula. On ``fuel_model_id``, which is nearly all ties, that
    would let the memory layout of the array decide how much leakage a channel
    reports.
    """
    assert avg_ranks(np.array([10.0, 20.0, 20.0, 30.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert avg_ranks(np.array([5.0, 5.0, 5.0])).tolist() == [2.0, 2.0, 2.0]
    assert avg_ranks(np.array([3.0, 1.0, 2.0])).tolist() == [3.0, 1.0, 2.0]

    shuffled = np.array([20.0, 30.0, 10.0, 20.0])
    assert sorted(avg_ranks(shuffled).tolist()) == [1.0, 2.5, 2.5, 4.0]


# --------------------------------------------------------------------------
# the estimator, against its own definition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 17, 101, 2718])
def test_the_ranked_auc_equals_the_pairwise_definition_on_tie_heavy_fields(seed: int) -> None:
    """A differential test against an independent quadratic implementation.

    Values are drawn from five integer classes over sixty cells, so ties are the
    normal case rather than an edge case, which is the regime the categorical
    channels in the artifact actually live in.

    FAILS WHEN: the ``n_pos * (n_pos + 1) / 2`` correction is off by one, or the
    tie-averaged ranks are replaced by ordinal ones. Both produce a smooth,
    plausible, systematically wrong leakage number for every categorical channel
    in the corpus, and neither changes any degenerate verdict.
    """
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 5, size=60).astype(float)
    burned = rng.random(60) < 0.4
    if burned.all() or not burned.any():  # pragma: no cover - guarded by the seeds used
        pytest.skip("degenerate draw")

    score = score_field(values, burned)
    assert score.auc == pytest.approx(_brute_force_auc(values, burned), abs=1e-12)
    assert score.leakage == pytest.approx(abs(2.0 * score.auc - 1.0), abs=1e-12)
    assert score.tie_fraction > 0.5, "the premise: this field is mostly ties"


def test_perfect_separation_scores_one_in_both_directions() -> None:
    """The statistic is direction-blind by construction, and that is deliberate.

    A channel that predicts burning and a channel that predicts not-burning are
    equally leaky; only the magnitude is the leakage. The AUC beside it keeps the
    direction so a reader can still tell which way it went.

    FAILS WHEN: the absolute value is dropped, at which point a perfectly
    anti-correlated channel reports leakage -1.0 and sorts to the bottom of every
    quantile table in the artifact.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    burned = np.array([False, False, True, True])

    forward = score_field(values, burned)
    assert forward.leakage == pytest.approx(1.0)
    assert forward.auc == pytest.approx(1.0)

    reversed_field = score_field(-values, burned)
    assert reversed_field.leakage == pytest.approx(1.0)
    assert reversed_field.auc == pytest.approx(0.0), "the sign survives in the AUC"


def test_the_mask_restricts_the_sample_before_anything_is_scored() -> None:
    """The burnable mask is the measurement of record, so it must actually bite.

    FAILS WHEN: the mask is applied to the values and not to the labels, or
    after the counts are taken, which would score the burnable subset against
    the whole footprint and silently mix the two masks.
    """
    values = np.array([0.0, 0.0, 1.0, 1.0, 5.0, 5.0])
    burned = np.array([False, True, False, True, False, True])
    mask = np.array([True, True, True, True, False, False])

    masked = score_field(values, burned, mask)
    assert masked.n_cells == 4
    assert masked.n_burned == 2 and masked.n_unburned == 2
    assert masked.auc == pytest.approx(_brute_force_auc(values[mask], burned[mask]), abs=1e-12)
    assert score_field(values, burned).n_cells == 6


def test_the_null_sd_carries_the_tie_correction_and_shrinks_with_sample_size() -> None:
    """The scale the leakage is read against, so it must respond to both terms.

    FAILS WHEN: the tie correction term is dropped, which inflates the null
    spread on exactly the categorical channels where ties dominate and makes a
    real separation look like noise.
    """
    rng = np.random.default_rng(5)
    small = np.arange(20.0)
    large = np.arange(200.0)
    burned_small = rng.random(20) < 0.5
    burned_large = rng.random(200) < 0.5

    sd_small = score_field(small, burned_small).null_sd
    sd_large = score_field(large, burned_large).null_sd
    assert sd_small is not None and sd_large is not None
    assert sd_large < sd_small, "more cells, tighter null"

    tied = np.repeat(np.arange(10.0), 20)
    tied[0] = -1.0  # keep it non-constant so it is scored rather than refused
    burned_tied = rng.random(tied.size) < 0.5
    tied_score = score_field(tied, burned_tied)
    untied_score = score_field(np.arange(float(tied.size)), burned_tied)
    assert tied_score.null_sd is not None and untied_score.null_sd is not None
    assert tied_score.null_sd < untied_score.null_sd


# --------------------------------------------------------------------------
# the four ways a measurement can be absent
# --------------------------------------------------------------------------


def test_each_degenerate_case_is_named_and_none_of_them_returns_a_number() -> None:
    """Four different reasons for no number, and none of them is 0.0.

    A numeric 0.0 in this artifact is the claim "this channel separates nothing".
    ``None`` with a reason is the claim "this pair could not be measured". They
    are different statements and one of them would be false.

    FAILS WHEN: any branch returns ``leakage=0.0`` instead of ``None``, which
    would put unmeasurable pairs into the quantile tables as the strongest
    possible evidence of no leakage and drag every published percentile down.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    burned = np.array([False, False, True, True])

    cases = {
        DEGENERATE_EMPTY: score_field(values, burned, np.zeros(4, dtype=bool)),
        DEGENERATE_NO_BURNED: score_field(values, np.zeros(4, dtype=bool)),
        DEGENERATE_NO_UNBURNED: score_field(values, np.ones(4, dtype=bool)),
        DEGENERATE_CONSTANT: score_field(np.ones(4), burned),
    }
    for expected, score in cases.items():
        assert score.degenerate == expected
        assert score.leakage is None, f"{expected} must not report a number"
        assert score.auc is None

    assert cases[DEGENERATE_CONSTANT].n_distinct_values == 1
    assert cases[DEGENERATE_NO_BURNED].n_burned == 0
    assert cases[DEGENERATE_NO_UNBURNED].n_unburned == 0
    assert score_field(values, burned).degenerate is None


def test_a_two_valued_channel_is_scored_rather_than_called_constant() -> None:
    """The {0,1} barrier channels must reach the estimator, not the refusal.

    FAILS WHEN: the constant guard is written as ``n_distinct <= 2``, which would
    silently drop every binary mask channel out of the artifact while the file
    still looks complete.
    """
    values = np.array([0.0, 0.0, 1.0, 1.0])
    burned = np.array([False, True, False, True])
    score = score_field(values, burned)
    assert score.degenerate is None
    assert score.n_distinct_values == 2
    assert score.leakage == pytest.approx(0.0), "a channel independent of burning scores 0"


# --------------------------------------------------------------------------
# aggregation into the published tables
# --------------------------------------------------------------------------


def test_degenerate_rows_are_excluded_from_the_grouped_quantiles() -> None:
    """The denominator of every published quantile.

    FAILS WHEN: the ``leakage is None`` filter is dropped and ``float(None)``
    raises, or worse, is replaced by a default that inserts a fabricated value
    into the distribution the ruling was made on.
    """
    rows = [
        {"mask": "burnable", "channel": "slope", "leakage": 0.2},
        {"mask": "burnable", "channel": "slope", "leakage": 0.4},
        {"mask": "burnable", "channel": "slope", "leakage": None},
        {"mask": "all_cells", "channel": "slope", "leakage": 0.9},
    ]
    grouped = _group(rows, "channel", "burnable")
    assert grouped["slope"]["n"] == 2
    assert grouped["slope"]["mean"] == pytest.approx(0.3)
    assert grouped["slope"]["p50"] == pytest.approx(0.3)
    assert _group(rows, "channel", "all_cells")["slope"]["n"] == 1


def test_an_empty_quantile_block_reports_a_count_and_no_percentiles() -> None:
    """FAILS WHEN: an empty group returns percentiles computed over an empty
    array, which are NaN and serialise to null, so a group with no data becomes
    indistinguishable from a group whose median is unknown for another reason."""
    assert _quantile_block([]) == {"n": 0}
    block = _quantile_block([0.0, 1.0])
    assert block["n"] == 2
    assert block["p0"] == 0.0 and block["p100"] == 1.0 and block["p50"] == pytest.approx(0.5)


def test_the_tail_is_defined_by_RANK_and_never_by_a_value_threshold() -> None:
    """The tail table describes the sample's shape and must not read as a bar.

    FAILS WHEN: the tail is cut at a fixed leakage value, which would turn a
    descriptive table into an implied threshold and would silently change size
    with the corpus.
    """
    rows = [
        {
            "mask": "burnable",
            "leakage": v,
            "fire_id": f"fire_{i}",
            "channel": "slope",
            "spatial_block_id": i % 2,
            "label_source": "gofer",
        }
        for i, v in enumerate([0.9, 0.8, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])
    ]
    tail = _tail_composition(rows, "burnable", 0.2)
    assert tail["k"] == 2
    assert tail["cut_value_at_rank_k"] == pytest.approx(0.8)
    assert sorted(name for name, _, _ in tail["by_fire"]) == ["fire_0", "fire_1"]
    assert sum(n for _, n, _ in tail["by_channel"]) == 2


def test_the_exceedance_curve_is_strictly_greater_than_each_grid_point() -> None:
    """It is the survival function; an inclusive comparison shifts the whole curve.

    FAILS WHEN: ``>`` becomes ``>=``, a same-length edit that adds every value
    sitting exactly on a grid point into the bin below it and moves a published
    curve without moving any underlying measurement.
    """
    curve = _exceedance([0.0, 0.05, 0.5, 1.0])
    as_dict = {g: n for g, n in curve}
    assert as_dict[0.0] == 3, "the exact 0.0 is not counted above 0.0"
    assert as_dict[0.05] == 2
    assert as_dict[0.5] == 1
    assert as_dict[1.0] == 0, "nothing exceeds the maximum of the statistic"
    assert curve[0][0] == 0.0 and curve[-1][0] == 1.0
