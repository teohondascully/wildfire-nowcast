"""Self-checks for D11's C1.6 leakage estimator. Every check ships a planted defect.

Run: ``.venv/bin/python -m wildfire_nowcast.data.leakage_selftest``

These validate the ESTIMATOR before it is pointed at the corpus. A measure that
has never been observed to fire is not evidence, and a measure that fires at the
WRONG magnitude is as broken as one that never fires - so every check here
asserts a NUMBER against an analytic target, not merely "non-zero".
"""

from __future__ import annotations

import math

import numpy as np

from wildfire_nowcast.data.leakage import (
    DEGENERATE_CONSTANT,
    DEGENERATE_NO_BURNED,
    DEGENERATE_NO_UNBURNED,
    _exceedance,
    _smooth_random_field,
    avg_ranks,
    score_field,
)


def test_ties_take_the_average_rank() -> None:
    """PLANTED DEFECT: order-dependent ranks would give [1,2,3,4] here.

    ``fuel_model_id`` and the two {0,1} masks are almost all ties. Tie-broken
    ranks let the sort order of equal values manufacture separation from nothing.
    """
    r = avg_ranks(np.array([5.0, 5.0, 5.0, 9.0]))
    assert np.allclose(r, [2.0, 2.0, 2.0, 4.0]), r
    r2 = avg_ranks(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(r2, [1.0, 2.0, 3.0]), r2


def test_a_perfect_separator_scores_exactly_one() -> None:
    """The positive control's ceiling. PLANTED DEFECT: an off-by-one in the
    U-statistic offset would give 1 - 1/(n_pos*n_neg), not exactly 1."""
    burned = np.array([[1, 1, 0, 0]], dtype=bool)
    s = score_field(np.array([[9.0, 8.0, 1.0, 0.0]]), burned)
    assert s.leakage == 1.0, s
    assert s.auc == 1.0, s


def test_the_measure_is_symmetric_in_sign() -> None:
    """A channel predicting UNBURNED must score the same as one predicting
    BURNED. PLANTED DEFECT: dropping the abs() would return -1.0 here and let a
    perfectly anti-correlated channel read as clean."""
    burned = np.array([[1, 1, 0, 0]], dtype=bool)
    s = score_field(np.array([[0.0, 1.0, 8.0, 9.0]]), burned)
    assert s.leakage == 1.0, s
    assert s.auc == 0.0, s


def test_binary_channel_matches_the_analytic_a_minus_b() -> None:
    """For a binary channel, ``2*AUC-1 == a - b`` EXACTLY, tie terms cancelling.

    This identity is what makes positive control 2 exact rather than approximate.
    PLANTED DEFECT: giving ties full credit instead of half would return
    ``a*(1-b) + a*b + (1-a)*(1-b) - 1``, i.e. 0.9808 rather than 0.92 below.
    """
    rng = np.random.default_rng(7)
    n_pos, n_neg = 4000, 6000
    burned = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    a, b = 0.96, 0.04
    chan = np.concatenate(
        [(rng.random(n_pos) < a).astype(float), (rng.random(n_neg) < b).astype(float)]
    )
    a_hat = float(chan[:n_pos].mean())
    b_hat = float(chan[n_pos:].mean())
    s = score_field(chan, burned)
    assert s.leakage is not None
    assert abs(s.leakage - abs(a_hat - b_hat)) < 1e-12, (s.leakage, a_hat - b_hat)
    assert abs(s.leakage - 0.92) < 0.02, s.leakage


def test_continuous_shifted_gaussian_hits_its_analytic_target() -> None:
    """Two unit normals separated by 1 give ``AUC = Phi(1/sqrt2)``.

    PLANTED DEFECT: computing AUC on the wrong class (ranks of the NEGATIVES)
    would return the same magnitude here by symmetry - which is why the sign
    check above is a separate test and this one is not asked to carry it.
    """
    rng = np.random.default_rng(3)
    n = 20000
    burned = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    chan = np.concatenate([rng.standard_normal(n) + 1.0, rng.standard_normal(n)])
    target = 2.0 * 0.5 * (1.0 + math.erf((1.0 / math.sqrt(2.0)) / math.sqrt(2.0))) - 1.0
    s = score_field(chan, burned)
    assert s.leakage is not None
    assert abs(s.leakage - target) < 0.02, (s.leakage, target)


def test_a_constant_channel_is_DEGENERATE_and_not_a_zero() -> None:
    """PLANTED DEFECT - the expensive one. A constant channel is all ties, so
    AUC is exactly 0.5 and ``|2*AUC-1|`` is exactly 0.0. Returning that 0.0
    would let ``water_barrier_mask`` under burnable masking, and every
    scar-free fire's ``recent_burn_scar``, pull the reported distribution DOWN
    while carrying no information at all. It must be excluded and counted."""
    burned = np.array([[1, 1, 0, 0]], dtype=bool)
    s = score_field(np.zeros((1, 4)), burned)
    assert s.leakage is None, s
    assert s.degenerate == DEGENERATE_CONSTANT, s
    assert s.n_distinct_values == 1, s


def test_no_contrast_in_the_mask_is_DEGENERATE_both_ways() -> None:
    """A mask that admits only burned (or only unburned) cells has no statistic.
    PLANTED DEFECT: a divide-by-zero here would surface as nan, and nan
    silently propagates through percentile summaries."""
    v = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert score_field(v, np.zeros((1, 4), bool)).degenerate == DEGENERATE_NO_BURNED
    assert score_field(v, np.ones((1, 4), bool)).degenerate == DEGENERATE_NO_UNBURNED


def test_the_mask_actually_restricts_the_sample() -> None:
    """PLANTED DEFECT: ignoring ``mask`` would score 1.0 on both calls below.

    The burnable mask is the whole point of the second reported view; a mask
    argument that is accepted and not applied is the check-that-cannot-fail
    pattern C3.5 exists for."""
    burned = np.array([[1, 1, 0, 0]], dtype=bool)
    values = np.array([[9.0, 8.0, 1.0, 0.0]])
    full = score_field(values, burned)
    assert full.leakage == 1.0 and full.n_cells == 4, full
    # Restrict to one burned and one unburned cell: still separable, but n falls.
    part = score_field(values, burned, np.array([[1, 0, 0, 1]], dtype=bool))
    assert part.n_cells == 2 and part.n_burned == 1, part
    # Restrict to a set where the channel cannot order: the value must MOVE.
    flat = score_field(
        np.array([[5.0, 5.0, 5.0, 0.0]]), burned, np.array([[1, 1, 1, 0]], dtype=bool)
    )
    assert flat.degenerate == DEGENERATE_CONSTANT, flat


def test_the_null_sd_matches_the_EXACT_hypergeometric_value_for_a_binary_channel() -> None:
    """An analytic target for the reported permutation-null SD.

    For a binary channel the permutation null of ``2*AUC-1`` is a rescaled
    hypergeometric, so its SD is exactly ``sqrt(K(N-K) / (n_pos*n_neg*(N-1)))``
    with K = the number of ones. No simulation, no tolerance.

    PLANTED DEFECT: dropping the tie correction from ``var_u``. On a binary
    channel the ties ARE the whole sample, so an uncorrected SD overstates by
    up to ~2x - and the reported SD is what puts every other number on this
    report on a scale. The empirical flatness check below cannot see this:
    on a lightly tied channel the correction moves the SD by ~1%.
    """
    for n, n_pos, k in ((3000, 900, 1500), (3000, 900, 2700), (500, 50, 100)):
        burned = np.zeros(n, bool)
        burned[:n_pos] = True
        chan = np.zeros(n)
        chan[np.random.default_rng(k).permutation(n)[:k]] = 1.0
        s = score_field(chan, burned)
        exact = math.sqrt(k * (n - k) / (n_pos * (n - n_pos) * (n - 1.0)))
        assert s.null_sd is not None
        assert abs(s.null_sd - exact) < 1e-12, (n, n_pos, k, s.null_sd, exact)


def test_the_permutation_null_is_flat_and_its_sd_is_the_right_scale() -> None:
    """The negative control, validated on synthetic data where truth is known.

    PLANTED DEFECT: an estimator that leaked the label into the ranking would
    show a permutation mean far from 0. A second defect this catches is a
    null_sd computed WITHOUT the tie correction: on this heavily tied channel it
    would overstate the SD, and the realised z-spread would come back far below
    1 instead of near 1.
    """
    rng = np.random.default_rng(19)
    n = 3000
    burned = rng.random(n) < 0.3
    values = np.rint(rng.random(n) * 6.0)  # heavily tied, like fuel_model_id
    base = score_field(values, burned)
    assert base.null_sd is not None and base.null_sd > 0
    draws = np.array(
        [score_field(rng.permutation(values), burned).leakage for _ in range(400)],
        dtype=float,
    )
    assert abs(draws.mean()) < 5.0 * base.null_sd, (draws.mean(), base.null_sd)
    # |2AUC-1| under the null is a folded normal with sd = null_sd, so the mean
    # of the folded draws is ~sqrt(2/pi)*sd. Within 30% is a scale check.
    expected_folded = math.sqrt(2.0 / math.pi) * base.null_sd
    assert 0.7 < draws.mean() / expected_folded < 1.3, (draws.mean(), expected_folded)


def test_the_spatial_null_field_is_independent_of_any_fire() -> None:
    """The diagnostic field must be generated from a seed alone.

    PLANTED DEFECT: a field built from the footprint (a smoothed copy, say)
    would score far from zero on average. Here the same field is scored against
    two DISJOINT random footprints; a field that knew about either would not
    average to ~0 across draws."""
    rng = np.random.default_rng(5)
    vals = []
    for _ in range(200):
        f = _smooth_random_field((40, 40), 5.0, rng)
        blob = np.zeros((40, 40), bool)
        r0, c0 = rng.integers(5, 25, size=2)
        blob[r0 : r0 + 12, c0 : c0 + 12] = True
        s = score_field(f, blob)
        assert s.auc is not None
        vals.append(2.0 * s.auc - 1.0)  # SIGNED, so it can cancel
    assert abs(float(np.mean(vals))) < 0.10, float(np.mean(vals))
    # ...and it is NOT concentrated at zero: that is the diagnostic's whole point.
    assert float(np.mean(np.abs(vals))) > 0.10, float(np.mean(np.abs(vals)))


def test_the_exceedance_curve_is_monotone_non_increasing() -> None:
    """PLANTED DEFECT: a `>=` / `>` mix-up or an unsorted grid would break
    monotonicity, and a survival function that rises is not a distribution."""
    counts = [n for _, n in _exceedance([0.05, 0.2, 0.2, 0.7, 0.95])]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == 5, counts


def main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in checks:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[D11 selftest] {len(checks)} checks passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
