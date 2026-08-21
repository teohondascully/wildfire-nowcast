"""G3's calibration criterion (ADR-020), validated against KNOWN answers.

``common/calibration.py`` is the module G3's rewritten bar NAMES, and until this
file existed it had ZERO test coverage — 469 lines deciding half a gate, checked
by nothing. This is the same ordering C6.4 and C6.0 were built under and the one
we keep getting right: **fix the gate's instrument before the result exists.**
No ensemble has been trained yet, so nobody here knows which way any of these
numbers cuts.

What this file establishes, in order of importance:

1. **The headline numbers are what the module claims.** An ORACLE scores exactly
   0, a forecast that predicts nothing scores exactly the base rate of the scored
   set, and CLIMATOLOGY — the forecast ADR-020 was written about — passes the
   forecast-bin term and FAILS the frontier term. If that last one ever stops
   holding, ADR-020's second mechanism is gone and the criterion is back to being
   satisfiable by zero information.
2. **The criterion is the WORST family, and the arithmetic is exact.** ``max`` of
   the two terms, pooled by sufficient statistics so many windows give bitwise
   the same answer as one big one.
3. **It is in the units of the bar.** Linear (points), not squared, which is the
   defect that let silence win REL.
4. **Unscoreable is not zero.** An empty scored set returns ``None``; returning
   0.0 would score an unevaluated forecast as perfect (C-1).

Model-blind by construction: nothing here imports ``model/``, loads a checkpoint
or reads ``runs/`` — asserted below by grep rather than promised in prose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wildfire_nowcast.common import calibration as K
from wildfire_nowcast.common.paths import repo_root

# --------------------------------------------------------------------------
# the primitive: a deviation in POINTS, on numbers whose answer is arithmetic
# --------------------------------------------------------------------------


def test_weighted_abs_deviation_is_occupancy_weighted_and_linear() -> None:
    """Hand-computed: (10*|0.2-0.5| + 30*|0.8-0.7|) / 40 = (3.0 + 3.0)/40 = 0.15."""
    strata = [
        K.Stratum(key=0, n=10, sum_p=2.0, sum_y=5.0),  # mean p 0.2, obs 0.5
        K.Stratum(key=1, n=30, sum_p=24.0, sum_y=21.0),  # mean p 0.8, obs 0.7
    ]
    assert K.weighted_abs_deviation(strata) == pytest.approx(0.15)


def test_the_aggregation_is_LINEAR_which_is_the_whole_ADR_020_fix() -> None:
    """Squaring is what let a silent forecast win. Assert the two disagree.

    One stratum deviating by 0.30 and one by 0.00 gives a LINEAR mean of 0.15 and
    a SQUARED mean of 0.045 — the squared form flatters the larger error, which
    is exactly how REL scored a do-nothing null (0.0050) above genuine skill
    (0.0099) while the bar was written in points.
    """
    strata = [
        K.Stratum(key=0, n=50, sum_p=10.0, sum_y=25.0),  # p 0.2, y 0.5 -> dev 0.30
        K.Stratum(key=1, n=50, sum_p=25.0, sum_y=25.0),  # p 0.5, y 0.5 -> dev 0.00
    ]
    linear = K.weighted_abs_deviation(strata)
    squared = sum(s.n * ((s.sum_p / s.n) - (s.sum_y / s.n)) ** 2 for s in strata) / 100
    assert linear == pytest.approx(0.15)
    assert squared == pytest.approx(0.045)
    assert linear > squared, "if these ever agree, the units of the bar have moved"


def test_an_empty_scored_set_is_None_and_not_zero() -> None:
    """C-1: unverifiable is a failure, not a pass. 0.0 here would read PERFECT."""
    assert K.weighted_abs_deviation([]) is None
    assert K.weighted_abs_deviation([K.Stratum(key=0, n=0, sum_p=0.0, sum_y=0.0)]) is None


def test_strata_stats_retains_empty_strata_and_validates_its_index() -> None:
    idx = np.array([0, 0, 3])
    strata = K.strata_stats(idx, np.array([0.5, 0.7, 0.1]), np.array([1.0, 0.0, 0.0]), 5)
    assert len(strata) == 5, "an unoccupied subgroup must be visibly unoccupied, not absent"
    assert [s.n for s in strata] == [2, 0, 0, 1, 0]
    assert strata[0].mean_forecast == pytest.approx(0.6)
    assert strata[0].observed_frequency == pytest.approx(0.5)
    assert strata[1].mean_forecast is None

    with pytest.raises(ValueError, match="outside"):
        K.strata_stats(np.array([7]), np.array([0.5]), np.array([1.0]), 5)
    with pytest.raises(ValueError, match="same length"):
        K.strata_stats(np.array([0, 1]), np.array([0.5]), np.array([1.0]), 5)


# --------------------------------------------------------------------------
# pooling — sufficient statistics, so many windows == one big window
# --------------------------------------------------------------------------


def test_pooling_many_windows_equals_scoring_them_as_one() -> None:
    """Exactness, not agreement to a tolerance: these are sums of the same numbers."""
    rng = np.random.default_rng(11)
    p = rng.random(600)
    y = (rng.random(600) < p).astype(np.float64)
    idx = rng.integers(0, 6, size=600)

    whole = K.strata_stats(idx, p, y, 6)
    chunks = [
        K.strata_stats(idx[a:b], p[a:b], y[a:b], 6) for a, b in ((0, 137), (137, 400), (400, 600))
    ]
    pooled = K.pool_strata(chunks)
    assert [(s.key, s.n) for s in pooled] == [(s.key, s.n) for s in whole]
    for a, b in zip(pooled, whole, strict=True):
        assert a.sum_p == pytest.approx(b.sum_p, abs=1e-12)
        assert a.sum_y == pytest.approx(b.sum_y, abs=1e-12)
    assert K.weighted_abs_deviation(pooled) == pytest.approx(
        K.weighted_abs_deviation(whole), abs=1e-15
    )


def test_pool_first_then_deviate_because_the_other_order_is_biased_upward() -> None:
    """The docstring's ordering claim, measured rather than asserted in prose.

    A PERFECTLY calibrated forecast (p = 0.30 everywhere, outcomes drawn at 0.30)
    scores ~0 when the strata are pooled first. Averaging per-window deviations
    instead scores it well above 0, because |mean p - mean y| in a 3-cell stratum
    is dominated by sampling noise and |.| cannot cancel. A gate criterion
    computed the wrong way round would penalise the very property G3 certifies.
    """
    rng = np.random.default_rng(3)
    windows = []
    for _ in range(400):
        p = np.full(3, 0.30)
        y = (rng.random(3) < 0.30).astype(np.float64)
        windows.append(K.strata_stats(np.zeros(3, dtype=int), p, y, 1))

    pooled_then_deviated = K.weighted_abs_deviation(K.pool_strata(windows))
    deviated_then_pooled = float(np.mean([K.weighted_abs_deviation(w) for w in windows]))

    assert pooled_then_deviated == pytest.approx(0.0, abs=0.02)
    assert deviated_then_pooled > 0.15
    assert deviated_then_pooled > 5 * pooled_then_deviated


# --------------------------------------------------------------------------
# the frontier partition — from x0 ALONE, and it must match the band it strata-fies
# --------------------------------------------------------------------------


def test_frontier_rings_are_chebyshev_and_start_at_the_burned_region() -> None:
    x0 = np.zeros((11, 11), dtype=np.uint8)
    x0[5, 5] = 1
    rings = K.frontier_rings(x0, 3)
    assert rings[5, 5] == 0
    assert rings[5, 6] == 1 and rings[4, 4] == 1, "8-connected: diagonals are ring 1"
    assert rings[5, 7] == 2 and rings[3, 3] == 2
    assert rings[5, 8] == 3
    assert rings[5, 9] == 4, "beyond the cap is its own bucket, not folded into the last ring"
    with pytest.raises(ValueError, match="max_radius"):
        K.frontier_rings(x0, 0)


def test_the_rings_partition_the_growth_band_exactly() -> None:
    """The docstring says rings 1..R partition ``growth_band``. Assert it.

    If these two ever used different stencils, the criterion would be scoring
    cells the mask excludes (or missing cells it includes) and the number would
    quietly stop being about the decision set G3 names.
    """
    from wildfire_nowcast.eval.masks import growth_band

    rng = np.random.default_rng(5)
    x0 = (rng.random((30, 30)) < 0.06).astype(np.uint8)
    for radius in (1, 2, 4, 7):
        rings = K.frontier_rings(x0, radius)
        band = growth_band(x0, radius)
        assert np.array_equal(band, (rings >= 1) & (rings <= radius))
        assert np.array_equal(rings == 0, x0 > 0)


# --------------------------------------------------------------------------
# THE HEADLINE NUMBERS — null / skill / oracle, on a constructed scenario
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenario() -> dict:
    """One lead of a spread problem with a KNOWN radial structure.

    Burn probability falls off with distance from the frontier — ring 1 burns at
    0.40, ring 2 at 0.20, ring 3 at 0.05 — so a forecast with the wrong radial
    profile is genuinely miscalibrated and a forecast with the right one is not.
    Outcomes are drawn from those rates, so the ground truth of every assertion
    below is arithmetic rather than a fitted expectation.

    The domain is deliberately LARGE (1,476 scored cells). At the 132 cells a
    60x60 domain gives, the correctly-specified forecast scores 0.061 — not
    because it is miscalibrated but because ``|mean p - mean y|`` in a small
    stratum is sampling noise that ``|.|`` cannot cancel. A fixture that noisy
    cannot tell "calibrated" from "close", so it would let a real regression
    through while looking green. Measured across sizes: n=132 -> 0.0606,
    n=756 -> 0.0423, n=1476 -> 0.0156 against a null of 0.229.
    """
    rng = np.random.default_rng(20260808)
    x0 = np.zeros((240, 240), dtype=np.uint8)
    x0[60:180, 60:180] = 2
    x0[62:178, 62:178] = 1
    radius = 3
    rings = K.frontier_rings(x0, radius)
    mask = (rings >= 1) & (rings <= radius)

    rate_by_ring = {1: 0.40, 2: 0.20, 3: 0.05}
    true_rate = np.zeros(x0.shape, dtype=np.float64)
    for r, rate in rate_by_ring.items():
        true_rate[rings == r] = rate
    y = ((rng.random(x0.shape) < true_rate) & mask).astype(np.float64)

    return {
        "x0": x0,
        "rings": rings,
        "n_rings": radius + 2,
        "mask": mask,
        "y": y,
        "true_rate": true_rate,
        "base_rate": float(y[mask].mean()),
    }


def _terms(scenario: dict, prob: np.ndarray) -> K.CalibrationTerms:
    return K.calibration_terms(
        prob,
        scenario["y"],
        scenario["mask"],
        rings=scenario["rings"],
        n_rings=scenario["n_rings"],
    )


def test_the_oracle_scores_exactly_zero(scenario) -> None:
    """A forecast equal to the observed outcome deviates from it by 0 everywhere.

    Exactly 0, on both families and therefore on the criterion — the minimum of
    the range, which is what makes a criterion fit to gate (the same standard
    ``best_member_iou_shape_masked`` had to meet).
    """
    terms = _terms(scenario, scenario["y"])
    assert terms.bins == pytest.approx(0.0, abs=1e-12)
    assert terms.frontier == pytest.approx(0.0, abs=1e-12)
    assert terms.error == pytest.approx(0.0, abs=1e-12)


def test_a_forecast_that_predicts_nothing_scores_exactly_the_base_rate(scenario) -> None:
    """The null floor is a LABEL STATISTIC, and the module reports it as one.

    p = 0 everywhere: every stratum deviates by its own observed frequency, and
    the occupancy-weighted mean of those is the base rate of the scored set. That
    the number is EXACTLY the base rate — on both families — is what makes the
    floor auditable without running a model.
    """
    terms = _terms(scenario, np.zeros_like(scenario["y"]))
    base = scenario["base_rate"]
    assert base > 0.05, "a fixture with no growth would make this vacuous"
    assert terms.bins == pytest.approx(base, abs=1e-12)
    assert terms.frontier == pytest.approx(base, abs=1e-12)
    assert terms.error == pytest.approx(base, abs=1e-12)
    assert terms.silent_floor == pytest.approx(base, abs=1e-12)


def test_the_well_specified_forecast_beats_the_null_by_a_wide_margin(scenario) -> None:
    """Genuine skill: the true per-ring rate. It must beat the null decisively.

    This is the assertion that would fail if the criterion were unwinnable — the
    complement of the null test, and just as necessary. A bar no correct forecast
    can clear is as broken as one silence can clear.
    """
    terms = _terms(scenario, scenario["true_rate"])
    null = _terms(scenario, np.zeros_like(scenario["y"]))
    clim = _terms(scenario, np.where(scenario["mask"], scenario["base_rate"], 0.0))
    assert terms.error is not None and null.error is not None
    # Measured on this fixture: skill 0.0156 vs null 0.2290, ratio 0.068.
    assert terms.error < 0.15 * null.error
    assert terms.error < 0.03, "what is left is the scored set's own sampling noise"
    assert terms.error < 0.25 * clim.error


def test_CLIMATOLOGY_passes_the_bin_term_and_FAILS_the_frontier_term(scenario) -> None:
    """ADR-020's second mechanism, which is the entire reason for family B.

    A forecast issuing the single marginal base rate on every band cell is
    perfectly calibrated against its OWN bins — one bin, sitting on the diagonal —
    while carrying zero information. It is wrong inside every ring: it
    under-predicts ring 1 and over-predicts ring 3. Family A cannot see that;
    family B is defined so that it can, and ``max`` is what makes the criterion
    inherit the stronger of the two.

    **If this test ever goes green by both terms being small, the criterion has
    become satisfiable by zero information and G3's calibration half is void.**
    """
    clim = np.where(scenario["mask"], scenario["base_rate"], 0.0)
    terms = _terms(scenario, clim)

    assert terms.bins == pytest.approx(0.0, abs=0.01), "climatology IS marginally calibrated"
    assert terms.frontier > 0.08, "and it is miscalibrated inside every ring"
    assert terms.error == pytest.approx(terms.frontier), "max() takes the worse family"
    assert terms.error > 10 * max(terms.bins, 1e-6)

    # ...and the criterion ranks it below genuine skill, which the bin term alone
    # would not: on family A, climatology (0.00) BEATS the true-rate forecast.
    skill = _terms(scenario, scenario["true_rate"])
    assert terms.error > skill.error
    assert clim_beats_skill_on_bins(terms, skill), (
        "the bin term is supposed to be fooled here — if it is not, this fixture no "
        "longer reproduces the pathology ADR-020 was written about"
    )


def clim_beats_skill_on_bins(clim: K.CalibrationTerms, skill: K.CalibrationTerms) -> bool:
    return (clim.bins or 0.0) <= (skill.bins or 0.0)


def test_a_confidently_wrong_forecast_scores_worse_than_the_null(scenario) -> None:
    """Sanity floor in the other direction: over-confidence must be punished.

    p = 0.9 on the whole band against a ~20% base rate. A criterion that scored
    this at or below silence would be rewarding noise, which is the mirror image
    of rewarding silence and just as disqualifying.
    """
    terms = _terms(scenario, np.where(scenario["mask"], 0.9, 0.0))
    null = _terms(scenario, np.zeros_like(scenario["y"]))
    assert terms.error is not None and null.error is not None
    assert terms.error > null.error
    assert terms.error > 0.5


def test_the_criterion_is_flat_in_how_the_scored_set_is_chunked(scenario) -> None:
    """Pooled over windows == scored whole. The gate number must not move with batching."""
    prob = scenario["true_rate"]
    whole = _terms(scenario, prob)

    halves = []
    for sel in (np.s_[:30, :], np.s_[30:, :]):
        sub_mask = np.zeros_like(scenario["mask"])
        sub_mask[sel] = scenario["mask"][sel]
        halves.append(
            (
                K.bin_strata(prob, scenario["y"], sub_mask),
                K.ring_strata(
                    prob, scenario["y"], sub_mask, scenario["rings"], scenario["n_rings"]
                ),
            )
        )
    pooled = K.terms_from_strata(
        K.pool_strata([h[0] for h in halves]), K.pool_strata([h[1] for h in halves])
    )
    assert pooled.error == pytest.approx(whole.error, abs=1e-12)
    assert pooled.bins == pytest.approx(whole.bins, abs=1e-12)
    assert pooled.frontier == pytest.approx(whole.frontier, abs=1e-12)


# --------------------------------------------------------------------------
# C0 — one implementation. The bin term must BE the ECE C6 already computes.
# --------------------------------------------------------------------------


def test_the_bin_term_is_bit_identical_to_c6s_own_ece_on_the_production_path(
    scenario,
) -> None:
    """C0, with the boundary drawn where it actually is.

    The PRODUCTION path is the one C6 uses: it bins the forecast ONCE for its
    reliability diagram and hands those sufficient statistics to
    ``terms_from_strata``. There is one binning, so the criterion's bin term is
    BITWISE the ECE C6 reports — asserted with ``==``, because "two
    implementations of one quantity" is how a producer and a verifier disagree
    while both look right.
    """
    from wildfire_nowcast.eval.metrics import _reliability_summary, reliability

    prob = scenario["true_rate"] * 0.8 + 0.03
    mask, y = scenario["mask"], scenario["y"]
    c6_bins = reliability(prob[mask], y[mask])
    c6_ece = _reliability_summary(c6_bins)["ece"]

    assert K.terms_from_strata(c6_bins, None).bins == c6_ece


def test_the_standalone_binner_agrees_with_c6_to_floating_point_not_bitwise(
    scenario,
) -> None:
    """The overclaim I found in my own docstring, pinned at its true strength.

    ``calibration.py`` said the bin term is "asserted bit-identical" to C6's ECE.
    That is true only on the production path above. ``bin_strata`` is the
    convenience path for callers that have NOT already binned, and it sums by
    ``bincount`` where C6 sums in a Python loop — same quantity, different
    association order, so they differ in the last 1-2 ulp (measured 6.9e-17 on
    this fixture). Docstring corrected; the achievable guarantee is asserted here
    rather than a stronger one being claimed and never checked.

    The distinction is not pedantry: C0 is satisfied because the number that
    GATES comes from one binning. What this test protects is the weaker promise
    that the two paths use the same edges and the same right-open convention — a
    real bug (an off-by-one in ``digitize``) would show up here as a gross
    disagreement, not as an ulp.
    """
    from wildfire_nowcast.eval.metrics import _reliability_summary, reliability

    prob = scenario["true_rate"] * 0.8 + 0.03
    mask, y = scenario["mask"], scenario["y"]
    c6_ece = _reliability_summary(reliability(prob[mask], y[mask]))["ece"]
    standalone = K.weighted_abs_deviation(K.bin_strata(prob, y, mask))

    assert standalone == pytest.approx(c6_ece, rel=0.0, abs=1e-14)

    # Same edges and right-open convention, asserted structurally: a value
    # exactly on a bin edge must land in the SAME stratum in both.
    edge_probs = np.array([0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
    sel = np.zeros(mask.shape, dtype=bool)
    flat = np.flatnonzero(mask.ravel())[: edge_probs.size]
    sel.ravel()[flat] = True
    p_field = np.zeros(mask.shape)
    p_field.ravel()[flat] = edge_probs
    ours = [s.n for s in K.bin_strata(p_field, y, sel)]
    theirs = [b["n"] for b in reliability(edge_probs, y.ravel()[flat])]
    assert ours == theirs


def test_the_gate_key_and_mask_are_named_in_code_not_in_a_table(scenario) -> None:
    """A downstream table must not be able to pick a different criterion silently."""
    assert K.GATE_CRITERION_KEY == "calibration_error"
    assert K.GATE_MASK == "growth_band"
    keys = _terms(scenario, scenario["true_rate"]).as_dict()
    assert K.GATE_CRITERION_KEY in keys
    assert f"{K.GATE_CRITERION_KEY}_bins" in keys
    assert f"{K.GATE_CRITERION_KEY}_frontier" in keys
    assert f"{K.GATE_CRITERION_KEY}_silent_floor" in keys
    assert set(K.SUBGROUP_FAMILIES) == {"bins", "frontier"}


# --------------------------------------------------------------------------
# check() raises — the IouTerms standard, so a shape change cannot pass quietly
# --------------------------------------------------------------------------


def test_check_raises_when_the_criterion_is_not_the_worst_family() -> None:
    good = K.CalibrationTerms(
        error=0.4,
        bins=0.1,
        frontier=0.4,
        silent_floor=0.2,
        n_scored=10,
        n_occupied_bins=2,
        n_occupied_rings=3,
    )
    assert good.check() is good

    with pytest.raises(AssertionError, match="WORST subgroup family"):
        K.CalibrationTerms(
            error=0.1,
            bins=0.1,
            frontier=0.4,
            silent_floor=0.2,
            n_scored=10,
            n_occupied_bins=2,
            n_occupied_rings=3,
        ).check()


def test_check_raises_when_a_term_is_not_a_probability() -> None:
    for bad in (1.5, -0.2, float("nan"), float("inf")):
        with pytest.raises(AssertionError, match="probability"):
            K.CalibrationTerms(
                error=bad,
                bins=bad,
                frontier=None,
                silent_floor=None,
                n_scored=10,
                n_occupied_bins=1,
                n_occupied_rings=0,
            ).check()


def test_a_missing_frontier_family_says_why_rather_than_scoring_silently() -> None:
    """Without x0 only the term climatology satisfies trivially is checked.

    The criterion still returns a number — it has to, the bin term is real — so
    the DECLARATION is what stops that number being read as the gate criterion.
    C-1 again: a silently weaker check is worse than a declared one.
    """
    terms = K.terms_from_strata([K.Stratum(key=0, n=8, sum_p=2.0, sum_y=2.0)], None)
    assert terms.frontier is None
    assert "climatology" in terms.unavailable_reason
    assert "x0" in terms.unavailable_reason
    assert terms.block_dict()["calibration_unavailable_reason"] == terms.unavailable_reason


def test_no_family_scored_means_no_criterion() -> None:
    terms = K.terms_from_strata(None, None)
    assert terms.error is None and terms.bins is None and terms.frontier is None
    assert terms.silent_floor is None
    assert terms.n_scored == 0


# --------------------------------------------------------------------------
# model-blindness, by grep — the property that licenses every number above
# --------------------------------------------------------------------------


def test_calibration_is_model_blind_by_construction() -> None:
    """Checkable, not promised. A gate instrument that can see the model under
    test can be tuned by it, and this file is the one that says it cannot.

    Checked by AST rather than by ``grep``, because grep cannot tell a reference
    from a mention: both files DISCUSS ``model/`` and ``runs/`` at length in their
    own docstrings, and a substring check flags that prose. The real property is
    that no import reaches ``model/`` and no executable string literal names a
    run directory — which is what an AST walk can state and a substring cannot.
    """
    import ast

    for rel in ("src/wildfire_nowcast/common/calibration.py", "tests/test_calibration.py"):
        tree = ast.parse(Path(repo_root() / rel).read_text())

        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [m for m in imported if m.startswith("wildfire_nowcast.model")], (
            f"{rel} imports model/: {imported}"
        )

        docstrings = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)
        }
        if rel.startswith("tests/"):
            # The live-string half cannot be applied to THIS file: its own
            # forbidden-token list and assertion messages are live string
            # literals, so it would always flag itself. The substantive property
            # for a test file is the import check above.
            continue

        live_strings = [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
        ]
        assert not [s for s in live_strings if "runs/" in s or "checkpoint" in s], (
            f"{rel} names a run directory or a checkpoint in live code"
        )
