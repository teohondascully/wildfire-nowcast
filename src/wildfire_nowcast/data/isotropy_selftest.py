"""Self-checks for D10's isotropy machinery. Every check ships a planted defect.

Run: ``.venv/bin/python -m wildfire_nowcast.data.isotropy_selftest``

These validate the ESTIMATOR and the INTERVAL before either is pointed at the
corpus. A statistic that has never been observed to fire is not evidence, and an
interval that cannot widen under clustering is not an interval.
"""

from __future__ import annotations

import numpy as np

from wildfire_nowcast.data.crossings import _cosine
from wildfire_nowcast.data.isotropy import (
    BodyRecord,
    _ceiling,
    _lattice_realisable_gaps,
    _ring_ceiling,
    _tie_averaged_ring_direction,
    cluster_bootstrap,
    negative_controls,
    positive_control,
)


def _body(
    fire: str, block: int, hour: int, u: float, v: float, east: float, north: float
) -> BodyRecord:
    n = float(np.hypot(east, north)) or 1.0
    c = _cosine(u, v, -north / n, east / n)
    return BodyRecord(
        fire_id=fire,
        spatial_block_id=block,
        cv_fold=0,
        split_role="train",
        label_source="gofer",
        hour=hour,
        n_cells=1,
        gap_km=2.0,
        merges_later=True,
        wind_u=u,
        wind_v=v,
        wind_speed=float(np.hypot(u, v)),
        cos_anchor=c,
        cos_centroid=c,
        cos_anchor_tie_avg=c,
        n_tied_anchors=1,
        cos_ceiling=1.0,
        disp_dr=0,
        disp_dc=2,
        d_anchor=(east / n, north / n),
        d_centroid=(east / n, north / n),
        parent_growth_cells=5,
        n_prior_burned_cells=100,
    )


def test_sign_convention_downwind_is_positive() -> None:
    """+row is SOUTH on the C1 lattice (C1.4). Getting this wrong inverts D10."""
    # Wind blowing due EAST (u=+5, v=0); body displaced due EAST (dc=+1, dr=0).
    assert _cosine(5.0, 0.0, 0.0, 1.0) == 1.0
    # Wind blowing due NORTH (v=+5); body displaced NORTH means dr = -1.
    assert _cosine(0.0, 5.0, -1.0, 0.0) == 1.0
    # PLANTED DEFECT this catches: if north were read as +dr, this would be -1.
    assert _cosine(0.0, 5.0, 1.0, 0.0) == -1.0


def test_cluster_bootstrap_widens_when_data_are_clustered() -> None:
    """The whole point: 636 correlated draws are not 636 independent draws.

    Averaged over replications, because a design-effect read off ONE draw is a
    property of that draw - the error that made the first version of this very
    test fail against correct code.
    """
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(30):
        offs = rng.normal(0, 0.5, size=20)
        vals = np.concatenate([o + rng.normal(0, 0.05, size=30) for o in offs])
        res = cluster_bootstrap(
            vals, np.repeat(np.arange(20), 30), draws=2000, seed=int(rng.integers(1e6))
        )
        ratios.append(res["design_effect_se_ratio"])
    assert float(np.mean(ratios)) > 3.0, np.mean(ratios)
    # PLANTED DEFECT: with iid data the two SEs must AGREE (ratio near 1) -
    # a bootstrap that always inflates is not measuring clustering.
    iid_ratios = []
    for _ in range(30):
        vals = rng.normal(0, 1, size=600)
        res = cluster_bootstrap(
            vals, np.repeat(np.arange(20), 30), draws=2000, seed=int(rng.integers(1e6))
        )
        iid_ratios.append(res["design_effect_se_ratio"])
    assert 0.85 < float(np.mean(iid_ratios)) < 1.15, np.mean(iid_ratios)


def test_cluster_bootstrap_interval_has_nominal_COVERAGE() -> None:
    """A CI is tested by COVERAGE over replications, never by one draw.

    Under a true mean of 0 the 95% interval must miss ~5% of the time, and it
    must not miss ~0% either - an interval that always covers is not a 95%
    interval, it is a vacuous one.
    """
    rng = np.random.default_rng(11)
    misses = 0
    reps = 300
    for _ in range(reps):
        offs = rng.normal(0, 0.4, size=20)
        vals = np.concatenate([o + rng.normal(0, 0.3, size=20) for o in offs])
        r = cluster_bootstrap(
            vals,
            np.repeat(np.arange(20), 20),
            draws=1500,
            seed=int(rng.integers(1e6)),
        )
        misses += bool(r["excludes_zero"])
    rate = misses / reps
    assert 0.01 < rate < 0.14, rate  # nominal 0.05, bootstrap-of-20-clusters slack


def test_cluster_bootstrap_detects_a_real_shift() -> None:
    """PLANTED SIGNAL: a genuinely shifted mean must be flagged."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(30):
        vals = rng.normal(0.8, 0.3, size=400)
        r = cluster_bootstrap(
            vals,
            np.repeat(np.arange(20), 20),
            draws=2000,
            seed=int(rng.integers(1e6)),
        )
        hits += bool(r["excludes_zero"])
    assert hits == 30, hits


def test_cluster_bootstrap_refuses_one_cluster() -> None:
    """An interval computed on one cluster is not an interval."""
    r = cluster_bootstrap(np.array([0.1, 0.2, 0.3]), np.array(["a", "a", "a"]))
    assert r["ci95"] is None and r["n_clusters"] == 1


def test_positive_control_fires_on_a_planted_skew() -> None:
    rng = np.random.default_rng(2)
    pool = []
    for f in range(12):
        for h in range(40):
            ang = rng.uniform(-np.pi, np.pi)
            pool.append(
                _body(
                    f"fire{f}",
                    f % 5,
                    h,
                    6 * np.cos(ang),
                    6 * np.sin(ang),
                    np.cos(rng.uniform(-np.pi, np.pi)),
                    np.sin(rng.uniform(-np.pi, np.pi)),
                )
            )
    pos = positive_control(pool)
    assert pos["sign_check_exactly_downwind_scores_plus_one"] is True, pos
    assert pos["fired"] is True, pos
    strong = [s for s in pos["sweep"] if s["planted_kappa"] == 0.8][0]
    assert strong["detected_excludes_zero"] is True, strong
    assert strong["realised_mean_cosine"] > 0.2, strong


def test_negative_controls_are_flat_on_an_isotropic_pool() -> None:
    rng = np.random.default_rng(3)
    pool = []
    for f in range(12):
        for h in range(40):
            wa = rng.uniform(-np.pi, np.pi)
            da = rng.uniform(-np.pi, np.pi)
            pool.append(
                _body(f"fire{f}", f % 5, h, 6 * np.cos(wa), 6 * np.sin(wa), np.cos(da), np.sin(da))
            )
    neg = negative_controls(pool)
    assert neg["passed"] is True, neg


def test_negative_controls_catch_a_rigged_estimator() -> None:
    """PLANTED DEFECT: a pool whose displacement is ALWAYS due east while the
    wind is also always due east makes the crosswind control flat but the
    permutation control STAY high - which is exactly the corpus-marginal
    confound the control exists to expose."""
    pool = [_body(f"fire{f}", f % 4, h, 6.0, 0.0, 1.0, 0.0) for f in range(8) for h in range(30)]
    neg = negative_controls(pool)
    assert neg["wind_permuted_across_corpus"]["mean"] > 0.9, neg
    assert neg["wind_rotated_90deg"]["mean"] == 0.0, neg


def test_tie_averaged_direction_points_away_from_the_burned_neighbour() -> None:
    prev = np.zeros((5, 5), dtype=bool)
    prev[2, 1] = True  # burned cell WEST of the landing cell (2,2)
    dr, dc, ok = _tie_averaged_ring_direction(prev, np.array([2]), np.array([2]))
    assert ok[0] and dr[0] == 0.0 and dc[0] == 1.0, (dr, dc, ok)
    # PLANTED DEFECT: symmetric neighbours must AVERAGE to a null direction, not
    # silently pick the first one - that is the row-major tie-break this
    # estimand exists to remove.
    sym = np.zeros((5, 5), dtype=bool)
    sym[1, 2] = sym[3, 2] = True
    dr2, dc2, ok2 = _tie_averaged_ring_direction(sym, np.array([2]), np.array([2]))
    assert ok2[0] and dr2[0] == 0.0 and dc2[0] == 0.0, (dr2, dc2)
    # A cell with no burned neighbour at all must report ok=False, never 0.
    empty = np.zeros((5, 5), dtype=bool)
    _, _, ok3 = _tie_averaged_ring_direction(empty, np.array([2]), np.array([2]))
    assert not ok3[0]


def test_quantisation_ceilings_are_what_the_lattice_allows() -> None:
    """A gap-2.0 body has 4 directions; a 1-cell step has 8. Different caps."""
    assert _ceiling(5.0, 0.0, 2.0) == 1.0  # wind due east
    assert abs(_ceiling(5.0, 5.0, 2.0) - np.cos(np.pi / 4)) < 1e-9  # 45 deg: 0.707
    # sqrt(5) offsets include (1,2)-type directions, so a 45 deg wind does far
    # better than the 4-cardinal cap.
    assert _ceiling(5.0, 5.0, float(np.hypot(1, 2))) > 0.94
    ring = _ring_ceiling(np.array([5.0, 5.0]), np.array([0.0, 5.0]))
    assert abs(ring[0] - 1.0) < 1e-9 and abs(ring[1] - 1.0) < 1e-9  # 8 dirs incl 45
    # PLANTED DEFECT: zero wind has no direction and must be nan, not 1.0.
    assert np.isnan(_ring_ceiling(np.array([0.0]), np.array([0.0]))[0])


def test_lattice_band_claim_is_checked_not_assumed() -> None:
    got = _lattice_realisable_gaps(2.236, 3.606)
    assert 2.828 in got and 3.0 in got and 3.162 in got, got
    # PLANTED DEFECT: an empty band that really IS arithmetically empty.
    assert _lattice_realisable_gaps(2.0, 2.2) == []


def run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} checks passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_all())
