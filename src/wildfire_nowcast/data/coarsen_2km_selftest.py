"""Self-checks for the 1 km -> 2 km coarsener (ADR-054 R1-prep). Every check
ships a planted defect, and every positive control asserts a MAGNITUDE against
an analytic identity rather than merely returning non-zero.

Run: ``.venv/bin/python -m wildfire_nowcast.data.coarsen_2km_selftest``

Coarsening is one of the rare things that is exactly checkable, so the bar here
is exactness where exactness exists:

* the 2 x 2 decomposition is a PARTITION, so ``4 * sum(coverage_fraction)``
  equals the 1 km burned-cell count with residual exactly 0;
* ``refine = 1`` is the IDENTITY, so it must reproduce its input bit-for-bit;
* the nested-snap identity is algebra, so it must hold with residual exactly 0;
* the disc's area is ``pi r^2``, so the coarse area must sit inside the
  analytic boundary-band bound ``0.5 * P * dx`` - scored by
  ``sim.coarsen.score_coarsening``, the existing scorer, not a second one.

Where exactness does NOT exist (a threshold rule loses features below the coarse
cell), the loss is MEASURED and reported, never asserted away.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from wildfire_nowcast.common.contract import (
    BURNED_OUT,
    BURNING,
    UNBURNED,
    fire_state_violations,
)
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.states import fireline_v2
from wildfire_nowcast.data.coarsen_2km import (
    REFINE,
    Padding,
    _pad_constant,
    block_mean,
    block_occupancy,
    modal_class,
    padding_for,
    resnap_bounds,
    target_grid,
)
from wildfire_nowcast.sim.coarsen import (
    DEFECTIVE_COARSENERS,
    coverage_fraction,
    disc_mask,
    score_coarsening,
    sub_cell_texture,
)

_RESULTS: dict[str, Any] = {}


def test_the_block_decomposition_is_an_exact_partition() -> None:
    """POSITIVE CONTROL, MAGNITUDE: ``4 * sum(coverage) == count(fine)``, residual 0.

    This is the identity that makes a block-mean an aggregation rather than a
    resampling. It asserts the ACTUAL 1 km burned-cell count, not "non-zero",
    and it is the check that catches a misaligned reshape, a dropped row or a
    double-counted pad - none of which any area-tolerance check could see.
    """
    rng = np.random.default_rng(7)
    residuals = []
    for _ in range(25):
        fine = rng.random((2 * rng.integers(3, 20), 2 * rng.integers(3, 20))) < 0.37
        exact = float(np.count_nonzero(fine))
        via_blocks = float(coverage_fraction(fine, REFINE).astype(np.float64).sum()) * REFINE**2
        residuals.append(abs(via_blocks - exact))
    assert max(residuals) == 0.0, f"partition identity broke: max residual {max(residuals)}"
    _RESULTS["partition_identity_max_residual"] = max(residuals)


def test_the_partition_identity_CATCHES_a_dropped_row() -> None:
    """PLANTED DEFECT: crop one row instead of padding. The identity must fail."""
    rng = np.random.default_rng(11)
    fine = rng.random((18, 18)) < 0.4
    exact = float(np.count_nonzero(fine))
    cropped = fine[:-2, :]
    via = float(coverage_fraction(cropped, REFINE).astype(np.float64).sum()) * REFINE**2
    assert abs(via - exact) > 0.0, "the identity did not notice a dropped row"
    _RESULTS["dropped_row_detected_residual"] = abs(via - exact)


def test_refine_one_is_the_identity_bit_for_bit() -> None:
    """POSITIVE CONTROL: the whole pipeline at ``refine=1`` must change nothing."""
    rng = np.random.default_rng(3)
    cont = (rng.normal(size=(9, 11)) * 37.0).astype(np.float32)
    binary = (rng.random((9, 11)) < 0.3).astype(np.float32)
    cats = rng.choice([98.0, 101.0, 142.0, 165.0], size=(9, 11))
    assert np.array_equal(block_mean(cont, 1), cont)
    assert np.array_equal(block_occupancy(binary, 1), binary)
    assert np.array_equal(modal_class(cats, 1)[0], cats.astype(np.float32))
    _RESULTS["refine_1_is_identity"] = True


def test_the_nested_snap_identity_is_exact() -> None:
    """POSITIVE CONTROL, MAGNITUDE: re-snapping 1 km -> 2 km == snapping raw -> 2 km.

    This is what licenses recomputing the 2 km spatial blocks from the manifests
    instead of re-reading every perimeter, so it is asserted rather than argued.
    """
    rng = np.random.default_rng(19)
    worst = 0.0
    for _ in range(500):
        x0 = float(rng.integers(-3_000_000, 3_000_000)) + float(rng.random())
        y0 = float(rng.integers(1_000_000, 3_000_000)) + float(rng.random())
        w = 1000.0 * float(rng.integers(5, 90))
        h = 1000.0 * float(rng.integers(5, 90))
        raw = (x0, y0, x0 + w, y0 + h)
        direct = Grid.from_bounds(raw, cell_size_m=2000.0, snap=True).bounds
        via_1km = resnap_bounds(Grid.from_bounds(raw, cell_size_m=1000.0, snap=True).bounds, 2000.0)
        worst = max(worst, max(abs(a - b) for a, b in zip(direct, via_1km, strict=True)))
    assert worst == 0.0, f"nested-snap identity is not exact: worst {worst} m"
    _RESULTS["nested_snap_max_residual_m"] = worst


def _textured_disc(n: int, r_km: float) -> np.ndarray:
    """A disc on the 1 km lattice perforated at 1 km scale (sub-2 km structure).

    A SMOOTH disc does not discriminate: ``sim/coarsen``'s own module doc records
    that point-sampling and area-fraction agree to within a cell on smooth shapes,
    and the rules only separate where there is structure below the coarse cell.
    At a 1 km -> 2 km step "below the coarse cell" means 1 km features, so the
    texture is applied on the fine lattice itself. Duty cycle 8/10, so every
    interior 2 km cell is >= 50% covered and the CORRECT coarse answer is the
    whole disc, area ``pi r^2``, known by construction.
    """
    disc = disc_mask(n, n, 1, cx_km=n / 2.0, cy_km=n / 2.0, r_km=r_km)
    return np.asarray(disc & sub_cell_texture(*disc.shape), dtype=bool)


def test_the_disc_area_matches_pi_r_squared_inside_the_analytic_bound() -> None:
    """POSITIVE CONTROL, MAGNITUDE: an analytically known area, scored by sim/coarsen.

    ``relative_tol`` is passed explicitly as ``dx / r`` - the analytic relative
    boundary-band bound for a disc - instead of inheriting ``RELATIVE_TOL``,
    which was calibrated for a 33x step where the band is a far smaller fraction
    of the shape. Reusing a tolerance across a different resolution ratio is how
    a bound stops meaning anything.
    """
    r_km, n = 40.0, 100
    fine = _textured_disc(n, r_km)
    coarse = block_occupancy(fine.astype(np.float32), REFINE) > 0
    verdict = score_coarsening(
        scenario="textured_disc_r40km_1km_to_2km",
        coarsener="occupancy_0.5",
        fine=fine,
        coarse=coarse,
        true_area_km2=math.pi * r_km**2,
        perimeter_km=2.0 * math.pi * r_km,
        coarse_cell_km=float(REFINE),
        true_components=1,
        relative_tol=float(REFINE) / r_km,
    )
    assert verdict.passed, verdict.as_dict()
    _RESULTS["disc_verdict"] = verdict.as_dict()


def test_the_planted_defects_are_CAUGHT_and_the_one_that_is_NOT_is_named() -> None:
    """NEGATIVE CONTROL, with an honest exception rather than a forced pass.

    ``nearest`` and ``all_subcells`` are caught on the textured disc. ``any_subcell``
    is **NOT** separable there, and that is a real property of a 2x step rather
    than a weak harness: at ``refine = 2`` any-touch and area-occupancy differ
    ONLY on blocks covered exactly 25%, which on a large blob is a sliver of the
    boundary band. Recorded as a number, and the scenario where the two rules DO
    separate exactly is the next check.
    """
    r_km, n = 40.0, 100
    fine = _textured_disc(n, r_km)
    caught: dict[str, list[str]] = {}
    for name, fn in DEFECTIVE_COARSENERS.items():
        v = score_coarsening(
            scenario="textured_disc_r40km_1km_to_2km",
            coarsener=name,
            fine=fine,
            coarse=fn(fine, REFINE),
            true_area_km2=math.pi * r_km**2,
            perimeter_km=2.0 * math.pi * r_km,
            coarse_cell_km=float(REFINE),
            true_components=1,
            relative_tol=float(REFINE) / r_km,
        )
        why = []
        if not (v.area_ok and v.relative_ok):
            why.append("area")
        if not v.connectivity_ok:
            why.append("connectivity")
        caught[name] = why
    assert caught["nearest"], "nearest-neighbour sampling went undetected"
    assert caught["all_subcells"], "the erosion defect went undetected"
    assert not caught["any_subcell"], (
        "any_subcell became detectable on a large blob — the note above is now stale"
    )
    _RESULTS["planted_defects_caught_by"] = caught
    _RESULTS["any_subcell_not_separable_on_a_large_blob_at_refine_2"] = True


def test_any_touch_and_area_occupancy_separate_EXACTLY_on_sub_cell_speckle() -> None:
    """POSITIVE CONTROL, MAGNITUDE, EXACT: the two rules' analytic difference.

    ``n_spots`` isolated 1 km cells, one per 2 km block. Each covers exactly 25%
    of its block, so the declared rule scores **exactly 0 km2** and any-touch
    scores **exactly 4 * n_spots km2**. Both are analytic, both are asserted, and
    the difference is the whole content of the "a mask is not a mean" choice -
    which is why ``water_barrier_mask`` carries both numbers in its per-fire QA.
    """
    n_spots, n = 24, 40
    fine = np.zeros((n, n), dtype=bool)
    rng = np.random.default_rng(41)
    picks = rng.choice((n // REFINE) ** 2, size=n_spots, replace=False)
    for p_ in picks:
        by, bx = divmod(int(p_), n // REFINE)
        fine[REFINE * by, REFINE * bx] = True
    assert int(np.count_nonzero(fine)) == n_spots
    occ_km2 = float(np.count_nonzero(block_occupancy(fine.astype(np.float32), REFINE))) * REFINE**2
    any_cells = np.count_nonzero(DEFECTIVE_COARSENERS["any_subcell"](fine, REFINE))
    any_km2 = float(any_cells) * REFINE**2
    expected = float(REFINE**2 * n_spots)
    assert occ_km2 == 0.0, f"the declared rule kept {occ_km2} km2 of 25%-covered cells"
    assert any_km2 == expected, f"any-touch scored {any_km2}, expected {expected}"
    _RESULTS["speckle_occupancy_km2"] = occ_km2
    _RESULTS["speckle_any_touch_km2"] = any_km2
    _RESULTS["speckle_true_fine_km2"] = float(n_spots)


def test_the_modal_rule_never_invents_a_class_and_the_mean_rule_DOES() -> None:
    """PLANTED DEFECT: averaging a class id. The count of invented classes is the check."""
    rng = np.random.default_rng(23)
    classes = np.array([91.0, 98.0, 101.0, 142.0, 165.0, 188.0])
    field = rng.choice(classes, size=(24, 24))
    modal, ties = modal_class(field, REFINE)
    invented_modal = int(np.count_nonzero(~np.isin(modal, classes)))
    averaged = block_mean(field, REFINE)
    invented_mean = int(np.count_nonzero(~np.isin(averaged, classes)))
    assert invented_modal == 0, f"modal invented {invented_modal} fuel classes"
    assert invented_mean > 0, "the planted defect (mean of a class id) invented nothing"
    _RESULTS["modal_invented_classes"] = invented_modal
    _RESULTS["mean_of_categorical_invented_classes"] = invented_mean
    _RESULTS["modal_tied_blocks_on_uniform_random"] = int(ties)


def test_the_modal_tiebreak_is_deterministic_and_not_id_ascending() -> None:
    """A 2-2 tie between water (98) and shrub (142) must NOT default to water.

    Ascending-id alone would systematically grow NON-BURNABLE ground, because
    the non-burnable FBFM40 classes are the low ids. The declared rule sends the
    tie to the field's commonest class.
    """
    field = np.array([[98.0, 142.0], [142.0, 98.0]])
    field = np.pad(field, ((0, 2), (0, 2)), constant_values=142.0)
    out, ties = modal_class(field, REFINE)
    assert ties >= 1, "the 2-2 tie was not detected"
    assert out[0, 0] == 142.0, f"tie went to the low id ({out[0, 0]}), not to the commonest class"
    alt, _ = modal_class(field, REFINE, priority=[98.0, 142.0])
    assert alt[0, 0] == 98.0, "the alternative tiebreak is not reachable, so it cannot be compared"
    _RESULTS["tiebreak_goes_to_commonest_class"] = True


def test_the_state_rule_survives_coarsening() -> None:
    """C1.1's guarantees hold by construction after re-application at 2 km."""
    t, n = 8, 16
    ever = np.zeros((t, n, n), dtype=bool)
    for i in range(t):
        ever[i, : 2 + i, : 2 + i] = True
    ever2 = block_occupancy(ever.astype(np.float32), REFINE) > 0
    act2 = block_occupancy(ever.astype(np.float32), REFINE) > 0
    state2 = fireline_v2(ever2, act2, line_dilation=0, validate=True)
    assert set(np.unique(state2)).issubset({UNBURNED, BURNING, BURNED_OUT})
    assert not fire_state_violations(state2), fire_state_violations(state2)
    _RESULTS["coarsened_state_passes_C1_1"] = True


def test_averaging_fire_state_is_WRONG_and_C1_CANNOT_SEE_IT() -> None:
    """PLANTED DEFECT, and a contract blind spot reported rather than assumed.

    Mean-then-round of ``{0, 1, 2}`` is the plausible-looking wrong rule for the
    label channel. It satisfies **every** guarantee ``fire_state_violations``
    checks - values, monotonicity, no 0->2 skip, one contiguous burning run - so
    the C1 checker cannot distinguish it from the declared rule. What DOES see it
    is the burned-area comparison: the mean rule silently DROPS cells whose
    sub-cells are in state 1, because ``mean([1,1,0,0]) == 0.5`` rounds to 0
    while the declared occupancy rule keeps a half-covered cell.
    """
    rng = np.random.default_rng(5)
    t, n = 12, 16
    ever = np.zeros((t, n, n), dtype=bool)
    order = rng.permutation(n * n)
    flat = ever.reshape(t, -1)
    for i in range(t):
        flat[i, order[: int((i + 1) / t * n * n)]] = True
    state = np.zeros((t, n, n), dtype=np.uint8)
    prev = np.zeros((n, n), dtype=bool)
    age = np.zeros((n, n), dtype=np.int64)
    for i in range(t):
        age[ever[i]] += 1
        state[i][ever[i]] = BURNED_OUT
        state[i][ever[i] & (age <= 2)] = BURNING
        prev = ever[i]
    assert prev.any()
    assert not fire_state_violations(state)

    averaged = np.rint(block_mean(state, REFINE)).astype(np.uint8)
    declared = block_occupancy((state >= BURNING).astype(np.float32), REFINE) > 0
    blind = not fire_state_violations(averaged)
    dropped = int(np.count_nonzero(declared & (averaged == UNBURNED)))
    assert blind, "the planted defect was caught by C1 after all — update this note"
    assert dropped > 0, "the mean rule did not differ from the declared rule here"
    _RESULTS["c1_state_checks_are_blind_to_an_averaged_state_field"] = blind
    _RESULTS["burned_cells_an_averaged_state_field_would_drop"] = dropped


def test_coarsening_a_monotone_field_stays_monotone() -> None:
    """The occupancy rule is monotone in the coverage fraction, so ``ever`` cannot
    go backwards. Asserted on a shape that grows one sub-cell at a time - the
    hardest case for a threshold rule."""
    t, n = 40, 12
    ever = np.zeros((t, n, n), dtype=bool)
    flat = ever.reshape(t, -1)
    for i in range(t):
        flat[i, : i + 1] = True
    ever2 = block_occupancy(ever.astype(np.float32), REFINE)
    areas = ever2.sum(axis=(1, 2))
    assert bool(np.all(np.diff(areas) >= 0)), f"coarsened area decreased: {areas}"
    _RESULTS["monotone_preserved_on_one_subcell_growth"] = True


def test_padding_is_outward_and_tiles_exactly() -> None:
    """A cropped domain would shrink C1.2's buffer; assert we pad, never crop."""
    for ny, nx, x0, y1 in ((51, 39, -2296000.0, 2087000.0), (45, 45, -2001000.0, 2000000.0)):
        fine = Grid(x_min=x0, y_max=y1, nx=nx, ny=ny, cell_size_m=1000.0)
        coarse = target_grid(fine, REFINE)
        pad = padding_for(fine, coarse)
        assert min(pad.top, pad.bottom, pad.left, pad.right) >= 0
        assert fine.ny + pad.top + pad.bottom == coarse.ny * REFINE
        assert fine.nx + pad.left + pad.right == coarse.nx * REFINE
        assert coarse.x_min <= fine.x_min and coarse.y_max >= fine.y_max
    _RESULTS["padding_is_outward"] = True


def test_the_fire_state_pad_value_cannot_invent_fire() -> None:
    """Padding with UNBURNED must be a no-op on the burned count, exactly."""
    rng = np.random.default_rng(31)
    state = (rng.random((5, 9, 9)) < 0.2).astype(np.uint8)
    padded = _pad_constant(state, Padding(1, 0, 0, 1), UNBURNED)
    assert int(np.count_nonzero(padded)) == int(np.count_nonzero(state))
    _RESULTS["pad_adds_zero_burned_cells"] = True


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures: list[str] = []
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:  # pragma: no cover - reported, not raised
            failures.append(f"{fn.__name__}: {exc}")
    report = {
        "selftest": "coarsen_2km",
        "n_checks": len(tests),
        "n_failed": len(failures),
        "failures": failures,
        "results": _RESULTS,
        "verdict": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=1, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
