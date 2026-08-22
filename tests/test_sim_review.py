"""Unit tests for the number-extraction half of :mod:`wildfire_nowcast.sim.review`.

WHAT THIS FILE PROTECTS
-----------------------
``review.py`` builds ``reports/review.html``, the single self-contained page an
external reader is pointed at. Its module docstring makes the promise the page
rests on: *"Numbers are never typed by hand into the prose. They are pulled from
records into collect() and substituted, so a stale number cannot survive a
re-render."* That promise moves the risk off transcription and onto the
``_collect_*`` functions, and every one of them was at **0 percent coverage**
(527 statements, none executed by any test) when this file was written.

Four published claims are computed here and NOWHERE else:

1. **The seed-SD / block-SD distinction.** ``_collect_g2`` computes
   ``sd_across_seed`` and ``sd_across_block`` side by side. STATE carries this as
   a standing correction (ADR-042): *"SEPARATION DENOMINATOR = SD ACROSS HELD-OUT
   BLOCKS, NEVER SEEDS. G2 is ~1.8 block-SD, not the ~16 seed-SD everywhere
   quoted."* The page prints both so a reader can see the 9x. If the two were
   ever computed off the same axis the correction would render as its own error,
   on the one page written for someone who cannot check it.
2. **G2's unanimity count.** G2's magnitude is retracted; what it still stands on
   is *"UNANIMITY across independent blocks (sign test, p~0.03)"*. That count is
   ``unanimity_hits`` / ``unanimity_cells``, accumulated over seed x block x
   horizon. A sign test is a function of its DENOMINATOR, so a cell dropped from
   the denominator moves the p-value in the flattering direction.
3. **"Block 5 has the lowest dispersion in N of N arms, without exception."**
   ``_collect_block5`` is the sole computer of both halves of that fraction, and
   it silently EXCLUDES arms whose per-block dispersion carries a ``None`` or is
   all non-positive. An exclusion that reached only one half would manufacture
   unanimity.
4. **The latent-ablation split at ratio 2.0.** ``_collect_dispersion`` separates
   the ablation pairs into strong and weak families, with the reason written in
   the source: *"pooling the two families would let 'collapses 1.1x' hide inside
   'collapses up to 7.8x'."*

WHAT IS NOT TESTED HERE, AND WHY
--------------------------------
``fig_g2``, ``fig_dispersion``, ``fig_overprediction_map``, ``fig_identity``,
``fig_block5``, ``_data_uri``, ``_shrink_gif``, ``build_html`` and ``main`` are
NOT covered. They need the real ``runs/`` records (92 MB, untracked, on one
disk), the C1 tensor store and the full ``reports/figures`` tree, and they
produce a 5.2 MB HTML blob whose correctness is a visual judgement. A test that
imported them and asserted a file was non-empty would run, pass, and could not
have failed. They are declared uncovered rather than decorated.

THE FIXTURES
------------
Synthetic records with numbers chosen so every statistic has a closed form
written out by hand in the test, never obtained by running the code under test.
The per-block values deliberately vary across seeds in one block and not in
another, so an estimator that collapsed the block axis onto the seed axis cannot
agree by accident.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from wildfire_nowcast.sim import review
from wildfire_nowcast.sim.review import (
    G2_SEEDS,
    GATE_IOU_KEY,
    PageData,
    _collect_block5,
    _collect_dispersion,
    _collect_elmfire,
    _collect_g2,
)

# --------------------------------------------------------------------------
# G2 fixture. Every number below makes a hand-computed answer exact.
# --------------------------------------------------------------------------

#: Headline value per seed. Mean 0.25 exactly; sd(ddof=1) = sqrt(0.05/3).
SEED_VALUES = (0.1, 0.2, 0.3, 0.4)
SEED_MEAN = 0.25
SEED_SD = math.sqrt(0.05 / 3.0)

#: Per-block candidate values BY SEED, the block's single opponent value, and
#: the per-seed win flags. Block 0 varies across seeds, blocks 1 and 2 do not.
#: The three block-level differences are 2.0, 1.0, 3.0 -> mean 2.0, sd(ddof=1)
#: exactly 1.0. Under ddof=0 the same three give sqrt(2/3), 1.22x away, so the
#: two conventions cannot be confused on this fixture.
BLOCKS: tuple[tuple[tuple[float, ...], float, tuple[bool, ...]], ...] = (
    ((1.0, 2.0, 3.0, 4.0), 0.5, (True, True, True, True)),
    ((2.0, 2.0, 2.0, 2.0), 1.0, (True, True, True, False)),
    ((0.0, 0.0, 0.0, 0.0), -3.0, (False, False, False, False)),
)
BLOCK_DIFFS = (2.0, 1.0, 3.0)
BLOCK_MEAN = 2.0
BLOCK_SD = 1.0
#: 7 wins of 12 seed x block cells per horizon, over three horizons.
WINS_PER_HORIZON = 7
CELLS_PER_HORIZON = 12
N_HORIZONS = 3

RULE_VALUE = 0.15
ENVELOPE_VALUE = 0.20


def _metric_block() -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for si, seed in enumerate(G2_SEEDS):
        candidates[seed] = {
            "value": SEED_VALUES[si],
            "per_block": [
                {
                    "fire_id": f"fire_{bi}",
                    "spatial_block_id": 3 + bi,
                    "candidate": cvals[si],
                    "opponent": opp,
                    "candidate_wins": wins[si],
                }
                for bi, (cvals, opp, wins) in enumerate(BLOCKS)
            ],
        }
    candidates["kernel_init"] = {"value": 0.05, "per_block": []}
    return {
        "candidates": candidates,
        "rule_opponent_value": RULE_VALUE,
        "envelope_value": ENVELOPE_VALUE,
        "envelope_from": "ellipse_cal3h",
        "persistence": 0.0,
    }


def _brier_block() -> dict[str, Any]:
    return {
        "candidates": {s: {"value": 0.4 + 0.01 * i} for i, s in enumerate(G2_SEEDS)},
        "persistence": 0.5,
        "rule_opponent_value": 0.45,
    }


def _g2_record() -> dict[str, Any]:
    by_horizon = {
        h: {
            "rule_opponent": "ellipse_cal3h",
            "metrics": {GATE_IOU_KEY: _metric_block(), "band Brier": _brier_block()},
        }
        for h in ("1", "2", "3")
    }
    return {
        "g2_per_horizon": {"by_horizon": by_horizon},
        "split_before": {"fingerprint": "4848f491e8d588fa", "n_fires": 12},
        "scope": {
            "heldout_fire_ids": ["fire_0", "fire_1", "fire_2"],
            "heldout_block_ids": [3, 4, 5],
            "n_heldout_blocks": 3,
        },
        "n_members": 24,
        "gate_criterion": {"key": "best_member_iou_shape_masked"},
    }


@pytest.fixture
def g2(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run ``_collect_g2`` against the synthetic record and return ``d.g2``."""
    monkeypatch.setattr(review, "_load", lambda rel: _g2_record())
    d = PageData()
    _collect_g2(d)
    return d.g2


def test_the_seed_SD_and_the_block_SD_are_computed_from_DIFFERENT_axes(
    g2: dict[str, Any],
) -> None:
    """The ADR-042 correction, rendered on the reviewer page.

    ``sd_across_seed`` is the spread of the four seed headline values;
    ``sd_across_block`` is the spread of the per-block candidate-minus-opponent
    differences. Different axes, different numbers, both printed. Each is
    asserted against its own closed form, and against the fact that they are not
    equal on a fixture built so they cannot be.

    WHAT WOULD MAKE THIS FAIL: computing the block SD over the seed array (or
    the reverse), or switching either to ``ddof=0``, which moves the block SD by
    1.22x here.
    """
    for h in ("1", "2", "3"):
        row = g2["horizons"][h]
        assert row["sd_across_seed"] == pytest.approx(SEED_SD, rel=1e-12)
        assert row["sd_across_block"] == pytest.approx(BLOCK_SD, rel=1e-12)
        assert row["kernel_mean"] == pytest.approx(SEED_MEAN, rel=1e-12)
        assert row["kernel_by_seed"] == [float(v) for v in SEED_VALUES]
        # The whole point of ADR-042: these are not interchangeable.
        assert row["sd_across_seed"] != pytest.approx(row["sd_across_block"], rel=1e-6)


def test_the_block_margin_uses_the_block_SD_and_the_seed_margins_use_the_seed_SD(
    g2: dict[str, Any],
) -> None:
    """Each separation is divided by its OWN denominator.

    STATE's correction is not "the SD was miscalculated", it is "the wrong SD was
    used as the denominator", which is why the published magnitude was ~9x
    inflated. The margins are therefore checked against closed forms built from
    the matching axis.

    WHAT WOULD MAKE THIS FAIL: dividing the block-level mean difference by
    ``sd_seed``, which on this fixture reads 15.49 instead of 2.00.
    """
    for h in ("1", "2", "3"):
        row = g2["horizons"][h]
        assert row["margin_block_sd"] == pytest.approx(BLOCK_MEAN / BLOCK_SD, rel=1e-12)
        assert row["margin_seed_sd_vs_rule"] == pytest.approx(
            (SEED_MEAN - RULE_VALUE) / SEED_SD, rel=1e-12
        )
        assert row["margin_seed_sd_vs_env"] == pytest.approx(
            (SEED_MEAN - ENVELOPE_VALUE) / SEED_SD, rel=1e-12
        )
        # The wrong denominator is 7.7x away here, not a rounding nudge.
        assert row["margin_block_sd"] != pytest.approx(BLOCK_MEAN / SEED_SD, rel=1e-3)


def test_margin_block_t_is_a_t_STATISTIC_and_carries_the_sqrt_of_the_block_count(
    g2: dict[str, Any],
) -> None:
    """``margin_block_t`` divides by the standard ERROR, not the standard deviation.

    With three blocks the two differ by sqrt(3) = 1.73x, and that difference is
    the whole distinction between "the blocks agree with each other" and "the
    mean over blocks is far from zero".

    WHAT WOULD MAKE THIS FAIL: dropping the ``sqrt(len(pb))``, which collapses
    ``margin_block_t`` onto ``margin_block_sd``.
    """
    n = len(BLOCK_DIFFS)
    for h in ("1", "2", "3"):
        row = g2["horizons"][h]
        assert row["margin_block_t"] == pytest.approx(
            BLOCK_MEAN / (BLOCK_SD / math.sqrt(n)), rel=1e-12
        )
        assert row["margin_block_t"] == pytest.approx(
            math.sqrt(n) * row["margin_block_sd"], rel=1e-12
        )


def test_unanimity_counts_every_seed_x_block_x_horizon_CELL_including_the_losses(
    g2: dict[str, Any],
) -> None:
    """The denominator of G2's only surviving leg.

    G2's magnitude is retracted; it stands on unanimity across blocks, scored by
    a sign test. A sign test is a function of hits AND cells, so the denominator
    is load-bearing in a way a mean's denominator is not: dropping the losing
    cells turns 21/36 into 21/21.

    The fixture keeps the two far apart: 7 wins of 12 cells per horizon over
    three horizons is 21 of 36.

    WHAT WOULD MAKE THIS FAIL: resetting the accumulator per horizon (12 cells),
    counting blocks rather than seed x block cells (9 cells), or skipping the
    block whose seeds all lost (24 cells).
    """
    assert g2["unanimity"] == {
        "hits": WINS_PER_HORIZON * N_HORIZONS,
        "cells": CELLS_PER_HORIZON * N_HORIZONS,
    }
    assert g2["unanimity"]["hits"] < g2["unanimity"]["cells"]


def test_the_per_block_rows_carry_every_seed_so_a_reader_can_recompute_the_SD(
    g2: dict[str, Any],
) -> None:
    """The page publishes the inputs to the block SD, not only the SD.

    ``per_block[i]["candidate_by_seed"]`` must hold all four seed values for that
    block, in seed order, beside the single opponent value they are differenced
    against. Without those a reader cannot audit the denominator ADR-042
    corrected, which is the only reason the row is on the page.

    WHAT WOULD MAKE THIS FAIL: emitting the seed MEAN in place of the per-seed
    list, which is the reduction the SD row already reports.
    """
    rows = g2["horizons"]["1"]["per_block"]
    assert len(rows) == len(BLOCKS)
    for bi, (cvals, opp, wins) in enumerate(BLOCKS):
        assert rows[bi]["candidate_by_seed"] == [float(v) for v in cvals]
        assert rows[bi]["opponent"] == pytest.approx(opp)
        assert rows[bi]["wins"] == [bool(w) for w in wins]
        assert rows[bi]["block"] == 3 + bi
        assert rows[bi]["fire_id"] == f"fire_{bi}"


# --------------------------------------------------------------------------
# Block 5 unanimity: "lowest in N of N arms, without exception"
# --------------------------------------------------------------------------

_DISPERSION_KEY = "ensemble dispersion (area spread-skill)"


def _m8_record(per_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "split_before": {"fingerprint": "4848f491e8d588fa"},
        "g3": {
            "models": {
                name: {"criteria": {_DISPERSION_KEY: {"per_block": pb}}}
                for name, pb in per_arm.items()
            }
        },
    }


def _anatomy() -> dict[str, Any]:
    return {
        "train_support_distance": {
            "blocks": {f"fire_{b}": {"mahalanobis": 1.0 + b} for b in (3, 4, 5, 6)},
            "channels": ["elevation", "canopy_cover"],
        },
        "frontier_rate": {f"fire_{b}": {"growth_per_frontier_cell": 0.1 * b} for b in (3, 4, 5, 6)},
        "block_truth": {},
    }


def _block5(monkeypatch: pytest.MonkeyPatch, per_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    monkeypatch.setattr(review, "_load", lambda rel: _m8_record(per_arm))
    monkeypatch.setattr(review, "_fig", lambda rel: _anatomy())
    d = PageData()
    _collect_block5(d)
    return d.block5


def test_the_lowest_block_is_found_by_block_ID_and_not_by_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Block 5 is the narrowest" is a statement about block 5, not about a slot.

    Arm A has its minimum at block 5; arm B has its minimum at block 6 while
    block 5 sits FIRST in its iteration order. Counting positions instead of ids
    would score both arms as hits and publish 2 of 2 where the truth is 1 of 2.

    WHAT WOULD MAKE THIS FAIL: taking the position of ``min(vals)``, or
    comparing the first key rather than the key AT the minimum.
    """
    out = _block5(
        monkeypatch,
        {
            "arm_min_at_5": {"3": 0.9, "4": 0.8, "5": 0.2, "6": 0.7},
            "arm_min_at_6": {"5": 0.9, "3": 0.8, "4": 0.7, "6": 0.1},
        },
    )
    assert out["n_arms_scored"] == 2
    assert out["n_arms_block5_lowest"] == 1


def test_an_unscorable_arm_is_dropped_from_BOTH_halves_of_the_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exclusion that reached only one half would manufacture unanimity.

    ``_collect_block5`` skips an arm whose per-block dispersion contains a
    ``None`` (a block that could not be scored), one whose values are all
    non-positive, and one with no per-block entry at all. The published claim is
    a fraction, so both halves have to move together: a skipped arm must appear
    in neither, and in ``per_arm`` neither.

    WHAT WOULD MAKE THIS FAIL: incrementing ``total`` before the ``None`` guard,
    which reads 1 of 4 instead of 1 of 1; or incrementing ``lowest`` for an
    unscorable arm, which reads as unanimity over arms that were never scored.
    """
    out = _block5(
        monkeypatch,
        {
            "arm_ok": {"3": 0.9, "4": 0.8, "5": 0.2, "6": 0.7},
            "arm_has_none": {"3": 0.9, "4": None, "5": 0.2, "6": 0.7},
            "arm_all_zero": {"3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0},
            "arm_empty": {},
        },
    )
    assert out["n_arms_scored"] == 1
    assert out["n_arms_block5_lowest"] == 1
    assert [r["arm"] for r in out["per_arm"]] == ["arm_ok"]


# --------------------------------------------------------------------------
# The latent-ablation split and the two dispersion bars
# --------------------------------------------------------------------------

#: One fixture for both dispersion tests. Ratios 8.0 / 2.0 / 1.1 straddle the
#: declared 2.0 boundary and land exactly ON it once; 0.80 is inside the old
#: asymmetric bar and outside the geometric one; ``ellipse_cal3h`` is inside the
#: geometric bar and is not a trained arm, so the trained filter has something
#: to remove.
_ABLATION_ARMS: dict[str, float] = {
    "m8_wide": 0.80,
    "m8_wide__ABL": 0.10,
    "m8_edge": 0.50,
    "m8_edge__ABL": 0.25,
    "m8_narrow": 1.10,
    "m8_narrow__ABL": 1.00,
    "ellipse_cal3h": 0.90,
}


def _dispersion(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec = {
        "split_before": {"fingerprint": "4848f491e8d588fa"},
        "g3": {
            "models": {
                name: {"criteria": {_DISPERSION_KEY: {"equal_block": v, "interval": [0.8, 1.2]}}}
                for name, v in _ABLATION_ARMS.items()
            }
        },
    }
    monkeypatch.setattr(review, "_load", lambda rel: rec)
    d = PageData()
    _collect_dispersion(d)
    return d.dispersion


def test_the_ablation_families_are_split_at_2x_and_reported_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source states the reason: pooling lets a 1.1x collapse hide inside 8x.

    Two pairs sit either side of the declared boundary and one sits exactly ON
    it. The boundary is ``>= 2.0``, so the arm at exactly 2.0 is STRONG; a strict
    ``>`` would move it into the weak family and change both families' published
    extrema in one edit.

    WHAT WOULD MAKE THIS FAIL: pooling the families -- the pooled range is
    [1.1, 8.0] and neither family's own range survives it -- or flipping the
    boundary to a strict ``>``.
    """
    out = _dispersion(monkeypatch)
    assert out["n_ablation_pairs"] == 3
    assert out["n_ablation_strong"] == 2
    assert out["n_ablation_weak"] == 1
    assert out["ablation_strong_min"] == pytest.approx(2.0)
    assert out["ablation_strong_max"] == pytest.approx(8.0)
    assert out["ablation_weak_min"] == pytest.approx(1.1, rel=1e-9)
    assert out["ablation_weak_max"] == pytest.approx(1.1, rel=1e-9)
    assert out["weak_arm_full_max"] == pytest.approx(1.10)
    # The pooled range would have hidden the weak family entirely.
    assert out["ablation_ratio_min"] == pytest.approx(1.1, rel=1e-9)
    assert out["ablation_ratio_max"] == pytest.approx(8.0)


def test_the_geometric_bar_is_one_over_1_2_and_is_counted_apart_from_the_old_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-039 replaced ``[0.8, 1.2]`` with ``[1/1.2, 1.2]``, and the page shows both.

    An arm at 0.80 is inside the old bar and OUTSIDE the geometric one, so the
    two counts must be able to disagree; a page that printed one count under both
    labels would report the correction as having had no effect. The trained
    filter is exercised too: ``ellipse_cal3h`` is inside the geometric bar and is
    not a trained arm, so the two geometric counts differ by exactly it.

    WHAT WOULD MAKE THIS FAIL: spelling the low endpoint as the rounded 0.8333
    is invisible here, but reusing the record's own ``interval`` for the
    geometric count reads 4 and 4 instead of 4 and 3, and dropping the trained
    filter reads 3 and 3.
    """
    out = _dispersion(monkeypatch)
    assert out["n_arms"] == len(_ABLATION_ARMS)
    assert out["old_interval"] == [0.8, 1.2]
    assert out["geometric_interval"] == [pytest.approx(1.0 / 1.2), 1.2]
    assert out["n_in_old_bar"] == 4  # 0.80, 1.10, 1.00, 0.90
    assert out["n_in_geo_bar"] == 3  # 0.80 drops out
    assert out["n_trained_in_geo_bar"] == 2  # the ellipse is not a trained arm


# --------------------------------------------------------------------------
# The ELMFIRE adapter's degeneracy verdict
# --------------------------------------------------------------------------


def _elmfire(monkeypatch: pytest.MonkeyPatch, windows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "note": "ADAPTER VALIDATION. NOT a G5 head-to-head.",
        "tensor": "data/fires/2019_kincade/tensor.zarr",
        "n_members": 6,
        "windows": windows,
    }
    monkeypatch.setattr(review, "_fig", lambda rel: payload)
    d = PageData()
    _collect_elmfire(d)
    return d.elmfire


def _window(t0: int, truth: int, *, native_deg: bool, lobo_deg: bool) -> dict[str, Any]:
    return {
        "t0": t0,
        "truth_new_cells": truth,
        "arms": {
            "native": {
                "median_new_cells": 19.5,
                "ratio_to_truth": 0.36,
                "distinct_members": 6,
                "degenerate": native_deg,
            },
            "lobotomised": {
                "median_new_cells": 2.0,
                "ratio_to_truth": 0.04,
                "distinct_members": 2,
                "degenerate": lobo_deg,
            },
        },
    }


def test_the_adapter_and_its_control_are_judged_with_OPPOSITE_quantifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both quantifiers point the strict way for their own role, and they differ.

    The working adapter fails if it is degenerate on ANY window; the lobotomised
    positive control only counts as demonstrated if it is degenerate on EVERY
    window. Swapping them weakens both at once: ``all`` over native would forgive
    a single degenerate window, and ``any`` over lobotomised would let one
    degenerate window stand in for a control that mostly worked. Since ELMFIRE is
    the external check the whole comparison rests on, an adapter certified by the
    weaker quantifier is worse than no certification.

    The fixture makes the swap visible: native is degenerate on 1 of 3 windows
    and lobotomised on 2 of 3, so the correct verdicts are (True, False) and the
    swapped ones are (False, True).

    WHAT WOULD MAKE THIS FAIL: exchanging ``any`` and ``all`` in
    ``_collect_elmfire``.
    """
    out = _elmfire(
        monkeypatch,
        [
            _window(7, 54, native_deg=False, lobo_deg=True),
            _window(8, 40, native_deg=True, lobo_deg=True),
            _window(9, 30, native_deg=False, lobo_deg=False),
        ],
    )
    assert out["native_degenerate"] is True
    assert out["lobo_degenerate"] is False


def test_the_elmfire_extrema_are_taken_over_windows_and_the_worst_truth_is_a_MAX(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published ratios are a RANGE over windows, not one window's number.

    ``worst_truth`` is the largest truth growth any window asked the adapter to
    reproduce, so it is a max; the medians and ratios are reported as min/max
    pairs so one flattering window cannot stand for the set.

    WHAT WOULD MAKE THIS FAIL: reading window 0 instead of aggregating, or
    taking ``min`` for ``worst_truth``, which reads 30 instead of 54 here and
    understates the target the adapter missed.
    """
    out = _elmfire(
        monkeypatch,
        [
            _window(7, 54, native_deg=False, lobo_deg=True),
            _window(8, 40, native_deg=False, lobo_deg=True),
            _window(9, 30, native_deg=False, lobo_deg=True),
        ],
    )
    assert out["worst_truth"] == 54
    assert out["n_windows"] == 3
    assert out["native_ratio_min"] == pytest.approx(0.36)
    assert out["native_ratio_max"] == pytest.approx(0.36)
    assert out["lobo_distinct_min"] == 2
    assert out["artifact"] == "reports/figures/elmfire_degeneracy_verdict.json"
