"""Equal-block pooling: a dropped block is a HARD failure (A14, ADR-039 (6)).

The defect being closed is not an arithmetic error - the mean of the surviving
blocks is computed correctly. The defect is that the result is *reported as if it
were complete*. Every test here plants a gap and asserts it is refused, or asserts
the number that says how many blocks actually contributed.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.pooling import (
    IncompleteBlockCoverageError,
    equal_block_mean,
    equal_block_mean_of,
)


def _per_fire(values: dict[str, tuple[int, object]]) -> dict[str, object]:
    """``{fire_id: (block, value)}`` -> the C6 per-fire payload shape."""
    return {
        fid: {"spatial_block_id": block, "models": {"m": {"growth_windows": {"k": value}}}}
        for fid, (block, value) in values.items()
    }


# --------------------------------------------------------------------------
# the planted defect
# --------------------------------------------------------------------------


def test_PLANTED_a_block_that_scores_None_is_a_HARD_failure() -> None:
    """PLANTED DEFECT: one block's only fire scores ``None``.

    The pre-A14 behaviour returned the mean of the other three blocks with
    ``n_blocks: 3`` and nothing saying the sample had shrunk below the 4 distinct
    held-out blocks C6.3 requires - so ``in_interval_equal_block: true`` could be
    reported on a criterion computed over three quarters of the evidence.
    """
    per_fire = _per_fire(
        {"a": (4, 0.9), "b": (5, 1.0), "c": (6, 1.1), "d": (7, None)},
    )
    with pytest.raises(IncompleteBlockCoverageError, match="contributed NOTHING"):
        equal_block_mean(per_fire, "m", "k", "growth_windows")


def test_the_error_names_the_block_and_both_counts() -> None:
    """A failure a reader cannot act on gets disabled. Name the gap."""
    per_fire = _per_fire({"a": (4, 0.9), "b": (5, None), "c": (6, None)})
    with pytest.raises(IncompleteBlockCoverageError) as exc:
        equal_block_mean(per_fire, "m", "k", "growth_windows")
    message = str(exc.value)
    assert "[5, 6]" in message
    assert "2 of 3 block(s)" in message
    assert "m.growth_windows.k" in message, "which criterion, not just which block"


def test_the_opt_in_is_EXPLICIT_and_reports_the_gap_instead_of_hiding_it() -> None:
    """``allow_missing_blocks=True`` must not restore the silent behaviour."""
    per_fire = _per_fire({"a": (4, 0.9), "b": (5, 1.1), "c": (6, None)})
    out = equal_block_mean(per_fire, "m", "k", "growth_windows", allow_missing_blocks=True)
    assert out["equal_block_mean"] == pytest.approx(1.0)
    assert out["n_blocks_contributing"] == 2
    assert out["n_blocks_expected"] == 3
    assert out["dropped_blocks"] == [6]
    assert out["complete"] is False, (
        "the permissive path still has to SAY the coverage was partial; a permissive path that "
        "returns an indistinguishable result is the original defect with an extra keyword"
    )


def test_PLANTED_a_dropped_FIRE_inside_a_surviving_block_is_still_declared() -> None:
    """PLANTED DEFECT: partial coverage that does not lose a whole block.

    Block 4 keeps a value, so nothing is raised - but half its evidence is gone
    and the return must say so. This is the sub-case that would otherwise pass
    unremarked once the block-level guard exists.
    """
    per_fire = _per_fire({"a": (4, 0.9), "a2": (4, None), "b": (5, 1.1)})
    out = equal_block_mean(per_fire, "m", "k", "growth_windows")
    assert out["n_blocks_contributing"] == 2
    assert out["dropped_blocks"] == []
    assert out["n_dropped_values"] == 1
    assert out["complete"] is False


def test_PLANTED_a_fire_with_an_unusable_block_id_is_not_silently_skipped() -> None:
    """PLANTED DEFECT: a manifest without a usable ``spatial_block_id``.

    Skipping it would be the same silent shrink one level down - the fire simply
    stops existing and every count still looks right.
    """
    per_fire = _per_fire({"a": (4, 0.9), "b": (5, 1.1)})
    per_fire["ghost"] = {"models": {"m": {"growth_windows": {"k": 2.0}}}}  # no block id
    out = equal_block_mean(per_fire, "m", "k", "growth_windows")
    assert None in out["per_block"], "the unlabelled fire must appear, not vanish"
    assert out["n_blocks_contributing"] == 3


# --------------------------------------------------------------------------
# the arithmetic it must not change
# --------------------------------------------------------------------------


def test_it_reproduces_the_pre_A14_number_on_complete_coverage() -> None:
    """The refactor must be behaviour-preserving where nothing was dropped.

    Computed against the original expression (``np.mean`` over the per-block
    means of ``np.mean`` over each block's fires) rather than against a
    hand-copied constant.
    """
    per_fire = _per_fire(
        {"a": (4, 0.9), "a2": (4, 0.7), "b": (5, 1.1), "c": (6, 0.3), "d": (7, 2.0)}
    )
    out = equal_block_mean(per_fire, "m", "k", "growth_windows")
    blocks = {4: [0.9, 0.7], 5: [1.1], 6: [0.3], 7: [2.0]}
    expected = float(np.mean([float(np.mean(v)) for v in blocks.values()]))
    assert out["equal_block_mean"] == pytest.approx(expected)
    assert out["per_block"] == {k: pytest.approx(float(np.mean(v))) for k, v in blocks.items()}
    assert out["n_blocks"] == out["n_blocks_contributing"] == 4
    assert out["n_fires"] == 5
    assert out["complete"] is True


def test_a_block_is_one_vote_regardless_of_how_many_hours_it_burned() -> None:
    """The whole point of equal-block: Creek was 47% of the window-pooled mass."""
    heavy = _per_fire({f"f{i}": (4, 0.0) for i in range(20)} | {"g": (5, 1.0)})
    assert equal_block_mean(heavy, "m", "k", "growth_windows")["equal_block_mean"] == 0.5


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.9", True])
def test_non_measurements_never_pool_as_values(bad: object) -> None:
    """``float("0.9")`` and ``float(True)`` both succeed - so both are refused."""
    with pytest.raises(IncompleteBlockCoverageError):
        equal_block_mean_of({4: 1.0, 5: bad})


def test_numpy_scalars_are_real_measurements_and_must_pool() -> None:
    """The mirror of the test above: refusing ``np.float32`` would reject real data."""
    out = equal_block_mean_of({4: np.float32(1.0), 5: np.float64(3.0)})
    assert out["equal_block_mean"] == pytest.approx(2.0)
    assert out["complete"] is True


def test_no_blocks_at_all_is_None_and_not_zero() -> None:
    out = equal_block_mean_of({})
    assert out["equal_block_mean"] is None
    assert out["n_blocks_contributing"] == 0
    assert out["complete"] is True, "vacuously complete: there was nothing to drop"
