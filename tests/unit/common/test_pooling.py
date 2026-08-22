"""``common/pooling.py`` - ADR-021 (4), the average over BLOCKS rather than over windows.

Equal-block pooling is a scientific convention here, not a formatting choice: a
window-pooled number gives Creek 785 of 1372 windows and turns a five-block result
into one fire's result. The loud part is ``IncompleteBlockCoverageError``, which
is what stops a pooled mean silently shrinking to the blocks that happened to
report.
"""

from __future__ import annotations

import pytest

from wildfire_nowcast.common.pooling import (
    IncompleteBlockCoverageError,
    equal_block_mean,
    equal_block_mean_of,
)


def test_each_block_gets_one_vote_regardless_of_how_many_fires_it_holds() -> None:
    """Three fires in one block and one in another is 2 blocks, not 4 windows."""
    out = equal_block_mean_of({1: [0.0, 0.0, 0.0], 2: [1.0]})
    assert out["equal_block_mean"] == pytest.approx(0.5), (
        f"got {out['equal_block_mean']}: a block with three fires outvoted a block with one"
    )
    assert out["n_blocks"] == 2

    # The control: window pooling would give 0.25 here, so the assertion above
    # separates the two conventions rather than agreeing with both.
    assert out["equal_block_mean"] != pytest.approx(0.25)


def test_a_block_that_contributes_nothing_is_loud_and_can_only_be_silenced_by_name() -> None:
    """A silent shrink is how a five-block claim becomes a four-block claim.

    ``allow_missing_blocks`` is the only way past it, and it must be spelled at
    the call site so a reviewer sees the permissive path. If it could be bound
    positionally, a stray fifth argument would switch off the coverage error
    without anyone writing its name.
    """
    incomplete = {1: [0.0, 1.0], 2: [None, float("nan")]}
    with pytest.raises(IncompleteBlockCoverageError):
        equal_block_mean_of(incomplete)

    out = equal_block_mean_of(incomplete, allow_missing_blocks=True)
    assert out["n_blocks"] == 1 and out["dropped_blocks"] == [2]

    with pytest.raises(TypeError):
        equal_block_mean_of(incomplete, True)  # type: ignore[misc]


def test_a_fire_with_an_unusable_block_label_is_kept_and_reported() -> None:
    """Dropping it would be the same silent-shrink defect one level down."""
    per_fire = {
        "a": {"spatial_block_id": 1, "models": {"m": {"growth": {"iou": 0.4}}}},
        "b": {"spatial_block_id": "not an int", "models": {"m": {"growth": {"iou": 0.8}}}},
    }
    out = equal_block_mean(per_fire, "m", "iou", "growth")
    assert out["n_blocks"] == 2, "the fire with a malformed block label was silently skipped"
    assert out["equal_block_mean"] == pytest.approx(0.6)

    with pytest.raises(TypeError):
        equal_block_mean(per_fire, "m", "iou", "growth", True)  # type: ignore[misc]
