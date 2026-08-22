"""``eval/stage.py``: the block-count boundary of the proportional-closure ceiling.

``proportional_closure_separation`` states the separation ANY arm reaches if it
closes a common fraction of every block's distance to truth. That number is a
CEILING, so the only thing it must never do is go undefined on a fold where the
separation it bounds is defined: an arm would then be reported with a separation
and no ceiling to read it against, which is worse than reporting neither.

The boundary is therefore a differential property rather than a constant. Both
sides of it are checked here against
:func:`~wildfire_nowcast.eval.stage.separation_of_blocks`, which is the function
the ceiling bounds, so neither can move without the other. Nothing here is a
threshold, a bar, or a pass/fail: ``stage_decay`` and everything derived from it
is NON-ADJUDICATING under C6.6, and a ceiling on detectability is a statement
about power, never about a verdict.

None of the four functions in ``ESTIMAND_FUNCTIONS`` is touched, imported or
re-derived by this file. Their source is hashed into
``D3_LICENSED_ESTIMAND_SHA256``, and a test that made a hashed span convenient to
edit would put the pin one refactor away from being re-pinned.
"""

from __future__ import annotations

import math

import pytest

from wildfire_nowcast.eval.stage import (
    OUTCOME_OK,
    proportional_closure_separation,
    separation_of_blocks,
)

#: Two blocks, distances chosen so every quantity below is exact in binary
#: floating point and can be done on paper: mean 2, sample SD sqrt(2), so the
#: ceiling is 2 / sqrt(2) = sqrt(2) and the distance CV is sqrt(2) / 2.
TWO_BLOCKS: dict[int, float] = {4: 1.0, 12: 3.0}

#: mean(d) / sd(d) for :data:`TWO_BLOCKS`, written from the definition of the
#: sample SD rather than obtained from the module under test.
TWO_BLOCK_CEILING = 2.0 / math.sqrt(2.0)


def test_the_proportional_closure_ceiling_is_DEFINED_on_exactly_two_blocks() -> None:
    """Two blocks are enough for a sample SD, so they are enough for a ceiling.

    WHAT WOULD MAKE THIS FAIL: a guard that refuses a block count at which the
    sample SD is defined, i.e. any minimum above two blocks.

    The expected numbers are arithmetic on the two declared distances, not a
    second call into the module: mean 2.0, sample SD sqrt(2) because the
    deviations are +/- 1 and the divisor is n - 1 = 1.
    """
    got = proportional_closure_separation(TWO_BLOCKS)

    assert got["outcome"] == OUTCOME_OK, got
    assert got["n_blocks"] == 2
    assert got["mean_distance"] == pytest.approx(2.0, abs=1e-15)
    assert got["sd_distance"] == pytest.approx(math.sqrt(2.0), abs=1e-15)
    assert got["separation_sd"] is not None
    assert got["separation_sd"] == pytest.approx(TWO_BLOCK_CEILING, abs=1e-15)
    assert got["distance_cv"] == pytest.approx(math.sqrt(2.0) / 2.0, abs=1e-15)
    assert got["invariant_to_the_fraction_closed"] is True


def test_the_ceiling_is_undefined_on_one_block_because_the_sample_sd_is() -> None:
    """The other side of the same boundary, so it cannot be moved down either.

    WHAT WOULD MAKE THIS FAIL: a guard that returns a separation from a single
    block, which would be a number with no dispersion under it and exactly the
    n = 1 calibration C-3 forbids.
    """
    got = proportional_closure_separation({4: 1.0})

    assert got["outcome"] == "UNDEFINED_fewer_than_two_blocks", got
    assert got["separation_sd"] is None
    assert got["n_blocks"] == 1
    assert "mean_distance" not in got, "an undefined ceiling must not ship a level to quote"


def test_the_ceiling_is_defined_on_exactly_the_folds_the_separation_is() -> None:
    """The ceiling and the quantity it bounds share one block-count boundary.

    WHAT WOULD MAKE THIS FAIL: a fold size at which one of the two returns a
    number and the other returns ``None``, in either direction, which would let a
    measured separation be quoted with no ceiling or a ceiling be quoted with no
    separation it could bound.

    This is the property, and the two constants agreeing today is only how it is
    satisfied. Checking it as a differential means the boundary cannot drift on
    one side alone, which a hard-coded 2 in this file would have allowed.
    """
    for n_blocks in (1, 2, 3):
        distances = {block: 1.0 + block for block in range(n_blocks)}
        # An arm that closes half of every block's gap: the separation is defined
        # exactly when a sample SD of the margins is.
        closed = {block: 0.5 * d for block, d in distances.items()}

        ceiling = proportional_closure_separation(distances)
        measured = separation_of_blocks(closed, distances, lower_is_better=True)

        assert (ceiling["separation_sd"] is None) == (measured.separation_sd is None), (
            f"at {n_blocks} block(s) the ceiling and the separation disagree about being "
            f"defined: ceiling={ceiling['outcome']} separation={measured.undefined_reason!r}"
        )


def test_the_two_block_ceiling_is_still_the_fraction_free_identity() -> None:
    """At the boundary the ceiling must still BE a ceiling, not merely a number.

    WHAT WOULD MAKE THIS FAIL: a two-block fold on which closing 1 percent and
    closing 100 percent of the gap separate differently, or on which either
    differs from ``mean(d) / sd(d)``, since the whole reason the ceiling can be
    quoted is that the fraction closed cancels.
    """
    ceiling = proportional_closure_separation(TWO_BLOCKS)
    assert ceiling["separation_sd"] is not None

    for fraction in (0.01, 0.5, 1.0):
        arm = {block: d * (1.0 - fraction) for block, d in TWO_BLOCKS.items()}
        measured = separation_of_blocks(arm, TWO_BLOCKS, lower_is_better=True)
        assert measured.separation_sd is not None, fraction
        assert measured.separation_sd == pytest.approx(TWO_BLOCK_CEILING, abs=1e-12), (
            fraction,
            measured.separation_sd,
        )
