"""Which fold is held out, and the guard that stops a block moving under a corpus.

WHAT STANDS BEHIND THIS. The split of record holds out one fold of five, and the
choice is not "fold 3" written down somewhere: it is the output of
``swap.select_heldout_fold``, which picks the fold covering the most distinct
SPATIAL BLOCKS. Every held-out number in this project is computed on whatever
that function returns, and ``data/swap.py`` was at zero line coverage.

WHY THE RULE IS BLOCKS AND NOT FIRES, restated because the test below is built
on exactly this. More fires from the same block are the same evidence with false
confidence: they share terrain, fuels and often weather, so a fold with five
fires in one block is one observation wearing five hats, and a fold with two
fires in two blocks is two. The precedent was set when the only fold that could
reach four blocks was not the fold with the most fires, so picking by fire count
would have produced a weaker split that looked stronger.

THE SECOND HALF is ``interim_build.check_existing_blocks_unmoved``. Spatial block
ids are computed over the whole fire universe, and inserting a fire whose id
sorts in the middle renumbers everything after it. A silent renumber is invisible
to every per-tensor check because each manifest stays individually conformant;
the first thing that notices is a fingerprint moving, and by then there is no way
on disk to tell a legitimate corpus growth from a bug. That guard is the only
thing standing between the two, and it was untested.
"""

from __future__ import annotations

import pytest

from wildfire_nowcast.data.interim_build import check_existing_blocks_unmoved
from wildfire_nowcast.data.swap import (
    EXCLUDED_FIRES,
    EXTENSION_LABEL_SOURCE,
    PUBLISHED_LABEL_SOURCE,
    CorpusAssignment,
    select_heldout_fold,
)


def _assignment(
    folds: dict[str, int],
    blocks: dict[str, int],
    *,
    hours: int = 100,
    label_source: dict[str, str] | None = None,
) -> CorpusAssignment:
    return CorpusAssignment(
        folds=dict(folds),
        blocks=dict(blocks),
        n_hours={fire: hours for fire in folds},
        label_source=label_source or {fire: PUBLISHED_LABEL_SOURCE for fire in folds},
        universe_n_fires=len(folds),
        universe_n_blocks=len(set(blocks.values())),
    )


# --------------------------------------------------------------------------
# the held-out fold
# --------------------------------------------------------------------------


def test_the_held_out_fold_is_chosen_by_BLOCK_count_not_by_FIRE_count() -> None:
    """The one input where the two rules give different answers.

    Fold 0 holds three fires and one block. Fold 1 holds two fires and two
    blocks. Counting fires picks 0 and yields one independent observation;
    counting blocks picks 1 and yields two. Every held-out claim's denominator
    depends on which of those two the function does.

    FAILS WHEN: the key becomes the number of member fires, or the negation is
    dropped so the fold with the FEWEST blocks is held out. Both produce a
    perfectly valid-looking split with less evidence behind it than the numbers
    computed on it would suggest.
    """
    assignment = _assignment(
        folds={"a": 0, "b": 0, "c": 0, "d": 1, "e": 1, "f": 2},
        blocks={"a": 5, "b": 5, "c": 5, "d": 1, "e": 2, "f": 3},
    )
    assert assignment.blocks_of_fold(0) == [5], "three fires, one block"
    assert assignment.blocks_of_fold(1) == [1, 2], "two fires, two blocks"

    assert select_heldout_fold(assignment) == 1


def test_a_tie_on_block_count_breaks_to_the_lowest_fold_id() -> None:
    """Determinism, so the split is not a judgement call made on the day.

    FAILS WHEN: the tiebreak is dropped and ``min`` returns whichever fold the
    set iteration happened to reach first, which makes the held-out fold a
    function of dict ordering rather than of the corpus.
    """
    tied = _assignment(folds={"x": 0, "y": 1, "z": 2}, blocks={"x": 7, "y": 9, "z": 4})
    assert [len(tied.blocks_of_fold(f)) for f in (0, 1, 2)] == [1, 1, 1]
    assert select_heldout_fold(tied) == 0


def test_two_fires_in_one_block_count_as_one_block_everywhere_they_are_counted() -> None:
    """Block counting is a set operation, in the selector and in the summary alike.

    FAILS WHEN: ``blocks_of_fold`` returns a list rather than a de-duplicated
    sorted set, which inflates ``n_blocks`` in the summary and can flip the
    held-out choice at the same time, in the same direction, so the two agree
    with each other and both are wrong.
    """
    assignment = _assignment(
        folds={"a": 0, "b": 0, "c": 1},
        blocks={"a": 3, "b": 3, "c": 8},
    )
    assert assignment.blocks_of_fold(0) == [3]
    summary = assignment.summary()
    assert summary["folds"]["0"]["n_blocks"] == 1
    assert summary["folds"]["0"]["fires"] == ["a", "b"]
    assert summary["n_blocks"] == 2
    assert summary["n_fires"] == 3


def test_the_summary_counts_the_two_label_sources_separately() -> None:
    """A table spanning the published product and the reimplementation must say
    which fires are which; the counts are how a reader sees the mix.

    FAILS WHEN: the label source counts are taken over the declared set rather
    than over the corpus members, so a source with zero members still appears
    and a reader concludes the corpus is more balanced than it is.
    """
    mixed = _assignment(
        folds={"a": 0, "b": 1, "c": 1},
        blocks={"a": 1, "b": 2, "c": 3},
        label_source={
            "a": PUBLISHED_LABEL_SOURCE,
            "b": EXTENSION_LABEL_SOURCE,
            "c": EXTENSION_LABEL_SOURCE,
        },
    )
    counts = mixed.summary()["label_source_counts"]
    assert counts == {PUBLISHED_LABEL_SOURCE: 1, EXTENSION_LABEL_SOURCE: 2}


def test_the_summary_hour_totals_are_per_fold_sums() -> None:
    """FAILS WHEN: the per-fold hour total is taken over the whole corpus, which
    makes every fold look identically sized and hides the imbalance a leave-fold-
    out claim rests on."""
    assignment = CorpusAssignment(
        folds={"a": 0, "b": 1, "c": 1},
        blocks={"a": 1, "b": 2, "c": 3},
        n_hours={"a": 10, "b": 20, "c": 30},
        label_source={f: PUBLISHED_LABEL_SOURCE for f in ("a", "b", "c")},
        universe_n_fires=3,
        universe_n_blocks=3,
    )
    folds = assignment.summary()["folds"]
    assert folds["0"]["n_hours"] == 10
    assert folds["1"]["n_hours"] == 50


def test_the_deliberately_excluded_fire_is_recorded_with_its_reason() -> None:
    """An exclusion without a reason is indistinguishable from an omission.

    FAILS WHEN: the exclusion registry is emptied or its reasons are reduced to
    a bare list of ids, at which point a reader of the corpus cannot tell a
    considered exclusion from a fire nobody got to.
    """
    assert EXCLUDED_FIRES, "the registry must not be empty while a fire is excluded"
    for fire_id, reason in EXCLUDED_FIRES.items():
        assert fire_id.strip()
        assert len(reason) > 40, f"{fire_id} is excluded without a stated reason"


# --------------------------------------------------------------------------
# the guard that stops a block moving
# --------------------------------------------------------------------------


def test_a_fire_whose_block_id_would_change_is_named_by_the_guard() -> None:
    """The whole point: it must return the offenders, not a boolean.

    FAILS WHEN: the comparison is inverted, or the guard returns a truthiness
    verdict instead of the ids. A boolean cannot be escalated: the difference
    between "one fire moved" and "nine fires moved" is the difference between a
    fixable ordering problem and a corpus that has to be rebuilt.
    """
    existing = {
        "kincade": {"spatial_block_id": 3},
        "glass": {"spatial_block_id": 4},
        "zogg": {"spatial_block_id": 5},
    }
    proposed = {"kincade": 3, "glass": 9, "zogg": 5}

    assert check_existing_blocks_unmoved(existing, proposed) == ["glass"]
    assert check_existing_blocks_unmoved(existing, {"kincade": 3, "glass": 4, "zogg": 5}) == []


def test_a_fire_absent_from_the_proposal_is_not_reported_as_moved() -> None:
    """Absence and movement are different findings and must not be conflated.

    FAILS WHEN: the ``fid in blocks`` membership test is dropped and a missing
    key raises, or defaults to a sentinel that compares unequal, which would
    report every not-yet-assigned fire as a block movement and bury the real one.
    """
    existing = {"kincade": {"spatial_block_id": 3}, "creek": {"spatial_block_id": 4}}
    assert check_existing_blocks_unmoved(existing, {"kincade": 3}) == []
    assert check_existing_blocks_unmoved(existing, {}) == []
    assert check_existing_blocks_unmoved({}, {"kincade": 3}) == []


def test_the_guard_reports_every_offender_and_not_only_the_first() -> None:
    """FAILS WHEN: the scan returns on the first mismatch, which understates the
    blast radius of a renumbering by however many fires come after it."""
    existing = {name: {"spatial_block_id": i} for i, name in enumerate("abcdef")}
    shifted = {name: i + 1 for i, name in enumerate("abcdef")}
    assert sorted(check_existing_blocks_unmoved(existing, shifted)) == list("abcdef")


@pytest.mark.parametrize("declared", [0, 1, 13])
def test_block_zero_is_a_real_block_id_and_not_a_missing_one(declared: int) -> None:
    """FAILS WHEN: the guard tests truthiness rather than equality, which would
    make block 0 unwatchable, and block 0 is a real member of the split."""
    existing = {"f": {"spatial_block_id": declared}}
    assert check_existing_blocks_unmoved(existing, {"f": declared}) == []
    assert check_existing_blocks_unmoved(existing, {"f": 99}) == ["f"]
