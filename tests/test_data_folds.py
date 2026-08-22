"""C2 fold and spatial-block assignment: the code that computes the split.

WHY THIS FILE EXISTS AND WHY IT IS FIRST. ``data/folds.py`` had **zero** line
coverage while every published number in this repository is bound to a split
fingerprint (21 fires, 14 spatial blocks, train folds [0,1,2,4], held out fold
3). ``assign_blocks`` and ``assign_folds`` are the two functions that produce
that split. They are pure, deterministic, and take dataclasses in and dicts out,
so there is no excuse of "it needs Earth Engine" here: they were simply never
tested.

A defect in either one is invisible to every per-tensor check. A manifest with a
wrong ``cv_fold`` is individually conformant; C1 does not look across fires. The
first thing that would notice is the fingerprint moving, and a fingerprint moves
for legitimate reasons too, so by then the question "did the split change
because the corpus grew or because the code broke" has no answer on disk.

THE FOUR PROPERTIES THAT MATTER, in the order the module's own docstring puts
them:

1. *Overlap is spatial leakage.* Two fires whose buffered domains overlap share
   terrain, fuels and barriers. They must land in ONE fold and ONE block.
   Transitively: A-B overlap and B-C overlap puts A and C together even when A
   and C are disjoint.
2. *Touching is not overlapping.* The predicate is half-open. Two domains that
   share exactly an edge are independent, and merging them would cost a block
   for nothing. Blocks are the denominator of every held-out claim here, so a
   predicate that is one comparison too generous silently shrinks the evidence.
3. *Block ids are computed over the WHOLE fire universe, never the built
   subset.* This is the property with a live consequence: inserting a fire whose
   id sorts in the middle RENUMBERS the fires after it. That is why
   ``interim_build.check_existing_blocks_unmoved`` exists at all, and it is
   asserted here rather than left as a warning in a docstring.
4. *Determinism.* No RNG, no dict-order dependence, no dependence on the order
   the caller happens to supply fires in.
"""

from __future__ import annotations

import pytest

from wildfire_nowcast.data.folds import (
    FireFoldInput,
    _overlaps,
    _spatial_groups,
    assign_blocks,
    assign_folds,
    fold_summary,
)


def _fire(fire_id: str, x0: float, y0: float, x1: float, y1: float, hours: int = 10):
    return FireFoldInput(fire_id=fire_id, bbox_5070=(x0, y0, x1, y1), n_hours=hours)


# --------------------------------------------------------------------------
# 1. the overlap predicate itself
# --------------------------------------------------------------------------


def test_the_overlap_predicate_is_half_open_so_touching_domains_stay_apart() -> None:
    """Sharing an edge is not sharing terrain.

    FAILS WHEN: ``_overlaps`` uses ``<`` instead of ``<=`` on the separating
    axis, which merges every pair of edge-adjacent domains into one block and
    shrinks the held-out block count for free.
    """
    left = (0.0, 0.0, 10.0, 10.0)
    right_touching = (10.0, 0.0, 20.0, 10.0)
    right_overlapping = (9.99, 0.0, 20.0, 10.0)

    assert not _overlaps(left, right_touching)
    assert not _overlaps(right_touching, left), "the predicate must be symmetric"
    assert _overlaps(left, right_overlapping)
    assert _overlaps(right_overlapping, left)

    above_touching = (0.0, 10.0, 10.0, 20.0)
    assert not _overlaps(left, above_touching), "the y axis must be half-open too"


def test_overlap_is_transitive_through_the_union_find() -> None:
    """A and C never touch, but B bridges them, so all three are one block.

    FAILS WHEN: grouping is pairwise rather than a union-find, i.e. the bridge
    fire is put with A and C is left on its own, which splits one landscape
    across two folds.
    """
    fires = [
        _fire("a", 0.0, 0.0, 10.0, 10.0),
        _fire("b", 8.0, 0.0, 18.0, 10.0),
        _fire("c", 16.0, 0.0, 26.0, 10.0),
    ]
    assert not _overlaps(fires[0].bbox_5070, fires[2].bbox_5070), "the premise: a and c are apart"

    groups = _spatial_groups(sorted(fires, key=lambda f: f.fire_id))
    assert groups == [[0, 1, 2]]

    blocks = assign_blocks(fires)
    assert len({blocks["a"], blocks["b"], blocks["c"]}) == 1

    folds = assign_folds(fires, k=3)
    assert len({folds["a"], folds["b"], folds["c"]}) == 1, (
        "spatially overlapping fires must share a fold or the held-out fire is "
        "scored on landscape the model trained on"
    )


# --------------------------------------------------------------------------
# 2. block ids
# --------------------------------------------------------------------------


def test_block_ids_are_ordered_by_the_alphabetically_first_member() -> None:
    """The id is a stable function of the fire names, not of input order.

    FAILS WHEN: the ordering key changes to group size, hour count, or the order
    the caller passed the fires in, all of which produce a different but equally
    self-consistent numbering that no manifest on disk agrees with.
    """
    fires = [
        _fire("z_fire", 0.0, 0.0, 1.0, 1.0),
        _fire("m_fire", 100.0, 100.0, 101.0, 101.0),
        _fire("a_fire", 200.0, 200.0, 201.0, 201.0),
    ]
    assert assign_blocks(fires) == {"a_fire": 0, "m_fire": 1, "z_fire": 2}
    assert assign_blocks(list(reversed(fires))) == assign_blocks(fires)


def test_inserting_a_fire_RENUMBERS_the_blocks_after_it() -> None:
    """The hazard that makes the universe rule mandatory, asserted not asserted about.

    ``assign_blocks``'s docstring says to always compute over the full fire
    universe and never over the subset built so far. This test is the evidence
    for that instruction: a fire whose id sorts between two existing ones pushes
    every later block id up by one. Recompute over the built subset as the
    corpus grows and manifests written on different days disagree about which
    block a fire is in, while each one stays individually conformant.

    FAILS WHEN: someone "fixes" the renumbering by hashing the fire id into a
    block id, which would make this test green and would silently change every
    block id already on disk.
    """
    base = [_fire("a", 0.0, 0.0, 1.0, 1.0), _fire("c", 100.0, 100.0, 101.0, 101.0)]
    assert assign_blocks(base) == {"a": 0, "c": 1}

    grown = [*base, _fire("b", 50.0, 50.0, 51.0, 51.0)]
    assert assign_blocks(grown) == {"a": 0, "b": 1, "c": 2}
    assert assign_blocks(grown)["c"] != assign_blocks(base)["c"]


def test_appending_a_fire_that_sorts_last_leaves_existing_ids_alone() -> None:
    """The benign direction of the same mechanism, so the test above is not read
    as "block ids are unstable" in general.

    FAILS WHEN: the id is derived from position in the input sequence rather
    than from the sorted component order.
    """
    base = [_fire("a", 0.0, 0.0, 1.0, 1.0), _fire("b", 100.0, 100.0, 101.0, 101.0)]
    grown = [*base, _fire("z", 500.0, 500.0, 501.0, 501.0)]
    before, after = assign_blocks(base), assign_blocks(grown)
    assert all(after[k] == v for k, v in before.items())
    assert after["z"] == 2


# --------------------------------------------------------------------------
# 3. fold packing
# --------------------------------------------------------------------------


def test_folds_pack_largest_group_first_into_the_least_loaded_fold() -> None:
    """Greedy balance by hour count, deterministic tiebreak on the low fold id.

    THE FIXTURE IS CHOSEN SO THAT ALPHABETICAL PACKING GIVES A DIFFERENT ANSWER,
    which is the only way this assertion can see the ordering key at all. The
    small fire sorts FIRST alphabetically and LAST by hours: packing by hours
    puts the 100 h fire alone in fold 0, packing alphabetically puts the 10 h
    fire there and lands the 100 h fire beside the 90 h one.

    An earlier version of this test used a fixture on which both orderings
    happened to agree. It passed with the hour term deleted, which made it a
    test of nothing; it is recorded here rather than quietly replaced.

    FAILS WHEN: the ordering drops the ``-sum(n_hours)`` key, or the tiebreak
    stops preferring the lowest fold id and the assignment becomes dependent on
    set iteration order.
    """
    fires = [
        _fire("a_small", 0.0, 0.0, 1.0, 1.0, hours=10),
        _fire("b_large", 10.0, 10.0, 11.0, 11.0, hours=100),
        _fire("c_medium", 20.0, 20.0, 21.0, 21.0, hours=90),
    ]
    by_hours = assign_folds(fires, k=2)
    assert by_hours == {"b_large": 0, "c_medium": 1, "a_small": 1}

    alphabetical_would_give = {"a_small": 0, "b_large": 1, "c_medium": 0}
    assert by_hours != alphabetical_would_give, (
        "the fixture must separate the two orderings or the assertion above is vacuous"
    )
    assert sum(f.n_hours for f in fires if by_hours[f.fire_id] == 0) == 100
    assert sum(f.n_hours for f in fires if by_hours[f.fire_id] == 1) == 100


def test_fold_assignment_does_not_depend_on_the_order_fires_are_supplied() -> None:
    """FAILS WHEN: the internal ``sorted(..., key=fire_id)`` is removed, which
    makes the split a function of whatever order the caller's glob returned."""
    fires = [
        _fire("kincade", 0.0, 0.0, 5.0, 5.0, hours=134),
        _fire("creek", 100.0, 0.0, 105.0, 5.0, hours=700),
        _fire("glass", 3.0, 3.0, 8.0, 8.0, hours=200),
        _fire("zogg", 400.0, 0.0, 405.0, 5.0, hours=90),
    ]
    reference = assign_folds(fires, k=3)
    for rotation in range(1, len(fires)):
        rotated = fires[rotation:] + fires[:rotation]
        assert assign_folds(rotated, k=3) == reference
    assert assign_folds(list(reversed(fires)), k=3) == reference


def test_an_empty_corpus_gives_an_empty_split_and_k_below_one_raises() -> None:
    """FAILS WHEN: ``k < 1`` is allowed through and ``min(range(0), ...)`` raises
    a bare ``ValueError`` from the standard library instead of the module's own
    message, or an empty corpus returns something other than an empty mapping."""
    assert assign_folds([], k=5) == {}
    assert assign_blocks([]) == {}
    with pytest.raises(ValueError, match="k must be >= 1"):
        assign_folds([_fire("a", 0.0, 0.0, 1.0, 1.0)], k=0)


# --------------------------------------------------------------------------
# 4. the balance report that goes into norm-stats provenance
# --------------------------------------------------------------------------


def test_fold_summary_reports_empty_folds_rather_than_hiding_them() -> None:
    """An empty fold is a real outcome of the spatial constraint and must be
    visible, because it changes the denominator of a leave-fold-out claim.

    FAILS WHEN: ``empty_folds`` is computed from the folds that appear in the
    assignment (so an empty fold is simply absent and reads as no anomaly), or
    ``imbalance_ratio`` divides by zero instead of returning ``None``.
    """
    fires = [
        _fire("a", 0.0, 0.0, 10.0, 10.0, hours=100),
        _fire("b", 5.0, 5.0, 15.0, 15.0, hours=50),
    ]
    folds = assign_folds(fires, k=4)
    report = fold_summary(fires, folds, k=4)

    assert report["n_fires"] == 2
    assert len(report["empty_folds"]) == 3, "one occupied fold, three empty"
    assert report["hours_min"] == 0
    assert report["hours_max"] == 150
    assert report["imbalance_ratio"] is None, "a zero-hour fold must not be divided by"


def test_fold_summary_hours_are_the_sum_of_the_member_fires() -> None:
    """FAILS WHEN: the per-fold hour total counts fires instead of hours, which
    makes the balance report look even while one fold holds five times the
    cell-hours of another."""
    fires = [
        _fire("a", 0.0, 0.0, 1.0, 1.0, hours=7),
        _fire("b", 10.0, 10.0, 11.0, 11.0, hours=11),
    ]
    folds = {"a": 0, "b": 1}
    report = fold_summary(fires, folds, k=2)
    assert report["folds"]["0"]["n_hours"] == 7
    assert report["folds"]["1"]["n_hours"] == 11
    assert report["folds"]["0"]["fires"] == ["a"]
    assert report["imbalance_ratio"] == pytest.approx(11 / 7)
