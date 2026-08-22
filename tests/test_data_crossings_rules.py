"""The crossing-detection rules behind ``data/events/crossings.json``.

WHAT STANDS BEHIND THIS. ``crossings.json`` is the artifact a barrier-crossing
gate was adjudicated against, and the adjudication turned on a distinction that
lives entirely in these functions: of the raw detached bodies the detector finds,
which are rasterisation jitter, which are separate ignitions filed under one
fire id, and which are real crossings. ``data/crossings.py`` is 302 statements
at **zero** line coverage. The verdict recorded against that file is that the
detector finds long-range DETACHED IGNITION rather than barrier crossing, and
that a visually confirmed highway crossing yields zero detached bodies because
the barrier cell itself burns. Both of those statements are claims about the
behaviour of the four functions tested here, and neither had a test.

THE FOUR, and why each one is not a formality:

* ``supercover_line`` is the corridor. A Bresenham line from (0,0) to (3,3)
  visits only the diagonal, so a one-cell-wide river laid on the anti-diagonal
  is invisible to it and the corridor reads clean. Supercover visits the
  off-diagonal cells too. That difference is the entire reason the module does
  not use the cheaper algorithm, and it is asserted below on the exact
  configuration that separates them.
* ``nearest_pair`` returns ``inf`` on an empty set. A zero would read as
  "adjacent", which is the opposite verdict.
* ``detect_detached_bodies`` decides what counts as detached at all, on an
  8-connected neighbourhood.
* ``classify_body`` is ordered time, then genealogy, then distance. The one
  input where those three tiers disagree is a far body that later merges, and
  that case is the difference between the ratified rule and a distance
  threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.contract import FBFM40_NONBURNABLE
from wildfire_nowcast.data.crossings import (
    FBFM40_OPEN_WATER,
    MIN_EVENT_CELLS,
    MIN_GAP_KM,
    RIDGE_RELIEF_M,
    SEPARATE_IGNITION_KM,
    DetachedBody,
    barrier_evidence,
    classify_body,
    detect_detached_bodies,
    nearest_pair,
    supercover_line,
)

CELL_M = 1000.0
H = W = 14


def _burnable_fuel() -> np.ndarray:
    return np.full((H, W), 101, dtype=np.int32)


def _body(**kwargs) -> DetachedBody:
    defaults = {
        "hour": 3,
        "n_cells": 2,
        "gap_km": 6.0,
        "merges_later": False,
        "landing": (2, 9),
        "anchor": (2, 3),
        "cells": ((2, 9), (2, 10)),
        "n_prior_burned_cells": 5,
        "prior_centroid": (2.0, 2.0),
    }
    defaults.update(kwargs)
    return DetachedBody(**defaults)


# --------------------------------------------------------------------------
# the corridor
# --------------------------------------------------------------------------


def test_the_corridor_does_not_slip_between_two_diagonal_barrier_cells() -> None:
    """The property that rules out Bresenham, on the configuration that shows it.

    A Bresenham segment from (0,0) to (3,3) is exactly the four diagonal cells,
    so a barrier occupying (1,2) and (2,1) is stepped straight through and the
    corridor is reported clean across a river. Supercover includes both.

    FAILS WHEN: the sub-step count drops so the line is sampled once per cell
    rather than four times, which reintroduces the diagonal slip on oblique
    corridors while leaving the axis-aligned cases correct.
    """
    diagonal = supercover_line(0, 0, 3, 3)
    bresenham_only = {(0, 0), (1, 1), (2, 2), (3, 3)}

    assert bresenham_only <= set(diagonal)
    assert (1, 2) in diagonal and (2, 1) in diagonal, "the anti-diagonal neighbours are visited"
    assert diagonal[0] == (0, 0) and diagonal[-1] == (3, 3), "endpoints are included"


def test_an_axis_aligned_corridor_is_exactly_the_cells_between_the_endpoints() -> None:
    """FAILS WHEN: the sub-sampling picks up neighbouring rows through a rounding
    slop, which would attribute a barrier one row off the corridor to it."""
    assert supercover_line(0, 0, 0, 3) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert supercover_line(2, 2, 2, 2) == [(2, 2)], "a zero-length segment is one cell"


def test_nearest_pair_returns_infinity_rather_than_zero_on_an_empty_set() -> None:
    """No source is not the same as adjacent, and zero would read as adjacent.

    FAILS WHEN: the empty guard is dropped and ``min`` over an empty array
    raises, or is replaced by a 0.0 default, which would classify every body in
    a fire's first frame as rasterisation jitter.
    """
    empty = np.array([], dtype=int)
    distance, a_cell, b_cell = nearest_pair(empty, empty, np.array([1]), np.array([1]))
    assert distance == float("inf")
    assert a_cell == (-1, -1) and b_cell == (-1, -1)

    distance, a_cell, b_cell = nearest_pair(
        np.array([0]), np.array([0]), np.array([3]), np.array([4])
    )
    assert distance == pytest.approx(5.0)
    assert a_cell == (0, 0) and b_cell == (3, 4)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_a_detached_body_records_its_gap_in_km_and_its_anchor() -> None:
    """The gap is what every downstream classification is made on.

    FAILS WHEN: the cell distance is not multiplied by the cell size, which
    reports 6 km as 6 cells; identical on a 1 km grid and wrong by a factor of
    two on the 2 km store, where nothing else would notice.
    """
    state = np.zeros((3, H, W), dtype=np.uint8)
    state[0, 2, 2] = 1
    state[1, 2, 2] = 2
    state[1, 2, 3] = 1
    state[2, 2, 2] = 2
    state[2, 2, 3] = 2
    state[2, 2, 9] = 1

    bodies = detect_detached_bodies(state, cell_size_m=CELL_M)
    assert len(bodies) == 1
    body = bodies[0]
    assert body.hour == 2
    assert body.n_cells == 1
    assert body.gap_km == pytest.approx(6.0)
    assert body.landing == (2, 9)
    assert body.anchor == (2, 3)
    assert body.n_prior_burned_cells == 2

    at_2km = detect_detached_bodies(state, cell_size_m=2 * CELL_M)
    assert at_2km[0].gap_km == pytest.approx(12.0)


def test_growth_that_touches_the_front_diagonally_is_not_detached() -> None:
    """8-connected, so a diagonal step is ordinary spread and not an event.

    FAILS WHEN: the neighbourhood narrows to 4-connected, which would fill the
    inventory with ordinary diagonal growth and inflate the event count that a
    gate's block coverage is computed from.
    """
    state = np.zeros((2, H, W), dtype=np.uint8)
    state[0, 5, 5] = 1
    state[1, 5, 5] = 2
    state[1, 6, 6] = 1
    assert detect_detached_bodies(state, cell_size_m=CELL_M) == []


def test_two_separate_detached_blobs_in_one_hour_are_two_bodies() -> None:
    """FAILS WHEN: the connected-component labelling of the new cells is dropped
    and the whole detached mask is treated as one body, which merges two spot
    events into one and puts the landing cell between them."""
    state = np.zeros((2, H, W), dtype=np.uint8)
    state[0, 1, 1] = 1
    state[1, 1, 1] = 2
    state[1, 6, 6] = 1
    state[1, 11, 11] = 1

    bodies = detect_detached_bodies(state, cell_size_m=CELL_M)
    assert len(bodies) == 2
    assert sorted(b.landing for b in bodies) == [(6, 6), (11, 11)]


def test_a_state_field_that_never_burns_is_refused_rather_than_reported_empty() -> None:
    """An empty result and an unusable input must not look the same.

    FAILS WHEN: the guard is removed, at which point a fire whose labels failed
    to rasterise contributes zero events and is indistinguishable from a fire
    that genuinely had none.
    """
    with pytest.raises(ValueError, match="never burns"):
        detect_detached_bodies(np.zeros((3, H, W), dtype=np.uint8), cell_size_m=CELL_M)
    with pytest.raises(ValueError, match=r"\(T, H, W\)"):
        detect_detached_bodies(np.zeros((H, W), dtype=np.uint8), cell_size_m=CELL_M)


# --------------------------------------------------------------------------
# classification: time, then genealogy, then distance
# --------------------------------------------------------------------------


def test_a_far_body_that_MERGES_is_a_crossing_and_not_a_separate_ignition() -> None:
    """The single input where the three tiers of the rule disagree.

    Merging is the normal fate of a real spot fire: an ember lands ahead of the
    front and the front overruns it. A definition that demands permanent
    separation selects for the pathological case and deletes the signal. So
    genealogy outranks distance, and a body far past the separate-ignition
    distance is still a crossing if it later merges.

    FAILS WHEN: the ``not body.merges_later`` term is dropped from the
    separate-ignition branch, which turns every long-range spot that the fire
    later overran into a filing artifact and empties the inventory of exactly
    the events it exists to hold.
    """
    far = SEPARATE_IGNITION_KM + 10.0

    assert (
        classify_body(
            _body(gap_km=far, merges_later=True), min_gap_km=MIN_GAP_KM, min_cells=MIN_EVENT_CELLS
        )
        == "crossing"
    )
    assert (
        classify_body(
            _body(gap_km=far, merges_later=False), min_gap_km=MIN_GAP_KM, min_cells=MIN_EVENT_CELLS
        )
        == "separate_ignition"
    )


def test_a_body_inside_the_jitter_distance_is_jitter_however_it_ends() -> None:
    """Distance decides first below the bar, before genealogy is consulted.

    FAILS WHEN: the comparison flips to ``<=``, which reclassifies every body at
    exactly the threshold and moves the count without moving the rule.
    """
    for merges in (True, False):
        assert (
            classify_body(
                _body(gap_km=MIN_GAP_KM - 0.01, merges_later=merges),
                min_gap_km=MIN_GAP_KM,
                min_cells=MIN_EVENT_CELLS,
            )
            == "rasterisation_jitter"
        )
    assert (
        classify_body(
            _body(gap_km=MIN_GAP_KM, merges_later=True),
            min_gap_km=MIN_GAP_KM,
            min_cells=MIN_EVENT_CELLS,
        )
        == "crossing"
    ), "a body exactly at the bar is retained"


def test_the_size_screen_is_applied_after_the_distance_and_genealogy_tiers() -> None:
    """Order matters: a small body is ``too_small``, not ``separate_ignition``.

    FAILS WHEN: the size check is hoisted above the genealogy check, which
    relabels single-cell long-range spots and makes the two exclusion counts in
    the artifact swap places without changing the total.
    """
    small = _body(gap_km=6.0, n_cells=1, merges_later=True)
    assert classify_body(small, min_gap_km=MIN_GAP_KM, min_cells=1) == "crossing"
    assert classify_body(small, min_gap_km=MIN_GAP_KM, min_cells=2) == "too_small"
    assert MIN_EVENT_CELLS == 1, "the shipped screen keeps one-cell spots"


# --------------------------------------------------------------------------
# what was crossed
# --------------------------------------------------------------------------


def test_barrier_evidence_reports_none_mapped_when_the_corridor_is_clear() -> None:
    """The honest negative, which is what made the gate unadjudicable.

    A detached body with nothing between it and the front is long-range ignition,
    not a barrier crossing. This branch has to be reachable or every event would
    be attributed to something.

    FAILS WHEN: the final branch defaults to a barrier kind rather than to
    ``none_mapped``, which would make every detached body look like a crossing.
    """
    evidence = barrier_evidence(
        (2, 3),
        (2, 9),
        np.zeros((H, W), dtype=np.uint8),
        _burnable_fuel(),
        np.zeros((H, W), dtype=np.float32),
    )
    assert evidence.kind == "none_mapped"
    assert evidence.n_barrier_cells == 0
    assert evidence.corridor_cells == 5, "strictly between the endpoints"
    assert evidence.ridge_relief_m == 0.0


def test_the_barrier_mask_outranks_a_non_burnable_fuel_class_in_the_corridor() -> None:
    """Precedence is water or road, then non-burnable fuel, then ridge.

    FAILS WHEN: the branches are reordered, which silently reassigns every event
    whose corridor holds both a mapped barrier and a non-burnable class; the
    event count is unchanged and the attribution table is wrong.
    """
    barrier = np.zeros((H, W), dtype=np.uint8)
    barrier[2, 6] = 1
    fuel = _burnable_fuel()
    fuel[2, 6] = 93  # a non-burnable class that is not open water
    assert 93 in FBFM40_NONBURNABLE

    evidence = barrier_evidence((2, 3), (2, 9), barrier, fuel, np.zeros((H, W), dtype=np.float32))
    assert evidence.kind == "road_or_narrow_water"
    assert evidence.n_nonburnable_cells == 1, "the fuel evidence is still reported beside it"

    fuel[2, 6] = FBFM40_OPEN_WATER
    assert (
        barrier_evidence((2, 3), (2, 9), barrier, fuel, np.zeros((H, W), dtype=np.float32)).kind
        == "water"
    )


def test_a_non_burnable_class_alone_is_attributed_to_fuel_not_to_a_ridge() -> None:
    """FAILS WHEN: the fuel branch stops consulting the contract's non-burnable
    class set and hardcodes a subset, which would drop whichever class was
    omitted into ``none_mapped``."""
    fuel = _burnable_fuel()
    fuel[2, 6] = 91
    evidence = barrier_evidence(
        (2, 3),
        (2, 9),
        np.zeros((H, W), dtype=np.uint8),
        fuel,
        np.zeros((H, W), dtype=np.float32),
    )
    assert evidence.kind == "nonburnable_fuel"
    assert evidence.nonburnable_classes == [91]


def test_the_ridge_threshold_is_relief_ABOVE_both_endpoints_and_is_strict() -> None:
    """Relief is measured against the higher endpoint, not against sea level.

    FAILS WHEN: the endpoint subtraction is dropped, at which point every
    corridor in mountainous terrain is a ridge, or the comparison becomes
    inclusive and a corridor exactly at the bar flips class.
    """
    elevation = np.zeros((H, W), dtype=np.float32)
    elevation[2, 6] = RIDGE_RELIEF_M
    at_the_bar = barrier_evidence(
        (2, 3), (2, 9), np.zeros((H, W), dtype=np.uint8), _burnable_fuel(), elevation
    )
    assert at_the_bar.kind == "none_mapped", "strictly greater, so the bar itself is not a ridge"

    elevation[2, 6] = RIDGE_RELIEF_M + 1.0
    over_the_bar = barrier_evidence(
        (2, 3), (2, 9), np.zeros((H, W), dtype=np.uint8), _burnable_fuel(), elevation
    )
    assert over_the_bar.kind == "ridge"
    assert over_the_bar.ridge_relief_m == pytest.approx(RIDGE_RELIEF_M + 1.0)

    raised = np.full((H, W), 500.0, dtype=np.float32)
    raised[2, 6] = 500.0 + RIDGE_RELIEF_M + 1.0
    assert (
        barrier_evidence(
            (2, 3), (2, 9), np.zeros((H, W), dtype=np.uint8), _burnable_fuel(), raised
        ).kind
        == "ridge"
    ), "an offset landscape gives the same relief"
