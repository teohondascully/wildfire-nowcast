"""C1.1 — the one implementation of the ``fireline_v2`` state rule (C0).

These tests are written against the rule as INTERFACES states it, on tiny hand
built masks where the answer can be checked by eye. They are the reason
data can delete its copy at A5 without re-deriving anything: whatever the
real ingestion path and the synthetic fixture disagree about, it will not be
what state 1 means.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, UNBURNED, fire_state_violations
from wildfire_nowcast.common.states import (
    FIRELINE_V2,
    apply_state_rule,
    burning_residence_hours,
    cumulative_or,
    dilate,
    fireline_v2,
    frames_without_burning,
)


def _masks(rows: list[list[list[int]]]) -> np.ndarray:
    return np.asarray(rows, dtype=bool)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def test_dilate_grows_by_one_ring_and_clips_at_the_edge() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert int(dilate(mask, 1).sum()) == 9
    assert int(dilate(mask, 1, connectivity=4).sum()) == 5
    assert int(dilate(mask, 2).sum()) == 25
    corner = np.zeros((5, 5), dtype=bool)
    corner[0, 0] = True
    assert int(dilate(corner, 1).sum()) == 4, "dilation must clip, not wrap"
    assert not dilate(np.zeros((4, 4), dtype=bool), 3).any()


def test_dilate_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="connectivity"):
        dilate(np.zeros((3, 3), dtype=bool), 1, connectivity=6)
    with pytest.raises(ValueError, match="2-D"):
        dilate(np.zeros((2, 3, 3), dtype=bool), 1)


def test_cumulative_or_makes_burned_area_monotone_by_construction() -> None:
    """GOFER perimeters occasionally shrink; `ever` absorbs that rather than
    letting it become a 2 -> 0 transition."""
    masks = _masks([[[1, 0]], [[0, 0]], [[0, 1]]])
    ever = cumulative_or(masks)
    assert ever.tolist() == [[[True, False]], [[True, False]], [[True, True]]]


# --------------------------------------------------------------------------
# the rule itself
# --------------------------------------------------------------------------


def test_a_cell_burns_the_hour_it_is_enclosed() -> None:
    """burning(t) contains new(t) unconditionally, so no cell can go 0 -> 2."""
    perims = _masks([[[0, 0]], [[1, 0]], [[1, 1]]])
    lines = np.zeros_like(perims)  # no fire line at all
    state = fireline_v2(perims, lines)
    assert state[0].tolist() == [[UNBURNED, UNBURNED]]
    assert state[1].tolist() == [[BURNING, UNBURNED]]
    assert state[2].tolist() == [[BURNED_OUT, BURNING]]


def test_burning_persists_while_the_fire_line_covers_the_cell() -> None:
    """The whole point of the rule: with zero perimeter growth the fire line
    still keeps state 1 populated, so a contagion kernel has a seed."""
    perims = _masks([[[1, 0, 0]]] * 4)
    lines = _masks([[[1, 0, 0]], [[1, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]])
    state = fireline_v2(perims, lines)
    assert [row[0][0] for row in state.tolist()] == [BURNING, BURNING, BURNED_OUT, BURNED_OUT]
    assert burning_residence_hours(state).tolist() == [2]


def test_a_wandering_fire_line_cannot_reignite_a_burned_out_cell() -> None:
    """`burning(t-1) or new(t)` is what makes the state absorbing. A naive
    `new or active` would put this cell back into state 1 at t=3."""
    perims = _masks([[[1, 0]]] * 4)
    lines = _masks([[[1, 0]], [[0, 0]], [[0, 0]], [[1, 0]]])
    state = fireline_v2(perims, lines)
    assert [row[0][0] for row in state.tolist()] == [BURNING, BURNED_OUT, BURNED_OUT, BURNED_OUT]
    assert fire_state_violations(state) == []


def test_line_dilation_reaches_one_cell_beyond_the_line() -> None:
    """cfireLine is a boundary that falls between cells, so C1.1 dilates it."""
    perims = _masks([[[1, 1, 1]]] * 2)
    lines = _masks([[[1, 0, 0]], [[1, 0, 0]]])
    assert fireline_v2(perims, lines, line_dilation=1)[1].tolist() == [
        [BURNING, BURNING, BURNED_OUT]
    ]
    assert fireline_v2(perims, lines, line_dilation=0)[1].tolist() == [
        [BURNING, BURNED_OUT, BURNED_OUT]
    ]


def test_empty_state_1_is_a_legal_outcome_not_an_error() -> None:
    """After a dormancy every cell is closed. C1.1 says so explicitly; no
    absorbing rule can avoid it, and the rule must not pretend otherwise."""
    perims = _masks([[[1, 0]]] * 3)
    lines = _masks([[[1, 0]], [[0, 0]], [[0, 0]]])
    state = fireline_v2(perims, lines)
    assert frames_without_burning(state).tolist() == [1, 2]
    assert fire_state_violations(state) == []


def test_output_always_satisfies_the_c1_1_guarantees() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        perims = rng.random((8, 6, 6)) < rng.uniform(0.05, 0.6)
        lines = rng.random((8, 6, 6)) < 0.2
        state = fireline_v2(perims, lines)
        assert fire_state_violations(state) == []
        assert set(np.unique(state).tolist()) <= {UNBURNED, BURNING, BURNED_OUT}


def test_shrinking_perimeters_do_not_resurrect_cells() -> None:
    perims = _masks([[[1, 1]], [[0, 0]], [[1, 1]]])
    lines = np.ones_like(perims)
    assert fire_state_violations(fireline_v2(perims, lines)) == []


# --------------------------------------------------------------------------
# the named entry point
# --------------------------------------------------------------------------


def test_apply_state_rule_rejects_the_retired_rule() -> None:
    masks = np.zeros((2, 3, 3), dtype=bool)
    with pytest.raises(ValueError, match="RETIRED by ADR-006"):
        apply_state_rule(masks, rule="provisional_p0", fire_line_masks=masks)
    with pytest.raises(ValueError, match="unknown state rule"):
        apply_state_rule(masks, rule="something_else", fire_line_masks=masks)


def test_apply_state_rule_refuses_to_guess_a_fire_line() -> None:
    """Without cfireLine there is no observational basis for state 1, so the
    rule must refuse rather than silently degrade to the retired behaviour."""
    with pytest.raises(ValueError, match="cfireLine"):
        apply_state_rule(np.zeros((2, 3, 3), dtype=bool), rule=FIRELINE_V2)


def test_shape_errors_are_reported_not_broadcast() -> None:
    with pytest.raises(ValueError, match=r"\(T, H, W\)"):
        fireline_v2(np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="!="):
        fireline_v2(np.zeros((2, 3, 3), dtype=bool), np.zeros((2, 4, 4), dtype=bool))
