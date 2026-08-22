"""``common/states.py`` - C1.1 ``fireline_v2``, and one mutation site proved unkillable.

Fire is absorbing: a cell goes 0 unburned -> 1 burning -> 2 burned out and never
comes back. The first frame is where the rule is seeded, and the seed is the
site the mutation sweep singles out. It is not debt, and the second test below is
the proof rather than a test written to make a survivor go away.
"""

from __future__ import annotations

import itertools

import numpy as np

from wildfire_nowcast.common.states import BURNED_OUT, BURNING, UNBURNED, fireline_v2


def test_the_first_frame_treats_every_perimeter_cell_as_newly_burned() -> None:
    """Nothing burned before t=0, so the whole first perimeter is NEW at t=0.

    This is the observable content of seeding ``prev_ever`` with an all-false
    field, and it is the condition the equivalence proof below depends on: seed
    it with anything else and this fails first.
    """
    perimeter = np.zeros((3, 7, 7), dtype=bool)
    perimeter[:, 1:6, 1:6] = True
    line = np.zeros((3, 7, 7), dtype=bool)
    line[0, 3, 3] = True

    state = fireline_v2(perimeter, line)
    inside = perimeter[0]
    assert np.all(state[0][inside] == BURNING), (
        "a cell inside the first perimeter was not scored as newly burning, so the rule was "
        "seeded as if something had already burned before t=0"
    )
    assert np.all(state[:, 0, 0] == UNBURNED), "a cell outside every perimeter ignited"

    # By t=1 the perimeter has not moved and the fire line has gone, so the same
    # cells are burned out. That is the control: t=0 is not simply "everything
    # inside a perimeter burns".
    assert np.all(state[1][inside] == BURNED_OUT)


def test_the_state_field_is_absorbing_in_time() -> None:
    """C1.1's guarantee: burned out is terminal and unburned is never revisited."""
    rng = np.random.default_rng(20260822)
    perimeter = np.zeros((6, 7, 7), dtype=bool)
    for t in range(6):
        perimeter[t, 3 - t // 2 : 4 + t // 2, 3 - t // 2 : 4 + t // 2] = True
    line = rng.random((6, 7, 7)) < 0.25

    state = fireline_v2(perimeter, line)
    ever_burned = state > 0
    assert np.all(ever_burned[1:] >= ever_burned[:-1]), "a burned cell became unburned"
    burned_out = state == BURNED_OUT
    assert np.all(burned_out[1:] >= burned_out[:-1]), "a burned-out cell re-ignited"


def test_the_prev_ever_seed_survivor_is_an_EQUIVALENT_MUTANT_and_this_is_the_proof() -> None:
    """``masks.shape[1:]`` -> ``masks.shape[2:]`` on the seed cannot change any output.

    The seed is read exactly once, at ``i == 0``, as ``cur_ever & ~prev_ever``,
    and is then rebound to ``cur_ever`` which is always full shape. So the only
    question is whether an all-false ``(W,)`` field and an all-false ``(H, W)``
    field are distinguishable under those two operators against an ``(H, W)``
    operand. They are not: numpy broadcasts the trailing axis, and false is the
    identity of the operation either way.

    Proved by arithmetic rather than by running the mutant, deliberately. ADR-084
    showed that an empirical check ("the output was byte-identical with the
    mutant on disk") is exactly what a stale ``.pyc`` also produces, so the only
    evidence immune to that class is an argument about what the operation IS.

    THE PROOF EXPIRES IF THE SEED STOPS BEING ALL-FALSE, and the test above is
    what fails in that case: it asserts the observable consequence at ``t = 0``.
    """
    for height, width in itertools.product(range(1, 8), repeat=2):
        rng = np.random.default_rng(height * 100 + width)
        current = rng.random((height, width)) < 0.5
        narrow = np.zeros((width,), dtype=bool)
        wide = np.zeros((height, width), dtype=bool)

        broadcast = current & ~narrow
        full = current & ~wide
        assert broadcast.shape == (height, width)
        assert np.array_equal(broadcast, full), (height, width)
        assert np.array_equal(broadcast, current)

    # The negative control, so the comparison above is capable of failing: a seed
    # that is NOT all-false separates the two shapes immediately.
    current = np.ones((3, 4), dtype=bool)
    assert not np.array_equal(
        current & ~np.ones((4,), dtype=bool), current & ~np.zeros((3, 4), dtype=bool)
    )
