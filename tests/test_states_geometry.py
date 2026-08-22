"""The exact geometry of ``common.states.dilate``, written against two mutation survivors.

``dilate`` is not a utility. It is the ``active`` term of the ratified
``fireline_v2`` state rule (C1.1), so its structuring element decides which cells
are labelled BURNING in every tensor this project has produced, and its slice
arithmetic decides what happens at a domain edge. The external audit mutated both
and the suite did not notice:

* ``common/states.py:104`` - the 4-connectivity offset table, ``(1, 0)`` made
  ``(1, 1)``. The plus becomes a bent four-cell stencil with a diagonal in it, and
  745 tests still passed. Killed here by asserting the stencil itself, twice: once
  as an exact output and once as the symmetry property no single-entry corruption
  can satisfy.
* ``common/states.py:112`` - the destination row slice, ``h + min(dy, 0)`` made
  ``h + min(dy, 1)``. **That one is an EQUIVALENT MUTANT and no test can kill it**;
  the last test in this file is the proof and the guard on the assumption it rests
  on. Writing a test that appeared to kill it would have been the more comfortable
  answer and a false one.

The edge cases carry the weight here. A fire touching the north edge of its domain
is the ordinary case, not the exotic one, and every slice in that loop is an
off-by-one waiting for a mask that reaches a border.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from wildfire_nowcast.common.states import dilate

#: A single lit cell in the middle of a 5x5 grid. Every expected array below is
#: written out in full rather than computed, because computing the expectation
#: from the same offsets the code uses is how a test comes to assert that a
#: function equals itself.
_CENTRE = np.zeros((5, 5), dtype=bool)
_CENTRE[2, 2] = True


def _grid(rows: str) -> np.ndarray:
    return np.array([[c == "X" for c in row] for row in rows.split()], dtype=bool)


def test_four_connectivity_is_a_plus_and_eight_connectivity_is_a_block() -> None:
    """The stencil, spelled out. This is what ``active`` means in C1.1."""
    assert np.array_equal(
        dilate(_CENTRE, connectivity=4),
        _grid(".....  ..X..  .XXX.  ..X..  ....."),
    ), "4-connectivity is no longer a plus"
    assert np.array_equal(
        dilate(_CENTRE, connectivity=8),
        _grid(".....  .XXX.  .XXX.  .XXX.  ....."),
    ), "8-connectivity is no longer a full 3x3 block"


def test_the_offset_tables_have_the_symmetry_a_single_corrupted_entry_cannot_have() -> None:
    """A property, not a picture: dilation by a set that is not its own negation is not dilation.

    Reading the stencil out of the function's OUTPUT rather than out of its source,
    so this cannot pass by agreeing with a copy of the table.
    """
    for connectivity, size in ((4, 5), (8, 9)):
        stencil = dilate(_CENTRE, connectivity=connectivity)
        offsets = {(int(y) - 2, int(x) - 2) for y, x in zip(*np.nonzero(stencil), strict=True)}
        assert len(offsets) == size, f"connectivity {connectivity} covers {offsets}"
        assert offsets == {(-y, -x) for y, x in offsets}, (
            f"connectivity {connectivity} is not symmetric under negation, so the fire spreads "
            f"further in one direction than the other for no physical reason: {sorted(offsets)}"
        )
        assert all(abs(y) <= 1 and abs(x) <= 1 for y, x in offsets), sorted(offsets)
    four = dilate(_CENTRE, connectivity=4)
    eight = dilate(_CENTRE, connectivity=8)
    assert np.array_equal(four & eight, four), "4-connectivity is not a subset of 8"


def test_dilation_at_a_corner_and_along_an_edge_is_exact() -> None:
    """The slice arithmetic, where every off-by-one in this function lives.

    A perimeter touching the domain edge is the ordinary case: the buffer margin
    is finite and C1 fires are cropped to it.
    """
    corner = np.zeros((4, 4), dtype=bool)
    corner[0, 0] = True
    assert np.array_equal(dilate(corner, connectivity=8), _grid("XX..  XX..  ....  ....")), (
        "the north-west corner wrapped, clipped or spilled"
    )
    assert np.array_equal(dilate(corner, connectivity=4), _grid("XX..  X...  ....  ....")), (
        "the 4-connected corner is wrong"
    )

    far = np.zeros((4, 4), dtype=bool)
    far[3, 3] = True
    assert np.array_equal(dilate(far, connectivity=8), _grid("....  ....  ..XX  ..XX")), (
        "the south-east corner is wrong"
    )

    edge = np.zeros((4, 4), dtype=bool)
    edge[0, 2] = True
    assert np.array_equal(dilate(edge, connectivity=8), _grid(".XXX  .XXX  ....  ....")), (
        "a cell on the north edge did not dilate correctly"
    )


def test_iterations_compose_and_a_non_positive_count_is_the_identity() -> None:
    """``iterations`` is a repeat count, and 0 must copy rather than dilate once."""
    once = dilate(_CENTRE, 1, connectivity=8)
    assert np.array_equal(dilate(_CENTRE, 2, connectivity=8), dilate(once, 1, connectivity=8))
    for count in (0, -1):
        out = dilate(_CENTRE, count, connectivity=8)
        assert np.array_equal(out, _CENTRE), f"iterations={count} changed the mask"
        assert out is not _CENTRE, "a non-positive count must return a copy, not the input"


def test_an_unsupported_connectivity_raises_rather_than_choosing_one() -> None:
    """The stencil is contractual, so an unknown one is refused, not defaulted."""
    with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
        dilate(_CENTRE, connectivity=6)
    with pytest.raises(ValueError, match="2-D"):
        dilate(np.zeros((2, 2, 2), dtype=bool))


def _dilate_with_the_survivor_applied(
    mask: np.ndarray, iterations: int, connectivity: int
) -> np.ndarray:
    """``common/states.py:112`` as the sweep mutates it: ``min(dy, 0)`` -> ``min(dy, 1)``."""
    out = np.asarray(mask, dtype=bool)
    if iterations <= 0:
        return out.copy()
    offsets = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 4
        else [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    )
    h, w = out.shape
    for _ in range(int(iterations)):
        acc = out.copy()
        for dy, dx in offsets:
            ys_dst = slice(max(dy, 0), h + min(dy, 1))
            ys_src = slice(max(-dy, 0), h + min(-dy, 0))
            xs_dst = slice(max(dx, 0), w + min(dx, 0))
            xs_src = slice(max(-dx, 0), w + min(-dx, 0))
            acc[ys_dst, xs_dst] |= out[ys_src, xs_src]
        out = acc
    return out


def test_the_dilation_slice_survivor_is_an_EQUIVALENT_MUTANT_and_this_is_the_proof() -> None:
    """``states.py:112`` cannot be killed, and the honest thing is to say so.

    ``dy`` only ever takes ``-1, 0, 1``, so ``h + min(dy, 1)`` differs from
    ``h + min(dy, 0)`` at ``dy == 1`` alone, where the end index ``h + 1`` clips to
    ``h`` on a length-``h`` axis. The two forms therefore agree on every input the
    offset table can produce. The plan's Task 5.5 lists this site as a survivor to
    kill; no test can, and a test that claimed to would be measuring nothing.

    This is not a decoration. It fails the moment the assumption behind the
    argument stops holding, which is what makes it worth having: an offset with
    ``|dy| > 1``, or an axis long enough for the clip to stop saving it, turns the
    survivor into a real defect and this test into a red one.
    """
    checked = 0
    for height in range(1, 4):
        for width in range(1, 4):
            for bits in itertools.product((False, True), repeat=height * width):
                mask = np.array(bits, dtype=bool).reshape(height, width)
                for connectivity in (4, 8):
                    for iterations in (1, 2, 3):
                        checked += 1
                        assert np.array_equal(
                            dilate(mask, iterations, connectivity=connectivity),
                            _dilate_with_the_survivor_applied(mask, iterations, connectivity),
                        ), (
                            "the mutant is no longer equivalent, which is GOOD NEWS: "
                            f"{mask.astype(int).tolist()} at connectivity {connectivity} now "
                            "separates them. Write the killing test and lower the mutation "
                            "budget in the same commit."
                        )
    assert checked > 3000, f"the exhaustive corpus collapsed to {checked} cases"

    rng = np.random.default_rng(0)
    for _ in range(50):
        mask = rng.random((9, 7)) < 0.3
        for connectivity in (4, 8):
            assert np.array_equal(
                dilate(mask, 2, connectivity=connectivity),
                _dilate_with_the_survivor_applied(mask, 2, connectivity),
            )
