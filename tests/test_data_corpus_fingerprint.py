"""The content fingerprint, and the coarsening geometry it was built to
distinguish.

WHAT STANDS BEHIND THIS. The split fingerprint hashes ``(fire_id, cv_fold,
spatial_block_id, n_hours)`` and the train folds. It is therefore INVARIANT
under a change of resolution: the 1 km corpus and the 2 km corpus hash to the
same value, which is a known and recorded limitation, and it means a number
computed on one can be filed beside a number computed on the other with nothing
on disk objecting.

``coarsen_2km.corpus_content_fingerprint`` is the function that CAN tell them
apart, because it hashes the grid geometry and the array bytes rather than the
membership table. It was at zero line coverage, along with the whole 261-statement
coarsening module, whose per-block aggregations decide what a coarser corpus
would even contain. The resolution study those functions were written for was
abandoned and the 2 km store is kept as provenance, so this file is not about
reviving it: it is about the fingerprint that makes provenance auditable, and
about the block decomposition that any future resolution change would go through.

THE ONE ASSERTION TO READ FIRST is that two corpora that are byte-identical in
every respect EXCEPT cell size produce different fingerprints. That is the exact
discrimination the split fingerprint lacks, demonstrated rather than argued, on
two corpora built in a temporary directory from the same arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildfire_nowcast.common.contract import STATIC_CHANNELS
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.assemble import C1_CHANNELS, ChannelBundle, write_fire_tensor
from wildfire_nowcast.data.coarsen_2km import (
    AGGREGATION,
    REFINE,
    TARGET_CELL_SIZE_M,
    Padding,
    block_mean,
    block_occupancy,
    corpus_content_fingerprint,
    modal_class,
    padding_for,
    resnap_bounds,
    target_grid,
)

N_HOURS = 2
N_CELLS = 4


def _write_fire(root: Path, fire_id: str, *, cell_size_m: float, fill: float = 0.0) -> None:
    """One tiny conformant fire.

    The origin and the cell COUNTS are held fixed across cell sizes on purpose.
    Everything the fingerprint hashes except ``cell_size_m`` is therefore
    identical between a 1 km and a 2 km build of this fire, including the array
    bytes, which is what makes the test below a test of the cell-size term
    rather than of the domain extent.
    """
    grid = Grid(
        x_min=0.0,
        y_max=4000.0,
        nx=N_CELLS,
        ny=N_CELLS,
        cell_size_m=cell_size_m,
    )
    bundle = ChannelBundle(
        fire_id=fire_id,
        grid=grid,
        times=pd.date_range("2020-09-05T00:00:00", periods=N_HOURS, freq="h"),
        cv_fold=0,
        spatial_block_id=0,
        fuel_vintage_lag_years=1,
        n_ignition_components=1,
    )
    for name in C1_CHANNELS:
        shape = (N_CELLS, N_CELLS) if name in STATIC_CHANNELS else (N_HOURS, N_CELLS, N_CELLS)
        bundle.add(name, np.full(shape, fill, dtype=np.float32))
    write_fire_tensor(
        bundle,
        norm_stats_path="norm_stats.json",
        tensor_path=root / fire_id / "tensor.zarr",
        manifest_path=root / fire_id / "manifest.json",
    )


# --------------------------------------------------------------------------
# the fingerprint
# --------------------------------------------------------------------------


def test_two_corpora_differing_ONLY_in_cell_size_get_different_fingerprints(
    tmp_path: Path,
) -> None:
    """The discrimination the split fingerprint does not have, ISOLATED.

    Both corpora hold the same fire ids, the same hour count, the same cell
    counts, the same origin and byte-identical arrays. The single differing
    input is ``cell_size_m``. The membership-based fingerprint cannot see that
    at all; this one must, and it must see it without help from any other term.

    An earlier version of this test varied the domain extent along with the cell
    size, so it stayed green with the cell-size term deleted from the hash: the
    corner coordinates alone were carrying the difference. It is recorded here
    because a test that passes for a reason other than the one it names is the
    defect this project keeps finding.

    FAILS WHEN: ``cell_size_m`` is dropped from the hashed grid string, at which
    point a resolution change becomes invisible to every content-provenance
    check and a number computed at one resolution can be filed beside a number
    computed at another.
    """
    one_km, two_km = tmp_path / "km1", tmp_path / "km2"
    for fire_id in ("fire_a", "fire_b"):
        _write_fire(one_km, fire_id, cell_size_m=1000.0)
        _write_fire(two_km, fire_id, cell_size_m=2000.0)

    fine = corpus_content_fingerprint(one_km, ["fire_a", "fire_b"])
    coarse = corpus_content_fingerprint(two_km, ["fire_a", "fire_b"])

    assert len(fine) == 16 and len(coarse) == 16
    assert fine != coarse


def test_the_fingerprint_is_membership_ordered_and_content_sensitive(tmp_path: Path) -> None:
    """Order must not matter; membership and bytes must.

    FAILS WHEN: the fire ids stop being sorted before hashing, which makes the
    fingerprint a function of whatever order a glob returned and turns every
    comparison between two runs into a coin flip.
    """
    root = tmp_path / "corpus"
    _write_fire(root, "fire_a", cell_size_m=1000.0, fill=0.0)
    _write_fire(root, "fire_b", cell_size_m=1000.0, fill=0.0)

    reference = corpus_content_fingerprint(root, ["fire_a", "fire_b"])
    assert corpus_content_fingerprint(root, ["fire_b", "fire_a"]) == reference
    assert corpus_content_fingerprint(root, ["fire_a"]) != reference, "membership matters"

    changed = tmp_path / "changed"
    _write_fire(changed, "fire_a", cell_size_m=1000.0, fill=0.0)
    _write_fire(changed, "fire_b", cell_size_m=1000.0, fill=1.0)
    assert corpus_content_fingerprint(changed, ["fire_a", "fire_b"]) != reference, (
        "one changed value in one channel must move the fingerprint"
    )


# --------------------------------------------------------------------------
# the lattice
# --------------------------------------------------------------------------


def test_resnapping_moves_edges_OUTWARD_only() -> None:
    """A coarser lattice must contain the finer domain, never clip it.

    FAILS WHEN: floor and ceil are swapped, or replaced by round-to-nearest,
    which quietly discards the outermost row of a fire's domain: the buffer the
    model is supposed to spread into.
    """
    assert resnap_bounds((100.0, -100.0, 2100.0, 1900.0)) == (0.0, -2000.0, 4000.0, 2000.0)
    already_aligned = (0.0, 0.0, 4000.0, 4000.0)
    assert resnap_bounds(already_aligned) == already_aligned, "an aligned box is a fixed point"
    assert TARGET_CELL_SIZE_M == 1000.0 * REFINE


def test_the_target_grid_contains_the_fine_grid_and_the_padding_tiles_it_exactly() -> None:
    """The two functions have to agree, so they are checked against each other.

    FAILS WHEN: ``padding_for`` rounds instead of checking, which produces a pad
    that is off by a cell and silently shifts every coarse cell half a cell east
    while the shape arithmetic still works out.
    """
    fine = Grid(x_min=1000.0, y_max=5000.0, nx=3, ny=3, cell_size_m=1000.0)
    coarse = target_grid(fine)

    assert coarse.cell_size_m == 2000.0
    assert tuple(coarse.bounds) == (0.0, 2000.0, 4000.0, 6000.0)
    assert (coarse.ny, coarse.nx) == (2, 2)

    pad = padding_for(fine, coarse)
    assert pad.as_dict() == {"top": 1, "bottom": 0, "left": 1, "right": 0}
    assert pad.total_cells == 2
    assert fine.ny + pad.top + pad.bottom == coarse.ny * REFINE
    assert fine.nx + pad.left + pad.right == coarse.nx * REFINE


def test_a_coarse_grid_that_does_not_contain_the_fine_one_is_refused() -> None:
    """FAILS WHEN: the negative-pad check is dropped, at which point a mismatched
    pair produces a negative pad width and numpy pads on the opposite side,
    reflecting the domain instead of extending it."""
    fine = Grid(x_min=0.0, y_max=8000.0, nx=8, ny=8, cell_size_m=1000.0)
    too_small = Grid(x_min=2000.0, y_max=6000.0, nx=2, ny=2, cell_size_m=2000.0)
    with pytest.raises(ValueError, match="does not contain the fine grid"):
        padding_for(fine, too_small)


def test_the_padding_dataclass_reports_its_own_total() -> None:
    """FAILS WHEN: ``total_cells`` sums the wrong pair of edges, which understates
    how much of a coarsened domain is fabricated pad rather than measured data."""
    pad = Padding(top=1, bottom=2, left=3, right=4)
    assert pad.total_cells == 10
    assert pad.as_dict() == {"top": 1, "bottom": 2, "left": 3, "right": 4}


# --------------------------------------------------------------------------
# the three aggregations
# --------------------------------------------------------------------------


def test_the_continuous_rule_is_an_exact_block_mean() -> None:
    """FAILS WHEN: the block reshape mixes rows, which produces a plausible field
    built from the wrong sub-cells, or the accumulation happens in float32 and
    drifts on a large domain."""
    field = np.arange(16, dtype=float).reshape(4, 4)
    assert block_mean(field).tolist() == [[2.5, 4.5], [10.5, 12.5]]

    stacked = np.stack([field, field + 100.0])
    out = block_mean(stacked)
    assert out.shape == (2, 2, 2)
    assert out[1].tolist() == [[102.5, 104.5], [110.5, 112.5]]


def test_the_binary_rule_sets_a_coarse_cell_at_EXACTLY_half_occupancy() -> None:
    """The boundary of the occupancy rule, asserted at the boundary.

    Two of four sub-cells is exactly 0.5. Whether that coarse cell is set decides
    the area of every coarsened perimeter, and a strictly-greater comparison is a
    same-length edit that no aggregate area check would notice.

    FAILS WHEN: the threshold comparison becomes ``>``, which drops every
    exactly-half-covered cell and shrinks the coarsened footprint by a boundary
    layer over the whole corpus.
    """
    half = np.zeros((4, 4))
    half[0, 0] = half[0, 1] = 1.0
    assert block_occupancy(half).tolist() == [[1.0, 0.0], [0.0, 0.0]]

    quarter = np.zeros((4, 4))
    quarter[0, 0] = 1.0
    assert block_occupancy(quarter).tolist() == [[0.0, 0.0], [0.0, 0.0]]

    full = np.ones((4, 4))
    assert block_occupancy(full).tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_the_class_rule_breaks_ties_by_global_frequency_not_by_ascending_id() -> None:
    """Ascending class id would systematically grow non-burnable ground.

    The low fuel-model ids are the non-burnable classes, so breaking every tie
    toward the lowest id converts a tied block into unburnable terrain, and the
    coarser the grid the more of the landscape stops being able to burn at all.

    FAILS WHEN: the default ordering drops the frequency term, or the tie counter
    stops counting, so the number of blocks whose class was decided by the
    tiebreak is no longer reported and the effect becomes unmeasurable.
    """
    field = np.array(
        [[1, 1, 2, 2], [2, 2, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        dtype=float,
    )
    classes, ties = modal_class(field)
    assert ties == 2, "the two top blocks are 2-2 ties"
    assert classes.tolist() == [[1.0, 1.0], [1.0, 1.0]], "class 1 is the global majority"

    forced, _ = modal_class(field, priority=[2.0, 1.0])
    assert forced.tolist() == [[2.0, 2.0], [1.0, 1.0]], "an explicit priority wins the tie"


def test_a_priority_order_that_omits_a_present_class_is_refused() -> None:
    """Silently dropping a class would delete terrain from the coarse field.

    FAILS WHEN: the completeness check on the priority list is removed, at which
    point an omitted class simply never wins a block and disappears from the
    coarse corpus with nothing recording that it was there.
    """
    field = np.array([[1, 1, 5, 5], [1, 1, 5, 5], [6, 6, 7, 7], [6, 6, 7, 7]], dtype=float)
    with pytest.raises(ValueError, match="omits classes present"):
        modal_class(field, priority=[1.0, 5.0])
    with pytest.raises(ValueError, match="expects a 2-D field"):
        modal_class(np.zeros((2, 4, 4)))


def test_every_C1_channel_has_a_declared_aggregation() -> None:
    """A channel with no rule would be silently dropped from a coarsened fire.

    FAILS WHEN: a channel is added to the contract and not to the table, which
    is invisible until a coarsened tensor is short one channel.
    """
    missing = [name for name in C1_CHANNELS if name not in AGGREGATION]
    assert missing == [], f"no coarsening rule declared for {missing}"
    assert set(AGGREGATION) == set(C1_CHANNELS), (
        "the table must not name channels that do not exist"
    )
