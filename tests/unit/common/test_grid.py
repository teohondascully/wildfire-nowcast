"""``common/grid.py`` - the EPSG:5070 lattice every tensor is cut on.

The grid is shared by four leads, so an off-by-one in it is an off-by-one in
every fire at once. What is checked here is the arithmetic no downstream check
can see: two fires only line up cell for cell because the snap is to a GLOBAL
multiple of the cell size, and a padded domain is only a buffer if the padding
goes OUTWARD on all four sides.
"""

from __future__ import annotations

import numpy as np

from wildfire_nowcast.common.grid import CELL_SIZE_M, CRS_STRING, Grid


def test_padding_grows_the_domain_outward_on_all_four_sides() -> None:
    """``pad_cells`` is a buffer, and a buffer that shrinks is worse than none.

    Every one of the four terms carries its own sign, and three of the four
    survive a single flip without changing the cell COUNT: pad the west edge
    inward and outward by the same amount and ``nx`` is unchanged. So the origin
    is asserted as well as the extent, which is what makes each sign visible on
    its own.
    """
    bounds = (0.0, 0.0, 10_000.0, 6_000.0)
    base = Grid.from_bounds(bounds, cell_size_m=1000.0)
    assert (base.nx, base.ny) == (10, 6)
    assert (base.x_min, base.y_max) == (0.0, 6_000.0)

    for pad_cells in (1, 3):
        padded = Grid.from_bounds(bounds, cell_size_m=1000.0, pad_cells=pad_cells)
        pad = pad_cells * 1000.0
        assert padded.nx == base.nx + 2 * pad_cells, (
            f"pad_cells={pad_cells} gave nx={padded.nx}; a buffer adds a margin at BOTH ends"
        )
        assert padded.ny == base.ny + 2 * pad_cells
        assert padded.x_min == base.x_min - pad, "the west edge moved the wrong way"
        assert padded.y_max == base.y_max + pad, "the north edge moved the wrong way"

        # The buffer is only a buffer if the original bounds are strictly inside
        # it: this is the property C1.2's edge reserve rests on.
        assert padded.x_min < bounds[0] and padded.y_max > bounds[3]
        assert padded.x_min + padded.nx * padded.cell_size_m > bounds[2]
        assert padded.y_max - padded.ny * padded.cell_size_m < bounds[1]


def test_snapping_puts_two_fires_on_one_global_lattice() -> None:
    """Cell edges land on multiples of the cell size, whatever the bounds are.

    Without this, two fires 500 m out of phase produce grids that can never be
    compared cell for cell, and nothing downstream would say so.
    """
    ragged = Grid.from_bounds((1_234.0, -5_678.0, 9_876.0, 4_321.0), cell_size_m=1000.0)
    assert ragged.x_min % 1000.0 == 0.0 and ragged.y_max % 1000.0 == 0.0
    assert ragged.x_min <= 1_234.0 and ragged.y_max >= 4_321.0

    unsnapped = Grid.from_bounds(
        (1_234.0, -5_678.0, 9_876.0, 4_321.0), cell_size_m=1000.0, snap=False
    )
    assert unsnapped.x_min == 1_234.0, "snap=False is the control and it must differ"


def test_rowcol_and_xy_are_inverse_and_the_defaults_are_the_contract_defaults() -> None:
    """C1's grid is 1 km on EPSG:5070; the round trip is what indexing rests on."""
    grid = Grid.from_bounds((0.0, 0.0, 5_000.0, 4_000.0))
    assert grid.cell_size_m == CELL_SIZE_M and grid.crs == CRS_STRING

    for row in range(grid.ny):
        for col in range(grid.nx):
            x, y = grid.xy(row, col)
            assert grid.rowcol(x, y) == (row, col)
            assert grid.contains(row, col)
    assert not grid.contains(-1, 0) and not grid.contains(0, grid.nx)


def test_a_grid_recovered_from_a_dataset_is_the_grid_that_wrote_it() -> None:
    """``from_dataset`` reads cell CENTRES back into an outer-edge origin."""
    import xarray as xr

    grid = Grid.from_bounds((0.0, 0.0, 7_000.0, 3_000.0))
    coords = grid.coord_arrays()
    ds = xr.Dataset(
        {"fire_state": (("y", "x"), np.zeros((grid.ny, grid.nx), dtype=np.uint8))},
        coords={"y": coords["y"], "x": coords["x"]},
        attrs={"crs": grid.crs},
    )
    recovered = Grid.from_dataset(ds)
    assert (recovered.nx, recovered.ny) == (grid.nx, grid.ny)
    assert recovered.x_min == grid.x_min and recovered.y_max == grid.y_max
    assert recovered.cell_size_m == grid.cell_size_m and recovered.crs == grid.crs
