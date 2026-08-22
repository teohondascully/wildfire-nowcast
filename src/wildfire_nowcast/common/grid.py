"""The analysis grid: EPSG:5070, 1 km cells, north-up.

One :class:`Grid` object carries everything anyone needs to place a raster in
space, and knows how to emit the coordinate arrays that C1 requires. Build one
with :meth:`Grid.from_bounds` (snapping to the 1 km lattice) or recover one from
an existing store with :meth:`Grid.from_dataset`.

Conventions, fixed once here so nobody has to re-derive them:

* ``x`` coordinates are cell **centres**, strictly ascending (east-positive).
* ``y`` coordinates are cell **centres**, strictly descending (north-up), which
  is the GDAL/rasterio raster convention.
* ``bounds`` are outer **edges** ``(xmin, ymin, xmax, ymax)`` - the same
  convention as ``manifest["bbox_5070"]`` (C2) and ``rasterio``.
* array axis 0 is ``y`` (north to south), axis 1 is ``x`` (west to east).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from wildfire_nowcast.common.contract import CELL_SIZE_M, CRS_STRING

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

__all__ = ["Grid", "BBox"]

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Grid:
    """A north-up, square-cell raster grid in a projected CRS.

    Attributes
    ----------
    x_min, y_max
        Outer edge of the north-west corner, in CRS units (metres).
    nx, ny
        Number of columns and rows.
    """

    x_min: float
    y_max: float
    nx: int
    ny: int
    cell_size_m: float = CELL_SIZE_M
    crs: str = CRS_STRING

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError(f"grid must have >= 1 cell per axis, got nx={self.nx} ny={self.ny}")
        if self.cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be positive, got {self.cell_size_m}")

    # -- derived geometry ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)`` - the array shape of one time slice."""
        return (self.ny, self.nx)

    @property
    def x_coords(self) -> np.ndarray:
        """Cell-centre eastings, ascending, float64."""
        return self.x_min + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.cell_size_m

    @property
    def y_coords(self) -> np.ndarray:
        """Cell-centre northings, descending, float64."""
        return self.y_max - (np.arange(self.ny, dtype=np.float64) + 0.5) * self.cell_size_m

    @property
    def x_max(self) -> float:
        return self.x_min + self.nx * self.cell_size_m

    @property
    def y_min(self) -> float:
        return self.y_max - self.ny * self.cell_size_m

    @property
    def bounds(self) -> BBox:
        """Outer edges ``(xmin, ymin, xmax, ymax)``; the C2 ``bbox_5070`` value."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Affine coefficients ``(a, b, c, d, e, f)`` in rasterio order.

        Feed straight to ``rasterio.transform.Affine(*grid.transform)`` or use
        :meth:`rasterio_transform`.
        """
        return (self.cell_size_m, 0.0, self.x_min, 0.0, -self.cell_size_m, self.y_max)

    def rasterio_transform(self) -> Any:
        """``affine.Affine`` for this grid (lazy import; rasterio not required
        to import this module)."""
        from rasterio.transform import Affine

        return Affine(*self.transform)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_bounds(
        cls,
        bounds: BBox,
        *,
        cell_size_m: float = CELL_SIZE_M,
        crs: str = CRS_STRING,
        pad_cells: int = 0,
        snap: bool = True,
    ) -> Grid:
        """Smallest grid covering ``bounds`` (outer edges, CRS units).

        With ``snap=True`` the edges are expanded outward to multiples of
        ``cell_size_m``, so grids built from different fires share one global
        lattice and can be compared cell-for-cell.
        """
        x0, y0, x1, y1 = (float(v) for v in bounds)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"bounds must be (xmin, ymin, xmax, ymax) with extent, got {bounds}")
        if snap:
            x0 = np.floor(x0 / cell_size_m) * cell_size_m
            y0 = np.floor(y0 / cell_size_m) * cell_size_m
            x1 = np.ceil(x1 / cell_size_m) * cell_size_m
            y1 = np.ceil(y1 / cell_size_m) * cell_size_m
        pad = pad_cells * cell_size_m
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        nx = max(1, int(round((x1 - x0) / cell_size_m)))
        ny = max(1, int(round((y1 - y0) / cell_size_m)))
        return cls(x_min=x0, y_max=y1, nx=nx, ny=ny, cell_size_m=cell_size_m, crs=crs)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Grid:
        """Recover the grid from a tensor store's ``x``/``y`` coordinates."""
        x = np.asarray(ds["x"].values, dtype=np.float64)
        y = np.asarray(ds["y"].values, dtype=np.float64)
        if x.size < 2 or y.size < 2:
            raise ValueError("cannot infer cell size from a grid with fewer than 2 cells per axis")
        cell = float(abs(x[1] - x[0]))
        crs = str(ds.attrs.get("crs", CRS_STRING))
        return cls(
            x_min=float(x[0] - cell / 2.0),
            y_max=float(y[0] + cell / 2.0),
            nx=int(x.size),
            ny=int(y.size),
            cell_size_m=cell,
            crs=crs,
        )

    # -- indexing -----------------------------------------------------------

    def rowcol(self, x: float, y: float) -> tuple[int, int]:
        """Array index ``(row, col)`` containing the point ``(x, y)``."""
        col = int(np.floor((x - self.x_min) / self.cell_size_m))
        row = int(np.floor((self.y_max - y) / self.cell_size_m))
        return row, col

    def xy(self, row: int, col: int) -> tuple[float, float]:
        """Cell-centre coordinates of array index ``(row, col)``."""
        return (
            self.x_min + (col + 0.5) * self.cell_size_m,
            self.y_max - (row + 0.5) * self.cell_size_m,
        )

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.ny and 0 <= col < self.nx

    def coord_arrays(self) -> dict[str, np.ndarray]:
        """``{"y": ..., "x": ...}`` ready to hand to :class:`xarray.DataArray`."""
        return {"y": self.y_coords, "x": self.x_coords}

    def attrs(self) -> dict[str, Any]:
        """C1 grid attributes for the store root."""
        return {"crs": self.crs, "cell_size_m": float(self.cell_size_m)}
