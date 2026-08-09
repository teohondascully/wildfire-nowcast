"""Vector -> C1 grid rasterisation.

Two rules, deliberately different:

* **Polygons** (perimeters) are converted by *area fraction*: rasterise on a
  ``factor``x finer lattice, block-average, threshold. Centroid-in-polygon
  throws away half a cell of information at 1 km, and ``all_touched=True``
  dilates every perimeter by a cell, which at GOFER's ~2 km effective resolution
  is a systematic positive area bias.
* **Lines** (active fire lines) are converted with ``all_touched=True``, because
  a line has zero area and any fractional rule would erase it.

Grid geometry comes from :class:`wildfire_nowcast.common.grid.Grid` — this module
places vectors on a grid, it does not define one (C0, ADR-007). Binary dilation
lives in :mod:`wildfire_nowcast.common.states` because it is part of the
contract-adjudicated C1.1 state rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.features import rasterize as _rio_rasterize
from shapely.geometry.base import BaseGeometry

from wildfire_nowcast.common.grid import Grid

__all__ = [
    "DEFAULT_OVERSAMPLE",
    "COVER_THRESHOLD",
    "oversampled_transform",
    "polygon_coverage",
    "polygon_mask",
    "line_mask",
]

#: 10x oversampling => 100 m sub-cells => coverage fraction quantised to 1%.
DEFAULT_OVERSAMPLE = 10
#: A cell is "inside the perimeter" when the perimeter covers at least this much
#: of it. 0.5 is the area-preserving choice (unbiased for straight boundaries).
COVER_THRESHOLD = 0.5


def oversampled_transform(grid: Grid, factor: int) -> Any:
    """``affine.Affine`` for a ``factor``-times finer lattice over ``grid``.

    Derived from the canonical :class:`Grid` corner and cell size; it introduces
    no second definition of the grid, only a sub-cell view of the same one.
    """
    from rasterio.transform import Affine  # noqa: PLC0415

    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    r = grid.cell_size_m / factor
    return Affine(r, 0.0, grid.x_min, 0.0, -r, grid.y_max)


def polygon_coverage(
    geom: BaseGeometry | None,
    grid: Grid,
    *,
    factor: int = DEFAULT_OVERSAMPLE,
) -> np.ndarray:
    """Fraction of each grid cell covered by ``geom``, shape ``grid.shape``, float32.

    ``geom`` must already be in ``grid.crs``. ``None``/empty gives all zeros.
    """
    h, w = grid.shape
    if geom is None or geom.is_empty:
        return np.zeros((h, w), dtype=np.float32)
    fine = _rio_rasterize(
        [(geom, 1)],
        out_shape=(h * factor, w * factor),
        transform=oversampled_transform(grid, factor),
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return (
        fine.reshape(h, factor, w, factor).mean(axis=(1, 3)).astype(np.float32)
    )


def polygon_mask(
    geom: BaseGeometry | None,
    grid: Grid,
    *,
    factor: int = DEFAULT_OVERSAMPLE,
    threshold: float = COVER_THRESHOLD,
) -> np.ndarray:
    """Boolean "inside the perimeter" mask via the area-fraction rule."""
    return polygon_coverage(geom, grid, factor=factor) >= threshold


def line_mask(geom: BaseGeometry | None, grid: Grid) -> np.ndarray:
    """Boolean mask of cells touched by a (multi)line, at native 1 km resolution."""
    h, w = grid.shape
    if geom is None or geom.is_empty:
        return np.zeros((h, w), dtype=bool)
    return _rio_rasterize(
        [(geom, 1)],
        out_shape=(h, w),
        transform=grid.rasterio_transform(),
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
