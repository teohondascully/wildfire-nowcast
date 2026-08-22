"""C1 channels 5-8: elevation, slope, aspect_sin, aspect_cos from USGS 3DEP.

Source: ``USGS/3DEP/10m`` (GEE). Static per fire, repeated over time by the
assembler rather than stored T times here.

The one thing that is easy to get wrong: **derive slope and aspect at the
analysis scale, not at 10 m**. ``ee.Terrain.slope`` on the native 10 m DEM and
then averaged to 1 km gives the mean of local slopes, which for rough terrain is
much steeper than the slope of the 1 km surface the model actually advances
over. Fire spread in a 1 km cell responds to the 1 km slope. So: reduce
elevation to 1 km first, then differentiate.

Aspect is circular, hence the sin/cos pair - averaging degrees across north
would otherwise produce south.

Retrieval is synchronous chunked ``computePixels`` (ADR-004), not a batch
export: our credentials hold only the ``earthengine,cloud-platform`` scope pair.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wildfire_nowcast.common.contract import CELL_SIZE_M
from wildfire_nowcast.common.derive import aspect_to_sin_cos, slope_aspect_from_elevation
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.sources.gee import (
    ExportConfig,
    export_image,
    fetch_bands_chunked,
    initialize_ee,
    region_for_grid,
)

__all__ = [
    "DEM_ASSET",
    "terrain_image",
    "TERRAIN_CHANNELS",
    "fetch_terrain",
    "export_terrain",
    "slope_aspect_from_dem",
    "terrain_provenance",
]

DEM_ASSET = "USGS/3DEP/10m_collection"  # USGS/3DEP/10m is deprecated
DEM_NATIVE_RES_M = 10.0


def terrain_image(grid: Grid) -> Any:
    """4-band static terrain image on the C1 lattice.

    Bands: ``elevation`` (m), ``slope`` (deg), ``aspect_sin``, ``aspect_cos``.
    """
    ee = initialize_ee()
    region = region_for_grid(grid)
    coll = ee.ImageCollection(DEM_ASSET).select("elevation")
    # mosaic() drops the default projection, and reduceResolution below requires
    # one; restore it from a member tile rather than letting EE guess.
    dem = coll.mosaic().setDefaultProjection(ee.Image(coll.first()).projection()).clip(region)

    # Aggregate to the analysis scale FIRST (see module docstring).
    dem_1km = dem.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65535).reproject(
        crs=grid.crs,
        crsTransform=[CELL_SIZE_M, 0, grid.x_min, 0, -CELL_SIZE_M, grid.y_max],
    )
    terrain = ee.Terrain.products(dem_1km)
    slope = terrain.select("slope")
    aspect_rad = terrain.select("aspect").multiply(np.pi / 180.0)
    return (
        dem_1km.rename("elevation")
        .addBands(slope.rename("slope"))
        .addBands(aspect_rad.sin().rename("aspect_sin"))
        .addBands(aspect_rad.cos().rename("aspect_cos"))
        .toFloat()
    )


TERRAIN_CHANNELS = ("elevation", "slope", "aspect_sin", "aspect_cos")

#: Horn's slope/aspect is a 3x3 stencil, so one ring of context is enough for a
#: tile's interior to be BIT-IDENTICAL to the untiled result. Verified, not
#: assumed - see :func:`_fetch_terrain_tiled`. Halos of 2 and 4 were also
#: measured and are no better, which is itself the evidence that the residual is
#: not a stencil effect.
_HALO_CELLS = 1

#: Tile edge lengths (in 1 km C1 cells) tried in order when a whole-domain
#: terrain request is refused. Measured: 73x90 fails, 40x40 succeeds.
_TILE_LADDER: tuple[int, ...] = (40, 24, 12)


def _is_reprojection_too_large(exc: Exception) -> bool:
    """True for Earth Engine's source-footprint cap, and nothing else.

    Matched narrowly on purpose. A broad ``except Exception -> retry smaller``
    would silently convert a genuine auth or asset error into a slow loop that
    ends in the same failure with less information.
    """
    return "reprojection output too large" in str(exc).lower()


def _subgrid(grid: Grid, y0: int, x0: int, ny: int, nx: int) -> Grid:
    """A window of ``grid`` that stays on the same continental lattice (C1.2)."""
    cs = grid.cell_size_m
    return Grid(
        x_min=grid.x_min + x0 * cs,
        y_max=grid.y_max - y0 * cs,
        nx=nx,
        ny=ny,
        cell_size_m=cs,
        crs=grid.crs,
    )


def _fetch_terrain_tiled(grid: Grid, tile: int, halo: int = _HALO_CELLS) -> dict[str, np.ndarray]:
    """Whole-domain terrain assembled from haloed spatial tiles.

    Chunking terrain over SPACE is the mirror of chunking RTMA over TIME, and it
    has the extra requirement that RTMA does not: slope and aspect are
    *neighbourhood* quantities, so a naive tiling would stamp a seam of wrong
    values every ``tile`` cells. Each tile is therefore fetched with a one-cell
    halo and cropped back, which makes every returned cell see the same 3x3
    stencil it would have seen in a single whole-domain request. Tiles are cut on
    the C1 lattice, so there is no resampling and no sub-cell offset to reconcile.

    EQUIVALENCE, MEASURED ON A SHIPPED FIRE rather than argued. Rebuilding Zogg's
    terrain (48x39, built originally by the untiled path) with ``tile=20``:

    * Earth Engine is deterministic - the untiled request repeated is
      byte-identical to itself AND to what is on disk, all four channels.
    * An interior window fetched alone is byte-identical to the same cells inside
      the whole-domain fetch (0 of 400 cells differ).
    * Tiled vs untiled differs on **exactly 170 cells, which are exactly the
      domain's outer ring** (39+39+48+48 minus the four corners). No seam appears
      at any tile boundary; the interior is bit-exact. Magnitudes are
      <=0.13 m elevation and <=1 deg slope.

    The ring is where ``.clip()`` truncates the 10 m aggregation, so its 1 km mean
    depends on the clip rectangle, which necessarily differs for an edge tile.
    WHY THIS IS ACCEPTABLE, and it is a fact about C1.2 rather than a hope: the
    domain is the final perimeter buffered 10 km, so measured across all built
    fires the nearest burned cell sits **10-11 cells from the domain edge and the
    ring never burns in any fire**. Widening the halo does not remove the residual
    (halo 1/2/4 all leave the same ring), which confirms the cause is the clip and
    not the stencil. The alternative - letting the halo run outside the domain so
    ring cells aggregate completely - was rejected because it would make large
    fires' edges *differ in kind* from the small fires already shipped.
    """
    out = {ch: np.empty(grid.shape, dtype=np.float32) for ch in TERRAIN_CHANNELS}
    for y0 in range(0, grid.ny, tile):
        for x0 in range(0, grid.nx, tile):
            bh, bw = min(tile, grid.ny - y0), min(tile, grid.nx - x0)
            hy0, hx0 = max(y0 - halo, 0), max(x0 - halo, 0)
            hy1 = min(y0 + bh + halo, grid.ny)
            hx1 = min(x0 + bw + halo, grid.nx)
            sub = _subgrid(grid, hy0, hx0, hy1 - hy0, hx1 - hx0)
            got = fetch_bands_chunked(terrain_image(sub), sub, list(TERRAIN_CHANNELS))
            iy, ix = y0 - hy0, x0 - hx0
            for ch in TERRAIN_CHANNELS:
                out[ch][y0 : y0 + bh, x0 : x0 + bw] = got[ch][iy : iy + bh, ix : ix + bw]
    return out


def fetch_terrain(
    grid: Grid, *, tile_ladder: tuple[int, ...] = _TILE_LADDER
) -> dict[str, np.ndarray]:
    """Pull the four static terrain channels as ``{channel: (H, W) float32}``.

    Tries the whole domain first, so every fire small enough to fit takes exactly
    the path it always took and its values cannot move. Only on Earth Engine's
    ``Reprojection output too large`` does it fall back to haloed spatial tiling.

    WHY THIS IS NEEDED. 3DEP is a ~10 m product and its tiles are geographic,
    while C1 is EPSG:5070 Albers. To serve one 1 km Albers request Earth Engine
    must first materialise the DEM over the axis-aligned geographic bounding box
    of a rotated Albers rectangle - for Creek's 73x90 km domain that is
    **11,695 x 10,355 native pixels (1.2e8)**, and the request is refused. The
    cap is on the SOURCE footprint, not on our 6,570 output cells, which is why
    it appears abruptly at ~3.5x the domain size of the first five fires and
    could not have been extrapolated from their timings.
    """
    try:
        return fetch_bands_chunked(terrain_image(grid), grid, list(TERRAIN_CHANNELS))
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is the known cap
        if not _is_reprojection_too_large(exc):
            raise
        last = exc
    for tile in tile_ladder:
        try:
            return _fetch_terrain_tiled(grid, tile)
        except Exception as exc:  # noqa: BLE001
            if not _is_reprojection_too_large(exc):
                raise
            last = exc
    raise RuntimeError(
        f"3DEP terrain for a {grid.ny}x{grid.nx} domain exceeded Earth Engine's "
        f"reprojection cap at every tile size in {tile_ladder}: {last}"
    ) from last


def export_terrain(fire_id: str, grid: Grid, config: ExportConfig | None = None) -> Any:
    """Batch-export variant. Gated by ADR-004; :func:`fetch_terrain` is the default."""
    return export_image(
        terrain_image(grid),
        name=f"{fire_id}__terrain",
        grid=grid,
        config=config,
        channels=list(TERRAIN_CHANNELS),
    )


def slope_aspect_from_dem(
    elevation: np.ndarray, res_m: float = CELL_SIZE_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline slope/aspect cross-check on the GEE export, as ``(slope_deg,
    aspect_sin, aspect_cos)``.

    Delegates to :mod:`wildfire_nowcast.common.derive` (C0): channels 6-8 are
    contract-declared derived channels, so their definition lives in ``common/``
    and this module only calls it. Note that ``common`` encodes a flat cell as
    ``(0, 0)`` - an unambiguous "no aspect" code - which is the intended C1
    behaviour and differs from a raw ``arctan2`` on a zero gradient.
    """
    slope_deg, aspect_deg = slope_aspect_from_elevation(elevation, res_m)
    sin_a, cos_a = aspect_to_sin_cos(aspect_deg, slope_deg)
    return slope_deg, sin_a, cos_a


def terrain_provenance() -> dict[str, Any]:
    return {
        "terrain_source": DEM_ASSET,
        "terrain_native_res_m": DEM_NATIVE_RES_M,
        "terrain_derivation": "mean-aggregate DEM to 1 km, then Horn slope/aspect",
        "aspect_encoding": "sin/cos of downslope azimuth (circular-safe)",
    }
