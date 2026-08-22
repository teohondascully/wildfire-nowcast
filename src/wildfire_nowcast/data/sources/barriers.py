"""C1 channel 12: ``water_barrier_mask`` {0,1}.

A barrier is anything a 1 km cell of fire has to *jump*: open water, a major
river, a multi-lane highway. Channel 12 is the feature the barrier-crossing
episode mining (P3, ``data/events/crossings.json``) is defined against, so its
definition has to be stable before crossings are mined.

Sources, all GEE:

* ``JRC/GSW1_4/GlobalSurfaceWater`` band ``occurrence`` - permanent water. Chosen
  over NHD because NHD is not a first-class GEE raster and GSW is a single
  global, temporally consistent, well-validated layer. Lakes and wide rivers
  come out of this.
* ``TIGER/2016/Roads`` filtered to primary/secondary (``rttyp`` in S/U/I) -
  highways wide enough to hold a fire, rasterised as lines.
* Narrow rivers are the gap: at 1 km a 60 m river is 6% of a cell and vanishes
  under any area threshold. They are captured by the *line* rasterisation of
  GSW's skeleton rather than by its area fraction - otherwise every documented
  river jump would have no river in the tensor to jump over.

A cell is a barrier when it is >= ``WATER_FRACTION_THRESHOLD`` water by area, OR
is touched by a major-road line, OR is touched by a permanent-water line.
Deliberately *not* merged with FBFM40's non-burnable classes (91-99): those stay
in channel 9 so the model can learn "non-burnable fuel" and "physical barrier"
separately, and so the two can be cross-checked in QA.

Retrieval is synchronous chunked ``computePixels`` (ADR-004), not a batch
export: our credentials hold only the ``earthengine,cloud-platform`` scope pair.
"""

from __future__ import annotations

from typing import Any

from wildfire_nowcast.common.contract import CELL_SIZE_M
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.sources.gee import (
    ExportConfig,
    export_image,
    fetch_bands_chunked,
    initialize_ee,
    region_for_grid,
)

__all__ = [
    "GSW_ASSET",
    "ROADS_ASSET",
    "WATER_FRACTION_THRESHOLD",
    "WATER_OCCURRENCE_THRESHOLD",
    "barrier_image",
    "fetch_barriers",
    "export_barriers",
    "barrier_provenance",
]

GSW_ASSET = "JRC/GSW1_4/GlobalSurfaceWater"
ROADS_ASSET = "TIGER/2016/Roads"

#: Water present in >= 80% of observations counts as permanent.
WATER_OCCURRENCE_THRESHOLD = 80
#: Areal fraction of a 1 km cell that must be permanent water to flag it.
WATER_FRACTION_THRESHOLD = 0.30
#: TIGER route types treated as major: Interstate, US, State.
MAJOR_ROUTE_TYPES = ("I", "U", "S")


def barrier_image(grid: Grid) -> Any:
    """Single-band uint8 ``water_barrier_mask`` on the C1 lattice."""
    ee = initialize_ee()
    region = region_for_grid(grid)
    crs_transform = [CELL_SIZE_M, 0, grid.x_min, 0, -CELL_SIZE_M, grid.y_max]

    water30 = ee.Image(GSW_ASSET).select("occurrence").gte(WATER_OCCURRENCE_THRESHOLD).unmask(0)
    # areal rule: what fraction of the 1 km cell is permanent water
    water_frac = water30.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65535).reproject(
        crs=grid.crs, crsTransform=crs_transform
    )
    water_area = water_frac.gte(WATER_FRACTION_THRESHOLD)
    # connectivity rule: any permanent water at all in the cell keeps narrow
    # rivers alive, which the areal rule would erase.
    water_line = water30.reduceResolution(reducer=ee.Reducer.max(), maxPixels=65535).reproject(
        crs=grid.crs, crsTransform=crs_transform
    )

    roads = (
        ee.FeatureCollection(ROADS_ASSET)
        .filterBounds(region)
        .filter(ee.Filter.inList("rttyp", list(MAJOR_ROUTE_TYPES)))
    )
    road_mask = (
        ee.Image(0).byte().paint(roads, 1, 1).reproject(crs=grid.crs, crsTransform=crs_transform)
    )

    return (
        water_area.Or(water_line)
        .Or(road_mask)
        .rename("water_barrier_mask")
        .unmask(0)
        .toUint8()
        .clip(region)
    )


def fetch_barriers(grid: Grid) -> Any:
    """Pull ``water_barrier_mask`` as a ``(H, W)`` array (ADR-004 sync fetch)."""
    return fetch_bands_chunked(barrier_image(grid), grid, ["water_barrier_mask"])[
        "water_barrier_mask"
    ]


def export_barriers(fire_id: str, grid: Grid, config: ExportConfig | None = None) -> Any:
    """Batch-export variant. Gated by ADR-004; :func:`fetch_barriers` is the default."""
    return export_image(
        barrier_image(grid),
        name=f"{fire_id}__barriers",
        grid=grid,
        config=config,
        channels=["water_barrier_mask"],
    )


def barrier_provenance() -> dict[str, Any]:
    return {
        "barrier_water_source": GSW_ASSET,
        "barrier_road_source": ROADS_ASSET,
        "barrier_water_occurrence_threshold_pct": WATER_OCCURRENCE_THRESHOLD,
        "barrier_water_fraction_threshold": WATER_FRACTION_THRESHOLD,
        "barrier_major_route_types": list(MAJOR_ROUTE_TYPES),
        "barrier_rule": "areal water OR any permanent water in cell OR major road line",
    }
