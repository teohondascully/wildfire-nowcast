"""GOES ABI Fire Detection Characterization (FDC) access - the GOFER input.

GOFER (Zenodo 14638647) publishes hourly perimeters for **28 California fires,
2019-2021, and nothing else**: all five Zenodo versions of concept record
8327264 carry the same 28 fires, and the newest (v0.2, 2025-01-13) is still
titled "...for 28 California wildfires from 2019-2021". Extending the corpus to
2022-2025 therefore means *running their algorithm*, not downloading more of
their output. The algorithm is open (github.com/tianjialiu/GOFER, ESSD 2024,
doi:10.5194/essd-16-1395-2024) and its inputs are all in the Earth Engine
catalog, so this is mechanical rather than speculative - but it is a
REIMPLEMENTATION, and this module says so everywhere rather than letting a
downstream reader assume the labels came from Zenodo.

What lives here is the *satellite-facing* half: which GOES collection is
authoritative on a given date, the smoothing kernel radius, the per-hour fire
detection confidence field, and the terrain-parallax displacement. The
perimeter algorithm itself is :mod:`wildfire_nowcast.data.gofer_ext`.

Three facts that make the extension mechanically identical to 2019-2021, all
verified live against the catalog rather than assumed:

* **The GOES fixed grid does not move.** GOES-16/17/18/19 FDCF all report the
  same ``nominalScale`` (2004.017315487541 m) and the same affine, and their
  GEOS projections carry ``central_meridian`` -75 (East) / -137 (West) for
  *every* satellite in the series. GOFER's hardcoded ``lon_0`` values are
  therefore correct for GOES-19 and GOES-18 too, and a kernel radius computed
  for GOES-16 is the same number as for GOES-19.
* **The satellite handover dates are already in GOFER's own code.**
  ``GOFER_functions.js`` maps GOES-East to GOES-16 then GOES-19 (2025-04-07)
  and GOES-West to GOES-17 then GOES-18 (2023-01-04). The upstream authors
  wrote the extension path in; :data:`SAT_BREAKPOINTS` is a transcription.
* **The mask codes are unambiguous in a fire AOI.** ``Mask <= 35`` selects only
  the fire classes: measured over the Kincade AOI, the only value <= 35 present
  is 32, and non-fire codes (100 "processed, no fire", 245) are all > 35. The
  ``0`` no-data value is masked by the catalog, so it never enters ``mod(10)``.

Validation of record for the port: :func:`kernel_resolutions` reproduces the
published Kincade kernels **exactly** - ``[3453, 2550, 1725]`` for
East/West/Combined against ``largeFires_metadata.js``'s ``kernels:
[3453,2550,1725]``. That is an exact integer match on a quantity derived from
the GOES pixel geometry, which is the cheapest available proof that this
environment reproduces theirs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.sources.gee import fetch_bands_chunked, initialize_ee

__all__ = [
    "GOES_NOMINAL_SCALE_M",
    "SAT_BREAKPOINTS",
    "SAT_LON0",
    "MASK_FIRE_MAX",
    "CONFIDENCE_BY_CODE",
    "SatelliteChoice",
    "satellite_for",
    "goes_collection",
    "kernel_resolutions",
    "hourly_confidence",
    "parallax_offsets_5070",
    "goes_provenance",
]

#: GOES ABI fixed-grid nominal scale, identical for 16/17/18/19 (verified live).
GOES_NOMINAL_SCALE_M = 2004.017315487541

#: GOES-East / GOES-West handover dates, transcribed from GOFER_functions.js.
#: ``(first_valid_date, collection_id)``, evaluated as "latest breakpoint <= date".
SAT_BREAKPOINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "East": (("2017-07-10", "NOAA/GOES/16/FDCF"), ("2025-04-07", "NOAA/GOES/19/FDCF")),
    "West": (("2018-08-28", "NOAA/GOES/17/FDCF"), ("2023-01-04", "NOAA/GOES/18/FDCF")),
}

#: Sub-satellite longitude of the GEOS projection, per side. Constant across the
#: series in the GEE catalog (the products are on the standard fixed grid).
SAT_LON0: dict[str, float] = {"East": -75.0, "West": -137.0}

#: FDC ``Mask`` codes at or below this are fire classes (10-15 nominal, 30-35
#: temporally filtered). Everything above is cloud / water / no-fire.
MASK_FIRE_MAX = 35

#: ``mask.mod(10).add(1)`` -> GOFER confidence value. Code 1 is "processed fire"
#: (best), 6 is "low probability". Transcribed from ``Export_FireConf.js``.
CONFIDENCE_BY_CODE: dict[int, float] = {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.8, 4: 0.5, 5: 0.3, 6: 0.1}

#: WGS84 / GRS80 constants and the GOES orbit radius, from ``Export_Parallax.js``
#: (itself adapted from goes-ortho by Steven Pestana).
_REQ = 6378137.0
_RPOL = 6356752.31414
_H = 35786023.0 + _REQ
_ECC = 0.0818191910435

#: GOFER zeroes the DEM below this height: the displacement is imprecise at
#: elevations under half the DEM's own resolution.
_DEM_FLOOR_M = 5.0


@dataclass(frozen=True)
class SatelliteChoice:
    """Which physical satellite serves a side on a given date."""

    side: str
    collection_id: str
    lon0: float

    @property
    def short(self) -> str:
        return self.collection_id.split("/")[2]


def satellite_for(when: datetime, side: str) -> SatelliteChoice:
    """The GOES collection authoritative for ``side`` on ``when``.

    Mirrors ``GOFER_functions.getGOEScol``: take the latest breakpoint that does
    not postdate the request, falling back to the first entry. Raising on a date
    before the first breakpoint would be wrong for the same reason GOFER falls
    back - the earliest GOES-16 data predates the operational handover.
    """
    if side not in SAT_BREAKPOINTS:
        raise ValueError(f"side must be 'East' or 'West', got {side!r}")
    when_utc = when.astimezone(UTC) if when.tzinfo else when.replace(tzinfo=UTC)
    chosen = SAT_BREAKPOINTS[side][0]
    for date_str, coll in SAT_BREAKPOINTS[side]:
        if datetime.fromisoformat(date_str).replace(tzinfo=UTC) <= when_utc:
            chosen = (date_str, coll)
    return SatelliteChoice(side=side, collection_id=chosen[1], lon0=SAT_LON0[side])


def goes_collection(when: datetime, side: str) -> Any:
    """``ee.ImageCollection`` of the FDC ``Mask`` band for ``side`` on ``when``."""
    ee = initialize_ee()
    return ee.ImageCollection(satellite_for(when, side).collection_id).select("Mask")


def kernel_resolutions(aoi: Any, when: datetime) -> dict[str, int]:
    """GOFER's per-fire smoothing radii, in metres, for East / West / Combined.

    Faithful port of ``Calc_KernelRes.js``: rasterise a random image in each
    satellite's native projection, vectorise the resulting cells at 10 m, and
    take the **area-weighted mean of sqrt(cell area)** over the cells touching
    the AOI. The Combined kernel uses the *intersection* lattice (the sum of the
    two random images), which is finer than either alone - that is why it is
    ~1700 m while East is ~3400 m.

    Verified exact on Kincade: this returns ``{'East': 3453, 'West': 2550,
    'Combined': 1725}``, matching ``largeFires_metadata.js`` digit for digit.
    """
    ee = initialize_ee()
    east = satellite_for(when, "East")
    west = satellite_for(when, "West")
    stamp = when.strftime("%Y-%m-%d")
    nxt = (when + timedelta(days=1)).strftime("%Y-%m-%d")
    e_proj = ee.ImageCollection(east.collection_id).filterDate(stamp, nxt).first().projection()
    w_proj = ee.ImageCollection(west.collection_id).filterDate(stamp, nxt).first().projection()

    e_rand = (
        ee.Image.random(0).multiply(1e4).toInt().reproject(crs=e_proj, scale=e_proj.nominalScale())
    )
    w_rand = (
        ee.Image.random(20).multiply(1e4).toInt().reproject(crs=w_proj, scale=w_proj.nominalScale())
    )

    def _res(img: Any) -> Any:
        vec = img.reduceToVectors(
            geometry=aoi.buffer(10000), crs="EPSG:4326", scale=10, maxPixels=int(1e12)
        ).filterBounds(aoi)

        def _tag(feat: Any) -> Any:
            area = ee.Number(feat.geometry().area(10))
            return feat.set("area", area).set("res_warea", area.sqrt().multiply(area))

        vec = vec.map(_tag)
        return vec.aggregate_sum("res_warea").divide(vec.aggregate_sum("area")).round()

    values = ee.List([_res(e_rand), _res(w_rand), _res(e_rand.add(w_rand))]).getInfo()
    return {"East": int(values[0]), "West": int(values[1]), "Combined": int(values[2])}


def _hourly_confidence_image(side_choice: SatelliteChoice, start: datetime, n_hours: int) -> Any:
    """One image, ``n_hours`` bands, band *i* = max confidence over hour *i*.

    Built with a server-side ``ee.List.sequence().map`` rather than a Python
    loop: a client-side loop over hundreds of bands raises ``RecursionError``
    inside ``ee.serializer`` before a request is ever sent (insights item 15).
    """
    ee = initialize_ee()
    coll = ee.ImageCollection(side_choice.collection_id).select("Mask")
    start_ee = ee.Date(start.strftime("%Y-%m-%dT%H:%M:%S"))
    codes = list(CONFIDENCE_BY_CODE)
    values = [CONFIDENCE_BY_CODE[c] for c in codes]

    def _hour(i_hour: Any) -> Any:
        i_hour = ee.Number(i_hour)
        end = start_ee.advance(i_hour, "hour")
        window = coll.filterDate(end.advance(-1, "hour"), end)

        def _code(img: Any) -> Any:
            return img.updateMask(img.lte(MASK_FIRE_MAX)).mod(10).add(1).toInt()

        best = ee.Image(
            ee.Algorithms.If(window.size().gt(0), window.map(_code).min().unmask(0), ee.Image(0))
        ).toInt()
        return best.remap(codes, values, 0).toFloat().rename("conf")

    stack = ee.ImageCollection(ee.List.sequence(1, n_hours, 1).map(_hour)).toBands()
    return stack.rename([band_name(i) for i in range(1, n_hours + 1)])


def band_name(i_hour: int) -> str:
    """Band label for hour ``i_hour`` (1-based, matching GOFER's ``timeStep``)."""
    return f"h{i_hour:05d}"


def hourly_confidence(
    grid: Grid, start: datetime, n_hours: int, side: str, *, chunk_hours: int = 48
) -> np.ndarray:
    """``(n_hours, ny, nx)`` per-hour GOES fire confidence on ``grid``.

    ``grid`` is the fine (sub-kilometre) EPSG:5070 analysis grid, not the C1
    lattice: the parallax shift and the ~1.7 km boxcar both need sub-GOES-pixel
    sampling to be meaningful. Values are GOFER's confidence scale in [0, 1].

    Hour *i* covers ``(start + (i-1) h, start + i h]``, i.e. the label at
    ``timeStep = i`` is END-OF-HOUR, exactly as C1.3 requires and exactly as
    GOFER's ``tUTC`` is defined.
    """
    choice = satellite_for(start, side)
    image = _hourly_confidence_image(choice, start, n_hours)
    out = np.zeros((n_hours, grid.ny, grid.nx), dtype=np.float32)
    names = [band_name(i) for i in range(1, n_hours + 1)]
    for i in range(0, n_hours, chunk_hours):
        block = names[i : i + chunk_hours]
        fetched = fetch_bands_chunked(image, grid, block)
        for j, name in enumerate(block):
            out[i + j] = fetched[name]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _abi_scan_angles(
    lon_deg: np.ndarray, lat_deg: np.ndarray, z_m: np.ndarray, lon0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ground point (lon, lat, z) -> ABI (x, y) scan angles. Port of ``lonlat2abi``."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    lon0 = np.radians(lon0_deg)
    lat_geo = np.arctan((_RPOL**2 / _REQ**2) * np.tan(lat))
    rc = _RPOL / np.sqrt(1.0 - (_ECC**2) * np.cos(lat_geo) ** 2) + z_m
    sx = _H - rc * np.cos(lat_geo) * np.cos(lon - lon0)
    sy = -rc * np.cos(lat_geo) * np.sin(lon - lon0)
    sz = rc * np.sin(lat_geo)
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    return x, y


def _abi_to_lonlat(x: np.ndarray, y: np.ndarray, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """ABI scan angles -> the ELLIPSOID lon/lat they intersect. Port of ``abi2latlon``.

    This is where parallax comes from: the satellite places every detection at
    the point where its line of sight crosses the ellipsoid, so a fire at
    elevation ``z`` is drawn at the wrong ground location.
    """
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (_REQ**2 / _RPOL**2) * np.sin(y) ** 2)
    b = -2.0 * _H * np.cos(x) * np.cos(y)
    c = _H**2 - _REQ**2
    rs = (-b - np.sqrt(np.maximum(b**2 - 4.0 * a * c, 0.0))) / (2.0 * a)
    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)
    lat = np.arctan((_REQ**2 / _RPOL**2) * (sz / np.sqrt((_H - sx) ** 2 + sy**2)))
    lon = np.radians(lon0_deg) - np.arctan(sy / (_H - sx))
    return np.degrees(lon), np.degrees(lat)


def parallax_offsets_5070(
    grid: Grid, elevation_m: np.ndarray, lon0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Terrain-parallax displacement in **EPSG:5070 metres**, ``(dx, dy)``.

    Returns ``apparent - ground`` per cell, so that a *pull* warp
    ``corrected[p] = raw[p + d(p)]`` moves the observation from where the
    satellite drew it back to where it burned. That is the same convention
    ``Export_Parallax.js`` feeds to ``Image.displace``.

    DELIBERATE DEVIATION, stated rather than buried: GOFER computes the
    displacement in a GEOGRAPHIC frame and then divides the x-component by
    ``cos(lat)`` because ``ee.Image.displace`` under-displaces in longitude.
    Doing the whole calculation in EPSG:5070 removes both the hack and a real
    error source - Albers is rotated ~10-20 deg from true north over California,
    so an (east, north) offset applied as an (x, y) offset would be wrong by
    that angle. Here both the ground point and the apparent point are projected
    to 5070 and differenced there, so the vector is correct by construction and
    no convergence angle is approximated.
    """
    from pyproj import Transformer  # noqa: PLC0415

    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    to_xy = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = np.meshgrid(grid.x_coords, grid.y_coords)
    lon, lat = to_ll.transform(xs, ys)

    z = np.where(elevation_m > _DEM_FLOOR_M, elevation_m, 0.0)
    sx, sy = _abi_scan_angles(lon, lat, z, lon0_deg)
    lon_app, lat_app = _abi_to_lonlat(sx, sy, lon0_deg)
    x_app, y_app = to_xy.transform(lon_app, lat_app)
    return (x_app - xs).astype(np.float32), (y_app - ys).astype(np.float32)


def goes_provenance(start: datetime) -> dict[str, Any]:
    """C2 provenance fragment naming the exact satellites used."""
    east, west = satellite_for(start, "East"), satellite_for(start, "West")
    return {
        "goes_east_collection": east.collection_id,
        "goes_west_collection": west.collection_id,
        "goes_lon0_east_deg": east.lon0,
        "goes_lon0_west_deg": west.lon0,
        "goes_nominal_scale_m": GOES_NOMINAL_SCALE_M,
        "goes_mask_fire_max_code": MASK_FIRE_MAX,
    }
