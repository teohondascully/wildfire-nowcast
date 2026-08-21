"""Native 30 m LANDFIRE for ELMFIRE — the input half of ADR-026 (3).

ELMFIRE is built for ~30 m LANDFIRE. The original G5 playbook told me to map OUR
1 km tensor into its inputs, which lobotomises it: ADR-026 (3) retired that and
ruled **native inputs, contract outputs**. This module is the native input path.
It touches nothing in ``data/`` — it pulls the same public LFPS service
``data/sources/fuels.py`` documents, onto a FINER grid, into a cache under
``vendor/`` which simviz owns.

WHAT IS FETCHED, AND WHY EACH ONE
---------------------------------
============ ================================= ==========================
raster       LFPS service                       why ELMFIRE needs it
============ ================================= ==========================
``fbfm40``   ``Landfire_LF{v}/LF{v}_FBFM40``    surface fuel model (Rothermel)
``cc``       ``Landfire_LF{v}/LF{v}_CC``        wind adjustment factor + crown
``ch``       ``Landfire_LF{v}/LF{v}_CH``        canopy height -> WAF, crown HPUA
``cbh``      ``Landfire_LF{v}/LF{v}_CBH``       crown initiation threshold
``cbd``      ``Landfire_LF{v}/LF{v}_CBD``       active-crown criterion
``dem``      ``Landfire_Topo/LF2020_Elev``      elevation
``slp``      ``Landfire_Topo/LF2020_SlpD``      slope, degrees
``asp``      ``Landfire_Topo/LF2020_Asp``       aspect, degrees
============ ================================= ==========================

``ch``/``cbh``/``cbd`` are the three C1 does not carry, and their absence is what
switched ELMFIRE's crown fire model OFF in S3. **They are fetched here rather
than added to C1**: ADR-026 (3) forbids growing the tensor contract to feed a
baseline, and this module honours that literally — nothing here can write to
``data/``.

ENCODINGS ARE LEFT NATIVE ON PURPOSE
------------------------------------
LANDFIRE ships CC in percent, CH and CBH in metres x 10, CBD in kg/m3 x 100.
ELMFIRE's own namelist defaults are ``CC_IN_PERCENT=.TRUE.``,
``CH_TIMES_10=.TRUE.``, ``CBH_TIMES_10=.TRUE.``, ``CBD_TIMES_100=.TRUE.``
(``elmfire_namelists.f90:112-118``), i.e. **its defaults already expect exactly
this encoding.** Passing the native integers through unconverted removes a whole
class of unit-conversion compromise, and it is independent evidence for
ADR-026 (3)'s direction: the simulator was designed around these products.

VINTAGE
-------
Fuels vintage is chosen by ``data.sources.fuels.vintage_for_fire`` — the same
leakage rule as C1 channel 9, imported rather than re-derived, so the baseline
cannot end up on a *newer* fuels layer than the model it is compared against.
Topography has one published vintage (``LF2020``) and is time-invariant to the
precision that matters, so no leakage rule applies to it.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.sources.fuels import (
    LFPS_BASE,
    LFPS_NODATA,
    vintage_for_fire,
)

__all__ = [
    "NativeLayer",
    "NATIVE_LAYERS",
    "CACHE_ROOT",
    "layer_url",
    "fetch_layer",
    "fetch_native_stack",
    "NativeStack",
    "synthetic_stack",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
#: simviz owns ``vendor/``. Never ``data/`` — that is frozen to this lead.
CACHE_ROOT = REPO_ROOT / "vendor" / "landfire_cache"


@dataclass(frozen=True)
class NativeLayer:
    """One ELMFIRE raster and where LFPS serves it."""

    stub: str
    #: ``"fuels"`` -> the vintage-selected folder; ``"topo"`` -> ``Landfire_Topo``.
    family: str
    code: str
    #: What the value means once ELMFIRE has applied its own default scaling.
    units: str
    #: Value written where LFPS has no data (ocean, off-CONUS).
    nodata_fill: int


NATIVE_LAYERS: tuple[NativeLayer, ...] = (
    NativeLayer("fbfm40", "fuels", "FBFM40", "Scott & Burgan class", 98),
    NativeLayer("cc", "fuels", "CC", "percent (CC_IN_PERCENT)", 0),
    NativeLayer("ch", "fuels", "CH", "m x 10 (CH_TIMES_10)", 0),
    NativeLayer("cbh", "fuels", "CBH", "m x 10 (CBH_TIMES_10)", 0),
    NativeLayer("cbd", "fuels", "CBD", "kg/m3 x 100 (CBD_TIMES_100)", 0),
    NativeLayer("dem", "topo", "Elev", "m", 0),
    NativeLayer("slp", "topo", "SlpD", "degrees", 0),
    NativeLayer("asp", "topo", "Asp", "degrees", 0),
)

#: The one topographic vintage LFPS publishes. Terrain is time-invariant, so the
#: fuels leakage rule does not apply and using LF2020 for a 2019 fire is not a
#: reach forward in any meaningful sense — the ground did not move.
TOPO_FOLDER = "Landfire_Topo"
TOPO_TAG = "LF2020"


def layer_url(layer: NativeLayer, folder: str, grid: Grid) -> str:
    """``exportImage`` URL clipping one layer onto ``grid`` at its own cell size.

    ``interpolation=RSP_NearestNeighbor`` for everything. On a ~30 m request grid
    that returns LANDFIRE's actual cell value rather than a blend of neighbours:
    FBFM40 is categorical and must never be interpolated (C1.5's own reasoning —
    an interpolated class is a fuel model that does not exist), and for the
    canopy layers nearest keeps the native integer encoding exact.
    """
    if layer.family == "topo":
        service = f"{TOPO_FOLDER}/{TOPO_TAG}_{layer.code}_CONUS"
    else:
        tag = folder.replace("Landfire_", "")
        service = f"{folder}/{tag}_{layer.code}_CONUS"
    minx, miny, maxx, maxy = grid.bounds
    return (
        f"{LFPS_BASE}/{service}/ImageServer/exportImage"
        f"?bbox={minx},{miny},{maxx},{maxy}"
        f"&bboxSR=5070&imageSR=5070&size={grid.nx},{grid.ny}"
        f"&format=tiff&interpolation=RSP_NearestNeighbor&f=image"
    )


def _cache_key(layer: NativeLayer, folder: str, grid: Grid) -> Path:
    payload = json.dumps(
        {
            "stub": layer.stub,
            "folder": folder if layer.family == "fuels" else TOPO_FOLDER,
            "bounds": [round(b, 3) for b in grid.bounds],
            "nx": grid.nx,
            "ny": grid.ny,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return CACHE_ROOT / f"{layer.stub}_{digest}.npy"


def fetch_layer(
    layer: NativeLayer,
    folder: str,
    grid: Grid,
    *,
    timeout_s: float = 300.0,
    use_cache: bool = True,
) -> np.ndarray:
    """One native layer as ``int16[ny, nx]`` on ``grid``, LFPS NoData filled.

    Cached under :data:`CACHE_ROOT` keyed by layer, folder and exact geometry, so
    a re-run of the same window is offline and byte-identical — which is what
    makes ELMFIRE's determinism checkable at all.
    """
    import rasterio  # noqa: PLC0415

    cache = _cache_key(layer, folder, grid)
    if use_cache and cache.exists():
        return np.load(cache)
    url = layer_url(layer, folder, grid)
    with urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
        blob = resp.read()
    with rasterio.open(io.BytesIO(blob)) as src:
        arr = src.read(1).astype(np.float32)
    if arr.shape != grid.shape:
        raise RuntimeError(
            f"{layer.code}: LFPS returned {arr.shape}, expected {grid.shape}. A "
            "silently different shape would misregister every fuel cell."
        )
    arr = np.where(arr <= LFPS_NODATA + 1.0, float(layer.nodata_fill), arr)
    out = np.rint(arr).astype(np.int16)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


@dataclass(frozen=True)
class NativeStack:
    """Every ELMFIRE input raster on one fine grid, plus its provenance."""

    grid: Grid
    layers: dict[str, np.ndarray]
    provenance: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Per-layer stats — the cheapest detector of a silently empty fetch."""
        out: dict[str, Any] = {}
        for stub, arr in sorted(self.layers.items()):
            out[stub] = {
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": round(float(arr.mean()), 3),
                "nonzero_frac": round(float(np.count_nonzero(arr)) / arr.size, 4),
            }
        return out


def fetch_native_stack(
    grid: Grid,
    fire_year: int,
    *,
    timeout_s: float = 300.0,
    use_cache: bool = True,
) -> NativeStack:
    """All eight native rasters for ``grid``, fuels vintage chosen by fire year."""
    folder = vintage_for_fire(int(fire_year))
    layers = {
        layer.stub: fetch_layer(layer, folder, grid, timeout_s=timeout_s, use_cache=use_cache)
        for layer in NATIVE_LAYERS
    }
    return NativeStack(
        grid=grid,
        layers=layers,
        provenance={
            "source": "USGS LANDFIRE Product Service (lfps.usgs.gov)",
            "fuels_folder": folder,
            "topo_folder": f"{TOPO_FOLDER}/{TOPO_TAG}",
            "fire_year": int(fire_year),
            "fuels_staleness_years": int(fire_year)
            - int("".join(c for c in folder if c.isdigit())),
            "cell_size_m": round(grid.cell_size_m, 4),
            "shape": list(grid.shape),
            "bounds_5070": [round(b, 2) for b in grid.bounds],
            "interpolation": "RSP_NearestNeighbor for every layer",
            "encodings_left_native": (
                "CC percent, CH/CBH m x 10, CBD kg/m3 x 100 — ELMFIRE's own "
                "namelist defaults expect exactly this"
            ),
        },
    )


def synthetic_stack(
    grid: Grid,
    *,
    fbfm40: int,
    canopy_cover_pct: int,
    canopy_height_m10: int,
    canopy_base_height_m10: int,
    canopy_bulk_density_kgm3_100: int,
    elevation_m: int = 500,
    slope_deg: int = 0,
    aspect_deg: int = 180,
) -> NativeStack:
    """A uniform, network-free stack for the non-degeneracy playthrough.

    Every value is stated explicitly so the expected fire behaviour is a property
    of the scenario, not of whatever LANDFIRE happens to say about a real place.
    That is what makes playthrough 2 runnable with no network and no human.
    """
    shape = grid.shape
    values = {
        "fbfm40": fbfm40,
        "cc": canopy_cover_pct,
        "ch": canopy_height_m10,
        "cbh": canopy_base_height_m10,
        "cbd": canopy_bulk_density_kgm3_100,
        "dem": elevation_m,
        "slp": slope_deg,
        "asp": aspect_deg,
    }
    return NativeStack(
        grid=grid,
        layers={k: np.full(shape, v, dtype=np.int16) for k, v in values.items()},
        provenance={
            "source": "SYNTHETIC uniform stack (no LFPS call)",
            "values": values,
            "cell_size_m": round(grid.cell_size_m, 4),
            "shape": list(shape),
        },
    )
