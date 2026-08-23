"""C1 channels 9 (``fuel_model_id``, FBFM40) and 10 (``canopy_cover``).

**Finding that changes the plan (2026-08-07).** The original design assumed
FBFM40 and canopy come from GEE. They do not. The Earth Engine public catalog
carries only
LANDFIRE ``v1.2.0``/``v1.4.0`` - EVC, EVH, EVT, BPS, ESP and the fire-regime
products. There is **no FBFM40 and no canopy-cover asset**, and v1.4.0
represents circa-2014 conditions, i.e. 5-7 years stale for 2019-2021 fires. That
is worse than the "LANDFIRE lags reality by 1-2 years" premise the mandatory
MTBS/NIFC correction was sized for.

The authoritative source is instead the **LANDFIRE Product Service (LFPS)** at
``lfps.usgs.gov`` - a public USGS REST/ArcGIS ImageServer service, no OAuth, no
Cloud project (an email string is requested for usage statistics only). It
serves every LANDFIRE vintage including ``FBFM40``, ``CC``, ``CH``, ``CBD``,
``CBH`` for CONUS. Like GOFER on Zenodo, this decouples two more channels from
the Earth Engine blocker.

Vintage selection is a **leakage** question, not a convenience one: a LANDFIRE
release published after a fire has that fire's own burn scar baked into its
fuels, so training on it leaks the label. :func:`vintage_for_fire` therefore
picks the newest vintage whose *effective year* is strictly before the fire's
ignition year, and the residual staleness is closed forward with MTBS and
current-season NIFC perimeters (:mod:`.burn_scar`) - the MTBS/NIFC correction the
feature spec in ``README.md`` makes mandatory rather than optional.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LFPS_BASE",
    "LANDFIRE_VINTAGES",
    "FUELS_PUBLISHING_FOLDERS",
    "GEE_LANDFIRE_FALLBACK",
    "LandfireLayer",
    "vintage_for_fire",
    "verify_fuels_catalog",
    "lfps_image_server",
    "lfps_export_url",
    "FBFM40_NONBURNABLE",
    "LFPS_NODATA",
    "NODATA_FILL",
    "fetch_lfps_layer",
    "nodata_report",
    "fuel_provenance",
]

logger = logging.getLogger(__name__)

LFPS_BASE = "https://lfps.usgs.gov/arcgis/rest/services"

#: LANDFIRE service folder -> the year of ground conditions it represents.
#: Effective year matters, not release year: LF2016 Remap was published in 2019.
LANDFIRE_VINTAGES: dict[str, int] = {
    "Landfire_LF2016": 2016,
    "Landfire_LF2020": 2020,
    "Landfire_LF2022": 2022,
    "Landfire_LF2023": 2023,
    "Landfire_LF2024": 2024,
}

#: Folders that actually PUBLISH the two C1 fuel layers (FBFM40 + CC) on LFPS.
#:
#: **This is not the same set as** :data:`LANDFIRE_VINTAGES`, and assuming it was
#: is how every 2021 fire broke. ``Landfire_LF2020`` exists as a folder and is
#: therefore a legitimate-looking vintage, but it serves exactly ONE product -
#: ``LF2020_BPS_CONUS`` - and no fuel model or canopy layer at all. A 2021 fire
#: selects LF2020 by the vintage-precedes-ignition rule (ADR-005) and then gets
#: an HTTP 404 from a URL that is correctly formed against a folder that is
#: genuinely there. Enumerated from the live LFPS REST catalog on 2026-08-08;
#: :func:`verify_fuels_catalog` re-probes it so this stays a checkable claim
#: rather than folklore.
FUELS_PUBLISHING_FOLDERS: frozenset[str] = frozenset(
    {"Landfire_LF2016", "Landfire_LF2022", "Landfire_LF2023", "Landfire_LF2024"}
)

#: What GEE actually has, kept only as a degraded fallback and never silently:
#: circa-2014 vegetation structure, no fuel model at all.
GEE_LANDFIRE_FALLBACK = {
    "existing_vegetation_cover": "LANDFIRE/Vegetation/EVC/v1_4_0",
    "existing_vegetation_height": "LANDFIRE/Vegetation/EVH/v1_4_0",
    "existing_vegetation_type": "LANDFIRE/Vegetation/EVT/v1_4_0",
}


@dataclass(frozen=True)
class LandfireLayer:
    """One LANDFIRE raster: the C1 channel it feeds and its LFPS layer code."""

    c1_channel: int
    c1_name: str
    code: str  # e.g. "FBFM40"; the LFPS product code is "{yy}{code}"
    dtype: str


FBFM40 = LandfireLayer(9, "fuel_model_id", "FBFM40", "int16")
CANOPY_COVER = LandfireLayer(10, "canopy_cover", "CC", "float32")
FUEL_LAYERS: tuple[LandfireLayer, ...] = (FBFM40, CANOPY_COVER)

#: FBFM40 non-burnable classes (Scott & Burgan 2005): 91 urban, 92 snow/ice,
#: 93 agriculture, 98 water, 99 barren. These double as a barrier signal and are
#: cross-checked against channel 12 rather than merged into it.
FBFM40_NONBURNABLE: frozenset[int] = frozenset({91, 92, 93, 98, 99})

#: LFPS returns this where LANDFIRE has no data. For a CONUS fire that is the
#: OCEAN: LANDFIRE stops at the coastline. Measured on CZU (Santa Cruz Mtns),
#: 15.7% of the buffered domain - every one of those cells already flagged water
#: by channel 12, at elevation 0.0, and never burned.
LFPS_NODATA = -9999.0

#: Where LFPS has no data, encode the ground truth rather than the sentinel:
#: FBFM40 class 98 is "NB8 - Open Water", an EXISTING class, not an invented one,
#: and open water carries no canopy. Leaving -9999 in place is not an option:
#: it is finite and integral, so it passes every C1.5 declaration, while dragging
#: the channel-10 train mean to -492 % and poisoning every normalisation.
NODATA_FILL: dict[str, float] = {"fuel_model_id": 98.0, "canopy_cover": 0.0}


def vintage_for_fire(fire_year: int) -> str:
    """Newest LANDFIRE folder that predates ``fire_year`` **and publishes fuels**.

    Two constraints, and the second one is not cosmetic:

    1. The vintage must strictly precede the fire year. A post-fire LANDFIRE
       release has the fire's own scar baked into its fuels, so reaching forward
       is label leakage (ADR-005). This raises rather than reaching forward.
    2. The folder must actually serve FBFM40 and CC
       (:data:`FUELS_PUBLISHING_FOLDERS`). Selecting a folder that exists but
       publishes neither layer produces a 404 from a well-formed URL - which is
       what ``Landfire_LF2020`` does to every 2021 fire.

    Consequence worth stating out loud rather than burying: because LFPS skips
    fuels for LF2020, a 2021 fire falls back to **LF2016 - five years stale**,
    not the one-to-two years the MTBS correction is sized for. That is still
    the correct choice (it is the newest *legal* vintage), and the staleness is
    recorded in provenance as ``fuels_staleness_years`` and partly corrected by
    channel 13, whose MTBS window is anchored to this same vintage year and so
    widens automatically. It is not silently absorbed.
    """
    eligible = {
        k: v
        for k, v in LANDFIRE_VINTAGES.items()
        if v < fire_year and k in FUELS_PUBLISHING_FOLDERS
    }
    if not eligible:
        raise ValueError(
            f"no LANDFIRE vintage both strictly precedes {fire_year} and publishes "
            f"FBFM40+CC on LFPS (publishing folders: {sorted(FUELS_PUBLISHING_FOLDERS)}); "
            "refusing to use a post-fire fuels layer (label leakage)"
        )
    return max(eligible, key=lambda k: eligible[k])


def verify_fuels_catalog(timeout_s: float = 60.0) -> dict[str, Any]:
    """Re-probe LFPS and report drift against :data:`FUELS_PUBLISHING_FOLDERS`.

    Keeps the hardcoded set honest. Network call, so it is never on the build
    path - run it when a build 404s, or before extending to a new fire year.
    """
    import json as _json  # noqa: PLC0415
    from urllib.request import urlopen  # noqa: PLC0415

    observed: dict[str, list[str]] = {}
    for folder in sorted(LANDFIRE_VINTAGES):
        try:
            with urlopen(f"{LFPS_BASE}/{folder}?f=json", timeout=timeout_s) as r:  # noqa: S310
                cat = _json.loads(r.read())
        except Exception as exc:  # pragma: no cover - network
            # ADR-103 (4). The marker string below is NOT a trace, and that is
            # the whole problem: a folder whose probe failed has fewer than
            # `len(FUEL_LAYERS)` codes, so it drops out of `publishing` and
            # lands in `drift` looking EXACTLY like a folder that was reached
            # and found not to publish fuels. One of those means the catalog
            # constant is stale; the other means the network blinked. The
            # returned dict cannot tell them apart, so the log does.
            logger.warning(
                "LFPS catalog probe failed for %s (%s: %s); it will be reported as "
                "NOT publishing, which is indistinguishable in `drift` from a folder "
                "that was reached and genuinely publishes nothing",
                folder,
                type(exc).__name__,
                exc,
            )
            observed[folder] = [f"<probe failed: {exc}>"]
            continue
        names = {s["name"].split("/")[-1] for s in cat.get("services", [])}
        observed[folder] = sorted(
            layer.code
            for layer in FUEL_LAYERS
            if f"{folder.replace('Landfire_', '')}_{layer.code}_CONUS" in names
        )
    publishing = {f for f, codes in observed.items() if len(codes) == len(FUEL_LAYERS)}
    return {
        "probed_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layers_seen": observed,
        "publishing_folders_observed": sorted(publishing),
        "publishing_folders_declared": sorted(FUELS_PUBLISHING_FOLDERS),
        "drift": sorted(publishing.symmetric_difference(FUELS_PUBLISHING_FOLDERS)),
    }


def lfps_image_server(folder: str, layer: LandfireLayer, area: str = "CONUS") -> str:
    """ImageServer endpoint, e.g. ``.../Landfire_LF2016/LF2016_FBFM40_CONUS/ImageServer``."""
    tag = folder.replace("Landfire_", "")
    return f"{LFPS_BASE}/{folder}/{tag}_{layer.code}_{area}/ImageServer"


def lfps_export_url(
    folder: str,
    layer: LandfireLayer,
    bbox_5070: tuple[float, float, float, float],
    size_xy: tuple[int, int],
    *,
    area: str = "CONUS",
) -> str:
    """Direct ``exportImage`` URL clipping one layer to the C1 grid.

    ``bboxSR``/``imageSR`` are both 5070 and ``size`` is the exact cell count, so
    the service resamples onto our lattice instead of us resampling afterwards.
    ``nearest`` interpolation is mandatory for FBFM40 (a categorical field);
    canopy cover overrides it to ``bilinear`` at the call site.
    """
    minx, miny, maxx, maxy = bbox_5070
    nx, ny = size_xy
    return (
        f"{lfps_image_server(folder, layer, area)}/exportImage"
        f"?bbox={minx},{miny},{maxx},{maxy}"
        f"&bboxSR=5070&imageSR=5070&size={nx},{ny}"
        f"&format=tiff&pixelType=S16&interpolation=RSP_NearestNeighbor&f=image"
    )


def fetch_lfps_layer(
    folder: str,
    layer: LandfireLayer,
    grid: Any,
    *,
    area: str = "CONUS",
    interpolation: str | None = None,
    timeout_s: float = 180.0,
    nodata_out: dict[str, Any] | None = None,
) -> Any:
    """Clip one LANDFIRE layer onto the C1 grid via LFPS ``exportImage``.

    No Earth Engine, no auth (ADR-005). The service reprojects and resamples to
    our exact lattice, so nothing is resampled afterwards. Categorical layers
    (FBFM40) must use nearest neighbour; continuous ones (canopy cover) get
    bilinear. Returns a ``(H, W)`` float32 array.

    NoData (``-9999``, i.e. off the LANDFIRE coastline) is replaced by
    :data:`NODATA_FILL`. The count is returned by :func:`nodata_report` for the
    manifest, because a fill that nobody counts is indistinguishable from data.
    """
    import io  # noqa: PLC0415
    from urllib.request import urlopen  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import rasterio  # noqa: PLC0415

    interp = interpolation or (
        "RSP_NearestNeighbor" if layer.dtype.startswith("int") else "RSP_BilinearInterpolation"
    )
    minx, miny, maxx, maxy = grid.bounds
    url = (
        f"{lfps_image_server(folder, layer, area)}/exportImage"
        f"?bbox={minx},{miny},{maxx},{maxy}"
        f"&bboxSR=5070&imageSR=5070&size={grid.nx},{grid.ny}"
        f"&format=tiff&interpolation={interp}&f=image"
    )
    from urllib.error import HTTPError  # noqa: PLC0415

    try:
        with urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            blob = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"LFPS 404 for {folder}/{layer.code}: the folder exists but does not "
                f"publish this layer. Declared publishing folders are "
                f"{sorted(FUELS_PUBLISHING_FOLDERS)}; run "
                "`verify_fuels_catalog()` to re-probe LFPS and see the drift. "
                "URL was: " + url
            ) from exc
        raise
    with rasterio.open(io.BytesIO(blob)) as src:
        arr = src.read(1).astype(np.float32)
    if arr.shape != grid.shape:
        raise RuntimeError(f"{layer.code}: LFPS returned {arr.shape}, expected {grid.shape}")
    nodata = arr <= LFPS_NODATA + 1.0
    if nodata_out is not None:
        nodata_out["mask"] = nodata
    fill = NODATA_FILL.get(layer.c1_name)
    if fill is not None and nodata.any():
        arr = np.where(nodata, np.float32(fill), arr).astype(np.float32)
    return arr


def nodata_report(nodata_mask: Any, layer: LandfireLayer, water_mask: Any = None) -> dict[str, Any]:
    """How much of a layer was LFPS NoData, and whether channel 12 calls it water.

    QA measures, it does not fix (the fix happened in :func:`fetch_lfps_layer`).
    ``nodata_cells_not_water`` is the number that matters: NoData that channel 12
    does NOT call water is NoData we have coded as ocean without evidence - a
    defect rather than a coastline, and it must be visible in the manifest.
    """
    import numpy as np  # noqa: PLC0415

    nod = np.asarray(nodata_mask, dtype=bool)
    out: dict[str, Any] = {
        "nodata_cells": int(nod.sum()),
        "nodata_fraction": round(float(nod.mean()), 4),
        "fill_value": NODATA_FILL.get(layer.c1_name),
        "fill_rationale": "LANDFIRE stops at the coastline; FBFM40 98 = NB8 open water",
    }
    if water_mask is not None:
        w = np.asarray(water_mask) > 0
        out["nodata_cells_not_water"] = int((nod & ~w).sum())
    return out


def fuel_provenance(fire_year: int, area: str = "CONUS") -> dict[str, Any]:
    """C2 ``provenance`` fragment for the fuels channels."""
    folder = vintage_for_fire(fire_year)
    return {
        "fuels_source": "LANDFIRE Product Service (lfps.usgs.gov)",
        "fuels_vintage_folder": folder,
        "fuels_vintage_year": LANDFIRE_VINTAGES[folder],
        "fuels_area": area,
        "fuels_staleness_years": fire_year - LANDFIRE_VINTAGES[folder],
        "fuels_layers": {
            layer.c1_name: {
                "c1_channel": layer.c1_channel,
                "endpoint": lfps_image_server(folder, layer, area),
            }
            for layer in FUEL_LAYERS
        },
        "fuels_leakage_policy": (
            "vintage strictly precedes fire year; residual staleness corrected "
            "forward with MTBS + current-season NIFC perimeters"
        ),
        "gee_landfire_not_used_reason": (
            "Earth Engine catalog has no FBFM40 or canopy-cover asset; its LANDFIRE "
            "holdings are v1.2.0/v1.4.0 (~2014 conditions)"
        ),
    }
