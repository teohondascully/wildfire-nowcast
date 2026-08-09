"""WFIGS interagency fire perimeters — the fire CATALOG for the 2022-2025 extension.

GOFER hand-curated its 28 fires; an extension needs a source of record for
"which fires exist, how big, and when did they start". This module reads the
public NIFC **WFIGS Interagency Perimeters** ArcGIS service (no auth, no Cloud
project — the same "plain published endpoint" logic as ADR-003/ADR-005).

Two distinct jobs, deliberately kept apart:

1. :func:`large_fires` — CANDIDATE SELECTION. Which CA wildfires cleared GOFER's
   own inclusion rule (>50,000 acres / 202 km2, since GOES cannot resolve the
   progression of anything smaller) in a year range, with their discovery time
   and initial point.
2. :func:`final_perimeter` — the STRAY-REMOVAL reference. GOFER's
   ``Export_FireProgQA.js`` keeps only perimeter parts that intersect the FRAP
   ground-truth footprint for that incident. That step is not cosmetic and it is
   not optional **for G4 specifically**: without it, a second fire burning 30 km
   away in the same AOI enters the labels as a spectacular never-merging
   "spot event". The reference footprint is the only thing standing between a
   crossing-episode table and a table of neighbouring fires.

MTBS is NOT usable for either job here, and that is measured rather than
assumed: querying the GEE MTBS burned-area boundaries for California returns 48
records for 2021, 29 for 2022, 34 for 2023, **7 for 2024 and 0 for 2025** —
MTBS's assessment lag means the two most recent seasons are essentially
unmapped. It stays the right source for channel 13 on older fires and the wrong
one for anything recent.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.paths import interim_dir

__all__ = [
    "WFIGS_PERIMETERS_URL",
    "GOFER_MIN_ACRES",
    "WfigsFire",
    "large_fires",
    "final_perimeter",
    "nifc_provenance",
]

WFIGS_PERIMETERS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)

#: GOFER's stated inclusion rule: "large wildfires (over 50,000 acres or 202
#: sq. km)". Below this GOES's ~2 km footprint cannot resolve an hourly front.
GOFER_MIN_ACRES = 50_000.0

_TIMEOUT_S = 180


def _query(params: dict[str, str], *, attempts: int = 4) -> dict[str, Any]:
    """One WFIGS query, with backoff on the service's own 429.

    WFIGS meters in "request units per minute" and a full-geometry query for a
    large multipart perimeter costs thousands of them, so ten candidate fires in
    a loop trips the quota. Backing off is the documented remedy; retrying
    blindly is not.
    """
    url = WFIGS_PERIMETERS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "wildfire-nowcast/data"})
    last: dict[str, Any] = {}
    for attempt in range(attempts):
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            payload = json.load(resp)
        if "error" not in payload:
            return payload
        last = payload["error"]
        if int(last.get("code", 0)) != 429 or attempt == attempts - 1:
            break
        time.sleep(65)
    raise RuntimeError(f"WFIGS query failed: {last}")


def _cache_path(irwin_id: str, to_crs: str) -> Path:
    """Cache location for one reference perimeter.

    Under ``data/interim/`` deliberately: C-4 confines this task's writes there,
    and a cache that lands anywhere else is still a write to a frozen tree.
    """
    key = hashlib.sha256(f"{irwin_id}|{to_crs}".encode()).hexdigest()[:16]
    return interim_dir() / "_cache" / "wfigs" / f"{key}.geojson"


def _epoch_ms_to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)


@dataclass(frozen=True)
class WfigsFire:
    """One candidate fire from WFIGS."""

    name: str
    discovery_utc: datetime
    gis_acres: float
    lat: float
    lon: float
    irwin_id: str

    @property
    def year(self) -> int:
        return self.discovery_utc.year

    @property
    def area_km2(self) -> float:
        return self.gis_acres * 0.00404686


def large_fires(
    *,
    year_min: int,
    year_max: int,
    state: str = "US-CA",
    min_acres: float = GOFER_MIN_ACRES,
) -> list[WfigsFire]:
    """CA wildfires at or above GOFER's size rule, discovered in ``[year_min, year_max]``.

    Deduplicated on ``(name, discovery day)``: WFIGS carries several polygon
    revisions per incident and the query returns each one.
    """
    where = (
        f"attr_POOState='{state}' AND attr_IncidentTypeCategory='WF' "
        f"AND attr_FireDiscoveryDateTime >= TIMESTAMP '{year_min}-01-01 00:00:00' "
        f"AND attr_FireDiscoveryDateTime < TIMESTAMP '{year_max + 1}-01-01 00:00:00' "
        f"AND poly_GISAcres >= {min_acres}"
    )
    payload = _query(
        {
            "where": where,
            "outFields": ",".join(
                [
                    "attr_IncidentName",
                    "attr_FireDiscoveryDateTime",
                    "poly_GISAcres",
                    "attr_InitialLatitude",
                    "attr_InitialLongitude",
                    "attr_IrwinID",
                    "poly_IRWINID",
                ]
            ),
            "returnGeometry": "false",
            "orderByFields": "poly_GISAcres DESC",
            "resultRecordCount": "500",
            "f": "json",
        }
    )
    seen: set[tuple[str, str]] = set()
    out: list[WfigsFire] = []
    for feat in payload.get("features", []):
        attrs = feat["attributes"]
        disc = _epoch_ms_to_utc(attrs.get("attr_FireDiscoveryDateTime"))
        name = str(attrs.get("attr_IncidentName") or "").strip()
        if disc is None or not name:
            continue
        key = (name.upper(), disc.strftime("%Y-%m-%d"))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            WfigsFire(
                name=name,
                discovery_utc=disc,
                gis_acres=float(attrs.get("poly_GISAcres") or 0.0),
                lat=float(attrs.get("attr_InitialLatitude") or 0.0),
                lon=float(attrs.get("attr_InitialLongitude") or 0.0),
                # Two spellings exist in the same service and neither is always
                # populated; taking whichever is present avoids dropping a fire
                # for a schema quirk.
                irwin_id=str(
                    attrs.get("attr_IrwinID") or attrs.get("poly_IRWINID") or ""
                ),
            )
        )
    return sorted(out, key=lambda f: -f.gis_acres)


def final_perimeter(irwin_id: str, *, to_crs: str = "EPSG:5070") -> Any:
    """The LARGEST WFIGS polygon for one incident, as a shapely geometry.

    Largest rather than latest: WFIGS revisions include partial-day maps, and a
    stray-removal reference that is smaller than the fire would amputate the
    real perimeter. Taking the maximum-area revision is conservative in the
    direction that keeps real fire and drops neighbours.
    """
    import geopandas as gpd  # noqa: PLC0415
    from shapely.geometry import mapping, shape  # noqa: PLC0415

    cache = _cache_path(irwin_id, to_crs)
    if cache.exists():
        return shape(json.loads(cache.read_text()))

    payload = _query(
        {
            # Both spellings are queried: WFIGS populates `attr_IrwinID` on some
            # records and only `poly_IRWINID` on others, and matching the wrong
            # one returns an empty set that looks exactly like "no such fire".
            "where": f"attr_IrwinID='{irwin_id}' OR poly_IRWINID='{irwin_id}'",
            "outFields": "poly_GISAcres",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
    )
    geoms = [shape(f["geometry"]) for f in payload.get("features", []) if f.get("geometry")]
    if not geoms:
        raise KeyError(f"no WFIGS polygon for irwin id {irwin_id!r}")
    series = gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(to_crs)
    best = max(series, key=lambda g: g.area)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mapping(best)))
    return best


def nifc_provenance() -> dict[str, Any]:
    """C2 provenance fragment for the WFIGS catalog + stray-removal reference."""
    return {
        "wfigs_service": WFIGS_PERIMETERS_URL,
        "wfigs_role": (
            "fire catalog (candidate selection) and stray-removal reference, "
            "standing in for GOFER's FRAP/ICS-209 crosswalk (Export_FireProgQA.js)"
        ),
        "wfigs_pull_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
