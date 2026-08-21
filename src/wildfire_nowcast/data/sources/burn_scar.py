"""C1 channel 13: ``recent_burn_scar`` {0,1} — the mandatory LANDFIRE correction.

LANDFIRE lags reality, and :mod:`.fuels` deliberately makes that lag *worse* by
refusing any vintage published after the fire (leakage). This channel closes the
gap: everything that burned between the fuels vintage and the fire's own
ignition is flagged, so the model can learn that recently burned ground does not
carry fire again.

Window: ``(vintage_year, ignition_time)``. Two sources, because neither alone
covers it:

* **MTBS** ``USFS/GTAC/MTBS/burned_area_boundaries/v1`` — authoritative, but only
  published ~1-2 years in arrears, so it cannot cover the fire's own season.
* **NIFC current-season perimeters** — the public NIFC/WFIGS ArcGIS service,
  no auth, which covers the season in progress.
  Important scoping point for the 2019-2021 training set: the WFIGS service is
  *YearToDate*, i.e. the CURRENT season, so it is useless for historical fires.
  It is retained for real-time inference only. For the training fires, MTBS
  (which by 2026 covers 2019-2024) already includes the within-season scars that
  NIFC would have supplied, so the historical path is MTBS-only and complete.

The exclusion that matters: a burn scar from **the fire itself** must never be
included. The window is closed strictly at ignition time.

MTBS is fetched synchronously via ``computePixels`` (ADR-004). The NIFC half
needs no Earth Engine at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = [
    "MTBS_ASSET",
    "NIFC_PERIMETER_SERVICE",
    "SCAR_LOOKBACK_YEARS",
    "SELF_EXCLUSION_GUARD_DAYS",
    "DNBR_THRESHOLD",
    "scar_window",
    "mtbs_scar_image",
    "mtbs_coverage_end",
    "wfigs_scar_mask",
    "dnbr_scar_mask",
    "fetch_burn_scar",
    "nifc_query_url",
    "burn_scar_provenance",
]

MTBS_ASSET = "USFS/GTAC/MTBS/burned_area_boundaries/v1"
#: WFIGS interagency perimeters, public ArcGIS FeatureServer, no authentication.
NIFC_PERIMETER_SERVICE = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0/query"
)
#: How far back a scar still suppresses spread. Chaparral recovers to carrying
#: fire in roughly 5-10 years; 6 is a compromise and is a config knob, not a law.
SCAR_LOOKBACK_YEARS = 6

#: Guard window before ignition in which an MTBS event is NOT treated as a prior
#: scar. Two independent reasons, and the first one bit us for real:
#:
#: 1. MTBS ``Ig_Date`` is **day-resolution and independently sourced**, so a
#:    fire's own record can be stamped *earlier* than the hourly GOFER ignition.
#:    Kincade: MTBS says 2019-10-23 07:00Z, GOFER says 2019-10-24 04:00Z. A naive
#:    ``Ig_Date < ignition`` filter therefore admitted Kincade's own 77,780-acre
#:    scar into channel 13 — i.e. handed the model the answer.
#: 2. A fire that started days before ours is probably still burning, so its
#:    final MTBS perimeter contains ground that burns *after* our ignition. That
#:    is leakage too, just less obvious.
SELF_EXCLUSION_GUARD_DAYS = 14


def _epoch_ms(when: datetime) -> int:
    """MTBS stores ``Ig_Date`` as epoch **milliseconds**, not as a date string."""
    aware = when if when.tzinfo else when.replace(tzinfo=UTC)
    return int(aware.timestamp() * 1000)


def scar_window(ignition_utc: datetime, vintage_year: int) -> tuple[datetime, datetime]:
    """``[start, end)`` of MTBS ignition dates that count as *prior* scars."""
    start_year = max(vintage_year, ignition_utc.year - SCAR_LOOKBACK_YEARS)
    return (
        datetime(start_year, 1, 1, tzinfo=UTC),
        ignition_utc - timedelta(days=SELF_EXCLUSION_GUARD_DAYS),
    )


def mtbs_scar_image(
    grid: Any,
    vintage_year: int,
    ignition_utc: datetime,
    *,
    fire_name: str | None = None,
) -> Any:
    """Binary MTBS scar mask of fires that burned *strictly before* this one.

    Self-exclusion is belt and braces — a guard window (see
    :data:`SELF_EXCLUSION_GUARD_DAYS`) *and* an incident-name match — because
    the date filter alone provably fails on day-resolution MTBS ignition dates.
    """
    from wildfire_nowcast.common.contract import CELL_SIZE_M  # noqa: PLC0415
    from wildfire_nowcast.data.sources.gee import initialize_ee, region_for_grid  # noqa: PLC0415

    ee = initialize_ee()
    region = region_for_grid(grid)
    start, end = scar_window(ignition_utc, vintage_year)
    scars = (
        ee.FeatureCollection(MTBS_ASSET)
        .filterBounds(region)
        .filter(ee.Filter.gte("Ig_Date", _epoch_ms(start)))
        .filter(ee.Filter.lt("Ig_Date", _epoch_ms(end)))
    )
    if fire_name:
        # second line of defence: drop this fire's own record whatever its date
        token = str(fire_name).split()[0].upper()
        scars = scars.filter(ee.Filter.stringContains("Incid_Name", token).Not())
    crs_transform = [CELL_SIZE_M, 0, grid.x_min, 0, -CELL_SIZE_M, grid.y_max]
    return (
        ee.Image(0)
        .byte()
        .paint(scars, 1)
        .reproject(crs=grid.crs, crsTransform=crs_transform)
        .rename("recent_burn_scar")
        .unmask(0)
        .toUint8()
        .clip(region)
    )


#: dNBR at or above this counts as a burn scar. 0.27 is the USGS/MTBS
#: "low / moderate-low" severity break — the lowest cut that is a burn rather
#: than noise. It is a CONFIG KNOB and it is calibrated against MTBS on the
#: years where both exist (see :func:`dnbr_vs_mtbs`), never asserted.
DNBR_THRESHOLD = 0.27

#: WFIGS all-years perimeters. NOT the YearToDate service the docstring above
#: warns about — this one carries every season and is published within weeks of
#: containment, which is exactly the window MTBS cannot reach.
WFIGS_ALL_YEARS = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)


#: Region over which MTBS completeness is judged. Every fire in this project is
#: Californian; judging completeness on a single fire's 50 km domain would be
#: noise, not a coverage measurement.
_CA_BBOX_4326 = (-124.6, 32.4, -114.0, 42.1)
#: A year counts as ASSESSED if its record count reaches this fraction of the
#: median of the five years before it. Fitted on nothing — it is a presence
#: test, not a threshold on an outcome — and the observed separation is 5x
#: (2024: 7 records vs a 2019-2023 median of ~34), so no value in [0.25, 0.75]
#: changes the verdict.
_MTBS_COMPLETENESS_FRACTION = 0.5


def mtbs_year_counts(year_min: int, year_max: int) -> dict[int, int]:
    """California MTBS records per ignition year. The raw evidence for the gap."""
    from wildfire_nowcast.data.sources.gee import initialize_ee  # noqa: PLC0415

    ee = initialize_ee()
    region = ee.Geometry.Rectangle(list(_CA_BBOX_4326))
    coll = ee.FeatureCollection(MTBS_ASSET).filterBounds(region)
    out: dict[int, int] = {}
    for year in range(year_min, year_max + 1):
        lo = _epoch_ms(datetime(year, 1, 1, tzinfo=UTC))
        hi = _epoch_ms(datetime(year + 1, 1, 1, tzinfo=UTC))
        out[year] = int(
            coll.filter(ee.Filter.gte("Ig_Date", lo))
            .filter(ee.Filter.lt("Ig_Date", hi))
            .size()
            .getInfo()
        )
    return out


def mtbs_coverage_end(reference_year: int) -> tuple[datetime, dict[str, Any]]:
    """End of the last MTBS year that is actually ASSESSED. DERIVED, not assumed.

    **The naive version of this function is a trap and I nearly shipped it.**
    ``aggregate_max("Ig_Date")`` returns 2024-12-17, which reads as "MTBS covers
    2024" — but California has **7** records in 2024 against 48 / 29 / 34 in
    2021 / 2022 / 2023, and **0** in 2025. A single early-released record makes
    a year look covered. Presence of a maximum date is not coverage, exactly as
    a finite value is not a plausible one (R11).

    So completeness is measured by record COUNT against the preceding five
    years, and the returned datetime is the end of the last year that passes.
    """
    counts = mtbs_year_counts(reference_year - 8, reference_year)
    assessed = reference_year - 8
    for year in sorted(counts):
        prior = [counts[y] for y in range(year - 5, year) if y in counts]
        if not prior:
            continue
        import statistics  # noqa: PLC0415

        bar = _MTBS_COMPLETENESS_FRACTION * statistics.median(prior)
        if counts[year] >= bar:
            assessed = year
    return datetime(assessed + 1, 1, 1, tzinfo=UTC), {
        "ca_records_per_year": counts,
        "last_assessed_year": assessed,
        "completeness_rule": (
            f"a year is ASSESSED when its CA record count reaches "
            f"{_MTBS_COMPLETENESS_FRACTION} x the median of the five years before "
            "it; aggregate_max(Ig_Date) is NOT coverage — one early record makes "
            "an unassessed year look complete"
        ),
    }


def _wfigs_query(params: dict[str, str], *, attempts: int = 4) -> dict[str, Any]:
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    url = WFIGS_ALL_YEARS + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "wildfire-nowcast/data"})
    last: dict[str, Any] = {}
    for attempt in range(attempts):
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            payload = json.load(resp)
        if "error" not in payload:
            return payload
        last = payload["error"]
        if int(last.get("code", 0)) != 429 or attempt == attempts - 1:
            break
        time.sleep(65)
    raise RuntimeError(f"WFIGS scar query failed: {last}")


def wfigs_scar_mask(
    grid: Any,
    start: datetime,
    end: datetime,
    *,
    fire_name: str | None = None,
    exclude_irwin: str | None = None,
) -> Any:
    """``(H, W)`` {0,1} scar mask from WFIGS final perimeters in ``[start, end)``.

    The substitute MTBS cannot be: WFIGS publishes an incident's final perimeter
    within weeks of containment rather than years after it, so it covers exactly
    the one-to-two seasons before a recent fire that MTBS has not assessed.

    Self-exclusion carries the SAME two-part guard as the MTBS path — the caller
    closes the window at ``ignition - guard`` and this also drops the fire's own
    IRWIN id and name token. ADR-008's lesson (a strict date inequality is not a
    self-exclusion) does not stop applying because the source changed.
    """
    import numpy as np  # noqa: PLC0415
    from rasterio.features import rasterize as _rio_rasterize  # noqa: PLC0415

    minx, miny, maxx, maxy = grid.bounds
    where = (
        "attr_IncidentTypeCategory='WF' "
        f"AND attr_FireDiscoveryDateTime >= TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}' "
        f"AND attr_FireDiscoveryDateTime < TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}'"
    )
    payload = _wfigs_query(
        {
            "where": where,
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "5070",
            "outSR": "5070",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "attr_IncidentName,attr_IrwinID,poly_IRWINID",
            "returnGeometry": "true",
            "f": "geojson",
        }
    )
    from shapely.geometry import shape  # noqa: PLC0415

    token = str(fire_name).split()[0].upper() if fire_name else None
    geoms = []
    for feat in payload.get("features", []):
        props = feat.get("properties", {}) or {}
        name = str(props.get("attr_IncidentName") or "").upper()
        ids = {str(props.get("attr_IrwinID") or ""), str(props.get("poly_IRWINID") or "")}
        if exclude_irwin and exclude_irwin in ids:
            continue
        if token and token in name:
            continue
        if feat.get("geometry"):
            geoms.append(shape(feat["geometry"]))
    if not geoms:
        return np.zeros(grid.shape, dtype=np.float32)
    return _rio_rasterize(
        [(g, 1) for g in geoms],
        out_shape=grid.shape,
        transform=grid.rasterio_transform(),
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(np.float32)


def _nbr(collection: Any, region: Any, start: str, end: str) -> Any:
    """Cloud-masked median NBR over ``[start, end)``. Sentinel-2 SR harmonised."""
    from wildfire_nowcast.data.sources.gee import initialize_ee  # noqa: PLC0415

    ee = initialize_ee()

    def _mask(img: Any) -> Any:
        scl = img.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(clear)

    coll = (
        ee.ImageCollection(collection)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(_mask)
    )
    return coll.median().normalizedDifference(["B8", "B12"]).rename("nbr"), coll.size()


def dnbr_scar_mask(
    grid: Any,
    start: datetime,
    end: datetime,
    *,
    threshold: float = DNBR_THRESHOLD,
    pre_days: int = 365,
    post_days: int = 120,
) -> tuple[Any, dict[str, Any]]:
    """``(H, W)`` {0,1} scar mask measured from Sentinel-2 dNBR, plus diagnostics.

    ``dNBR = NBR(before the gap) - NBR(just before OUR ignition)``, thresholded.
    Scoped to the MTBS GAP only, never to the whole lookback: over a six-year
    baseline dNBR stops measuring fire and starts measuring logging, drought
    mortality and phenology. Over a one-to-two-season gap it is a burn detector.

    **BOTH WINDOWS END BEFORE OUR FIRE IGNITES, and the first version of this
    function did not.** Taking the post-composite *after* ``end`` looks natural
    — you want imagery after the scar formed — and it walks straight through our
    own fire, so dNBR then measures OUR burn and hands the model the answer.
    Measured on 2024_bridge before the fix: 188 flagged cells against WFIGS's
    14, IoU 0.025. That is ADR-008's Kincade self-scar leak reproduced in a new
    source, and it was caught only because an INDEPENDENT source was scored
    against it rather than trusted.
    """
    import numpy as np  # noqa: PLC0415

    from wildfire_nowcast.data.sources.gee import (  # noqa: PLC0415
        fetch_bands_chunked,
        region_for_grid,
    )

    region = region_for_grid(grid)
    pre_start = (start - timedelta(days=pre_days)).strftime("%Y-%m-%d")
    pre_nbr, n_pre = _nbr(
        "COPERNICUS/S2_SR_HARMONIZED", region, pre_start, start.strftime("%Y-%m-%d")
    )
    post_start = (end - timedelta(days=post_days)).strftime("%Y-%m-%d")
    post_nbr, n_post = _nbr(
        "COPERNICUS/S2_SR_HARMONIZED", region, post_start, end.strftime("%Y-%m-%d")
    )
    post_end = end.strftime("%Y-%m-%d")
    counts = {"n_pre_images": int(n_pre.getInfo()), "n_post_images": int(n_post.getInfo())}
    if counts["n_pre_images"] == 0 or counts["n_post_images"] == 0:
        return np.zeros(grid.shape, dtype=np.float32), {
            **counts,
            "usable": False,
            "reason": "no cloud-clear Sentinel-2 imagery on one side of the window",
        }
    dnbr = pre_nbr.subtract(post_nbr).rename("dnbr")
    arr = fetch_bands_chunked(dnbr, grid, ["dnbr"])["dnbr"]
    finite = np.isfinite(arr)
    mask = (finite & (arr >= threshold)).astype(np.float32)
    return mask, {
        **counts,
        "usable": True,
        "threshold": threshold,
        "coverage_fraction": round(float(finite.mean()), 4),
        "scar_fraction": round(float(mask.mean()), 4),
        "pre_window": [pre_start, start.strftime("%Y-%m-%d")],
        "post_window": [post_start, post_end],
        "both_windows_precede_ignition": True,
    }


def fetch_burn_scar(
    grid: Any,
    vintage_year: int,
    ignition_utc: datetime,
    *,
    fire_name: str | None = None,
    exclude_irwin: str | None = None,
    report: dict[str, Any] | None = None,
) -> Any:
    """``(H, W)`` {0,1} scar mask, MTBS plus a gap-fill for what MTBS has not
    assessed yet (ADR-004 synchronous fetch).

    THE CHANNEL IS BUILT TWO DIFFERENT WAYS ACROSS THE CORPUS, so it says which
    way, per fire, in ``report`` -> manifest provenance. A channel that cannot
    name its own source is the two-numbers-for-one-fact hazard C2 already
    rejects, pointed at a raster instead of a scalar.
    """
    import numpy as np  # noqa: PLC0415

    from wildfire_nowcast.data.sources.gee import fetch_bands_chunked  # noqa: PLC0415

    start, end = scar_window(ignition_utc, vintage_year)
    img = mtbs_scar_image(grid, vintage_year, ignition_utc, fire_name=fire_name)
    mtbs = fetch_bands_chunked(img, grid, ["recent_burn_scar"])["recent_burn_scar"]
    out = (mtbs > 0).astype(np.float32)
    detail: dict[str, Any] = {
        "sources_used": ["mtbs"],
        "mtbs_cells": int((mtbs > 0).sum()),
        "window_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    gap_start, coverage = mtbs_coverage_end(ignition_utc.year)
    detail["mtbs_coverage"] = coverage
    detail["mtbs_coverage_end_utc"] = gap_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    if gap_start < end:
        gap_start = max(gap_start, start)
        detail["gap_window_utc"] = [
            gap_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ]
        wfigs = wfigs_scar_mask(
            grid, gap_start, end, fire_name=fire_name, exclude_irwin=exclude_irwin
        )
        detail["wfigs_cells"] = int((wfigs > 0).sum())
        detail["sources_used"].append("wfigs_all_years")
        dnbr, dnbr_info = dnbr_scar_mask(grid, gap_start, end)
        detail["dnbr"] = dnbr_info
        detail["dnbr_cells"] = int((dnbr > 0).sum())
        if dnbr_info.get("usable"):
            detail["sources_used"].append("sentinel2_dnbr")
            # Agreement between the two substitutes, on the SAME window. This is
            # the only place we can see whether the dNBR cut is trustworthy on a
            # fire where MTBS has nothing to check it against.
            inter = int(((wfigs > 0) & (dnbr > 0)).sum())
            union = int(((wfigs > 0) | (dnbr > 0)).sum())
            detail["wfigs_dnbr_iou"] = round(inter / union, 4) if union else None
            detail["dnbr_recall_of_wfigs"] = (
                round(inter / int((wfigs > 0).sum()), 4) if (wfigs > 0).any() else None
            )
        out = np.maximum(out, np.maximum(wfigs, dnbr))
    else:
        detail["gap_window_utc"] = None
    detail["final_cells"] = int((out > 0).sum())
    detail["final_fraction"] = round(float((out > 0).mean()), 4)
    if report is not None:
        report.update(detail)
    return out


def nifc_query_url(bbox_5070: tuple[float, float, float, float], ignition_utc: datetime) -> str:
    """No-auth NIFC/WFIGS query for perimeters that existed before ignition."""
    minx, miny, maxx, maxy = bbox_5070
    epoch_ms = int(ignition_utc.timestamp() * 1000)
    return (
        f"{NIFC_PERIMETER_SERVICE}?f=geojson&outFields=*"
        f"&geometry={minx},{miny},{maxx},{maxy}"
        f"&geometryType=esriGeometryEnvelope&inSR=5070&outSR=5070"
        f"&spatialRel=esriSpatialRelIntersects"
        f"&where=attr_FireDiscoveryDateTime<{epoch_ms}"
    )


def burn_scar_provenance(vintage_year: int, ignition_utc: datetime) -> dict[str, Any]:
    start, end = scar_window(ignition_utc, vintage_year)
    return {
        "burn_scar_sources": [MTBS_ASSET],
        "burn_scar_window_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "burn_scar_window_end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "burn_scar_lookback_years": SCAR_LOOKBACK_YEARS,
        "burn_scar_self_exclusion_guard_days": SELF_EXCLUSION_GUARD_DAYS,
        "burn_scar_self_exclusion": (
            "guard window before ignition PLUS incident-name match; a bare "
            "Ig_Date < ignition filter leaks the fire's own scar because MTBS "
            "ignition dates are day-resolution and independently sourced"
        ),
        "burn_scar_nifc_not_used_reason": (
            "WFIGS YearToDate service is current-season only; for 2019-2021 fires "
            "MTBS already covers the within-season scars. For 2024+ fires the "
            "ALL-YEARS WFIGS service plus Sentinel-2 dNBR fill the MTBS gap — see "
            "provenance.qa.burn_scar_detail.sources_used, which is per fire"
        ),
        "burn_scar_gap_fill_sources": [WFIGS_ALL_YEARS, "COPERNICUS/S2_SR_HARMONIZED"],
        "burn_scar_dnbr_threshold": DNBR_THRESHOLD,
    }
