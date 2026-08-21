"""C1 channels 1-4 + the inputs to channel 11: RTMA hourly weather.

Source: ``NOAA/NWS/RTMA`` (GEE), 2.5 km, hourly, from 2011-01-01. This is the
"truth weather" for training — no forecast, no reanalysis blend.

Two unit traps, both confirmed from the Earth Engine catalog entry and both
places where a silent error would be invisible in training:

* ``TMP`` and ``DPT`` are in **degrees Celsius**, while C1 channel 3 is **kelvin**.
* RTMA publishes **no relative humidity band**. C1 channel 4 (``rh_2m``, %) is
  derived from ``TMP`` and ``DPT``; the formula is pinned in :func:`rh_from_dewpoint`
  and recorded in the manifest so it is never re-derived differently elsewhere.

Resampling 2.5 km -> 1 km is an upsample. It is done bilinearly, which invents no
detail but keeps gradients continuous; the effective weather resolution stays
2.5 km and that is recorded in provenance rather than pretended away.

Retrieval is synchronous chunked ``computePixels`` (ADR-004), not a batch
export: our credentials hold only the ``earthengine,cloud-platform`` scope pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.data.sources.gee import (
    ExportConfig,
    export_image,
    fetch_bands_chunked,
    hourly_window,
    initialize_ee,
    region_for_grid,
)

__all__ = [
    "RTMA_COLLECTION",
    "RTMA_NATIVE_RES_M",
    "RTMA_BANDS",
    "C1_WEATHER_CHANNELS",
    "rh_from_dewpoint",
    "celsius_to_kelvin",
    "rtma_hourly_stack",
    "fetch_rtma",
    "export_rtma",
    "missing_hours",
]

RTMA_COLLECTION = "NOAA/NWS/RTMA"
RTMA_NATIVE_RES_M = 2500.0
RTMA_START = "2011-01-01T00:00:00"

#: RTMA band -> (C1 channel index, C1 name). SPFH/WIND/WDIR/GUST/TCDC/VIS/PRES/
#: ACPC01 are available but unused by C1 v1; GUST and ACPC01 are the two most
#: likely v2 additions (gusts drive spotting, precip terminates runs).
RTMA_BANDS: dict[str, tuple[int, str]] = {
    "UGRD": (1, "wind_u10"),
    "VGRD": (2, "wind_v10"),
    "TMP": (3, "temp_2m"),
    "DPT": (-1, "dewpoint_2m"),  # not a C1 channel; feeds rh_2m and the FM proxy
}
RTMA_EXTRA_BANDS = ("GUST", "ACPC01", "TCDC", "PRES", "SPFH")


def celsius_to_kelvin(t_c: Any) -> Any:
    """C -> K. Works on numpy arrays and on ``ee.Image``.

    ``ee.Image`` does not implement ``__add__``, so dispatch on ``.add``.
    """
    if hasattr(t_c, "add"):  # ee.Image
        return t_c.add(273.15)
    return t_c + 273.15


def rh_from_dewpoint(temp_c: Any, dewpoint_c: Any) -> Any:
    """Relative humidity (%) from temperature and dew point, both in Celsius.

    Magnus-Tetens over water with the Alduchov & Eskridge (1996) coefficients
    (a = 17.625, b = 243.04 C), which is accurate to <0.4% over -40..+50 C::

        RH = 100 * exp(a*Td/(b+Td) - a*T/(b+T))

    Pinned here because channel 4 and channel 11 must not disagree about what
    humidity means.
    """
    a, b = 17.625, 243.04
    if hasattr(temp_c, "expression"):  # ee.Image
        import ee  # noqa: PLC0415

        img = ee.Image(temp_c).rename("T").addBands(ee.Image(dewpoint_c).rename("Td"))
        return (
            img.expression(
                "100 * exp(a*Td/(b+Td) - a*T/(b+T))",
                {"a": a, "b": b, "T": img.select("T"), "Td": img.select("Td")},
            )
            .clamp(0, 100)
            .rename("rh_2m")
        )
    t = np.asarray(temp_c, dtype=np.float64)
    td = np.asarray(dewpoint_c, dtype=np.float64)
    rh = 100.0 * np.exp(a * td / (b + td) - a * t / (b + t))
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


@dataclass(frozen=True)
class RtmaRequest:
    fire_id: str
    grid: Grid
    start_utc: datetime
    end_utc: datetime


def rtma_hourly_stack(req: RtmaRequest) -> Any:
    """Build the per-hour multi-band ``ee.Image`` stack for one fire domain.

    Returns a single ``ee.Image`` whose bands are ``{var}_{hhh}`` for each hour
    offset, so the whole time series exports as one batch task instead of T
    tasks. Hour ordering is explicit and dense: a missing RTMA hour becomes an
    all-masked band, never a silently shifted one.
    """
    ee = initialize_ee()
    lo, hi = hourly_window(req.start_utc, req.end_utc, pad_h=1)
    region = region_for_grid(req.grid)

    coll = (
        ee.ImageCollection(RTMA_COLLECTION)
        .filterDate(lo.strftime("%Y-%m-%dT%H:%M:%S"), hi.strftime("%Y-%m-%dT%H:%M:%S"))
        .filterBounds(region)
    )

    hours = _hour_sequence(req.start_utc, req.end_utc)
    bands = []
    for i, t in enumerate(hours):
        t0 = ee.Date(t.strftime("%Y-%m-%dT%H:%M:%S"))
        # RTMA is stamped on the hour; take the single image whose start_time is t.
        img = ee.Image(coll.filterDate(t0, t0.advance(1, "hour")).first())
        img = ee.Image(ee.Algorithms.If(img, img, _masked_placeholder(ee)))
        u = img.select("UGRD").rename(f"wind_u10_{i:04d}")
        v = img.select("VGRD").rename(f"wind_v10_{i:04d}")
        tk = celsius_to_kelvin(img.select("TMP")).rename(f"temp_2m_{i:04d}")
        rh = rh_from_dewpoint(img.select("TMP"), img.select("DPT")).rename(f"rh_2m_{i:04d}")
        bands.extend([u, v, tk, rh])
    return ee.Image.cat(bands).resample("bilinear")


def _masked_placeholder(ee: Any) -> Any:
    """A fully masked RTMA-shaped image, for hours RTMA did not publish."""
    return (
        ee.Image.constant([0, 0, 0, 0])
        .rename(["UGRD", "VGRD", "TMP", "DPT"])
        .updateMask(ee.Image.constant(0))
        .toFloat()
    )


def _hour_sequence(start: datetime, end: datetime) -> list[datetime]:
    import pandas as pd  # noqa: PLC0415

    return list(pd.date_range(start, end, freq="1h", inclusive="both").to_pydatetime())


C1_WEATHER_CHANNELS = ("wind_u10", "wind_v10", "temp_2m", "rh_2m")


#: Hours per request. Bounded by graph depth, not by response size: one band is
#: an `ee.Image.cat` node, and serialising a few hundred of them overflows the
#: Python recursion limit inside `ee.serializer` before it ever reaches the API.
#: 24 h keeps the graph shallow and matches RTMA's daily publication rhythm.
FETCH_CHUNK_HOURS = 24


def fetch_rtma(req: RtmaRequest, *, chunk_hours: int = FETCH_CHUNK_HOURS) -> dict[str, np.ndarray]:
    """Pull one fire's weather block as ``{channel: (T, H, W) float32}``.

    Synchronous chunked fetch (ADR-004). Chunking is over **time**, and it is
    required for correctness, not just for size: a single ``ee.Image`` carrying
    every hour of a long fire is a deep enough expression graph to blow the
    recursion limit during client-side serialisation.

    Hours RTMA did not publish come back as NaN and are reported by
    :func:`missing_hours` rather than interpolated away.
    """
    hours = _hour_sequence(req.start_utc, req.end_utc)
    per_channel: dict[str, list[np.ndarray]] = {ch: [] for ch in C1_WEATHER_CHANNELS}

    for lo in range(0, len(hours), chunk_hours):
        block = hours[lo : lo + chunk_hours]
        sub = RtmaRequest(req.fire_id, req.grid, block[0], block[-1])
        image = rtma_hourly_stack(sub)
        names = [f"{ch}_{i:04d}" for i in range(len(block)) for ch in C1_WEATHER_CHANNELS]
        flat = fetch_bands_chunked(image, req.grid, names)
        for ch in C1_WEATHER_CHANNELS:
            per_channel[ch].extend(flat[f"{ch}_{i:04d}"] for i in range(len(block)))

    return {ch: np.stack(per_channel[ch]).astype(np.float32) for ch in C1_WEATHER_CHANNELS}


def export_rtma(req: RtmaRequest, config: ExportConfig | None = None) -> Any:
    """Batch-export variant. Gated by ADR-004; :func:`fetch_rtma` is the default."""
    return export_image(
        rtma_hourly_stack(req),
        name=f"{req.fire_id}__rtma",
        grid=req.grid,
        config=config,
        channels=list(C1_WEATHER_CHANNELS),
    )


def missing_hours(values: np.ndarray, times: Any) -> dict[str, Any]:
    """QA for weather completeness: which hours came back entirely NaN.

    "Weather coverage complete" is a per-fire QA criterion; RTMA does drop hours,
    and a silently interpolated gap would look like a calm hour to the model.
    """
    arr = np.asarray(values)
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[:, None]
    all_nan = np.isnan(flat).all(axis=1)
    any_nan = np.isnan(flat).any(axis=1)
    idx = np.nonzero(all_nan)[0]
    return {
        "n_hours": int(arr.shape[0]),
        "n_hours_fully_missing": int(all_nan.sum()),
        "n_hours_partially_missing": int((any_nan & ~all_nan).sum()),
        "missing_hour_indices": [int(i) for i in idx],
        "missing_hours_utc": [str(times[i]) for i in idx] if times is not None else [],
        "coverage_complete": bool(not any_nan.any()),
    }
