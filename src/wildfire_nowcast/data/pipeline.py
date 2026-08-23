"""End-to-end per-fire build: GOFER + GEE + LFPS -> a contract-true C1 v2 tensor.

One function, :func:`build_fire_tensor`, walks all 14 channels and hands a
complete :class:`~wildfire_nowcast.data.assemble.ChannelBundle` to the writer.
The writer is the only thing that may touch ``data/fires/`` and it refuses
anything short of 14 channels (ADR-003b).

The two conventions that are easy to violate and expensive to debug:

* **C1.3 time base.** GOFER ``tUTC`` is the *end* of the hour it describes, so
  the weather for step ``t`` is the RTMA hour at ``tUTC - 1h``. :func:`weather_hours`
  is the single place that lag is applied.
* **C1.2 lattice.** Every source is pulled with the fire's exact affine pinned,
  never with a bare ``scale``, so no channel is resampled after retrieval and
  all 14 share cell centres exactly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from wildfire_nowcast.common.derive import dead_fuel_moisture_simard
from wildfire_nowcast.common.states import FIRELINE_V2
from wildfire_nowcast.data.assemble import ChannelBundle
from wildfire_nowcast.data.audit import audit_channels
from wildfire_nowcast.data.gofer import GOFER_VERSION, GoferArchive
from wildfire_nowcast.data.ignitions import count_ignition_components
from wildfire_nowcast.data.labels import LabelBuild, build_fire_state
from wildfire_nowcast.data.sources.barriers import barrier_provenance, fetch_barriers
from wildfire_nowcast.data.sources.burn_scar import burn_scar_provenance, fetch_burn_scar
from wildfire_nowcast.data.sources.fuels import (
    CANOPY_COVER,
    FBFM40,
    FBFM40_NONBURNABLE,
    LANDFIRE_VINTAGES,
    NODATA_FILL,
    fetch_lfps_layer,
    fuel_provenance,
    nodata_report,
    vintage_for_fire,
)
from wildfire_nowcast.data.sources.rtma import (
    C1_WEATHER_CHANNELS,
    RtmaRequest,
    fetch_rtma,
    missing_hours,
)
from wildfire_nowcast.data.sources.terrain import fetch_terrain, terrain_provenance

__all__ = ["weather_hours", "BuildTimings", "build_fire_tensor"]

#: ADR-103/106. Module scope, `__name__`, and nothing else: no handler, no level,
#: no format. A build that is slow in the label stage is now made chatty in the
#: label stage alone (`--log-levels wildfire_nowcast.data.pipeline=DEBUG`), which
#: is the capability the `_log(msg)` shim this replaced could not express.
logger = logging.getLogger(__name__)

#: C1.3 - GOFER tUTC is end-of-hour, so weather is lagged by exactly this much.
WEATHER_LAG = timedelta(hours=1)

#: C1 channel 11 asks for the formula to be documented; the implementation is
#: `common.derive.dead_fuel_moisture_simard` (C0 - one implementation).
FUEL_MOISTURE_FORMULA = (
    "Simard (1968) equilibrium moisture content of fine dead fuels, as used by NFDRS 1-h "
    "timelag fuels and the Fosberg index; inputs are RTMA TMP + RH (RH derived from RTMA "
    "TMP/DPT via Alduchov-Eskridge Magnus). Implementation: "
    "wildfire_nowcast.common.derive.dead_fuel_moisture_simard, clipped to [1, 60] %."
)


def weather_hours(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map perimeter timestamps to the RTMA hours that drove them (C1.3).

    The perimeter stamped ``05:00`` describes 04:00 -> 05:00, so its weather is
    the RTMA analysis at 04:00. Applying this in one place is the only defence
    against training every fire an hour out of phase.
    """
    return times - WEATHER_LAG


@dataclass
class BuildTimings:
    """Wall-clock per stage - the input to any honest ETA."""

    stages: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        self.stages[name] = round(seconds, 2)

    @property
    def total_s(self) -> float:
        return round(sum(self.stages.values()), 2)


def build_fire_tensor(
    fire_id: str,
    *,
    archive: GoferArchive | None = None,
    cv_fold: int | None = None,
    spatial_block_id: int | None = None,
    label_build: LabelBuild | None = None,
) -> tuple[ChannelBundle, BuildTimings]:
    """Assemble all 14 C1 channels for one fire. Does not write anything.

    ``label_build`` may come either from :func:`~wildfire_nowcast.data.labels.
    build_fire_state` (published GOFER, 2019-2021) or from
    :func:`~wildfire_nowcast.data.gofer_ext.to_label_build` (the 2022-2025
    extension). Everything downstream of channel 0 reads the label build's
    PROVENANCE rather than the GOFER archive, so the two sources travel the
    identical code path from here on - which is the only way a 2022 fire and a
    2019 fire can end up in the same table without a second implementation
    quietly diverging.
    """
    timings = BuildTimings()

    # -- channel 0: labels (GOFER, no auth) -----------------------------------
    t0 = time.time()
    if label_build is None:
        archive = archive or GoferArchive()
        label_build = build_fire_state(fire_id, archive=archive, rule=FIRELINE_V2)
    labels = label_build
    timings.record("labels", time.time() - t0)
    grid, times = labels.grid, labels.times
    logger.info(
        "[%s] labels %s grid %s (%ss)",
        fire_id,
        labels.state.shape,
        grid.shape,
        timings.stages["labels"],
    )

    # The label build already recorded these when it rasterised the perimeters,
    # with byte-identical values to the archive lookup this replaced. Reading
    # them from ONE place is what lets published-GOFER fires (2019-2021) and
    # gofer_ext fires (2022-2025) share this code path instead of forking it.
    fire_name = str(labels.provenance["gofer_fname"])
    fire_year = int(labels.provenance["gofer_fyear"])
    ignition_utc = datetime.strptime(
        str(labels.provenance["goes_ignition_utc"]), "%Y-%m-%d %H"
    ).replace(tzinfo=UTC)
    label_version = str(
        labels.provenance.get("gofer_version")
        or labels.provenance.get("gofer_ext_version")
        or GOFER_VERSION
    )

    bundle = ChannelBundle(
        fire_id=fire_id,
        grid=grid,
        times=times,
        gofer_version=label_version,
        ignition_time_utc=ignition_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        cv_fold=cv_fold,
        spatial_block_id=spatial_block_id,
    )
    bundle.add("fire_state", labels.state)

    # -- channels 1-4: RTMA weather, lagged one hour (C1.3) -------------------
    t0 = time.time()
    w_hours = weather_hours(times)
    weather = fetch_rtma(
        RtmaRequest(fire_id, grid, w_hours[0].to_pydatetime(), w_hours[-1].to_pydatetime())
    )
    timings.record("rtma", time.time() - t0)
    for ch in C1_WEATHER_CHANNELS:
        arr = weather[ch]
        if arr.shape[0] != len(times):
            raise RuntimeError(f"{ch}: got {arr.shape[0]} hours, expected {len(times)}")
        bundle.add(ch, arr)
    weather_qa = {ch: missing_hours(weather[ch], w_hours) for ch in C1_WEATHER_CHANNELS}
    logger.info("[%s] rtma %s (%ss)", fire_id, weather["temp_2m"].shape, timings.stages["rtma"])

    # -- channels 5-8: 3DEP terrain -------------------------------------------
    t0 = time.time()
    terrain = fetch_terrain(grid)
    timings.record("terrain", time.time() - t0)
    for ch in ("elevation", "slope", "aspect_sin", "aspect_cos"):
        bundle.add(ch, terrain[ch])
    logger.info("[%s] terrain (%ss)", fire_id, timings.stages["terrain"])

    # -- channels 9-10: LANDFIRE fuels via LFPS, no auth (ADR-005) ------------
    t0 = time.time()
    folder = vintage_for_fire(fire_year)
    fbfm_nd: dict[str, Any] = {}
    canopy_nd: dict[str, Any] = {}
    fbfm = fetch_lfps_layer(folder, FBFM40, grid, nodata_out=fbfm_nd)
    canopy = fetch_lfps_layer(folder, CANOPY_COVER, grid, nodata_out=canopy_nd)
    timings.record("fuels", time.time() - t0)
    bundle.add("fuel_model_id", fbfm)
    bundle.add("canopy_cover", canopy)
    logger.info("[%s] fuels %s (%ss)", fire_id, folder, timings.stages["fuels"])

    # -- channel 11: derived dead fuel moisture -------------------------------
    t0 = time.time()
    fm = dead_fuel_moisture_simard(weather["temp_2m"], weather["rh_2m"])
    timings.record("moisture", time.time() - t0)
    bundle.add("fuel_moisture_proxy", fm)

    # -- channel 12: water / barrier mask -------------------------------------
    t0 = time.time()
    barriers = fetch_barriers(grid)
    timings.record("barriers", time.time() - t0)
    bundle.add("water_barrier_mask", barriers)
    logger.info("[%s] barriers (%ss)", fire_id, timings.stages["barriers"])

    # -- channel 13: recent burn scar -----------------------------------------
    t0 = time.time()
    scar_report: dict[str, Any] = {}
    scar = fetch_burn_scar(
        grid,
        LANDFIRE_VINTAGES[folder],
        ignition_utc,
        fire_name=fire_name,
        exclude_irwin=labels.provenance.get("gofer_ext_irwin_id") or None,
        report=scar_report,
    )
    timings.record("burn_scar", time.time() - t0)
    bundle.add("recent_burn_scar", scar)
    logger.info("[%s] burn_scar (%ss)", fire_id, timings.stages["burn_scar"])

    # -- provenance + QA ------------------------------------------------------
    bundle.provenance = {
        **labels.provenance,
        **terrain_provenance(),
        **fuel_provenance(fire_year),
        **barrier_provenance(),
        **burn_scar_provenance(LANDFIRE_VINTAGES[folder], ignition_utc),
        "rtma_collection": "NOAA/NWS/RTMA",
        "rtma_native_res_m": 2500.0,
        "rtma_resampling": "bilinear upsample 2.5 km -> 1 km (effective res stays 2.5 km)",
        "time_convention": "end_of_hour",
        "weather_lag_h": int(WEATHER_LAG.total_seconds() // 3600),
        "fuel_moisture_formula": FUEL_MOISTURE_FORMULA,
        "gee_retrieval": "ee.data.computePixels, synchronous chunked (ADR-004)",
        "build_timings_s": dict(timings.stages),
        "built_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if spatial_block_id is not None:
        bundle.provenance["spatial_block_id"] = int(spatial_block_id)

    # -- C2 [v2.7] keys (ADR-014). Derived here, never defaulted. --------------
    bundle.fuel_vintage_lag_years = fire_year - int(LANDFIRE_VINTAGES[folder])
    ignition_report = count_ignition_components(labels.state, cell_size_m=grid.cell_size_m)
    bundle.n_ignition_components = ignition_report.n_ignition_components
    bundle.provenance["ignition_components"] = ignition_report.to_provenance()
    if ignition_report.n_ignition_components > 1:
        # WARNING, not INFO, and the change is deliberate. R16: `2020_july_complex`
        # and SCU are several fires filed under one GOFER id, and the 47 km apparent
        # teleport that follows is a filing artefact. Under the `_log` shim this
        # line was suppressed by `--quiet` along with the routine stage timings;
        # at WARNING it reaches stderr even in a program that configured nothing,
        # via `logging.lastResort`.
        logger.warning(
            "[%s] MULTI-IGNITION: %d components filed under one GOFER id; see manifest "
            "provenance.ignition_components",
            fire_id,
            ignition_report.n_ignition_components,
        )

    bundle.provenance["fuels_nodata_policy"] = (
        f"LFPS NoData (-9999, off the LANDFIRE coastline) -> {NODATA_FILL}; "
        "FBFM40 98 is NB8 open water, an existing class"
    )

    bundle.qa = {
        **labels.qa,
        "weather": weather_qa,
        "static": _static_qa(bundle, fbfm, canopy, barriers, scar, terrain),
        "fuels_nodata": {
            "fuel_model_id": nodata_report(fbfm_nd["mask"], FBFM40, barriers),
            "canopy_cover": nodata_report(canopy_nd["mask"], CANOPY_COVER, barriers),
        },
        # R11: the contract checks structure, not plausibility. This runs on every
        # build so a new region's sentinel is found by machine, not by a human
        # happening to read a mean (which is how CZU's -9999 was caught).
        "physical_audit": audit_channels(bundle.channels, fire_id=fire_id),
        # Channel 13 is built from MTBS for older fires and from MTBS + WFIGS
        # all-years + Sentinel-2 dNBR where MTBS has not assessed the season
        # yet. A channel built two ways across a corpus must say which way, per
        # fire, or it is C2's two-numbers-for-one-fact hazard aimed at a raster.
        "burn_scar_detail": scar_report,
    }
    if bundle.qa["physical_audit"]["verdict"] != "ok":
        # WARNING for the same reason: R11's class. CZU shipped a `-9999` LFPS
        # sentinel past 42 structural checks, and this is the line that says so.
        logger.warning(
            "[%s] PHYSICAL AUDIT: suspect channels %s; see manifest provenance.qa.physical_audit",
            fire_id,
            bundle.qa["physical_audit"]["suspect_channels"],
        )
    return bundle, timings


def _static_qa(
    bundle: ChannelBundle,
    fbfm: np.ndarray,
    canopy: np.ndarray,
    barriers: np.ndarray,
    scar: np.ndarray,
    terrain: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Sanity checks on the static channels, plus the barrier/fuel cross-check."""
    nonburn = np.isin(fbfm.astype(int), sorted(FBFM40_NONBURNABLE))
    barrier = barriers > 0
    ever_burned = (bundle.channels["fire_state"] != 0).any(axis=0)
    return {
        "fuel_classes_present": sorted(int(v) for v in np.unique(fbfm)),
        "fuel_nonburnable_fraction": round(float(nonburn.mean()), 4),
        "canopy_cover_pct_mean": round(float(np.nanmean(canopy)), 2),
        "barrier_cell_fraction": round(float(barrier.mean()), 4),
        # channels 9 and 12 are deliberately separate; agreement is informative
        "barrier_nonburnable_agreement": round(
            float((barrier & nonburn).sum() / max(barrier.sum(), 1)), 4
        ),
        "burn_scar_cell_fraction": round(float((scar > 0).mean()), 4),
        "burn_scar_overlapping_this_fire_frac": round(
            float(((scar > 0) & ever_burned).sum() / max(ever_burned.sum(), 1)), 4
        ),
        "elevation_m_range": [
            round(float(terrain["elevation"].min()), 1),
            round(float(terrain["elevation"].max()), 1),
        ],
        "slope_deg_max": round(float(terrain["slope"].max()), 2),
        "aspect_unit_circle_max_err": round(
            float(np.max(np.abs(terrain["aspect_sin"] ** 2 + terrain["aspect_cos"] ** 2 - 1))), 5
        ),
        "burned_cells_on_barrier": int((ever_burned & barrier).sum()),
    }
