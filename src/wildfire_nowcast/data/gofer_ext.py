"""GOFER-Combined, reimplemented for fires GOFER never published (2022-2025).

**Read this first: these labels are NOT GOFER.** All five Zenodo versions of the
GOFER concept record cover the same 28 California fires, 2019-2021. Nothing
downstream may treat a fire built here as if it came out of ``GOFER.zip``, so
every artifact this module produces carries ``gofer_version =
"gofer-ext-<algo>-<date>"`` and a ``label_source`` of ``"gofer_ext"``. The
algorithm is a faithful port of the published one (github.com/tianjialiu/GOFER,
ESSD 2024) — the *code* is theirs, the *execution* is ours, and the difference
between those two is exactly what the fidelity measurement below exists to
quantify.

Pipeline, transcribed from ``Export_FireConf.js`` -> ``Export_ScaleVal.js`` ->
``Export_FireProg.js`` -> ``Export_FireProgQA.js`` -> ``6a - Export_cFireLine.js``::

    conf[h]        per-hour max GOES fire-detection confidence, per satellite
    conf[h]        <- pull-warped by the terrain-parallax field x 0.85
    raw_cum[h]     running max over hours 1..h            (UNSCALED)
    scale[h]       max over AOIbuf of boxcar_R(mean_sat(raw_cum[h]))   (=1 for h>=500)
    cum[h]         running max over hours 1..h of conf[j]/scale[j]
    smooth[h]      boxcar_R(mean_sat(cum[h]))
    perim[h]       smooth[h] > 0.95
    perim[h]       <- cumulative UNION over hours (monotone by construction)
    perim[h]       <- components intersecting the WFIGS reference footprint only
    fline[h]       boundary(perim[h]) AND boxcar_R(mean_sat(conf[h]/scale[h])) > fconf

Two operations are done locally in EPSG:5070 that GOFER does server-side in
EPSG:4326, and both are *improvements* rather than shortcuts:

* **Parallax.** GOFER computes the displacement in a geographic frame and then
  divides x by ``cos(lat)`` to work around ``Image.displace`` under-displacing
  in longitude. Working in 5070 removes the workaround and removes a genuine
  error: Albers is rotated 10-20 deg from true north over California, so an
  (east, north) offset applied as an (x, y) offset is wrong by that angle.
* **Cumulative max.** GOFER recomputes the running max server-side for every
  hour, which is O(T^2) work; the displacement is a static nearest-neighbour
  gather, so it commutes with ``max`` and the running max can be done once,
  locally, in O(T). Identical output, and it is what makes a 1,500-hour fire
  affordable under ADR-004's synchronous-fetch constraint.

The one approximation that is NOT free: GOFER smooths and thresholds on a 50 m
lattice before vectorising, and this module works on a 250 m lattice. That is a
1.4 % difference in the effective boxcar half-width. It is measured rather than
argued — see :func:`validate_against_published`, which scores this
implementation against the *published* perimeters on fires GOFER did publish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from wildfire_nowcast.common.contract import CELL_SIZE_M, CRS_STRING
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.states import FIRELINE_V2, StateRule, apply_state_rule, dilate
from wildfire_nowcast.data.gofer import DEFAULT_CFIRE_CONF
from wildfire_nowcast.data.labels import LabelBuild, fire_domain_grid
from wildfire_nowcast.data.qa import fire_qa_report
from wildfire_nowcast.data.rasterize import COVER_THRESHOLD
from wildfire_nowcast.data.sources.goes_fdc import (
    MASK_FIRE_MAX,
    goes_provenance,
    hourly_confidence,
    kernel_resolutions,
    parallax_offsets_5070,
    satellite_for,
)
from wildfire_nowcast.data.sources.nifc import WfigsFire, final_perimeter, nifc_provenance
from wildfire_nowcast.data.sources.terrain import fetch_terrain

__all__ = [
    "ALGO_VERSION",
    "GOFER_EXT_VERSION",
    "CONFIDENCE_CUTOFF",
    "PARALLAX_ADJ",
    "FINE_RES_M",
    "FireSpec",
    "GoferRun",
    "spec_from_wfigs",
    "detection_window",
    "run_gofer_ext",
    "to_label_build",
    "validate_against_published",
]

#: Bump when any numeric step below changes. It lands in every manifest.
ALGO_VERSION = "gofer-ext-1.0"
GOFER_EXT_VERSION = f"{ALGO_VERSION}-goes-fdcf"

#: GOFER-Combined constants (largeFires_metadata.js). NOT refitted here: they
#: were optimised in the published parameter-sensitivity step against FRAP/FEDS
#: for 2019-2021, and refitting them on fires we have no ground truth for would
#: be exactly the n=1 calibration C-3 forbids.
CONFIDENCE_CUTOFF = 0.95
PARALLAX_ADJ = 0.85
SAT_MODE = "C"

#: Analysis lattice for the algorithm: an exact 4x refinement of the C1 1 km
#: lattice, so block-reduction to C1 is exact and shares cell centres (C1.2).
FINE_RES_M = 250.0
_REFINE = int(round(CELL_SIZE_M / FINE_RES_M))

#: Hour index at and beyond which GOFER stops rescaling (Export_ScaleVal.js).
_SCALE_CUTOFF_HOUR = 500
#: A scale factor below this is treated as zero, which zeroes that hour.
_SCALE_FLOOR = 0.1
#: GOFER's AOIbuf: the AOI bounds buffered by this before taking the scale max.
_AOI_BUF_M = 5_000.0
#: Extra margin on the compute domain so the C1 10 km buffer always fits inside.
_COMPUTE_MARGIN_M = 12_000.0
#: Slack allowed when testing a perimeter component against the WFIGS reference.
_STRAY_TOLERANCE_M = 2_000.0
#: km2 per acre.
_ACRE_KM2 = 0.00404686


@dataclass
class FireSpec:
    """Everything the algorithm needs for one fire. The analogue of GOFER's
    hand-authored ``largeFires_metadata.js`` entry, but derived, not typed."""

    fire_id: str
    name: str
    start_utc: datetime
    n_hours: int
    aoi_bounds_5070: tuple[float, float, float, float]
    kernels: dict[str, int]
    irwin_id: str
    official_acres: float

    @property
    def kernel_m(self) -> int:
        return int(self.kernels["Combined" if SAT_MODE == "C" else SAT_MODE])


@dataclass
class GoferRun:
    """Output of the algorithm on the fine lattice, before C1 rasterisation."""

    spec: FireSpec
    fine: Grid
    times: pd.DatetimeIndex
    perimeter: np.ndarray  # (T, ny, nx) bool, monotone
    fire_line: np.ndarray  # (T, ny, nx) bool
    scale_vals: np.ndarray  # (T,) float
    provenance: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------


def _compute_grids(bounds: tuple[float, float, float, float]) -> tuple[Grid, Grid]:
    """``(coarse 1 km, fine 250 m)`` grids over ``bounds`` + the compute margin."""
    minx, miny, maxx, maxy = bounds
    coarse = Grid.from_bounds(
        (
            minx - _COMPUTE_MARGIN_M,
            miny - _COMPUTE_MARGIN_M,
            maxx + _COMPUTE_MARGIN_M,
            maxy + _COMPUTE_MARGIN_M,
        ),
        cell_size_m=CELL_SIZE_M,
        snap=True,
    )
    fine = Grid(
        x_min=coarse.x_min,
        y_max=coarse.y_max,
        nx=coarse.nx * _REFINE,
        ny=coarse.ny * _REFINE,
        cell_size_m=CELL_SIZE_M / _REFINE,
        crs=coarse.crs,
    )
    return coarse, fine


def _box1d(a: np.ndarray, half_cells: float) -> np.ndarray:
    """Exact 1-D box mean of half-width ``half_cells`` along axis 0, edge-replicated.

    FRACTIONAL on purpose. An integer-cell box quantises the kernel width to the
    lattice, and at a 250 m lattice a 1,702 m radius rounds to a 1,875 m
    half-width — 10 % too wide, which over-smooths and shrinks every perimeter
    by ~11 % (measured before this was fixed). Weighting the two partially
    covered cells by their overlap makes the effective half-width exactly
    ``radius``, which is what Earth Engine's 50 m evaluation delivers to within
    its own +-25 m quantisation.
    """
    n = max(int(np.floor(half_cells - 0.5)), 0)
    frac = max(half_cells - 0.5 - n, 0.0)
    pad = n + 1
    padded = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    csum = np.zeros((padded.shape[0] + 1, padded.shape[1]), dtype=np.float64)
    csum[1:] = padded.cumsum(axis=0)
    rows = a.shape[0]
    start = pad - n
    window = csum[start + 2 * n + 1 : start + 2 * n + 1 + rows] - csum[start : start + rows]
    left = padded[start - 1 : start - 1 + rows]
    right = padded[start + 2 * n + 1 : start + 2 * n + 1 + rows]
    return (window + frac * (left + right)) / (2.0 * half_cells)


def _boxcar(arr: np.ndarray, radius_m: float, res_m: float) -> np.ndarray:
    """``ee.Kernel.square(radius_m, 'meters', normalize=True)`` + mean reducer.

    Separable, pure numpy. **Deliberately not scipy**: this package has no scipy
    dependency, and adding one to the shared virtualenv while another lead is
    mid-training is exactly the shared-state change C-4 exists to stop.
    ``common.states.dilate`` already made the same call for the same reason.
    """
    half = max(float(radius_m) / float(res_m), 0.5 + 1e-9)
    work = np.asarray(arr, dtype=np.float64)
    work = _box1d(work, half)
    work = _box1d(work.T, half).T
    return work.astype(np.float32)


def _erode3(mask: np.ndarray) -> np.ndarray:
    """3x3 binary erosion with a zero border, numpy only."""
    out = np.asarray(mask, dtype=bool).copy()
    h, w = out.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(out)
            ys_dst = slice(max(dy, 0), h + min(dy, 0))
            ys_src = slice(max(-dy, 0), h + min(-dy, 0))
            xs_dst = slice(max(dx, 0), w + min(dx, 0))
            xs_src = slice(max(-dx, 0), w + min(-dx, 0))
            shifted[ys_dst, xs_dst] = mask[ys_src, xs_src]
            out &= shifted
    return out


def _reconstruct(seed: np.ndarray, mask: np.ndarray, *, max_iter: int = 10_000) -> np.ndarray:
    """Geodesic reconstruction: everything in ``mask`` 8-connected to ``seed``.

    Used instead of full connected-component labelling because the perimeter
    stack is MONOTONE, so hour ``h``'s result seeds hour ``h+1`` and each step
    only has to propagate one hour of growth. A per-hour BFS over the whole fine
    lattice would be ~10^7 Python-level cell visits per fire.
    """
    current = np.asarray(seed, dtype=bool) & mask
    for _ in range(max_iter):
        grown = dilate(current, 1) & mask
        if np.array_equal(grown, current):
            return current
        current = grown
    return current


def _pull_warp(arr: np.ndarray, dx: np.ndarray, dy: np.ndarray, grid: Grid) -> np.ndarray:
    """``out[p] = arr[p + d(p)]`` with nearest-neighbour sampling (``Image.displace``)."""
    ny, nx = arr.shape[-2:]
    rows, cols = np.indices((ny, nx))
    col_s = np.clip(np.rint(cols + dx / grid.cell_size_m).astype(np.int64), 0, nx - 1)
    row_s = np.clip(np.rint(rows - dy / grid.cell_size_m).astype(np.int64), 0, ny - 1)
    return arr[..., row_s, col_s]


def _block_fraction(fine_mask: np.ndarray, factor: int) -> np.ndarray:
    """Area fraction of each coarse cell covered, from a fine boolean mask."""
    t, ny, nx = fine_mask.shape
    return (
        fine_mask.astype(np.float32)
        .reshape(t, ny // factor, factor, nx // factor, factor)
        .mean(axis=(2, 4))
    )


def _block_any(fine_mask: np.ndarray, factor: int) -> np.ndarray:
    """``all_touched``-style reduction: a coarse cell is set if any sub-cell is."""
    t, ny, nx = fine_mask.shape
    return (
        fine_mask.reshape(t, ny // factor, factor, nx // factor, factor).any(axis=(2, 4))
    )


def _crop_to(arr: np.ndarray, src: Grid, dst: Grid) -> np.ndarray:
    """Crop a ``src``-gridded array to ``dst``. Both must share the lattice."""
    r0 = int(round((src.y_max - dst.y_max) / src.cell_size_m))
    c0 = int(round((dst.x_min - src.x_min) / src.cell_size_m))
    if r0 < 0 or c0 < 0 or r0 + dst.ny > src.ny or c0 + dst.nx > src.nx:
        raise ValueError(
            f"C1 domain {dst.shape} at ({r0},{c0}) does not fit inside the compute "
            f"domain {src.shape}; widen _COMPUTE_MARGIN_M rather than clipping"
        )
    return arr[..., r0 : r0 + dst.ny, c0 : c0 + dst.nx]


# ---------------------------------------------------------------------------
# staging: replaces GOFER's manual "0a - Calc_StagingAOI.js"
# ---------------------------------------------------------------------------


def detection_window(
    aoi_bounds_5070: tuple[float, float, float, float],
    start: datetime,
    *,
    max_days: int = 75,
    min_pixels: int = 1,
) -> int:
    """Hours from ``start`` to the last day with any GOES fire pixel in the AOI.

    GOFER sets ``nHour`` by "manually inspecting GOES active fire pixels and
    timeseries" (``Calc_StagingAOI.js``). This derives the same number: count
    fire-coded pixels per day over both satellites and take the last day above
    ``min_pixels``. Deriving it means a new fire is self-serve and the number is
    reproducible, which a hand-inspected one is not.
    """
    from wildfire_nowcast.data.sources.gee import initialize_ee  # noqa: PLC0415

    ee = initialize_ee()
    minx, miny, maxx, maxy = aoi_bounds_5070
    region = ee.Geometry.Rectangle(
        coords=[minx, miny, maxx, maxy], proj=CRS_STRING, geodesic=False
    )
    start_ee = ee.Date(start.strftime("%Y-%m-%dT%H:%M:%S"))
    colls = [
        ee.ImageCollection(satellite_for(start, side).collection_id).select("Mask")
        for side in ("East", "West")
    ]

    def _day(i_day: Any) -> Any:
        i_day = ee.Number(i_day)
        t0 = start_ee.advance(i_day, "day")
        t1 = t0.advance(1, "day")
        counts = [
            ee.Image(
                ee.Algorithms.If(
                    c.filterDate(t0, t1).size().gt(0),
                    c.filterDate(t0, t1).map(lambda im: im.lte(MASK_FIRE_MAX)).max(),
                    ee.Image(0),
                )
            ).unmask(0)
            for c in colls
        ]
        total = ee.Image(counts[0]).add(counts[1]).rename("n")
        n = total.reduceRegion(
            reducer=ee.Reducer.sum().unweighted(),
            geometry=region,
            scale=2000,
            maxPixels=int(1e9),
        ).get("n")
        return ee.Feature(None, {"day": i_day, "n": n})

    days = ee.FeatureCollection(ee.List.sequence(0, max_days - 1, 1).map(_day))
    rows = days.getInfo()["features"]
    counts = [(int(r["properties"]["day"]), float(r["properties"]["n"] or 0.0)) for r in rows]
    active = [d for d, n in counts if n >= min_pixels]
    if not active:
        raise RuntimeError(
            f"no GOES fire pixels in the AOI within {max_days} days of {start:%Y-%m-%d %HZ}"
        )
    return int((max(active) + 1) * 24)


def next_overlapping_fire_hours(
    fire: WfigsFire, reference: Any, start: datetime, *, buffer_m: float = 15_000.0
) -> int | None:
    """Hours from ``start`` until the NEXT large fire starts on the same ground.

    **Found the hard way.** `2025_madre` (SLO, 2025-07-02) and `2025_gifford`
    (2025-08-01) ignite **3 km apart**. Scanning Madre for 64 days runs the
    detection window straight through Gifford, and stray removal cannot save
    you: it keeps components that TOUCH the reference footprint, and an adjacent
    fire touches. Madre's derived area came out at **726.5 km² against a mapped
    326.9 km²** — 2.2x, i.e. two fires in one label set. The `_crop_to` guard
    caught it (the C1 domain no longer fit the compute domain) rather than a
    contaminated tensor shipping.

    GOFER solves this by hand — `AOIsmall` / `AOIsmallTS` and a manually chosen
    `end` per fire. This derives the same bound: the scan stops when another
    WFIGS large fire whose footprint comes within ``buffer_m`` of ours is
    discovered. Returns ``None`` when no such fire exists.
    """
    from wildfire_nowcast.data.sources.nifc import large_fires  # noqa: PLC0415

    zone = reference.buffer(buffer_m)
    soonest: datetime | None = None
    for other in large_fires(year_min=start.year, year_max=start.year):
        if other.irwin_id == fire.irwin_id or other.discovery_utc <= fire.discovery_utc:
            continue
        try:
            other_geom = final_perimeter(other.irwin_id)
        except (KeyError, RuntimeError):
            continue
        if zone.intersects(other_geom) and (soonest is None or other.discovery_utc < soonest):
            soonest = other.discovery_utc
    if soonest is None:
        return None
    return max(1, int((soonest - start).total_seconds() // 3600))


def spec_from_wfigs(fire: WfigsFire, *, max_days: int = 75) -> FireSpec:
    """Derive a :class:`FireSpec` from one WFIGS record. No hand-typed numbers."""
    reference = final_perimeter(fire.irwin_id)
    bounds = tuple(float(v) for v in reference.bounds)
    start = fire.discovery_utc.replace(minute=0, second=0, microsecond=0)
    n_hours = detection_window(bounds, start, max_days=max_days)  # type: ignore[arg-type]
    neighbour_cap = next_overlapping_fire_hours(fire, reference, start)
    if neighbour_cap is not None and neighbour_cap < n_hours:
        n_hours = neighbour_cap
    from wildfire_nowcast.data.sources.gee import initialize_ee  # noqa: PLC0415

    ee = initialize_ee()
    minx, miny, maxx, maxy = bounds  # type: ignore[misc]
    aoi_ll = ee.Geometry.Rectangle(
        coords=[minx, miny, maxx, maxy], proj=CRS_STRING, geodesic=False
    ).transform("EPSG:4326", 1.0)
    kernels = kernel_resolutions(aoi_ll, start)
    slug = "".join(ch if ch.isalnum() else "_" for ch in fire.name.lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return FireSpec(
        fire_id=f"{start.year}_{slug}",
        name=fire.name,
        start_utc=start,
        n_hours=n_hours,
        aoi_bounds_5070=bounds,  # type: ignore[arg-type]
        kernels=kernels,
        irwin_id=fire.irwin_id,
        official_acres=fire.gis_acres,
    )


# ---------------------------------------------------------------------------
# the algorithm
# ---------------------------------------------------------------------------


def run_gofer_ext(
    spec: FireSpec,
    *,
    fine_res_m: float = FINE_RES_M,
    cfire_conf: float = DEFAULT_CFIRE_CONF,
    reference_geom: Any | None = None,
    wfigs_fire: WfigsFire | None = None,
    verbose: bool = True,
) -> GoferRun:
    """Run GOFER-Combined for one fire and return fine-lattice masks."""
    coarse, fine = _compute_grids(spec.aoi_bounds_5070)
    if abs(fine.cell_size_m - fine_res_m) > 1e-9:
        raise ValueError(
            f"fine_res_m={fine_res_m} is not an exact refinement of {CELL_SIZE_M} m"
        )

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    _log(
        f"[{spec.fire_id}] compute grid {fine.shape} @ {fine.cell_size_m:.0f} m, "
        f"{spec.n_hours} h, kernel {spec.kernel_m} m"
    )

    elevation_coarse = fetch_terrain(coarse)["elevation"]
    elevation = np.repeat(np.repeat(elevation_coarse, _REFINE, axis=0), _REFINE, axis=1)

    conf: dict[str, np.ndarray] = {}
    for side in ("East", "West"):
        raw = hourly_confidence(fine, spec.start_utc, spec.n_hours, side)
        dx, dy = parallax_offsets_5070(fine, elevation, satellite_for(spec.start_utc, side).lon0)
        dx = _boxcar(dx, spec.kernel_m, fine.cell_size_m) * PARALLAX_ADJ
        dy = _boxcar(dy, spec.kernel_m, fine.cell_size_m) * PARALLAX_ADJ
        conf[side] = _pull_warp(raw, dx, dy, fine)
        _log(f"[{spec.fire_id}] {side}: confidence fetched + parallax-corrected")

    # -- scale factors (Export_ScaleVal.js): max over AOIbuf of the smoothed,
    #    UNSCALED cumulative confidence. Early hours have low absolute
    #    confidence, so without this no early perimeter clears the 0.95 cutoff.
    aoi_slice = _aoi_window(fine, spec.aoi_bounds_5070, _AOI_BUF_M)
    scale_vals = np.ones(spec.n_hours, dtype=np.float64)
    run_e = np.zeros(fine.shape, dtype=np.float32)
    run_w = np.zeros(fine.shape, dtype=np.float32)
    for h in range(spec.n_hours):
        if h + 1 >= _SCALE_CUTOFF_HOUR:
            scale_vals[h:] = 1.0
            break
        np.maximum(run_e, conf["East"][h], out=run_e)
        np.maximum(run_w, conf["West"][h], out=run_w)
        smoothed = _boxcar(0.5 * (run_e + run_w), spec.kernel_m, fine.cell_size_m)
        scale_vals[h] = float(smoothed[aoi_slice].max())
    scale_vals[scale_vals < _SCALE_FLOOR] = 0.0
    del run_e, run_w

    # -- perimeters + fire lines --------------------------------------------
    perim = np.zeros((spec.n_hours, fine.ny, fine.nx), dtype=bool)
    fline = np.zeros_like(perim)
    cum_e = np.zeros(fine.shape, dtype=np.float32)
    cum_w = np.zeros(fine.shape, dtype=np.float32)
    for h in range(spec.n_hours):
        scale = scale_vals[h]
        # EE returns 0 for x/0 (verified live), which is what zeroes an
        # under-confident hour rather than sending it to +inf.
        inv = 0.0 if scale == 0.0 else 1.0 / scale
        e_h = conf["East"][h] * inv
        w_h = conf["West"][h] * inv
        np.maximum(cum_e, e_h, out=cum_e)
        np.maximum(cum_w, w_h, out=cum_w)
        cum = _boxcar(0.5 * (cum_e + cum_w), spec.kernel_m, fine.cell_size_m)
        perim[h] = cum > CONFIDENCE_CUTOFF
        hourly = _boxcar(0.5 * (e_h + w_h), spec.kernel_m, fine.cell_size_m)
        fline[h] = hourly > cfire_conf

    # -- QA post-processing (Export_FireProgQA.js + eePro_FireProg.R) --------
    np.logical_or.accumulate(perim, axis=0, out=perim)
    reference = reference_geom if reference_geom is not None else final_perimeter(spec.irwin_id)
    foreign = _neighbour_exclusion(fine, reference, wfigs_fire)
    if foreign.any():
        perim &= ~foreign[None, :, :]
        _log(f'[{spec.fire_id}] neighbour exclusion removed {int(foreign.sum())} fine cells '
             'mapped to another incident and outside our own footprint')
    perim = _remove_stray(perim, fine, reference)
    fline &= perim

    keep = _trim_to_growth(perim)
    perim, fline = perim[keep], fline[keep]
    times = pd.DatetimeIndex(
        [spec.start_utc.replace(tzinfo=None) + timedelta(hours=int(i + 1)) for i in keep]
    )
    final_km2 = float(perim[-1].sum()) * (fine.cell_size_m**2) / 1e6
    _log(
        f"[{spec.fire_id}] {len(times)} h kept of {spec.n_hours}; final area "
        f"{final_km2:.1f} km2 (WFIGS {spec.official_acres * _ACRE_KM2:.1f} km2)"
    )

    provenance = {
        "label_source": "gofer_ext",
        "gofer_ext_version": GOFER_EXT_VERSION,
        "gofer_ext_algorithm": (
            "port of github.com/tianjialiu/GOFER (ESSD 2024, doi:10.5194/essd-16-1395-2024) "
            "GOFER-Combined; perimeters computed here, NOT downloaded from Zenodo 14638647"
        ),
        "gofer_ext_sat_mode": SAT_MODE,
        "gofer_ext_confidence_cutoff": CONFIDENCE_CUTOFF,
        "gofer_ext_parallax_adj": PARALLAX_ADJ,
        "gofer_ext_kernel_m": spec.kernel_m,
        "gofer_ext_kernels": dict(spec.kernels),
        "gofer_ext_fine_res_m": fine.cell_size_m,
        "gofer_ext_start_utc": spec.start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gofer_ext_n_hours_scanned": spec.n_hours,
        "gofer_ext_cfire_conf": cfire_conf,
        # Carried so channel 13 can exclude THIS fire from the WFIGS gap-fill by
        # id, not only by date and name. ADR-008's lesson generalises: a strict
        # date inequality is not a self-exclusion.
        "gofer_ext_irwin_id": spec.irwin_id,
        **goes_provenance(spec.start_utc),
        **nifc_provenance(),
    }
    diagnostics = {
        "scale_vals_first24": [round(float(v), 4) for v in scale_vals[:24]],
        "hours_scanned": spec.n_hours,
        "hours_kept": int(len(times)),
        "first_kept_hour": int(keep[0] + 1) if len(keep) else None,
        "final_area_km2": round(float(perim[-1].sum()) * (fine.cell_size_m**2) / 1e6, 2),
        "wfigs_acres": spec.official_acres,
    }
    return GoferRun(
        spec=spec,
        fine=fine,
        times=times,
        perimeter=perim,
        fire_line=fline,
        scale_vals=scale_vals,
        provenance=provenance,
        diagnostics=diagnostics,
    )


def _aoi_window(
    grid: Grid, bounds: tuple[float, float, float, float], buffer_m: float
) -> tuple[slice, slice]:
    minx, miny, maxx, maxy = bounds
    c0 = int(np.clip((minx - buffer_m - grid.x_min) / grid.cell_size_m, 0, grid.nx - 1))
    c1 = int(np.clip((maxx + buffer_m - grid.x_min) / grid.cell_size_m, 1, grid.nx))
    r0 = int(np.clip((grid.y_max - (maxy + buffer_m)) / grid.cell_size_m, 0, grid.ny - 1))
    r1 = int(np.clip((grid.y_max - (miny - buffer_m)) / grid.cell_size_m, 1, grid.ny))
    return slice(r0, r1), slice(c0, c1)


def _neighbour_exclusion(grid: Grid, reference: Any, fire: WfigsFire | None) -> np.ndarray:
    """Cells mapped to a DIFFERENT incident and outside our own footprint.

    The companion to the temporal cap. The cap stops the scan before the *next*
    fire starts; this removes ground belonging to a fire that started *earlier*
    and was still burning when ours began — which the cap cannot reach, because
    it is already inside hour 1's cumulative maximum.

    Only cells OUTSIDE our own WFIGS reference are ever removed, so a genuine
    overlap between two mapped perimeters can never carve into our fire. Ground
    that another incident owns and ours does not is not ours.
    """
    from rasterio.features import rasterize as _rio_rasterize  # noqa: PLC0415

    from wildfire_nowcast.data.sources.nifc import large_fires  # noqa: PLC0415

    if fire is None:
        return np.zeros(grid.shape, dtype=bool)
    others = []
    for other in large_fires(year_min=fire.discovery_utc.year, year_max=fire.discovery_utc.year):
        if other.irwin_id == fire.irwin_id:
            continue
        try:
            geom = final_perimeter(other.irwin_id)
        except (KeyError, RuntimeError):
            continue
        if geom.intersects(reference.buffer(_COMPUTE_MARGIN_M)):
            others.append(geom)
    if not others:
        return np.zeros(grid.shape, dtype=bool)
    foreign = _rio_rasterize(
        [(g, 1) for g in others], out_shape=grid.shape,
        transform=grid.rasterio_transform(), fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    ours = _rio_rasterize(
        [(reference, 1)], out_shape=grid.shape,
        transform=grid.rasterio_transform(), fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    return foreign & ~dilate(ours, 2)


def _remove_stray(perim: np.ndarray, grid: Grid, reference: Any) -> np.ndarray:
    """Drop connected components that do not touch the WFIGS reference footprint.

    This is GOFER's ``removeStray`` with FRAP swapped for WFIGS, and it is the
    single most consequential QA step for G4: without it, a *different* fire
    burning in the same AOI enters this fire's labels as a never-merging body
    tens of kilometres away, i.e. as a textbook long-range spot event that never
    happened. Since the whole point of the 2022-2025 extension is to raise the
    spot-candidate count above n=2, a false positive here is worse than a miss.
    """
    from rasterio.features import rasterize as _rio_rasterize  # noqa: PLC0415

    ref = _rio_rasterize(
        [(reference, 1)],
        out_shape=grid.shape,
        transform=grid.rasterio_transform(),
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    pad = max(1, int(round(_STRAY_TOLERANCE_M / grid.cell_size_m)))
    ref = dilate(ref, pad)

    out = np.zeros_like(perim)
    previous = np.zeros(perim.shape[1:], dtype=bool)
    for h in range(perim.shape[0]):
        if not perim[h].any():
            continue
        seed = (perim[h] & ref) | previous
        if not seed.any():
            continue
        previous = _reconstruct(seed, perim[h])
        out[h] = previous
    return out


def _trim_to_growth(perim: np.ndarray) -> np.ndarray:
    """Indices to keep: first non-empty hour .. last hour that grew.

    ``eePro_FireProg.R`` cuts "extraneous timesteps after fire stops growing";
    ``Export_FireProg.js`` drops zero-area timesteps at the front. Both matter
    for us: a trailing run of bitwise-identical frames is 100 % zero-growth
    hours, which is R5's persistence attractor handed to the model for free.
    """
    area = perim.reshape(perim.shape[0], -1).sum(axis=1)
    nonzero = np.flatnonzero(area > 0)
    if nonzero.size == 0:
        raise RuntimeError("no non-empty perimeter in the whole run")
    first = int(nonzero[0])
    grew = np.flatnonzero(np.diff(area) > 0)
    last = int(grew[-1] + 1) if grew.size else int(nonzero[-1])
    return np.arange(first, max(last, first) + 1)


# ---------------------------------------------------------------------------
# C1 handoff
# ---------------------------------------------------------------------------


def to_label_build(
    run: GoferRun,
    *,
    rule: StateRule = FIRELINE_V2,
    cover_threshold: float = COVER_THRESHOLD,
) -> LabelBuild:
    """Rasterise a run onto the C1 1 km lattice and apply the C0 state rule.

    The state rule itself is NOT reimplemented here: ``apply_state_rule`` in
    ``common/`` is the single implementation (C0), and it is handed exactly the
    two mask stacks C1.1 names.
    """
    fine = run.fine
    final_bounds = _mask_bounds(run.perimeter[-1], fine)
    grid = fire_domain_grid(final_bounds)

    coarse_frac = _block_fraction(run.perimeter, _REFINE)
    coarse_grid = Grid(
        x_min=fine.x_min,
        y_max=fine.y_max,
        nx=fine.nx // _REFINE,
        ny=fine.ny // _REFINE,
        cell_size_m=fine.cell_size_m * _REFINE,
        crs=fine.crs,
    )
    perim_masks = _crop_to(coarse_frac >= cover_threshold, coarse_grid, grid)
    # C1.1's active(t) is a *line* term: `all_touched` semantics, not area.
    line_masks = _crop_to(_block_any(run.fire_line, _REFINE), coarse_grid, grid)
    # A line only means anything on the boundary of the perimeter; interior
    # cells that happen to be hot are already burning by construction.
    line_masks &= _boundary(perim_masks)

    state = apply_state_rule(perim_masks, rule=rule, fire_line_masks=line_masks)

    perims = _perimeter_frame(perim_masks, grid, run.times)
    provenance: dict[str, Any] = {
        **run.provenance,
        "state_rule": rule,
        "state_rule_status": "RATIFIED — INTERFACES C1.1 (ADR-006 P1)",
        "state_rule_impl": (
            "wildfire_nowcast.common.states.apply_state_rule (C0: one implementation)"
        ),
        "grid_buffer_m": 10_000.0,
        "polygon_rasterization": {
            "method": "area_fraction_from_250m_lattice",
            "oversample_factor": _REFINE,
            "cover_threshold": cover_threshold,
        },
        "gofer_fname": run.spec.name,
        "gofer_fyear": run.spec.start_utc.year,
        "acres_official": run.spec.official_acres,
        "goes_ignition_utc": run.spec.start_utc.strftime("%Y-%m-%d %H"),
        "cfire_conf": run.provenance.get("gofer_ext_cfire_conf"),
        "cfireline_timesteps_missing": int((~line_masks.any(axis=(1, 2))).sum()),
        "gofer_ext_diagnostics": run.diagnostics,
    }
    qa = fire_qa_report(
        fire_id=run.spec.fire_id,
        perims=perims,
        state=state,
        grid=grid,
        extra={"state_rule": rule, "label_source": "gofer_ext"},
    )
    return LabelBuild(
        fire_id=run.spec.fire_id,
        grid=grid,
        times=run.times,
        state=state,
        qa=qa,
        provenance=provenance,
    )


def _boundary(masks: np.ndarray) -> np.ndarray:
    """Cells inside the mask that touch its complement (8-connected)."""
    out = np.zeros_like(masks)
    for h in range(masks.shape[0]):
        if masks[h].any():
            out[h] = masks[h] & ~_erode3(masks[h])
    return out


def _mask_bounds(mask: np.ndarray, grid: Grid) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise RuntimeError("final perimeter is empty")
    x0 = grid.x_min + cols.min() * grid.cell_size_m
    x1 = grid.x_min + (cols.max() + 1) * grid.cell_size_m
    y1 = grid.y_max - rows.min() * grid.cell_size_m
    y0 = grid.y_max - (rows.max() + 1) * grid.cell_size_m
    return (x0, y0, x1, y1)


def _perimeter_frame(masks: np.ndarray, grid: Grid, times: pd.DatetimeIndex) -> Any:
    """Vectorise the C1-grid masks so the existing vector QA applies unchanged."""
    import geopandas as gpd  # noqa: PLC0415
    from rasterio.features import shapes as _rio_shapes  # noqa: PLC0415
    from shapely.geometry import shape as _shape  # noqa: PLC0415
    from shapely.ops import unary_union  # noqa: PLC0415

    geoms = []
    for h in range(masks.shape[0]):
        parts = [
            _shape(geom)
            for geom, value in _rio_shapes(
                masks[h].astype(np.uint8), mask=masks[h], transform=grid.rasterio_transform()
            )
            if value == 1
        ]
        geoms.append(unary_union(parts) if parts else unary_union([]))
    return gpd.GeoDataFrame(
        {"timestep": np.arange(1, masks.shape[0] + 1), "tUTC": times},
        geometry=geoms,
        crs=grid.crs,
    )


# ---------------------------------------------------------------------------
# fidelity: score this port against the PUBLISHED product
# ---------------------------------------------------------------------------


def validate_against_published(
    fire_id: str,
    *,
    fine_res_m: float = FINE_RES_M,
    archive: Any | None = None,
    max_hours: int | None = None,
) -> dict[str, Any]:
    """Run this port on a fire GOFER DID publish and score it against Zenodo.

    This is the only thing that entitles anyone to put a 2022-2025 fire and a
    2019-2021 fire in the same table. Returns per-hour IoU on the C1 lattice,
    the final-area ratio, and the centroid offset — the same quantities used to
    quantify GOFER-East vs GOFER-West label noise, so the port's error can be
    read directly against the product's own known noise floor.
    """
    from wildfire_nowcast.data.gofer import GoferArchive  # noqa: PLC0415
    from wildfire_nowcast.data.labels import build_fire_state  # noqa: PLC0415

    arch = archive or GoferArchive()
    published = build_fire_state(fire_id, archive=arch, with_east_west_noise=False)
    meta = arch.lookup(fire_id)
    perims = arch.perimeters(fire_id, to_crs=CRS_STRING)
    start = pd.to_datetime(meta.GOESIg_UTC).to_pydatetime().replace(tzinfo=UTC)
    n_hours = int(perims["timestep"].max())
    if max_hours is not None:
        n_hours = min(n_hours, max_hours)

    ee_geom = _aoi_geometry(tuple(perims.total_bounds))  # type: ignore[arg-type]
    spec = FireSpec(
        fire_id=fire_id,
        name=str(meta.fname),
        start_utc=start,
        n_hours=n_hours,
        aoi_bounds_5070=tuple(float(v) for v in perims.total_bounds),  # type: ignore[arg-type]
        kernels=kernel_resolutions(ee_geom, start),
        irwin_id="",
        official_acres=float(meta.acres_official),
    )
    reference = perims.geometry.iloc[-1]
    run = run_gofer_ext(spec, fine_res_m=fine_res_m, reference_geom=reference, verbose=False)
    ours = to_label_build(run)

    return _score_against(published, ours, spec)


def _aoi_geometry(bounds: tuple[float, float, float, float]) -> Any:
    from wildfire_nowcast.data.sources.gee import initialize_ee  # noqa: PLC0415

    ee = initialize_ee()
    minx, miny, maxx, maxy = bounds
    return ee.Geometry.Rectangle(
        coords=[minx, miny, maxx, maxy], proj=CRS_STRING, geodesic=False
    ).transform("EPSG:4326", 1.0)


def _score_against(published: LabelBuild, ours: LabelBuild, spec: FireSpec) -> dict[str, Any]:
    """IoU / area / centroid agreement between two label builds of one fire."""
    grid = published.grid
    pub = published.state != 0
    mine = _reproject_mask(ours.state != 0, ours.grid, grid)

    shared = min(len(published.times), len(ours.times))
    ious, area_ratio, centroid_km = [], [], []
    for h in range(shared):
        a, b = pub[h], mine[h]
        union = int((a | b).sum())
        ious.append(float((a & b).sum()) / union if union else np.nan)
        area_ratio.append(float(b.sum()) / float(a.sum()) if a.sum() else np.nan)
        if a.any() and b.any():
            ra, ca = np.nonzero(a)
            rb, cb = np.nonzero(b)
            centroid_km.append(
                float(
                    np.hypot(
                        (cb.mean() - ca.mean()) * grid.cell_size_m,
                        (rb.mean() - ra.mean()) * grid.cell_size_m,
                    )
                    / 1000.0
                )
            )

    cell_km2 = (grid.cell_size_m**2) / 1e6
    return {
        "fire_id": spec.fire_id,
        "algo_version": ALGO_VERSION,
        "kernel_m": spec.kernel_m,
        "n_hours_published": int(len(published.times)),
        "n_hours_ours": int(len(ours.times)),
        "n_hours_scored": int(shared),
        "iou_mean": round(float(np.nanmean(ious)), 4),
        "iou_median": round(float(np.nanmedian(ious)), 4),
        "iou_p10": round(float(np.nanpercentile(ious, 10)), 4),
        "iou_final_hour": round(float(ious[shared - 1]), 4) if shared else None,
        "area_ratio_median": round(float(np.nanmedian(area_ratio)), 4),
        "final_area_published_km2": round(float(pub[shared - 1].sum()) * cell_km2, 2),
        "final_area_ours_km2": round(float(mine[shared - 1].sum()) * cell_km2, 2),
        "centroid_offset_km_median": round(float(np.median(centroid_km)), 3)
        if centroid_km
        else None,
        "centroid_offset_km_p90": round(float(np.percentile(centroid_km, 90)), 3)
        if centroid_km
        else None,
    }


def _reproject_mask(masks: np.ndarray, src: Grid, dst: Grid) -> np.ndarray:
    """Move a boolean stack between two grids on the same 1 km lattice."""
    out = np.zeros((masks.shape[0], dst.ny, dst.nx), dtype=bool)
    r0 = int(round((src.y_max - dst.y_max) / dst.cell_size_m))
    c0 = int(round((dst.x_min - src.x_min) / dst.cell_size_m))
    cs = np.arange(dst.nx) + c0
    col_ok = (cs >= 0) & (cs < src.nx)
    for r in range(dst.ny):
        sr = r + r0
        if not 0 <= sr < src.ny:
            continue
        out[:, r, col_ok] = masks[:, sr, cs[col_ok]]
    return out


def dump_run(run: GoferRun, path: Any) -> None:
    """Write the run's provenance + diagnostics beside its interim artifacts."""
    from pathlib import Path  # noqa: PLC0415

    Path(path).write_text(
        json.dumps({"provenance": run.provenance, "diagnostics": run.diagnostics}, indent=2)
        + "\n"
    )
