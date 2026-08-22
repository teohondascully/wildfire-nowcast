"""Per-fire QA. The output is a JSON-serialisable dict that goes verbatim into
the C2 manifest under ``provenance.qa``.

Design rule: QA **measures**, it does not fix. Where the pipeline does repair
something (the cumulative-OR that enforces monotone burned area), QA reports how
much repair was needed, so label noise stays visible instead of being absorbed.

Three families of check:

1. *Vector-level* - what GOFER itself says: hour gaps, area monotonicity,
   perimeter nesting, geometry validity, growth burstiness.
2. *Raster-level* - what survived onto the C1 grid: monotone burned-cell count,
   illegal state transitions, teleporting (new burned area disconnected from the
   existing fire), frames with no burning cell.
3. *Label-noise* - GOFER East vs West disagreement, which is a direct
   observation of viewing-geometry/parallax error, in kilometres.
"""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from wildfire_nowcast.common.contract import CRS_STRING
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.states import BURNED_OUT, BURNING, UNBURNED, dilate
from wildfire_nowcast.data.rasterize import polygon_mask

__all__ = ["vector_qa", "raster_qa", "east_west_noise", "fire_qa_report"]


def _f(x: Any) -> float | None:
    """JSON-safe float."""
    if x is None:
        return None
    v = float(x)
    return None if math.isnan(v) or math.isinf(v) else round(v, 6)


def vector_qa(perims: gpd.GeoDataFrame) -> dict[str, Any]:
    """Checks on the raw GOFER perimeter series (any CRS; areas use EPSG:5070)."""
    g = perims.sort_values("timestep").reset_index(drop=True)
    t = pd.to_datetime(g["tUTC"])
    gaps_h = (t.diff().dropna().dt.total_seconds() / 3600.0).to_numpy()

    metric = g.to_crs(CRS_STRING) if str(g.crs).upper() != CRS_STRING else g
    area_km2 = metric.geometry.area.to_numpy() / 1e6
    d_area = np.diff(area_km2) if len(area_km2) > 1 else np.array([0.0])

    # nesting: area present at t-1 that is absent at t ("un-burning")
    geoms = list(metric.geometry)
    unburn = (
        np.array([geoms[i - 1].difference(geoms[i]).area / 1e6 for i in range(1, len(geoms))])
        if len(geoms) > 1
        else np.array([0.0])
    )

    cent = metric.geometry.centroid
    jump_km = (
        np.hypot(np.diff(cent.x.to_numpy()), np.diff(cent.y.to_numpy())) / 1000.0
        if len(cent) > 1
        else np.array([0.0])
    )

    return {
        "n_timesteps": int(len(g)),
        "t_start_utc": t.iloc[0].strftime("%Y-%m-%dT%H:%M:%S"),
        "t_end_utc": t.iloc[-1].strftime("%Y-%m-%dT%H:%M:%S"),
        "hours_missing": int((gaps_h != 1.0).sum()),
        "max_gap_h": _f(gaps_h.max()) if gaps_h.size else 1.0,
        "invalid_geometries": int((~metric.geometry.is_valid).sum()),
        "multipart_timesteps": int(metric.geometry.geom_type.eq("MultiPolygon").sum()),
        "timesteps_with_holes": int(
            sum(
                1
                for x in metric.geometry
                if (
                    len(x.interiors)
                    if x.geom_type == "Polygon"
                    else sum(len(p.interiors) for p in x.geoms)
                )
                > 0
            )
        ),
        "final_area_km2": _f(area_km2[-1]),
        "area_monotone": bool((d_area >= -1e-9).all()),
        "n_area_decreases": int((d_area < -1e-9).sum()),
        "max_area_decrease_km2": _f(-d_area.min()) if (d_area < 0).any() else 0.0,
        "unburning_area_max_km2": _f(unburn.max()),
        "unburning_area_total_km2": _f(unburn.sum()),
        "zero_growth_hour_fraction": _f((d_area == 0).mean()),
        "longest_zero_growth_run_h": int(_longest_run(d_area == 0)),
        "hourly_growth_km2_p50": _f(np.median(d_area)),
        "hourly_growth_km2_p95": _f(np.percentile(d_area, 95)),
        "hourly_growth_km2_max": _f(d_area.max()),
        "centroid_jump_km_p95": _f(np.percentile(jump_km, 95)),
        "centroid_jump_km_max": _f(jump_km.max()),
    }


def _longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def raster_qa(state: np.ndarray, grid: Grid) -> dict[str, Any]:
    """Checks on the rasterised ``(T, H, W)`` uint8 state field."""
    st = np.asarray(state)
    if st.ndim != 3:
        raise ValueError("state must be (T, H, W)")
    cell_km2 = (grid.cell_size_m / 1000.0) ** 2
    ever = st != UNBURNED
    burned_cells = ever.reshape(st.shape[0], -1).sum(axis=1)
    d_cells = np.diff(burned_cells) if st.shape[0] > 1 else np.array([0])

    # illegal transitions: anything that leaves the burned set, or 0 -> 2 directly.
    revert = int((ever[:-1] & ~ever[1:]).sum()) if st.shape[0] > 1 else 0
    skip = int(((st[:-1] == UNBURNED) & (st[1:] == BURNED_OUT)).sum()) if st.shape[0] > 1 else 0

    # teleporting: new burned cells with no burned 8-neighbour at t-1.
    tele_cells = 0
    tele_steps = 0
    max_tele_km = 0.0
    if st.shape[0] > 1:
        for i in range(1, st.shape[0]):
            new = ever[i] & ~ever[i - 1]
            if not new.any():
                continue
            detached = new & ~dilate(ever[i - 1], 1)
            n_det = int(detached.sum())
            if n_det:
                tele_cells += n_det
                tele_steps += 1
                max_tele_km = max(max_tele_km, _max_gap_km(detached, ever[i - 1], grid))

    burning_per_step = (st == BURNING).reshape(st.shape[0], -1).sum(axis=1)
    active = burned_cells > 0
    return {
        "grid_shape_yx": [int(grid.ny), int(grid.nx)],
        "grid_bbox_5070": list(grid.bounds),
        "cell_km2": cell_km2,
        "state_values_present": sorted(int(v) for v in np.unique(st)),
        "burned_area_monotone": bool((d_cells >= 0).all()),
        "n_burned_cell_decreases": int((d_cells < 0).sum()),
        "cells_reverting_from_burned": revert,
        "cells_skipping_burning_state": skip,
        "final_burned_cells": int(burned_cells[-1]),
        "final_burned_area_km2": _f(burned_cells[-1] * cell_km2),
        "teleport_steps": tele_steps,
        "teleport_cells_total": tele_cells,
        "teleport_max_gap_km": _f(max_tele_km),
        "frames_with_no_burning_cell": int((burning_per_step == 0).sum()),
        "frames_with_no_burning_cell_while_active": int(((burning_per_step == 0) & active).sum()),
        "burning_cells_p50": _f(np.median(burning_per_step)),
        "burning_cells_max": int(burning_per_step.max()),
    }


def _max_gap_km(detached: np.ndarray, prev_burned: np.ndarray, grid: Grid) -> float:
    """Largest distance from a detached new cell to the nearest previously burned
    cell, in km. Brute force over cell indices; domains are ~10^3-10^4 cells."""
    dy, dx = np.nonzero(detached)
    py, px = np.nonzero(prev_burned)
    if not len(py) or not len(dy):
        return 0.0
    res_km = grid.cell_size_m / 1000.0
    best = 0.0
    for y, x in zip(dy, dx, strict=True):
        d = np.hypot(py - y, px - x).min() * res_km
        best = max(best, float(d))
    return best


def east_west_noise(
    east: gpd.GeoDataFrame,
    west: gpd.GeoDataFrame,
    grid: Grid | None = None,
) -> dict[str, Any]:
    """GOFER-East vs GOFER-West disagreement = observed label noise, in km.

    The two variants see the same fire from different geostationary slots, so
    their disagreement bounds below the parallax + viewing-geometry component of
    the label error. The mean displacement *vector* is the systematic part; the
    equivalent-radius mismatch is the scale the model's observation-noise
    augmentation has to cover.
    """
    e = east.to_crs(CRS_STRING).set_index("timestep")
    w = west.to_crs(CRS_STRING).set_index("timestep")
    common = sorted(set(e.index) & set(w.index))
    if not common:
        return {"n_common_timesteps": 0}

    iou, dxs, dys, dr = [], [], [], []
    for ts in common:
        ge, gw = e.loc[ts, "geometry"], w.loc[ts, "geometry"]
        inter, union = ge.intersection(gw).area, ge.union(gw).area
        iou.append(inter / union if union > 0 else np.nan)
        ce, cw = ge.centroid, gw.centroid
        dxs.append((ce.x - cw.x) / 1000.0)
        dys.append((ce.y - cw.y) / 1000.0)
        dr.append((math.sqrt(ge.area / math.pi) - math.sqrt(gw.area / math.pi)) / 1000.0)

    iou_a, dx_a, dy_a, dr_a = map(np.asarray, (iou, dxs, dys, dr))
    dist = np.hypot(dx_a, dy_a)
    out = {
        "n_east_timesteps": int(len(e)),
        "n_west_timesteps": int(len(w)),
        "n_common_timesteps": int(len(common)),
        "iou_mean": _f(np.nanmean(iou_a)),
        "iou_p05": _f(np.nanpercentile(iou_a, 5)),
        "iou_min": _f(np.nanmin(iou_a)),
        "centroid_offset_km_mean": _f(dist.mean()),
        "centroid_offset_km_p90": _f(np.percentile(dist, 90)),
        "centroid_offset_km_max": _f(dist.max()),
        # systematic (vector-mean) component: the parallax signature
        "centroid_offset_vector_km": [_f(dx_a.mean()), _f(dy_a.mean())],
        "equiv_radius_mismatch_km_mean": _f(np.abs(dr_a).mean()),
        "equiv_radius_mismatch_km_p90": _f(np.percentile(np.abs(dr_a), 90)),
        "equiv_radius_mismatch_km_max": _f(np.abs(dr_a).max()),
        "east_larger_fraction": _f((dr_a > 0).mean()),
    }
    if grid is not None:
        # disagreement expressed on the actual analysis grid
        diff_cells = []
        for ts in common:
            me = polygon_mask(e.loc[ts, "geometry"], grid)
            mw = polygon_mask(w.loc[ts, "geometry"], grid)
            diff_cells.append(int((me ^ mw).sum()))
        out["symmetric_difference_cells_mean"] = _f(np.mean(diff_cells))
        out["symmetric_difference_cells_max"] = int(np.max(diff_cells))
    return out


def fire_qa_report(
    *,
    fire_id: str,
    perims: gpd.GeoDataFrame,
    state: np.ndarray,
    grid: Grid,
    east: gpd.GeoDataFrame | None = None,
    west: gpd.GeoDataFrame | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full per-fire QA block for the manifest."""
    report: dict[str, Any] = {
        "fire_id": fire_id,
        "vector": vector_qa(perims),
        "raster": raster_qa(state, grid),
    }
    if east is not None and west is not None:
        report["label_noise_east_west"] = east_west_noise(east, west, grid)
    if extra:
        report.update(extra)
    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Boil the numbers down to pass/warn flags. Warnings are not failures - GOFER
    genuinely behaves this way - but they must be visible in the manifest."""
    v, r = report["vector"], report["raster"]
    fails, warns = [], []
    if not r["burned_area_monotone"]:
        fails.append("burned area decreases on the grid")
    if r["cells_reverting_from_burned"]:
        fails.append(f"{r['cells_reverting_from_burned']} cell-steps leave the burned set")
    if r["cells_skipping_burning_state"]:
        fails.append(
            f"{r['cells_skipping_burning_state']} cell-steps go unburned->burned-out "
            "without passing through burning"
        )
    if v["hours_missing"]:
        fails.append(f"{v['hours_missing']} non-hourly gaps in the perimeter series")
    if v["invalid_geometries"]:
        fails.append(f"{v['invalid_geometries']} invalid geometries")
    if v["unburning_area_total_km2"]:
        warns.append(
            f"{v['unburning_area_total_km2']} km2 of perimeter area vanishes between "
            "steps in the source; repaired by cumulative-OR"
        )
    if r["teleport_steps"]:
        warns.append(
            f"{r['teleport_steps']} steps with burned area detached from the previous "
            f"footprint (max gap {r['teleport_max_gap_km']} km) — spotting or label jump"
        )
    if r["frames_with_no_burning_cell_while_active"]:
        warns.append(
            f"{r['frames_with_no_burning_cell_while_active']} frames have an active fire "
            "but no cell in state 1 — the contagion kernel has no seed in those steps"
        )
    if v["zero_growth_hour_fraction"] and v["zero_growth_hour_fraction"] > 0.4:
        warns.append(
            f"{100 * v['zero_growth_hour_fraction']:.0f}% of hours have exactly zero "
            f"perimeter growth (longest run {v['longest_zero_growth_run_h']} h)"
        )
    return {"pass": not fails, "failures": fails, "warnings": warns}
