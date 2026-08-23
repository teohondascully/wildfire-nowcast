"""C4 - the synthetic fire generator, i.e. the parallelism unlock.

``make_synthetic_fire`` produces a complete, C1/C2/C3-conformant fire on disk in
well under five seconds and with no network, no credentials and no GEE. Model
and simulation/visualisation work can therefore start against real-shaped data
long before the real ingestion path exists.

Because it is a stand-in for GOFER labels, it deliberately exercises the awkward
paths rather than the easy ones:

* all three states ``{0, 1, 2}`` occur, and the perimeter -> state mapping is
  the ratified C1.1 rule ``fireline_v2``, applied through the single canonical
  implementation in :mod:`wildfire_nowcast.common.states` - the same function
  the real ingestion path calls, so the fixture cannot drift from the labels;
* fire is absorbing, so ``fire_state`` never decreases in time, and no cell
  jumps 0 -> 2 without a burning hour;
* **a scripted dormancy** (the fire lies down, GOES sees no fire line) makes
  state 1 legitimately EMPTY for several hours. C1.1 records this in 6-37% of
  real frames, so the fixture must *exercise* the phenomenon rather than hide
  it: a consumer that conditions solely on state 1 must break here, on a 0.6 s
  fixture, and not later on real data;
* a ``water_barrier_mask`` river runs the full height of the domain, is never
  burned, and stops the main front;
* at a scripted hour a **spot fire ignites on the far side of that river**, far
  enough away that no contiguous spread could have produced it. Long-range
  spotting is an explicit, separate component of the model (``README.md``,
  *the model has two components*), so every
  downstream consumer needs a fixture where it actually happens.

Physics here is a caricature - an anisotropic wind/slope-driven dilation with
correlated noise. It is *not* a baseline and must never be scored as one.

CLI::

    python -m wildfire_nowcast.common.synthetic --out outputs/synthetic_fire/tensor.zarr
    make synth
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import xarray as xr

from wildfire_nowcast.common import zarr_io as zio
from wildfire_nowcast.common.contract import CHANNELS, MIN_BUFFER_MARGIN_CELLS
from wildfire_nowcast.common.derive import (
    aspect_to_sin_cos,
    dead_fuel_moisture_simard,
    slope_aspect_from_elevation,
)
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.common.paths import outputs_dir
from wildfire_nowcast.common.states import apply_state_rule, dilate

__all__ = [
    "SyntheticFire",
    "SyntheticGeometry",
    "DEFAULT_GRID",
    "DEFAULT_IGNITION_UTC",
    "default_grid_for",
    "make_synthetic_fire",
    "build_synthetic_dataset",
]

#: North-west corner of the synthetic domain: the northern Sierra foothills
#: (Camp Fire country), snapped to the EPSG:5070 1 km lattice.
DOMAIN_NW_CORNER = (-2_190_000.0, 2_200_000.0)

#: The domain for the documented 24 h default: 128 x 128 km.
DEFAULT_GRID = Grid(x_min=DOMAIN_NW_CORNER[0], y_max=DOMAIN_NW_CORNER[1], nx=128, ny=128)

DEFAULT_IGNITION_UTC = "2020-08-17T21:00:00"

#: A synthetic fire belongs to no real landscape block, so it can never be
#: selected into a spatial CV fold or counted toward C3.3's `n_train_blocks`.
SYNTHETIC_BLOCK_ID = -1

MIN_HOURS = 3
MIN_CELLS = 32
MAX_DEFAULT_CELLS = 320

#: C1.2 - the fixture reserves this many cells of unburnable-by-construction
#: frame around the domain, so the final footprint always sits at least the
#: mandated 10 km inside every edge.
#:
#: A10 measured what the docstring below used to merely assert: at 48 h the fire
#: reached within 2-3 cells of the east edge, at 96 h and 168 h it touched it,
#: and at 168 h the north edge too. Every one of those fixtures passed the whole
#: contract, because C1.2's buffer sentence had never been implemented - a fire
#: that stops growing because it ran out of domain is exactly the artefact this
#: fixture exists NOT to teach. Reserving the frame is the same construction as
#: sizing the domain to the final perimeter plus 10 km, applied from the other
#: end, and unlike a growth heuristic it cannot silently stop holding.
EDGE_RESERVE_CELLS = MIN_BUFFER_MARGIN_CELLS


def default_grid_for(n_hours: int) -> Grid:
    """Domain sized so the fire never reaches the edge within ``n_hours``.

    A fire clipped by the domain boundary is a *misleading* fixture: spread
    stops for a reason that has nothing to do with weather, fuel or barriers,
    and anything trained or debugged against it learns an artefact. The default
    domain therefore grows with the horizon (roughly 3.5 cells per hour, the
    downwind head rate plus margin), staying at the documented 128 km box for
    the 24 h default.

    This sizing is a *budget*, not a guarantee - it was wrong above ~30 h. The
    guarantee is :data:`EDGE_RESERVE_CELLS`, enforced during simulation and
    asserted against the finished tensor.
    """
    cells = int(np.clip(int(np.ceil(3.5 * n_hours + 24)), 128, MAX_DEFAULT_CELLS))
    cells = int(np.ceil(cells / 8) * 8)
    return Grid(x_min=DOMAIN_NW_CORNER[0], y_max=DOMAIN_NW_CORNER[1], nx=cells, ny=cells)


def _interior_mask(shape: tuple[int, int], reserve: int = EDGE_RESERVE_CELLS) -> np.ndarray:
    """Cells the fire may occupy: everything but a ``reserve``-cell frame (C1.2)."""
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    r = int(max(0, min(reserve, (min(ny, nx) - 1) // 2)))
    mask[r : ny - r, r : nx - r] = True
    return mask


# FBFM40 palette: grass, grass-shrub, shrub, timber-understory, timber-litter.
_FUEL_PALETTE: tuple[int, ...] = (101, 102, 121, 122, 142, 145, 161, 165, 181, 186)
#: Relative rate-of-spread multiplier per fuel class (caricature, not Rothermel).
_FUEL_RATE: dict[int, float] = {
    101: 1.15,
    102: 1.20,
    121: 1.05,
    122: 1.10,
    142: 1.00,
    145: 0.95,
    161: 0.80,
    165: 0.85,
    181: 0.70,
    186: 0.75,
}
_FUEL_NB_WATER = 98  # FBFM40 NB8, open water: non-burnable


class SyntheticGeometry(NamedTuple):
    """Where and when the scripted events are, in array indices / hour indices."""

    ignition_rc: tuple[int, int]
    spot_rc: tuple[int, int]
    river_col_range: tuple[int, int]
    crossing_hour: int
    #: ``[start, stop)`` hours during which the fire lies down: no growth and no
    #: fire line, so ``fireline_v2`` closes every cell and state 1 is EMPTY.
    dormancy_hours: tuple[int, int] = (0, 0)


class SyntheticFire(NamedTuple):
    """Return value of :func:`make_synthetic_fire`.

    A tuple, so ``tensor, manifest, *_ = make_synthetic_fire(0)`` works, and
    named, so ``result.tensor_path`` also works.
    """

    tensor_path: Path
    manifest_path: Path
    norm_stats_path: Path
    fire_id: str
    geometry: SyntheticGeometry


# --------------------------------------------------------------------------
# small numeric primitives
# --------------------------------------------------------------------------


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter over a ``(2r+1)^2`` window via a summed-area table, O(N)."""
    if radius < 1:
        return a.astype(np.float64, copy=False)
    pad = np.pad(a.astype(np.float64, copy=False), radius, mode="edge")
    csum = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)))
    k = 2 * radius + 1
    ny, nx = a.shape
    total = (
        csum[k : k + ny, k : k + nx]
        - csum[0:ny, k : k + nx]
        - csum[k : k + ny, 0:nx]
        + csum[0:ny, 0:nx]
    )
    return total / float(k * k)


def _smooth_field(
    rng: np.random.Generator, shape: tuple[int, int], radius: int, passes: int = 2
) -> np.ndarray:
    """Spatially correlated noise in ``[0, 1]``.

    Correlated, not white: fire perimeters and fuel beds are lumpy at the
    kilometre scale, and independent per-pixel noise would produce
    salt-and-pepper perimeters that no downstream code should ever be trained
    or debugged against.
    """
    field = rng.random(shape)
    for _ in range(max(1, passes)):
        field = _box_blur(field, radius)
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-12:
        return np.zeros(shape, dtype=np.float64)
    return (field - lo) / (hi - lo)


def _shift_or(out: np.ndarray, mask: np.ndarray, dr: int, dc: int) -> None:
    """``out |= mask`` shifted by ``(dr, dc)``, clipped at the domain edge."""
    ny, nx = mask.shape
    dst_r = slice(max(0, dr), ny + min(0, dr))
    src_r = slice(max(0, -dr), ny + min(0, -dr))
    dst_c = slice(max(0, dc), nx + min(0, dc))
    src_c = slice(max(0, -dc), nx + min(0, -dc))
    if dst_r.start >= dst_r.stop or dst_c.start >= dst_c.stop:
        return
    out[dst_r, dst_c] |= mask[src_r, src_c]


def _spread_kernel(u: float, v: float, scale: float = 1.0) -> np.ndarray:
    """Boolean structuring element: the cells reachable in one hour.

    An egg-shaped (offset-ellipse) neighbourhood whose major axis lies along the
    wind: long downwind head, short backing edge, intermediate flanks. This is
    the classic elliptical-spread caricature, discretised.

    ``scale`` shrinks the whole ellipse; the simulator uses a full-size kernel
    for stochastic candidates and a shrunken one for the deterministic core, so
    the front always advances somewhere while its shape stays ragged.
    """
    speed = float(np.hypot(u, v))
    head = float(np.clip(1.2 + 0.38 * speed, 1.2, 6.0)) * scale
    back = max(0.4, 0.20 * head)
    flank = max(0.6, 0.30 * head)

    radius = int(np.ceil(max(head, flank, back)))
    offs = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offs, offs, indexing="ij")
    d_east = dc.astype(np.float64)
    d_north = -dr.astype(np.float64)  # row index increases southward

    if speed < 1e-6:
        wu, wv = 1.0, 0.0
    else:
        wu, wv = u / speed, v / speed

    along = d_east * wu + d_north * wv
    across = -d_east * wv + d_north * wu

    semi_major = (head + back) / 2.0
    focus = (head - back) / 2.0
    kernel = ((along - focus) ** 2) / semi_major**2 + (across**2) / flank**2 <= 1.0
    kernel[radius, radius] = True
    return kernel


def _dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    radius = kernel.shape[0] // 2
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if kernel[dr + radius, dc + radius]:
                _shift_or(out, mask, dr, dc)
    return out


def _disc(shape: tuple[int, int], row: int, col: int, radius: float) -> np.ndarray:
    ny, nx = shape
    rr, cc = np.ogrid[:ny, :nx]
    return cast("np.ndarray[Any, Any]", ((rr - row) ** 2 + (cc - col) ** 2) <= radius**2)


# --------------------------------------------------------------------------
# static and weather fields
# --------------------------------------------------------------------------


def _terrain(rng: np.random.Generator, grid: Grid) -> np.ndarray:
    """Ridge-and-valley elevation, roughly 150-1900 m."""
    ny, nx = grid.shape
    rows = np.linspace(0.0, 1.0, ny)[:, None]
    cols = np.linspace(0.0, 1.0, nx)[None, :]
    ridges = (
        420.0 * np.sin(2.0 * np.pi * (1.4 * cols + 0.15))
        + 260.0 * np.cos(2.0 * np.pi * (1.9 * rows - 0.35))
        + 180.0 * np.sin(2.0 * np.pi * (0.8 * (rows + cols)))
    )
    ramp = 500.0 * (1.0 - cols)
    rough = 260.0 * (_smooth_field(rng, (ny, nx), radius=4) - 0.5)
    elev = 900.0 + ridges * 0.6 + ramp * 0.5 + rough
    return np.clip(elev, 120.0, 2400.0)


def _river_mask(grid: Grid, col_fraction: float = 0.55) -> tuple[np.ndarray, tuple[int, int]]:
    """A sinuous, full-height, 3-cell-wide river and one lake."""
    ny, nx = grid.shape
    rows = np.arange(ny)
    centre = col_fraction * nx
    amplitude = max(2.0, nx / 40.0)
    river_col = centre + amplitude * np.sin(2.0 * np.pi * rows / max(ny / 1.5, 1.0))

    cols = np.arange(nx)[None, :]
    mask = np.abs(cols - river_col[:, None]) <= 1.0

    # A lake, parked in the north-east so it never interacts with the fire path.
    lake = _disc((ny, nx), row=int(0.12 * ny), col=int(0.86 * nx), radius=max(3.0, nx / 26.0))
    mask = mask | lake
    return mask, (int(np.floor(river_col.min() - 1)), int(np.ceil(river_col.max() + 1)))


def _static_fields(rng: np.random.Generator, grid: Grid) -> dict[str, np.ndarray]:
    ny, nx = grid.shape
    elevation = _terrain(rng, grid)
    slope_deg, aspect_deg = slope_aspect_from_elevation(elevation, grid.cell_size_m)
    aspect_sin, aspect_cos = aspect_to_sin_cos(aspect_deg, slope_deg)

    water, river_cols = _river_mask(grid)

    fuel_field = _smooth_field(rng, (ny, nx), radius=6)
    idx = np.clip((fuel_field * len(_FUEL_PALETTE)).astype(int), 0, len(_FUEL_PALETTE) - 1)
    fuel_model_id = np.asarray(_FUEL_PALETTE, dtype=np.float32)[idx]
    fuel_model_id[water] = float(_FUEL_NB_WATER)

    canopy = 85.0 * _smooth_field(rng, (ny, nx), radius=5)
    canopy[water] = 0.0

    # Last season's burn scar, in the south-west, off the main run.
    scar = _disc((ny, nx), row=int(0.82 * ny), col=int(0.22 * nx), radius=max(4.0, nx / 12.0))

    return {
        "elevation": elevation.astype(np.float32),
        "slope": slope_deg.astype(np.float32),
        "aspect_sin": aspect_sin.astype(np.float32),
        "aspect_cos": aspect_cos.astype(np.float32),
        "fuel_model_id": fuel_model_id.astype(np.float32),
        "canopy_cover": canopy.astype(np.float32),
        "water_barrier_mask": water.astype(np.float32),
        "recent_burn_scar": scar.astype(np.float32),
        "_river_cols": np.asarray(river_cols),
    }


def _weather(
    rng: np.random.Generator,
    grid: Grid,
    times: np.ndarray,
    elevation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Diurnal RTMA-like weather: hot dry afternoons, a steady WSW wind."""
    ny, nx = grid.shape
    n_t = int(times.size)
    hours = (times.astype("datetime64[h]").astype("int64") % 24).astype(np.float64)

    # Diurnal cycle: temperature peaks ~22 UTC (mid-afternoon in California).
    phase = 2.0 * np.pi * (hours - 16.0) / 24.0
    temp_base = 297.0 + 11.0 * np.sin(phase)
    rh_base = np.clip(42.0 - 26.0 * np.sin(phase), 8.0, 96.0)

    lapse = -6.5e-3 * (elevation - float(elevation.mean()))
    spatial_t = 1.2 * (_smooth_field(rng, (ny, nx), radius=7) - 0.5)
    spatial_rh = 6.0 * (_smooth_field(rng, (ny, nx), radius=7) - 0.5)

    # Wind blows toward ~70 deg (ENE), veering slowly, gusting in the afternoon.
    bearing = np.radians(70.0 + 14.0 * np.sin(2.0 * np.pi * np.arange(n_t) / 19.0))
    speed = 5.0 + 4.0 * np.clip(np.sin(phase), 0.0, None) + rng.normal(0.0, 0.35, n_t)
    speed = np.clip(speed, 1.5, 14.0)

    gust = 0.9 + 0.2 * (_smooth_field(rng, (ny, nx), radius=8) - 0.5)

    temp = (temp_base[:, None, None] + lapse[None] + spatial_t[None]).astype(np.float32)
    rh = np.clip(rh_base[:, None, None] + spatial_rh[None], 3.0, 100.0).astype(np.float32)
    u = ((speed * np.sin(bearing))[:, None, None] * gust[None]).astype(np.float32)
    v = ((speed * np.cos(bearing))[:, None, None] * gust[None]).astype(np.float32)

    return {
        "wind_u10": u,
        "wind_v10": v,
        "temp_2m": temp,
        "rh_2m": rh,
        "fuel_moisture_proxy": dead_fuel_moisture_simard(temp, rh).astype(np.float32),
    }


# --------------------------------------------------------------------------
# fire simulation
# --------------------------------------------------------------------------


def _scripted_hours(n_hours: int) -> tuple[int, tuple[int, int]]:
    """``(crossing_hour, (dormancy_start, dormancy_stop))`` for a horizon.

    The dormancy is placed strictly AFTER the barrier crossing: a spot fire
    landing mid-dormancy would create new cells, and ``fireline_v2`` puts every
    new cell in state 1, which would silently cancel the empty-state-1 frames
    the fixture exists to produce. Its length is ~12% of the horizon (floor 2 h),
    which lands inside C1.1's measured 6-37% band for real fires.
    """
    crossing = int(np.clip(round(n_hours * 0.30), 1, n_hours - 1))
    start = int(np.clip(round(n_hours * 0.55), crossing + 1, n_hours - 1))
    length = max(2, int(round(n_hours * 0.12)))
    return crossing, (start, min(start + length, n_hours))


def _simulate_perimeters(
    rng: np.random.Generator,
    grid: Grid,
    n_hours: int,
    static: dict[str, np.ndarray],
    weather: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, SyntheticGeometry]:
    """Grow a perimeter hour by hour and emit a matching active fire line.

    Returns ``(perimeters, fire_lines, geometry)`` - the two inputs C1.1 takes.
    The fire line stands in for GOFER ``cfireLine``: the band of cells around
    the hour's advance, i.e. where a satellite would still see flaming front. It
    is EMPTY through the scripted dormancy, which is what makes ``fireline_v2``
    close every cell and leave state 1 empty for those hours.
    """
    ny, nx = grid.shape
    barrier = static["water_barrier_mask"] > 0.5
    river_lo, river_hi = (int(v) for v in static["_river_cols"])

    interior = _interior_mask((ny, nx))
    ignition_row = int(0.52 * ny)
    ignition_col = int(0.22 * nx)
    spot_row = int(0.44 * ny)
    # The spot must land beyond the river (it is the barrier CROSSING) and
    # inside the C1.2 reserve. If those two cannot both hold the domain is too
    # small for a crossing at all, which is a generator bug, not a fire.
    spot_col = min(nx - 1 - EDGE_RESERVE_CELLS, river_hi + max(5, nx // 20))
    crossing_hour, dormancy = _scripted_hours(n_hours)

    if barrier[spot_row, spot_col]:
        spot_col = min(nx - 1 - EDGE_RESERVE_CELLS, spot_col + 3)
    if not (river_hi < spot_col and interior[spot_row, spot_col]):
        raise ValueError(
            f"cannot place the scripted spot fire: river ends at column {river_hi} and the "
            f"C1.2 reserve leaves nothing east of it in a {nx}-cell domain. Enlarge the grid "
            "rather than moving the spot back across the river — a spot on the near bank is "
            "not a barrier crossing, and the crossing is what the fixture exists to exercise"
        )

    # Per-cell susceptibility: fuel, slope and last season's scar.
    fuel_rate = np.full((ny, nx), 0.9, dtype=np.float64)
    for code, rate in _FUEL_RATE.items():
        fuel_rate[static["fuel_model_id"] == float(code)] = rate
    fuel_rate[barrier] = 0.0
    slope_boost = 1.0 + 0.018 * static["slope"].astype(np.float64)
    scar_damp = 1.0 - 0.85 * static["recent_burn_scar"].astype(np.float64)
    susceptibility = np.clip(0.95 * fuel_rate * slope_boost * scar_damp, 0.0, 1.0)

    burned = _disc((ny, nx), ignition_row, ignition_col, radius=1.2) & ~barrier & interior
    perimeters: list[np.ndarray] = [burned.copy()]
    fire_lines: list[np.ndarray] = [burned.copy()]

    for t in range(1, n_hours):
        dormant = dormancy[0] <= t < dormancy[1]
        previous = burned
        if not dormant:
            u = float(weather["wind_u10"][t].mean())
            v = float(weather["wind_v10"][t].mean())
            reachable = ~burned & ~barrier & interior
            candidates = _dilate(burned, _spread_kernel(u, v)) & reachable
            core = _dilate(burned, _spread_kernel(u, v, scale=0.6)) & reachable

            if candidates.any():
                draw = _smooth_field(rng, (ny, nx), radius=2)
                # Core advance is deterministic wherever fuel allows, so the
                # front never stalls; the outer band is stochastic, so it stays
                # ragged.
                accepted = (candidates & (draw < susceptibility)) | (core & (susceptibility > 0.25))
                if not accepted.any():
                    score = np.where(candidates, susceptibility - draw, -np.inf)
                    accepted = np.zeros_like(candidates)
                    accepted.flat[int(np.argmax(score))] = True
                burned = burned | accepted

            if t == crossing_hour:
                spot = _disc((ny, nx), spot_row, spot_col, radius=1.2) & ~barrier & interior
                burned = burned | spot

        new = burned & ~previous
        # The flaming front is the advance plus one cell of shoulder, so a cell
        # keeps burning for a few hours after it is enclosed (GOFER's residence
        # p50 is 3-5 h) instead of the flat 1 h the retired rule produced.
        line = dilate(new, 1) & burned if new.any() else np.zeros((ny, nx), dtype=bool)
        perimeters.append(burned.copy())
        fire_lines.append(line)

    geometry = SyntheticGeometry(
        ignition_rc=(ignition_row, ignition_col),
        spot_rc=(spot_row, spot_col),
        river_col_range=(river_lo, river_hi),
        crossing_hour=crossing_hour,
        dormancy_hours=dormancy,
    )
    return np.asarray(perimeters), np.asarray(fire_lines), geometry


def _simulate_fire_state(
    rng: np.random.Generator,
    grid: Grid,
    n_hours: int,
    static: dict[str, np.ndarray],
    weather: dict[str, np.ndarray],
) -> tuple[np.ndarray, SyntheticGeometry]:
    """Perimeters + fire lines -> ``{0, 1, 2}`` through the C1.1 rule.

    The mapping is NOT implemented here. It is delegated to
    :func:`wildfire_nowcast.common.states.apply_state_rule`, the one
    implementation the contract adjudicates (C0). If the fixture and the real
    labels ever disagree about what state 1 means, that is a bug in one shared
    function, not a discrepancy between two lookalike ones.
    """
    perimeters, fire_lines, geometry = _simulate_perimeters(rng, grid, n_hours, static, weather)
    state = apply_state_rule(perimeters, rule="fireline_v2", fire_line_masks=fire_lines)
    return state, geometry


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def build_synthetic_dataset(
    seed: int,
    n_hours: int = 24,
    *,
    grid: Grid | None = None,
    ignition_time_utc: str = DEFAULT_IGNITION_UTC,
) -> tuple[xr.Dataset, SyntheticGeometry]:
    """Build the C1 dataset in memory, without touching disk."""
    if n_hours < MIN_HOURS:
        raise ValueError(
            f"n_hours must be >= {MIN_HOURS} so that all three states and the scripted "
            f"barrier crossing can occur; got {n_hours}"
        )
    grid = grid or default_grid_for(n_hours)
    if min(grid.shape) < MIN_CELLS:
        raise ValueError(
            f"grid must be at least {MIN_CELLS} cells on each side to fit the scripted "
            f"river and its far-side spot fire; got {grid.shape}"
        )

    rng = np.random.default_rng(seed)
    times = zio.hourly_times(ignition_time_utc, n_hours)

    static = _static_fields(rng, grid)
    weather = _weather(rng, grid, times, static["elevation"])
    fire_state, geometry = _simulate_fire_state(rng, grid, n_hours, static, weather)

    channels: dict[str, np.ndarray] = {"fire_state": fire_state}
    channels.update(weather)
    channels.update({k: v for k, v in static.items() if not k.startswith("_")})
    missing = [c for c in CHANNELS if c not in channels]
    if missing:  # pragma: no cover - guards against future channel additions
        raise AssertionError(f"synthetic generator is missing C1 channels {missing}")

    ds = zio.build_tensor_dataset(
        channels,
        grid,
        times,
        attrs={
            "title": "synthetic wildfire (C4)",
            "synthetic": "true",
            "synthetic_seed": int(seed),
            "synthetic_crossing_hour": int(geometry.crossing_hour),
            "synthetic_dormancy_hours": list(geometry.dormancy_hours),
            "state_rule": "fireline_v2",
            "comment": (
                "Generated by wildfire_nowcast.common.synthetic. Caricature physics; "
                "never use as a baseline or as training data for reported results."
            ),
        },
    )
    return ds, geometry


def make_synthetic_fire(
    seed: int,
    n_hours: int = 24,
    *,
    out: str | Path | None = None,
    fire_id: str | None = None,
    grid: Grid | None = None,
    ignition_time_utc: str = DEFAULT_IGNITION_UTC,
    cv_fold: int = -1,
    with_norm_stats: bool = True,
) -> SyntheticFire:
    """C4 - write one synthetic fire and return its paths.

    Parameters
    ----------
    seed
        Fully determines the fire; the same seed always yields the same tensor.
    n_hours
        Number of hourly steps (>= 3).
    out
        Path of the ``tensor.zarr`` store. ``manifest.json`` and
        ``norm_stats.json`` are written beside it, so the result is a valid
        C1/C2/C3 triple and ``manifest["norm_stats_path"]`` resolves.
        Defaults to ``outputs/synthetic/{fire_id}/tensor.zarr``.
    cv_fold
        Defaults to ``-1``, meaning "not a member of any leave-fire-out fold".
        A synthetic fire must never be selected into a real CV split, and for
        the same reason its ``spatial_block_id`` is ``-1``: it belongs to no
        real landscape block (C3.1).
    """
    fire_id = fire_id or f"synthetic_{seed:04d}"
    tensor_path = Path(out) if out else outputs_dir() / "synthetic" / fire_id / "tensor.zarr"
    out_dir = tensor_path.parent

    ds, geometry = build_synthetic_dataset(
        seed, n_hours, grid=grid, ignition_time_utc=ignition_time_utc
    )
    grid_used = grid or default_grid_for(n_hours)

    norm_stats_path = out_dir / "norm_stats.json"
    manifest = zio.build_manifest(
        fire_id=fire_id,
        gofer_version="synthetic (no GOFER; wildfire_nowcast.common.synthetic)",
        bbox_5070=grid_used.bounds,
        ignition_time_utc=ignition_time_utc,
        n_hours=int(n_hours),
        cv_fold=int(cv_fold),
        spatial_block_id=SYNTHETIC_BLOCK_ID,
        provenance={
            "generator": "wildfire_nowcast.common.synthetic.make_synthetic_fire",
            "seed": str(seed),
            "generated_utc": zio.utc_now_iso(),
            "state_rule": "fireline_v2",
            "state_rule_source": "wildfire_nowcast.common.states.apply_state_rule (C0/C1.1)",
            "time_convention": "end_of_hour",
            # C2 [v2] requires the LANDFIRE vintage and the fconf used. The
            # honest answer for a generated fire is "not applicable", and
            # DECLARING that satisfies the clause while pretending to a vintage
            # would not (C-1: declaring a weakness is a gate, omitting it is a
            # failure). Nothing here is a stand-in for a real source.
            "fuels_vintage_year": "n/a (synthetic: fuels are generated, never sourced)",
            "cfire_conf": "n/a (synthetic: no GOFER perimeters; the C1.1 rule is applied to "
            "generated perimeters and their generated fire lines)",
        },
        norm_stats_path="norm_stats.json",
        fuel_vintage_lag_years=0,
        n_ignition_components=1,
        extra={
            "synthetic": {
                "seed": int(seed),
                "ignition_rowcol": list(geometry.ignition_rc),
                "spot_rowcol": list(geometry.spot_rc),
                "river_col_range": list(geometry.river_col_range),
                "crossing_hour": int(geometry.crossing_hour),
                "dormancy_hours": list(geometry.dormancy_hours),
                "grid": {
                    "x_min": grid_used.x_min,
                    "y_max": grid_used.y_max,
                    "nx": grid_used.nx,
                    "ny": grid_used.ny,
                },
            }
        },
    )

    written_tensor, written_manifest = zio.write_fire(ds, manifest, out_dir)
    if with_norm_stats:
        # One synthetic fire is one landscape, so these stats are BOOTSTRAP by
        # construction (C3.3) and the file says so. A synthetic fire is never
        # reporting-ready, and the contract report should keep saying that.
        stats = zio.compute_norm_stats(
            [ds], train_folds=[int(cv_fold)], spatial_block_ids=[SYNTHETIC_BLOCK_ID]
        )
        stats["note"] = (
            "synthetic fire (C4): plumbing fixture only. Never normalise real data with these."
        )
        zio.write_norm_stats(stats, norm_stats_path)
    return SyntheticFire(
        tensor_path=written_tensor,
        manifest_path=written_manifest,
        norm_stats_path=norm_stats_path,
        fire_id=fire_id,
        geometry=geometry,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.common.synthetic",
        description="Generate one C1/C2/C3-conformant synthetic fire (contract C4).",
    )
    p.add_argument("--out", type=Path, default=None, help="path of the tensor.zarr to write")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--hours", type=int, default=24, help="number of hourly steps (default: 24)")
    p.add_argument("--fire-id", type=str, default=None, help="fire id (default: synthetic_SEED)")
    p.add_argument("--quiet", action="store_true", help="print nothing on success")
    add_logging_arguments(p)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    # ADR-103: the ONE place this program is allowed to configure logging.
    configure_from_args(args)
    result = make_synthetic_fire(
        seed=args.seed, n_hours=args.hours, out=args.out, fire_id=args.fire_id
    )
    if not args.quiet:
        from wildfire_nowcast.common.states import burning_residence_hours, frames_without_burning

        ds = zio.open_tensor(result.tensor_path)
        state = zio.fire_state_of(ds)
        counts = {value: int((state == value).sum()) for value in (0, 1, 2)}
        empty = frames_without_burning(state)
        residence = burning_residence_hours(state)
        print(f"fire_id     : {result.fire_id}")
        print(f"tensor      : {result.tensor_path}")
        print(f"manifest    : {result.manifest_path}")
        print(f"norm_stats  : {result.norm_stats_path}")
        print(f"shape       : {tuple(int(ds.sizes[d]) for d in ('time', 'y', 'x'))} (t, y, x)")
        print(f"state counts: unburned={counts[0]} burning={counts[1]} burned_out={counts[2]}")
        print(f"crossing_hr : {result.geometry.crossing_hour} (spot at {result.geometry.spot_rc})")
        print(
            f"dormancy    : hours {result.geometry.dormancy_hours[0]}-"
            f"{result.geometry.dormancy_hours[1] - 1} -> {empty.size}/{state.shape[0]} frames "
            f"({100.0 * empty.size / state.shape[0]:.0f}%) with NO cell in state 1 (C1.1: 6-37% "
            "of real frames)"
        )
        print(
            f"residence   : burning hours per cell p50="
            f"{float(np.median(residence)) if residence.size else 0.0:.0f} "
            f"max={int(residence.max()) if residence.size else 0}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
