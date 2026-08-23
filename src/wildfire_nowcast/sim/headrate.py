"""How fast did the fire's HEAD actually move, against ELMFIRE's own ceiling.

WHY THIS EXISTS
---------------
ELMFIRE clamps its crown fire spread rate at ``CROWN_FIRE_SPREAD_RATE_LIMIT``,
250 ft/min (``elmfire_namelists.f90:596``), applied as
``CROSA = MIN(CROSA, CROWN_FIRE_SPREAD_RATE_LIMIT)``
(``elmfire_spread_rate.f90:184``). That is **4.572 km/h**, and on a synthetic
scene three wind speeds an octave apart (8, 16, 32 m/s) returned bit-identical
output, which is what saturation against a constant looks like.

A published claim about ELMFIRE over-predicting growth means one thing if the
simulator was RESPONDING to the weather and a different thing if it was sitting
on a hard-coded constant. Settling that needs the **HEAD**, and the quantity our
run rows carry is an **AREA**: added cells divided by frontier length is a MEAN
advance over the whole front, and a mean far below the ceiling is perfectly
compatible with a head pinned at it. So the mean cannot answer the question and
this module does not compute one.

WHAT IS MEASURED
----------------
The distance from the ``t0`` burned set to the farthest cell ELMFIRE's own
time-of-arrival raster says was reached by each lead hour, on the **fine
analysis lattice** (30.303 m in the native configuration), BEFORE any coarsening
to the 1 km C1 grid. Two reasons for the fine lattice rather than the returned
C5 samples:

* resolution. At 1 km cells and a 3 h horizon the ceiling is 13.716 cells, and
  the neighbouring representable values are 13 and 14, so a 1 km read of the
  head is quantised at 0.333 km/h.
* direction of the error. :func:`~wildfire_nowcast.sim.coarsen.coarsen_occupancy`
  needs an occupancy threshold to light a 1 km cell, so a THIN fast finger can
  coarsen away entirely. A 1 km head therefore has a bias toward reading LOW,
  which is the same direction as the conclusion "the ceiling is slack". A
  measurement must not be biased toward its own answer.

Distances are straight-line from the initial burned set, so they are a LOWER
BOUND on path length and hence on the spread rate that produced them. Reading
below the ceiling on this statistic is therefore evidence, and reading at it is
proof.

NO SCIPY. ``distance_transform_edt`` is not available in this environment
(``sim/growth.py`` says the same and substitutes an offset scan). The exact
distance here is cheaper than a general EDT anyway: the initial state is a 1 km
field replicated onto sub-cells, so the burned set is a union of ``refine`` by
``refine`` axis-aligned blocks and the distance to a block has a closed form.
Only blocks on the burned set's boundary can be nearest to an outside cell, so
the scan is over a perimeter, not an area. Checked against brute force in
:mod:`wildfire_nowcast.sim.selftest`, not asserted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "CROWN_FIRE_SPREAD_RATE_LIMIT_FT_MIN",
    "CROWN_FIRE_SPREAD_RATE_LIMIT_KMH",
    "RADIAL_RATE_FLOOR_KM",
    "block_boundary",
    "distance_to_burned_fine",
    "head_advance",
]

#: ELMFIRE's own constant, ``elmfire_namelists.f90:596``.
CROWN_FIRE_SPREAD_RATE_LIMIT_FT_MIN = 250.0
#: 250 ft/min in km/h: ``250 * 0.3048 * 60 / 1000``.
CROWN_FIRE_SPREAD_RATE_LIMIT_KMH = CROWN_FIRE_SPREAD_RATE_LIMIT_FT_MIN * 0.3048 * 60.0 / 1000.0

#: Cells nearer than this to the ``t0`` set are excluded from the per-cell radial
#: rate. A cell one fine cell out with an arrival time of a few seconds gives a
#: meaningless rate of tens of km/h; the quantity is only interpretable once the
#: distance is large against the lattice.
RADIAL_RATE_FLOOR_KM = 1.0

_CHUNK = 50_000


def block_boundary(burned_coarse: np.ndarray) -> np.ndarray:
    """Burned coarse cells that have at least one unburned 8-neighbour.

    Cells outside the array count as BURNED for this test, which makes the
    boundary smaller. That is safe for the use here: every query cell is inside
    the array, the segment from a query to its nearest burned cell stays inside
    the array, and the cell one step before the end of that segment is a
    neighbour of it, so a fully surrounded burned cell is never the unique
    nearest one.
    """
    burned = np.asarray(burned_coarse) > 0
    if not burned.any():
        return np.zeros_like(burned)
    padded = np.ones((burned.shape[0] + 2, burned.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = burned
    interior = np.ones_like(burned)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= padded[1 + dy : 1 + dy + burned.shape[0], 1 + dx : 1 + dx + burned.shape[1]]
    return burned & ~interior


def distance_to_burned_fine(
    x0_coarse: np.ndarray, refine: int, *, chunk: int = _CHUNK
) -> np.ndarray:
    """Distance in FINE CELLS from every fine cell to the nearest burned fine cell.

    ``x0_coarse`` is the window's ``t0`` state on the coarse lattice; the fine
    burned set is that field replicated ``refine`` by ``refine``, which is what
    :meth:`~wildfire_nowcast.sim.elmfire.ElmfireNativeModel.predict` hands
    ELMFIRE. Burned fine cells get 0. If nothing is burned every cell gets
    ``inf``, because "distance from a fire that does not exist" has no value and
    a 0 there would silently read as "at the fire".
    """
    coarse = np.asarray(x0_coarse) > 0
    r = int(refine)
    ny, nx = coarse.shape
    fine_shape = (ny * r, nx * r)
    if not coarse.any():
        return np.full(fine_shape, np.inf, dtype=np.float64)

    edge = block_boundary(coarse)
    rows, cols = np.nonzero(edge)
    br0 = (rows * r).astype(np.int64)
    bc0 = (cols * r).astype(np.int64)

    burned_fine = np.repeat(np.repeat(coarse, r, axis=0), r, axis=1)
    qi, qj = np.nonzero(~burned_fine)
    out = np.zeros(fine_shape, dtype=np.float64)
    if qi.size == 0:
        return out
    best = np.empty(qi.size, dtype=np.int64)
    for start in range(0, qi.size, int(chunk)):
        stop = min(qi.size, start + int(chunk))
        a = qi[start:stop, None].astype(np.int64)
        b = qj[start:stop, None].astype(np.int64)
        dy = np.maximum(np.maximum(br0[None, :] - a, a - (br0[None, :] + r - 1)), 0)
        dx = np.maximum(np.maximum(bc0[None, :] - b, b - (bc0[None, :] + r - 1)), 0)
        best[start:stop] = (dy * dy + dx * dx).min(axis=1)
    out[qi, qj] = np.sqrt(best.astype(np.float64))
    return out


def head_advance(
    arrival_s: np.ndarray,
    distance_fine_cells: np.ndarray,
    burned_fine: np.ndarray,
    *,
    cell_size_m: float,
    horizon_h: int,
) -> dict[str, Any]:
    """Realised head advance for ONE member, from its time-of-arrival raster.

    ``arrival_s`` is ELMFIRE's ``time_of_arrival`` in seconds with a negative
    value where the fire never arrived. Returns kilometres and km/h beside
    ELMFIRE's own ceiling, so the comparison never has to be reconstructed by a
    reader holding one number.
    """
    arrival = np.asarray(arrival_s, dtype=np.float64)
    dist_km = np.asarray(distance_fine_cells, dtype=np.float64) * float(cell_size_m) / 1000.0
    new = (~np.asarray(burned_fine)) & (arrival >= 0.0)

    head_km: list[float] = []
    for k in range(int(horizon_h)):
        reached = new & (arrival <= (k + 1) * 3600.0 + 1e-6)
        head_km.append(float(dist_km[reached].max()) if reached.any() else 0.0)
    increments = [head_km[0]] + [head_km[k] - head_km[k - 1] for k in range(1, len(head_km))]

    reached_all = new & (arrival <= int(horizon_h) * 3600.0 + 1e-6)
    far = reached_all & (dist_km >= RADIAL_RATE_FLOOR_KM) & (arrival > 0.0)
    n_far = int(far.sum())
    if n_far:
        radial = dist_km[far] / (arrival[far] / 3600.0)
        max_radial = float(radial.max())
        share_at_cap = float((radial >= 0.9 * CROWN_FIRE_SPREAD_RATE_LIMIT_KMH).mean())
    else:
        max_radial = 0.0
        share_at_cap = 0.0

    sustained = head_km[-1] / float(horizon_h) if head_km else 0.0
    peak = max(increments) if increments else 0.0
    return {
        "cap_kmh": round(CROWN_FIRE_SPREAD_RATE_LIMIT_KMH, 6),
        "fine_cell_m": round(float(cell_size_m), 4),
        "head_km_by_lead": [round(v, 4) for v in head_km],
        "head_advance_kmh_by_lead": [round(v, 4) for v in increments],
        "sustained_head_kmh": round(sustained, 4),
        "peak_hourly_head_kmh": round(peak, 4),
        "sustained_head_over_cap": round(sustained / CROWN_FIRE_SPREAD_RATE_LIMIT_KMH, 4),
        "peak_hourly_head_over_cap": round(peak / CROWN_FIRE_SPREAD_RATE_LIMIT_KMH, 4),
        "max_radial_rate_kmh": round(max_radial, 4),
        "radial_rate_floor_km": RADIAL_RATE_FLOOR_KM,
        "n_reached_fine_cells": int(reached_all.sum()),
        "n_reached_beyond_floor": n_far,
        "share_beyond_floor_at_90pct_of_cap": round(share_at_cap, 4),
    }
