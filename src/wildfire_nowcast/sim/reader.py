"""C1 read layer for the visualisation package.

This is the ONLY place in :mod:`wildfire_nowcast.sim` that touches a tensor
store, and it touches it exclusively through
:mod:`wildfire_nowcast.common.zarr_io` accessors keyed BY CHANNEL NAME. No
integer ever indexes ``features`` here.

Why that matters more than it looks: the v2 store holds ``features`` with 13
channels and ``channel_index_offset = 1``, so position ``i`` is C1 channel
``i + 1``. A literal ``features[:, 5]`` therefore returns ``slope`` where the
author meant ``elevation`` - a *plausible* raster, correlated with the right
one, that renders as a physics anomaly rather than as an indexing bug.

The reader also derives the quantities the renderers need and the contract
asks consumers to think in:

* ``ever`` - ``state > 0``, the cumulative burned region. Monotone by C1.4.
* ``frontier`` - the edge of ``ever``. C1.1 is explicit that **the contagion
  source is the frontier of the burned region, not state 1 alone**, because
  state 1 is legitimately empty in 6-37% of frames. Every renderer draws the
  frontier so a dormant frame still shows where the fire can restart.
* ``dormant`` - frames with zero cells in state 1. LEGAL, not missing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, CELL_SIZE_M
from wildfire_nowcast.common.zarr_io import channel_values, get_channel, open_tensor
from wildfire_nowcast.sim.style import PlotGeometry, plot_extent

__all__ = ["FireFrames", "load_fire", "ever_burned", "frontier_of", "arrival_hour"]


# -- pure array helpers (no I/O, trivially testable) ------------------------


def ever_burned(state: np.ndarray) -> np.ndarray:
    """``state > 0`` - the cumulative burned region, per timestep."""
    return np.asarray(state) > 0


def frontier_of(mask: np.ndarray, *, connectivity: int = 8) -> np.ndarray:
    """Cells inside ``mask`` that touch at least one cell outside it.

    Implemented with shifts rather than ``scipy.ndimage`` to keep the viz layer
    free of a dependency the contract does not already require. ``mask`` may be
    ``(y, x)`` or ``(t, y, x)``; the leading axis is untouched.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or 3-D mask, got {m.ndim}-D")
    pad = [(0, 0)] * (m.ndim - 2) + [(1, 1), (1, 1)]
    # Pad with True so the domain edge is NOT reported as frontier: a fire that
    # has run off the tile is a different phenomenon from an interior front, and
    # conflating them would draw a bright edge around every clipped fire.
    p = np.pad(m, pad, constant_values=True)
    offsets = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 4
        else [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    )
    ny, nx = m.shape[-2:]
    exposed = np.zeros_like(m)
    for dy, dx in offsets:
        sl = (..., slice(1 + dy, 1 + dy + ny), slice(1 + dx, 1 + dx + nx))
        exposed |= ~p[sl]
    return m & exposed


def arrival_hour(state: np.ndarray, *, fill: float = np.nan) -> np.ndarray:
    """First time index at which each cell is burned; ``fill`` where it never is.

    Note the censoring: "never arrived within the window" is NOT a late arrival
    time, and encoding it as one (e.g. ``T``) biases every arrival-time
    statistic downward-in-spread. It is returned as ``fill`` (NaN by default) so
    the caller is forced to decide.
    """
    ever = ever_burned(state)
    any_ever = ever.any(axis=0)
    first = np.argmax(ever, axis=0).astype(np.float64)
    return np.where(any_ever, first, fill)


# -- the loaded fire -------------------------------------------------------


@dataclass
class FireFrames:
    """One fire, loaded from a C1 store, with the derived views renderers need."""

    fire_id: str
    state: np.ndarray  # uint8 (t, y, x)
    times: np.ndarray  # datetime64[ns] (t,)
    wind_u: np.ndarray  # f32 (t, y, x) eastward
    wind_v: np.ndarray  # f32 (t, y, x) northward
    barrier: np.ndarray  # bool (y, x)
    elevation: np.ndarray  # f32 (y, x)
    geom: PlotGeometry
    source: str
    attrs: dict[str, Any]

    # -- derived ----------------------------------------------------------

    @property
    def n_hours(self) -> int:
        return int(self.state.shape[0])

    @cached_property
    def ever(self) -> np.ndarray:
        return ever_burned(self.state)

    @cached_property
    def burning(self) -> np.ndarray:
        return self.state == BURNING

    @cached_property
    def burned_out(self) -> np.ndarray:
        return self.state == BURNED_OUT

    @cached_property
    def frontier(self) -> np.ndarray:
        """Edge of the cumulative burned region - the C1.1 contagion source."""
        return frontier_of(self.ever)

    @cached_property
    def area_km2(self) -> np.ndarray:
        cell_km2 = (self.geom.cell_size_m / 1000.0) ** 2
        return self.ever.sum(axis=(1, 2)) * cell_km2

    @cached_property
    def growth_km2(self) -> np.ndarray:
        """Per-hour new burned area. ``growth[0]`` is the ignition footprint."""
        g = np.diff(self.area_km2, prepend=0.0)
        return np.asarray(g)

    @cached_property
    def n_burning(self) -> np.ndarray:
        return self.burning.sum(axis=(1, 2))

    @cached_property
    def dormant(self) -> np.ndarray:
        """Frames with NO cell in state 1. Legal under C1.1 (6-37% of real frames)."""
        return self.n_burning == 0

    @cached_property
    def wind_speed(self) -> np.ndarray:
        return np.hypot(self.wind_u, self.wind_v)

    def dormant_run_length(self, t: int) -> int:
        """How many consecutive frames up to and including ``t`` are dormant."""
        n = 0
        while t - n >= 0 and bool(self.dormant[t - n]):
            n += 1
        return n

    def label(self, t: int) -> str:
        return str(np.datetime_as_string(self.times[t], unit="h")).replace("T", " ") + "Z"

    def summary(self) -> dict[str, Any]:
        zero_growth = float((self.growth_km2[1:] == 0).mean()) if self.n_hours > 1 else 0.0
        return {
            "fire_id": self.fire_id,
            "source": self.source,
            "n_hours": self.n_hours,
            "shape": list(self.geom.shape),
            "final_area_km2": float(self.area_km2[-1]),
            "dormant_frames": int(self.dormant.sum()),
            "dormant_frac": float(self.dormant.mean()),
            "zero_growth_frac": zero_growth,
            "max_hourly_growth_km2": float(self.growth_km2.max()),
            "max_wind_speed_ms": float(self.wind_speed.max()),
            "barrier_cells": int(self.barrier.sum()),
        }


def load_fire(path: str | Path, *, fire_id: str | None = None) -> FireFrames:
    """Open a C1 tensor store and materialise the views the renderers need.

    Channels are read BY NAME. The C1.4 axis orientation is verified before any
    geometry is built (:func:`wildfire_nowcast.sim.style.plot_extent` raises on
    a store that is not north-up).
    """
    p = Path(path)
    ds: xr.Dataset = open_tensor(p)

    state = np.asarray(get_channel(ds, "fire_state").values, dtype=np.uint8)
    cell = float(ds.attrs.get("cell_size_m", CELL_SIZE_M))
    geom = plot_extent(ds["x"].values, ds["y"].values, cell_size_m=cell)

    wind_u = channel_values(ds, "wind_u10", dtype=np.float32)
    wind_v = channel_values(ds, "wind_v10", dtype=np.float32)
    barrier = channel_values(ds, "water_barrier_mask", dtype=np.float32)[0] > 0.5
    elevation = channel_values(ds, "elevation", dtype=np.float32)[0]

    resolved = fire_id or str(ds.attrs.get("fire_id") or p.parent.name)
    return FireFrames(
        fire_id=resolved,
        state=state,
        times=np.asarray(ds["time"].values),
        wind_u=wind_u,
        wind_v=wind_v,
        barrier=barrier,
        elevation=elevation,
        geom=geom,
        source=str(p),
        attrs=dict(ds.attrs),
    )
