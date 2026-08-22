"""Derived-channel formulas, defined once.

Channels 6-8 (slope, aspect_sin, aspect_cos) and channel 11
(fuel_moisture_proxy) are *derived*, and C1 requires the derivation to be
documented. It is documented here, and both the synthetic generator and the
real ingestion path call these same functions, so synthetic and real tensors
cannot disagree about what a channel means.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kelvin_to_fahrenheit",
    "dead_fuel_moisture_simard",
    "slope_aspect_from_elevation",
    "aspect_to_sin_cos",
    "wind_speed",
    "wind_direction_to",
]

_ABS_ZERO_F = -459.67


def kelvin_to_fahrenheit(temp_k: np.ndarray | float) -> np.ndarray:
    """K -> degrees Fahrenheit."""
    return np.asarray(temp_k, dtype=np.float64) * 9.0 / 5.0 - 459.67


def dead_fuel_moisture_simard(
    temp_k: np.ndarray | float,
    rh_pct: np.ndarray | float,
    *,
    clip: tuple[float, float] = (1.0, 60.0),
) -> np.ndarray:
    """C1 channel 11 - ``fuel_moisture_proxy``, in percent.

    Equilibrium moisture content (EMC) of fine dead fuels from air temperature
    and relative humidity, using the piecewise fit of **Simard (1968)**, as used
    by the US National Fire Danger Rating System for 1-hour timelag fuels:

    .. math::

        H < 10       &: 0.03229 + 0.281073 H - 0.000578 H T \\\\
        10 \\le H \\le 50 &: 2.22749 + 0.160107 H - 0.014784 T \\\\
        H > 50       &: 21.0606 + 0.005565 H^2 - 0.00035 H T - 0.483199 H

    with ``H`` = relative humidity (%) and ``T`` = temperature (deg F). The
    result is EMC in percent, clipped to ``clip``.

    This is a *proxy*: it ignores timelag dynamics, solar heating and
    precipitation, so it is an equilibrium estimate rather than actual fuel
    moisture. It is a deterministic function of RTMA ``temp_2m``/``rh_2m``,
    which is exactly what C1 asks for, and it is monotonically decreasing in
    temperature and increasing in humidity - the two dependencies that matter
    for spread.

    Reference: Simard, A.J. (1968), *The Moisture Content of Forest Fuels I*,
    Forest Fire Research Institute, Ottawa, Information Report FF-X-14.
    """
    t_f = np.clip(kelvin_to_fahrenheit(temp_k), _ABS_ZERO_F, 200.0)
    h = np.clip(np.asarray(rh_pct, dtype=np.float64), 0.0, 100.0)
    t_f, h = np.broadcast_arrays(t_f, h)

    low = 0.03229 + 0.281073 * h - 0.000578 * h * t_f
    mid = 2.22749 + 0.160107 * h - 0.014784 * t_f
    high = 21.0606 + 0.005565 * h**2 - 0.00035 * h * t_f - 0.483199 * h

    emc = np.where(h < 10.0, low, np.where(h <= 50.0, mid, high))
    return np.clip(emc, clip[0], clip[1]).astype(np.float32)


def slope_aspect_from_elevation(
    elevation: np.ndarray, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """C1 channels 6 and (via :func:`aspect_to_sin_cos`) 7-8.

    Returns ``(slope_deg, aspect_deg)`` for a north-up elevation raster whose
    axis 0 runs north->south and axis 1 west->east.

    Slope is ``atan(|grad z|)`` in degrees. Aspect is the compass bearing the
    slope *faces* (downhill), 0 deg = north, 90 deg = east. Gradients are
    central differences (edge-replicated), i.e. the 4-neighbour simplification
    of Horn's method - adequate at 1 km where the DEM is already heavily
    aggregated.
    """
    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"elevation must be 2-D (y, x), got {z.ndim}-D")
    dz_drow, dz_dcol = np.gradient(z, float(cell_size_m))
    dz_deast = dz_dcol
    dz_dnorth = -dz_drow  # row index increases southward

    slope_deg = np.degrees(np.arctan(np.hypot(dz_deast, dz_dnorth)))
    # Downhill direction is the negated uphill gradient; bearing = atan2(E, N).
    aspect_deg = np.degrees(np.arctan2(-dz_deast, -dz_dnorth)) % 360.0
    aspect_deg = np.where(slope_deg <= 0.0, 0.0, aspect_deg)
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def aspect_to_sin_cos(
    aspect_deg: np.ndarray, slope_deg: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """C1 channels 7-8. Flat cells (slope == 0) encode as ``(0, 0)``.

    Aspect is circular, so it is stored as its sine and cosine rather than as a
    raw angle; ``(0, 0)`` is an unambiguous "no aspect" code because it is the
    only point in the unit disc that no real bearing maps to.
    """
    a = np.radians(np.asarray(aspect_deg, dtype=np.float64))
    sin_a = np.sin(a)
    cos_a = np.cos(a)
    if slope_deg is not None:
        flat = np.asarray(slope_deg) <= 0.0
        sin_a = np.where(flat, 0.0, sin_a)
        cos_a = np.where(flat, 0.0, cos_a)
    return sin_a.astype(np.float32), cos_a.astype(np.float32)


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Wind speed (m/s) from eastward ``u`` and northward ``v`` components."""
    return np.hypot(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)).astype(
        np.float32
    )


def wind_direction_to(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compass bearing the wind is blowing *toward*, degrees, 0 = north.

    Note this is the opposite of the meteorological convention (which reports
    the direction wind blows *from*). Fire spreads toward this bearing, so this
    is the useful one here; the name says which it is.
    """
    return (
        np.degrees(np.arctan2(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)))
        % 360.0
    ).astype(np.float32)
