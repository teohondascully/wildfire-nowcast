"""Adapter from a C1 tensor to the C5 ``predict()`` call signature.

C5 is::

    predict(x0: uint8[H,W], static: f32[C_s,H,W], weather: f32[T,C_w,H,W],
            n_members: int, horizon_h: int, seed: int)
        -> samples: uint8[n_members, horizon_h, H, W]

**C5 names the shapes but not the contents.** It does not say which C1 channels
are "static", which are "weather", in what order, or how ``weather``'s time axis
lines up with ``x0``. Two leads implementing against it independently — which is
exactly what is happening — will each pick something reasonable and the pair
will be silently wrong: a swapped ``aspect_sin``/``aspect_cos`` or an
off-by-one weather index does not raise, it just makes the model look mediocre.
The gap is raised in ``docs/decisions.md`` (modelling).

Until it is ratified, this module implements the only choice that follows
mechanically from the contract itself, so it can be checked rather than
remembered:

* ``static``  = ``FEATURE_CHANNELS ∩ STATIC_CHANNELS``, in C1 channel order.
  8 channels: elevation, slope, aspect_sin, aspect_cos, fuel_model_id,
  canopy_cover, water_barrier_mask, recent_burn_scar.
* ``weather`` = ``FEATURE_CHANNELS \\ STATIC_CHANNELS``, in C1 channel order.
  5 channels: wind_u10, wind_v10, temp_2m, rh_2m, fuel_moisture_proxy.
* ``weather[i]`` is the weather driving the step that PRODUCES ``samples[:, i]``.
  Under C1.3 times are END-OF-HOUR, so with ``x0 = fire_state[t0]`` that is
  ``weather_channels[t0 + 1 + i]``, and ``T == horizon_h``. Taking
  ``weather[0] = t0`` instead trains every fire one hour out of phase with its
  weather — ADR-006 calls this out as silently catastrophic, and it is the same
  off-by-one wearing a different hat.

:data:`C5_CONVENTION` records all of the above in a form a consumer can assert
against, so a future disagreement is a failed assertion rather than a plot that
looks a bit off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr

from wildfire_nowcast.common.contract import FEATURE_CHANNELS, STATIC_CHANNELS
from wildfire_nowcast.common.zarr_io import channel_values, get_channel

__all__ = [
    "STATIC_C5",
    "WEATHER_C5",
    "C5_CONVENTION",
    "C5Inputs",
    "PredictFn",
    "c5_inputs",
]

#: C1 static channels, in C1 channel order. ``C_s = 8``.
STATIC_C5: tuple[str, ...] = tuple(c for c in FEATURE_CHANNELS if c in STATIC_CHANNELS)

#: C1 time-varying feature channels, in C1 channel order. ``C_w = 5``.
WEATHER_C5: tuple[str, ...] = tuple(c for c in FEATURE_CHANNELS if c not in STATIC_CHANNELS)

C5_CONVENTION: dict[str, Any] = {
    "static_channels": list(STATIC_C5),
    "weather_channels": list(WEATHER_C5),
    "c_s": len(STATIC_C5),
    "c_w": len(WEATHER_C5),
    "weather_time_origin": "t0+1",
    "weather_time_len": "horizon_h",
    "note": (
        "PROVISIONAL — C5 does not specify channel membership, order, or the weather time "
        "origin. Derived here from common.contract STATIC_CHANNELS so it is reproducible, not "
        "invented. See BLOCKERS.md '@model C5 does not specify static/weather membership'."
    ),
}


@runtime_checkable
class PredictFn(Protocol):
    """The C5 callable. Baselines and the learned model share this signature."""

    def __call__(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class C5Inputs:
    """Everything one C5 call needs, plus the truth to score it against."""

    x0: np.ndarray  # uint8 (H, W)     — state at t0
    static: np.ndarray  # f32   (C_s, H, W)
    weather: np.ndarray  # f32   (T, C_w, H, W), T == horizon_h
    truth: np.ndarray  # uint8 (horizon_h, H, W) — states t0+1 .. t0+horizon_h
    t0: int
    horizon_h: int
    times: np.ndarray  # datetime64 (horizon_h,) — stamps of the predicted hours

    def check(self) -> None:
        """Assert the shape relations C5 implies. Cheap, and catches adapter drift."""
        h, w = self.x0.shape
        assert self.static.shape == (len(STATIC_C5), h, w), self.static.shape
        assert self.weather.shape == (self.horizon_h, len(WEATHER_C5), h, w), self.weather.shape
        assert self.truth.shape == (self.horizon_h, h, w), self.truth.shape


def c5_inputs(ds: xr.Dataset, t0: int, horizon_h: int) -> C5Inputs:
    """Slice a C1 store into one C5 ``predict()`` call at ``t0``.

    Channels are read BY NAME (never by an integer index into ``features``).
    """
    n_t = int(ds.sizes["time"])
    if not 0 <= t0 < n_t - 1:
        raise IndexError(f"t0={t0} leaves no future to predict in a {n_t}-hour store")
    if horizon_h < 1:
        raise ValueError(f"horizon_h must be >= 1, got {horizon_h}")
    if t0 + horizon_h >= n_t:
        raise IndexError(
            f"t0={t0} + horizon_h={horizon_h} runs past the store's last hour ({n_t - 1}). "
            "Refusing to pad: a padded truth silently scores the model against invented data."
        )

    state = np.asarray(get_channel(ds, "fire_state").values, dtype=np.uint8)
    static = np.stack([channel_values(ds, c, dtype=np.float32)[0] for c in STATIC_C5], axis=0)
    weather = np.stack(
        [channel_values(ds, c, dtype=np.float32)[t0 + 1 : t0 + 1 + horizon_h] for c in WEATHER_C5],
        axis=1,
    )
    out = C5Inputs(
        x0=state[t0],
        static=static.astype(np.float32, copy=False),
        weather=weather.astype(np.float32, copy=False),
        truth=state[t0 + 1 : t0 + 1 + horizon_h],
        t0=int(t0),
        horizon_h=int(horizon_h),
        times=np.asarray(ds["time"].values)[t0 + 1 : t0 + 1 + horizon_h],
    )
    out.check()
    return out
