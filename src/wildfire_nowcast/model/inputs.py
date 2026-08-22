"""C5 input assembly - turning a C1 tensor into ``predict()`` arguments.

C5 fixes the *shapes* of ``predict``'s arguments but not their *content*::

    predict(x0: uint8[H,W], static: f32[C_s,H,W], weather: f32[T,C_w,H,W], ...)

Three things are therefore unspecified in the contract and are pinned HERE, once,
so that a baseline, the learned kernel and sim's driver all agree. If any
of the three drifts, every comparison at G5 is apples-to-oranges and nothing
warns you.

1. **Which channels, in what order.** ``C_s`` is :data:`STATIC_INPUT_CHANNELS`
   and ``C_w`` is :data:`WEATHER_INPUT_CHANNELS`, both in C1 index order and
   both derived from :mod:`wildfire_nowcast.common.contract` rather than
   retyped. Index them with :func:`static_index` / :func:`weather_index`; never
   with a literal integer. (The C1 store is two variables with
   ``channel_index_offset: 1``, so a literal is off by one *and* the split into
   static/weather renumbers again - two independent ways to silently read the
   wrong field.)

2. **Units.** ``static`` and ``weather`` carry RAW C1 PHYSICAL UNITS - m/s, K,
   %, degrees, FBFM40 class ids. They are NOT normalised. A baseline needs real
   wind speeds to compute a real rate of spread, and C3 says a model reads
   ``data/norm_stats.json`` itself, so normalisation belongs inside the model,
   after this boundary, not before it.

3. **Time phase.** This is the one C1.3 calls silently catastrophic. Time index
   ``i`` labels the hour ENDING at ``time[i]`` (``time_convention:
   end_of_hour``). So the step that carries the fire from ``x0 = state[t0]`` to
   the first predicted hour is driven by the weather of the hour ending at
   ``time[t0 + 1]``::

       weather[k]  <-  features[t0 + 1 + k]      # drives predicted step k
       truth[k]    <-  fire_state[t0 + 1 + k]    # the label for step k

   Off by one here and every model trains an hour out of phase with its
   weather, which presents as a mediocre model rather than as a bug. Use
   :func:`forecast_inputs` and you cannot get it wrong.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from wildfire_nowcast.common.contract import (
    BURNING,
    CHANNELS,
    FEATURE_CHANNELS,
    STATIC_CHANNELS,
    UNBURNED,
)
from wildfire_nowcast.common.zarr_io import channel_values, get_channel

__all__ = [
    "STATIC_INPUT_CHANNELS",
    "WEATHER_INPUT_CHANNELS",
    "N_STATIC",
    "N_WEATHER",
    "static_index",
    "weather_index",
    "ForecastWindow",
    "static_from_dataset",
    "weather_from_dataset",
    "forecast_inputs",
    "iter_windows",
    "growth_cells_per_step",
]

#: ``C_s`` - the static half of C1's feature channels, in C1 index order.
#: (5 elevation, 6 slope, 7 aspect_sin, 8 aspect_cos, 9 fuel_model_id,
#: 10 canopy_cover, 12 water_barrier_mask, 13 recent_burn_scar)
STATIC_INPUT_CHANNELS: tuple[str, ...] = tuple(c for c in CHANNELS if c in STATIC_CHANNELS)

#: ``C_w`` - the time-varying half, in C1 index order.
#: (1 wind_u10, 2 wind_v10, 3 temp_2m, 4 rh_2m, 11 fuel_moisture_proxy)
WEATHER_INPUT_CHANNELS: tuple[str, ...] = tuple(
    c for c in FEATURE_CHANNELS if c not in STATIC_CHANNELS
)

N_STATIC = len(STATIC_INPUT_CHANNELS)
N_WEATHER = len(WEATHER_INPUT_CHANNELS)

_STATIC_POS = {name: i for i, name in enumerate(STATIC_INPUT_CHANNELS)}
_WEATHER_POS = {name: i for i, name in enumerate(WEATHER_INPUT_CHANNELS)}


def static_index(name: str) -> int:
    """Position of ``name`` along ``static``'s channel axis."""
    try:
        return _STATIC_POS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a static input channel; expected one of {STATIC_INPUT_CHANNELS}"
        ) from None


def weather_index(name: str) -> int:
    """Position of ``name`` along ``weather``'s channel axis."""
    try:
        return _WEATHER_POS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a weather input channel; expected one of {WEATHER_INPUT_CHANNELS}"
        ) from None


@dataclass(frozen=True)
class ForecastWindow:
    """One evaluable ``(inputs, labels)`` window, with its phase already applied.

    Attributes
    ----------
    x0
        ``uint8[H,W]`` fire_state at the analysis time ``time[t0]``.
    static
        ``f32[C_s,H,W]`` in :data:`STATIC_INPUT_CHANNELS` order, raw units.
    weather
        ``f32[horizon_h,C_w,H,W]`` in :data:`WEATHER_INPUT_CHANNELS` order, raw
        units. ``weather[k]`` is the hour that DRIVES predicted step ``k``.
    truth
        ``uint8[horizon_h,H,W]`` fire_state at ``time[t0+1 .. t0+horizon_h]``,
        aligned element-for-element with a ``samples[member]`` trajectory.
    t0
        Index of the analysis time in the source dataset.
    times
        ``datetime64[s][horizon_h]`` end-of-hour stamps of the predicted steps.
    fire_id
        Provenance, so a window can never be mixed up between fires.
    """

    x0: np.ndarray
    static: np.ndarray
    weather: np.ndarray
    truth: np.ndarray
    t0: int
    times: np.ndarray
    fire_id: str = ""

    @property
    def horizon_h(self) -> int:
        return int(self.weather.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.x0.shape[0]), int(self.x0.shape[1]))

    def predict_args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x0, static, weather)`` - the first three C5 positional arguments."""
        return self.x0, self.static, self.weather

    def truth_growth_cells(self) -> int:
        """Cells that newly burn across the whole window.

        Zero for ~4 windows in 5 (insights/data item 1: 51-91% of GOFER hours
        have BITWISE zero growth, median ~0.79). That is an observation
        artefact -- GOES cannot see new front at night or under cloud -- not the
        fire stopping, so a zero here is not evidence of a barrier and must not
        be scored as if the model were asked an interesting question.
        """
        before = self.x0 > UNBURNED
        after = self.truth[-1] > UNBURNED
        return int(np.count_nonzero(after & ~before))


def static_from_dataset(ds: xr.Dataset, t_index: int = 0) -> np.ndarray:
    """``f32[C_s,H,W]`` in :data:`STATIC_INPUT_CHANNELS` order.

    C1.5 enforces that static channels really are constant over time, so
    ``t_index`` is a formality; it is exposed only so a caller debugging a
    suspect tensor can read a specific hour.
    """
    return np.stack(
        [channel_values(ds, name, dtype=np.float32)[t_index] for name in STATIC_INPUT_CHANNELS],
        axis=0,
    ).astype(np.float32, copy=False)


def weather_from_dataset(ds: xr.Dataset, t0: int, horizon_h: int) -> np.ndarray:
    """``f32[horizon_h,C_w,H,W]`` - the hours that DRIVE steps ``0..horizon_h-1``.

    Applies the C1.3 end-of-hour phase: ``weather[k]`` is ``features[t0+1+k]``.
    """
    if horizon_h < 1:
        raise ValueError(f"horizon_h must be >= 1, got {horizon_h}")
    n_t = int(ds.sizes["time"])
    stop = t0 + 1 + horizon_h
    if t0 < 0 or stop > n_t:
        raise IndexError(
            f"window t0={t0} horizon={horizon_h} needs feature indices "
            f"{t0 + 1}..{stop - 1} but the store has {n_t} hours"
        )
    slabs = [
        get_channel(ds, name).isel(time=slice(t0 + 1, stop)).values.astype(np.float32, copy=False)
        for name in WEATHER_INPUT_CHANNELS
    ]
    return np.stack(slabs, axis=1).astype(np.float32, copy=False)


def forecast_inputs(
    ds: xr.Dataset,
    t0: int,
    horizon_h: int = 3,
    *,
    fire_id: str = "",
) -> ForecastWindow:
    """Build one :class:`ForecastWindow` from a C1 store. The phase is applied here."""
    n_t = int(ds.sizes["time"])
    if horizon_h < 1:
        raise ValueError(f"horizon_h must be >= 1, got {horizon_h}")
    if t0 < 0 or t0 + horizon_h >= n_t:
        raise IndexError(
            f"t0={t0} with horizon_h={horizon_h} runs past the end of a {n_t}-hour store "
            f"(need t0 + horizon_h <= {n_t - 1})"
        )
    state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    return ForecastWindow(
        x0=state[t0],
        static=static_from_dataset(ds),
        weather=weather_from_dataset(ds, t0, horizon_h),
        truth=state[t0 + 1 : t0 + 1 + horizon_h],
        t0=int(t0),
        times=np.asarray(ds["time"].values[t0 + 1 : t0 + 1 + horizon_h], dtype="datetime64[s]"),
        fire_id=str(fire_id or ds.attrs.get("fire_id", "")),
    )


def iter_windows(
    ds: xr.Dataset,
    horizon_h: int = 3,
    *,
    stride: int = 1,
    require_ignited: bool = True,
    growth_only: bool = False,
    fire_id: str = "",
) -> Iterator[ForecastWindow]:
    """Every leave-fire-out evaluable window in a store.

    Parameters
    ----------
    require_ignited
        Skip windows whose ``x0`` has nothing burned at all. Nowcasting starts
        from an observed fire; scoring "will an unignited landscape ignite" is a
        different (and much easier) problem that would flatter every model.
    growth_only
        Keep only windows where the truth actually grows. **Never use this to
        produce a headline number** -- it conditions the evaluation set on the
        outcome. It exists so the growth-conditioned stratum can be reported
        ALONGSIDE the all-windows number (insights/data item 1), because a
        pooled score over raw hourly steps is ~79% "nothing happened" and
        persistence wins it.
    """
    n_t = int(ds.sizes["time"])
    for t0 in range(0, n_t - horizon_h, max(1, int(stride))):
        window = forecast_inputs(ds, t0, horizon_h, fire_id=fire_id)
        if require_ignited and not np.any(window.x0 > UNBURNED):
            continue
        if growth_only and window.truth_growth_cells() == 0:
            continue
        yield window


def growth_cells_per_step(state: np.ndarray) -> np.ndarray:
    """Newly-burned cell count for each hourly step of a ``(T,H,W)`` state field.

    Length ``T-1``. The fraction of zeros here is the single most important
    property of the labels (insights/data item 1) and every evaluation should
    report it next to its scores.
    """
    arr = np.asarray(state)
    burned = arr > UNBURNED
    return np.count_nonzero(burned[1:] & ~burned[:-1], axis=(1, 2)).astype(np.int64)


def contagion_seed(x0: np.ndarray) -> np.ndarray:
    """The frontier of the BURNED REGION -- the contagion source under C1.1.

    C1.1 is explicit that state 1 is legitimately EMPTY in 6-37% of frames after
    a long dormancy, and no absorbing rule can avoid it. A kernel (or a
    baseline) that seeds from ``state == 1`` therefore has nothing to propagate
    from in a large minority of steps and will silently predict "no growth"
    exactly when it is least justified. Seed from this instead: every burned
    cell adjacent to an unburned one.
    """
    arr = np.asarray(x0)
    burned = arr > UNBURNED
    if not burned.any():
        return np.zeros_like(burned)
    interior = np.ones_like(burned)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        rolled = np.roll(burned, shift, axis=axis)
        # A cell on the domain edge has no neighbour there; treat off-grid as
        # unburned so the domain boundary counts as frontier rather than as
        # interior (over-seeding is recoverable, under-seeding is not).
        if axis == 0:
            rolled[0 if shift == 1 else -1, :] = False
        else:
            rolled[:, 0 if shift == 1 else -1] = False
        interior &= rolled
    return burned & ~interior


def describe_inputs() -> dict[str, Any]:
    """Machine-readable description of the C5 input convention.

    Written into every run directory so a result carries its own definition of
    what ``static``/``weather`` meant when it was produced.
    """
    return {
        "static_channels": list(STATIC_INPUT_CHANNELS),
        "weather_channels": list(WEATHER_INPUT_CHANNELS),
        "units": "raw C1 physical units; NOT normalised (C3 normalisation is the model's job)",
        "time_phase": "weather[k] and truth[k] are the hour ENDING at time[t0+1+k] (C1.3)",
        "fire_state_values": {"unburned": UNBURNED, "burning": BURNING, "burned_out": 2},
    }


def _assert_channel_partition() -> None:
    """Static + weather must be exactly C1's 13 feature channels, no overlap."""
    union = set(STATIC_INPUT_CHANNELS) | set(WEATHER_INPUT_CHANNELS)
    overlap = set(STATIC_INPUT_CHANNELS) & set(WEATHER_INPUT_CHANNELS)
    if overlap or union != set(FEATURE_CHANNELS):
        raise AssertionError(
            "static/weather split is not a partition of C1's feature channels: "
            f"overlap={sorted(overlap)} missing={sorted(set(FEATURE_CHANNELS) - union)}"
        )


_assert_channel_partition()


def channel_report(names: Sequence[str]) -> str:
    """``'0 elevation, 1 slope, ...'`` -- for error messages and logs."""
    return ", ".join(f"{i} {n}" for i, n in enumerate(names))
