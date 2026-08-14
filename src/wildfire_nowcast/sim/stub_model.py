"""A C5-shaped **visualisation fixture**. NOT a baseline, NOT a model.

modelling owns C5 (``src/wildfire_nowcast/model/api.py``) and C6
(``src/wildfire_nowcast/eval/metrics.py``); both are being built in parallel
with this package. This module exists so the ensemble viewer can be written,
exercised and debugged *before* they land, and so that the day they land the
only thing that changes is which callable gets passed in.

Read the following as a warning label:

* It is a caricature — a wind-biased contagion with a shared per-step latent.
  It has no Rothermel physics, no fuels response beyond a barrier mask, and no
  calibration of any kind. **It must never appear in a gate, a report, a figure
  intended for anyone outside this repo, or a comparison against ELMFIRE.**
* Its one scientific commitment is structural, and is the project's (see
  ``README.md``): pixels are conditionally independent Bernoulli **only given a
  shared per-step latent** ``z_t``. ``latent_sigma=0`` reduces it to the
  independent-per-pixel-noise model that is known-broken by ensemble collapse,
  and that degenerate setting is retained deliberately as
  the POSITIVE CONTROL for the ensemble-collapse detector in
  :mod:`wildfire_nowcast.sim.ensemble`. A collapse detector that has never been
  shown to fire is not a detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, UNBURNED
from wildfire_nowcast.sim.c5 import STATIC_C5, WEATHER_C5

__all__ = ["StubEnsemble", "stub_predict"]

_U = WEATHER_C5.index("wind_u10")
_V = WEATHER_C5.index("wind_v10")
_RH = WEATHER_C5.index("rh_2m")
_BARRIER = STATIC_C5.index("water_barrier_mask")


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``mask`` translated by ``(dy, dx)``, zero-filled at the edges."""
    out = np.zeros_like(mask)
    ny, nx = mask.shape
    ys = slice(max(0, -dy), ny + min(0, -dy))
    xs = slice(max(0, -dx), nx + min(0, -dx))
    yd = slice(max(0, dy), ny + min(0, dy))
    xd = slice(max(0, dx), nx + min(0, dx))
    out[yd, xd] = mask[ys, xs]
    return out


def _neighbourhood(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out |= _shift(mask, dy, dx)
    return out


@dataclass
class StubEnsemble:
    """C5-shaped fixture. See the module docstring before using this anywhere.

    Parameters
    ----------
    latent_sigma
        Std of the shared per-step latent ``z_t``. ``0.0`` gives the known-broken
        independent-per-pixel model and is the collapse-detector control.
    base_rate, wind_gain
        Logit intercept and the weight on downwind alignment.
    residence_h
        Hours a cell stays in state 1 before becoming state 2. Keeps the fixture
        C1.1-shaped (absorbing, no 0->2 skip) so the viewer sees real semantics.
    """

    latent_sigma: float = 0.9
    base_rate: float = -1.4
    wind_gain: float = 1.1
    residence_h: int = 3
    name: str = "stub-contagion (viz fixture, NOT a baseline)"

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        """C5 ``predict``. Returns ``uint8[n_members, horizon_h, H, W]``."""
        x0 = np.asarray(x0, dtype=np.uint8)
        static = np.asarray(static, dtype=np.float32)
        weather = np.asarray(weather, dtype=np.float32)
        h, w = x0.shape
        if static.shape != (len(STATIC_C5), h, w):
            raise ValueError(f"static must be {(len(STATIC_C5), h, w)}, got {static.shape}")
        if weather.shape[0] < horizon_h or weather.shape[1] != len(WEATHER_C5):
            raise ValueError(
                f"weather must be (>={horizon_h}, {len(WEATHER_C5)}, {h}, {w}), got {weather.shape}"
            )

        barrier = static[_BARRIER] > 0.5
        rng = np.random.default_rng(seed)
        out = np.empty((n_members, horizon_h, h, w), dtype=np.uint8)

        for m in range(n_members):
            state = x0.copy()
            age = np.where(state == BURNING, 1, 0).astype(np.int16)
            for step in range(horizon_h):
                u = float(np.nanmean(weather[step, _U]))
                v = float(np.nanmean(weather[step, _V]))
                rh = float(np.nanmean(weather[step, _RH]))
                # ONE latent per step, shared by every pixel. This is the whole
                # reason the ensemble has usable spread.
                z = float(rng.normal(0.0, self.latent_sigma))

                ever = state > UNBURNED
                cand = _neighbourhood(ever) & ~ever & ~barrier
                if cand.any():
                    speed = float(np.hypot(u, v))
                    dy = int(np.sign(-v)) if abs(v) > 0.25 * max(speed, 1e-6) else 0
                    dx = int(np.sign(u)) if abs(u) > 0.25 * max(speed, 1e-6) else 0
                    downwind = _shift(ever, dy, dx) & cand
                    dryness = np.clip((60.0 - rh) / 60.0, -1.0, 1.0)
                    logit = np.full((h, w), self.base_rate + z + 0.8 * dryness, dtype=np.float64)
                    logit += self.wind_gain * np.minimum(speed / 8.0, 2.0) * downwind
                    p = 1.0 / (1.0 + np.exp(-logit))
                    new = cand & (rng.random((h, w)) < p)
                else:
                    new = np.zeros((h, w), dtype=bool)

                age = np.where(state == BURNING, age + 1, age)
                still = (state == BURNING) & (age <= self.residence_h)
                state = np.where(
                    new,
                    BURNING,
                    np.where(still, BURNING, np.where(state > 0, BURNED_OUT, UNBURNED)),
                ).astype(np.uint8)
                age = np.where(new, 1, age).astype(np.int16)
                out[m, step] = state
        return out


def stub_predict(
    x0: np.ndarray,
    static: np.ndarray,
    weather: np.ndarray,
    n_members: int,
    horizon_h: int,
    seed: int,
) -> np.ndarray:
    """Module-level C5 callable using :class:`StubEnsemble` defaults."""
    return StubEnsemble().predict(x0, static, weather, n_members, horizon_h, seed)
