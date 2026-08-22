"""[M10] The CONTROL arms - the things that license a metric channel to speak.

ADR-045 (4): *"an indistinguishability claim is exactly the claim this project has
gotten wrong most often"*. M10's likely outcome is "B is indistinguishable from
A", and a null result is only a result if the instrument has been shown to be
capable of detecting a difference **on the same data, in the same run, through the
same code**. So each metric channel carries its own control:

    dispersion channel  <- C1, the latent held at its prior mean
    accuracy channel    <- C2, the wind covariates zeroed

If a control does not separate from A, that channel returns ``not_a_verdict`` and
NO conclusion is drawn from it - not "no difference". Three-valued, exactly as
C6.5 made the G3 conditions three-valued.

C1 needs no code: :meth:`ContagionKernel.with_sampler` already produces the
shared-parameter ablation (same fit, same seed, ``z`` at its prior mean). The
project's scientific ground truth (see README.md) admits an
independent-per-pixel-noise-only sampler *only* as an ablation, never as a
candidate, because it is known to collapse the ensemble; this arm is that
ablation and nothing else.

C2 is here. It is applied at PREDICTION TIME rather than by retraining, and that
choice is the point: the control has to isolate the CHANNEL, and a retrained
no-wind model would differ from A in its fitted parameters as well as in its
inputs, so a separation could not be attributed to wind. Zeroing at prediction
time leaves exactly one thing different.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wildfire_nowcast.model.api import validate_predict_inputs
from wildfire_nowcast.model.inputs import weather_index

__all__ = ["ZEROED_WEATHER_CHANNELS", "ZeroedCovariateModel"]

#: The covariates C2 removes. Named here, once, so the arm cannot silently change
#: what it is a control FOR.
ZEROED_WEATHER_CHANNELS: tuple[str, ...] = ("wind_u10", "wind_v10")


class ZeroedCovariateModel:
    """A C5 predictor that blanks named weather channels before delegating.

    Holds the wrapped model's ``provenance`` so C8's split check reads the
    checkpoint's real fingerprint rather than being exempted - an unstamped
    control is an unverifiable control, and C-1 makes unverifiable a failure.
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str,
        channels: tuple[str, ...] = ZEROED_WEATHER_CHANNELS,
    ) -> None:
        self.model = model
        self.name = name
        self.channels = tuple(channels)
        self._indices = tuple(weather_index(c) for c in self.channels)
        # Not a copy of a summary: the same dict contents, so a C8 mismatch on the
        # wrapped checkpoint is a C8 mismatch here.
        self.provenance: dict[str, Any] = dict(getattr(model, "provenance", {}) or {})
        self.provenance["control"] = (
            f"M10 arm C2 — weather channels {list(self.channels)} ZEROED at prediction time. "
            "Same fit as the arm it wraps; the ONLY difference is the input."
        )

    @property
    def kind(self) -> str:
        return f"zeroed_covariate_control({getattr(self.model, 'kind', '?')})"

    def _blank(self, weather: np.ndarray) -> np.ndarray:
        out = np.array(weather, copy=True)
        for index in self._indices:
            out[:, index] = 0.0
        return out

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        """C5 ``predict``, with the named channels zeroed. Seed-exact, like the wrapped model."""
        validate_predict_inputs(x0, static, weather, n_members, horizon_h, seed)
        return self.model.predict(x0, static, self._blank(weather), n_members, horizon_h, seed)

    def predict_proba(
        self, x0: np.ndarray, static: np.ndarray, weather: np.ndarray, horizon_h: int, **kwargs: Any
    ) -> np.ndarray:
        return self.model.predict_proba(x0, static, self._blank(weather), horizon_h, **kwargs)

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "zeroed_weather_channels": list(self.channels),
            "wrapped": self.model.to_spec() if hasattr(self.model, "to_spec") else None,
        }
