"""Model-lead's package: the C5 prediction API, baselines, and the kernel.

Import surface is deliberately thin - :func:`predict` and :func:`load_model` are
the C5 contract, and everything else (input assembly, spread physics, the
baselines themselves) is reached through its own module so that a caller always
names what it is depending on.
"""

from __future__ import annotations

from wildfire_nowcast.model.api import Predictor, available_models, load_model, predict

__all__ = ["Predictor", "predict", "load_model", "available_models"]
