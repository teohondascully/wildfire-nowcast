"""C5 - the model prediction API. Baselines and the learned kernel share it.

INTERFACES C5::

    predict(x0: uint8[H,W], static: f32[C_s,H,W], weather: f32[T,C_w,H,W],
            n_members: int, horizon_h: int, seed: int)
        -> samples: uint8[n_members, horizon_h, H, W]
    load_model(path) -> object exposing predict
    Baselines (persistence, ellipse) implement the SAME signature.

"the SAME signature" is the load-bearing clause. sim consumes every
predictor identically and the G5 head-to-head against ELMFIRE depends on there
being exactly one calling convention, so this module defines it as a
:class:`Predictor` protocol and validates both ends of it. The channel order,
units and time phase of ``static``/``weather`` are pinned in
:mod:`wildfire_nowcast.model.inputs`; read that module's header before
constructing arguments by hand, and prefer
:func:`~wildfire_nowcast.model.inputs.forecast_inputs`, which applies the C1.3
end-of-hour phase for you.

Note on the free function. C5 lists ``predict`` under ``model/api.py`` with the
six positional arguments above, and separately says ``load_model`` returns an
object exposing ``predict``. A free function cannot predict without something to
predict *with*, so :func:`predict` takes the six C5 arguments positionally, in
order, plus a keyword-only ``model``. A predictor's bound method has exactly the
C5 signature and nothing else. (Flagged as a PROPOSAL in status/model.md rather
than assumed: this is the one place the C5 text is ambiguous.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from wildfire_nowcast.common.contract import FIRE_STATE_VALUES
from wildfire_nowcast.model.inputs import N_STATIC, N_WEATHER, channel_report
from wildfire_nowcast.model.inputs import (
    STATIC_INPUT_CHANNELS as _STATIC,
)
from wildfire_nowcast.model.inputs import (
    WEATHER_INPUT_CHANNELS as _WEATHER,
)

__all__ = [
    "Predictor",
    "predict",
    "load_model",
    "save_model",
    "available_models",
    "validate_predict_inputs",
    "validate_samples",
    "SPEC_NAME",
]

#: Filename a saved predictor writes into its directory.
SPEC_NAME = "model.json"


@runtime_checkable
class Predictor(Protocol):
    """Anything that can be scored through C6. The learned kernel, every
    baseline, and (via a shim) ELMFIRE all satisfy this."""

    #: Short stable identifier used in run directories and figure legends.
    name: str

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        """``uint8[n_members, horizon_h, H, W]`` of fire_state trajectories."""
        ...


# --------------------------------------------------------------------------
# validation - applied on both sides of the boundary
# --------------------------------------------------------------------------


def validate_predict_inputs(
    x0: np.ndarray,
    static: np.ndarray,
    weather: np.ndarray,
    n_members: int,
    horizon_h: int,
    seed: int,
) -> tuple[int, int]:
    """Check the C5 argument contract; return ``(H, W)``.

    Deliberately strict. Every one of these has a silent-wrong-answer failure
    mode: a transposed ``static``, a ``weather`` slab one hour out of phase
    (C1.3), a float ``x0`` that quietly compares ``> 0`` on a probability. An
    exception now is worth a week of "the model is mediocre" later.
    """
    x0_arr = np.asarray(x0)
    if x0_arr.ndim != 2:
        raise ValueError(f"x0 must be [H,W] fire_state, got shape {x0_arr.shape}")
    if x0_arr.dtype != np.uint8:
        raise TypeError(
            f"x0 must be uint8 fire_state (C5), got {x0_arr.dtype}. A float x0 is usually a "
            "probability field that has been passed where a state was expected."
        )
    bad = np.setdiff1d(np.unique(x0_arr), np.asarray(FIRE_STATE_VALUES, dtype=np.uint8))
    if bad.size:
        raise ValueError(f"x0 has values outside {FIRE_STATE_VALUES}: {bad.tolist()}")
    height, width = int(x0_arr.shape[0]), int(x0_arr.shape[1])

    static_arr = np.asarray(static)
    if static_arr.shape != (N_STATIC, height, width):
        raise ValueError(
            f"static must be [C_s,H,W] = {(N_STATIC, height, width)}, got {static_arr.shape}. "
            f"C_s is ({channel_report(_STATIC)})"
        )

    weather_arr = np.asarray(weather)
    if weather_arr.ndim != 4 or weather_arr.shape[1:] != (N_WEATHER, height, width):
        raise ValueError(
            f"weather must be [T,C_w,H,W] with C_w,H,W = {(N_WEATHER, height, width)}, got "
            f"{weather_arr.shape}. C_w is ({channel_report(_WEATHER)})"
        )
    if int(horizon_h) < 1:
        raise ValueError(f"horizon_h must be >= 1, got {horizon_h}")
    if weather_arr.shape[0] < int(horizon_h):
        raise ValueError(
            f"weather covers {weather_arr.shape[0]} steps but horizon_h={horizon_h}. "
            "weather[k] is the hour that DRIVES step k (C1.3 end-of-hour phase); a short "
            "slab means the last steps would be forced with the wrong hour."
        )
    if int(n_members) < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members}")
    int(seed)  # raises TypeError on a non-integer seed rather than silently reseeding
    if not np.all(np.isfinite(np.asarray(weather_arr, dtype=np.float64))):
        raise ValueError("weather contains non-finite values; C1.5 requires finite features")
    return height, width


def validate_samples(
    samples: np.ndarray,
    x0: np.ndarray,
    n_members: int,
    horizon_h: int,
) -> np.ndarray:
    """Check a predictor's OUTPUT against C5 and against C1.1's state algebra.

    Fire is absorbing (CLAUDE.md ground truth), so a trajectory that decreases
    in time, or that unburns a cell that was already burned at ``x0``, is
    invalid regardless of which model produced it. Catching that here means the
    metrics never have to defend against it, and a broken sampler fails loudly
    in its own code rather than as a strange reliability diagram.
    """
    arr = np.asarray(samples)
    height, width = np.asarray(x0).shape
    expected = (int(n_members), int(horizon_h), int(height), int(width))
    if arr.shape != expected:
        raise ValueError(f"samples must be {expected}, got {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"samples must be uint8 fire_state (C5), got {arr.dtype}")
    bad = np.setdiff1d(np.unique(arr), np.asarray(FIRE_STATE_VALUES, dtype=np.uint8))
    if bad.size:
        raise ValueError(f"samples have values outside {FIRE_STATE_VALUES}: {bad.tolist()}")
    if arr.shape[1] >= 2 and not bool(np.all(arr[:, 1:] >= arr[:, :-1])):
        raise ValueError("samples decrease in time; fire is absorbing (0 -> 1 -> 2 only)")
    if not bool(np.all(arr[:, 0] >= np.asarray(x0)[None])):
        raise ValueError("samples unburn a cell that is already burned in x0; fire is absorbing")
    return arr


# --------------------------------------------------------------------------
# the C5 free function
# --------------------------------------------------------------------------


def predict(
    x0: np.ndarray,
    static: np.ndarray,
    weather: np.ndarray,
    n_members: int,
    horizon_h: int,
    seed: int,
    *,
    model: Predictor | str,
) -> np.ndarray:
    """C5 ``predict``: sample ``n_members`` fire_state trajectories.

    ``model`` may be a :class:`Predictor` or a registered baseline name
    (``"persistence"``, ``"ellipse"``). Inputs and outputs are validated, so
    this is also the recommended way to call a predictor you did not write.
    """
    resolved = load_model(model) if isinstance(model, str) else model
    validate_predict_inputs(x0, static, weather, n_members, horizon_h, seed)
    samples = resolved.predict(x0, static, weather, int(n_members), int(horizon_h), int(seed))
    return validate_samples(samples, x0, int(n_members), int(horizon_h))


# --------------------------------------------------------------------------
# checkpoint / baseline loading
# --------------------------------------------------------------------------


def _registry() -> dict[str, Any]:
    # Imported lazily: baselines import this module for validation, and the
    # learned kernel pulls in torch (~1 s) that a baseline-only caller does not
    # need to pay for.
    from wildfire_nowcast.model.baselines import BASELINES

    registry = dict(BASELINES)
    try:
        from wildfire_nowcast.model.kernel import ContagionKernel

        registry[ContagionKernel.kind] = ContagionKernel
    except ImportError:  # pragma: no cover - torch missing
        pass
    return registry


def available_models() -> list[str]:
    """Names :func:`load_model` accepts without a path."""
    return sorted(_registry())


def load_model(path: str | Path) -> Predictor:
    """C5 ``load_model``: a baseline name, or a directory holding ``model.json``.

    A saved predictor is a directory, not a file, so a learned checkpoint can
    put weights beside its spec later without changing this signature.
    """
    key = str(path)
    registry = _registry()
    if key in registry:
        return registry[key]()

    candidate = Path(path)
    spec_path = candidate / SPEC_NAME if candidate.is_dir() else candidate
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"no model at {path!r}: not a registered baseline ({', '.join(sorted(registry))}) "
            f"and no {SPEC_NAME} at {spec_path}"
        )
    spec = json.loads(spec_path.read_text())
    kind = spec.get("kind")
    if kind not in registry:
        raise ValueError(
            f"{spec_path} declares kind={kind!r}, which is not a known predictor "
            f"({', '.join(sorted(registry))})"
        )
    return registry[kind].from_spec(spec)


def save_model(model: Any, path: str | Path) -> Path:
    """Write ``{path}/model.json`` for a predictor that exposes ``to_spec()``."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / SPEC_NAME
    spec_path.write_text(json.dumps(model.to_spec(), indent=2, sort_keys=False) + "\n")
    return spec_path
