"""C5 - the model prediction API. Baselines and the learned kernel share it.

INTERFACES C5::

    predict(x0: uint8[H,W], static: f32[C_s,H,W], weather: f32[T,C_w,H,W],
            n_members: int, horizon_h: int, seed: int)
        -> samples: uint8[n_members, horizon_h, H, W]
    load_model(path) -> object exposing predict
    load_model(f"{path}__independent") -> that model's ABLATION ARM
    Baselines (persistence, ellipse) implement the SAME signature.

THE ABLATION ARM IS PART OF THIS INTERFACE, and it has to be, because every
C5-shaped consumer obtains a predictor BY NAME. G3 (d) asks whether removing the
shared latent collapses the ensemble; that is a question about the model, so the
arm it is asked of must be the model. ``<address>__independent`` resolves by
loading ``<address>`` and taking a parameter-sharing view of it, and
:func:`ablation_arm` refuses to return an arm whose parameters are not the very
objects the model holds. A consumer that could only reach the ablation by
constructing a kernel in Python would silently fall back to whatever fixture it
had, which is how a visualisation stub came to stand in for the model in every
collapse number this repository held (ADR-118, ADR-119).

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
C5 signature and nothing else. (Flagged and escalated as a PROPOSAL rather
than assumed: this is the one place the C5 text is ambiguous.)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from wildfire_nowcast.common.contract import FIRE_STATE_VALUES
from wildfire_nowcast.model.inputs import N_STATIC, N_WEATHER, channel_report
from wildfire_nowcast.model.inputs import (
    STATIC_INPUT_CHANNELS as _STATIC,
)
from wildfire_nowcast.model.inputs import (
    WEATHER_INPUT_CHANNELS as _WEATHER,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ABLATION_ARM_MODE",
    "ABLATION_ARM_SUFFIX",
    "ARM_SEPARATOR",
    "Predictor",
    "ablation_arm",
    "ablation_arm_is_demonstrative",
    "arm_name",
    "assert_ablation_arm_is_demonstrative",
    "available_models",
    "load_model",
    "predict",
    "save_model",
    "split_arm_name",
    "validate_predict_inputs",
    "validate_samples",
    "SPEC_NAME",
]

#: Filename a saved predictor writes into its directory.
SPEC_NAME = "model.json"

#: Separates a model's identifier from the SAMPLER ARM drawn off it.
#:
#: There are two name spaces in C5 and they used to disagree. A model has an
#: ADDRESS (the string ``load_model`` accepts) and a LABEL (``predictor.name``,
#: which reaches run directories and figure legends). Before this module owned
#: the join, ``load_model("contagion_kernel")`` returned an object whose label
#: was ``"kernel"``, so the arm derived from it was addressed
#: ``contagion_kernel__independent`` and labelled ``kernel__independent`` - two
#: spellings for one thing, and neither round-tripped. The suffix below is now
#: the ONLY spelling: :func:`load_model` resolves with it and
#: :meth:`~wildfire_nowcast.model.kernel.ContagionKernel.with_sampler` labels
#: with it, and the kernel's default label is its ``kind``, so address == label
#: for the registry model and for its arm.
ARM_SEPARATOR: Final[str] = "__"

#: The sampler mode that IS the G3 ablation: ``z_t`` held at its prior mean, so
#: the only randomness left is independent per-pixel Bernoulli. That model is
#: known-broken here - independent per-pixel noise averages out over thousands
#: of cells, so the ensemble has no area spread left - and exists ONLY as the
#: ablation. It is never a candidate.
ABLATION_ARM_MODE: Final[str] = "independent"

#: ``<model>__independent``. Appending this to any address ``load_model``
#: accepts addresses that model's latent-off ablation arm.
ABLATION_ARM_SUFFIX: Final[str] = ARM_SEPARATOR + ABLATION_ARM_MODE


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

    ``model`` may be a :class:`Predictor` or any address :func:`load_model`
    accepts: a registered baseline name (``"persistence"``, ``"ellipse"``), a
    checkpoint directory, or either with :data:`ABLATION_ARM_SUFFIX` appended
    for the latent-off ablation arm. Inputs and outputs are validated, so this
    is also the recommended way to call a predictor you did not write.
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
        # ADR-103 (4). This swallow has a real and silent consequence: the
        # learned kernel simply vanishes from `available_models()`, so
        # `load_model("contagion")` then fails with "unknown model" rather than
        # with "torch is not installed", which sends the reader to the wrong
        # place. WARNING rather than DEBUG for exactly that reason - the cost is
        # one stderr line to a caller who deliberately has no torch, and the
        # benefit is that a missing optional dependency stops impersonating a
        # typo in a model name.
        logger.warning(
            "torch is unavailable, so the learned kernel is absent from available_models(); "
            "load_model('%s') will report an unknown model rather than a missing dependency",
            "contagion",
        )
    return registry


def arm_name(base: str, mode: str = ABLATION_ARM_MODE) -> str:
    """The address of ``base``'s ``mode`` sampler arm."""
    return f"{base}{ARM_SEPARATOR}{mode}"


def split_arm_name(name: str) -> tuple[str, str] | None:
    """``("contagion_kernel", "independent")`` for an arm address, else ``None``."""
    if name.endswith(ABLATION_ARM_SUFFIX) and len(name) > len(ABLATION_ARM_SUFFIX):
        return name[: -len(ABLATION_ARM_SUFFIX)], ABLATION_ARM_MODE
    return None


def available_models() -> list[str]:
    """Names :func:`load_model` accepts without a path.

    Includes the latent-off ABLATION ARM of every registered predictor that has
    one. The arm is not a second registry entry and could not be: it is derived
    from its base at resolution time, so an arm name cannot exist without the
    model it ablates and cannot drift away from it.
    """
    registry = _registry()
    names = set(registry)
    names.update(arm_name(key) for key, cls in registry.items() if hasattr(cls, "with_sampler"))
    return sorted(names)


def _assert_arm_shares_parameters(model: Any, arm: Any) -> None:
    """RAISE unless ``arm``'s parameters are the SAME OBJECTS as ``model``'s.

    This is the guard, not a comment. The whole value of the ablation is that it
    is the same fit: if the arm's parameters were merely EQUAL - a copy, a
    re-load, a second construction from the same spec - then any difference
    between the two ensembles would be attributable to the sampler OR to the
    parameters, and the collapse comparison would be confounded exactly where it
    claims not to be. Equality would still read as a clean result, which is why
    the check is on identity and why it raises rather than warns.
    """
    named = getattr(model, "named_parameters", None)
    arm_named = getattr(arm, "named_parameters", None)
    if named is None or arm_named is None:
        raise TypeError(
            f"cannot verify that the ablation arm of {getattr(model, 'name', model)!r} shares "
            "its parameters: the predictor exposes no named_parameters(). An unverifiable "
            "sharing claim is refused rather than assumed - the arm's ONLY guarantee is that "
            "it is the same fit."
        )
    base = dict(named())
    view = dict(arm_named())
    if set(base) != set(view):
        raise RuntimeError(
            "the ablation arm has a different parameter SET from the model it ablates "
            f"(only in model: {sorted(set(base) - set(view))}; only in arm: "
            f"{sorted(set(view) - set(base))}). It is a different model, not an ablation."
        )
    copied = [k for k in sorted(base) if view[k] is not base[k]]
    if copied:
        raise RuntimeError(
            f"the ablation arm does NOT share the model's parameters: {copied} are distinct "
            "objects. Something constructed a second model instead of taking a view of this "
            "one; the arms would then differ in the sampler AND in the parameters, and the "
            "collapse comparison would be confounded even when its numbers look right."
        )


def ablation_arm(model: Predictor | str | Path, mode: str = ABLATION_ARM_MODE) -> Predictor:
    """The latent-off ABLATION of ``model``, sharing its parameters.

    ``model`` may be a loaded predictor or any address :func:`load_model`
    accepts. The arm is always ``model.with_sampler(mode)`` - never a second
    construction - and :func:`_assert_arm_shares_parameters` checks that on
    every call, so a future implementation that built a look-alike would raise
    here instead of returning plausible numbers.
    """
    resolved = load_model(model) if isinstance(model, (str, Path)) else model
    make = getattr(resolved, "with_sampler", None)
    if make is None:
        raise ValueError(
            f"{getattr(resolved, 'name', resolved)!r} has no with_sampler(...), so it has no "
            "latent to switch off and no ablation arm. Baselines are deterministic or "
            "independent by construction; the arm exists only for predictors with a shared "
            "per-step latent."
        )
    arm = make(mode)
    _assert_arm_shares_parameters(resolved, arm)
    return arm


def ablation_arm_is_demonstrative(model: Any) -> bool:
    """Would ablating ``model`` remove anything?

    ``False`` when the base has no shared latent. Such a model's independent
    arm is BIT-IDENTICAL to it, so a collapse ratio read off the pair is 1.0 by
    construction and is a statement about the pair being the same forecast, not
    about ``z_t``.
    """
    return getattr(model, "latent", None) is not None


def assert_ablation_arm_is_demonstrative(model: Any) -> None:
    """RAISE if a collapse verdict is about to be taken on a vacuous ablation."""
    if not ablation_arm_is_demonstrative(model):
        raise ValueError(
            f"{getattr(model, 'name', model)!r} has NO shared latent, so holding z_t at its "
            "prior mean removes nothing and the ablation arm is the same forecast under "
            "another name. A collapse comparison on this pair reads 1.0 by construction. "
            "G3 (d) asks whether removing z_t collapses the ensemble; it can only be asked "
            "of a model that has one."
        )


def load_model(path: str | Path) -> Predictor:
    """C5 ``load_model``: a baseline name, a directory holding ``model.json``, or
    either of those with ``__independent`` appended for the ABLATION ARM.

    A saved predictor is a directory, not a file, so a learned checkpoint can
    put weights beside its spec later without changing this signature.

    **THE ARM IS DERIVED, NOT REGISTERED.** ``<address>__independent`` resolves
    by loading ``<address>`` through this same function and taking
    ``with_sampler("independent")`` on THAT object, so the arm shares the
    parameters of the model it was derived from within the call that produced
    it. A registry entry that constructed its own kernel would be a second model
    that merely resembles the first, and no downstream number could tell the two
    apart. An address that exists on disk always wins over the derived reading,
    so a checkpoint directory whose name happens to end in ``__independent`` is
    still loaded as itself.
    """
    key = str(path)
    registry = _registry()
    if key in registry:
        return registry[key]()

    candidate = Path(path)
    spec_path = candidate / SPEC_NAME if candidate.is_dir() else candidate
    if spec_path.is_file():
        return _from_spec_file(spec_path, registry)

    arm = split_arm_name(key)
    if arm is not None:
        base_key, mode = arm
        base_candidate = Path(base_key)
        base_spec = base_candidate / SPEC_NAME if base_candidate.is_dir() else base_candidate
        if base_key in registry or base_spec.is_file():
            return ablation_arm(load_model(base_key), mode)

    raise FileNotFoundError(
        f"no model at {path!r}: not a registered baseline ({', '.join(available_models())}) "
        f"and no {SPEC_NAME} at {spec_path}. A name ending {ABLATION_ARM_SUFFIX!r} is the "
        "latent-off ablation ARM of the address before it and resolves only when that "
        "address does."
    )


def _from_spec_file(spec_path: Path, registry: Mapping[str, Any]) -> Predictor:
    spec = json.loads(spec_path.read_text())
    kind = spec.get("kind")
    if kind not in registry:
        raise ValueError(
            f"{spec_path} declares kind={kind!r}, which is not a known predictor "
            f"({', '.join(sorted(registry))})"
        )
    predictor: Predictor = registry[kind].from_spec(spec)
    return predictor


def save_model(model: Any, path: str | Path) -> Path:
    """Write ``{path}/model.json`` for a predictor that exposes ``to_spec()``."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / SPEC_NAME
    spec_path.write_text(json.dumps(model.to_spec(), indent=2, sort_keys=False) + "\n")
    return spec_path
