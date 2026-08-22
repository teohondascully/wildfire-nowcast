"""C1.1 - the ratified ``fireline_v2`` perimeter -> ``fire_state`` rule.

**This module is the single implementation of the state rule (C0, ADR-007).**
The rule is contractual: it decides what the label channel *means*, so a producer
and a verifier that compute it through different code is precisely how a tensor
passes its own check and is still wrong. Everything that turns GOFER perimeters
into ``{0, 1, 2}`` - the real ingestion path, the synthetic fixture, any future
re-labelling - imports :func:`apply_state_rule` from here. Nothing re-implements
it.

The rule (ADR-006 P1, INTERFACES C1.1)::

    ever(t)       = OR_{s<=t} inside perimeter(s)
    new(t)        = ever(t) and not ever(t-1)
    active(t)     = ever(t) and dilate(cfireLine(t, fconf=0.50), 1 cell)
    burning(t)    = (burning(t-1) or new(t)) and (new(t) or active(t))
    burned_out(t) = ever(t) and not burning(t)

Why each term is there
----------------------
``ever`` is a running OR, so burned area is monotone *by construction* rather
than by hope; the cells it rescues (perimeters GOFER shrank) are a QA number,
not a silent repair.

``active`` is what keeps state 1 observationally grounded. 51-91% of GOFER hours
have bitwise-zero perimeter growth because GOES stops detecting the front at
night or under cloud - not because the fire stopped. The retired provisional
rule made state 1 an artefact of Δt and left it empty in 62-87% of frames, so a
contagion kernel had no seed in most training steps. ``cfireLine`` is non-empty
in 100% of Kincade's zero-growth hours.

``(burning(t-1) or new(t))`` is what keeps fire **absorbing**. A naive
``burning = new or active`` lets a cell go 2 -> 1 when the fire line wanders back
over it, which is a state decrease. Fire is ABSORBING here (``README.md``):
burned area never decreases, so that transition is forbidden outright.
Because ``new`` can never fire twice for one cell, dropping out of ``burning`` is
permanent: every cell traces 0 -> 1 (one contiguous run) -> 2 and never returns.

Guarantees, verified by :func:`wildfire_nowcast.common.contract.fire_state_violations`
and asserted in-line when ``validate=True``: values in ``{0,1,2}``, non-decreasing
in time, no 0 -> 2 skip, one contiguous burning run per cell.

Accepted caveat (ADR-006, and a contract line in C1.1): this does NOT reach 0%
empty-burning frames and no absorbing rule can - after a long dormancy every cell
is closed and state 1 is legitimately empty (6-37% of real frames). Consumers
must treat the **frontier of the burned region**, not state 1 alone, as the
contagion source.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from wildfire_nowcast.common.contract import (
    BURNED_OUT,
    BURNING,
    UNBURNED,
    fire_state_violations,
)

__all__ = [
    "UNBURNED",
    "BURNING",
    "BURNED_OUT",
    "StateRule",
    "FIRELINE_V2",
    "RETIRED_RULES",
    "dilate",
    "cumulative_or",
    "fireline_v2",
    "apply_state_rule",
    "burning_residence_hours",
    "frames_without_burning",
]

#: The one supported rule. ``provisional_p0`` is RETIRED (ADR-006 P1) and is
#: rejected rather than quietly re-implemented, so no artefact can be produced
#: under it by accident.
StateRule = Literal["fireline_v2"]
#: Annotated with the alias rather than inferred as ``str``: without this the
#: literal type is widened and ``rule: StateRule = FIRELINE_V2`` below stops
#: type-checking, which is the one place the retired rule could re-enter.
FIRELINE_V2: StateRule = "fireline_v2"
RETIRED_RULES = frozenset({"provisional_p0", "provisional", "p0"})


def dilate(mask: np.ndarray, iterations: int = 1, *, connectivity: int = 8) -> np.ndarray:
    """Binary dilation with a 4- or 8-connected structuring element (numpy only).

    Kept here rather than pulled from scipy so the contract-adjudicated path has
    no optional dependency: a lead who cannot import scipy must still get the
    same labels, not a fallback.
    """
    out = np.asarray(mask, dtype=bool)
    if iterations <= 0:
        return out.copy()
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if out.ndim != 2:
        raise ValueError(f"dilate expects a 2-D mask, got shape {out.shape}")
    offsets = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 4
        else [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    )
    h, w = out.shape
    for _ in range(int(iterations)):
        acc = out.copy()
        for dy, dx in offsets:
            ys_dst = slice(max(dy, 0), h + min(dy, 0))
            ys_src = slice(max(-dy, 0), h + min(-dy, 0))
            xs_dst = slice(max(dx, 0), w + min(dx, 0))
            xs_src = slice(max(-dx, 0), w + min(-dx, 0))
            acc[ys_dst, xs_dst] |= out[ys_src, xs_src]
        out = acc
    return out


def cumulative_or(masks: np.ndarray) -> np.ndarray:
    """Running OR over the leading (time) axis - the monotone ``ever`` field."""
    return np.logical_or.accumulate(np.asarray(masks, dtype=bool), axis=0)


def fireline_v2(
    perimeter_masks: np.ndarray,
    fire_line_masks: np.ndarray,
    *,
    line_dilation: int = 1,
    validate: bool = True,
) -> np.ndarray:
    """Apply C1.1 to ``(T, H, W)`` boolean perimeter and fire-line masks.

    Parameters
    ----------
    perimeter_masks
        ``(T, H, W)`` bool: "inside the GOFER perimeter at t". Need not be
        nested; ``ever`` is cumulative, so shrinking perimeters are absorbed.
    fire_line_masks
        ``(T, H, W)`` bool from GOFER ``cfireLine`` at ``fconf`` (0.50 is the
        contract default; the six shipped levels ARE the label-perturbation
        ensemble for the observation-noise augmentation).
    line_dilation
        Cells to grow the fire line by before intersecting with ``ever``. 1
        accounts for the line being a boundary that falls *between* cells.
    validate
        Assert the C1.1 guarantees on the result. Cheap relative to building
        the masks, and a violation here means the inputs are malformed.

    Returns
    -------
    ``(T, H, W)`` uint8 in ``{0, 1, 2}``.
    """
    masks = np.asarray(perimeter_masks, dtype=bool)
    lines = np.asarray(fire_line_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"expected (T, H, W) perimeter masks, got shape {masks.shape}")
    if lines.shape != masks.shape:
        raise ValueError(f"fire_line_masks {lines.shape} != perimeter masks {masks.shape}")

    state = np.zeros(masks.shape, dtype=np.uint8)
    ever = cumulative_or(masks)
    prev_ever = np.zeros(masks.shape[1:], dtype=bool)
    burning = np.zeros(masks.shape[1:], dtype=bool)
    for i in range(masks.shape[0]):
        cur_ever = ever[i]
        new = cur_ever & ~prev_ever
        active = cur_ever & dilate(lines[i], line_dilation)
        burning = (burning | new) & (new | active)
        state[i][cur_ever] = BURNED_OUT
        state[i][burning] = BURNING
        prev_ever = cur_ever

    if validate:
        violations = fire_state_violations(state)
        if violations:  # pragma: no cover - only reachable with malformed input
            raise ValueError(
                "fireline_v2 produced an invalid state field: " + "; ".join(violations)
            )
    return state


def apply_state_rule(
    perimeter_masks: np.ndarray,
    *,
    rule: StateRule = FIRELINE_V2,
    fire_line_masks: np.ndarray | None = None,
    line_dilation: int = 1,
    validate: bool = True,
) -> np.ndarray:
    """Named entry point for the C1.1 rule; the name is recorded in C2 provenance.

    Only ``"fireline_v2"`` is accepted. The provisional P0 rule is RETIRED
    (ADR-006 P1) and asking for it raises rather than silently producing labels
    under semantics no consumer expects.
    """
    if rule in RETIRED_RULES:
        raise ValueError(
            f"state rule {rule!r} was RETIRED by ADR-006 P1 and must not be used anywhere. "
            f"Use {FIRELINE_V2!r} (INTERFACES C1.1)."
        )
    if rule != FIRELINE_V2:
        raise ValueError(f"unknown state rule {rule!r}; C1.1 defines only {FIRELINE_V2!r}")
    if fire_line_masks is None:
        raise ValueError(
            "fireline_v2 requires fire_line_masks (GOFER cfireLine). Without an active "
            "fire line there is no observational basis for state 1 (ADR-006 P1)."
        )
    return fireline_v2(
        perimeter_masks,
        fire_line_masks,
        line_dilation=line_dilation,
        validate=validate,
    )


# --------------------------------------------------------------------------
# descriptive helpers (shared by QA, the synthetic fixture and figures)
# --------------------------------------------------------------------------


def burning_residence_hours(state: np.ndarray) -> np.ndarray:
    """Hours each ever-burning cell spent in state 1, as a flat array.

    Under C1.1 the burning hours of a cell form one contiguous run, so this is
    just a per-cell count. The retired rule made every value exactly 1; GOFER
    under ``fireline_v2`` gives p50 3-5 h, p90 8-19 h.
    """
    arr = np.asarray(state)
    counts = (arr == BURNING).sum(axis=0).ravel()
    return cast("np.ndarray[Any, Any]", counts[counts > 0])


def frames_without_burning(state: np.ndarray) -> np.ndarray:
    """Indices of frames with no cell in state 1.

    Legitimately non-empty: C1.1 records 6-37% of real frames. Use this to
    *measure* the phenomenon, never as a reason to reject a tensor.
    """
    arr = np.asarray(state)
    return np.flatnonzero(~(arr == BURNING).any(axis=(1, 2)))
