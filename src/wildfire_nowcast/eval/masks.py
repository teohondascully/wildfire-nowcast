"""Scoring masks and event definitions for C6.

The choice of *which pixels to score* decides more about a wildfire nowcasting
number than the choice of scoring rule does, so it is made explicitly here
rather than implicitly inside a metric.

Why a domain-wide score is not enough
-------------------------------------
A buffered fire domain is mostly far-field unburned landscape that every model
gets right for free. Kincade's domain is 51 x 39 = 1,989 cells and its FINAL
footprint is 347 of them; over a 3 h window the disputed set is a few dozen
cells. A Brier score averaged over the domain is therefore ~99% agreement on
cells nobody was ever uncertain about, and it compresses the difference between
a good model and a useless one into the third decimal place.

Combine that with insights/data item 1 — 51-91% of hours have BITWISE zero
growth, median ~0.79 — and a domain-wide, all-hours score is a measurement of
how often nothing happened. Persistence maximises it.

So C6 computes every metric under at least two masks:

``domain``
    Every cell. Unambiguous, reproducible, and the number that is comparable
    across models on the same fire. It is the contract's headline because it
    cannot be gamed by a mask choice — not because it is the informative one.
``growth_band``
    Cells UNBURNED at ``t0`` and within ``band_radius_cells`` of the burned
    frontier: the cells where the forecast is actually a decision. This mask is
    computed from ``x0`` ALONE, never from the outcome, so restricting to it is
    not conditioning on the answer. It is the number a G2/G5 verdict should
    quote, next to the domain number, never instead of it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, UNBURNED
from wildfire_nowcast.common.states import dilate

__all__ = [
    "EVENTS",
    "DEFAULT_EVENT",
    "event_field",
    "frontier",
    "growth_band",
    "scoring_masks",
    "default_band_radius",
]

#: Binary events a state field can be scored on.
#:
#: ``burned`` is the default and the only one that should carry a headline. It
#: is the monotone, absorbing quantity — "has the fire arrived here yet" — which
#: is what a nowcast is for. ``burning`` (state 1) is tempting and wrong as a
#: target: under C1.1 state 1 tracks whether GOES could SEE a fire line, so
#: scoring it measures satellite visibility as much as fire behaviour, and it is
#: legitimately empty in 6-37% of frames.
EVENTS: Mapping[str, int] = {
    "burned": UNBURNED,  # state > 0
    "burning": BURNING,  # state == 1
    "burned_out": BURNED_OUT,  # state == 2
}

DEFAULT_EVENT = "burned"


def event_field(state: np.ndarray, event: str = DEFAULT_EVENT) -> np.ndarray:
    """Boolean indicator of ``event`` over an arbitrarily-shaped state array."""
    arr = np.asarray(state)
    if event == "burned":
        return arr > UNBURNED
    if event == "burning":
        return arr == BURNING
    if event == "burned_out":
        return arr == BURNED_OUT
    raise ValueError(f"unknown event {event!r}; expected one of {sorted(EVENTS)}")


def frontier(x0: np.ndarray) -> np.ndarray:
    """Burned cells adjacent to an unburned cell — the contagion source (C1.1).

    Explicitly NOT ``state == 1``. C1.1 records that state 1 is legitimately
    empty in 6-37% of frames (Kincade: 43 of 134, every one of them while the
    fire was still active), so a band built from state 1 would be EMPTY in a
    third of Kincade's windows and the growth-band metrics would silently
    evaluate nothing at all.
    """
    burned = np.asarray(x0) > UNBURNED
    if not burned.any():
        return np.zeros_like(burned)
    return burned & dilate(~burned, 1)


def default_band_radius(horizon_h: int, *, cells_per_hour: float = 4.0) -> int:
    """Band radius in cells for a horizon, from a generous head-rate ceiling.

    4 cells/h at 1 km is ~4 km/h, comfortably above the fastest head rate this
    project's baselines produce and above Kincade's documented Diablo run. The
    band is meant to be generous: a too-tight band would exclude cells the model
    could plausibly have reached and would flatter every model by hiding its
    false alarms.
    """
    return max(1, int(np.ceil(float(cells_per_hour) * max(1, int(horizon_h)))))


def growth_band(x0: np.ndarray, radius_cells: int) -> np.ndarray:
    """Unburned cells within ``radius_cells`` of the ``t0`` burned frontier.

    Depends on ``x0`` only. If nothing is burned at ``t0`` the band is empty and
    the caller should skip the window — nowcasting starts from an observed fire.
    """
    burned = np.asarray(x0) > UNBURNED
    seed = frontier(x0)
    if not seed.any():
        return np.zeros_like(burned)
    return dilate(seed, max(1, int(radius_cells))) & ~burned


def scoring_masks(
    x0: np.ndarray | None,
    shape: tuple[int, int],
    horizon_h: int,
    *,
    band_radius_cells: int | None = None,
) -> dict[str, np.ndarray]:
    """``{"domain": ..., "growth_band": ...}``; band omitted when ``x0`` is None.

    ``x0`` is not part of the C6 signature, so a caller that does not supply it
    gets the domain mask only — with the metrics dict recording that the band
    was unavailable, rather than silently reporting domain numbers as if they
    were band numbers.
    """
    masks = {"domain": np.ones(shape, dtype=bool)}
    if x0 is not None:
        radius = (
            int(band_radius_cells)
            if band_radius_cells is not None
            else default_band_radius(horizon_h)
        )
        masks["growth_band"] = growth_band(x0, radius)
    return masks
