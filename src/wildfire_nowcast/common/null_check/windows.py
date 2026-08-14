"""EVALUATION WINDOWS, built from labels alone.

A :class:`Window` is one ``t0`` state plus the ``L`` label frames after it —
the only input any forecaster in this harness is allowed to see. Two sources:
a generated label sequence with a DECLARED zero-growth rate (so the pathology's
magnitude is an input rather than an accident of whichever fire was on disk),
and any C1 store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Window:
    """One evaluation window: the state at ``t0`` and the ``L`` labels after it."""

    t0: int
    x0: np.ndarray  # uint8 [H, W]
    truth: np.ndarray  # uint8 [L, H, W]


def _ellipse(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2 <= 1.0


def synthetic_windows(
    *,
    n_hours: int = 40,
    horizon_h: int = 3,
    shape: tuple[int, int] = (40, 40),
    p_grow: float = 0.55,
    seed: int = 20260808,
) -> tuple[list[Window], dict[str, Any]]:
    """A label sequence with a KNOWN zero-growth rate, and windows over it.

    Deliberately synthetic rather than a real fire: the pathology's magnitude is
    a function of the zero-growth rate, so that rate must be a declared input,
    not something inherited from whichever fire happened to be on disk. The
    generated statistic is returned alongside so the report states it.

    Growth is an anisotropic ellipse with a drifting centre, made absorbing by
    construction (C1.1's guarantee, reproduced here so the fixture is legal
    state data rather than an arbitrary boolean field).
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    grow = rng.random(n_hours) < float(p_grow)
    steps = np.cumsum(grow.astype(np.float64))
    # Radius AND centre advance only on a growth hour, so a dormant hour adds
    # bitwise zero cells. Drifting the centre every hour instead would make every
    # hour a growth hour and quietly destroy the property being tested — the
    # zero-growth rate is the whole point of this fixture.
    radius = 3.0 + steps * 1.10
    cy = height / 2.0 - steps * 0.32
    cx = width / 2.0 + steps * 0.46

    ever = np.zeros((n_hours, height, width), dtype=bool)
    prev = np.zeros((height, width), dtype=bool)
    for t in range(n_hours):
        prev = prev | _ellipse(shape, cy[t], cx[t], radius[t], radius[t] * 0.72)
        ever[t] = prev

    state = np.zeros((n_hours, height, width), dtype=np.uint8)
    for t in range(n_hours):
        older = ever[t - 1] if t else np.zeros_like(ever[t])
        state[t] = np.where(ever[t] & ~older, 1, np.where(ever[t], 2, 0)).astype(np.uint8)

    windows = [
        Window(t0=t0, x0=state[t0], truth=state[t0 + 1 : t0 + 1 + horizon_h])
        for t0 in range(n_hours - horizon_h)
        if state[t0].any()
    ]
    total_leads = sum(int(w.truth.shape[0]) for w in windows)
    zero_growth = sum(
        1
        for w in windows
        for k in range(w.truth.shape[0])
        if not np.any((w.truth[k] > 0) & ~(w.x0 > 0))
    )
    stats = {
        "source": "synthetic_windows",
        "n_windows": len(windows),
        "n_leads": total_leads,
        "grid_shape": list(shape),
        "horizon_h": horizon_h,
        "zero_growth_lead_fraction": (zero_growth / total_leads) if total_leads else None,
        "seed": seed,
    }
    return windows, stats


def windows_from_tensor(
    tensor_path: str | Path, *, horizon_h: int = 3, stride: int = 1, max_windows: int | None = None
) -> tuple[list[Window], dict[str, Any]]:
    """Windows read from any C1 store — the C4 synthetic fire or a real fire."""
    from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: PLC0415

    ds = open_tensor(Path(tensor_path))
    state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    n_t = int(state.shape[0])
    windows: list[Window] = []
    for t0 in range(0, n_t - horizon_h, max(1, int(stride))):
        if not state[t0].any():
            continue
        windows.append(Window(t0=t0, x0=state[t0], truth=state[t0 + 1 : t0 + 1 + horizon_h]))
        if max_windows is not None and len(windows) >= max_windows:
            break
    total_leads = sum(int(w.truth.shape[0]) for w in windows)
    zero_growth = sum(
        1
        for w in windows
        for k in range(w.truth.shape[0])
        if not np.any((w.truth[k] > 0) & ~(w.x0 > 0))
    )
    stats = {
        "source": str(tensor_path),
        "n_windows": len(windows),
        "n_leads": total_leads,
        "grid_shape": list(state.shape[1:]),
        "horizon_h": horizon_h,
        "zero_growth_lead_fraction": (zero_growth / total_leads) if total_leads else None,
    }
    return windows, stats
