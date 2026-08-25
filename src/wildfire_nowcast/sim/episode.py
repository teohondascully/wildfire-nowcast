"""Episode replay: what the model expected, beside what actually burned.

    python -m wildfire_nowcast.sim.episode \
        --fire 2020_creek --episode 1234,1235,1236 --contrast 1255,82 \
        --calibration <a run artifact carrying the ellipse calibration> \
        --out <the episode page>.png --regime-out <the regime page>.png \
        --run-out <the timeline page>.png --json <every number on them>.json

Every output is named by the caller and none is defaulted, so two runs cannot
overwrite each other and a reader who finds two pages can tell which run each
belongs to.

The model's worst published behaviour is not a statistic: it is 162 cells put at
``p >= 0.5`` at the 3 h lead of which none burned, and 158 of them sit in three
consecutive hours of one fire. A number that concentrated is a picture, and this
module draws it: per window, the ensemble's burn probability, the member fronts,
the cells it was confident about, and the cells that actually burned, on the
same axes and at the same zoom.

**A figure of a failure alone would mislead, so this one refuses to be made
without a control.** The published discriminator is a CONJUNCTION of wind AND
dryness AND a long perimeter, not wind on its own; a page showing only the
extreme-wind hours invites the reader to conclude "high wind breaks it", which
the same corpus refutes. ``--contrast`` windows are therefore rendered in the
same figure, with the same columns, the same colour scale and the same zoom
logic, and the regime page plots every growth window in the held-out set so the
reader can see how few of them the model is ever confident in.

Three labelling rules are enforced here rather than left to a caption.

**A window is named by its ``t0``, and the hour printed beside it is the hour of
``t0``, not the hour the forecast is valid.** Under the end-of-hour convention a
window at ``t0`` is driven by the weather of ``t0+1 .. t0+horizon`` and is scored
against the state at ``t0+horizon``, so the same episode can be quoted with two
different clock times three hours apart, both correct, neither labelled. Every
panel prints both.

**Every probability field names its seed draw.** The two draws differ only in
how member seeds are assigned to windows, and a claim that holds in one is not a
claim that holds in both. This module computes both and prints both counts; the
raster it draws is draw A and says so.

**The confident cells are recomputed here through C5, never read from a
table.** ``--cross-check`` takes a run artifact carrying a confident-cell dump
and asserts set equality with what this module computed; the comparison and its
result are printed on the page. A figure that agrees with a table it copied is
not evidence that the table is right.

Nothing in this module reaches into model internals. Members come from
``predict()``, the scored band and the burned-event definition come from the
evaluation masks, and the calibrated ellipse is built by the single constructor
in :mod:`wildfire_nowcast.sim.replay`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from wildfire_nowcast.common.contract import UNBURNED  # noqa: E402
from wildfire_nowcast.common.paths import fire_tensor_path  # noqa: E402
from wildfire_nowcast.common.states import dilate  # noqa: E402
from wildfire_nowcast.common.zarr_io import channel_values, open_tensor  # noqa: E402
from wildfire_nowcast.eval.masks import (  # noqa: E402
    DEFAULT_EVENT,
    event_field,
    scoring_masks,
)
from wildfire_nowcast.model.inputs import forecast_inputs  # noqa: E402
from wildfire_nowcast.sim.absent import refuse_if_empty  # noqa: E402
from wildfire_nowcast.sim.ensemble import draw_burn_probability  # noqa: E402
from wildfire_nowcast.sim.style import (  # noqa: E402
    COL_MEMBER,
    COL_TEXT,
    COL_WARN,
    COL_WIND,
    STATE_COLORS,
    PlotGeometry,
    add_scale_bar,
    plot_extent,
    quiver_grid,
    stamp,
)

__all__ = [
    "CONFIDENT_P",
    "DRAWS",
    "WindowFacts",
    "ArmReading",
    "window_positions",
    "seed_offsets_for_fire",
    "confident_mask",
    "distance_to_burned",
    "window_facts",
    "read_arm",
    "sweep_growth_windows",
    "render_episode_page",
    "render_regime_page",
    "render_episode_movie",
    "sweep_window_range",
    "render_run_page",
    "main",
]

#: The probability at or above which a forecast is called a commitment. Not a
#: free parameter here: it is the threshold the confident-cell finding was
#: published at, and moving it would produce a different set with the same name.
CONFIDENT_P = 0.5

#: Member-seed base and ensemble size of the run being explained.
SEED_BASE = 20260807
MEMBERS = 24
HORIZON = 3

#: The two seed draws. They differ ONLY in how a window's index maps to its seed
#: offset: draw A counts every evaluable window, draw B counts within the
#: growth / dormant stratum. Same model, same weather, same truth.
DRAWS = ("A", "B")

_GREEN_BY_LEAD = ("#a1d99b", "#41ab5d", "#00701a")
_COL_CONFIDENT = COL_WARN
_COL_NEARMISS = "#b45309"


# -- window bookkeeping ----------------------------------------------------


def window_positions(state: np.ndarray, horizon_h: int) -> tuple[list[int], dict[int, int]]:
    """``(t0s, position_of_t0)`` for every evaluable window of a fire.

    Reproduces the enumeration the evaluation harness uses at stride 1: ``t0``
    in ``range(0, T - horizon_h)``, skipping windows with nothing burned at
    ``t0``. The POSITION matters as much as the window, because member seeds are
    ``seed + position``; taking ``t0`` for the position instead would reseed
    every ensemble and draw a neighbouring experiment.

    Derived from the state array rather than by materialising every window,
    which is a shortcut and is therefore checked: ``--cross-check`` compares the
    resulting cell sets against the run's own dump, and a wrong position moves
    every cell.
    """
    arr = np.asarray(state)
    ignited = (arr > UNBURNED).any(axis=(1, 2))
    t0s = [t for t in range(0, arr.shape[0] - int(horizon_h)) if bool(ignited[t])]
    return t0s, {t: i for i, t in enumerate(t0s)}


def seed_offsets_for_fire(
    state: np.ndarray, horizon_h: int, draw: str
) -> tuple[list[int], dict[int, int]]:
    """``(t0s, offset_of_t0)`` for one fire under one seed draw.

    Draw A is the shipped rule and the offset is the window's position in the
    full list. Draw B indexes within the stratum, so a growth window's offset is
    the number of growth windows before it. Both are needed because a claim that
    survives one draw and not the other is a claim about seeds.
    """
    if draw not in DRAWS:
        raise ValueError(f"unknown draw {draw!r}; expected one of {DRAWS}")
    arr = np.asarray(state)
    t0s, position = window_positions(arr, horizon_h)
    if draw == "A":
        return t0s, dict(position)
    burned = arr > UNBURNED
    offsets: dict[int, int] = {}
    n_growth = 0
    n_dormant = 0
    for t0 in t0s:
        grows = bool(np.count_nonzero(burned[t0 + horizon_h] & ~burned[t0]))
        if grows:
            offsets[t0] = n_growth
            n_growth += 1
        else:
            offsets[t0] = n_dormant
            n_dormant += 1
    return t0s, offsets


def confident_mask(
    prob: np.ndarray, band: np.ndarray, threshold: float = CONFIDENT_P
) -> np.ndarray:
    """Band cells whose ensemble probability at the final lead is a commitment."""
    hit = np.asarray(band, dtype=bool) & (np.asarray(prob)[-1] >= float(threshold))
    return np.asarray(hit, dtype=bool)


def distance_to_burned(x0: np.ndarray, limit: int = 32) -> np.ndarray:
    """Chebyshev distance in cells to the nearest cell burned at ``t0``.

    ``-1`` where the fire never reaches within ``limit``. Repeated 8-connected
    dilation rather than a distance transform, to stay on the same neighbourhood
    definition the contagion and the band use.
    """
    burned = np.asarray(x0) > UNBURNED
    out = np.full(burned.shape, -1, dtype=np.int16)
    out[burned] = 0
    cur = burned
    for d in range(1, int(limit) + 1):
        nxt = dilate(cur)
        new = nxt & ~cur
        if not new.any():
            break
        out[new] = d
        cur = nxt
    return out


# -- readings --------------------------------------------------------------


@dataclass
class WindowFacts:
    """Everything about a window that does not depend on a model."""

    fire_id: str
    t0: int
    position: int
    time_t0: str
    time_valid: str
    burned_cells: int
    band_cells: int
    truth_growth_cells: int
    max_wind_ms: float
    mean_wind_ms: float
    min_rh_pct: float
    mean_wind_u: float
    mean_wind_v: float
    rank_max_wind: int = 0
    rank_min_rh: int = 0
    rank_burned: int = 0
    n_growth_windows: int = 0


@dataclass
class ArmReading:
    """One arm's ensemble at one window, reduced to what a page needs."""

    name: str
    draw: str
    seed: int
    prob: np.ndarray = field(repr=False)
    members: np.ndarray = field(repr=False)
    confident: np.ndarray = field(repr=False)
    n_confident: int = 0
    n_confident_burned: int = 0
    max_p: float = 0.0
    n_p_ge_025: int = 0
    mean_pred_new_cells: float = 0.0
    n_band_cells: int = 0
    #: Counts of band cells per 0.1-wide probability bin at the final lead.
    hist: tuple[int, ...] = ()


def window_facts(ds: Any, fire_id: str, t0: int, position: int, horizon_h: int) -> WindowFacts:
    """Covariates of one window, read from the tensor by channel name.

    Wind and humidity are summarised over the hours that DRIVE the forecast,
    ``t0+1 .. t0+horizon``, over the whole domain. The domain reading is the one
    the published table used; restricting to the burned footprint changes the
    dryness of some windows and is reported in the sidecar, not silently
    substituted here.
    """
    w = forecast_inputs(ds, t0, horizon_h, fire_id=fire_id)
    times = np.asarray(ds["time"].values)
    sl = slice(t0 + 1, t0 + 1 + horizon_h)
    u = channel_values(ds, "wind_u10", dtype=np.float32)[sl]
    v = channel_values(ds, "wind_v10", dtype=np.float32)[sl]
    rh = channel_values(ds, "rh_2m", dtype=np.float32)[sl]
    spd = np.hypot(u, v)
    band = scoring_masks(w.x0, (w.x0.shape[0], w.x0.shape[1]), horizon_h)["growth_band"]
    return WindowFacts(
        fire_id=fire_id,
        t0=int(t0),
        position=int(position),
        time_t0=str(times[t0]),
        time_valid=str(times[t0 + horizon_h]),
        burned_cells=int(np.count_nonzero(np.asarray(w.x0) > UNBURNED)),
        band_cells=int(np.count_nonzero(band)),
        truth_growth_cells=int(w.truth_growth_cells()),
        max_wind_ms=float(spd.max()),
        mean_wind_ms=float(spd.mean()),
        min_rh_pct=float(rh.min()),
        mean_wind_u=float(u.mean()),
        mean_wind_v=float(v.mean()),
    )


def read_arm(
    model: Any,
    window: Any,
    *,
    name: str,
    draw: str,
    seed: int,
    n_members: int = MEMBERS,
    horizon_h: int = HORIZON,
) -> ArmReading:
    """Call C5 ``predict()`` once and reduce the ensemble to a page's worth."""
    samples = model.predict(window.x0, window.static, window.weather, n_members, horizon_h, seed)
    ev = event_field(samples, DEFAULT_EVENT)
    prob = ev.mean(axis=0, dtype=np.float64)
    x0 = np.asarray(window.x0)
    band = scoring_masks(x0, (x0.shape[0], x0.shape[1]), horizon_h)["growth_band"]
    truth_ev = event_field(np.asarray(window.truth)[:horizon_h], DEFAULT_EVENT)
    conf = confident_mask(prob, band, CONFIDENT_P)
    was_burned = np.asarray(x0) > UNBURNED
    new_per_member = np.count_nonzero(ev[:, horizon_h - 1] & ~was_burned, axis=(1, 2))
    in_band = prob[horizon_h - 1][band] if band.any() else np.zeros(0)
    hist, _ = np.histogram(in_band, bins=np.linspace(0.0, 1.0, 11))
    return ArmReading(
        name=name,
        draw=draw,
        seed=int(seed),
        prob=prob,
        members=ev,
        confident=conf,
        n_confident=int(np.count_nonzero(conf)),
        n_confident_burned=int(np.count_nonzero(conf & truth_ev[horizon_h - 1])),
        max_p=float(in_band.max()) if in_band.size else 0.0,
        n_p_ge_025=int(np.count_nonzero(band & (prob[horizon_h - 1] >= 0.25))),
        mean_pred_new_cells=float(new_per_member.mean()),
        n_band_cells=int(np.count_nonzero(band)),
        hist=tuple(int(v) for v in hist),
    )


# -- the sweep over every growth window ------------------------------------


def sweep_growth_windows(
    fire_ids: Sequence[str],
    model: Any,
    *,
    draws: Sequence[str] = DRAWS,
    n_members: int = MEMBERS,
    horizon_h: int = HORIZON,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Covariates and confident-cell counts for every growth window of every fire.

    This is what turns one rendered episode into a statement about a population:
    without it the reader cannot tell whether the model commits like this once or
    every windy afternoon. Cost is one ``predict()`` per window per draw.
    """
    rows: list[dict[str, Any]] = []
    for fire_id in fire_ids:
        ds = open_tensor(fire_tensor_path(fire_id))
        try:
            state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
            burned = state > UNBURNED
            t0s, position = window_positions(state, horizon_h)
            offsets = {d: seed_offsets_for_fire(state, horizon_h, d)[1] for d in draws}
            for t0 in t0s:
                growth = int(np.count_nonzero(burned[t0 + horizon_h] & ~burned[t0]))
                if growth <= 0:
                    continue
                facts = window_facts(ds, fire_id, t0, position[t0], horizon_h)
                w = forecast_inputs(ds, t0, horizon_h, fire_id=fire_id)
                row: dict[str, Any] = asdict(facts)
                for d in draws:
                    r = read_arm(
                        model,
                        w,
                        name="model",
                        draw=d,
                        seed=SEED_BASE + offsets[d][t0],
                        n_members=n_members,
                        horizon_h=horizon_h,
                    )
                    row[f"n_confident_{d}"] = r.n_confident
                    row[f"n_confident_burned_{d}"] = r.n_confident_burned
                    row[f"max_p_{d}"] = r.max_p
                    row[f"seed_{d}"] = r.seed
                rows.append(row)
            if progress:
                print(f"  swept {fire_id}: {len(rows)} growth windows so far", flush=True)
        finally:
            ds.close()
    refuse_if_empty(
        "the growth-window sweep",
        {"fires": len(fire_ids), "growth_windows": len(rows)},
        because="A regime map over zero windows would draw an empty axis and read as 'the "
        "model is never confident', which is a finding it did not measure.",
    )
    return _rank_rows(rows)


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the three ranks the episode is quoted by. Ties share the best rank."""
    n = len(rows)
    for key, dest, reverse in (
        ("max_wind_ms", "rank_max_wind", True),
        ("min_rh_pct", "rank_min_rh", False),
        ("burned_cells", "rank_burned", True),
    ):
        order = sorted((r[key] for r in rows), reverse=reverse)
        index: dict[Any, int] = {}
        for i, v in enumerate(order):
            index.setdefault(v, i + 1)
        for r in rows:
            r[dest] = int(index[r[key]])
            r["n_growth_windows"] = n
    return rows


# -- panels ----------------------------------------------------------------


def _sub_geometry(geom: PlotGeometry, box: tuple[int, int, int, int]) -> PlotGeometry:
    r0, r1, c0, c1 = box
    return plot_extent(geom.x_centres[c0:c1], geom.y_centres[r0:r1], cell_size_m=geom.cell_size_m)


def zoom_box(
    masks: Sequence[np.ndarray], shape: tuple[int, int], *, margin: int = 4, minimum: int = 18
) -> tuple[int, int, int, int]:
    """A window on the action: rows/cols spanning every mask, padded and squared.

    Returned as ``(r0, r1, c0, c1)`` half-open. If nothing is marked at all the
    box centres on the domain, because an empty box is a crash and an empty
    PANEL is a legitimate answer that the reader still has to be able to see.
    """
    ny, nx = shape
    rows: list[int] = []
    cols: list[int] = []
    for m in masks:
        rr, cc = np.nonzero(np.asarray(m, dtype=bool))
        rows.extend(int(v) for v in rr)
        cols.extend(int(v) for v in cc)
    if not rows:
        return (0, ny, 0, nx)
    r0, r1 = min(rows) - margin, max(rows) + margin + 1
    c0, c1 = min(cols) - margin, max(cols) + margin + 1
    height, width = r1 - r0, c1 - c0
    side = max(height, width, minimum)
    rc, cc_ = (r0 + r1) // 2, (c0 + c1) // 2
    r0, r1 = rc - side // 2, rc - side // 2 + side
    c0, c1 = cc_ - side // 2, cc_ - side // 2 + side
    r0 = max(0, min(r0, ny - 1))
    c0 = max(0, min(c0, nx - 1))
    r1 = min(ny, max(r1, r0 + 2))
    c1 = min(nx, max(c1, c0 + 2))
    return (r0, r1, c0, c1)


def _outline_cells(ax: Any, geom: PlotGeometry, mask: np.ndarray, *, color: str, lw: float) -> int:
    """Draw one square per marked cell. Returns how many were drawn.

    Per-cell rectangles rather than a contour, because the cells this figure is
    about are frequently single and isolated, and a contour of an isolated cell
    is a diamond through its own centre: smaller than the cell, and read as a
    point rather than as a claim about that cell.
    """
    half = 0.5 * geom.cell_size_m
    rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
    for r, c in zip(rr, cc, strict=True):
        ax.add_patch(
            Rectangle(
                (geom.x_centres[c] - half, geom.y_centres[r] - half),
                geom.cell_size_m,
                geom.cell_size_m,
                fill=False,
                edgecolor=color,
                linewidth=lw,
                zorder=9,
            )
        )
    return int(rr.size)


def _fill_cells(ax: Any, geom: PlotGeometry, mask: np.ndarray, *, color: str, alpha: float) -> None:
    half = 0.5 * geom.cell_size_m
    rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
    for r, c in zip(rr, cc, strict=True):
        ax.add_patch(
            Rectangle(
                (geom.x_centres[c] - half, geom.y_centres[r] - half),
                geom.cell_size_m,
                geom.cell_size_m,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=6,
            )
        )


def _axes(ax: Any, geom: PlotGeometry, title: str, *, title_color: str = COL_TEXT) -> None:
    ax.set_xlim(geom.extent[0], geom.extent[1])
    ax.set_ylim(geom.extent[2], geom.extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, pad=4, color=title_color)


def _panel_context(
    ax: Any,
    geom: PlotGeometry,
    x0: np.ndarray,
    truth_new: np.ndarray,
    confident: np.ndarray,
    wind: tuple[np.ndarray, np.ndarray],
    box: tuple[int, int, int, int],
    facts: WindowFacts,
) -> None:
    """The whole fire, so the zoom is locatable and the wind field is visible."""
    ax.imshow(
        np.asarray(x0),
        cmap=ListedColormap(list(STATE_COLORS)),
        vmin=0,
        vmax=2,
        **geom.imshow_kwargs,
    )
    u, v = wind
    xg, yg, rows, cols, step = quiver_grid(geom, 12)
    ref = max(float(np.hypot(u, v).max()), 1e-6)
    ax.quiver(
        xg,
        yg,
        u[rows, cols],
        v[rows, cols],
        angles="xy",
        scale_units="xy",
        scale=ref / (0.85 * step * geom.cell_size_m),
        color=COL_WIND,
        alpha=0.75,
        width=0.005,
        zorder=5,
    )
    _outline_cells(ax, geom, confident, color=_COL_CONFIDENT, lw=0.5)
    _fill_cells(ax, geom, truth_new, color=_GREEN_BY_LEAD[2], alpha=1.0)
    r0, r1, c0, c1 = box
    half = 0.5 * geom.cell_size_m
    ax.add_patch(
        Rectangle(
            (geom.x_centres[c0] - half, geom.y_centres[r1 - 1] - half),
            (c1 - c0) * geom.cell_size_m,
            (r1 - r0) * geom.cell_size_m,
            fill=False,
            edgecolor=COL_TEXT,
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            zorder=10,
        )
    )
    add_scale_bar(ax, geom)
    _axes(
        ax,
        geom,
        f"t0 = {facts.t0}  ({facts.time_t0[:16]}Z)   burned {facts.burned_cells} cells",
    )
    ax.text(
        0.012,
        0.985,
        f"wind to leeward, max {facts.max_wind_ms:.1f} m/s\nmin RH {facts.min_rh_pct:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COL_WIND,
        bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.8},
    )


def _draw_burned_at_t0(ax: Any, geom: PlotGeometry, burned: np.ndarray) -> None:
    """The t0 footprint, as the background every other layer sits on."""
    ax.imshow(
        np.where(np.asarray(burned, dtype=bool), 1.0, np.nan),
        cmap=ListedColormap([STATE_COLORS[2]]),
        vmin=0,
        vmax=1,
        zorder=1,
        **geom.imshow_kwargs,
    )


def _ring_cells(ax: Any, geom: PlotGeometry, mask: np.ndarray, *, color: str) -> None:
    """A ring around each marked cell, so two cells in a 50-cell fire are findable.

    Without this the truth is a couple of coloured pixels that the eye slides
    past, and the reader concludes the panel is empty. An empty panel and a
    two-cell panel are the whole point of this figure.
    """
    rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
    for r, c in zip(rr, cc, strict=True):
        ax.plot(
            geom.x_centres[c],
            geom.y_centres[r],
            marker="o",
            mfc="none",
            mec=color,
            ms=13,
            mew=1.3,
            zorder=12,
        )


def _panel_model(
    ax: Any,
    geom: PlotGeometry,
    reading: ArmReading,
    box: tuple[int, int, int, int],
    facts: WindowFacts,
    x0: np.ndarray,
    truth_new: np.ndarray,
    *,
    title: str,
    show_members: bool = True,
) -> Any:
    """P(this cell is NEWLY burned by 3 h), which is the only interesting half.

    Cells already burned at ``t0`` carry probability 1 in every member by the
    absorbing rule, so drawing them saturates the panel and hides the forecast.
    They are drawn as the char-coloured footprint instead, and the colour scale
    is spent entirely on the cells the model was actually asked about.
    """
    r0, r1, c0, c1 = box
    sub = _sub_geometry(geom, box)
    burned = (np.asarray(x0) > UNBURNED)[r0:r1, c0:c1]
    prob3 = np.asarray(reading.prob)[HORIZON - 1][r0:r1, c0:c1].copy()
    prob3[burned] = 0.0
    _draw_burned_at_t0(ax, sub, burned)
    im = draw_burn_probability(ax, sub, prob3)
    if show_members:
        for m in np.asarray(reading.members)[:, HORIZON - 1][:, r0:r1, c0:c1]:
            front = m & ~burned
            if front.any():
                ax.contour(
                    sub.x_centres,
                    sub.y_centres,
                    m.astype(float),
                    levels=[0.5],
                    colors=[COL_MEMBER],
                    linewidths=0.5,
                    alpha=0.22,
                    zorder=7,
                )
    conf = np.asarray(reading.confident)[r0:r1, c0:c1]
    n = _outline_cells(ax, sub, conf, color=_COL_CONFIDENT, lw=1.0)
    if n == 0 and reading.max_p > 0:
        _outline_cells(ax, sub, prob3 >= 0.999 * reading.max_p, color=_COL_NEARMISS, lw=1.4)
    _fill_cells(ax, sub, truth_new[r0:r1, c0:c1], color=_GREEN_BY_LEAD[2], alpha=1.0)
    _ring_cells(ax, sub, truth_new[r0:r1, c0:c1], color=_GREEN_BY_LEAD[2])
    _draw_wind_arrow(ax, sub, facts)
    _axes(ax, sub, title)
    return im


def _panel_truth(
    ax: Any,
    geom: PlotGeometry,
    x0: np.ndarray,
    truth: np.ndarray,
    confident: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    title: str,
) -> None:
    r0, r1, c0, c1 = box
    sub = _sub_geometry(geom, box)
    was = (np.asarray(x0) > UNBURNED)[r0:r1, c0:c1]
    _draw_burned_at_t0(ax, sub, was)
    ev = event_field(np.asarray(truth)[:HORIZON], DEFAULT_EVENT)
    seen = was.copy()
    newest = np.zeros_like(was)
    for lead in range(HORIZON):
        new = ev[lead][r0:r1, c0:c1] & ~seen
        _fill_cells(ax, sub, new, color=_GREEN_BY_LEAD[lead], alpha=1.0)
        newest = newest | new
        seen = seen | ev[lead][r0:r1, c0:c1]
    _outline_cells(ax, sub, np.asarray(confident)[r0:r1, c0:c1], color=_COL_CONFIDENT, lw=1.0)
    _ring_cells(ax, sub, newest, color=_GREEN_BY_LEAD[2])
    _axes(ax, sub, title)


def _draw_wind_arrow(ax: Any, geom: PlotGeometry, facts: WindowFacts) -> None:
    """One arrow for the mean wind of the driving hours, in metre coordinates."""
    u, v = facts.mean_wind_u, facts.mean_wind_v
    norm = math.hypot(u, v)
    if norm <= 0:
        return
    span = geom.extent[1] - geom.extent[0]
    length = 0.22 * span
    x = geom.extent[0] + 0.16 * span
    y = geom.extent[2] + 0.14 * (geom.extent[3] - geom.extent[2])
    ax.arrow(
        x,
        y,
        length * u / norm,
        length * v / norm,
        width=0.012 * span,
        color=COL_WIND,
        alpha=0.9,
        length_includes_head=True,
        zorder=11,
    )
    ax.text(
        x,
        y - 0.06 * (geom.extent[3] - geom.extent[2]),
        f"{norm:.1f} m/s mean",
        fontsize=6.5,
        color=COL_WIND,
        ha="left",
        va="top",
    )


# -- pages -----------------------------------------------------------------


def _ratio(pred: float, truth: int) -> str:
    """``NNx truth``, or a refusal when truth is zero rather than an infinity."""
    if truth <= 0:
        return "undefined (truth grew 0)"
    return f"{pred / truth:.0f}x"


def _panel_commitment(ax: Any, model: ArmReading, ell: ArmReading) -> None:
    """Where the ensemble put its probability, over the whole scored band.

    The map panels can show that a commitment happened; they cannot show that
    everything else was low. This says how many band cells sit in each 0.1-wide
    probability bin, on a log count axis, with the commitment threshold drawn.
    A row with nothing right of the line is a model that was UNCERTAIN, which is
    a different statement from a model that was silent.
    """
    centres = np.arange(10) * 0.1 + 0.05
    width = 0.042
    m = np.asarray(model.hist, dtype=float)
    e = np.asarray(ell.hist, dtype=float)
    ax.bar(
        centres - width / 2, np.maximum(m, 0.4), width=width, color=_COL_CONFIDENT, label="model"
    )
    ax.bar(centres + width / 2, np.maximum(e, 0.4), width=width, color="#6b7280", label=ell.name)
    ax.axvline(CONFIDENT_P, color=COL_TEXT, lw=1.0, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, max(m.max(), e.max(), 10.0) * 3)
    ax.set_xlabel("P(newly burned by 3 h)", fontsize=7)
    ax.set_ylabel("band cells", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(
        f"where the probability mass sits: {model.n_confident} model cells and "
        f"{ell.n_confident} {ell.name} cells right of the line",
        fontsize=7,
        pad=3,
    )
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _rank(value: int, facts: WindowFacts) -> str:
    """``rank k of n``, or a statement that no sweep was run. Never a bare 0.

    A rank printed as ``0 of 0`` because the population was never enumerated
    looks like a measurement and is not one.
    """
    if facts.n_growth_windows <= 0:
        return "(rank NOT COMPUTED: no sweep)"
    return f"(rank {value} of {facts.n_growth_windows})"


def _row_caption(
    facts: WindowFacts, model: ArmReading, model_b: ArmReading, ell: ArmReading
) -> str:
    return (
        f"t0 = {facts.t0}   state hour {facts.time_t0[:16]}Z   3 h valid "
        f"{facts.time_valid[:16]}Z\n"
        f"max wind {facts.max_wind_ms:5.2f} m/s {_rank(facts.rank_max_wind, facts)}   "
        f"min RH {facts.min_rh_pct:5.2f}% {_rank(facts.rank_min_rh, facts)}\n"
        f"perimeter {facts.burned_cells} cells {_rank(facts.rank_burned, facts)}\n"
        f"TRUTH grew {facts.truth_growth_cells} cells by 3 h\n"
        f"MODEL p>=0.5: {model.n_confident} cells, {model.n_confident_burned} burned "
        f"(draw A)   {model_b.n_confident} cells, {model_b.n_confident_burned} burned "
        f"(draw B)\n"
        f"MODEL max p {model.max_p:.3f}   cells at p>=0.25: {model.n_p_ge_025}   "
        f"band {model.n_band_cells} cells\n"
        f"MODEL mean member growth {model.mean_pred_new_cells:.1f} cells "
        f"= {_ratio(model.mean_pred_new_cells, facts.truth_growth_cells)} truth\n"
        f"ELLIPSE p>=0.5: {ell.n_confident} cells, {ell.n_confident_burned} burned   "
        f"max p {ell.max_p:.3f}   mean member growth {ell.mean_pred_new_cells:.1f} cells"
    )


def render_episode_page(
    fire_id: str,
    episode: Sequence[int],
    contrast: Sequence[int],
    model: Any,
    ellipse: Any,
    out: str | Path,
    *,
    ranks: dict[int, WindowFacts] | None = None,
    dpi: int = 150,
    provenance: str = "",
) -> dict[str, Any]:
    """Draw the episode and its contrast windows as one page of four columns."""
    t0s = list(episode) + list(contrast)
    refuse_if_empty(
        "the episode page",
        {"windows": len(t0s)},
        because="A page with no windows would still render axes and a title.",
    )
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
        geom = plot_extent(ds["x"].values, ds["y"].values)
        offsets = {d: seed_offsets_for_fire(state, HORIZON, d)[1] for d in DRAWS}
        _, position = window_positions(state, HORIZON)
        u_all = channel_values(ds, "wind_u10", dtype=np.float32)
        v_all = channel_values(ds, "wind_v10", dtype=np.float32)

        n_rows = len(t0s)
        # Header and footer are fixed INCHES, not fractions, so the legend does
        # not slide under the last row when a caller asks for one more window.
        head_in, foot_in, row_in = 0.95, 1.35, 3.35
        height = row_in * n_rows + head_in + foot_in
        fig = plt.figure(figsize=(15.6, height), dpi=dpi)
        gs = fig.add_gridspec(
            n_rows,
            5,
            width_ratios=[1.28, 1.0, 1.0, 1.0, 1.25],
            hspace=0.19,
            wspace=0.05,
            left=0.012,
            right=0.995,
            top=1 - head_in / height,
            bottom=foot_in / height,
        )
        record: list[dict[str, Any]] = []
        im = None
        for i, t0 in enumerate(t0s):
            w = forecast_inputs(ds, t0, HORIZON, fire_id=fire_id)
            facts = window_facts(ds, fire_id, t0, position[t0], HORIZON)
            if ranks is not None and t0 in ranks:
                facts.rank_max_wind = ranks[t0].rank_max_wind
                facts.rank_min_rh = ranks[t0].rank_min_rh
                facts.rank_burned = ranks[t0].rank_burned
                facts.n_growth_windows = ranks[t0].n_growth_windows
            m_a = read_arm(model, w, name="model", draw="A", seed=SEED_BASE + offsets["A"][t0])
            m_b = read_arm(model, w, name="model", draw="B", seed=SEED_BASE + offsets["B"][t0])
            e_a = read_arm(ellipse, w, name="ellipse", draw="A", seed=SEED_BASE + offsets["A"][t0])
            truth_new = (
                event_field(np.asarray(w.truth)[:HORIZON], DEFAULT_EVENT)[HORIZON - 1]
            ) & ~(np.asarray(w.x0) > UNBURNED)
            # The crop is the FIRE, not the confident cells: cropping to where
            # the model committed would size the frame to the answer and make a
            # scattered commitment look compact.
            x0_burned = np.asarray(w.x0) > UNBURNED
            band = scoring_masks(w.x0, (int(w.x0.shape[0]), int(w.x0.shape[1])), HORIZON)[
                "growth_band"
            ]
            box = zoom_box(
                [x0_burned | band, truth_new],
                (state.shape[1], state.shape[2]),
                margin=2,
            )
            in_episode = t0 in set(episode)
            tag = "EPISODE" if in_episode else "CONTRAST"

            ax0 = fig.add_subplot(gs[i, 0])
            _panel_context(
                ax0,
                geom,
                w.x0,
                truth_new,
                m_a.confident,
                (u_all[t0 + 1], v_all[t0 + 1]),
                box,
                facts,
            )
            ax0.text(
                0.012,
                0.02,
                tag,
                transform=ax0.transAxes,
                fontsize=9,
                fontweight="bold",
                color=_COL_CONFIDENT if in_episode else COL_TEXT,
                ha="left",
                va="bottom",
                bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 2.0},
            )

            ax1 = fig.add_subplot(gs[i, 1])
            im = _panel_model(
                ax1,
                geom,
                m_a,
                box,
                facts,
                w.x0,
                truth_new,
                title=(
                    f"MODEL P(NEWLY burned by 3 h), draw A\n"
                    f"{m_a.n_confident} cells at p>=0.5, max p {m_a.max_p:.2f}"
                ),
            )
            ax2 = fig.add_subplot(gs[i, 2])
            _panel_truth(
                ax2,
                geom,
                w.x0,
                w.truth,
                m_a.confident,
                box,
                title=(
                    f"TRUTH: {facts.truth_growth_cells} new cells by 3 h\n"
                    f"model's confident cells outlined"
                ),
            )
            ax3 = fig.add_subplot(gs[i, 3])
            _panel_model(
                ax3,
                geom,
                e_a,
                box,
                facts,
                w.x0,
                truth_new,
                title=(
                    f"{ellipse.name} P(NEWLY burned by 3 h)\n"
                    f"{e_a.n_confident} cells at p>=0.5, max p {e_a.max_p:.2f}"
                ),
                show_members=False,
            )
            inner = gs[i, 4].subgridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42)
            ax4 = fig.add_subplot(inner[0])
            ax4.axis("off")
            ax4.text(
                0.0,
                1.0,
                _row_caption(facts, m_a, m_b, e_a),
                transform=ax4.transAxes,
                fontsize=8,
                family="DejaVu Sans Mono",
                color=COL_TEXT,
                ha="left",
                va="top",
            )
            ax5 = fig.add_subplot(inner[1])
            _panel_commitment(ax5, m_a, e_a)
            record.append(
                {
                    "fire_id": fire_id,
                    "t0": t0,
                    "role": tag.lower(),
                    "facts": asdict(facts),
                    "zoom_box_rows_cols": list(box),
                    "model_drawA": _arm_record(m_a),
                    "model_drawB": _arm_record(m_b),
                    "ellipse_drawA": _arm_record(e_a),
                    "confident_cells_drawA": _cells(m_a, w),
                    "confident_cells_drawB": _cells(m_b, w),
                }
            )
        if im is not None:
            cax = fig.add_axes((0.62, 0.60 / height, 0.20, 0.11 / height))
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_label(
                "P(cell NEWLY burned by 3 h) over 24 members; cells burned at t0 are char",
                fontsize=7,
            )
            cb.ax.tick_params(labelsize=6)
        _episode_legend(fig, y=0.42 / height)
        fig.suptitle(
            f"{fire_id}: what the model expected against what burned",
            fontsize=13.5,
            y=1 - 0.20 / height,
        )
        if provenance:
            stamp(fig, provenance)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return {"figure": str(out), "windows": record}
    finally:
        ds.close()


def _arm_record(r: ArmReading) -> dict[str, Any]:
    return {
        "name": r.name,
        "draw": r.draw,
        "seed": r.seed,
        "n_confident": r.n_confident,
        "n_confident_burned": r.n_confident_burned,
        "max_p": r.max_p,
        "n_p_ge_025": r.n_p_ge_025,
        "mean_pred_new_cells": r.mean_pred_new_cells,
    }


def _cells(r: ArmReading, window: Any) -> list[dict[str, Any]]:
    dist = distance_to_burned(window.x0)
    truth_ev = event_field(np.asarray(window.truth)[:HORIZON], DEFAULT_EVENT)
    rr, cc = np.nonzero(np.asarray(r.confident))
    return [
        {
            "row": int(a),
            "col": int(b),
            "p_3h": float(np.asarray(r.prob)[HORIZON - 1, a, b]),
            "burned_by_3h": bool(truth_ev[HORIZON - 1, a, b]),
            "dist_to_burned_cells": int(dist[a, b]),
        }
        for a, b in zip(rr, cc, strict=True)
    ]


def _episode_legend(fig: Any, *, y: float = 0.002) -> None:
    handles = [
        Line2D([], [], color=_COL_CONFIDENT, lw=1.6, label="model p >= 0.5 (a commitment)"),
        Line2D([], [], color=_COL_NEARMISS, lw=1.4, label="model's highest-p cells (none >= 0.5)"),
        Line2D([], [], color=COL_MEMBER, lw=0.9, alpha=0.5, label="member front at 3 h"),
        Line2D([], [], color=_GREEN_BY_LEAD[0], lw=6, label="truth: burned by 1 h"),
        Line2D([], [], color=_GREEN_BY_LEAD[1], lw=6, label="truth: by 2 h"),
        Line2D([], [], color=_GREEN_BY_LEAD[2], lw=6, label="truth: by 3 h"),
        Line2D([], [], color=STATE_COLORS[2], lw=6, label="burned at t0"),
        Line2D([], [], color=COL_WIND, lw=1.4, label="wind (to leeward)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=4,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.012, y),
    )


def render_regime_page(
    rows: Sequence[dict[str, Any]],
    out: str | Path,
    *,
    highlight: dict[str, Sequence[int]] | None = None,
    fire_id: str = "",
    dpi: int = 150,
    provenance: str = "",
) -> dict[str, Any]:
    """Every growth window in the held-out set, in the covariates that are quoted.

    This page exists to stop the episode being read as "high wind breaks it".
    It carries the negative evidence in two forms: the scatter, where the windows
    the model committed in sit apart from the merely windy and the merely dry
    ones, and two tables listing the windiest and the driest windows that
    produced NOTHING, with the highest probability the model reached in each.
    Where the separation is imperfect it stays visible: a window with a one-cell
    commitment is drawn at its own size, not promoted to look like the episode.
    """
    refuse_if_empty(
        "the regime page",
        {"growth_windows": len(rows)},
        because="An empty scatter still draws axes and reads as 'no window is ever confident'.",
    )
    wind = np.array([r["max_wind_ms"] for r in rows])
    rh = np.array([r["min_rh_pct"] for r in rows])
    burned = np.array([r["burned_cells"] for r in rows])
    conf_a = np.array([r.get("n_confident_A", 0) for r in rows])
    conf_b = np.array([r.get("n_confident_B", 0) for r in rows])
    hot_a = conf_a > 0
    hot_b = conf_b > 0
    cold = ~(hot_a | hot_b)

    fig = plt.figure(figsize=(15.6, 6.4), dpi=dpi)
    # Bottom margin reserved INSIDE the figure so the legend does not land on
    # the provenance stamp once the saved bounding box tightens around it.
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.12], wspace=0.14, bottom=0.24, top=0.93, left=0.07, right=0.99
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(
        wind[cold],
        rh[cold],
        s=5 + 26 * (burned[cold] / max(burned.max(), 1)),
        c="#b9b4ab",
        alpha=0.55,
        linewidths=0,
        label=f"{int(cold.sum())} windows: no cell reaches p>=0.5 in either draw",
    )
    ax.scatter(
        wind[hot_b],
        rh[hot_b],
        s=60,
        facecolors="none",
        edgecolors=COL_TEXT,
        linewidths=0.9,
        label=f"{int(hot_b.sum())} windows with a commitment in draw B",
    )
    ax.scatter(
        wind[hot_a],
        rh[hot_a],
        s=14 + 4.0 * conf_a[hot_a],
        c=_COL_CONFIDENT,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        label=f"{int(hot_a.sum())} windows with a commitment in draw A (area = cells)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("max wind over the 3 driving hours (m/s)")
    ax.set_ylabel("min relative humidity over the 3 driving hours (%), log scale")
    ax.set_title(
        f"all {len(rows)} growth windows of the 5 held-out blocks\n"
        "grey marker area grows with perimeter length, the third conjunct",
        fontsize=9,
    )
    seen: dict[tuple[float, float], list[str]] = {}
    for label, t0s in (highlight or {}).items():
        for t0 in t0s:
            row = next((r for r in rows if r["t0"] == t0 and r["fire_id"] == fire_id), None)
            if row is None:
                continue
            key = (round(row["max_wind_ms"], 3), round(row["min_rh_pct"], 3))
            seen.setdefault(key, []).append(f"{label} t0={t0}")
    # Offsets are staggered by GROUP INDEX, not by value: three windows one hour
    # apart sit on top of each other, and three labels on top of each other read
    # as one illegible label rather than as three windows.
    for i, ((x, y), labels) in enumerate(sorted(seen.items())):
        ax.annotate(
            " / ".join(labels),
            xy=(x, y),
            xytext=(-150, 30 - 34 * i),
            textcoords="offset points",
            fontsize=7.5,
            color=COL_TEXT,
            arrowprops={"arrowstyle": "-", "color": COL_TEXT, "lw": 0.6},
        )
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.10), frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    header = (
        f"{'fire / t0':>26} {'wind':>6} {'RH':>6} {'perim':>6} {'truth':>6} "
        f"{'maxp A':>7} {'p>=.5 A':>8} {'burned':>7} {'p>=.5 B':>8}"
    )

    def _line(r: dict[str, Any]) -> str:
        return (
            f"{r['fire_id'][:18] + '@' + str(r['t0']):>26} {r['max_wind_ms']:6.2f} "
            f"{r['min_rh_pct']:6.2f} {r['burned_cells']:6d} {r['truth_growth_cells']:6d} "
            f"{r.get('max_p_A', 0.0):7.3f} {r.get('n_confident_A', 0):8d} "
            f"{r.get('n_confident_burned_A', 0):7d} {r.get('n_confident_B', 0):8d}"
        )

    committed = sorted(
        (r for r in rows if r.get("n_confident_A", 0) or r.get("n_confident_B", 0)),
        key=lambda r: -max(r.get("n_confident_A", 0), r.get("n_confident_B", 0)),
    )
    quiet = [r for r in rows if not r.get("n_confident_A", 0) and not r.get("n_confident_B", 0)]
    windiest = sorted(quiet, key=lambda r: -r["max_wind_ms"])[:8]
    driest = sorted(quiet, key=lambda r: r["min_rh_pct"])[:8]
    lines = [
        "EVERY window where the model committed, either draw",
        header,
        "-" * len(header),
        *[_line(r) for r in committed],
        "",
        "the WINDIEST windows that produced no commitment at all",
        *[_line(r) for r in windiest],
        "",
        "the DRIEST windows that produced no commitment at all",
        *[_line(r) for r in driest],
        "",
        f"TOTALS over {len(rows)} growth windows: draw A {int(conf_a.sum())} cells at p>=0.5, "
        f"{sum(r.get('n_confident_burned_A', 0) for r in rows)} burned; "
        f"draw B {int(conf_b.sum())} cells, "
        f"{sum(r.get('n_confident_burned_B', 0) for r in rows)} burned.",
        "Wind alone does not do it and dryness alone does not do it: the rows above are "
        "windier, and drier,",
        "than nearly everything in the corpus and the model stays under the line in all of them.",
    ]
    ax2.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax2.transAxes,
        fontsize=7.2,
        family="DejaVu Sans Mono",
        va="top",
        ha="left",
        color=COL_TEXT,
    )
    if provenance:
        stamp(fig, provenance)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return {
        "figure": str(out),
        "n_growth_windows": len(rows),
        "n_windows_with_commitment_drawA": int(hot_a.sum()),
        "n_windows_with_commitment_drawB": int(hot_b.sum()),
        "n_confident_cells_drawA": int(conf_a.sum()),
        "n_confident_cells_drawB": int(conf_b.sum()),
        "n_confident_cells_that_burned_drawA": int(
            sum(r.get("n_confident_burned_A", 0) for r in rows)
        ),
        "n_confident_cells_that_burned_drawB": int(
            sum(r.get("n_confident_burned_B", 0) for r in rows)
        ),
        "windows_with_commitment": [
            {
                "fire_id": r["fire_id"],
                "t0": r["t0"],
                "n_confident_A": r.get("n_confident_A", 0),
                "n_confident_B": r.get("n_confident_B", 0),
                "max_wind_ms": r["max_wind_ms"],
                "min_rh_pct": r["min_rh_pct"],
                "burned_cells": r["burned_cells"],
                "truth_growth_cells": r["truth_growth_cells"],
            }
            for r in committed
        ],
        "windiest_quiet_windows": [
            {k: r[k] for k in ("fire_id", "t0", "max_wind_ms", "min_rh_pct", "max_p_A")}
            for r in windiest
        ],
        "driest_quiet_windows": [
            {k: r[k] for k in ("fire_id", "t0", "max_wind_ms", "min_rh_pct", "max_p_A")}
            for r in driest
        ],
    }


def render_episode_movie(
    fire_id: str,
    t0_first: int,
    t0_last: int,
    model: Any,
    ellipse: Any,
    out: str | Path,
    *,
    episode: Sequence[int] = (),
    fps: int = 1,
    dpi: int = 110,
    provenance: str = "",
) -> dict[str, Any]:
    """The same three panels, one frame per hour, across and around the episode.

    A still page shows that the model committed; the movie shows the commitment
    ARRIVING and LEAVING while the fire underneath it does almost nothing. The
    crop is computed once over every frame, so the frame does not jitter and the
    eye can attribute a change to the fire rather than to the camera.
    """
    frames = list(range(int(t0_first), int(t0_last) + 1))
    refuse_if_empty(
        "the episode movie",
        {"frames": len(frames)},
        because="A zero-frame movie writes a valid file that plays nothing.",
    )
    from wildfire_nowcast.sim.movie import _writer  # noqa: PLC0415

    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
        geom = plot_extent(ds["x"].values, ds["y"].values)
        offsets = seed_offsets_for_fire(state, HORIZON, "A")[1]
        _, position = window_positions(state, HORIZON)
        u_all = channel_values(ds, "wind_u10", dtype=np.float32)
        v_all = channel_values(ds, "wind_v10", dtype=np.float32)
        shape = (int(state.shape[1]), int(state.shape[2]))
        spans = []
        for t0 in frames:
            x0b = state[t0] > UNBURNED
            band = scoring_masks(state[t0], shape, HORIZON)["growth_band"]
            spans.append(x0b | band)
        box = zoom_box([np.logical_or.reduce(spans)], shape, margin=2)

        writer, out_path = _writer(Path(out), fps)
        fig = plt.figure(figsize=(12.6, 5.2), dpi=dpi)
        rows: list[dict[str, Any]] = []
        with writer.saving(fig, str(out_path), dpi=dpi):
            for t0 in frames:
                w = forecast_inputs(ds, t0, HORIZON, fire_id=fire_id)
                facts = window_facts(ds, fire_id, t0, position[t0], HORIZON)
                m = read_arm(model, w, name="model", draw="A", seed=SEED_BASE + offsets[t0])
                e = read_arm(ellipse, w, name="ellipse", draw="A", seed=SEED_BASE + offsets[t0])
                truth_new = event_field(np.asarray(w.truth)[:HORIZON], DEFAULT_EVENT)[
                    HORIZON - 1
                ] & ~(np.asarray(w.x0) > UNBURNED)
                fig.clear()
                gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.0, 1.0, 1.0], wspace=0.05)
                ax0 = fig.add_subplot(gs[0, 0])
                _panel_context(
                    ax0,
                    geom,
                    w.x0,
                    truth_new,
                    m.confident,
                    (u_all[t0 + 1], v_all[t0 + 1]),
                    box,
                    facts,
                )
                ax1 = fig.add_subplot(gs[0, 1])
                _panel_model(
                    ax1,
                    geom,
                    m,
                    box,
                    facts,
                    w.x0,
                    truth_new,
                    title=f"MODEL: {m.n_confident} cells at p>=0.5, max p {m.max_p:.2f}",
                )
                ax2 = fig.add_subplot(gs[0, 2])
                _panel_truth(
                    ax2,
                    geom,
                    w.x0,
                    w.truth,
                    m.confident,
                    box,
                    title=f"TRUTH: {facts.truth_growth_cells} new cells by 3 h",
                )
                ax3 = fig.add_subplot(gs[0, 3])
                _panel_model(
                    ax3,
                    geom,
                    e,
                    box,
                    facts,
                    w.x0,
                    truth_new,
                    title=f"{ellipse.name}: {e.n_confident} cells at p>=0.5",
                    show_members=False,
                )
                mark = "  <<< EPISODE" if t0 in set(episode) else ""
                fig.suptitle(
                    f"{fire_id}   t0 = {t0}   {facts.time_t0[:16]}Z  ->  3 h valid "
                    f"{facts.time_valid[:16]}Z{mark}\n"
                    f"max wind {facts.max_wind_ms:.1f} m/s   min RH {facts.min_rh_pct:.2f}%   "
                    f"model commits {m.n_confident} cells at p>=0.5   truth burns "
                    f"{facts.truth_growth_cells} cells",
                    fontsize=10,
                    color=_COL_CONFIDENT if mark else COL_TEXT,
                )
                if provenance:
                    stamp(fig, provenance)
                writer.grab_frame(facecolor="white")
                rows.append(
                    {
                        "t0": t0,
                        "time_t0": facts.time_t0,
                        "max_wind_ms": facts.max_wind_ms,
                        "min_rh_pct": facts.min_rh_pct,
                        "truth_growth_cells": facts.truth_growth_cells,
                        "model_n_confident": m.n_confident,
                        "model_max_p": m.max_p,
                        "model_mean_member_growth": m.mean_pred_new_cells,
                        "ellipse_n_confident": e.n_confident,
                    }
                )
        plt.close(fig)
        return {"movie": str(out_path), "frames": rows}
    finally:
        ds.close()


def sweep_window_range(
    fire_id: str,
    t0_first: int,
    t0_last: int,
    model: Any,
    ellipse: Any,
    *,
    draws: Sequence[str] = DRAWS,
    n_members: int = MEMBERS,
    horizon_h: int = HORIZON,
) -> list[dict[str, Any]]:
    """Every window in a t0 range, INCLUDING the ones the gate never scores.

    The scored stratum is growth windows only. That is a defensible choice -- a
    window where the label records no new cell asks the model nothing it can be
    graded on -- but it means a commitment made in a zero-growth hour is
    invisible to every published number. This function looks at both kinds and
    labels each one, so the size of the invisible part is measurable rather than
    assumed to be small.
    """
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
        burned = state > UNBURNED
        offsets = {d: seed_offsets_for_fire(state, horizon_h, d)[1] for d in draws}
        _, position = window_positions(state, horizon_h)
        rows: list[dict[str, Any]] = []
        for t0 in range(int(t0_first), int(t0_last) + 1):
            if t0 not in position:
                continue
            w = forecast_inputs(ds, t0, horizon_h, fire_id=fire_id)
            facts = window_facts(ds, fire_id, t0, position[t0], horizon_h)
            growth = int(np.count_nonzero(burned[t0 + horizon_h] & ~burned[t0]))
            row: dict[str, Any] = asdict(facts)
            row["scored_by_the_gate"] = bool(growth > 0)
            row["new_cells_this_hour"] = int(np.count_nonzero(burned[t0 + 1] & ~burned[t0]))
            for d in draws:
                r = read_arm(
                    model,
                    w,
                    name="model",
                    draw=d,
                    seed=SEED_BASE + offsets[d][t0],
                    n_members=n_members,
                    horizon_h=horizon_h,
                )
                row[f"n_confident_{d}"] = r.n_confident
                row[f"n_confident_burned_{d}"] = r.n_confident_burned
                row[f"max_p_{d}"] = r.max_p
                row[f"mean_member_growth_{d}"] = r.mean_pred_new_cells
            e = read_arm(
                ellipse,
                w,
                name="ellipse",
                draw="A",
                seed=SEED_BASE + offsets["A"][t0],
                n_members=n_members,
                horizon_h=horizon_h,
            )
            row["ellipse_n_confident"] = e.n_confident
            row["ellipse_mean_member_growth"] = e.mean_pred_new_cells
            rows.append(row)
        refuse_if_empty(
            "the window-range sweep",
            {"windows": len(rows)},
            because="A range with no evaluable window would draw a flat timeline reading "
            "'the model never commits here'.",
        )
        return rows
    finally:
        ds.close()


def render_run_page(
    rows: Sequence[dict[str, Any]],
    out: str | Path,
    *,
    fire_id: str = "",
    dpi: int = 150,
    provenance: str = "",
) -> dict[str, Any]:
    """The commitment, the label and the weather on one time axis.

    Three panels sharing ``t0`` so that the scored windows can be seen in their
    context. The published episode is whatever part of this run happens to fall
    in a window the gate scores; that is a property of the SCORING SET as much
    as of the model, and it is only visible when the unscored hours are drawn
    beside the scored ones instead of dropped.
    """
    refuse_if_empty(
        "the run page",
        {"windows": len(rows)},
        because="An empty timeline is indistinguishable from a quiet one.",
    )
    t0 = np.array([r["t0"] for r in rows])
    conf_a = np.array([r.get("n_confident_A", 0) for r in rows])
    conf_b = np.array([r.get("n_confident_B", 0) for r in rows])
    scored = np.array([bool(r["scored_by_the_gate"]) for r in rows])
    truth = np.array([r["truth_growth_cells"] for r in rows])
    grow_h = np.array([r["new_cells_this_hour"] for r in rows])
    mean_growth = np.array([r.get("mean_member_growth_A", 0.0) for r in rows])
    wind = np.array([r["max_wind_ms"] for r in rows])
    rh = np.array([r["min_rh_pct"] for r in rows])
    perim = np.array([r["burned_cells"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.4), dpi=dpi, sharex=True)
    for ax in axes:
        for i in range(len(rows)):
            if scored[i]:
                ax.axvspan(t0[i] - 0.5, t0[i] + 0.5, color="#dff0d8", zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ell = np.array([r.get("ellipse_n_confident", 0) for r in rows])
    ax = axes[0]
    ax.bar(t0, conf_a, width=0.82, color=_COL_CONFIDENT, label="model, draw A")
    ax.plot(t0, conf_b, color=COL_TEXT, lw=1.0, marker="o", ms=2.6, label="model, draw B")
    ax.plot(t0, ell, color="#6b7280", lw=1.2, label="calibrated ellipse, draw A")
    ax.add_patch(
        Rectangle((0, 0), 0, 0, facecolor="#dff0d8", label="hours the gate scores (truth grew)")
    )
    ax.set_ylabel("cells at p >= 0.5")
    ax.set_title(
        f"{fire_id}: the model's commitments over the whole wind event, scored hours shaded green",
        fontsize=11,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    in_scored = int(conf_a[scored].sum())
    total = int(conf_a.sum())
    ax.text(
        0.005,
        0.96,
        f"draw A: {total} cells at p>=0.5 over these {len(rows)} windows, of which "
        f"{in_scored} ({100 * in_scored / max(total, 1):.1f}%) are in the "
        f"{int(scored.sum())} windows the gate scores.\n"
        f"draw B: {int(conf_b.sum())} cells, {int(conf_b[scored].sum())} of them scored. "
        f"Cells that burned: {sum(r.get('n_confident_burned_A', 0) for r in rows)} (draw A), "
        f"{sum(r.get('n_confident_burned_B', 0) for r in rows)} (draw B).",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        color=COL_TEXT,
    )

    ax = axes[1]
    ax.bar(t0, mean_growth, width=0.82, color="#f0a58f", label="model mean member growth, 3 h")
    ax.bar(t0, truth, width=0.5, color=_GREEN_BY_LEAD[2], label="truth new cells by 3 h")
    ax.set_ylabel("new cells in 3 h")
    ax.plot([], [], color=COL_TEXT, lw=1.2, ls=(0, (4, 2)), label="cells burned at t0 (right)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(t0, perim, color=COL_TEXT, lw=1.2, ls=(0, (4, 2)))
    ax2.set_ylabel("cells burned at t0 (dashed)", fontsize=8)
    ax2.spines["top"].set_visible(False)
    steps = [i for i, v in enumerate(grow_h) if v > 0]
    for i in steps:
        ax.annotate(
            f"the label moves: +{int(grow_h[i])} cells at {str(rows[i]['time_t0'])[:16]}Z",
            xy=(t0[i], truth[i]),
            xytext=(-30, 46),
            textcoords="offset points",
            fontsize=8,
            color=COL_TEXT,
            arrowprops={"arrowstyle": "->", "color": COL_TEXT, "lw": 0.8},
        )

    ax = axes[2]
    ax.plot(t0, wind, color=COL_WIND, lw=1.4, label="max wind over the 3 driving hours")
    ax.set_ylabel("m/s", color=COL_WIND)
    ax.set_xlabel(
        f"t0, hour index into the fire's own tensor "
        f"({str(rows[0]['time_t0'])[:16]}Z to {str(rows[-1]['time_t0'])[:16]}Z; "
        "each window is scored 3 h after its own t0)"
    )
    ax3 = ax.twinx()
    ax3.plot(t0, rh, color="#7c3aed", lw=1.4, label="min RH")
    ax3.set_yscale("log")
    ax3.set_ylabel("min relative humidity (%), log", color="#7c3aed")
    ax3.spines["top"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax3.legend(fontsize=7.5, frameon=False, loc="upper right")

    if provenance:
        stamp(fig, provenance)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return {
        "figure": str(out),
        "fire_id": fire_id,
        "t0_first": int(t0[0]),
        "t0_last": int(t0[-1]),
        "n_windows": len(rows),
        "n_windows_scored": int(scored.sum()),
        "n_confident_cells_drawA": int(conf_a.sum()),
        "n_confident_cells_drawA_in_scored_windows": int(conf_a[scored].sum()),
        "n_confident_cells_drawB": int(conf_b.sum()),
        "n_confident_cells_drawB_in_scored_windows": int(conf_b[scored].sum()),
        "n_confident_cells_that_burned_drawA": int(
            sum(r.get("n_confident_burned_A", 0) for r in rows)
        ),
        "n_confident_cells_that_burned_drawB": int(
            sum(r.get("n_confident_burned_B", 0) for r in rows)
        ),
        "label_increments": [
            {
                "t0_of_step": int(rows[i]["t0"]),
                "time_utc": str(rows[i]["time_t0"]),
                "new_cells": int(grow_h[i]),
            }
            for i in steps
        ],
        "rows": list(rows),
    }


# -- cross-check -----------------------------------------------------------


def _load_dump(artifact: str | Path) -> list[dict[str, Any]]:
    path = Path(artifact)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        payload = json.load(fh)
    return list(payload.get("confident_cells") or [])


def cross_check_sweep(
    rows: Sequence[dict[str, Any]], artifact: str | Path, draw: str
) -> dict[str, Any]:
    """Compare the WHOLE sweep's per-window commitment counts with a run's dump.

    The per-window cross-check proves three windows agree. This proves the
    population does: every window where either side found a commitment appears,
    with both counts, so a window this module missed is as visible as a window
    it invented.
    """
    dump = _load_dump(artifact)
    theirs: dict[tuple[str, int], int] = {}
    for c in dump:
        key = (str(c["fire_id"]), int(c["t0"]))
        theirs[key] = theirs.get(key, 0) + 1
    mine = {
        (str(r["fire_id"]), int(r["t0"])): int(r.get(f"n_confident_{draw}", 0))
        for r in rows
        if int(r.get(f"n_confident_{draw}", 0)) > 0
    }
    keys = sorted(set(mine) | set(theirs))
    windows = [
        {"fire_id": k[0], "t0": k[1], "n_mine": mine.get(k, 0), "n_artifact": theirs.get(k, 0)}
        for k in keys
    ]
    return {
        "artifact": str(artifact),
        "draw": draw,
        "n_growth_windows_swept": len(rows),
        "n_cells_mine": int(sum(int(r.get(f"n_confident_{draw}", 0)) for r in rows)),
        "n_cells_artifact": len(dump),
        "n_cells_mine_that_burned": int(
            sum(int(r.get(f"n_confident_burned_{draw}", 0)) for r in rows)
        ),
        "n_cells_artifact_that_burned": int(sum(1 for c in dump if c.get("burned_by_3h"))),
        "windows_with_a_commitment": windows,
        "every_window_agrees": all(w["n_mine"] == w["n_artifact"] for w in windows),
    }


def cross_check(page: dict[str, Any], artifact: str | Path, draw: str) -> dict[str, Any]:
    """Compare this module's confident cells with a run artifact's own dump.

    Set equality per window, not counts: two different 92-cell sets agree on
    every count anyone would print and disagree about the entire finding.
    """
    path = Path(artifact)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        payload = json.load(fh)
    dump = payload.get("confident_cells") or []
    theirs: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for c in dump:
        theirs.setdefault((str(c["fire_id"]), int(c["t0"])), set()).add(
            (int(c["row"]), int(c["col"]))
        )
    out: dict[str, Any] = {
        "artifact": str(path),
        "draw": draw,
        "n_cells_in_artifact": len(dump),
        "windows": [],
        "identical_on_every_window": True,
    }
    for w in page["windows"]:
        key = (str(w["fire_id"]), int(w["t0"]))
        mine = {(c["row"], c["col"]) for c in w[f"confident_cells_draw{draw}"]}
        ref = theirs.get(key, set())
        same = mine == ref
        out["identical_on_every_window"] = bool(out["identical_on_every_window"] and same)
        out["windows"].append(
            {
                "fire_id": key[0],
                "t0": key[1],
                "n_mine": len(mine),
                "n_artifact": len(ref),
                "identical": same,
                "only_mine": sorted(mine - ref)[:8],
                "only_artifact": sorted(ref - mine)[:8],
            }
        )
    return out


# -- CLI -------------------------------------------------------------------


def _calibrated_ellipse(calibration_artifact: str | Path, horizon_h: int, workdir: Path) -> Any:
    """The gate's own ellipse at the matching horizon, built by ONE constructor.

    The run artifact stores the calibration in two flat keys; the constructor in
    :mod:`wildfire_nowcast.sim.replay` reads the nested shape a results file
    has. This copies the two blocks verbatim into that shape rather than
    rebuilding the ellipse here, because a second construction site is a second
    set of choices about which scale is "the" calibrated one.
    """
    from wildfire_nowcast.sim.replay import load_gate_models  # noqa: PLC0415

    path = Path(calibration_artifact)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        payload = json.load(fh)
    if "ellipse_calibration_alt" in payload:
        shim = {
            "ellipse_calibration": {
                "rule_of_record": payload["ellipse_calibration"],
                "alternative_horizons": payload["ellipse_calibration_alt"],
            }
        }
    else:
        shim = {"ellipse_calibration": payload["ellipse_calibration"]}
    workdir.mkdir(parents=True, exist_ok=True)
    shim_path = workdir / "ellipse_calibration_shim.json"
    shim_path.write_text(json.dumps(shim, indent=1) + "\n")
    gate = load_gate_models(shim_path, include_persistence=False)
    name = f"ellipse_cal{int(horizon_h)}h"
    if name not in gate.models:
        raise KeyError(
            f"{name} is not in the calibration artifact; it carries "
            f"{sorted(gate.models)}. Refusing to substitute a different horizon: "
            "the per-horizon rule exists so that a comparison cannot be made at "
            "the lead where the opponent is weakest."
        )
    return gate.models[name]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.episode",
        description="Render an episode of confident-and-wrong forecasting beside its controls.",
    )
    ap.add_argument("--fire", required=True)
    ap.add_argument("--episode", required=True, help="comma-separated t0 values")
    ap.add_argument("--contrast", default="", help="comma-separated t0 values, the controls")
    ap.add_argument("--out", required=True, help="episode page (.png)")
    ap.add_argument("--regime-out", default=None, help="regime page (.png); needs the sweep")
    ap.add_argument("--json", dest="json_out", required=True, help="every number on the pages")
    ap.add_argument("--calibration", required=True, help="run artifact carrying ellipse scales")
    ap.add_argument("--cross-check", default=None, help="run artifact carrying a cell dump")
    ap.add_argument("--cross-check-draw", default="A", choices=list(DRAWS))
    ap.add_argument("--sweep-json", default=None, help="read/write the growth-window sweep here")
    ap.add_argument("--sweep", action="store_true", help="run the sweep even if the file exists")
    ap.add_argument("--run-out", default=None, help="commitment-over-time page (.png)")
    ap.add_argument("--run-window", default="", help="first,last t0 for the run page")
    ap.add_argument("--movie-out", default=None, help="animated episode (.gif or .mp4)")
    ap.add_argument("--movie-window", default="", help="first,last t0 for the movie")
    ap.add_argument("--movie-fps", type=int, default=1)
    ap.add_argument("--members", type=int, default=MEMBERS)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    from wildfire_nowcast.model.reference import (  # noqa: PLC0415
        REFERENCE_FIT_ADDRESS,
        load_reference_fit,
        reference_fit_sha256,
    )

    episode = [int(v) for v in args.episode.split(",") if v.strip()]
    contrast = [int(v) for v in args.contrast.split(",") if v.strip()]
    out_dir = Path(args.out).parent
    model = load_reference_fit()
    model.name = "reference_fit"
    ellipse = _calibrated_ellipse(args.calibration, HORIZON, out_dir)
    provenance = (
        f"model {REFERENCE_FIT_ADDRESS} sha256 {reference_fit_sha256()[:16]} | "
        f"{args.members} members, horizon {HORIZON} h, seed base {SEED_BASE} | "
        f"ellipse {ellipse.name} from {args.calibration} | "
        f"raster is draw A; both draws are computed and printed"
    )

    rows: list[dict[str, Any]] = []
    sweep_path = Path(args.sweep_json) if args.sweep_json else None
    if sweep_path is not None and sweep_path.is_file() and not args.sweep:
        rows = json.loads(sweep_path.read_text())
    elif sweep_path is not None:
        from wildfire_nowcast.eval.baseline_run import load_splits  # noqa: PLC0415

        fires = [s.fire_id for s in load_splits([0, 1, 2, 4]) if not s.is_train]
        rows = sweep_growth_windows(fires, model, n_members=args.members, progress=True)
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(json.dumps(rows) + "\n")

    ranks = {
        int(r["t0"]): WindowFacts(
            **{k: v for k, v in r.items() if k in WindowFacts.__annotations__}
        )
        for r in rows
        if r["fire_id"] == args.fire
    }
    page = render_episode_page(
        args.fire,
        episode,
        contrast,
        model,
        ellipse,
        args.out,
        ranks=ranks or None,
        dpi=args.dpi,
        provenance=provenance,
    )
    payload: dict[str, Any] = {
        "kind": "sim.episode",
        "fire_id": args.fire,
        "episode_t0": episode,
        "contrast_t0": contrast,
        "provenance": provenance,
        "confident_p": CONFIDENT_P,
        "page": page,
    }
    if args.cross_check:
        payload["cross_check"] = cross_check(page, args.cross_check, args.cross_check_draw)
        if rows:
            payload["cross_check_sweep"] = cross_check_sweep(
                rows, args.cross_check, args.cross_check_draw
            )
    if args.run_out:
        first, last = (
            [int(v) for v in args.run_window.split(",")]
            if args.run_window
            else [min(episode) - 24, max(episode) + 24]
        )
        run_rows = sweep_window_range(
            args.fire, first, last, model, ellipse, n_members=args.members
        )
        payload["run"] = render_run_page(
            run_rows, args.run_out, fire_id=args.fire, dpi=args.dpi, provenance=provenance
        )
        r = payload["run"]
        print(
            f"run page {first}..{last}: {r['n_confident_cells_drawA']} cells at p>=0.5 "
            f"(draw A), {r['n_confident_cells_drawA_in_scored_windows']} of them in the "
            f"{r['n_windows_scored']} of {r['n_windows']} windows the gate scores; "
            f"{r['n_confident_cells_that_burned_drawA']} burned"
        )
    if args.movie_out:
        first, last = (
            [int(v) for v in args.movie_window.split(",")]
            if args.movie_window
            else [min(episode) - 4, max(episode) + 4]
        )
        payload["movie"] = render_episode_movie(
            args.fire,
            first,
            last,
            model,
            ellipse,
            args.movie_out,
            episode=episode,
            fps=args.movie_fps,
            dpi=110,
            provenance=provenance,
        )
    if rows and args.regime_out:
        payload["regime"] = render_regime_page(
            rows,
            args.regime_out,
            highlight={"episode": episode, "control": contrast},
            fire_id=args.fire,
            dpi=args.dpi,
            provenance=provenance,
        )
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(payload, indent=1) + "\n")
    for w in page["windows"]:
        f = w["facts"]
        print(
            f"{w['role']:>8} t0={w['t0']:>5} wind {f['max_wind_ms']:6.2f} "
            f"RH {f['min_rh_pct']:6.2f} truth +{f['truth_growth_cells']:>3} "
            f"model p>=0.5 A {w['model_drawA']['n_confident']:>3} "
            f"B {w['model_drawB']['n_confident']:>3} "
            f"ellipse {w['ellipse_drawA']['n_confident']:>3}"
        )
    if "cross_check" in payload:
        ok = payload["cross_check"]["identical_on_every_window"]
        print(f"cross-check: cell sets identical on every rendered window: {ok}")
    if "cross_check_sweep" in payload:
        cs = payload["cross_check_sweep"]
        print(
            f"cross-check over the whole sweep: {cs['n_cells_mine']} cells here vs "
            f"{cs['n_cells_artifact']} in the artifact, "
            f"{cs['n_cells_mine_that_burned']} vs {cs['n_cells_artifact_that_burned']} burned, "
            f"every window agrees: {cs['every_window_agrees']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
