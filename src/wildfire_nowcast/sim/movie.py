"""Fire-movie renderer: ``fire_state`` over time with a wind quiver overlay.

    make movie TENSOR=data/fires/2019_kincade/tensor.zarr MOVIE=reports/figures/kincade.mp4

Three deliberate design choices, each answering a way this movie could lie.

**1. Dormancy is drawn, not dropped.** Under C1.1 ``fireline_v2``, 6-37% of real
frames have ZERO cells in state 1 - the fire is still there, GOES just cannot
see a fire line. A renderer that only draws state 1 goes blank on those frames
and reads as missing data. So every frame draws the burned region *and* the
frontier of the burned region (which C1.1 names as the true contagion source),
and a dormant frame gets an explicit badge with its run length. Blank is never
ambiguous.

**2. Growth is bursty and hour-locked, so the movie carries a clock.** 51-91% of
GOFER hours have bitwise-zero growth; Kincade did ~250 of its 347 km2 in about
12 hours. At a uniform frame rate that is a still image punctuated by two
violent minutes, and a viewer cannot tell "nothing happening" from "renderer
stuck". Every frame therefore carries a growth timeline strip showing where the
current hour sits in the fire's whole history. ``--pacing event`` additionally
*dwells* on high-growth hours; it never removes an hour, so it cannot hide
anything, and the pacing mode is stamped into the figure because a screenshot
outlives its caption.

**3. New cells with no burned neighbour last hour are marked.** Those are
"teleports" - genuine spotting, or GOFER's polygon hull snapping outward. data
QA measured 17 such steps on Kincade alone with a 4.47 km max gap. Marking them
in the base movie means the P3 crossing episodes are visible without a special
tool, and a model that teleports for the wrong reason is visible immediately.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from wildfire_nowcast.common.logs import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
)
from wildfire_nowcast.sim.reader import FireFrames, load_fire  # noqa: E402
from wildfire_nowcast.sim.style import (  # noqa: E402
    COL_BARRIER,
    COL_FRONTIER,
    COL_TEXT,
    COL_WARN,
    COL_WIND,
    STATE_COLORS,
    STATE_LABELS,
    add_north_arrow,
    add_scale_bar,
    quiver_grid,
    stamp,
    state_cmap,
)

# ADR-103: a logger, and NOTHING else at import. `main` configures. This module
# is a LIBRARY to `review.py` and to every lead who renders a fire, so the
# container it fell back to and the probe that failed are diagnostics, not the
# program's output. The summary table `main` prints is the output.
logger = logging.getLogger(__name__)

__all__ = [
    "MovieSpec",
    "render_movie",
    "render_frame",
    "frame_order",
    "teleport_cells",
    "max_front_gap_cells",
    "gap_summary",
    "main",
]

COL_NEW = "#ffffff"
COL_TELEPORT = "#e0218a"


# -- pacing ----------------------------------------------------------------


def frame_order(
    growth_km2: np.ndarray, *, pacing: str = "uniform", max_dwell: int = 4
) -> list[int]:
    """Hour indices in render order.

    ``uniform`` is one frame per hour: the honest default, and the only pacing
    from which a viewer can read elapsed time off the frame counter.

    ``event`` repeats high-growth hours up to ``max_dwell`` times so a burst is
    watchable at normal fps. It is strictly ADDITIVE - every hour still appears
    exactly once before any repeat - so event pacing can slow the movie down but
    can never drop an hour, and therefore cannot hide a defect.
    """
    n = int(np.asarray(growth_km2).size)
    if pacing == "uniform":
        return list(range(n))
    if pacing != "event":
        raise ValueError(f"pacing must be 'uniform' or 'event', got {pacing!r}")
    g = np.asarray(growth_km2, dtype=np.float64)
    g = np.where(np.isfinite(g), np.maximum(g, 0.0), 0.0)
    top = float(g.max())
    if top <= 0:
        return list(range(n))
    # Square-root so a single 30 km2 hour does not swamp a run of 5 km2 hours.
    dwell = 1 + np.floor((max_dwell - 1) * np.sqrt(g / top)).astype(int)
    order: list[int] = []
    for t in range(n):
        order.extend([t] * int(dwell[t]))
    return order


DEFAULT_MIN_GAP_CELLS = 3


def _dilate(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """``k`` rounds of 8-connected binary dilation (Chebyshev ball of radius k)."""
    m = np.asarray(mask, dtype=bool)
    ny, nx = m.shape
    for _ in range(max(0, k)):
        out = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ys = slice(max(0, dy), ny + min(0, dy))
                xs = slice(max(0, dx), nx + min(0, dx))
                yd = slice(max(0, -dy), ny + min(0, -dy))
                xd = slice(max(0, -dx), nx + min(0, -dx))
                out[yd, xd] |= m[ys, xs]
        m = out
    return m


def teleport_cells(
    ever: np.ndarray, t: int, *, min_gap_cells: int = DEFAULT_MIN_GAP_CELLS
) -> np.ndarray:
    """Cells newly burned at ``t`` further than ``min_gap_cells`` from any cell
    burned at ``t-1`` (Chebyshev distance, i.e. 8-connected dilation rounds).

    The threshold is not decoration. data QA's definition - "no burned
    8-NEIGHBOUR at t-1", i.e. ``min_gap_cells=1`` - is the right detector on real
    GOFER, where the front advances about a cell an hour, and it is useless on
    anything faster: the C4 synthetic fire advances 2-3 cells/hour, so every
    single step trips it and the marker stops meaning anything. At the default
    of 3 a flagged cell has skipped at least two full cells of unburned ground,
    which no contiguous 1 km/h front can do.

    Genuine long-range spotting and GOFER hull-snapping look identical here;
    the point of drawing them is that both are worth a human glance.
    """
    ever = np.asarray(ever, dtype=bool)
    if t <= 0 or not ever[t - 1].any():
        # FIRST APPEARANCE. With no prior burned region there is nothing to have
        # jumped away FROM, so every cell would measure as an infinite teleport.
        # Reporting an ignition as a spot event is a false positive that scales
        # with the domain, not with the fire.
        return np.zeros(ever.shape[1:], dtype=bool)
    new = ever[t] & ~ever[t - 1]
    if not new.any():
        return new
    return new & ~_dilate(ever[t - 1], max(1, int(min_gap_cells)))


def max_front_gap_cells(ever: np.ndarray, t: int, *, limit: int = 24) -> int:
    """Largest Chebyshev jump made by any cell newly burned at ``t``.

    ``0`` means every new cell touched last hour's burned region; ``limit`` means
    "at least ``limit``" (the search is capped so this stays cheap).

    Returns ``0`` on a FIRST-APPEARANCE step (``ever[t-1]`` empty). This was a
    real false positive: Zogg reported a 24 km max front gap and CZU 24 km, both
    of which were simply the fire's ignition hour measured against an empty prior
    region, and both of which read in the summary as a domain-scale teleport.
    Kincade's genuine maximum is 3 km.
    """
    ever = np.asarray(ever, dtype=bool)
    if t <= 0 or not ever[t - 1].any() or not (ever[t] & ~ever[t - 1]).any():
        return 0
    new = ever[t] & ~ever[t - 1]
    grown = ever[t - 1]
    for k in range(1, limit + 1):
        grown = _dilate(grown, 1)
        if not (new & ~grown).any():
            return k - 1
    return limit


# -- rendering -------------------------------------------------------------


@dataclass
class MovieSpec:
    """Rendering options. Defaults are the honest ones."""

    pacing: str = "uniform"
    fps: int = 4
    dpi: int = 130
    quiver_target: int = 13
    show_wind: bool = True
    show_teleports: bool = True
    max_dwell: int = 4
    min_gap_cells: int = DEFAULT_MIN_GAP_CELLS


#: Figure furniture, in INCHES. Laid out explicitly rather than through
#: gridspec fractions because the map axes must keep the data's aspect ratio: a
#: fraction-based layout leaves a different amount of dead space for every fire
#: shape, and the timeline strip drifts around underneath it.
_M_LEFT, _M_RIGHT, _M_TOP, _M_BOTTOM = 0.30, 0.75, 1.02, 0.52
_TIMELINE_H, _GAP = 1.00, 0.62
_MAP_W_RANGE = (4.6, 8.2)
_MAP_H_RANGE = (2.6, 7.6)


def _new_figure(fire: FireFrames, dpi: int) -> tuple[Any, Any, Any]:
    """Figure sized so the map axes exactly fits the fire's aspect ratio."""
    aspect = fire.geom.height_km / max(fire.geom.width_km, 1e-9)
    map_w = float(np.clip(_MAP_H_RANGE[1] / aspect, *_MAP_W_RANGE))
    map_h = float(np.clip(map_w * aspect, *_MAP_H_RANGE))
    map_w = float(np.clip(map_h / aspect, *_MAP_W_RANGE))

    fig_w = _M_LEFT + map_w + _M_RIGHT
    fig_h = _M_BOTTOM + _TIMELINE_H + _GAP + map_h + _M_TOP
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    ax_map = fig.add_axes(
        [_M_LEFT / fig_w, (_M_BOTTOM + _TIMELINE_H + _GAP) / fig_h, map_w / fig_w, map_h / fig_h]
    )
    ax_time = fig.add_axes(
        [(_M_LEFT + 0.35) / fig_w, _M_BOTTOM / fig_h, (map_w - 0.35) / fig_w, _TIMELINE_H / fig_h]
    )
    fig._layout_in = (fig_w, fig_h)  # noqa: SLF001 - used to place the legend
    return fig, ax_map, ax_time


def _draw_timeline(ax: Any, fire: FireFrames, t: int) -> None:
    """Growth bars + cumulative area, with dormancy shaded. The movie's clock."""
    hours = np.arange(fire.n_hours)
    ax.clear()
    for start, stop in _runs(fire.dormant):
        ax.axvspan(start - 0.5, stop - 0.5, color="#d7d3cb", alpha=0.75, lw=0, zorder=0)
    ax.bar(hours, fire.growth_km2, width=0.9, color="#ff5a1f", zorder=2)
    ax.axvline(t, color=COL_WARN, lw=1.5, zorder=4)
    ax.set_xlim(-0.5, fire.n_hours - 0.5)
    ax.set_ylim(0, max(float(fire.growth_km2.max()) * 1.15, 1e-6))
    ax.set_ylabel("growth\nkm²/h", fontsize=7.5, color=COL_TEXT)
    ax.set_xlabel("hour index (end-of-hour stamps, C1.3)", fontsize=7.5, color=COL_TEXT)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    twin = getattr(ax, "_area_twin", None)
    if twin is None:
        twin = ax.twinx()
        ax._area_twin = twin  # noqa: SLF001
    twin.clear()
    # Axes.clear() undoes what twinx() set up, so the right-hand axis has to be
    # re-established every frame or both y-labels stack on the left.
    twin.yaxis.set_label_position("right")
    twin.yaxis.tick_right()
    twin.plot(hours, fire.area_km2, color="#3a3128", lw=1.2, zorder=3)
    twin.plot([t], [fire.area_km2[t]], "o", ms=4, color=COL_WARN, zorder=5)
    twin.set_ylim(0, max(float(fire.area_km2.max()) * 1.12, 1e-6))
    twin.set_ylabel("cumulative\nkm²", fontsize=7.5, color="#3a3128")
    twin.tick_params(labelsize=7)
    twin.spines["top"].set_visible(False)

    zero_frac = float((fire.growth_km2[1:] == 0).mean()) if fire.n_hours > 1 else 0.0
    ax.text(
        0.006,
        0.93,
        f"{zero_frac:.0%} of hours have bitwise-zero growth",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#6b7280",
        zorder=6,
    )


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs where ``flags`` is True."""
    f = np.asarray(flags, dtype=bool)
    if not f.any():
        return []
    d = np.diff(f.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def render_frame(ax: Any, fire: FireFrames, t: int, spec: MovieSpec) -> None:
    """Draw hour ``t`` of ``fire`` onto ``ax`` (map panel only)."""
    geom = fire.geom
    ax.clear()

    ax.imshow(fire.state[t], cmap=state_cmap(), vmin=-0.5, vmax=2.5, **geom.imshow_kwargs)

    # Barrier: hatched outline, never a fill - a solid barrier would occlude the
    # very cells we need to watch for a crossing event.
    if fire.barrier.any():
        ax.contour(
            geom.x_centres,
            geom.y_centres,
            fire.barrier.astype(float),
            levels=[0.5],
            colors=[COL_BARRIER],
            linewidths=1.1,
            alpha=0.85,
        )

    # C1.1: the contagion source is the FRONTIER of the burned region. Drawn on
    # every frame, so a dormant frame still shows where the fire can restart.
    if fire.ever[t].any():
        ax.contour(
            geom.x_centres,
            geom.y_centres,
            fire.ever[t].astype(float),
            levels=[0.5],
            colors=[COL_FRONTIER],
            linewidths=1.6,
        )

    # Cells that arrived THIS hour: distinguishes advance from smouldering.
    if t > 0:
        new = fire.ever[t] & ~fire.ever[t - 1]
        if new.any():
            ys, xs = np.nonzero(new)
            size = 9.0 if new.sum() < 250 else 4.0
            ax.scatter(
                geom.x_centres[xs],
                geom.y_centres[ys],
                s=size,
                marker="s",
                facecolors="none",
                edgecolors=COL_NEW,
                linewidths=0.6,
                zorder=6,
            )
        if spec.show_teleports:
            tele = teleport_cells(fire.ever, t, min_gap_cells=spec.min_gap_cells)
            if tele.any():
                ys, xs = np.nonzero(tele)
                ax.scatter(
                    geom.x_centres[xs],
                    geom.y_centres[ys],
                    s=70,
                    marker="o",
                    facecolors="none",
                    edgecolors=COL_TELEPORT,
                    linewidths=1.4,
                    zorder=7,
                )

    if spec.show_wind:
        _draw_wind(ax, fire, t, spec)

    ax.set_xlim(geom.extent[0], geom.extent[1])
    ax.set_ylim(geom.extent[2], geom.extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    add_north_arrow(ax)
    add_scale_bar(ax, geom)
    _draw_status(ax, fire, t)


def _draw_wind(ax: Any, fire: FireFrames, t: int, spec: MovieSpec) -> None:
    """Quiver in METRE coordinates, so northward v10 renders upward. See style.py."""
    xg, yg, rows, cols, step = quiver_grid(fire.geom, spec.quiver_target)
    u = fire.wind_u[t][rows, cols]
    v = fire.wind_v[t][rows, cols]
    # Scale is fixed by the WHOLE fire's peak wind, not this frame's, so a gust
    # looks like a gust; and the longest arrow stays inside its own sample cell
    # so the field stays readable instead of turning into a hatch.
    ref = max(float(fire.wind_speed.max()), 1e-6)
    target_len_m = 0.85 * step * fire.geom.cell_size_m
    ax.quiver(
        xg,
        yg,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=ref / target_len_m,
        color=COL_WIND,
        alpha=0.8,
        width=0.0038,
        headwidth=3.4,
        headlength=4.0,
        zorder=5,
    )
    spd = float(np.hypot(fire.wind_u[t], fire.wind_v[t]).mean())
    mx = float(np.hypot(fire.wind_u[t], fire.wind_v[t]).max())
    ax.text(
        0.012,
        0.985,
        f"wind → downwind  mean {spd:.1f} / max {mx:.1f} m s⁻¹",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COL_WIND,
        bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.8},
    )


def _draw_status(ax: Any, fire: FireFrames, t: int) -> None:
    n_burn = int(fire.n_burning[t])
    txt = (
        f"hour {t:>3d}/{fire.n_hours - 1}   {fire.label(t)}\n"
        f"burned {fire.area_km2[t]:7.1f} km²   +{fire.growth_km2[t]:.1f} km²/h   "
        f"state-1 cells {n_burn}"
    )
    ax.text(
        0.012,
        0.012,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        color=COL_TEXT,
        bbox={"fc": "white", "ec": "none", "alpha": 0.93, "pad": 2.5},
        zorder=11,
    )
    if n_burn == 0:
        run = fire.dormant_run_length(t)
        ax.text(
            0.5,
            0.965,
            f"DORMANT — no active fire line (hour {run} of this run). "
            "LEGAL under C1.1; the fire has not gone out.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#7c2d12",
            bbox={"fc": "#fde68a", "ec": "#b45309", "alpha": 0.95, "pad": 3.0},
            zorder=10,
        )


def _legend(fig: Any, spec: MovieSpec) -> None:
    handles = [
        Patch(fc=STATE_COLORS[0], ec="#9ca3af", label=STATE_LABELS[0]),
        Patch(fc=STATE_COLORS[1], ec="none", label=f"{STATE_LABELS[1]} (state 1)"),
        Patch(fc=STATE_COLORS[2], ec="none", label=STATE_LABELS[2]),
        Line2D([], [], color=COL_FRONTIER, lw=1.8, label="frontier (C1.1 contagion source)"),
        Line2D([], [], color=COL_BARRIER, lw=1.2, label="water/barrier mask"),
        Line2D(
            [], [], marker="s", ls="none", mfc="none", mec="#6b7280", ms=5, label="new this hour"
        ),
    ]
    if spec.show_teleports:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                mfc="none",
                mec=COL_TELEPORT,
                ms=8,
                label="teleport (no burned 8-neighbour at t−1)",
            )
        )
    fig_h = getattr(fig, "_layout_in", (8.0, 10.0))[1]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - 0.42 / fig_h),
        ncol=2 if len(handles) <= 4 else 4,
        frameon=False,
        fontsize=6.8,
        handlelength=1.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )


@lru_cache(maxsize=1)
def ffmpeg_usable() -> bool:
    """Whether ffmpeg is present AND actually runs.

    ``shutil.which`` is not enough. On this machine the Homebrew ffmpeg is on
    PATH but aborts at dynamic-link time (it wants ``libjxl.0.9`` against an
    installed 0.11), so ``which`` says yes and the first frame dies with a
    broken pipe several hundred frames into a render. Probe it once.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        return (
            subprocess.run(
                [exe, "-version"], capture_output=True, timeout=20, check=False
            ).returncode
            == 0
        )
    except Exception:  # pragma: no cover - environment dependent
        # A probe that fails is not the same as an absent ffmpeg, and the caller
        # cannot tell them apart from a bool. Say which one happened.
        logger.warning("ffmpeg at %s did not answer -version; treating it as unusable", exe)
        return False


def _writer(out: Path, fps: int) -> tuple[Any, Path]:
    """Pick a writer. Falls back to GIF rather than failing on a broken ffmpeg."""
    if out.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        if ffmpeg_usable():
            return FFMpegWriter(fps=fps, bitrate=2600, codec="libx264"), out
        out = out.with_suffix(".gif")
        logger.warning(
            "ffmpeg is missing or non-functional; writing %s instead. Rendering is "
            "unaffected, only the container changes",
            out,
        )
    if out.suffix.lower() != ".gif":
        out = out.with_suffix(".gif")
    return PillowWriter(fps=fps), out


def gap_summary(gaps: Sequence[int], cell_km: float, min_gap_cells: int) -> dict[str, Any]:
    """The front-gap statistics for a movie summary, and their DENOMINATOR.

    ``n_gap_steps`` is always present; the four statistics are present only when
    there was a step to measure. A fire with ``n_hours <= 1`` has no step, and
    the four used to read ``0``, ``0``, ``0.0`` and ``0.0`` - exactly what a fire
    with no teleports reads. "Nothing jumped" and "no hour was examined" were the
    same four numbers in the summary a reader scans to decide whether a store
    needs a human look, and neither the summary nor its consumer could tell them
    apart. Omitted rather than set to ``None`` so a consumer indexing the key
    gets a ``KeyError`` at the point of use.

    Split out of :func:`render_movie` so the empty case is reachable without
    encoding a video.
    """
    out: dict[str, Any] = {"min_gap_cells": int(min_gap_cells), "n_gap_steps": len(gaps)}
    if not gaps:
        return out
    # Both definitions are reported. `detached_steps` reproduces data QA's
    # "no burned 8-neighbour" count so the two are comparable; the render
    # threshold is stricter (see teleport_cells).
    out["detached_steps"] = int(sum(1 for g in gaps if g >= 1))
    out["teleport_steps"] = int(sum(1 for g in gaps if g >= min_gap_cells))
    out["max_front_gap_km"] = float(max(gaps) * cell_km)
    out["median_front_advance_cells"] = float(np.median([g for g in gaps if g >= 1] or [0]))
    return out


def render_movie(
    tensor: str | Path,
    out: str | Path,
    spec: MovieSpec | None = None,
    *,
    stills: int = 0,
) -> dict[str, Any]:
    """Render a fire movie and return a summary dict (also useful as a QA record)."""
    spec = spec or MovieSpec()
    fire = load_fire(tensor)
    order = frame_order(fire.growth_km2, pacing=spec.pacing, max_dwell=spec.max_dwell)

    fig, ax_map, ax_time = _new_figure(fire, spec.dpi)
    _legend(fig, spec)
    pacing_note = (
        "uniform pacing: 1 frame = 1 hour"
        if spec.pacing == "uniform"
        else f"EVENT pacing: high-growth hours dwell up to {spec.max_dwell}x — NOT real time"
    )
    stamp(fig, f"{fire.fire_id} · {fire.source} · C1 v2.3 · {pacing_note} · {spec.fps} fps")

    writer, out_path = _writer(Path(out), spec.fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.suptitle(
        f"{fire.fire_id} — fire state + 10 m wind",
        fontsize=11,
        y=1.0 - 0.14 / fig._layout_in[1],  # noqa: SLF001
        va="top",
    )

    with writer.saving(fig, str(out_path), dpi=spec.dpi):
        for t in order:
            render_frame(ax_map, fire, t, spec)
            _draw_timeline(ax_time, fire, t)
            writer.grab_frame()

    still_paths: list[str] = []
    if stills > 0:
        still_paths = _write_stills(fig, ax_map, ax_time, fire, spec, out_path, stills)
    plt.close(fig)

    gaps = [max_front_gap_cells(fire.ever, t) for t in range(1, fire.n_hours)]
    summary = fire.summary()
    summary.update(
        {
            "movie": str(out_path),
            "pacing": spec.pacing,
            "n_frames": len(order),
            "fps": spec.fps,
            "stills": still_paths,
        }
    )
    summary.update(gap_summary(gaps, fire.geom.cell_size_m / 1000.0, spec.min_gap_cells))
    return summary


def _write_stills(
    fig: Any, ax_map: Any, ax_time: Any, fire: FireFrames, spec: MovieSpec, out: Path, n: int
) -> list[str]:
    """Key stills: the biggest-growth hours plus the middle of each dormant run.

    Deliberately not evenly spaced - on a fire that is 70% flat, evenly spaced
    stills are 70% identical.
    """
    picks = list(np.argsort(fire.growth_km2)[::-1][: max(1, n - 2)])
    picks += [int((a + b) // 2) for a, b in _runs(fire.dormant)[:2]]
    picks = sorted({int(t) for t in picks})[:n]
    paths: list[str] = []
    for t in picks:
        render_frame(ax_map, fire, t, spec)
        _draw_timeline(ax_time, fire, t)
        p = out.parent / f"{out.stem}_t{t:03d}.png"
        fig.savefig(p, dpi=spec.dpi)
        paths.append(str(p))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.movie",
        description="Render a C1 fire tensor as a movie (state + wind quiver).",
    )
    ap.add_argument("--tensor", required=True, help="path to a C1 tensor.zarr")
    ap.add_argument("--out", required=True, help="output .mp4 (or .gif)")
    ap.add_argument("--pacing", choices=("uniform", "event"), default="uniform")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--max-dwell", type=int, default=4)
    ap.add_argument("--stills", type=int, default=0, help="also write N key stills as PNG")
    ap.add_argument("--no-wind", action="store_true")
    ap.add_argument("--no-teleports", action="store_true")
    ap.add_argument(
        "--min-gap-cells",
        type=int,
        default=DEFAULT_MIN_GAP_CELLS,
        help="mark a new cell as a teleport if it is this many cells (Chebyshev) from "
        "last hour's burned region; 1 reproduces data QA's no-8-neighbour definition",
    )
    ap.add_argument("--summary-json", default=None)
    add_logging_arguments(ap)
    args = ap.parse_args(argv)
    # ADR-103: the ONE place this program configures logging.
    configure_from_args(args)

    spec = MovieSpec(
        pacing=args.pacing,
        fps=args.fps,
        dpi=args.dpi,
        max_dwell=args.max_dwell,
        show_wind=not args.no_wind,
        show_teleports=not args.no_teleports,
        min_gap_cells=args.min_gap_cells,
    )
    summary = render_movie(args.tensor, args.out, spec, stills=args.stills)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    for k, v in summary.items():
        print(f"{k:>22} : {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
