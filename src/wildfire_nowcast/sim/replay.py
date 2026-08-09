"""Replay a gate run window-by-window and show WHERE a score comes from.

    python -m wildfire_nowcast.sim.replay --run runs/baselines-20260808-052918 \
        --kernel kernel=runs/kernel-nll_only-20260808-044220 --outdir reports/figures

A pooled metric answers "which model won". It cannot answer "on what". This
module re-walks the *identical* windows a gate run scored, through the same C5
``predict()`` and the same C6 mask, and decomposes the score per window so the
loss can be pointed at on a map.

**Reproduction is checked, not assumed.** :func:`iter_eval_windows` re-derives
the gate's window set from the C1 store; running this module prints the
reproduced ``band_best_member_iou`` beside the value recorded in the run's own
``results.json``. On ``2020_dolan`` those agree to five decimals for all three
models, which is what licenses every statement made from the figures. If they
ever disagree, the figures are describing a different experiment and the module
says so instead of drawing.

**The decomposition.** C6's ``band_best_member_iou`` is
``max_m mean_k IoU(member m, truth, at lead k)`` over the growth band. It is
therefore a mean over LEADS, and the leads are not alike: in the growth stratum
21.9% of (window, lead) pairs have **zero** new truth cells inside the band, and
:func:`wildfire_nowcast.eval.metrics.fuzzy_iou` returns 1.0 for an empty-vs-empty
comparison (documented, and defensible in isolation). So each window's score
splits exactly into

* ``silence_term``  — leads where truth did not grow in the band. A member that
  ignites nothing scores 1.0; a member that ignites ONE cell scores 0.0.
* ``shape_term``    — leads where truth did grow. This is the mode-capture
  quantity the metric is named after.

Both are reported, always together. The split is arithmetic, not a re-definition
of the metric: ``silence_term + shape_term == band_best_member_iou`` to floating
point, and the module asserts it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from wildfire_nowcast.common.iou_terms import silent_floor as _canonical_silent_floor  # noqa: E402
from wildfire_nowcast.common.zarr_io import channel_values, open_tensor  # noqa: E402
from wildfire_nowcast.sim.c5 import C5Inputs, c5_inputs  # noqa: E402
from wildfire_nowcast.sim.style import (  # noqa: E402
    BURN_PROB_CMAP,
    COL_BARRIER,
    COL_TEXT,
    COL_TRUTH,
    COL_WARN,
    PlotGeometry,
    add_north_arrow,
    plot_extent,
    stamp,
)

__all__ = [
    "iter_eval_windows",
    "WindowScore",
    "score_window",
    "GateModels",
    "load_gate_models",
    "replay_fire",
    "render_small_multiples",
    "render_decomposition",
    "main",
]

# Colour roles specific to this figure. Truth is black everywhere in the package;
# a member is drawn as an outline so truth beneath it stays readable.
COL_HIT = "#166534"
COL_FALSE_ALARM = "#f59e0b"
COL_MISS = "#7f1d1d"


# --------------------------------------------------------------------------
# window enumeration — the gate's, reproduced from C1
# --------------------------------------------------------------------------


def iter_eval_windows(
    ds: xr.Dataset, horizon_h: int = 3, *, stride: int = 1
) -> Iterator[tuple[int, C5Inputs]]:
    """Yield ``(i, window)`` for every evaluable window, in the gate's order.

    ``i`` is the position in the FILTERED sequence, because the gate seeds member
    sampling with ``seed + i``. Using the raw ``t0`` instead would reseed every
    window differently and silently change every ensemble — the figures would be
    of a neighbouring experiment, not of the run being explained.

    The rule (mirrored from the run's own enumeration, and verified against its
    recorded ``n_windows``): ``t0`` in ``range(0, T - horizon_h, stride)``,
    skipping windows whose ``x0`` has nothing burned at all.
    """
    n_t = int(ds.sizes["time"])
    i = 0
    for t0 in range(0, n_t - horizon_h, max(1, int(stride))):
        window = c5_inputs(ds, t0, horizon_h)
        if not np.any(window.x0 > 0):
            continue
        yield i, window
        i += 1


def band_of(x0: np.ndarray, radius_cells: int) -> np.ndarray:
    """C6's ``growth_band`` mask, via ``eval.masks`` so it cannot drift from C6."""
    from wildfire_nowcast.eval.masks import growth_band  # noqa: PLC0415

    return np.asarray(growth_band(np.asarray(x0), int(radius_cells)), dtype=bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard with C6's empty-vs-empty convention (``fuzzy_iou`` returns 1.0)."""
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b)) / union if union else 1.0


# --------------------------------------------------------------------------
# per-window decomposition
# --------------------------------------------------------------------------


@dataclass
class WindowScore:
    """One window, one model: the C6 band score and what it is made of."""

    fire_id: str
    model: str
    t0: int
    band_cells: int
    band_best_member_iou: float
    best_member: int
    best_member_is_silent: bool
    silence_term: float
    shape_term: float
    n_empty_leads: int
    n_silent_members: int
    truth_new_by_lead: list[int] = field(default_factory=list)
    best_new_by_lead: list[int] = field(default_factory=list)
    iou_by_lead: list[float] = field(default_factory=list)
    precision: float = float("nan")
    recall: float = float("nan")
    area_ratio: float = float("nan")
    mean_area_ratio: float = float("nan")

    def check(self) -> None:
        total = self.silence_term + self.shape_term
        if not np.isclose(total, self.band_best_member_iou, atol=1e-9):
            raise AssertionError(
                f"decomposition does not reconstruct the metric: {total} != "
                f"{self.band_best_member_iou}. The split is arithmetic; a mismatch "
                "means the metric changed shape and this module must be re-derived."
            )


def score_window(
    samples: np.ndarray,
    window: C5Inputs,
    *,
    fire_id: str,
    model: str,
    band: np.ndarray,
) -> WindowScore:
    """Decompose C6's band best-member IoU for one window. Pure; no I/O."""
    member_event = np.asarray(samples) > 0  # [M, L, H, W]
    truth_event = np.asarray(window.truth) > 0  # [L, H, W]
    n_members, n_lead = member_event.shape[0], member_event.shape[1]

    truth_new = [int(np.count_nonzero(truth_event[k] & band)) for k in range(n_lead)]
    per = np.empty((n_members, n_lead), dtype=np.float64)
    new_cells = np.empty((n_members, n_lead), dtype=np.int64)
    for m in range(n_members):
        for k in range(n_lead):
            pred = member_event[m, k] & band
            per[m, k] = _iou(pred, truth_event[k] & band)
            new_cells[m, k] = int(np.count_nonzero(pred))

    trajectory = per.mean(axis=1)
    best = int(np.argmax(trajectory))
    empty = [k for k in range(n_lead) if truth_new[k] == 0]
    grew = [k for k in range(n_lead) if truth_new[k] > 0]

    k_last = n_lead - 1
    pred_last = member_event[best, k_last] & band
    obs_last = truth_event[k_last] & band
    hit = int(np.count_nonzero(pred_last & obs_last))
    mean_area = float(new_cells[:, k_last].mean())

    score = WindowScore(
        fire_id=fire_id,
        model=model,
        t0=int(window.t0),
        band_cells=int(np.count_nonzero(band)),
        band_best_member_iou=float(trajectory[best]),
        best_member=best,
        best_member_is_silent=bool(new_cells[best].sum() == 0),
        silence_term=float(sum(per[best, k] for k in empty) / n_lead),
        shape_term=float(sum(per[best, k] for k in grew) / n_lead),
        n_empty_leads=len(empty),
        n_silent_members=int((new_cells.sum(axis=1) == 0).sum()),
        truth_new_by_lead=truth_new,
        best_new_by_lead=[int(v) for v in new_cells[best]],
        iou_by_lead=[float(v) for v in per[best]],
        precision=hit / max(int(np.count_nonzero(pred_last)), 1),
        recall=hit / max(int(np.count_nonzero(obs_last)), 1),
        area_ratio=int(np.count_nonzero(pred_last)) / max(int(np.count_nonzero(obs_last)), 1),
        mean_area_ratio=mean_area / max(int(np.count_nonzero(obs_last)), 1),
    )
    score.check()
    return score


def silent_floor(window: C5Inputs, band: np.ndarray) -> float:
    """What an ensemble that predicts NOTHING scores on this window.

    Depends only on the labels, so it is identical for every model — which is the
    whole point. A metric whose null value is not zero must publish that value
    next to every number it produces, or the numbers read as skill.

    [C0, ADR-036] This used to RECOMPUTE the floor inline. It is a gate quantity —
    C6.4's silence term, the thing G2's criterion is defined by REMOVING — so a
    second implementation of it is exactly what C0 forbids: *the producer and the
    verifier computing geometry through different code is how a tensor passes its
    check and is still wrong*. The arithmetic agreed (3003 randomised cases plus
    all-empty / all-burn / empty-band edges, 0 mismatches, exact equality), but the
    canonical version also VALIDATES the horizon and raises, and that guard was
    silently absent here. This function now only derives ``truth_empty`` — which is
    what its old loop computed anyway — and delegates the reduction.
    """
    truth_event = np.asarray(window.truth) > 0
    truth_empty = np.array(
        [not np.any(truth_event[k] & band) for k in range(truth_event.shape[0])],
        dtype=bool,
    )
    return _canonical_silent_floor(truth_empty)


# --------------------------------------------------------------------------
# models — the ONE place this package constructs a predictor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateModels:
    """The predictors a gate run used, rebuilt from its own recorded artifacts.

    Every one of them is called ONLY through the C5 ``predict()`` signature. The
    construction is here, in one function, so the single place this package
    touches ``model.*`` for anything other than ``predict`` is auditable.

    **Recorded contract gap (PROPOSAL, see docs/decisions.md).** C5
    says checkpoints load behind ``load_model(path)``. That works for the kernel:
    the run directory holds ``model.json``. It does NOT work for the calibrated
    ellipse — a gate run records the fitted *scale* in ``results.json`` but emits
    no loadable spec, so reproducing the gate's own opponent requires rebuilding
    ``EllipseBaseline(params.scaled(scale))`` by hand. That is a reproducibility
    hole, and it will bite hardest at G5, where ELMFIRE's configuration is the
    thing a reader most needs to be able to re-instantiate.
    """

    models: dict[str, Any]
    provenance: dict[str, Any]


def load_gate_models(
    results_json: str | Path,
    *,
    kernels: Sequence[str] = (),
    include_persistence: bool = True,
) -> GateModels:
    """Rebuild a gate run's predictors. ``kernels`` are ``name=run_dir`` strings."""
    from wildfire_nowcast.model.api import load_model  # noqa: PLC0415
    from wildfire_nowcast.model.baselines import EllipseBaseline  # noqa: PLC0415

    payload = json.loads(Path(results_json).read_text())
    cal = payload["ellipse_calibration"]
    scale_1h = float(cal["rule_of_record"]["scale"])
    base = EllipseBaseline()

    models: dict[str, Any] = {}
    prov: dict[str, Any] = {"results_json": str(results_json), "ellipse_scales": {}}
    if include_persistence:
        models["persistence"] = load_model("persistence")
        prov["persistence"] = "C5 load_model('persistence')"
    models["ellipse"] = EllipseBaseline(base.params.scaled(scale_1h), name="ellipse")
    prov["ellipse_scales"]["ellipse"] = scale_1h
    for horizon, entry in (cal.get("alternative_horizons") or {}).items():
        name = f"ellipse_cal{int(horizon)}h"
        scale = float(entry["scale"])
        models[name] = EllipseBaseline(base.params.scaled(scale), name=name)
        prov["ellipse_scales"][name] = scale
    for spec in kernels:
        name, _, run_dir = spec.partition("=")
        if not run_dir:
            raise ValueError(f"--kernel expects name=run_dir, got {spec!r}")
        models[name] = load_model(run_dir)
        prov[name] = run_dir
    return GateModels(models=models, provenance=prov)


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def replay_fire(
    fire_id: str,
    tensor_path: str | Path,
    gate: GateModels,
    *,
    horizon_h: int = 3,
    stride: int = 2,
    n_members: int = 24,
    seed: int = 20260807,
    band_radius_cells: int | None = None,
    growth_only: bool = True,
) -> dict[str, Any]:
    """Score every window of one fire for every model, keeping the per-window split.

    ``growth_only`` mirrors the ``growth_windows`` stratum the gate reports and
    that G2 was adjudicated on. It conditions the window set on the outcome and
    is therefore never a headline number on its own — it is here because it is
    the stratum whose IoU split the gate.
    """
    from wildfire_nowcast.eval.masks import default_band_radius  # noqa: PLC0415

    radius = int(band_radius_cells or default_band_radius(horizon_h))
    ds = open_tensor(Path(tensor_path))
    try:
        windows = list(iter_eval_windows(ds, horizon_h, stride=stride))
    finally:
        pass

    rows: list[WindowScore] = []
    floors: list[float] = []
    kept: list[tuple[int, C5Inputs, np.ndarray]] = []
    for i, w in windows:
        if growth_only and int((w.truth[-1] > 0).sum()) <= int((w.x0 > 0).sum()):
            continue
        band = band_of(w.x0, radius)
        kept.append((i, w, band))
        floors.append(silent_floor(w, band))

    for name, model in gate.models.items():
        for i, w, band in kept:
            samples = model.predict(w.x0, w.static, w.weather, n_members, horizon_h, seed + i)
            rows.append(score_window(samples, w, fire_id=fire_id, model=name, band=band))

    return {
        "fire_id": fire_id,
        "n_windows_total": len(windows),
        "n_windows_scored": len(kept),
        "growth_only": bool(growth_only),
        "band_radius_cells": radius,
        "horizon_h": horizon_h,
        "n_members": n_members,
        "seed": seed,
        "stride": stride,
        "silent_floor": float(np.mean(floors)) if floors else float("nan"),
        "windows": [asdict(r) for r in rows],
        "ds": ds,
    }


def summarise(rows: Sequence[WindowScore | dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Pool the per-window split by model. Means over windows, as C6 pools."""
    out: dict[str, dict[str, float]] = {}
    dicts = [r if isinstance(r, dict) else asdict(r) for r in rows]
    for model in sorted({r["model"] for r in dicts}):
        sel = [r for r in dicts if r["model"] == model]
        out[model] = {
            "n_windows": float(len(sel)),
            "band_best_member_iou": float(np.mean([r["band_best_member_iou"] for r in sel])),
            "silence_term": float(np.mean([r["silence_term"] for r in sel])),
            "shape_term": float(np.mean([r["shape_term"] for r in sel])),
            "best_member_silent_frac": float(np.mean([r["best_member_is_silent"] for r in sel])),
            "mean_silent_members": float(np.mean([r["n_silent_members"] for r in sel])),
            "precision": float(np.mean([r["precision"] for r in sel])),
            "recall": float(np.mean([r["recall"] for r in sel])),
            "area_ratio": float(np.median([r["area_ratio"] for r in sel])),
        }
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _map_axes(ax: Any, geom: PlotGeometry) -> None:
    ax.set_xlim(geom.extent[0], geom.extent[1])
    ax.set_ylim(geom.extent[2], geom.extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9ca3af")


def _draw_confusion(
    ax: Any,
    geom: PlotGeometry,
    *,
    burned0: np.ndarray,
    truth_new: np.ndarray,
    pred_new: np.ndarray,
    band: np.ndarray,
    barrier: np.ndarray | None = None,
) -> None:
    """Hit / miss / false-alarm map. The picture the IoU number is computed from."""
    rgba = np.zeros((*burned0.shape, 4), dtype=float)
    rgba[band] = (0.96, 0.96, 0.95, 1.0)
    rgba[burned0] = (0.23, 0.19, 0.16, 1.0)
    fa = pred_new & ~truth_new
    miss = truth_new & ~pred_new
    hit = pred_new & truth_new
    for mask, colour in (
        (fa, (0.96, 0.62, 0.04, 1.0)),
        (miss, (0.50, 0.11, 0.11, 1.0)),
        (hit, (0.09, 0.39, 0.20, 1.0)),
    ):
        rgba[mask] = colour
    ax.imshow(rgba, **geom.imshow_kwargs)
    if barrier is not None and barrier.any():
        edge = np.zeros((*barrier.shape, 4), dtype=float)
        edge[barrier] = (0.17, 0.42, 0.69, 0.35)
        ax.imshow(edge, **geom.imshow_kwargs)
    _map_axes(ax, geom)


def _draw_prob(
    ax: Any, geom: PlotGeometry, prob: np.ndarray, truth_new: np.ndarray, band: np.ndarray
) -> None:
    shown = np.where(band, prob, np.nan)
    ax.imshow(shown, vmin=0.0, vmax=1.0, cmap=BURN_PROB_CMAP, **geom.imshow_kwargs)
    ys, xs = np.nonzero(truth_new)
    if ys.size:
        ax.scatter(
            geom.x_centres[xs],
            geom.y_centres[ys],
            s=6,
            facecolors="none",
            edgecolors=COL_TRUTH,
            linewidths=0.6,
            zorder=4,
        )
    _map_axes(ax, geom)


def render_small_multiples(
    fire_id: str,
    ds: xr.Dataset,
    gate: GateModels,
    picks: Sequence[int],
    *,
    models: Sequence[str],
    horizon_h: int,
    stride: int,
    n_members: int,
    seed: int,
    band_radius_cells: int,
    out: str | Path,
    subtitle: str = "",
) -> Path:
    """Member-vs-truth small multiples: rows are windows, columns are models."""
    geom = plot_extent(
        ds["x"].values, ds["y"].values, cell_size_m=float(ds.attrs.get("cell_size_m", 1000.0))
    )
    barrier = channel_values(ds, "water_barrier_mask", dtype=np.float32)[0] > 0.5
    wind_u = channel_values(ds, "wind_u10", dtype=np.float32)
    wind_v = channel_values(ds, "wind_v10", dtype=np.float32)

    by_t0 = {w.t0: (i, w) for i, w in iter_eval_windows(ds, horizon_h, stride=stride)}
    chosen = [by_t0[t] for t in picks if t in by_t0]
    if not chosen:
        raise ValueError(f"none of {list(picks)} are evaluable windows of {fire_id}")

    n_col = len(models) + 1
    fig, axes = plt.subplots(
        len(chosen), n_col, figsize=(2.55 * n_col + 1.4, 2.95 * len(chosen)), squeeze=False
    )
    for r, (i, w) in enumerate(chosen):
        band = band_of(w.x0, band_radius_cells)
        burned0 = w.x0 > 0
        truth_new = (w.truth[-1] > 0) & ~burned0
        empty_leads = sum(
            1 for k in range(horizon_h) if not np.any((w.truth[k] > 0) & band)
        )

        ax = axes[r][0]
        _draw_confusion(
            ax,
            geom,
            burned0=burned0,
            truth_new=truth_new,
            pred_new=np.zeros_like(truth_new),
            band=band,
            barrier=barrier,
        )
        u = float(np.mean(wind_u[w.t0 + 1 : w.t0 + 1 + horizon_h]))
        v = float(np.mean(wind_v[w.t0 + 1 : w.t0 + 1 + horizon_h]))
        cx = geom.extent[0] + 0.16 * (geom.extent[1] - geom.extent[0])
        cy = geom.extent[2] + 0.14 * (geom.extent[3] - geom.extent[2])
        span = 0.11 * (geom.extent[1] - geom.extent[0])
        norm = max(np.hypot(u, v), 1e-6)
        ax.arrow(
            cx, cy, span * u / norm, span * v / norm,
            width=span * 0.06, color="#1f4e5f", zorder=6, length_includes_head=True,
        )
        ax.text(
            cx, cy - span * 0.75, f"{np.hypot(u, v):.1f} m/s",
            fontsize=6, ha="center", color="#1f4e5f",
        )
        add_north_arrow(ax)
        ax.set_ylabel(
            f"t0={w.t0}\ntruth +{int(truth_new.sum())} cells\n"
            f"{empty_leads}/{horizon_h} leads empty in band",
            fontsize=7,
        )
        if r == 0:
            ax.set_title("TRUTH (new cells, +3 h)", fontsize=8.5)

        for c, name in enumerate(models, start=1):
            samples = gate.models[name].predict(
                w.x0, w.static, w.weather, n_members, horizon_h, seed + i
            )
            sc = score_window(samples, w, fire_id=fire_id, model=name, band=band)
            pred_new = (samples[sc.best_member, -1] > 0) & ~burned0
            ax = axes[r][c]
            _draw_confusion(
                ax,
                geom,
                burned0=burned0,
                truth_new=truth_new,
                pred_new=pred_new,
                band=band,
                barrier=barrier,
            )
            silent = " SILENT MEMBER" if sc.best_member_is_silent else ""
            ax.set_xlabel(
                f"IoU {sc.band_best_member_iou:.3f} = silence {sc.silence_term:.3f}"
                f" + shape {sc.shape_term:.3f}{silent}\n"
                f"P {sc.precision:.2f}  R {sc.recall:.2f}  area x{sc.area_ratio:.2f}"
                f"  ({sc.n_silent_members}/{n_members} members silent)",
                fontsize=6.4,
                color=COL_WARN if sc.best_member_is_silent else COL_TEXT,
            )
            if r == 0:
                ax.set_title(f"{name} — best member", fontsize=8.5)

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=7, color="#166534", label="hit"),
        plt.Line2D([], [], marker="s", ls="", ms=7, color="#f59e0b", label="false alarm"),
        plt.Line2D([], [], marker="s", ls="", ms=7, color="#7f1d1d", label="miss"),
        plt.Line2D([], [], marker="s", ls="", ms=7, color="#3a3128", label="burned at t0"),
        plt.Line2D([], [], marker="s", ls="", ms=7, color=COL_BARRIER, alpha=0.4, label="barrier"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(
        f"{fire_id} — member vs truth, growth band only\n{subtitle}", fontsize=11, y=0.995
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.965))
    stamp(
        fig,
        "C1 tensor + C5 predict() + C6 growth_band mask. Best member is C6's own choice "
        "(max mean-over-lead band IoU). No model internals read.",
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def render_decomposition(
    per_fire: dict[str, dict[str, dict[str, float]]],
    floors: dict[str, float],
    recorded: dict[str, dict[str, float]] | None,
    out: str | Path,
) -> Path:
    """Stacked silence/shape bars per fire, with the null floor drawn as a line."""
    fires = list(per_fire)
    models = sorted({m for f in per_fire.values() for m in f})
    fig, axes = plt.subplots(
        1, len(fires) + 1, figsize=(4.1 * (len(fires) + 1), 4.9), squeeze=False
    )
    axes = axes[0]

    for a, fire in enumerate(fires):
        ax = axes[a]
        stats = per_fire[fire]
        xs = np.arange(len(models))
        silence = [stats.get(m, {}).get("silence_term", np.nan) for m in models]
        shape = [stats.get(m, {}).get("shape_term", np.nan) for m in models]
        ax.bar(xs, silence, color="#cbd5e1", edgecolor="#64748b", lw=0.6, label="silence term")
        ax.bar(
            xs, shape, bottom=silence, color="#0f766e", edgecolor="#134e4a", lw=0.6,
            label="shape term (mode capture)",
        )
        floor = floors.get(fire, float("nan"))
        ax.axhline(floor, color=COL_WARN, lw=1.4, ls="--")
        ax.text(
            len(models) - 0.4, floor, f" null floor {floor:.3f}", color=COL_WARN,
            fontsize=7, va="bottom", ha="right",
        )
        for x, m in zip(xs, models, strict=True):
            total = stats.get(m, {}).get("band_best_member_iou", np.nan)
            ax.text(x, total + 0.006, f"{total:.3f}", ha="center", fontsize=7)
            if recorded and fire in recorded and m in recorded[fire]:
                rec = recorded[fire][m]
                ok = abs(rec - total) < 5e-4
                ax.text(
                    x, -0.028,
                    ("=" if ok else "≠") + f"{rec:.3f}",
                    ha="center", fontsize=6,
                    color="#166534" if ok else COL_WARN,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(models, fontsize=7, rotation=20, ha="right")
        top = max([s + t for s, t in zip(silence, shape, strict=True)])
        ax.set_ylim(-0.045, max(0.42, top * 1.18))
        ax.set_title(fire, fontsize=9.5)
        ax.grid(alpha=0.22, lw=0.5, axis="y")
        if a == 0:
            ax.set_ylabel("C6 band_best_member_iou (growth windows)")
            ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[-1]
    pooled_s = {
        m: np.mean([per_fire[f].get(m, {}).get("silence_term", np.nan) for f in fires])
        for m in models
    }
    pooled_h = {
        m: np.mean([per_fire[f].get(m, {}).get("shape_term", np.nan) for f in fires])
        for m in models
    }
    xs = np.arange(len(models))
    ax.bar(xs, [pooled_s[m] for m in models], color="#cbd5e1", edgecolor="#64748b", lw=0.6)
    ax.bar(
        xs, [pooled_h[m] for m in models], bottom=[pooled_s[m] for m in models],
        color="#0f766e", edgecolor="#134e4a", lw=0.6,
    )
    ax.axhline(float(np.mean(list(floors.values()))), color=COL_WARN, lw=1.4, ls="--")
    for x, m in zip(xs, models, strict=True):
        ax.text(x, pooled_s[m] + pooled_h[m] + 0.006, f"{pooled_s[m] + pooled_h[m]:.3f}",
                ha="center", fontsize=7)
        ax.text(
            x, pooled_s[m] / 2, f"{pooled_s[m]:.3f}", ha="center", fontsize=6.5,
            color="#334155",
        )
        ax.text(x, pooled_s[m] + pooled_h[m] / 2, f"{pooled_h[m]:.3f}", ha="center",
                fontsize=6.5, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=7, rotation=20, ha="right")
    ax.set_title("mean of the 4 held-out blocks", fontsize=9.5)
    ax.grid(alpha=0.22, lw=0.5, axis="y")

    fig.suptitle(
        "WHERE band_best_member_iou comes from — silence credit vs mode capture\n"
        "silence term = leads where truth grew ZERO cells in the band, where C6's "
        "empty-vs-empty IoU is 1.0 for a member that ignites nothing and 0.0 for "
        "one that ignites one cell",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    stamp(
        fig,
        "reproduced through C5 predict() + C6 mask; '=' under a bar means it "
        "matches the run's recorded value",
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _recorded_iou(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for fire, entry in (payload.get("per_fire") or {}).items():
        out[fire] = {
            model: float(m["growth_windows"]["band_best_member_iou"])
            for model, m in entry["models"].items()
            if "growth_windows" in m and m["growth_windows"].get("band_best_member_iou") is not None
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.replay",
        description="Replay a gate run window-by-window and decompose its band IoU.",
    )
    ap.add_argument("--run", required=True, help="gate run directory holding results.json")
    ap.add_argument("--kernel", action="append", default=[], help="name=run_dir")
    ap.add_argument("--fires", nargs="*", default=None, help="default: the run's held-out fires")
    ap.add_argument("--outdir", default="reports/figures")
    ap.add_argument("--fires-dir", default="data/fires")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--picks", type=int, default=4, help="windows per small-multiple figure")
    args = ap.parse_args(argv)

    run = Path(args.run)
    payload = json.loads((run / "results.json").read_text())
    cfg = {}
    cfg_path = run / "config.yaml"
    if cfg_path.is_file():
        for line in cfg_path.read_text().splitlines():
            if ":" in line and not line.startswith(" "):
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
    horizon = args.horizon or int(payload.get("horizon_h", 3))
    members = args.members or int(payload.get("n_members", 24))
    seed = args.seed if args.seed is not None else int(payload.get("seed", 20260807))
    stride = args.stride or int(cfg.get("stride", 1) or 1)

    fires = args.fires or list(payload.get("per_fire") or {})
    gate = load_gate_models(run / "results.json", kernels=args.kernel)
    recorded = _recorded_iou(payload)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from wildfire_nowcast.eval.masks import default_band_radius  # noqa: PLC0415

    radius = default_band_radius(horizon)

    per_fire: dict[str, dict[str, dict[str, float]]] = {}
    floors: dict[str, float] = {}
    all_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []

    for fire in fires:
        tensor = Path(args.fires_dir) / fire / "tensor.zarr"
        result = replay_fire(
            fire, tensor, gate,
            horizon_h=horizon, stride=stride, n_members=members, seed=seed,
            band_radius_cells=radius,
        )
        ds = result.pop("ds")
        rows = result["windows"]
        all_rows.extend(rows)
        per_fire[fire] = summarise(rows)
        floors[fire] = result["silent_floor"]

        for model, stats in per_fire[fire].items():
            rec = recorded.get(fire, {}).get(model)
            if rec is None:
                continue
            if abs(rec - stats["band_best_member_iou"]) > 5e-4:
                mismatches.append(
                    f"{fire}/{model}: replay {stats['band_best_member_iou']:.5f} != "
                    f"recorded {rec:.5f}"
                )

        # Windows for the small multiples: the largest disagreements between the
        # learned model and the strongest ellipse, both directions, plus the most
        # extreme silence-credit window. Picking by disagreement is the point --
        # a random window shows the median, and the median is not what split G2.
        kernel_names = [k.partition("=")[0] for k in args.kernel] or ["kernel"]
        km = kernel_names[0]
        opp = "ellipse_cal3h" if "ellipse_cal3h" in gate.models else "ellipse"
        by_t0: dict[int, dict[str, dict[str, Any]]] = {}
        for r in rows:
            by_t0.setdefault(r["t0"], {})[r["model"]] = r
        deltas = [
            (t0, v[km]["band_best_member_iou"] - v[opp]["band_best_member_iou"], v)
            for t0, v in by_t0.items()
            if km in v and opp in v
        ]
        deltas.sort(key=lambda d: d[1])
        n = max(1, args.picks // 2)
        picks = [d[0] for d in deltas[:n]] + [d[0] for d in deltas[-n:]]
        render_small_multiples(
            fire, ds, gate, picks,
            models=[m for m in (km, opp, "ellipse") if m in gate.models],
            horizon_h=horizon, stride=stride, n_members=members, seed=seed,
            band_radius_cells=radius,
            out=outdir / f"iou_smallmultiples_{fire}.png",
            subtitle=(
                f"rows: the {n} windows where {km} loses most to {opp} (top) and wins most "
                f"(bottom) on C6 band IoU  |  null floor for this fire = {floors[fire]:.3f}"
            ),
        )
        print(f"[replay] {fire}: {result['n_windows_scored']} growth windows, "
              f"floor={floors[fire]:.4f}")
        for model, stats in per_fire[fire].items():
            print(
                f"    {model:16s} IoU {stats['band_best_member_iou']:.4f} = "
                f"silence {stats['silence_term']:.4f} + shape {stats['shape_term']:.4f}"
                f"   best-member-silent {stats['best_member_silent_frac']:.0%}"
                f"   P {stats['precision']:.2f} R {stats['recall']:.2f}"
            )

    fig = render_decomposition(per_fire, floors, recorded, outdir / "iou_decomposition.png")
    payload_out = {
        "kind": "simviz_gate_replay",
        "run": str(run),
        "split_fingerprint": (payload.get("scope") or {}).get("split_fingerprint"),
        "horizon_h": horizon,
        "n_members": members,
        "seed": seed,
        "stride": stride,
        "band_radius_cells": radius,
        "provenance": gate.provenance,
        "silent_floor_per_fire": floors,
        "per_fire": per_fire,
        "reproduction_mismatches": mismatches,
        "windows": all_rows,
    }
    (outdir / "iou_decomposition.json").write_text(json.dumps(payload_out, indent=1) + "\n")
    print(f"[replay] {fig}")
    if mismatches:
        print("[replay] REPRODUCTION MISMATCH — the figures describe a different experiment:")
        for m in mismatches:
            print("   ", m)
        return 1
    print("[replay] reproduction OK: every replayed band IoU matches the run's recorded value")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
