"""Ensemble viewer: spaghetti fronts, burn probability, arrival-time quantiles.

    python -m wildfire_nowcast.sim.ensemble --tensor outputs/synthetic_fire/tensor.zarr \
        --out reports/figures/synthetic_ensemble.png --t0 8 --horizon 3 --members 24

Input is whatever a C5 ``predict()`` returns — ``uint8[n_members, horizon_h,
H, W]`` — so a baseline, the learned model and (later) ELMFIRE all render
identically and are therefore comparable by eye as well as through C6.

Two things here are easy to get wrong and are handled explicitly.

**Arrival-time quantiles are CENSORED.** A pixel that burns in 3 of 24 members
has no median arrival time. Taking a ``nanquantile`` over only the members that
did arrive reports a confident, early median for a pixel that almost never
burns — the map then looks like a fast, certain fire everywhere the ensemble is
actually unsure. :func:`arrival_quantiles` treats "never arrived in the window"
as ``+inf``, so the q-quantile is undefined (NaN) whenever fewer than a fraction
``q`` of members arrive at all, and the figure draws those cells as censored.

**Ensemble collapse is checked, not assumed.** Models with independent
per-pixel noise and no shared per-step latent are known-broken by collapse
(``README.md``): ten thousand independent Bernoullis average out and total
burned area concentrates on its mean. A spaghetti plot
of a collapsed ensemble is a single thick line and looks tidy, which is the
danger. :func:`ensemble_diagnostics` measures spread directly and the figure
carries a loud banner when it is degenerate. These are RENDERING diagnostics for
triage; the headline calibration numbers are C6's and are not recomputed here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: E402
from wildfire_nowcast.sim.c5 import C5_CONVENTION, C5Inputs, c5_inputs  # noqa: E402
from wildfire_nowcast.sim.reader import load_fire  # noqa: E402
from wildfire_nowcast.sim.style import (  # noqa: E402
    ARRIVAL_CMAP,
    BURN_PROB_CMAP,
    COL_BARRIER,
    COL_MEMBER,
    COL_TEXT,
    COL_TRUTH,
    COL_WARN,
    PlotGeometry,
    add_north_arrow,
    stamp,
)

__all__ = [
    "burn_probability",
    "arrival_quantiles",
    "ensemble_diagnostics",
    "EnsembleView",
    "render_ensemble",
    "main",
]

CENSORED_COLOR = "#dcd8d0"


# -- pure statistics -------------------------------------------------------


def burn_probability(samples: np.ndarray, lead: int | None = None) -> np.ndarray:
    """Fraction of members in which each cell has burned by ``lead`` (default last)."""
    s = np.asarray(samples)
    lead = s.shape[1] - 1 if lead is None else int(lead)
    return (s[:, lead] > 0).mean(axis=0).astype(np.float64)


def arrival_quantiles(
    samples: np.ndarray, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell arrival-time quantiles in LEAD HOURS, honestly censored.

    Returns ``(q_maps, burn_prob)`` where ``q_maps[i]`` is NaN wherever fewer
    than ``quantiles[i]`` of members ever burn the cell — i.e. where the
    quantile genuinely does not exist inside the forecast window.

    Arrival is 1-based in lead hours: ``1`` means "burning by the first
    predicted hour".
    """
    s = np.asarray(samples) > 0
    n_members, horizon = s.shape[0], s.shape[1]
    arrived = s.any(axis=1)  # (M, H, W)
    first = np.argmax(s, axis=1) + 1  # (M, H, W), 1-based
    a = np.where(arrived, first.astype(np.float64), np.inf)
    a.sort(axis=0)  # +inf sorts last, which is exactly right

    out = np.empty((len(quantiles), *s.shape[2:]), dtype=np.float64)
    for i, q in enumerate(quantiles):
        # Order statistic: the smallest k with k/M >= q.
        k = int(np.ceil(q * n_members))
        k = min(max(k, 1), n_members) - 1
        vals = a[k]
        out[i] = np.where(np.isfinite(vals), vals, np.nan)
    prob = arrived.mean(axis=0).astype(np.float64)
    assert horizon >= 1
    return out, prob


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = float((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


#: Below this, the ensemble is no more dispersed than independent pixel noise
#: would make it — i.e. the shared latent is doing nothing. See
#: :func:`independence_dispersion_index`.
COLLAPSE_INDEX_THRESHOLD = 1.5


def independence_dispersion_index(samples: np.ndarray, lead: int = -1) -> float:
    """Observed burned-area spread ÷ the spread independent pixels would give.

    This is the collapse test that does not need a magic number. If every pixel
    were an independent Bernoulli with the ensemble's own marginal ``p_i``, the
    burned-area variance would be exactly ``Σ p_i(1 − p_i)``. So::

        index = std(member areas) / sqrt(Σ p_i (1 − p_i))

    ``index ≈ 1`` means the ensemble carries no correlated innovation at all —
    precisely the independent-per-pixel model this project treats as known-broken
    and keeps only as an ablation.
    ``index >> 1`` means a shared per-step latent is actually moving the whole
    front together, which is what ``z_t`` exists to do.

    Why not a coefficient of variation: CV of an independent ensemble falls like
    ``1/sqrt(N_front)``, so any CV threshold is really a threshold on fire size
    and will quietly change meaning between a 40-cell fire and a 400-cell one.
    Measured on the viz stub, this index is 1.09-1.31 at ``latent_sigma=0``
    against 5.1-6.0 at ``0.9``, and is stable across 12-60 members where CV is
    not.
    """
    final = np.asarray(samples)[:, lead] > 0
    areas = final.sum(axis=(1, 2)).astype(np.float64)
    p = final.mean(axis=0)
    expected = float(np.sqrt(np.sum(p * (1.0 - p))))
    if expected <= 0:
        # Every member identical: zero spread predicted AND observed.
        return 0.0 if areas.std() == 0 else float("inf")
    return float(areas.std() / expected)


def ensemble_diagnostics(
    samples: np.ndarray, truth: np.ndarray | None = None, *, max_pairs: int = 400
) -> dict[str, Any]:
    """Spread diagnostics for triage. Model-agnostic: samples + truth only.

    ``collapsed`` is the headline: an ensemble whose members are near-identical
    produces a tidy-looking spaghetti plot and a near-binary probability map,
    both of which read as *confidence* rather than as the defect they are.
    """
    s = np.asarray(samples) > 0
    n_members = s.shape[0]
    final = s[:, -1]
    areas = final.sum(axis=(1, 2)).astype(np.float64)

    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(n_members) for j in range(i + 1, n_members)]
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    ious = [_iou(final[i], final[j]) for i, j in pairs] or [1.0]

    prob = burn_probability(samples)
    touched = prob > 0
    uncertain = (prob > 0) & (prob < 1)
    mean_area = float(areas.mean())
    index = independence_dispersion_index(samples)

    diag: dict[str, Any] = {
        "n_members": int(n_members),
        "horizon_h": int(s.shape[1]),
        "independence_dispersion_index": index,
        "mean_pairwise_iou": float(np.mean(ious)),
        "min_pairwise_iou": float(np.min(ious)),
        "member_area_cells_mean": mean_area,
        "member_area_cells_std": float(areas.std()),
        "member_area_cv": float(areas.std() / mean_area) if mean_area > 0 else 0.0,
        "uncertain_cell_frac": float(uncertain.sum() / touched.sum()) if touched.any() else 0.0,
        "n_distinct_members": int(np.unique(final.reshape(n_members, -1), axis=0).shape[0]),
    }
    # Primary criterion is the dispersion index; the rest catch the degenerate
    # cases it cannot see (a literally identical ensemble has index 0/0).
    diag["collapsed"] = bool(
        index < COLLAPSE_INDEX_THRESHOLD
        or diag["n_distinct_members"] <= max(1, n_members // 10)
        or diag["uncertain_cell_frac"] < 0.02
    )

    if truth is not None:
        t = np.asarray(truth) > 0
        truth_area = float(t[-1].sum())
        diag.update(
            {
                "truth_area_cells": truth_area,
                "truth_within_member_range": bool(areas.min() <= truth_area <= areas.max()),
                "truth_area_rank": int((areas < truth_area).sum()),
                "best_member_iou": float(max(_iou(final[i], t[-1]) for i in range(n_members))),
                "ensemble_mean_iou": float(
                    np.mean([_iou(final[i], t[-1]) for i in range(n_members)])
                ),
            }
        )
        # A rank at either extreme means the truth falls outside the ensemble —
        # under-dispersion, the failure this whole viewer exists to expose.
        diag["truth_outside_envelope"] = not diag["truth_within_member_range"]
    return diag


# -- figure ----------------------------------------------------------------


@dataclass
class EnsembleView:
    """One rendered ensemble figure and the numbers behind it."""

    figure_path: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _map_axes(ax: Any, geom: PlotGeometry, title: str) -> None:
    ax.set_xlim(geom.extent[0], geom.extent[1])
    ax.set_ylim(geom.extent[2], geom.extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=5)


def _draw_barrier(ax: Any, geom: PlotGeometry, barrier: np.ndarray) -> None:
    if barrier.any():
        ax.contour(
            geom.x_centres,
            geom.y_centres,
            barrier.astype(float),
            levels=[0.5],
            colors=[COL_BARRIER],
            linewidths=0.9,
            alpha=0.8,
        )


def _truth_front(ax: Any, geom: PlotGeometry, mask: np.ndarray, **kw: Any) -> None:
    if mask.any():
        ax.contour(
            geom.x_centres,
            geom.y_centres,
            mask.astype(float),
            levels=[0.5],
            colors=[kw.pop("color", COL_TRUTH)],
            linewidths=kw.pop("lw", 2.0),
            zorder=8,
            **kw,
        )


def render_ensemble(
    tensor: str | Path,
    predict: Any,
    out: str | Path,
    *,
    t0: int,
    horizon_h: int = 3,
    n_members: int = 24,
    seed: int = 0,
    model_name: str = "unnamed C5 predict()",
    dpi: int = 140,
) -> EnsembleView:
    """Call a C5 ``predict()`` at ``t0`` and render the four ensemble panels."""
    fire = load_fire(tensor)
    ds = open_tensor(Path(tensor))
    inp: C5Inputs = c5_inputs(ds, t0, horizon_h)

    samples = np.asarray(
        predict(
            x0=inp.x0,
            static=inp.static,
            weather=inp.weather,
            n_members=n_members,
            horizon_h=horizon_h,
            seed=seed,
        )
    )
    if samples.shape != (n_members, horizon_h, *inp.x0.shape):
        raise ValueError(
            "C5 violation: predict() returned "
            f"{samples.shape}, expected {(n_members, horizon_h, *inp.x0.shape)}"
        )

    diag = ensemble_diagnostics(samples, inp.truth)
    geom = fire.geom
    prob = burn_probability(samples)
    qmaps, arrive_prob = arrival_quantiles(samples)
    q10, q50, q90 = qmaps
    spread = q90 - q10

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.6), dpi=dpi)
    (ax_sp, ax_pb), (ax_at, ax_sd) = axes

    # -- 1. spaghetti of member fronts ------------------------------------
    ax_sp.imshow(
        np.where(inp.x0 > 0, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.6, **geom.imshow_kwargs
    )
    _draw_barrier(ax_sp, geom, fire.barrier)
    for m in range(samples.shape[0]):
        front = samples[m, -1] > 0
        if front.any():
            ax_sp.contour(
                geom.x_centres,
                geom.y_centres,
                front.astype(float),
                levels=[0.5],
                colors=[COL_MEMBER],
                linewidths=0.7,
                alpha=0.45,
            )
    _truth_front(ax_sp, geom, inp.truth[-1] > 0)
    _map_axes(ax_sp, geom, f"member fronts at +{horizon_h} h  (n={n_members})")
    add_north_arrow(ax_sp)
    ax_sp.legend(
        handles=[
            Line2D([], [], color=COL_MEMBER, lw=1.0, alpha=0.7, label="member front"),
            Line2D([], [], color=COL_TRUTH, lw=2.0, label="truth front"),
            Line2D([], [], color="#9ca3af", lw=4.0, label="state at t0"),
        ],
        loc="lower left",
        fontsize=7,
        frameon=False,
    )

    # -- 2. burn probability ----------------------------------------------
    im = ax_pb.imshow(
        np.where(prob > 0, prob, np.nan), cmap=BURN_PROB_CMAP, vmin=0, vmax=1, **geom.imshow_kwargs
    )
    _draw_barrier(ax_pb, geom, fire.barrier)
    _truth_front(ax_pb, geom, inp.truth[-1] > 0, lw=1.6)
    _map_axes(ax_pb, geom, f"burn probability at +{horizon_h} h")
    fig.colorbar(im, ax=ax_pb, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)

    # -- 3. median arrival time -------------------------------------------
    censored = np.isnan(q50) & (arrive_prob > 0)
    ax_at.imshow(
        np.where(censored, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.8, **geom.imshow_kwargs
    )
    im = ax_at.imshow(
        q50, cmap=ARRIVAL_CMAP, vmin=0.5, vmax=horizon_h + 0.5, **geom.imshow_kwargs
    )
    _draw_barrier(ax_at, geom, fire.barrier)
    _truth_front(ax_at, geom, inp.truth[-1] > 0, lw=1.6)
    _map_axes(ax_at, geom, "median arrival (lead h); grey = censored, p(burn) < 0.5")
    cb = fig.colorbar(im, ax=ax_at, fraction=0.046, pad=0.02, ticks=range(1, horizon_h + 1))
    cb.ax.tick_params(labelsize=7)

    # -- 4. arrival spread -------------------------------------------------
    im = ax_sd.imshow(spread, cmap="viridis", vmin=0, **geom.imshow_kwargs)
    _draw_barrier(ax_sd, geom, fire.barrier)
    _truth_front(ax_sd, geom, inp.truth[-1] > 0, lw=1.6)
    _map_axes(ax_sd, geom, "arrival p90 − p10 (h); NaN where either is censored")
    fig.colorbar(im, ax=ax_sd, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)

    _headline(fig, fire.fire_id, model_name, inp, diag)
    stamp(
        fig,
        f"{fire.source} · C1 v2.3 · C5 provisional split "
        f"C_s={C5_CONVENTION['c_s']} C_w={C5_CONVENTION['c_w']}, weather origin "
        f"{C5_CONVENTION['weather_time_origin']} (see BLOCKERS @model) · seed={seed}",
    )
    fig.tight_layout(rect=(0, 0.012, 1, 0.925))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    meta = {
        "fire_id": fire.fire_id,
        "tensor": str(tensor),
        "model": model_name,
        "t0": t0,
        "t0_time": str(np.datetime_as_string(fire.times[t0], unit="h")),
        "horizon_h": horizon_h,
        "n_members": n_members,
        "seed": seed,
        "c5_convention": C5_CONVENTION,
    }
    return EnsembleView(figure_path=str(out_path), diagnostics=diag, meta=meta)


def _headline(fig: Any, fire_id: str, model_name: str, inp: C5Inputs, diag: dict[str, Any]) -> None:
    t_lbl = str(np.datetime_as_string(inp.times[0], unit="h")).replace("T", " ")
    fig.suptitle(
        f"{fire_id} — ensemble from t0={inp.t0} (+1 h = {t_lbl}Z), horizon {inp.horizon_h} h\n"
        f"{model_name}",
        fontsize=11,
        y=0.985,
    )
    line = (
        f"dispersion vs independent-pixel {diag['independence_dispersion_index']:.2f}×   "
        f"pairwise IoU {diag['mean_pairwise_iou']:.3f}   "
        f"member-area CV {diag['member_area_cv']:.3f}   "
        f"uncertain cells {diag['uncertain_cell_frac']:.2f}   "
        f"distinct {diag['n_distinct_members']}/{diag['n_members']}"
    )
    if "best_member_iou" in diag:
        line += (
            f"   best-member IoU {diag['best_member_iou']:.3f}   "
            f"truth in envelope: {'yes' if diag['truth_within_member_range'] else 'NO'}"
        )
    fig.text(0.5, 0.938, line, ha="center", va="top", fontsize=7.6, color=COL_TEXT)

    warnings: list[str] = []
    if diag["collapsed"]:
        warnings.append(
            f"ENSEMBLE COLLAPSE: dispersion is only "
            f"{diag['independence_dispersion_index']:.2f}× the independent-pixel floor"
        )
    if diag.get("truth_outside_envelope"):
        warnings.append("UNDER-DISPERSED: truth area lies outside the member range")
    if warnings:
        fig.text(
            0.5,
            0.918,
            "  ‖  ".join(warnings),
            ha="center",
            va="top",
            fontsize=8.2,
            color="white",
            bbox={"fc": COL_WARN, "ec": "none", "pad": 3.0},
        )


# -- CLI -------------------------------------------------------------------


def _resolve_predict(name: str) -> tuple[Any, str]:
    """Resolve a C5 predict() by name, through the contract only.

    NO SILENT FALLBACK. An earlier version of this function fell back to the viz
    stub on any exception and merely printed a message. That was written while
    ``model/api.py` did not yet exist; now that it does, the fallback is a
    hazard, not a convenience. If ``load_model('ellipse')`` starts raising for a
    real reason during a gate run, a fallback renders a plausible figure from a
    caricature that ``stub_model``'s own docstring forbids from appearing in a
    gate — and the only trace is a line of scrolled-past stdout. Failing is the
    safe behaviour; the stub is reachable only by asking for it BY NAME.
    """
    if name == "stub":
        from wildfire_nowcast.sim.stub_model import StubEnsemble  # noqa: PLC0415

        stub = StubEnsemble()
        return stub.predict, stub.name

    from wildfire_nowcast.model import api as model_api  # noqa: PLC0415

    if hasattr(model_api, name):
        return getattr(model_api, name), f"C5 {name} (model/api.py)"
    return model_api.load_model(name).predict, f"C5 load_model({name})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.ensemble",
        description="Render the ensemble viewer for a C5 predict() on a C1 tensor.",
    )
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="stub", help="'stub', or a name in model/api.py")
    ap.add_argument("--t0", type=int, default=None, help="default: the hour before peak growth")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--diagnostics-json", default=None)
    args = ap.parse_args(argv)

    t0 = args.t0
    if t0 is None:
        fire = load_fire(args.tensor)
        # Default to just before the biggest run: a viewer aimed at a flat hour
        # would show a correct, useless figure on a fire that is 70% flat.
        t0 = int(np.clip(int(np.argmax(fire.growth_km2)) - 1, 0, fire.n_hours - args.horizon - 1))

    predict, model_name = _resolve_predict(args.model)
    view = render_ensemble(
        args.tensor,
        predict,
        args.out,
        t0=t0,
        horizon_h=args.horizon,
        n_members=args.members,
        seed=args.seed,
        model_name=model_name,
        dpi=args.dpi,
    )
    payload = {"figure": view.figure_path, **view.meta, "diagnostics": view.diagnostics}
    if args.diagnostics_json:
        Path(args.diagnostics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.diagnostics_json).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
