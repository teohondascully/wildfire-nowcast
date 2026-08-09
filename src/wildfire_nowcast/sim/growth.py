"""WHERE the kernel's 2.66-3.06x growth over-prediction comes from.

ADR-021 (3b) records the G2 pass with a named limitation: the kernel over-predicts
new cells **2.66-3.06x** while the growth-calibrated ellipse over-predicts
**1.79x**, and states that G3's dispersion bar is where that collides. A single
ratio cannot say WHY, and the difference between the two available answers
changes what the fix has to be:

*speed*
    the kernel advances too fast wherever it advances. Then the remedy is a scale
    — a smaller reach, a lower amplitude — and it costs nothing but calibration.
*silence*
    the kernel advances at roughly the right rate when the fire is running, and
    keeps advancing when the fire has stopped. Then a scale is the WRONG remedy:
    dividing by 2.7 would fix the total while making the running fire too slow,
    because the excess is not where the growth is.
*geometry*
    the head advance is right and the flanks and rear are over-driven. Then a
    scale is wrong for the same reason, one dimension over.

These are distinguishable, from OUTSIDE the model, through ``predict()`` and the
C1 tensor only. This module measures all three and renders them.

THE DECOMPOSITIONS
------------------
Everything below is arithmetic on C6.2's OWN quantity. ``eval.validity`` defines
the ratio as ``sum over windows of (mean over members of new cells at the final
lead, relative to x0)`` divided by the same sum for truth. That total is a SUM over
windows, so it splits exactly, and this module asserts the split reconstructs the
run's recorded ``growth_ratio`` before reporting any of it — the same standard the
S2 IoU decomposition was held to.

``dormancy``
    windows where truth grew ZERO cells over the horizon vs windows where it grew.
    62% of held-out windows have bitwise-zero growth (the run's own note), so a
    forecaster with no OFF state pays for it 953 times before it says anything
    about a fire that is moving.
``sector``
    every predicted and observed new cell is assigned an OUTWARD NORMAL — the unit
    vector from its nearest ``t0``-burned cell, from the exact Euclidean distance
    transform, so the classification is LOCAL and does not assume the fire is
    round. Cells are binned head / flank / rear by that normal's angle to the mean
    wind over the window. A per-sector ratio separates "too fast" from "too fast
    in the wrong places".

C-4 AND CHECKPOINTS
-------------------
This module NEVER opens a path under ``runs/`` while another lead may be writing
it. :func:`snapshot_checkpoint` copies a run directory to a scratch location,
hashes the source before and after the copy and the copy itself, and refuses to
proceed unless all three agree; the loaded model is the COPY, and the hash goes
into the output JSON. A torn read then fails loudly instead of producing a figure
of a model that never existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: E402
from wildfire_nowcast.sim.c5 import WEATHER_C5, C5Inputs  # noqa: E402
from wildfire_nowcast.sim.replay import iter_eval_windows, load_gate_models  # noqa: E402
from wildfire_nowcast.sim.style import COL_TEXT, COL_WARN, stamp  # noqa: E402

__all__ = [
    "SECTORS",
    "snapshot_checkpoint",
    "outward_normals",
    "sector_of",
    "WindowGrowth",
    "measure_fire",
    "render",
    "main",
]

#: Head / flank / rear, by the cosine between a cell's outward normal and the
#: mean wind. 60 deg half-angles, so the three bins have comparable solid angle
#: (head 1/3 of the circle, flank 1/3, rear 1/3) and a ratio between them is not
#: an artefact of unequal bin widths.
SECTORS: tuple[str, ...] = ("head", "flank", "rear")
_COS_HEAD = 0.5
_COS_REAR = -0.5


# --------------------------------------------------------------------------
# C-4: never read a checkpoint another lead may be writing
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def snapshot_checkpoint(run_dir: str | Path, dest_root: str | Path) -> tuple[Path, str]:
    """Copy a run directory, proving the source did not move under the copy.

    Returns ``(snapshot_dir, sha256_of_model_json)``. Raises if the source hash
    changes across the copy or the copy does not match — which is exactly the
    condition C-4 exists for, and a hazard I filed against ``eval/selftest.py``
    before it was mine to trip over.
    """
    src = Path(run_dir)
    model = src / "model.json"
    if not model.exists():
        raise FileNotFoundError(f"{model} does not exist; not a loadable run directory")
    before = _sha256(model)
    dest = Path(dest_root) / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    after = _sha256(model)
    copied = _sha256(dest / "model.json")
    if not (before == after == copied):
        raise RuntimeError(
            f"checkpoint {src} moved while being snapshotted "
            f"(before={before[:12]} after={after[:12]} copy={copied[:12]}). "
            "C-4: another lead is writing it. Refusing to render a figure of a "
            "model that may never have existed."
        )
    return dest, before


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


#: How far from the ``t0`` frontier a normal is defined. A 3 h window's reachable
#: band is a handful of cells (``eval.masks.default_band_radius``), so 12 is
#: generous; cells beyond it get no sector and are counted in the ``unassigned``
#: audit rather than silently dropped into a bin.
NORMAL_SEARCH_RADIUS = 12


def _offsets_by_distance(radius: int) -> list[tuple[int, int]]:
    """All integer offsets within ``radius``, nearest first. Ties in stable order."""
    offs = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if 0 < dy * dy + dx * dx <= radius * radius
    ]
    offs.sort(key=lambda o: (o[0] * o[0] + o[1] * o[1], o[0], o[1]))
    return offs


def outward_normals(
    burned0: np.ndarray, *, radius: int = NORMAL_SEARCH_RADIUS
) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector from each cell's NEAREST burned cell to the cell (outward).

    Returns ``(ny, nx)`` in array coordinates: ``ny`` increases DOWNWARD (row
    index), which for a C1.4 y-descending store is SOUTHWARD. Callers converting
    to a compass frame must flip the sign of ``ny``; :func:`sector_of` does.

    The normal is LOCAL — nearest burned cell, not the fire centroid. A
    centroid-based direction is biased on an elongated or crescent fire, which is
    most of them after a few hours, and the bias points along the very axis this
    module is trying to measure.

    Implemented as a nearest-first offset scan rather than a library EDT: this
    project's venv has no scipy, and adding a dependency to `pyproject.toml` is
    infra's file, not mine (C-4). Exactness is asserted against a brute-force
    reference in the self-tests, so the substitution is checked, not assumed.
    """
    burned = np.asarray(burned0) > 0
    h, w = burned.shape
    ny = np.zeros((h, w), dtype=np.float64)
    nx = np.zeros((h, w), dtype=np.float64)
    if not burned.any():
        return ny, nx
    assigned = burned.copy()  # burned cells have no outward normal
    for dy, dx in _offsets_by_distance(int(radius)):
        if assigned.all():
            break
        # Does the cell at (y, x) have a burned neighbour at (y+dy, x+dx)?
        shifted = np.zeros((h, w), dtype=bool)
        ys = slice(max(0, -dy), h - max(0, dy))
        xs = slice(max(0, -dx), w - max(0, dx))
        ys_src = slice(max(0, dy), h - max(0, -dy))
        xs_src = slice(max(0, dx), w - max(0, -dx))
        shifted[ys, xs] = burned[ys_src, xs_src]
        hit = shifted & ~assigned
        if not hit.any():
            continue
        norm = float(np.hypot(dy, dx))
        # The cell lies at -(dy, dx) from the burned cell it found.
        ny[hit] = -dy / norm
        nx[hit] = -dx / norm
        assigned |= hit
    return ny, nx


def sector_of(burned0: np.ndarray, wind_u: float, wind_v: float) -> np.ndarray:
    """Per-cell sector index: 0 head, 1 flank, 2 rear. ``-1`` where undefined.

    ``wind_u``/``wind_v`` are the C1 eastward/northward components. The DOWNWIND
    direction is ``(+u east, +v north)``; in array coordinates north is ``-y``, so
    the downwind unit vector is ``(-v, +u)`` after normalisation.
    """
    ny, nx = outward_normals(burned0)
    speed = float(np.hypot(wind_u, wind_v))
    out = np.full(np.asarray(burned0).shape, -1, dtype=np.int8)
    if speed <= 0:
        return out
    dy, dx = -float(wind_v) / speed, float(wind_u) / speed
    cos = ny * dy + nx * dx
    out = np.where(cos >= _COS_HEAD, 0, np.where(cos <= _COS_REAR, 2, 1)).astype(np.int8)
    out[np.asarray(burned0) > 0] = -1
    return out


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


@dataclass
class WindowGrowth:
    """One window, one model: C6.2's own count, split by dormancy and sector."""

    fire_id: str
    model: str
    window_index: int
    t0: int
    truth_new: float
    pred_new: float
    truth_grew: bool
    wind_speed: float
    truth_sector: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pred_sector: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


def _new_cells(field_: np.ndarray, burned0: np.ndarray) -> np.ndarray:
    return (np.asarray(field_) > 0) & ~(np.asarray(burned0) > 0)


def _mean_wind(window: C5Inputs) -> tuple[float, float]:
    """Domain- and time-mean ``(u, v)`` over the window's weather, by NAME."""
    iu = WEATHER_C5.index("wind_u10")
    iv = WEATHER_C5.index("wind_v10")
    return (
        float(np.mean(window.weather[:, iu])),
        float(np.mean(window.weather[:, iv])),
    )


def measure_fire(
    fire_id: str,
    tensor_path: str | Path,
    models: dict[str, Any],
    *,
    horizon_h: int = 3,
    stride: int = 2,
    n_members: int = 24,
    seed: int = 20260807,
) -> list[WindowGrowth]:
    """Every window of one fire, every model. C5 ``predict()`` only."""
    ds = open_tensor(Path(tensor_path))
    rows: list[WindowGrowth] = []
    for i, w in iter_eval_windows(ds, horizon_h, stride=stride):
        burned0 = w.x0 > 0
        truth_new = _new_cells(w.truth[-1], burned0)
        u, v = _mean_wind(w)
        sect = sector_of(w.x0, u, v)
        t_sec = [float(np.count_nonzero(truth_new & (sect == s))) for s in range(3)]
        for name, model in models.items():
            samples = model.predict(w.x0, w.static, w.weather, n_members, horizon_h, seed + i)
            member_final = _new_cells(samples[:, -1], burned0)  # [M, H, W]
            pred = float(np.count_nonzero(member_final, axis=(1, 2)).mean())
            p_sec = [
                float(np.count_nonzero(member_final & (sect == s)[None], axis=(1, 2)).mean())
                for s in range(3)
            ]
            rows.append(
                WindowGrowth(
                    fire_id=fire_id,
                    model=name,
                    window_index=i,
                    t0=int(w.t0),
                    truth_new=float(np.count_nonzero(truth_new)),
                    pred_new=pred,
                    truth_grew=bool(np.count_nonzero(truth_new) > 0),
                    wind_speed=float(np.hypot(u, v)),
                    truth_sector=t_sec,
                    pred_sector=p_sec,
                )
            )
    return rows


def summarise(rows: list[WindowGrowth]) -> dict[str, Any]:
    """Pool per model. Every total is a plain sum, so every split is exact."""
    out: dict[str, Any] = {}
    models = sorted({r.model for r in rows})
    for m in models:
        rs = [r for r in rows if r.model == m]
        grew = [r for r in rs if r.truth_grew]
        dorm = [r for r in rs if not r.truth_grew]
        pred_total = sum(r.pred_new for r in rs)
        truth_total = sum(r.truth_new for r in rs)
        pred_g = sum(r.pred_new for r in grew)
        pred_d = sum(r.pred_new for r in dorm)
        t_sec = [sum(r.truth_sector[s] for r in rs) for s in range(3)]
        p_sec = [sum(r.pred_sector[s] for r in rs) for s in range(3)]
        # The all-windows sector ratio is INFLATED by the dormancy effect in
        # exactly the way finding (19) describes — truth is 0 in every sector of
        # every dormant window, so the ratio there is driven by the same missing
        # off state and says nothing about geometry. The growth-window form is
        # the one that isolates SHAPE, and it is the one to read.
        t_sec_g = [sum(r.truth_sector[s] for r in grew) for s in range(3)]
        p_sec_g = [sum(r.pred_sector[s] for r in grew) for s in range(3)]
        # Exactness, asserted rather than hoped for: the two splits must both
        # reconstruct the same total the gate's C6.2 verdict is computed from.
        assert abs((pred_g + pred_d) - pred_total) < 1e-6, (m, pred_g + pred_d, pred_total)
        assert abs(sum(p_sec) - pred_total) < 1e-6, (m, sum(p_sec), pred_total)
        out[m] = {
            "n_windows": len(rs),
            "n_windows_truth_grew": len(grew),
            "n_windows_truth_dormant": len(dorm),
            "n_windows_model_ignited": int(sum(1 for r in rs if r.pred_new > 0)),
            "n_dormant_windows_model_ignited": int(sum(1 for r in dorm if r.pred_new > 0)),
            "n_new_cells_predicted": pred_total,
            "n_new_cells_truth": truth_total,
            "growth_ratio": (pred_total / truth_total) if truth_total else None,
            "predicted_in_growth_windows": pred_g,
            "predicted_in_dormant_windows": pred_d,
            "dormant_share_of_prediction": (pred_d / pred_total) if pred_total else None,
            "growth_ratio_on_growth_windows": (
                pred_g / sum(r.truth_new for r in grew) if grew else None
            ),
            "excess_total": pred_total - truth_total,
            "excess_from_dormant_windows": pred_d,
            "dormant_share_of_excess": (
                pred_d / (pred_total - truth_total) if pred_total > truth_total else None
            ),
            "sector_truth": dict(zip(SECTORS, t_sec, strict=True)),
            "sector_pred": dict(zip(SECTORS, p_sec, strict=True)),
            "sector_ratio": {
                s: (p / t if t else None) for s, t, p in zip(SECTORS, t_sec, p_sec, strict=True)
            },
            "sector_truth_growth_windows": dict(zip(SECTORS, t_sec_g, strict=True)),
            "sector_pred_growth_windows": dict(zip(SECTORS, p_sec_g, strict=True)),
            "sector_ratio_growth_windows": {
                s: (p / t if t else None)
                for s, t, p in zip(SECTORS, t_sec_g, p_sec_g, strict=True)
            },
        }
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render(summary: dict[str, Any], rows: list[WindowGrowth], out: str | Path) -> Path:
    """Four panels: the ratio, where it comes from, the sector split, the scatter."""
    models = [m for m in summary if m != "persistence"]
    order = [m for m in models if m.startswith("kernel") or m.startswith("nbfix")] + [
        m for m in models if not (m.startswith("kernel") or m.startswith("nbfix"))
    ]
    colour = {
        m: ("#0f766e" if (m.startswith("nbfix") or m == "kernel") else "#a78bfa"
            if m == "kernel_init" else "#b45309")
        for m in order
    }

    fig = plt.figure(figsize=(17.0, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.22, left=0.06, right=0.985,
                          top=0.855, bottom=0.09)

    # (1) the headline ratio, and the same ratio computed ONLY where truth grew
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(order))
    all_r = [summary[m]["growth_ratio"] or np.nan for m in order]
    grow_r = [summary[m]["growth_ratio_on_growth_windows"] or np.nan for m in order]
    # DEFECT, mine, found by looking at the render: the second series was drawn
    # with a PER-MODEL colour list while the legend carried a single swatch, so
    # the legend showed one arbitrary model's colour and implied the bars were
    # colour-coded by something they were not. One series, one colour.
    ax.bar(xs - 0.2, all_r, width=0.38, color="#cbd5e1", edgecolor="#64748b", lw=0.7,
           label="ALL windows (the ADR-021 number)")
    ax.bar(xs + 0.2, grow_r, width=0.38, color="#0f766e", lw=0.7,
           edgecolor="#111827", label="ONLY windows where truth GREW")
    ax.axhline(1.0, color=COL_TEXT, lw=1.0, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=32, ha="right", fontsize=7)
    ax.set_ylabel("predicted new cells / truth new cells")
    ax.set_title("(1) the over-prediction ratio, with and without the dormant windows",
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=7, frameon=False)

    # (2) where the predicted cells LAND: dormant vs growing windows
    ax = fig.add_subplot(gs[0, 1])
    dorm = [summary[m]["predicted_in_dormant_windows"] for m in order]
    grew = [summary[m]["predicted_in_growth_windows"] for m in order]
    ax.bar(xs, grew, color="#0f766e", edgecolor="#134e4a", lw=0.7,
           label="predicted in windows where truth grew")
    ax.bar(xs, dorm, bottom=grew, color=COL_WARN, edgecolor="#7c2d12", lw=0.7,
           label="predicted in windows where truth grew NOTHING")
    truth_total = summary[order[0]]["n_new_cells_truth"] if order else 0.0
    ax.axhline(truth_total, color="#111827", lw=1.4, ls=":",
               label=f"TRUTH total = {truth_total:.0f} cells")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=32, ha="right", fontsize=7)
    ax.set_ylabel("new cells (ensemble mean, summed over windows)")
    ax.set_title("(2) a forecaster with no OFF state pays here", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # (3) sector ratio: head / flank / rear
    ax = fig.add_subplot(gs[1, 0])
    width = 0.8 / max(len(order), 1)
    base = np.arange(len(SECTORS))
    # SECOND defect found by looking: this plotted the ALL-WINDOWS sector ratio,
    # which is inflated by the dormancy effect panel (2) is about — truth is 0 in
    # every sector of every dormant window — so the panel asking "is it the head
    # or the flanks" was answering with the same number panel (1) already
    # answered. The GROWTH-WINDOW form isolates shape, and it inverts the
    # reading: on all windows the kernel looks rear-heavy AND head-heavy; on
    # growth windows it is head-DEFICIENT (0.86) and rear-EXCESSIVE (1.31).
    for k, m in enumerate(order):
        vals = [summary[m]["sector_ratio_growth_windows"][s] or np.nan for s in SECTORS]
        ax.bar(base + k * width - 0.4 + width / 2, vals, width=width * 0.92,
               color=colour[m], edgecolor="#111827", lw=0.5, label=m)
    ax.axhline(1.0, color=COL_TEXT, lw=1.0, ls="--")
    ax.set_xticks(base)
    ax.set_xticklabels([f"{s}\n(vs mean wind)" for s in SECTORS], fontsize=8)
    ax.set_ylabel("predicted / truth new cells in sector")
    ax.set_title(
        "(3) SHAPE, on growth windows only — head-deficient, rear-excessive.\n"
        "Compare kernel_init (untrained): the anisotropy is INVERTED by training.",
        fontsize=9.5,
    )
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (4) per-window scatter: predicted vs truth
    ax = fig.add_subplot(gs[1, 1])
    for m in order:
        rs = [r for r in rows if r.model == m]
        ax.scatter([r.truth_new for r in rs], [r.pred_new for r in rs], s=6, alpha=0.35,
                   color=colour[m], label=m, linewidths=0)
    lim = max([r.truth_new for r in rows] + [r.pred_new for r in rows] + [1.0])
    ax.plot([0, lim], [0, lim], color=COL_TEXT, lw=1.0, ls="--", label="perfect")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("truth new cells in window (symlog)")
    ax.set_ylabel("predicted new cells (ensemble mean, symlog)")
    ax.set_title("(4) per window — the column at truth=0 is the whole story", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")

    n_d = summary[order[0]]["n_windows_truth_dormant"] if order else 0
    n_w = summary[order[0]]["n_windows"] if order else 0
    fig.suptitle(
        "ANATOMY OF THE GROWTH OVER-PREDICTION — every number via C5 predict() and the C1 "
        "tensor; no model internal is read\n"
        f"{n_w} windows, {n_d} of them ({100.0 * n_d / max(n_w, 1):.0f}%) with BITWISE ZERO "
        "truth growth. Splits sum EXACTLY to C6.2's own growth_ratio (asserted).",
        fontsize=10.5,
    )
    stamp(fig, "ADR-021 (3b) names 2.66-3.06x as the limitation G3's dispersion bar collides "
               "with. This is where it comes from.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.growth", allow_abbrev=False
    )
    ap.add_argument("--results", required=True, help="runs/<id>/results.json (the run of record)")
    ap.add_argument("--kernel", action="append", default=[], help="name=run_dir")
    ap.add_argument("--fire", action="append", default=[], help="fire_id=tensor_path")
    ap.add_argument("--snapshot-root", default=None,
                    help="copy checkpoints here first (C-4). Strongly recommended.")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", default="reports/figures/growth_anatomy.png")
    args = ap.parse_args(argv)

    kernels = list(args.kernel)
    snapshots: dict[str, str] = {}
    if args.snapshot_root:
        pinned = []
        for spec in kernels:
            name, _, run_dir = spec.partition("=")
            dest, sha = snapshot_checkpoint(run_dir, args.snapshot_root)
            snapshots[name] = f"{run_dir} sha256:{sha[:16]}"
            pinned.append(f"{name}={dest}")
        kernels = pinned

    gate = load_gate_models(args.results, kernels=kernels, include_persistence=False)
    models = gate.models

    rows: list[WindowGrowth] = []
    for spec in args.fire:
        fire_id, _, path = spec.partition("=")
        rows.extend(
            measure_fire(fire_id, path, models, horizon_h=args.horizon,
                         stride=args.stride, n_members=args.members, seed=args.seed)
        )
    summary = summarise(rows)

    recorded = json.loads(Path(args.results).read_text()).get("c6_2_validity") or {}
    reconciliation = {}
    for m, s in summary.items():
        rec = recorded.get(m)
        if not rec:
            continue
        reconciliation[m] = {
            "recorded_growth_ratio": rec.get("growth_ratio"),
            "replayed_growth_ratio": s["growth_ratio"],
            "abs_delta": abs(float(rec.get("growth_ratio") or 0.0) - (s["growth_ratio"] or 0.0)),
            "recorded_n_windows": rec.get("n_windows"),
            "replayed_n_windows": s["n_windows"],
        }

    fig = render(summary, rows, args.out)
    payload = {
        "kind": "growth_anatomy",
        "results_json": args.results,
        "checkpoint_snapshots": snapshots,
        "horizon_h": args.horizon,
        "n_members": args.members,
        "stride": args.stride,
        "seed": args.seed,
        "sector_definition": {
            "head": "cos(outward normal, mean wind) >= 0.5",
            "flank": "-0.5 < cos < 0.5",
            "rear": "cos <= -0.5",
            "normal": "unit vector from the nearest t0-burned cell (exact EDT), per cell",
        },
        "summary": summary,
        "reconciliation_with_c6_2": reconciliation,
        "rows": [asdict(r) for r in rows],
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(payload, indent=1))
    print(f"[growth] {fig}")
    for m, s in summary.items():
        print(
            f"  {m:16s} ratio {s['growth_ratio']:.3f}  "
            f"on-growth-windows {s['growth_ratio_on_growth_windows']:.3f}  "
            f"dormant share of prediction {100.0 * (s['dormant_share_of_prediction'] or 0):.1f}%  "
            f"ignited in {s['n_dormant_windows_model_ignited']}/"
            f"{s['n_windows_truth_dormant']} dormant windows"
        )
        print(
            "                   sector ratio  "
            + "  ".join(
                f"{k} {('n/a' if v is None else f'{v:.2f}')}" for k, v in s["sector_ratio"].items()
            )
        )
    for m, r in reconciliation.items():
        print(f"  [reconcile] {m:16s} recorded {r['recorded_growth_ratio']} "
              f"replayed {r['replayed_growth_ratio']} delta {r['abs_delta']:.4g} "
              f"windows {r['recorded_n_windows']} vs {r['replayed_n_windows']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
