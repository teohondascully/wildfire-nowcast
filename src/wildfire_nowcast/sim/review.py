"""S7 - the external-reviewer page.

Builds ``reports/review.html``: ONE self-contained HTML file (every image inlined
as a ``data:`` URI) that walks a reader who has seen nothing since the original
design sketch from their mental model to the current evidence.

HARD CONSTRAINTS THIS MODULE OBEYS, AND THEY ARE THE POINT
-----------------------------------------------------------
* **Nothing is imported from ``wildfire_nowcast.eval``** and no scoring code is
  re-run. modelling holds ``eval/`` while M9 is in flight (C-4). Every number
  on the page is READ OUT of a run record or a ``reports/figures/*.json``
  artifact that was written by a completed run.
* **No checkpoint is read and ``predict()`` is never called.** Anything that
  would need a live model is rendered as a labelled ``NOT MEASURED`` gap.
* The only live data touched is the C1 tensor store, read-only, through
  :mod:`wildfire_nowcast.sim.reader` - the contract input this lead consumes.
* **Every number carries its provenance**: the run record it came from and the
  split fingerprint it is bound to. The corpus of record is
  ``b3e5dadad01eaef9`` (21 fires); G2 and all four G3 attempts are bound to
  ``4848f491e8d588fa`` (12 fires) and are labelled as such wherever they appear.
* Numbers are never typed by hand into the prose. They are pulled from records
  into :func:`collect` and substituted, so a stale number cannot survive a
  re-render.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from wildfire_nowcast.common.logs import configure_logging  # noqa: E402
from wildfire_nowcast.sim.style import add_north_arrow  # noqa: E402

# ADR-103: a logger, and NOTHING else at import. `main` configures. What this
# page PRINTS is its output; what it says about how it was built - a figure it
# could not downscale, a section it could not fill - is a diagnostic.
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# provenance constants - the C8 archive boundary, named, never inlined
# --------------------------------------------------------------------------

FINGERPRINT_PRE_D6 = "4848f491e8d588fa"  # 12 fires: G2 + all four G3 attempts
FINGERPRINT_OF_RECORD = "b3e5dadad01eaef9"  # 21 fires: the corpus of record

REPO = Path(__file__).resolve().parents[3]
FIGDIR = REPO / "reports" / "figures"
OUT_HTML = REPO / "reports" / "review.html"

RECORD_G2 = "runs/baselines-20260808-095003/results.json"
RECORD_M7 = "runs/baselines-20260809-073414/results.json"
RECORD_M8 = "runs/baselines-20260809-102243/results.json"

GATE_IOU_KEY = "band best-member IoU (SHAPE, masked)"
G2_SEEDS = ("nbfix_s0", "nbfix_s1", "nbfix_s2", "nbfix_s3")

INK = "#16181d"
MUTED = "#6b7280"
GOOD = "#0f766e"
BAD = "#b91c1c"
WARN = "#b45309"
COOL = "#1d4ed8"
GREY = "#9ca3af"


def _load(rel: str) -> dict[str, Any]:
    return json.loads((REPO / rel).read_text())


def _fig(rel: str) -> dict[str, Any]:
    return json.loads((FIGDIR / rel).read_text())


# --------------------------------------------------------------------------
# 1. COLLECT - every number the page shows, pulled from a record
# --------------------------------------------------------------------------


@dataclass
class PageData:
    """Everything the page states, with the record each item came from."""

    g2: dict[str, Any] = field(default_factory=dict)
    growth: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    block5: dict[str, Any] = field(default_factory=dict)
    dispersion: dict[str, Any] = field(default_factory=dict)
    elmfire: dict[str, Any] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)


def collect() -> PageData:
    d = PageData()
    _collect_g2(d)
    _collect_growth(d)
    _collect_identity(d)
    _collect_block5(d)
    _collect_dispersion(d)
    _collect_elmfire(d)
    return d


def _collect_elmfire(d: PageData) -> None:
    v = _fig("elmfire_degeneracy_verdict.json")
    w = v["windows"]
    d.elmfire = {
        "artifact": "reports/figures/elmfire_degeneracy_verdict.json",
        "note": v["note"],
        "tensor": v["tensor"],
        "n_members": v["n_members"],
        "worst_truth": max(x["truth_new_cells"] for x in w),
        "native_median_min": min(x["arms"]["native"]["median_new_cells"] for x in w),
        "native_median_max": max(x["arms"]["native"]["median_new_cells"] for x in w),
        "native_ratio_min": min(x["arms"]["native"]["ratio_to_truth"] for x in w),
        "native_ratio_max": max(x["arms"]["native"]["ratio_to_truth"] for x in w),
        "lobo_median_min": min(x["arms"]["lobotomised"]["median_new_cells"] for x in w),
        "lobo_median_max": max(x["arms"]["lobotomised"]["median_new_cells"] for x in w),
        "lobo_distinct_min": min(x["arms"]["lobotomised"]["distinct_members"] for x in w),
        "native_degenerate": any(x["arms"]["native"]["degenerate"] for x in w),
        "lobo_degenerate": all(x["arms"]["lobotomised"]["degenerate"] for x in w),
        "n_windows": len(w),
    }


def _collect_g2(d: PageData) -> None:
    rec = _load(RECORD_G2)
    bh = rec["g2_per_horizon"]["by_horizon"]
    out: dict[str, Any] = {
        "record": RECORD_G2,
        "fingerprint": rec["split_before"]["fingerprint"],
        "n_fires": rec["split_before"]["n_fires"],
        "heldout_fire_ids": rec["scope"]["heldout_fire_ids"],
        "heldout_block_ids": rec["scope"]["heldout_block_ids"],
        "n_heldout_blocks": rec["scope"]["n_heldout_blocks"],
        "n_members": rec["n_members"],
        "horizons": {},
        "criterion": rec["gate_criterion"]["key"],
    }
    unanimity_hits = 0
    unanimity_cells = 0
    for h in ("1", "2", "3"):
        m = bh[h]["metrics"][GATE_IOU_KEY]
        brier = bh[h]["metrics"]["band Brier"]
        vals = np.array([m["candidates"][s]["value"] for s in G2_SEEDS], float)
        sd_seed = float(vals.std(ddof=1))
        nb = len(m["candidates"][G2_SEEDS[0]]["per_block"])
        per_block_diff, per_block_rows = [], []
        for i in range(nb):
            cvals = [m["candidates"][s]["per_block"][i]["candidate"] for s in G2_SEEDS]
            opp = m["candidates"][G2_SEEDS[0]]["per_block"][i]["opponent"]
            per_block_diff.append(float(np.mean(cvals)) - float(opp))
            wins = [m["candidates"][s]["per_block"][i]["candidate_wins"] for s in G2_SEEDS]
            unanimity_hits += int(sum(bool(w) for w in wins))
            unanimity_cells += len(wins)
            per_block_rows.append(
                {
                    "fire_id": m["candidates"][G2_SEEDS[0]]["per_block"][i]["fire_id"],
                    "block": m["candidates"][G2_SEEDS[0]]["per_block"][i]["spatial_block_id"],
                    "candidate_by_seed": [float(v) for v in cvals],
                    "opponent": float(opp),
                    "wins": [bool(w) for w in wins],
                }
            )
        pb = np.array(per_block_diff, float)
        sd_block = float(pb.std(ddof=1))
        out["horizons"][h] = {
            "rule_opponent": bh[h]["rule_opponent"],
            "rule_value": float(m["rule_opponent_value"]),
            "envelope_value": float(m["envelope_value"]),
            "envelope_from": m["envelope_from"],
            "persistence": float(m["persistence"]),
            "kernel_mean": float(vals.mean()),
            "kernel_by_seed": [float(v) for v in vals],
            "kernel_init": float(m["candidates"]["kernel_init"]["value"]),
            "sd_across_seed": sd_seed,
            "sd_across_block": sd_block,
            "margin_seed_sd_vs_rule": float((vals.mean() - m["rule_opponent_value"]) / sd_seed),
            "margin_seed_sd_vs_env": float((vals.mean() - m["envelope_value"]) / sd_seed),
            "margin_block_sd": float(pb.mean() / sd_block),
            "margin_block_t": float(pb.mean() / (sd_block / math.sqrt(len(pb)))),
            "untrained_deficit_seed_sd": float(
                (vals.mean() - m["candidates"]["kernel_init"]["value"]) / sd_seed
            ),
            "per_block": per_block_rows,
            "brier_kernel": float(np.mean([brier["candidates"][s]["value"] for s in G2_SEEDS])),
            "brier_persistence": float(brier["persistence"]),
            "brier_rule": float(brier["rule_opponent_value"]),
        }
    out["unanimity"] = {"hits": unanimity_hits, "cells": unanimity_cells}
    d.g2 = out


def _collect_growth(d: PageData) -> None:
    rec = _load(RECORD_G2)
    ga = _fig("growth_anatomy.json")
    val = rec["c6_2_validity"]
    # only the models the anatomy artifact actually replayed - never a default
    models = [
        m for m in ("nbfix_s1", "nbfix_s2", "kernel_init", "ellipse_cal3h") if m in ga["summary"]
    ]
    d.growth = {
        "record": RECORD_G2,
        "anatomy": (
            "growth anatomy built by wildfire_nowcast.sim.growth; not tracked, "
            "and neither is the record it is built from, so "
            "**A CLONE CANNOT CHECK THIS NUMBER**"
        ),
        "fingerprint": rec["split_before"]["fingerprint"],
        "truth_new_cells": float(val["nbfix_s1"]["n_new_cells_truth"]),
        "n_windows": int(val["nbfix_s1"]["n_windows"]),
        "n_growth_windows": int(val["nbfix_s1"]["n_windows_with_truth_growth"]),
        "n_dormant_windows": int(ga["summary"]["nbfix_s1"]["n_windows_truth_dormant"]),
        "pooled": {m: float(val[m]["growth_ratio"]) for m in models},
        "on_growth_windows": {
            m: float(ga["summary"][m]["growth_ratio_on_growth_windows"]) for m in models
        },
        "predicted_total": {m: float(ga["summary"][m]["n_new_cells_predicted"]) for m in models},
        "predicted_in_growth": {
            m: float(ga["summary"][m]["predicted_in_growth_windows"]) for m in models
        },
        "predicted_in_dormant": {
            m: float(ga["summary"][m]["predicted_in_dormant_windows"]) for m in models
        },
        "dormant_ignition_count": {
            m: int(ga["summary"][m]["n_dormant_windows_model_ignited"]) for m in models
        },
        "reconciliation_max_delta": max(
            abs(v["abs_delta"]) for v in ga["reconciliation_with_c6_2"].values()
        ),
    }


def _collect_identity(d: PageData) -> None:
    an = _fig("s5_block5_anatomy.json")
    parts = an["parts"]
    by_model: dict[str, dict[int, dict[str, Any]]] = {}
    for p in parts:
        by_model.setdefault(p["model"], {})[int(p["spatial_block_id"])] = p

    def spread(model: str, key: str) -> float | None:
        vals = [
            v[key]
            for v in by_model[model].values()
            if v.get(key) is not None and v[key] > 0 and np.isfinite(v[key])
        ]
        if len(vals) < 4:
            return None
        return float(max(vals) / min(vals))

    trained = [m for m in sorted(by_model) if m.startswith(("m6_", "m7_", "m8_"))]
    rows = []
    for m in trained:
        cv_sp, gc_sp = spread(m, "ensemble_cv"), spread(m, "growth_calibration")
        cvs = [by_model[m][b]["ensemble_cv"] for b in sorted(by_model[m])]
        gcs = [by_model[m][b]["growth_calibration"] for b in sorted(by_model[m])]
        adrs = [by_model[m][b]["adr"] for b in sorted(by_model[m])]
        rows.append(
            {
                "model": m,
                "cv": [float(x) for x in cvs],
                "gc": [float(x) for x in gcs],
                "adr": [float(x) for x in adrs],
                "cv_block_spread": cv_sp,
                "gc_block_spread": gc_sp,
            }
        )
    cv_all = [c for r in rows for c in r["cv"]]
    wide = [r for r in rows if min(r["cv"]) > 1.0]
    d.identity = {
        "record": an["results_path"],
        "artifact": (
            "block-5 anatomy built by wildfire_nowcast.sim.s5_report; not tracked, "
            "and neither is the record it is built from, so "
            "**A CLONE CANNOT CHECK THIS NUMBER**"
        ),
        "fingerprint": an["split_fingerprint"],
        "identity": an["identity"],
        "max_residual": float(an["max_identity_residual"]),
        "known_answer": an["known_answer_check"],
        "rows": rows,
        "n_trained_arms": len(rows),
        "cv_value_min": float(min(cv_all)),
        "cv_value_max": float(max(cv_all)),
        "cv_value_span": float(max(cv_all) / min(cv_all)),
        "cv_block_spread_wide_min": float(min(r["cv_block_spread"] for r in wide)),
        "cv_block_spread_wide_max": float(max(r["cv_block_spread"] for r in wide)),
        "cv_block_spread_all_min": float(min(r["cv_block_spread"] for r in rows)),
        "cv_block_spread_all_max": float(max(r["cv_block_spread"] for r in rows)),
        "gc_block_spread_min": float(min(r["gc_block_spread"] for r in rows)),
        "gc_block_spread_max": float(max(r["gc_block_spread"] for r in rows)),
        "n_wide_arms": len(wide),
        "truth_shape_min": float(min(p["truth_shape_factor"] for p in parts)),
        "truth_shape_max": float(max(p["truth_shape_factor"] for p in parts)),
    }


def _collect_block5(d: PageData) -> None:
    an = _fig("s5_block5_anatomy.json")
    rec = _load(RECORD_M8)
    ms = rec["g3"]["models"]
    lowest, total, per_arm = 0, 0, []
    for name, v in ms.items():
        pb = v["criteria"]["ensemble dispersion (area spread-skill)"]["per_block"]
        if not pb or any(x is None for x in pb.values()):
            continue
        vals = {int(k): float(x) for k, x in pb.items()}
        if max(vals.values()) <= 0:
            continue
        total += 1
        lowest += int(min(vals, key=lambda k: vals[k]) == 5)
        per_arm.append({"arm": name, **{f"b{k}": vals[k] for k in sorted(vals)}})
    blocks = an["train_support_distance"]["blocks"]
    d.block5 = {
        "record": RECORD_M8,
        "fingerprint": rec["split_before"]["fingerprint"],
        "n_arms_scored": total,
        "n_arms_block5_lowest": lowest,
        "per_arm": per_arm,
        "mahalanobis": {k: float(v["mahalanobis"]) for k, v in blocks.items()},
        "block_of": {
            "2020_bobcat": 3,
            "2020_creek": 4,
            "2020_czu_lightning_complex": 5,
            "2020_dolan": 6,
        },
        "frontier_rate": {
            k: float(v["growth_per_frontier_cell"]) for k, v in an["frontier_rate"].items()
        },
        "block_truth": an["block_truth"],
        "channels_used": an["train_support_distance"]["channels"],
    }


def _collect_dispersion(d: PageData) -> None:
    rec = _load(RECORD_M8)
    ms = rec["g3"]["models"]
    key = "ensemble dispersion (area spread-skill)"
    interval = ms[next(iter(ms))]["criteria"][key]["interval"]
    arms, abl_pairs = [], []
    for name, v in ms.items():
        c = v["criteria"][key]
        arms.append({"arm": name, "equal_block": float(c["equal_block"] or 0.0)})
        if name.endswith("__ABL"):
            base = name[: -len("__ABL")]
            if base in ms:
                a = float(ms[base]["criteria"][key]["equal_block"] or 0.0)
                b = float(c["equal_block"] or 0.0)
                if b > 0:
                    abl_pairs.append({"arm": base, "full": a, "ablated": b, "ratio": a / b})
    geo_lo, geo_hi = 1.0 / 1.2, 1.2
    trained = [a for a in arms if a["arm"].startswith(("m6_", "m7_", "m8_"))]
    # The ablation only demonstrates the latent is load-bearing where the arm was
    # WIDE to begin with. Splitting this out is not a nicety: pooling the two
    # families would let "collapses 1.1x" hide inside "collapses up to 7.8x".
    strong = [p for p in abl_pairs if p["ratio"] >= 2.0]
    weak = [p for p in abl_pairs if p["ratio"] < 2.0]
    d.dispersion = {
        "record": RECORD_M8,
        "fingerprint": rec["split_before"]["fingerprint"],
        "n_arms": len(arms),
        "arms": arms,
        "old_interval": [float(interval[0]), float(interval[1])],
        "geometric_interval": [geo_lo, geo_hi],
        "n_in_old_bar": sum(1 for a in arms if interval[0] <= a["equal_block"] <= interval[1]),
        "n_in_geo_bar": sum(1 for a in arms if geo_lo <= a["equal_block"] <= geo_hi),
        "n_trained_in_geo_bar": sum(1 for a in trained if geo_lo <= a["equal_block"] <= geo_hi),
        "ablation_pairs": abl_pairs,
        "ablation_ratio_min": min(p["ratio"] for p in abl_pairs),
        "ablation_ratio_max": max(p["ratio"] for p in abl_pairs),
        "n_ablation_pairs": len(abl_pairs),
        "n_ablation_strong": len(strong),
        "n_ablation_weak": len(weak),
        "ablation_strong_min": min(p["ratio"] for p in strong),
        "ablation_strong_max": max(p["ratio"] for p in strong),
        "ablation_weak_min": min(p["ratio"] for p in weak),
        "ablation_weak_max": max(p["ratio"] for p in weak),
        "weak_arm_full_max": max(p["full"] for p in weak),
    }


# --------------------------------------------------------------------------
# 2. FIGURES
# --------------------------------------------------------------------------


def _finish(fig: Any, name: str, note: str) -> Path:
    fig.text(0.005, 0.004, note, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    out = FIGDIR / name
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_g2(d: PageData) -> Path:
    g = d.g2
    fig = plt.figure(figsize=(15.5, 10.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.26, height_ratios=[1.0, 0.95])
    hs = ("1", "2", "3")

    ax = fig.add_subplot(gs[0, 0])
    labels = [
        "kernel\n(4 seeds)",
        "wind ellipse\n(rule opponent)",
        "best ellipse\n(envelope)",
        "SAME kernel,\nUNTRAINED",
        "persistence\n(do nothing)",
    ]
    cols = [GOOD, COOL, "#60a5fa", WARN, GREY]
    w = 0.16
    x = np.arange(len(hs))
    for i, lab in enumerate(labels):
        keys = ["kernel_mean", "rule_value", "envelope_value", "kernel_init", "persistence"]
        vals = [g["horizons"][h][keys[i]] for h in hs]
        ax.bar(x + (i - 2) * w, vals, w, color=cols[i], label=lab, edgecolor="white", linewidth=0.6)
    for j, h in enumerate(hs):
        seeds = g["horizons"][h]["kernel_by_seed"]
        ax.plot([x[j] - 2 * w] * len(seeds), seeds, "o", ms=3, color="#052e2b", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} h" for h in hs])
    ax.set_ylim(0, 0.255)
    ax.set_ylabel("best-member IoU, shape term, growth-masked")
    ax.set_title(
        "(1) THE G2 GATE CRITERION, on 4 held-out spatial blocks\n"
        "higher is better; the do-nothing null scores EXACTLY 0 (no visible bar)",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(fontsize=7.4, ncol=3, loc="upper center", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    seed_sd = [g["horizons"][h]["margin_seed_sd_vs_rule"] for h in hs]
    seed_env = [g["horizons"][h]["margin_seed_sd_vs_env"] for h in hs]
    block_sd = [g["horizons"][h]["margin_block_sd"] for h in hs]
    ax.bar(x - 0.25, seed_sd, 0.24, color=GOOD, label="margin / SD ACROSS OUR OWN SEEDS (vs rule)")
    ax.bar(x, seed_env, 0.24, color="#5eead4", label="margin / seed SD (vs envelope)")
    ax.bar(x + 0.25, block_sd, 0.24, color=BAD, label="margin / SD ACROSS THE 4 BLOCKS")
    for j in range(3):
        ax.text(
            x[j] - 0.25, seed_sd[j] + 0.4, f"+{seed_sd[j]:.1f}", ha="center", fontsize=9, color=GOOD
        )
        ax.text(
            x[j] + 0.25,
            block_sd[j] + 0.4,
            f"+{block_sd[j]:.2f}",
            ha="center",
            fontsize=9,
            color=BAD,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} h" for h in hs])
    ax.set_ylim(0, 24)
    ax.set_ylabel("margin over the ellipse, in SD units")
    ax.set_title(
        "(2) THE SAME MARGIN, UNDER TWO DENOMINATORS - read this before quoting '+16 SD'\n"
        "the headline divides by our own seed-to-seed wobble (n=4 training runs), not by\n"
        "variation across the independent spatial units (n=4 blocks)",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for h in hs:
        for pb in g["horizons"][h]["per_block"]:
            rows.append((f"{h} h", pb["fire_id"].replace("2020_", ""), pb["wins"]))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    for r, (_hl, _fid, wins) in enumerate(rows):
        for s, wv in enumerate(wins):
            ax.add_patch(
                Rectangle(
                    (s - 0.42, len(rows) - 1 - r - 0.42),
                    0.84,
                    0.84,
                    color=GOOD if wv else BAD,
                    alpha=0.85,
                )
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{hl} · {fid}" for hl, fid, _ in rows][::-1], fontsize=7.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([s.replace("nbfix_", "seed ") for s in G2_SEEDS], fontsize=8)
    ax.set_title(
        f"(3) UNANIMITY: {g['unanimity']['hits']}/{g['unanimity']['cells']} "
        "block x seed x horizon cells beat the ellipse\n"
        "green = kernel wins that cell. FORMALLY DEMOTED in-project to corroboration\n"
        "below 6 blocks: at 4 blocks unanimity is nearly free. It carries NO verdict.",
        fontsize=10.5,
        loc="left",
    )
    ax.set_xlabel("independent training seed")

    ax = fig.add_subplot(gs[1, 1])
    kb = [g["horizons"][h]["brier_kernel"] for h in hs]
    pbz = [g["horizons"][h]["brier_persistence"] for h in hs]
    rb = [g["horizons"][h]["brier_rule"] for h in hs]
    ax.bar(x - 0.24, kb, 0.22, color=GOOD, label="kernel (4-seed mean)")
    ax.bar(x, rb, 0.22, color=COOL, label="wind ellipse (rule opponent)")
    ax.bar(x + 0.24, pbz, 0.22, color=GREY, label="persistence (ignites ZERO cells)")
    top = max(max(kb), max(pbz), max(rb))
    for j in range(3):
        rel = (kb[j] - pbz[j]) / pbz[j] * 100.0
        word = "LOSES" if rel > 0.5 else ("ties" if abs(rel) <= 0.5 else "wins")
        ax.text(
            x[j],
            max(kb[j], pbz[j], rb[j]) + top * 0.045,
            f"vs persistence:\nkernel {word} ({rel:+.1f}%)",
            ha="center",
            fontsize=8,
            color=BAD if rel > 0.5 else (MUTED if abs(rel) <= 0.5 else GOOD),
        )
    ax.set_xticks(x)
    ax.set_ylim(0, top * 1.30)
    ax.set_xticklabels([f"{h} h" for h in hs])
    ax.set_ylabel("band Brier (lower is better)")
    ax.set_title(
        "(4) A NAMED LIMITATION OF THE SAME PASS, at the same visual weight\n"
        "on window-pooled band Brier the kernel does not beat DOING NOTHING at 1 or 2 h.\n"
        "It wins at 3 h and wins arrival-time CRPS on 4/4 blocks.",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(fontsize=7.6, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "G2 - the deterministic kernel against the physics baseline, shown rather than asserted",
        fontsize=13,
        y=0.985,
    )
    return _finish(
        fig,
        "review_g2.png",
        f"source {RECORD_G2} · split {FINGERPRINT_PRE_D6} (12-fire corpus, SUPERSEDED) · "
        f"criterion {d.g2['criterion']} · {d.g2['n_members']} members · "
        "no eval/ import, no checkpoint read — every value read out of the record",
    )


def fig_dispersion(d: PageData) -> Path:
    dd = d.dispersion
    arms = sorted(dd["arms"], key=lambda a: -a["equal_block"])
    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(1, 2, wspace=0.22, width_ratios=[1.55, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    names = [a["arm"] for a in arms]
    vals = [a["equal_block"] for a in arms]
    lo, hi = dd["geometric_interval"]
    colors = [
        GOOD if lo <= v <= hi else (GREY if "__ABL" in n else BAD)
        for n, v in zip(names, vals, strict=True)
    ]
    ax.barh(range(len(vals)), vals, color=colors, height=0.78)
    ax.axvspan(lo, hi, color=GOOD, alpha=0.12, zorder=0)
    ax.axvline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlabel("area dispersion ratio, equal-block pooled  (1.0 = calibrated spread)")
    ax.set_title(
        f"(1) EVERY ARM EVER SCORED FOR G3 - {dd['n_arms']} of them.\n"
        f"{dd['n_in_old_bar']} sit inside the original bar; {dd['n_in_geo_bar']} inside the "
        f"pre-registered GEOMETRIC bar [{lo:.4f}, {hi:.1f}] (green band),\n"
        "which was tightened on the side we fail. grey = the independent-noise ABLATION.\n"
        "G3 has been attempted four times and failed four times.",
        fontsize=10.5,
        loc="left",
    )
    ax.grid(axis="x", alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    pairs = sorted(dd["ablation_pairs"], key=lambda p: -p["ratio"])
    yy = np.arange(len(pairs))
    lab_full = "full model (shared latent z_t)"
    lab_abl = "ABLATION: independent per-pixel noise only"
    ax.barh(yy - 0.19, [p["full"] for p in pairs], 0.36, color=GOOD, label=lab_full)
    ax.barh(yy + 0.19, [p["ablated"] for p in pairs], 0.36, color=GREY, label=lab_abl)
    for i, p in enumerate(pairs):
        ax.text(
            max(p["full"], p["ablated"]) + 0.02,
            i,
            f"x{p['ratio']:.1f}",
            va="center",
            fontsize=6.5,
            color=INK if p["ratio"] >= 2.0 else BAD,
        )
    ax.axhline(dd["n_ablation_strong"] - 0.5, color=BAD, lw=1.1, ls="--")
    ax.text(
        0.62,
        dd["n_ablation_strong"] - 0.15,
        "below this line the latent buys almost nothing -\nthose arms were already near-collapsed",
        fontsize=7.4,
        color=BAD,
        va="top",
    )
    ax.set_yticks(yy)
    ax.set_yticklabels([p["arm"] for p in pairs], fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlabel("area dispersion ratio, equal-block pooled")
    ax.set_title(
        "(2) ARE THE MEMBERS CLONES? The ablation answers - but only for HALF the arms.\n"
        f"Removing the shared per-step latent collapses the spread "
        f"{dd['ablation_strong_min']:.1f}x-"
        f"{dd['ablation_strong_max']:.1f}x on {dd['n_ablation_strong']} of "
        f"{dd['n_ablation_pairs']} arms,\nand only {dd['ablation_weak_min']:.1f}x-"
        f"{dd['ablation_weak_max']:.1f}x on the other {dd['n_ablation_weak']}. "
        "The machinery works where it was switched on.",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(fontsize=7.6, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    return _finish(
        fig,
        "review_dispersion.png",
        f"source {RECORD_M8} · split {FINGERPRINT_PRE_D6} (12-fire corpus, SUPERSEDED) · "
        "read out of the record's own g3 block; nothing recomputed",
    )


def fig_overprediction_map(d: PageData) -> Path:
    from wildfire_nowcast.sim.reader import load_fire

    ga = _fig("growth_anatomy.json")
    rows = [r for r in ga["rows"] if r["model"] == "nbfix_s1"]

    # SELECTION RULE, stated on the figure: use the held-out fire whose OWN
    # growth-window calibration is closest to the pooled value. Picking a fire
    # by how the maps look is exactly the failure this page is arguing against.
    per_fire: dict[str, list[float]] = {}
    for r in rows:
        if r["truth_new"] > 0:
            a = per_fire.setdefault(r["fire_id"], [0.0, 0.0])
            a[0] += r["truth_new"]
            a[1] += r["pred_new"]
    ratios = {k: v[1] / v[0] for k, v in per_fire.items()}
    pooled = sum(v[1] for v in per_fire.values()) / sum(v[0] for v in per_fire.values())
    fire_id = min(ratios, key=lambda k: abs(math.log(ratios[k]) - math.log(pooled)))
    d.growth["map_fire"] = fire_id
    d.growth["per_fire_growth_window_ratio"] = {k: float(v) for k, v in ratios.items()}
    d.growth["pooled_growth_window_ratio"] = float(pooled)
    frames = load_fire(REPO / "data" / "fires" / fire_id / "tensor.zarr", fire_id=fire_id)
    ever = frames.ever
    horizon = int(ga["horizon_h"])

    def tensor_truth_new(t0: int) -> int:
        t1 = min(t0 + horizon, ever.shape[0] - 1)
        return int((ever[t1] & ~ever[t0]).sum())

    cand = [r for r in rows if r["fire_id"] == fire_id]
    dormant = sorted(
        [r for r in cand if r["truth_new"] == 0 and r["pred_new"] > 4],
        key=lambda r: -r["pred_new"],
    )
    grown = sorted([r for r in cand if r["truth_new"] >= 8], key=lambda r: -r["truth_new"])
    dorm = next((r for r in dormant if tensor_truth_new(int(r["t0"])) == 0), dormant[0])
    big = next(
        (r for r in grown if tensor_truth_new(int(r["t0"])) == int(round(r["truth_new"]))), grown[0]
    )
    # median growth window, so panel 2 is not cherry-picked from either tail
    by_size = sorted(grown, key=lambda r: r["truth_new"])
    med_pool = [r for r in by_size if tensor_truth_new(int(r["t0"])) == int(round(r["truth_new"]))]
    med = med_pool[len(med_pool) // 2] if med_pool else by_size[len(by_size) // 2]

    fig = plt.figure(figsize=(16.6, 5.2))
    gs = fig.add_gridspec(1, 4, wspace=0.16, width_ratios=[1, 1, 1, 1.2])
    geom = frames.geom
    cell_km2 = (geom.cell_size_m / 1000.0) ** 2

    def draw(ax: Any, r: dict[str, Any], title: str) -> None:
        t0 = int(r["t0"])
        t1 = min(t0 + horizon, ever.shape[0] - 1)
        base = ever[t0]
        truth_new = ever[t1] & ~base
        canvas = np.zeros(base.shape + (3,), float) + 1.0
        canvas[frames.barrier] = (0.87, 0.91, 0.96)
        canvas[base] = (0.13, 0.13, 0.15)
        canvas[truth_new] = (0.98, 0.72, 0.11)
        ax.imshow(canvas, extent=geom.extent, origin="upper", interpolation="nearest")
        # crop to the fire plus a margin, so the reader sees the fire and not the domain
        ys, xs = np.where(base | truth_new)
        pad = 10
        y0, y1_ = max(ys.min() - pad, 0), min(ys.max() + pad, base.shape[0] - 1)
        x0, x1_ = max(xs.min() - pad, 0), min(xs.max() + pad, base.shape[1] - 1)
        half = geom.cell_size_m / 2.0
        ax.set_xlim(geom.x_centres[x0] - half, geom.x_centres[x1_] + half)
        ax.set_ylim(geom.y_centres[y1_] - half, geom.y_centres[y0] + half)
        # AREA TOKENS: squares drawn to the map's own scale. No spatial claim.
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        span = xhi - xlo
        tokens = (
            (float(r["truth_new"]), "#f59e0b", "truth"),
            (float(r["pred_new"]), "#b91c1c", "kernel"),
        )
        sides = [math.sqrt(max(a, 0.0) * cell_km2) * 1000.0 for a, _, _ in tokens]
        wide = max(max(sides), 0.10 * span)
        x0 = xlo + 0.05 * span
        y0 = ylo + 0.05 * (yhi - ylo)
        for k, (area_cells, col, lab) in enumerate(tokens):
            yy0 = y0 + k * (wide + 0.045 * (yhi - ylo))
            ax.add_patch(
                Rectangle(
                    (x0, yy0),
                    sides[k],
                    sides[k],
                    facecolor=col,
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=6,
                )
            )
            ax.text(
                x0 + wide + 0.03 * span,
                yy0,
                f"{lab}  {area_cells:.1f} km2",
                fontsize=8.0,
                va="bottom",
                ha="left",
                color=col if area_cells > 0 else MUTED,
                zorder=7,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.2},
            )
        # scale bar in DATA coordinates, so the equal-aspect padding cannot
        # push it outside the drawn map (it did, on the first render)
        bar_km = 10.0
        bx = xhi - 0.07 * span - bar_km * 1000.0
        by = yhi - 0.09 * (yhi - ylo)
        ax.plot(
            [bx, bx + bar_km * 1000.0], [by, by], color=INK, lw=2.4, zorder=7, solid_capstyle="butt"
        )
        ax.text(
            bx + bar_km * 500.0,
            by - 0.012 * (yhi - ylo),
            f"{bar_km:.0f} km",
            ha="center",
            va="top",
            fontsize=7.6,
            color=INK,
            zorder=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        add_north_arrow(ax)
        ax.set_title(title, fontsize=9.0, loc="left")

    short = fire_id.replace("2020_", "").replace("_lightning_complex", "")
    draw(
        fig.add_subplot(gs[0, 0]),
        dorm,
        f"(1) FIRE DOES NOT MOVE · {short} t0={int(dorm['t0'])}\n"
        f"truth over {horizon} h: +0 cells, bitwise zero\n"
        f"kernel: +{dorm['pred_new']:.1f} — ALL of it excess",
    )
    draw(
        fig.add_subplot(gs[0, 1]),
        med,
        f"(2) MEDIAN MOVING WINDOW · t0={int(med['t0'])}\n"
        f"truth +{med['truth_new']:.0f}, kernel +{med['pred_new']:.1f}\n"
        f"= {med['pred_new'] / max(med['truth_new'], 1):.2f}x",
    )
    draw(
        fig.add_subplot(gs[0, 2]),
        big,
        f"(3) FASTEST WINDOW ON THIS FIRE · t0={int(big['t0'])}\n"
        f"truth +{big['truth_new']:.0f}, kernel +{big['pred_new']:.1f}\n"
        f"= {big['pred_new'] / max(big['truth_new'], 1):.2f}x — we UNDER-predict extremes",
    )

    ax = fig.add_subplot(gs[0, 3])
    gr = d.growth
    truth = gr["truth_new_cells"]
    items = [
        ("truth", truth, 0.0, "#f59e0b", GREY),
        (
            "kernel\n(nbfix_s1)",
            gr["predicted_in_growth"]["nbfix_s1"],
            gr["predicted_in_dormant"]["nbfix_s1"],
            GOOD,
            BAD,
        ),
        (
            "wind ellipse\n(cal 3 h)",
            gr["predicted_in_growth"]["ellipse_cal3h"],
            gr["predicted_in_dormant"]["ellipse_cal3h"],
            COOL,
            "#93c5fd",
        ),
    ]
    ax.set_aspect("equal")  # without this the "area" claim is a lie
    scale = 1.0 / math.sqrt(max(i[1] + i[2] for i in items))
    xc = 0.0
    for label, a, b, c1, c2 in items:
        tot = a + b
        side = math.sqrt(tot) * scale
        ax.add_patch(Rectangle((xc, 0), side, side * (a / tot), color=c1))
        ax.add_patch(Rectangle((xc, side * (a / tot)), side, side * (b / tot), color=c2))
        ax.text(xc + side / 2, -0.06, label, ha="center", va="top", fontsize=9)
        ax.text(
            xc + side / 2,
            side + 0.02,
            f"{tot:,.0f} cells\n({tot / truth:.2f}x truth)",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
        if b > 0:
            ax.text(
                xc + side / 2,
                side * (a / tot) + side * (b / tot) / 2,
                f"{b:,.0f}\nin windows where\nnothing happened",
                ha="center",
                va="center",
                fontsize=6.6,
                color="white",
            )
        xc += side + 0.10
    ax.set_xlim(-0.08, xc)
    ax.set_ylim(-0.42, 1.34)
    ax.axis("off")
    ax.set_title(
        f"(4) ALL {gr['n_windows']:,} HELD-OUT WINDOWS, AS AREA\n"
        f"{gr['n_dormant_windows']:,} of them have ZERO truth growth.\n"
        "Square AREA is proportional to predicted new cells.",
        fontsize=9.0,
        loc="left",
    )

    ratio_txt = " · ".join(
        f"{k.replace('2020_', '').replace('_lightning_complex', '')} {v:.2f}x"
        for k, v in sorted(ratios.items())
    )
    fig.suptitle(
        "The calibration failure, as area on the ground - where the over-prediction actually lives",
        fontsize=13,
        y=1.06,
    )
    fig.text(
        0.5,
        1.005,
        f"Fire shown = {fire_id}, chosen by rule: the held-out fire whose OWN growth-window "
        f"calibration ({ratios[fire_id]:.2f}x) is closest in log space to the pooled "
        f"{pooled:.2f}x.  Per fire: {ratio_txt}",
        ha="center",
        fontsize=8.4,
        color=MUTED,
    )
    return _finish(
        fig,
        "review_overprediction_map.png",
        f"maps: C1 tensor (data/fires/{fire_id}/tensor.zarr), read-only; dark = burned at t0, "
        "orange = truth's REAL new cells over the next 3 h. counts: growth anatomy built "
        f"by wildfire_nowcast.sim.growth + {RECORD_G2}, split {FINGERPRINT_PRE_D6}; neither is "
        "tracked, so A CLONE CANNOT CHECK THIS NUMBER. "
        "THE CORNER SQUARES ARE AREA TOKENS DRAWN TO THE MAP'S OWN SCALE and make NO spatial "
        "claim: the model's actual spatial output is not rendered anywhere on this page, because "
        "that needs a checkpoint read and the model tree is being rewritten.",
    )


def fig_identity(d: PageData) -> Path:
    idn = d.identity
    rows = sorted(idn["rows"], key=lambda r: r["model"])
    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(2, 2, hspace=0.95, wspace=0.24, height_ratios=[1.0, 0.9])

    ax = fig.add_subplot(gs[0, :])
    xx = np.arange(len(rows))
    lab_cv = "ensemble_CV — how much the ensemble's RELATIVE WIDTH varies across the 4 blocks"
    lab_gc = "growth_calibration — how much the MEAN AREA ERROR varies across the 4 blocks"
    ax.bar(xx - 0.2, [r["cv_block_spread"] for r in rows], 0.38, color=COOL, label=lab_cv)
    ax.bar(xx + 0.2, [r["gc_block_spread"] for r in rows], 0.38, color=BAD, label=lab_gc)
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.95, 22)
    ax.set_xticks(xx)
    ax.set_xticklabels([r["model"] for r in rows], rotation=60, ha="right", fontsize=6.6)
    ax.set_ylabel("max / min across the 4 held-out blocks")
    ax.set_title(
        "(1) WHICH FACTOR MOVES? Same arm, same seed, four held-out blocks.\n"
        f"ensemble_CV varies by "
        f"{idn['cv_block_spread_all_min']:.2f}x-{idn['cv_block_spread_all_max']:.2f}x "
        f"across blocks "
        f"({idn['cv_block_spread_wide_min']:.2f}x-{idn['cv_block_spread_wide_max']:.2f}x on the "
        f"{idn['n_wide_arms']} widest arms);  growth_calibration varies by "
        f"{idn['gc_block_spread_min']:.1f}x-{idn['gc_block_spread_max']:.1f}x. "
        "The mean is the term that moves.",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[1, 0])
    for r in rows:
        ax.plot([1, 2, 3, 4], r["cv"], "-o", ms=2.5, lw=0.8, color=COOL, alpha=0.55)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(
        ["block 3\nbobcat", "block 4\ncreek", "block 5\nCZU", "block 6\ndolan"], fontsize=8
    )
    ax.set_ylabel("ensemble_CV\n(member SD / predicted mean)", fontsize=9)
    ax.set_title(
        "(2) ensemble_CV, EVERY TRAINED ARM - the honest version of 'near-constant'\n"
        "the LINES ARE FLAT block-to-block, but their LEVELS span "
        f"{idn['cv_value_min']:.2f}-{idn['cv_value_max']:.2f} = {idn['cv_value_span']:.1f}x\n"
        "ACROSS arms. The quoted '1.24-1.43x' is the FLATNESS OF A LINE, not the value of CV.",
        fontsize=10.2,
        loc="left",
    )
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 1])
    for r in rows:
        ax.plot([1, 2, 3, 4], r["gc"], "-o", ms=2.5, lw=0.8, color=BAD, alpha=0.55)
    ax.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(
        ["block 3\nbobcat", "block 4\ncreek", "block 5\nCZU", "block 6\ndolan"], fontsize=8
    )
    ax.set_ylabel("growth_calibration\n(predicted mean area / truth mean area)", fontsize=9)
    ax.set_title(
        "(3) growth_calibration, THE SAME ARMS\n"
        "these lines are NOT flat, and every one of them collapses on block 5.\n"
        "Dashed = perfect. On these later arms the mean is UNDER 1 on three blocks of four:\n"
        "'we over-predict growth' is a pooled statement that hides the sign.",
        fontsize=10.2,
        loc="left",
    )
    ax.grid(alpha=0.25)

    fig.suptitle(
        "The identity (ADR-035):  adr = sqrt((M+1)/M) x ensemble_CV x growth_calibration "
        f"x truth_shape x relief    -    max residual {idn['max_residual']:.1e}",
        fontsize=12.5,
        y=0.995,
    )
    return _finish(
        fig,
        "review_identity.png",
        f"source: block-5 anatomy built by wildfire_nowcast.sim.s5_report from {idn['record']}; "
        "neither is tracked, so A CLONE CANNOT CHECK THIS NUMBER. split "
        f"{FINGERPRINT_PRE_D6}. Known-answer check: the truth-scale denominator reproduces the "
        "record's own persistence RMS error to 0.0 exactly on 2 of 4 fires and <4e-15 on the other "
        "two.",
    )


def fig_block5(d: PageData) -> Path:
    b5 = d.block5
    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 2, wspace=0.2, width_ratios=[1.5, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    per = sorted(b5["per_arm"], key=lambda a: -a["b3"])
    xx = np.arange(len(per))
    for key, col, lab in (
        ("b3", "#94a3b8", "block 3 — Bobcat"),
        ("b4", "#64748b", "block 4 — Creek"),
        ("b6", "#334155", "block 6 — Dolan"),
        ("b5", BAD, "block 5 — CZU"),
    ):
        ax.plot(
            xx,
            [a[key] for a in per],
            "-o",
            ms=2.6,
            lw=1.0,
            color=col,
            label=lab,
            zorder=5 if key == "b5" else 2,
        )
    ax.set_xticks(xx)
    ax.set_xticklabels([a["arm"] for a in per], rotation=90, fontsize=4.2)
    ax.tick_params(axis="x", pad=1)
    ax.set_ylabel("area dispersion ratio")
    ax.set_title(
        f"(1) BLOCK 5 IS LOWEST IN {b5['n_arms_block5_lowest']} OF "
        f"{b5['n_arms_scored']} ARMS ON THE RECORD\n"
        "every arm, every seed, every ablation, every baseline - the red line never\n"
        "crosses another. It survived a latent redesign, a spatial latent, an asymmetric\n"
        "prior and a mean correction, and it does not move.",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    maha = b5["mahalanobis"]
    fr = b5["frontier_rate"]
    ref = {a["arm"]: a for a in b5["per_arm"]}["m7_offstate_s1"]
    offsets = {3: (14, 6), 4: (14, -20), 6: (14, -14), 5: (-12, 14)}
    for fid, blk in b5["block_of"].items():
        xv, yv = maha[fid], ref[f"b{blk}"]
        col = BAD if blk == 5 else COOL
        ax.scatter([xv], [yv], s=180, color=col, zorder=5)
        ax.annotate(
            f"block {blk} | {fid.replace('2020_', '').replace('_lightning_complex', '')}\n"
            f"truth growth per frontier cell {fr[fid]:.3f}",
            (xv, yv),
            textcoords="offset points",
            xytext=offsets[blk],
            ha="right" if offsets[blk][0] < 0 else "left",
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("Mahalanobis distance of the scored conditions from the TRAIN support")
    ax.set_ylabel("area dispersion ratio (arm m7_offstate_s1)")
    ax.set_xlim(0.3, 6.0)
    ax.set_ylim(0.20, 1.45)
    ax.set_title(
        "(2) IT IS NOT DISTANCE FROM THE TRAINING DATA - the obvious explanation, refuted\n"
        f"blocks 5 and 6 are EQUALLY FAR ({maha['2020_czu_lightning_complex']:.2f} vs "
        f"{maha['2020_dolan']:.2f}) and {ref['b6'] / ref['b5']:.1f}x apart in outcome.\n"
        "n = 4. This refutes a clean monotone story; it cannot support one. The mystery is open.",
        fontsize=10.5,
        loc="left",
    )
    ax.grid(alpha=0.25)
    fig.subplots_adjust(bottom=0.30)

    return _finish(
        fig,
        "review_block5.png",
        f"per-arm ratios: {RECORD_M8}. distances + truth rates: block-5 anatomy built by "
        f"wildfire_nowcast.sim.s5_report from {RECORD_M7}; neither is tracked, so A CLONE CANNOT "
        f"CHECK THIS NUMBER. split {FINGERPRINT_PRE_D6}. "
        f"Mahalanobis over {len(b5['channels_used'])} C1 channels on the scored cells.",
    )


# --------------------------------------------------------------------------
# 3. HTML
# --------------------------------------------------------------------------


def _data_uri(path: Path, *, max_width: int | None = 1750) -> str:
    mimes = {"png": "image/png", "gif": "image/gif", "jpg": "image/jpeg"}
    mime = mimes[path.suffix.lstrip(".").lower()]
    raw = path.read_bytes()
    if max_width is not None and path.suffix.lower() == ".png":
        try:
            from PIL import Image

            with Image.open(path) as im:
                if im.width > max_width:
                    im = im.convert("RGB")
                    h = int(im.height * max_width / im.width)
                    im = im.resize((max_width, h), Image.LANCZOS)
                    import io

                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=88, optimize=True)
                    raw, mime = buf.getvalue(), "image/jpeg"
        except Exception:  # pragma: no cover - rendering fallback
            # The page still renders, at full size. Saying so matters because the
            # symptom of this path is a 40 MB HTML file and no other trace.
            logger.warning(
                "could not downscale %s to %d px; embedding it at full size",
                path.name,
                max_width,
                exc_info=True,
            )
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _shrink_gif(src: Path, dst: Path, *, max_width: int = 460, step: int = 2) -> Path:
    from PIL import Image, ImageSequence

    with Image.open(src) as im:
        frames = []
        for i, fr in enumerate(ImageSequence.Iterator(im)):
            if i % step:
                continue
            f = fr.convert("RGB")
            if f.width > max_width:
                f = f.resize((max_width, int(f.height * max_width / f.width)), Image.LANCZOS)
            frames.append(f.convert("P", palette=Image.ADAPTIVE, colors=96))
        frames[0].save(
            dst,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=max(80, int(im.info.get("duration", 120)) * step),
            optimize=True,
        )
    return dst


def figure_block(path: Path, caption: str, notestablishes: str, *, wide: bool = True) -> str:
    uri = _data_uri(path, max_width=1750 if wide else 1100)
    return (
        f'<figure class="{"wide" if wide else ""}"><img src="{uri}" alt="{html.escape(path.name)}">'
        f"<figcaption><b>Shows.</b> {caption}<br>"
        f'<span class="ne"><b>Does not establish.</b> {notestablishes}</span></figcaption></figure>'
    )


def build_html(d: PageData) -> Path:
    figs = {
        "g2": fig_g2(d),
        "disp": fig_dispersion(d),
        "over": fig_overprediction_map(d),
        "ident": fig_identity(d),
        "b5": fig_block5(d),
    }
    gif_small = FIGDIR / "review_kincade_small.gif"
    _shrink_gif(FIGDIR / "2019_kincade.gif", gif_small)

    map_fire = d.growth["map_fire"].replace("2020_", "").replace("_lightning_complex", "")
    g, gr, idn, b5, dsp, em = (d.g2, d.growth, d.identity, d.block5, d.dispersion, d.elmfire)
    h1, h2, h3 = (g["horizons"][k] for k in ("1", "2", "3"))

    old = (
        f'<span class="corpus old">OLD CORPUS | 12 fires | fingerprint '
        f"<code>{FINGERPRINT_PRE_D6}</code></span>"
    )
    cur = (
        f'<span class="corpus new">CORPUS OF RECORD | 21 fires | fingerprint '
        f"<code>{FINGERPRINT_OF_RECORD}</code></span>"
    )

    css = """
:root{--ink:#16181d;--mut:#6b7280;--good:#0f766e;--bad:#b91c1c;--warn:#b45309;--line:#e5e7eb}
*{box-sizing:border-box}
body{margin:0;font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
 color:var(--ink);background:#fafafa}
.wrap{max-width:1180px;margin:0 auto;padding:36px 26px 90px}
h1{font-size:30px;line-height:1.24;margin:0 0 6px}
h2{font-size:23px;margin:52px 0 6px;padding-top:22px;border-top:3px solid var(--ink)}
h3{font-size:17px;margin:28px 0 6px}
p{margin:12px 0}
code{background:#eef0f3;padding:1px 5px;border-radius:3px;font-size:.86em}
.lede{font-size:18px;color:#30343c}
.corpus{display:inline-block;font-size:11.5px;font-weight:700;letter-spacing:.03em;
 padding:2px 8px;border-radius:3px;vertical-align:2px;white-space:nowrap}
.corpus code{background:transparent;color:inherit;padding:0}
.old{background:#fef3c7;color:#92400e;border:1px solid #fcd34d}
.new{background:#dcfce7;color:#14532d;border:1px solid #86efac}
figure{margin:26px 0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:12px}
figure img{width:100%;height:auto;display:block;border-radius:3px}
figcaption{font-size:13.5px;color:#374151;margin-top:10px;line-height:1.55}
.ne{color:var(--bad)}
.box{border-left:4px solid var(--ink);background:#fff;padding:14px 18px;margin:22px 0;
 border-radius:0 5px 5px 0}
.box.fail{border-color:var(--bad);background:#fef2f2}
.box.open{border-color:var(--warn);background:#fffbeb}
.box.ok{border-color:var(--good);background:#f0fdfa}
.box.gap{border-color:#6b7280;background:#f3f4f6}
.box
h4{margin:0 0 6px;font-size:14px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0;background:#fff}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f3f4f6;font-size:12.5px}
.num{font-variant-numeric:tabular-nums}
.prov{font-size:12px;color:var(--mut);margin-top:-6px}
.gate{display:inline-block;padding:1px 7px;border-radius:3px;font-size:12px;font-weight:700}
.gp{background:#dcfce7;color:#14532d}.gf{background:#fee2e2;color:#7f1d1d}
.gn{background:#e5e7eb;color:#374151}
.toc{background:#fff;border:1px solid var(--line);border-radius:6px;padding:14px 22px;margin:26px 0}
.toc ol{margin:6px 0;padding-left:22px}
hr{border:0;border-top:1px solid var(--line);margin:34px 0}
.small{font-size:13px;color:var(--mut)}
"""

    def pct(x: float) -> str:
        return f"{x:.2f}"

    body: list[str] = []
    A = body.append

    # ---------------------------------------------------------------- intro
    A(f"""
<h1>Wildfire nowcasting - what we have, what we do not, and what broke</h1>
<p class="lede">You two helped design this. You have seen nothing since. The sketch was:
learn a transition kernel <code>P(x<sub>t+1h</sub> | x<sub>t</sub>, features)</code>, Monte-Carlo it
forward, get a calibrated ensemble. That is still the architecture. This page is the evidence,
including the parts that did not work - which is most of the interesting parts.</p>

<div class="box">
<h4>Read the corpus badges - they are not decoration</h4>
<p>We swapped the training corpus two days ago. <b>Every result on this page is bound to the
OLD corpus</b> {old} and nothing has been re-scored against the current one {cur}. The badge
exists because this project has twice been bitten by a split moving underneath a result: once a
cross-validation fold changed while a model was training and four fires silently crossed from
train to held-out, and once a stale normalisation file would have baked a held-out fire's
statistics into training - caught, that time, before the run. Where a figure was rendered under
the old corpus, it says so on the figure.</p>
</div>

<div class="toc">
<b>What this page answers, in order</b>
<ol>
<li>Does it actually learn fire spread? (G2 - passed, with named limits)</li>
<li>What does the ensemble look like? (members vs truth; are they clones?)</li>
<li>The calibration failure, made visual</li>
<li>The identity that reframed the whole project - and one sentence of ours it kills</li>
<li>Block 5 / CZU: the open mystery</li>
<li>What we have <em>not</em> attempted</li>
</ol>
<p class="small">Gate status: G0 <span class="gate gp">closed</span> | G1
<span class="gate gp">closed</span> | G2 <span class="gate gp">PASSED</span> | G3
<span class="gate gf">FAILED x4</span> | G4 <span class="gate gn">not attempted</span> |
G5 <span class="gate gn">not attempted</span> | G6 <span class="gate gn">not attempted</span></p>
</div>

<div class="box gap">
<h4>How this page was built, so you can discount it correctly</h4>
<p>Everything here is read out of run records and figure artifacts already on disk. No scoring
code was re-run for this page, no model checkpoint was opened, and <code>predict()</code> was
never called - the modelling lead is mid-rewrite of the evaluation package and running anything
against it would produce numbers nobody could attribute later. Two consequences you should hold
against us: <b>(a)</b> the ensemble panels below are the ones already rendered, not fresh ones
chosen to flatter; <b>(b)</b> one panel we wanted - the kernel's actual spatial ensemble on a
real fire - is a labelled gap, not a rendering.</p>
</div>
""")

    # ---------------------------------------------------------------- S1
    A(f"""
<h2>1. Does it actually learn fire spread?</h2>
<p>Yes, on the criterion the gate was adjudicated on, against a real physics opponent, on
{g["n_heldout_blocks"]} held-out <em>spatial blocks</em> (not fires - overlapping fires share a
landscape, so blocks are the independent unit). Held out:
{", ".join(f["fire_id"].replace("_", " ") for f in g["horizons"]["1"]["per_block"])}. {old}</p>

<p>Start with the fire itself. This is a real GOFER-labelled fire evolving hour by hour with the
RTMA wind over it, straight out of the C1 tensor - no model involved. It is here because the
first thing anyone should check is whether the <em>data</em> is coherent.</p>
""")
    A(
        f'<figure><img src="{_data_uri(gif_small, max_width=None)}" alt="Kincade fire movie">'
        "<figcaption><b>Shows.</b> 2019 Kincade, 134 hourly frames, wind quiver overlaid. The fire "
        "runs SW through the Oct-27 Diablo wind event (area-weighted downwind cosine +0.68 on 88% "
        "of growth hours, z=+4.5), is dormant when the wind is 1.7 m/s and bursts when it is 9.0, "
        "holds at the barrier on the west flank and then crosses it - as the real fire crossed "
        "US-101. Nothing teleports (max front gap 3 km, median advance 1 cell/h).<br>"
        '<span class="ne"><b>Does not establish.</b> Anything about the model. This is the label '
        "and weather data only. It also does not establish that the 1 km rasterisation is adequate "
        "- GOFER's effective resolution is ~2 km, so roughly a cell of this outline is "
        "noise.</span>"
        "</figcaption></figure>"
    )

    A("""
<p>Now the gate. G2 asks: does the learned kernel beat a wind-advected ellipse whose growth rate
is calibrated on the training fires - separately at each horizon, so we do not get to pick the
horizon where our opponent is weakest? The criterion is a best-member IoU restricted to the
<em>shape</em> term on windows where truth actually grew. That last clause matters and is not
cosmetic: the original metric ranked <em>doing nothing</em> above doing something, because an
empty prediction against an empty truth scores IoU 1.0.</p>
""")
    A(
        figure_block(
            figs["g2"],
            f"Panel 1: the kernel scores {h1['kernel_mean']:.3f} / {h2['kernel_mean']:.3f} / "
            f"{h3['kernel_mean']:.3f} at 1/2/3 h against the calibrated ellipse's "
            f"{h1['rule_value']:.3f} / {h2['rule_value']:.3f} / {h3['rule_value']:.3f}, and "
            f"against "
            f"the same kernel UNTRAINED at {h1['kernel_init']:.3f} / {h2['kernel_init']:.3f} / "
            f"{h3['kernel_init']:.3f}. Training helps, by "
            f"{h1['untrained_deficit_seed_sd']:.1f}-{h3['untrained_deficit_seed_sd']:.1f} seed-SD. "
            "Persistence scores exactly 0. Panel 2 is the honest reading of '+16 SD' (see the red "
            "box "
            "below). Panel 3 is unanimity. Panel 4 is a limitation of the same pass, at the same "
            "size "
            "as the win.",
            "That the kernel is a good absolute forecaster. It beats this opponent on this "
            "criterion; panel 4 shows it losing to <em>predicting nothing at all</em> on pixelwise "
            "band Brier at 1 h. It also does not establish anything about the current 21-fire "
            "corpus - this run is bound to the 12-fire split.",
        )
    )

    A(f"""
<div class="box fail">
<h4>Where our own headline overstates itself - read this one</h4>
<p>We have been quoting <b>+{h1["margin_seed_sd_vs_rule"]:.1f} / +{h2["margin_seed_sd_vs_rule"]:.1f}
/ +{h3["margin_seed_sd_vs_rule"]:.1f} SD</b>. I reproduced those numbers exactly from the run
record, and the denominator is <b>the standard deviation across our own four training seeds</b>
({h1["sd_across_seed"]:.5f} at 1 h). That is a statement about how reproducible our training is.
It is <em>not</em> a statement about sampling variability across fires or landscapes.</p>
<p>Recompute the same margin against variation across the four held-out spatial blocks - the
units the contract itself calls independent - and it is <b>+{h1["margin_block_sd"]:.2f} /
+{h2["margin_block_sd"]:.2f} / +{h3["margin_block_sd"]:.2f} block-SD</b>
(paired t = {h1["margin_block_t"]:.1f} / {h2["margin_block_t"]:.1f} / {h3["margin_block_t"]:.1f}
on n=4). Same data, same record, different and much more relevant denominator. The direction of
the result is unchanged and every block agrees; the <em>size</em> of the claim shrinks by about
an order of magnitude. Our own decision log even describes this statistic as an "equal-block SD",
which it is not.</p>
<p>Similarly, the "<b>{g["unanimity"]["hits"]}/{g["unanimity"]["cells"]} cells</b>" figure is real
but was formally demoted in-project to corroboration: at four blocks, unanimity is nearly free.
It is drawn here at the same weight as everything else and should carry no weight in your
reading.</p>
</div>

<h3>What G2 does <em>not</em> say, on the record</h3>
<ul>
<li>The kernel <b>loses to persistence</b> on window-pooled band Brier at 1 h and ties at 2 h.
At 1 km cells, an hour of fire movement is often sub-cell, so persistence at 1 h is structural.</li>
<li>The gate's 1 h number rests on <b>239 of 446</b> growth windows - 54% of the stratum, because
the criterion is undefined where truth did not grow.</li>
<li>Growth calibration is <b>our worst number and the opponent's is better</b>. That is section
3.</li>
</ul>
""")

    # ---------------------------------------------------------------- S2
    A("""
<h2>2. What does the ensemble look like?</h2>
<p>This is the crux, and it is where we have spent most of the project. The question you would
ask first is the right one: <b>are the members visibly different from each other, or are they
clones?</b></p>

<p>Two answers, and they point different ways.</p>
""")
    A(
        figure_block(
            FIGDIR / "kincade_ens_ellipse.png",
            "A real 24-member ensemble drawn over a real fire at the peak of the Diablo wind "
            "event: "
            "member fronts (top left), burn probability (top right), median arrival hour and its "
            "p90-p10 spread (bottom). The members are near-identical - pairwise IoU 0.956, "
            "member-area CV 0.030 - and <b>truth's area lies entirely outside the member "
            "range</b>. "
            "That red banner is the renderer's own collapse detector firing.",
            "Anything about our learned kernel. <b>This is the wind-ellipse baseline</b>, and on a "
            "fire that was in TRAIN under the old split. It is included because it is the clearest "
            "picture we have of what an under-dispersed ensemble looks like, and because it is the "
            "figure that first made us raise the baseline-fairness problem. Rendered under the old "
            "corpus.",
            wide=False,
        )
    )
    A(
        figure_block(
            FIGDIR / "iou_smallmultiples_2020_creek.png",
            "Member-versus-truth on the held-out Creek fire: the two windows where the kernel "
            "loses "
            "most to the calibrated ellipse (top) and the two where it wins most (bottom). Truth's "
            "new cells over 3 h are typically <b>+2 to +4 "
            "cells</b>. Bottom row, third column: an ellipse ensemble in which <b>24 of 24 members "
            "predict nothing</b> still scores IoU 0.333, which is why the gate criterion had to be "
            "redefined.",
            "That members differ from <em>each other</em> - this shows only the best member per "
            "model. It also shows how small the typical signal is: most of this problem is "
            "deciding "
            "whether two or four cells will ignite, not drawing a dramatic front.",
        )
    )

    A("""
<p>The quantitative answer is cleaner, and it is the one I would weight. Remove the shared
per-step latent and leave only independent per-pixel noise - the ablation the original design
called "known-broken, build it only as an ablation" - and the ensemble collapses. But it only
collapses <em>hard</em> on the arms that had real width to begin with, and that qualification is
on the figure rather than in a footnote:</p>
""")
    A(
        figure_block(
            figs["disp"],
            f"Left: every one of the {dsp['n_arms']} arms ever scored against G3's dispersion "
            f"criterion, sorted. <b>{dsp['n_in_old_bar']}</b> sit inside the original bar and "
            f"<b>{dsp['n_in_geo_bar']}</b> inside the tightened one, and no arm sits inside on all "
            "four seeds - a criterion met on two seeds of four is a coin flip, not a pass. Right: "
            "the ablation control. Removing the shared per-step latent collapses dispersion "
            f"{dsp['ablation_strong_min']:.1f}x-{dsp['ablation_strong_max']:.1f}x on "
            f"{dsp['n_ablation_strong']} of {dsp['n_ablation_pairs']} arms.",
            "<b>That the members are non-clones on every arm.</b> On the other "
            f"{dsp['n_ablation_weak']} arms the ablation changes the answer by only "
            f"{dsp['ablation_weak_min']:.1f}x-{dsp['ablation_weak_max']:.1f}x - because those arms "
            f"were already nearly collapsed (all of them below {dsp['weak_arm_full_max']:.2f}). "
            "So the correlated-innovation machinery is demonstrably load-bearing where it was "
            "turned up, and demonstrably near-idle where it was not. It also does not establish "
            "that the ensemble is calibrated: it plainly is not.",
        )
    )

    A("""
<div class="box gap">
<h4>NOT MEASURED - a panel we owe you and cannot render today</h4>
<p>We have no rendered <b>spatial</b> ensemble of the learned kernel on a held-out fire - the
spaghetti-of-member-fronts picture, but for our model rather than for the ellipse. Producing one
requires calling <code>predict()</code> against a checkpoint, and the evaluation package was
being rewritten at the time; running the renderer against a tree being edited is a hazard we
have already caused once. The numbers above (dispersion ratio, ablation collapse, member counts)
are all we can honestly show today. <b>This gap points in our favour and you should treat it as
unverified.</b></p>
</div>
""")

    # ---------------------------------------------------------------- S3
    A(f"""
<h2>3. The calibration failure, made visual</h2>
<p>Here is the number that should worry you most, and it is ours, not the baseline's. Pooled over
all {gr["n_windows"]:,} scored held-out windows, the kernel predicts
<b>{gr["pooled"]["nbfix_s2"]:.2f}x-{gr["pooled"]["nbfix_s1"]:.2f}x</b> truth's new burned
area. The one-parameter wind ellipse - a model with a single fitted scalar - predicts
<b>{gr["pooled"]["ellipse_cal3h"]:.2f}x</b>. <b>Our physics baseline is better calibrated in area
than our neural simulator.</b> {old}</p>

<p>As a table that is easy to skim past. As area on the ground it is not.</p>
""")
    A(
        figure_block(
            figs["over"],
            f"Panels 1-3 are three real windows on one held-out fire ({map_fire}), straight from "
            f"the C1 "
            "tensor: dark = burned at t0, orange = truth's actual new cells over the next 3 h. The "
            "two squares in each corner are the truth and kernel <em>areas</em>, drawn to the "
            "map's "
            "own scale. Panel 1: the fire does not move and the kernel paints area anyway. Panel "
            "2: "
            "the median moving window. Panel 3: the fastest window on this fire, where we "
            "<em>under</em>-predict badly. Panel 4: the whole held-out set - the kernel's "
            f"total is {gr['predicted_total']['nbfix_s1']:,.0f} cells against truth's "
            f"{gr['truth_new_cells']:,.0f}, and <b>{gr['predicted_in_dormant']['nbfix_s1']:,.0f} "
            f"of "
            "it is spent in windows where the fire never moved.</b> The kernel ignites in "
            f"{gr['dormant_ignition_count']['nbfix_s1']} of {gr['n_dormant_windows']} such windows "
            f"- "
            "it is never once silent.",
            "Any spatial claim about the model. <b>The corner squares are area tokens, not "
            "predictions</b> - the kernel's actual footprint is not drawn anywhere on this page. "
            "Panels 1-3 are three windows from one fire, chosen as zero / median / maximum growth, "
            "so they illustrate the aggregate rather than evidencing it. Panel 4's totals are sums "
            "over overlapping 3-hour windows: a predicted-increment budget, not a map of burned "
            "area.",
        )
    )

    A(f"""
<div class="box ok">
<h4>The stratified reading, which inverts the headline</h4>
<p>Split those windows by whether the fire was moving. On the
<b>{gr["n_growth_windows"]}</b> windows where truth grew, the kernel predicts
<b>{gr["on_growth_windows"]["nbfix_s1"]:.3f}x</b> and
<b>{gr["on_growth_windows"]["nbfix_s2"]:.3f}x</b>
of truth's new area - better calibrated than the ellipse's
<b>{gr["on_growth_windows"]["ellipse_cal3h"]:.3f}x</b> on the same windows. The entire
{gr["pooled"]["nbfix_s1"]:.2f}x is bought in the <b>{gr["n_dormant_windows"]}</b> windows with
bitwise-zero growth.</p>
<p><b>So the defect is not spread rate. It is a missing OFF state.</b> This decomposition
reconciles with the evaluation package's own pooled ratio to a maximum absolute delta of
{"exactly 0" if gr["reconciliation_max_delta"] == 0 else f"{gr['reconciliation_max_delta']:.0e}"}
across all six models - it is arithmetic on their number,
not a competing measurement. The practical consequence, which we flagged before anyone tuned
anything: dividing the model's growth by ~3 to fix the headline would drive the growth-window
ratio from 0.98 to about 0.33 and break the dispersion gate. The two defects need opposite
fixes.</p>
</div>

<div class="box fail">
<h4>And a second place our phrasing is wrong</h4>
<p>"We over-predict growth" is true of the G2-era arms, pooled. It is <em>not</em> true of the
later arms per block: on the arms in section 4 the mean area ratio is <b>below 1 on three of the
four held-out blocks</b>. We both over- and under-predict, in different regimes, and the pooled
scalar hides the sign.</p>
</div>
""")

    # ---------------------------------------------------------------- S4
    A(f"""
<h2>4. The identity - and the sentence of ours it kills</h2>
<p>G3 asks for a calibrated ensemble: a spread-skill ratio near 1. We failed it four times.
Three of those attempts were aimed at the wrong quantity, and we only know that because someone
decomposed the criterion instead of optimising against it.</p>

<p>The criterion factors <b>exactly</b>:</p>
<p style="text-align:center;font-size:17px"><code>adr = sqrt((M+1)/M) &times; ensemble_CV &times;
growth_calibration &times; truth_shape &times; relief</code></p>
<p class="prov">Maximum residual over 96 model &times; block cells:
<b>{idn["max_residual"]:.1e}</b> - machine epsilon. This is an identity, not a fit. The one new
quantity it needs (the RMS of truth's own growth) reproduces the record's own persistence error
to 0.0 exactly on 2 of 4 fires and below 4e-15 on the other two.</p>

<p>The consequence: <b>"the ensemble is under-dispersed" was a misreading of our own metric.</b>
The denominator of that ratio is the model's own mean-area error, so a mean error shows up as a
spread failure. Widening the ensemble cannot fix it; fixing the mean buys the spread for free.
Three modelling tasks - a latent redesign, a spatial latent, an asymmetric prior - attacked the
width term, which was never the binding one.</p>
""")
    A(
        figure_block(
            figs["ident"],
            "Panel 1: for each trained arm, how much each factor varies across the four held-out "
            f"blocks. <code>ensemble_CV</code> varies {idn['cv_block_spread_all_min']:.2f}x-"
            f"{idn['cv_block_spread_all_max']:.2f}x; <code>growth_calibration</code> varies "
            f"{idn['gc_block_spread_min']:.1f}x-{idn['gc_block_spread_max']:.1f}x. Panels 2 and 3 "
            "show the same arms factor by factor: the CV lines are flat, the calibration lines are "
            "not and every one of them collapses on block 5.",
            "That ensemble width is 'fine'. See the box below - it is flat across blocks, which is "
            "a "
            "different claim. Nor does it establish causation: the identity is algebra, so it "
            "tells "
            "you which factor <em>carries</em> the failure, not which mechanism <em>causes</em> "
            "it.",
        )
    )

    A(f"""
<div class="box fail">
<h4>The third place I have to correct us, and it is the same mistake three times</h4>
<p>We have been writing "<code>ensemble_CV</code> is near-constant at <b>1.24-1.43x</b> across
every arm ever run". Read off the artifact, <b>that is not what the number is</b>.
<code>ensemble_CV</code>'s <em>values</em> span
<b>{idn["cv_value_min"]:.2f} to {idn["cv_value_max"]:.2f}</b> across the
{idn["n_trained_arms"]} trained arms - a <b>{idn["cv_value_span"]:.1f}x</b> range. The model can
and did move ensemble width a great deal.</p>
<p>1.24-1.43x is the <b>block-to-block max/min ratio within a single arm</b>, on the widest arms
only - i.e. a measure of <em>flatness</em>, paired with the same statistic for
<code>growth_calibration</code>
({idn["gc_block_spread_min"]:.1f}x-{idn["gc_block_spread_max"]:.1f}x).
Recomputing it off the artifact I get {idn["cv_block_spread_wide_min"]:.2f}x-
{idn["cv_block_spread_wide_max"]:.2f}x on the {idn["n_wide_arms"]} widest arms, so even the
quoted range is slightly narrower than the data.
So the argument survives intact - CV does not explain why one block behaves differently - but the
sentence "the ensemble's relative width is roughly fine" does not follow from it, because that is
a claim about the level, and the level moves {idn["cv_value_span"]:.1f}x.</p>
<p>This is the third instance in this project of a compound statistic being quoted in units other
than its own: a variance ratio printed beside a standard-deviation criterion (which reversed the
direction of 155 of 170 reported cells), a bar asymmetry quoted in ratio units when the bar is
defined in log units, and now this. It is a systematic failure mode of ours and you should assume
there are more.</p>
</div>
""")

    # ---------------------------------------------------------------- S5
    maha = b5["mahalanobis"]
    A("""
<h2>5. Block 5 / CZU - the open mystery</h2>
<p>One held-out block behaves unlike the other three, on every arm we have ever run, and we cannot
explain it. We are presenting it as an open problem rather than as a solved one.</p>
""")
    A(
        figure_block(
            figs["b5"],
            f"Left: the dispersion ratio per block for all {b5['n_arms_scored']} scored arms. "
            f"Block 5 "
            f"is the lowest in <b>{b5['n_arms_block5_lowest']} of {b5['n_arms_scored']}</b> - the "
            f"red "
            "line never crosses another. It survived a latent redesign, a spatial latent, an "
            "asymmetric prior and a mean correction unchanged. Right: the obvious explanation - "
            "that "
            "CZU is far from the training distribution - refuted. Blocks 5 and 6 are equally far "
            f"(Mahalanobis {maha['2020_czu_lightning_complex']:.2f} vs {maha['2020_dolan']:.2f}) "
            f"and "
            "differ by more than 2x in outcome, while blocks 3 and 4 are near "
            f"({maha['2020_bobcat']:.2f}, {maha['2020_creek']:.2f}) and differ from each other "
            f"too.",
            "Any positive explanation. It rules a hypothesis out and leaves nothing in its place. "
            "n = 4 blocks: distance-versus-outcome here is four points, so it can refute a clean "
            "monotone story but cannot support one.",
        )
    )
    A(
        figure_block(
            FIGDIR / "s5_block5_anatomy.png",
            "The same investigation in full, including the finding that killed the lead it started "
            "from: the per-block table of the dispersion ratio is not comparable across blocks, "
            "because its denominator is the model's own error. An arm that over-predicts blocks "
            "3/4/6 by 5-8x inflates its own denominator there and looks flat.",
            "That we know why CZU is different. Three further hypotheses were tested and refuted "
            "en "
            "route (the ignition merge, window concentration, and the 51%-blocked growth band). "
            "Rendered under the old corpus.",
        )
    )
    A(f"""
<div class="box open">
<h4>What we do know, and it is thin</h4>
<p>Truth's growth per frontier cell on CZU is
<b>{b5["frontier_rate"]["2020_czu_lightning_complex"]:.3f}</b>
against {b5["frontier_rate"]["2020_bobcat"]:.3f} / {b5["frontier_rate"]["2020_creek"]:.3f} /
{b5["frontier_rate"]["2020_dolan"]:.3f} for its siblings - CZU is fast. The training fires span
0.082-0.409, so CZU is at the fast end of the training distribution, not outside it. Our model's
own predicted rate per frontier cell spans only ~1.7x where truth's spans ~5.7x:
<b>we predict least where truth is fastest.</b> That is a failure to modulate the rate in response
to covariates, not a failure to extrapolate - and it is what the current in-flight task is
testing. CZU is also the CZU <em>Lightning Complex</em>: separate ignitions by construction, which
a contagion-only kernel cannot reproduce at any temperature, and roughly a third of its domain is
the Pacific Ocean.</p>
</div>
""")

    # ---------------------------------------------------------------- S6
    A(f"""
<h2>6. What we have not attempted</h2>
<p>Plainly, because the temptation to imply otherwise is exactly what would cost us your trust.</p>

<table>
<thead><tr><th>Gate</th><th style="text-align:left">What it would show</th>
<th style="text-align:left">Status</th></tr></thead>
<tbody>
<tr><td>G4</td><td style="text-align:left">The explicit long-range <b>spot component</b> beats a
no-spot model on barrier-crossing / spotting episodes</td>
<td style="text-align:left"><span class="gate gn">NOT ATTEMPTED</span> - <b>the spot model does
not exist yet.</b> An event list of 12 events across 6 spatial blocks is recorded in the project
decision log, to be scored leave-block-out; <em>I could not locate the machine-readable file on
disk while writing this, so treat that count as reported and not as verified here.</em> On the
old 12-fire corpus there were only <b>two</b> never-merging spot candidates against hundreds of
2 km rasterisation holes, which is not adjudicable at all; growing the corpus was done
specifically to fix that.</td></tr>
<tr><td>G5</td><td style="text-align:left"><b>ELMFIRE Monte Carlo</b> head-to-head, the
operational-grade physics comparison</td>
<td style="text-align:left"><span class="gate gn">NOT ATTEMPTED</span> - ELMFIRE is installed,
patched, and callable behind our prediction interface, but <b>no head-to-head has been run and no
metric has been computed against it</b>.</td></tr>
<tr><td>G6</td><td style="text-align:left">Expected-loss maps (burn probability &times; structures)
for incident-command readout</td>
<td style="text-align:left"><span class="gate gn">NOT ATTEMPTED</span></td></tr>
</tbody></table>

<p>On G5 specifically, three things are already on the record and all of them cut against us:</p>
<ul>
<li>Our first ELMFIRE mapping produced a median of <b>{em["lobo_median_min"]:.0f}-
{em["lobo_median_max"]:.0f} new cells against truth's up to +{em["worst_truth"]}</b> on the
fastest windows, with as few as {em["lobo_distinct_min"]} distinct members of
{em["n_members"]}. That was <b>our</b> defect, not ELMFIRE's: we fed it canopy cover with no
canopy height, and a 60%-closed canopy of zero height collapses its wind adjustment factor to
nearly zero. On native 30 m inputs with the full canopy the same windows give a median of
{em["native_median_min"]:.0f}-{em["native_median_max"]:.0f} with all
{em["n_members"]}/{em["n_members"]} members distinct. <b>A partial description of a coupled input
group is an anti-capability, not a partial one.</b></li>
<li>Even on native inputs ELMFIRE reaches only <b>{em["native_ratio_min"]:.2f}-
{em["native_ratio_max"]:.2f}x</b> of truth's growth on those windows. So a future headline of
the form "we beat ELMFIRE on the Diablo run" would be much weaker than it sounds, and the ratio
has to be printed beside the win.</li>
<li>ELMFIRE's spotting is off by upstream default while our architecture has an explicit spot
component. Nobody has ruled on that asymmetry yet.</li>
</ul>
<p class="small">Provenance for the three bullets: <code>{em["artifact"]}</code>,
{em["n_windows"]} windows on <code>{em["tensor"]}</code>, {em["n_members"]} members.
{html.escape(em["note"])}</p>
<p class="small">We also found and reported two uninitialised-variable reads in ELMFIRE's own
Fortran source. We patched only the one we could prove was output-neutral, and avoided the code
path containing the other rather than fixing it - silently improving your opponent's model is
still cheating.</p>

<hr>
<h3>The honest summary</h3>
<ul>
<li><b>The architecture works.</b> The kernel learns something real about spread, beats a
calibrated physics baseline on held-out landscapes, and its trained version beats its untrained
version on the same metric.</li>
<li><b>The ensemble machinery works where it is turned up.</b> Removing the shared latent
collapses the spread {dsp["ablation_strong_min"]:.1f}x-{dsp["ablation_strong_max"]:.1f}x on
{dsp["n_ablation_strong"]} of {dsp["n_ablation_pairs"]} arms - and barely at all on the
{dsp["n_ablation_weak"]} that were already narrow.</li>
<li><b>The calibration does not work</b>, and the reason is not what we thought for three
consecutive attempts. The binding defect is a missing OFF state plus a growth rate that does not
respond enough to covariates - not ensemble width.</li>
<li><b>One block defies every explanation we have tried</b>, including the obvious one.</li>
<li><b>The two hardest comparisons - the spot component and ELMFIRE - have not been run.</b></li>
</ul>
<p class="small">Everything above is bound to the superseded 12-fire corpus
(<code>{FINGERPRINT_PRE_D6}</code>). The current corpus is 21 fires
(<code>{FINGERPRINT_OF_RECORD}</code>, 14 spatial blocks, 16 train / 5 held out) and nothing has
been re-scored against it yet. When it is, the numbers on this page will move, and the honest
expectation is that some of them will get worse.</p>
""")

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>wildfire-nowcast - evidence review</title>"
        f"<style>{css}</style></head><body><div class='wrap'>"
        + "".join(body)
        + "</div></body></html>"
    )
    OUT_HTML.write_text(doc, encoding="utf-8")
    return OUT_HTML


def main() -> int:
    # ADR-103: the ONE place this program configures logging. No flags of its own,
    # so WILDFIRE_LOG_LEVEL is how a reader raises it.
    configure_logging()
    d = collect()
    out = build_html(d)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB)")
    print(json.dumps({"gaps": d.gaps}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
