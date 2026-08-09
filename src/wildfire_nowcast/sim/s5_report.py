"""S5 — assemble the block-5 (CZU) diagnosis into one artifact and one page.

Reads: a C6 run record's ``results.json`` and the C1 tensors. Calls no model,
loads no checkpoint, writes only under ``reports/figures``.

WHAT THIS ANSWERS
-----------------
"Why can CZU's ensemble not spread?" — decomposed rather than scored, in the
shape of the growth-anatomy work (ADR-025): take one scalar apart until the
strata disagree.

THE FOUR-FACTOR FORM. Combining :mod:`sim.blockanatomy`'s identity with C6's own
bias/scatter split gives, exactly::

    adr = sqrt((M+1)/M) x CV x  (pred_mean/truth_mean)  x (truth_mean/truthRMS) x relief
                          ^ensemble ^GROWTH CALIBRATION   ^truth shape           ^model's own
                           relative                        (1.74-2.01 across      denominator
                           width                           our four blocks)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.paths import repo_root
from wildfire_nowcast.sim.blockanatomy import (
    adr_parts,
    block_truth,
    frontier_rate,
    support_distance,
)

__all__ = ["build_report", "render", "main"]

#: Arms named in ADR-034 (5)'s per-block table, so the artifact can be read
#: against the observation it was built to test.
FOCUS_ARMS: tuple[str, ...] = (
    "m6_fair_brier0_s1",
    "m7_offstate_s1",
    "m7_gate_nofix_s3",
    "m6_fair_s1",
)


def build_report(
    results_path: str | Path,
    *,
    stride: int = 2,
    with_support: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(results_path).read_text())
    horizon = int(payload["horizon_h"])
    n_members = int(payload["n_members"])
    per_fire = payload["per_fire"]
    scope = payload["scope"]

    truths = {
        fid: block_truth(fid, int(pf["spatial_block_id"]), horizon_h=horizon, stride=stride)
        for fid, pf in per_fire.items()
    }
    rates = {fid: frontier_rate(fid, horizon_h=horizon, stride=stride) for fid in per_fire}

    known: list[dict[str, Any]] = []
    for fid, pf in per_fire.items():
        p = pf["models"]["persistence"]["growth_windows"]
        rec = float(np.hypot(p["band_area_error_bias"], p["band_area_error_scatter"]))
        known.append(
            {
                "fire_id": fid,
                "persistence_rms_err_from_record": rec,
                "truth_rms_recomputed_from_c1": truths[fid].truth_rms,
                "abs_diff": abs(rec - truths[fid].truth_rms),
            }
        )

    arms = [m for m in next(iter(per_fire.values()))["models"] if not m.endswith("__ABL")]
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for fid, pf in per_fire.items():
            gb = pf["models"].get(arm, {}).get("growth_windows")
            if not gb or gb.get("band_area_dispersion_ratio") is None:
                continue
            t = truths[fid]
            parts = adr_parts(arm, fid, int(pf["spatial_block_id"]), gb, t.truth_rms, n_members)
            pred_mean = t.truth_mean + parts.bias
            rows.append(
                {
                    **{
                        k: v
                        for k, v in parts.__dict__.items()
                        if k not in {"model", "fire_id"}
                    },
                    "model": arm,
                    "fire_id": fid,
                    "truth_mean": t.truth_mean,
                    "pred_mean": pred_mean,
                    "growth_calibration": pred_mean / t.truth_mean,
                    "ensemble_cv": (parts.member_sd / pred_mean) if pred_mean > 0 else None,
                    "truth_shape_factor": t.truth_mean / t.truth_rms,
                    "n_units": t.n_units,
                    "n_eff_units": t.n_eff_units,
                }
            )

    support = (
        support_distance(
            scope["train_fire_ids"],
            scope["heldout_fire_ids"],
            horizon_h=horizon,
            stride=stride,
        )
        if with_support
        else {}
    )

    return {
        "task": "S5 — why block 5 (CZU) cannot spread",
        "results_path": str(results_path),
        "split_fingerprint": payload["split_before"]["fingerprint"],
        "code_fingerprints_agree": payload["code_fingerprints_agree"]["verdict"],
        "horizon_h": horizon,
        "n_members": n_members,
        "stride": stride,
        "identity": (
            "adr == sqrt((M+1)/M) * ensemble_cv * growth_calibration * "
            "truth_shape_factor * denominator_relief, exactly"
        ),
        "known_answer_check": known,
        "max_identity_residual": max(float(r["identity_residual"]) for r in rows),
        "block_truth": {k: v.__dict__ for k, v in truths.items()},
        "frontier_rate": rates,
        "train_support_distance": support,
        "parts": rows,
        "focus_arms": list(FOCUS_ARMS),
        "not_a_verdict": (
            "G3 is adjudicated by the maintainer. Nothing here is a pass/fail for any gate."
        ),
    }


def _factor_spread(rows: list[dict[str, Any]], arm: str, key: str) -> float:
    vals = [abs(float(r[key])) for r in rows if r["model"] == arm and r[key] is not None]
    return max(vals) / min(vals) if vals and min(vals) > 0 else float("nan")


def render(report: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wildfire_nowcast.sim.style import stamp

    rows = report["parts"]
    blocks = sorted({int(r["spatial_block_id"]) for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))
    fig.suptitle(
        "S5 — block 5 (CZU) cannot spread because its MEAN is 3-7x too small.\n"
        "The 'CZU is ordinary at w_brier=0' reading is the ratio's denominator, not its "
        "numerator.",
        fontsize=13,
        fontweight="bold",
    )

    def pick(arm: str, key: str) -> list[float]:
        by = {int(r["spatial_block_id"]): r for r in rows if r["model"] == arm}
        return [float(by[b][key]) if by.get(b) and by[b][key] is not None else np.nan
                for b in blocks]

    # (A) adr vs s2s for the arm that launched the task
    ax = axes[0, 0]
    w = 0.38
    xs = np.arange(len(blocks))
    arm = "m6_fair_brier0_s1"
    ax.bar(xs - w / 2, pick(arm, "adr"), w, label="adr (G3 criterion)", color="#8899aa")
    ax.bar(xs + w / 2, pick(arm, "spread_to_signal"), w,
           label="s2s (spread / truth's own scale)", color="#c0392b")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"block {b}" for b in blocks])
    ax.set_title(f"(A) {arm}: adr looks flat, the ensemble width does NOT", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhspan(0.8, 1.2, color="#2e7d32", alpha=0.10)
    for i, b in enumerate(blocks):
        if b == 5:
            ax.annotate("block 5\nNARROWEST", (i + w / 2, pick(arm, "spread_to_signal")[i]),
                        textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=8, color="#c0392b", fontweight="bold")

    # (B) s2s across every arm — block 5 is last, always
    ax = axes[0, 1]
    arms = sorted({r["model"] for r in rows if r["model"] != "persistence"})
    for b, col in zip(blocks, ["#1f77b4", "#ff7f0e", "#c0392b", "#2ca02c"], strict=True):
        ys = [next((float(r["spread_to_signal"]) for r in rows
                    if r["model"] == a and int(r["spatial_block_id"]) == b), np.nan)
              for a in arms]
        ax.plot(range(len(arms)), ys, marker="o", ms=3,
                lw=2.2 if b == 5 else 1.0, color=col, label=f"block {b}",
                zorder=5 if b == 5 else 2)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=90, fontsize=5.5)
    ax.set_ylabel("spread-to-signal")
    ax.set_title("(B) block 5 has the lowest s2s in EVERY arm on the record", fontsize=10)
    ax.legend(fontsize=8)

    # (C) the mechanism: predicted mean vs truth mean
    ax = axes[1, 0]
    for i, a in enumerate(report["focus_arms"]):
        ax.plot(xs + (i - 1.5) * 0.06, pick(a, "growth_calibration"),
                marker="s", ms=6, lw=0, label=a)
    ax.axhline(1.0, color="#333", lw=1, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"block {b}" for b in blocks])
    ax.set_yscale("log")
    ax.set_ylabel("predicted mean area / truth mean area")
    ax.set_title("(C) MECHANISM: every arm under-predicts block 5 by 3-7x", fontsize=10)
    ax.legend(fontsize=7)

    # (D) the transfer test
    ax = axes[1, 1]
    sup = report.get("train_support_distance", {}).get("blocks", {})
    if sup:
        bt = report["block_truth"]
        for fid, s in sup.items():
            b = int(bt[fid]["spatial_block_id"])
            y = next(float(r["spread_to_signal"]) for r in rows
                     if r["model"] == "m7_offstate_s1" and int(r["spatial_block_id"]) == b)
            ax.scatter(s["mahalanobis"], y, s=90,
                       color="#c0392b" if b == 5 else "#1f77b4", zorder=3)
            ax.annotate(f"block {b}\n{fid.replace('2020_', '')}",
                        (s["mahalanobis"], y), textcoords="offset points",
                        xytext=(7, -4), fontsize=7.5)
        ax.set_xlabel("Mahalanobis distance of scored conditions from TRAIN support")
        ax.set_ylabel("spread-to-signal (m7_offstate_s1)")
        ax.set_title(
            "(D) TRANSFER TEST, n=4 — a TREND, not a law.\n"
            "Blocks 5 and 6 are equally far from train (4.17 vs 4.07)\n"
            "and 2.2x apart in spread. Distance cannot be the operative variable.",
            fontsize=9,
        )
    stamp(
        fig,
        "C6 run record + C1 tensors only; no model called, no checkpoint read. "
        f"split {report['split_fingerprint']}. NOT a gate verdict.",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="runs/baselines-20260809-073414/results.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--no-support", action="store_true")
    ap.add_argument("--stem", default="s5_block5_anatomy")
    args = ap.parse_args(argv)
    out_dir = repo_root() / "reports" / "figures"
    report = build_report(args.results, stride=args.stride, with_support=not args.no_support)
    (out_dir / f"{args.stem}.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.stem}.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    render(report, out_dir / f"{args.stem}.png")
    print(f"wrote {out_dir / args.stem}.{{json,png}}")
    print(f"max identity residual {report['max_identity_residual']:.3g}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
