"""[M9 item 1] The CONDITIONAL RESPONSE, measured directly. Not an arm sweep.

Regresses ``log(growth per frontier cell)`` on standardised window covariates,
separately for TRUTH, for the MODEL's ensemble mean and for the WIND ELLIPSE,
on the same windows, and compares the slopes. Then asks whether the fitted slope
difference accounts for each block's log-rate deficit - the CZU question, made
arithmetic.

    .venv/bin/python runs/_m9_response.py [--stride 2] [--train-stride 6]

THE ELLIPSE IS RUN AT SCALE 1 AND UNCALIBRATED, ON PURPOSE. Its growth
calibration is a single multiplicative constant, i.e. a pure INTERCEPT shift in
log space, so it cannot move a single slope. Calibrating it would cost a full
pass over 16 train fires and change nothing in this table. Stated here rather
than left for a reader to notice.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.paths import fire_tensor_path, norm_stats_path
from wildfire_nowcast.common.zarr_io import open_tensor, read_norm_stats
from wildfire_nowcast.eval.baseline_run import load_splits
from wildfire_nowcast.eval.masks import default_band_radius
from wildfire_nowcast.eval.reporting import scoring_code_fingerprint, split_fingerprint
from wildfire_nowcast.eval.response import (
    COVARIATES,
    compare_responses,
    explain_block_deficit,
    fit_response,
    window_row,
)
from wildfire_nowcast.model.baselines import EllipseBaseline
from wildfire_nowcast.model.inputs import iter_windows
from wildfire_nowcast.model.kernel import ContagionKernel

OUT = Path("runs/m9_response.json")
HORIZON = 3
MEMBERS = 24
SEED = 20260807


def _windows(fire_id: str, stride: int) -> list[Any]:
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        return list(iter_windows(ds, HORIZON, stride=stride, fire_id=fire_id))
    finally:
        ds.close()


def _rows_for(fire, kernel, ellipse, stride: int, band_radius: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, w in enumerate(_windows(fire.fire_id, stride)):
        samples = kernel.predict(w.x0, w.static, w.weather, MEMBERS, HORIZON, SEED + i)
        row = window_row(
            w,
            samples,
            band_radius=band_radius,
            fire_id=fire.fire_id,
            spatial_block_id=fire.spatial_block_id,
        )
        ell = ellipse.predict(w.x0, w.static, w.weather, MEMBERS, HORIZON, SEED + i)
        band = (np.asarray(ell) > 0)[:, -1]
        from wildfire_nowcast.eval.masks import growth_band

        mask = growth_band(np.asarray(w.x0), band_radius)
        row["ellipse_growth"] = float(band[:, mask].sum(axis=1).mean())
        row["role"] = fire.role
        rows.append(row)
    return rows


def _per_block(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    blocks = sorted({int(r["spatial_block_id"]) for r in rows})
    fits = {}
    for b in blocks:
        fits[str(b)] = fit_response([r for r in rows if int(r["spatial_block_id"]) == b], target)
    usable = {k: v for k, v in fits.items() if not v.get("insufficient")}
    summary = {}
    for name in COVARIATES:
        joint = [v["joint"][name]["slope"] for v in usable.values()]
        uni = [v["univariate"][name]["slope"] for v in usable.values()]
        summary[name] = {
            "joint_mean": float(np.mean(joint)) if joint else None,
            "joint_sd": float(np.std(joint, ddof=1)) if len(joint) > 1 else None,
            "univariate_mean": float(np.mean(uni)) if uni else None,
            "univariate_sd": float(np.std(uni, ddof=1)) if len(uni) > 1 else None,
            "n_blocks": len(joint),
        }
    return {"per_block": fits, "equal_block": summary, "n_blocks_used": len(usable)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--train-stride", type=int, default=6)
    ap.add_argument("--checkpoint", type=str, default="")
    args = ap.parse_args(argv)

    scoring_before = scoring_code_fingerprint()
    split = split_fingerprint()
    stats = read_norm_stats(norm_stats_path())
    splits = load_splits([int(f) for f in stats["train_folds"]])
    heldout = [s for s in splits if not s.is_train]
    train = [s for s in splits if s.is_train]

    ckpt = args.checkpoint or json.loads(Path("runs/m9_train.json").read_text())["run_dir"]
    kernel = ContagionKernel.load(Path(ckpt) / "model.json")
    ellipse = EllipseBaseline(name="ellipse_uncalibrated")

    band_radius = default_band_radius(HORIZON)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    for fire in heldout:
        n0 = len(rows)
        rows += _rows_for(fire, kernel, ellipse, args.stride, band_radius)
        print(
            f"held-out {fire.fire_id:<32} block {fire.spatial_block_id:>2}  "
            f"{len(rows) - n0:>4} windows  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    train_rows: list[dict[str, Any]] = []
    for fire in train:
        n0 = len(train_rows)
        train_rows += _rows_for(fire, kernel, ellipse, args.train_stride, band_radius)
        print(
            f"train    {fire.fire_id:<32} block {fire.spatial_block_id:>2}  "
            f"{len(train_rows) - n0:>4} windows  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    heldout_cmp = compare_responses(rows)
    train_cmp = compare_responses(train_rows)
    ellipse_cmp = compare_responses(rows, model_key="ellipse_growth")
    deficit = explain_block_deficit(rows, heldout_cmp)
    deficit_uni = explain_block_deficit(rows, heldout_cmp, kind="univariate")

    payload = {
        "task": "M9 item 1 — the rate's conditional response to covariates",
        "split_fingerprint": split["fingerprint"],
        "scoring_code_before": scoring_before,
        "scoring_code_after": scoring_code_fingerprint(),
        "checkpoint": str(ckpt),
        "horizon_h": HORIZON,
        "n_members": MEMBERS,
        "stride_heldout": args.stride,
        "stride_train": args.train_stride,
        "band_radius_cells": band_radius,
        "covariates": list(COVARIATES),
        "n_rows_heldout": len(rows),
        "n_rows_train": len(train_rows),
        "heldout": heldout_cmp,
        "train": train_cmp,
        "ellipse_vs_truth_heldout": ellipse_cmp,
        "per_block_truth": _per_block(rows, "truth_growth"),
        "per_block_model": _per_block(rows, "model_growth"),
        "per_block_ellipse": _per_block(rows, "ellipse_growth"),
        "block_deficit_joint": deficit,
        "block_deficit_univariate": deficit_uni,
        "rows": rows,
        "not_a_verdict": "Diagnostic. No gate is adjudicated here and no arm is proposed.",
    }
    OUT.write_text(json.dumps(payload, indent=1, default=float))

    print()
    print(
        f"held-out rows {len(rows)}  usable in the paired fit {heldout_cmp['n']}   "
        f"train rows {len(train_rows)}  usable {train_cmp['n']}"
    )
    print()
    hdr = (
        f"{'covariate':<22}{'truth b':>10}{'+-':>8}{'model b':>10}{'+-':>8}"
        f"{'compress':>10}{'z_win':>8}{'ellipse b':>11}"
    )
    for kind in ("joint", "univariate"):
        print(
            f"--- {kind.upper()} slopes, POOLED over held-out blocks "
            f"(log rate per 1 SD of covariate)"
        )
        print(hdr)
        for name in COVARIATES:
            c = heldout_cmp["compression"][kind][name]
            e = ellipse_cmp["compression"][kind][name]["model_slope"]
            comp = c["compression"]
            print(
                f"{name:<22}{c['truth_slope']:>10.4f}{c['truth_se']:>8.4f}"
                f"{c['model_slope']:>10.4f}{c['model_se']:>8.4f}"
                f"{('     --' if comp is None else f'{comp:>10.3f}')}"
                f"{(0.0 if c['z_window_ols'] is None else c['z_window_ols']):>8.2f}{e:>11.4f}"
            )
        print()

    print("--- EQUAL-BLOCK slopes (mean +- SD over held-out blocks), JOINT")
    pb_t = _per_block(rows, "truth_growth")["equal_block"]
    pb_m = _per_block(rows, "model_growth")["equal_block"]
    print(f"{'covariate':<22}{'truth':>10}{'sd':>9}{'model':>10}{'sd':>9}{'compress':>10}")
    for name in COVARIATES:
        t, m = pb_t[name], pb_m[name]
        comp = (
            m["joint_mean"] / t["joint_mean"]
            if t["joint_mean"] and abs(t["joint_mean"]) > 1e-9
            else None
        )
        print(
            f"{name:<22}{(t['joint_mean'] or 0):>10.4f}{(t['joint_sd'] or 0):>9.4f}"
            f"{(m['joint_mean'] or 0):>10.4f}{(m['joint_sd'] or 0):>9.4f}"
            f"{('     --' if comp is None else f'{comp:>10.3f}')}"
        )
    print()
    print("--- TRAIN vs HELD-OUT: is the response absent, or does it fail to transfer?")
    for name in COVARIATES:
        h = heldout_cmp["compression"]["joint"][name]
        t = train_cmp["compression"]["joint"][name]
        print(
            f"{name:<22} train truth {t['truth_slope']:>8.4f} model {t['model_slope']:>8.4f}"
            f"   |   heldout truth {h['truth_slope']:>8.4f} model {h['model_slope']:>8.4f}"
        )
    print()
    print("--- BLOCK DEFICIT: does the fitted slope difference account for each block?")
    print(
        f"{'block':>6}{'n':>6}{'log r truth':>13}{'log r model':>13}{'deficit':>10}"
        f"{'obs dev':>10}{'pred dev':>10}{'residual':>10}"
    )
    for b in deficit["blocks"]:
        print(
            f"{b['spatial_block_id']:>6}{b['n_windows']:>6}{b['mean_log_rate_truth']:>13.4f}"
            f"{b['mean_log_rate_model']:>13.4f}{b['observed_deficit']:>10.4f}"
            f"{b['observed_deviation_from_level']:>10.4f}"
            f"{b['predicted_deviation_from_level']:>10.4f}{b['residual']:>10.4f}"
        )
    print(f"  level (removed from both sides) = {deficit['mean_deficit_level']:.4f}")
    print(
        f"  variance in the block SPREAD explained by the slope difference = "
        f"{deficit['variance_explained_across_blocks']:.3f}"
    )
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
