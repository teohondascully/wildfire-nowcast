"""[M9 item 1, robustness] Does truth's growth scale with FRONTIER LENGTH?

`runs/_m9_response.py` finds the largest truth/model slope gap on
`log_frontier_cells` (z = 8.68, joint), implying truth's growth is nearly
INDEPENDENT of frontier length while the model's is very nearly PROPORTIONAL to
it. That conclusion is drawn from a log-ratio regression fitted on GROWTH windows
only, and both of those choices can manufacture it:

* **division bias** - ``y = log(g) - log(f)`` regressed on ``log(f)`` shares a
  noisy term with its own regressor, which biases the elasticity DOWN;
* **selection** - conditioning on ``g > 0`` keeps the high-rate windows among
  short-frontier fires, which also biases the elasticity DOWN.

So the claim is re-tested here with an estimator that uses **no logarithm and no
conditioning on growth**: bin EVERY held-out window by frontier length and read
the mean growth in each bin. If truth's mean growth is flat across bins while the
model's rises in proportion, the finding survives both objections.

**THE INTERNAL CONTROL THAT MATTERS.** Truth and the model are measured on the
SAME windows with the SAME frontier count, by the SAME estimator. A bias in the
estimator applies to both. The model's elasticity is ~1.0 under it; truth's is
~0. A shared bias cannot produce a difference.

    .venv/bin/python runs/_m9_scaling.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path("runs/m9_response.json")
OUT = Path("runs/m9_scaling.json")
TARGETS = ("truth_growth", "model_growth", "ellipse_growth")
N_BINS = 8


def _loglog_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Weighted OLS of ``log(mean growth)`` on ``log(mean frontier)`` over BINS.

    Fitted on BIN MEANS, so a zero-growth window contributes a 0 to its bin's
    mean rather than being dropped - which is the whole point. Weights are bin
    counts. The slope IS the elasticity: 1.0 means growth is proportional to
    frontier length, 0.0 means it does not depend on it at all.
    """
    keep = (y > 0) & (x > 0)
    lx, ly, lw = np.log(x[keep]), np.log(y[keep]), w[keep]
    if lx.size < 3:
        return {"elasticity": float("nan"), "se": float("nan"), "n_bins": int(lx.size)}
    design = np.column_stack([np.ones(lx.size), lx])
    sw = np.sqrt(lw)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], ly * sw, rcond=None)
    resid = ly - design @ beta
    dof = max(lx.size - 2, 1)
    s2 = float((lw * resid**2).sum() / lw.sum()) * lx.size / dof
    cov = np.linalg.pinv((design * lw[:, None]).T @ design) * s2
    return {
        "elasticity": float(beta[1]),
        "intercept": float(beta[0]),
        "se": float(math.sqrt(max(cov[1, 1], 0.0))),
        "n_bins": int(lx.size),
    }


def main() -> int:
    payload = json.loads(SRC.read_text())
    rows = [r for r in payload["rows"] if r.get("role") == "heldout"]
    f = np.array([r["_n_frontier_cells"] for r in rows], dtype=float)
    order = np.argsort(f)
    bins = np.array_split(order, N_BINS)

    table: list[dict[str, Any]] = []
    for idx in bins:
        entry: dict[str, Any] = {
            "n_windows": int(idx.size),
            "frontier_mean": float(f[idx].mean()),
            "frontier_min": float(f[idx].min()),
            "frontier_max": float(f[idx].max()),
            "zero_growth_fraction": float(np.mean([rows[i]["truth_growth"] == 0 for i in idx])),
        }
        for target in TARGETS:
            values = np.array([float(rows[i][target]) for i in idx])
            entry[target] = float(values.mean())
            entry[f"{target}_per_frontier"] = float(values.mean() / f[idx].mean())
        table.append(entry)

    counts = np.array([e["n_windows"] for e in table], dtype=float)
    frontier = np.array([e["frontier_mean"] for e in table])
    fits = {
        target: _loglog_fit(frontier, np.array([e[target] for e in table]), counts)
        for target in TARGETS
    }

    # Per block, for the CZU question: where does each block sit in frontier
    # length, and what is its realised vs predicted rate?
    per_block: dict[str, Any] = {}
    for block in sorted({int(r["spatial_block_id"]) for r in rows}):
        sub = [r for r in rows if int(r["spatial_block_id"]) == block]
        grow = [r for r in sub if r["truth_growth"] > 0]
        fb = np.array([r["_n_frontier_cells"] for r in sub], dtype=float)
        per_block[str(block)] = {
            "fire_id": sub[0]["fire_id"],
            "n_windows": len(sub),
            "n_growth_windows": len(grow),
            "frontier_mean": float(fb.mean()),
            "frontier_median": float(np.median(fb)),
            "truth_growth_mean_all": float(np.mean([r["truth_growth"] for r in sub])),
            "model_growth_mean_all": float(np.mean([r["model_growth"] for r in sub])),
            "truth_rate_growth_windows": (
                float(np.mean([r["truth_growth"] / r["_n_frontier_cells"] for r in grow]))
                if grow
                else None
            ),
            "model_rate_growth_windows": (
                float(np.mean([r["model_growth"] / r["_n_frontier_cells"] for r in grow]))
                if grow
                else None
            ),
        }

    # WITHIN-BLOCK, because the pooled elasticity is partly a BETWEEN-FIRE
    # comparison: Creek is 785 of 1372 held-out windows and owns the long-frontier
    # bins outright. A negative pooled elasticity could therefore mean "big fires
    # grow less per unit perimeter" (a scaling law) OR "late windows grow less"
    # (fire stage, since a long frontier is also an OLD fire). Only the
    # within-fire version separates them, and they are different findings.
    within: dict[str, Any] = {}
    for block in sorted({int(r["spatial_block_id"]) for r in rows}):
        sub = [r for r in rows if int(r["spatial_block_id"]) == block]
        if len(sub) < 40:
            within[str(block)] = {
                "fire_id": sub[0]["fire_id"],
                "n": len(sub),
                "skipped": "fewer than 40 windows",
            }
            continue
        fb = np.array([r["_n_frontier_cells"] for r in sub], dtype=float)
        idxs = np.array_split(np.argsort(fb), min(N_BINS, max(3, len(sub) // 20)))
        b_front = np.array([fb[i].mean() for i in idxs])
        b_n = np.array([float(i.size) for i in idxs])
        entry: dict[str, Any] = {"fire_id": sub[0]["fire_id"], "n": len(sub), "n_bins": len(idxs)}
        for target in TARGETS:
            means = np.array([np.mean([float(sub[i][target]) for i in ix]) for ix in idxs])
            entry[target] = _loglog_fit(b_front, means, b_n)
        within[str(block)] = entry

    # THE SAME ROBUST ESTIMATOR, APPLIED TO EVERY COVARIATE. "Which channel
    # operated" must not be answered from joint OLS coefficients here: RH and
    # `fuel_moisture_proxy` are near-collinear by construction (the proxy is
    # RTMA-derived), and the joint decomposition of a block's deficit shows
    # contributions of +0.97 and -1.69 cancelling to -0.7 - individually
    # meaningless. This is the MARGINAL response, on ALL windows, with no design
    # matrix: bin by the covariate, read mean growth per bin, fit
    # log(mean growth) against the STANDARDISED covariate. Marginal responses
    # conflate correlated covariates, which is stated rather than hidden; what
    # they do not do is trade coefficients between collinear columns.
    marginal: dict[str, Any] = {}
    for name in payload["covariates"]:
        values = np.array([float(r[name]) for r in rows])
        mu, sd = float(values.mean()), float(values.std(ddof=1))
        if sd <= 0:
            continue
        idxs = np.array_split(np.argsort(values), N_BINS)
        z = np.array([(values[i].mean() - mu) / sd for i in idxs])
        n_b = np.array([float(i.size) for i in idxs])
        entry: dict[str, Any] = {"covariate_mean": mu, "covariate_sd": sd}
        for target in TARGETS:
            means = np.array([np.mean([float(rows[i][target]) for i in ix]) for ix in idxs])
            keep = means > 0
            if keep.sum() < 3:
                entry[target] = {"slope_per_sd": float("nan"), "se": float("nan")}
                continue
            design = np.column_stack([np.ones(keep.sum()), z[keep]])
            w = np.sqrt(n_b[keep])
            beta, *_ = np.linalg.lstsq(design * w[:, None], np.log(means[keep]) * w, rcond=None)
            resid = np.log(means[keep]) - design @ beta
            dof = max(int(keep.sum()) - 2, 1)
            s2 = float((n_b[keep] * resid**2).sum() / n_b[keep].sum()) * int(keep.sum()) / dof
            cov_m = np.linalg.pinv((design * n_b[keep][:, None]).T @ design) * s2
            entry[target] = {
                "slope_per_sd": float(beta[1]),
                "se": float(math.sqrt(max(cov_m[1, 1], 0.0))),
                "n_bins": int(keep.sum()),
            }
        marginal[name] = entry

    # ...AND THE SAME THING WITHIN EACH BLOCK, WHICH IS THE ONE TO READ.
    # The pooled marginal table above is CONTAMINATED BY BLOCK COMPOSITION and
    # the artifact says so with its own numbers: the four longest-frontier bins
    # are 100% Creek, and the four coldest temperature bins are 96-100% Creek. So
    # a pooled "response to temperature" is partly "response to being Creek in
    # winter". Binning WITHIN a block and averaging over blocks removes that
    # completely, and it is the same equal-block convention ADR-021 (4) adopted
    # for every criterion. Where the two tables disagree, this one is the
    # measurement and the pooled one is the confound.
    within_marginal: dict[str, Any] = {}
    for name in payload["covariates"]:
        per_block_slopes: dict[str, dict[str, float]] = {}
        for block in sorted({int(r["spatial_block_id"]) for r in rows}):
            sub = [r for r in rows if int(r["spatial_block_id"]) == block]
            if len(sub) < 40:
                continue
            values = np.array([float(r[name]) for r in sub])
            mu, sd = float(values.mean()), float(values.std(ddof=1))
            if sd <= 0:
                continue
            k = min(N_BINS, max(3, len(sub) // 20))
            idxs = np.array_split(np.argsort(values), k)
            z = np.array([(values[i].mean() - mu) / sd for i in idxs])
            n_b = np.array([float(i.size) for i in idxs])
            cell: dict[str, float] = {}
            for target in TARGETS:
                means = np.array([np.mean([float(sub[i][target]) for i in ix]) for ix in idxs])
                keep = means > 0
                if keep.sum() < 3:
                    continue
                design = np.column_stack([np.ones(keep.sum()), z[keep]])
                w = np.sqrt(n_b[keep])
                beta, *_ = np.linalg.lstsq(design * w[:, None], np.log(means[keep]) * w, rcond=None)
                cell[target] = float(beta[1])
            if cell:
                per_block_slopes[str(block)] = cell
        summary: dict[str, Any] = {"per_block": per_block_slopes}
        for target in TARGETS:
            vals = [v[target] for v in per_block_slopes.values() if target in v]
            summary[target] = {
                "equal_block_mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_blocks": len(vals),
                "n_wrong_sign_vs_truth": None,
            }
        # Equal-block z on the DIFFERENCE, with the standard error taken across
        # BLOCKS (n=5), never across windows. A window-level SE here would be a
        # fiction with a plausible magnitude: windows overlap in time and within
        # a fire, and blocks are the independent unit (C6.3).
        for target in ("model_growth", "ellipse_growth"):
            t_list = [v["truth_growth"] for v in per_block_slopes.values() if "truth_growth" in v]
            m_list = [v[target] for v in per_block_slopes.values() if target in v]
            if len(t_list) > 1 and len(m_list) == len(t_list):
                diffs = [m - t for m, t in zip(m_list, t_list, strict=True)]
                sd_d = float(np.std(diffs, ddof=1))
                se = sd_d / math.sqrt(len(diffs))
                summary[target]["paired_difference"] = float(np.mean(diffs))
                summary[target]["paired_difference_se_block_sd"] = se
                summary[target]["z_paired_block_sd"] = (
                    float(np.mean(diffs) / se) if se > 0 else None
                )
        t_vals = {b: v.get("truth_growth") for b, v in per_block_slopes.items()}
        for target in ("model_growth", "ellipse_growth"):
            summary[target]["n_wrong_sign_vs_truth"] = sum(
                1
                for b, v in per_block_slopes.items()
                if target in v and t_vals.get(b) is not None and v[target] * t_vals[b] < 0
            )
        within_marginal[name] = summary

    # FIRE STAGE, because within a fire the frontier is nearly monotone in time.
    # "Growth falls as the perimeter grows" and "growth falls as the fire ages"
    # are the same measurement inside one fire and DIFFERENT MECHANISMS, and a
    # kernel has no representation of either. Binned by `t0` so a reader can see
    # which one the data actually shows without taking my word for the framing.
    stage: dict[str, Any] = {}
    for block in sorted({int(r["spatial_block_id"]) for r in rows}):
        sub = sorted(
            (r for r in rows if int(r["spatial_block_id"]) == block), key=lambda r: r["t0"]
        )
        if len(sub) < 40:
            continue
        idxs = np.array_split(np.arange(len(sub)), min(N_BINS, max(3, len(sub) // 20)))
        stage[str(block)] = {
            "fire_id": sub[0]["fire_id"],
            "bins": [
                {
                    "t0_mean": float(np.mean([sub[i]["t0"] for i in ix])),
                    "n": int(ix.size),
                    "frontier_mean": float(np.mean([sub[i]["_n_frontier_cells"] for i in ix])),
                    "truth_growth": float(np.mean([sub[i]["truth_growth"] for i in ix])),
                    "model_growth": float(np.mean([sub[i]["model_growth"] for i in ix])),
                }
                for ix in idxs
            ],
        }

    out = {
        "task": "M9 — frontier-length scaling, without logs and without conditioning on growth",
        "fire_stage_by_t0": stage,
        "marginal_response_within_block": within_marginal,
        "marginal_response_all_windows": marginal,
        "within_block_elasticities": within,
        "source": str(SRC),
        "split_fingerprint": payload["split_fingerprint"],
        "n_heldout_windows": len(rows),
        "n_bins": N_BINS,
        "bins": table,
        "elasticities_from_bin_means": fits,
        "per_block": per_block,
        "not_a_verdict": "Diagnostic.",
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))

    print(f"held-out windows {len(rows)} (ALL windows, dormant included)")
    print(
        f"{'frontier bin':>14}{'n':>6}{'zero-growth':>13}{'truth g':>10}{'model g':>10}"
        f"{'ellipse g':>11}{'truth/f':>10}{'model/f':>10}"
    )
    for e in table:
        print(
            f"{e['frontier_min']:>6.0f}-{e['frontier_max']:<7.0f}{e['n_windows']:>6}"
            f"{e['zero_growth_fraction']:>13.3f}{e['truth_growth']:>10.3f}"
            f"{e['model_growth']:>10.3f}{e['ellipse_growth']:>11.3f}"
            f"{e['truth_growth_per_frontier']:>10.4f}{e['model_growth_per_frontier']:>10.4f}"
        )
    print()
    print(
        "ELASTICITY of mean growth w.r.t. frontier length (1.0 = proportional, 0.0 = no dependence)"
    )
    for target, fit in fits.items():
        print(
            f"  {target:<16}{fit['elasticity']:>8.4f} +- {fit['se']:.4f}   ({fit['n_bins']} bins)"
        )
    print()
    print(
        f"{'block':>6}{'fire':<30}{'n':>6}{'grow':>6}{'frontier':>10}"
        f"{'truth g':>10}{'model g':>10}{'truth rate':>12}{'model rate':>12}"
    )
    for block, e in per_block.items():
        print(
            f"{block:>6}{e['fire_id']:<30}{e['n_windows']:>6}{e['n_growth_windows']:>6}"
            f"{e['frontier_mean']:>10.1f}{e['truth_growth_mean_all']:>10.3f}"
            f"{e['model_growth_mean_all']:>10.3f}"
            f"{(e['truth_rate_growth_windows'] or 0):>12.4f}"
            f"{(e['model_rate_growth_windows'] or 0):>12.4f}"
        )
    print()
    print("MARGINAL RESPONSE on ALL held-out windows (d log mean growth per 1 SD of covariate)")
    print(
        f"{'covariate':<22}{'truth':>10}{'+-':>8}{'model':>10}{'+-':>8}{'ellipse':>10}{'gap/SE':>9}"
    )
    for name, e in marginal.items():
        t, m = e["truth_growth"], e["model_growth"]
        el = e["ellipse_growth"]["slope_per_sd"]
        gap = m["slope_per_sd"] - t["slope_per_sd"]
        se = math.hypot(t["se"], m["se"])
        print(
            f"{name:<22}{t['slope_per_sd']:>10.4f}{t['se']:>8.4f}"
            f"{m['slope_per_sd']:>10.4f}{m['se']:>8.4f}{el:>10.4f}"
            f"{(gap / se if se > 0 else float('nan')):>9.1f}"
        )
    print()
    print(
        "WITHIN-BLOCK MARGINAL RESPONSE, equal-block mean +- SD over blocks. THE TABLE TO "
        "READ:\n  the pooled one above is confounded (its 4 longest-frontier bins are 100% "
        "Creek)."
    )
    print(
        f"{'covariate':<22}{'truth':>10}{'sd':>9}{'model':>10}{'sd':>9}{'ellipse':>10}"
        f"{'blocks':>8}{'sign-flip':>11}{'z(paired)':>11}"
    )
    for name, e in within_marginal.items():
        t, m, el = e["truth_growth"], e["model_growth"], e["ellipse_growth"]
        flip = f"{m['n_wrong_sign_vs_truth']}/{m['n_blocks']}"
        print(
            f"{name:<22}{(t['equal_block_mean'] or 0):>10.4f}{(t['sd'] or 0):>9.4f}"
            f"{(m['equal_block_mean'] or 0):>10.4f}{(m['sd'] or 0):>9.4f}"
            f"{(el['equal_block_mean'] or 0):>10.4f}{t['n_blocks']:>8}"
            f"{flip:>11}{(m.get('z_paired_block_sd') or 0):>11.2f}"
        )
    print()
    print("WITHIN-BLOCK elasticities: separates a between-fire scaling law from fire STAGE")
    print(f"{'block':>6}{'fire':<30}{'n':>6}{'truth':>18}{'model':>18}{'ellipse':>18}")
    for block, e in within.items():
        if e.get("skipped"):
            print(f"{block:>6}{e['fire_id']:<30}{e['n']:>6}   SKIPPED ({e['skipped']})")
            continue

        def _c(entry: dict[str, Any], key: str) -> str:
            fit = entry[key]
            return f"{fit['elasticity']:>10.3f}+-{fit['se']:<6.3f}"

        print(
            f"{block:>6}{e['fire_id']:<30}{e['n']:>6}"
            f"{_c(e, 'truth_growth'):>18}{_c(e, 'model_growth'):>18}"
            f"{_c(e, 'ellipse_growth'):>18}"
        )
    print()
    print("FIRE STAGE: the same fire, binned by t0. Frontier grows monotonically; does truth?")
    for block, e in stage.items():
        print(f"  block {block} {e['fire_id']}")
        print(f"    {'t0':>7}{'n':>5}{'frontier':>10}{'truth g':>10}{'model g':>10}")
        for b in e["bins"]:
            print(
                f"    {b['t0_mean']:>7.0f}{b['n']:>5}{b['frontier_mean']:>10.1f}"
                f"{b['truth_growth']:>10.3f}{b['model_growth']:>10.3f}"
            )
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
