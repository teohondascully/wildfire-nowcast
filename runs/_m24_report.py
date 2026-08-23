"""[M24] Read ADR-128 (4)'s four acceptance numbers off `runs/m24_frontdist.json`.

Everything here is pooled EQUAL-BLOCK (mean over the 5 held-out spatial blocks of
each block's own mean over its windows), because C3.1 says buffered domains
overlap and ADR-128 (5) says origins inside a fire are autocorrelated. Separation
is |mean paired block difference| / SD of that difference ACROSS BLOCKS - the
ADR-042 denominator, never a seed SD. Intervals are a BLOCK bootstrap.

    .venv/bin/python runs/_m24_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.eval.power import _spearman

SRC = Path("runs/m24_frontdist.json")
OUT = Path("runs/m24_acceptance.json")
LEADS = (1, 2, 3)
N_BOOT = 10000


def _mean(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def per_block(payload: dict[str, Any], rung: str, getter: Any) -> dict[str, float]:
    """One value per FIRE (= one per block on fold 3), each a mean over its windows."""
    out: dict[str, float] = {}
    for fire, block in payload["per_fire"].items():
        value = _mean([getter(row) for row in block["rows"][rung]])
        if value is not None:
            out[fire] = value
    return out


def equal_block(values: dict[str, float]) -> float | None:
    return float(np.mean(list(values.values()))) if values else None


def separation(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    """|mean paired block difference| / SD across blocks. ADR-042's denominator."""
    fires = sorted(set(a) & set(b))
    if len(fires) < 2:
        return {"separation": None, "n_blocks": len(fires)}
    diff = np.array([a[f] - b[f] for f in fires], dtype=np.float64)
    sd = float(diff.std(ddof=1))
    return {
        "mean_diff": float(diff.mean()),
        "sd_diff_blocks": sd,
        "separation": (abs(float(diff.mean())) / sd) if sd > 0 else None,
        "n_blocks": len(fires),
        "unanimous": bool(np.all(diff > 0) or np.all(diff < 0)),
    }


def block_bootstrap(values: dict[str, float], seed: int = 20260823) -> dict[str, Any]:
    fires = sorted(values)
    arr = np.array([values[f] for f in fires], dtype=np.float64)
    if arr.size < 2:
        return {"lo": None, "hi": None, "n_blocks": int(arr.size)}
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, arr.size, size=(N_BOOT, arr.size))].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "se_blocks": float(arr.std(ddof=1) / np.sqrt(arr.size)),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n_blocks": int(arr.size),
    }


CHANNELS: dict[str, Any] = {
    "front_combined": lambda h: lambda r: r["front"][str(h)]["combined"],
    "front_truth_to_pred": lambda h: lambda r: r["front"][str(h)]["truth_to_pred"],
    "front_pred_to_truth": lambda h: lambda r: r["front"][str(h)]["pred_to_truth"],
    "iou_shape_masked": lambda h: lambda r: r["iou_shape_masked"][h - 1],
    "brier": lambda h: lambda r: r["brier"][h - 1],
    "arrival_crps": lambda _h: lambda r: r["arrival_crps"],
}
#: +1 when a LOWER score means a better forecast. ADR-053 reported Spearman on
#: lower-is-better channels against |log area error|, so a correctly-ordered
#: channel is POSITIVE there; `iou_shape_masked` is higher-is-better and its
#: correctly-ordered sign is NEGATIVE. Both raw signs are printed and the
#: ORIENTED one is printed beside them so no reader has to remember which.
LOWER_IS_BETTER = {
    "front_combined": True,
    "front_truth_to_pred": True,
    "front_pred_to_truth": True,
    "iou_shape_masked": False,
    "brier": True,
    "arrival_crps": True,
}


def main() -> int:
    payload = json.loads(SRC.read_text())
    fires = sorted(payload["per_fire"])
    area_rungs = [f"area_k{lv:.3f}" for lv in payload["area_levels"]]
    shift_rungs = [f"shift_d{lv:04.1f}" for lv in payload["shift_levels"]]
    shape_rungs = [f"shape_f{lv:.3f}" for lv in payload.get("shape_levels", [])]

    # --- severity units, MEASURED on the samples ---------------------------
    severity: dict[str, dict[str, Any]] = {}
    for rung in area_rungs + shift_rungs + shape_rungs + ["reference"]:
        log_ratio: dict[str, float] = {}
        disp: dict[str, float] = {}
        for fire in fires:
            rows = payload["per_fire"][fire]["rows"][rung]
            pred = sum(float(r["pred_cells"][2]) for r in rows)
            truth = sum(float(r["truth_cells"][2]) for r in rows)
            if pred > 0 and truth > 0:
                log_ratio[fire] = float(np.log(pred / truth))
            km = _mean([s["displacement_km"] for s in payload["per_fire"][fire]["severity"][rung]])
            if km is not None:
                disp[fire] = km
        severity[rung] = {
            "log_area_ratio": equal_block(log_ratio),
            "abs_log_area_ratio": abs(equal_block(log_ratio)) if log_ratio else None,
            "displacement_km": equal_block(disp),
        }

    # --- per-rung, per-lead, per-block pooled values ------------------------
    table: dict[str, Any] = {}
    for rung in ["reference"] + area_rungs + shift_rungs + shape_rungs:
        row: dict[str, Any] = {}
        for name, make in CHANNELS.items():
            for lead in LEADS:
                blocks = per_block(payload, rung, make(lead))
                row[f"{name}_{lead}h"] = {
                    "equal_block": equal_block(blocks),
                    "by_block": blocks,
                }
        table[rung] = row

    def series(rungs: list[str], key: str) -> np.ndarray:
        return np.array([table[r][key]["equal_block"] for r in rungs], dtype=np.float64)

    # --- LEG 1: monotone in AREA error --------------------------------------
    area_sev = np.array([severity[r]["abs_log_area_ratio"] for r in area_rungs], dtype=np.float64)
    leg1: dict[str, Any] = {
        "severity_abs_log_area_ratio": dict(zip(area_rungs, area_sev.tolist(), strict=True))
    }
    for name in CHANNELS:
        for lead in LEADS:
            key = f"{name}_{lead}h"
            rho = _spearman(area_sev, series(area_rungs, key))
            leg1[key] = {
                "spearman_raw": rho,
                "spearman_oriented": (
                    None if rho is None else (rho if LOWER_IS_BETTER[name] else -rho)
                ),
                "values": series(area_rungs, key).tolist(),
            }

    # --- LEG 2: monotone in DISPLACEMENT ------------------------------------
    disp_sev = np.array([severity[r]["displacement_km"] for r in shift_rungs], dtype=np.float64)
    leg2: dict[str, Any] = {
        "severity_displacement_km": dict(zip(shift_rungs, disp_sev.tolist(), strict=True))
    }
    for name in CHANNELS:
        for lead in LEADS:
            key = f"{name}_{lead}h"
            values = series(shift_rungs, key)
            rho = _spearman(disp_sev, values)
            steps = np.diff(values)
            leg2[key] = {
                "spearman_raw": rho,
                "spearman_oriented": (
                    None if rho is None else (rho if LOWER_IS_BETTER[name] else -rho)
                ),
                "strictly_monotone_worse": bool(
                    np.all(steps > 0) if LOWER_IS_BETTER[name] else np.all(steps < 0)
                ),
                "values": values.tolist(),
            }

    # --- LEG 3: do the two TERMS separate area error from displacement? -----
    leg3: dict[str, Any] = {}
    for rung in area_rungs + shift_rungs + shape_rungs:
        ttp = table[rung]["front_truth_to_pred_3h"]["equal_block"]
        ptt = table[rung]["front_pred_to_truth_3h"]["equal_block"]
        leg3[rung] = {
            "combined": table[rung]["front_combined_3h"]["equal_block"],
            "truth_to_pred": ttp,
            "pred_to_truth": ptt,
            "asymmetry_log": (
                float(np.log(ttp / ptt)) if ttp and ptt and ttp > 0 and ptt > 0 else None
            ),
            "family": (
                "area" if rung in area_rungs else ("shift" if rung in shift_rungs else "shape")
            ),
            "abs_log_area_ratio": severity[rung]["abs_log_area_ratio"],
            "displacement_km": severity[rung]["displacement_km"],
        }

    # --- LEG 4: variance against the incumbent, on the SAME episodes --------
    leg4: dict[str, Any] = {"level": {}, "paired_separation": {}}
    for name in (
        "front_combined",
        "front_truth_to_pred",
        "front_pred_to_truth",
        "iou_shape_masked",
        "brier",
        "arrival_crps",
    ):
        for lead in LEADS:
            key = f"{name}_{lead}h"
            leg4["level"][key] = block_bootstrap(table["reference"][key]["by_block"])
            boot = leg4["level"][key]
            leg4["level"][key]["relative_se"] = (
                abs(boot["se_blocks"] / boot["mean"]) if boot.get("mean") else None
            )
    for rung in shift_rungs + area_rungs + shape_rungs:
        for name in ("front_combined", "iou_shape_masked", "brier"):
            for lead in LEADS:
                key = f"{name}_{lead}h"
                leg4["paired_separation"].setdefault(rung, {})[key] = separation(
                    table[rung][key]["by_block"], table["reference"][key]["by_block"]
                )

    out = {
        "task": "M24 (ADR-128 (4)) acceptance test read-off",
        "source": str(SRC),
        "n_growth_windows": payload["n_growth_windows"],
        "n_blocks": payload["n_blocks"],
        "fires": fires,
        "split_before": payload["split_before"],
        "split_after": payload["split_after"],
        "scoring_fingerprints_agree": payload["scoring_fingerprint_before"]
        == payload["scoring_fingerprint_after"],
        "common_fingerprints_agree": payload["common_fingerprint_before"]
        == payload["common_fingerprint_after"],
        "severity": severity,
        "leg1_area_monotonicity": leg1,
        "leg2_displacement_monotonicity": leg2,
        "leg3_separation": leg3,
        "leg4_variance": leg4,
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
