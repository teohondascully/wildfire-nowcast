"""[M24] Read ADR-128 (4)'s four acceptance numbers off `runs/m24_frontdist.json`.

Everything here is pooled EQUAL-BLOCK (mean over the 5 held-out spatial blocks of
each block's own mean over its windows), because C3.1 says buffered domains
overlap and ADR-128 (5) says origins inside a fire are autocorrelated. Separation
is |mean paired block difference| / SD of that difference ACROSS BLOCKS - the
ADR-042 denominator, never a seed SD. Intervals are a BLOCK bootstrap.

[M25, ADR-130 (4)] The same read-off now also carries the two questions
ADR-130 (3) proved the score conflates: ``p_silent`` as its OWN channel, and
front-distance CRPS CONDITIONAL on non-silence. The all-silent episode-lead is a
DEFINED case and is reported BOTH ways - dropped (treatment A) and kept at the
unconditional value, which for an all-silent ensemble is exactly ``cap/2``
(treatment B) - with its count, because dropping it silently would select
exactly the episodes where the ensemble was active.

    .venv/bin/python runs/_m24_report.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.eval.power import _spearman

SRC = Path("runs/m24_frontdist.json")
#: The same table, gzipped. [ADR-130 (5)] The plain `.json` form is 6.5 MB and
#: the disk holds the compressed one, so a reader who runs this script after a
#: `git clone` gets `FileNotFoundError` - which is exactly how a verification
#: diff once reported "identical" for a script that had CRASHED and never
#: rewritten its output. Reading either form removes the trap; the EXIT CODE is
#: still the thing to check before trusting any diff.
SRC_GZ = Path("runs/m24_frontdist.json.gz")
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


def _cond_kept(key: str) -> Any:
    """[M25] Treatment B: the all-silent episode-lead KEEPS its unconditional value.

    For an ensemble in which every member is silent the unconditional terms are
    exactly ``truth_to_pred = cap`` and ``pred_to_truth = 0``, so ``combined`` is
    ``cap/2`` - the worst location score the channel can express. Imputing it is
    therefore not an invention: it is the same number the incumbent
    unconditional channel already assigns, and it keeps the episode in the
    denominator. Reported BESIDE treatment A, never instead of it.
    """

    def make(h: int) -> Any:
        def getter(row: dict[str, Any]) -> float | None:
            cell = row["front"][str(h)]
            value = cell[f"{key}_cond"]
            return cell[key] if value is None else value

        return getter

    return make


def _p_silent(h: int) -> Any:
    def getter(row: dict[str, Any]) -> float | None:
        cell = row["front"][str(h)]
        members = int(cell["n_members"])
        return None if members == 0 else float(cell["n_empty_members"]) / float(members)

    return getter


CHANNELS: dict[str, Any] = {
    "front_combined": lambda h: lambda r: r["front"][str(h)]["combined"],
    "front_truth_to_pred": lambda h: lambda r: r["front"][str(h)]["truth_to_pred"],
    "front_pred_to_truth": lambda h: lambda r: r["front"][str(h)]["pred_to_truth"],
    "iou_shape_masked": lambda h: lambda r: r["iou_shape_masked"][h - 1],
    "brier": lambda h: lambda r: r["brier"][h - 1],
    "arrival_crps": lambda _h: lambda r: r["arrival_crps"],
    # [M25] TREATMENT A - conditional on non-silence, all-silent episode-leads
    # UNDEFINED and therefore absent from the mean. `_mean` already skips None,
    # so the drop happens here and its SIZE is counted in `m25_silence`.
    "front_combined_cond": lambda h: lambda r: r["front"][str(h)]["combined_cond"],
    "front_truth_to_pred_cond": lambda h: lambda r: r["front"][str(h)]["truth_to_pred_cond"],
    "front_pred_to_truth_cond": lambda h: lambda r: r["front"][str(h)]["pred_to_truth_cond"],
    # [M25] TREATMENT B - the same score with the all-silent leads KEPT.
    "front_combined_cond_kept": _cond_kept("combined"),
    # [M25] Silence as its own channel. A DIAGNOSTIC, not a score: it is listed
    # in LOWER_IS_BETTER only because the monotonicity legs iterate this dict,
    # and a Spearman on it answers "does this degradation change WHETHER the
    # ensemble spoke", which is the confound the conditional score removes.
    "p_silent": _p_silent,
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
    "front_combined_cond": True,
    "front_truth_to_pred_cond": True,
    "front_pred_to_truth_cond": True,
    "front_combined_cond_kept": True,
    "p_silent": True,
}


def _read_source() -> dict[str, Any]:
    if SRC.exists():
        return dict(json.loads(SRC.read_text()))
    if SRC_GZ.exists():
        with gzip.open(SRC_GZ, "rt") as handle:
            return dict(json.load(handle))
    raise FileNotFoundError(f"neither {SRC} nor {SRC_GZ} exists")


def main() -> int:
    payload = _read_source()
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
        "front_combined_cond",
        "front_truth_to_pred_cond",
        "front_pred_to_truth_cond",
        "front_combined_cond_kept",
        "p_silent",
    ):
        for lead in LEADS:
            key = f"{name}_{lead}h"
            leg4["level"][key] = block_bootstrap(table["reference"][key]["by_block"])
            boot = leg4["level"][key]
            leg4["level"][key]["relative_se"] = (
                abs(boot["se_blocks"] / boot["mean"]) if boot.get("mean") else None
            )
    for rung in shift_rungs + area_rungs + shape_rungs:
        for name in (
            "front_combined",
            "iou_shape_masked",
            "brier",
            "front_combined_cond",
            "front_combined_cond_kept",
            "p_silent",
        ):
            for lead in LEADS:
                key = f"{name}_{lead}h"
                leg4["paired_separation"].setdefault(rung, {})[key] = separation(
                    table[rung][key]["by_block"], table["reference"][key]["by_block"]
                )

    # --- [M25] SILENCE AS ITS OWN CHANNEL, AND THE ALL-SILENT ACCOUNTING ----
    silence: dict[str, Any] = {
        "definition": (
            "a member is SILENT at (window, lead) when its increment inside the SCORED MASK "
            "(growth_band & ~burned_at_t0) is empty; p_silent = n_empty_members / n_members"
        ),
        "p_silent_by_lead": {},
        "p_silent_by_block": {},
        "censored_fraction_by_lead": {},
        "censored_fraction_cond_by_lead": {},
        "all_silent_episode_leads": {},
        "defined_episode_leads": {},
        "all_silent_unconditional_combined": {},
    }
    for lead in LEADS:
        silence["p_silent_by_lead"][str(lead)] = table["reference"][f"p_silent_{lead}h"][
            "equal_block"
        ]
        silence["p_silent_by_block"][str(lead)] = table["reference"][f"p_silent_{lead}h"][
            "by_block"
        ]
        cens = per_block(
            payload, "reference", lambda r, h=lead: r["front"][str(h)]["censored_fraction"]
        )
        cens_c = per_block(
            payload, "reference", lambda r, h=lead: r["front"][str(h)]["censored_fraction_cond"]
        )
        silence["censored_fraction_by_lead"][str(lead)] = equal_block(cens)
        silence["censored_fraction_cond_by_lead"][str(lead)] = equal_block(cens_c)
    for rung in ["reference", "shape_f1.000"]:
        per_rung_all_silent: dict[str, dict[str, int]] = {}
        per_rung_defined: dict[str, dict[str, int]] = {}
        imputed: list[float] = []
        for lead in LEADS:
            per_rung_all_silent[str(lead)] = {}
            per_rung_defined[str(lead)] = {}
            for fire in fires:
                rows = payload["per_fire"][fire]["rows"][rung]
                cells = [r["front"][str(lead)] for r in rows]
                defined_cells = [c for c in cells if c["combined"] is not None]
                mute = [c for c in defined_cells if c["combined_cond"] is None]
                per_rung_defined[str(lead)][fire] = len(defined_cells)
                per_rung_all_silent[str(lead)][fire] = len(mute)
                imputed.extend(float(c["combined"]) for c in mute)
        silence["all_silent_episode_leads"][rung] = per_rung_all_silent
        silence["defined_episode_leads"][rung] = per_rung_defined
        silence["all_silent_unconditional_combined"][rung] = {
            "n": len(imputed),
            "distinct_values": sorted({round(v, 9) for v in imputed}),
            "note": (
                "treatment B imputes these; an all-silent ensemble scores truth_to_pred = cap "
                "and pred_to_truth = 0 by construction, so combined must be exactly cap/2"
            ),
        }

    # --- [M25] THE PRE-REGISTERED BAR, ADR-130 (4) --------------------------
    BAR = 2.536
    RUNG = "shape_f1.000"
    LEAD = 3
    incumbent = leg4["paired_separation"][RUNG][f"iou_shape_masked_{LEAD}h"]
    verdict: dict[str, Any] = {
        "preregistered_bar": BAR,
        "bar_source": (
            "ADR-130 (4): the incumbent best_member_iou_shape_masked's ALREADY-MEASURED "
            f"separation on {RUNG} at {LEAD} h, chosen so it could not be tuned after the table"
        ),
        "incumbent_recomputed_here": incumbent,
        "rung": RUNG,
        "lead_h": LEAD,
        "cap_km_unchanged": True,
        "rung_set_unchanged": True,
        "split_unchanged": payload["split_before"]["fingerprint"],
        "arms": {},
        # C6.7 [v2.18]: an instrument may adjudicate where it has power and MUST
        # report at 1/2/3 h. The pre-registered CELL is 3 h; the other two leads
        # are reported beside it so this cannot become a published SELECTION.
        "all_leads": {},
    }
    ARMS = (
        ("A_all_silent_dropped", "front_combined_cond"),
        ("B_all_silent_kept_at_cap", "front_combined_cond_kept"),
        ("unconditional_M24_control", "front_combined"),
        ("incumbent_iou_shape_masked", "iou_shape_masked"),
    )

    def _arm(channel: str, lead: int) -> dict[str, Any]:
        key = f"{channel}_{lead}h"
        sep = dict(leg4["paired_separation"][RUNG][key])
        degraded = table[RUNG][key]["by_block"]
        base = table["reference"][key]["by_block"]
        sep["by_block_diff"] = {f: degraded[f] - base[f] for f in sorted(set(degraded) & set(base))}
        sep["degraded_by_block"] = degraded
        sep["reference_by_block"] = base
        sep["clears_bar"] = bool(
            sep.get("separation") is not None
            and float(sep["separation"]) >= BAR
            and bool(sep.get("unanimous"))
        )
        return sep

    for label, channel in ARMS:
        verdict["arms"][label] = _arm(channel, LEAD)
        verdict["all_leads"][label] = {str(lead): _arm(channel, lead) for lead in LEADS}
    verdict["verdict"] = (
        "PASS"
        if (
            verdict["arms"]["A_all_silent_dropped"]["clears_bar"]
            and verdict["arms"]["B_all_silent_kept_at_cap"]["clears_bar"]
        )
        else "FAIL"
    )
    verdict["verdict_rule"] = (
        "PRE-COMMITTED in the M25 pre-registration: PASS requires the bar cleared under "
        "treatment A (the primary, answering 'given it predicted something, where') AND not "
        "contradicted by treatment B. A result that depends on the all-silent convention is a "
        "choice, not a measurement."
    )

    out = {
        "task": "M24 (ADR-128 (4)) acceptance test read-off + M25 (ADR-130 (4)) conditional score",
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
        "m25_silence": silence,
        "m25_conditional_verdict": verdict,
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
