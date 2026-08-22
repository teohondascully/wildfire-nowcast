"""[M9] ``growth_calibration`` per BLOCK and per REGIME, from a scored artifact.

Two jobs, and they are separable on purpose.

**1. RE-SCORE AN ARCHIVED RECORD.** ADR-039 (5) made the first moment a G3
condition after every existing result was already on disk. Those artifacts do not
carry ``band_growth_calibration`` - it did not exist in ``eval/`` until M9 - so
the condition cannot be evaluated on them by reading one key. It CAN be
reconstructed exactly, because the record already carries everything the ratio is
made of::

    persistence predicts ZERO new cells in the growth band, by construction
      => its band_area_error_bias == -(mean truth area)
    growth_calibration(model) == 1 + band_area_error_bias(model) / (mean truth area)

The reconstruction is not asserted, it is CHECKED, twice: against sim's
independent recomputation from the C1 tensors
(``reports/figures/s5_block5_anatomy.json``, ``abs_diff 0.0`` on 4 of 4 fires),
and - on any run made after M9 - against the emitted key itself, which must agree
to 1e-9. A reconstruction that is only self-consistent is the kind of check this
project has been bitten by six times.

**2. STRATIFY BY REGIME.** ADR-026 makes dormant-vs-growth reporting the standard
from G3 onward. The same ratio is computed on ALL windows, on GROWTH windows and
on DORMANT windows, per block, because those are three different numbers and the
project has already published one of them as if it were another: ADR-025's
correction ("we over-predict 2.66-3.06x") was an ALL-WINDOW number quoted against
a GROWTH-WINDOW claim, and ADR-035's 3-7x block spread was measured on the
GROWTH-WINDOW stratum. Putting all three in one table is the only way to see
which hypothesis a number belongs to.

On a dormant stratum truth grew nothing, so the RATIO is undefined and this
module reports ``None`` for it and the absolute predicted area beside it. That is
the honest answer - not 0, not inf - and it is what makes "the model cannot say
nothing happens this hour" a measurable statement rather than a metaphor.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wildfire_nowcast.common import dispersion as g3

__all__ = [
    "STRATA",
    "NULL_MODEL",
    "truth_area_mean",
    "growth_calibration_of",
    "regime_table",
    "block_spread",
    "load_results",
]

#: The three strata, in the order a reader should meet them. ``all_windows``
#: FIRST because it is the number that has been quoted, and the point of the
#: table is to show what it is made of.
STRATA: tuple[str, ...] = ("all_windows", "growth_windows", "dormant_windows")

#: The model whose predicted growth is ZERO BY CONSTRUCTION, so its signed area
#: error IS the truth. Named once; `eval/baseline_run.NULL_MODELS` declares the
#: same fact for C6.2 and a second spelling is how the two would drift.
NULL_MODEL = "persistence"

_BIAS_KEY = "band_area_error_bias"
_EMITTED_KEY = f"band_{g3.FIRST_MOMENT_KEY}"
_TRUTH_SUM_KEY = "band_first_moment_truth_area_sum"


def load_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _row(per_fire_entry: Mapping[str, Any], model: str, stratum: str) -> dict[str, Any]:
    return ((per_fire_entry.get("models", {}).get(model) or {}).get(stratum)) or {}


#: The stratum on which truth is ZERO BY THE STRATUM'S OWN DEFINITION. A window
#: enters it when ``truth_growth_cells() == 0``, and every ``growth_band`` cell is
#: unburned at t0, so band truth growth is zero there too - no measurement
#: needed, and none available: persistence's area error is identically 0 on this
#: stratum, so ``_area_error_decomposition`` returns ``None`` for every term and
#: the NULL-model reconstruction goes blind exactly where hypothesis (A) lives.
#: **That is a property of the reconstruction, not of the world**, and it is
#: handled by construction rather than by giving up.
DORMANT_STRATUM = "dormant_windows"


def truth_area_mean(per_fire_entry: Mapping[str, Any], stratum: str) -> float | None:
    """Mean truth event area per scored lead, in the growth band.

    ``0.0`` on the dormant stratum by construction. Otherwise from the NULL
    model's signed area error, which IS the truth because persistence ignites
    nothing. ``None`` only when the stratum was not scored at all - 0 and absent
    are different facts and this project has conflated them before.
    """
    if stratum == DORMANT_STRATUM:
        return 0.0
    row = _row(per_fire_entry, NULL_MODEL, stratum)
    bias = row.get(_BIAS_KEY)
    return None if bias is None else -float(bias)


def growth_calibration_of(
    per_fire_entry: Mapping[str, Any], model: str, stratum: str
) -> dict[str, Any]:
    """One cell of the table: the ratio, its two parts, and how it was obtained.

    ``source`` is ``"emitted"`` when the artifact carries
    ``band_growth_calibration`` (post-M9 runs) and ``"reconstructed"`` otherwise.
    When BOTH are available they are compared and the disagreement is reported -
    never silently preferred - because a reconstruction that has quietly stopped
    matching its measurement is exactly the failure this module would otherwise
    hide.
    """
    row = _row(per_fire_entry, model, stratum)
    null_row = _row(per_fire_entry, NULL_MODEL, stratum)
    out: dict[str, Any] = {
        "model": model,
        "stratum": stratum,
        "n_windows": row.get("n_windows"),
        "growth_calibration": None,
        "truth_area_mean": None,
        "pred_area_mean": None,
        "source": "unscored",
        "emitted": row.get(_EMITTED_KEY),
        "reconstructed": None,
        "agreement": None,
    }
    if not row:
        return out

    del null_row
    truth_mean = truth_area_mean(per_fire_entry, stratum)
    bias = row.get(_BIAS_KEY)
    if truth_mean is not None and bias is not None:
        pred_mean = truth_mean + float(bias)
        out["truth_area_mean"] = truth_mean
        out["pred_area_mean"] = pred_mean
        # `growth_calibration` returns None at a zero denominator, which is the
        # right answer on the dormant stratum: the model's absolute predicted
        # area is the measurement there, and a ratio against nothing is not.
        out["reconstructed"] = g3.growth_calibration(pred_mean, truth_mean)

    emitted = out["emitted"]
    reconstructed = out["reconstructed"]
    if emitted is not None and reconstructed is not None:
        out["agreement"] = abs(float(emitted) - float(reconstructed))
    out["growth_calibration"] = emitted if emitted is not None else reconstructed
    out["source"] = (
        "emitted"
        if emitted is not None
        else (
            "reconstructed"
            if reconstructed is not None
            else ("undefined_ratio" if out["pred_area_mean"] is not None else "unscored")
        )
    )
    return out


def regime_table(
    results: Mapping[str, Any],
    models: Sequence[str],
    *,
    strata: Sequence[str] = STRATA,
) -> dict[str, Any]:
    """``growth_calibration`` for every (model, stratum, held-out block).

    The block id, the window count and the truth area travel WITH every ratio.
    A calibration ratio quoted without its denominator is how this project
    published "2.66-3.06x over-prediction" as a property of the model when it was
    a property of the dormant stratum (ADR-025 (1)).
    """
    per_fire = results["per_fire"]
    rows: list[dict[str, Any]] = []
    for fid, entry in per_fire.items():
        for model in models:
            for stratum in strata:
                cell = growth_calibration_of(entry, model, stratum)
                rows.append(
                    {
                        "fire_id": fid,
                        "spatial_block_id": int(entry["spatial_block_id"]),
                        "n_windows_total": entry.get("n_windows"),
                        "n_growth_windows": entry.get("n_growth_windows"),
                        "zero_growth_fraction": entry.get("zero_growth_fraction"),
                        **cell,
                    }
                )
    disagreements = [r for r in rows if r["agreement"] is not None and r["agreement"] > 1e-9]
    reconciliation = _stratum_reconciliation(rows)
    return {
        "split_fingerprint": (results.get("split_before") or {}).get("fingerprint"),
        "strata": list(strata),
        "models": list(models),
        "rows": rows,
        "stratum_reconciliation": reconciliation,
        "stratum_reconciliation_note": (
            "growth + dormant must equal all, in CELL-LEADS, for both truth and prediction. "
            "This is an arithmetic identity the record has no obligation to satisfy if any of "
            "the three strata were reconstructed wrongly, so it is a READ-A-VALUE check on the "
            "reconstruction and on the claim that dormant truth is exactly 0 — not a proof that "
            "some set is empty. `max_abs_residual` is the number to look at."
        ),
        "reconstruction_disagreements": disagreements,
        "reconstruction_note": (
            "`reconstructed` uses persistence's band_area_error_bias as -(mean truth area), "
            "which is exact because persistence ignites zero cells by construction (C6.2 "
            "NULL_MODELS). Cross-checked against simviz's independent recomputation from the "
            "C1 tensors at abs_diff 0.0 on 4 of 4 held-out fires "
            "(reports/figures/s5_block5_anatomy.json). Any cell where BOTH the emitted and the "
            "reconstructed value exist and differ by more than 1e-9 is listed above; an empty "
            "list here is only meaningful when `emitted` is non-null somewhere, which is why "
            "the source of every cell is printed."
        ),
    }


def _stratum_reconciliation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``growth + dormant == all`` in cell-leads, for truth AND for prediction.

    The identity that makes the whole table trustworthy. It is checked rather
    than asserted, and its worst residual is reported as a NUMBER, because "no
    disagreements found" is the shape of claim this project has been wrong about
    four times.
    """
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r["fire_id"], r["model"]), {})[r["stratum"]] = r
    out: list[dict[str, Any]] = []
    for (fire, model), strata in sorted(by_key.items()):
        need = ("all_windows", "growth_windows", DORMANT_STRATUM)
        if not all(s in strata for s in need):
            continue
        totals = {}
        for what in ("truth_area_mean", "pred_area_mean"):
            parts = {}
            for s in need:
                value, n = strata[s].get(what), strata[s].get("n_windows")
                parts[s] = None if value is None or n is None else float(value) * float(n)
            if any(v is None for v in parts.values()):
                continue
            totals[what] = {
                "all": parts["all_windows"],
                "growth_plus_dormant": parts["growth_windows"] + parts[DORMANT_STRATUM],
                "residual": parts["all_windows"] - parts["growth_windows"] - parts[DORMANT_STRATUM],
            }
        if totals:
            out.append({"fire_id": fire, "model": model, **totals})
    residuals = [
        abs(v["residual"])
        for row in out
        for k, v in row.items()
        if isinstance(v, dict) and "residual" in v
    ]
    return {
        "rows": out,
        "n_checked": len(out),
        "max_abs_residual": max(residuals) if residuals else None,
    }


def block_spread(table: Mapping[str, Any], model: str, stratum: str) -> dict[str, Any]:
    """max/min of ``growth_calibration`` across BLOCKS - the quantity ADR-035 put at 3-7x.

    Reported as a plain ratio of the extremes AND in log units, because the
    hypothesis being tested is about whether restricting to growth windows makes
    the spread COLLAPSE, and "collapse" has to be a number.
    """
    by_block: dict[int, list[float]] = {}
    for r in table["rows"]:
        if r["model"] != model or r["stratum"] != stratum:
            continue
        value = r["growth_calibration"]
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            continue
        by_block.setdefault(int(r["spatial_block_id"]), []).append(float(value))
    means = {b: sum(v) / len(v) for b, v in by_block.items()}
    if not means:
        return {"model": model, "stratum": stratum, "n_blocks": 0, "spread": None}
    lo_block = min(means, key=lambda b: means[b])
    hi_block = max(means, key=lambda b: means[b])
    lo, hi = means[lo_block], means[hi_block]
    return {
        "model": model,
        "stratum": stratum,
        "n_blocks": len(means),
        "per_block": means,
        "equal_block_mean": sum(means.values()) / len(means),
        "min_block": lo_block,
        "max_block": hi_block,
        "spread": (hi / lo) if lo > 0 else None,
        "spread_log": (math.log(hi) - math.log(lo)) if lo > 0 else None,
    }
