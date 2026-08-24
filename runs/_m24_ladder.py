"""[M24] ADR-128 (4)'s ACCEPTANCE TEST for `front_distance_crps`, on two ladders.

Scores every rung of M11's AREA family and a NEW exact-area DISPLACEMENT family
on the same held-out fold-3 windows, through the same C6 `evaluate()` call, and
records the incumbent `best_member_iou_shape_masked` and the ADR-053 channels
(Brier, arrival CRPS) in the SAME rows. So "the new score orders the ladder and
Brier does not" is one table, on one set of episodes, not a comparison against a
remembered number.

Base forecast: `runs/m9_probe_s1-20260809-222011`, which is the checkpoint M11
itself wrapped. Same stride, members, seed and horizon as M11.

    .venv/bin/python runs/_m24_ladder.py
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY
from wildfire_nowcast.common.paths import fire_tensor_path, norm_stats_path
from wildfire_nowcast.common.zarr_io import open_tensor, read_norm_stats
from wildfire_nowcast.eval.baseline_run import load_splits
from wildfire_nowcast.eval.metrics import evaluate
from wildfire_nowcast.eval.reporting import (
    common_code_fingerprint,
    scoring_code_fingerprint,
    split_fingerprint,
)
from wildfire_nowcast.model.api import load_model
from wildfire_nowcast.model.degrade import (
    MODE_AREA,
    MODE_SHAPE,
    MODE_SHIFT,
    CellOrder,
    degrade_samples,
    increment_overlap,
    realised_displacement_km,
)
from wildfire_nowcast.model.inputs import iter_windows

#: [M25] Written GZIPPED, and the plain form is never materialised. ADR-130 (5)
#: records a verification diff that read "identical" because the report script
#: had crashed looking for a `.json` beside a `.json.gz`; the 6.5 MB plain file
#: existed only to be deleted again. The report reads either form.
OUT = Path("runs/m24_frontdist.json.gz")
BASE_CHECKPOINT = "runs/m9_probe_s1-20260809-222011"
HORIZON = 3
MEMBERS = 24
SEED = 20260807
STRIDE = 2

AREA_LEVELS = (0.10, 0.28, 0.50, 0.75, 0.90, 1.00, 1.15, 1.50, 2.00, 2.80, 5.00, 8.00)
SHIFT_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
#: M11's OWN shape family, re-scored here for one reason: `shape_f1.000` is the
#: rung ADR-128 (2) is about - 98.2% of the increment relocated at EXACT area,
#: where `iou_shape_masked_3h` reads 1.9749 against a 2.0 bar. Any claim that a
#: new channel is more powerful has to be readable on THAT rung, not only on a
#: family this task invented.
SHAPE_LEVELS = (0.05, 0.15, 0.40, 1.00)


def rung_rows(samples: np.ndarray, window: Any, order: CellOrder) -> dict[str, np.ndarray]:
    rungs: dict[str, np.ndarray] = {"reference": samples}
    for level in AREA_LEVELS:
        rungs[f"area_k{level:.3f}"] = degrade_samples(
            samples, window.x0, mode=MODE_AREA, level=level, order=order
        )
    for level in SHIFT_LEVELS:
        rungs[f"shift_d{level:04.1f}"] = degrade_samples(
            samples, window.x0, mode=MODE_SHIFT, level=level, order=order
        )
    for level in SHAPE_LEVELS:
        rungs[f"shape_f{level:.3f}"] = degrade_samples(
            samples, window.x0, mode=MODE_SHAPE, level=level, order=order
        )
    return rungs


def score(samples: np.ndarray, window: Any) -> dict[str, Any]:
    res = evaluate(samples, window.truth, x0=window.x0, leads=(1, 2, 3))
    band = res["by_mask"]["growth_band"]
    front = band["front_distance_crps"]
    burned0 = window.x0 > 0
    row: dict[str, Any] = {
        "iou_shape_masked": [band[f"{GATE_CRITERION_KEY}_by_horizon"][h - 1] for h in (1, 2, 3)],
        "brier": [band["brier_by_lead"].get(h) for h in (1, 2, 3)],
        "arrival_crps": band["arrival_crps"],
        "front": {
            str(h): {
                # [M25, ADR-130 (4)] The two questions the score conflates, kept
                # apart in the record: `n_empty` / `n_members` is P(silent), the
                # `*_cond` terms are the score over the members that spoke, and
                # `combined_cond is None` on a DEFINED lead means EVERY member
                # was silent - a case that is counted, never dropped in silence.
                k: front["by_horizon"][str(h)][k]
                for k in (
                    "combined",
                    "truth_to_pred",
                    "pred_to_truth",
                    "combined_cond",
                    "truth_to_pred_cond",
                    "pred_to_truth_cond",
                    "censored_fraction",
                    "censored_fraction_cond",
                    "n_empty_members",
                    "n_members",
                )
            }
            for h in (1, 2, 3)
        },
    }
    # area severity, measured on the samples inside the SAME band the scores use
    from wildfire_nowcast.eval.masks import default_band_radius, growth_band

    band_mask = growth_band(window.x0, default_band_radius(HORIZON)) & ~burned0
    member_inc = (samples > 0) & (~burned0)[None, None] & band_mask[None, None]
    truth_inc = (window.truth > 0) & ~burned0[None] & band_mask[None]
    row["pred_cells"] = [float(member_inc[:, h - 1].sum()) / samples.shape[0] for h in (1, 2, 3)]
    row["truth_cells"] = [float(truth_inc[h - 1].sum()) for h in (1, 2, 3)]
    return row


def main() -> int:
    started = time.time()
    split_before = split_fingerprint()
    scoring_before = scoring_code_fingerprint()["fingerprint"]
    common_before = common_code_fingerprint()["fingerprint"]

    model = load_model(BASE_CHECKPOINT)
    stats = read_norm_stats(norm_stats_path())
    train_folds = [int(f) for f in stats["train_folds"]]
    splits = [s for s in load_splits(train_folds) if not s.is_train]
    per_fire: dict[str, Any] = {}
    n_windows_total = 0
    for split in sorted(splits, key=lambda s: s.fire_id):
        ds = open_tensor(fire_tensor_path(split.fire_id))
        try:
            windows = [
                w
                for w in iter_windows(ds, HORIZON, stride=STRIDE, fire_id=split.fire_id)
                if w.truth_growth_cells() > 0
            ]
        finally:
            ds.close()
        rows: dict[str, list[dict[str, Any]]] = {}
        severity: dict[str, list[dict[str, Any]]] = {}
        for i, window in enumerate(windows):
            samples = model.predict(
                window.x0, window.static, window.weather, MEMBERS, HORIZON, SEED + i
            )
            order = CellOrder(samples, window.x0)
            for name, arr in rung_rows(samples, window, order).items():
                rows.setdefault(name, []).append(score(arr, window))
                severity.setdefault(name, []).append(
                    {
                        "displacement_km": realised_displacement_km(arr, samples, window.x0),
                        "increment_iou": increment_overlap(arr, samples, window.x0).iou,
                    }
                )
        per_fire[split.fire_id] = {
            "spatial_block_id": split.spatial_block_id,
            "n_windows": len(windows),
            "rows": rows,
            "severity": severity,
        }
        n_windows_total += len(windows)
        print(f"{split.fire_id}: {len(windows)} growth windows", flush=True)

    payload = {
        "task": "M24 (ADR-128 (4)) - front_distance_crps acceptance test",
        "base_checkpoint": BASE_CHECKPOINT,
        "horizon_h": HORIZON,
        "n_members": MEMBERS,
        "seed": SEED,
        "stride": STRIDE,
        "area_levels": list(AREA_LEVELS),
        "shift_levels": list(SHIFT_LEVELS),
        "shape_levels": list(SHAPE_LEVELS),
        "n_growth_windows": n_windows_total,
        "n_blocks": len({v["spatial_block_id"] for v in per_fire.values()}),
        "split_before": split_before,
        "split_after": split_fingerprint(),
        "scoring_fingerprint_before": scoring_before,
        "scoring_fingerprint_after": scoring_code_fingerprint()["fingerprint"],
        "common_fingerprint_before": common_before,
        "common_fingerprint_after": common_code_fingerprint()["fingerprint"],
        "elapsed_s": round(time.time() - started, 1),
        "per_fire": per_fire,
    }
    with gzip.open(OUT, "wt") as handle:
        json.dump(payload, handle, default=float)
    print("windows", n_windows_total, "elapsed_s", payload["elapsed_s"])
    print("split", payload["split_before"], payload["split_after"])
    print("scoring", scoring_before, payload["scoring_fingerprint_after"])
    print("common", common_before, payload["common_fingerprint_after"])
    print("bytes", OUT.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
