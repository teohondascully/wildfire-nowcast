"""[S1] Score ONE checkpoint over a set of fires, emitting ``eval.response`` window rows.

The rows are what ``eval.stage.stage_decay_by_block`` consumes, and truth and the
forecast are read from the SAME rows, so the two arms and the truth are
necessarily measured on the same windows. That is not a convenience: the
criterion is a PAIRED per-block sign, and a pairing across two different window
sets is not a pairing.

Every arm is scored with the same members, the same per-window seed and the same
stride, so ``truth_growth`` is bit-identical across arms and only
``model_growth`` moves. The ellipse is NOT run here - ADR-061 (6)'s criterion is
S against A against truth, and the ellipse would be a third of the compute for a
column nothing reads.

    .venv/bin/python runs/_s1_score.py --checkpoint runs/<dir> --fold 0 \
        --out runs/s1_rows_a_s1_f0.json

**THE FOLD IS AN ARGUMENT HERE TOO, AND IT IS THE SAME OBJECT AS THE FIT.**
``--fold k`` resolves ONE ``SplitContext`` over ``data/norm_stats_heldout_fold{k}.json``
(C8.2), and the fires scored are that context's HELD-OUT fires - never a list a
caller typed. The checkpoint is then checked against that context with
``assert_model_split_matches``, which HARD FAILS if the model was trained under a
different partition. That check is the whole reason a 15-cell matrix can be run
by hand at all: the way this experiment fails silently is fold k's checkpoint
being scored on fold j's held-out fires, which produces a perfectly plausible
number on fires the model trained on, and no per-window check could see it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.paths import fire_tensor_path, repo_relative
from wildfire_nowcast.common.splits import resolve_split_context
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.eval.baseline_run import load_splits
from wildfire_nowcast.eval.masks import default_band_radius
from wildfire_nowcast.eval.reporting import assert_model_split_matches, scoring_code_fingerprint
from wildfire_nowcast.eval.response import window_row
from wildfire_nowcast.model.inputs import iter_windows
from wildfire_nowcast.model.kernel import ContagionKernel
from wildfire_nowcast.model.train import stamp_expected_c6_3

HORIZON = 3
MEMBERS = 24
SEED = 20260807
STRIDE = 2
#: The five leave-fold-out normalisations (task D12) and their fingerprints
#: of record (ADR-065 (1)). Duplicated from `_s1_train.py` deliberately: the two
#: scripts must agree about which file is fold k, and a shared import would make
#: them agree by construction even if the file on disk moved.
FOLD_STATS = {k: Path(f"data/norm_stats_heldout_fold{k}.json") for k in range(5)}
FOLD_FINGERPRINTS = {
    0: "485706acb537b9ac",
    1: "606d5904b7254f1a",
    2: "27d06a388cfd8946",
    3: "b3e5dadad01eaef9",
    4: "5847b50cda6e0e9a",
}


def _windows(fire_id: str, stride: int) -> list[Any]:
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        return list(iter_windows(ds, HORIZON, stride=stride, fire_id=fire_id))
    finally:
        ds.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument(
        "--fold",
        type=int,
        choices=sorted(FOLD_STATS),
        required=True,
        help="the HELD-OUT fold; its held-out fires are what gets scored",
    )
    args = ap.parse_args(argv)

    scoring_before = scoring_code_fingerprint()
    ctx = resolve_split_context(stats_path=FOLD_STATS[args.fold])
    split = stamp_expected_c6_3(ctx.fingerprint())
    if split["fingerprint"] != FOLD_FINGERPRINTS[args.fold]:
        raise SystemExit(
            f"fold {args.fold} stats reproduce {split['fingerprint']}, not the "
            f"{FOLD_FINGERPRINTS[args.fold]} of record (ADR-065 (1)). Refusing to score."
        )
    stats = ctx.norm_stats()
    splits = load_splits([int(f) for f in stats["train_folds"]])
    wanted = {s.fire_id for s in splits if not s.is_train}
    chosen = [s for s in splits if s.fire_id in wanted]
    if len(chosen) != len(wanted):
        missing = sorted(wanted - {s.fire_id for s in chosen})
        raise SystemExit(f"unknown fire ids: {missing}")

    model = ContagionKernel.load(Path(args.checkpoint) / "model.json")
    # C8 - THE CHECK THAT MAKES A 15-CELL MATRIX RUNNABLE BY HAND. Scoring fold
    # k's checkpoint on fold j's held-out fires is the ADR-015 defect with a
    # keyboard slip as its cause, and it yields a plausible number on trained-on
    # fires. This raises instead.
    model_split = assert_model_split_matches(
        model, split, name=f"arm checkpoint {repo_relative(args.checkpoint)}"
    )
    band_radius = default_band_radius(HORIZON)
    rows: list[dict[str, Any]] = []
    per_fire: dict[str, Any] = {}
    t0 = time.time()
    for fire in chosen:
        n0, t_fire = len(rows), time.time()
        for i, w in enumerate(_windows(fire.fire_id, args.stride)):
            samples = model.predict(w.x0, w.static, w.weather, MEMBERS, HORIZON, SEED + i)
            row = window_row(
                w,
                samples,
                band_radius=band_radius,
                fire_id=fire.fire_id,
                spatial_block_id=fire.spatial_block_id,
            )
            row["is_train_fire"] = bool(fire.is_train)
            rows.append(row)
        per_fire[fire.fire_id] = {
            "spatial_block_id": fire.spatial_block_id,
            "cv_fold": fire.cv_fold,
            "is_train_fire": bool(fire.is_train),
            "n_windows": len(rows) - n0,
            "elapsed_s": round(time.time() - t_fire, 1),
        }
        print(
            f"{fire.fire_id:<32} block {fire.spatial_block_id:>2} fold {fire.cv_fold} "
            f"{len(rows) - n0:>4} windows  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    elapsed = time.time() - t0
    payload = {
        "task": "S1 — window rows for one arm",
        "checkpoint": repo_relative(args.checkpoint),
        "model_kind": model.kind,
        "stage_scalar": bool(model.stage is not None),
        "stage_report": (None if model.stage is None else model.stage.report()),
        "model_provenance": dict(model.provenance),
        "fold": args.fold,
        "stats_path": repo_relative(FOLD_STATS[args.fold]),
        "split_fingerprint": split["fingerprint"],
        "split": split,
        "model_split_check": model_split,
        "heldout_fire_ids": sorted(wanted),
        "scoring_code_before": scoring_before,
        "scoring_code_after": scoring_code_fingerprint(),
        "horizon_h": HORIZON,
        "n_members": MEMBERS,
        "seed": SEED,
        "stride": args.stride,
        "band_radius_cells": band_radius,
        "n_rows": len(rows),
        "elapsed_s": round(elapsed, 1),
        "seconds_per_window": round(elapsed / max(1, len(rows)), 4),
        "per_fire": per_fire,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, default=float))
    print(
        f"\n{len(rows)} rows in {elapsed:.0f}s ({payload['seconds_per_window']:.4f} s/window) "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
