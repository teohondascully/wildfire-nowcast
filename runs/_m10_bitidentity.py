"""[M10] Prove that the `kernel.py` refactor arm B needs leaves arm A BITWISE UNCHANGED.

M10's whole question is A vs B. Arm A is the INCUMBENT checkpoint
(`runs/m9_train.json` -> `m9_probe_s1`), and to give B a place to hook into the
same code path I have to route `ContagionKernel.rollout` and `.predict` through a
new `step_probability_at` indirection. **A refactor of the incumbent's forward
pass, made by the person who wants the challenger to look good, is exactly the
kind of change that must be measured rather than asserted.**

So this script hashes A's ACTUAL OUTPUTS - the sampled C5 ensemble, the mean-field
rollout and the marginal `predict_proba` - on real held-out windows, and is run
BEFORE and AFTER the edit::

    .venv/bin/python runs/_m10_bitidentity.py --write     # before the edit
    .venv/bin/python runs/_m10_bitidentity.py --verify    # after it

`--verify` READS THE STORED VALUES and compares them; it does not merely check
that some set is empty. A mismatch exits non-zero and prints the first offending
key. The digests are of raw bytes, so "bitwise" means bitwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.eval.baseline_run import _windows_for
from wildfire_nowcast.model.kernel import ContagionKernel

OUT = Path("runs/m10_bitidentity.json")
FIRE = "2020_dolan"  # held out under b3e5dadad01eaef9; small enough to be quick
HORIZON = 3
MEMBERS = 8
SEED = 20260807
N_WINDOWS = 4


def _digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:32]


def measure() -> dict[str, Any]:
    ckpt = json.loads(Path("runs/m9_train.json").read_text())["run_dir"]
    model = ContagionKernel.load(Path(ckpt) / "model.json")
    windows = _windows_for(FIRE, HORIZON, 40)[:N_WINDOWS]
    if len(windows) < 2:
        raise RuntimeError(f"{FIRE}: {len(windows)} windows at stride 40 — nothing to compare")
    rows: dict[str, Any] = {}
    for i, w in enumerate(windows):
        samples = model.predict(w.x0, w.static, w.weather, MEMBERS, HORIZON, SEED + i)
        proba_z0 = model.predict_proba(w.x0, w.static, w.weather, HORIZON)
        proba_marginal = model.predict_proba(
            w.x0, w.static, w.weather, HORIZON, n_latent_samples=3, seed=SEED + i
        )
        rows[f"w{i}"] = {
            "t0": int(w.t0),
            "samples_sha256": _digest(samples),
            "samples_sum": int(np.asarray(samples, dtype=np.int64).sum()),
            "proba_z0_sha256": _digest(proba_z0),
            "proba_z0_sum": float(proba_z0.sum()),
            "proba_marginal_sha256": _digest(proba_marginal),
            "proba_marginal_sum": float(proba_marginal.sum()),
        }
    return {
        "checkpoint": str(ckpt),
        "fire_id": FIRE,
        "horizon_h": HORIZON,
        "n_members": MEMBERS,
        "seed": SEED,
        "rows": rows,
        "why": (
            "Arm A's forward pass is refactored in M10 so arm B can share it. These digests "
            "are the evidence that the refactor changed nothing about A, taken before the "
            "edit and re-read after it."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.write == args.verify:
        raise SystemExit("pass exactly one of --write / --verify")

    now = measure()
    if args.write:
        OUT.write_text(json.dumps(now, indent=1, default=float))
        print(f"WROTE {OUT}")
        for key, row in now["rows"].items():
            print(
                f"  {key} t0={row['t0']:>4} samples {row['samples_sha256'][:16]} "
                f"sum {row['samples_sum']}"
            )
        return 0

    before = json.loads(OUT.read_text())
    if before["checkpoint"] != now["checkpoint"]:
        print(f"CHECKPOINT MOVED: {before['checkpoint']} -> {now['checkpoint']}")
        return 2
    keys = (
        "samples_sha256",
        "samples_sum",
        "proba_z0_sha256",
        "proba_z0_sum",
        "proba_marginal_sha256",
        "proba_marginal_sum",
    )
    compared = 0
    for name, row in before["rows"].items():
        after = now["rows"].get(name)
        if after is None:
            print(f"MISSING ROW {name} after the edit")
            return 3
        for key in keys:
            compared += 1
            if row[key] != after[key]:
                print(f"MISMATCH {name}.{key}: before {row[key]!r} after {after[key]!r}")
                return 4
    # POSITIVE CONTROL: a comparison that compared nothing is not a comparison.
    expected = len(before["rows"]) * len(keys)
    if compared != expected or compared == 0:
        print(f"VACUOUS CHECK: compared {compared} values, expected {expected}")
        return 5
    print(
        f"BITWISE IDENTICAL: {compared} values over {len(before['rows'])} windows agree "
        f"(digest of window 0 samples = {now['rows']['w0']['samples_sha256'][:16]}, "
        f"sum = {now['rows']['w0']['samples_sum']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
