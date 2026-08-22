"""[S1] Arm A is BITWISE unchanged by the stage-head edit, and arm S starts AT arm A.

Two claims, both about the incumbent, both made by the person who wants the
challenger to look good, and therefore both MEASURED on real held-out windows
rather than argued from the diff:

1. ``stage_scalar=False`` executes the pre-S1 expressions. Verified by re-running
   ``runs/_m10_bitidentity.py --verify``, whose digests were recorded BEFORE the
   M10 edit and therefore long before this one. That script is not duplicated
   here; it is invoked, so there is one set of reference digests and not two.

2. ``stage_scalar=True`` with the head at its ZERO INITIALISATION is bitwise the
   same model. This is the stronger statement and the one arm S's fairness rests
   on: S does not begin the fit at a different place from A, it begins at A
   exactly. If it were merely close, "S beat A" would be partly a statement about
   a different initialisation.

POSITIVE CONTROL. Clause 2 compares digests of a model against digests of a model
and would pass trivially if the stage head were never reached. So the same
checkpoint is reloaded a third time with ONE coefficient moved off zero, and its
digests must DIFFER. If the control agrees, clause 2 proved nothing and this
script exits non-zero saying so.

    .venv/bin/python runs/_s1_bitidentity.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wildfire_nowcast.eval.baseline_run import _windows_for
from wildfire_nowcast.model.kernel import ContagionKernel

OUT = Path("runs/s1_bitidentity.json")
FIRE = "2020_dolan"  # held out under b3e5dadad01eaef9; the M10 reference fire
HORIZON = 3
MEMBERS = 8
SEED = 20260807
N_WINDOWS = 4
CONTROL_COEFFICIENT = 1e-3


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:32]


def _outputs(model: ContagionKernel, windows: list[Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for i, w in enumerate(windows):
        samples = model.predict(w.x0, w.static, w.weather, MEMBERS, HORIZON, SEED + i)
        proba = model.predict_proba(w.x0, w.static, w.weather, HORIZON)
        rows[f"w{i}"] = {
            "t0": int(w.t0),
            "samples_sha256": _digest(samples),
            "samples_sum": int(np.asarray(samples, dtype=np.int64).sum()),
            "proba_sha256": _digest(proba),
            "proba_sum": float(proba.sum()),
        }
    return rows


def _load(ckpt: Path, *, stage: bool) -> ContagionKernel:
    spec = json.loads((ckpt / "model.json").read_text())
    spec["config"]["stage_scalar"] = stage
    return ContagionKernel.from_spec(spec)


def main() -> int:
    ckpt = Path(json.loads(Path("runs/m9_train.json").read_text())["run_dir"])
    windows = _windows_for(FIRE, HORIZON, 40)[:N_WINDOWS]
    if len(windows) < 2:
        raise RuntimeError(f"{FIRE}: {len(windows)} windows at stride 40 — nothing to compare")

    m10 = subprocess.run(  # noqa: S603
        [sys.executable, "runs/_m10_bitidentity.py", "--verify"],
        capture_output=True,
        text=True,
        check=False,
    )
    arm_a_unchanged = m10.returncode == 0 and "BITWISE IDENTICAL" in m10.stdout

    arm_a = _load(ckpt, stage=False)
    arm_s_zero = _load(ckpt, stage=True)
    control = _load(ckpt, stage=True)
    assert control.stage is not None
    with torch.no_grad():
        control.stage.log_amplitude_coeff[0] += CONTROL_COEFFICIENT

    out_a = _outputs(arm_a, windows)
    out_s = _outputs(arm_s_zero, windows)
    out_c = _outputs(control, windows)

    keys = ("samples_sha256", "samples_sum", "proba_sha256", "proba_sum")
    compared = 0
    mismatches: list[str] = []
    for name, row in out_a.items():
        for key in keys:
            compared += 1
            if row[key] != out_s[name][key]:
                mismatches.append(f"{name}.{key}: A={row[key]!r} S0={out_s[name][key]!r}")
    control_differs = sum(
        1 for name, row in out_a.items() for key in keys if row[key] != out_c[name][key]
    )
    # WHICH values the control moves is the informative part, and it decides how
    # much clause 2 is worth. A 1e-3 nudge to the log hazard moves the MARGINAL
    # field in float32 but is far too small to flip a Bernoulli draw, so
    # `samples_*` agreeing is weak evidence and `proba_*` agreeing is strong. The
    # control is therefore required to move the marginal in EVERY window, not
    # merely somewhere: "differs > 0" would be satisfied by one lucky window.
    control_moves_proba = {
        name: bool(row["proba_sha256"] != out_c[name]["proba_sha256"])
        for name, row in out_a.items()
    }
    control_effective = all(control_moves_proba.values()) and bool(control_moves_proba)

    expected = len(out_a) * len(keys)
    payload = {
        "task": "S1 — arm A bitwise unchanged; arm S at zero-init IS arm A",
        "checkpoint": str(ckpt),
        "fire_id": FIRE,
        "n_windows": len(windows),
        "n_members": MEMBERS,
        "n_values_compared": compared,
        "n_values_expected": expected,
        "arm_a_unchanged_since_m10": arm_a_unchanged,
        "m10_verify_stdout": m10.stdout.strip(),
        "zero_init_is_bitwise_arm_a": not mismatches and compared == expected and compared > 0,
        "mismatches": mismatches[:6],
        "positive_control_coefficient": CONTROL_COEFFICIENT,
        "positive_control_n_values_differing": control_differs,
        "positive_control_moves_the_marginal_per_window": control_moves_proba,
        "positive_control_effective": control_effective,
        "arm_a": out_a,
        "arm_s_zero_init": out_s,
        "control": out_c,
    }
    ok = bool(
        payload["arm_a_unchanged_since_m10"]
        and payload["zero_init_is_bitwise_arm_a"]
        and payload["positive_control_effective"]
    )
    payload["outcome"] = "OK" if ok else "FAILED"
    OUT.write_text(json.dumps(payload, indent=1, default=float))

    print(f"arm A unchanged since M10           : {arm_a_unchanged}")
    print(
        f"arm S at zero init is bitwise arm A : {payload['zero_init_is_bitwise_arm_a']} "
        f"({compared}/{expected} values)"
    )
    print(
        f"positive control differs            : {control_differs}/{expected} values "
        f"(coefficient moved by {CONTROL_COEFFICIENT}); moves the marginal in "
        f"{sum(control_moves_proba.values())}/{len(control_moves_proba)} windows"
    )
    for line in mismatches[:6]:
        print(f"  MISMATCH {line}")
    print(f"outcome: {payload['outcome']}   written: {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
