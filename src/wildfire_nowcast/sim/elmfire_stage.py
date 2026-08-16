"""[E1] Does the PHYSICS baseline decelerate? ELMFIRE against ``stage_decay``.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
ADR-064 promotes ELMFIRE from "a benchmark we owe" to the experiment that decides
whether this project's negative result is about OUR kernel or about the whole
model class. The reasoning is one sentence: **ELMFIRE is Rothermel — rate of
spread applied along a perimeter — so it is perimeter-proportional too.** If it
also accelerates as a fire ages, "a contagion kernel cannot decelerate" stops
being a defect report about us and becomes a statement about the field's standard
tool. If it decelerates, the defect is ours and we know what to copy.

**This is NOT G5.** No gate is attempted, no C6 gate criterion is computed, and
``eval.stage.licence`` still refuses ``stage_decay`` the right to adjudicate
anything. Every artifact this module writes carries that refusal.

HOW THE COMPARISON IS KEPT HONEST (ADR-064 (6))
-----------------------------------------------
1. **One estimator, not two.** ``stage_decay`` is imported from
   :mod:`wildfire_nowcast.eval.stage` and called. Nothing here re-implements it.
   A second implementation is a C0 breach and is how a comparison silently
   becomes two different measurements.
2. **The same windows.** Windows come from
   :func:`wildfire_nowcast.model.inputs.iter_windows` at ``horizon_h=3``,
   ``stride=2`` — the identical call ``runs/_s1_score.py`` made for arm A and
   arm S — and rows are built by the identical
   :func:`wildfire_nowcast.eval.response.window_row`. ``truth_growth`` therefore
   comes out bit-identical to the arm A/S rows, which is checked rather than
   hoped (:func:`check_truth_pairing`).
3. **ELMFIRE is not handicapped.** It reads native 30 m LANDFIRE fuels, canopy
   and topography with crown fire ON and stock Rothermel parameters; only its
   OUTPUT is coarsened to our 1 km lattice. Nothing here tunes it. The full list
   of places our world and its world do not line up is
   :data:`wildfire_nowcast.sim.elmfire.MAPPING_COMPROMISES`, and it is copied
   into every artifact.

WHY A WHOLE-DOMAIN FUEL FETCH
-----------------------------
``sim.landfire.fetch_native_stack`` is keyed on the exact window geometry, and
every ``t0`` has a different window, so scoring 1372 windows the naive way is
~11k LFPS requests and tens of GB of cache. This module fetches each fire's
WHOLE domain once at 30.303 m and slices it per window. That is exact, not an
approximation — ``fine_grid`` shares the north-west corner and refines by an
integer, and ``window_grids`` snaps to whole 1 km cells, so each window's fine
grid is a sub-grid at offset ``(row0 * refine, col0 * refine)``.
:func:`verify_slice_equivalence` is the positive control: it re-fetches one
window straight from LFPS and requires every layer to match the slice EXACTLY,
and it must also detect a deliberately shifted slice, or it is not a check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import fire_tensor_path, norm_stats_path
from wildfire_nowcast.common.zarr_io import open_tensor, read_norm_stats
from wildfire_nowcast.eval.baseline_run import load_splits
from wildfire_nowcast.eval.masks import default_band_radius
from wildfire_nowcast.eval.reporting import scoring_code_fingerprint, split_fingerprint
from wildfire_nowcast.eval.response import window_row
from wildfire_nowcast.eval.stage import (
    STAGE_DECAY_KEY,
    licence,
    sign_test,
    stage_decay_by_block,
)
from wildfire_nowcast.model.inputs import forecast_inputs, iter_windows
from wildfire_nowcast.sim.coarsen import DEFAULT_REFINE, fine_grid
from wildfire_nowcast.sim.elmfire import (
    MAPPING_COMPROMISES,
    ElmfireConfig,
    ElmfireNativeModel,
    InputMode,
    Window,
    find_binary,
    window_grids,
)
from wildfire_nowcast.sim.landfire import (
    CACHE_ROOT,
    NATIVE_LAYERS,
    NativeStack,
    fetch_layer,
    fetch_native_stack,
)

__all__ = [
    "HORIZON_H",
    "STRIDE",
    "SEED",
    "DEFAULT_MEMBERS",
    "TASK",
    "DomainStack",
    "build_domain_stack",
    "domain_stack_path",
    "held_out_fires",
    "verify_slice_equivalence",
    "window_t0s",
    "run_fire",
    "check_truth_pairing",
    "score",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The three constants ``runs/_s1_score.py`` used for arm A and arm S. They are
#: repeated here rather than imported because that file is untracked scratch; the
#: pairing they buy is CHECKED against the arm A rows by
#: :func:`check_truth_pairing`, so a drift is a failed check, not a silent one.
HORIZON_H: Final = 3
STRIDE: Final = 2
SEED: Final = 20260807

#: Members per window. ELMFIRE itself is deterministic; the ensemble is OUR
#: construction (ws/wd/moisture perturbed from a declared PDF), which is recorded
#: in ``MAPPING_COMPROMISES``. 4 is a COST choice, not a scientific one, so every
#: run also reports ``stage_decay`` recomputed on the nested member prefixes
#: 1, 2, ... M — if the verdict moved with the member count we would have to say
#: so, and that is cheaper to measure than to argue about.
DEFAULT_MEMBERS: Final = 4

TASK: Final = "E1 — does the physics baseline decelerate? (ADR-064)"

#: Held-out block ids of the split of record, in the order ADR-064 quotes them.
HELD_OUT_BLOCKS: Final = (4, 5, 6, 7, 12)

#: ADR-064 (4), copied verbatim so the artifact carries the prediction it tests.
PRE_REGISTRATION: Final = {
    "source": "ADR-064 (4), written before any ELMFIRE output existed",
    "E-P1": (
        "ELMFIRE's stage_decay is POSITIVE (accelerating) on >= 4 of the 5 "
        "currently held-out blocks"
    ),
    "E-P2": "its magnitude sits BETWEEN our kernel's and truth's",
    "E-P3": (
        "it gets DOLAN (block 6) closer to right than arm S does, because Dolan "
        "is remote and least suppressed"
    ),
    "decision_rule": (
        ">=4/5 positive -> the defect belongs to the model CLASS. >=4/5 negative "
        "-> E-P1 REFUTED and the defect is ours. 3/5 either way -> not_a_verdict."
    ),
}

#: A window whose per-member WALL-CLOCK time reaches this fraction of ELMFIRE's
#: ``MAX_RUNTIME`` is treated as possibly truncated and is refused. 0.9 rather
#: than 1.0 because the cap is checked at the top of a timestep, so a run can
#: exceed it slightly, and because our own measurement is the parent's view of
#: the subprocess and includes a little setup.
TRUNCATION_FRACTION: Final = 0.9

TRUNCATION_MARGIN_NOTE: Final = (
    "235 s/member against a 600 s cap on the four blocks scored here (2.55x "
    "margin, 0 windows within 50% of the cap), so those blocks are cap-"
    "independent; Creek reaches 385 s/member by t0=124 and rises, which is why "
    "it is the block that did not finish"
)

#: Places E1's configuration departs from ELMFIRE's OWN namelist defaults, over
#: and above ``sim.elmfire.MAPPING_COMPROMISES``. ADR-064 (6) says "default
#: Rothermel parameters, no tuning by us", so a deviation that is ours has to be
#: named even when it is the conservative one.
DECLARED_DEVIATIONS_FROM_ELMFIRE_DEFAULTS: Final = [
    {
        "namelist": "&TIME_CONTROL SIMULATION_DT",
        "elmfire_default": 5.0,
        "ours": 1.0,
        "where_set": "sim.elmfire.ElmfireConfig.simulation_dt_s, chosen at S3/S4",
        "direction": (
            "FINER than the default, i.e. a more accurate level-set solve, not a "
            "faster or more favourable one. It is not a Rothermel parameter."
        ),
        "cost": (
            "5x the runtime. This is the single reason block 4 (Creek, 785 windows, "
            "78% of the total compute) is expensive: at t0=62 one window already "
            "costs ~717 s. It was NOT changed mid-experiment — altering a solver "
            "setting after seeing that a run is slow is how one experiment becomes "
            "two. A future full-14-block run should decide it BEFORE starting."
        ),
    },
    {
        "namelist": "&SIMULATOR MAX_RUNTIME",
        "elmfire_default": 999999.0,
        "ours": 600.0,
        "where_set": "sim.elmfire.ElmfireConfig.max_runtime_s, chosen at S3/S4",
        "direction": (
            "DANGEROUS, and it is why this is declared rather than left implicit. "
            "`elmfire_level_set.f90:1263` compares it against WALL-CLOCK elapsed "
            "time and, when exceeded, sets `T = TSTOP + 1` and stops the "
            "simulation early. Nothing in the rasters says so. A capped run "
            "therefore UNDER-reports late-window growth, which biases stage_decay "
            "DOWN — toward deceleration, i.e. AGAINST E-P1 — and, because the cap "
            "is wall-clock, whether it bites depends on machine load and worker "
            "count rather than on the fire."
        ),
        "cost": (
            "MEASURED, not assumed: across the scored blocks the worst window "
            f"costs {TRUNCATION_MARGIN_NOTE}. Any window within "
            "`TRUNCATION_FRACTION` of the cap is refused by `score`."
        ),
    },
    {
        "namelist": "&SPOTTING ENABLE_SPOTTING",
        "elmfire_default": False,
        "ours": False,
        "where_set": "upstream default, elmfire_namelists.f90:695 — untouched",
        "direction": (
            "we take ELMFIRE's own default. Recorded because our kernel HAS a "
            "long-range spot component, so the comparison is a spotting model "
            "against a non-spotting one. Carried from S4's fairness note."
        ),
        "cost": "none",
    },
]

#: The published reference rows, same estimator, same held-out blocks
#: (STATE.md / ``runs/s1.json``). Carried so a reader never has to fetch them.
REFERENCE_STAGE_DECAY: Final[dict[str, dict[int, float]]] = {
    "truth": {4: -2.1751, 5: -1.1836, 6: 1.4560, 7: -0.0398, 12: -1.8179},
    "arm_a": {4: 0.5771, 5: 0.4637, 6: 1.1453, 7: 1.0891, 12: 0.1568},
    "arm_s": {4: 0.1960, 5: -1.2667, 6: -0.2844, 7: 0.0191, 12: -0.5305},
}


# --------------------------------------------------------------------------
# whole-domain 30 m stack, sliced per window
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainStack:
    """One fire's whole C1 domain at the fine lattice, plus its provenance.

    ``layers`` are ``int16[ny_fine, nx_fine]`` on ``fine``, which is
    ``fine_grid(coarse_domain, refine)``. Slicing is exact; see the module
    docstring and :func:`verify_slice_equivalence`.
    """

    fire_id: str
    coarse: Grid
    fine: Grid
    refine: int
    layers: dict[str, np.ndarray]
    provenance: dict[str, Any]

    def slice_for(self, window: Window) -> NativeStack:
        """The window's own :class:`NativeStack`, cut out of the domain stack."""
        if window.refine != self.refine:
            raise ValueError(
                f"window refine={window.refine} but the domain stack is at "
                f"refine={self.refine}; a mismatched lattice is not sliceable"
            )
        r0 = window.row0 * self.refine
        c0 = window.col0 * self.refine
        r1 = r0 + window.fine.ny
        c1 = c0 + window.fine.nx
        if r1 > self.fine.ny or c1 > self.fine.nx:
            raise ValueError(
                f"window fine block rows {r0}:{r1} cols {c0}:{c1} runs outside the "
                f"domain fine grid {self.fine.shape}"
            )
        layers = {k: np.ascontiguousarray(v[r0:r1, c0:c1]) for k, v in self.layers.items()}
        return NativeStack(
            grid=window.fine,
            layers=layers,
            provenance={
                **self.provenance,
                "sliced_from_domain": True,
                "slice_rows": [r0, r1],
                "slice_cols": [c0, c1],
            },
        )


def domain_stack_path(fire_id: str, refine: int) -> Path:
    return CACHE_ROOT / f"domain_{fire_id}_r{refine}.npz"


def _fire_year(fire_id: str) -> int:
    head = fire_id.split("_", 1)[0]
    if not head.isdigit():
        raise ValueError(f"cannot read a fire year off {fire_id!r}")
    return int(head)


def build_domain_stack(
    fire_id: str,
    *,
    refine: int = DEFAULT_REFINE,
    use_cache: bool = True,
) -> DomainStack:
    """Fetch (or load) every native layer for a fire's WHOLE domain at 30.303 m.

    One LFPS request per layer per fire instead of one per layer per window.
    """
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        coarse = Grid.from_dataset(ds)
    finally:
        ds.close()
    fine = fine_grid(coarse, refine)
    cache = domain_stack_path(fire_id, refine)
    if use_cache and cache.exists():
        blob = np.load(cache, allow_pickle=False)
        layers = {layer.stub: blob[layer.stub] for layer in NATIVE_LAYERS}
        provenance = json.loads(str(blob["provenance_json"].item()))
        return DomainStack(fire_id, coarse, fine, refine, layers, provenance)

    stack = fetch_native_stack(fine, _fire_year(fire_id), use_cache=False)
    provenance = {
        **stack.provenance,
        "fire_id": fire_id,
        "scope": "WHOLE C1 domain, one LFPS request per layer",
        "refine": refine,
    }
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
            **stack.layers,
        }
        np.savez_compressed(cache, **arrays)
    return DomainStack(fire_id, coarse, fine, refine, dict(stack.layers), provenance)


def verify_slice_equivalence(
    fire_id: str,
    *,
    refine: int = DEFAULT_REFINE,
    t0: int | None = None,
) -> dict[str, Any]:
    """POSITIVE CONTROL for the whole-domain fetch: slice == direct fetch.

    Re-fetches ONE window's layers straight from LFPS and requires every one to
    match the slice of the domain stack EXACTLY. The control half is the point:
    the same comparison against a slice shifted by one fine cell must FAIL, or
    the check is measuring nothing.
    """
    domain = build_domain_stack(fire_id, refine=refine)
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    finally:
        ds.close()
    if t0 is None:
        burned = [(int((state[t] > 0).sum()), t) for t in range(len(state) - HORIZON_H)]
        t0 = max(b for b in burned if b[0] > 0)[1] // 2 or max(burned)[1]
    window = window_grids(
        domain.coarse,
        state[t0],
        reach_cells=max(1, ElmfireConfig().reach_cells_per_hour * HORIZON_H),
        refine=refine,
    )
    sliced = domain.slice_for(window)
    folder = domain.provenance["fuels_folder"]
    per_layer: dict[str, Any] = {}
    shifted_differs: list[str] = []
    for layer in NATIVE_LAYERS:
        direct = fetch_layer(layer, folder, window.fine, use_cache=True)
        ours = sliced.layers[layer.stub]
        same = bool(np.array_equal(direct, ours))
        # Shift by one fine cell in whichever direction still FITS. A window that
        # is the whole domain cannot be shifted +1 without running off the end,
        # and a silently truncated slice would make the control unable to fail.
        shifted: np.ndarray | None = None
        how = "none"
        for delta in (1, -1):
            r0 = window.row0 * refine + delta
            c0 = window.col0 * refine + delta
            if r0 < 0 or c0 < 0:
                continue
            cand = domain.layers[layer.stub][
                r0 : r0 + window.fine.ny, c0 : c0 + window.fine.nx
            ]
            if cand.shape == direct.shape:
                shifted, how = cand, f"domain slice offset by {delta:+d} fine cells"
                break
        if shifted is None:
            # The window IS the whole domain, so no offset slice of the right
            # shape exists. Roll the array instead: same size, same values, one
            # cell out of register. Weaker than the offset slice, and labelled.
            shifted = np.roll(np.roll(ours, 1, axis=0), 1, axis=1)
            how = "one-cell roll of the slice (window is the whole domain)"
        shift_ok = not np.array_equal(direct, shifted)
        per_layer_how = how
        if shift_ok:
            shifted_differs.append(layer.stub)
        per_layer[layer.stub] = {
            "identical": same,
            "n_differing_cells": int(np.count_nonzero(direct != ours)),
            "shifted_slice_detected_as_different": shift_ok,
            "shift_construction": per_layer_how,
        }
    return {
        "check": "whole-domain fetch sliced per window == per-window LFPS fetch",
        "fire_id": fire_id,
        "t0": int(t0),
        "window_fine_shape": list(window.fine.shape),
        "per_layer": per_layer,
        "all_layers_identical": all(v["identical"] for v in per_layer.values()),
        "positive_control_layers_that_detect_a_1_cell_shift": sorted(shifted_differs),
        "positive_control_ok": len(shifted_differs) >= 1,
        "positive_control_note": (
            "an all-identical result means nothing unless the SAME comparison can "
            "fail; a slice offset by one fine cell must differ on at least one layer"
        ),
    }


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------


def window_t0s(fire_id: str, *, stride: int = STRIDE) -> list[int]:
    """The ``t0`` of every window ``iter_windows`` yields, in its own order.

    Read off ``iter_windows`` itself so the enumeration index used for the
    per-window seed is the same index ``runs/_s1_score.py`` used.
    """
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        return [
            int(w.t0)
            for w in iter_windows(ds, HORIZON_H, stride=stride, fire_id=fire_id)
        ]
    finally:
        ds.close()


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(fire_id: str, refine: int, members: int, binary: str) -> None:
    domain = build_domain_stack(fire_id, refine=refine)
    ds = open_tensor(fire_tensor_path(fire_id))
    cfg = ElmfireConfig(
        mode=InputMode.NATIVE,
        refine=refine,
        crown_fire=True,
        fire_year=_fire_year(fire_id),
        stack_provider=domain.slice_for,
    )
    _WORKER.update(
        fire_id=fire_id,
        ds=ds,
        model=ElmfireNativeModel(Grid.from_dataset(ds), binary=binary, config=cfg),
        members=int(members),
        band_radius=default_band_radius(HORIZON_H),
    )


def _worker_run(job: tuple[int, int, int]) -> dict[str, Any]:
    index, t0, block = job
    started = time.time()
    model: ElmfireNativeModel = _WORKER["model"]
    window = forecast_inputs(_WORKER["ds"], t0, HORIZON_H, fire_id=_WORKER["fire_id"])
    samples = model.predict(
        window.x0,
        window.static,
        window.weather,
        _WORKER["members"],
        HORIZON_H,
        SEED + index,
    )
    row = window_row(
        window,
        samples,
        band_radius=_WORKER["band_radius"],
        fire_id=_WORKER["fire_id"],
        spatial_block_id=block,
    )
    row["is_train_fire"] = False
    # Nested member PREFIXES: model_growth as it would have been at 1..M members.
    # Free (the members are already computed) and it is the only honest way to
    # say whether the member count moved the answer.
    prefix: dict[str, float] = {}
    for m in range(1, int(_WORKER["members"]) + 1):
        sub = window_row(
            window,
            samples[:m],
            band_radius=_WORKER["band_radius"],
            fire_id=_WORKER["fire_id"],
            spatial_block_id=block,
        )
        prefix[str(m)] = float(sub["model_growth"])
    row["model_growth_by_member_prefix"] = prefix
    row["_elapsed_s"] = round(time.time() - started, 2)
    return row


def run_fire(
    fire_id: str,
    *,
    spatial_block_id: int,
    members: int = DEFAULT_MEMBERS,
    workers: int = 1,
    refine: int = DEFAULT_REFINE,
    stride: int = STRIDE,
    limit: int | None = None,
    out: Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run ELMFIRE over every window of one fire and emit ``eval.response`` rows.

    **Kill-tolerant, because three bursts of this project have died mid-run.**
    Every finished row is appended to ``<out>.partial.jsonl`` and flushed
    immediately, and a re-run SKIPS the ``t0`` values already on disk. A run that
    is killed at window 310 of 346 therefore costs 36 windows, not 310. The
    partial file is kept after a successful run so the resume path is auditable
    rather than a claim.
    """
    binary = str(find_binary())
    t0s = window_t0s(fire_id, stride=stride)
    if limit is not None:
        t0s = t0s[: int(limit)]

    partial_path = (
        None if out is None else Path(str(out).removesuffix(".json") + ".partial.jsonl")
    )
    done: dict[int, dict[str, Any]] = {}
    if partial_path is not None and partial_path.exists():
        for line in partial_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A row half-written when the process died. Dropping it is right:
                # it will simply be recomputed. Silently KEEPING it would not be.
                continue
            if int(row.get("n_members_run", members)) == int(members):
                done[int(row["t0"])] = row
    jobs = [
        (i, t0, spatial_block_id) for i, t0 in enumerate(t0s) if t0 not in done
    ]
    if progress and done:
        print(
            f"[E1] {fire_id}: resuming — {len(done)} of {len(t0s)} windows already "
            f"on disk in {partial_path}",
            flush=True,
        )

    scoring_before = scoring_code_fingerprint()
    started = time.time()
    rows: list[dict[str, Any]] = []
    sink = None if partial_path is None else partial_path.open("a")
    try:
        def _keep(row: dict[str, Any]) -> None:
            row["n_members_run"] = int(members)
            rows.append(row)
            if sink is not None:
                sink.write(json.dumps(row, default=float) + "\n")
                sink.flush()
            if progress and len(rows) % 10 == 0:
                _report(fire_id, len(rows), len(jobs), started)

        if workers <= 1:
            if jobs:
                _worker_init(fire_id, refine, members, binary)
            for job in jobs:
                _keep(_worker_run(job))
        elif jobs:
            with ProcessPoolExecutor(
                max_workers=int(workers),
                initializer=_worker_init,
                initargs=(fire_id, refine, members, binary),
            ) as pool:
                for row in pool.map(_worker_run, jobs, chunksize=1):
                    _keep(row)
    finally:
        if sink is not None:
            sink.close()
    elapsed = time.time() - started
    n_resumed = len(done)
    rows = sorted([*done.values(), *rows], key=lambda r: int(r["t0"]))
    payload: dict[str, Any] = {
        "task": f"{TASK} — window rows for ONE fire",
        "fire_id": fire_id,
        "spatial_block_id": int(spatial_block_id),
        "model_kind": "elmfire_native_monte_carlo",
        "binary": binary,
        "split_fingerprint": split_fingerprint()["fingerprint"],
        "scoring_code_before": scoring_before,
        "scoring_code_after": scoring_code_fingerprint(),
        "horizon_h": HORIZON_H,
        "stride": stride,
        "n_members": int(members),
        "seed": SEED,
        "band_radius_cells": default_band_radius(HORIZON_H),
        "refine": refine,
        "fine_cell_m": round(1000.0 / refine, 4),
        "n_rows": len(rows),
        "n_windows_expected": len(t0s),
        "n_windows_resumed_from_disk": n_resumed,
        "n_windows_computed_this_pass": len(rows) - n_resumed,
        "partial_path": None if partial_path is None else str(partial_path),
        "workers": int(workers),
        "elapsed_s": round(elapsed, 1),
        "seconds_per_window": round(elapsed / max(1, len(rows) - n_resumed), 3),
        "elmfire_max_runtime_s": float(ElmfireConfig().max_runtime_s),
        "max_seconds_per_member": round(
            max((float(r["_elapsed_s"]) / max(1, members) for r in rows), default=0.0), 1
        ),
        "n_windows_near_max_runtime": sum(
            1
            for r in rows
            if float(r["_elapsed_s"]) / max(1, members)
            >= TRUNCATION_FRACTION * float(ElmfireConfig().max_runtime_s)
        ),
        "mapping_compromises": MAPPING_COMPROMISES,
        "not_a_gate": (
            "G5 is NOT attempted here and no C6 gate criterion is computed. "
            "ADR-064 (6): this adjudicates a scientific question, not a gate."
        ),
        "rows": rows,
    }
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=1, default=float))
    return payload


def _report(fire_id: str, done: int, total: int, started: float) -> None:
    dt = time.time() - started
    rate = dt / max(1, done)
    print(
        f"{fire_id:<30} {done:>4}/{total:<4} {dt:>7.0f}s  "
        f"{rate:>5.2f}s/window  eta {(total - done) * rate / 60:>6.1f} min",
        flush=True,
    )


# --------------------------------------------------------------------------
# pairing + scoring
# --------------------------------------------------------------------------


def check_truth_pairing(
    rows: list[dict[str, Any]], reference_rows_path: Path
) -> dict[str, Any]:
    """Are our windows the SAME windows arm A was scored on, with the same truth?

    The whole comparison rests on this. ``stage_decay`` is computed per block
    from ``truth_growth`` and ``model_growth`` on the same rows, so a different
    window set would silently produce a different truth reference and the
    reference rows ADR-064 quotes would no longer be the thing we are beside.
    """
    ref = json.loads(Path(reference_rows_path).read_text())["rows"]
    ours = {(r["fire_id"], int(r["t0"])): r for r in rows}
    theirs = {(r["fire_id"], int(r["t0"])): r for r in ref if r["fire_id"] in {
        r2["fire_id"] for r2 in rows
    }}
    shared = sorted(set(ours) & set(theirs))
    disagreements = [
        {"key": list(k), "ours": ours[k]["truth_growth"], "reference": theirs[k]["truth_growth"]}
        for k in shared
        if not math.isclose(
            float(ours[k]["truth_growth"]), float(theirs[k]["truth_growth"]), rel_tol=0.0,
            abs_tol=0.0,
        )
    ]
    return {
        "reference": str(reference_rows_path),
        "n_ours": len(ours),
        "n_reference_same_fires": len(theirs),
        "n_shared_windows": len(shared),
        "n_only_in_ours": len(set(ours) - set(theirs)),
        "n_only_in_reference": len(set(theirs) - set(ours)),
        "n_truth_disagreements": len(disagreements),
        "first_disagreements": disagreements[:5],
        "paired": bool(
            shared and not disagreements and len(shared) == len(ours) == len(theirs)
        ),
    }


def _stage_by_block(rows: list[dict[str, Any]], target: str) -> dict[int, Any]:
    return stage_decay_by_block(rows, target=target)


def score(
    rows_paths: list[Path],
    *,
    reference_rows_path: Path | None = None,
    member_prefix: int | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    """Score ELMFIRE's ``stage_decay`` per block with ``eval/stage.py`` UNCHANGED.

    Self-identifying by construction: the payload names exactly which fires
    exist, and it says ``not_a_verdict`` unless all five currently held-out
    blocks are present — a half-run that a later reader mistakes for a finished
    one is the expensive failure mode (ADR-064).

    ``member_prefix`` fixes the ensemble size EVERY block is read at. Fires may
    legitimately have been run with different member counts (cost varies ~50x
    across these five fires), and a headline that reads block 12 at 4 members and
    block 4 at 2 would be comparing two ensembles across blocks. The default is
    the MINIMUM available across the inputs, so the headline is always internally
    uniform and never depends on which fire happened to be cheap.
    """
    rows: list[dict[str, Any]] = []
    per_fire: dict[str, Any] = {}
    members = set()
    for path in rows_paths:
        blob = json.loads(Path(path).read_text())
        n_have, n_want = int(blob["n_rows"]), int(blob["n_windows_expected"])
        complete = n_have == n_want
        cap = float(blob.get("elmfire_max_runtime_s") or ElmfireConfig().max_runtime_s)
        n_members_run = int(blob["n_members"])
        # Absence of a timing is recorded, not treated as safety: a row with no
        # `_elapsed_s` cannot be checked against the cap, and saying so is the
        # difference between "checked and clear" and "not checked".
        timed = [
            float(r["_elapsed_s"]) / max(1, n_members_run)
            for r in blob["rows"]
            if r.get("_elapsed_s") is not None
        ]
        n_untimed = len(blob["rows"]) - len(timed)
        worst = max(timed, default=0.0)
        n_near = sum(1 for t in timed if t >= TRUNCATION_FRACTION * cap)
        per_fire[blob["fire_id"]] = {
            "spatial_block_id": int(blob["spatial_block_id"]),
            "n_windows": n_have,
            "n_windows_expected": n_want,
            "complete": complete,
            "scored": complete and n_near == 0,
            "n_members": n_members_run,
            "elapsed_s": blob["elapsed_s"],
            "elmfire_max_runtime_s": cap,
            "max_seconds_per_member": round(worst, 1),
            "max_runtime_margin": round(cap / worst, 2) if worst > 0 else None,
            "n_windows_near_max_runtime": n_near,
            "n_windows_without_timing_so_uncheckable": n_untimed,
            "rows_path": str(path),
        }
        if complete and n_near:
            # ELMFIRE's MAX_RUNTIME is WALL-CLOCK and its abort is SILENT
            # (elmfire_level_set.f90:1263 sets T = TSTOP + 1 and dumps). A capped
            # window under-reports growth, and it under-reports it on exactly the
            # late, large windows — which moves stage_decay toward deceleration.
            per_fire[blob["fire_id"]]["refused"] = (
                f"{n_near} window(s) reached >= {TRUNCATION_FRACTION:.0%} of the "
                f"{cap:.0f} s wall-clock MAX_RUNTIME, where ELMFIRE stops the "
                "simulation early and says so nowhere in its rasters. Not scored: a "
                "silently truncated late window biases stage_decay DOWN."
            )
            continue
        if not complete:
            # REFUSED, not truncated. `stage_decay` splits a block's windows at
            # their own median age, so scoring the first 60% of a fire's life
            # computes a DIFFERENT estimand — an early-vs-earlier contrast — and
            # would read as a block that decelerates less than it does.
            per_fire[blob["fire_id"]]["refused"] = (
                f"{n_have} of {n_want} windows. stage_decay is a late-half vs "
                "early-half contrast over a block's WHOLE window set; a prefix of "
                "a fire's life is a different estimand, not a noisier version of "
                "this one. Not scored."
            )
            continue
        rows.extend(blob["rows"])
        members.add(int(blob["n_members"]))
    headline_prefix = int(member_prefix) if member_prefix else (min(members) if members else 0)
    key = str(headline_prefix)
    rows = [
        (
            {**r, "model_growth": r["model_growth_by_member_prefix"][key]}
            if key in (r.get("model_growth_by_member_prefix") or {})
            else r
        )
        for r in rows
    ]
    truth = _stage_by_block(rows, "truth_growth")
    model = _stage_by_block(rows, "model_growth")

    blocks = sorted(set(truth) | set(model))
    per_block: dict[str, Any] = {}
    positives: list[int] = []
    for block in blocks:
        t, m = truth.get(block), model.get(block)
        tv = t.value if t is not None and t.defined else None
        mv = m.value if m is not None and m.defined else None
        if mv is not None and mv > 0:
            positives.append(block)
        per_block[str(block)] = {
            "outcome": (m.outcome if m is not None else "UNDEFINED_block_absent"),
            "elmfire": mv,
            "truth_this_run": tv,
            "truth_reference": REFERENCE_STAGE_DECAY["truth"].get(block),
            "arm_a_reference": REFERENCE_STAGE_DECAY["arm_a"].get(block),
            "arm_s_reference": REFERENCE_STAGE_DECAY["arm_s"].get(block),
            "sign": (None if mv is None else ("positive" if mv > 0 else "negative")),
            "distance_to_truth": (
                None
                if mv is None or REFERENCE_STAGE_DECAY["truth"].get(block) is None
                else abs(mv - REFERENCE_STAGE_DECAY["truth"][block])
            ),
            "truth_detail": t.as_dict() if t is not None else None,
            "elmfire_detail": m.as_dict() if m is not None else None,
        }

    # member-count sensitivity: the same estimator on the nested prefixes
    by_prefix: dict[str, Any] = {}
    max_m = max(members) if members else 0
    for m_count in range(1, max_m + 1):
        key = str(m_count)
        sub = [
            {**r, "model_growth": r["model_growth_by_member_prefix"][key]}
            for r in rows
            if key in (r.get("model_growth_by_member_prefix") or {})
        ]
        if not sub:
            continue
        got = _stage_by_block(sub, "model_growth")
        by_prefix[key] = {
            str(b): (v.value if v.defined else None) for b, v in sorted(got.items())
        }

    scored = [b for b in blocks if per_block[str(b)]["elmfire"] is not None]
    n_pos = len(positives)
    n_neg = len(scored) - n_pos
    n_expected = len(HELD_OUT_BLOCKS)
    n_missing = max(0, n_expected - len(scored))
    complete = sorted(scored) == sorted(HELD_OUT_BLOCKS)

    # ADR-064 (4)'s rule is a COUNT with a threshold, so it can be decided before
    # every block exists: the missing blocks can only move the count by their own
    # number. With 4 of 5 scored and 4 positive, the final count is 4 or 5 and
    # >=4/5 holds under BOTH completions. Saying "not_a_verdict" there would
    # discard a determination the pre-registered rule already makes; claiming a
    # verdict when a missing block COULD flip it would be the opposite error.
    # Both are refused explicitly rather than left to a reader.
    p1_held_under_every_completion = n_pos >= 4
    p1_refuted_under_every_completion = n_neg >= 4
    determined = p1_held_under_every_completion or p1_refuted_under_every_completion

    if p1_held_under_every_completion:
        verdict = "E-P1_HELD_defect_belongs_to_the_model_class"
        why = (
            f"{n_pos} of {len(scored)} scored blocks are POSITIVE (accelerating)"
            + (
                ". ADR-064 (4): >=4/5 positive -> perimeter-proportional spread "
                "models systematically fail to reproduce observed fire deceleration."
                if complete
                else (
                    f", with {n_missing} block(s) not scored. The count can only "
                    f"RISE, so the final tally is >= {n_pos}/5 under every possible "
                    "completion and ADR-064 (4)'s >=4/5 threshold is met either way. "
                    "The MAGNITUDE columns are still incomplete; only the sign count "
                    "is determined."
                )
            )
        )
    elif p1_refuted_under_every_completion:
        verdict = "E-P1_REFUTED_the_defect_is_ours"
        why = (
            f"{n_neg} of {len(scored)} scored blocks are NEGATIVE (decelerating)"
            + (
                ". ADR-064 (4): >=4/5 negative -> E-P1 is REFUTED and the defect "
                "is OURS specifically."
                if complete
                else (
                    f", with {n_missing} block(s) not scored; the count can only "
                    "RISE, so >=4/5 negative holds under every completion."
                )
            )
        )
    elif not complete:
        verdict = "not_a_verdict"
        why = (
            f"INCOMPLETE and UNDETERMINED: {len(scored)} of {n_expected} currently "
            f"held-out blocks scored ({n_pos} positive, {n_neg} negative). Present: "
            f"{sorted(scored)}; expected {list(HELD_OUT_BLOCKS)}. Fires on disk: "
            f"{sorted(per_fire)}. Neither threshold is reachable-or-not independently "
            "of the missing block(s), so no reading is licensed."
        )
    else:
        verdict = "not_a_verdict"
        why = (
            f"{n_pos}/5 positive — ADR-064 (4) fixes 3/5 either way as "
            "`not_a_verdict`; it waits for the full 14 blocks."
        )

    payload: dict[str, Any] = {
        "task": TASK,
        "question": (
            "ELMFIRE is Rothermel, i.e. rate of spread along a perimeter, so it is "
            "perimeter-proportional like our kernel. Does it ACCELERATE as a fire "
            "ages, the way our kernel does, or DECELERATE, the way real fires do?"
        ),
        "pre_registration": PRE_REGISTRATION,
        "estimator": (
            "eval/stage.py UNCHANGED — the same code that judged truth, arm A and "
            "arm S. No second implementation exists in this module (ADR-064 (6))."
        ),
        "split_fingerprint": split_fingerprint()["fingerprint"],
        "scoring_code_fingerprint": scoring_code_fingerprint(),
        "metric": STAGE_DECAY_KEY,
        "licence": licence(gate="the E1 ELMFIRE stage experiment"),
        "n_members_as_run": sorted(members),
        "headline_member_prefix": headline_prefix,
        "headline_member_prefix_note": (
            "every block is read at the SAME ensemble size — the minimum available "
            "across the fires — so no block's number comes from a bigger ensemble "
            "than another's. `member_prefix_sensitivity` shows the same estimator "
            "at every other available size."
        ),
        "horizon_h": HORIZON_H,
        "stride": STRIDE,
        "per_fire": per_fire,
        "n_rows": len(rows),
        "held_out_blocks_expected": list(HELD_OUT_BLOCKS),
        "blocks_scored": sorted(scored),
        "blocks_missing": sorted(set(HELD_OUT_BLOCKS) - set(scored)),
        "complete": complete,
        "verdict_determined_without_the_missing_blocks": determined,
        "declared_deviations_from_elmfire_defaults": DECLARED_DEVIATIONS_FROM_ELMFIRE_DEFAULTS,
        "per_block": per_block,
        "n_blocks_positive": n_pos,
        "n_blocks_scored": len(scored),
        "sign_test_positive_direction": sign_test(n_pos, len(scored)) if scored else None,
        "member_prefix_sensitivity": by_prefix,
        "reference_rows_pairing": (
            None
            if reference_rows_path is None
            else check_truth_pairing(rows, Path(reference_rows_path))
        ),
        "reference_values": {
            k: {str(b): v for b, v in sorted(d.items())}
            for k, d in REFERENCE_STAGE_DECAY.items()
        },
        "verdict": verdict,
        "verdict_reason": why,
        "mapping_compromises": MAPPING_COMPROMISES,
        "not_a_gate": (
            "G5 IS NOT ATTEMPTED AND IS NOT ADJUDICATED HERE. `stage_decay` is not "
            "licensed to decide a gate and the C6 registry refuses it; that refusal "
            "is recorded in `licence` above and is CORRECT."
        ),
    }
    if not complete:
        payload["incomplete"] = why
    if verdict == "not_a_verdict":
        payload["not_a_verdict"] = why
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=1, default=float))
    return payload


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def held_out_fires() -> list[Any]:
    """The held-out fires of the split ON DISK, never a hardcoded list."""
    stats = read_norm_stats(norm_stats_path())
    splits = load_splits([int(f) for f in stats["train_folds"]])
    return [s for s in splits if not s.is_train]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.elmfire_stage", allow_abbrev=False
    )
    ap.add_argument("--fires", nargs="*", default=None)
    ap.add_argument("--members", type=int, default=DEFAULT_MEMBERS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prefetch", action="store_true", help="fetch domain stacks and stop")
    ap.add_argument("--verify-slice", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--rows-dir", default="runs")
    ap.add_argument("--out", default="runs/e1.json")
    ap.add_argument("--reference-rows", default="runs/s1_rows_a_s1.json")
    args = ap.parse_args(argv)

    fires = held_out_fires()
    if args.fires:
        wanted = set(args.fires)
        fires = [f for f in fires if f.fire_id in wanted]
        if len(fires) != len(wanted):
            raise SystemExit(f"unknown or non-held-out fires: {sorted(wanted)}")

    rows_dir = Path(args.rows_dir)
    if args.prefetch:
        for f in fires:
            stack = build_domain_stack(f.fire_id)
            print(f"{f.fire_id:<32} fine {stack.fine.shape} cached "
                  f"{domain_stack_path(f.fire_id, stack.refine)}", flush=True)
        return 0
    if args.verify_slice:
        for f in fires:
            print(json.dumps(verify_slice_equivalence(f.fire_id), indent=1), flush=True)
        return 0

    if not args.score_only:
        for f in fires:
            out = rows_dir / f"e1_rows_{f.fire_id}.json"
            print(f"[E1] {f.fire_id} block {f.spatial_block_id} -> {out}", flush=True)
            run_fire(
                f.fire_id,
                spatial_block_id=int(f.spatial_block_id),
                members=args.members,
                workers=args.workers,
                limit=args.limit,
                out=out,
            )

    paths = sorted(rows_dir.glob("e1_rows_*.json"))
    if not paths:
        raise SystemExit("no e1_rows_*.json on disk; nothing to score")
    ref = Path(args.reference_rows)
    payload = score(paths, reference_rows_path=ref if ref.exists() else None, out=Path(args.out))
    print(json.dumps(
        {k: payload[k] for k in ("verdict", "verdict_reason", "blocks_scored",
                                 "n_blocks_positive", "per_block")},
        indent=1, default=float,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
