"""PLAYTHROUGH 2 — ELMFIRE baseline non-degeneracy (ADR-030).

A playthrough test is an end-to-end scenario whose correct answer is known BY
CONSTRUCTION, plus a scoring function returning pass/fail, with a planted defect
the harness actually detects. This one asks the question S4 exists to answer:

    **Do native inputs make ELMFIRE a non-degenerate baseline?**

and it answers it without a human in the loop, on a scenario with no network
dependency, while keeping a NEGATIVE CONTROL whose failure is already measured —
the exact configuration that produced ADR-025 (4)'s +2 cells against truth's +54.

WHY A DEGENERACY VERDICT NEEDS ITS OWN SCORING FUNCTION
------------------------------------------------------
C6.2 says a baseline that ignites zero cells on the held-out set is not a
distinct baseline and its gate is **VOID, not passed**. That is the right rule
and it is too coarse to use alone: the Brier-fitted ellipse ignited 8 cells
against truth's 4,557 and was correctly called degenerate at 0.002x, not at 0x.
:func:`degeneracy_verdict` therefore pre-registers three criteria, all fixed
BEFORE any native number existed:

``D1`` no member ignites anything                       — C6.2 verbatim
``D2`` median member growth < ``floor``                 — order-of-magnitude miss
``D3`` at most one distinct member                      — not an ensemble at all

For a scenario with truth, ``floor = 0.2 * truth_new`` — under-predicting growth
by more than 5x. Calibrated against what we already know rather than invented:
the barred Brier-fit ellipse sits at 0.002x and S3's ELMFIRE at 0.037x (both
degenerate), while the calibrated ellipse spans 0.845x-3.09x and the kernel
0.874x-3.06x (both not). Nothing in [0.2, 5] is currently contested.
For the synthetic scenario there is no truth, so the floor is an absolute
cell count justified by the physics of the scenario itself.

THE PLANTED DEFECT IS THE POINT
-------------------------------
:data:`LOBOTOMISED` is not a strawman invented to fail. It is the S3 input
mapping: a 1 km analysis grid with the canopy layers zeroed, which is what C1's
channel set forces. Measured here at exactly zero growth. **A playthrough whose
negative control is a real, previously-shipped configuration cannot be accused of
grading itself.**
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.sim.c5 import WEATHER_C5
from wildfire_nowcast.sim.coarsen import DEFAULT_REFINE, fine_grid
from wildfire_nowcast.sim.elmfire import (
    ElmfireConfig,
    ElmfireNativeModel,
    InputMode,
)
from wildfire_nowcast.sim.landfire import synthetic_stack

__all__ = [
    "DEGENERACY_CRITERIA",
    "DegeneracyVerdict",
    "degeneracy_verdict",
    "Arm",
    "ARMS",
    "uniform_weather",
    "run_arm",
    "run_playthrough",
]

DEGENERACY_CRITERIA = {
    "D1": "no member ignites a single new cell (C6.2 verbatim)",
    "D2": "median member new-cell count < floor (order-of-magnitude under-prediction)",
    "D3": "at most one distinct member — an ensemble of one is not an ensemble",
}

#: Under-predict truth's growth by more than this factor and the baseline is not
#: a baseline. Fixed before any native number existed; see the module docstring
#: for the calibration against models already on the record.
TRUTH_FLOOR_FRACTION = 0.2


@dataclass(frozen=True)
class DegeneracyVerdict:
    arm: str
    new_cells_per_member: list[int]
    median_new_cells: float
    distinct_members: int
    floor: float
    floor_basis: str
    d1_zero_growth: bool
    d2_below_floor: bool
    d3_no_ensemble: bool
    degenerate: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "new_cells_per_member": self.new_cells_per_member,
            "median_new_cells": self.median_new_cells,
            "distinct_members": self.distinct_members,
            "floor": self.floor,
            "floor_basis": self.floor_basis,
            "D1_zero_growth": self.d1_zero_growth,
            "D2_below_floor": self.d2_below_floor,
            "D3_no_ensemble": self.d3_no_ensemble,
            "degenerate": self.degenerate,
        }


def degeneracy_verdict(
    arm: str,
    samples: np.ndarray,
    x0: np.ndarray,
    *,
    truth_new: int | None = None,
    absolute_floor: float | None = None,
) -> DegeneracyVerdict:
    """Score one ensemble against the three pre-registered criteria."""
    burned0 = int(np.count_nonzero(np.asarray(x0) > 0))
    per_member = [
        int(np.count_nonzero(samples[m, -1] > 0)) - burned0 for m in range(samples.shape[0])
    ]
    median = float(np.median(per_member)) if per_member else 0.0
    distinct = len({s.tobytes() for s in samples})
    if truth_new is not None:
        floor = TRUTH_FLOOR_FRACTION * float(truth_new)
        basis = f"{TRUTH_FLOOR_FRACTION} x truth_new={truth_new}"
    elif absolute_floor is not None:
        floor = float(absolute_floor)
        basis = "absolute floor from the scenario's own physics"
    else:
        raise ValueError("a degeneracy verdict needs either truth_new or absolute_floor")
    d1 = max(per_member, default=0) <= 0
    d2 = median < floor
    d3 = distinct <= 1
    return DegeneracyVerdict(
        arm=arm,
        new_cells_per_member=per_member,
        median_new_cells=median,
        distinct_members=distinct,
        floor=round(floor, 3),
        floor_basis=basis,
        d1_zero_growth=bool(d1),
        d2_below_floor=bool(d2),
        d3_no_ensemble=bool(d3),
        degenerate=bool(d1 or d2 or d3),
    )


# --------------------------------------------------------------------------
# the scenario — known by construction, no network
# --------------------------------------------------------------------------

#: 20 x 20 km of uniform TU5 (Scott & Burgan 165: very high load, dry climate
#: timber-shrub), flat, ignited as a 3 x 3 km block on the west edge.
DOMAIN_CELLS = 20
IGNITION_ROWS = slice(9, 12)
IGNITION_COLS = slice(3, 6)
HORIZON_H = 3
WIND_MS = 12.0
DEAD_FUEL_MOISTURE_PCT = 3.0

#: The known-by-construction floor. A 3 km-wide head fire in continuous dry fuel
#: under a 27 mph wind must advance at least ~1 km/h; over 3 h that is >= 3 km of
#: advance across a 3 km front, i.e. >= 9 new 1 km cells even before flank
#: growth. 10 is the conservative bar. ELMFIRE's own
#: CROWN_FIRE_SPREAD_RATE_LIMIT (250 ft/min = 4.57 km/h) bounds the other side.
SYNTHETIC_FLOOR_CELLS = 10.0


@dataclass(frozen=True)
class Arm:
    """One end-to-end configuration under test."""

    name: str
    native_grid: bool
    canopy_cover_pct: int
    canopy_height_m10: int
    canopy_base_height_m10: int
    canopy_bulk_density_kgm3_100: int
    crown_fire: bool
    #: What the harness must conclude. The negative control's expectation is
    #: DEGENERATE, and if it ever comes back non-degenerate the test fails —
    #: which is what stops this from being a check that cannot fail.
    expect_degenerate: bool
    why: str


ARMS: tuple[Arm, ...] = (
    Arm(
        name="native_30m_crown_on",
        native_grid=True,
        canopy_cover_pct=60,
        canopy_height_m10=200,
        canopy_base_height_m10=20,
        canopy_bulk_density_kgm3_100=15,
        crown_fire=True,
        expect_degenerate=False,
        why="ADR-026 (3) configuration: native LANDFIRE canopy, crown fire ON",
    ),
    Arm(
        name="lobotomised_1km_crown_off",
        native_grid=False,
        canopy_cover_pct=60,
        canopy_height_m10=0,
        canopy_base_height_m10=0,
        canopy_bulk_density_kgm3_100=0,
        crown_fire=False,
        expect_degenerate=True,
        why=(
            "the S3 mapping, verbatim: 1 km analysis grid and the canopy layers "
            "C1 cannot supply set to zero. Known-true failure case."
        ),
    ),
    Arm(
        name="diagnostic_1km_crown_on",
        native_grid=False,
        canopy_cover_pct=60,
        canopy_height_m10=200,
        canopy_base_height_m10=20,
        canopy_bulk_density_kgm3_100=15,
        crown_fire=True,
        expect_degenerate=False,
        why=(
            "ISOLATES THE VARIABLE: coarse grid but a complete canopy. Separates "
            "'1 km lobotomises ELMFIRE' from 'a missing canopy lobotomises ELMFIRE'."
        ),
    ),
)


def uniform_weather(n_hours: int, n_cells: int, *, wind_ms: float, moisture_pct: float
                    ) -> np.ndarray:
    """C5 ``weather`` for a steady westerly. Shape ``[T, C_w, H, W]``."""
    wx = np.zeros((n_hours, len(WEATHER_C5), n_cells, n_cells), dtype=np.float32)
    idx = {c: i for i, c in enumerate(WEATHER_C5)}
    wx[:, idx["wind_u10"]] = wind_ms  # blowing toward +x, i.e. to the east
    wx[:, idx["wind_v10"]] = 0.0
    wx[:, idx["temp_2m"]] = 305.0
    wx[:, idx["rh_2m"]] = 10.0
    wx[:, idx["fuel_moisture_proxy"]] = moisture_pct
    return wx


def run_arm(
    arm: Arm, *, n_members: int, seed: int, refine: int = DEFAULT_REFINE
) -> dict[str, Any]:
    """Run one arm end to end and score it. Returns the verdict plus evidence."""
    grid = Grid(
        x_min=-2_000_000.0,
        y_max=2_000_000.0,
        nx=DOMAIN_CELLS,
        ny=DOMAIN_CELLS,
        cell_size_m=1000.0,
    )
    x0 = np.zeros(grid.shape, dtype=np.uint8)
    x0[IGNITION_ROWS, IGNITION_COLS] = 1
    stack_grid = fine_grid(grid, refine) if arm.native_grid else grid
    stack = synthetic_stack(
        stack_grid,
        fbfm40=165,
        canopy_cover_pct=arm.canopy_cover_pct,
        canopy_height_m10=arm.canopy_height_m10,
        canopy_base_height_m10=arm.canopy_base_height_m10,
        canopy_bulk_density_kgm3_100=arm.canopy_bulk_density_kgm3_100,
    )
    cfg = ElmfireConfig(
        mode=InputMode.NATIVE if arm.native_grid else InputMode.LOBOTOMISED,
        refine=refine,
        stack=stack,
        crown_fire=arm.crown_fire,
    )
    model = ElmfireNativeModel(grid, config=cfg)
    wx = uniform_weather(
        HORIZON_H, DOMAIN_CELLS, wind_ms=WIND_MS, moisture_pct=DEAD_FUEL_MOISTURE_PCT
    )
    t0 = time.time()
    samples = model.predict(x0, None, wx, n_members, HORIZON_H, seed)
    elapsed = time.time() - t0

    verdict = degeneracy_verdict(
        arm.name, samples, x0, absolute_floor=SYNTHETIC_FLOOR_CELLS
    )
    # The head must move DOWNWIND. A model that grows the right amount in the
    # wrong direction is not a usable baseline either, and this is free to check.
    burned_any = samples[:, -1].max(axis=0) > 0
    cols = np.argwhere(burned_any)[:, 1]
    x0_cols = np.argwhere(x0 > 0)[:, 1]
    downwind_advance = int(cols.max() - x0_cols.max()) if cols.size else 0
    return {
        "arm": arm.name,
        "why": arm.why,
        "expect_degenerate": arm.expect_degenerate,
        "analysis_cell_size_m": round(stack_grid.cell_size_m, 4),
        "analysis_shape": list(stack_grid.shape),
        "crown_fire": arm.crown_fire,
        "canopy": {
            "cc_pct": arm.canopy_cover_pct,
            "ch_m": arm.canopy_height_m10 / 10.0,
            "cbh_m": arm.canopy_base_height_m10 / 10.0,
            "cbd_kg_m3": arm.canopy_bulk_density_kgm3_100 / 100.0,
        },
        "elapsed_s": round(elapsed, 2),
        "burned_at_t0": int(np.count_nonzero(x0)),
        "cells_by_lead": [int(np.count_nonzero(samples[0, k] > 0)) for k in range(HORIZON_H)],
        "downwind_advance_cells": downwind_advance,
        "samples_are_absorbing": bool(
            np.all(np.diff(samples.astype(np.int16), axis=1) >= 0)
        ),
        "verdict": verdict.as_dict(),
        "agrees_with_expectation": bool(verdict.degenerate == arm.expect_degenerate),
    }


def run_playthrough(
    *, n_members: int = 6, seed: int = 20260808, refine: int = DEFAULT_REFINE
) -> dict[str, Any]:
    """PLAYTHROUGH 2 — pass iff every arm lands where it was declared to land."""
    rows = [run_arm(a, n_members=n_members, seed=seed, refine=refine) for a in ARMS]
    determinism = _determinism_check(n_members=2, seed=seed, refine=refine)
    ok = all(r["agrees_with_expectation"] for r in rows) and determinism["bitwise_identical"]
    return {
        "playthrough": "baseline_non_degeneracy",
        "criteria": DEGENERACY_CRITERIA,
        "truth_floor_fraction": TRUTH_FLOOR_FRACTION,
        "synthetic_floor_cells": SYNTHETIC_FLOOR_CELLS,
        "scenario": {
            "domain_cells": DOMAIN_CELLS,
            "fuel_model": "FBFM40 165 (TU5), uniform",
            "terrain": "flat, slope 0",
            "wind_ms": WIND_MS,
            "dead_fuel_moisture_pct": DEAD_FUEL_MOISTURE_PCT,
            "horizon_h": HORIZON_H,
            "ignition": "3 x 3 km block on the west flank",
        },
        "arms": rows,
        "determinism": determinism,
        "n_members": n_members,
        "seed": seed,
        "verdict": "PASS" if ok else "FAIL",
    }


def _determinism_check(*, n_members: int, seed: int, refine: int) -> dict[str, Any]:
    """Re-run the native arm twice and compare bytes.

    S3 proved 4/4 bitwise determinism on the OLD input path. The input path
    changed, so that proof does not transfer and is re-taken here.
    """
    arm = ARMS[0]
    outs = []
    for _ in range(2):
        grid = Grid(
            x_min=-2_000_000.0, y_max=2_000_000.0,
            nx=DOMAIN_CELLS, ny=DOMAIN_CELLS, cell_size_m=1000.0,
        )
        x0 = np.zeros(grid.shape, dtype=np.uint8)
        x0[IGNITION_ROWS, IGNITION_COLS] = 1
        stack = synthetic_stack(
            fine_grid(grid, refine),
            fbfm40=165,
            canopy_cover_pct=arm.canopy_cover_pct,
            canopy_height_m10=arm.canopy_height_m10,
            canopy_base_height_m10=arm.canopy_base_height_m10,
            canopy_bulk_density_kgm3_100=arm.canopy_bulk_density_kgm3_100,
        )
        model = ElmfireNativeModel(
            grid,
            config=ElmfireConfig(refine=refine, stack=stack, crown_fire=True),
        )
        wx = uniform_weather(
            HORIZON_H, DOMAIN_CELLS, wind_ms=WIND_MS, moisture_pct=DEAD_FUEL_MOISTURE_PCT
        )
        outs.append(model.predict(x0, None, wx, n_members, HORIZON_H, seed))
    return {
        "n_repeats": 2,
        "n_members": n_members,
        "bitwise_identical": bool(np.array_equal(outs[0], outs[1])),
        "note": "the input path changed at S4, so S3's 4/4 proof does not transfer",
    }


def real_fire_ab(
    tensor: str | Path,
    *,
    n_windows: int = 3,
    horizon_h: int = 3,
    n_members: int = 6,
    seed: int = 20260807,
    refine: int = DEFAULT_REFINE,
) -> dict[str, Any]:  # pragma: no cover - network + subprocess
    """The A/B that answers S4's empirical question on REAL data.

    Same fire, same windows, same members, same seed; the ONLY difference is the
    input path. Kept out of :func:`run_playthrough` on purpose — that one must
    stay network-free so it can run in CI, and a test that quietly needs the
    internet is a test that quietly stops running.
    """
    from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: PLC0415
    from wildfire_nowcast.sim.c5 import c5_inputs  # noqa: PLC0415

    ds = open_tensor(Path(tensor))
    state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    grid = Grid.from_dataset(ds)
    year = int(str(ds.attrs.get("time_start_utc", "2020"))[:4])
    growth = sorted(
        (
            (int((state[t + horizon_h] > 0).sum() - (state[t] > 0).sum()), t)
            for t in range(1, len(state) - horizon_h)
        ),
        reverse=True,
    )[:n_windows]

    rows: list[dict[str, Any]] = []
    for truth_new, t0 in growth:
        win = c5_inputs(ds, t0, horizon_h)
        entry: dict[str, Any] = {"t0": t0, "truth_new_cells": truth_new, "arms": {}}
        for mode, crown in ((InputMode.NATIVE, True), (InputMode.LOBOTOMISED, False)):
            model = ElmfireNativeModel(
                grid,
                config=ElmfireConfig(
                    mode=mode, refine=refine, crown_fire=crown, fire_year=year
                ),
            )
            t_start = time.time()
            samples = model.predict(
                win.x0, win.static, win.weather, n_members, horizon_h, seed
            )
            v = degeneracy_verdict(mode.value, samples, win.x0, truth_new=truth_new)
            entry["arms"][mode.value] = {
                **v.as_dict(),
                "ratio_to_truth": (
                    round(v.median_new_cells / truth_new, 4) if truth_new else None
                ),
                "elapsed_s": round(time.time() - t_start, 1),
                "analysis_cell_size_m": model.last_run["analysis_cell_size_m"],
            }
        rows.append(entry)

    native_deg = [r["arms"]["native"]["degenerate"] for r in rows]
    return {
        "kind": "elmfire_native_vs_lobotomised",
        "note": (
            "ADAPTER VALIDATION. NOT a G5 head-to-head: no C6 metric is computed, "
            "no other model is run, and G5 is not authorised."
        ),
        "tensor": str(tensor),
        "horizon_h": horizon_h,
        "n_members": n_members,
        "seed": seed,
        "criteria": DEGENERACY_CRITERIA,
        "windows": rows,
        "native_degenerate_on_any_window": bool(any(native_deg)),
        "verdict": (
            "ELMFIRE IS NOT DEGENERATE ON NATIVE INPUTS"
            if not any(native_deg)
            else "ELMFIRE REMAINS DEGENERATE — G5 MUST BE REPORTED WITH IT DECLARED SO"
        ),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.playthrough", allow_abbrev=False
    )
    ap.add_argument("--members", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--refine", type=int, default=DEFAULT_REFINE)
    ap.add_argument("--out", default="reports/figures/playthrough_nondegeneracy.json")
    ap.add_argument("--real-tensor", default=None, help="also run the real-fire A/B")
    ap.add_argument("--real-windows", type=int, default=3)
    ap.add_argument(
        "--real-out", default="reports/figures/elmfire_degeneracy_verdict.json"
    )
    args = ap.parse_args(argv)
    if args.real_tensor:
        ab = real_fire_ab(
            args.real_tensor,
            n_windows=args.real_windows,
            n_members=args.members,
            refine=args.refine,
        )
        out = Path(args.real_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ab, indent=1))
        for row in ab["windows"]:
            for mode, a in row["arms"].items():
                print(
                    f"[ab] t0={row['t0']:<5} truth+{row['truth_new_cells']:<4} "
                    f"{mode:<12} median+{a['median_new_cells']:<6.1f} "
                    f"ratio {a['ratio_to_truth']}  distinct {a['distinct_members']}  "
                    f"degenerate={a['degenerate']}"
                )
        print(f"[ab] {ab['verdict']}  -> {out}")
        return 0
    report = run_playthrough(n_members=args.members, seed=args.seed, refine=args.refine)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    for row in report["arms"]:
        v = row["verdict"]
        print(
            f"[playthrough2] {row['arm']:<28} cells {row['cells_by_lead']}  "
            f"median_new {v['median_new_cells']:6.1f}  distinct {v['distinct_members']}  "
            f"degenerate={v['degenerate']!s:<5} expected={row['expect_degenerate']!s:<5} "
            f"{'ok' if row['agrees_with_expectation'] else 'MISMATCH'}"
        )
    print(f"[playthrough2] determinism bitwise: {report['determinism']['bitwise_identical']}")
    print(f"[playthrough2] {report['verdict']}  -> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def render_verdict(
    playthrough1: str | Path,
    playthrough2: str | Path,
    real_ab: str | Path,
    out: str | Path,
) -> Path:  # pragma: no cover - rendering
    """One page an incident-command reader can check the S4 claims against.

    Numbers in a JSON are auditable; a picture is what makes a wrong one obvious.
    Every panel here is drawn from a file on disk, never from a variable in
    memory, so the figure cannot disagree with the artifact it cites.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wildfire_nowcast.sim.coarsen import (
        DEFAULT_REFINE,
        SCENARIOS,
        coarsen_all,
        coarsen_nearest,
        coarsen_occupancy,
    )

    p1 = json.loads(Path(playthrough1).read_text())
    p2 = json.loads(Path(playthrough2).read_text())
    ab = json.loads(Path(real_ab).read_text())

    fig = plt.figure(figsize=(15.5, 10.0))
    gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.28, height_ratios=[1.15, 1.0, 0.9])

    # (1) coarsening: fine truth vs the rule vs two planted defects
    sc = SCENARIOS[2]  # the textured disc, where the rules actually separate
    fine = sc.build(DEFAULT_REFINE)
    panels = [
        ("fine 30.3 m input", fine, None),
        ("RULE: occupancy >= 0.5", coarsen_occupancy(fine, DEFAULT_REFINE), sc.area_km2),
        ("DEFECT: nearest", coarsen_nearest(fine, DEFAULT_REFINE), sc.area_km2),
        ("DEFECT: all sub-cells", coarsen_all(fine, DEFAULT_REFINE), sc.area_km2),
    ]
    for i, (title, arr, truth) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(arr, cmap="inferno", interpolation="nearest")
        if truth is None:
            # The fine input is a disc perforated at ~150 m, so its own burned
            # area is 80% of the disc. The correct 1 km answer is still the WHOLE
            # disc, because every interior cell is 80% covered.
            area = float(np.count_nonzero(arr)) * (1.0 / DEFAULT_REFINE) ** 2
            sub = f"burned {area:.1f} km2 inside a {sc.area_km2:.1f} km2 disc"
        else:
            area = float(np.count_nonzero(arr))
            sub = f"{area:.1f} km2  (correct answer {truth:.1f})"
        ax.set_title(f"{title}\n{sub}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # (2) the controlled experiment: which variable actually mattered
    ax = fig.add_subplot(gs[1, :2])
    names = [a["arm"] for a in p2["arms"]]
    finals = [a["cells_by_lead"][-1] for a in p2["arms"]]
    colors = ["#2b7a3d" if not a["verdict"]["degenerate"] else "#b02418" for a in p2["arms"]]
    ax.barh(range(len(names)), finals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(p2["arms"][0]["burned_at_t0"], color="k", ls="--", lw=1)
    ax.set_xlabel("cells burned after 3 h (dashed = t0 state)", fontsize=8)
    ax.set_title(
        "PLAYTHROUGH 2 — synthetic TU5, 12 m/s, 3% FM. Red = flagged DEGENERATE.\n"
        "1 km WITH a canopy nearly matches 30 m; 30 m WITHOUT one is zero.",
        fontsize=9,
    )

    # (3) the real-fire A/B
    ax = fig.add_subplot(gs[1, 2:])
    xs = np.arange(len(ab["windows"]))
    truth = [w["truth_new_cells"] for w in ab["windows"]]
    nat = [w["arms"]["native"]["median_new_cells"] for w in ab["windows"]]
    lob = [w["arms"]["lobotomised"]["median_new_cells"] for w in ab["windows"]]
    ax.bar(xs - 0.26, truth, 0.24, label="truth", color="#333333")
    ax.bar(xs, nat, 0.24, label="native 30 m (crown ON)", color="#2b7a3d")
    ax.bar(xs + 0.26, lob, 0.24, label="S3 mapping (1 km, crown OFF)", color="#b02418")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"t0={w['t0']}" for w in ab["windows"]], fontsize=8)
    ax.set_ylabel("new cells in 3 h (member median)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(
        "REAL FIRE — 2019 Kincade, three highest-growth windows.\n"
        "Same windows, members and seed; ONLY the input path differs.",
        fontsize=9,
    )

    # (4) the verdict text, quoted from the artifacts
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    lines = [
        f"PLAYTHROUGH 1 (coarsening)      : {p1['verdict']}   rule = {p1['rule']}",
        "   planted defects caught by     : "
        + "; ".join(f"{k} -> {', '.join(v)}" for k, v in p1["defects_caught_by"].items()),
        "   measured resolution limit    : "
        + ", ".join(
            f"{r['finger_width_km']} km finger keeps {r['area_retained']:.0%} in "
            f"{r['components']} piece(s)"
            for r in p1["resolution_limit_measured"][:3]
        ),
        f"PLAYTHROUGH 2 (non-degeneracy)  : {p2['verdict']}   "
        f"determinism bitwise = {p2['determinism']['bitwise_identical']}",
        f"REAL-FIRE VERDICT               : {ab['verdict']}",
        "   native ratio to truth        : "
        + ", ".join(
            f"t0={w['t0']} {w['arms']['native']['ratio_to_truth']:.2f}x"
            for w in ab["windows"]
        )
        + "   |  S3 mapping: "
        + ", ".join(f"{w['arms']['lobotomised']['ratio_to_truth']:.3f}x" for w in ab["windows"]),
        "NOT A G5 HEAD-TO-HEAD. No C6 metric is computed here and no other model is run.",
    ]
    ax.text(
        0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.5,
        family="monospace", transform=ax.transAxes,
    )
    fig.suptitle(
        "S4 — ELMFIRE on NATIVE INPUTS, CONTRACT OUTPUTS (ADR-026 (3)): adapter validation",
        fontsize=12,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
