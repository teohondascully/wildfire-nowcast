"""The 30 m -> 1 km coarsening rule for ELMFIRE's output, and its playthrough test.

ADR-026 (3) moved ELMFIRE to **native inputs, contract outputs**: it consumes
30 m LANDFIRE at its own resolution, and only its OUTPUT ensemble is brought back
to the C1 1 km lattice and wrapped in C5. That makes this module the last step
before a baseline's numbers enter a gate, so a bug here is indistinguishable from
"the baseline is weak" - which is exactly how the 1 km input mapping produced a
degenerate ELMFIRE and nearly voided G5 in the flattering direction.

THE RULE, AND WHY IT IS NOT A PREFERENCE
----------------------------------------
``data/rasterize.py`` already defines how a GOFER perimeter becomes a 1 km TRUTH
cell: rasterise the polygon on a 10x finer lattice, block-**mean** to a coverage
fraction, and threshold at ``COVER_THRESHOLD = 0.5``. Its own reasoning is the
reasoning here - centroid-in-polygon throws away half a cell at 1 km, and
``all_touched=True`` dilates every perimeter by a cell.

    **A member's fine burned set is coarsened by FRACTIONAL-AREA OCCUPANCY:
    block-mean the binary fine mask over the exactly-nested R x R sub-cell block,
    then mark the 1 km cell burned iff the covered fraction is >= 0.5.**

Scoring ELMFIRE against truth under a *different* convention would be a
systematic area bias with no visible cause in any table, so matching truth's own
rule is the only choice that keeps G5 adjudicable.

WHY THE FINE LATTICE IS 30.303 m AND NOT 30 m
---------------------------------------------
30 m does not divide 1000 m. A true-30 m grid is therefore NOT nested inside the
1 km lattice, every coarse cell straddles partial sub-cells, and area
conservation becomes a property of an overlap-weight implementation rather than
of arithmetic. :data:`DEFAULT_REFINE` = 33 gives 30.303 m - 1.01% off native,
inside LANDFIRE's own resampling noise - and makes each 1 km cell exactly
33 x 33 = 1089 sub-cells, so the block-mean is exact by construction. Declared as
a mapping compromise, with the direction of bias: none, it is a 1% cell-size
change applied identically to every layer.

THE DEFECT FAMILY THIS MODULE EXISTS TO CATCH
---------------------------------------------
Three wrong rules are implemented here on purpose, as
:data:`DEFECTIVE_COARSENERS`, because ADR-030 requires a playthrough test to have
a planted defect the harness actually detects:

``nearest``  centroid / nearest-neighbour sampling. Area error is small and
             UNBIASED on blobs, so an area check alone passes it - it destroys
             thin features instead, and is caught by CONNECTIVITY.
``all``      require every sub-cell (an integer-cell boxcar / erosion). Loses the
             whole boundary band, which is the 11%-perimeter-shrink bug class.
             Caught by AREA, on the low side.
``any``      any-touch dilation. Caught by AREA, on the high side.

Two different assertions catch two different defects, and neither catches both.
That is the argument for keeping both, not redundancy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.sim.absent import refuse_if_empty

__all__ = [
    "DEFAULT_REFINE",
    "OCCUPANCY_THRESHOLD",
    "COARSENING_RULE",
    "fine_grid",
    "coverage_fraction",
    "coarsen_occupancy",
    "coarsen_nearest",
    "coarsen_all",
    "coarsen_any",
    "DEFECTIVE_COARSENERS",
    "n_components",
    "AreaVerdict",
    "score_coarsening",
    "disc_mask",
    "rotated_bar_mask",
    "diagonal_finger_mask",
    "SCENARIOS",
    "run_playthrough",
]

#: Sub-cells per 1 km cell edge. 33 -> 30.303 m, an EXACT nesting. See module doc.
DEFAULT_REFINE = 33
#: Truth's own threshold (``data.rasterize.COVER_THRESHOLD``); do not diverge.
OCCUPANCY_THRESHOLD = 0.5

COARSENING_RULE = (
    "fractional-area occupancy: block-mean the binary fine mask over the exactly "
    "nested R x R sub-cell block, burned iff covered fraction >= 0.5. Identical in "
    "form and threshold to data/rasterize.polygon_mask, which defines TRUTH."
)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def fine_grid(coarse: Grid, refine: int = DEFAULT_REFINE) -> Grid:
    """The exactly-nested refinement of ``coarse`` by an integer factor.

    Shares the north-west corner, so sub-cell block ``(i, j)`` of the fine grid
    tiles coarse cell ``(i // refine, j // refine)`` with no remainder.
    """
    if refine < 1:
        raise ValueError(f"refine must be >= 1, got {refine}")
    return Grid(
        x_min=coarse.x_min,
        y_max=coarse.y_max,
        nx=coarse.nx * refine,
        ny=coarse.ny * refine,
        cell_size_m=coarse.cell_size_m / refine,
        crs=coarse.crs,
    )


def _blocks(fine: np.ndarray, refine: int) -> np.ndarray:
    """View ``[R*h, R*w]`` as ``[h, R, w, R]``. Raises on a non-nested shape."""
    arr = np.asarray(fine)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D fine mask, got shape {arr.shape}")
    h, w = arr.shape
    if h % refine or w % refine:
        raise ValueError(
            f"fine shape {arr.shape} is not an exact multiple of refine={refine}. "
            "A non-nested lattice would need partial-overlap weights, which is "
            "the very thing DEFAULT_REFINE=33 exists to avoid."
        )
    return arr.reshape(h // refine, refine, w // refine, refine)


def coverage_fraction(fine: np.ndarray, refine: int = DEFAULT_REFINE) -> np.ndarray:
    """Fraction of each 1 km cell covered by the fine burned set, float32."""
    return _blocks(np.asarray(fine) > 0, refine).mean(axis=(1, 3)).astype(np.float32)


def coarsen_occupancy(
    fine: np.ndarray,
    refine: int = DEFAULT_REFINE,
    threshold: float = OCCUPANCY_THRESHOLD,
) -> np.ndarray:
    """THE RULE. Fractional-area occupancy at ``threshold``; boolean ``[h, w]``."""
    return coverage_fraction(fine, refine) >= threshold


# -- deliberately wrong rules, kept as the harness's planted defects --------


def coarsen_nearest(fine: np.ndarray, refine: int = DEFAULT_REFINE) -> np.ndarray:
    """DEFECT: nearest-neighbour / centroid sampling. Drops thin features."""
    off = refine // 2
    return np.asarray(fine)[off::refine, off::refine] > 0


def coarsen_all(fine: np.ndarray, refine: int = DEFAULT_REFINE) -> np.ndarray:
    """DEFECT: require EVERY sub-cell (integer-cell boxcar). Erodes the boundary."""
    return _blocks(np.asarray(fine) > 0, refine).all(axis=(1, 3))


def coarsen_any(fine: np.ndarray, refine: int = DEFAULT_REFINE) -> np.ndarray:
    """DEFECT: any-touch. Dilates every perimeter by a cell."""
    return _blocks(np.asarray(fine) > 0, refine).any(axis=(1, 3))


DEFECTIVE_COARSENERS: dict[str, Callable[..., np.ndarray]] = {
    "nearest": coarsen_nearest,
    "all_subcells": coarsen_all,
    "any_subcell": coarsen_any,
}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def n_components(mask: np.ndarray) -> int:
    """Number of 8-connected components of a boolean mask.

    Delegates to :func:`wildfire_nowcast.sim.components.label_components` - the
    union-find labeller this package already owns. scipy is deliberately absent
    from this project (``common/states.py`` says so in as many words), and a
    coarsening test that needs an optional dependency to run is a test that
    silently stops running.
    """
    from wildfire_nowcast.sim.components import label_components  # noqa: PLC0415

    _, count = label_components(np.asarray(mask) > 0)
    return int(count)


@dataclass(frozen=True)
class AreaVerdict:
    """One scenario's pass/fail, with every number it rests on."""

    scenario: str
    coarsener: str
    true_area_km2: float
    coarse_area_km2: float
    abs_err_km2: float
    rel_err: float
    analytic_tol_km2: float
    area_ok: bool
    relative_ok: bool
    components_true: int
    components_coarse: int
    connectivity_ok: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "coarsener": self.coarsener,
            "true_area_km2": round(self.true_area_km2, 4),
            "coarse_area_km2": round(self.coarse_area_km2, 4),
            "abs_err_km2": round(self.abs_err_km2, 4),
            "rel_err": round(self.rel_err, 5),
            "analytic_tol_km2": round(self.analytic_tol_km2, 4),
            "area_ok": self.area_ok,
            "relative_ok": self.relative_ok,
            "components_true": self.components_true,
            "components_coarse": self.components_coarse,
            "connectivity_ok": self.connectivity_ok,
            "passed": self.passed,
        }


#: Relative-area bound for shapes several cells across. Tighter than the analytic
#: boundary-band bound, and it is what actually catches a subtle rule.
RELATIVE_TOL = 0.05


def score_coarsening(
    *,
    scenario: str,
    coarsener: str,
    fine: np.ndarray,
    coarse: np.ndarray,
    true_area_km2: float,
    perimeter_km: float,
    coarse_cell_km: float = 1.0,
    true_components: int = 1,
    relative_tol: float = RELATIVE_TOL,
) -> AreaVerdict:
    """Score one coarsening against an ANALYTICALLY KNOWN area and component count.

    Two independent criteria, because they catch different defects:

    * **area** - ``|A_coarse - A_true| <= 0.5 * P * dx``, the boundary-band bound.
      A rule that keeps only whole cells loses the entire band and violates it on
      the low side; a rule that keeps any touched cell violates it on the high
      side. Plus a tighter ``relative_tol`` for shapes many cells across.
    * **connectivity** - the coarse component count must equal the true one. This
      is the only criterion that sees nearest-neighbour sampling, whose area error
      is small and unbiased while it shreds thin features.
    """
    coarse_area = float(np.count_nonzero(coarse)) * coarse_cell_km**2
    abs_err = abs(coarse_area - true_area_km2)
    rel_err = abs_err / true_area_km2 if true_area_km2 > 0 else math.inf
    tol = 0.5 * perimeter_km * coarse_cell_km
    comps = n_components(coarse)
    area_ok = abs_err <= tol
    relative_ok = rel_err <= relative_tol
    conn_ok = comps == true_components
    return AreaVerdict(
        scenario=scenario,
        coarsener=coarsener,
        true_area_km2=float(true_area_km2),
        coarse_area_km2=coarse_area,
        abs_err_km2=abs_err,
        rel_err=rel_err,
        analytic_tol_km2=tol,
        area_ok=area_ok,
        relative_ok=relative_ok,
        components_true=int(true_components),
        components_coarse=comps,
        connectivity_ok=conn_ok,
        passed=bool(area_ok and relative_ok and conn_ok),
    )


# --------------------------------------------------------------------------
# scenarios whose answer is known by construction
# --------------------------------------------------------------------------


def _fine_centres(n: int, cell_km: float) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) * cell_km


def disc_mask(
    ny_coarse: int, nx_coarse: int, refine: int, *, cx_km: float, cy_km: float, r_km: float
) -> np.ndarray:
    """Filled disc on the fine lattice. Area = ``pi r^2`` exactly."""
    cell = 1.0 / refine
    y = _fine_centres(ny_coarse * refine, cell)[:, None]
    x = _fine_centres(nx_coarse * refine, cell)[None, :]
    return ((x - cx_km) ** 2 + (y - cy_km) ** 2) <= r_km**2


def rotated_bar_mask(
    ny_coarse: int,
    nx_coarse: int,
    refine: int,
    *,
    cx_km: float,
    cy_km: float,
    length_km: float,
    width_km: float,
    angle_deg: float,
) -> np.ndarray:
    """Rotated rectangle on the fine lattice. Area = ``length * width`` exactly."""
    cell = 1.0 / refine
    y = _fine_centres(ny_coarse * refine, cell)[:, None] - cy_km
    x = _fine_centres(nx_coarse * refine, cell)[None, :] - cx_km
    th = math.radians(angle_deg)
    u = x * math.cos(th) + y * math.sin(th)
    v = -x * math.sin(th) + y * math.cos(th)
    return (np.abs(u) <= length_km / 2.0) & (np.abs(v) <= width_km / 2.0)


def sub_cell_texture(ny_fine: int, nx_fine: int, *, modulus: int = 10, keep: int = 8) -> np.ndarray:
    """A DETERMINISTIC ~150 m stripe texture with duty cycle ``keep / modulus``.

    Stands in for what a 30 m fire simulator actually emits: a burned region
    riddled with unburned interstices far below the 1 km output scale. It is the
    discriminator that matters, because on a SMOOTH shape nearest-neighbour and
    area-fraction agree to within a cell (measured: ring fronts of width
    0.8-1.6 km, 36 sub-cell offsets, mean areas within 1% of each other). The two
    rules only separate when there is structure below the coarse cell - which is
    the entire reason a 30 m -> 1 km step exists.
    """
    i, j = np.meshgrid(np.arange(ny_fine), np.arange(nx_fine), indexing="ij")
    return ((i * 7 + j * 11) % modulus) < keep


def textured_disc_mask(
    ny_coarse: int, nx_coarse: int, refine: int, *, cx_km: float, cy_km: float, r_km: float
) -> np.ndarray:
    """A disc perforated by :func:`sub_cell_texture`.

    KNOWN BY CONSTRUCTION: every interior 1 km cell is 80% covered, so under an
    area-fraction rule at threshold 0.5 the correct coarse answer is **the whole
    disc**, area ``pi r^2``. A low-pass rule recovers it; a point sample cannot,
    because it is sampling the texture rather than averaging it away.
    """
    disc = disc_mask(ny_coarse, nx_coarse, refine, cx_km=cx_km, cy_km=cy_km, r_km=r_km)
    return disc & sub_cell_texture(*disc.shape)


def diagonal_finger_mask(
    ny_coarse: int,
    nx_coarse: int,
    refine: int,
    *,
    cx_km: float,
    cy_km: float,
    length_km: float,
    width_km: float,
    angle_deg: float,
) -> np.ndarray:
    """A thin finger. Below 1 km wide it is NOT representable - see
    :func:`resolution_limit_probe`, which measures the loss rather than
    pretending a threshold rule can avoid it."""
    return rotated_bar_mask(
        ny_coarse,
        nx_coarse,
        refine,
        cx_km=cx_km,
        cy_km=cy_km,
        length_km=length_km,
        width_km=width_km,
        angle_deg=angle_deg,
    )


@dataclass(frozen=True)
class Scenario:
    name: str
    ny: int
    nx: int
    build: Callable[[int], np.ndarray]
    area_km2: float
    perimeter_km: float
    components: int = 1
    check_relative: bool = True


_DISC_R = 6.0
_BAR_L, _BAR_W = 14.0, 3.0
_FINGER_L, _FINGER_W = 18.0, 0.75

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="disc_r6km",
        ny=20,
        nx=20,
        build=lambda r: disc_mask(20, 20, r, cx_km=10.0, cy_km=10.0, r_km=_DISC_R),
        area_km2=math.pi * _DISC_R**2,
        perimeter_km=2.0 * math.pi * _DISC_R,
    ),
    Scenario(
        name="bar_14x3km_rot30",
        ny=22,
        nx=22,
        build=lambda r: rotated_bar_mask(
            22,
            22,
            r,
            cx_km=11.0,
            cy_km=11.0,
            length_km=_BAR_L,
            width_km=_BAR_W,
            angle_deg=30.0,
        ),
        area_km2=_BAR_L * _BAR_W,
        perimeter_km=2.0 * (_BAR_L + _BAR_W),
    ),
    Scenario(
        name="textured_disc_r6km",
        ny=20,
        nx=20,
        build=lambda r: textured_disc_mask(20, 20, r, cx_km=10.0, cy_km=10.0, r_km=_DISC_R),
        area_km2=math.pi * _DISC_R**2,
        perimeter_km=2.0 * math.pi * _DISC_R,
    ),
)


def resolution_limit_probe(refine: int = DEFAULT_REFINE) -> list[dict[str, Any]]:
    """MEASURE what a 1 km output cannot represent. Reported, never pass/fail.

    A finger narrower than a coarse cell is not recoverable by ANY binary rule at
    that resolution - an area-fraction threshold loses it when it straddles a cell
    boundary, and a point sample loses it when it misses a centre. Asserting
    otherwise would be a test that cannot fail in the direction that matters.
    This is the honest statement of the limit, with numbers, so that ELMFIRE's
    native fingering is known to be partly un-representable in C5's 1 km output
    BEFORE anyone reads a G5 table.
    """
    rows: list[dict[str, Any]] = []
    for width in (0.5, 0.75, 1.0, 1.5, 2.0):
        fine = diagonal_finger_mask(
            26,
            26,
            refine,
            cx_km=13.0,
            cy_km=13.0,
            length_km=_FINGER_L,
            width_km=width,
            angle_deg=20.0,
        )
        coarse = coarsen_occupancy(fine, refine)
        true_area = _FINGER_L * width
        area = float(np.count_nonzero(coarse))
        rows.append(
            {
                "finger_width_km": width,
                "true_area_km2": round(true_area, 3),
                "coarse_area_km2": area,
                "area_retained": round(area / true_area, 4) if true_area else None,
                "components": n_components(coarse),
                "representable": bool(width >= 1.0),
            }
        )
    return rows


def translation_sweep(
    refine: int = DEFAULT_REFINE, *, n_offsets: int = 6
) -> dict[str, dict[str, float]]:
    """Sub-cell translation stability on the textured disc.

    The true area is translation-invariant, so a faithful coarsening's must be
    too. Reported for the rule and for every planted defect, so that a defect's
    failure cannot be dismissed as one unlucky phase.
    """
    out: dict[str, dict[str, float]] = {}
    fns: dict[str, Callable[..., np.ndarray]] = {"occupancy_0.5": coarsen_occupancy}
    fns.update(DEFECTIVE_COARSENERS)
    offsets = np.linspace(0.0, 1.0, n_offsets, endpoint=False)
    for name, fn in fns.items():
        areas: list[float] = []
        for dy in offsets:
            for dx in offsets:
                fine = disc_mask(
                    22, 22, refine, cx_km=11.0 + dx, cy_km=11.0 + dy, r_km=_DISC_R
                ) & sub_cell_texture(22 * refine, 22 * refine)
                areas.append(float(np.count_nonzero(fn(fine, refine))))
        arr = np.asarray(areas)
        out[name] = {
            "mean_area_km2": round(float(arr.mean()), 3),
            "sd_area_km2": round(float(arr.std()), 3),
            "min_area_km2": float(arr.min()),
            "max_area_km2": float(arr.max()),
            "n_offsets": int(arr.size),
        }
    return out


def run_playthrough(refine: int = DEFAULT_REFINE) -> dict[str, Any]:
    """PLAYTHROUGH 1 - coarsening correctness, no human in the loop.

    Every scenario's area and component count are known analytically. The rule
    must pass all of them; each planted defect must be CAUGHT by at least one
    criterion on at least one scenario, and the report says which criterion caught
    which defect, so "the harness detected it" is checkable rather than asserted.

    Both denominators are refused when empty, and the second is the one that
    matters. ``all_defects_caught = all(v for v in caught.values())`` is
    vacuously True over an emptied :data:`DEFECTIVE_COARSENERS`, so a
    playthrough that planted NO defects would report ``every_planted_defect_
    caught: true`` and PASS. ADR-030 makes this target a gate on the grounds
    that a playthrough that cannot fail turns the suite red; with no defects
    declared it cannot fail, and it would turn the suite GREEN. That is the
    gate's own premise inverted, and nothing in ``src/``, ``tests/`` or
    ``tools/`` pinned the dict's non-emptiness. The sibling playthrough in
    ``sim/playthrough.py`` had the equivalent guard and this one did not, which
    is invisible from either report.

    Refusal is local and deliberate: the shared helper that turns a
    ``defects_caught_by`` map into a coverage verdict lives in ``common/``,
    which this package does not own and does not import. Guarding here means
    the report cannot be BUILT over zero defects, independently of what any
    downstream consumer decides to do with it.
    """
    refuse_if_empty(
        "coarsening playthrough",
        {"scenarios": len(SCENARIOS), "planted_defects": len(DEFECTIVE_COARSENERS)},
        because=(
            "a rule that passed no scenarios and a harness that caught all zero of its "
            "planted defects both read PASS."
        ),
    )
    rule_rows: list[dict[str, Any]] = []
    defect_rows: list[dict[str, Any]] = []
    for sc in SCENARIOS:
        fine = sc.build(refine)
        rel_tol = RELATIVE_TOL if sc.check_relative else math.inf
        good = coarsen_occupancy(fine, refine)
        rule_rows.append(
            score_coarsening(
                scenario=sc.name,
                coarsener="occupancy_0.5",
                fine=fine,
                coarse=good,
                true_area_km2=sc.area_km2,
                perimeter_km=sc.perimeter_km,
                true_components=sc.components,
                relative_tol=rel_tol,
            ).as_dict()
        )
        for name, fn in DEFECTIVE_COARSENERS.items():
            defect_rows.append(
                score_coarsening(
                    scenario=sc.name,
                    coarsener=name,
                    fine=fine,
                    coarse=fn(fine, refine),
                    true_area_km2=sc.area_km2,
                    perimeter_km=sc.perimeter_km,
                    true_components=sc.components,
                    relative_tol=rel_tol,
                ).as_dict()
            )

    rule_passes = all(r["passed"] for r in rule_rows)
    caught: dict[str, list[str]] = {}
    for name in DEFECTIVE_COARSENERS:
        rows = [r for r in defect_rows if r["coarsener"] == name and not r["passed"]]
        why: list[str] = []
        for r in rows:
            if not (r["area_ok"] and r["relative_ok"]):
                why.append(f"{r['scenario']}:area")
            if not r["connectivity_ok"]:
                why.append(f"{r['scenario']}:connectivity")
        caught[name] = why
    all_defects_caught = all(v for v in caught.values())

    return {
        "playthrough": "coarsening_correctness",
        "rule": COARSENING_RULE,
        "refine": int(refine),
        "fine_cell_m": round(1000.0 / refine, 4),
        "threshold": OCCUPANCY_THRESHOLD,
        "n_scenarios": len(SCENARIOS),
        "n_planted_defects": len(DEFECTIVE_COARSENERS),
        "rule_rows": rule_rows,
        "defect_rows": defect_rows,
        "defects_caught_by": caught,
        "resolution_limit_measured": resolution_limit_probe(refine),
        "translation_sweep": translation_sweep(refine),
        "rule_passes_every_scenario": bool(rule_passes),
        "every_planted_defect_caught": bool(all_defects_caught),
        "verdict": "PASS" if (rule_passes and all_defects_caught) else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="python -m wildfire_nowcast.sim.coarsen", allow_abbrev=False)
    ap.add_argument("--refine", type=int, default=DEFAULT_REFINE)
    ap.add_argument("--out", default="reports/figures/playthrough_coarsening.json")
    args = ap.parse_args(argv)
    report = run_playthrough(args.refine)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    for row in report["rule_rows"]:
        print(
            f"[coarsen] {row['scenario']:<26} true {row['true_area_km2']:8.3f} km2  "
            f"coarse {row['coarse_area_km2']:8.3f}  rel {row['rel_err']:.4f}  "
            f"comp {row['components_coarse']}  {'ok' if row['passed'] else 'FAIL'}"
        )
    for name, why in report["defects_caught_by"].items():
        print(f"[coarsen] planted defect {name:<13} caught by: {why or 'NOTHING — BAD'}")
    print(f"[coarsen] {report['verdict']}  -> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
