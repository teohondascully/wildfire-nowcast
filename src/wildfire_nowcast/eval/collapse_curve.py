"""M19: the curve of ``independence_dispersion_index`` at ``latent_sigma=0``.

    .venv/bin/python -m wildfire_nowcast.eval.collapse_curve --out <artifact>.json

THE PREMISE THIS MODULE RESTS ON, STATED HERE RATHER THAN CITED
---------------------------------------------------------------
This project models pixels as conditionally independent Bernoulli ONLY given a
shared per-step latent ``z_t``. A kernel whose only stochasticity is
independent per-pixel noise is retained as an ABLATION and is never a
candidate. The reason is arithmetic before it is empirical: a sum of order
10^4 independent Bernoullis concentrates on its mean, so members differ pixel
by pixel while their total burned areas do not, and the ensemble carries no
usable spread. It is also measured, on this repository's own arms: removing the
shared latent narrows the ensemble by 3.7 to 7.8 times on 16 of 25 arms, and by
1.1 to 1.6 times on the other nine, which had almost no dispersion left to
lose. README.md states the same commitment publicly and carries those numbers.

THIS FILE CARRIES THE DIRECT MEASUREMENT OF IT, SO IT POINTS NOWHERE FOR IT
---------------------------------------------------------------------------
The INSTRUMENT is controlled before it is used as a control. On a field that is
independent BY CONSTRUCTION the index reads 1.0338, 1.0052, 1.0109, 1.0106,
0.9936 at 24, 48, 96, 192, 384 members, with the seed spread falling 0.1215 to
0.0429, and 0.9674 at 24 members over 200 seeds, judged COLLAPSED in 200 of 200
draws. Without that reading, an index above 1.0 could mean a broken estimator
just as easily as a correlated field, and nothing below could tell the two
apart.

Then the premise itself. With ``latent_sigma=0`` at horizon 1, one step of this
ensemble sits ON the independent-pixel floor: 1.0154, 1.0273, 0.9998, 1.0133,
1.0048 at the same five member counts, 12 of 12 seeds judged COLLAPSED at every
one of them. Switching off the shared latent leaves nothing above the floor,
and that is what collapse means here. The claim is measured in this file rather
than repeated from somewhere a reader of this file cannot open.

The same sweep is equally clear about what it does NOT establish, which is why
the module exists at all. At ``latent_sigma=0`` and horizon 3 the index reads
1.5114 over 200 seeds (sd 0.2073, range 0.9690 to 2.2568) and judges the
documented control COLLAPSED in only 97 of 200 draws, a coin flip. It does not
converge to 1.0 with size either: at 384 members and horizon 3 it reads 1.4713,
1.4296, 1.3928, 1.3985, 1.3716 on domains 30, 60, 120, 240, 480. So a
MULTI-STEP reading of this index is not a statement about the latent, and the
premise above is legible at horizon 1 and unreadable above it. The section on
the null, below, says why from the algebra rather than from these numbers.

WHAT IS BEING ASKED, AND WHY IT IS A MEASUREMENT AND NOT A DECISION
------------------------------------------------------------------
G3's wording is "ablation demonstrates collapse". The documented positive
control for that is
``StubEnsemble(latent_sigma=0)``: with the shared latent held at zero the
ensemble should behave as independent pixels, so
:func:`wildfire_nowcast.sim.ensemble.independence_dispersion_index` should read
near its null and the ensemble should be judged COLLAPSED. On a 30x30 / 3 h /
24-member domain it reads 1.5299 against ``COLLAPSE_INDEX_THRESHOLD = 1.5``,
i.e. NOT collapsed, 2% the wrong side of the bar.

**The threshold is not moved here and must not be moved anywhere.** This module
computes the curve that the question actually needs: the index at
``latent_sigma=0`` as a function of ensemble MEMBERS, DOMAIN and HORIZON, with
several seeds per cell so the answer carries a spread rather than a point.

THE NULL IS 1.0 FOR A ONE-STEP FIELD AND FOR NOTHING ELSE
---------------------------------------------------------
The index is ``std(member areas) / sqrt(sum_i p_i (1 - p_i))``. The denominator
is the burned-area standard deviation of a field of independent Bernoullis
carrying the ensemble's own marginals, so ``1.0`` is the independent-pixel null
by construction, with no fitted constant anywhere. That derivation holds
exactly when the indicators being summed are independent.

They are independent for ONE step of the stub with ``z`` held at zero: given a
frozen state, each candidate cell ignites on its own draw. They are NOT
independent for two steps or more, because the step-2 candidate set is a
function of the step-1 draw. A member that happened to ignite more cells at
step 1 offers more front to step 2, so member areas at step 3 are
over-dispersed relative to the independent-pixel floor **with no shared latent
anywhere in the model**. Contagion is itself a correlating mechanism.

So the quantity this module reports for each cell is not only the index but the
horizon it was measured at, and the headline question - does it converge to
1.0 - is asked separately per horizon rather than pooled.

TWO CONTROLS SIT BESIDE THE CURVE, FOR THE TWO WAYS IT COULD MISLEAD
--------------------------------------------------------------------
:func:`constructed_independent_index` measures the index on a field that IS
independent by construction, at the same member counts. Any member dependence
there is estimator bias and belongs to the instrument; member dependence in the
stub that is absent here belongs to the process. Without it, a curve that moves
with members cannot be attributed.

:func:`stub_index` at ``horizon_h=1`` is the process control at the other end:
the stub's own dynamics, with the multi-step coupling removed. It has a known
answer, 1.0, forced by the algebra above rather than by a previous run.

WHAT THIS MODULE MAY NOT BE USED FOR
------------------------------------
``sim/stub_model.py``'s docstring forbids the stub from appearing in a gate, a
report for anyone outside this repo, or a comparison against ELMFIRE. Nothing
here is a result about the learned kernel or about G3's outcome. It is a
measurement of a CONTROL, which is the one purpose ``latent_sigma=0`` is
retained for.

``sim/ensemble.py`` is simviz's file. It is imported, never edited, and the
index used here is the shipped one rather than a copy, so the curve describes
the instrument in the tree and not a re-derivation of it.

**A CLONE CANNOT OPEN THIS MODULE'S SWEEP ARTIFACTS**, and no path to one is
written here for that reason. They land wherever ``--out`` says, and the run
directory is excluded from the public tree by ``.gitignore``; naming a path a
clone cannot open is the ADR-105 (3) defect and would trade one unopenable
reference for another. Two things stand in its place, and both are better than
a path. Every number this module asserts is written out above, in full, at the
setting it was measured at. And THE CURVE IS RE-DERIVABLE by the one command at
the top of this docstring: nothing here reads a stored value, so the artifacts
are a record of a run and never the support for a claim.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from wildfire_nowcast.common.contract import BURNING
from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.sim.absent import EXIT_NOTHING_EXAMINED, refuse_if_empty
from wildfire_nowcast.sim.c5 import STATIC_C5, WEATHER_C5

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DOMAINS",
    "DEFAULT_HORIZONS",
    "DEFAULT_MEMBERS",
    "FIRE_SCALINGS",
    "SweepCell",
    "CellSummary",
    "collapse_scene",
    "collapse_threshold",
    "convergence_at_largest",
    "constructed_independent_index",
    "fire_radius",
    "format_report",
    "main",
    "smallest_firing_configuration",
    "stub_index",
    "summarise",
    "sweep",
]

#: The member axis. 24 is the documented control and the CLI default in
#: `sim/ensemble.py`; the rest double from there so a 1/M bias would be visible
#: as a straight line on a log axis rather than as three scattered points.
DEFAULT_MEMBERS: Final[tuple[int, ...]] = (24, 48, 96, 192, 384)

#: The domain axis. 30 is the domain the 1.5299 was measured on.
DEFAULT_DOMAINS: Final[tuple[int, ...]] = (30, 60, 120)

#: The horizon axis, which the brief did not ask for and which is the axis the
#: mechanism above predicts the effect lives on. 1 is the known-answer control.
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 6)

#: How the initial fire is sized against the domain.
#:
#: ``fixed`` keeps a one-cell ignition, so growing the domain adds empty margin
#: and nothing else: at horizon 3 the fire never reaches the edge of a 30x30
#: box, so 60x60 and 120x120 are the SAME experiment with padding. That is what
#: "domain size" naively means and it is reported because it is what a reader
#: assumes was swept.
#:
#: ``scaled`` grows the ignition disc with the domain, so the number of front
#: cells - the actual denominator of the index - grows too. That is the axis
#: worth a curve, and the two are reported separately rather than averaged,
#: because averaging them would hide that one of them is inert.
FIRE_SCALINGS: Final[tuple[str, ...]] = ("fixed", "scaled")

_U: Final[int] = WEATHER_C5.index("wind_u10")
_V: Final[int] = WEATHER_C5.index("wind_v10")
_RH: Final[int] = WEATHER_C5.index("rh_2m")

#: Weather held constant across the sweep: a steady 5 m/s wind toward the
#: south-east and 25% RH. Constant on purpose - a varying driver would put a
#: second source of member-to-member spread into a measurement whose whole
#: subject is where the spread comes from.
_WIND_U: Final[float] = 4.0
_WIND_V: Final[float] = -3.0
_RH_PCT: Final[float] = 25.0


def fire_radius(domain: int, scaling: str) -> int:
    """Initial burning-disc radius in cells for ``domain`` under ``scaling``."""
    if scaling == "fixed":
        return 1
    if scaling == "scaled":
        return max(1, domain // 15)
    raise ValueError(f"scaling must be one of {FIRE_SCALINGS}, got {scaling!r}")


def collapse_scene(
    domain: int, radius: int, horizon_h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A C5 ``(x0, static, weather)`` triple: a centred burning disc, steady wind.

    No barriers and no terrain. Every static channel is zero, which makes
    ``water_barrier_mask`` zero, so nothing in the scene can stop the front and
    the only variation between members is the ignition draw. That is the point:
    this is an instrument test, not a fire.
    """
    if domain < 3 or radius < 1 or horizon_h < 1:
        raise ValueError(f"domain={domain}, radius={radius}, horizon_h={horizon_h}")
    x0 = np.zeros((domain, domain), dtype=np.uint8)
    yy, xx = np.mgrid[0:domain, 0:domain]
    centre = domain // 2
    x0[((yy - centre) ** 2 + (xx - centre) ** 2) <= radius * radius] = BURNING

    static = np.zeros((len(STATIC_C5), domain, domain), dtype=np.float32)
    weather = np.zeros((horizon_h, len(WEATHER_C5), domain, domain), dtype=np.float32)
    weather[:, _U] = _WIND_U
    weather[:, _V] = _WIND_V
    weather[:, _RH] = _RH_PCT
    return x0, static, weather


def _shipped_instruments() -> tuple[Callable[..., float], Callable[..., dict[str, Any]]]:
    """``(independence_dispersion_index, ensemble_diagnostics)`` from ``sim/``.

    Imported inside a function rather than at module scope because
    ``sim/ensemble.py`` runs ``matplotlib.use("Agg")`` at import time. A module
    scope import would make selecting a matplotlib backend a side effect of
    importing anything in ``eval/``, which is a thing a scoring package must not
    do to a program that has its own opinion about backends.
    """
    from wildfire_nowcast.sim.ensemble import (  # noqa: PLC0415
        ensemble_diagnostics,
        independence_dispersion_index,
    )

    return independence_dispersion_index, ensemble_diagnostics


def collapse_threshold() -> float:
    """``COLLAPSE_INDEX_THRESHOLD`` read from ``sim/``, never restated here.

    A curve that compares against a locally written 1.5 would keep agreeing with
    itself after the shipped constant moved. It is read so that if anyone does
    move it, every number in this artifact moves with it and says so.
    """
    from wildfire_nowcast.sim.ensemble import COLLAPSE_INDEX_THRESHOLD  # noqa: PLC0415

    return float(COLLAPSE_INDEX_THRESHOLD)


@dataclass(frozen=True)
class SweepCell:
    """One measurement: one scene, one member count, one seed."""

    family: str
    domain: int
    radius: int
    horizon_h: int
    members: int
    seed: int
    index: float
    collapsed: bool
    n_distinct_members: int
    uncertain_cell_frac: float
    area_mean: float
    area_sd: float
    independent_floor_sd: float


def stub_index(
    domain: int,
    horizon_h: int,
    members: int,
    seed: int,
    *,
    scaling: str = "scaled",
    latent_sigma: float = 0.0,
) -> SweepCell:
    """Measure the shipped index on one ``StubEnsemble(latent_sigma)`` ensemble.

    ``latent_sigma`` is a parameter and defaults to the control's value rather
    than being hard-wired, so the same function draws the healthy arm and the
    curve can be read against something other than itself.
    """
    from wildfire_nowcast.sim.stub_model import StubEnsemble  # noqa: PLC0415

    index_fn, diagnostics_fn = _shipped_instruments()
    radius = fire_radius(domain, scaling)
    x0, static, weather = collapse_scene(domain, radius, horizon_h)
    samples = StubEnsemble(latent_sigma=latent_sigma).predict(
        x0, static, weather, members, horizon_h, seed
    )
    diag = diagnostics_fn(samples)
    final = samples[:, -1] > 0
    areas = final.sum(axis=(1, 2)).astype(np.float64)
    marginal = final.mean(axis=0)
    return SweepCell(
        family=f"stub_sigma{latent_sigma:g}_{scaling}",
        domain=domain,
        radius=radius,
        horizon_h=horizon_h,
        members=members,
        seed=seed,
        index=float(index_fn(samples)),
        collapsed=bool(diag["collapsed"]),
        n_distinct_members=int(diag["n_distinct_members"]),
        uncertain_cell_frac=float(diag["uncertain_cell_frac"]),
        area_mean=float(areas.mean()),
        area_sd=float(areas.std()),
        independent_floor_sd=float(np.sqrt(np.sum(marginal * (1.0 - marginal)))),
    )


def constructed_independent_index(
    n_cells: int, members: int, seed: int, *, p_low: float = 0.15, p_high: float = 0.85
) -> SweepCell:
    """The ESTIMATOR control: independent Bernoullis, no dynamics at all.

    The known answer is 1.0 and it is forced by algebra, not by a prior run:
    ``Var(sum of independent indicators) = sum p_i (1 - p_i)``, which is exactly
    what the index divides by. Any drift with ``members`` here is a property of
    the estimator; drift in the stub that does NOT appear here is a property of
    the process. That separation is the reason this control exists, and without
    it "the curve moves with members" is unattributable.
    """
    index_fn, diagnostics_fn = _shipped_instruments()
    rng = np.random.default_rng(seed)
    p = rng.uniform(p_low, p_high, size=n_cells)
    side = int(round(math.sqrt(n_cells)))
    if side * side != n_cells:
        raise ValueError(f"n_cells must be a perfect square for the [M,1,H,W] shape, got {n_cells}")
    draws = (rng.random((members, n_cells)) < p).astype(np.uint8)
    samples = draws.reshape(members, 1, side, side)
    diag = diagnostics_fn(samples)
    final = samples[:, -1] > 0
    areas = final.sum(axis=(1, 2)).astype(np.float64)
    marginal = final.mean(axis=0)
    return SweepCell(
        family="constructed_independent",
        domain=side,
        radius=0,
        horizon_h=1,
        members=members,
        seed=seed,
        index=float(index_fn(samples)),
        collapsed=bool(diag["collapsed"]),
        n_distinct_members=int(diag["n_distinct_members"]),
        uncertain_cell_frac=float(diag["uncertain_cell_frac"]),
        area_mean=float(areas.mean()),
        area_sd=float(areas.std()),
        independent_floor_sd=float(np.sqrt(np.sum(marginal * (1.0 - marginal)))),
    )


def sweep(
    *,
    members: Sequence[int] = DEFAULT_MEMBERS,
    domains: Sequence[int] = DEFAULT_DOMAINS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    scalings: Sequence[str] = FIRE_SCALINGS,
    n_seeds: int = 8,
    latent_sigma: float = 0.0,
) -> list[SweepCell]:
    """Every (scaling, domain, horizon, members, seed) cell, plus the controls."""
    refuse_if_empty(
        "collapse_curve.sweep",
        {
            "members": len(members),
            "domains": len(domains),
            "horizons": len(horizons),
            "scalings": len(scalings),
            "seeds": int(n_seeds),
        },
        because="a curve over an empty axis is not a flat curve, it is no curve.",
    )
    cells: list[SweepCell] = []
    total = len(scalings) * len(domains) * len(horizons) * len(members) * n_seeds
    logger.info("sweeping %d stub cells at latent_sigma=%g", total, latent_sigma)
    started = time.perf_counter()
    done = 0
    for scaling in scalings:
        for domain in domains:
            for horizon_h in horizons:
                for n_members in members:
                    for seed in range(n_seeds):
                        cells.append(
                            stub_index(
                                domain,
                                horizon_h,
                                n_members,
                                seed,
                                scaling=scaling,
                                latent_sigma=latent_sigma,
                            )
                        )
                        done += 1
                logger.debug(
                    "%s domain=%d horizon=%d done (%d/%d, %.1fs elapsed)",
                    scaling,
                    domain,
                    horizon_h,
                    done,
                    total,
                    time.perf_counter() - started,
                )

    logger.info("sweeping the constructed-independent estimator control")
    for n_members in members:
        for seed in range(n_seeds):
            cells.append(constructed_independent_index(900, n_members, seed))
    logger.info("%d cells in %.1fs", len(cells), time.perf_counter() - started)
    return cells


@dataclass(frozen=True)
class CellSummary:
    """One (family, domain, horizon, members) cell aggregated across seeds."""

    family: str
    domain: int
    horizon_h: int
    members: int
    n_seeds: int
    index_mean: float
    index_sd: float
    index_min: float
    index_max: float
    n_collapsed: int
    area_mean: float

    @property
    def fires_every_seed(self) -> bool:
        """Every seed in this cell was judged collapsed by the shipped verdict."""
        return self.n_collapsed == self.n_seeds


def summarise(cells: Sequence[SweepCell]) -> list[CellSummary]:
    """Aggregate seeds within each cell. Refuses an empty sweep."""
    refuse_if_empty(
        "collapse_curve.summarise",
        {"cells": len(cells)},
        because="a summary over zero measurements has no true value, only a tidy one.",
    )
    keys = sorted({(c.family, c.domain, c.horizon_h, c.members) for c in cells})
    out: list[CellSummary] = []
    for family, domain, horizon_h, members in keys:
        group = [
            c
            for c in cells
            if (c.family, c.domain, c.horizon_h, c.members) == (family, domain, horizon_h, members)
        ]
        values = np.array([c.index for c in group], dtype=np.float64)
        out.append(
            CellSummary(
                family=family,
                domain=domain,
                horizon_h=horizon_h,
                members=members,
                n_seeds=len(group),
                index_mean=float(values.mean()),
                index_sd=float(values.std(ddof=1)) if len(group) > 1 else float("nan"),
                index_min=float(values.min()),
                index_max=float(values.max()),
                n_collapsed=sum(1 for c in group if c.collapsed),
                area_mean=float(np.mean([c.area_mean for c in group])),
            )
        )
    return out


def smallest_firing_configuration(
    summaries: Sequence[CellSummary], *, family: str, horizon_h: int
) -> CellSummary | None:
    """The cheapest cell in which the control fires on EVERY seed, or ``None``.

    Cheapest is by ``members * domain**2``, which is what the sweep actually
    costs. ``None`` is returned, and must be rendered as "no such configuration",
    rather than the smallest element of an empty set or a nearest miss: a
    control that fires on 7 of 8 seeds has not been shown to fire.
    """
    firing = [
        s
        for s in summaries
        if s.family == family and s.horizon_h == horizon_h and s.fires_every_seed
    ]
    if not firing:
        return None
    return min(firing, key=lambda s: (s.members * s.domain * s.domain, s.members, s.domain))


def convergence_at_largest(
    summaries: Sequence[CellSummary], *, family: str, horizon_h: int
) -> dict[str, float] | None:
    """Distance from the 1.0 null at the largest member count, in SEM units.

    Reported in standard ERRORS of the cell mean rather than in seed SDs,
    because the question is whether the LIMIT is 1.0 and the limit is estimated
    by that mean. ``None`` when the family carries no row at this horizon; a
    caller must render that as absent rather than as zero distance.
    """
    rows = [s for s in summaries if s.family == family and s.horizon_h == horizon_h]
    if not rows:
        return None
    largest = max(rows, key=lambda s: s.members)
    sd = largest.index_sd
    sem = sd / math.sqrt(largest.n_seeds) if largest.n_seeds > 1 and math.isfinite(sd) else 0.0
    excess = largest.index_mean - 1.0
    return {
        "domain": float(largest.domain),
        "members": float(largest.members),
        "index_mean": largest.index_mean,
        "index_sd": sd,
        "excess_over_null": excess,
        "excess_in_sem": excess / sem if sem > 0 else float("inf"),
    }


def format_report(cells: Sequence[SweepCell], threshold: float) -> list[str]:
    """The program's OUTPUT: the curve, the controls and the one-sentence answer."""
    summaries = summarise(cells)
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("M19  independence_dispersion_index at latent_sigma=0, vs members and domain")
    lines.append("=" * 100)
    lines.append(
        f"threshold read from sim/ensemble.py: COLLAPSE_INDEX_THRESHOLD = {threshold:g} "
        "(READ, NEVER MOVED)"
    )
    lines.append(
        "null for the index is 1.0 and is exact ONLY for a one-step field; see the module "
        "docstring."
    )
    lines.append("")

    families = sorted({s.family for s in summaries})
    for family in families:
        rows = [s for s in summaries if s.family == family]
        lines.append(f"-- {family} " + "-" * max(0, 96 - len(family)))
        lines.append(
            f"{'domain':>7}{'horizon':>9}{'members':>9}{'seeds':>7}"
            f"{'index':>10}{'sd':>9}{'min':>9}{'max':>9}{'collapsed':>11}{'area':>9}"
        )
        for s in sorted(rows, key=lambda r: (r.domain, r.horizon_h, r.members)):
            sd = "n/a" if math.isnan(s.index_sd) else f"{s.index_sd:.4f}"
            lines.append(
                f"{s.domain:>7}{s.horizon_h:>9}{s.members:>9}{s.n_seeds:>7}"
                f"{s.index_mean:>10.4f}{sd:>9}{s.index_min:>9.4f}{s.index_max:>9.4f}"
                f"{s.n_collapsed:>7}/{s.n_seeds:<3}{s.area_mean:>9.1f}"
            )
        lines.append("")

    lines.append("-- convergence at the largest member count " + "-" * 57)
    lines.append(
        f"{'family':>34}{'horizon':>9}{'members':>9}{'index':>10}{'sd':>9}"
        f"{'index-1.0':>12}{'in SEM':>10}"
    )
    for family in families:
        for horizon_h in sorted({s.horizon_h for s in summaries if s.family == family}):
            verdict = convergence_at_largest(summaries, family=family, horizon_h=horizon_h)
            if verdict is None:
                continue
            largest_sd = verdict["index_sd"]
            sd_text = "n/a" if not math.isfinite(largest_sd) else f"{largest_sd:.4f}"
            lines.append(
                f"{family:>34}{horizon_h:>9}{int(verdict['members']):>9}"
                f"{verdict['index_mean']:>10.4f}{sd_text:>9}"
                f"{verdict['excess_over_null']:>12.4f}{verdict['excess_in_sem']:>10.1f}"
            )
    lines.append("")

    lines.append("-- smallest configuration at which the CONTROL FIRES ON EVERY SEED " + "-" * 33)
    for family in families:
        for horizon_h in sorted({s.horizon_h for s in summaries if s.family == family}):
            best = smallest_firing_configuration(summaries, family=family, horizon_h=horizon_h)
            if best is None:
                lines.append(
                    f"  {family} h={horizon_h}: NONE. No swept configuration collapses on every "
                    "seed."
                )
            else:
                lines.append(
                    f"  {family} h={horizon_h}: domain {best.domain}x{best.domain}, "
                    f"{best.members} members, index {best.index_mean:.4f} "
                    f"(max over seeds {best.index_max:.4f} vs bar {threshold:g})"
                )
    lines.append("")
    lines.append(f"cells measured: {len(cells)}   summarised cells: {len(summaries)}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    """Parser as its own function so a test can assert PARSED BEHAVIOUR.

    ADR-104 (5): a wiring test that asserts a flag name is a substring of the
    help text cannot tell a flag from a flag that has been disabled by renaming.
    """
    p = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.eval.collapse_curve",
        description=(
            "Curve of independence_dispersion_index at latent_sigma=0 vs members, domain "
            "and horizon. Reads COLLAPSE_INDEX_THRESHOLD; never writes it."
        ),
    )
    p.add_argument("--out", default=None, help="write the full sweep as JSON to this path")
    p.add_argument("--members", default=",".join(str(m) for m in DEFAULT_MEMBERS))
    p.add_argument("--domains", default=",".join(str(d) for d in DEFAULT_DOMAINS))
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    p.add_argument("--scalings", default=",".join(FIRE_SCALINGS))
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument(
        "--latent-sigma",
        type=float,
        default=0.0,
        help="0.0 is the documented positive control; a positive value draws the healthy arm",
    )
    add_logging_arguments(p)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep, print the report, optionally write the artifact."""
    args = build_parser().parse_args(argv)
    configure_from_args(args, default_verbosity=1)

    members = tuple(int(v) for v in str(args.members).split(",") if v)
    domains = tuple(int(v) for v in str(args.domains).split(",") if v)
    horizons = tuple(int(v) for v in str(args.horizons).split(",") if v)
    scalings = tuple(v for v in str(args.scalings).split(",") if v)

    cells = sweep(
        members=members,
        domains=domains,
        horizons=horizons,
        scalings=scalings,
        n_seeds=int(args.seeds),
        latent_sigma=float(args.latent_sigma),
    )
    threshold = collapse_threshold()
    for line in format_report(cells, threshold):
        print(line)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": "M19",
            "question": (
                "does independence_dispersion_index at latent_sigma=0 converge to 1.0 as "
                "members and domain grow"
            ),
            "collapse_index_threshold_read_from_sim": threshold,
            "latent_sigma": float(args.latent_sigma),
            "n_cells": len(cells),
            "axes": {
                "members": list(members),
                "domains": list(domains),
                "horizons": list(horizons),
                "scalings": list(scalings),
                "seeds": int(args.seeds),
            },
            "cells": [asdict(c) for c in cells],
            "summary": [asdict(s) for s in summarise(cells)],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"written: {path}")
    return 0 if cells else EXIT_NOTHING_EXAMINED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
