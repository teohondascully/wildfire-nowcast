"""M21: what the two collapse BARS are worth, per horizon, derived and measured.

    .venv/bin/python -m wildfire_nowcast.eval.collapse_bars --out <artifact>.json

TWO INSTRUMENTS ADOPTED THE NUMBER 1.5 FOR DIFFERENT ESTIMANDS
---------------------------------------------------------------
They did so independently - ``eval``'s ``> 1.5`` is a topological ancestor of
``COLLAPSE_INDEX_THRESHOLD``, so neither was copied from the other - and each
was wrong in a different way: the index in its LEVEL, the ratio in its TAIL
(ADR-119 (4)). **``eval``'s has since moved to 5.0 (M22, below); the
description of the coincidence is kept because it is what the measurement was
for.**

``sim/ensemble.py`` holds ``COLLAPSE_INDEX_THRESHOLD = 1.5`` on
``independence_dispersion_index``, which is an EMPIRICAL spread over a
THEORETICAL one: ``std(member areas) / sqrt(sum_i p_i (1 - p_i))``. Its null is
1.0 by algebra for a one-step field and, as
:mod:`wildfire_nowcast.eval.collapse_curve` measured, 1.0048 / 1.25 /
1.47-1.52 / 1.75-1.79 / 2.19-2.29 at horizons 1 / 2 / 3 / 4 / 6 with no latent
anywhere. Its null is not horizon stable.

:data:`wildfire_nowcast.eval.selftest.ABLATION_SD_RATIO_BAR` held the same
number, 1.5, on a DIFFERENT quantity: one EMPIRICAL spread over another
EMPIRICAL spread, the latent arm's across-member area SD divided by the
ablation arm's. Same literal, different estimand, so agreement between the two
was a coincidence and disagreement between them would have been a puzzle nobody
had posed. This module measures the second one the way ``collapse_curve`` measured
the first, and derives the first one's ONE-STEP bar from the algebra of the
construction it is asserted on.

THE RATIO'S NULL IS 1.0 AT EVERY HORIZON, AND THAT PART IS SOUND
-----------------------------------------------------------------
The index compares an observed spread against a MODEL of what an independent
field would give, and that model stops being right after one step because the
step-2 candidate set is a function of the step-1 draw. The ratio compares an
observed spread against ANOTHER OBSERVED SPREAD from the same process with one
switch flipped. Multi-step contagion inflates BOTH arms, so it cancels in the
quotient. Under the null, that is, when the shared latent contributes nothing,
the two arms are draws from the SAME law and ``E[ratio]`` is 1 at every
horizon. This is a real structural advantage over the index and it is measured
below rather than asserted: the null MEDIAN reads 1.000, 0.972, 0.991, 0.992,
1.008, 0.999 at horizons 1 to 6.

ITS TAIL IS NOT HORIZON STABLE, AND THE BAR USED TO SIT IN THE TAIL
--------------------------------------------------------------------
A ratio of two sample SDs is only as well behaved as its DENOMINATOR, and the
ablation arm's area SD is not stable in the horizon. On the scene the shipped
check uses it runs 0.843, 1.254, 1.419, 1.551, 1.139, 0.486 at horizons 1 to 6:
the fire decelerates, the ablation ensemble becomes near deterministic, and the
quotient acquires a heavy right tail exactly where the denominator is smallest.
Over 400 seeds with the latent OFF ON BOTH ARMS, the fraction of draws that
would have PASSED the shipped check at the retired bar of 1.5 reads

    h = 1   0.0%      h = 4    0.8%
    h = 2   1.5%      h = 5    5.2%
    h = 3   2.5%      h = 6   26.8%

for ``null_sigma_zero`` and 0.5 / 2.2 / 2.8 / 1.8 / 7.2 / 24.8% for
``null_no_latent``, which removes the head outright and is the more adversarial
of the two. So 1.5 was about the 99th percentile of this instrument's own null
at horizons 1 to 4 and about the 73rd at horizon 6, which is the horizon the
verdict is taken at. The tail is the process and not a guard: only 1.2% of the
400 draws are the degenerate case where the denominator is exactly zero, and
those are counted as FAILURES here, exactly as the shipped check counts them.
The null's 99th percentile among measurable draws runs 1.420, 1.543, 1.589,
1.480, 1.811, 3.973.

[M22] **THE BAR IS NOW 5.0 AND THE HORIZON IS STILL 6.** The bar was derived
from this instrument's own null at the lead it runs at, under the rule the check
applies, over four independent blocks: the ``null_no_latent`` 99th percentile
among measurable draws reads 4.965 (seeds 0-199), 2.888 (seeds 400-599), 3.648
(``null_sigma_zero``, seeds 400-599) and 3.973 (``null_sigma_zero``, seeds
0-399), so 5.0 is the smallest round value at or above every estimate. It buys a
false fire rate of 1.00 / 0.00 / 0.00 / 0.75% on those same four blocks against
the 24-28% that 1.5 bought, and it costs 1.8 points of power: over 400
replications the TREATMENT clears 5.0 in 97.0% of draws against 98.8% at 1.5. At
the shipped seed the ratio is 6.517.

**CALIBRATING A BAR AGAINST THIS NULL IS NOT THE VACUITY ADR-114 (3) KILLED, AND
THE DIFFERENCE IS WORTH STATING BECAUSE IT LOOKS THE SAME FROM OUTSIDE.** That
ruling rejected calibrating the INDEX against a null measured from a no-latent
run, because the index's subject IS the no-latent run, so the comparison reduces
to "the ablation equals itself". Here the verdict compares a latent-ON arm
against a latent-OFF one and the null is the sampling distribution of that
quotient when there is no latent to find. The bar is a threshold on a statistic,
calibrated against the distribution of that statistic under the hypothesis the
check exists to reject. Nothing in the verdict is measured from the null.

**WHY THE HORIZON DID NOT MOVE INSTEAD.** Relocating the verdict to lead 3, the
longest lead G3's 1-3 h wording covers, repairs nothing: at lead 3 the treatment
clears its own null's 99th percentile in 64.2% of draws, so a single-draw
verdict there trades a 27% false FIRE rate for a ~36% false SILENCE rate, and at
the shipped seed it reads 1.460 - red for want of power rather than for want of
a latent. At lead 1 the instrument has no power at all: 3.2% of TREATMENT draws
clear 1.5 against 0.0% of null draws. **ADR-114 (b)'s three verdict-bearing
calls at k = 1, 2, 3 are therefore not executable on THIS instrument**, and that
is a fact about the estimand rather than about the code: one instrument's null
is exact at one step, the other's power is absent there, and they are
complementary in the horizon rather than substitutes (ADR-119 (4)).

THE HORIZON IS LOAD BEARING AND WAS NOT RECORDED ANYWHERE
----------------------------------------------------------
At the shipped configuration (32 members, seed 4, the 25x25 grass plain with a
12 m/s east wind) the ratio reads 0.922, 1.410, 1.460, 1.488, 2.690, 6.517 at
horizons 1 to 6. It cleared the retired 1.5 at horizons 5 and 6 and at no other
horizon, and it clears the shipped 5.0 at horizon 6 ALONE, so changing one
integer in the call turns a green gate-path check red, and until this module
landed neither the horizon nor the member count appeared in the record the check
emits. The pass at horizon 6 is not itself an artefact of the zero-denominator
guard: at that seed the ablation SD is 0.803, not 0.

WHAT THIS DOES AND DOES NOT SETTLE
-----------------------------------
``ABLATION_SD_RATIO_BAR`` and ``horizon_h = 6`` are read from
:mod:`wildfire_nowcast.eval.selftest` and never restated here, so if either
moves every number this module prints moves with it. The instrument is not
vacuous: with the latent ON it separates from its own null from horizon 3
upward, reaching 77.2% and 98.0% of draws above the retired bar at horizons 3
and 4 and 100% at 5, against a null of 2.5% and 0.8%. What is established is
that the bar's false fire rate is a function of the horizon it is evaluated at,
that the horizon is 6, and that at 6 the retired bar's rate was 26.8%. The
degenerate denominator reaches the healthy arm too: 5 of the treatment's 400
draws at horizon 6 divide by the 1e-9 floor rather than by being wide, and are
scored as failures on both arms alike.

It also establishes that the two collapse instruments cannot both be relocated
to one step. At horizon 1 the ratio separates almost not at all: 3.2% of
draws WITH the latent on clear 1.5, against 0.0% with it off. An instrument
whose null is exact at one step and an instrument whose POWER is absent at one
step are not substitutes for one another.

THE ONE-STEP INDEX BAR, DERIVED HERE INDEPENDENTLY
---------------------------------------------------
:func:`derived_one_step_index` derives what ``independence_dispersion_index``
must read on the correlated ensemble that ``sim/selftest.py`` constructs: 900
cells with ``p_i ~ U(0.2, 0.8)``, 60 members, one lead step, and a single shared
logit shift ``z ~ N(0, 1.2^2)``. That construction is ALREADY one step, so the
multi-step defect never touched it; what is at stake is only the magnitude.

By the law of total variance, with ``q_i(z) = sigmoid(logit(p_i) + z)``,

    Var(area) = E_z[ sum_i q_i (1 - q_i) ]  +  Var_z( sum_i q_i )   = D + V
    denominator^2 = sum_i pbar_i (1 - pbar_i)                       = D + W
    W = sum_i Var_z(q_i),   index = sqrt((D + V) / (D + W))

``V`` scales like the square of the cell count because one scalar ``z`` moves
every cell together, while ``D`` and ``W`` scale linearly, so the index grows
like ``sqrt(N_cells)`` and any bar on it is a statement about the construction
as much as about the latent. Evaluated by Gauss-Hermite quadrature this gives
**13.726** on the realised ``p`` vector and **13.718** on the seed free
population, and the shipped function returns mean 13.700, sd 0.903, min 10.463
over 3000 replications of the construction. The assertion in the tree is
``> 2.0``, which is **12.99 sd below the derived mean**, was cleared by 3000 of
3000 replications, and would still be cleared if the shared latent's ``sigma_z``
fell from 1.2 to 0.1234, a 90% destruction of the effect the assertion exists to
witness. A bar that survives a tenfold weakening of its subject is a bar on
nothing.

**The derived value is 13.72 and the derived bar is 9.0**: five sd below the
mean, 1.6 sd below the smallest of 3000 replications, and first violated when
``sigma_z`` falls to 0.691, i.e. by a 42% weakening. That number is derived and
REPORTED. ``sim/selftest.py`` is not this package's to edit and nothing here
writes it.

THE PUBLISHED-CLAIM SWEEP THAT CAME WITH THIS, AND ITS OWN FALSE NEGATIVE
--------------------------------------------------------------------------
The class swept for is a tracked sentence asserting a dispersion, collapse or
ablation MAGNITUDE with no horizon recorded, over the 31 tracked modules of
``eval/`` and ``model/``. Four were repaired in the same commit as this file:
the 3.7-7.8x and 1.1-1.6x ablation separations at ``eval/baseline_run.py`` and
restated in ``eval/collapse_curve.py``, which are C6 ``band_area_dispersion_ratio``
and therefore the **1-3 h band pooled**; ``collapse_curve``'s
independent-by-construction control readings, which are at ONE lead step; and
the ~100x ``area_dispersion_ratio`` separation in ``eval/selftest.py``, which is
also one lead step and whose body asserts the weaker 50x.

**A number that cannot be split by lead has to say so, and one of them could
not.** ``band_area_dispersion_ratio`` had no ``_by_horizon`` sibling anywhere in
this tree while the G2 shape criterion and the G3 calibration criterion both
did, so the ruling that a 1-3 h statement is three verdict-bearing calls could
not be carried out for G3's dispersion half at all (ADR-120 (1)). **M22 built
the sibling** and it is exact: the per-lead sums recombine to the pooled
criterion to floating point at every level of pooling. **The magnitudes quoted
above are still the POOLED quantity** and stay quoted that way - re-labelling a
number measured on the pooled statistic with a decomposition that did not exist
when it was measured would be the reverse of the repair.

**THE SWEEP MISSED ONE OF ITS OWN TARGETS FIRST TIME AND THAT IS RECORDED
RATHER THAN TIDIED.** The horizon vocabulary it treats as sufficient included
the bare word "step", so a paragraph containing the phrase "shared PER-STEP
latent" was read as having recorded a horizon when it had recorded a model
property. The paragraph it silenced was the 3.7-7.8x claim, which is the
loudest instance in the class and the one already known by reading. A scan
whose keyword list is drawn from the vocabulary of the thing being scanned will
be silenced by that vocabulary; the word was removed and the corpus re-run.

WHAT THIS MODULE MAY NOT BE USED FOR
-------------------------------------
Nothing here is a result about the learned kernel or about G3's outcome. Both
arms are the untrained ``ContagionKernel`` at its initialisation, which is a
model of the ensemble MACHINERY and not of a fire. The scene is the same 25x25
toy plain the shipped check uses, imported rather than copied, so that the
numbers describe the instrument in the tree.

**A CLONE CANNOT OPEN THIS MODULE'S SWEEP ARTIFACTS**, and no path to one is
written here for that reason: they land wherever ``--out`` says and the run
directory is excluded from the public tree. Every number asserted above is
written out in full at the setting it was measured at, and all of it is
re-derivable by the one command at the top of this docstring, because nothing
here reads a stored value.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.sim.absent import EXIT_NOTHING_EXAMINED

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FAMILIES",
    "DEFAULT_SEEDS",
    "IndexBarDerivation",
    "RatioCell",
    "RatioSummary",
    "build_parser",
    "derived_one_step_index",
    "format_report",
    "index_bar_monte_carlo",
    "main",
    "null_tail_by_lead",
    "ratio_by_horizon",
    "ratio_sweep",
    "shipped_ratio_bar",
    "shipped_ratio_horizon",
    "summarise_ratio",
]

#: The arms swept. ``null_*`` are the two ways to make the shared latent
#: contribute nothing while the rest of the process is untouched; ``treatment``
#: is the shipped configuration. Both nulls are carried because they fail
#: differently: ``null_sigma_zero`` keeps the latent HEAD wired and forces its
#: scale to zero, so it also proves the head is not leaking through some path
#: other than sigma, while ``null_no_latent`` removes the head entirely and
#: pays for its independence with two different seeds.
DEFAULT_FAMILIES: Final[tuple[str, ...]] = (
    "null_sigma_zero",
    "null_no_latent",
    "treatment",
)

#: Replications per family. 400 puts about four draws in the 1% tail, which is
#: enough to say a rate is 28% and not enough to put three digits on a 99th
#: percentile; the CLI takes more.
DEFAULT_SEEDS: Final[int] = 400

#: The seed offset given to the SECOND arm of ``null_no_latent``. With no latent
#: head the two sampler modes consume the generator identically, so the same
#: seed makes the arms BITWISE identical and the ratio is exactly 1.0 with no
#: variance at all. That degenerate reading is a fact about the harness, not
#: about the instrument, and an offset is what buys the sampling distribution
#: the question is actually about.
_NULL_SEED_OFFSET: Final[int] = 100_000

#: The construction asserted on in ``sim/selftest.py``. Restated as data here
#: because it is another package's TEST BODY rather than an importable value,
#: and a derivation has to name the thing it derives.
INDEX_CONSTRUCTION: Final[dict[str, float]] = {
    "n_cells": 900.0,
    "n_members": 60.0,
    "p_low": 0.2,
    "p_high": 0.8,
    "sigma_z": 1.2,
}


def shipped_ratio_bar() -> float:
    """``ABLATION_SD_RATIO_BAR`` read from the self-test, never restated here.

    A module that compared against a locally written 1.5 would keep agreeing
    with itself after the shipped bar moved.
    """
    from wildfire_nowcast.eval.selftest import ABLATION_SD_RATIO_BAR  # noqa: PLC0415

    return float(ABLATION_SD_RATIO_BAR)


def shipped_ratio_horizon() -> int:
    """``ABLATION_SD_RATIO_HORIZON_H`` read from the self-test.

    The horizon is the variable this module exists to expose, so reading it is
    the whole point: if the check is ever moved to another lead time, the report
    below says so instead of continuing to describe horizon 6.
    """
    from wildfire_nowcast.eval.selftest import ABLATION_SD_RATIO_HORIZON_H  # noqa: PLC0415

    return int(ABLATION_SD_RATIO_HORIZON_H)


@dataclass(frozen=True)
class RatioCell:
    """One replication: one family, one seed, one lead step.

    ``degenerate`` records the case the shipped check hides behind a
    ``max(sd, 1e-9)`` floor: an ablation arm whose members all finish at the
    same area has NO measurable spread, and dividing by the floor turns an
    absent denominator into a ratio of order 1e9 that reads as a spectacular
    pass.
    """

    family: str
    horizon_h: int
    seed: int
    mean_latent: float
    mean_independent: float
    sd_latent: float
    sd_independent: float
    ratio: float
    degenerate: bool
    fires: bool


@dataclass(frozen=True)
class RatioSummary:
    """One family at one lead step, over every replication."""

    family: str
    horizon_h: int
    n: int
    ratio_mean: float
    ratio_median: float
    ratio_p95: float
    ratio_p99: float
    n_fires: int
    fire_rate: float
    n_degenerate: int
    n_measurable: int
    n_fires_measurable: int
    fire_rate_measurable: float
    ratio_p99_measurable: float
    mean_sd_latent: float
    mean_sd_independent: float
    mean_area_latent: float
    mean_area_independent: float
    bar: float


def _toy_scene_and_kernel(family: str) -> tuple[Any, Any, Any, Any]:
    """``(x0, static, weather, model)`` for one family.

    The scene is IMPORTED from the self-test rather than rebuilt, so the numbers
    describe the shipped check's own scene. A local copy would make every
    reading a statement about the copy, which is the duplication C0 forbids for
    exactly this reason.
    """
    from wildfire_nowcast.eval.selftest import _toy_scene  # noqa: PLC0415
    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig  # noqa: PLC0415
    from wildfire_nowcast.model.latent import LatentConfig  # noqa: PLC0415

    x0, static, weather = _toy_scene(wind_u=12.0)
    if family == "treatment":
        model = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=3))
    elif family == "null_sigma_zero":
        # `init_sigma` is validated strictly positive, so the scale is driven to
        # zero through the parameter itself: sigma = max_sigma * sigmoid(logit),
        # and sigmoid(-50) is 2e-22. Zeroing the CONFIG would only reach 1e-4 of
        # max_sigma, which is small but is not zero and would leave a reader
        # unable to tell a residual from a leak.
        import torch  # noqa: PLC0415

        model = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=3))
        head = model.latent
        if head is None:  # pragma: no cover - constructed with a latent one line above
            raise RuntimeError("the null arm needs a latent head to drive to zero")
        with torch.no_grad():
            head.sigma_logit.fill_(-50.0)
    elif family == "null_no_latent":
        model = ContagionKernel(KernelConfig(), latent_config=None)
    else:
        raise ValueError(f"family must be one of {DEFAULT_FAMILIES}, got {family!r}")
    return x0, static, weather, model


def ratio_by_horizon(
    family: str,
    seed: int,
    *,
    n_members: int,
    horizon_h: int,
    bar: float,
) -> list[RatioCell]:
    """The shipped estimand at EVERY lead step of one pair of rollouts.

    The shipped check reads ``samples[:, -1]``, i.e. the cumulative burned area
    at the final lead. Reading every lead from the same pair costs nothing and
    is what makes the horizon dependence visible at all.
    """
    if n_members < 2 or horizon_h < 1:
        raise ValueError(f"n_members={n_members}, horizon_h={horizon_h}")
    x0, static, weather, model = _toy_scene_and_kernel(family)
    seed_b = seed + (_NULL_SEED_OFFSET if family == "null_no_latent" else 0)
    a = model.with_sampler("latent").predict(x0, static, weather, n_members, horizon_h, seed)
    b = model.with_sampler("independent").predict(x0, static, weather, n_members, horizon_h, seed_b)

    out: list[RatioCell] = []
    for k in range(horizon_h):
        na = ((a[:, k] > 0) & (x0[None] == 0)).sum(axis=(1, 2)).astype(float)
        nb = ((b[:, k] > 0) & (x0[None] == 0)).sum(axis=(1, 2)).astype(float)
        sd_a, sd_b = float(na.std(ddof=1)), float(nb.std(ddof=1))
        ratio = sd_a / max(sd_b, 1e-9)
        out.append(
            RatioCell(
                family=family,
                horizon_h=k + 1,
                seed=seed,
                mean_latent=float(na.mean()),
                mean_independent=float(nb.mean()),
                sd_latent=sd_a,
                sd_independent=sd_b,
                ratio=ratio,
                degenerate=sd_b == 0.0,
                # [M22] `fires` is the SHIPPED VERDICT, which since M21 refuses a
                # zero denominator rather than dividing by the 1e-9 floor. It read
                # `ratio > bar` alone until now, so this module scored a rule the
                # gate-path check no longer applies and its tail counted the
                # degenerate draws as passes. That is the one defect a bar's own
                # control may not have: measuring a verdict nobody takes.
                fires=(ratio > bar) and sd_b > 0.0,
            )
        )
    return out


def ratio_sweep(
    *,
    families: Sequence[str] = DEFAULT_FAMILIES,
    n_seeds: int = DEFAULT_SEEDS,
    n_members: int = 32,
    horizon_h: int | None = None,
    bar: float | None = None,
) -> list[RatioCell]:
    """Every family at every lead step over ``n_seeds`` replications.

    ``bar`` defaults to the shipped one. It is an argument because the horizon
    dependence of the tail was DISCOVERED at a bar that no longer ships, and a
    finding you can only reproduce at the value you have since replaced is a
    finding you have deleted.
    """
    horizon = shipped_ratio_horizon() if horizon_h is None else int(horizon_h)
    bar = shipped_ratio_bar() if bar is None else float(bar)
    cells: list[RatioCell] = []
    for family in families:
        logger.info("ratio sweep: family=%s seeds=%d horizon=%d", family, n_seeds, horizon)
        for seed in range(int(n_seeds)):
            cells.extend(
                ratio_by_horizon(family, seed, n_members=n_members, horizon_h=horizon, bar=bar)
            )
    return cells


def summarise_ratio(cells: Sequence[RatioCell], bar: float | None = None) -> list[RatioSummary]:
    """Collapse the replications to one row per (family, lead step).

    The MEDIAN is reported beside the mean because the mean of this quantity is
    not a summary of it: one degenerate denominator moves the mean of 400 draws
    into the millions while the median does not move at all. Reporting only the
    mean would have hidden the very case that matters.

    ``bar`` is recorded, not re-applied: ``fires`` was decided when the cell was
    made. Passing a bar the cells were not swept at would produce a row whose
    ``bar`` and whose ``fire_rate`` describe two different thresholds.
    """
    bar = shipped_ratio_bar() if bar is None else float(bar)
    keys = sorted({(c.family, c.horizon_h) for c in cells})
    rows: list[RatioSummary] = []
    for family, horizon in keys:
        group = [c for c in cells if c.family == family and c.horizon_h == horizon]
        ratios = np.array([c.ratio for c in group], dtype=np.float64)
        # The degenerate draws are separated rather than winsorised. A ratio
        # built on a zero denominator is not a large value of the quantity, it
        # is the absence of one, and pooling the two makes a p99 that describes
        # the 1e-9 floor instead of the process.
        measurable = np.array([c.ratio for c in group if not c.degenerate], dtype=np.float64)
        fires_measurable = [c.fires for c in group if not c.degenerate]
        rows.append(
            RatioSummary(
                family=family,
                horizon_h=horizon,
                n=len(group),
                ratio_mean=float(ratios.mean()),
                ratio_median=float(np.median(ratios)),
                ratio_p95=float(np.percentile(ratios, 95)),
                ratio_p99=float(np.percentile(ratios, 99)),
                n_fires=int(sum(c.fires for c in group)),
                fire_rate=float(np.mean([c.fires for c in group])),
                n_degenerate=int(sum(c.degenerate for c in group)),
                n_measurable=int(measurable.size),
                n_fires_measurable=int(sum(fires_measurable)),
                fire_rate_measurable=(
                    float(np.mean(fires_measurable)) if fires_measurable else float("nan")
                ),
                ratio_p99_measurable=(
                    float(np.percentile(measurable, 99)) if measurable.size else float("nan")
                ),
                mean_sd_latent=float(np.mean([c.sd_latent for c in group])),
                mean_sd_independent=float(np.mean([c.sd_independent for c in group])),
                mean_area_latent=float(np.mean([c.mean_latent for c in group])),
                mean_area_independent=float(np.mean([c.mean_independent for c in group])),
                bar=bar,
            )
        )
    return rows


@lru_cache(maxsize=8)
def _null_cells(
    family: str, horizon_h: int, n_seeds: int, n_members: int
) -> tuple[tuple[int, float, bool], ...]:
    """``(lead, ratio, degenerate)`` for one null family. Cached by argument.

    Cached because the tail is read at more than one bar per invocation and the
    sweep is the expensive half: the RATIOS do not depend on the bar, only the
    counting does. Re-running the rollouts per bar would also make two readings
    of the same null cost twice as much as they are worth.
    """
    out: list[tuple[int, float, bool]] = []
    for seed in range(int(n_seeds)):
        for cell in ratio_by_horizon(
            family, seed, n_members=int(n_members), horizon_h=int(horizon_h), bar=float("inf")
        ):
            out.append((cell.horizon_h, cell.ratio, cell.degenerate))
    return tuple(out)


def null_tail_by_lead(
    *,
    bar: float,
    horizon_h: int,
    n_seeds: int,
    n_members: int = 32,
    family: str = "null_no_latent",
) -> dict[int, float]:
    """Fraction of NO-LATENT draws that would PASS the shipped check, per lead.

    This is the number a collapse verdict has to be read against, and until M22
    no verdict carried it. It applies the shipped rule exactly - a draw passes
    only if it clears the bar AND its denominator is measurable - so the rate is
    commensurable with the verdict it is published beside rather than merely
    similar to it.

    ``null_no_latent`` is the default because it is the more adversarial of the
    two nulls: it removes the latent head entirely and pays for the arms'
    independence with two different seeds, and its tail at lead 6 is the heavier
    of the two at every bar measured.
    """
    cells = _null_cells(family, int(horizon_h), int(n_seeds), int(n_members))
    rates: dict[int, float] = {}
    for lead in range(1, int(horizon_h) + 1):
        rows = [(r, d) for h, r, d in cells if h == lead]
        passes = [(r > bar) and not d for r, d in rows]
        rates[lead] = float(np.mean(passes)) if passes else float("nan")
    return rates


@dataclass(frozen=True)
class IndexBarDerivation:
    """The closed-form one-step index for a shared-logit-shift ensemble."""

    n_cells: int
    sigma_z: float
    conditional_variance: float
    latent_variance: float
    marginal_excess: float
    index: float


def derived_one_step_index(
    p: np.ndarray,
    sigma_z: float = 1.2,
    n_quadrature: int = 201,
) -> IndexBarDerivation:
    """``independence_dispersion_index`` in the many-member limit, in closed form.

    Cells are conditionally independent GIVEN the shared shift ``z``, so the law
    of total variance splits the numerator exactly and the denominator is the
    marginal Bernoulli variance. Gauss-Hermite quadrature over ``z`` makes both
    integrals deterministic, which is what separates a DERIVATION from another
    Monte Carlo run with the same sampling noise as the thing it is checking.
    """
    if sigma_z <= 0 or n_quadrature < 5:
        raise ValueError(f"sigma_z={sigma_z}, n_quadrature={n_quadrature}")
    nodes, weights = hermegauss(int(n_quadrature))
    weights = weights / weights.sum()
    z = nodes * float(sigma_z)
    logit = np.log(np.asarray(p, dtype=np.float64) / (1.0 - np.asarray(p, dtype=np.float64)))
    q = 1.0 / (1.0 + np.exp(-(logit[None, :] + z[:, None])))
    pbar = (weights[:, None] * q).sum(axis=0)
    conditional = float((weights[:, None] * (q * (1.0 - q))).sum())
    totals = q.sum(axis=1)
    latent = float((weights * totals**2).sum() - (weights * totals).sum() ** 2)
    excess = float((weights[:, None] * q**2).sum(axis=0).sum() - (pbar**2).sum())
    return IndexBarDerivation(
        n_cells=int(np.asarray(p).size),
        sigma_z=float(sigma_z),
        conditional_variance=conditional,
        latent_variance=latent,
        marginal_excess=excess,
        index=float(np.sqrt((conditional + latent) / (conditional + excess))),
    )


def index_bar_monte_carlo(n_replications: int = 3000) -> dict[str, float]:
    """The same construction through the SHIPPED index, replication by replication.

    The derivation above is a limit; this is what the function in the tree
    actually returns on the construction the assertion is made about, including
    its ``ddof=0`` numerator and its use of the EMPIRICAL marginals in the
    denominator. Agreement between the two is the check that the derivation
    describes the shipped estimand and not an idealisation of it.
    """
    from wildfire_nowcast.sim.ensemble import independence_dispersion_index  # noqa: PLC0415

    n_cells = int(INDEX_CONSTRUCTION["n_cells"])
    n_members = int(INDEX_CONSTRUCTION["n_members"])
    side = int(round(float(np.sqrt(n_cells))))
    values = np.empty(int(n_replications), dtype=np.float64)
    for i in range(int(n_replications)):
        rng = np.random.default_rng(i)
        p = rng.uniform(INDEX_CONSTRUCTION["p_low"], INDEX_CONSTRUCTION["p_high"], size=n_cells)
        rng.random((n_members, n_cells))  # the independent arm, drawn and discarded
        z = rng.normal(0.0, INDEX_CONSTRUCTION["sigma_z"], size=(n_members, 1))
        q = 1.0 / (1.0 + np.exp(-(np.log(p / (1.0 - p))[None, :] + z)))
        corr = (rng.random((n_members, n_cells)) < q).astype(np.uint8)
        values[i] = independence_dispersion_index(corr.reshape(n_members, 1, side, side))
    return {
        "n_replications": float(n_replications),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def format_report(rows: Sequence[RatioSummary], index: dict[str, Any]) -> list[str]:
    """The one-page reading. Every verdict-bearing number carries its horizon."""
    bar = rows[0].bar if rows else shipped_ratio_bar()
    horizon = shipped_ratio_horizon()
    lines = [
        "M21  the two collapse bars, per horizon",
        f"     eval SD-ratio bar {bar:g} at horizon_h = {horizon} (read from eval/selftest.py)",
        "",
        "  median and p95 over ALL draws; p99 and fires over the MEASURABLE ones; deg counts",
        "  the draws whose ablation arm had NO spread, where the quotient has no value at all",
        "",
        "  family            h |  median     p95     p99 |  fires   rate | sd_lat sd_abl | deg",
    ]
    for row in rows:
        lines.append(
            f"  {row.family:<16} {row.horizon_h} | {row.ratio_median:7.3f} {row.ratio_p95:7.3f} "
            f"{row.ratio_p99_measurable:7.3f} | {row.n_fires_measurable:4d}/{row.n_measurable:<4d} "
            f"{100 * row.fire_rate_measurable:5.1f}% | "
            f"{row.mean_sd_latent:6.3f} {row.mean_sd_independent:6.3f} | {row.n_degenerate:3d}"
        )
    lines.append("")
    lines.append("  one-step index bar, derived (sim/selftest.py's correlated construction)")
    lines.append(
        f"     derived {index['derived_realised']:.4f} (realised p) / "
        f"{index['derived_population']:.4f} (population)"
    )
    mc = index["monte_carlo"]
    lines.append(
        f"     shipped fn over {int(mc['n_replications'])} replications: mean {mc['mean']:.4f} "
        f"sd {mc['sd']:.4f} min {mc['min']:.4f}"
    )
    lines.append(
        f"     asserted in the tree: > {index['asserted_bar']:g}, which is "
        f"{index['asserted_bar_sd_below_mean']:.2f} sd below the derived mean"
    )
    lines.append(f"     derived bar, reported and NOT written anywhere: > {index['derived_bar']:g}")
    return lines


def _index_section() -> dict[str, Any]:
    """The ITEM 2 derivation, assembled once so the report and the artifact agree."""
    n_cells = int(INDEX_CONSTRUCTION["n_cells"])
    sigma_z = float(INDEX_CONSTRUCTION["sigma_z"])
    realised = np.random.default_rng(0).uniform(
        INDEX_CONSTRUCTION["p_low"], INDEX_CONSTRUCTION["p_high"], size=n_cells
    )
    derived_realised = derived_one_step_index(realised, sigma_z)
    grid = np.linspace(INDEX_CONSTRUCTION["p_low"], INDEX_CONSTRUCTION["p_high"], 20001)[1:-1]
    fine = derived_one_step_index(grid, sigma_z)
    scale = n_cells / grid.size
    population = float(
        np.sqrt(
            (fine.conditional_variance * scale + fine.latent_variance * scale**2)
            / (fine.conditional_variance * scale + fine.marginal_excess * scale)
        )
    )
    mc = index_bar_monte_carlo()
    # The derived bar: five sd below the derived mean, rounded DOWN to a whole
    # number so it is quotable, and stated with the weakening it first refuses.
    derived_bar = float(np.floor(derived_realised.index - 5.0 * mc["sd"]))
    weakened = [
        {"sigma_z": s, "index": derived_one_step_index(realised, s).index}
        for s in (1.2, 0.9, 0.691, 0.6, 0.4, 0.2, 0.1234, 0.1)
    ]
    return {
        "construction": dict(INDEX_CONSTRUCTION),
        "derived_realised": derived_realised.index,
        "derived_population": population,
        "derived_terms": asdict(derived_realised),
        "monte_carlo": mc,
        "asserted_bar": 2.0,
        "asserted_bar_sd_below_mean": (derived_realised.index - 2.0) / mc["sd"],
        "derived_bar": derived_bar,
        "derived_bar_sd_below_mean": (derived_realised.index - derived_bar) / mc["sd"],
        "index_under_a_weakened_latent": weakened,
    }


def build_parser() -> argparse.ArgumentParser:
    """Parser as its own function so a test can assert PARSED BEHAVIOUR."""
    p = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.eval.collapse_bars",
        description=(
            "Per-horizon null and power of the ablation SD-ratio check, and the "
            "closed-form one-step bar for the dispersion index. Reads both bars; "
            "writes neither."
        ),
    )
    p.add_argument("--out", default=None, help="write the full sweep as JSON to this path")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--members", type=int, default=32)
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="lead steps to roll out; the default is the horizon the shipped check uses",
    )
    add_logging_arguments(p)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep and the derivation, print the report, optionally write it."""
    args = build_parser().parse_args(argv)
    configure_from_args(args, default_verbosity=1)

    families = tuple(v for v in str(args.families).split(",") if v)
    cells = ratio_sweep(
        families=families,
        n_seeds=int(args.seeds),
        n_members=int(args.members),
        horizon_h=args.horizon,
    )
    rows = summarise_ratio(cells)
    index = _index_section()
    for line in format_report(rows, index):
        print(line)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": "M21",
            "question": (
                "is the ablation SD-ratio check horizon-sound, and what is the one-step "
                "bar for the dispersion index"
            ),
            "ablation_sd_ratio_bar_read_from_eval": shipped_ratio_bar(),
            "ablation_sd_ratio_horizon_read_from_eval": shipped_ratio_horizon(),
            "n_members": int(args.members),
            "n_seeds": int(args.seeds),
            "families": list(families),
            "n_cells": len(cells),
            "cells": [asdict(c) for c in cells],
            "summary": [asdict(r) for r in rows],
            "one_step_index_bar": index,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"written: {path}")
    return 0 if cells else EXIT_NOTHING_EXAMINED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
