"""Per-horizon ensemble-collapse verdicts, scored on the ONE-STEP increment.

    python -m wildfire_nowcast.sim.collapse --tensor outputs/synthetic_fire/tensor.zarr \
        --out reports/figures/collapse_per_horizon.json --t0 20 --horizon 3 --members 24

ADR-114 adopts this in four clauses and all four are here, because three of
them is half a repair.

(a) THE VERDICT-BEARING CALL IS AT ONE LEAD STEP.
    :func:`~wildfire_nowcast.sim.ensemble.independence_dispersion_index` divides
    the observed spread of burned area by ``sqrt(sum p_i (1 - p_i))``, which is
    the variance of a sum of indicators **only while those indicators are
    conditionally independent**. That holds for the cells a member adds in ONE
    step from a state every member shares. It fails at two steps: the step-2
    candidate set is a function of the step-1 draw, so a member that ignited
    more cells early has more front to ignite from later, and contagion is
    itself a correlating mechanism that needs no latent. The index then measures
    the shared innovation AND the dynamics and attributes all of it to the
    innovation. Measured with the shipped instrument, a no-latent field reads
    1.0048 at one step and 1.25 / 1.47-1.52 at two and three cumulative steps,
    so a fixed threshold on the cumulative index is partly a threshold on
    horizon. On the one-step increment ``1.0`` is exact by algebra at every
    lead, and ``COLLAPSE_INDEX_THRESHOLD`` keeps meaning one thing.

(b) A 1-3 H STATEMENT IS THREE VERDICTS, NOT ONE.
    :func:`per_horizon_collapse` calls ``predict()`` with ``horizon_h=1`` at
    ``k = 1, 2, 3``, re-conditioning every member on the same state at the start
    of hour ``k``. That is inside C5 and touches no model internal. The
    cumulative multi-step index is KEPT, as
    ``cumulative_index_description``, and is never a verdict.

(c) THE INSTRUMENT'S OWN CONTROLS RUN IN THE SAME INVOCATION AS THE VERDICT
    and their readings are published beside it. No control reading, no verdict.
    See :func:`instrument_controls` for what they are and, more importantly,
    for why they are not the thing ADR-114 (3) killed.

(d) EVERY VERDICT PUBLISHES ITS HORIZON. :class:`CollapseVerdict` carries
    ``lead_h``, ``estimand``, ``conditioning`` and both control readings in the
    same record as the verdict, so the one variable that decides the verdict
    cannot be dropped from the record. That omission is why the defect survived
    for weeks.

WHAT IS COMPROMISED, SAID HERE RATHER THAN DISCOVERED LATER
-----------------------------------------------------------
Re-conditioning every member on a shared state at hour ``k`` is the price of an
exact null. The ``k=2`` and ``k=3`` verdicts are therefore NOT statements about
the ensemble a forecaster would be holding at hour 2 or 3, whose members have
diverged; they are statements about the one-step innovation the model injects at
that hour, from the state at the start of it. The honest wording of a three-hour
claim is "the ablation's one-step increment is indistinguishable from
independent-per-pixel noise at each of hours 1, 2 and 3, conditioned on the state
at the start of that hour", and the diverged-ensemble quantity is the demoted
cumulative description. Both travel together or neither is quotable.

The conditioning state is a choice and it is recorded in every verdict.
``truth`` uses the observed state, which is available from C1 and is the least
arbitrary; ``rollout`` uses one shared member trajectory from the model itself,
for a scene with no truth. Measured on the synthetic fire the two agree on all
six verdicts, so on that scene the choice is not outcome-determinative, which is
a fact about that scene and not a general one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.seeds import stable_seed
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.sim.absent import (
    EXIT_NOTHING_EXAMINED,
    AbsentMeasurementError,
    refuse_if_empty,
)
from wildfire_nowcast.sim.c5 import C5Inputs, c5_inputs
from wildfire_nowcast.sim.ensemble import (
    COLLAPSE_INDEX_THRESHOLD,
    COLLAPSED,
    CUMULATIVE_FROM_T0,
    NOT_A_VERDICT,
    NOT_COLLAPSED,
    ONE_STEP_INCREMENT,
    independence_dispersion_index,
)

__all__ = [
    "ONE_STEP_INCREMENT",
    "CUMULATIVE_FROM_T0",
    "COLLAPSED",
    "NOT_COLLAPSED",
    "NOT_A_VERDICT",
    "CONTROL_REPLICATES",
    "one_step_increment",
    "increment_dispersion_index",
    "InstrumentControls",
    "instrument_controls",
    "CollapseVerdict",
    "one_step_collapse_verdict",
    "PerHorizonCollapse",
    "per_horizon_collapse",
    "analytic_shared_latent_index",
    "main",
]

#: Replicates of each control ensemble. A sample size, not a bar: it buys the
#: control a small standard error so the gate lands on the estimator's BIAS
#: rather than on one draw's noise. The per-replicate spread is published too.
CONTROL_REPLICATES = 32


# -- the estimand ----------------------------------------------------------


def one_step_increment(samples: np.ndarray, given: np.ndarray) -> np.ndarray:
    """Cells each member ADDS in one step, given a state every member shares.

    ``samples`` is a C5 return with exactly one lead step, ``uint8[M, 1, H, W]``.
    ``given`` is the ``uint8[H, W]`` state passed in as ``x0``. Returns
    ``bool[M, H, W]``.

    The masking is explicit rather than implied. With a shared ``given`` the
    already-burned cells have a marginal of exactly 1 and contribute nothing to
    either side of the ratio, so on a state-preserving model the increment and
    the full field give the identical index. The mask is here for the model that
    is NOT state-preserving: a cell that un-burns has a marginal strictly
    between 0 and 1 and would enter the denominator as though it were a fresh
    ignition. Writing the estimand out means it is checkable rather than
    conditional on an absorbing-state assumption this module cannot enforce.
    """
    arr = np.asarray(samples)
    refuse_if_empty(
        "one_step_increment",
        {
            "members": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            "lead_steps": int(arr.shape[1]) if arr.ndim >= 2 else 0,
        },
        because="the one-step increment is the estimand a collapse verdict is taken from.",
    )
    if arr.shape[1] != 1:
        raise ValueError(
            f"one_step_increment needs exactly one lead step, got {arr.shape[1]}. "
            "A multi-step array is the cumulative estimand, which is a description "
            f"({CUMULATIVE_FROM_T0}) and never a verdict."
        )
    prior = np.asarray(given) > 0
    if prior.shape != arr.shape[2:]:
        raise ValueError(f"given is {prior.shape}, samples are {arr.shape[2:]}")
    return (arr[:, 0] > 0) & ~prior[None, :, :]


def increment_dispersion_index(increment: np.ndarray) -> float:
    """The SHIPPED index, measured on a one-step increment.

    This deliberately re-wraps the increment as a one-lead-step sample array and
    calls :func:`~wildfire_nowcast.sim.ensemble.independence_dispersion_index`
    rather than re-deriving the ratio. A second implementation of an instrument
    is a second set of choices about ``ddof``, about the empirical ``p_hat`` and
    about the zero-spread branch, and the two would agree until the day they did
    not. The M19 curve driver imports the same function, so a number
    measured here and a number measured there come from one estimator.
    """
    inc = np.asarray(increment, dtype=bool)
    if inc.ndim < 2:
        raise ValueError(f"an increment needs a member axis and cells, got shape {inc.shape}")
    # The index sums over cells and is invariant to how they are laid out, so a
    # flat control field and a (H, W) increment reach the same estimator.
    flat = inc.reshape(inc.shape[0], 1, -1, 1).astype(np.uint8)
    return independence_dispersion_index(flat)


# -- the instrument's own controls ------------------------------------------


@dataclass(frozen=True)
class InstrumentControls:
    """Both control readings for one verdict, and whether a verdict may be taken."""

    independent_index: float
    independent_sd: float
    comonotone_index: float
    comonotone_sd: float
    n_replicates: int
    n_members: int
    n_uncertain_cells: int
    threshold: float

    @property
    def independent_ok(self) -> bool:
        """A by-construction-independent field must read inside the bar, both ways."""
        lo, hi = 1.0 / self.threshold, self.threshold
        return bool(lo < self.independent_index < hi)

    @property
    def comonotone_ok(self) -> bool:
        """A maximally dependent field with the same marginals must clear the bar."""
        return bool(self.comonotone_index >= self.threshold)

    @property
    def ok(self) -> bool:
        return self.independent_ok and self.comonotone_ok

    @property
    def reason(self) -> str:
        """Why no verdict may be taken, or the empty string."""
        parts: list[str] = []
        if not self.independent_ok:
            parts.append(
                f"the independent-by-construction control reads "
                f"{self.independent_index:.4f}, outside "
                f"({1.0 / self.threshold:.4f}, {self.threshold:.4f}) around the null of 1.0, "
                "so the estimator is biased on this scene"
            )
        if not self.comonotone_ok:
            parts.append(
                f"the maximally-dependent control reads {self.comonotone_index:.4f}, "
                f"under the {self.threshold:g} bar, so this scene "
                f"({self.n_uncertain_cells} uncertain cells, {self.n_members} members) "
                "cannot separate dependence from independence at all"
            )
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "independent_index": self.independent_index,
            "independent_sd": self.independent_sd,
            "independent_ok": self.independent_ok,
            "comonotone_index": self.comonotone_index,
            "comonotone_sd": self.comonotone_sd,
            "comonotone_ok": self.comonotone_ok,
            "n_replicates": self.n_replicates,
            "n_members": self.n_members,
            "n_uncertain_cells": self.n_uncertain_cells,
            "threshold": self.threshold,
            "ok": self.ok,
            "reason": self.reason,
        }


def instrument_controls(
    marginals: np.ndarray,
    n_members: int,
    *,
    seed: int,
    n_replicates: int = CONTROL_REPLICATES,
    threshold: float = COLLAPSE_INDEX_THRESHOLD,
) -> InstrumentControls:
    """Two controls for the estimator, built from the ensemble's own marginals.

    ``marginals`` is the per-cell ``p_i`` of the increment under test.

    POSITIVE CONTROL, independent by construction. Redraw the same marginals
    with every cell independent. The index must read ``1.0``; the M19 sweep
    measured the shipped estimator at ``0.9936 +/- 0.0429`` over 384 members on
    exactly this kind of field. This certifies the ESTIMATOR, at the member
    count and the marginal distribution of the ensemble actually being judged,
    which a canned 30x30 control cannot do.

    NEGATIVE CONTROL, maximally dependent with identical marginals. One uniform
    per member, shared by every cell: cell ``i`` burns iff ``u_m < p_i``. That
    is the comonotone extreme of the dependence family, so it must land above
    the bar. It fails exactly when the scene has too few uncertain cells for any
    amount of dependence to be visible, which is a power check the positive
    control cannot perform, and which fires on precisely the scenes where a
    "collapsed" reading would mean nothing.

    NEITHER CONTROL USES A NEW NUMBER. Both are gated on
    ``COLLAPSE_INDEX_THRESHOLD``, the positive one on the multiplicative band
    ``(1/T, T)`` around the null, the negative one on ``T`` itself.

    WHY THIS IS NOT THE PROPOSAL ADR-114 (3) KILLED, and the difference is the
    whole point. That proposal set the THRESHOLD to the value measured from a
    no-latent run, which is the ablation, so the ablation would have matched
    itself forever. These controls take only the MARGINALS, a first moment, and
    then generate the dependence structure from an assumption. The quantity
    under test is the dispersion, and the dispersion of each control comes from
    the construction, never from the ensemble. A collapsed ablation and its
    positive control both reading 1.0 is the RESULT, not a tautology: they are
    two different draws and the control could read 1.3 while the ablation reads
    1.0.

    WHAT THESE CONTROLS CANNOT DO. They certify the estimator, not the estimand.
    A synthetic field has no dynamics, so the positive control reads 1.0 at any
    horizon and would have read 1.0 during the whole period the cumulative index
    was drifting to 1.52. The estimand is fixed by clause (a), not by clause (c),
    and no control here would have caught that defect on its own.
    """
    p = np.asarray(marginals, dtype=np.float64)
    refuse_if_empty(
        "instrument_controls",
        {"cells": int(p.size), "members": int(n_members)},
        because="a control that examined nothing certifies nothing.",
    )
    rng = np.random.default_rng(seed)
    indep: list[float] = []
    comon: list[float] = []
    for _ in range(int(n_replicates)):
        draws = rng.random((n_members, *p.shape)) < p[None, ...]
        indep.append(increment_dispersion_index(draws))
        u = rng.random((n_members, *(1 for _ in p.shape)))
        comon.append(increment_dispersion_index(u < p[None, ...]))
    ind = np.asarray(indep, dtype=np.float64)
    com = np.asarray(comon, dtype=np.float64)
    return InstrumentControls(
        independent_index=float(np.nanmean(ind)),
        independent_sd=float(np.nanstd(ind, ddof=1)) if ind.size > 1 else 0.0,
        comonotone_index=float(np.nanmean(com)),
        comonotone_sd=float(np.nanstd(com, ddof=1)) if com.size > 1 else 0.0,
        n_replicates=int(n_replicates),
        n_members=int(n_members),
        n_uncertain_cells=int(((p > 0.0) & (p < 1.0)).sum()),
        threshold=float(threshold),
    )


# -- the verdict -----------------------------------------------------------


@dataclass(frozen=True)
class CollapseVerdict:
    """One collapse verdict and everything needed to read it, including WHEN.

    ``lead_h`` is the hour the verdict is about, 1-based in lead hours. It is a
    field of the verdict rather than a field of the run because the run may
    carry three of these, and because a verdict whose horizon lives somewhere
    else is a verdict whose deciding variable can be dropped.
    """

    lead_h: int
    estimand: str
    conditioning: str
    index: float
    threshold: float
    n_members: int
    n_new_cells_mean: float
    controls: InstrumentControls

    @property
    def verdict(self) -> str:
        """``collapsed`` / ``not_collapsed`` / ``not_a_verdict: <why>``."""
        if not self.controls.ok:
            return f"{NOT_A_VERDICT}: {self.controls.reason}"
        if self.estimand != ONE_STEP_INCREMENT:
            return (
                f"{NOT_A_VERDICT}: the estimand is {self.estimand}, whose null is not 1.0 "
                "and moves with horizon"
            )
        if not np.isfinite(self.index):
            return f"{NOT_A_VERDICT}: the index is {self.index}, so nothing was measured"
        return COLLAPSED if self.index < self.threshold else NOT_COLLAPSED

    @property
    def is_a_verdict(self) -> bool:
        return not self.verdict.startswith(NOT_A_VERDICT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lead_h": self.lead_h,
            "estimand": self.estimand,
            "conditioning": self.conditioning,
            "independence_dispersion_index": self.index,
            "threshold": self.threshold,
            "n_members": self.n_members,
            "n_new_cells_mean": self.n_new_cells_mean,
            "verdict": self.verdict,
            "is_a_verdict": self.is_a_verdict,
            "controls": self.controls.to_dict(),
        }


def one_step_collapse_verdict(
    samples: np.ndarray,
    given: np.ndarray,
    *,
    lead_h: int,
    conditioning: str,
    seed: int,
    n_replicates: int = CONTROL_REPLICATES,
) -> CollapseVerdict:
    """Score one ``horizon_h=1`` C5 return, with its controls, at ``lead_h``."""
    inc = one_step_increment(samples, given)
    p = inc.mean(axis=0).astype(np.float64)
    return CollapseVerdict(
        lead_h=int(lead_h),
        estimand=ONE_STEP_INCREMENT,
        conditioning=conditioning,
        index=increment_dispersion_index(inc),
        threshold=float(COLLAPSE_INDEX_THRESHOLD),
        n_members=int(inc.shape[0]),
        n_new_cells_mean=float(inc.sum(axis=(1, 2)).mean()),
        controls=instrument_controls(p, int(inc.shape[0]), seed=seed, n_replicates=n_replicates),
    )


@dataclass(frozen=True)
class PerHorizonCollapse:
    """Three verdicts, the demoted cumulative description, and the provenance."""

    verdicts: tuple[CollapseVerdict, ...]
    cumulative_index_description: float
    cumulative_estimand: str = CUMULATIVE_FROM_T0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[int, str]:
        return {v.lead_h: v.verdict for v in self.verdicts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_horizon_verdicts": [v.to_dict() for v in self.verdicts],
            "cumulative_index_description": self.cumulative_index_description,
            "cumulative_estimand": self.cumulative_estimand,
            "cumulative_is_a_verdict": False,
            "cumulative_note": (
                "DESCRIPTION ONLY. The cumulative index has no exact null: a no-latent "
                "field reads 1.0048 at h=1 and 1.25 / 1.47-1.52 at h=2 / h=3 with the "
                "shared latent switched off, so a fixed threshold on it is partly a "
                "threshold on horizon. Never a verdict."
            ),
            "threshold": float(COLLAPSE_INDEX_THRESHOLD),
            "meta": self.meta,
        }


def per_horizon_collapse(
    predict: Any,
    inp: C5Inputs,
    *,
    n_members: int = 24,
    seed: int = 0,
    conditioning: str = "truth",
    n_replicates: int = CONTROL_REPLICATES,
    cumulative_samples: np.ndarray | None = None,
) -> PerHorizonCollapse:
    """One verdict per lead hour, each from its own ``horizon_h=1`` C5 call.

    ``conditioning='truth'`` shares the observed state at the start of each hour.
    ``conditioning='rollout'`` shares one model trajectory instead, for a scene
    with no truth; the trajectory is a single-member ``predict()`` call and is
    therefore still a state every re-conditioned member starts from.

    The cumulative description comes from ONE ``horizon_h=inp.horizon_h`` call,
    which is the quantity a forecaster actually holds. ``cumulative_samples``
    lets a caller that already made that call hand its array over instead.
    """
    if conditioning not in {"truth", "rollout"}:
        raise ValueError(f"conditioning must be 'truth' or 'rollout', got {conditioning!r}")
    horizon = int(inp.horizon_h)
    if horizon < 1:
        raise ValueError(f"horizon_h must be >= 1, got {horizon}")

    shared: list[np.ndarray] = [np.asarray(inp.x0)]
    if conditioning == "truth":
        shared.extend(np.asarray(inp.truth[i]) for i in range(horizon - 1))
    else:
        roll = np.asarray(
            predict(
                x0=inp.x0,
                static=inp.static,
                weather=inp.weather,
                n_members=1,
                horizon_h=horizon,
                seed=stable_seed("sim.collapse", "rollout", seed),
            )
        )[0]
        shared.extend(np.asarray(roll[i]) for i in range(horizon - 1))

    verdicts: list[CollapseVerdict] = []
    for k in range(1, horizon + 1):
        step_seed = stable_seed("sim.collapse", "lead", k, seed)
        samples = np.asarray(
            predict(
                x0=shared[k - 1],
                static=inp.static,
                weather=inp.weather[k - 1][None, ...],
                n_members=n_members,
                horizon_h=1,
                seed=step_seed,
            )
        )
        if samples.shape != (n_members, 1, *inp.x0.shape):
            raise ValueError(
                "C5 violation: predict() returned "
                f"{samples.shape}, expected {(n_members, 1, *inp.x0.shape)} at lead {k}"
            )
        verdicts.append(
            one_step_collapse_verdict(
                samples,
                shared[k - 1],
                lead_h=k,
                conditioning=conditioning,
                seed=stable_seed("sim.collapse", "control", k, seed),
                n_replicates=n_replicates,
            )
        )

    # The demoted description. A caller that has already made this call passes
    # its array in rather than paying for a second one; the seed is then the
    # caller's, which is why `cumulative_seed` is recorded in `meta` either way.
    cumulative_seed = (
        int(seed)
        if cumulative_samples is not None
        else stable_seed("sim.collapse", "cumulative", seed)
    )
    cumulative = (
        np.asarray(cumulative_samples)
        if cumulative_samples is not None
        else np.asarray(
            predict(
                x0=inp.x0,
                static=inp.static,
                weather=inp.weather,
                n_members=n_members,
                horizon_h=horizon,
                seed=cumulative_seed,
            )
        )
    )
    return PerHorizonCollapse(
        verdicts=tuple(verdicts),
        cumulative_index_description=independence_dispersion_index(cumulative),
        meta={
            "t0": int(inp.t0),
            "horizon_h": horizon,
            "n_members": int(n_members),
            "seed": int(seed),
            "conditioning": conditioning,
            "control_replicates": int(n_replicates),
            "cumulative_seed": int(cumulative_seed),
            "cumulative_samples_supplied": cumulative_samples is not None,
        },
    )


# -- the analytic identity the one-step bar is derived from ------------------


def analytic_shared_latent_index(
    marginals: np.ndarray, latent_sigma: float, *, n_quad: int = 201, span: float = 8.0
) -> float:
    """Exact one-step index for a shared-logit-shift ensemble. No sampling.

    The model is the one this package's fixture implements and the one this
    project commits to: cell ``i`` is Bernoulli with logit ``logit(p_i) + z`` where ``z``
    is ONE normal draw per member, shared by every cell. Then, over the joint,

        var(area) = E_z[sum_i q_i (1 - q_i)] + var_z[sum_i q_i]

    and the index is that over ``sqrt(sum_i pbar_i (1 - pbar_i))`` with
    ``pbar_i = E_z[q_i]``. Gauss-Legendre over a truncated normal, renormalised,
    so the return is a deterministic function of ``marginals`` and
    ``latent_sigma`` alone.

    THIS EXISTS TO REPLACE A BAR, NOT TO ADD ONE. This package's self-test asserted
    that a shared-latent ensemble scores ``> 2.0``. That construct's exact index
    is 13.73, so the bar sat 6.9x below the value it was policing and could not
    have failed for any defect short of the instrument returning nothing. A
    control that asserts a magnitude against an identity can fail in both
    directions; one that asserts non-triviality can fail in neither.
    """
    p = np.asarray(marginals, dtype=np.float64).ravel()
    if p.size == 0:
        raise AbsentMeasurementError(
            "analytic_shared_latent_index over zero cells: an identity with an empty "
            "sum is 0/0, not an answer."
        )
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("marginals must lie strictly inside (0, 1) to have a finite logit")
    if latent_sigma < 0.0:
        raise ValueError(f"latent_sigma must be >= 0, got {latent_sigma}")
    if latent_sigma == 0.0:
        return 1.0

    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    lo, hi = -span * latent_sigma, span * latent_sigma
    z = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * weights * np.exp(-0.5 * (z / latent_sigma) ** 2)
    w = w / w.sum()

    logit = np.log(p / (1.0 - p))
    q = 1.0 / (1.0 + np.exp(-(logit[None, :] + z[:, None])))
    total = q.sum(axis=1)
    within = (q * (1.0 - q)).sum(axis=1)
    e_total = float((w * total).sum())
    e_within = float((w * within).sum())
    var_total = float((w * total * total).sum()) - e_total * e_total
    pbar = (w[:, None] * q).sum(axis=0)
    denom = float((pbar * (1.0 - pbar)).sum())
    return float(np.sqrt((e_within + max(var_total, 0.0)) / denom))


# -- CLI -------------------------------------------------------------------


def _render(result: PerHorizonCollapse) -> str:
    """The table a reader checks the json against. ASCII only."""
    lines = [
        "per-horizon ensemble-collapse verdicts (ADR-114, one-step increment)",
        f"  t0={result.meta['t0']}  members={result.meta['n_members']}  "
        f"conditioning={result.meta['conditioning']}  "
        f"threshold={COLLAPSE_INDEX_THRESHOLD:g}",
        f"  {'lead':>5}{'index':>10}{'indep ctl':>12}{'comono ctl':>12}{'new cells':>11}  verdict",
    ]
    for v in result.verdicts:
        lines.append(
            f"  {v.lead_h:>5}{v.index:>10.4f}{v.controls.independent_index:>12.4f}"
            f"{v.controls.comonotone_index:>12.4f}{v.n_new_cells_mean:>11.1f}  {v.verdict}"
        )
    lines.append(
        f"  cumulative over {result.meta['horizon_h']} h: "
        f"{result.cumulative_index_description:.4f}  DESCRIPTION ONLY, never a verdict"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.collapse",
        description=(
            "Per-horizon ensemble-collapse verdicts on the one-step increment, with "
            "the instrument's own controls in the same invocation (ADR-114)."
        ),
    )
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--out", default=None, help="write the verdict json here")
    ap.add_argument(
        "--model", default="stub", help="'stub', 'stub-nolatent', or a name in model/api.py"
    )
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditioning", default="truth", choices=("truth", "rollout"))
    ap.add_argument("--replicates", type=int, default=CONTROL_REPLICATES)
    args = ap.parse_args(argv)

    from wildfire_nowcast.sim.ensemble import _resolve_predict  # noqa: PLC0415

    predict, model_name = _resolve_predict(args.model)
    inp = c5_inputs(open_tensor(Path(args.tensor)), args.t0, args.horizon)
    try:
        result = per_horizon_collapse(
            predict,
            inp,
            n_members=args.members,
            seed=args.seed,
            conditioning=args.conditioning,
            n_replicates=args.replicates,
        )
    except AbsentMeasurementError as exc:
        print(str(exc))
        return EXIT_NOTHING_EXAMINED

    payload = {"tensor": str(args.tensor), "model": model_name, **result.to_dict()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(_render(result))
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
