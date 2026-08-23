"""The paired BLOCK TEST: a test on block differences instead of an effect size.

WHY A TEST AND NOT AN SD
-----------------------
The acceptance rule this repository has used to adjudicate a degradation ladder
is ``|mean| / sd`` across held-out spatial blocks - an EFFECT SIZE with no
``sqrt(n)`` anywhere. Three properties of it were measured rather than argued:

* its expectation does not improve with more blocks, so a value below the bar
  becomes MORE reliably below it as blocks are added;
* the unanimity conjunct it is paired with is algebraically vacuous at
  ``n <= 8``, because the Samuelson bound caps ``|mean| / sd`` at
  ``sqrt(n - 1) = 1.7889`` for ``n = 5`` - below the 2.536 bar, so at the sample
  size where the rule was written unanimity could not exclude anything;
* ``|mean| / sd`` is upward biased at ``n = 5`` (E = 1.9956 against a truth of
  1.5949), so five blocks read the statistic higher than the quantity it
  estimates - and the bar itself was measured at ``n = 5``.

None of that is a defect of the corpus; all of it is a defect of the statistic.
The replacement is the textbook default and it is not new to this repository:
the entry that first quarantined four headline channels already recommended *"a
test rather than an SD"* and recorded that *"there is precedent and we ignored
it."*

WHAT THIS MODULE IS AND IS NOT
------------------------------
It is three tests on ONE vector of per-block differences, plus their agreement:

``paired_t``
    one-sided paired Student t on the block differences. The pre-registered
    instrument.
``paired_sign``
    the exact one-sided binomial tail on the DIRECTION of the differences,
    delegated to :func:`wildfire_nowcast.eval.stage.sign_test` - one
    implementation, not a second copy.
``sign_flip_permutation``
    the exact randomisation test: every one of the ``2**n`` sign assignments is
    enumerated, so it is exact rather than sampled and needs no normality.

It contains no gate criterion, no bar on a metric, and no knowledge of which arm
is ours. It takes numbers and returns numbers plus a per-test outcome string. A
lead's own code should never contain the word that closes a gate.

THE ORIENTATION IS FIXED BY THE QUESTION, NOT CHOSEN AFTER THE NUMBERS
----------------------------------------------------------------------
Callers pass differences ALREADY ORIENTED so that a POSITIVE value means the
alternative hypothesis. On a degradation ladder that is fixed by what a rung is:
a rung is a deliberately worse forecast, so ``H1`` is "this channel scores the
rung worse than the reference" and :func:`oriented_differences` builds the sign
from the channel's own ``lower_is_better`` flag. One-sidedness is therefore a
property of the design and not a direction picked once the table existed.

WHAT THE THREE TESTS CAN AND CANNOT DO AT n = 5, MEASURED NOT ASSUMED
---------------------------------------------------------------------
Both distribution-free tests have a floor on the p-value they can report: the
smallest attainable one-sided p is ``1 / 2**n``, which is 0.03125 at ``n = 5``.
So at ``alpha = 0.05`` and five blocks BOTH of them require every block to point
the same way, while the t does not. That is reported by
:func:`min_attainable_p` beside every verdict rather than left for a reader to
derive, because "the sign test disagreed" and "the sign test cannot express this
p at this n" are different statements and only the second is about power.

The t's own assumption is normality of the five differences, and five points
cannot support a normality check worth running. That limitation is stated, and
the two distribution-free tests are reported beside it precisely so a
disagreement is visible instead of resolved by preference.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.eval.stage import sign_test

__all__ = [
    "DEFAULT_ALPHA",
    "MAX_BLOCKS_FOR_EXACT_PERMUTATION",
    "BlockDifferences",
    "block_test",
    "min_attainable_p",
    "oriented_differences",
    "paired_sign",
    "paired_t",
    "sign_flip_permutation",
    "student_t_sf",
]

#: The conventional significance level. **Not a fitted constant and it has no
#: fitting sample**: it is the textbook default, named in advance so that the
#: level could not be chosen by looking at which side the numbers landed on.
#: Every entry point takes ``alpha`` as an argument, so this is a default and
#: never a hidden bar; a caller that wants another level states it.
DEFAULT_ALPHA = 0.05

#: Above this many blocks the exact enumeration of ``2**n`` sign assignments
#: stops being cheap. It is a COMPUTE guard, not a statistical threshold: over
#: it, :func:`sign_flip_permutation` refuses rather than silently switching to a
#: sampled approximation, because a test that quietly changes estimand between
#: two runs is the defect this module exists to avoid.
MAX_BLOCKS_FOR_EXACT_PERMUTATION = 20

_OUTCOME_REJECT = "REJECT_H0"
_OUTCOME_KEEP = "DID_NOT_REJECT_H0"


# --------------------------------------------------------------------------
# the Student t tail, computed rather than looked up
# --------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    eps = 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge at a={a}, b={b}")


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: int) -> float:
    """Upper tail ``P(T > t)`` of Student's t on ``df`` degrees of freedom.

    Computed from the regularised incomplete beta rather than sampled. A
    simulated tail is fine for a one-off report and wrong for an instrument that
    must return the same p-value twice: this repository has already recorded a
    p of 0.0117 obtained from two million simulated null draws, and a number that
    moves in the fourth decimal between runs cannot decide a cell that sits near
    the level.

    Checked against three independent references (see the module self-tests):
    the closed forms at ``df = 1`` and ``df = 2``, published critical values, and
    a Monte-Carlo null.
    """
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if math.isnan(t):
        raise ValueError("t is NaN")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    half = 0.5 * _betai(0.5 * df, 0.5, df / (df + t * t))
    return half if t >= 0.0 else 1.0 - half


def min_attainable_p(n_blocks: int) -> dict[str, float | None]:
    """The smallest one-sided p each test can report at this many blocks.

    The t's tail is continuous and has no floor. Both distribution-free tests
    are built on ``2**n`` equally likely outcomes, so neither can report below
    ``1 / 2**n`` however large the effect is. At ``n = 5`` that is 0.03125, which
    is under 0.05 - so they CAN reject, but only on a unanimous set of blocks.
    """
    n = int(n_blocks)
    floor = (1.0 / float(2**n)) if 0 < n <= MAX_BLOCKS_FOR_EXACT_PERMUTATION else None
    return {"paired_t": None, "paired_sign": floor, "sign_flip_permutation": floor}


# --------------------------------------------------------------------------
# building the differences
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockDifferences:
    """One oriented difference per block, with the blocks' labels kept beside it.

    ``label`` travels with the numbers on purpose. Every outcome string this
    module builds repeats it, so a planted set of differences names itself in
    the text that reports the rejection, and a rejection cannot be read as
    belonging to the real data.
    """

    labels: tuple[str, ...]
    values: tuple[float, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.values):
            raise ValueError(f"{len(self.labels)} labels against {len(self.values)} values")
        for v in self.values:
            if not math.isfinite(v):
                raise ValueError(f"non-finite block difference {v} in {self.label!r}")

    @property
    def n(self) -> int:
        return len(self.values)

    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def signs(self) -> dict[str, int]:
        """Per-block signs. Published beside every separation, never in an annex."""
        return {k: int(np.sign(v)) for k, v in zip(self.labels, self.values, strict=True)}


def oriented_differences(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    lower_is_better: bool,
    label: str = "",
) -> BlockDifferences:
    """Per-block differences oriented so POSITIVE means the alternative.

    Only blocks present and finite on BOTH sides contribute; a block missing on
    one side is dropped from the pairing rather than imputed, and the caller can
    see it happened because ``n`` falls.
    """
    keys = sorted(set(candidate) & set(reference))
    rows: list[tuple[str, float]] = []
    for key in keys:
        a, b = float(candidate[key]), float(reference[key])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        rows.append((key, (a - b) if lower_is_better else (b - a)))
    return BlockDifferences(
        labels=tuple(k for k, _ in rows), values=tuple(v for _, v in rows), label=label
    )


def _as_differences(diffs: BlockDifferences | Sequence[float], label: str) -> BlockDifferences:
    if isinstance(diffs, BlockDifferences):
        return diffs if not label else BlockDifferences(diffs.labels, diffs.values, label)
    values = tuple(float(v) for v in diffs)
    return BlockDifferences(
        labels=tuple(f"block_{i}" for i in range(len(values))), values=values, label=label
    )


def _undefined(d: BlockDifferences, reason: str, alpha: float) -> dict[str, Any]:
    return {
        "outcome": f"UNDEFINED_{reason}",
        "rejects": False,
        "p_one_sided": None,
        "alpha": alpha,
        "n_blocks": d.n,
        "label": d.label,
        "detail": (
            f"{d.label or 'unlabelled'}: undefined ({reason}) on {d.n} block(s); "
            "an undefined test is NOT a rejection and is NOT a pass"
        ),
    }


# --------------------------------------------------------------------------
# the three tests
# --------------------------------------------------------------------------
def paired_t(
    diffs: BlockDifferences | Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    label: str = "",
) -> dict[str, Any]:
    """One-sided paired Student t on block differences. ``H1: mean > 0``.

    Two degenerate cases are refused rather than reported, both for the reason
    the block separation already refuses them: a block-to-block SD estimated as
    exactly zero from a handful of blocks is not an SD, and dividing by it turns
    any margin into infinite significance. All-zero differences (an identity
    rung) are ``0 / 0`` and are refused separately, because a rung that changed
    nothing has produced no evidence in either direction.
    """
    d = _as_differences(diffs, label)
    if d.n < 2:
        return _undefined(d, "fewer_than_two_blocks", alpha)
    arr = d.array()
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        reason = "all_differences_identically_zero" if mean == 0.0 else "zero_block_sd"
        out = _undefined(d, reason, alpha)
        out.update({"mean_diff": mean, "sd_diff": sd, "df": d.n - 1})
        return out
    df = d.n - 1
    t_stat = mean / (sd / math.sqrt(d.n))
    p = student_t_sf(t_stat, df)
    rejects = bool(p <= alpha)
    return {
        "outcome": _OUTCOME_REJECT if rejects else _OUTCOME_KEEP,
        "rejects": rejects,
        "p_one_sided": float(p),
        "alpha": alpha,
        "t_statistic": float(t_stat),
        "df": df,
        "n_blocks": d.n,
        "mean_diff": mean,
        "sd_diff": sd,
        "effect_size_abs_mean_over_sd": abs(mean) / sd,
        "per_block_sign": d.signs(),
        "label": d.label,
        "detail": (
            f"{d.label or 'unlabelled'}: t = {t_stat:.4f} on {df} df, one-sided p = {p:.6f} "
            f"against alpha = {alpha}, {'REJECTS' if rejects else 'does not reject'} H0"
        ),
    }


def paired_sign(
    diffs: BlockDifferences | Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    label: str = "",
) -> dict[str, Any]:
    """Exact one-sided sign test on the DIRECTION of the block differences.

    Blocks whose difference is exactly zero are evidence in neither direction and
    are removed from ``n`` before the tail is taken, which is the tie rule
    :func:`wildfire_nowcast.eval.stage.sign_test` requires of its caller. Their
    count is reported, because silently counting them as unfavourable would be a
    choice made inside a p-value.
    """
    d = _as_differences(diffs, label)
    arr = d.array()
    n_zero = int((arr == 0.0).sum())
    kept = arr[arr != 0.0]
    if kept.size == 0:
        out = _undefined(d, "every_block_difference_is_zero", alpha)
        out["n_zero_blocks"] = n_zero
        return out
    k = int((kept > 0).sum())
    res = sign_test(k, int(kept.size))
    p = res["p_one_sided"]
    rejects = bool(p is not None and p <= alpha)
    return {
        "outcome": _OUTCOME_REJECT if rejects else _OUTCOME_KEEP,
        "rejects": rejects,
        "p_one_sided": None if p is None else float(p),
        "alpha": alpha,
        "n_blocks": d.n,
        "n_nonzero_blocks": int(kept.size),
        "n_zero_blocks": n_zero,
        "k_favourable": k,
        "unanimous": bool(k == kept.size),
        "per_block_sign": d.signs(),
        "label": d.label,
        "detail": (
            f"{d.label or 'unlabelled'}: {k} of {kept.size} blocks favour H1"
            f"{f' ({n_zero} exactly zero, removed)' if n_zero else ''}, exact one-sided "
            f"p = {p:.6f} against alpha = {alpha}, "
            f"{'REJECTS' if rejects else 'does not reject'} H0"
        ),
    }


def sign_flip_permutation(
    diffs: BlockDifferences | Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    label: str = "",
    statistic: str = "mean",
) -> dict[str, Any]:
    """Exact paired randomisation test: all ``2**n`` sign assignments, enumerated.

    Under the null that each block's difference is symmetric about zero, every
    sign assignment is equally likely, so the exact one-sided p is the fraction
    of assignments whose statistic reaches the observed one. Nothing is sampled
    and no seed is involved, so this returns the same number every time it runs.

    ``statistic`` is reported for BOTH choices by :func:`block_test`. The mean is
    the classical randomisation statistic; the t is the permutation form of the
    parametric test and is not a monotone function of the mean under sign flips,
    since flipping a sign moves the SD as well. They are reported together
    because choosing between them after seeing the table is exactly the freedom
    a pre-registration removes.
    """
    d = _as_differences(diffs, label)
    if d.n < 2:
        return _undefined(d, "fewer_than_two_blocks", alpha)
    if d.n > MAX_BLOCKS_FOR_EXACT_PERMUTATION:
        return _undefined(d, f"more_than_{MAX_BLOCKS_FOR_EXACT_PERMUTATION}_blocks", alpha)
    if statistic not in ("mean", "t"):
        raise ValueError(f"statistic must be 'mean' or 't', got {statistic!r}")
    arr = d.array()

    def stat(values: np.ndarray) -> float:
        if statistic == "mean":
            return float(values.mean())
        sd = float(values.std(ddof=1))
        if sd == 0.0:
            return math.inf if values.mean() > 0 else (-math.inf if values.mean() < 0 else 0.0)
        return float(values.mean() / (sd / math.sqrt(values.size)))

    observed = stat(arr)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=d.n)), dtype=np.float64)
    draws = np.array([stat(row * arr) for row in signs], dtype=np.float64)
    n_total = int(draws.size)
    n_at_least = int((draws >= observed).sum())
    p = n_at_least / float(n_total)
    rejects = bool(p <= alpha)
    return {
        "outcome": _OUTCOME_REJECT if rejects else _OUTCOME_KEEP,
        "rejects": rejects,
        "p_one_sided": float(p),
        "alpha": alpha,
        "statistic": statistic,
        "observed_statistic": observed,
        "n_assignments": n_total,
        "n_at_least_as_extreme": n_at_least,
        "min_attainable_p": 1.0 / float(n_total),
        "n_blocks": d.n,
        "per_block_sign": d.signs(),
        "label": d.label,
        "detail": (
            f"{d.label or 'unlabelled'}: {n_at_least} of {n_total} sign assignments reach the "
            f"observed {statistic}, exact one-sided p = {p:.6f} against alpha = {alpha}, "
            f"{'REJECTS' if rejects else 'does not reject'} H0"
        ),
    }


# --------------------------------------------------------------------------
# all three, and whether they agree
# --------------------------------------------------------------------------
def block_test(
    diffs: BlockDifferences | Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    label: str = "",
) -> dict[str, Any]:
    """The pre-registered t, with the two distribution-free tests beside it.

    ``agreement`` is reported and no test is preferred. A disagreement between
    the t and either exact test at ``n = 5`` is a statement about power, not a
    tie to be broken: see :func:`min_attainable_p`.
    """
    d = _as_differences(diffs, label)
    t_res = paired_t(d, alpha=alpha)
    sign_res = paired_sign(d, alpha=alpha)
    perm_mean = sign_flip_permutation(d, alpha=alpha, statistic="mean")
    perm_t = sign_flip_permutation(d, alpha=alpha, statistic="t")
    votes = {
        "paired_t": t_res["rejects"],
        "paired_sign": sign_res["rejects"],
        "sign_flip_permutation_mean": perm_mean["rejects"],
        "sign_flip_permutation_t": perm_t["rejects"],
    }
    return {
        "label": d.label,
        "n_blocks": d.n,
        "alpha": alpha,
        "per_block_diff": dict(zip(d.labels, d.values, strict=True)),
        "per_block_sign": d.signs(),
        "paired_t": t_res,
        "paired_sign": sign_res,
        "sign_flip_permutation_mean": perm_mean,
        "sign_flip_permutation_t": perm_t,
        "min_attainable_p": min_attainable_p(d.n),
        "rejects": votes,
        "all_three_agree": len(
            {votes["paired_t"], votes["paired_sign"], votes["sign_flip_permutation_mean"]}
        )
        == 1,
        "t_and_sign_agree": votes["paired_t"] == votes["paired_sign"],
        "t_and_permutation_agree": votes["paired_t"] == votes["sign_flip_permutation_mean"],
    }
