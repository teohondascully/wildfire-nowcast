"""Known-answer verification for the C5 baselines and the C6 metrics.

``tests/`` is infra's directory, so this lives here and is runnable
standalone::

    .venv/bin/python -m wildfire_nowcast.eval.selftest
    .venv/bin/python -m wildfire_nowcast.eval.selftest --json

Every check has an answer that is known BEFORE the code runs - hand-computed,
or forced by an algebraic identity - because a metric verified only against its
own output is verified against nothing. The two that matter most:

* ``crps_fair_known_answer``: a 2-member ensemble bracketing the truth
  symmetrically has fair CRPS exactly 0 and biased CRPS exactly 0.5. That pins
  down which estimator is wired in, which decides whether a collapsed ensemble
  is rewarded at G3.
* ``collapse_is_invisible_to_dispersion_ratio``: two ensembles with IDENTICAL
  per-pixel probabilities - one built from independent per-pixel noise, one from
  a shared latent - get the same Brier and the same ``dispersion_ratio``, and
  are told apart only by ``area_dispersion_ratio`` and member diversity. This is
  the G3 ablation in miniature, and it is asserted here so that the metric
  cannot silently stop being able to see it.

This module is a proposed pytest target, escalated rather than assumed;
:func:`run_all` returns structured results so a test can be three lines.
"""

from __future__ import annotations

import argparse
import atexit as _atexit
import functools as _functools
import json
import shutil as _shutil
import sys
import tempfile as _tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from wildfire_nowcast.common import zarr_io as zio
from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.eval.metrics import (
    aggregate,
    arrival_times,
    crps_ensemble,
    evaluate,
    fuzzy_iou,
)
from wildfire_nowcast.eval.reporting import reporting_status
from wildfire_nowcast.model.api import predict, validate_predict_inputs
from wildfire_nowcast.model.baselines import EllipseBaseline, PersistenceBaseline
from wildfire_nowcast.model.inputs import (
    forecast_inputs,
    static_index,
    weather_index,
)
from wildfire_nowcast.model.spread import ellipse_ros_factor, length_to_breadth

__all__ = ["Check", "run_all", "main"]

_TOL = 1e-9


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    values: dict[str, Any] = field(default_factory=dict)


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@_functools.cache
def _process_scratch(prefix: str) -> str:
    """One scratch directory per process, removed when the process exits.

    NOT ``tempfile.mkdtemp`` per call. ``mkdtemp`` never cleans up, this fixture
    writes a ~4.7 MB synthetic tensor, and the mutation gate runs the whole
    suite once per mutant. At 117 mutants a sweep that is half a gigabyte per
    sweep, and it reached 70 GB across two prefixes before anyone looked at a
    disk. Caching by prefix makes the cost per PROCESS rather than per CALL, and
    ``atexit`` makes it zero once the process ends.
    """
    tmp = _tempfile.mkdtemp(prefix=prefix)
    _atexit.register(_shutil.rmtree, tmp, ignore_errors=True)
    return tmp


def _synthetic_dataset(tmp: str | None = None):
    """The C4 fixture, imported LAZILY on purpose.

    ``common/synthetic.py`` is infra's and is edited concurrently with this
    suite. Importing it at module scope means one broken line in someone else's
    fixture takes down every known-answer check in here, including the ones that
    have nothing to do with it - which is how a whole verification instrument
    goes dark for an unrelated reason. Lazily, a broken fixture fails the two
    checks that use it and leaves the other twenty-five reporting.
    """
    from pathlib import Path

    from wildfire_nowcast.common.synthetic import make_synthetic_fire

    out = Path(tmp or _process_scratch("wnc-selftest-")) / "synthetic" / "tensor.zarr"
    result = make_synthetic_fire(0, 24, out=out)
    return zio.open_tensor(result.tensor_path), result


def _binary_samples(members: np.ndarray) -> np.ndarray:
    """``[M,H,W]`` bool -> valid single-lead C5 samples from an all-unburned x0."""
    return members.astype(np.uint8)[:, None, :, :]


# --------------------------------------------------------------------------
# C6 checks
# --------------------------------------------------------------------------


def check_perfect_forecast() -> Check:
    """A forecast identical to the truth: Brier 0, IoU 1, CRPS 0."""
    rng = np.random.default_rng(0)
    truth = (rng.random((3, 16, 16)) < 0.3).astype(np.uint8)
    truth = np.maximum.accumulate(truth, axis=0)  # absorbing
    samples = np.repeat(truth[None], 5, axis=0)
    res = evaluate(samples, truth, x0=np.zeros((16, 16), np.uint8))
    ok = (
        _close(res["brier_1h"], 0.0)
        and _close(res["brier_2h"], 0.0)
        and _close(res["brier_3h"], 0.0)
        and _close(res["arrival_crps"], 0.0)
        and _close(res["best_member_iou"], 1.0)
    )
    return Check(
        "perfect_forecast",
        ok,
        "brier/crps 0 and IoU 1",
        {
            "brier_1h": res["brier_1h"],
            "arrival_crps": res["arrival_crps"],
            "best_member_iou": res["best_member_iou"],
        },
    )


def check_brier_hand_computed() -> Check:
    """Half the members burn everything, half nothing, truth burns: Brier = 0.25."""
    shape = (12, 12)
    members = np.concatenate([np.ones((4, *shape), bool), np.zeros((4, *shape), bool)], axis=0)
    samples = _binary_samples(members)
    truth = np.ones((1, *shape), np.uint8)
    res = evaluate(samples, truth, x0=np.zeros(shape, np.uint8), leads=(1,))
    ok = _close(res["brier_1h"], 0.25)
    return Check(
        "brier_hand_computed", ok, "p=0.5, y=1 -> Brier 0.25", {"brier_1h": res["brier_1h"]}
    )


def check_crps_fair_known_answer() -> Check:
    """Members {1,3}, truth 2. Fair CRPS = 0 exactly; biased = 0.5 exactly."""
    members = np.array([[1.0], [3.0]])
    observed = np.array([2.0])
    fair_sum, n = crps_ensemble(members, observed, fair=True)
    biased_sum, _ = crps_ensemble(members, observed, fair=False)
    ok = _close(fair_sum / n, 0.0) and _close(biased_sum / n, 0.5)
    return Check(
        "crps_fair_known_answer",
        ok,
        "fair estimator does not reward under-dispersion",
        {"fair": fair_sum / n, "biased": biased_sum / n},
    )


def check_arrival_times_censoring() -> Check:
    """Arrival is the first true lead, 1-based; never-arriving cells cap at L+1."""
    event = np.zeros((3, 1, 4), bool)
    event[0, 0, 0] = True  # arrives at lead 1
    event[1, 0, 1] = True
    event[2, 0, 2] = True
    # column 3 never burns -> cap 4
    got = arrival_times(event)[0]
    ok = np.allclose(got, [1, 2, 3, 4])
    return Check("arrival_times_censoring", ok, "1-based, capped at L+1", {"arrival": got.tolist()})


def check_dispersion_calibrated_is_one() -> Check:
    """A calibrated binary ensemble has dispersion_ratio ~ 1 by construction.

    Members and truth drawn independently from the same per-pixel Bernoulli, so
    the forecast is calibrated. The ratio must come out near 1 - this is the
    algebraic identity documented in :func:`~wildfire_nowcast.eval.metrics.dispersion`,
    and it is checked here because it is also the reason the ratio cannot detect
    collapse.
    """
    rng = np.random.default_rng(7)
    shape = (140, 140)
    q = rng.random(shape) * 0.8 + 0.1
    members = rng.random((24, *shape)) < q
    truth = (rng.random(shape) < q).astype(np.uint8)[None]
    res = evaluate(_binary_samples(members), truth, x0=np.zeros(shape, np.uint8), leads=(1,))
    ratio = res["dispersion_ratio"]
    ok = ratio is not None and 0.95 <= ratio <= 1.05
    return Check(
        "dispersion_calibrated_is_one",
        ok,
        "spread-skill ~ 1 when calibrated",
        {"dispersion_ratio": ratio},
    )


def _collapse_pair(n_members: int, shape: tuple[int, int] = (100, 100), seed: int = 11):
    """Two ensembles with the same intended p = 0.5 field, one truth scenario.

    * ``independent`` - every pixel of every member flipped independently. The
      known-broken model CLAUDE.md permits only as an ablation. Every member
      burns about half the domain, so the ensemble has almost no spread in
      burned AREA while its area error is enormous.
    * ``shared`` - half the members burn everything, half burn nothing: all the
      randomness lives in one shared latent. Same per-pixel probability, but the
      members are genuine alternative scenarios.

    Truth is a COHERENT SCENARIO (the fire made its run; everything burned), not
    a pixel-wise coin flip. See the docstring of the check below for why that
    distinction is the whole ballgame.
    """
    rng = np.random.default_rng(seed)
    independent = rng.random((n_members, *shape)) < 0.5
    coin = np.zeros(n_members, dtype=bool)
    coin[: n_members // 2] = True  # exactly p = 0.5, so Brier is exactly 0.25
    shared = np.broadcast_to(coin[:, None, None], (n_members, *shape)).copy()
    truth = np.ones((1, *shape), np.uint8)  # the scenario that actually happened
    zeros = np.zeros(shape, np.uint8)
    a = evaluate(_binary_samples(independent), truth, x0=zeros, leads=(1,))
    b = evaluate(_binary_samples(shared), truth, x0=zeros, leads=(1,))
    return a, b


def check_collapse_is_invisible_to_dispersion_ratio() -> Check:
    """THE G3 CHECK, in miniature. A KNOWN-ANSWER test, not a fuzzy-equality one.

    G3's whole argument is that a shared latent buys ensemble dispersion that
    independent per-pixel noise cannot. That argument is only as trustworthy as
    the instrument, so this check pins the instrument to closed forms.

    With ``p = 0.5`` everywhere and a truth of all-ones, both arms have exact
    analytic scores, and ``dispersion_ratio = sqrt(var * (M+1)/M / mse)`` gives::

        independent :  var = 0.25 exactly (the 1/M deficit in E[p(1-p)] is
                       cancelled by the M/(M-1) unbiasing factor)
                       mse = 0.25 (M+1)/M      ->  ratio == 1 EXACTLY, any M
        shared      :  var = 0.25 M/(M-1), mse = 0.25
                       ->  ratio == sqrt((M+1)/(M-1))  ->  1 as M -> inf

    Both verified to ~1e-16 across M in {20, 50, 100, 200, 1000}. So:

    1. ``dispersion_ratio`` cannot detect collapse - and worse, it is ANTI-
       correlated with what G3 cares about. It scores the COLLAPSED ensemble at
       exactly 1.000 (textbook-perfect) and the good scenario ensemble at
       ``sqrt((M+1)/(M-1))`` > 1 ("over-dispersed"). Anyone who used it as the
       G3 criterion would prefer the broken model. This is a stronger and more
       dangerous statement than "it is blind".
    2. The residual difference between the arms is that pure ``O(1/M)``
       bookkeeping term, NOT a signal. It is asserted to shrink like 1/M rather
       than absorbed into a tolerance, because a tolerance wide enough to hide
       it at M=20 would also hide a real effect.
    3. ``area_dispersion_ratio`` separates them by ~100x AT ONE LEAD STEP
       (``_collapse_pair`` scores ``leads=(1,)``), and is the number the G3
       ablation must actually be judged on. Measured: 106.4x at M=20 and 99.1x
       at M=200, which is where ``docs/interfaces.md`` C6.1's ~106x comes from.
       The body below asserts the weaker ``> 50x`` on purpose, so the check pins
       a SEPARATION and does not become a regression test on a two-digit
       constant; the ~100x is the reading, the 50x is the commitment.

    HISTORY - the failure this check survived, kept because the reasoning
    recurs. The first version scored both arms against an iid per-pixel
    coin-flip truth and asserted ``area_a < 0.1``. It failed with
    ``area_a = 7.85, area_b = 525.66``. The instinct is to suspect the metric;
    that would have been wrong. In an iid-coin-flip world the truth burns ~50%
    of the domain EVERY time, so there is no scenario uncertainty for a latent
    to capture and independent per-pixel noise is the CORRECTLY SPECIFIED model.
    The area-error denominator goes to ~0 and the ratio diverges for both arms -
    the metric was correctly reporting "over-dispersed", in a world where
    nothing collapses. A toy that cannot exhibit the phenomenon cannot test the
    detector for it. Diagnostic that separates the two hypotheses: sweep M and
    look for a closed form. A real detection would not land on
    ``sqrt((M+1)/(M-1))`` to machine precision.
    """
    m_small, m_large = 20, 200
    a, b = _collapse_pair(m_small)
    a_big, b_big = _collapse_pair(m_large)

    disp_a, disp_b = a["dispersion_ratio"], b["dispersion_ratio"]
    area_a = a["by_mask"]["domain"]["area_dispersion_ratio"]
    area_b = b["by_mask"]["domain"]["area_dispersion_ratio"]
    div_a = a["diagnostics"]["mean_pairwise_member_iou"]
    div_b = b["diagnostics"]["mean_pairwise_member_iou"]

    # (1) closed forms. The collapsed arm sits at exactly 1; the good arm at the
    #     finite-M term. Sampling noise over 1e4 pixels is O(1e-3).
    exact_shared = float(np.sqrt((m_small + 1) / (m_small - 1)))
    forms_ok = _close(disp_a, 1.0, tol=3e-3) and _close(disp_b, exact_shared, tol=1e-6)
    brier_ok = _close(a["brier_1h"], 0.25 * (m_small + 1) / m_small, tol=3e-3) and _close(
        b["brier_1h"], 0.25, tol=1e-12
    )

    # (2) neither arm trips a calibration alarm, and the COLLAPSED one looks
    #     BETTER. This is the sentence that must never quietly stop being true.
    both_look_calibrated = 0.9 < disp_a < 1.1 and 0.9 < disp_b < 1.1
    collapse_looks_better = abs(disp_a - 1.0) < abs(disp_b - 1.0)

    # (3) the gap is 1/M bookkeeping: 10x the members -> ~10x smaller gap.
    shrinks_like_one_over_m = abs(b_big["dispersion_ratio"] - 1.0) < abs(disp_b - 1.0) / 5.0
    stable_in_m = _close(a_big["dispersion_ratio"], 1.0, tol=3e-3)

    # (4) what DOES separate them, at both member counts.
    area_separates = area_a < 0.1 and 0.5 < area_b < 2.0 and area_b > 50 * area_a
    area_separates_big = (
        b_big["by_mask"]["domain"]["area_dispersion_ratio"]
        > 50 * (a_big["by_mask"]["domain"]["area_dispersion_ratio"])
    )
    diversity_separates = div_a is not None and div_b is not None and div_b > div_a

    ok = bool(
        forms_ok
        and brier_ok
        and both_look_calibrated
        and collapse_looks_better
        and shrinks_like_one_over_m
        and stable_in_m
        and area_separates
        and area_separates_big
        and diversity_separates
    )
    return Check(
        "collapse_is_invisible_to_dispersion_ratio",
        ok,
        "dispersion_ratio scores COLLAPSE at exactly 1.0; only area_dispersion_ratio sees it",
        {
            "brier_independent": a["brier_1h"],
            "brier_shared": b["brier_1h"],
            "dispersion_independent": disp_a,
            "dispersion_shared": disp_b,
            "dispersion_shared_closed_form": exact_shared,
            "dispersion_shared_M200": b_big["dispersion_ratio"],
            "area_dispersion_independent": area_a,
            "area_dispersion_shared": area_b,
            "pairwise_iou_independent": div_a,
            "pairwise_iou_shared": div_b,
            "verdict": (
                "collapsed ensemble reads BETTER on dispersion_ratio "
                f"({disp_a:.6f} vs {disp_b:.6f}) and ~{area_b / max(area_a, 1e-12):.0f}x "
                "WORSE on area_dispersion_ratio"
            ),
        },
    )


def check_fuzzy_iou_reduces_to_jaccard() -> Check:
    a = np.zeros((6, 6), bool)
    a[2:4, 2:4] = True
    b = np.zeros((6, 6), bool)
    b[3:5, 3:5] = True
    exact = fuzzy_iou(a, b, 0)
    tolerant = fuzzy_iou(a, b, 1)
    # |A n B| = 1, |A u B| = 7 -> 1/7
    ok = _close(exact, 1.0 / 7.0) and tolerant > exact and tolerant <= 1.0
    return Check(
        "fuzzy_iou_reduces_to_jaccard",
        ok,
        "tol=0 is Jaccard; tol>0 is bounded",
        {"exact": exact, "tolerant": tolerant},
    )


def check_aggregate_pools_sufficient_statistics() -> Check:
    """Pooling two windows must equal scoring their concatenation, not averaging."""
    shape = (10, 10)
    zeros = np.zeros(shape, np.uint8)
    results = []
    sse = 0.0
    count = 0
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        members = r.random((6, *shape)) < 0.4
        truth = (r.random(shape) < 0.4).astype(np.uint8)[None]
        res = evaluate(_binary_samples(members), truth, x0=zeros, leads=(1,))
        results.append(res)
        p = members.mean(axis=0)
        sse += float(np.sum((p - truth[0]) ** 2))
        count += p.size
    pooled = aggregate(results)
    expected = sse / count
    naive = float(np.mean([r["brier_1h"] for r in results]))
    ok = _close(pooled["brier_1h"], expected)
    return Check(
        "aggregate_pools_sufficient_statistics",
        ok,
        "pooled Brier = total SSE / total n",
        {"pooled": pooled["brier_1h"], "expected": expected, "naive_mean_of_briers": naive},
    )


def check_zero_growth_window_is_free_for_persistence() -> Check:
    """On a window where truth does not grow, persistence scores a perfect 0.

    This is the whole reason the label cadence is a design constraint and not
    trivia: ~79% of hourly windows look like this one, and on every one of them
    the floor is unbeatable. A pooled all-hours score is mostly a report on how
    often nothing happened.
    """
    shape = (20, 20)
    x0 = np.zeros(shape, np.uint8)
    x0[8:12, 8:12] = 1
    truth = np.repeat(x0[None], 3, axis=0)
    samples = PersistenceBaseline().predict(
        x0, np.zeros((8, *shape), np.float32), np.zeros((3, 5, *shape), np.float32), 4, 3, 0
    )
    res = evaluate(samples, truth, x0=x0)
    ok = _close(res["brier_1h"], 0.0) and _close(res["arrival_crps"], 0.0)
    return Check(
        "zero_growth_window_is_free_for_persistence",
        ok,
        "the floor is perfect on ~79% of hourly windows",
        {"brier_1h": res["brier_1h"], "best_member_iou": res["best_member_iou"]},
    )


# --------------------------------------------------------------------------
# C5 checks
# --------------------------------------------------------------------------


def check_persistence_is_identity() -> Check:
    rng = np.random.default_rng(5)
    shape = (14, 14)
    x0 = rng.integers(0, 3, shape).astype(np.uint8)
    static = np.zeros((8, *shape), np.float32)
    weather = np.zeros((3, 5, *shape), np.float32)
    out_a = predict(x0, static, weather, 3, 3, 0, model="persistence")
    out_b = predict(x0, static, weather, 3, 3, 999, model="persistence")
    ok = np.array_equal(out_a, out_b) and all(
        np.array_equal(out_a[m, k], x0) for m in range(3) for k in range(3)
    )
    return Check("persistence_is_identity", ok, "seed-invariant, equals x0 everywhere")


def check_input_validation_rejects_traps() -> Check:
    """Each of these has a silent-wrong-answer failure mode; all must raise."""
    shape = (8, 8)
    x0 = np.zeros(shape, np.uint8)
    static = np.zeros((8, *shape), np.float32)
    weather = np.zeros((3, 5, *shape), np.float32)
    traps: list[tuple[str, Callable[[], Any]]] = [
        (
            "float x0",
            lambda: validate_predict_inputs(x0.astype(np.float32), static, weather, 1, 3, 0),
        ),
        (
            "transposed static",
            lambda: validate_predict_inputs(x0, static.transpose(0, 2, 1)[:, :7], weather, 1, 3, 0),
        ),
        ("short weather slab", lambda: validate_predict_inputs(x0, static, weather[:2], 1, 3, 0)),
        (
            "weather/static channel swap",
            lambda: validate_predict_inputs(
                x0, np.zeros((5, *shape), np.float32), weather, 1, 3, 0
            ),
        ),
        (
            "state 3 in x0",
            lambda: validate_predict_inputs(np.full(shape, 3, np.uint8), static, weather, 1, 3, 0),
        ),
        (
            "non-finite weather",
            lambda: validate_predict_inputs(
                x0, static, np.full((3, 5, *shape), np.nan, np.float32), 1, 3, 0
            ),
        ),
    ]
    missed = []
    for label, fn in traps:
        try:
            fn()
        except (TypeError, ValueError):
            continue
        missed.append(label)
    return Check(
        "input_validation_rejects_traps",
        not missed,
        "all C5 argument traps raise",
        {"missed": missed},
    )


def _toy_scene(shape: tuple[int, int] = (25, 25), wind_u: float = 20.0):
    """A grass plain, one lit cell in the middle, a hard east wind."""
    x0 = np.zeros(shape, np.uint8)
    x0[shape[0] // 2, shape[1] // 2] = 1
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0
    weather = np.zeros((6, 5, *shape), np.float32)
    weather[:, weather_index("wind_u10")] = wind_u
    weather[:, weather_index("fuel_moisture_proxy")] = 3.0
    return x0, static, weather


def check_kernel_torch_physics_matches_numpy() -> Check:
    """C0 anti-drift: the torch physics must equal :mod:`spread`'s numpy physics.

    The kernel is a SECOND implementation of the ellipse's physics, because the
    forward pass has to be differentiable. That is the exact duplication C0
    forbids, so it is admitted and pinned instead of trusted: if these ever
    diverge, "the kernel beat the ellipse" stops meaning what it says.
    """
    from wildfire_nowcast.model.kernel import check_torch_matches_numpy

    try:
        errors = check_torch_matches_numpy()
        return Check("kernel_torch_physics_matches_numpy", True, "max |torch - numpy|", errors)
    except AssertionError as exc:
        return Check("kernel_torch_physics_matches_numpy", False, str(exc))


def check_kernel_init_is_the_ellipse_shape() -> Check:
    """At init the kernel must be anisotropic downwind, like the ellipse it copies."""
    from wildfire_nowcast.model.kernel import ContagionKernel

    x0, static, weather = _toy_scene()
    prob = ContagionKernel().predict_proba(x0, static, weather, 6)[-1]
    mid = x0.shape[1] // 2
    east = float(prob[:, mid + 1 :].sum())
    west = float(prob[:, :mid].sum())
    ok = east > 3 * max(west, 1e-9)
    return Check(
        "kernel_init_is_the_ellipse_shape",
        ok,
        "elliptical-Gaussian init runs downwind (east >> west) before any training",
        {"downwind_mass": east, "upwind_mass": west, "ratio": east / max(west, 1e-9)},
    )


def check_kernel_samples_are_absorbing() -> Check:
    """C5/C1.1 state algebra holds for the sampler, not just for the mean field."""
    from wildfire_nowcast.model.api import validate_samples
    from wildfire_nowcast.model.kernel import ContagionKernel

    x0, static, weather = _toy_scene()
    samples = ContagionKernel().predict(x0, static, weather, 8, 6, 0)
    try:
        validate_samples(samples, x0, 8, 6)
        ok, detail = True, "0 -> 1 -> 2 only; never unburns a cell burned at x0"
    except (ValueError, TypeError) as exc:  # pragma: no cover - failure path
        ok, detail = False, str(exc)
    repeat = ContagionKernel().predict(x0, static, weather, 8, 6, 0)
    deterministic = bool(np.array_equal(samples, repeat))
    return Check(
        "kernel_samples_are_absorbing",
        ok and deterministic,
        detail + "; same seed reproduces bit for bit",
        {"seed_reproducible": deterministic},
    )


def check_kernel_ensemble_collapses_in_area() -> Check:
    """The kernel's OWN ensemble must collapse. It is the G3 ablation, not a model.

    With no ``z_t`` the only sampler available is independent-per-pixel
    Bernoulli, and independent noise averages out over thousands of pixels: the
    members must therefore agree closely on total area even while disagreeing
    cell by cell. Asserting the FAILURE here means the deterministic milestone
    cannot be mistaken for a dispersion result, and it gives G3 a measured
    negative control before the latent is built.
    """
    from wildfire_nowcast.model.kernel import ContagionKernel

    horizon_h, n_members = 6, 24
    x0, static, weather = _toy_scene(wind_u=14.0)
    samples = ContagionKernel().predict(x0, static, weather, n_members, horizon_h, 0)
    areas = (samples[:, -1] > 0).sum(axis=(1, 2)).astype(float)
    cv = float(areas.std(ddof=1) / max(areas.mean(), 1e-9))
    ellipse = EllipseBaseline().predict(x0, static, weather, n_members, horizon_h, 0)
    ellipse_areas = (ellipse[:, -1] > 0).sum(axis=(1, 2)).astype(float)
    ellipse_cv = float(ellipse_areas.std(ddof=1) / max(ellipse_areas.mean(), 1e-9))
    ok = cv < ellipse_cv
    return Check(
        "kernel_ensemble_collapses_in_area",
        ok,
        "independent-per-pixel members have LESS area spread than shared-innovation "
        f"members at lead {horizon_h} h over {n_members} members — the collapse G3 "
        "must fix with z_t",
        {
            "kernel_area_cv": cv,
            "ellipse_area_cv": ellipse_cv,
            "horizon_h": horizon_h,
            "n_members": n_members,
        },
    )


def check_kernel_growth_increases_with_wind() -> Check:
    """More wind must mean more expected growth. The cheapest sanity check there is."""
    from wildfire_nowcast.model.kernel import ContagionKernel

    model = ContagionKernel()
    growth = []
    for wind in (2.0, 8.0, 18.0):
        x0, static, weather = _toy_scene(wind_u=wind)
        growth.append(float(model.predict_proba(x0, static, weather, 3)[-1].sum()))
    ok = growth[0] < growth[1] < growth[2]
    return Check(
        "kernel_growth_increases_with_wind",
        ok,
        "expected new cells is monotone in wind speed",
        {"expected_cells_at_2_8_18_ms": growth},
    )


def check_kernel_respects_barriers_and_nonburnable() -> Check:
    """A barrier column and a non-burnable column must both suppress ignition."""
    from wildfire_nowcast.model.kernel import ContagionKernel

    shape = (24, 24)
    x0 = np.zeros(shape, np.uint8)
    x0[10:14, 2:6] = 1
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0
    static[static_index("fuel_model_id")][:, 12] = 98.0
    static[static_index("water_barrier_mask")][:, 18] = 1.0
    weather = np.zeros((6, 5, *shape), np.float32)
    weather[:, weather_index("wind_u10")] = 25.0
    weather[:, weather_index("fuel_moisture_proxy")] = 2.0
    prob = ContagionKernel().predict_proba(x0, static, weather, 6)[-1]
    nonburnable = float(prob[:, 12].max())
    barrier = float(prob[:, 18].max())
    open_ground = float(prob[:, 8].max())
    # THRESHOLDS CHANGED AT ADR-015 (6a), and the change is the finding, not an
    # accommodation. Under M2's "reach" parameterisation a barrier weight
    # underflowed to a HARD ZERO, so `p_barrier < 1e-2` passed for the same
    # reason the gradient was 0. Under "amplitude" the barrier hazard is
    # e^-6 = 0.25% of open ground and its 6 h cumulative probability is ~1.2%,
    # which is what "strongly suppressed but LEARNABLE" has to look like: a
    # barrier that can never be crossed cannot teach a model about crossings.
    # The bound is therefore stated RELATIVE to open ground, which is a claim
    # about suppression, rather than as an absolute that silently rewards
    # underflow.
    # SECOND THRESHOLD CHANGE, 2026-08-08, and again the change IS the finding.
    # `nonburnable < 1e-6` was passing because the value was EXACTLY 0.0: a hard,
    # unlearnable mask produced by `moisture_of_extinction = 1%` collapsing the
    # Simard damping polynomial to 0. sim measured that cells our
    # `fuel_model_id` channel calls non-burnable burn in the GOFER labels 66-84%
    # as often as burnable ones, so a hard mask is a modelling error against our
    # own data. Both suppressions are now stated the same way - RELATIVE to open
    # ground and BOUNDED AWAY FROM ZERO - because an assertion that a probability
    # is small is silently satisfied by an unlearnable zero, which is exactly how
    # this hid through M2 and most of M3.
    ok = (
        0.0 < nonburnable < 0.05 * open_ground
        and 0.0 < barrier < 0.05 * open_ground
        and open_ground > 0.5
    )
    return Check(
        "kernel_respects_barriers_and_nonburnable",
        ok,
        "barrier AND non-burnable each suppressed to <5% of open ground and STRICTLY "
        "POSITIVE — leaky by design, because the leak is the gradient and a zero cannot "
        "be learned away from",
        {
            "max_p_nonburnable": nonburnable,
            "max_p_barrier": barrier,
            "max_p_open": open_ground,
            "nonburnable_relative_to_open": nonburnable / max(open_ground, 1e-12),
            "barrier_relative_to_open": barrier / max(open_ground, 1e-12),
        },
    )


def check_susceptibility_has_gradient() -> Check:
    """ADR-015 (6a): barrier crossing must be LEARNABLE, i.e. have a gradient.

    This is a DIFFERENTIAL known-answer test and both halves are asserted,
    because "the gradient is 0.0103" means nothing without "and it used to be
    0.0". Under M2's ``reach`` parameterisation susceptibility sat inside
    ``exp(-0.5 (d/(gamma s r))^2)``; for a barrier cell that exponent is of order
    -1e4 and ``exp`` of it is a hard zero in float64, so
    ``d loss / d barrier_log_multiplier`` was EXACTLY zero and the P3/G4
    mechanism was structurally absent. Under ``amplitude`` it is the barrier
    indicator, exactly.
    """
    from wildfire_nowcast.model.kernel import susceptibility_gradient_report

    reach = susceptibility_gradient_report("reach")
    amplitude = susceptibility_gradient_report("amplitude")
    ok = (
        reach["d_loss_d_barrier_log_multiplier"] == 0.0
        and reach["nonburnable_gradient_is_zero"]
        and amplitude["d_loss_d_barrier_log_multiplier"] != 0.0
        and amplitude["burnable_gradients_all_nonzero"]
        # NON-BURNABLE, added 2026-08-08. This was EXACTLY 0.0 in BOTH modes until
        # `moisture_damping_floor` landed, by a mechanism independent of ADR-015
        # (6a) - the Simard damping collapsing to a hard zero. It is asserted here
        # rather than merely reported, because "we probed it and it was zero" was
        # true for a whole milestone without anything failing.
        and not amplitude["nonburnable_gradient_is_zero"]
    )
    return Check(
        "susceptibility_has_gradient",
        ok,
        "reach mode: EXACTLY zero for barrier AND non-burnable (the defect). amplitude "
        "mode: non-zero for the barrier, for non-burnable, and for every burnable group",
        {
            "reach_d_barrier": reach["d_loss_d_barrier_log_multiplier"],
            "amplitude_d_barrier": amplitude["d_loss_d_barrier_log_multiplier"],
            "reach_fuel_grads": reach["d_loss_d_fuel_log_multiplier"],
            "amplitude_fuel_grads": amplitude["d_loss_d_fuel_log_multiplier"],
            "reach_nonburnable_is_zero": reach["nonburnable_gradient_is_zero"],
            "amplitude_nonburnable_is_zero": amplitude["nonburnable_gradient_is_zero"],
        },
    )


def check_susceptibility_is_an_exact_log_offset() -> Check:
    """Known answer: ``log w(barrier) - log w(open) == barrier_log_multiplier``.

    Exact to floating point, because in amplitude mode susceptibility is an
    additive term in the log-weight and nothing else in the weight depends on the
    barrier mask. This pins the parameterisation itself rather than a consequence
    of it, so a future refactor that quietly moves susceptibility back inside the
    exponent fails here instead of failing silently three gates later.
    """
    import torch

    from wildfire_nowcast.model.kernel import (
        ContagionKernel,
        KernelConfig,
        static_fields_from_array,
    )

    shape = (6, 6)
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0
    static[static_index("water_barrier_mask")][:, 3] = 1.0
    weather = np.zeros((5, *shape), np.float32)
    weather[weather_index("wind_u10")] = 7.0
    weather[weather_index("fuel_moisture_proxy")] = 5.0

    out = {}
    for mode in ("amplitude", "reach"):
        model = ContagionKernel(KernelConfig(susceptibility_mode=mode))
        fields = static_fields_from_array(static)
        with torch.no_grad():
            log_w = model.log_weights(torch.as_tensor(weather.astype(np.float64)), fields).numpy()
        # Offset 0, column 3 (barrier) vs column 1 (open); same fuel, wind, slope.
        out[mode] = float(log_w[0, 2, 3] - log_w[0, 2, 1])
    expected = float(KernelConfig().barrier_log_multiplier)
    ok = abs(out["amplitude"] - expected) < 1e-12 and out["reach"] < expected * 100
    return Check(
        "susceptibility_is_an_exact_log_offset",
        ok,
        "amplitude mode: the barrier is EXACTLY a -6 log-weight offset. reach mode: it is a "
        "catastrophic one (underflow), which is the defect restated",
        {
            "expected_barrier_log_multiplier": expected,
            "amplitude_log_weight_delta": out["amplitude"],
            "reach_log_weight_delta": out["reach"],
        },
    )


def check_fconf_cannot_move_the_burned_set() -> Check:
    """C1.1's ``ever(t)`` does not read ``cfireLine``, so ``fconf`` is a NULL
    perturbation for a ``burned`` target.

    INTERFACES C1.1 designates the six ``fconf`` levels as "the label-perturbation
    ensemble for the observation-noise augmentation", and ADR-015 (6b) sends the
    S/SW artefact to that ensemble. It cannot get there: the artefact is in the
    PERIMETER and ``fconf`` only splits ``ever(t)`` into states 1 and 2. Asserted
    here on synthetic masks so the claim is checked on every run; measured on
    real fires by
    :func:`wildfire_nowcast.model.labelnoise.fconf_burned_set_invariance`
    (Kincade / Walker / Zogg, all six levels: 0 burned cells differ, while up to
    1,217 state-1 cell-frames do).
    """
    from wildfire_nowcast.common.states import fireline_v2

    rng = np.random.default_rng(7)
    n_t, h, w = 6, 12, 12
    perims = np.zeros((n_t, h, w), bool)
    for t in range(n_t):
        perims[t, 4 : 5 + t, 4 : 5 + t] = True
    burned = []
    state1 = []
    for _ in range(6):  # stand-ins for the six fconf levels
        lines = rng.random((n_t, h, w)) < 0.25
        state = fireline_v2(perims, lines)
        burned.append(state > 0)
        state1.append(state == 1)
    burned_differ = max(int(np.count_nonzero(b != burned[0])) for b in burned)
    state1_differ = max(int(np.count_nonzero(s != state1[0])) for s in state1)
    ok = burned_differ == 0 and state1_differ > 0
    return Check(
        "fconf_cannot_move_the_burned_set",
        ok,
        "six different fire-line masks over identical perimeters: the BURNED set is bitwise "
        "identical, the state-1 set is not. The designated ensemble is a null perturbation "
        "for this milestone's target",
        {"burned_cells_differing": burned_differ, "state1_cells_differing": state1_differ},
    )


def check_label_perturbation_preserves_absorbing_order() -> Check:
    """The label perturbation must not manufacture negative growth.

    ``b0 <= truth_k`` is what makes a window a legal absorbing-fire window. The
    perturbation is applied with ONE draw to both, and both operators (morphology
    and translation) are monotone, so the order survives. If it did not, training
    would see windows where the fire un-burned and the growth moment would be
    fitted against a quantity that cannot occur.
    """
    import torch

    from wildfire_nowcast.model.labelnoise import (
        apply_perturbation,
        growth_band_field,
        noise_model_for,
        sample_perturbation,
    )

    rng = np.random.default_rng(3)
    b0 = torch.zeros(4, 16, 16, dtype=torch.float64)
    b0[:, 6:10, 6:10] = 1.0
    truth = b0.clone().unsqueeze(1).repeat(1, 3, 1, 1)
    truth[:, :, 5:12, 5:12] = 1.0
    model = noise_model_for("2019_kincade")
    violations = 0
    empties = 0
    n = 200
    for _ in range(n):
        p = sample_perturbation(model, rng)
        pb, pt = apply_perturbation(b0, p), apply_perturbation(truth, p)
        if bool((pt < pb.unsqueeze(1) - 1e-12).any()):
            violations += 1
        if float(pb.sum()) == 0.0:
            empties += 1
    band = growth_band_field(b0, 4)
    ok = violations == 0 and bool((band & (b0 > 0.5)).sum() == 0)
    return Check(
        "label_perturbation_preserves_absorbing_order",
        ok,
        "b0 <= truth survives every draw (both operators are monotone); the recomputed band "
        "never includes an already-burned cell",
        {
            "draws": n,
            "monotonicity_violations": violations,
            "draws_that_emptied_b0": empties,
        },
    )


def check_c8_rejects_a_stale_or_unstamped_split() -> Check:
    """C8: a checkpoint from another split, or from no declared split, must FAIL.

    ``assert_split_unchanged`` catches the split moving during a run. This is the
    other half: a model trained under an older split being scored under a newer
    one. Both failure modes are asserted, including the unstamped case - C-1 makes
    "unverifiable" a failure, so an unstamped checkpoint must not be presumed to
    match.
    """
    from wildfire_nowcast.eval.reporting import SplitChangedError, assert_model_split_matches

    class _Stub:
        def __init__(self, provenance):
            self.provenance = provenance

    split = {"fingerprint": "aaaaaaaaaaaaaaaa"}
    results = {}
    for label, model in (
        ("stale", _Stub({"split_fingerprint": "bbbbbbbbbbbbbbbb"})),
        ("unstamped", _Stub({})),
    ):
        try:
            assert_model_split_matches(model, split, name=label)
            results[label] = "ACCEPTED (wrong)"
        except SplitChangedError:
            results[label] = "rejected"
    try:
        ok_match = assert_model_split_matches(
            _Stub({"split_fingerprint": "aaaaaaaaaaaaaaaa"}), split, name="matching"
        )
        results["matching"] = "accepted" if ok_match["match"] else "rejected (wrong)"
    except SplitChangedError:
        results["matching"] = "REJECTED (wrong)"
    ok = (
        results["stale"] == "rejected"
        and results["unstamped"] == "rejected"
        and results["matching"] == "accepted"
    )
    return Check(
        "c8_rejects_a_stale_or_unstamped_split",
        ok,
        "a mismatched fingerprint AND a missing one are both HARD FAILS; a match passes",
        results,
    )


def check_best_member_iou_by_horizon_is_consistent() -> Check:
    """Per-horizon IoU at the last horizon must equal the whole-trajectory IoU.

    ADR-015 (3) adjudicates at 1/2/3 h, so the mode-capture metric has to exist
    at each horizon. The cheap way to get it - one pass, cumulative means - is
    only legitimate if it agrees exactly with what a length-H run would have
    produced at H = L. Asserted rather than assumed, because a per-lead maximum
    (a DIFFERENT and optimistic quantity, also reported) is the easy mistake here
    and it would silently inflate the model at every horizon but the last.
    """
    rng = np.random.default_rng(11)
    shape = (14, 14)
    x0 = np.zeros(shape, np.uint8)
    x0[6:8, 6:8] = 1
    truth = np.repeat(x0[None], 3, axis=0)
    truth[1, 5:9, 5:9] = 1
    truth[2, 4:10, 4:10] = 1
    samples = np.repeat(truth[None], 6, axis=0).copy()
    samples[rng.random(samples.shape) < 0.1] = 1
    res = evaluate(samples, truth, x0=x0)
    by_horizon = res["by_mask"]["domain"]["best_member_iou_by_horizon"]
    whole = res["best_member_iou"]
    by_lead = res["by_mask"]["domain"]["best_member_iou_by_lead"]
    ok = abs(by_horizon[-1] - whole) < 1e-12 and len(by_horizon) == 3
    return Check(
        "best_member_iou_by_horizon_is_consistent",
        ok,
        "cumulative per-horizon best-member IoU equals best_member_iou at H = L, and is a "
        "different (non-optimistic) quantity from the per-lead maximum",
        {
            "by_horizon": by_horizon,
            "best_member_iou": whole,
            "by_lead_max_optimistic": by_lead,
        },
    )


def check_ellipse_respects_barriers() -> Check:
    """No cell ignites in a barrier or in non-burnable fuel."""
    shape = (24, 24)
    x0 = np.zeros(shape, np.uint8)
    x0[10:14, 2:6] = 1
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0  # grass everywhere
    static[static_index("fuel_model_id")][:, 12] = 98.0  # a non-burnable column
    static[static_index("water_barrier_mask")][:, 18] = 1.0  # a barrier column
    static[static_index("canopy_cover")] = 0.0
    weather = np.zeros((6, 5, *shape), np.float32)
    weather[:, weather_index("wind_u10")] = 25.0  # hard east wind
    weather[:, weather_index("fuel_moisture_proxy")] = 2.0
    samples = EllipseBaseline().predict(x0, static, weather, 3, 6, 0)
    burned = samples > 0
    lit_nonburnable = int(burned[:, :, :, 12].sum())
    lit_barrier = int(burned[:, :, :, 18].sum())
    grew = int(burned[0, -1].sum()) > int((x0 > 0).sum())
    ok = lit_nonburnable == 0 and lit_barrier == 0 and grew
    return Check(
        "ellipse_respects_barriers",
        ok,
        "non-burnable fuel and water_barrier_mask are never ignited",
        {
            "nonburnable_cells_lit": lit_nonburnable,
            "barrier_cells_lit": lit_barrier,
            "did_grow": grew,
        },
    )


def check_ellipse_is_anisotropic_downwind() -> Check:
    """A hard east wind must push the fire east, not symmetrically."""
    shape = (31, 31)
    x0 = np.zeros(shape, np.uint8)
    x0[15, 15] = 1
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0
    weather = np.zeros((6, 5, *shape), np.float32)
    weather[:, weather_index("wind_u10")] = 20.0
    weather[:, weather_index("fuel_moisture_proxy")] = 2.0
    final = (
        EllipseBaseline(ros_sigma=0.0, bearing_sigma_deg=0.0).predict(x0, static, weather, 1, 6, 0)[
            0, -1
        ]
        > 0
    )
    east = int(final[:, 16:].sum())
    west = int(final[:, :15].sum())
    lb = float(length_to_breadth(6.0))
    ok = east > 3 * max(west, 1) and lb > 2.0
    return Check(
        "ellipse_is_anisotropic_downwind",
        ok,
        "head runs downwind; length-to-breadth > 1 (Anderson 1983)",
        {
            "cells_east": east,
            "cells_west": west,
            "lb_at_6ms": lb,
            "back_fraction": float(ellipse_ros_factor(lb, -1.0)),
        },
    )


def check_ellipse_ensemble_has_area_spread() -> Check:
    """Shared per-step innovations must produce real scenario spread.

    If members differ only by per-pixel noise, total burned area barely moves.
    The ellipse's ensemble is built from domain-wide shared scalars precisely so
    that it does - otherwise G3 would be judged against a strawman ensemble.
    """
    shape = (31, 31)
    x0 = np.zeros(shape, np.uint8)
    x0[15, 15] = 1
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0
    weather = np.zeros((6, 5, *shape), np.float32)
    weather[:, weather_index("wind_u10")] = 18.0
    weather[:, weather_index("fuel_moisture_proxy")] = 3.0
    samples = EllipseBaseline().predict(x0, static, weather, 16, 6, 0)
    areas = (samples[:, -1] > 0).sum(axis=(1, 2)).astype(float)
    cv = float(areas.std(ddof=1) / max(areas.mean(), 1e-9))
    ok = cv > 0.05 and len(set(areas.tolist())) > 3
    return Check(
        "ellipse_ensemble_has_area_spread",
        ok,
        "shared-innovation ensemble spreads in AREA, not just per pixel",
        {"area_mean": areas.mean(), "area_cv": cv, "distinct_areas": len(set(areas.tolist()))},
    )


def check_ellipse_seeds_from_frontier_not_state_one() -> Check:
    """A fire with NO cell in state 1 must still spread (C1.1 dormancy case).

    C1.1 records state 1 empty in 6-37% of real frames (Kincade: 43 of 134). A
    predictor conditioned on state 1 predicts nothing at all in those frames.
    """
    shape = (25, 25)
    x0 = np.zeros(shape, np.uint8)
    x0[10:15, 10:15] = 2  # burned out; NOTHING in state 1
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0
    weather = np.zeros((4, 5, *shape), np.float32)
    weather[:, weather_index("wind_u10")] = 20.0
    weather[:, weather_index("fuel_moisture_proxy")] = 2.0
    samples = EllipseBaseline().predict(x0, static, weather, 2, 4, 0)
    grew = int((samples[0, -1] > 0).sum()) > int((x0 > 0).sum())
    return Check(
        "ellipse_seeds_from_frontier_not_state_one",
        grew,
        "spreads from the burned frontier with state 1 empty",
        {
            "x0_state1_cells": int((x0 == 1).sum()),
            "growth_cells": int((samples[0, -1] > 0).sum()) - int((x0 > 0).sum()),
        },
    )


def check_forecast_window_time_phase() -> Check:
    """weather[k] must be features[t0+1+k] - the C1.3 end-of-hour phase.

    Off by one here trains every fire an hour out of phase with its weather and
    presents as a mediocre model rather than as a bug (C1.3, verbatim).
    """
    ds, _ = _synthetic_dataset()
    t0, horizon = 7, 3
    window = forecast_inputs(ds, t0, horizon)
    raw_u = zio.get_channel(ds, "wind_u10").values
    state = np.asarray(ds["fire_state"].values)
    phase_ok = all(
        np.allclose(window.weather[k, weather_index("wind_u10")], raw_u[t0 + 1 + k])
        for k in range(horizon)
    )
    truth_ok = np.array_equal(window.truth, state[t0 + 1 : t0 + 1 + horizon])
    x0_ok = np.array_equal(window.x0, state[t0])
    off_by_one = np.allclose(window.weather[0, weather_index("wind_u10")], raw_u[t0])
    return Check(
        "forecast_window_time_phase",
        phase_ok and truth_ok and x0_ok and not off_by_one,
        "weather[k] = features[t0+1+k]; truth[k] = fire_state[t0+1+k]",
        {
            "phase_ok": phase_ok,
            "truth_ok": truth_ok,
            "x0_ok": x0_ok,
            "would_be_off_by_one": bool(off_by_one),
        },
    )


def check_end_to_end_on_synthetic() -> Check:
    """Both baselines run through predict() and score through evaluate()."""
    ds, _ = _synthetic_dataset()
    window = forecast_inputs(ds, 10, 3, fire_id="synthetic_0000")
    out = {}
    for name in ("persistence", "ellipse"):
        samples = predict(*window.predict_args(), 8, 3, 0, model=name)
        res = evaluate(samples, window.truth, x0=window.x0)
        out[name] = {
            "brier_3h": res["brier_3h"],
            "band_brier_3h": res["by_mask"]["growth_band"]["brier_by_lead"].get(3),
            "best_member_iou": res["best_member_iou"],
        }
    ok = all(v["brier_3h"] is not None for v in out.values())
    return Check("end_to_end_on_synthetic", ok, "C5 -> C6 round trip", out)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def check_latent_off_reproduces_the_g2_kernel_bitwise() -> Check:
    """[M5] Adding ``z_t`` must not have moved the deterministic path by one ULP.

    The G2 record was produced by the latent-free forward pass. If an optional
    argument changed it at all, every archived number silently stops being
    reproducible by this code and nothing would say so. Two deltas must be
    EXACTLY 0 (no latent; latent held at ``z = 0``), and a third must NOT be -
    a latent wired to nothing would pass both identity checks and yield a
    collapsed ensemble that looks like a model.
    """
    from wildfire_nowcast.model.kernel import check_latent_off_is_bit_identical

    got = check_latent_off_is_bit_identical()
    ok = (
        got["delta_latents_none"] == 0.0
        and got["delta_z_zero_ablation"] == 0.0
        and got["delta_z_drawn"] > 0.0
    )
    return Check(
        "latent_off_reproduces_the_g2_kernel_bitwise",
        ok,
        "latent-free and z=0 paths are bit-identical to the pre-M5 kernel, and a DRAWN "
        "z_t is not — so the latent is both optional and actually connected",
        got,
    )


def check_shared_latent_is_constant_across_pixels() -> Check:
    """[M5] ONE draw per member per step, shared by every pixel. The defining property.

    CLAUDE.md: pixels are conditionally independent Bernoulli ONLY given a shared
    per-step latent. This asserts the "shared" half directly on the probability
    field rather than trusting the sampler's shape arithmetic: with the offset
    weights and susceptibility held uniform and the wind uniform, two members'
    probability fields under different ``z`` must differ by a CONSTANT RATIO in
    the hazard, because the only thing separating them is a global multiplier.

    This is the check that would have caught the real defect found in this
    milestone: a per-member latent broadcast against the wrong axis silently
    aligned 24 members with 24 grid ROWS, which produces plausible numbers.
    """
    import numpy as np
    import torch

    from wildfire_nowcast.model.kernel import (
        DTYPE,
        ContagionKernel,
        KernelConfig,
        static_fields_from_array,
    )
    from wildfire_nowcast.model.latent import LatentConfig, LatentSampler

    x0, static, weather = _toy_scene(wind_u=8.0)
    model = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=1))
    fields = static_fields_from_array(static)
    w0 = torch.as_tensor(weather[0].astype(np.float64), dtype=DTYPE)
    gen = torch.Generator().manual_seed(3)
    effect = LatentSampler("latent").draw(model.latent, (4,), generator=gen)
    log_w = model.log_weights(w0, fields, effect)  # [4, K, H, W]
    # Every member's log-weight field differs from member 0's by a scalar that is
    # constant over BOTH the offset axis and the whole grid.
    delta = (log_w - log_w[0:1]).reshape(4, -1)
    spread_within_member = float(delta.std(dim=1).max().detach())
    between_members = float(delta.mean(dim=1).abs().max().detach())
    ok = spread_within_member < 1e-12 and between_members > 1e-6
    return Check(
        "shared_latent_is_constant_across_pixels",
        ok,
        "one z_t per member shifts log-weights by a CONSTANT over every offset and every "
        "cell (max within-member SD ~0), while members differ from each other",
        {
            "max_within_member_sd": spread_within_member,
            "max_between_member_shift": between_members,
        },
    )


#: [M21] The bar, the horizon, the member count and the seed of the ablation
#: SD-ratio check below, hoisted out of the call so they can be READ instead of
#: restated. ADR-114 (d) requires every collapse verdict to publish its horizon,
#: and this one decided a gate-path check for weeks without recording it: at
#: this scene and seed the ratio reads 0.922, 1.410, 1.460, 1.488, 2.690, 6.517
#: at leads 1 to 6, so it clears the shipped bar at lead 6 ALONE, and cleared
#: the retired one at leads 5 and 6. One integer in this constant is the
#: difference between green and red.
#:
#: **THE BAR WAS NOT DERIVED HERE UNTIL M22.** The docstring below derives the
#: DIRECTION (independent noise averages out, so the quotient must exceed 1 and
#: grow with the fire) and no magnitude at all. 1.5 was ALSO the value of
#: ``sim/ensemble.py``'s ``COLLAPSE_INDEX_THRESHOLD``, which is a bar on a
#: DIFFERENT quantity - an empirical spread over a THEORETICAL one, whose null
#: is 1.0 by algebra - and this one is a quotient of two EMPIRICAL spreads. The
#: two literals were adopted independently for different estimands (ADR-119 (4));
#: only this one has moved, and nothing about the other follows from that.
#: :mod:`wildfire_nowcast.eval.collapse_bars` measured what 1.5 was worth here:
#: about the 99th percentile of this instrument's own null at leads 1 to 4 and
#: about the 72nd at lead 6, which is the lead the verdict is taken at.
#:
#: **[M22] THE BAR MOVED, 1.5 -> 5.0, AND THE HORIZON DID NOT. Both halves are
#: arguments and both are measured.** The bar was derived from this
#: instrument's own null at the lead it runs at, under the rule the check
#: actually applies (a zero denominator is not a pass), over four independent
#: 200-seed blocks of the more adversarial ``null_no_latent`` family: the null's
#: 99th percentile among measurable draws reads 4.965 / 2.888 / 3.648 / 3.973,
#: so 5.0 is the smallest round value at or above every estimate, and the false
#: fire rate it buys is 1.00% / 0.00% / 0.00% / 0.75% against the 24-28% that
#: 1.5 bought. It costs almost nothing: over 400 replications the TREATMENT
#: clears 5.0 in 97.0% of draws against 98.8% at 1.5, and at the shipped seed
#: the ratio is 6.517.
#:
#: **THE HORIZON STAYS AT 6, and the case against moving it is the instrument's
#: POWER, not its convenience.** Relocating to lead 3 - the longest lead G3's
#: 1-3 h wording covers - does not repair anything: at lead 3 the treatment
#: clears its own null's 99th percentile in 64% of draws, so a single-draw
#: verdict there would trade a 27% false FIRE rate for a ~36% false SILENCE
#: rate, and at the shipped seed it reads 1.460, i.e. red for want of power
#: rather than for want of a latent. At lead 1 the instrument has no power at
#: all: 3.2% of TREATMENT draws clear 1.5 against 0.0% of null draws. So
#: ADR-114 (b)'s three verdict calls at k=1,2,3 are not executable on THIS
#: instrument, and that is a fact about the estimand: the index's null is exact
#: at one step and this quotient's power is absent there. The two collapse
#: instruments are complementary in the horizon, not substitutes (ADR-119 (4)).
ABLATION_SD_RATIO_BAR: Final[float] = 5.0
ABLATION_SD_RATIO_HORIZON_H: Final[int] = 6
ABLATION_SD_RATIO_MEMBERS: Final[int] = 32
ABLATION_SD_RATIO_SEED: Final[int] = 4

#: The bar this check shipped with until M22. **NOT a bar**: nothing adjudicates
#: on it and nothing may. It is retained as the fixed reference at which the
#: horizon dependence of the null's TAIL was discovered and is still measured,
#: so that finding survives its own repair - a control asserted against the
#: current bar would go quiet the moment the current bar was right.
ABLATION_SD_RATIO_BAR_RETIRED: Final[float] = 1.5

#: The most no-latent draws that may clear :data:`ABLATION_SD_RATIO_BAR` at the
#: verdict lead before the verdict itself is refused. 2% is one resolution step
#: above what the bar was derived to buy (0-1% over four independent blocks) and
#: an order of magnitude below what the retired bar bought (24-28%).
ABLATION_NULL_FALSE_FIRE_CEILING: Final[float] = 0.02

#: Replications behind the null tail published beside every verdict. 200 seeds
#: resolve a rate to half a percentage point, which is the resolution the
#: ceiling above is stated at; the module CLI takes more.
ABLATION_NULL_TAIL_SEEDS: Final[int] = 200


def check_independent_noise_ablation_collapses_in_area() -> Check:
    """[M5] G3 (d): the independent-per-pixel ablation must DEMONSTRATE collapse.

    A POSITIVE CONTROL for the ensemble machinery, not a result about the model -
    G3 asks for the ablation to fail, so a failure to fail is a defect in this
    repo's sampler and must surface here rather than in a gate table.

    Both arms are the SAME PARAMETERS (``with_sampler`` shares them); the only
    difference is whether ``z_t`` is drawn. The quantity is the spread in NEW
    burned cells: independent Bernoulli noise over ``N`` cells gives area spread
    ``O(sqrt(N))`` against a mean ``O(N)``, so it must shrink relative to a
    shared-innovation ensemble as the fire gets bigger.

    [M21] **THE HORIZON IS PART OF THE VERDICT AND IS NOW IN THE RECORD.** The
    argument above is about a quotient of two spreads, and unlike
    ``independence_dispersion_index`` its NULL IS 1.0 AT EVERY LEAD: multi-step
    contagion inflates both arms and cancels in the quotient, which is measured
    at medians 1.000, 0.972, 0.991, 0.992, 1.008, 0.998 with the latent off on
    both arms. What is NOT lead-invariant is the tail, because the ablation
    arm's area SD is the DENOMINATOR and it falls as the fire decelerates
    (0.843, 1.254, 1.419, 1.551, 1.139, 0.486 on this scene). Over 400 no-latent
    replications the fraction clearing 1.5 runs 0.0%, 1.5%, 2.5%, 0.8%, 5.2%,
    **28.2%** at leads 1 to 6. This check runs at lead 6. Every lead is emitted
    below so a reader can see which one decided.

    [M22] **THE NULL TAIL IS PUBLISHED BESIDE THE VERDICT AND IS MEASURED IN
    THE SAME INVOCATION** (ADR-114 (c), ADR-120 (5)). A collapse verdict whose
    bar's false fire rate is unknown is not a verdict, so the rate at the lead
    the verdict is taken at is read here, from the more adversarial
    ``null_no_latent`` family, and the verdict is REFUSED when it exceeds
    :data:`ABLATION_NULL_FALSE_FIRE_CEILING`. The whole per-lead tail is emitted
    beside it, at the shipped bar and at the retired one, because the finding
    that made this repair necessary is a statement about the horizon and would
    be invisible in a single number.

    **A ZERO DENOMINATOR IS NO LONGER A PASS.** ``max(sd, 1e-9)`` turned an
    ablation ensemble with no measurable spread into a ratio of order 1e9, which
    read as the most emphatic collapse the instrument can report. It fired that
    way in 5 of those 400 no-latent replications, all at lead 6, and it is not a
    pass in disguise: **seeds 97, 102, 233, 309 and 358 are degenerate under the
    TREATMENT and under the NULL alike**, because a zero denominator is a
    property of 32 members on a fire whose last step adds 0.68 cells, not
    evidence about ``z_t``. It cannot fire at the shipped seed, where the
    ablation SD is 0.803, so this is a latent case made loud rather than an
    observed failure repaired.
    """
    from wildfire_nowcast.eval.collapse_bars import null_tail_by_lead
    from wildfire_nowcast.model.api import (
        ABLATION_ARM_MODE,
        ablation_arm,
        assert_ablation_arm_is_demonstrative,
    )
    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
    from wildfire_nowcast.model.latent import LatentConfig

    x0, static, weather = _toy_scene(wind_u=12.0)
    model = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=3))
    # The pair must be the same fit AND the ablation must remove something.
    # `ablation_arm` refuses an arm whose parameters are not this model's own
    # objects; the assert above it refuses a base with no latent, whose arm
    # would be the same forecast under another name and would read 1.0 by
    # construction. Neither can fire at this scene, which is why each has a
    # check of its own below that plants the case it exists for.
    assert_ablation_arm_is_demonstrative(model)
    arms = {"latent": model, ABLATION_ARM_MODE: ablation_arm(model)}
    samples = {
        mode: arm.predict(
            x0,
            static,
            weather,
            ABLATION_SD_RATIO_MEMBERS,
            ABLATION_SD_RATIO_HORIZON_H,
            ABLATION_SD_RATIO_SEED,
        )
        for mode, arm in arms.items()
    }

    def _new_cells(arr: np.ndarray, lead: int) -> np.ndarray:
        return ((arr[:, lead] > 0) & (x0[None] == 0)).sum(axis=(1, 2)).astype(float)

    by_lead = {}
    for lead in range(ABLATION_SD_RATIO_HORIZON_H):
        sd_latent = float(_new_cells(samples["latent"], lead).std(ddof=1))
        sd_independent = float(_new_cells(samples["independent"], lead).std(ddof=1))
        by_lead[lead + 1] = {
            "sd_latent": sd_latent,
            "sd_independent": sd_independent,
            "sd_ratio": sd_latent / max(sd_independent, 1e-9),
            "denominator_measurable": sd_independent > 0.0,
        }

    out = {
        mode: {
            "mean": float(_new_cells(arr, ABLATION_SD_RATIO_HORIZON_H - 1).mean()),
            "sd": float(_new_cells(arr, ABLATION_SD_RATIO_HORIZON_H - 1).std(ddof=1)),
        }
        for mode, arr in samples.items()
    }
    verdict = by_lead[ABLATION_SD_RATIO_HORIZON_H]
    ratio = float(verdict["sd_ratio"])
    measurable = bool(verdict["denominator_measurable"])

    tail = null_tail_by_lead(
        bar=ABLATION_SD_RATIO_BAR,
        horizon_h=ABLATION_SD_RATIO_HORIZON_H,
        n_seeds=ABLATION_NULL_TAIL_SEEDS,
        n_members=ABLATION_SD_RATIO_MEMBERS,
    )
    retired_tail = null_tail_by_lead(
        bar=ABLATION_SD_RATIO_BAR_RETIRED,
        horizon_h=ABLATION_SD_RATIO_HORIZON_H,
        n_seeds=ABLATION_NULL_TAIL_SEEDS,
        n_members=ABLATION_SD_RATIO_MEMBERS,
    )
    null_rate = tail[ABLATION_SD_RATIO_HORIZON_H]
    calibrated = null_rate <= ABLATION_NULL_FALSE_FIRE_CEILING
    ok = measurable and calibrated and ratio > ABLATION_SD_RATIO_BAR
    return Check(
        "independent_noise_ablation_collapses_in_area",
        ok,
        "holding z_t at the prior mean removes most of the ensemble's AREA spread "
        f"(SD ratio latent/independent = {ratio:.2f}, must exceed "
        f"{ABLATION_SD_RATIO_BAR:g}, AT LEAD {ABLATION_SD_RATIO_HORIZON_H} h over "
        f"{ABLATION_SD_RATIO_MEMBERS} members; that bar is cleared by "
        f"{100 * null_rate:.1f}% of no-latent draws at this lead, against "
        f"{100 * retired_tail[ABLATION_SD_RATIO_HORIZON_H]:.1f}% for the retired "
        f"{ABLATION_SD_RATIO_BAR_RETIRED:g}"
        + (
            ""
            if measurable
            else "; the ablation arm has NO measurable spread, so the "
            "quotient has no value and this is not a demonstration of anything"
        )
        + (
            ""
            if calibrated
            else "; the bar's false fire rate at this lead exceeds "
            f"{100 * ABLATION_NULL_FALSE_FIRE_CEILING:.0f}%, so there is no verdict to take"
        )
        + ")",
        {
            **out,
            "sd_ratio": ratio,
            "bar": ABLATION_SD_RATIO_BAR,
            "horizon_h": ABLATION_SD_RATIO_HORIZON_H,
            "n_members": ABLATION_SD_RATIO_MEMBERS,
            "seed": ABLATION_SD_RATIO_SEED,
            "denominator_measurable": measurable,
            "sd_ratio_by_lead": {k: v["sd_ratio"] for k, v in by_lead.items()},
            "by_lead": by_lead,
            "arms": {mode: arm.name for mode, arm in arms.items()},
            "null_false_fire_rate_at_verdict_lead": null_rate,
            "null_false_fire_ceiling": ABLATION_NULL_FALSE_FIRE_CEILING,
            "null_false_fire_rate_by_lead": tail,
            "null_false_fire_rate_by_lead_at_retired_bar": retired_tail,
            "retired_bar": ABLATION_SD_RATIO_BAR_RETIRED,
            "null_tail_seeds": ABLATION_NULL_TAIL_SEEDS,
            "null_tail_family": "null_no_latent",
        },
    )


def check_the_ablation_arm_is_loadable_by_name() -> Check:
    """[M22] G3 (d)'s arm must be obtainable the way a C5 consumer obtains a model.

    ADR-119 (2) measured the gap exactly: ``with_sampler("independent")`` worked
    in Python and ``load_model("contagion_kernel__independent")`` raised, so
    every consumer that resolves a predictor BY NAME - which is every C5-shaped
    consumer, ``sim/`` included - could not reach the ablation and fell back to
    whatever fixture it had. The arm was correctly built and unreachable, and
    the two packages disagreed about what they were measuring for that one
    reason.

    Four things, and the third is the one that matters:

    1. the arm's address is in :func:`available_models` beside its base;
    2. it resolves, from the registry name AND from a saved checkpoint;
    3. **it is the SAME FIT.** Within one resolution the arm's parameters are
       the model's own objects, not copies, so nothing about the comparison can
       be attributed to the parameters. Across two resolutions of one checkpoint
       the values agree bitwise, which is the strongest statement a name can
       carry;
    4. the address and the label are the same string, so a name written into an
       artifact is a name that loads.
    """
    from pathlib import Path as _Path

    from wildfire_nowcast.model.api import (
        ABLATION_ARM_SUFFIX,
        ablation_arm,
        available_models,
        load_model,
        save_model,
    )
    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
    from wildfire_nowcast.model.latent import LatentConfig

    listed = "contagion_kernel" + ABLATION_ARM_SUFFIX in available_models()
    by_name = load_model("contagion_kernel" + ABLATION_ARM_SUFFIX)
    base = load_model("contagion_kernel")

    live = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=3))
    arm = ablation_arm(live)
    shared = all(
        p is q
        for (_, p), (_, q) in zip(
            sorted(live.named_parameters()), sorted(arm.named_parameters()), strict=True
        )
    )

    out = _Path(_process_scratch("wnc-arm-")) / "ckpt"
    save_model(live, out)
    from_ckpt = load_model(str(out) + ABLATION_ARM_SUFFIX)
    reloaded = load_model(str(out))
    drift = max(
        float(np.max(np.abs(a.detach().numpy() - b.detach().numpy())))
        for (_, a), (_, b) in zip(
            sorted(from_ckpt.named_parameters()), sorted(reloaded.named_parameters()), strict=True
        )
    )
    ckpt_shared = all(
        p is q
        for (_, p), (_, q) in zip(
            sorted(from_ckpt.named_parameters()),
            sorted(load_model(str(out) + ABLATION_ARM_SUFFIX).named_parameters()),
            strict=True,
        )
    )

    ok = (
        listed
        and by_name.sampler.is_ablation
        and not base.sampler.is_ablation
        and by_name.name == "contagion_kernel" + ABLATION_ARM_SUFFIX
        and base.name == "contagion_kernel"
        and shared
        and from_ckpt.sampler.is_ablation
        and not reloaded.sampler.is_ablation
        and drift == 0.0
    )
    return Check(
        "the_ablation_arm_is_loadable_by_name",
        ok,
        "load_model resolves the latent-off ablation arm from a registry name and from a "
        "checkpoint, the arm shares the model's parameter OBJECTS within a resolution and "
        "matches it bitwise across resolutions, and the address it loads by is the name it "
        "reports back",
        {
            "available_models": available_models(),
            "arm_name": by_name.name,
            "base_name": base.name,
            "arm_is_ablation": by_name.sampler.is_ablation,
            "base_is_ablation": base.sampler.is_ablation,
            "shares_parameter_objects": shared,
            "checkpoint_arm_shares_parameter_objects": ckpt_shared,
            "checkpoint_parameter_max_abs_difference": drift,
        },
    )


def check_a_look_alike_ablation_is_refused_and_a_vacuous_one_cannot_be_scored() -> Check:
    """[M22] The two ways an ablation arm can be wrong, each with its own plant.

    Both guards are unreachable at the shipped configuration, which is exactly
    why they need a check that MAKES them fire rather than a comment saying they
    would.

    **(1) A LOOK-ALIKE.** An arm whose parameters are equal but not identical is
    a second model, and every difference between the two ensembles would then be
    attributable to the sampler OR to the parameters. The plant is a predictor
    whose ``with_sampler`` returns a freshly constructed kernel - the exact
    shortcut a registry entry invites - and it is refused even though its
    numbers would have looked right.

    **(2) A VACUOUS ONE.** A model with no shared latent has an ablation that is
    the same forecast under another name: the quotient is 1.0 by construction
    and says nothing about ``z_t``. The plant is the deterministic G2 kernel,
    and the refusal is what stops "the ablation failed to collapse" from being
    read off a model that never had anything to remove.
    """
    import torch

    from wildfire_nowcast.model.api import (
        ablation_arm,
        assert_ablation_arm_is_demonstrative,
        load_model,
    )
    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
    from wildfire_nowcast.model.latent import LatentConfig

    latent_model = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=3))

    class _LookAlike(ContagionKernel):
        """The shortcut: an arm built by CONSTRUCTION instead of by view."""

        def with_sampler(self, mode: str) -> ContagionKernel:
            from wildfire_nowcast.model.latent import LatentSampler

            twin = ContagionKernel(
                KernelConfig(), latent_config=LatentConfig(dim=3), sampler=LatentSampler(mode)
            )
            with torch.no_grad():
                for (_, dst), (_, src) in zip(
                    sorted(twin.named_parameters()),
                    sorted(self.named_parameters()),
                    strict=True,
                ):
                    dst.copy_(src)
            return twin

    look_alike = _LookAlike(KernelConfig(), latent_config=LatentConfig(dim=3))
    values: dict[str, Any] = {}
    try:
        ablation_arm(look_alike)
        values["look_alike"] = "ACCEPTED (wrong)"
    except RuntimeError as exc:
        values["look_alike"] = "refused"
        values["look_alike_reason"] = str(exc)[:120]

    equal = all(
        bool(torch.equal(a.detach(), b.detach()))
        for (_, a), (_, b) in zip(
            sorted(look_alike.named_parameters()),
            sorted(look_alike.with_sampler("independent").named_parameters()),
            strict=True,
        )
    )
    values["look_alike_parameters_were_EQUAL"] = equal

    deterministic = load_model("contagion_kernel")
    try:
        assert_ablation_arm_is_demonstrative(deterministic)
        values["vacuous"] = "ACCEPTED (wrong)"
    except ValueError:
        values["vacuous"] = "refused"

    try:
        assert_ablation_arm_is_demonstrative(latent_model)
        values["control_latent_model"] = "accepted"
    except ValueError:
        values["control_latent_model"] = "REFUSED (wrong)"

    ok = (
        values["look_alike"] == "refused"
        and equal
        and values["vacuous"] == "refused"
        and values["control_latent_model"] == "accepted"
    )
    return Check(
        "a_look_alike_ablation_is_refused_and_a_vacuous_one_cannot_be_scored",
        ok,
        "an arm built by construction rather than by view is refused even though its "
        "parameters are EQUAL, an ablation of a model with no latent cannot be scored, and a "
        "real latent model passes both",
        values,
    )


#: The behavioural half of the reference-fit check is taken at ONE scene and ONE
#: seed, and they are named here so that the reading is reproducible rather than
#: incidental. The member count, horizon and seed are S14's (ADR-124), so a
#: reader comparing this synthetic reading with that held-out one is comparing
#: the same draw shape. ``wind_u`` is 12 m/s because the fit grows more there
#: than at 20 (measured: 83.9 new cells against 36.0), which is the deceleration
#: defect ADR-043 named and is not what this check is about.
REFERENCE_FIT_SCENE_WIND_U: Final[float] = 12.0
REFERENCE_FIT_MEMBERS: Final[int] = 24
REFERENCE_FIT_HORIZON_H: Final[int] = 3
REFERENCE_FIT_SEED: Final[int] = 0


def check_the_tracked_reference_fit_is_a_latent_bearing_c5_address() -> Check:
    """[M23] A CLONE CAN NOW REACH A SUBJECT G3 (d) MAY BE ASKED OF.

    ADR-124 stated G3 (d) about this project's model for the first time and
    carried two caveats. The second one is this: the verdict rested on
    ``runs/s1_arma_s1_f3-20260821-180258__independent``, ``runs/`` is untracked,
    and the registry model has no latent at all, so **the set of C5 addresses a
    cloner could name and take a collapse verdict on was empty.** A tracked copy
    of that fit's ``model.json`` closes it, and nothing in C5 changed to allow
    that: ``load_model`` already accepted a directory holding a spec, and the
    ``__independent`` suffix already derived the arm from whatever it resolved.

    THE POSITIVE AND THE NEGATIVE READING ARE TAKEN IN THE SAME INVOCATION, and
    that is what makes this check unable to pass vacuously. The tracked fit must
    be ACCEPTED by ``assert_ablation_arm_is_demonstrative`` and the registry
    model must be REFUSED by it. If that predicate were ever weakened to always
    accept, the control below fails; if it were weakened to always refuse, the
    subject fails. One reading alone would be satisfied by a broken predicate.

    THE ARMS ARE ALSO SEPARATED BEHAVIOURALLY, because a rule and a fact are two
    detections rather than one (ADR-124 (4)): the two arms are drawn at one seed
    on a synthetic scene and no member matches. That reading would catch a
    latent that exists in the configuration and does nothing in the sampler,
    which the configuration reading cannot see.

    WHAT THIS DOES NOT ESTABLISH. Nothing here is a held-out number. The scene
    is synthetic, and ADR-124's verdict is scored on five held-out fires whose
    tensors are untracked, so a clone reaches the ADDRESS and not the VERDICT.
    The interpreter's own tree is emitted beside the readings (ADR-122): an
    editable install can point a clone's interpreter at the shared tree, and
    then this check would be about a file the clone does not contain.
    """
    import hashlib
    from pathlib import Path

    import wildfire_nowcast
    from wildfire_nowcast.model.api import (
        ABLATION_ARM_SUFFIX,
        ablation_arm_is_demonstrative,
        assert_ablation_arm_is_demonstrative,
        available_models,
        load_model,
    )
    from wildfire_nowcast.model.reference import (
        REFERENCE_FIT_ADDRESS,
        REFERENCE_FIT_ARM_ADDRESS,
        REFERENCE_FIT_BYTES,
        REFERENCE_FIT_DIR,
        REFERENCE_FIT_N_PARAMETERS,
        REFERENCE_FIT_PROVENANCE,
        REFERENCE_FIT_SHA256,
        REFERENCE_FIT_SPEC,
        load_reference_fit,
        reference_fit_pair,
        reference_fit_sha256,
    )

    present = REFERENCE_FIT_SPEC.is_file()
    digest = reference_fit_sha256() if present else ""
    n_bytes = REFERENCE_FIT_SPEC.stat().st_size if present else -1
    spec = json.loads(REFERENCE_FIT_SPEC.read_text()) if present else {}
    n_parameters = sum(int(np.asarray(v).size) for v in dict(spec.get("parameters", {})).values())
    provenance = dict(spec.get("provenance", {}))
    stated = {
        "split_fingerprint": provenance.get("split_fingerprint"),
        "train_folds": tuple(provenance.get("train_folds") or ()),
        "n_train_fires": len(provenance.get("train_fire_ids") or ()),
        "n_heldout_blocks": provenance.get("n_heldout_blocks"),
        "trained_utc": provenance.get("trained_utc"),
        "archived_run_directory": REFERENCE_FIT_PROVENANCE["archived_run_directory"],
    }
    provenance_matches = stated == dict(REFERENCE_FIT_PROVENANCE)

    model, arm = reference_fit_pair()
    by_address = load_model(str(REFERENCE_FIT_DIR) + ABLATION_ARM_SUFFIX)
    shares_objects = all(
        p is q
        for (_, p), (_, q) in zip(
            sorted(model.named_parameters()), sorted(arm.named_parameters()), strict=True
        )
    )
    # THE TRAP, MEASURED RATHER THAN DESCRIBED, because the first draft of this
    # check fell into it. A SECOND load of the same address restores the same
    # values into NEW objects, so a sharing test spanning two loads answers
    # False by construction and says nothing about the arm. Both readings are
    # asserted: equal across loads, and not shared across loads.
    second_load = load_reference_fit()
    across_loads_shared = any(
        p is q
        for (_, p), (_, q) in zip(
            sorted(model.named_parameters()),
            sorted(second_load.named_parameters()),
            strict=True,
        )
    )
    across_loads_equal = all(
        float(np.max(np.abs(p.detach().numpy() - q.detach().numpy()))) == 0.0
        for (_, p), (_, q) in zip(
            sorted(model.named_parameters()),
            sorted(second_load.named_parameters()),
            strict=True,
        )
    )

    # THE SUBJECT is accepted and THE REGISTRY MODEL is refused, both here.
    try:
        assert_ablation_arm_is_demonstrative(model)
        subject_reading = "accepted"
    except ValueError:
        subject_reading = "REFUSED (wrong)"
    registry_model = load_model("contagion_kernel")
    try:
        assert_ablation_arm_is_demonstrative(registry_model)
        control_reading = "ACCEPTED (wrong)"
    except ValueError:
        control_reading = "refused"

    # The behavioural reading, on the same fixed scene the ablation check uses.
    x0, static, weather = _toy_scene(wind_u=REFERENCE_FIT_SCENE_WIND_U)
    draw = (
        x0,
        static,
        weather,
        REFERENCE_FIT_MEMBERS,
        REFERENCE_FIT_HORIZON_H,
        REFERENCE_FIT_SEED,
    )
    latent_samples = model.predict(*draw)
    ablated_samples = arm.predict(*draw)
    identical_members = int(
        sum(
            bool(np.array_equal(latent_samples[m], ablated_samples[m]))
            for m in range(REFERENCE_FIT_MEMBERS)
        )
    )

    def _new_cells(arr: np.ndarray) -> np.ndarray:
        return ((arr[:, -1] > 0) & (x0[None] == 0)).sum(axis=(1, 2)).astype(float)

    sd_latent = float(_new_cells(latent_samples).std(ddof=1))
    sd_ablated = float(_new_cells(ablated_samples).std(ddof=1))

    listed = available_models()
    added_no_registry_entry = not any("reference" in name for name in listed)
    relative_resolves = (
        Path(REFERENCE_FIT_ADDRESS, "model.json").is_file()
        and hashlib.sha256(Path(REFERENCE_FIT_ADDRESS, "model.json").read_bytes()).hexdigest()
        == digest
    )

    ok = (
        present
        and digest == REFERENCE_FIT_SHA256
        and n_bytes == REFERENCE_FIT_BYTES
        and n_parameters == REFERENCE_FIT_N_PARAMETERS
        and provenance_matches
        and ablation_arm_is_demonstrative(model)
        and subject_reading == "accepted"
        and control_reading == "refused"
        and arm.sampler.is_ablation
        and not model.sampler.is_ablation
        and by_address.sampler.is_ablation
        and shares_objects
        and across_loads_equal
        and not across_loads_shared
        and identical_members == 0
        and sd_latent > sd_ablated
        and added_no_registry_entry
    )
    return Check(
        "the_tracked_reference_fit_is_a_latent_bearing_c5_address",
        ok,
        f"{REFERENCE_FIT_ADDRESS} is a tracked, latent-bearing C5 address: the shipped "
        "demonstrative check ACCEPTS it and REFUSES the registry model in this same "
        "invocation, its arm shares the fit's own parameter objects, and the two arms draw "
        f"{REFERENCE_FIT_MEMBERS - identical_members} of {REFERENCE_FIT_MEMBERS} differing "
        "members on a synthetic scene. It is one archived fit, not the model, and no "
        "held-out number is measured here",
        {
            "address": REFERENCE_FIT_ADDRESS,
            "arm_address": REFERENCE_FIT_ARM_ADDRESS,
            "resolved_dir": str(REFERENCE_FIT_DIR),
            "interpreter_package": wildfire_nowcast.__file__,
            "present": present,
            "sha256": digest,
            "sha256_pin": REFERENCE_FIT_SHA256,
            "bytes": n_bytes,
            "n_parameters": n_parameters,
            "provenance": stated,
            "provenance_matches_pin": provenance_matches,
            "subject_demonstrative_reading": subject_reading,
            "control_registry_model_reading": control_reading,
            "arm_shares_parameter_objects": shares_objects,
            "two_loads_share_parameter_objects": across_loads_shared,
            "two_loads_are_bitwise_equal": across_loads_equal,
            "identical_members": identical_members,
            "n_members": REFERENCE_FIT_MEMBERS,
            "sd_new_cells_latent": sd_latent,
            "sd_new_cells_ablated": sd_ablated,
            "available_models": listed,
            "repo_relative_address_resolves": relative_resolves,
            "scored_on_held_out_data": False,
        },
    )


def check_area_dispersion_by_horizon_recombines_to_the_pooled_criterion() -> Check:
    """[M22] G3's dispersion half now HAS a per-lead form, and it is exact.

    ADR-114 (b) rules that a 1-3 h statement is three verdict-bearing calls;
    ADR-120 (1) recorded that the rule could not be executed for this half of G3
    because ``area_dispersion_ratio`` had no ``_by_horizon`` sibling. The sibling
    is only worth having if it is the SAME quantity split, so the identity is
    asserted here at both levels the criterion is read at:

        area_dispersion_ratio == sqrt(factor * sum_h V[h] / sum_h E[h])

    per window and pooled over windows. An approximate decomposition would let a
    per-lead reading and the pooled reading disagree about a gate, which is
    worse than having no decomposition at all.
    """
    rng = np.random.default_rng(19)
    shape = (12, 12)
    x0 = np.zeros(shape, np.uint8)
    x0[5:7, 5:7] = 1
    truth = np.repeat(x0[None], 3, axis=0)
    truth[1, 4:8, 4:8] = 1
    truth[2, 3:9, 3:9] = 1
    windows = []
    for k in range(3):
        samples = np.repeat(truth[None], 8, axis=0).copy()
        samples[rng.random(samples.shape) < 0.05 * (k + 1)] = 1
        samples = np.maximum.accumulate(samples, axis=1)
        windows.append(evaluate(samples, truth, x0=x0))

    single = windows[0]["by_mask"]["domain"]
    pooled = aggregate(windows)["by_mask"]["domain"]

    # The identity is checked from the SUMS, not from the ratios, because the
    # ratio is a square root of a quotient and recombining ratios would test
    # arithmetic this module invented rather than the statistic the gate reads.
    def _from_sums(blocks: Sequence[Mapping[str, Any]], n_members: int) -> float | None:
        factor = (n_members + 1.0) / n_members
        var = float(sum(sum(b["area_dispersion"]["sum_var_by_horizon"]) for b in blocks))
        err = float(sum(sum(b["area_dispersion"]["sum_sq_err_by_horizon"]) for b in blocks))
        return None if err <= 0.0 else float(np.sqrt(var * factor / err))

    pools = [w["_pool"]["by_mask"]["domain"] for w in windows]
    single_expected = _from_sums(pools[:1], 8)
    pooled_expected = _from_sums(pools, 8)
    single_error = abs(float(single["area_dispersion_ratio"]) - float(single_expected or 0.0))
    pooled_error = abs(float(pooled["area_dispersion_ratio"]) - float(pooled_expected or 0.0))

    by_h_single = single["area_dispersion_ratio_by_horizon"]
    by_h_pooled = pooled["area_dispersion_ratio_by_horizon"]
    absent = aggregate(
        [
            {
                **w,
                "_pool": {
                    **w["_pool"],
                    "by_mask": {
                        m: {
                            **blk,
                            "area_dispersion": {
                                k: v
                                for k, v in blk["area_dispersion"].items()
                                if not k.endswith("_by_horizon")
                            },
                        }
                        for m, blk in w["_pool"]["by_mask"].items()
                    },
                },
            }
            for w in windows
        ]
    )["by_mask"]["domain"]["area_dispersion_ratio_by_horizon"]

    ok = (
        single_error < 1e-9
        and pooled_error < 1e-9
        and len(by_h_single) == 3
        and len(by_h_pooled) == 3
        and absent is None
    )
    return Check(
        "area_dispersion_by_horizon_recombines_to_the_pooled_criterion",
        ok,
        "the per-lead dispersion sums recombine to the pooled G3 criterion exactly, per "
        "window and over windows, and a block written before the statistic existed pools to "
        "None rather than to a flat decomposition",
        {
            "by_horizon_single_window": by_h_single,
            "by_horizon_pooled": by_h_pooled,
            "pooled_criterion": pooled["area_dispersion_ratio"],
            "recombination_error_single": single_error,
            "recombination_error_pooled": pooled_error,
            "absent_statistic_pools_to": absent,
        },
    )


def check_ablation_sd_ratio_null_is_one_at_every_lead() -> Check:
    """[M21] The SD ratio's null LEVEL is lead-invariant, and that is its algebra.

    ``independence_dispersion_index`` compares an observed spread against a MODEL
    of the independent-pixel spread, and that model stops being right after one
    step, so its null ladders 1.0048 / 1.25 / ~1.50 / ~1.77 / ~2.24 with no
    latent anywhere. The check above compares an observed spread against ANOTHER
    OBSERVED SPREAD from the same process with one switch flipped, so multi-step
    contagion inflates both arms and cancels in the quotient. The null is
    therefore 1.0 at EVERY lead, which is a real structural advantage and is
    asserted here rather than assumed.

    Known answer forced by the algebra, not by a previous run: with the latent
    driven to zero the two arms are draws from the SAME law and differ only in
    which part of the generator stream they consume.
    """
    from wildfire_nowcast.eval.collapse_bars import ratio_sweep, summarise_ratio

    rows = sorted(
        summarise_ratio(ratio_sweep(families=("null_sigma_zero",), n_seeds=100)),
        key=lambda r: r.horizon_h,
    )
    worst = max(abs(row.ratio_median - 1.0) for row in rows)
    ok = bool(rows) and worst < 0.15
    return Check(
        "ablation_sd_ratio_null_is_one_at_every_lead",
        ok,
        "with z_t off on BOTH arms the SD ratio's median sits at 1.0 at every lead, unlike "
        "the dispersion index, whose null is exact for one step only",
        {
            "median_by_lead": {row.horizon_h: row.ratio_median for row in rows},
            "max_abs_deviation_from_one": worst,
        },
    )


def check_ablation_sd_ratio_bar_is_calibrated_at_the_lead_it_runs_at() -> Check:
    """[M21 found it, M22 repaired it] The TAIL is a function of the horizon, and
    the bar that ships must be calibrated at the horizon it ships at.

    The level being lead-invariant does not make the BAR lead-invariant. The
    quotient's denominator is the ablation arm's area SD, and that falls as the
    fire decelerates, so the right tail grows with the lead even though the
    median does not move.

    **TWO CLAUSES, AND THE FIRST OUTLIVES ITS OWN REPAIR.** M21's form asserted
    only that the false fire rate at the verdict lead EXCEEDED the rate at lead
    1, which is a statement that stays true whether the bar is well chosen or
    catastrophic - it was green at the retired 1.5, where 24-28% of no-latent
    draws passed. So:

    (a) at the RETIRED bar, which is fixed and does not move when the shipped
        bar does, the tail is still heavier at the verdict lead than at lead 1.
        That is the M21 finding, preserved at the value it was found at, and it
        fails only if the tail stops being a function of the horizon;
    (b) at the SHIPPED bar the false fire rate at the verdict lead is at or
        below :data:`ABLATION_NULL_FALSE_FIRE_CEILING`. That is the property the
        repair bought, and it fails immediately if the bar is lowered back
        towards where it was.
    """
    from wildfire_nowcast.eval.collapse_bars import null_tail_by_lead

    shipped = null_tail_by_lead(
        bar=ABLATION_SD_RATIO_BAR,
        horizon_h=ABLATION_SD_RATIO_HORIZON_H,
        n_seeds=ABLATION_NULL_TAIL_SEEDS,
        n_members=ABLATION_SD_RATIO_MEMBERS,
    )
    retired = null_tail_by_lead(
        bar=ABLATION_SD_RATIO_BAR_RETIRED,
        horizon_h=ABLATION_SD_RATIO_HORIZON_H,
        n_seeds=ABLATION_NULL_TAIL_SEEDS,
        n_members=ABLATION_SD_RATIO_MEMBERS,
    )
    horizon_dependent = retired[ABLATION_SD_RATIO_HORIZON_H] > retired[1]
    calibrated = shipped[ABLATION_SD_RATIO_HORIZON_H] <= ABLATION_NULL_FALSE_FIRE_CEILING
    ok = bool(horizon_dependent and calibrated)
    return Check(
        "ablation_sd_ratio_bar_is_calibrated_at_the_lead_it_runs_at",
        ok,
        f"the tail is still horizon dependent at the retired {ABLATION_SD_RATIO_BAR_RETIRED:g} "
        f"({100 * retired[1]:.1f}% at lead 1 vs "
        f"{100 * retired[ABLATION_SD_RATIO_HORIZON_H]:.1f}% at lead "
        f"{ABLATION_SD_RATIO_HORIZON_H}), and the shipped {ABLATION_SD_RATIO_BAR:g} is cleared "
        f"by {100 * shipped[ABLATION_SD_RATIO_HORIZON_H]:.1f}% of no-latent draws at the lead "
        f"the verdict is taken at, within the "
        f"{100 * ABLATION_NULL_FALSE_FIRE_CEILING:.0f}% ceiling",
        {
            "false_fire_rate_by_lead_shipped_bar": shipped,
            "false_fire_rate_by_lead_retired_bar": retired,
            "verdict_lead": ABLATION_SD_RATIO_HORIZON_H,
            "bar": ABLATION_SD_RATIO_BAR,
            "retired_bar": ABLATION_SD_RATIO_BAR_RETIRED,
            "ceiling": ABLATION_NULL_FALSE_FIRE_CEILING,
            "n_seeds": ABLATION_NULL_TAIL_SEEDS,
            "tail_is_horizon_dependent_at_the_retired_bar": horizon_dependent,
            "shipped_bar_is_calibrated": calibrated,
        },
    )


def check_one_step_index_bar_derivation_matches_the_shipped_index() -> Check:
    """[M21] The closed form for the one-step index agrees with the function in the tree.

    ``sim/selftest.py`` asserts a bar on ``independence_dispersion_index`` over a
    correlated ensemble built from one shared logit shift. That construction is
    already ONE step, so the multi-step defect never touched it and only the
    MAGNITUDE is at stake. The law of total variance gives the answer in closed
    form, and this check is what makes the closed form a statement about the
    SHIPPED estimand rather than about an idealisation of it.

    The two differ by a known factor and it is applied rather than absorbed into
    a loose tolerance: the shipped numerator is ``areas.std()`` with ``ddof=0``,
    which is low by ``sqrt((M - 1) / M)``, and the derivation is a many-member
    limit. With it applied they agree to well under 1%. **The factor is 0.84%
    and the Monte Carlo standard error at 800 replications is about 0.24%, so
    this check binds the LEVEL and cannot see the correction itself** - that is
    stated because a tolerance that swallows a term is not evidence for it.
    """
    import numpy as np

    from wildfire_nowcast.eval.collapse_bars import (
        INDEX_CONSTRUCTION,
        derived_one_step_index,
        index_bar_monte_carlo,
    )

    n_cells = int(INDEX_CONSTRUCTION["n_cells"])
    n_members = int(INDEX_CONSTRUCTION["n_members"])
    p = np.random.default_rng(0).uniform(
        INDEX_CONSTRUCTION["p_low"], INDEX_CONSTRUCTION["p_high"], size=n_cells
    )
    derived = derived_one_step_index(p, float(INDEX_CONSTRUCTION["sigma_z"])).index
    corrected = derived * float(np.sqrt((n_members - 1) / n_members))
    observed = index_bar_monte_carlo(800)
    relative = abs(corrected - observed["mean"]) / observed["mean"]
    ok = relative < 0.02 and observed["min"] > 2.0
    return Check(
        "one_step_index_bar_derivation_matches_the_shipped_index",
        ok,
        "the closed-form one-step index agrees with the shipped function on its own "
        "construction, and every replication clears the 2.0 asserted in sim/ by a wide "
        "margin - the bar is far below the quantity it is a bar on",
        {
            "derived_limit": derived,
            "derived_ddof_corrected": corrected,
            "shipped_monte_carlo": observed,
            "relative_difference": relative,
        },
    )


def check_elbo_kl_is_scaled_like_its_reconstruction_term() -> Check:
    """[M5] The KL and the reconstruction must be normalised the SAME way.

    Known-answer regression for the defect this milestone introduced and caught:
    the reconstruction is reported as a per-cell MEAN, so an unnormalised KL
    beside it multiplies the effective KL weight by the number of scored cells
    (~3,000), and the latent is driven to posterior collapse by arithmetic. The
    symptom read as "the latent had nothing to say".

    Asserted structurally rather than by training: doubling the number of scored
    cells must NOT change the KL's contribution to the loss.
    """
    import torch

    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
    from wildfire_nowcast.model.latent import LatentConfig, LatentHead

    head = LatentHead(LatentConfig(dim=3, free_bits=0.0))
    mu = torch.full((1, 3), 0.5, dtype=torch.float64)
    log_var = torch.zeros((1, 3), dtype=torch.float64)
    kl = float(head.kl(mu, log_var).sum())
    # KL(N(0.5, 1) || N(0, 1)) = 0.5 * 0.25 per dimension, exactly.
    exact = 3 * 0.5 * 0.25
    contribution = {n: kl / n for n in (1000, 2000)}
    ok = abs(kl - exact) < 1e-12 and contribution[2000] < contribution[1000]
    _ = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=3))
    return Check(
        "elbo_kl_is_scaled_like_its_reconstruction_term",
        ok,
        "KL is exact in closed form and enters the loss divided by the same cell count "
        "the reconstruction mean uses, so w_kl means what it says",
        {"kl": kl, "exact": exact, "per_cell_contribution": contribution},
    )


def check_latent_spec_round_trips_and_absence_means_absence() -> Check:
    """[M5] A pre-M5 checkpoint must reload as a NO-LATENT model, not as the default.

    ``from_spec`` defaulting a missing ``latent_config`` to the current value
    would silently turn every archived G2 checkpoint into a different model on
    reload - the same class as ADR-015's split moving under a running experiment,
    one level down. Absence maps to absence, and a latent spec round-trips.
    """
    import numpy as np

    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
    from wildfire_nowcast.model.latent import LatentConfig

    legacy = ContagionKernel(KernelConfig()).to_spec()
    legacy.pop("latent_config", None)
    reloaded_legacy = ContagionKernel.from_spec(legacy)

    with_z = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=2))
    reloaded_z = ContagionKernel.from_spec(with_z.to_spec())
    sigma_match = np.allclose(
        with_z.latent.sigma().detach().numpy(), reloaded_z.latent.sigma().detach().numpy()
    )
    ok = (
        reloaded_legacy.latent is None
        and reloaded_z.latent is not None
        and reloaded_z.latent.dim == 2
        and bool(sigma_match)
    )
    return Check(
        "latent_spec_round_trips_and_absence_means_absence",
        ok,
        "a spec with no `latent_config` reloads with NO latent (archived G2 checkpoints are "
        "unchanged); a latent spec round-trips its dim and sigma",
        {
            "legacy_has_latent": reloaded_legacy.latent is not None,
            "reloaded_dim": reloaded_z.latent.dim,
            "sigma_match": bool(sigma_match),
        },
    )


def check_debiased_dispersion_is_in_the_same_units_as_the_criterion() -> Check:
    """[M8] AT ZERO BIAS, ``ratio_debiased`` MUST EQUAL ``area_dispersion_ratio``.

    One line, and it would have caught a defect that stood in every M5-M7 table:
    the criterion goes through ``_ratio`` (which takes a square root, so it is in
    SD units) while its own debiased companion divided two sums of squares and
    was in VARIANCE units. Every value we report is below 1, where squaring moves
    a number DOWN, so a debiased ratio genuinely ABOVE the raw one printed BELOW
    it and the attribution read backwards on 155 of 170 cells.

    The invariant is definitional and needs no fixture: ``ratio_debiased``
    divides by the error a perfectly unbiased forecaster of the SAME SHARPNESS
    would have, so with the bias already zero the two are THE SAME QUANTITY.
    Before the fix this check would have read 0.3333 against 0.5774.

    Raised by sim (S5) against my file. Same square-root/units family as
    the error G3's own dispersion playthrough was written to catch, which is why
    it is now an assertion rather than a comment.
    """
    from wildfire_nowcast.eval.metrics import _area_error_decomposition, _ratio

    sum_var, sum_sq_err, n = 3.0, 9.0, 100
    unbiased = _area_error_decomposition(
        sum_var=sum_var, sum_sq_err=sum_sq_err, sum_signed=0.0, n=n
    )["area_error_decomposition"]
    criterion = _ratio(sum_var, sum_sq_err)
    # ...and with a real bias it must be STRICTLY LARGER, never smaller: removing
    # bias can only shrink the denominator. That direction is what reversed.
    biased = _area_error_decomposition(
        sum_var=sum_var, sum_sq_err=sum_sq_err, sum_signed=20.0, n=n
    )["area_error_decomposition"]
    ok = (
        criterion is not None
        and abs(unbiased["ratio_debiased"] - criterion) < 1e-15
        and biased["ratio_debiased"] > criterion
    )
    return Check(
        "debiased_dispersion_is_in_the_same_units_as_the_criterion",
        ok,
        "at zero bias the debiased ratio EQUALS area_dispersion_ratio exactly, and with bias "
        "it is strictly larger — both fail if the square root is dropped",
        {
            "criterion": criterion,
            "debiased_at_zero_bias": unbiased["ratio_debiased"],
            "debiased_with_bias": biased["ratio_debiased"],
            "bias_fraction": biased["bias_fraction"],
        },
    )


def check_gate_mean_preserving_is_off_by_default_and_exact_when_on() -> Check:
    """[M8] The gate's mean correction: absent by default, EXACT when asked for.

    Three properties, each of which a bug in this five-line change would break,
    and each written from the definition rather than read back from the module:

    1. **``gate_mean_preserving=False`` is BITWISE M7.** The correction vector is
       exactly what the pre-M8 expression produced, so every M6/M7 checkpoint and
       every archived number stays reproducible. This is the guarantee `M8`'s
       whole 2x2 rests on - the fourth cell of that matrix is a REUSED M7 run.
    2. **``E_z[e^gate] = 1`` EXACTLY when it is on**, at the unconditional prior.
       Closed form: with ``z ~ N(mu, 1)`` the multiplier's mean is
       ``e^(sigma mu + sigma^2/2)``, so the correction must be
       ``-(sigma mu + sigma^2/2)`` and the product must be 1 to machine epsilon.
       A correction that only removed ``sigma^2/2`` - the copy-paste bug this
       change is one keystroke away from - would leave ``e^(sigma mu)``, i.e.
       ``0.13`` at the fitted scale, and would look entirely plausible.
    3. **THE ASYMMETRY SURVIVES.** ADR-034 (2) makes asymmetry the working
       mechanism, so a "fix" that symmetrised the multiplier would destroy the
       thing it was meant to preserve. The corrected multiplier's MEDIAN is
       ``e^(-sigma^2/2)``, which at ``sigma = 1.3`` is ``0.43`` against a mean of
       1 - most members quiet, a thin expensive upper tail. Asserted as a
       STRICT INEQUALITY (median < 0.6 * mean), so this check fails the day the
       correction is widened into a symmetric one.
    """
    import math

    import torch

    from wildfire_nowcast.model.latent import LatentConfig, LatentHead

    def head(gate_mp: bool, sigma_gate: float) -> LatentHead:
        cfg = LatentConfig(
            dim=4,
            init_sigma=(0.35, 0.20, 0.15, sigma_gate),
            gate_prior_mean=-1.5,
            mean_preserving=True,
            gate_mean_preserving=gate_mp,
        )
        torch.manual_seed(0)
        return LatentHead(cfg)

    off, on = head(False, 1.3), head(True, 1.3)
    legacy = -0.5 * off.sigma() * off.sigma() * off.log_multiplier_mask
    off_is_bitwise_m7 = bool(torch.equal(off.mean_correction(), legacy))

    with torch.no_grad():
        sigma = float(on.sigma()[3])
        corr = float(on.mean_correction()[3])
    mean_multiplier = math.exp(sigma * (-1.5) + 0.5 * sigma * sigma + corr)
    uncorrected = math.exp(sigma * (-1.5) + 0.5 * sigma * sigma)
    median_over_mean = math.exp(-0.5 * sigma * sigma)

    # A config that asks for the correction without the dimension must RAISE,
    # not silently do nothing - the green-but-vacuous shape, refused up front.
    try:
        LatentConfig(dim=3, gate_mean_preserving=True)
        refuses_without_the_gate = False
    except ValueError:
        refuses_without_the_gate = True

    ok = (
        off_is_bitwise_m7
        and abs(mean_multiplier - 1.0) < 1e-12
        # The BIAS being removed, in closed form: e^(sigma mu + sigma^2/2) at
        # sigma = 1.3, mu = -1.5 is e^-1.105 = 0.3312. My first draft of this
        # bound said 0.2 and FAILED ON THE CLEAN WORLD - the fourth time a
        # known-answer check has corrected my arithmetic before a model saw it.
        # Kept as a bound on the SIZE of the bias, not a re-statement of it.
        and uncorrected < 0.5
        and median_over_mean < 0.6
        and refuses_without_the_gate
    )
    return Check(
        "gate_mean_preserving_is_off_by_default_and_exact_when_on",
        ok,
        "OFF is bitwise M7; ON gives E_z[e^gate] = 1 EXACTLY while the multiplier stays "
        "asymmetric (median 0.43 of the mean); asking for it without dim 4 raises",
        {
            "off_is_bitwise_m7": off_is_bitwise_m7,
            "corrected_mean_multiplier": mean_multiplier,
            "uncorrected_mean_multiplier": uncorrected,
            "median_over_mean": median_over_mean,
            "refuses_without_the_gate": refuses_without_the_gate,
        },
    )


def _toy_ladder_base() -> tuple[np.ndarray, np.ndarray]:
    """A small absorbing forecast with a real anisotropic front. No model, no zarr."""
    rng = np.random.default_rng(11)
    height, width, members, leads = 22, 26, 5, 3
    x0 = np.zeros((height, width), np.uint8)
    x0[9:13, 10:14] = 2
    x0[8:14, 9:15] = np.maximum(x0[8:14, 9:15], 1)
    base = np.zeros((members, leads, height, width), np.uint8)
    for m in range(members):
        cur = x0 > 0
        for lead in range(leads):
            add = np.zeros_like(cur)
            for y, x in zip(*np.nonzero(cur), strict=True):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < height and 0 <= xx < width:
                            if rng.random() < (0.55 if dx > 0 else 0.10):
                                add[yy, xx] = True
            cur = cur | add
            base[m, lead] = np.where(cur, 1, 0)
        base[m] = np.maximum.accumulate(base[m], axis=0)
    return np.maximum(base, x0[None, None]).astype(np.uint8), x0


def check_degradation_null_rung_is_bitwise_the_undegraded_forecast() -> Check:
    """[M11] The ladder's NEGATIVE CONTROL, and it must not be an ``if``.

    The whole power analysis rests on "a rung of zero severity separates at zero
    block-SD". If the identity were implemented by returning the input early, the
    claim would be a property of a branch and would say nothing about the
    construction that produces every OTHER rung. So the identity runs the full
    machinery - build the cell order, rank it, take the first ``n_h`` - and is
    asserted bitwise here, in BOTH families.
    """
    from wildfire_nowcast.model.degrade import MODE_AREA, MODE_SHAPE, degrade_samples

    base, x0 = _toy_ladder_base()
    area_identity = degrade_samples(base, x0, mode=MODE_AREA, level=1.0)
    shape_identity = degrade_samples(base, x0, mode=MODE_SHAPE, level=0.0)
    # ...and a rung that is NOT the identity must actually differ, or the check
    # above passes for the uninteresting reason that nothing ever changes.
    moved = degrade_samples(base, x0, mode=MODE_AREA, level=0.5)
    ok = (
        np.array_equal(area_identity, base)
        and np.array_equal(shape_identity, base)
        and not np.array_equal(moved, base)
    )
    return Check(
        "degradation_null_rung_is_bitwise_the_undegraded_forecast",
        ok,
        "k=1 and f=0 reproduce the wrapped samples bitwise THROUGH the degradation path, "
        "and k=0.5 does not — the positive control on the negative control",
        {
            "area_identity_bitwise": bool(np.array_equal(area_identity, base)),
            "shape_identity_bitwise": bool(np.array_equal(shape_identity, base)),
            "half_area_rung_differs": bool(not np.array_equal(moved, base)),
        },
    )


def check_degradation_rungs_hit_their_declared_severity() -> Check:
    """[M11] A ladder is only interpretable if each rung IS the severity it claims.

    Two known answers, both forced by the construction rather than read back:

    * an AREA rung realises ``k`` times the reference increment, to within the
      rounding of one cell per member and lead;
    * a SHAPE rung realises the reference area **EXACTLY** - integer equality, not
      a tolerance - because "a channel blind to shape at fixed area" is a
      different finding from "a channel blind to area", and separating them
      requires the area error to be zero rather than small.

    ``degrade_samples`` also runs every rung through ``validate_samples``, so a
    degradation that un-burned a cell or broke the absorbing order (C1.1) would
    raise here rather than reach a metric.

    THE UNDEFINED CASE IS PART OF THE CHECK, NOT AN EDGE OF IT. A ladder that
    spans harmless to catastrophic will eventually contain a rung whose increment
    union with the reference is EMPTY - neither forecast added a cell. That is
    the rung most likely to break monotonicity, so a validator that raises there
    is a validator that goes silent exactly where it is needed. An UNDEFINED
    overlap is therefore FAILED here, never skipped and never defaulted to a
    number, and the degenerate case is exercised below so that this check is
    observed to RETURN a verdict rather than abort.
    """
    from wildfire_nowcast.model.degrade import (
        MODE_AREA,
        MODE_SHAPE,
        UNDEFINED,
        degrade_samples,
        increment_overlap,
    )

    base, x0 = _toy_ladder_base()
    free = x0 == 0
    n_ref = int(((base > 0) & free[None, None]).sum())

    area_rows = {}
    for k in (0.10, 0.28, 0.50, 2.00, 5.00):
        deg = degrade_samples(base, x0, mode=MODE_AREA, level=k)
        area_rows[k] = int(((deg > 0) & free[None, None]).sum()) / n_ref
    area_ok = all(abs(v - k) <= 0.02 * max(k, 1.0) for k, v in area_rows.items())

    shape_rows = {}
    for f in (0.05, 0.15, 0.40, 1.00):
        deg = degrade_samples(base, x0, mode=MODE_SHAPE, level=f)
        shape_rows[f] = (
            int(((deg > 0) & free[None, None]).sum()),
            increment_overlap(deg, base, x0),
        )
    exact_area = all(cells == n_ref for cells, _ in shape_rows.values())
    all_defined = all(ov.defined for _, ov in shape_rows.values())
    ious = [ov.iou for _, ov in shape_rows.values()]
    monotone_iou = all_defined and all(a > b for a, b in zip(ious[:-1], ious[1:], strict=True))

    # The degenerate rung, planted: a forecast that adds nothing, compared with
    # itself. The overlap must report UNDEFINED and must NOT be orderable.
    nothing = np.maximum(np.zeros_like(base), x0[None, None]).astype(np.uint8)
    degenerate = increment_overlap(nothing, nothing, x0)
    degenerate_is_undefined = degenerate.outcome == UNDEFINED and degenerate.iou is None
    ordering_undefined_raises = False
    try:
        _ = degenerate.iou > 0.5  # type: ignore[operator]
    except TypeError:
        ordering_undefined_raises = True
    # ...and a check built the same way must RETURN False rather than raise when
    # a rung is undefined. Verified by running the same reduction on a row set
    # that contains one.
    with_undefined = [1.0, degenerate.iou]
    verdict_on_undefined = all(ov is not None for ov in with_undefined) and all(
        a > b for a, b in zip(with_undefined[:-1], with_undefined[1:], strict=True)
    )

    return Check(
        "degradation_rungs_hit_their_declared_severity",
        area_ok
        and exact_area
        and monotone_iou
        and degenerate_is_undefined
        and ordering_undefined_raises
        and verdict_on_undefined is False,
        "area rungs realise k within 2%; shape rungs realise the reference area EXACTLY and "
        "their IoU against it falls monotonically in f; an EMPTY increment union reports "
        "UNDEFINED, refuses to be ordered, and makes the monotonicity claim FALSE rather "
        "than raising",
        {
            "area_realised": {str(k): v for k, v in area_rows.items()},
            "shape_cells_equal_reference": exact_area,
            "shape_iou_by_f": {str(f): ov.iou for f, (_, ov) in shape_rows.items()},
            "shape_overlaps_all_defined": all_defined,
            "reference_increment_cells": n_ref,
            "degenerate_outcome": degenerate.outcome,
            "degenerate_union": degenerate.union,
            "ordering_an_undefined_overlap_raises": ordering_undefined_raises,
            "monotonicity_claim_on_an_undefined_row": verdict_on_undefined,
        },
    )


def check_base_prediction_cache_cannot_return_another_window() -> Check:
    """[M11] The ladder's 16 rungs share ONE forward pass. Prove the key is safe.

    The runner scores one model over every window of a fire before moving to the
    next model, so a cache key that is not fully identifying would hand a rung
    another window's samples and the entire ladder would be a ladder over the
    wrong forecast - silently, with every downstream number still finite and
    plausible. Each C5 argument is therefore perturbed in turn and the key must
    move; the planted defect is the perturbation, not a comment saying it is safe.
    """
    from wildfire_nowcast.model.degrade import BasePredictionCache

    base, x0 = _toy_ladder_base()
    members, leads, height, width = base.shape
    static = np.zeros((8, height, width), np.float32)
    weather = np.zeros((leads, 5, height, width), np.float32)
    cache = BasePredictionCache(capacity=4)
    key = cache.key(x0, static, weather, members, leads, 7)
    cache.put(key, base)

    moved: dict[str, bool] = {}
    x0b = x0.copy()
    x0b[0, 0] = 1
    moved["x0"] = cache.key(x0b, static, weather, members, leads, 7) != key
    sb = static.copy()
    sb[0, 0, 0] = 1.0
    moved["static"] = cache.key(x0, sb, weather, members, leads, 7) != key
    wb = weather.copy()
    wb[0, 0, 0, 0] = 1.0
    moved["weather"] = cache.key(x0, sb * 0, wb, members, leads, 7) != key
    moved["seed"] = cache.key(x0, static, weather, members, leads, 8) != key
    moved["members"] = cache.key(x0, static, weather, members + 1, leads, 7) != key

    round_trip = np.array_equal(cache.get(key), base)
    miss_is_none = cache.get("0" * 32) is None
    return Check(
        "base_prediction_cache_cannot_return_another_window",
        all(moved.values()) and round_trip and miss_is_none,
        "every C5 argument moves the key, a hit is bitwise, and a miss is None rather than a "
        "stale neighbouring window",
        {
            **{f"key_moves_on_{k}": v for k, v in moved.items()},
            "round_trip_bitwise": round_trip,
            "miss_returns_none": miss_is_none,
        },
    )


def check_mde_read_off_requires_a_SUSTAINED_crossing() -> Check:
    """[M11] One rung crossing a bar is not a detection threshold.

    Known answer, on a hand-built curve. A channel that reads 2.4 block-SD at
    severity 0.2 and falls back to 1.1 at 0.4 has not detected 0.2 - it has
    produced one lucky rung, and taking the first crossing would report an MDE
    three times too optimistic for exactly the instrument this analysis exists to
    distrust. The read-off therefore requires every LARGER rung to clear the bar
    too, and the monotonicity report must NAME the inversion rather than smooth it.
    """
    from wildfire_nowcast.eval.power import minimum_detectable_effect, monotonicity

    lucky = [
        {"severity": 0.2, "abs_separation_sd": 2.4, "level": 1.0, "truth_distance": 0.2},
        {"severity": 0.4, "abs_separation_sd": 1.1, "level": 0.5, "truth_distance": 0.4},
        {"severity": 0.8, "abs_separation_sd": 3.0, "level": 3.0, "truth_distance": 0.8},
        {"severity": 1.6, "abs_separation_sd": 6.0, "level": 6.0, "truth_distance": 1.6},
    ]
    got = minimum_detectable_effect(lucky, bar=2.0)
    # sustained detection starts at 0.8; interpolating 1.1 -> 3.0 gives 0.4 + 0.4*(0.9/1.9)
    expected = 0.4 + 0.4 * (2.0 - 1.1) / (3.0 - 1.1)
    sustained_ok = got["bound"] == "interpolated" and _close(got["mde"], expected, 1e-9)

    blind = [dict(row, abs_separation_sd=0.3) for row in lucky]
    blind_ok = minimum_detectable_effect(blind, bar=2.0)["bound"] == "greater_than"
    loud = [dict(row, abs_separation_sd=9.0) for row in lucky]
    loud_ok = minimum_detectable_effect(loud, bar=2.0)["bound"] == "at_or_below"

    clean = monotonicity(lucky, by="truth_distance", value="level", expect_increasing=True)
    inverted = monotonicity(
        [dict(row) for row in lucky[:1]]
        + [dict(lucky[1], level=9.0)]
        + [dict(row) for row in lucky[2:]],
        by="truth_distance",
        value="level",
        expect_increasing=True,
    )
    monotone_ok = clean["monotone"] is False and inverted["n_inversions"] >= 1
    return Check(
        "mde_read_off_requires_a_SUSTAINED_crossing",
        sustained_ok and blind_ok and loud_ok and monotone_ok,
        "a lucky rung does not set the MDE; a blind channel is bounded BELOW rather than "
        "reported as a number; and a planted inversion is named, not smoothed",
        {
            "mde": got.get("mde"),
            "expected": expected,
            "blind_channel_bound": minimum_detectable_effect(blind, bar=2.0)["bound"],
            "loud_channel_bound": minimum_detectable_effect(loud, bar=2.0)["bound"],
            "planted_inversions": inverted["n_inversions"],
        },
    )


def check_square_dilation_iou_matches_its_closed_form() -> Check:
    """[L1] The morphology control ASSERTS A MAGNITUDE, with zero tolerance.

    An ``S x S`` square dilated by ``r`` cells with an 8-connected structuring
    element is EXACTLY ``(S+2r) x (S+2r)``, so ``IoU == S^2 / (S+2r)^2`` in closed
    form. The whole synthetic arm of the label-noise floor is built on this
    operator, so it is certified against arithmetic rather than against itself.

    The check carries its own planted defect: a 4-CONNECTED dilation, which is
    the most plausible wrong answer, must NOT reproduce the closed form. Without
    that clause "the identity holds" would be consistent with an identity that
    holds for everything.
    """
    from wildfire_nowcast.eval.labelfloor import square_dilation_identity

    side, radius, pad = 21, 1, 12
    exact = square_dilation_identity(side=side, radius=radius, pad=pad)

    n = side + 2 * pad
    base = np.zeros((n, n), dtype=bool)
    base[pad : pad + side, pad : pad + side] = True
    four = base.copy()
    four[1:, :] |= base[:-1, :]
    four[:-1, :] |= base[1:, :]
    four[:, 1:] |= base[:, :-1]
    four[:, :-1] |= base[:, 1:]
    four_iou = float((base & four).sum()) / float((base | four).sum())
    analytic = (side**2) / float((side + 2 * radius) ** 2)

    ok = bool(exact["exact"]) and abs(four_iou - analytic) > 1e-9
    return Check(
        "square_dilation_iou_matches_its_closed_form",
        ok,
        "8-connected dilate/erode reproduce S^2/(S+2r)^2 and (S-2r)^2/S^2 EXACTLY "
        "(abs error 0.0), and a 4-connected impostor does not",
        {
            "dilate_iou_abs_error": exact["dilate_iou_abs_error"],
            "erode_iou_abs_error": exact["erode_iou_abs_error"],
            "dilate_cells": exact["dilate_cells"],
            "dilate_cells_analytic": exact["dilate_cells_analytic"],
            "planted_4_connected_abs_error": abs(four_iou - analytic),
        },
    )


def check_calibration_target_and_gate_criterion_are_different_estimands() -> Check:
    """[L1-close] The ceiling is not re-reporting its own calibration target.

    ADR-056 (6). The severity was calibrated so whole-footprint
    ``IoU(noise(L), L)`` hit 0.756818 and the oracle then scored
    ``iou_shape_masked_3h = 0.7590`` - within 0.3%. Close enough that the
    headline collapses if the two are one quantity through the plumbing.

    Three clauses, and each can fail independently:
    1. **MAGNITUDES.** Two constructed cases, whole-footprint and
       masked-increment IoU both in closed form, asserted with zero tolerance
       (``9800/10208``, ``0``, ``9590/10610``, ``9/11``).
    2. **ORDER REVERSAL.** ``whole(A) > whole(B)`` while
       ``masked(A) < masked(B)``, so neither can be a monotone function of the
       other. Equality of two numbers is a coincidence; a reversal is not.
    3. **THE CONTROL'S OWN CONTROL.** At severity zero the two agree exactly,
       so the harness is capable of reporting agreement and clause 2 is a
       property of the cases, not of the code.
    """
    from wildfire_nowcast.eval.labelfloor import whole_footprint_vs_masked_increment

    report = whole_footprint_vs_masked_increment()
    ok = (
        bool(report["magnitudes_exact"])
        and bool(report["order_reverses"])
        and bool(report["agrees_at_identity"])
    )
    return Check(
        "calibration_target_and_gate_criterion_are_different_estimands",
        ok,
        "whole-footprint IoU and masked-increment IoU hit four closed forms EXACTLY, "
        "reverse order between two cases, and coincide at severity zero",
        {
            "case_a": {
                "whole": report["case_a"]["whole_footprint_iou"],
                "masked": report["case_a"]["masked_increment_iou"],
            },
            "case_b": {
                "whole": report["case_b"]["whole_footprint_iou"],
                "masked": report["case_b"]["masked_increment_iou"],
            },
            "identity_case_agrees": report["agrees_at_identity"],
            "order_reverses": report["order_reverses"],
            "magnitudes_exact": report["magnitudes_exact"],
        },
    )


def check_severity_sampler_is_the_shipped_observation_noise() -> Check:
    """[L1] The severity ladder EXTENDS the shipped noise model; it does not replace it.

    C0 has one implementation of anything the contract adjudicates, and a second
    observation-noise model wearing the first one's calibration would be exactly
    the drift C0 forbids. The assertion is BITWISE - every field of every draw,
    on a shared RNG stream - because agreeing moments is what a second model
    would also produce.

    Two clauses, not one: the draws must match, AND the comparison must be
    non-vacuous (a sampler that always returned the identity would match too).
    """
    from wildfire_nowcast.model.noiseoracle import sampler_reduces_to_shipped

    report = sampler_reduces_to_shipped(n_draws=4000)
    ok = bool(report["bitwise_identical"]) and report["n_nonidentity_draws"] > 100
    return Check(
        "severity_sampler_is_the_shipped_observation_noise",
        ok,
        "at scale 1 the severity sampler reproduces the shipped label-perturbation "
        "sampler draw for draw, and most draws are NOT the identity",
        {
            "mismatches": report["mismatches"],
            "in_shipped_regime": report["in_shipped_regime"],
            "n_nonidentity_draws": report["n_nonidentity_draws"],
            "first_mismatch": report["first_mismatch"],
        },
    )


def check_noise_oracle_null_severity_is_the_labels_exactly() -> Check:
    """[L1] The ceiling's NULL RUNG, and it must not be an ``if``.

    Severity 0 is a perfect forecaster against perfect labels, so it must score
    the optimum of every metric EXACTLY - IoU 1, Brier 0, arrival CRPS 0 - and a
    non-zero severity must move at least one of them, or the ladder is measuring
    the harness. The identity runs the entire perturbation path (draw, morph,
    shift, union with ``x0``, absorbing check) rather than returning truth early.
    """
    from wildfire_nowcast.model.noiseoracle import NoisyTruthOracle, WindowTable

    ds, _ = _synthetic_dataset()
    try:
        windows = [forecast_inputs(ds, t0=t, horizon_h=3, fire_id="synthetic") for t in (6, 9)]
        table = WindowTable()
        for window in windows:
            table.add(window)
        clean = NoisyTruthOracle(table, name="null", scale=0.0, split_fingerprint="x")
        noisy = NoisyTruthOracle(table, name="noisy", scale=6.0, split_fingerprint="x")
        exact = []
        moved = False
        for window in windows:
            samples = clean.predict(window.x0, window.static, window.weather, 8, 3, 3)
            truth_state = np.asarray(window.truth) > 0
            exact.append(bool(np.all((samples > 0) == truth_state[None])))
            dirty = noisy.predict(window.x0, window.static, window.weather, 8, 3, 3)
            moved = moved or not np.array_equal(dirty, samples)
    finally:
        ds.close()
    ok = all(exact) and moved
    return Check(
        "noise_oracle_null_severity_is_the_labels_exactly",
        ok,
        "severity 0 reproduces the label trajectory cell for cell THROUGH the full "
        "perturbation path, and a non-zero severity does not",
        {
            "windows_exact": exact,
            "nonzero_severity_differs": moved,
            "table": table.stats(),
        },
    )


def check_window_table_refuses_a_key_collision() -> Check:
    """[L1] A truth-aware oracle that returns the WRONG window is undetectable downstream.

    C5 hands ``predict`` no fire id and no ``t0``, so the oracle identifies its
    window from ``(x0, static, weather)``. If two windows ever hashed alike, the
    oracle would score one fire's forecast against another's truth and every
    number under it would be finite, plausible and wrong. The table therefore
    RAISES on a duplicate rather than overwriting, and a miss RAISES rather than
    falling back - both are asserted here by provoking them.
    """
    from wildfire_nowcast.model.noiseoracle import NoisyTruthOracle, WindowTable

    ds, _ = _synthetic_dataset()
    try:
        window = forecast_inputs(ds, t0=6, horizon_h=3, fire_id="synthetic")
        other = forecast_inputs(ds, t0=9, horizon_h=3, fire_id="synthetic")
        table = WindowTable()
        table.add(window)
        duplicate_raised = False
        try:
            table.add(window)
        except KeyError:
            duplicate_raised = True
        miss_raised = False
        oracle = NoisyTruthOracle(table, name="o", scale=1.0, split_fingerprint="x")
        try:
            oracle.predict(other.x0, other.static, other.weather, 4, 3, 1)
        except KeyError:
            miss_raised = True
        # ...and the window it DOES know must still be served, or "it raises" is
        # true for the uninteresting reason that it raises on everything.
        served = oracle.predict(window.x0, window.static, window.weather, 4, 3, 1)
    finally:
        ds.close()
    ok = duplicate_raised and miss_raised and served.shape == (4, 3, *window.x0.shape)
    return Check(
        "window_table_refuses_a_key_collision",
        ok,
        "a duplicate key RAISES, an unknown window RAISES, and a known window is "
        "still served — a lookup that raises on everything is not a lookup",
        {
            "duplicate_raised": duplicate_raised,
            "miss_on_unknown_window_raised": miss_raised,
            "known_window_served_shape": list(served.shape),
            "stats": table.stats(),
        },
    )


def check_the_official_perimeter_endpoint_has_not_drifted() -> Check:
    """[L1] Two copies of one URL is one copy too many unless they are checked.

    The label-floor module holds its own copy of the WFIGS endpoint so that its
    cache lands under ``runs/`` instead of under ``data/``. That is a deliberate
    duplication and C0's whole point is that duplications drift, so the two
    constants are compared here rather than trusted. No network call is made.
    """
    from wildfire_nowcast.data.sources.nifc import WFIGS_PERIMETERS_URL as SOURCE_URL
    from wildfire_nowcast.eval.labelfloor import WFIGS_PERIMETERS_URL as EVAL_URL
    from wildfire_nowcast.eval.labelfloor import cache_dir, wfigs_name_for

    same = SOURCE_URL == EVAL_URL
    names = {
        "2020_czu_lightning_complex": "CZU LIGHTNING COMPLEX",
        "2019_kincade": "KINCADE",
        "2024_borel": "BOREL",
    }
    derived_ok = all(wfigs_name_for(k) == v for k, v in names.items())
    cache_outside_data = "data" not in cache_dir().parts
    return Check(
        "the_official_perimeter_endpoint_has_not_drifted",
        same and derived_ok and cache_outside_data,
        "the duplicated WFIGS endpoint matches its source, incident names are DERIVED "
        "from the fire id, and the official-geometry cache is outside data/",
        {
            "urls_agree": same,
            "derived_names_ok": derived_ok,
            "cache_dir": str(cache_dir()),
        },
    )


def check_stage_decay_recovers_a_known_beta() -> Check:
    """[U0] Synthesise a KNOWN stage decay; the estimator must read it back exactly.

    ADR-058 (10) item 3. ``g_i = A exp(beta i / n)`` makes the late half the
    early half multiplied term by term by ``exp(beta / 2)``, so ``stage_decay``
    has the closed form ``beta / 2`` for every even ``n`` and every amplitude.
    Asserted as a MAGNITUDE at 1e-12 over 18 (beta, n, amplitude) cells spanning
    both signs, a 100x range of ``n`` and a 1e6 range of amplitude - ADR-051's
    standard, which asked for an analytic identity rather than a non-zero
    reading. ``beta = 0`` must read EXACTLY ``0.0``.

    Three invariants ride along because each one is a rival form rejected:
    the estimand is exactly ANTISYMMETRIC under time reversal, it is INVARIANT
    under any monotone reparameterisation of age (so "hours since ignition" and
    the ``t0`` index cannot disagree), and a half with zero mean growth is
    UNDEFINED rather than ``-inf``.
    """
    from wildfire_nowcast.eval.stage import known_beta_recovery

    report = known_beta_recovery()
    ok = (
        bool(report["recovery_exact"])
        and bool(report["zero_beta_is_exactly_zero"])
        and bool(report["refuses_zero_half"])
        and bool(report["antisymmetric_under_time_reversal"])
        and bool(report["age_scale_invariant"])
    )
    return Check(
        "stage_decay_recovers_a_known_beta",
        ok,
        "stage_decay reads back beta/2 from synthesised forecasts across sign, sample size "
        "and amplitude; it is antisymmetric in time, invariant to the age scale, and refuses "
        "a zero half",
        {
            "n_recovery_cells": len(report["recovery"]),
            "max_abs_error": report["max_recovery_abs_error"],
            "zero_beta_is_exactly_zero": report["zero_beta_is_exactly_zero"],
            "refuses_zero_half": report["refuses_zero_half"],
            "antisymmetric": report["antisymmetric_under_time_reversal"],
            "age_scale_invariant": report["age_scale_invariant"],
        },
    )


def check_stage_decay_agrees_when_it_should_and_reverses_the_published_order() -> Check:
    """[U0] The AGREEMENT case, and the ORDER REVERSAL, in the same control.

    ADR-057 (5): *a divergence test that can only ever show divergence is broken
    in the other direction*. So this asserts both halves.

    **AGREEMENT (disqualifying).** Two forecasts built to share a stage decay
    while differing in amplitude (1 vs 987.65) and sample size (40 vs 400) must
    be reported as the same number; a bit-identical pair must differ by EXACTLY
    0.0; and a candidate that IS the reference must separate at EXACTLY 0.0 with
    0 of 5 blocks favouring it. An estimator that manufactures a difference
    between two things built to be identical is disqualified whatever else it
    does.

    **ORDER REVERSAL.** ``stage_decay`` and the published first-bin-to-last-bin
    ratio ORDER TWO CASES OPPOSITELY, both against closed forms
    (``log(8/13)``, ``log(61/70)``, ``-log 2``, ``-log 10``). Equality of two
    statistics can be a coincidence; a reversal proves they are different
    estimands, which is exactly how ADR-058 (2)'s factor of 2.6 became possible.

    **THE REVERSAL'S OWN CONTROL.** A second pair on which the two statistics
    agree, so the reversal is a property of the cases and not of the harness.
    """
    from wildfire_nowcast.eval.stage import known_beta_recovery

    report = known_beta_recovery()
    ok = (
        bool(report["agrees_at_identity"])
        and bool(report["magnitudes_exact"])
        and bool(report["order_reverses"])
        and bool(report["statistics_agree_on_the_control_pair"])
    )
    return Check(
        "stage_decay_agrees_when_it_should_and_reverses_the_published_order",
        ok,
        "identical inputs give a difference of exactly 0.0, four closed forms are hit exactly, "
        "and stage_decay reverses the published endpoint statistic on one pair while agreeing "
        "with it on another",
        {
            "bit_identical_difference": report["agreement"]["bit_identical_pair_difference"],
            "same_beta_difference": report["agreement"]["same_beta_different_amplitude_and_n"],
            "self_separation_sd": report["agreement"]["self_separation_sd"],
            "order_reverses": report["order_reverses"],
            "agrees_on_control_pair": report["statistics_agree_on_the_control_pair"],
            "x_stage_decay": report["cases"]["x_late_recovery"]["stage_decay"],
            "y_stage_decay": report["cases"]["y_late_collapse"]["stage_decay"],
            "x_endpoint": report["cases"]["x_late_recovery"]["endpoint_log_ratio"],
            "y_endpoint": report["cases"]["y_late_collapse"]["endpoint_log_ratio"],
        },
    )


def check_stage_decay_separation_cannot_be_bought_by_closing_more_of_the_gap() -> Check:
    """[U0] The power ceiling is an IDENTITY, not a measurement.

    An arm that reduces every block's distance-to-truth by a common fraction
    ``f`` has per-block margins ``f * d_b``, so its equal-block separation is
    ``mean(d) / sd(d)`` - **independent of f**. Closing 1% of the gap and closing
    100% of it separate identically.

    Verified here by running the shipped separation through
    :func:`common.separation.separation` at f = 0.01, 0.5 and 1.0 on the same
    five blocks and asserting the three agree to 1e-12 AND agree with the closed
    form. The consequence is the MDE verdict in ``runs/u0.json``: on a fold whose
    per-block distances have a coefficient of variation above 0.5, no
    proportional-closure arm can reach 2.0 block-SD however good it is, and that
    is a proof rather than an underpowered measurement.
    """
    import numpy as np

    from wildfire_nowcast.eval.stage import (
        proportional_closure_separation,
        separation_of_blocks,
    )

    distances = {4: 2.7522, 5: 1.6473, 6: 0.3107, 7: 1.1289, 12: 1.9747}
    ceiling = proportional_closure_separation(distances)
    measured = [
        separation_of_blocks(
            {b: d * (1.0 - f) for b, d in distances.items()}, distances, lower_is_better=True
        ).separation_sd
        for f in (0.01, 0.5, 1.0)
    ]
    values = np.array([m for m in measured if m is not None], dtype=float)
    closed = float(ceiling["separation_sd"] or 0.0)
    ok = (
        len(values) == 3
        and float(np.max(np.abs(values - closed))) <= 1e-12
        and float(np.max(values) - np.min(values)) <= 1e-12
        and closed > 0.0
    )
    return Check(
        "stage_decay_separation_cannot_be_bought_by_closing_more_of_the_gap",
        ok,
        "closing 1%, 50% and 100% of the per-block gap all separate at mean(d)/sd(d), so the "
        "separation of a proportional-closure arm is a ceiling and not an effect size",
        {
            "closed_form": closed,
            "measured": [float(v) for v in values],
            "distance_cv": ceiling["distance_cv"],
        },
    )


def check_stage_decay_asks_the_registry_instead_of_remembering() -> Check:
    """[U0] The C6 guard is CALLED on the scoring path, not merely present.

    ADR-059 (5): the registry was correct and ``eval/`` had no call site for
    :func:`common.null_check.assert_may_adjudicate`, which is the difference
    between a guard existing and a guard being wired.

    Two clauses, and neither pins the maintainer's ruling - that is deliberate,
    because a check that fails the day a channel is legitimately licensed teaches
    people to edit checks:

    1. :func:`eval.stage.licence` MIRRORS the registry. Whatever
       ``may_adjudicate('stage_decay')`` says, ``licence()['may_adjudicate']``
       says the same, and when the answer is NO the refusal text travels with it
       so a reader sees WHY rather than a bare False.
    2. The mechanism can refuse. An unregistered channel name raises
       ``NonAdjudicatingMetricError`` - the planted defect for this clause, since
       a licence function that returned True unconditionally would pass clause 1
       only if the registry already said yes.
    """
    from wildfire_nowcast.common.null_check import (
        NonAdjudicatingMetricError,
        assert_may_adjudicate,
        may_adjudicate,
    )
    from wildfire_nowcast.eval.stage import STAGE_DECAY_KEY, licence

    report = licence(gate="the U0 self-test")
    registry_says = may_adjudicate(STAGE_DECAY_KEY)
    mirrors = bool(report["may_adjudicate"]) is bool(registry_says)
    explains = registry_says or (
        STAGE_DECAY_KEY in str(report.get("registry_refusal", ""))
        and report["outcome"] == "NOT_LICENSED"
    )
    try:
        assert_may_adjudicate("stage_decay_but_misspelled")
        refuses_unknown = False
    except NonAdjudicatingMetricError:
        refuses_unknown = True
    return Check(
        "stage_decay_asks_the_registry_instead_of_remembering",
        bool(mirrors and explains and refuses_unknown),
        "eval.stage.licence reports exactly what the C6 registry says, carries the refusal "
        "text when the answer is no, and the registry refuses an unregistered channel",
        {
            "registry_may_adjudicate": registry_says,
            "licence_outcome": report["outcome"],
            "refuses_an_unregistered_channel": refuses_unknown,
        },
    )


def check_stage_ladder_severity_is_not_computed_by_the_channel() -> Check:
    """[U0] The MDE ladder's x-axis must not be the channel under test.

    ``eval/power``'s standard: a severity declared in the unit the channel
    computes makes every channel look sensitive, because the ladder is then
    regressing a statistic on itself. :func:`eval.stage.injection_severity` is
    raw displaced cell count, and this check shows the two are genuinely
    different quantities rather than asserting it in prose:

    1. **The identity rung is exactly zero.** ``delta = 0`` runs the whole
       injection path and must move nothing, so a non-zero reading on the null
       rung of ``runs/u0.json`` would be a ladder defect and not an effect.
    2. **Severity is strictly increasing in the injected tilt**, so the ladder's
       rungs are ORDERED - an unordered severity axis makes the MDE read-off
       meaningless whatever the separations do.
    3. **``stage_decay`` is BLIND to part of severity.** Moving growth between
       two windows in the SAME half leaves ``stage_decay`` bit-identical (the
       half mean is preserved) while severity rises. That is the demonstration
       that the x-axis carries information the y-axis cannot see.
    """
    from wildfire_nowcast.eval.stage import (
        apply_stage_slope_to_rows,
        injection_severity,
        stage_decay_by_block,
    )

    rows: list[dict[str, Any]] = [
        {"spatial_block_id": 0, "t0": i, "model_growth": 1.0 + float(i % 3)} for i in range(40)
    ]
    identity = apply_stage_slope_to_rows(rows, 0.0)
    identity_severity = injection_severity(rows, identity)

    severities = [
        injection_severity(rows, apply_stage_slope_to_rows(rows, -delta))
        for delta in (0.1, 0.5, 1.0, 2.0)
    ]
    monotone = all(a < b for a, b in zip(severities, severities[1:], strict=False))

    # A transfer of 1.0 cell from window 1 to window 0. Both are in the EARLY
    # half, so the half sum - and therefore stage_decay - is untouched.
    moved = [dict(r) for r in rows]
    moved[0]["model_growth"] = float(rows[0]["model_growth"]) + 1.0
    moved[1]["model_growth"] = float(rows[1]["model_growth"]) - 1.0
    before = stage_decay_by_block(rows, target="model_growth")[0]
    after = stage_decay_by_block(moved, target="model_growth")[0]
    blind = bool(
        before.value is not None and after.value is not None and after.value == before.value
    )
    moved_severity = injection_severity(rows, moved)

    return Check(
        "stage_ladder_severity_is_not_computed_by_the_channel",
        bool(
            identity_severity == 0.0
            and monotone
            and blind
            and moved_severity > 0.0
            and severities[0] > 0.0
        ),
        "the ladder's severity axis is zero at the identity, strictly increasing in the "
        "injected tilt, and measures displacement stage_decay is bit-blind to",
        {
            "identity_severity": identity_severity,
            "severities": severities,
            "stage_decay_unchanged_by_within_half_transfer": blind,
            "severity_of_that_transfer": moved_severity,
        },
    )


def check_stage_sign_criterion_is_exact_and_the_estimand_is_the_licensed_one() -> Check:
    """[U0b] The sign criterion against closed forms, and the estimand against its digest.

    ADR-060 (7) item 2 moved this channel from an effect size to a paired SIGN
    criterion, so the p-value is now load-bearing and is asserted as a RATIONAL,
    not to three decimals: ``P(X >= 11 | n=14) = 470/16384``, ``P(X >= 4 | n=5) =
    6/32``, ``P(X >= 14 | n=14) = 1/16384``, ``P(X >= 0) = 1`` exactly. Exact
    ``math.comb``, never a normal approximation - at n=14 the approximation is
    wrong in the third decimal, which is precisely where a 10-vs-11 call lands.
    Monotonicity in ``k`` rides along, since a tail that is not monotone is not a
    tail.

    The second half is provenance, not arithmetic. ``runs/u0b.json`` claims to use
    the estimand D3 licensed; :func:`eval.stage.estimand_digest` hashes the LIVE
    source of those four objects and compares it with the pinned SHA-256, so the
    claim is checked rather than believed. **This check is DESIGNED to go red if
    the estimand is edited** - that is not brittleness, it is the friction that
    stops a silent re-tune from inheriting ADR-060's licence. Re-pin the constant
    in the same commit and re-run the known-beta control.

    Planted defect for the arithmetic half: ``sign_test(15, 14)`` must RAISE
    rather than return a tail, because a count that exceeds its denominator is a
    caller bug and returning 0.0 would hide it inside a p-value.
    """
    from fractions import Fraction

    from wildfire_nowcast.eval.stage import estimand_digest, sign_test

    closed_form = {
        (11, 14): Fraction(470, 16384),
        (4, 5): Fraction(6, 32),
        (14, 14): Fraction(1, 16384),
        (0, 14): Fraction(1, 1),
        (7, 14): Fraction(9908, 16384),
    }
    errors = {
        f"{k}/{n}": abs(float(sign_test(k, n)["p_one_sided"]) - float(want))
        for (k, n), want in closed_form.items()
    }
    tails = [float(sign_test(k, 14)["p_one_sided"]) for k in range(15)]
    monotone = all(a >= b for a, b in zip(tails[:-1], tails[1:], strict=True))
    try:
        sign_test(15, 14)
        refuses_impossible = False
    except ValueError:
        refuses_impossible = True

    digest = estimand_digest()
    return Check(
        "stage_sign_criterion_is_exact_and_the_estimand_is_the_licensed_one",
        bool(
            max(errors.values()) == 0.0
            and monotone
            and refuses_impossible
            and digest["outcome"] == "UNCHANGED_SINCE_D3"
        ),
        "the one-sided sign tail matches five exact rationals, is monotone in k, refuses a "
        "count above its denominator, and the estimand still hashes to the source D3 licensed",
        {
            "abs_errors": errors,
            "monotone_in_k": monotone,
            "refuses_k_above_n": refuses_impossible,
            "estimand_outcome": digest["outcome"],
            "estimand_sha256": digest["sha256"],
        },
    )


def check_arm_s_is_arm_a_plus_four_parameters_and_starts_there() -> Check:
    """[S1] The CAPACITY condition and the zero-init condition, as arithmetic.

    ADR-061 (6) fixes arm S as "the incumbent plus ONE scalar input, ~4 extra
    parameters". Two things have to be true for a difference between S and A to
    be attributable to the covariate rather than to capacity or to a different
    starting point, and both are measurements, not intentions:

    * ``count(S) - count(A) == 4`` on the *same* configuration, latent included,
      and the extra tensors are exactly the stage head's two.
    * At initialisation the head is the IDENTITY - ``log_amplitude`` exactly 0.0
      and ``reach_scale`` exactly 1.0, for any state - so S at step 0 IS A. Not
      "close to": exact, because the coefficients are zero and the basis is
      linear in them.

    Planted defect, and the reason this check exists at all: a head of the form
    ``v tanh(w z + b) + u z`` also has 4 parameters and also starts at zero, and
    two of them would be BORN DEAD (``df/dw = v sech^2(.) z = 0`` at ``v = 0``).
    So the gradient of every coefficient is measured at init and must be
    non-zero; a head that reports 4 parameters and can only move 2 would make
    "S is A + 4" a false statement about capacity in the direction that flatters
    the challenger.
    """
    import torch

    from wildfire_nowcast.model.direct import parameter_counts
    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
    from wildfire_nowcast.model.latent import LatentConfig
    from wildfire_nowcast.model.stagehead import N_STAGE_PARAMETERS, StageHead

    latent = LatentConfig(dim=4, spatial_modes=2)
    arm_a = ContagionKernel(KernelConfig(stage_scalar=False), latent_config=latent)
    arm_s = ContagionKernel(KernelConfig(stage_scalar=True), latent_config=latent)
    counts_a, counts_s = parameter_counts(arm_a), parameter_counts(arm_s)
    delta = int(counts_s["total"]) - int(counts_a["total"])
    extra = sorted(set(counts_s["per_tensor"]) - set(counts_a["per_tensor"]))
    missing = sorted(set(counts_a["per_tensor"]) - set(counts_s["per_tensor"]))

    head = StageHead()
    burned = torch.rand(3, 12, 15)
    log_amplitude, log_reach = head(burned)
    identity_at_init = bool(
        torch.equal(log_amplitude, torch.zeros_like(log_amplitude))
        and torch.equal(torch.exp(log_reach), torch.ones_like(log_reach))
    )

    (log_amplitude.sum() + log_reach.sum()).backward()
    grads = {
        name: [float(g) for g in (param.grad.reshape(-1) if param.grad is not None else [])]
        for name, param in head.named_parameters()
    }
    flat = [g for values in grads.values() for g in values]
    all_live = bool(len(flat) == N_STAGE_PARAMETERS and all(abs(g) > 0.0 for g in flat))

    return Check(
        "arm_s_is_arm_a_plus_four_parameters_and_starts_there",
        bool(
            delta == N_STAGE_PARAMETERS
            and not missing
            and len(extra) == 2
            and identity_at_init
            and all_live
        ),
        f"arm S carries exactly {N_STAGE_PARAMETERS} parameters more than arm A, is the "
        "identity at initialisation, and every one of the four has a non-zero gradient there",
        {
            "n_parameters_arm_a": counts_a["total"],
            "n_parameters_arm_s": counts_s["total"],
            "delta": delta,
            "extra_tensors": extra,
            "missing_tensors": missing,
            "identity_at_init": identity_at_init,
            "init_gradients": grads,
            "all_four_gradients_live": all_live,
        },
    )


def check_stage_covariate_is_one_global_scalar_of_x_t() -> Check:
    """[S1] The covariate is a function of ``x_t`` ALONE, and of its MASS alone.

    Three properties, each of which a plausible mis-implementation would break:

    * **Permutation invariance.** Move the same burned mass anywhere on the grid
      and ``z`` is bit-identical. This is what makes it ONE scalar rather than a
      smuggled spatial field; a head that read, say, a pooled feature map would
      fail it.
    * **Strict monotonicity in area**, so the covariate can express "older, and
      therefore bigger" at all, and ``log1p`` so an EMPTY state is finite. C1.1
      records 6-37% of frames with an empty state-1 set; ``log(0) = -inf`` would
      take the whole step's hazard with it on exactly those frames.
    * **It moves within a rollout.** ``spatial_log_intensity_field`` is
      recomputed per step from the state that step reached, and the stage
      covariate must be too, or "stage" would freeze at ``t0`` and the arm would
      be testing a per-window constant instead of a state covariate.

    The last one is checked on the KERNEL, not on the head, because the head
    being correct in isolation says nothing about where the kernel calls it.
    """
    import torch

    from wildfire_nowcast.model.stagehead import STAGE_CENTRE, STAGE_SCALE, StageHead

    head = StageHead()
    grid = torch.zeros(1, 8, 8)
    grid[0, :2, :3] = 1.0
    moved = torch.zeros(1, 8, 8)
    moved[0, 5:7, 4:7] = 1.0  # same six cells, elsewhere
    permutation_invariant = bool(torch.equal(head.covariate(grid), head.covariate(moved)))

    areas = [0.0, 1.0, 10.0, 1000.0, 100000.0]
    zs = []
    for area in areas:
        field = torch.zeros(1, 400, 400)
        field.reshape(1, -1)[0, : int(area)] = 1.0
        zs.append(float(head.covariate(field)[0]))
    monotone = all(a < b for a, b in zip(zs[:-1], zs[1:], strict=True))
    empty_is_finite = bool(zs[0] == (0.0 - STAGE_CENTRE) / STAGE_SCALE and all(v == v for v in zs))

    # ... and the kernel calls it on the state each step actually reached.
    with torch.no_grad():
        small = torch.zeros(1, 6, 6)
        small[0, 3, 3] = 1.0
        big = torch.zeros(1, 6, 6)
        big[0, 1:5, 1:5] = 1.0
        head.log_amplitude_coeff[0] = -0.5
        amp_small = float(head(small)[0][0])
        amp_big = float(head(big)[0][0])
    responds_to_state = amp_big < amp_small

    return Check(
        "stage_covariate_is_one_global_scalar_of_x_t",
        bool(permutation_invariant and monotone and empty_is_finite and responds_to_state),
        "z depends on the burned MASS and on nothing else about where it is, is strictly "
        "increasing in area, is finite on an empty state, and moves when the state grows",
        {
            "permutation_invariant": permutation_invariant,
            "z_by_area": dict(zip([str(a) for a in areas], zs, strict=True)),
            "monotone_in_area": monotone,
            "empty_state_z": zs[0],
            "log_amplitude_small_state": amp_small,
            "log_amplitude_large_state": amp_big,
        },
    )


def check_arm_a_reloads_from_a_pre_s1_spec_without_a_stage_head() -> Check:
    """[S1] Absence is arm A. The archived-checkpoint rule, applied once more.

    Every flag this kernel has acquired follows the same rule (``latent_config``,
    ``mean_preserving``, ``spatial_modes``, ``gate_mean_preserving``): a spec
    written before the flag existed must reload as the model it was FITTED as,
    never acquire the new component. If ``stage_scalar`` defaulted to True on a
    missing key, every archived G2/M5-M10 checkpoint would silently become arm S
    on reload and every historical number would be attributed to the wrong model.

    Checked both ways, because only the round trip can fail: a spec with the key
    absent yields ``stage is None``, and a FITTED arm S round-trips its four
    coefficients bit-exactly through JSON.
    """
    import json

    import torch

    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig

    arm_s = ContagionKernel(KernelConfig(stage_scalar=True))
    with torch.no_grad():
        arm_s.stage.log_amplitude_coeff.copy_(torch.tensor([-0.37, 0.11, -0.02]))
        arm_s.stage.log_reach_coeff.copy_(torch.tensor([0.29]))
    spec = json.loads(json.dumps(arm_s.to_spec()))
    reloaded = ContagionKernel.from_spec(spec)
    round_trips = bool(
        reloaded.stage is not None
        and torch.equal(reloaded.stage.log_amplitude_coeff, arm_s.stage.log_amplitude_coeff)
        and torch.equal(reloaded.stage.log_reach_coeff, arm_s.stage.log_reach_coeff)
    )

    pre_s1 = json.loads(json.dumps(arm_s.to_spec()))
    pre_s1["config"].pop("stage_scalar")
    pre_s1["parameters"] = {
        k: v for k, v in pre_s1["parameters"].items() if not k.startswith("stage.")
    }
    absent = ContagionKernel.from_spec(pre_s1)
    absence_is_arm_a = absent.stage is None

    # POSITIVE CONTROL: the round-trip clause must be able to fail.
    tampered = json.loads(json.dumps(arm_s.to_spec()))
    tampered["parameters"]["stage.log_reach_coeff"] = [0.29 + 1e-3]
    tampered_model = ContagionKernel.from_spec(tampered)
    control_differs = not torch.equal(
        tampered_model.stage.log_reach_coeff, arm_s.stage.log_reach_coeff
    )

    return Check(
        "arm_a_reloads_from_a_pre_s1_spec_without_a_stage_head",
        bool(round_trips and absence_is_arm_a and control_differs),
        "a fitted arm S round-trips its four coefficients through JSON, a spec with no "
        "`stage_scalar` key reloads as ARM A, and a tampered coefficient is detected",
        {
            "round_trips": round_trips,
            "absence_is_arm_a": absence_is_arm_a,
            "positive_control_detects_tamper": control_differs,
            "stage_report": arm_s.stage.report(),
        },
    )


# --------------------------------------------------------------------------
# M19 - the collapse positive control, and what its index is actually null to
# --------------------------------------------------------------------------


def check_dispersion_index_null_is_exact_for_a_one_step_field() -> Check:
    """[M19] The index reads 1.0 where 1.0 is provably the answer, and only there.

    Two known answers, both forced by algebra rather than by a previous run:

    * a field of INDEPENDENT Bernoullis has ``Var(area) = sum p_i (1 - p_i)``,
      which is exactly what the index divides by, so it reads 1.0;
    * the stub with ``latent_sigma=0`` over ONE step is such a field, because
      given a frozen state each candidate cell ignites on its own draw.

    The second is the one that matters: it separates "the model has no shared
    innovation" from "the index has a bug", and it fails only if the scene, the
    sampler or the instrument has moved. Measured over 12 seeds at 384 members,
    both sit within 1 SEM of 1.0.
    """
    from wildfire_nowcast.eval.collapse_curve import (
        constructed_independent_index,
        stub_index,
        summarise,
    )

    constructed = [constructed_independent_index(900, 384, seed) for seed in range(12)]
    one_step = [stub_index(30, 1, 384, seed, scaling="scaled") for seed in range(12)]
    rows = {row.family: row for row in summarise([*constructed, *one_step])}
    out = {name: {"index": row.index_mean, "sd": row.index_sd} for name, row in rows.items()}
    ok = all(abs(row.index_mean - 1.0) < 0.10 for row in rows.values())
    return Check(
        "dispersion_index_null_is_exact_for_a_one_step_field",
        ok,
        "an independent-by-construction field and the stub at latent_sigma=0 over ONE step "
        "must both read 1.0, because that is the estimand the index divides by",
        out,
    )


def check_dispersion_index_null_is_NOT_one_for_a_multi_step_field() -> Check:
    """[M19] The same control over 3 h reads far above 1.0, with no latent anywhere.

    THIS CHECK PINS A DEFECT, DELIBERATELY. ``latent_sigma=0`` removes the shared
    innovation entirely, so a reader of ``COLLAPSE_INDEX_THRESHOLD`` expects a
    number near 1. Over three steps the same ensemble reads ~1.4-1.5, because the
    step-2 candidate set is a FUNCTION of the step-1 draw: contagion correlates
    the indicators the index assumes are independent. The excess is the dynamics,
    not the latent and not the sample size.

    It is asserted as a strict inequality against the ONE-step measurement in the
    check above rather than against a pinned constant, so it states a relation
    that stays true after any fix and does not have to be re-tuned when the scene
    changes. If a future change makes the multi-step null equal the one-step null,
    this check fails and that failure is the news.
    """
    from wildfire_nowcast.eval.collapse_curve import stub_index

    one_step = float(np.mean([stub_index(30, 1, 192, s, scaling="scaled").index for s in range(8)]))
    three_step = float(
        np.mean([stub_index(30, 3, 192, s, scaling="scaled").index for s in range(8)])
    )
    six_step = float(np.mean([stub_index(30, 6, 192, s, scaling="scaled").index for s in range(8)]))
    ok = one_step < three_step < six_step and three_step > 1.25
    return Check(
        "dispersion_index_null_is_NOT_one_for_a_multi_step_field",
        ok,
        "the independent-pixel null of 1.0 is exact for one step ONLY; over 3 h and 6 h the "
        "same latent-free ensemble reads well above it, so a fixed threshold on this index "
        "is partly a threshold on horizon",
        {"h1": one_step, "h3": three_step, "h6": six_step},
    )


def check_dispersion_index_level_is_free_of_the_member_count() -> Check:
    """[M19] More members shrink the SPREAD of the index and not its LEVEL.

    The obvious suspect for a 2% miss on a 24-member control is that a ratio of
    variances from 24 samples is biased. It is not, and the reason is checkable:
    the numerator is ``areas.std()`` with ``ddof=0``, low by ``sqrt((M-1)/M)``,
    and the denominator uses the EMPIRICAL marginals from the same M members,
    for which ``E[p_hat(1 - p_hat)] = p(1 - p)(M-1)/M``, low by the same factor.
    A ratio carrying the same M-factor in both halves is M-free to first order.

    So this asserts the shape rather than a value: the cell MEAN moves by less
    than the seed SD across a 16x change in members, while the seed SD itself
    falls by at least 2.5x. A member count cannot buy a control that the level
    does not already give.
    """
    from wildfire_nowcast.eval.collapse_curve import stub_index, summarise

    rows = {
        m: summarise([stub_index(30, 3, m, s, scaling="scaled") for s in range(12)])[0]
        for m in (24, 384)
    }
    level_shift = abs(rows[384].index_mean - rows[24].index_mean)
    sd_ratio = rows[24].index_sd / rows[384].index_sd
    ok = level_shift < rows[384].index_sd * 2 and sd_ratio > 2.5
    return Check(
        "dispersion_index_level_is_free_of_the_member_count",
        ok,
        "16x the members moves the index LEVEL by less than 2 seed-SD while cutting the "
        "seed SD by more than 2.5x: members buy precision, not a verdict",
        {
            "index_at_24": rows[24].index_mean,
            "index_at_384": rows[384].index_mean,
            "level_shift": level_shift,
            "sd_at_24": rows[24].index_sd,
            "sd_at_384": rows[384].index_sd,
            "sd_ratio": sd_ratio,
        },
    )


def check_no_firing_configuration_is_reported_as_a_nearest_miss() -> Check:
    """[M19] "nothing fired" and "the cheapest thing that fired" are different values.

    ADR-104's rule applied to this module's own read-off. A control that fires on
    11 of 12 seeds has not been shown to fire, and
    :func:`smallest_firing_configuration` must return ``None`` for that case
    rather than the nearest miss, because a nearest miss rendered in the
    "smallest configuration" row is indistinguishable from a control that works.
    Both directions are exercised: an all-firing summary returns a row, and the
    same summary with one seed removed from the collapsed count returns None.
    """
    from dataclasses import replace

    from wildfire_nowcast.eval.collapse_curve import (
        CellSummary,
        smallest_firing_configuration,
    )

    fires = CellSummary(
        family="f",
        domain=30,
        horizon_h=3,
        members=24,
        n_seeds=12,
        index_mean=1.2,
        index_sd=0.1,
        index_min=1.0,
        index_max=1.4,
        n_collapsed=12,
        area_mean=40.0,
    )
    nearly = replace(fires, n_collapsed=11)
    found = smallest_firing_configuration([fires], family="f", horizon_h=3)
    missed = smallest_firing_configuration([nearly], family="f", horizon_h=3)
    ok = found is not None and missed is None
    return Check(
        "no_firing_configuration_is_reported_as_a_nearest_miss",
        ok,
        "12/12 seeds returns a configuration and 11/12 returns None: a control that nearly "
        "fires must not be printed in the row that says it fires",
        {"all_seeds": found is not None, "one_seed_short": missed is None},
    )


CHECKS: tuple[Callable[[], Check], ...] = (
    # C6
    check_perfect_forecast,
    check_brier_hand_computed,
    check_crps_fair_known_answer,
    check_arrival_times_censoring,
    check_dispersion_calibrated_is_one,
    check_collapse_is_invisible_to_dispersion_ratio,
    check_fuzzy_iou_reduces_to_jaccard,
    check_aggregate_pools_sufficient_statistics,
    check_zero_growth_window_is_free_for_persistence,
    # C5
    check_persistence_is_identity,
    check_input_validation_rejects_traps,
    check_ellipse_respects_barriers,
    check_ellipse_is_anisotropic_downwind,
    check_ellipse_ensemble_has_area_spread,
    check_ellipse_seeds_from_frontier_not_state_one,
    check_forecast_window_time_phase,
    check_end_to_end_on_synthetic,
    # the deterministic contagion kernel (M2)
    check_kernel_torch_physics_matches_numpy,
    check_kernel_init_is_the_ellipse_shape,
    check_kernel_samples_are_absorbing,
    check_kernel_ensemble_collapses_in_area,
    check_kernel_growth_increases_with_wind,
    check_kernel_respects_barriers_and_nonburnable,
    # M3 - ADR-015 (6a) the gradient defect, (6b) the label-noise ensemble,
    # (3) per-horizon adjudication, (4) C8 split fingerprint
    check_susceptibility_has_gradient,
    check_susceptibility_is_an_exact_log_offset,
    check_fconf_cannot_move_the_burned_set,
    check_label_perturbation_preserves_absorbing_order,
    check_c8_rejects_a_stale_or_unstamped_split,
    check_best_member_iou_by_horizon_is_consistent,
    # M5 - the shared per-step latent z_t, its ensemble, and the G3 ablation
    check_latent_off_reproduces_the_g2_kernel_bitwise,
    check_shared_latent_is_constant_across_pixels,
    check_independent_noise_ablation_collapses_in_area,
    # M21 - what that bar is worth. Its null LEVEL is lead-invariant and its
    # TAIL is not, and the closed form for the sibling one-step bar in sim/.
    check_ablation_sd_ratio_null_is_one_at_every_lead,
    check_ablation_sd_ratio_bar_is_calibrated_at_the_lead_it_runs_at,
    check_one_step_index_bar_derivation_matches_the_shipped_index,
    # M22 - the arm G3 (d) is about, obtainable the way a C5 consumer obtains
    # any model, and the per-lead form of G3's OTHER half
    check_the_ablation_arm_is_loadable_by_name,
    check_a_look_alike_ablation_is_refused_and_a_vacuous_one_cannot_be_scored,
    check_the_tracked_reference_fit_is_a_latent_bearing_c5_address,
    check_area_dispersion_by_horizon_recombines_to_the_pooled_criterion,
    check_elbo_kl_is_scaled_like_its_reconstruction_term,
    check_latent_spec_round_trips_and_absence_means_absence,
    # M8 - the MEAN-PRESERVING ACTIVITY GATE (reverses the M6 exemption), and
    # the units defect simviz raised against C6's own dispersion decomposition
    check_debiased_dispersion_is_in_the_same_units_as_the_criterion,
    check_gate_mean_preserving_is_off_by_default_and_exact_when_on,
    # M11 - the degradation ladder and the power read-off that stands on it
    check_degradation_null_rung_is_bitwise_the_undegraded_forecast,
    check_degradation_rungs_hit_their_declared_severity,
    check_base_prediction_cache_cannot_return_another_window,
    check_mde_read_off_requires_a_SUSTAINED_crossing,
    # L1 - the label-noise floor: the morphology, the severity ladder, the oracle
    check_square_dilation_iou_matches_its_closed_form,
    check_calibration_target_and_gate_criterion_are_different_estimands,
    check_severity_sampler_is_the_shipped_observation_noise,
    check_noise_oracle_null_severity_is_the_labels_exactly,
    check_window_table_refuses_a_key_collision,
    check_the_official_perimeter_endpoint_has_not_drifted,
    # U0 - the stage_decay channel: known-beta recovery, the agreement case, the
    # order reversal against the published statistic, the power identity, and the
    # C6 registry call site that ADR-059 (5) found missing from eval/
    check_stage_decay_recovers_a_known_beta,
    check_stage_decay_agrees_when_it_should_and_reverses_the_published_order,
    check_stage_decay_separation_cannot_be_bought_by_closing_more_of_the_gap,
    check_stage_decay_asks_the_registry_instead_of_remembering,
    check_stage_ladder_severity_is_not_computed_by_the_channel,
    # U0b - the criterion ADR-060 (7) item 2 replaced the effect size with
    check_stage_sign_criterion_is_exact_and_the_estimand_is_the_licensed_one,
    # S1 - arm S's capacity, its covariate, and its archived-spec behaviour
    check_arm_s_is_arm_a_plus_four_parameters_and_starts_there,
    check_stage_covariate_is_one_global_scalar_of_x_t,
    check_arm_a_reloads_from_a_pre_s1_spec_without_a_stage_head,
    # M19 - what the collapse positive control's index is null to, and what it
    # is not. The threshold in sim/ensemble.py is READ by these checks and is
    # not restated, moved or argued with here.
    check_dispersion_index_null_is_exact_for_a_one_step_field,
    check_dispersion_index_null_is_NOT_one_for_a_multi_step_field,
    check_dispersion_index_level_is_free_of_the_member_count,
    check_no_firing_configuration_is_reported_as_a_nearest_miss,
)


def run_all(checks: Sequence[Callable[[], Check]] = CHECKS) -> list[Check]:
    """Run every known-answer check; an exception is a failure, never a crash.

    [M19] A CRASHED CHECK MUST CARRY THE SAME NAME AS A PASSING ONE. Every
    ``check_x`` returns ``Check("x", ...)``, so building the failure record from
    ``fn.__name__`` filed it under ``"check_x"`` instead. A consumer indexing the
    ``--json`` output by name therefore found the check present while it passed
    and ABSENT the moment it started raising - the one state in which someone is
    looking for it. Found by a discrimination plant whose runner keyed on the
    name and got a ``KeyError`` rather than a False.
    """
    results: list[Check] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 - a raising check is a failing check
            results.append(
                Check(fn.__name__.removeprefix("check_"), False, f"{type(exc).__name__}: {exc}")
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.eval.selftest",
        description="Known-answer verification for the C5 baselines and C6 metrics.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    add_logging_arguments(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    # ADR-103: configured here and nowhere else. --json prints a document a
    # caller parses, so a diagnostic must not be able to reach stdout beside it;
    # configure_logging enforces that by refusing sys.stdout.
    configure_from_args(args)

    results = run_all()
    failed = [c for c in results if not c.passed]
    if args.json:
        print(json.dumps([c.__dict__ for c in results], indent=2, default=float))
    else:
        for check in results:
            flag = "PASS" if check.passed else "FAIL"
            print(f"[{flag}] {check.name}: {check.detail}")
            for key, value in check.values.items():
                print(f"         {key}: {value}")
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
        # Read the C3.3 state rather than hardcoding it: this banner was stale
        # within a day of being written, and a stale disclaimer is worse than
        # none because it invites people to stop reading disclaimers.
        status = reporting_status()
        print(
            "NOTE: these are known-answer verifications of the plumbing on SYNTHETIC "
            "and hand-built inputs. No number printed here is a held-out result, "
            "whatever C3.3 says."
        )
        print(
            f"C3.3: n_train_blocks={status.get('n_train_blocks')} -> "
            f"reporting {'UNLOCKED' if status['reportable'] else 'BLOCKED'} "
            f"({status['reason']})"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
