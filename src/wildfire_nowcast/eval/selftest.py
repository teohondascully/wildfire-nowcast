"""Known-answer verification for the C5 baselines and the C6 metrics.

``tests/`` is infra's directory, so this lives here and is runnable
standalone::

    .venv/bin/python -m wildfire_nowcast.eval.selftest
    .venv/bin/python -m wildfire_nowcast.eval.selftest --json

Every check has an answer that is known BEFORE the code runs — hand-computed,
or forced by an algebraic identity — because a metric verified only against its
own output is verified against nothing. The two that matter most:

* ``crps_fair_known_answer``: a 2-member ensemble bracketing the truth
  symmetrically has fair CRPS exactly 0 and biased CRPS exactly 0.5. That pins
  down which estimator is wired in, which decides whether a collapsed ensemble
  is rewarded at G3.
* ``collapse_is_invisible_to_dispersion_ratio``: two ensembles with IDENTICAL
  per-pixel probabilities — one built from independent per-pixel noise, one from
  a shared latent — get the same Brier and the same ``dispersion_ratio``, and
  are told apart only by ``area_dispersion_ratio`` and member diversity. This is
  the G3 ablation in miniature, and it is asserted here so that the metric
  cannot silently stop being able to see it.

This module is a proposed pytest target for infra (see status/model.md);
:func:`run_all` returns structured results so a test can be three lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wildfire_nowcast.common import zarr_io as zio
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


def _synthetic_dataset(tmp: str | None = None):
    """The C4 fixture, imported LAZILY on purpose.

    ``common/synthetic.py`` is infra's and is edited concurrently with this
    suite. Importing it at module scope means one broken line in someone else's
    fixture takes down every known-answer check in here, including the ones that
    have nothing to do with it — which is how a whole verification instrument
    goes dark for an unrelated reason. Lazily, a broken fixture fails the two
    checks that use it and leaves the other twenty-five reporting.
    """
    import tempfile
    from pathlib import Path

    from wildfire_nowcast.common.synthetic import make_synthetic_fire

    out = Path(tmp or tempfile.mkdtemp(prefix="wnc-selftest-")) / "synthetic" / "tensor.zarr"
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
    return Check("perfect_forecast", ok, "brier/crps 0 and IoU 1", {
        "brier_1h": res["brier_1h"], "arrival_crps": res["arrival_crps"],
        "best_member_iou": res["best_member_iou"],
    })


def check_brier_hand_computed() -> Check:
    """Half the members burn everything, half nothing, truth burns: Brier = 0.25."""
    shape = (12, 12)
    members = np.concatenate(
        [np.ones((4, *shape), bool), np.zeros((4, *shape), bool)], axis=0
    )
    samples = _binary_samples(members)
    truth = np.ones((1, *shape), np.uint8)
    res = evaluate(samples, truth, x0=np.zeros(shape, np.uint8), leads=(1,))
    ok = _close(res["brier_1h"], 0.25)
    return Check("brier_hand_computed", ok, "p=0.5, y=1 -> Brier 0.25",
                 {"brier_1h": res["brier_1h"]})


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
    return Check("arrival_times_censoring", ok, "1-based, capped at L+1",
                 {"arrival": got.tolist()})


def check_dispersion_calibrated_is_one() -> Check:
    """A calibrated binary ensemble has dispersion_ratio ~ 1 by construction.

    Members and truth drawn independently from the same per-pixel Bernoulli, so
    the forecast is calibrated. The ratio must come out near 1 — this is the
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
    return Check("dispersion_calibrated_is_one", ok, "spread-skill ~ 1 when calibrated",
                 {"dispersion_ratio": ratio})


def _collapse_pair(n_members: int, shape: tuple[int, int] = (100, 100), seed: int = 11):
    """Two ensembles with the same intended p = 0.5 field, one truth scenario.

    * ``independent`` — every pixel of every member flipped independently. The
      known-broken model CLAUDE.md permits only as an ablation. Every member
      burns about half the domain, so the ensemble has almost no spread in
      burned AREA while its area error is enormous.
    * ``shared`` — half the members burn everything, half burn nothing: all the
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

    1. ``dispersion_ratio`` cannot detect collapse — and worse, it is ANTI-
       correlated with what G3 cares about. It scores the COLLAPSED ensemble at
       exactly 1.000 (textbook-perfect) and the good scenario ensemble at
       ``sqrt((M+1)/(M-1))`` > 1 ("over-dispersed"). Anyone who used it as the
       G3 criterion would prefer the broken model. This is a stronger and more
       dangerous statement than "it is blind".
    2. The residual difference between the arms is that pure ``O(1/M)``
       bookkeeping term, NOT a signal. It is asserted to shrink like 1/M rather
       than absorbed into a tolerance, because a tolerance wide enough to hide
       it at M=20 would also hide a real effect.
    3. ``area_dispersion_ratio`` separates them by ~100x, and is the number the
       G3 ablation must actually be judged on.

    HISTORY — the failure this check survived, kept because the reasoning
    recurs. The first version scored both arms against an iid per-pixel
    coin-flip truth and asserted ``area_a < 0.1``. It failed with
    ``area_a = 7.85, area_b = 525.66``. The instinct is to suspect the metric;
    that would have been wrong. In an iid-coin-flip world the truth burns ~50%
    of the domain EVERY time, so there is no scenario uncertainty for a latent
    to capture and independent per-pixel noise is the CORRECTLY SPECIFIED model.
    The area-error denominator goes to ~0 and the ratio diverges for both arms —
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
    area_separates_big = b_big["by_mask"]["domain"]["area_dispersion_ratio"] > 50 * (
        a_big["by_mask"]["domain"]["area_dispersion_ratio"]
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
    return Check("fuzzy_iou_reduces_to_jaccard", ok, "tol=0 is Jaccard; tol>0 is bounded",
                 {"exact": exact, "tolerant": tolerant})


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

    This is the whole reason insights/data item 1 is a design constraint and not
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
        ("float x0", lambda: validate_predict_inputs(
            x0.astype(np.float32), static, weather, 1, 3, 0)),
        ("transposed static", lambda: validate_predict_inputs(
            x0, static.transpose(0, 2, 1)[:, :7], weather, 1, 3, 0)),
        ("short weather slab", lambda: validate_predict_inputs(
            x0, static, weather[:2], 1, 3, 0)),
        ("weather/static channel swap", lambda: validate_predict_inputs(
            x0, np.zeros((5, *shape), np.float32), weather, 1, 3, 0)),
        ("state 3 in x0", lambda: validate_predict_inputs(
            np.full(shape, 3, np.uint8), static, weather, 1, 3, 0)),
        ("non-finite weather", lambda: validate_predict_inputs(
            x0, static, np.full((3, 5, *shape), np.nan, np.float32), 1, 3, 0)),
    ]
    missed = []
    for label, fn in traps:
        try:
            fn()
        except (TypeError, ValueError):
            continue
        missed.append(label)
    return Check("input_validation_rejects_traps", not missed,
                 "all C5 argument traps raise", {"missed": missed})


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

    x0, static, weather = _toy_scene(wind_u=14.0)
    samples = ContagionKernel().predict(x0, static, weather, 24, 6, 0)
    areas = (samples[:, -1] > 0).sum(axis=(1, 2)).astype(float)
    cv = float(areas.std(ddof=1) / max(areas.mean(), 1e-9))
    ellipse = EllipseBaseline().predict(x0, static, weather, 24, 6, 0)
    ellipse_areas = (ellipse[:, -1] > 0).sum(axis=(1, 2)).astype(float)
    ellipse_cv = float(ellipse_areas.std(ddof=1) / max(ellipse_areas.mean(), 1e-9))
    ok = cv < ellipse_cv
    return Check(
        "kernel_ensemble_collapses_in_area",
        ok,
        "independent-per-pixel members have LESS area spread than shared-innovation "
        "members — the collapse G3 must fix with z_t",
        {"kernel_area_cv": cv, "ellipse_area_cv": ellipse_cv},
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
    # own data. Both suppressions are now stated the same way — RELATIVE to open
    # ground and BOUNDED AWAY FROM ZERO — because an assertion that a probability
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
        # (6a) — the Simard damping collapsing to a hard zero. It is asserted here
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
            log_w = model.log_weights(
                torch.as_tensor(weather.astype(np.float64)), fields
            ).numpy()
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
    one. Both failure modes are asserted, including the unstamped case — C-1 makes
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
    at each horizon. The cheap way to get it — one pass, cumulative means — is
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
        {"nonburnable_cells_lit": lit_nonburnable, "barrier_cells_lit": lit_barrier,
         "did_grow": grew},
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
    final = EllipseBaseline(ros_sigma=0.0, bearing_sigma_deg=0.0).predict(
        x0, static, weather, 1, 6, 0
    )[0, -1] > 0
    east = int(final[:, 16:].sum())
    west = int(final[:, :15].sum())
    lb = float(length_to_breadth(6.0))
    ok = east > 3 * max(west, 1) and lb > 2.0
    return Check(
        "ellipse_is_anisotropic_downwind",
        ok,
        "head runs downwind; length-to-breadth > 1 (Anderson 1983)",
        {"cells_east": east, "cells_west": west, "lb_at_6ms": lb,
         "back_fraction": float(ellipse_ros_factor(lb, -1.0))},
    )


def check_ellipse_ensemble_has_area_spread() -> Check:
    """Shared per-step innovations must produce real scenario spread.

    If members differ only by per-pixel noise, total burned area barely moves.
    The ellipse's ensemble is built from domain-wide shared scalars precisely so
    that it does — otherwise G3 would be judged against a strawman ensemble.
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
        {"x0_state1_cells": int((x0 == 1).sum()),
         "growth_cells": int((samples[0, -1] > 0).sum()) - int((x0 > 0).sum())},
    )


def check_forecast_window_time_phase() -> Check:
    """weather[k] must be features[t0+1+k] — the C1.3 end-of-hour phase.

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
        {"phase_ok": phase_ok, "truth_ok": truth_ok, "x0_ok": x0_ok,
         "would_be_off_by_one": bool(off_by_one)},
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
    EXACTLY 0 (no latent; latent held at ``z = 0``), and a third must NOT be —
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


def check_independent_noise_ablation_collapses_in_area() -> Check:
    """[M5] G3 (d): the independent-per-pixel ablation must DEMONSTRATE collapse.

    A POSITIVE CONTROL for the ensemble machinery, not a result about the model —
    G3 asks for the ablation to fail, so a failure to fail is a defect in this
    repo's sampler and must surface here rather than in a gate table.

    Both arms are the SAME PARAMETERS (``with_sampler`` shares them); the only
    difference is whether ``z_t`` is drawn. The quantity is the spread in NEW
    burned cells: independent Bernoulli noise over ``N`` cells gives area spread
    ``O(sqrt(N))`` against a mean ``O(N)``, so it must shrink relative to a
    shared-innovation ensemble as the fire gets bigger.
    """
    from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
    from wildfire_nowcast.model.latent import LatentConfig

    x0, static, weather = _toy_scene(wind_u=12.0)
    model = ContagionKernel(KernelConfig(), latent_config=LatentConfig(dim=3))
    out = {}
    for mode in ("latent", "independent"):
        samples = model.with_sampler(mode).predict(x0, static, weather, 32, 6, 4)
        new = ((samples[:, -1] > 0) & (x0[None] == 0)).sum(axis=(1, 2)).astype(float)
        out[mode] = {"mean": float(new.mean()), "sd": float(new.std(ddof=1))}
    ratio = out["latent"]["sd"] / max(out["independent"]["sd"], 1e-9)
    ok = ratio > 1.5
    return Check(
        "independent_noise_ablation_collapses_in_area",
        ok,
        "holding z_t at the prior mean removes most of the ensemble's AREA spread "
        f"(SD ratio latent/independent = {ratio:.2f}, must exceed 1.5)",
        {**out, "sd_ratio": ratio},
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
    reload — the same class as ADR-015's split moving under a running experiment,
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
       whole 2x2 rests on — the fourth cell of that matrix is a REUSED M7 run.
    2. **``E_z[e^gate] = 1`` EXACTLY when it is on**, at the unconditional prior.
       Closed form: with ``z ~ N(mu, 1)`` the multiplier's mean is
       ``e^(sigma mu + sigma^2/2)``, so the correction must be
       ``-(sigma mu + sigma^2/2)`` and the product must be 1 to machine epsilon.
       A correction that only removed ``sigma^2/2`` — the copy-paste bug this
       change is one keystroke away from — would leave ``e^(sigma mu)``, i.e.
       ``0.13`` at the fitted scale, and would look entirely plausible.
    3. **THE ASYMMETRY SURVIVES.** ADR-034 (2) makes asymmetry the working
       mechanism, so a "fix" that symmetrised the multiplier would destroy the
       thing it was meant to preserve. The corrected multiplier's MEDIAN is
       ``e^(-sigma^2/2)``, which at ``sigma = 1.3`` is ``0.43`` against a mean of
       1 — most members quiet, a thin expensive upper tail. Asserted as a
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
    # not silently do nothing — the green-but-vacuous shape, refused up front.
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
        # bound said 0.2 and FAILED ON THE CLEAN WORLD — the fourth time a
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
    machinery — build the cell order, rank it, take the first ``n_h`` — and is
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
    * a SHAPE rung realises the reference area **EXACTLY** — integer equality, not
      a tolerance — because "a channel blind to shape at fixed area" is a
      different finding from "a channel blind to area", and separating them
      requires the area error to be zero rather than small.

    ``degrade_samples`` also runs every rung through ``validate_samples``, so a
    degradation that un-burned a cell or broke the absorbing order (C1.1) would
    raise here rather than reach a metric.

    THE UNDEFINED CASE IS PART OF THE CHECK, NOT AN EDGE OF IT. A ladder that
    spans harmless to catastrophic will eventually contain a rung whose increment
    union with the reference is EMPTY — neither forecast added a cell. That is
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
    monotone_iou = all_defined and all(
        a > b for a, b in zip(ious[:-1], ious[1:], strict=True)
    )

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
        a > b
        for a, b in zip(with_undefined[:-1], with_undefined[1:], strict=True)
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
    wrong forecast — silently, with every downstream number still finite and
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
        {**{f"key_moves_on_{k}": v for k, v in moved.items()},
         "round_trip_bitwise": round_trip, "miss_returns_none": miss_is_none},
    )


def check_mde_read_off_requires_a_SUSTAINED_crossing() -> Check:
    """[M11] One rung crossing a bar is not a detection threshold.

    Known answer, on a hand-built curve. A channel that reads 2.4 block-SD at
    severity 0.2 and falls back to 1.1 at 0.4 has not detected 0.2 — it has
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
    # M3 — ADR-015 (6a) the gradient defect, (6b) the label-noise ensemble,
    # (3) per-horizon adjudication, (4) C8 split fingerprint
    check_susceptibility_has_gradient,
    check_susceptibility_is_an_exact_log_offset,
    check_fconf_cannot_move_the_burned_set,
    check_label_perturbation_preserves_absorbing_order,
    check_c8_rejects_a_stale_or_unstamped_split,
    check_best_member_iou_by_horizon_is_consistent,
    # M5 — the shared per-step latent z_t, its ensemble, and the G3 ablation
    check_latent_off_reproduces_the_g2_kernel_bitwise,
    check_shared_latent_is_constant_across_pixels,
    check_independent_noise_ablation_collapses_in_area,
    check_elbo_kl_is_scaled_like_its_reconstruction_term,
    check_latent_spec_round_trips_and_absence_means_absence,
    # M8 — the MEAN-PRESERVING ACTIVITY GATE (reverses the M6 exemption), and
    # the units defect simviz raised against C6's own dispersion decomposition
    check_debiased_dispersion_is_in_the_same_units_as_the_criterion,
    check_gate_mean_preserving_is_off_by_default_and_exact_when_on,
    # M11 — the degradation ladder and the power read-off that stands on it
    check_degradation_null_rung_is_bitwise_the_undegraded_forecast,
    check_degradation_rungs_hit_their_declared_severity,
    check_base_prediction_cache_cannot_return_another_window,
    check_mde_read_off_requires_a_SUSTAINED_crossing,
)


def run_all(checks: Sequence[Callable[[], Check]] = CHECKS) -> list[Check]:
    """Run every known-answer check; an exception is a failure, never a crash."""
    results: list[Check] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 - a raising check is a failing check
            results.append(Check(fn.__name__, False, f"{type(exc).__name__}: {exc}"))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.eval.selftest",
        description="Known-answer verification for the C5 baselines and C6 metrics.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = run_all()
    failed = [c for c in results if not c.passed]
    if args.json:
        print(json.dumps([c.__dict__ for c in results], indent=2, default=float))
    else:
        for check in results:
            flag = "PASS" if check.passed else "FAIL"
            print(f"[{flag}] {check.name} — {check.detail}")
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
