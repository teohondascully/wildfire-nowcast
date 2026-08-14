"""THE MODELS THE HARNESS SCORES — every one synthesised from labels alone.

There is no checkpoint here, no import of ``model/`` and no run directory. The
harness cannot see which model is "ours", because none of them is: that is what
lets a gate's instrument be fixed BEFORE the result exists.

Three groups, and the membership of each is the argument:

* the DEGENERATES that must never win — two do-nothing nulls, the fitted
  climatology (information-free rather than merely silent), and the collapse
  ablation;
* the SKILL REFERENCES a degenerate must be measurably worse than;
* the ORACLE, which nothing may match.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from wildfire_nowcast.common.null_check.registry import FAMILY_SPREAD, MetricSpec
from wildfire_nowcast.common.null_check.windows import Window

Forecaster = Callable[[Window, int, np.random.Generator], np.ndarray]


def _as_samples(member_burned: np.ndarray) -> np.ndarray:
    """Boolean ``[M, L, H, W]`` -> C5 ``uint8`` fire_state (1 = burned)."""
    return np.where(np.asarray(member_burned, dtype=bool), 1, 0).astype(np.uint8)


def null_zero_ignition(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """PERSISTENCE. Ignites exactly zero new cells. ADR-017's null of record."""
    burned = window.x0 > 0
    n_lead = int(window.truth.shape[0])
    return _as_samples(np.broadcast_to(burned, (n_members, n_lead, *burned.shape)))


def null_empty(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Predicts NOTHING AT ALL — not even the fire that is already burning."""
    n_lead = int(window.truth.shape[0])
    return np.zeros((n_members, n_lead, *window.x0.shape), dtype=np.uint8)


def oracle(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Predicts the truth exactly. Nothing may outrank this on any metric."""
    truth = np.asarray(window.truth) > 0
    return _as_samples(np.broadcast_to(truth, (n_members, *truth.shape)))


def _growth_candidates(window: Window) -> tuple[np.ndarray, np.ndarray]:
    """``(true new cells at the final lead, plausible-but-wrong cells)``."""
    from wildfire_nowcast.common.states import dilate  # noqa: PLC0415

    burned = window.x0 > 0
    truth_final = (np.asarray(window.truth[-1]) > 0) & ~burned
    reach = dilate(burned, 3) & ~burned
    return truth_final, reach & ~truth_final


def _skillful_members(
    window: Window,
    n_members: int,
    rng: np.random.Generator,
    *,
    recall: float,
    false_alarm: float,
    shared_latent: bool,
) -> np.ndarray:
    """Genuine but imperfect skill: partial recall plus realistic false alarms.

    ``shared_latent`` is the project's correlated-innovation structure: one
    per-member multiplier scales the whole increment, so members are different
    SCENARIOS. With it off, members differ only by independent per-pixel noise —
    individually calibrated, no scenario spread — which is the G3 ablation and
    the degenerate case ``dispersion_ratio`` cannot see.
    """
    burned = window.x0 > 0
    hits, misses = _growth_candidates(window)
    n_lead = int(window.truth.shape[0])
    out = np.zeros((n_members, n_lead, *burned.shape), dtype=bool)
    for m in range(n_members):
        latent = float(rng.uniform(0.45, 1.55)) if shared_latent else 1.0
        take_hit = rng.random(burned.shape) < min(1.0, recall * latent)
        take_miss = rng.random(burned.shape) < min(1.0, false_alarm * latent)
        claimed = (hits & take_hit) | (misses & take_miss)
        for k in range(n_lead):
            frac = (k + 1) / n_lead
            phase = rng.random(burned.shape) < frac
            out[m, k] = burned | (claimed & phase)
        # absorbing in time, per C1.1
        for k in range(1, n_lead):
            out[m, k] |= out[m, k - 1]
    return _as_samples(out)


def skillful(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Informative and imperfect: recall 0.55, false-alarm 0.10, shared latent.

    **Read this before trusting a verdict that rests on it.** Those two constants
    were chosen against a generated fixture whose growth-band base rate is ~2%.
    On a real fire the base rate is far lower — Kincade's growth band is 0.24% at
    1 h — so a flat 10% false-alarm rate over a 3-cell reach makes this forecaster
    over-predict by a factor of tens. It is informative but grossly OVER-CONFIDENT
    off-fixture, and a proper score is supposed to rank an over-confident forecast
    below silence. See :func:`fit_calibrated_skillful`.
    """
    return _skillful_members(
        window, n_members, rng, recall=0.55, false_alarm=0.10, shared_latent=True
    )


@dataclass(frozen=True)
class CalibratedSkillful:
    """Genuine skill whose TOTAL predicted growth matches the observed growth.

    **This exists because the harness was not holding its own reference model to
    the project's own rule.** C6.2 [v2.8] says a baseline's scale must be
    CALIBRATED TO REPRODUCE OBSERVED MEAN GROWTH, never left free — an
    uncalibrated baseline is not a distinct baseline, it is a broken one. The
    null check's ``skillful`` reference was never held to that, and off its
    fixture it over-predicts by tens of times. The consequence is a FALSE ALARM
    on every proper score: measured on CZU (60% zero-growth leads, growth-band
    base rate 0.06%), ``brier_1h`` flags SILENCE_FAVOURING with null 0.0005
    against ``skillful`` 0.0007 — which is Brier working correctly on a forecast
    that claims tens of times too many cells, not Brier failing.

    A check whose reference model is unrealistic reports the reference's defects
    as the metric's. So this variant keeps the DISCRIMINATION of ``skillful``
    (the same recall on true growth cells, the same shared latent, so members are
    still scenarios) and fits ONE constant — the false-alarm rate — so that the
    expected number of claimed cells equals the observed number of new cells,
    summed over all scored windows. It is fitted on the labels of the windows
    being scored, exactly as :class:`Climatology` is, and for the same reason: a
    reference must be fair on THIS sample or it is not a reference.
    """

    recall: float
    #: One false-alarm rate PER LEAD, not one overall. C6.2 [v2.8] ratified
    #: exactly this for the ellipse baseline after ADR-015 measured that the
    #: calibration horizon is worth ~4.7x in over-prediction ratio: growth is not
    #: linear in the horizon, so a model calibrated in TOTAL is miscalibrated at
    #: every individual lead. Measured here on CZU, the truth's own band rate runs
    #: 0.00055 / 0.00184 / 0.00540 at 1/2/3 h — a 10x ratio where linear phasing
    #: would give 3x, so a total-calibrated reference over-predicts ~3x at 1 h and
    #: a per-horizon calibration criterion correctly says so.
    false_alarm_by_lead: tuple[float, ...]
    n_windows_fitted: int
    #: What it was fitted to, per lead: (true new cells, plausible-but-wrong).
    n_hits_by_lead: tuple[int, ...]
    n_misses_by_lead: tuple[int, ...]

    def __call__(
        self, window: Window, n_members: int, rng: np.random.Generator
    ) -> np.ndarray:
        from wildfire_nowcast.common.states import dilate  # noqa: PLC0415

        burned = window.x0 > 0
        reach = dilate(burned, 3) & ~burned
        n_lead = int(window.truth.shape[0])
        out = np.zeros((n_members, n_lead, *burned.shape), dtype=bool)
        for m in range(n_members):
            latent = float(rng.uniform(0.45, 1.55))
            u_hit = rng.random(burned.shape)
            u_miss = rng.random(burned.shape)
            claimed = np.zeros(burned.shape, dtype=bool)
            for k in range(n_lead):
                hits_k = (np.asarray(window.truth[k]) > 0) & ~burned
                fa = self.false_alarm_by_lead[min(k, len(self.false_alarm_by_lead) - 1)]
                claimed |= (hits_k & (u_hit < min(1.0, self.recall * latent))) | (
                    reach & ~hits_k & (u_miss < min(1.0, fa * latent))
                )
                out[m, k] = burned | claimed  # absorbing in time, per C1.1
        return _as_samples(out)


def fit_calibrated_skillful(
    windows: Sequence[Window], *, recall: float = 0.55
) -> CalibratedSkillful:
    """Fit :class:`CalibratedSkillful`'s per-lead constants from labels alone."""
    from wildfire_nowcast.common.states import dilate  # noqa: PLC0415

    if not windows:
        raise ValueError("fit_calibrated_skillful needs at least one window")
    horizon = max(int(w.truth.shape[0]) for w in windows)
    hits = np.zeros(horizon, dtype=np.int64)
    misses = np.zeros(horizon, dtype=np.int64)
    for w in windows:
        burned = w.x0 > 0
        reach = dilate(burned, 3) & ~burned
        for k in range(int(w.truth.shape[0])):
            hit_k = (np.asarray(w.truth[k]) > 0) & ~burned
            hits[k] += int(np.count_nonzero(hit_k))
            misses[k] += int(np.count_nonzero(reach & ~hit_k))
    # recall*H_k + fa_k*M_k == H_k  =>  fa_k = (1 - recall) * H_k / M_k. A lead
    # with no plausible-but-wrong cells gets 0, which is correct rather than a
    # division by zero.
    fa = [
        0.0 if m <= 0 else float(min(1.0, max(0.0, (1.0 - recall) * h / m)))
        for h, m in zip(hits.tolist(), misses.tolist(), strict=True)
    ]
    return CalibratedSkillful(
        recall=float(recall),
        false_alarm_by_lead=tuple(fa),
        n_windows_fitted=len(windows),
        n_hits_by_lead=tuple(int(v) for v in hits),
        n_misses_by_lead=tuple(int(v) for v in misses),
    )


def collapse_indep_noise(window: Window, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """The G3 ablation shape: same marginals, NO scenario spread.

    Members are individually calibrated per pixel and nearly identical in every
    aggregate, because independent noise averages out over thousands of cells.
    This is the ensemble G3 exists to reject, and ``dispersion_ratio`` scores it
    as perfect — which is why it is in the DEGENERATE set here.
    """
    return _skillful_members(
        window, n_members, rng, recall=0.55, false_alarm=0.10, shared_latent=False
    )


@dataclass(frozen=True)
class Climatology:
    """Persistence outside the reachable band, the AVERAGE growth rate inside it.

    **Why this belongs in the degenerate set, even though it "predicts
    something".** C6.0's literal text names a model that predicts nothing, and the
    two zero-ignition nulls are that. But the property those nulls actually
    exploit is not silence — it is carrying NO INFORMATION about the situation,
    and at a 1-7% base rate a silent forecast is very nearly the climatological
    one. Climatology is the exact, sharpened form: it knows where the fire is and
    what fraction of reachable cells burn on average, and nothing else. No wind,
    no fuel, no direction, no shape of the front. It cannot distinguish any cell
    of the band from any other.

    It is the sharpest possible test of a CALIBRATION statistic, because it is
    perfectly calibrated marginally BY CONSTRUCTION, and any statistic it beats
    is a statistic that cannot tell information from its absence. The rate is
    fitted on the labels of the very windows being scored — a free parameter the
    degenerate is GIVEN. That is deliberate and is the conservative direction for
    a safety check, the same reasoning as :data:`NOISE_FLOOR_SD` being 2 and not
    1: a check like this should flag more, not fewer.

    Members are independent per pixel, so this is also collapsed. That is a
    property of climatology, not a modelling choice — there is no scenario to
    vary.
    """

    #: Marginal P(a band cell is burned by lead k), fitted over ALL windows. One
    #: constant per lead, never per window: a rate re-fitted per window would see
    #: that window's answer, which is a different (and much stronger) model.
    rate_by_lead: tuple[float, ...]
    band_radius: int
    n_windows_fitted: int

    def __call__(
        self, window: Window, n_members: int, rng: np.random.Generator
    ) -> np.ndarray:
        from wildfire_nowcast.eval.masks import growth_band  # noqa: PLC0415

        burned = window.x0 > 0
        band = growth_band(window.x0, self.band_radius)
        n_lead = int(window.truth.shape[0])
        out = np.zeros((n_members, n_lead, *burned.shape), dtype=bool)
        for m in range(n_members):
            # One uniform per cell, thresholded by the CUMULATIVE rate, so a
            # member is absorbing in time (C1.1) and its marginals are exactly
            # `rate_by_lead` at every lead.
            u = rng.random(burned.shape)
            for k in range(n_lead):
                rate = self.rate_by_lead[min(k, len(self.rate_by_lead) - 1)]
                out[m, k] = burned | (band & (u < rate))
        return _as_samples(out)


def fit_climatology(
    windows: Sequence[Window], *, band_radius: int | None = None
) -> Climatology:
    """Fit the one constant :class:`Climatology` has, from labels alone."""
    from wildfire_nowcast.eval.masks import default_band_radius, growth_band  # noqa: PLC0415

    if not windows:
        raise ValueError("fit_climatology needs at least one window")
    horizon = max(int(w.truth.shape[0]) for w in windows)
    radius = int(band_radius) if band_radius is not None else default_band_radius(horizon)
    burned_by_lead = np.zeros(horizon, dtype=np.float64)
    cells_by_lead = np.zeros(horizon, dtype=np.float64)
    for w in windows:
        band = growth_band(w.x0, radius)
        n_band = int(np.count_nonzero(band))
        for k in range(int(w.truth.shape[0])):
            burned_by_lead[k] += int(np.count_nonzero((w.truth[k] > 0) & band))
            cells_by_lead[k] += n_band
    rates = np.divide(
        burned_by_lead, cells_by_lead, out=np.zeros_like(burned_by_lead), where=cells_by_lead > 0
    )
    return Climatology(
        rate_by_lead=tuple(float(r) for r in rates),
        band_radius=radius,
        n_windows_fitted=len(windows),
    )


FORECASTERS: dict[str, Forecaster] = {
    "null_zero_ignition": null_zero_ignition,
    "null_empty": null_empty,
    "collapse_indep_noise": collapse_indep_noise,
    "skillful": skillful,
    "oracle": oracle,
}

#: The name the fitted :class:`Climatology` is registered under.
CLIMATOLOGY = "null_climatology"

#: ...and the growth-calibrated skill reference.
SKILLFUL_CALIBRATED = "skillful_calibrated"


def forecasters_for(windows: Sequence[Window]) -> dict[str, Forecaster]:
    """:data:`FORECASTERS` plus the two models that must be FITTED on ``windows``.

    Separate from the static dict because these two have a parameter, and a
    parameter fitted on the wrong sample would be a different model. Callers that
    want the static set (constructed-case tests) still get it.
    """
    return {
        **FORECASTERS,
        CLIMATOLOGY: fit_climatology(windows),
        SKILLFUL_CALIBRATED: fit_calibrated_skillful(windows),
    }


#: The information-free models of C6.0. Tested against every metric.
#:
#: ``null_zero_ignition`` / ``null_empty`` are C6.0's literal do-nothing nulls;
#: ``null_climatology`` is the same idea sharpened — see :class:`Climatology`.
NULLS: frozenset[str] = frozenset({"null_zero_ignition", "null_empty", CLIMATOLOGY})

#: The collapse ablation. Tested against SPREAD metrics only — see FAMILY_SKILL.
COLLAPSE = "collapse_indep_noise"

#: Forecasters that never draw from the rng, so their score is IDENTICAL at every
#: seed. Declared rather than detected, and verified by a test that runs each one
#: under two different generators and asserts bit-equality — a mis-declaration
#: must break the build, not silently cache a stochastic model at one seed.
#:
#: :func:`run_null_check` scores these ONCE and replicates, which is bitwise the
#: same numbers (they were already identical) for a third less work: measured,
#: they are 2.86 s of the 7.7 s each report costs, and the null-check fixture is
#: over half the test suite's wall clock. The scoring itself is C6's, so this is
#: the only saving available here that does not touch another lead's module or
#: change a verdict.
DETERMINISTIC: frozenset[str] = frozenset({"null_zero_ignition", "null_empty", "oracle"})

#: THE SUBJECT OF THE ZERO-CAPTURE AXIOM: a forecast that claims nothing anywhere
#: at any lead, for every member.
#:
#: It is ``null_empty`` and NOT ``null_zero_ignition``, and that choice is
#: load-bearing rather than cosmetic. Persistence reproduces the already-burned
#: region, so on the DOMAIN mask it legitimately scores ``best_member_iou``
#: 0.857 — capture it earned by making a correct claim. An axiom pointed at it
#: would flag every capture metric in the table, which is a false-positive safety
#: check, which is how safety checks get switched off (the argument that has kept
#: C1.6 off the hard tier twice).
ZERO_CLAIM = "null_empty"

#: Everything that must never win, for reporting.
DEGENERATE: frozenset[str] = NULLS | {COLLAPSE}
SKILLFUL = "skillful"
ORACLE = "oracle"

#: Every forecaster that has GENUINE SKILL. A degenerate must be measurably worse
#: than the BEST of these, not than an arbitrary one of them.
#:
#: Two, not one, and the reason is a defect this harness had: a single reference
#: that is over-confident off-fixture makes every proper score look
#: silence-favouring, and the check then reports the REFERENCE's defect as the
#: METRIC's. Requiring "the degenerate beats every skilful forecaster we can
#: build" is both the conservative reading of C6.0 and the one that does not
#: manufacture false alarms — and a false-positive safety check is the thing that
#: teaches people to disable safety checks (the argument that kept C1.6 off the
#: hard tier twice). The BLIND comparison deliberately does NOT use this: collapse
#: is compared against ``skillful`` specifically, because ``collapse_indep_noise``
#: is that exact model with the shared latent removed, and a controlled ablation
#: must vary one thing.
SKILL_REFERENCES: frozenset[str] = frozenset({SKILLFUL, SKILLFUL_CALIBRATED})


def degenerates_for(spec: MetricSpec) -> frozenset[str]:
    return NULLS | {COLLAPSE} if spec.family == FAMILY_SPREAD else NULLS


def strongest_reference(spec: MetricSpec) -> str:
    """The model a degenerate may never match.

    For a skill metric that is the ORACLE: nothing may be as good as being
    exactly right. For a SPREAD metric it is ``skillful``, the healthy
    shared-latent ensemble — a perfect forecast has zero error, so its
    spread-skill ratio is undefined, and that is a property of the estimand
    rather than a defect of the metric.
    """
    return SKILLFUL if spec.family == FAMILY_SPREAD else ORACLE
