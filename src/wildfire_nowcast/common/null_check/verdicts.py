"""SCORING AND ADJUDICATION: run every forecaster, then judge each metric.

Every forecaster is scored on identical windows, with an identical member count,
under identical seeds, so the only thing that varies is the forecast — which is
what makes each verdict a property of the METRIC rather than of the sample.

Two verdicts per metric, deliberately not collapsed into one string: the
COMPARISON against a reference model, and the ZERO-CAPTURE AXIOM, which needs no
reference and no threshold. They measurably disagree; see the package docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wildfire_nowcast.common.null_check.forecasters import (
    COLLAPSE,
    DEGENERATE,
    DETERMINISTIC,
    SKILL_REFERENCES,
    SKILLFUL,
    ZERO_CLAIM,
    Forecaster,
    degenerates_for,
    forecasters_for,
    strongest_reference,
)
from wildfire_nowcast.common.null_check.registry import (
    C6_METRICS,
    CAPTURE_NOT_APPLICABLE,
    DIAGNOSTIC,
    FAMILY_SPREAD,
    HIGHER,
    LABEL_STATISTIC,
    LOWER,
    TARGET,
    VERDICT_BLIND,
    VERDICT_BROKEN,
    VERDICT_OK,
    VERDICT_PAYS_FOR_NOTHING,
    VERDICT_SILENCE_FAVOURING,
    VERDICT_UNDECIDABLE,
    MetricSpec,
)
from wildfire_nowcast.common.null_check.windows import Window


def _c6_scorer() -> Callable[[np.ndarray, np.ndarray, np.ndarray], Mapping[str, Any]]:
    """The default score function: C6 ``evaluate`` + ``aggregate``.

    Imported lazily and injected, so ``common/`` does not depend on ``eval/`` at
    module scope and the harness stays usable against ANY scoring function. C6
    is modelling's module; this file only calls its documented entry point.
    """
    from wildfire_nowcast.eval.metrics import aggregate, evaluate  # noqa: PLC0415

    def score(samples: np.ndarray, truth: np.ndarray, x0: np.ndarray) -> Mapping[str, Any]:
        return evaluate(samples, truth, x0=x0)

    score.aggregate = aggregate  # type: ignore[attr-defined]
    return score


#: Scalars a ``by_mask`` block carries that are BOOKKEEPING, not scores — counts,
#: denominators and censoring caps. Everything else numeric is treated as a
#: metric and must be registered in :data:`C6_METRICS`, so a metric added
#: tomorrow breaks this check loudly instead of passing by omission. That is the
#: C-2 lesson one level down: the burden belongs on the thing being added.
_STRUCTURAL_KEYS: frozenset[str] = frozenset(
    {
        "n_cells",
        "n_windows",
        "arrival_crps_active_cells",
        "arrival_crps_censor_cap_h",
        "best_member_iou_n_empty_leads",
        "best_member_iou_n_growing_leads",
        "base_rate",
        "uncertainty",
        "calibration_n_scored",
        "calibration_n_occupied_bins",
        "calibration_n_occupied_rings",
    }
)


def _flatten(block: Mapping[str, Any]) -> dict[str, float | None]:
    """Every numeric score in one ``by_mask`` block, registered or not."""
    out: dict[str, float | None] = {}
    for lead, value in (block.get("brier_by_lead") or {}).items():
        out[f"brier_{int(lead)}h"] = value
    for lead, summary in (block.get("reliability_summary") or {}).items():
        for stat, value in (summary or {}).items():
            if stat in _STRUCTURAL_KEYS:
                continue
            out[f"{stat}_{int(lead)}h"] = value
    for key, value in block.items():
        if key in _STRUCTURAL_KEYS or key.startswith("n_") or key.endswith("_by_horizon"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = float(value)
    return out


#: How a comparison came out, once seed noise is accounted for.
#: How many seed SDs a paired difference must exceed to count as real. A noise
#: floor measured from this run's own seeds, not a constant fitted to any sample
#: of fires — so C-3's "state the fitting sample" has nothing to bind to here,
#: and that is deliberate: the alternative is a magic separation threshold.
NOISE_FLOOR_SD = 2.0

CMP_BETTER = "better"
CMP_WORSE = "worse"
CMP_INDISTINGUISHABLE = "indistinguishable"


def _advantage(a: float, b: float, spec: MetricSpec) -> float:
    """Signed advantage of ``a`` over ``b``: positive means ``a`` is better."""
    if spec.direction == HIGHER:
        return a - b
    if spec.direction == LOWER:
        return b - a
    if spec.direction == TARGET:
        target = float(spec.target if spec.target is not None else 1.0)
        return abs(b - target) - abs(a - target)
    raise ValueError(f"{spec.direction} is not rankable")


def compare(
    a_by_seed: Sequence[float], b_by_seed: Sequence[float], spec: MetricSpec
) -> tuple[str, float, float]:
    """PAIRED comparison across seeds: ``(verdict, mean advantage, sd)``.

    Paired, because every forecaster is scored on identical windows with the same
    seed, so the difference has far less variance than either score. And
    seed-aware, because the alternative is a threshold — and a threshold here
    would be a constant deciding a pass/fail, which C-3 makes expensive for good
    reason. The noise floor is MEASURED from the same experiment: a difference
    smaller than its own seed-to-seed standard deviation is not a finding.

    That distinction is not academic. Measured on this fixture,
    ``dispersion_ratio`` prefers the COLLAPSED ensemble on 4 of 5 seeds and the
    healthy one on the 5th, all within 1%: reporting the 4/5 as a direction would
    be reporting a coin flip. ADR-018 quotes 100-159 seed SD for a real effect —
    this is the same discipline applied to the instruments.

    :data:`NOISE_FLOOR_SD` is a NOISE FLOOR, not a fitted threshold: the scale it
    is measured against is estimated from this run's own seeds, and it states no
    claim about any sample of fires (C-3). It is set to 2 rather than 1 because a
    5-seed SD carries ~35% relative error, and because the conservative direction
    for a SAFETY check is to flag more, not fewer — an undecided comparison here
    resolves to "the metric cannot tell these apart", which is a finding.
    """
    diffs = [
        _advantage(float(a), float(b), spec)
        for a, b in zip(a_by_seed, b_by_seed, strict=True)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
    ]
    if not diffs:
        return CMP_INDISTINGUISHABLE, float("nan"), float("nan")
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    floor = NOISE_FLOOR_SD * sd
    if mean > floor:
        return CMP_BETTER, mean, sd
    if mean < -floor:
        return CMP_WORSE, mean, sd
    return CMP_INDISTINGUISHABLE, mean, sd


@dataclass
class MetricVerdict:
    """One metric under one mask, carrying BOTH of C6.0's answers.

    ``verdict`` is the COMPARISON (beatable in principle by genuine skill?) and
    ``capture_verdict`` is the ZERO-CAPTURE AXIOM (does it pay for an empty
    forecast?). They are separate fields because they are separate questions and
    they measurably disagree — see the module docstring. ``is_flagged`` is the
    disjunction, for callers that only need "did anything fire".
    """

    metric: str
    mask: str
    verdict: str
    scores: dict[str, float | None]
    gate_eligible: bool
    quarantined_by: str = ""
    detail: str = ""
    capture_verdict: str = CAPTURE_NOT_APPLICABLE
    capture_detail: str = ""

    @property
    def is_failure(self) -> bool:
        """C-1 ``fail``, from EITHER verdict.

        Hard from the comparison when a degenerate ranks with the best forecast
        the metric admits (``BROKEN``) or when the metric cannot see collapse at
        all (``BLIND``). Hard from the axiom when the metric pays a strictly
        positive score for an empty forecast: that credit is bankable by any
        model, so a gate resting on it is measuring partly silence.
        """
        if not self.gate_eligible:
            return False
        return (
            self.verdict in (VERDICT_BROKEN, VERDICT_BLIND)
            or self.capture_verdict == VERDICT_PAYS_FOR_NOTHING
        )

    @property
    def is_reporting_gap(self) -> bool:
        """C-1 ``reporting``: gate-eligible and silence-favouring.

        Printed unconditionally and non-zero under ``--strict``, but not a hard
        failure, because a proper score at a 1% base rate legitimately prefers
        silence to a sub-coin-flip predictor (R14). Making that a hard fail would
        be a false positive that teaches people to disable the check — the exact
        argument that kept C1.6 off the hard tier.
        """
        return self.gate_eligible and self.verdict == VERDICT_SILENCE_FAVOURING

    @property
    def is_flagged(self) -> bool:
        """Either verdict says something other than ``ok``/not-applicable."""
        return self.verdict != VERDICT_OK or self.capture_verdict == VERDICT_PAYS_FOR_NOTHING


@dataclass
class NullCheckReport:
    scenario: dict[str, Any]
    verdicts: list[MetricVerdict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not any(v.is_failure for v in self.verdicts)

    @property
    def reporting_ok(self) -> bool:
        return self.ok and not any(v.is_reporting_gap for v in self.verdicts)

    def failures(self) -> list[MetricVerdict]:
        return [v for v in self.verdicts if v.is_failure]

    def reporting_gaps(self) -> list[MetricVerdict]:
        return [v for v in self.verdicts if v.is_reporting_gap]

    def pays_for_nothing(self) -> list[MetricVerdict]:
        """Metrics that pay a strictly positive score for an empty forecast."""
        return [v for v in self.verdicts if v.capture_verdict == VERDICT_PAYS_FOR_NOTHING]

    def quarantined_confirmed(self) -> list[MetricVerdict]:
        """Known-broken metrics this run reproduced — the positive controls.

        Reads BOTH verdicts (``is_flagged``). Before A12 it read only the
        comparison, so when the refit raised the reference above the null floor
        the controls emptied and the harness looked healthy while the pathology
        was untouched. A positive control that can be silenced by improving the
        reference model was never testing the metric.
        """
        return [v for v in self.verdicts if v.quarantined_by and v.is_flagged]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause": "C6.0",
            "ok": self.ok,
            "scenario": self.scenario,
            "problems": self.problems,
            "verdicts": [
                {
                    "metric": v.metric,
                    "mask": v.mask,
                    "verdict": v.verdict,
                    "capture_verdict": v.capture_verdict,
                    "gate_eligible": v.gate_eligible,
                    "quarantined_by": v.quarantined_by,
                    "scores": v.scores,
                    "detail": v.detail,
                    "capture_detail": v.capture_detail,
                }
                for v in self.verdicts
            ],
        }


def _best_skill_reference(means: Mapping[str, float | None], spec: MetricSpec) -> str:
    """Which of :data:`SKILL_REFERENCES` scores best on this metric."""
    scored = [n for n in sorted(SKILL_REFERENCES) if means.get(n) is not None]
    if not scored:
        return SKILLFUL
    best = scored[0]
    for name in scored[1:]:
        if _advantage(float(means[name]), float(means[best]), spec) > 0:
            best = name
    return best


def _capture_verdict(
    spec: MetricSpec, by_seed: Mapping[str, list[float | None]]
) -> tuple[str, str]:
    """THE ZERO-CAPTURE AXIOM — no reference model, no threshold, no member count.

    A ``higher_is_better`` capture metric must pay :data:`ZERO_CLAIM` the minimum
    of its range. Every such metric registered here has minimum 0 and every sound
    one lands on EXACTLY ``0.0``, so the line is strict positivity and there is no
    constant for C-3 to bind. Evaluated per seed rather than on the mean: "pays
    for nothing" is a claim about the metric, so one seed paying is enough.
    """
    if spec.direction != HIGHER:
        return CAPTURE_NOT_APPLICABLE, ""
    if ZERO_CLAIM not in by_seed:
        return (
            VERDICT_UNDECIDABLE,
            f"{ZERO_CLAIM!r} is not among the scored forecasters, so the zero-capture axiom "
            "could not be evaluated. Reported, never silently passed.",
        )
    paid = [
        float(v)
        for v in by_seed[ZERO_CLAIM]
        if v is not None and np.isfinite(v) and float(v) > 0.0
    ]
    if not paid:
        return VERDICT_OK, ""
    worst = max(paid)
    return (
        VERDICT_PAYS_FOR_NOTHING,
        f"a forecast that claims NOTHING anywhere at any lead is paid {worst:.5f} on "
        f"{len(paid)} of {len(by_seed[ZERO_CLAIM])} seeds, where the minimum of this metric's "
        "range is 0. That credit is bankable by any model without predicting anything, so this "
        "metric is a joint test of capture and of saying nothing and cannot isolate the first. "
        "Needs no reference model and no member count, which is exactly why it is the verdict "
        "that survives changing either (ADR-022 (1)).",
    )


def _verdict_for(
    metric: str,
    mask: str,
    spec: MetricSpec,
    by_seed: Mapping[str, list[float | None]],
) -> MetricVerdict:
    means = {
        name: (float(np.mean([v for v in vals if v is not None and np.isfinite(v)])) if any(
            v is not None and np.isfinite(v) for v in vals
        ) else None)
        for name, vals in by_seed.items()
    }
    capture, capture_detail = _capture_verdict(spec, by_seed)
    base = MetricVerdict(
        metric=metric,
        mask=mask,
        verdict=VERDICT_UNDECIDABLE,
        scores=means,
        gate_eligible=spec.gate_eligible,
        quarantined_by=spec.quarantined_by,
        capture_verdict=capture,
        capture_detail=capture_detail,
    )
    reference = strongest_reference(spec)
    if means.get(reference) is None or means.get(SKILLFUL) is None:
        base.detail = (
            f"reference model {reference!r} scores None/non-finite here, so there is nothing "
            "to rank against. Undecidable is reported, never silently passed."
        )
        return base
    degenerate = sorted(k for k in degenerates_for(spec) if means.get(k) is not None)
    if not degenerate:
        base.detail = "no degenerate model produced a score"
        return base

    if spec.family == FAMILY_SPREAD and means.get(COLLAPSE) is not None:
        outcome, mean, sd = compare(by_seed[COLLAPSE], by_seed[SKILLFUL], spec)
        if outcome == CMP_INDISTINGUISHABLE:
            base.verdict = VERDICT_BLIND
            base.detail = (
                f"collapsed {means[COLLAPSE]:.4f} vs healthy {means[SKILLFUL]:.4f}: paired "
                f"advantage {mean:+.4f} +- {sd:.4f} over seeds, i.e. INSIDE its own seed "
                "noise. This metric cannot separate a COLLAPSED ensemble from a healthy one, "
                "so which one it prefers is a coin flip — and ensemble collapse is this "
                "project's central claim (ADR-011)."
            )
            return base

    broken = {}
    for name in degenerate:
        outcome, mean, sd = compare(by_seed[name], by_seed[reference], spec)
        if outcome in (CMP_BETTER, CMP_INDISTINGUISHABLE):
            broken[name] = (outcome, mean, sd)
    if broken:
        base.verdict = VERDICT_BROKEN
        base.detail = (
            f"{sorted(broken)} are not measurably worse than {reference} "
            f"({means[reference]:.4f}): "
            + "; ".join(
                f"{k} {v[1]:+.4f} +- {v[2]:.4f} ({v[0]})" for k, v in sorted(broken.items())
            )
            + ". A degenerate model cannot be as good as the best forecast this metric admits; "
            "the metric is measuring something other than capability."
        )
        return base

    # A degenerate must be worse than the BEST skilful forecaster, not than an
    # arbitrary one — see SKILL_REFERENCES for why this is two models and not one.
    skill_ref = _best_skill_reference(means, spec)
    favoured = {}
    for name in degenerate:
        outcome, mean, sd = compare(by_seed[name], by_seed[skill_ref], spec)
        if outcome in (CMP_BETTER, CMP_INDISTINGUISHABLE):
            favoured[name] = (outcome, mean, sd)
    if favoured:
        base.verdict = VERDICT_SILENCE_FAVOURING
        base.detail = (
            f"{sorted(favoured)} are not measurably worse than the best forecaster with genuine "
            f"skill ({skill_ref} {means[skill_ref]:.4f}): "
            + "; ".join(
                f"{k} {v[1]:+.4f} +- {v[2]:.4f} ({v[0]})" for k, v in sorted(favoured.items())
            )
            + ". Not automatically a bug at a low base rate (R14/C6.2), but this metric cannot "
            "be read as capability without stating it, and it must not gate."
        )
        return base

    base.verdict = VERDICT_OK
    base.detail = "every degenerate model ranks below genuine skill, beyond seed noise"
    return base


#: Seeds the harness re-runs every forecaster under. More than one is not
#: optional: with a single seed, ``dispersion_ratio`` reports a DIRECTION for a
#: difference that is a coin flip, and the harness would publish it.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Members per forecast. **Not a performance knob — it changes verdicts**, and it
#: changes them in the flattering direction when it is small.
#:
#: Measured on this fixture (growth band, 3 h, ``null_climatology`` vs
#: ``skillful``), ECE at increasing member counts::
#:
#:     M      8      32     128     512
#:     clim   0.0795 0.0256 0.0110  0.0007
#:     skill  0.0347 0.0482 0.0484  0.0484
#:
#: The ensemble MEAN of ``M`` Bernoulli draws is itself noisy, and that noise is
#: uncorrelated with the truth, so at ``M = 8`` it inflates a no-information
#: forecast's ECE by ~0.08 and HIDES the fact that climatology drives ECE to
#: zero. Raising ``M`` does not create the pathology, it stops masking it. 32 is
#: the smallest power of two at which the masking is gone here (climatology
#: already beats genuine skill), and it costs ~12 s. Terms that pool many cells
#: per stratum — the frontier term, Brier, CRPS — are flat in ``M`` (0.1094 ->
#: 0.1098), which is precisely the property a gate criterion needs: a bar that
#: moves with the member count is not a bar.
DEFAULT_MEMBERS = 32


def run_null_check(
    windows: Sequence[Window],
    scenario: Mapping[str, Any] | None = None,
    *,
    n_members: int = DEFAULT_MEMBERS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    forecasters: Mapping[str, Forecaster] | None = None,
    score_fn: Callable[..., Any] | None = None,
) -> NullCheckReport:
    """Score every forecaster on identical windows and adjudicate each metric.

    Identical windows, an identical member count and identical seeds for every
    forecaster: the only thing that varies is the forecast, which is what makes
    each verdict a property of the METRIC rather than of the sample. Comparisons
    are PAIRED across seeds — see :func:`compare`.
    """
    if not windows:
        raise ValueError("run_null_check needs at least one window")
    if not seeds:
        raise ValueError("run_null_check needs at least one seed")
    models = dict(forecasters or forecasters_for(windows))
    scorer = score_fn or _c6_scorer()
    pool = getattr(scorer, "aggregate", None)
    if pool is None:  # pragma: no cover - only for an injected scorer
        from wildfire_nowcast.eval.metrics import aggregate as pool  # noqa: PLC0415

    report = NullCheckReport(scenario=dict(scenario or {}))
    report.scenario.setdefault("n_windows", len(windows))
    report.scenario["n_members"] = int(n_members)
    report.scenario["seeds"] = [int(s) for s in seeds]
    report.scenario["forecasters"] = sorted(models)
    report.scenario["degenerate"] = sorted(DEGENERATE)
    report.scenario["zero_claim"] = ZERO_CLAIM
    if ZERO_CLAIM not in models:
        report.problems.append(
            f"the zero-capture axiom's subject {ZERO_CLAIM!r} is absent from the forecaster set, "
            "so no capture metric can be adjudicated against an empty forecast. A missing "
            "control is a PROBLEM, not a skip."
        )

    # pooled[model][seed_index] = the aggregated score dict
    pooled: dict[str, list[Mapping[str, Any]]] = {}
    for name, fn in models.items():
        runs = []
        for seed in seeds:
            # A forecaster that never draws produces the same score at every
            # seed, so scoring it once and replicating is bitwise identical, not
            # an approximation. See DETERMINISTIC — declared, and asserted by a
            # test that runs each one under two generators.
            if runs and name in DETERMINISTIC:
                runs.append(runs[0])
                continue
            rng = np.random.default_rng(int(seed))
            runs.append(
                pool(
                    [
                        scorer(fn(w, n_members, rng), np.asarray(w.truth), np.asarray(w.x0))
                        for w in windows
                    ]
                )
            )
        pooled[name] = runs

    masks = sorted({m for runs in pooled.values() for m in runs[0].get("by_mask", {})})
    for mask in masks:
        flat = {
            name: [_flatten(run["by_mask"][mask]) for run in runs] for name, runs in pooled.items()
        }
        seen = sorted({k for runs in flat.values() for run in runs for k in run})
        unregistered = [k for k in seen if k not in C6_METRICS]
        if unregistered:
            report.problems.append(
                f"[{mask}] C6 emits {unregistered} but common/null_check.C6_METRICS does not "
                "declare their orientation, so they cannot be null-checked. An unregistered "
                "metric is an unchecked metric (C-2, one level down)."
            )
        for metric in seen:
            spec = C6_METRICS.get(metric)
            if spec is None:
                continue
            by_seed = {name: [run.get(metric) for run in runs] for name, runs in flat.items()}
            if spec.direction == LABEL_STATISTIC:
                values = {v for runs in by_seed.values() for v in runs if v is not None}
                if len(values) > 1:
                    report.problems.append(
                        f"[{mask}] {metric} is a property of the LABELS and must be identical "
                        f"for every forecaster and seed, but got {sorted(values)}. Something "
                        "model-dependent has leaked into it."
                    )
                continue
            if spec.direction == DIAGNOSTIC:
                continue
            report.verdicts.append(_verdict_for(metric, mask, spec, by_seed))
    return report
