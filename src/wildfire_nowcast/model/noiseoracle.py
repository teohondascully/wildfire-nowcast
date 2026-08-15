"""[L1 arm (b)] The NOISE CEILING: what a PERFECT forecaster scores against noisy labels.

ADR-054 (4). Arm (a) (:mod:`wildfire_nowcast.eval.labelfloor`) measures the real
GOFER-vs-official disagreement. This module reproduces that magnitude
SYNTHETICALLY on the labels themselves, so it can be pushed through the licensed
scoring path and turned into a per-metric ceiling.

THE CONSTRUCTION
----------------
Let ``T`` be the underlying fire and ``L = noise(T)`` the label we actually have.
A forecaster that is RIGHT ABOUT THE FIRE is still scored against ``L``, so its
score is bounded by ``score(T, L)`` — the disagreement the labels contribute on
their own. We cannot observe ``T``, so the standard substitution applies: take
the labels as the centre and generate ``noise_k(L)``. The severity of ``noise``
is CALIBRATED so that ``IoU(noise(L), L)`` on the final footprint reproduces the
measured ``IoU(GOFER, official)`` from arm (a). That calibration is the whole
reason the ceiling means anything; a severity picked for convenience would make
the ceiling an arbitrary number with a scientific-looking derivation.

Members are drawn INDEPENDENTLY per member, so the oracle is a genuine
ensemble whose spread is the label-noise spread.

WHAT IS AND IS NOT NEW HERE (C0)
--------------------------------
The morphology (``dilate_field``/``erode_field``), the translation
(``shift_field``), the composition order (``apply_perturbation``) and the
per-fire calibration source (``noise_model_for``, reading the measured
East-vs-West disagreement) ALL come from
:mod:`wildfire_nowcast.model.labelnoise`. Nothing is re-implemented.

The one extension is the DRAW at severities above the shipped one: the shipped
sampler expresses the morphological term as a Bernoulli over a ONE-cell
dilate/erode, which cannot express more than one cell. :func:`draw_perturbation`
generalises it to a stochastically-rounded integer magnitude with the same mean,
and caps the translation at a severity-scaled number of cells.
**At the shipped parameters it is BITWISE IDENTICAL to
:func:`wildfire_nowcast.model.labelnoise.sample_perturbation`, draw for draw, on
a shared RNG stream** — asserted by :func:`sampler_reduces_to_shipped`, which
compares every field of every draw and demands exact equality rather than
agreement of moments.

SEVERITY IS DECLARED IN KILOMETRES
----------------------------------
``Severity`` reports ``shift_sigma_km`` and ``morph_km`` — displacement of the
perimeter on the ground. Neither is a function of any metric being tested, so
the ceiling curve is not circular. IoU is the CALIBRATION TARGET, never the
severity axis.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from wildfire_nowcast.common.seeds import stable_seed
from wildfire_nowcast.model.api import validate_predict_inputs, validate_samples
from wildfire_nowcast.model.labelnoise import (
    FCONF_LEVELS,
    MAX_SHIFT_CELLS,
    FireNoiseModel,
    LabelPerturbation,
    apply_perturbation,
    noise_model_for,
    sample_perturbation,
)

__all__ = [
    "COHERENT",
    "PER_STEP",
    "Severity",
    "severity_for",
    "draw_perturbation",
    "perturb_burned",
    "sampler_reduces_to_shipped",
    "WindowKey",
    "WindowTable",
    "NoisyTruthOracle",
    "final_footprint_agreement",
]


def _stochastic_round(value: float, rng: np.random.Generator) -> int:
    """Mean-preserving integer rounding.

    Character-for-character the rule in ``labelnoise._stochastic_round`` (a
    private name, so it is restated rather than imported); it consumes EXACTLY
    one draw from ``rng``, which is what makes the shipped-equivalence assertion
    a bitwise one rather than a distributional one.
    """
    low = math.floor(value)
    return int(low + (1 if rng.random() < (value - low) else 0))


@dataclass(frozen=True)
class Severity:
    """One rung of the label-noise ladder, in units that no scored metric uses."""

    fire_id: str
    scale: float
    cell_km: float
    shift_sigma_cells: float
    morph_mean_cells: float
    max_shift_cells: int
    measured_centroid_offset_km: float
    measured_radius_mismatch_km: float
    source: str

    @property
    def shift_sigma_km(self) -> float:
        return self.shift_sigma_cells * self.cell_km

    @property
    def morph_km(self) -> float:
        return self.morph_mean_cells * self.cell_km

    @property
    def is_identity(self) -> bool:
        return self.scale == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fire_id": self.fire_id,
            "scale": self.scale,
            "shift_sigma_km": self.shift_sigma_km,
            "morph_km": self.morph_km,
            "shift_sigma_cells": self.shift_sigma_cells,
            "morph_mean_cells": self.morph_mean_cells,
            "max_shift_cells": self.max_shift_cells,
            "measured_centroid_offset_km": self.measured_centroid_offset_km,
            "measured_radius_mismatch_km": self.measured_radius_mismatch_km,
            "calibration_source": self.source,
            "unit_note": (
                "shift_sigma_km and morph_km are GROUND DISPLACEMENTS. They are independent "
                "of every metric scored against this ladder, so the ceiling curve cannot be "
                "circular. IoU is the calibration TARGET, never the severity axis."
            ),
        }


def severity_for(fire_id: str, scale: float, *, cell_size_m: float = 1000.0) -> Severity:
    """Per-fire severity at ``scale``, anchored on the MEASURED East-vs-West noise.

    ``scale = 1`` is the shipped observation-noise model exactly. ``scale = 0`` is
    the identity and must reproduce the labels bitwise.
    """
    cell_km = float(cell_size_m) / 1000.0
    base = noise_model_for(fire_id, cell_size_m=cell_size_m, scale=1.0)
    # `noise_model_for` clips `p_morph` at 0.5 because in the shipped model it is
    # a PROBABILITY. Here it is a mean magnitude in cells, so the clip is undone
    # from the measured quantity it was computed from — never from the clipped
    # value, which would silently cap the ladder for the noisiest fires.
    morph_unclipped = 0.5 * base.radius_mismatch_km / cell_km
    return Severity(
        fire_id=fire_id,
        scale=float(scale),
        cell_km=cell_km,
        shift_sigma_cells=float(scale) * base.offset_sigma_cells,
        morph_mean_cells=float(scale) * morph_unclipped,
        max_shift_cells=max(MAX_SHIFT_CELLS, int(math.ceil(MAX_SHIFT_CELLS * float(scale)))),
        measured_centroid_offset_km=base.centroid_offset_km,
        measured_radius_mismatch_km=base.radius_mismatch_km,
        source=base.source,
    )


def draw_perturbation(sev: Severity, rng: np.random.Generator) -> LabelPerturbation:
    """One realisation. RNG consumption order matches the shipped sampler exactly."""
    if sev.is_identity:
        return LabelPerturbation()
    magnitude = (
        abs(rng.normal(0.0, sev.shift_sigma_cells)) if sev.shift_sigma_cells else 0.0
    )
    magnitude = min(magnitude, float(sev.max_shift_cells))
    angle = rng.uniform(0.0, 2.0 * math.pi)
    # Row index increases SOUTHWARD (C1.4 north-up), matching labelnoise.
    shift_r = _stochastic_round(-magnitude * math.cos(angle), rng)
    shift_c = _stochastic_round(magnitude * math.sin(angle), rng)
    steps = _stochastic_round(sev.morph_mean_cells, rng)
    morph = 0
    if steps:
        morph = steps if rng.random() < 0.5 else -steps
    return LabelPerturbation(
        shift_r=shift_r,
        shift_c=shift_c,
        morph=morph,
        fconf=float(rng.choice(FCONF_LEVELS)),
    )


def sampler_reduces_to_shipped(
    fire_id: str = "2020_creek", n_draws: int = 50_000, seed: int = 20260814
) -> dict[str, Any]:
    """ASSERT — bitwise, every field, every draw — that scale=1 IS the shipped sampler.

    Not "the moments agree". Not "the distributions look the same". The two
    samplers are driven by two RNGs seeded identically and every one of the four
    fields of every one of ``n_draws`` draws must be EQUAL, which is only
    possible if they consume the same number of variates in the same order from
    the same distributions. The extension is then demonstrably an extension
    rather than a second noise model wearing the first one's name (C0).

    Valid only where the shipped model's own regime holds: ``p_morph < 1`` (it is
    a probability there) and the same shift cap. Both are asserted, not assumed.
    """
    sev = severity_for(fire_id, 1.0)
    shipped: FireNoiseModel = noise_model_for(fire_id, scale=1.0)
    in_regime = (
        shipped.p_morph < 1.0
        and abs(shipped.p_morph - sev.morph_mean_cells) < 1e-12
        and shipped.max_shift_cells == sev.max_shift_cells
        and abs(shipped.offset_sigma_cells - sev.shift_sigma_cells) < 1e-12
    )
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    nonidentity = 0
    for i in range(int(n_draws)):
        a = draw_perturbation(sev, rng_a)
        b = sample_perturbation(shipped, rng_b)
        if not a.is_identity:
            nonidentity += 1
        if (a.shift_r, a.shift_c, a.morph, a.fconf) != (b.shift_r, b.shift_c, b.morph, b.fconf):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {"draw": i, "extended": a.to_dict(), "shipped": b.to_dict()}
    return {
        "fire_id": fire_id,
        "n_draws": int(n_draws),
        "in_shipped_regime": bool(in_regime),
        "mismatches": mismatches,
        "bitwise_identical": mismatches == 0 and in_regime,
        "n_nonidentity_draws": nonidentity,
        "first_mismatch": first_mismatch,
        "why": (
            "A control that only asserted 'the two samplers have similar moments' would pass "
            "a second noise model with the same variance. Exact equality of every field of "
            "every draw can only hold if this is the shipped sampler with a wider magnitude "
            "schedule. n_nonidentity_draws must be large or the comparison is vacuous."
        ),
    }


def perturb_burned(burned: np.ndarray, p: LabelPerturbation) -> np.ndarray:
    """Apply one perturbation to a boolean burned-set field, via the SHIPPED operator."""
    if p.is_identity:
        return np.asarray(burned, dtype=bool)
    t = torch.from_numpy(np.ascontiguousarray(np.asarray(burned, dtype=np.float32)))
    out = apply_perturbation(t, p)
    return np.asarray(out.numpy() > 0.5)


# ---------------------------------------------------------------------------
# the C5 oracle
# ---------------------------------------------------------------------------

WindowKey = str


def _window_key(x0: np.ndarray, static: np.ndarray, weather: np.ndarray) -> WindowKey:
    """Identify a scoring window from the C5 arguments alone.

    C5's ``predict`` receives no fire id and no ``t0``, and a truth-aware oracle
    needs both. Keying on ``(x0, static, weather)`` is exact: ``static`` is the
    fire (elevation is unique per domain) and ``weather`` is a float32 RTMA slab
    that differs between any two hours. Collisions are not assumed away —
    :meth:`WindowTable.add` REFUSES a duplicate key, so a collision is a crash
    rather than a silently mis-scored window.
    """
    digest = hashlib.blake2b(digest_size=16)
    for arr in (x0, static, weather):
        a = np.ascontiguousarray(np.asarray(arr))
        digest.update(a.tobytes())
        digest.update(repr(a.shape).encode())
    return digest.hexdigest()


class WindowTable:
    """``(x0, static, weather) -> (fire_id, t0, truth)`` with hard uniqueness."""

    def __init__(self) -> None:
        self._rows: dict[WindowKey, tuple[str, int, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0

    def add(self, window: Any) -> None:
        key = _window_key(window.x0, window.static, window.weather)
        existing = self._rows.get(key)
        if existing is not None:
            raise KeyError(
                f"WINDOW KEY COLLISION: {existing[0]}@t0={existing[1]} and "
                f"{window.fire_id}@t0={window.t0} hash identically. The oracle would return "
                "the wrong fire's truth and every number below it would be wrong while "
                "looking fine. Refusing to build the table."
            )
        self._rows[key] = (str(window.fire_id), int(window.t0), np.asarray(window.truth))

    def get(
        self, x0: np.ndarray, static: np.ndarray, weather: np.ndarray
    ) -> tuple[str, int, np.ndarray]:
        key = _window_key(x0, static, weather)
        row = self._rows.get(key)
        if row is None:
            self.misses += 1
            raise KeyError(
                "the oracle was asked to predict a window it has no truth for. This is a "
                "HARD FAILURE on purpose: falling back to a guess would turn a perfect "
                "forecaster into an unspecified one on an unknown subset of windows."
            )
        self.hits += 1
        return row

    def stats(self) -> dict[str, Any]:
        return {"n_windows": len(self._rows), "hits": self.hits, "misses": self.misses}

    def __len__(self) -> int:
        return len(self._rows)


#: The label noise is ONE DRAW PER FIRE, shared by every cell and every hour —
#: that is the shipped model's stated design and it comes from the measurement
#: that the GOFER centroid offset is "a per-fire latent nuisance parameter,
#: resampled per fire, not per pixel and not per step".
#: Under that assumption the SAME displacement corrupts ``x_t`` and ``x_{t+h}``,
#: so it largely CANCELS in the increment, and the increment is what an hourly
#: nowcast is scored on.
COHERENT = "coherent"

#: The opposite extreme: a FRESH draw at every hour. Nobody claims this is the
#: label process; it is the upper bound on what label noise could cost an hourly
#: forecast, and it is scored because the honest answer to "how much of our error
#: is the labels'" is an interval whose two ends are declared.
PER_STEP = "per_step"


class NoisyTruthOracle:
    """A C5 predictor that knows the fire and mis-observes it exactly like GOFER.

    ``scale = 0`` is the PERFECT forecaster against perfect labels and must score
    the metric's optimum; it is the null rung and it runs the whole machinery
    rather than short-circuiting, so "the null rung is exact" is a property of
    the construction and not of an ``if``.

    ``temporal`` picks which end of the coherence interval this rung sits at.

    ``COHERENT`` — ``x0 | (perturb(truth_h) \\ perturb(x0))``. One displacement
        corrupts the whole trajectory, so what survives into the increment is
        only the part that does not cancel: the increment translated, and the
        increment's own boundary dilated. This is the construction that matches
        the measured per-fire structure of the noise.
    ``PER_STEP`` — ``x0 | perturb(truth_h)``. The initial condition is clean and
        every hour is mis-observed afresh. This displaces the WHOLE perimeter
        relative to a clean ``x0``, so a 2 km label error lands entirely inside a
        growth band whose real content is one hour of spread.

    The two are the same severity applied under two temporal assumptions; the
    severity is calibrated on the FINAL FOOTPRINT, which both share, so the
    calibration does not have to be redone for each.
    """

    kind = "noisy_truth_oracle"

    def __init__(
        self,
        table: WindowTable,
        *,
        name: str,
        scale: float,
        split_fingerprint: str,
        temporal: str = COHERENT,
        cell_size_m: float = 1000.0,
    ) -> None:
        if temporal not in (COHERENT, PER_STEP):
            raise ValueError(f"unknown temporal assumption {temporal!r}")
        self.table = table
        self.name = name
        self.scale = float(scale)
        self.temporal = temporal
        self.cell_size_m = float(cell_size_m)
        self._severity: dict[str, Severity] = {}
        # C8: the oracle is not trained, but an unstamped model is a HARD FAIL
        # (C-1: unverifiable is a failure). It is scored on exactly the split it
        # is handed, so it declares that split rather than claiming an exemption.
        self.provenance: dict[str, Any] = {
            "split_fingerprint": split_fingerprint,
            "role": (
                "L1 label-noise CEILING oracle — predicts the labels' own future, "
                f"mis-observed at severity scale {scale}. NOT A MODEL and NOT A BASELINE."
            ),
        }

    def severity(self, fire_id: str) -> Severity:
        if fire_id not in self._severity:
            self._severity[fire_id] = severity_for(
                fire_id, self.scale, cell_size_m=self.cell_size_m
            )
        return self._severity[fire_id]

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        validate_predict_inputs(x0, static, weather, n_members, horizon_h, seed)
        fire_id, t0, truth = self.table.get(x0, static, weather)
        sev = self.severity(fire_id)
        # Seeded on the window AND the severity, so two rungs never share a draw
        # and the same rung is reproducible window by window.
        # `stable_seed`, NOT `hash` (ADR-057): CPython randomises str hashing per
        # process, so this line used to draw a different stream in every run and
        # "same seed" was false across processes. Movement measured at 0.018.
        rng = np.random.default_rng(
            [int(seed), int(t0), stable_seed(fire_id), int(round(self.scale * 1e6))]
        )
        x0_arr = np.asarray(x0, dtype=np.uint8)
        burned0 = x0_arr > 0
        out = np.zeros((int(n_members), int(horizon_h), *x0_arr.shape), dtype=np.uint8)
        for m in range(int(n_members)):
            p = draw_perturbation(sev, rng)
            # The SAME displacement that corrupts the future corrupts the present.
            # Subtracting it is what makes the per-fire latent cancel in the
            # increment instead of being counted once for every hour.
            base = perturb_burned(burned0, p) if self.temporal == COHERENT else None
            running = burned0.copy()
            for h in range(int(horizon_h)):
                noisy = perturb_burned(np.asarray(truth[h]) > 0, p)
                if base is not None:
                    noisy = noisy & ~base
                # Union with x0 and with the previous lead: the forecaster is GIVEN
                # x0 exactly, and fire is absorbing (C1.1). Without this an erosion
                # would un-burn a cell that is burned in the initial condition,
                # which is not an observation error, it is an illegal trajectory.
                running = running | noisy
                out[m, h] = np.maximum(x0_arr, running.astype(np.uint8))
        return validate_samples(out, x0_arr, int(n_members), int(horizon_h))

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "scale": self.scale,
            "temporal": self.temporal,
        }


def final_footprint_agreement(
    fire_id: str,
    burned_final: np.ndarray,
    scale: float,
    *,
    n_draws: int = 64,
    seed: int = 20260814,
    cell_size_m: float = 1000.0,
) -> dict[str, Any]:
    """``IoU(noise(L), L)`` on the FINAL footprint — the calibration measurement.

    This is the quantity that is matched to arm (a)'s measured
    ``IoU(GOFER, official)``. Same estimand: two renderings of one fire's final
    footprint on the same 1 km grid, compared by intersection over union.
    """
    sev = severity_for(fire_id, scale, cell_size_m=cell_size_m)
    rng = np.random.default_rng([int(seed), stable_seed(fire_id)])
    base = np.asarray(burned_final, dtype=bool)
    ious: list[float] = []
    ratios: list[float] = []
    for _ in range(int(n_draws)):
        p = draw_perturbation(sev, rng)
        noisy = perturb_burned(base, p)
        union = int((base | noisy).sum())
        inter = int((base & noisy).sum())
        ious.append((inter / union) if union else 1.0)
        ratios.append((float(noisy.sum()) / float(base.sum())) if base.sum() else float("nan"))
    return {
        "fire_id": fire_id,
        "scale": scale,
        "severity": sev.to_dict(),
        "n_draws": int(n_draws),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "sd_iou": float(np.std(ious)),
        "mean_area_ratio": float(np.mean(ratios)),
        "base_cells": int(base.sum()),
    }
