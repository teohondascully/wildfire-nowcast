"""``z_t`` - the SHARED PER-STEP LATENT that makes the innovations correlated.

CLAUDE.md, ground truth, not re-litigated here::

    Pixels are conditionally independent Bernoulli ONLY given a shared per-step
    latent z_t (correlated innovations). Independent-per-pixel-noise-only models
    are known-broken (ensemble collapse) - do not build them except as ablation.

This module is the "ONLY given" clause, in code. :mod:`wildfire_nowcast.model.kernel`
supplies ``p(x)`` from ``(x_t, features)``; everything here supplies the ONE draw
per step that every pixel in that step sees.

WHY A LOW-DIMENSIONAL GLOBAL LATENT AND NOT A NOISE FIELD
---------------------------------------------------------
The phenomenon this exists for is MEASURED in the labels, not assumed:
*"Growth is bursty and hour-locked: Kincade's whole run is 12 hours doing ~250
of 347 km². Per-step Bernoulli rates are strongly non-stationary - this is
precisely the case the shared per-step latent z_t exists for."* The innovation
that is missing from the deterministic kernel is not per-pixel jitter, it is
**this hour was three times hotter than the weather said**, and that is a
property of the STEP, not of a pixel. So ``z_t`` is a small vector of global
step-level forcing errors, and its three components are named physical
quantities rather than anonymous features:

===========  ==============================================================
``z[0]``     log-multiplier on the whole hazard field - burst intensity.
``z[1]``     rotation (radians) of the effective head direction - the
             direction error RTMA and the terrain resolution leave behind.
``z[2]``     log-multiplier on the effective wind speed - reach AND
             length-to-breadth move together, because they do in the physics.
===========  ==============================================================

Each is scaled by a LEARNED ``sigma_k`` fitted through the ELBO, so the ensemble
spread is an estimate of the real step-to-step forcing error rather than a knob
tuned against a dispersion score. Nothing in this module reads a gate metric.

Consequences that are the point rather than side effects:

* one draw moves the whole field COHERENTLY, so a member is a *scenario* - a
  bigger fire, or one that ran further south - not speckle around a mean field;
* the ensemble's TOTAL AREA acquires spread that does not vanish as the grid
  grows. Independent per-pixel noise gives area spread ``O(sqrt(N))`` against a
  mean ``O(N)``, which is the collapse ``area_dispersion_ratio`` detects
  (C6.1/ADR-011).

THE ABLATION IS THE SAME OBJECT WITH ONE SWITCH
------------------------------------------------
:class:`LatentSampler` with ``mode="independent"`` fixes ``z = 0`` (the prior
mean) and leaves the per-pixel Bernoulli draw alone. That is the
known-broken model CLAUDE.md permits ONLY as an ablation, and G3 requires it to
DEMONSTRATE collapse - i.e. it is a POSITIVE CONTROL for the ensemble machinery
in this repo. If it fails to collapse, the finding is about the instrument, not
about the model.

THE INFERENCE NETWORK
---------------------
``q(z_k | b_{k-1}, y_k, p_k^0)`` is a small convolutional encoder that sees the
model's own one-step probability field ``p_k^0`` (at ``z = 0``) alongside the
observed outcome, i.e. it encodes the INNOVATION rather than the state. It is
used ONLY during training - it reads ``y_k``, the future - and is never invoked
by :meth:`~wildfire_nowcast.model.kernel.ContagionKernel.predict`. That
separation is asserted by a self-test, not left to discipline.

POSTERIOR COLLAPSE IS A REAL FAILURE MODE HERE, AND IT IS DETECTABLE
---------------------------------------------------------------------
If ``q`` collapses to the prior, ``z`` carries no information, the reconstruction
term gains nothing from a wider prior, and the fitted ``sigma_k`` are driven to
ZERO - which reproduces the ablation while looking like a trained latent model.
:func:`latent_report` emits ``sigma``, the per-dimension KL and the fraction of
the KL each dimension carries, so a collapsed latent is visible in the run
artifact instead of being inferred from a disappointing dispersion number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "LATENT_COMPONENTS",
    "SPATIAL_COMPONENTS",
    "ACTIVITY_GATE",
    "N_PRIOR_COVARIATES",
    "PRIOR_COVARIATES",
    "reparameterise",
    "SAMPLER_MODES",
    "LatentConfig",
    "LatentEffect",
    "LatentHead",
    "LatentSampler",
    "LatentPath",
    "latent_report",
]

DTYPE = torch.float64

#: What each latent dimension does, in order. Named so a run artifact can say
#: which physical innovation carried the spread, and so an anonymous "dim 2" can
#: never appear in a report.
LATENT_COMPONENTS: tuple[str, ...] = (
    "log_intensity",  # multiplies the whole hazard field
    "head_rotation",  # rotates the effective head direction (radians)
    "log_wind_speed",  # scales the effective wind speed
    "activity_gate",  # [M6] turns the WHOLE FIELD off - see below
)

#: [M6] THE ACTIVITY GATE, and why it belongs in ``z_t`` rather than beside it.
#:
#: **Dormancy is the most extreme correlated innovation this system has: the
#: entire front goes quiet together.** 953 of 1,399 held-out windows have BITWISE
#: zero growth, and every kernel this project has trained ignites in 953 of 953
#: of them (`eval/validity._off_state`). A latent that cannot turn the whole
#: field OFF is not modelling the dominant correlated mode, so this is z_t's job
#: by construction and not an add-on to it.
#:
#: Mechanically it is a FOURTH dimension whose effect on the hazard is a
#: log-multiplier like ``log_intensity``, but with a much larger admissible
#: ``sigma`` and a NEGATIVE mean offset, so that a draw in the lower tail
#: multiplies the whole field's hazard by ``e^-6`` or less - an OFF state, not a
#: quiet one. The two are separated deliberately: ``log_intensity`` is the
#: ordinary hour-to-hour variation of an ACTIVE fire and stays small and
#: symmetric; ``activity_gate`` is the regime switch. Folding them into one
#: dimension would let a single sigma have to be small enough to be a plausible
#: forcing error and large enough to reach zero, which it cannot be.
#:
#: **The prior on this dimension is CONDITIONAL on the step's own weather**
#: (:class:`ConditionalPrior`), which is the part of "CVAE-style" the M5
#: implementation skipped: with a fixed ``N(0, I)`` prior, ``z_t`` cannot know
#: that a cold, humid, still hour is a dormant hour, so the model has no route to
#: an OFF state even in principle. RH and temperature are C1 channels 3-4 and
#: carry the diurnal cycle, so the signal is already in the tensor.
#:
#: **PROVENANCE, LOOKED FOR AND FOUND, AND OUTSIDE THE TREE.** 953 and 1,399 are
#: ``c6_2_validity.<model>.off_state.n_dormant_windows`` and ``.n_windows`` in the
#: M5 scoring run ``runs/baselines-20260808-193208``, and "ignites in 953 of 953"
#: is ``n_dormant_windows_where_no_member_ignited == 0`` (equivalently
#: ``dormant_off_rate == 0.0``) on every trained arm in it, against 953 for
#: persistence, which cannot ignite by construction. That record is not tracked,
#: and tracking it would publish 19 occurrences of an internal coordination role
#: frozen inside run-time ``note`` strings that are evidence and may not be
#: edited, which is the exemption ``tools/cited_runs.py`` already records for its
#: four siblings. So this pair is checkable where the run lives.
#: **A CLONE CANNOT CHECK THIS NUMBER.**
ACTIVITY_GATE = "activity_gate"

#: ``"latent"`` - draw ``z_t ~ p(z)`` once per member per step, then draw pixels
#: independently GIVEN it. This is the model.
#: ``"independent"`` - hold ``z_t`` at the prior mean and draw pixels
#: independently. **This is the known-broken ablation** and the G3 positive
#: control; it is never a candidate.
SAMPLER_MODES: tuple[str, ...] = ("latent", "independent")

#: [M6] Global per-step covariates the CONDITIONAL PRIOR sees, in order. All are
#: domain means of C1 weather channels, standardised by fixed constants so the
#: prior net's scale is interpretable and does not depend on norm_stats (which
#: are per-channel over TRAIN folds and would make the prior a function of the
#: split). Chosen because RH and temperature carry the diurnal cycle, and the
#: diurnal cycle is what dormancy tracks.
PRIOR_COVARIATES: tuple[str, ...] = ("rh_2m", "temp_2m", "wind_speed")
N_PRIOR_COVARIATES = len(PRIOR_COVARIATES)

#: [M7] THE LOW-RANK SPATIAL LATENT - extra dimensions of ``z_t`` whose effect is
#: a FIELD rather than a scalar, so an ensemble can disagree about WHERE.
#:
#: **WHY THIS WAS BUILT, AND WHY THAT REASON DID NOT SURVIVE ITS OWN TEST.**
#: ADR-032 (3) diagnosed M6's dispersion failure as a missing degree of freedom:
#: a GLOBAL SCALAR latent can only widen the ensemble by scaling the whole hazard
#: field, i.e. by blurring the front everywhere, so every unit of spread it buys
#: lands in the wrong places, and a proper score is CORRECT to decline it. That
#: diagnosis is what authorised this module.
#:
#: **ADR-034 (2) SUPERSEDES IT, and does so on this module's own result.** The
#: spatial latent was built as specified and then measured: 0.2468 +/- 0.0260
#: dispersion against a 0.2331 +/- 0.0115 control, **+5.9%, inside seed noise**.
#: A blind negative control that denies the encoder its basis-projected pooling
#: lands at 0.2347, so the modes ARE being inferred; they are simply worth almost
#: nothing for AREA. Spatial rank was not the missing degree of freedom.
#: **What replaced it is SYMMETRIC vs ASYMMETRIC.** At this base rate, spread
#: bought DOWNWARD is cheap and spread bought UPWARD is unboundedly expensive, so
#: a symmetric latent pays the expensive side to buy the cheap one. The activity
#: gate widens the SAME physical channel with an asymmetric mixture prior and
#: reaches 0.75-0.80 with better Brier, CRPS and calibration.
#:
#: **THE M6 MEASUREMENTS ARE NOT RETRACTED** and ADR-034 (2) re-uses them: the
#: ``w_brier = 0`` arm reached 0.5799 dispersion while being a 2.6x-worse forecast
#: with 6.01x growth over-prediction. What was superseded is the MECHANISM they
#: were read as supporting, not the numbers. This module stays because that
#: negative result is load-bearing and because the basis it defines is pinned by a
#: playthrough; ``spatial_modes = 0`` is the default and this is not a route to G3.
#:
#: Each mode multiplies the hazard by ``exp(c_m phi_m(x))`` where ``phi_m`` is a
#: FIXED, FIRE-ANCHORED basis function (:func:`spatial_basis`) and ``c_m`` is one
#: scalar per draw. So the object is still a SHARED PER-STEP LATENT of a handful
#: of numbers - every pixel of a member-step sees the same ``c``, and the field
#: it induces is smooth by construction. **This is deliberately NOT a per-pixel
#: noise field**, which CLAUDE.md records as known-broken and admits only as an
#: ablation: a rank-``R`` expansion has ``R`` degrees of freedom, not ``H*W``.
#:
#: ================================  =========================================
#: ``intensity_grad_east``           east flank runs while the west does not
#: ``intensity_grad_north``          north flank runs while the south does not
#: ``intensity_radial``              the FRONT runs while the interior does not
#: ================================  =========================================
#:
#: All three modulate ``log_intensity`` only. Spatially modulating the WIND
#: DIRECTION as well would double the rank for a much less interpretable object,
#: and I am naming that as this parameterisation's declared cost rather than
#: discovering it later: if the real innovation is that one flank turns while the
#: other keeps its bearing, THIS latent cannot represent it either.
#:
#: **WHERE THESE NUMBERS COME FROM, LOOKED FOR RATHER THAN ASSUMED.** The M7
#: table of record is the scoring run ``runs/baselines-20260809-073414``. Its
#: ``results.json`` is cited elsewhere in this repository and is deliberately NOT
#: tracked; the exemption and its reason are in ``tools/cited_runs.py``. Four of
#: the numbers above are exact in it: the blind negative control's **0.2347** and
#: the ``w_brier = 0`` arm's **0.5799** are both
#: ``g3.models.<arm>.criteria["ensemble dispersion (area spread-skill)"].equal_block``;
#: **6.01x** is
#: ``c6_2_validity.m6_fair_brier0_s1.off_state.growth_stratum_ratio``; **2.6x** is
#: that arm's band Brier over the four-seed ``m6_fair`` mean (2.585).
#:
#: **THE PAIR 0.2468 +/- 0.0260 AGAINST 0.2331 +/- 0.0115 IS NOT RE-DERIVABLE
#: FROM ANY ARTIFACT ON DISK, TRACKED OR NOT, AND THIS SENTENCE IS HERE SO THAT A
#: READER IS TOLD RATHER THAN LEFT TO ASSUME IT WAS CHECKED.** That record's own
#: four-seed equal-block means are 0.2475 and 0.2337. The failure is confined and
#: measured: the gate-IoU and band-Brier columns of the SAME table reproduce to
#: the last published digit, while the dispersion column alone is uniformly
#: **0.9974x** across all six arms (spread 3.3e-4), and the value is identical
#: across three different scoring fingerprints. So a metric moved after the table
#: was written; it was not a different run, a different seed set or a different
#: block set, each of which would have moved the other columns too. The **+5.9%**
#: ratio is the one quantity unaffected, because it re-derives from either pair.
#: **A CLONE CANNOT CHECK THIS NUMBER.**
SPATIAL_COMPONENTS: tuple[str, ...] = (
    "intensity_grad_east",
    "intensity_grad_north",
    "intensity_radial",
)


@dataclass(frozen=True)
class LatentConfig:
    """Structure of ``z_t``. Everything here is fixed; ``sigma`` is learned."""

    #: Number of latent dimensions actually used, taken from the head of
    #: :data:`LATENT_COMPONENTS`. 0 disables the latent entirely and the kernel
    #: is then bit-identical to its G2 form (asserted by a self-test).
    dim: int = 3
    #: Initial generative scales, in the natural units of each component:
    #: dimensionless log-multiplier, radians, dimensionless log-multiplier, and
    #: (dim 4) the ACTIVITY GATE's log-multiplier. Deliberately MODEST on the
    #: first three, so a large fitted sigma is evidence the fires paid for it
    #: rather than an initialisation being reported back; deliberately LARGE on
    #: the gate, because a regime switch that cannot reach zero is not a switch.
    init_sigma: tuple[float, ...] = (0.35, 0.20, 0.15, 2.0)
    #: Hard cap on sigma, applied in the constrained parameterisation. A
    #: log_intensity sigma of 5 is not a wide ensemble, it is an unbounded
    #: hazard multiplier that saturates every cell in the domain. The gate is
    #: capped far higher ON PURPOSE: e^-6 of open ground is the OFF state.
    max_sigma: tuple[float, ...] = (2.0, 1.5, 1.0, 6.0)
    #: [M6] Mean offset of the ACTIVITY GATE dimension, in log-hazard units. The
    #: gate's prior is N(gate_prior_mean, 1) in z-space rather than N(0, 1), so
    #: the marginal is a MIXTURE of active and quiet hours rather than a
    #: symmetric wobble around "always on". 0.0 disables the offset.
    gate_prior_mean: float = 0.0
    #: [M6] Condition the prior on the step's own weather (mean RH, temperature,
    #: wind speed) so the model can learn WHICH hours are dormant. With a fixed
    #: N(0, I) prior it cannot: that is the part of "CVAE-style" M5 skipped.
    conditional_prior: bool = False
    #: [M6] MEAN-PRESERVING MULTIPLICATIVE LATENT - see the module docstring
    #: section "SPREAD MUST NOT MOVE THE MEAN". The log-multiplier dimensions
    #: (``log_intensity``, ``log_wind_speed``) carry ``-sigma_k^2 / 2`` so that
    #: ``E_z[e^effect] = 1`` exactly and ``sigma`` is a PURE spread parameter.
    #: ``False`` reproduces M5 bitwise. The ACTIVITY GATE is deliberately
    #: EXCLUDED: it is a regime switch whose whole purpose is an asymmetric
    #: prior, and mean-correcting it would compensate every OFF draw with a
    #: hotter ON draw, i.e. would undo the thing it exists to do.
    #: **[M8] THAT EXEMPTION IS NOW A SWITCH - see `gate_mean_preserving`, which
    #: reverses this paragraph's decision on measured grounds.**
    mean_preserving: bool = False
    #: [M8] MEAN-PRESERVE THE ACTIVITY GATE TOO. **This reverses a decision I
    #: documented at M6, four lines above, and the reason is that its stated
    #: premise has been falsified.**
    #:
    #: The M6 argument was: mean-correcting the gate "would compensate every OFF
    #: draw with a hotter ON draw, i.e. would undo the thing it exists to do".
    #: The thing it exists to do is the OFF state, and the OFF state scores
    #: ``dormant_off_rate = 0.0000`` on EVERY gate arm ever trained, with a
    #: MEASURED CEILING of 0.143 from the covariates C1 carries (ADR-034 (4)).
    #: So the exemption protects nothing measurable, and it costs something
    #: measurable: with ``sigma_gate ~ 1.3`` and ``gate_prior_mean + bias ~ -3.0``
    #: the gate multiplies the whole hazard field by ``E_z[e^gate] ~ 0.03``, which
    #: the kernel must undo by inflating its base rate. That is exactly the
    #: SPREAD-INFLATES-THE-MEAN confound `mean_preserving` was introduced to
    #: remove for dims 0 and 2, left in place on the dimension with the LARGEST
    #: sigma in the model.
    #:
    #: **WHAT IS CORRECTED, AND WHAT IS DELIBERATELY NOT.** The correction is
    #: ``-(sigma_3 * gate_prior_mean + sigma_3^2 / 2)`` - the UNCONDITIONAL part
    #: only. So:
    #:  * ``E_z[e^gate] = 1`` exactly at the unconditional prior, making
    #:    ``sigma_3`` a PURE SPREAD PARAMETER like every other dimension;
    #:  * the multiplier is still a LOG-NORMAL with ``sigma ~ 1.3``: median
    #:    ``e^{-sigma^2/2} = 0.43`` against mean 1, i.e. most members quiet and a
    #:    thin expensive upper tail. **The ASYMMETRY ADR-034 (2) identifies as the
    #:    working mechanism is preserved; only the BIAS is removed.**
    #:  * the CONDITIONAL deviation ``sigma_3 * w . cov`` is NOT corrected, so a
    #:    dormant hour can still be predicted quieter than an active one. Removing
    #:    that too would delete the OFF-state route entirely rather than merely
    #:    failing to reach it, and would also make dims 0 and 3 differ only by a
    #:    cap.
    #: Requires ``dim >= 4`` and RAISES otherwise - a silent no-op on a config
    #: that asks for a correction to a dimension that does not exist is the
    #: green-but-vacuous shape this project keeps finding. ``False`` reproduces
    #: M7 BITWISE (asserted by `eval/selftest`).
    gate_mean_preserving: bool = False
    #: [M7] Hand the inference network the INNOVATION DECOMPOSITION (realised new
    #: burn, expected new burn, and their difference) instead of the two large
    #: near-identical fields it is the difference of. **False reproduces M6
    #: BITWISE.** Not a capacity change - same widths, one extra input channel -
    #: it is a fix to a docstring's claim that the code did not implement. See
    #: :class:`_Encoder`; measured cause in `runs/m7_offstate_optimum.json` and
    #: the M6 gate checkpoint's own posterior.
    innovation_channels: bool = False
    #: [M7] NEGATIVE CONTROL, and the only reason it is a switch: give the encoder
    #: the spatial modes to infer while DENYING it the basis-projected pooling it
    #: needs to identify them. A globally-pooled summary returns the same value
    #: for "the east flank ran" and "the west flank ran", so ``q`` collapses to
    #: the prior on exactly those dimensions. If the blind arm matches the
    #: sighted one, the spatial modes are not being inferred and any dispersion
    #: gain came from somewhere else.
    spatial_encoder_pooling: bool = True
    #: [M7] Number of LOW-RANK SPATIAL modes taken from the head of
    #: :data:`SPATIAL_COMPONENTS`. **0 reproduces M6 BITWISE** and is the default,
    #: so no existing config is silently redefined. These dimensions are appended
    #: AFTER the global ones, so global dimension indices never move.
    spatial_modes: int = 0
    #: [M7] Initial and maximum generative scales for the spatial modes, in
    #: log-hazard units per unit of basis function. The basis is normalised to
    #: unit radius of gyration of the burned region, so ``sigma = 0.3`` means "the
    #: hazard one gyration-radius east of the fire centroid is e^0.3 = 1.35x its
    #: value at the centroid, one standard draw". Modest on purpose, for the same
    #: reason the global scales are: a large FITTED sigma is then evidence the
    #: fires paid for it.
    spatial_init_sigma: tuple[float, ...] = (0.30, 0.30, 0.30)
    spatial_max_sigma: tuple[float, ...] = (2.0, 2.0, 2.0)
    #: [M6] AR(1) TEMPORAL PERSISTENCE of ``z_t`` across steps, in the
    #: standardised coordinate: ``u_k = rho u_{k-1} + sqrt(1 - rho^2) eps_k``.
    #: Every step's MARGINAL stays exactly ``N(mu_p_k, I)``, so the one-step
    #: model, the per-step KL and every M5 identity check are unchanged and only
    #: the temporal dependence moves - which is what makes the intervention
    #: attributable. ``0.0`` is M5 (iid draws) and reproduces it bitwise;
    #: ``1.0`` is one draw held across the whole horizon.
    rho: float = 0.0
    #: Encoder width. Small on purpose: it is amortised inference over a 3-vector,
    #: not a feature extractor, and a large encoder would let the ELBO explain the
    #: innovation with inference capacity instead of with a wider prior.
    encoder_channels: int = 8
    #: FREE BITS (Kingma et al. 2016), per dimension, in nats. The KL of a
    #: dimension below this is not penalised. This is the standard remedy for
    #: posterior collapse and it is DECLARED rather than tuned: with the KL fully
    #: penalised, a 3-nat-per-step latent competing against a 10^4-cell
    #: reconstruction term is free to be switched off, and "the latent did
    #: nothing" would then be a statement about the optimiser.
    free_bits: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.dim <= len(LATENT_COMPONENTS):
            raise ValueError(
                f"dim={self.dim}; expected 0..{len(LATENT_COMPONENTS)} "
                f"(components: {LATENT_COMPONENTS})"
            )
        for name in ("init_sigma", "max_sigma"):
            values = getattr(self, name)
            if len(values) < self.dim:
                raise ValueError(f"{name} has {len(values)} entries, need >= dim={self.dim}")
            if any(v <= 0 for v in values[: self.dim]):
                raise ValueError(f"{name} must be strictly positive, got {values}")
        if not 0 <= self.spatial_modes <= len(SPATIAL_COMPONENTS):
            raise ValueError(
                f"spatial_modes={self.spatial_modes}; expected 0..{len(SPATIAL_COMPONENTS)} "
                f"(components: {SPATIAL_COMPONENTS})"
            )
        if self.spatial_modes and self.dim == 0:
            # A spatial mode multiplies `log_intensity`, which is global dim 0.
            # Allowing spatial modes with no global block would produce a model
            # whose ONLY latent is mean-zero in space - a different object that
            # must be asked for by name, not reached by leaving a field at 0.
            raise ValueError("spatial_modes requires dim >= 1 (they modulate log_intensity)")
        for name in ("spatial_init_sigma", "spatial_max_sigma"):
            values = getattr(self, name)
            if len(values) < self.spatial_modes:
                raise ValueError(
                    f"{name} has {len(values)} entries, need >= spatial_modes={self.spatial_modes}"
                )
            if any(v <= 0 for v in values[: self.spatial_modes]):
                raise ValueError(f"{name} must be strictly positive, got {values}")
        if self.gate_mean_preserving and self.dim < 4:
            # Refused rather than ignored: the whole point of the flag is to
            # correct dimension 3, and a config that asks for it without that
            # dimension is asking for something this object cannot do.
            raise ValueError(
                f"gate_mean_preserving requires dim >= 4 (the activity gate); got dim={self.dim}"
            )
        if self.free_bits < 0:
            raise ValueError(f"free_bits={self.free_bits}; expected >= 0")
        if not 0.0 <= self.rho < 1.0:
            # rho = 1 is excluded, not clamped: the AR(1) innovation variance is
            # `1 - rho^2` and a degenerate transition is a different model (one
            # draw for the whole horizon), which must be asked for by name.
            raise ValueError(f"rho={self.rho}; expected 0 <= rho < 1")

    @property
    def components(self) -> tuple[str, ...]:
        """Global components first, then spatial. Global indices never move."""
        return LATENT_COMPONENTS[: self.dim] + SPATIAL_COMPONENTS[: self.spatial_modes]

    @property
    def total_dim(self) -> int:
        """Length of ``z``. ``dim`` alone is the GLOBAL block, for M6 compatibility."""
        return self.dim + self.spatial_modes

    @property
    def all_init_sigma(self) -> tuple[float, ...]:
        return tuple(self.init_sigma[: self.dim]) + tuple(
            self.spatial_init_sigma[: self.spatial_modes]
        )

    @property
    def all_max_sigma(self) -> tuple[float, ...]:
        return tuple(self.max_sigma[: self.dim]) + tuple(
            self.spatial_max_sigma[: self.spatial_modes]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "components": list(self.components),
            "init_sigma": list(self.init_sigma[: self.dim]),
            "max_sigma": list(self.max_sigma[: self.dim]),
            "encoder_channels": self.encoder_channels,
            "free_bits": self.free_bits,
            "gate_prior_mean": self.gate_prior_mean,
            "conditional_prior": self.conditional_prior,
            "mean_preserving": self.mean_preserving,
            "gate_mean_preserving": self.gate_mean_preserving,
            "rho": self.rho,
            "spatial_modes": self.spatial_modes,
            "spatial_init_sigma": list(self.spatial_init_sigma[: self.spatial_modes]),
            "spatial_max_sigma": list(self.spatial_max_sigma[: self.spatial_modes]),
            "innovation_channels": self.innovation_channels,
            "spatial_encoder_pooling": self.spatial_encoder_pooling,
        }


@dataclass(frozen=True)
class LatentEffect:
    """What one draw of ``z_t`` does to the physics. All batch-shaped ``[...]``.

    Held as an explicit object rather than as three loose tensors so that
    ``latent=None`` (no latent) and ``LatentEffect.identity()`` (latent present,
    drawn at its mean) are DIFFERENT things a reader can tell apart - the second
    is the ablation and the first is the G2 model.
    """

    log_intensity: Tensor
    head_rotation: Tensor
    log_wind_speed: Tensor
    #: [M7] Sigma-scaled coefficients of the LOW-RANK SPATIAL modes, ``[..., R]``,
    #: or ``None`` when the model has none. The COEFFICIENTS live here and the
    #: FIELD is built by the kernel, which is the only object that holds the
    #: burned state the basis is anchored to - the same split
    #: ``head_rotation`` already uses (the latent says how far to rotate, the
    #: kernel knows what it is rotating).
    spatial_intensity: Tensor | None = None
    #: [M7] ``sigma_m^2`` per spatial mode, ``[..., R]``, or ``None``. Carried
    #: alongside the draw because the mean-preserving correction for a spatial
    #: mode is a FIELD (``-0.5 sum_m sigma_m^2 phi_m(x)^2``) and can only be
    #: formed where the basis is.
    spatial_variance: Tensor | None = None

    @classmethod
    def identity(cls, like: Tensor) -> LatentEffect:
        z = torch.zeros_like(like)
        return cls(log_intensity=z, head_rotation=z, log_wind_speed=z)


def spatial_basis(burned: Tensor, n_modes: int, *, clip: float = 3.0) -> Tensor:
    """[M7] The FIRE-ANCHORED spatial basis ``phi_m``. ``[..., R, H, W]``.

    ``burned`` is the mean burned field ``b`` in ``[0, 1]``, ``[..., H, W]``.

    **ANCHORED TO THE FIRE, NOT TO THE DOMAIN, AND THAT IS THE WHOLE DESIGN.**
    The domain is a final-perimeter bbox buffered 10 km (C1.2), so a
    domain-anchored basis would mean something different for a fire at hour 3
    and the same fire at hour 100, and something different again for two fires
    of different size. Anchoring on the burned region's centroid and normalising
    by its RADIUS OF GYRATION makes ``phi`` scale-free and comparable across
    fires and hours: ``phi = +1`` is always "one fire-radius east of the fire".

    Modes, in :data:`SPATIAL_COMPONENTS` order::

        phi_0 = (col - cx) / rg          east  gradient
        phi_1 = (cy - row) / rg          north gradient   (C1.4: y DESCENDS)
        phi_2 = phi_0^2 + phi_1^2 - 1    radial: FRONT vs interior

    The first two are the "one flank runs, the other does not" innovation ADR-032
    (3) says a global scalar cannot represent. The third is mean-zero under the
    gyration normalisation by construction (``<phi_0^2 + phi_1^2>_b = 1``), so it
    is not a disguised copy of the global ``log_intensity`` mode.

    ``clip`` bounds the basis far from the fire. Without it a cell 20 radii away
    would see ``e^(20 sigma)``, which is not a correlated innovation but an
    unbounded extrapolation of a linear model - and it would be applied where the
    hazard is numerically zero anyway, so the clip costs nothing real.
    """
    r = int(n_modes)
    if r <= 0:
        raise ValueError(f"n_modes={r}; expected >= 1")
    h, w = burned.shape[-2], burned.shape[-1]
    rows = torch.arange(h, dtype=burned.dtype, device=burned.device).reshape(-1, 1)
    cols = torch.arange(w, dtype=burned.dtype, device=burned.device).reshape(1, -1)
    mass = burned.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-9)
    cy = (burned * rows).sum(dim=(-2, -1), keepdim=True) / mass
    cx = (burned * cols).sum(dim=(-2, -1), keepdim=True) / mass
    dy, dx = rows - cy, cols - cx
    # Radius of gyration, floored at ONE CELL: a single-cell fire has rg = 0 and
    # would divide by zero, and at 1 km cells a sub-cell gyration radius is
    # quantisation rather than shape (STATE: "persistence at 1 h is STRUCTURAL").
    rg = torch.sqrt((burned * (dy * dy + dx * dx)).sum(dim=(-2, -1), keepdim=True) / mass)
    rg = rg.clamp(min=1.0)
    # `dx` is [..., 1, W] and `dy` is [..., H, 1]; broadcast both to the full grid
    # BEFORE stacking, or `torch.stack` silently refuses (and `cat` would not).
    zero = torch.zeros_like(burned)
    east = (dx / rg + zero).clamp(-clip, clip)
    north = (-dy / rg + zero).clamp(-clip, clip)
    modes = [east, north, east * east + north * north - 1.0]
    return torch.stack(modes[:r], dim=-3)


class _Encoder(nn.Module):
    """``q(z_k | b_{k-1}, y_k, p_k^0)`` - amortised inference, TRAINING ONLY.

    Input channels, in order, all ``[..., H, W]``:

    0. ``b_{k-1}``  - the burned mean field entering the step;
    1. ``y_k``      - the OBSERVED burned field leaving it (the future);
    2. ``p_k^0``    - the model's own step probability at ``z = 0``.

    **[M7] THAT LAST PARAGRAPH USED TO CLAIM A PROPERTY THIS CODE DID NOT HAVE,
    AND IT IS MEASURED.** It said channel 2 makes this "an innovation encoder"
    because ``y_k - b_{k-1}`` against ``p_k^0`` is the residual - but the network
    was never GIVEN that difference, only the two large fields it is the
    difference of, and it had to recover it through two ReLU convolutions and a
    GLOBAL MEAN. ``b_{k-1}`` and ``y_k`` are near-identical burned blobs of
    hundreds of cells whose difference is a handful, so the innovation arrives as
    a ~1% perturbation on a large common mode and does not survive pooling.
    Measured on the shipped M6 gate checkpoint, 235 TRAIN windows: the posterior
    mean of the ACTIVITY GATE dimension differs between dormant and growing
    windows by **+0.037 against a spread of 0.426 - under a tenth of an SD, and
    with the WRONG SIGN.** ``q`` cannot see dormancy, so the conditional prior has
    nothing to regress, which is why ADR-032 (5)'s "the blocker is the fit of
    ``p(z_t|weather)``" bottoms out one level further down than it looks.
    ``innovation_channels`` therefore hands the encoder the DECOMPOSITION
    explicitly - realised new burn, expected new burn, and their difference -
    instead of asking it to subtract two large fields. Same family as a
    normalisation applied to one term of a comparison, and as the M2
    zero-gradient defect: a quantity structurally unable to move, presenting as a
    quantity that had no reason to.

    Global mean AND max pooling, because the two statistics separate the two
    things a step can do: burn a little everywhere (mean) versus make one run
    (max). A mean-only summary cannot distinguish them and the direction
    component would have nothing to key on.

    **[M7] THE INFERENCE NETWORK'S SUFFICIENT STATISTICS MUST SPAN THE
    GENERATIVE MODEL'S MODES, OR THE NEW DIMENSIONS ARE UNIDENTIFIABLE.** A
    globally-pooled encoder returns the SAME summary for "the east flank ran" and
    "the west flank ran", so it cannot infer the sign of a spatial gradient
    coefficient. ``q`` would then collapse to the prior on exactly those
    dimensions, the reconstruction would gain nothing from them, and their fitted
    ``sigma`` would be a statement about my pooling rather than about fire.
    So when the model has ``R`` spatial modes the encoder gains ``R`` extra
    pooled statistics, ``<h, phi_m> / <1, phi_m^2>`` - the projection of its own
    feature map onto the SAME basis the decoder uses. This is a symmetry
    requirement, not a capacity increase: it adds ``R * channels`` inputs to one
    linear head and nothing else.
    """

    def __init__(
        self,
        dim: int,
        channels: int,
        spatial_modes: int = 0,
        innovation_channels: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.spatial_modes = int(spatial_modes)
        self.innovation_channels = bool(innovation_channels)
        self.conv1 = nn.Conv2d(
            4 if self.innovation_channels else 3, channels, 3, padding=1, dtype=DTYPE
        )
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, dtype=DTYPE)
        self.head = nn.Linear((2 + self.spatial_modes) * channels, 2 * self.dim, dtype=DTYPE)
        # Start at q == prior: mu = 0, log_var = 0. The latent then has to be
        # EARNED from the first step, and a non-zero KL in the artifact is
        # evidence the fires moved it rather than evidence of an initialisation.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, b_prev: Tensor, y_now: Tensor, p_zero: Tensor, basis: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if self.innovation_channels:
            # `realised` and `expected` are both the NEW burn of this step, so
            # the huge common-mode burned blob is differenced out BEFORE the
            # convolution rather than after a global pool. Channel 3 is their
            # difference: the innovation z_t is defined to explain.
            realised = y_now - b_prev
            expected = p_zero * (1.0 - b_prev)
            x = torch.stack([b_prev, realised, expected, realised - expected], dim=-3)
        else:
            x = torch.stack([b_prev, y_now, p_zero], dim=-3)
        flat = x.reshape(-1, *x.shape[-3:])
        h = torch.relu(self.conv1(flat))
        h = torch.relu(self.conv2(h))
        stats = [h.mean(dim=(-2, -1)), h.amax(dim=(-2, -1))]
        if self.spatial_modes:
            if basis is None:
                raise ValueError(
                    "this encoder has spatial modes and was called without a basis: "
                    "a globally-pooled summary cannot identify a spatial coefficient"
                )
            phi = basis.reshape(-1, *basis.shape[-3:])[:, : self.spatial_modes]  # [N, R, H, W]
            # <h, phi_m> normalised by <phi_m, phi_m>, i.e. the least-squares
            # projection coefficient. Normalising matters: `intensity_radial` has
            # a much larger norm than the gradients, so an unnormalised inner
            # product would hand the head three statistics on wildly different
            # scales for no reason of physics.
            norm = (phi * phi).sum(dim=(-2, -1)).clamp(min=1e-9)  # [N, R]
            proj = torch.einsum("nchw,nmhw->nmc", h, phi) / norm.unsqueeze(-1)  # [N, R, C]
            stats.append(proj.reshape(proj.shape[0], -1))
        out = self.head(torch.cat(stats, dim=-1))
        mu, log_var = out[..., : self.dim], out[..., self.dim :]
        batch = x.shape[:-3]
        # log_var is clamped, not free: an unclamped log_var can drive the KL
        # term to +/- inf and take the whole run with it, and a run that dies is
        # indistinguishable from a run that had nothing to say.
        return mu.reshape(*batch, self.dim), log_var.reshape(*batch, self.dim).clamp(-8.0, 4.0)


class LatentHead(nn.Module):
    """The generative scales ``sigma`` plus the inference network.

    ``sigma`` is the whole model of the ensemble: ``z ~ N(0, I)`` is fixed, and
    what the fires teach is how large a step-level forcing error actually is.
    Parameterised through a scaled sigmoid so it stays in ``(0, max_sigma)``
    under unconstrained optimisation and cannot walk to a hazard multiplier of
    ``e^12``.
    """

    def __init__(self, config: LatentConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentConfig()
        dim = self.config.total_dim
        self.register_buffer(
            "max_sigma",
            torch.tensor(list(self.config.all_max_sigma), dtype=DTYPE),
            persistent=False,
        )
        init = torch.tensor(list(self.config.all_init_sigma), dtype=DTYPE)
        frac = torch.clamp(init / torch.clamp(self.max_sigma, min=1e-9), 1e-4, 1 - 1e-4)
        self.sigma_logit = nn.Parameter(torch.log(frac / (1.0 - frac)))
        self.encoder = _Encoder(
            dim,
            int(self.config.encoder_channels),
            self.config.spatial_modes if self.config.spatial_encoder_pooling else 0,
            self.config.innovation_channels,
        )
        # [M6] The PRIOR's mean. Zero on every dimension except the activity
        # gate, which carries `gate_prior_mean` so its marginal is a mixture.
        base = torch.zeros(dim, dtype=DTYPE)
        if self.config.dim >= 4 and self.config.gate_prior_mean:
            base[3] = float(self.config.gate_prior_mean)
        self.register_buffer("prior_mean_base", base, persistent=False)
        # [M6] Which dimensions get the log-normal mean correction. A BUFFER
        # rather than an index loop so `mean_correction()` is one differentiable
        # expression: an in-place index_put_ into a fresh tensor is correct here
        # today and is the kind of thing that silently stops carrying a gradient.
        mask = torch.zeros(dim, dtype=DTYPE)
        if self.config.mean_preserving:
            for i in (0, 2):
                if self.config.dim > i:
                    mask[i] = 1.0
        # [M8] The gate joins the -sigma^2/2 family, and ALSO needs its prior
        # OFFSET removed, because unlike dims 0 and 2 its prior mean is not zero.
        # Two buffers rather than one because the two terms scale differently in
        # sigma (quadratic vs linear) and folding them would hide that.
        shift = torch.zeros(dim, dtype=DTYPE)
        if self.config.gate_mean_preserving and self.config.dim >= 4:
            mask[3] = 1.0
            shift[3] = float(self.config.gate_prior_mean)
        self.register_buffer("log_multiplier_mask", mask, persistent=False)
        self.register_buffer("gate_prior_shift", shift, persistent=False)
        # [M6] CONDITIONAL PRIOR - the part of "CVAE-style" M5 left out. With a
        # fixed N(0, I) prior, z_t cannot know that a cold, humid, still hour is
        # a dormant hour, so no amount of sigma buys an OFF state IN THE RIGHT
        # HOURS. Zero-initialised, so training starts EXACTLY at the
        # unconditional prior and any conditioning is something the fires taught.
        self.prior_net: nn.Linear | None = None
        if self.config.conditional_prior:
            self.prior_net = nn.Linear(N_PRIOR_COVARIATES, dim, dtype=DTYPE)
            nn.init.zeros_(self.prior_net.weight)
            nn.init.zeros_(self.prior_net.bias)

    @property
    def dim(self) -> int:
        """Length of ``z`` - GLOBAL dimensions plus [M7] spatial modes."""
        return self.config.total_dim

    @property
    def global_dim(self) -> int:
        return self.config.dim

    @property
    def spatial_modes(self) -> int:
        return self.config.spatial_modes

    def sigma(self) -> Tensor:
        return self.max_sigma * torch.sigmoid(self.sigma_logit)

    def mean_correction(self) -> Tensor:
        """[M6] ``-sigma_k^2 / 2`` on the LOG-MULTIPLIER dimensions, else 0.

        A multiplicative error term should have unit mean: ``e^(sigma eps)`` with
        ``eps ~ N(0, 1)`` has mean ``e^(sigma^2/2)``, so WIDENING the ensemble
        also multiplies its mean hazard. Every term in the objective that
        constrains the mean (the reconstruction NLL, the pushforward Brier, the
        growth moment) then constrains ``sigma`` through the back door, and the
        mean shift lands in ``area_dispersion_ratio``'s DENOMINATOR as bias.
        Subtracting ``sigma^2/2`` makes ``E_z[e^effect] = 1`` exactly and turns
        ``sigma`` into a pure spread parameter. This is the standard log-normal
        correction, not a tuning knob: it has no free parameter.

        Applied to ``log_intensity`` (0) and ``log_wind_speed`` (2), which are
        multiplicative errors. NOT to ``head_rotation`` (1), which is additive in
        radians and already zero-mean, and - under ``mean_preserving`` alone -
        NOT to ``activity_gate`` (3), whose asymmetric prior is the point of the
        dimension.

        [M8] ``gate_mean_preserving`` adds dimension 3, and it needs a SECOND
        term the other dimensions do not: its prior mean is ``gate_prior_mean``
        rather than 0, so ``E[e^(sigma z)] = e^(sigma mu + sigma^2/2)`` and both
        parts must come off. The correction is therefore
        ``-(sigma_3 mu_gate + sigma_3^2 / 2)``, LINEAR plus QUADRATIC in sigma.
        The conditional prior's contribution to ``mu`` is deliberately left
        uncorrected - see :attr:`LatentConfig.gate_mean_preserving`.
        """
        sigma = self.sigma()
        return -0.5 * sigma * sigma * self.log_multiplier_mask - sigma * self.gate_prior_shift

    def effect(self, z: Tensor) -> LatentEffect:
        """Map ``z`` ``[..., dim]`` to its three physical modulations.

        Absent dimensions contribute exactly zero, so ``dim=1`` is "intensity
        only" rather than a differently-shaped model.
        """
        scaled = z * self.sigma() + self.mean_correction()
        zero = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
        n_global = self.global_dim

        def take(i: int) -> Tensor:
            return scaled[..., i] if n_global > i else zero

        # The ACTIVITY GATE acts on the SAME physical quantity as log_intensity -
        # a global log-multiplier on the hazard - so it is ADDED to it rather
        # than given its own entry point. What separates them is their scale and
        # their prior: log_intensity is a small symmetric forcing error on an
        # ACTIVE fire, the gate is a regime switch with a large sigma and a
        # negative prior mean, so the marginal over z is a MIXTURE of active and
        # quiet hours instead of a wobble around "always on".
        # [M7] The SPATIAL coefficients travel as coefficients, not as a field:
        # only the kernel holds the burned state the basis is anchored to.
        spatial = spatial_var = None
        if self.spatial_modes:
            spatial = scaled[..., n_global:]
            if self.config.mean_preserving:
                # `E_z[exp(sum_m c_m phi_m(x))] = exp(0.5 sum_m sigma_m^2 phi_m(x)^2)`,
                # so the mean-preserving correction for a spatial mode is a FIELD
                # and cannot be folded into `mean_correction()`. The kernel forms
                # it where the basis lives; the variance travels with the draw.
                sig = self.sigma()[n_global:]
                spatial_var = (sig * sig).expand_as(spatial)
        return LatentEffect(
            log_intensity=take(0) + take(3),
            head_rotation=take(1),
            log_wind_speed=take(2),
            spatial_intensity=spatial,
            spatial_variance=spatial_var,
        )

    def prior_mean(self, covariates: Tensor | None = None) -> Tensor:
        """``mu_p`` - ``[dim]``, or ``[..., dim]`` when conditioned on weather."""
        base = self.prior_mean_base
        if self.prior_net is None or covariates is None:
            return base
        return base + self.prior_net(covariates)

    def prior_sample(
        self, shape: tuple[int, ...], generator: Any = None, covariates: Tensor | None = None
    ) -> Tensor:
        """``z ~ N(mu_p, I)`` with an explicit generator, so C5 stays seed-exact."""
        eps = torch.randn(
            (*shape, self.dim), dtype=DTYPE, generator=generator, device=self.sigma().device
        )
        return self.prior_mean(covariates) + eps

    def posterior(
        self, b_prev: Tensor, y_now: Tensor, p_zero: Tensor, basis: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.encoder(b_prev, y_now, p_zero, basis)

    def kl(
        self,
        mu: Tensor,
        log_var: Tensor,
        prior_mean: Tensor | None = None,
        prior_var: float = 1.0,
    ) -> Tensor:
        """``KL(N(mu, e^log_var) || N(mu_p, s^2))``, per sample per dimension.

        ``mu_p`` defaults to this head's own prior mean, so the KL is measured
        against the prior the ENSEMBLE will actually be drawn from. Measuring it
        against ``N(0, I)`` while sampling from ``N(mu_p, I)`` would charge the
        posterior for the prior's own offset and make the gate expensive to use
        exactly where it is right.

        [M6] ``prior_var`` is the AR(1) TRANSITION variance ``1 - rho^2``. At
        ``rho = 0`` it is 1.0 and this is bitwise M5's KL. It is a separate
        argument rather than read from the config because the caller is the only
        thing that knows whether it is conditioning on ``z_{k-1}``.
        """
        mp = self.prior_mean() if prior_mean is None else prior_mean
        d = mu - mp
        s2 = float(prior_var)
        if s2 == 1.0:
            return 0.5 * (d * d + torch.exp(log_var) - 1.0 - log_var)
        return 0.5 * (d * d / s2 + torch.exp(log_var) / s2 - 1.0 - log_var + math.log(s2))

    def to_spec(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "sigma": [float(v) for v in self.sigma().detach()],
            "components": list(self.config.components),
        }


def reparameterise(mu: Tensor, log_var: Tensor, generator: Any = None) -> Tensor:
    """``mu + sigma * eps`` - gradients reach ``mu`` and ``log_var``, not ``eps``."""
    eps = torch.randn(mu.shape, dtype=mu.dtype, generator=generator, device=mu.device)
    return mu + torch.exp(0.5 * log_var) * eps


@dataclass(frozen=True)
class LatentSampler:
    """How an ensemble member gets its ``z_t``. THE ABLATION LIVES HERE.

    ``mode="latent"`` draws one ``z_t`` per member per step: pixels are
    conditionally independent GIVEN it, which is CLAUDE.md's model.
    ``mode="independent"`` holds ``z_t`` at the prior mean, so the only
    randomness left is the per-pixel Bernoulli draw - CLAUDE.md's known-broken
    model, built here ONLY as the G3 ablation.
    """

    mode: str = "latent"

    def __post_init__(self) -> None:
        if self.mode not in SAMPLER_MODES:
            raise ValueError(f"mode={self.mode!r}; expected one of {SAMPLER_MODES}")

    @property
    def is_ablation(self) -> bool:
        return self.mode == "independent"

    def draw(
        self,
        head: LatentHead | None,
        shape: tuple[int, ...],
        generator: Any = None,
        covariates: Tensor | None = None,
    ) -> LatentEffect | None:
        """One draw per entry of ``shape`` (normally ``(n_members,)``).

        ``None`` means "this model has no latent at all" and is NOT the same as
        the ablation: the ablation is a latent model sampled at ``z = 0``.
        """
        if head is None or head.dim == 0:
            return None
        if self.is_ablation:
            # The ablation holds z at the PRIOR MEAN, which with a conditional
            # prior is not the origin. Anything else would change the ensemble's
            # mean as well as its spread and make the collapse comparison a
            # comparison of two different forecasts.
            z = head.prior_mean(covariates).expand(*shape, head.dim).clone()
        else:
            z = head.prior_sample(shape, generator=generator, covariates=covariates)
        return head.effect(z)

    def path(self, head: LatentHead | None) -> LatentPath:
        """[M6] A stateful AR(1) draw over one rollout. See :class:`LatentPath`."""
        return LatentPath(head, self)

    def draw_path(
        self,
        head: LatentHead | None,
        shape: tuple[int, ...],
        horizon: int,
        generator: Any = None,
        covariates: Any = None,
    ) -> list[LatentEffect | None]:
        """[M6] ``horizon`` draws as ONE AR(1) PATH, not ``horizon`` independent calls.

        ``covariates`` is ``None`` or a sequence of length ``horizon``.
        """
        p = self.path(head)
        return [
            p.step(
                shape,
                generator=generator,
                covariates=None if covariates is None else covariates[k],
            )
            for k in range(int(horizon))
        ]

    def describe(self) -> str:
        if self.mode == "latent":
            return (
                "z_t ~ N(0, I) drawn ONCE PER MEMBER PER STEP and shared by every pixel; "
                "pixels are conditionally independent Bernoulli GIVEN (x_t, z_t)"
            )
        return (
            "ABLATION (CLAUDE.md known-broken): z_t held at the prior mean, so the only "
            "randomness is independent per-pixel Bernoulli. Required by G3 to DEMONSTRATE "
            "collapse; never a candidate."
        )


class LatentPath:
    """[M6] The AR(1) draw of ``z_1..z_H`` for ONE rollout, one step at a time.

    In the standardised coordinate ``u_k = z_k - mu_p_k``::

        u_1 ~ N(0, I)
        u_k = rho * u_{k-1} + sqrt(1 - rho^2) * eps_k

    so **every step's marginal is exactly ``N(mu_p_k, I)``**. The one-step model,
    the per-step KL and every M5 identity check are therefore untouched, and the
    ONLY thing this changes is the temporal dependence - which is what makes the
    intervention attributable.

    WHY IT EXISTS. ``z_t`` carries the step-level forcing error (RTMA wind bias,
    fuel-moisture bias, unresolved terrain). That error is AUTOCORRELATED at
    multi-hour scale: it is a BIAS, not a fresh coin every hour. With iid draws
    the H-step ensemble spread of a cumulative quantity grows as
    ``sqrt(H) sigma`` while its mean grows as ``H``, so an iid per-step latent
    UNDERSTATES multi-hour dispersion BY CONSTRUCTION - and G3 is adjudicated at
    1/2/3 h.

    **It is STATEFUL AND INCREMENTAL rather than a batch call ON PURPOSE.**
    :meth:`~wildfire_nowcast.model.kernel.ContagionKernel.predict` interleaves
    the latent draw with the per-pixel Bernoulli draw from ONE seeded generator.
    Pre-drawing a whole path would reorder that stream, so every M5 number would
    move for a reason that is not the intervention. Consuming exactly one
    ``randn`` per step, in place, keeps ``rho = 0`` BITWISE identical to M5.
    """

    def __init__(self, head: LatentHead | None, sampler: LatentSampler) -> None:
        self.head = head
        self.sampler = sampler
        self.u: Tensor | None = None
        self.rho = 0.0 if head is None else float(head.config.rho)
        self.scale = math.sqrt(max(1.0 - self.rho * self.rho, 0.0))

    def step(
        self,
        shape: tuple[int, ...],
        generator: Any = None,
        covariates: Tensor | None = None,
    ) -> LatentEffect | None:
        head = self.head
        if head is None or head.dim == 0:
            return None
        if self.sampler.is_ablation:
            return self.sampler.draw(head, shape, generator=generator, covariates=covariates)
        eps = torch.randn(
            (*shape, head.dim), dtype=DTYPE, generator=generator, device=head.sigma().device
        )
        self.u = eps if self.u is None else self.rho * self.u + self.scale * eps
        return head.effect(head.prior_mean(covariates) + self.u)


def _ar1_gain(rho: float, horizon: int) -> float:
    """SD of an AR(1) sum over ``horizon`` steps, relative to the iid sum.

    ``Var(sum_k u_k) = H + 2 sum_{l=1}^{H-1} (H - l) rho^l`` for unit-variance
    AR(1); the iid case is ``H``. Closed form, so the predicted effect of ``rho``
    on multi-hour spread is a number written down BEFORE the run rather than
    something read off a fitted model afterwards.
    """
    h = int(horizon)
    total = float(h)
    for lag in range(1, h):
        total += 2.0 * (h - lag) * (rho**lag)
    return math.sqrt(total / h)


def latent_report(head: LatentHead | None) -> dict[str, Any]:
    """Per-dimension sigma, in physical units, for the run artifact.

    The posterior-collapse detector: a latent whose sigma has been driven to
    ~0 is an ablation wearing the model's name, and the number that says so
    belongs in the artifact rather than in a reader's inference from a
    disappointing dispersion score.
    """
    if head is None or head.dim == 0:
        return {"present": False, "note": "no latent — this is the deterministic G2 kernel"}
    sigma = [float(v) for v in head.sigma().detach()]
    init = list(head.config.all_init_sigma)
    n_global = head.global_dim
    return {
        "present": True,
        "dim": head.dim,
        "global_dim": n_global,
        # [M7] The rank of the SPATIAL block, printed beside the global one so a
        # reader can see at a glance whether an ensemble could disagree about
        # WHERE or only about HOW MUCH.
        "spatial_modes": head.spatial_modes,
        "components": list(head.config.components),
        "sigma": dict(zip(head.config.components, sigma, strict=True)),
        "init_sigma": dict(zip(head.config.components, init, strict=True)),
        "sigma_over_init": [s / i for s, i in zip(sigma, init, strict=True)],
        "spatial_sigma": dict(
            zip(SPATIAL_COMPONENTS[: head.spatial_modes], sigma[n_global:], strict=True)
        ),
        # e^sigma at one gyration radius from the fire centroid, i.e. what a
        # one-SD draw does to the hazard at the fire's own edge. The number that
        # says whether the spatial modes are doing anything physical.
        "implied_edge_intensity_ratio_1sd": (
            [math.exp(s) for s in sigma[n_global:]] if head.spatial_modes else None
        ),
        "implied_intensity_ratio_1sd": math.exp(sigma[0]) if n_global >= 1 else None,
        "implied_rotation_deg_1sd": math.degrees(sigma[1]) if n_global >= 2 else None,
        # [M6] Both are properties of the PRIOR, not fitted against any score.
        "mean_preserving": head.config.mean_preserving,
        "innovation_channels": head.config.innovation_channels,
        "spatial_encoder_pooling": head.config.spatial_encoder_pooling,
        "mean_correction": [float(v) for v in head.mean_correction().detach()],
        "rho": head.config.rho,
        # With AR(1) the H-step cumulative innovation has SD
        # sigma * sqrt(H + 2*sum_{l<H}(H-l)rho^l) instead of sigma*sqrt(H).
        "horizon3_spread_gain_vs_iid": _ar1_gain(head.config.rho, 3),
        "collapse_warning": (
            "sigma is within 1% of zero on every dimension: the latent is OFF in all but "
            "name and its ensemble is the independent-per-pixel ablation"
            if all(s < 1e-2 for s in sigma)
            else None
        ),
    }
