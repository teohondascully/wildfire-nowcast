"""Training the contagion kernel. TRAIN fires only, by construction.

CLAUDE.md's training recipe is "one-step ELBO/NLL + multi-step pushforward
(3-6 h) Brier/CRPS". Without a latent the ELBO reduces to the NLL and the
objective is::

    L =  w_nll   * mean-over-leads Bernoulli NLL on the growth band
       + w_brier * mean-over-leads Brier on the growth band
       + w_growth* (log1p(E[new cells]) - log1p(observed new cells))^2

all evaluated on ONE free-running rollout per window, so lead 1 is the one-step
term and leads 2..H are the pushforward. Computing them from separate rollouts
would silently condition them differently.

[M5] THE ELBO, once ``latent_dim > 0``
--------------------------------------
With the shared per-step latent ``z_t`` present the first term becomes a genuine
ELBO over the sequence ``z_1..z_H`` (:mod:`wildfire_nowcast.model.latent`)::

    L = w_nll   * NLL(y | POSTERIOR path)        <- reconstruction
      + w_kl    * KL(q(z_k | b_{k-1}, y_k, p_k^0) || N(0, I))   summed over k
      + w_brier * Brier(y | MARGINAL prior predictive)
      + w_growth* growth moment on the MARGINAL

Two things about that split are deliberate and are the difference between
training a generative ensemble and training a mean field:

* the RECONSTRUCTION term runs the rollout under posterior draws, so it asks
  *could this fire's actual hour be explained by some z?*, while
* the BRIER and GROWTH terms run it under PRIOR draws averaged over
  ``n_prior_samples``, so they ask *is what we will actually sample right on
  average?* Scoring the marginal at ``z = 0`` instead would be a different
  quantity: ``p`` is concave in the hazard and the hazard is log-normal in ``z``.

**Nothing here trains the ensemble SPREAD directly, and that is on purpose.**
The spread is ``sigma``, and ``sigma`` is fitted by the ELBO's own trade-off —
the KL pays for width, the reconstruction pays for narrowness. G3 is adjudicated
on ``area_dispersion_ratio``, so a term that optimised area spread would be
tuning on the gate metric. If the fitted ensemble is under-dispersed, that is a
result about the model and it is reported as one.
``w_area_crps`` exists (a fair CRPS on the ensemble's burned AREA — a proper
score, and literally the "Brier/CRPS" CLAUDE.md names) but defaults to 0 and any
arm using it is declared as exploratory, because it is adjacent enough to the
gate metric that quietly enabling it would be indistinguishable from tuning.

Three decisions here are scientific and are argued where they are made:

1. **Arrival-time CRPS is not a separate term, and adding one would be
   double-counting.** For a monotone binary arrival process with
   ``F(k) = b_k``, the discrete CRPS/RPS is exactly ``sum_k (b_k - y_k)^2`` —
   the sum of the per-lead Brier scores. ``w_brier`` IS the arrival-CRPS weight.

2. **The growth-moment term is the remedy for the persistence attractor, and it
   is NOT class weighting** (STATE R14). Reweighting changes how much each
   window counts; a moment constraint changes what quantity is estimated. It is
   also the same rule ADR-011 imposes on the ellipse baseline, so G2 compares
   two models whose overall rate is pinned the same way and differs in shape.

3. **Fires are sampled with equal probability, not windows** (STATE R13). CZU is
   a coastal outlier with 47.8% barrier cells, and Dolan and Bobcat are 696 and
   433 hours against Zogg's 71. Window-uniform sampling would let the three
   largest fires supply ~80% of every gradient, and "the kernel learned the
   landscape of whichever fire ran longest" is not a transferable kernel.

Model selection never touches a held-out fire: the step budget is fixed a
priori and every reported training diagnostic is a TRAIN diagnostic.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wildfire_nowcast.common.contract import UNBURNED
from wildfire_nowcast.common.paths import fire_tensor_path, norm_stats_path
from wildfire_nowcast.common.runs import create_run_dir
from wildfire_nowcast.common.states import dilate
from wildfire_nowcast.common.zarr_io import get_channel, open_tensor, read_norm_stats
from wildfire_nowcast.eval.masks import default_band_radius, frontier
from wildfire_nowcast.model.direct import DirectHorizonKernel, parameter_counts
from wildfire_nowcast.model.inputs import (
    WEATHER_INPUT_CHANNELS,
    static_from_dataset,
    static_index,
    weather_from_dataset,
    weather_index,
)
from wildfire_nowcast.model.kernel import (
    DTYPE,
    SUSCEPTIBILITY_MODES,
    ContagionKernel,
    KernelConfig,
    StaticFields,
    check_latent_off_is_bit_identical,
    check_torch_matches_numpy,
    offset_anisotropy,
    offset_kernel_table,
    parameter_report,
    static_fields_from_array,
    step_covariates,
    susceptibility_gradient_report,
)
from wildfire_nowcast.model.labelnoise import (
    IDENTITY,
    FireNoiseModel,
    LabelPerturbation,
    apply_perturbation,
    growth_band_field,
    noise_model_for,
    perturbation_report,
    sample_perturbation,
)
from wildfire_nowcast.model.latent import (
    LatentConfig,
    LatentSampler,
    latent_report,
    reparameterise,
    spatial_basis,
)
from wildfire_nowcast.model.spread import EllipseParams

__all__ = [
    "TrainConfig",
    "M5_MATRIX",
    "M6_MATRIX",
    "M6_ACTIVITY_PREREGISTRATION",
    "M5_SELECTION_RULE",
    "M5_DECLARED_ENTRIES",
    "train_ensemble_diagnostics",
    "wind_sector_report",
    "FireTensors",
    "load_fire",
    "train_kernel",
    "growth_diagnostics",
    "run_matrix",
    "M8_MATRIX",
    "M8_SELECTION_RULE",
    "M8_DECLARED_ENTRIES",
    "M8_PREDICTIONS",
    "M8_ELLIPSE_TRANSFER",
    "M8_GATE_EXTRA_TRANSFER_LOSS",
    "M8_CONDITIONAL_PRIOR_TRANSFER",
    "M3_MATRIX",
    "M3_SELECTION_RULE",
    "M3_DECLARED_ENTRIES",
    "M2_MATRIX",
    "main",
]

_PROB_EPS = 1e-7


@dataclass
class TrainConfig:
    """Everything that changes a run. Written verbatim into ``runs/{id}/config.yaml``."""

    horizon_h: int = 3
    radius: int = 3
    steps: int = 300
    batch_size: int = 12
    learning_rate: float = 0.05
    w_nll: float = 1.0
    w_brier: float = 2.0
    w_growth: float = 0.05
    #: "batch" = aggregate growth-moment constraint (the remedy of record).
    #: "window" = per-window log-growth regression (FALSIFIED — it inherits the
    #: zero-inflation and pulls toward silence; kept so the failure reproduces).
    growth_moment: str = "batch"
    #: EMA momentum for the aggregate growth moment (0 = raw batch estimate).
    growth_ema_momentum: float = 0.8
    #: Restrict the objective to windows whose truth grows. The R14 ABLATION,
    #: default off. Kept as a switch rather than a branch so P4 is one config
    #: change and not one code change.
    growth_windows_only: bool = False
    #: Multiply the loss of growth windows. The other half of the R14 ablation.
    growth_window_weight: float = 1.0
    #: Head-rate scale of the ellipse the kernel is INITIALISED from. Overridden
    #: at init by the growth calibration unless ``calibrate_alpha`` is False.
    ellipse_scale: float = 1.0
    calibrate_alpha: bool = True
    #: Where susceptibility enters the log-weight. "amplitude" is the ADR-015
    #: (6a) fix; "reach" reproduces M2's exactly-zero-gradient defect and is a
    #: CONTROL, never a candidate.
    susceptibility_mode: str = "amplitude"
    #: Marginalise the loss over the measured GOFER observation noise
    #: (:mod:`wildfire_nowcast.model.labelnoise`). ADR-015 (6b).
    label_perturbation: bool = False
    #: Multiplies both the offset sigma and the morph probability. 1.0 = the
    #: measured East-vs-West disagreement; 0 would be the identity.
    label_perturbation_scale: float = 1.0
    seed: int = 20260808
    fire_uniform_sampling: bool = True
    #: Cosine-decay the learning rate to 10% of its start. With the growth
    #: moment in the loss the raw-SGD trajectory oscillates (measured: the term
    #: excursed to 16.97 at step 100 of one run), and a final-step parameter
    #: vector is then a sample of the oscillation, not the optimum.
    lr_cosine_decay: bool = True
    #: Polyak-average the parameters over the last fraction of steps. Same
    #: reason: report the centre of the trajectory, not wherever it stopped.
    polyak_tail: float = 0.25
    # -- [M5] the shared per-step latent z_t --------------------------------
    #: Dimensions of ``z_t`` taken from the head of
    #: :data:`~wildfire_nowcast.model.latent.LATENT_COMPONENTS`. **0 = NO LATENT
    #: and is the default**, so every pre-M5 config reproduces its own result
    #: bitwise and the G2 kernel is not silently redefined by a new default.
    latent_dim: int = 0
    #: Weight on the KL term of the ELBO. 1.0 is the ELBO as written; below 1.0
    #: is a beta-VAE and is a declared departure, not a free knob.
    w_kl: float = 1.0
    #: Free bits per latent dimension, in nats (Kingma et al. 2016). The declared
    #: remedy for posterior collapse: a 3-nat latent competing with a 10^4-cell
    #: reconstruction term is otherwise free to switch itself off, and "the
    #: latent did nothing" would then be a statement about the optimiser.
    latent_free_bits: float = 0.02
    #: Prior draws used to estimate the MARGINAL predictive that the Brier and
    #: growth terms are computed on. More is a better estimate of the same
    #: quantity, never a different objective.
    n_prior_samples: int = 4
    #: [M6] Prior mean of the ACTIVITY GATE dimension (``latent_dim >= 4``), in
    #: z-space. Negative puts prior mass on an OFF hour, so the marginal over z
    #: is a MIXTURE of active and quiet hours rather than a wobble around
    #: "always on". 0 disables the offset.
    gate_prior_mean: float = 0.0
    #: [M6] Condition the prior on the step's own weather (mean RH, temperature,
    #: wind speed). Without it z_t cannot know WHICH hour is dormant, so no
    #: amount of sigma buys an OFF state in the right hours.
    conditional_prior: bool = False
    #: [M6] Use the FINITE-ENSEMBLE-UNBIASED ("fair") pushforward Brier instead of
    #: the plug-in. See :func:`_fair_brier_correction`: the plug-in adds
    #: ``Var_z(b)/S`` — the ensemble variance — to the loss as a sharpness bonus,
    #: which is ADR-027 (3)'s measured cause of the ~4x-too-narrow ensemble.
    #: This changes the ESTIMATOR of the mandated score, never the estimand, and
    #: has NO free parameter. ``False`` reproduces M5 bitwise.
    fair_brier: bool = False
    #: [M6] Mean-preserving multiplicative latent (``E_z[e^effect] = 1``), so
    #: ``sigma`` is a pure spread parameter and does not also move the ensemble
    #: mean. See :meth:`~wildfire_nowcast.model.latent.LatentHead.mean_correction`.
    mean_preserving_latent: bool = False
    #: [M8] Extend that correction to the ACTIVITY GATE (latent dim 3). Reverses
    #: an M6 exemption whose premise (protect the OFF state) ADR-034 (4)
    #: falsified. See `LatentConfig.gate_mean_preserving`.
    gate_mean_preserving_latent: bool = False
    #: [M6] AR(1) persistence of ``z_t`` across steps. 0 = M5's iid draws.
    #: MEASURED on TRAIN fires (:func:`innovation_autocorrelation`), never fitted
    #: against a held-out score or a gate criterion.
    latent_rho: float = 0.0
    #: [M7] LOW-RANK SPATIAL modes appended to ``z_t``
    #: (:data:`~wildfire_nowcast.model.latent.SPATIAL_COMPONENTS`). **0 = M6,
    #: bitwise**, and is the default. This is the degree of freedom ADR-032 (3)
    #: identifies as missing: a global scalar latent can only widen the ensemble
    #: by blurring it uniformly. It is NOT a per-pixel noise field — rank R, not
    #: rank H*W — because CLAUDE.md records per-pixel-independent noise as
    #: known-broken and admits it only as an ablation.
    spatial_modes: int = 0
    #: [M7] Give the inference network the innovation decomposition explicitly.
    #: **False = M6, bitwise.** MEASURED cause: on the shipped M6 gate checkpoint
    #: the posterior mean of the ACTIVITY GATE separates dormant from growing
    #: windows by +0.037 against a spread of 0.426 — under a tenth of an SD, and
    #: with the wrong sign. `q` cannot see dormancy, so `p(z_t|weather)` has
    #: nothing to regress. See `runs/m7_offstate_optimum.json`.
    innovation_encoder: bool = False
    #: [M7] NEGATIVE CONTROL for the spatial latent's identifiability. See
    #: :class:`~wildfire_nowcast.model.latent.LatentConfig`.
    spatial_encoder_pooling: bool = True
    #: Fair CRPS on the ensemble's burned AREA. **Defaults to 0 and any arm that
    #: turns it on is DECLARED EXPLORATORY**: it is a proper score of the area
    #: distribution, which is close enough to G3's `area_dispersion_ratio` that
    #: enabling it silently would be indistinguishable from tuning on the gate.
    w_area_crps: float = 0.0
    band_radius_cells: int | None = None
    #: [M10 / ADR-045] Fit the DIRECT-HORIZON head (arm B) instead of the
    #: free-running rollout (arm A). **False = arm A, bitwise** — verified by
    #: `runs/_m10_bitidentity.py` against the incumbent checkpoint's own outputs,
    #: not by inspection. See :mod:`wildfire_nowcast.model.direct` for why B's
    #: stencil is multi-scale with tied weights (the two obvious constructions
    #: violate ADR-045 (3)'s capacity or reach conditions).
    direct_horizon: bool = False
    #: North-south mirror ABLATION (:func:`mirror_north_south`). Diagnostic only:
    #: a model fitted with this on must never be evaluated against real held-out
    #: fires, because it was fitted to a world that does not exist. It answers one
    #: question — whether the learned S/SW anisotropy is in the data or in my code.
    mirror_ns: bool = False
    #: [S1 / ADR-061 (6)] Fit ARM S: the incumbent kernel plus ONE scalar input,
    #: log burned area (:class:`~wildfire_nowcast.model.stagehead.StageHead`,
    #: 4 parameters). **False = arm A, bitwise** — measured by
    #: ``runs/_s1_bitidentity.py`` against the incumbent checkpoint's own outputs.
    #: Orthogonal to ``direct_horizon``: S1 tests a COVARIATE, M10 tested a
    #: horizon treatment, and combining them would confound two arms.
    stage_scalar: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FireTensors:
    """One fire, fully resident: static fields, weather, labels, band masks."""

    fire_id: str
    fields: StaticFields
    weather: Tensor  # [T, C_w, H, W]
    burned: Tensor  # [T, H, W] in {0., 1.}
    band: Tensor  # [T, H, W] bool — growth band from x0 at each t0
    t0_index: np.ndarray  # usable analysis times
    shape: tuple[int, int]
    static_array: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def n_windows(self) -> int:
        return int(self.t0_index.size)


def mirror_north_south(
    state: np.ndarray, static: np.ndarray, weather: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflect one fire about an east-west axis, IN MEMORY ONLY. No tensor is touched.

    This is the decisive discriminator for insights item 25's remaining rival B:
    the learned offset weights grow a wind-INDEPENDENT preference at bearing
    ~201 deg (S/SSW) in 8 of 8 fits, and the label-perturbation ensemble did not
    remove it. Two explanations survive — a real property of the landscape/weather,
    or an artefact of my own offset parameterisation and array orientation.
    They make OPPOSITE predictions under a mirror, so one run separates them::

        physics / data  ->  bearing maps theta -> (180 - theta) mod 360, i.e. ~339
        code artefact   ->  bearing STAYS at ~201, because the model never sees
                            the world, only the array

    C1.4 fixes ``y`` DESCENDING (row 0 = north), so ``flip(axis=-2)`` is a true
    north-south reflection of the ground. The vector channels must be reflected
    too, or this tests nothing:

    * ``wind_v10`` is the NORTHWARD component            -> negate.
    * ``wind_u10`` is EASTWARD, unchanged by an N-S mirror.
    * aspect bearing maps ``theta -> 180 - theta``, so
      ``aspect_sin = sin(theta)`` is UNCHANGED (its east component survives) and
      ``aspect_cos = cos(theta)`` is NEGATED. Getting this pair backwards would
      make the ablation quietly meaningless, so it is spelled out.

    Everything else (elevation, slope, fuels, canopy, masks, scalar weather) is a
    scalar field and only moves spatially.
    """
    state_m = np.ascontiguousarray(np.flip(state, axis=-2))
    static_m = np.ascontiguousarray(np.flip(static, axis=-2))
    weather_m = np.ascontiguousarray(np.flip(weather, axis=-2))
    static_m[static_index("aspect_cos")] *= -1.0
    weather_m[:, weather_index("wind_v10")] *= -1.0
    return state_m, static_m, weather_m


def load_fire(
    fire_id: str,
    horizon_h: int,
    *,
    band_radius_cells: int | None = None,
    mirror_ns: bool = False,
) -> FireTensors:
    """Read one C1 store into resident tensors, with the C1.3 phase applied once.

    ``weather[t]`` here is indexed by ABSOLUTE time, and a window at ``t0`` uses
    ``weather[t0+1 : t0+1+H]`` — the same ``t0+1+k`` phase
    :func:`~wildfire_nowcast.model.inputs.forecast_inputs` applies, taken from
    that module's convention rather than re-derived.

    ``mirror_ns`` is the north-south reflection ablation of
    :func:`mirror_north_south`. It is an IN-MEMORY transform of a read-only load;
    it never writes, and data's tensors are not mine to modify.
    """
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
        static = static_from_dataset(ds)
        weather_np = np.stack(
            [get_channel(ds, name).values.astype(np.float64) for name in WEATHER_INPUT_CHANNELS],
            axis=1,
        )
    finally:
        ds.close()
    # Phase self-check: window t0 must see exactly what forecast_inputs would
    # hand it. Re-deriving the C1.3 phase in a second place is how a model ends
    # up an hour out of step with its weather and merely looks mediocre.
    if weather_np.shape[0] > 2:
        probe = weather_from_dataset(open_tensor(fire_tensor_path(fire_id)), 0, 1)
        if not np.allclose(probe[0], weather_np[1], equal_nan=True):
            raise AssertionError(
                f"{fire_id}: weather phase here disagrees with inputs.forecast_inputs "
                "(C1.3 end-of-hour). Refusing to train an hour out of phase."
            )

    if mirror_ns:
        # AFTER the phase check, deliberately: the check compares against a fresh
        # read of the store, so mirroring first would fail it for the wrong reason
        # and I would be debugging my own ablation instead of running it.
        state, static, weather_np = mirror_north_south(state, static, weather_np)

    burned = (state > UNBURNED).astype(np.float64)
    radius = (
        int(band_radius_cells) if band_radius_cells is not None else default_band_radius(horizon_h)
    )
    band = np.zeros(burned.shape, dtype=bool)
    usable = []
    for t0 in range(burned.shape[0] - horizon_h):
        b0 = burned[t0] > 0
        if not b0.any():
            continue
        seed = frontier(state[t0])
        band[t0] = dilate(seed, radius) & ~b0
        usable.append(t0)

    return FireTensors(
        fire_id=fire_id,
        fields=static_fields_from_array(static),
        weather=torch.as_tensor(weather_np, dtype=DTYPE),
        burned=torch.as_tensor(burned, dtype=DTYPE),
        band=torch.as_tensor(band),
        t0_index=np.asarray(usable, dtype=np.int64),
        shape=(int(burned.shape[1]), int(burned.shape[2])),
        static_array=static,
    )


def _batch(
    fire: FireTensors,
    t0s: np.ndarray,
    horizon_h: int,
    *,
    perturbation: LabelPerturbation | None = None,
    band_radius_cells: int | None = None,
) -> dict[str, Tensor]:
    """Assemble one batch of windows from ONE fire (they share static fields).

    ``perturbation`` is ONE draw of the observation noise, applied identically to
    ``b0`` and to every lead of ``truth`` — see
    :func:`wildfire_nowcast.model.labelnoise.apply_perturbation` for why the
    single shared draw is the whole point. The band is RECOMPUTED from the
    perturbed ``b0``, never carried over, or the model would be scored on cells
    the perturbed fire was never near.

    The weather is NOT perturbed. RTMA is a different instrument with a different
    error; displacing the labels to marginalise GOFER's geolocation while
    dragging the wind field along with them would model a nonexistent joint error.
    """
    b0 = torch.stack([fire.burned[int(t)] for t in t0s])
    weather = torch.stack(
        [fire.weather[int(t) + 1 : int(t) + 1 + horizon_h] for t in t0s]
    )  # [B, H, C_w, Hh, Ww]
    truth = torch.stack([fire.burned[int(t) + 1 : int(t) + 1 + horizon_h] for t in t0s])
    band = torch.stack([fire.band[int(t)] for t in t0s])
    applied = IDENTITY
    if perturbation is not None and not perturbation.is_identity:
        cand_b0 = apply_perturbation(b0, perturbation)
        # An erosion that empties a window's fire leaves nothing to spread from,
        # and a window with no seed is not a harder training example, it is an
        # absent one. Drop the morph rather than the window, so the batch keeps
        # its size and the shift (the part that matters for the artefact) stands.
        if float(cand_b0.sum(dim=(1, 2)).min()) < 1.0:
            perturbation = LabelPerturbation(
                perturbation.shift_r, perturbation.shift_c, 0, perturbation.fconf
            )
            cand_b0 = apply_perturbation(b0, perturbation)
        if not perturbation.is_identity and float(cand_b0.sum(dim=(1, 2)).min()) >= 1.0:
            radius = (
                int(band_radius_cells)
                if band_radius_cells is not None
                else default_band_radius(horizon_h)
            )
            b0 = cand_b0
            truth = apply_perturbation(truth, perturbation)
            band = growth_band_field(b0, radius)
            applied = perturbation
    return {
        "b0": b0,
        "weather": weather,
        "truth": truth,
        "band": band,
        "_perturbation": applied,
    }


def _losses(
    model: ContagionKernel,
    fire: FireTensors,
    batch: dict[str, Tensor],
    horizon_h: int,
    config: TrainConfig,
    ema: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    """One free-running rollout -> every loss term. Lead 1 IS the one-step term.

    [M5] With a latent present this returns the ELBO's two terms plus the
    MARGINAL Brier/growth; see the module docstring for why the reconstruction
    runs under the posterior and the scoring terms under the prior.
    """
    y = batch["truth"]
    latent_terms = (
        _elbo_terms(model, fire, batch, horizon_h, config) if model.latent is not None else None
    )
    if latent_terms is None:
        b = model.rollout(batch["b0"], batch["weather"], fire.fields, horizon_h)  # [B,L,H,W]
        b_score = b
    else:
        b = latent_terms["b_posterior"]  # reconstruction path
        b_score = latent_terms["b_marginal"]  # prior predictive, the thing we sample
    mask = batch["band"].unsqueeze(1).expand_as(b)
    n = torch.clamp(mask.sum(), min=1.0)

    p = torch.clamp(b, _PROB_EPS, 1.0 - _PROB_EPS)
    nll = -(y * torch.log(p) + (1.0 - y) * torch.log1p(-p))
    nll = (nll * mask).sum() / n
    brier_plugin = (((b_score - y) ** 2) * mask).sum() / n
    # [M6] THE FAIR (finite-ensemble-unbiased) PUSHFORWARD BRIER. See
    # `_fair_brier_correction` — this is the SAME score, estimated without the
    # sharpness bias a 4-sample plug-in carries.
    correction = torch.zeros((), dtype=DTYPE)
    if config.fair_brier and latent_terms is not None:
        correction = (latent_terms["marginal_var_over_s"] * mask).sum() / n
    brier = brier_plugin - correction
    b = b_score  # the growth moment constrains what we SAMPLE, not the posterior

    unburned0 = (batch["b0"] < 0.5).unsqueeze(1)
    predicted_growth = ((b[:, -1:] - batch["b0"].unsqueeze(1)) * unburned0).sum(dim=(1, 2, 3))
    observed_growth = ((y[:, -1:] - batch["b0"].unsqueeze(1)) * unburned0).sum(dim=(1, 2, 3))
    # AGGREGATE moment, not a per-window regression. Measured, and it is the
    # difference between a remedy and a second copy of the disease: the
    # per-window form `mean_w (log1p(pred_w) - log1p(obs_w))^2` targets each
    # window's own growth, and since 63% of windows observe bitwise zero it
    # pulls the model toward silence exactly like the loss it was added to
    # correct. The batch total is the quantity ADR-011 pins on the ellipse, so
    # this is the same rule applied to the kernel.
    if config.growth_moment == "window":
        growth = (
            (
                torch.log1p(torch.clamp(predicted_growth, min=0.0))
                - torch.log1p(torch.clamp(observed_growth, min=0.0))
            )
            ** 2
        ).mean()
    else:
        # ... and estimated with an EMA rather than from the batch alone.
        # Measured: a 12-window batch estimate of an aggregate moment is
        # dominated by whether a burst hour happened to be drawn, and the raw
        # batch form oscillated (growth term 0.51 -> 5.74 -> 0.08 within 40
        # steps) and overshot to 1.71x. The EMA makes the constraint apply to
        # the long-run rate, which is the quantity the rule actually names.
        pred_now = torch.clamp(predicted_growth.sum(), min=0.0)
        obs_now = float(torch.clamp(observed_growth.sum(), min=0.0))
        if ema is None:
            pred_eff, obs_eff = pred_now, obs_now
        else:
            m = float(config.growth_ema_momentum)
            if not ema:
                ema["pred"], ema["obs"] = float(pred_now.detach()), obs_now
            pred_eff = m * ema["pred"] + (1.0 - m) * pred_now
            obs_eff = m * ema["obs"] + (1.0 - m) * obs_now
            ema["pred"], ema["obs"] = float(pred_eff.detach()), float(obs_eff)
        growth = (torch.log1p(pred_eff) - math.log1p(obs_eff)) ** 2

    total = config.w_nll * nll + config.w_brier * brier + config.w_growth * growth
    out = {
        "loss": total,
        "nll": nll.detach(),
        "brier": brier.detach(),
        "brier_plugin": brier_plugin.detach(),
        "brier_sharpness_bias": correction.detach(),
        "growth": growth.detach(),
        "predicted_growth": predicted_growth.detach().sum(),
        "observed_growth": observed_growth.detach().sum(),
        "kl": torch.zeros((), dtype=DTYPE),
        "area_crps": torch.zeros((), dtype=DTYPE),
    }
    if latent_terms is not None:
        # THE KL IS DIVIDED BY THE SAME `n` THE RECONSTRUCTION TERM IS, AND THIS
        # IS A BUG FIX, NOT A WEIGHT CHOICE. A correct ELBO is
        # `sum_cells log p(y|z) - KL`; ours reports the reconstruction as a
        # per-cell MEAN. Leaving the KL unnormalised beside it therefore
        # multiplies the effective KL weight by the number of scored cells —
        # measured here as ~3,000 — so `w_kl = 1.0` was silently `w_kl = 3000`.
        # The observable symptom was posterior collapse that reads as "the
        # latent had nothing to say": KL per dimension sat at 0.002-0.016 nats,
        # i.e. exactly the free-bits allowance and not one nat more. Same class
        # as every other defect in this project's record — a normalisation
        # applied to one term of a comparison and not to the other.
        out["loss"] = total + config.w_kl * latent_terms["kl"] / n
        out["kl"] = latent_terms["kl"].detach()
        out["kl_scale_n_cells"] = n.detach()
        out["kl_raw"] = latent_terms["kl_raw"].detach()
        out["kl_per_dim"] = latent_terms["kl_per_dim"].detach()
        if config.w_area_crps > 0:
            out["loss"] = out["loss"] + config.w_area_crps * latent_terms["area_crps"]
            out["area_crps"] = latent_terms["area_crps"].detach()
    return out


def _elbo_terms(
    model: ContagionKernel,
    fire: FireTensors,
    batch: dict[str, Tensor],
    horizon_h: int,
    config: TrainConfig,
) -> dict[str, Tensor]:
    """The two ELBO terms plus the marginal prior predictive. ``model.latent`` required.

    The POSTERIOR path is a free-running rollout driven by ``z_k ~ q(. | b_{k-1},
    y_k, p_k^0)``: at each step the encoder sees the state the model has actually
    reached, the truth that step produced, and the model's own ``z = 0``
    prediction — i.e. it encodes the INNOVATION, which is the quantity ``z_t`` is
    defined to carry. ``p_k^0`` and ``b_{k-1}`` enter the encoder DETACHED so the
    inference network cannot back-propagate into the physics: ``q`` is there to
    invert the decoder, not to reshape it.

    The MARGINAL path averages ``n_prior_samples`` independent prior rollouts.
    Each sample draws a fresh ``z`` PER STEP, exactly as :meth:`predict` does, so
    the training-time marginal and the evaluation-time ensemble are estimates of
    the same distribution rather than two conventions that happen to be close.
    """
    latent = model.latent
    assert latent is not None
    b0, weather, y = batch["b0"], batch["weather"], batch["truth"]
    rho = float(latent.config.rho)
    transition_var = max(1.0 - rho * rho, 1e-12)

    # -- posterior path (reconstruction) ---------------------------------
    b = b0
    posterior = []
    kl_per_dim = torch.zeros(latent.dim, dtype=DTYPE)
    z_prev: Tensor | None = None
    mu_p_prev: Tensor | None = None
    # [M10] A DIRECT-HORIZON head anchors every lead on the window's ORIGIN; a
    # rollout anchors each step on the state it reached. The encoder must see the
    # same field the decoder modulates or the spatial latent dimensions describe a
    # different fire from the one they act on — which would not crash, and would
    # not show up anywhere except as a latent that mysteriously learned nothing.
    anchor_on_origin = bool(getattr(model, "anchors_on_origin", False))
    for k in range(horizon_h):
        w_k = weather[:, k]
        p_zero = model.step_probability_at(b, weather, fire.fields, k, None, b0).detach()
        # [M7] The encoder sees the SAME fire-anchored basis the decoder applies.
        # Without it a globally-pooled q cannot tell "the east flank ran" from
        # "the west flank ran", the spatial dimensions are UNIDENTIFIABLE, and
        # their fitted sigma would be a statement about my pooling rather than
        # about fire. Detached with the rest of the encoder's inputs: q inverts
        # the decoder, it does not reshape it.
        basis_anchor = b0 if anchor_on_origin else b
        basis_k = (
            spatial_basis(basis_anchor.detach(), latent.spatial_modes)
            if latent.spatial_modes
            else None
        )
        mu, log_var = latent.posterior(b.detach(), y[:, k], p_zero, basis_k)
        z = reparameterise(mu, log_var)
        p = model.step_probability_at(b, weather, fire.fields, k, latent.effect(z), b0)
        b = b + (1.0 - b) * p
        posterior.append(b)
        # [M6] The KL is measured against THIS STEP's prior, which under a
        # conditional prior depends on the step's own weather. Measuring it
        # against N(0, I) while sampling from N(mu_p(w), I) would charge the
        # posterior for the prior's own conditioning and make the activity gate
        # most expensive exactly in the hours where it is right.
        mu_p = latent.prior_mean(
            step_covariates(w_k) if latent.prior_net is not None else None
        )
        # [M6] Under AR(1) the sequence prior FACTORISES as
        # p(z_1) prod_k p(z_k | z_{k-1}), so the exact ELBO's KL for step k>1 is
        # against N(mu_p_k + rho(z_{k-1} - mu_p_{k-1}), (1 - rho^2) I) and NOT
        # against the marginal. Scoring against the marginal would charge the
        # posterior for information the prior already has, i.e. would price
        # persistence as if it were surprise. At rho = 0 both branches are the
        # same expression and the M5 KL reproduces bitwise.
        if rho > 0.0 and z_prev is not None and mu_p_prev is not None:
            kl_step = latent.kl(
                mu, log_var, mu_p + rho * (z_prev.detach() - mu_p_prev), transition_var
            )
        else:
            kl_step = latent.kl(mu, log_var, mu_p)
        kl_per_dim = kl_per_dim + kl_step.mean(dim=0)
        z_prev, mu_p_prev = z, mu_p

    # FREE BITS: a dimension carrying less than `latent_free_bits` nats is not
    # charged. Without it the KL of a 3-vector is negligible beside a
    # ~10^4-cell reconstruction term in absolute size but is the ONLY term that
    # can be driven to exactly zero, so the optimiser's cheapest move is to
    # switch the latent off and report an ablation as a model.
    kl_raw = kl_per_dim.sum()
    floor = float(config.latent_free_bits)
    kl = torch.clamp(kl_per_dim, min=floor).sum() - floor * latent.dim

    # -- marginal path (what predict() will actually sample) --------------
    samples = []
    covariates = (
        [step_covariates(weather[:, k]) for k in range(horizon_h)]
        if latent.prior_net is not None
        else None
    )
    for _ in range(max(1, int(config.n_prior_samples))):
        # [M6] ONE AR(1) path per sample, through the same entry point
        # `predict` uses, so the training-time marginal and the evaluation-time
        # ensemble stay estimates of the SAME distribution rather than two
        # conventions that happen to agree at rho = 0.
        draws = LatentSampler("latent").draw_path(
            latent, (b0.shape[0],), horizon_h, covariates=covariates
        )
        samples.append(model.rollout(b0, weather, fire.fields, horizon_h, draws))
    stacked = torch.stack(samples)  # [S, B, L, H, W]
    b_marginal = stacked.mean(dim=0)
    marginal_var_over_s = _fair_brier_correction(stacked)

    # Fair CRPS on the burned AREA inside the band. Off by default (w_area_crps
    # = 0) and declared exploratory wherever it is on — see the module docstring.
    mask = batch["band"].unsqueeze(1)
    areas = (stacked * mask.unsqueeze(0)).sum(dim=(-2, -1))  # [S, B, L]
    truth_area = (y * mask).sum(dim=(-2, -1))  # [B, L]
    s = areas.shape[0]
    if s > 1:
        spread = torch.abs(areas.unsqueeze(0) - areas.unsqueeze(1)).sum(dim=(0, 1)) / (
            2.0 * s * (s - 1)
        )
    else:
        spread = torch.zeros_like(truth_area)
    area_crps = (torch.abs(areas - truth_area.unsqueeze(0)).mean(dim=0) - spread).mean()

    return {
        "b_posterior": torch.stack(posterior, dim=1),
        "b_marginal": b_marginal,
        "marginal_var_over_s": marginal_var_over_s,
        "kl": kl,
        "kl_raw": kl_raw,
        "kl_per_dim": kl_per_dim,
        "area_crps": area_crps,
    }


def _fair_brier_correction(stacked: Tensor) -> Tensor:
    """[M6] ``s^2 / S`` — the SHARPNESS BIAS of a plug-in pushforward Brier.

    **THE DEFECT, DERIVED RATHER THAN SEARCHED FOR.** The pushforward term scores
    the predictive MARGINAL ``p_bar = E_z[b(z)]``, but it can only ever see a
    Monte-Carlo estimate ``p_hat_S`` built from ``S = n_prior_samples`` prior
    rollouts. For any unbiased estimator of a mean,

        E[(p_hat_S - y)^2] = (p_bar - y)^2 + Var_z(b) / S

    so **the plug-in estimator of the Brier score is BIASED, and the bias is
    exactly the ensemble variance divided by S.** Minimising it therefore adds an
    explicit ``+Var_z(b)/S`` PENALTY ON ENSEMBLE SPREAD that has nothing to do
    with forecast quality — it is an artifact of estimating a marginal with four
    samples. At ``S = 4``, ``w_brier = 2`` and ``n ~ 3,000`` scored cells, that
    penalty outweighs the correctly-normalised KL (``w_kl / n``) by three orders
    of magnitude, which is ADR-027 (3)'s measured "sigma moves 4.8x across
    `w_brier` and ~10% across a 16x `w_kl` sweep" — *derived*, not observed.

    **THE FIX CHANGES THE ESTIMATOR, NOT THE ESTIMAND.** Subtracting the unbiased
    sample variance over ``S`` gives

        BS_fair = (p_hat_S - y)^2 - s_S^2 / S,   E[BS_fair] = (p_bar - y)^2

    exactly, for every ``S >= 2``. This is the standard finite-ensemble ("fair",
    Ferro 2014) correction of a proper score, and it is the SAME multi-step
    pushforward Brier CLAUDE.md mandates — estimated without the bias.

    **WHY THIS IS NOT TUNING ON THE GATE (ADR-027 (6) is the standard I am held
    to).** It has NO free parameter, so there is nothing to tune. It never
    references area, spread, dispersion or any G3 quantity: its expectation is a
    pure function of the marginal ``p_bar``. And it is a bias correction whose
    size is fixed by ``S`` alone — the same correction, with the same sign and
    the same magnitude, whatever the ensemble turns out to score.

    ``S = 1`` returns exactly zero: with one sample there is no variance estimate
    and the plug-in is all there is. A model with no latent never reaches here.
    """
    s = int(stacked.shape[0])
    if s < 2:
        return torch.zeros_like(stacked[0])
    return stacked.var(dim=0, unbiased=True) / float(s)


def _marginal_rollout(
    model: ContagionKernel,
    batch: dict[str, Tensor],
    fire: FireTensors,
    horizon: int,
    n_prior_samples: int,
    generator: Any = None,
) -> Tensor:
    """``E_z[b_k]`` over ``n_prior_samples`` prior draws, or the ``z=0`` path at 0.

    ``generator`` makes the estimate a DETERMINISTIC function of the parameters.
    That is not cosmetic: :func:`calibrate_alpha_to_growth` bisects on this
    quantity, and bisection on a Monte-Carlo objective converges to wherever the
    noise happened to fall rather than to the calibrated scale.
    """
    if model.latent is None or n_prior_samples <= 0:
        return model.rollout(batch["b0"], batch["weather"], fire.fields, horizon)
    total = None
    covariates = (
        [step_covariates(batch["weather"][:, k]) for k in range(horizon)]
        if model.latent.prior_net is not None
        else None
    )
    for _ in range(int(n_prior_samples)):
        draws = LatentSampler("latent").draw_path(
            model.latent,
            (batch["b0"].shape[0],),
            horizon,
            generator=generator,
            covariates=covariates,
        )
        got = model.rollout(batch["b0"], batch["weather"], fire.fields, horizon, draws)
        total = got if total is None else total + got
    return total / float(n_prior_samples)


@torch.no_grad()
def innovation_autocorrelation(
    model: ContagionKernel,
    fires: Sequence[FireTensors],
    *,
    max_windows_per_fire: int | None = None,
) -> dict[str, Any]:
    """[M6] Lag-1 autocorrelation of the ONE-STEP GROWTH INNOVATION. TRAIN ONLY.

    This is the measurement that SETS ``latent_rho``, and it is deliberately a
    measurement of the very quantity ``z_t[0]`` (``log_intensity``) is defined to
    carry: the log-ratio between the new cells the hour actually produced and the
    new cells the deterministic (``z = 0``) kernel expected,

        e_t = log1p(observed new cells) - log1p(predicted new cells)

    on the growth band, at a ONE-hour step. ``rho`` is then the lag-1
    autocorrelation of ``e_t`` over CONSECUTIVE hours, pooled across fires after
    removing each fire's own mean (a per-fire mean offset is a calibration error,
    not persistence, and leaving it in would inflate rho toward 1 mechanically).

    **This is a generative parameter estimated on TRAIN fires, exactly as
    ADR-011/C6.2 estimates the ellipse's scale on TRAIN fires.** It reads no
    held-out fire, no ensemble, and no gate metric — so it cannot be, and is not,
    tuned toward ``area_dispersion_ratio``. C-3 applies and is satisfied by
    construction: the estimate spans 8 fires / 7 spatial blocks, and the per-fire
    values are returned so the pooled number can be audited against its spread.
    """
    per_fire: list[dict[str, Any]] = []
    num = den_a = den_b = 0.0
    n_pairs = 0
    for f in fires:
        idx = f.t0_index
        if max_windows_per_fire is not None and idx.size > max_windows_per_fire:
            idx = idx[:: max(1, idx.size // int(max_windows_per_fire))]
        resid: dict[int, float] = {}
        for t0 in idx:
            t = int(t0)
            b0 = f.burned[t]
            band = f.band[t]
            if not bool(band.any()):
                continue
            p = model.step_probability(b0, f.weather[t + 1], f.fields, None)
            unburned = (b0 < 0.5) & band
            pred = float((p * unburned).sum())
            obs = float(((f.burned[t + 1] - b0) * unburned).sum())
            resid[t] = math.log1p(max(obs, 0.0)) - math.log1p(max(pred, 0.0))
        if len(resid) < 3:
            per_fire.append({"fire_id": f.fire_id, "n_pairs": 0, "rho": None})
            continue
        mean = sum(resid.values()) / len(resid)
        pairs = [(resid[t] - mean, resid[t + 1] - mean) for t in resid if (t + 1) in resid]
        if not pairs:
            per_fire.append({"fire_id": f.fire_id, "n_pairs": 0, "rho": None})
            continue
        fa = sum(a * b for a, b in pairs)
        fb = sum(a * a for a, _ in pairs)
        fc = sum(b * b for _, b in pairs)
        per_fire.append(
            {
                "fire_id": f.fire_id,
                "n_pairs": len(pairs),
                "rho": (fa / math.sqrt(fb * fc)) if fb > 0 and fc > 0 else None,
                "innovation_sd": float(np.std([resid[t] for t in resid])),
            }
        )
        num += fa
        den_a += fb
        den_b += fc
        n_pairs += len(pairs)
    rho = (num / math.sqrt(den_a * den_b)) if den_a > 0 and den_b > 0 else 0.0
    seen = [p["rho"] for p in per_fire if p["rho"] is not None]
    return {
        "rho_lag1_pooled": float(rho),
        "n_pairs": int(n_pairs),
        "n_fires": len(seen),
        "per_fire_rho_min": (min(seen) if seen else None),
        "per_fire_rho_max": (max(seen) if seen else None),
        "per_fire": per_fire,
        "estimand": (
            "lag-1 autocorrelation of log1p(observed new cells) - log1p(predicted new "
            "cells) at a 1 h step on the growth band, per-fire mean removed; TRAIN fires "
            "only. This is the autocorrelation of the innovation z_t[0] carries."
        ),
    }


@torch.no_grad()
def growth_diagnostics(
    model: ContagionKernel,
    fires: Sequence[FireTensors],
    horizons: Sequence[int] = (1, 3),
    *,
    max_windows_per_fire: int | None = None,
    seed: int = 0,
    n_prior_samples: int = 0,
) -> dict[str, Any]:
    """Predicted vs observed new cells at each horizon. The P1/P2/P3 measurement.

    Mean-field expectation, not a sample: the quantity pre-registered is
    ``E[new cells]``, and estimating it by Monte Carlo would add sampling noise
    to a number whose closed form is available.

    [M5] With a latent, ``n_prior_samples > 0`` estimates the MARGINAL
    ``E_z E[new cells]`` instead. That is the quantity the ensemble realises and
    it is NOT the ``z = 0`` value — the hazard is log-normal in ``z`` and ``p``
    is concave in the hazard — so a growth ratio quoted at ``z = 0`` for a latent
    model would describe a forecast nobody samples.
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(seed) + 8191)
    out: dict[str, Any] = {}
    for horizon in horizons:
        pred_total = obs_total = 0.0
        per_fire: dict[str, Any] = {}
        for fire in fires:
            usable = fire.t0_index[fire.t0_index + horizon < fire.burned.shape[0]]
            if max_windows_per_fire and usable.size > max_windows_per_fire:
                usable = np.sort(rng.choice(usable, max_windows_per_fire, replace=False))
            pred = obs = 0.0
            for start in range(0, usable.size, 16):
                t0s = usable[start : start + 16]
                batch = _batch(fire, t0s, horizon)
                b = _marginal_rollout(model, batch, fire, horizon, n_prior_samples, gen)
                unburned0 = batch["b0"] < 0.5
                pred += float(((b[:, -1] - batch["b0"]) * unburned0).sum())
                obs += float(((batch["truth"][:, -1] - batch["b0"]) * unburned0).sum())
            per_fire[fire.fire_id] = {
                "predicted_new_cells": pred,
                "observed_new_cells": obs,
                "ratio": (pred / obs) if obs > 0 else None,
                "n_windows": int(usable.size),
            }
            pred_total += pred
            obs_total += obs
        out[f"{horizon}h"] = {
            "predicted_new_cells": pred_total,
            "observed_new_cells": obs_total,
            "growth_ratio": (pred_total / obs_total) if obs_total > 0 else None,
            "per_fire": per_fire,
        }
    return out


@torch.no_grad()
def band_scores(
    model: ContagionKernel,
    fires: Sequence[FireTensors],
    horizon_h: int,
    *,
    n_prior_samples: int = 0,
) -> dict[str, Any]:
    """TRAIN band NLL / Brier, and the reach diagnostics that explain them."""
    nll_sum = brier_sum = n_cells = 0.0
    unreachable = truth_new = 0.0
    reach = int(model.config.radius) * int(horizon_h)
    for fire in fires:
        for start in range(0, fire.t0_index.size, 16):
            t0s = fire.t0_index[start : start + 16]
            batch = _batch(fire, t0s, horizon_h)
            b = _marginal_rollout(model, batch, fire, horizon_h, n_prior_samples)
            y = batch["truth"]
            mask = batch["band"].unsqueeze(1).expand_as(b)
            p = torch.clamp(b, _PROB_EPS, 1.0 - _PROB_EPS)
            nll_sum += float(
                ((-(y * torch.log(p) + (1.0 - y) * torch.log1p(-p))) * mask).sum()
            )
            brier_sum += float((((b - y) ** 2) * mask).sum())
            n_cells += float(mask.sum())
            # Cells the truth burns beyond anything the kernel can reach.
            for i in range(len(t0s)):
                b0 = batch["b0"][i].numpy() > 0.5
                new = (batch["truth"][i, -1].numpy() > 0.5) & ~b0
                truth_new += float(new.sum())
                unreachable += float((new & ~dilate(b0, reach)).sum())
    return {
        "band_nll": nll_sum / max(n_cells, 1.0),
        "band_brier": brier_sum / max(n_cells, 1.0),
        "band_cells": n_cells,
        "truth_new_cells": truth_new,
        "unreachable_new_cells": unreachable,
        "unreachable_fraction": unreachable / max(truth_new, 1.0),
        "reach_cells": reach,
        "note": (
            "unreachable_* counts truth cells outside the kernel's radius*horizon reach. "
            "They are unlearnable by the contagion component BY CONSTRUCTION and are "
            "charged the clamped NLL ceiling; this count is the SPOT BUDGET for P3."
        ),
    }


@torch.no_grad()
def wind_sector_report(
    model: ContagionKernel,
    fires: Sequence[FireTensors],
    horizon_h: int,
    *,
    n_prior_samples: int = 0,
    max_windows_per_fire: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """[M6] Predicted / observed new cells by sector RELATIVE TO THE WIND.

    sim's S3 measured, on held-out fires, that TRAINING INVERTS our wind
    anisotropy: the untrained kernel scores head/flank/rear 1.507 / 1.133 / 0.847
    (correctly anisotropic — more growth downwind than the truth had, less
    upwind), and the trained one scores 0.863 / 0.974 / 1.315 (inverted: it
    UNDER-predicts downwind and OVER-predicts upwind). That is backwards for a
    wind-driven fire.

    This is the same estimand computed on TRAIN, so the effect of an intervention
    can be seen before any held-out fire is read. A cell is assigned to a sector
    by the angle between (cell - fire centroid) and the domain-mean wind at that
    step: head |theta| < 60 deg, flank 60-120, rear > 120. The reported number is
    predicted-new / observed-new within the sector, so 1.0 is unbiased and the
    SHAPE of the three-number profile is the anisotropy claim.
    """
    rng = np.random.default_rng(int(seed))
    gen = torch.Generator().manual_seed(int(seed) + 104729)
    edges = (60.0, 120.0)
    pred = {"head": 0.0, "flank": 0.0, "rear": 0.0}
    obs = {"head": 0.0, "flank": 0.0, "rear": 0.0}
    for fire in fires:
        usable = fire.t0_index
        if max_windows_per_fire and usable.size > max_windows_per_fire:
            usable = np.sort(rng.choice(usable, max_windows_per_fire, replace=False))
        rows, cols = np.indices(fire.shape)
        for start in range(0, usable.size, 8):
            t0s = usable[start : start + 8]
            batch = _batch(fire, t0s, horizon_h)
            b = _marginal_rollout(model, batch, fire, horizon_h, n_prior_samples, gen)
            for i, t0 in enumerate(t0s):
                b0 = batch["b0"][i].numpy() > 0.5
                if not b0.any():
                    continue
                cy, cx = rows[b0].mean(), cols[b0].mean()
                w = fire.weather[int(t0) + 1].numpy()
                u = float(w[weather_index("wind_u10")].mean())
                v = float(w[weather_index("wind_v10")].mean())
                if u * u + v * v < 1e-9:
                    continue
                # C1.4: row 0 is NORTH and y DESCENDS, so a cell's northward
                # displacement is -(row - centre). Getting this backwards would
                # silently swap head and rear, which is precisely the quantity
                # under test.
                east, north = cols - cx, -(rows - cy)
                norm = np.sqrt(east**2 + north**2) + 1e-9
                cos_t = np.clip((east * u + north * v) / (norm * np.hypot(u, v)), -1, 1)
                theta = np.degrees(np.arccos(cos_t))
                pred_new = (b[i, -1].numpy() - batch["b0"][i].numpy()) * (~b0)
                obs_new = (batch["truth"][i, -1].numpy() - batch["b0"][i].numpy()) * (~b0)
                for name, sel in (
                    ("head", theta < edges[0]),
                    ("flank", (theta >= edges[0]) & (theta < edges[1])),
                    ("rear", theta >= edges[1]),
                ):
                    pred[name] += float(pred_new[sel].sum())
                    obs[name] += float(obs_new[sel].sum())
    ratios = {k: (pred[k] / obs[k]) if obs[k] > 0 else None for k in pred}
    head, rear = ratios["head"], ratios["rear"]
    return {
        "sector_ratio": ratios,
        "predicted_new_cells": pred,
        "observed_new_cells": obs,
        "head_over_rear": (head / rear) if (head and rear) else None,
        "anisotropy_verdict": (
            "UNKNOWN"
            if not (head and rear)
            else ("CORRECT (head > rear)" if head > rear else "INVERTED (rear > head)")
        ),
        "definition": (
            "predicted new cells / observed new cells within a sector defined by the angle "
            "to the DOMAIN-MEAN WIND. head |theta|<60 deg, flank 60-120, rear >120. "
            "TRAIN fires only. Same estimand as simviz S3's held-out measurement."
        ),
    }


@torch.no_grad()
def train_ensemble_diagnostics(
    model: ContagionKernel,
    fires: Sequence[FireTensors],
    horizon_h: int,
    *,
    n_members: int = 16,
    max_windows_per_fire: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    """[M5] The independent-noise ABLATION, run on TRAIN windows, as a POSITIVE CONTROL.

    G3 requires the independent-per-pixel ensemble to DEMONSTRATE collapse.
    That makes it a test of this repo's own ensemble machinery as much as of the
    model: if the ablation fails to collapse, the finding is about the
    instrument. Running it on TRAIN means the control is available before any
    held-out fire is read, so a broken sampler is caught before it contaminates
    a gate number rather than after.

    The two arms are the SAME parameters (``with_sampler`` shares them), the same
    windows and the same seed. The reported quantity is the ensemble's spread in
    TOTAL BURNED AREA relative to its own mean — the collapse
    ``area_dispersion_ratio`` detects, expressed without a truth term so it
    cannot be confused with a skill score.
    """
    rng = np.random.default_rng(int(seed))
    out: dict[str, Any] = {
        "n_members": n_members,
        "definition": (
            "area_cv = SD over members of NEW burned cells at the final lead (cells unburned "
            "at t0), divided by the member mean. NEW cells, not total: the already-burned "
            "region is identical in every member and every arm, so including it dilutes both "
            "spreads by the same large constant and makes a collapsed ensemble look merely "
            "narrow. It is also the quantity the growth_band mask scores. NO truth term: "
            "this is SPREAD ONLY and is never a skill score."
        ),
    }
    windows: list[tuple[Any, int]] = []
    for fire in fires:
        usable = fire.t0_index
        if usable.size > max_windows_per_fire:
            usable = np.sort(rng.choice(usable, max_windows_per_fire, replace=False))
        windows.extend((fire, int(t0)) for t0 in usable)

    for mode in ("latent", "independent"):
        view = model.with_sampler(mode) if model.latent is not None else model
        sum_var = sum_mean = 0.0
        cvs: list[float] = []
        for fire, w0 in windows:
            x0 = np.where(fire.burned[w0].numpy() > 0.5, 1, 0).astype(np.uint8)
            weather = fire.weather[w0 + 1 : w0 + 1 + horizon_h].numpy()
            samples = view.predict(
                x0, fire.static_array, weather, n_members, horizon_h, int(seed) + w0
            )
            areas = ((samples[:, -1] > 0) & (x0[None] == 0)).sum(axis=(1, 2)).astype(np.float64)
            mean = float(areas.mean())
            sum_var += float(areas.var(ddof=1))
            sum_mean += mean
            # A per-window CV is dominated by SMALL windows: at a mean of 4 new
            # cells even pure counting noise gives CV ~ 0.5, so an unrestricted
            # mean-of-CVs measures how often the fire was quiet, not how wide the
            # ensemble is. Restricted, and POOLED sums are carried alongside,
            # because the gate metric pools.
            if mean >= 10.0:
                cvs.append(float(areas.std(ddof=1)) / mean)
        out[mode] = {
            "pooled_member_area_variance": sum_var,
            "pooled_member_area_mean": sum_mean,
            "area_cv_mean_windows_ge10": float(np.mean(cvs)) if cvs else None,
            "n_windows": len(windows),
            "n_windows_ge10": len(cvs),
        }
        if model.latent is None:
            out["note"] = "no latent: both arms are the SAME independent-per-pixel sampler"
            out["latent"] = out["independent"] = out[mode]
            break
    lat, ind = out.get("latent", {}), out.get("independent", {})
    if lat.get("pooled_member_area_variance") and ind.get("pooled_member_area_variance"):
        out["variance_ratio_latent_over_independent"] = (
            lat["pooled_member_area_variance"] / ind["pooled_member_area_variance"]
        )
        if lat.get("area_cv_mean_windows_ge10") and ind.get("area_cv_mean_windows_ge10"):
            out["cv_ratio_latent_over_independent"] = (
                lat["area_cv_mean_windows_ge10"] / ind["area_cv_mean_windows_ge10"]
            )
    return out


def calibrate_alpha_to_growth(
    model: ContagionKernel,
    fires: Sequence[FireTensors],
    *,
    horizon_h: int = 1,
    tolerance: float = 0.02,
    max_iterations: int = 30,
    n_prior_samples: int = 0,
) -> dict[str, Any]:
    """Set ``log_alpha`` so the UNTRAINED kernel reproduces observed train growth.

    Same rule as the ellipse baseline's C6.2 calibration, applied to the
    kernel's initialisation. The point is that training starts from a physics
    model that is already right about the *rate*, so anything the optimiser
    subsequently buys is about the *shape* — which is what the anisotropic
    kernel is a claim about.
    """
    lo, hi = -12.0, 12.0

    def ratio_at(value: float) -> float:
        with torch.no_grad():
            model.log_alpha.fill_(value)
        d = growth_diagnostics(model, fires, (horizon_h,), n_prior_samples=n_prior_samples)[
            f"{horizon_h}h"
        ]
        return float("inf") if d["growth_ratio"] is None else float(d["growth_ratio"])

    history = []
    chosen = 0.0
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        r = ratio_at(mid)
        history.append({"log_alpha": mid, "growth_ratio": r})
        chosen = mid
        if r > 0 and abs(math.log(r)) <= tolerance:
            break
        if r < 1.0:
            lo = mid
        else:
            hi = mid
    with torch.no_grad():
        model.log_alpha.fill_(chosen)
    return {
        "log_alpha": chosen,
        "alpha": math.exp(chosen),
        "horizon_h": horizon_h,
        "final_growth_ratio": history[-1]["growth_ratio"] if history else None,
        "n_iterations": len(history),
        "rule": "same growth-matching rule as the C6.2 ellipse calibration (ADR-011)",
    }


def train_kernel(
    config: TrainConfig | None = None,
    *,
    train_fire_ids: Sequence[str] | None = None,
    write_run: bool = True,
    run_prefix: str = "kernel",
    log_every: int = 25,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit the kernel on TRAIN fires. Returns the run payload; writes ``runs/``."""
    cfg = config or TrainConfig()
    check_torch_matches_numpy()  # C0: refuse to train a drifted physics path
    torch.manual_seed(cfg.seed)

    from wildfire_nowcast.eval.reporting import (
        assert_split_unchanged,
        check_common_code_unchanged,
        common_code_fingerprint,
        scoring_code_fingerprint,
        split_fingerprint,
    )

    split = split_fingerprint()
    code = common_code_fingerprint()
    # [v2.11 / C-4.2] BOTH ENDS, on the training side too. A checkpoint whose
    # scoring/model code moved mid-fit is as unattributable as a score whose did.
    scoring_before = scoring_code_fingerprint()
    stats = read_norm_stats(norm_stats_path())
    train_folds = {int(f) for f in stats["train_folds"]}
    from wildfire_nowcast.eval.baseline_run import load_splits

    splits = load_splits(sorted(train_folds))
    train = [s for s in splits if s.is_train]
    heldout = [s for s in splits if not s.is_train]
    if train_fire_ids is not None:
        wanted = set(train_fire_ids)
        stray = wanted & {s.fire_id for s in heldout}
        if stray:
            raise RuntimeError(f"refusing to train on held-out fires {sorted(stray)}")
        train = [s for s in train if s.fire_id in wanted]
    if not train:
        raise RuntimeError("no train fires")

    t_load = time.time()
    fires = [
        load_fire(
            s.fire_id,
            cfg.horizon_h,
            band_radius_cells=cfg.band_radius_cells,
            mirror_ns=cfg.mirror_ns,
        )
        for s in train
    ]
    load_s = time.time() - t_load

    # [M10] One constructor call, one config, two arms. The direct head differs
    # from the rollout in its CLASS and in nothing else that this function does —
    # same losses, same optimiser, same seed, same calibration, same diagnostics —
    # so a difference in the result is a difference in the horizon treatment and
    # not in the training recipe.
    kernel_cls = DirectHorizonKernel if cfg.direct_horizon else ContagionKernel
    model = kernel_cls(
        KernelConfig(
            radius=cfg.radius,
            susceptibility_mode=cfg.susceptibility_mode,
            # [S1] The ONLY difference between arm A and arm S. Same class, same
            # losses, same optimiser, same seed, same calibration — so a
            # difference in the result is a difference in the COVARIATE.
            stage_scalar=cfg.stage_scalar,
        ),
        name="direct_horizon_kernel" if cfg.direct_horizon else "contagion_kernel",
        ellipse_params=EllipseParams().scaled(cfg.ellipse_scale),
        latent_config=(
            None
            if cfg.latent_dim <= 0
            else LatentConfig(
                dim=int(cfg.latent_dim),
                free_bits=float(cfg.latent_free_bits),
                gate_prior_mean=float(cfg.gate_prior_mean),
                conditional_prior=bool(cfg.conditional_prior),
                mean_preserving=bool(cfg.mean_preserving_latent),
                gate_mean_preserving=bool(cfg.gate_mean_preserving_latent),
                rho=float(cfg.latent_rho),
                spatial_modes=int(cfg.spatial_modes),
                innovation_channels=bool(cfg.innovation_encoder),
                spatial_encoder_pooling=bool(cfg.spatial_encoder_pooling),
            )
        ),
    )
    marginal_samples = cfg.n_prior_samples if cfg.latent_dim > 0 else 0
    # C8 (INTERFACES v2.8). Stamped BEFORE training, so a checkpoint that outlives
    # this session carries the split it was fitted on and an evaluator can hard
    # fail on a mismatch instead of silently scoring a trained-on fire.
    model.provenance = {
        "split_fingerprint": split["fingerprint"],
        "train_fire_ids": [s.fire_id for s in train],
        "train_folds": sorted(train_folds),
        "n_heldout_blocks": split["n_heldout_blocks"],
        "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg.to_dict(),
    }
    init_calibration = (
        calibrate_alpha_to_growth(model, fires, n_prior_samples=marginal_samples)
        if cfg.calibrate_alpha
        else {"skipped": True}
    )
    init_growth = growth_diagnostics(
        model, fires, (1, cfg.horizon_h), n_prior_samples=marginal_samples
    )
    init_scores = band_scores(model, fires, cfg.horizon_h, n_prior_samples=marginal_samples)

    # -- window pool, with the R14 ablation switches applied ---------------
    pools: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for fire in fires:
        t0s = fire.t0_index
        grew = np.array(
            [
                float(
                    (
                        (fire.burned[int(t) + cfg.horizon_h] - fire.burned[int(t)])
                        * (fire.burned[int(t)] < 0.5)
                    ).sum()
                )
                > 0
                for t in t0s
            ]
        )
        if cfg.growth_windows_only:
            t0s = t0s[grew > 0]
            grew = grew[grew > 0]
        w = np.where(grew > 0, float(cfg.growth_window_weight), 1.0)
        pools.append(t0s)
        weights.append(w / max(w.sum(), 1e-9))
    if any(p.size == 0 for p in pools):
        raise RuntimeError("a train fire has no usable window under this config")

    # -- the observation-noise model, one calibration per TRAIN fire ---------
    noise_models: dict[int, FireNoiseModel] = {
        i: noise_model_for(fire.fire_id, scale=cfg.label_perturbation_scale)
        for i, fire in enumerate(fires)
    }

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, float]] = []
    perturbations_drawn: list[dict[str, Any]] = []
    # One EMA PER FIRE: fires differ by an order of magnitude in growth rate, so
    # a single EMA mixed across fire-uniform batches targets a rate no fire has.
    growth_emas: dict[int, dict[str, float]] = {i: {} for i in range(len(fires))}
    polyak: dict[str, torch.Tensor] = {}
    polyak_count = 0
    tail_start = int(cfg.steps * (1.0 - float(cfg.polyak_tail)))
    t_train = time.time()
    for step in range(int(cfg.steps)):
        f = int(rng.integers(len(fires))) if cfg.fire_uniform_sampling else int(
            rng.choice(len(fires), p=np.array([p.size for p in pools], dtype=float) / sum(
                p.size for p in pools
            ))
        )
        pool, prob = pools[f], weights[f]
        take = min(cfg.batch_size, pool.size)
        t0s = rng.choice(pool, size=take, replace=False, p=prob if take < pool.size else None)
        draw = sample_perturbation(noise_models[f], rng) if cfg.label_perturbation else None
        batch = _batch(
            fires[f],
            np.sort(t0s),
            cfg.horizon_h,
            perturbation=draw,
            band_radius_cells=cfg.band_radius_cells,
        )
        if cfg.label_perturbation:
            perturbations_drawn.append(
                {"step": step, "fire": fires[f].fire_id, **batch["_perturbation"].to_dict()}
            )
        if cfg.lr_cosine_decay and cfg.steps > 1:
            frac = step / (cfg.steps - 1)
            for group in optimiser.param_groups:
                group["lr"] = cfg.learning_rate * (
                    0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac))
                )
        terms = _losses(model, fires[f], batch, cfg.horizon_h, cfg, ema=growth_emas[f])
        optimiser.zero_grad(set_to_none=True)
        terms["loss"].backward()
        grad_norm = float(
            torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if cfg.polyak_tail > 0 and step >= tail_start:
            polyak_count += 1
            with torch.no_grad():
                for name, param in model.named_parameters():
                    polyak[name] = (
                        param.detach().clone()
                        if name not in polyak
                        else polyak[name] + (param.detach() - polyak[name]) / polyak_count
                    )
        row = {
            "step": step,
            "fire": fires[f].fire_id,
            "loss": float(terms["loss"].detach()),
            "nll": float(terms["nll"]),
            "brier": float(terms["brier"]),
            "growth": float(terms["growth"]),
            "kl": float(terms["kl"]),
            "grad_norm": grad_norm,
        }
        if "kl_per_dim" in terms:
            row["kl_per_dim"] = [float(v) for v in terms["kl_per_dim"]]
            row["sigma"] = [float(v) for v in model.latent.sigma().detach()]
        history.append(row)
        if verbose and (step % log_every == 0 or step == cfg.steps - 1):
            print(
                f"  step {step:>4}  {row['fire']:<28} loss {row['loss']:.5f}  "
                f"nll {row['nll']:.5f}  brier {row['brier']:.6f}  "
                f"growth {row['growth']:.4f}  kl {row['kl']:.4f}  |g| {grad_norm:.3g}"
            )
    if polyak_count > 1:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(polyak[name])
    train_s = time.time() - t_train

    final_growth = growth_diagnostics(
        model, fires, (1, cfg.horizon_h), n_prior_samples=marginal_samples
    )
    final_scores = band_scores(model, fires, cfg.horizon_h, n_prior_samples=marginal_samples)
    collapse = train_ensemble_diagnostics(model, fires, cfg.horizon_h, seed=cfg.seed)
    sectors = wind_sector_report(
        model, fires, cfg.horizon_h, n_prior_samples=marginal_samples,
        max_windows_per_fire=48, seed=cfg.seed,
    )
    init_sectors = wind_sector_report(
        ContagionKernel(
            KernelConfig(radius=cfg.radius, susceptibility_mode=cfg.susceptibility_mode),
            ellipse_params=EllipseParams().scaled(cfg.ellipse_scale),
        ),
        fires, cfg.horizon_h, max_windows_per_fire=48, seed=cfg.seed,
    )

    scoring_after = scoring_code_fingerprint()

    payload: dict[str, Any] = {
        "kind": "contagion_kernel_training",
        "config": cfg.to_dict(),
        "scope": {
            "train_fire_ids": [f.fire_id for f in fires],
            "train_blocks": sorted({s.spatial_block_id for s in train}),
            "heldout_fire_ids": [s.fire_id for s in heldout],
            "heldout_blocks": sorted({s.spatial_block_id for s in heldout}),
            "n_heldout_blocks": len({s.spatial_block_id for s in heldout}),
            "norm_stats_n_train_blocks": stats.get("n_train_blocks"),
            "split_fingerprint": split["fingerprint"],
            "c6_3_satisfied": split["c6_3_satisfied"],
            "warning": (
                "TRAIN diagnostics only. No held-out fire was read during training and no "
                "hyperparameter was selected on one; the step budget is fixed a priori."
            ),
        },
        "split_before": split,
        "split_after": assert_split_unchanged(split),
        "common_code_before": code,
        "common_code_after": check_common_code_unchanged(code),
        "scoring_code_before": scoring_before,
        "scoring_code_after": scoring_after,
        "code_fingerprints_agree": {
            "scoring_code": scoring_before["fingerprint"] == scoring_after["fingerprint"],
            "verdict": (
                "OK — one version of the model/eval code produced this fit"
                if scoring_before["fingerprint"] == scoring_after["fingerprint"]
                else "C-4.2 HARD FAIL — code moved during this training run"
            ),
        },
        "init_calibration": init_calibration,
        "train_diagnostics": {
            "init": {"growth": init_growth, "scores": init_scores},
            "final": {"growth": final_growth, "scores": final_scores},
        },
        # [M5] The latent, and whether it survived training. `sigma` is the whole
        # model of the ensemble; a sigma driven to ~0 is posterior collapse and
        # reproduces the ablation while wearing the model's name, so the number
        # that says so travels in the artifact rather than being inferred from a
        # disappointing dispersion score downstream.
        "latent": latent_report(model.latent),
        # [M5] The G3 POSITIVE CONTROL, measured on TRAIN windows only, so it is
        # available before any held-out fire is read. Same fit, same windows,
        # same seed; the ONLY difference is whether z_t is drawn.
        "train_ensemble_collapse_check": collapse,
        # [M6] The pre-registered anisotropy test (see insights item 49). The
        # UNTRAINED control is measured in the SAME call on the SAME windows, so
        # "training inverts the anisotropy" is a within-run comparison rather
        # than a comparison against a number from another session.
        "wind_sectors": {"final": sectors, "untrained_control": init_sectors},
        "latent_bit_identity": check_latent_off_is_bit_identical(),
        # [M10 / ADR-045 (3)] "B is not allowed to win on capacity." Every run
        # carries its own parameter census so the comparison is a number in the
        # artifact rather than an argument in a status entry.
        "parameter_counts": parameter_counts(model),
        "parameters": parameter_report(model),
        "offset_kernel": offset_kernel_table(model),
        # ADR-015 (6a): the fix is only a fix if the gradient is measurably
        # non-zero, so the measurement travels in every run artifact next to the
        # M2 form it replaces.
        "susceptibility_gradient": {
            mode: susceptibility_gradient_report(mode) for mode in ("amplitude", "reach")
        },
        # ADR-015 (6b): the wind-independent directional preference, as ONE
        # number, before and after training.
        "offset_anisotropy": {
            "final": offset_anisotropy(model),
            "note": (
                "At init every c_d is 0, so magnitude is exactly 0 and the bearing is "
                "undefined. Any non-zero value here was learned from the fires — or from "
                "the labels' own displacement, which is what the perturbation tests."
            ),
        },
        "label_perturbation": (
            {
                "enabled": True,
                **perturbation_report(
                    [f.fire_id for f in fires], scale=cfg.label_perturbation_scale
                ),
                "n_draws_used": len(perturbations_drawn),
                "fraction_of_steps_identity": (
                    float(
                        np.mean(
                            [
                                d["shift_r"] == 0 and d["shift_c"] == 0 and d["morph"] == 0
                                for d in perturbations_drawn
                            ]
                        )
                    )
                    if perturbations_drawn
                    else None
                ),
            }
            if cfg.label_perturbation
            else {"enabled": False}
        ),
        "history": history,
        "timing_s": {"load": round(load_s, 1), "train": round(train_s, 1)},
    }

    if write_run:
        run = create_run_dir(
            {"experiment": "contagion_kernel", **cfg.to_dict()}, prefix=run_prefix
        )
        model.save(run.path)
        (run.path / "training.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
        payload["run_dir"] = str(run.path)
        payload["model_path"] = str(run.path)
    payload["model"] = model
    return payload


#: The M3 matrix, pre-registered in docs/decisions.md BEFORE it was
#: run. It is a 2x2 over ADR-015's two interventions plus two controls, because
#: (6a) and (6b) are different claims and a single "fixed" run could not say
#: which one moved anything.
M3_MATRIX: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "m2_reach",
        {"susceptibility_mode": "reach", "w_growth": 0.0, "label_perturbation": False},
        "CONTROL: M2's selected config, reproduced exactly. Zero barrier gradient.",
    ),
    (
        "amplitude",
        {"susceptibility_mode": "amplitude", "w_growth": 0.0, "label_perturbation": False},
        "ADR-015 (6a) ONLY: susceptibility moved to the hazard amplitude. Tests P11/P12.",
    ),
    (
        "amplitude_perturbed",
        {
            "susceptibility_mode": "amplitude",
            "w_growth": 0.0,
            "label_perturbation": True,
            "label_perturbation_scale": 1.0,
        },
        "THE PRE-REGISTERED G2 CANDIDATE: (6a) + (6b) at the measured noise scale.",
    ),
    (
        "amplitude_perturbed_2x",
        {
            "susceptibility_mode": "amplitude",
            "w_growth": 0.0,
            "label_perturbation": True,
            "label_perturbation_scale": 2.0,
        },
        "DOSE-RESPONSE for P15: twice the measured noise. Not a candidate.",
    ),
    (
        "reach_perturbed",
        {"susceptibility_mode": "reach", "w_growth": 0.0, "label_perturbation": True},
        "SEPARATION: (6b) without (6a), so the two interventions are not confounded.",
    ),
    (
        "init_only",
        {"susceptibility_mode": "amplitude", "steps": 0},
        "CONTROL: the growth-calibrated elliptical-Gaussian, untrained.",
    ),
)

#: Declared BEFORE the M3 matrix ran.
#:
#: **No config is selected by score, and none is selected on held-out data.**
#: ADR-015 fixes both interventions a priori, so the G2 candidate is named rather
#: than fitted: ``amplitude_perturbed`` is (6a) + (6b) at their specified
#: settings. ``amplitude`` is scored alongside it as a SECOND declared entry
#: because (6a) and (6b) answer different questions and ADR-015 asks for both.
#: The other four are controls and are never promoted.
#:
#: If a declared entry's TRAIN 3 h growth ratio falls outside [0.8, 1.25] it is
#: reported as failing its own sanity condition. It is NOT swapped for a config
#: that passes — swapping is how a selection rule becomes a search.
M3_SELECTION_RULE = (
    "PRE-DECLARED, NOT FITTED. G2 candidate = 'amplitude_perturbed' (ADR-015 6a + 6b at "
    "their specified settings). Second declared entry = 'amplitude' (6a alone). Neither is "
    "chosen by score and no held-out fire is read until both are frozen. A declared entry "
    "whose TRAIN 3 h growth ratio leaves [0.8, 1.25] is REPORTED AS FAILING, never replaced."
)

M3_DECLARED_ENTRIES: tuple[str, ...] = ("amplitude_perturbed", "amplitude")

#: The M5 matrix (P2: z_t, ensembles, the independent-noise ablation),
#: pre-registered in docs/decisions.md BEFORE it was run.
#:
#: **ONE intervention separates the candidate from the G2 kernel: the latent.**
#: `zt` and `nozt` differ ONLY in `latent_dim`, so "the ensemble got wider" cannot
#: be confounded with "the kernel got better". Everything else — amplitude
#: susceptibility, the NB moisture floor, `w_growth = 0`, 300 steps — is M4's
#: configuration held fixed.
#:
#: **The growth-moment ladder that would normally appear here is DELIBERATELY
#: ABSENT, and the reason is a measurement.** ADR-021 (3b) names our 2.66-3.06x
#: held-out growth over-prediction as the thing G3's dispersion bar collides
#: with, and the obvious response is to raise `w_growth`. But the four M4
#: checkpoints score a TRAIN growth ratio of **1.00-1.23** — they are already
#: calibrated where they were fitted. The over-prediction is therefore a
#: TRAIN -> HELD-OUT TRANSFER GAP, and a loss term that pins the train moment
#: cannot close it. Adding the ladder would have produced four runs, a
#: selection, and no effect on the number it was chosen for.
#:
#: The two `w_kl` arms are a DECLARED EXPLORATORY SENSITIVITY, not candidates:
#: `w_kl` moves the fitted `sigma`, `sigma` is the ensemble width, and G3 is
#: adjudicated on dispersion. Reporting how sigma responds to the KL weight is
#: characterisation; promoting the arm that widened the ensemble would be tuning
#: on the gate, and the matrix says so in its own entries.
M5_MATRIX: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "zt",
        {"latent_dim": 3, "w_kl": 1.0, "n_prior_samples": 4, "w_growth": 0.0},
        "CANDIDATE: M4's kernel + the shared per-step latent z_t. The ONLY change.",
    ),
    (
        "nozt",
        {"latent_dim": 0, "w_growth": 0.0},
        "CONTROL: M4's kernel reproduced under M5 code. Isolates the latent; also "
        "the separately-trained form of the independent-noise ablation.",
    ),
    (
        "zt_wkl_025",
        {"latent_dim": 3, "w_kl": 0.25, "n_prior_samples": 4, "w_growth": 0.0},
        "DECLARED EXPLORATORY (not a candidate): sigma's response to a weaker KL. "
        "Characterisation of the spread knob, never a route to the dispersion bar.",
    ),
    (
        "zt_wkl_4",
        {"latent_dim": 3, "w_kl": 4.0, "n_prior_samples": 4, "w_growth": 0.0},
        "DECLARED EXPLORATORY (not a candidate): the other side of the same knob.",
    ),
    (
        "zt_areacrps",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "w_area_crps": 0.02,
        },
        "DECLARED EXPLORATORY: CLAUDE.md's 'Brier/CRPS' read literally as a proper "
        "score of the ensemble's AREA. Adjacent to the gate metric, so it is "
        "declared, reported, and cannot be promoted without that being said.",
    ),
)

#: Declared BEFORE the M5 matrix ran.
#:
#: **No config is selected by score and no held-out fire is read until the
#: candidate is frozen.** The candidate is NAMED (`zt`), not fitted: P2's
#: question is "does the shared latent produce correlated innovations and a
#: dispersed ensemble", and that question has one intervention. `nozt` is the
#: control. The three `zt_*` arms are EXPLORATORY and may not be promoted to the
#: candidate; if one of them is quoted anywhere, the sentence quoting it says it
#: was exploratory.
#:
#: SANITY CONDITION, on TRAIN only, checked before any held-out scoring:
#:   (a) TRAIN 3 h growth ratio in [0.8, 1.25]  — the M3 band, unchanged;
#:   (b) the fitted `sigma` is not collapsed (> 0.01 on at least one dimension);
#:   (c) the TRAIN collapse control shows the latent arm with MORE pooled member
#:       area variance than its own independent-noise ablation.
#: A declared entry failing any of these is REPORTED AS FAILING, never replaced —
#: (c) especially, because a failure there is evidence about the ensemble
#: machinery and suppressing it would remove the only positive control we have.
M5_SELECTION_RULE = (
    "PRE-DECLARED, NOT FITTED. Candidate = 'zt' (M4's kernel + z_t, one intervention). "
    "Control = 'nozt'. 'zt_wkl_025', 'zt_wkl_4' and 'zt_areacrps' are DECLARED EXPLORATORY "
    "and may not be promoted. Sanity, TRAIN only: 3 h growth ratio in [0.8, 1.25]; sigma "
    "not collapsed; the latent arm carries more pooled member area variance than its own "
    "independent-noise ablation. Failing an entry is reported, never replaced."
)

M5_DECLARED_ENTRIES: tuple[str, ...] = ("zt", "nozt")

#: [M6] PRE-REGISTERED BEFORE THE ACTIVITY GATE WAS TRAINED EVEN ONCE.
#: Full text and rationale: docs/decisions.md items 49-50. Kept in
#: source as well as in insights because a pre-registration that lives only in a
#: prose file is a recollection.
M6_ACTIVITY_PREREGISTRATION = (
    "ONE-DEFECT HYPOTHESIS (maintainer): the missing OFF state and the inverted wind "
    "anisotropy are ONE defect — a model with no zero mass must place probability somewhere "
    "every hour, so dormant hours train it to spread thinly and isotropically into label "
    "noise. PREDICTION: adding the activity gate PARTIALLY RESTORES head/rear anisotropy "
    "with NO change to the kernel's spatial structure. "
    "P21 CONFIRMED if wind_sector_report head_over_rear moves >=20% of the way back to the "
    "UNTRAINED control; FALSIFIED if it moves <5% or moves away — and a falsification is a "
    "REAL RESULT (the inversion would then be an independent kernel defect). "
    "P22 (checked FIRST, else P21 is uninterpretable): dormant_off_rate rises from 0.000 to "
    ">0.05. "
    "P23: area_dispersion_ratio rises above the M5 candidate's 0.2257 — a CONSEQUENCE of a "
    "mixture prior, NOT evidence of capability, and it may not be quoted as progress on G3. "
    "PREMISE CORRECTION carried rather than the loose version: dormant vs growth windows are "
    "953:446 (~2:1), so if the mechanism operates it is through the PER-CELL loss, not the "
    "window counts — and it is per-cell: the growth-band mask makes the negative:positive "
    "cell ratio of order 100:1."
)

#: [M6B] The AR(1) persistence used by the candidate. **MEASURED, NOT CHOSEN.**
#: :func:`innovation_autocorrelation` on the 8 TRAIN fires, 2,350 consecutive
#: hour-pairs, against the M5 no-latent checkpoint: pooled lag-1 autocorrelation
#: of the one-step growth innovation = **0.5144**, per-fire 0.2171-0.6224, and
#: positive on 8 of 8. Recorded at `runs/m6_innovation_autocorr.json`.
#: C-3 is satisfied by construction — 8 fires across 7 spatial blocks — and no
#: held-out fire and no gate metric enters the estimate.
M6B_RHO = 0.5144

M6_MATRIX: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "gate",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": True,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
        },
        "M6 CANDIDATE: M5's z_t + the ACTIVITY GATE and a weather-conditional prior. "
        "The kernel's spatial structure is UNCHANGED, which is what makes P21 clean.",
    ),
    (
        "gate_uncond",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": False,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
        },
        "CONTROL: the gate WITHOUT the conditional prior. Separates 'a wider, "
        "skewed prior' from 'a prior that knows WHICH hour is dormant'.",
    ),
)

#: [M6B] PRE-REGISTERED BEFORE THE FIRST LOSS-FIX RUN. Full text: insights 52-54.
#:
#: **THE FIX IS DERIVED, NOT SEARCHED, AND THE MECHANISM IS ARITHMETIC.** ADR-027
#: (3) measured that the multi-step pushforward Brier — not the ELBO — sets the
#: ensemble width. Reading the loss rather than sweeping it says why, in one line:
#: the term scores the predictive MARGINAL but can only see an ``S = 4`` sample
#: estimate of it, and ``E[(p_hat_S - y)^2] = (p_bar - y)^2 + Var_z(b)/S``. **The
#: plug-in estimator ADDS THE ENSEMBLE VARIANCE TO THE LOSS.** See
#: :func:`_fair_brier_correction`.
#:
#: Three corrections, each of which removes a CONFOUND BETWEEN SPREAD AND MEAN
#: rather than adding a knob. None references area, dispersion or any G3
#: quantity; none has a free parameter fitted against a score:
#:  1. ``fair_brier`` — score the mandated Brier with the finite-ensemble
#:     unbiased estimator. Changes the ESTIMATOR, never the estimand.
#:  2. ``mean_preserving_latent`` — ``E_z[e^effect] = 1`` exactly, so ``sigma`` is
#:     a pure spread parameter and does not also inflate the ensemble mean.
#:  3. ``latent_rho`` — AR(1) persistence of ``z_t``, at the value MEASURED on
#:     TRAIN fires by :func:`innovation_autocorrelation` (0.5144 pooled over
#:     2,350 hour-pairs, 8 fires, per-fire 0.217-0.622, all 8 positive).
#:
#: The candidate is the COMPOSITION and is NAMED IN ADVANCE. The three
#: single-mechanism arms exist so the result is ATTRIBUTABLE, not so the best one
#: can be promoted — none may be.
M6B_MATRIX: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "m6_fair",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
        },
        "THE CANDIDATE, named before the first run: all three derived corrections.",
    ),
    (
        "m6_fairbrier",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
        },
        "ATTRIBUTION: correction 1 alone. Tests P24/P25 — is the finite-ensemble "
        "bias the dominant sharpening channel?",
    ),
    (
        "m6_meanpres",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "mean_preserving_latent": True,
        },
        "ATTRIBUTION: correction 2 alone. Isolates the log-normal mean inflation.",
    ),
    (
        "m6_ar1",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "latent_rho": M6B_RHO,
        },
        "ATTRIBUTION: correction 3 alone, at the TRAIN-measured rho. Its predicted "
        "3 h spread gain is a CLOSED FORM (1.364x) written down before the run.",
    ),
    (
        "m5_zt_repro",
        {"latent_dim": 3, "w_kl": 1.0, "n_prior_samples": 4, "w_growth": 0.0},
        "CONTAMINATION CONTROL (P28): M5's candidate config under M6 code. It must "
        "reproduce ADR-027's 0.2147, or the rho=0 path is not bitwise M5 and every "
        "comparison in this milestone is contaminated.",
    ),
    (
        "m6_fair_brier0",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "w_brier": 0.0,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
        },
        "DECLARED EXPLORATORY, NEVER A CANDIDATE: w_brier = 0 departs from the loss "
        "CLAUDE.md mandates. It exists only to measure P25 — how much of the "
        "sigma-vs-w_brier sensitivity the fair estimator removed.",
    ),
)

#: Declared BEFORE the M6B matrix ran.
M6B_SELECTION_RULE = (
    "PRE-DECLARED, NOT FITTED. Candidate = 'm6_fair' (all three derived corrections), named "
    "before the first run and NOT chosen by score. 'm6_fairbrier' / 'm6_meanpres' / 'm6_ar1' "
    "are ATTRIBUTION arms and may not be promoted, even if one scores better than the "
    "candidate — promoting a single-mechanism arm on its score is exactly the search this "
    "task was told not to run. 'm5_zt_repro' is the contamination control. 'm6_fair_brier0' "
    "is DECLARED EXPLORATORY (w_brier = 0 is not the mandated loss). Sanity, TRAIN only: "
    "3 h growth ratio in [0.8, 1.25]; sigma not collapsed. Failing is reported, never replaced."
)

M6B_DECLARED_ENTRIES: tuple[str, ...] = ("m6_fair", "m5_zt_repro")

#: [M7] The SPATIAL LEVER, MEASURED ON TRAIN FIRES BEFORE THE FIRST M7 ARM RAN.
#: ``runs/m7_spatial_lever.json``, 2,358 windows, 8 fires, 7 spatial blocks, the
#: `m6_fair_s1` checkpoint's own one-step hazard restricted to the growth band.
#:
#: ``lever_m = <phi_m>_lambda`` is exactly ``d log(area) / d c_m``, so a spatial
#: mode with scale ``sigma_m`` contributes ``|lever_m| * sigma_m`` to the SD of
#: log burned area against the GLOBAL mode's ``sigma_0 * 1``. Pooled RMS:
#:
#:   ``intensity_grad_east``   **0.3013**   (mean +0.086)
#:   ``intensity_grad_north``  **0.2953**   (mean -0.122)
#:   ``intensity_radial``      **1.1451**   (mean **+1.0034**)
#:
#: **THE RADIAL MODE'S MEAN LEVER IS 1.00, WHICH MAKES IT THE GLOBAL MODE IN A
#: SPATIAL COSTUME** — growth happens at the frontier, where ``phi_radial ~ +1``
#: by construction, so for AREA it is nearly a relabelled ``log_intensity``. It
#: is therefore DECLARED EXPLORATORY here, before any number exists, and may
#: never be promoted: passing G3 through it would be the uniform blur ADR-032 (3)
#: rejects, wearing a new name.
#: The two GRADIENT modes have mean lever ~0 and RMS ~0.30 — genuinely
#: window-specific, which is what "the ensemble disagrees about WHERE" means.
#: Geographic east/north is chosen over a WIND-FRAME basis for exactly this
#: reason: an along-wind mode would sit on the head every hour and inherit the
#: radial mode's mean-1 lever.
M7_SPATIAL_LEVER_RMS: tuple[float, float, float] = (0.3013, 0.2953, 1.1451)

#: [M7] PRE-REGISTERED BEFORE THE FIRST M7 TRAINING RUN. Full text: insights 59-63.
#:
#: TWO INDEPENDENT PROBLEMS, TWO CANDIDATES, NEITHER COMPOSED WITH THE OTHER —
#: ADR-029 (7)'s one-variable-at-a-time rule applies inside a milestone too.
#:  A. DISPERSION (ADR-032 (3)): a global scalar latent can only widen by
#:     blurring. ``m7_spatial`` gives ``z_t`` two SPATIAL degrees of freedom.
#:  B. THE OFF STATE (ADR-032 (5)): measured one level below the brief's
#:     diagnosis. It is not that ``p(z_t|weather)`` cannot be fitted — it is that
#:     ``q`` never saw dormancy to hand it. ``m7_offstate`` fixes the encoder.
#:
#: `m6_fair` is the CONTROL and is NOT retrained: its four recorded seeds enter
#: the scoring run unchanged, so the control cannot drift under the candidate.
M7_MATRIX: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "m7_spatial",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "spatial_modes": 2,
            "innovation_encoder": True,
        },
        "CANDIDATE A (dispersion), named before the first run: m6_fair + the two "
        "LOW-RANK SPATIAL GRADIENT modes. Rank 2, not H*W — a correlated "
        "innovation, not the known-broken per-pixel noise field.",
    ),
    (
        "m7_offstate",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": True,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "innovation_encoder": True,
        },
        "CANDIDATE B (OFF state): the M6 activity gate + weather-conditional "
        "prior, with the INNOVATION ENCODER fix. One variable against "
        "`m7_gate_nofix`.",
    ),
    (
        "m7_gate_nofix",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": True,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
        },
        "CONTROL for B: the same gate WITHOUT the encoder fix. Isolates the one "
        "change; without it a gate that starts working proves nothing about why.",
    ),
    (
        "m7_spatial_blind",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "spatial_modes": 2,
            "innovation_encoder": True,
            "spatial_encoder_pooling": False,
        },
        "NEGATIVE CONTROL for A: the spatial modes exist but `q` is denied the "
        "basis-projected pooling that identifies them. If this matches the "
        "candidate, the modes are not being inferred and the gain came from "
        "somewhere else.",
    ),
    (
        "m7_spatial_r3",
        {
            "latent_dim": 3,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "spatial_modes": 3,
            "innovation_encoder": True,
        },
        "DECLARED EXPLORATORY, NEVER A CANDIDATE: adds `intensity_radial`, whose "
        "MEAN LEVER IS 1.0034 — i.e. the global blur mode relabelled. It exists "
        "to MEASURE that a spatial-looking mode can move G3's criterion without "
        "the ensemble disagreeing about WHERE.",
    ),
)

#: Declared BEFORE the M7 matrix ran.
M7_SELECTION_RULE = (
    "PRE-DECLARED, NOT FITTED. Two candidates, each named before the first run and neither "
    "chosen by score: 'm7_spatial' for DISPERSION (G3) and 'm7_offstate' for the OFF STATE. "
    "'m7_gate_nofix' and 'm7_spatial_blind' are CONTROLS and may never be promoted. "
    "'m7_spatial_r3' is DECLARED EXPLORATORY on a measurement made before it ran (radial mean "
    "lever 1.0034 = the global mode in a spatial costume) and may never be promoted EVEN IF IT "
    "SCORES BEST — which, given its lever, is the outcome I expect. The M6 control 'm6_fair' is "
    "REUSED from its recorded runs and is not retrained. Sanity, TRAIN only: 3 h growth ratio in "
    "[0.8, 1.25]; sigma not collapsed. Failing is reported, never replaced. No arm's loss "
    "references area, dispersion, calibration or any G3 quantity."
)

M7_DECLARED_ENTRIES: tuple[str, ...] = ("m7_spatial", "m7_offstate")

#: [M8] MEASURED BEFORE THE FIRST M8 ARM RAN — `runs/m8_partition_probe.json` and
#: `runs/m8_transfer_probe.json`. Both read labels/weather only; neither reads a
#: held-out score, and neither is fitted.
#:
#: **EVERY MODEL THIS PROJECT HAS TRAINED LOSES THE SAME ~18% OF ITS GROWTH
#: BETWEEN TRAIN AND HELD-OUT, AND SO DOES A ONE-PARAMETER PHYSICS BASELINE.**
#: The wind-advected ellipse's ONLY fitted number is a scale set by C6.2 to
#: reproduce TRAIN mean hourly growth; at 3 h that scale lands at TRAIN ratio
#: **1.0086** and scores held-out **0.8447** — transfer **0.838**. Our arms:
#: `m6_fair` 0.819, `m7_spatial` 0.813, gate arms 0.719. A baseline with no
#: latent, no CNN, no Brier term and no gate shows the SAME loss, so ~0.84 of
#: ADR-034 (1)(4)'s "new failure mode" is a PROPERTY OF THE PARTITION and is not
#: reachable by any change to the kernel. Only the gate's EXTRA 0.719/0.838 =
#: **0.858** is ours to fix, and M8 aims at that number, not at the 0.84.
M8_ELLIPSE_TRANSFER = 0.8447 / 1.0086
M8_GATE_EXTRA_TRANSFER_LOSS = 0.719 / 0.838

#: [M8] FALSIFIED BEFORE THE CANDIDATE WAS DESIGNED, and it is why the candidate
#: is not what I first reached for. HYPOTHESIS: the gate's extra transfer loss is
#: its CONDITIONAL PRIOR extrapolating — `mu_p = gate_prior_mean + b + w . cov`
#: multiplies the field by `e^(sigma mu_p)`, and held-out weather is a different
#: distribution. Predicted `e^(sigma (mu_ho - mu_tr)) ~ 0.878` if true.
#: **MEASURED over all 8 M7 gate seeds: 1.057 — the WRONG SIGN.** The conditional
#: prior makes held-out slightly HOTTER, not cooler. Hypothesis dead; the fix it
#: implied (centre the conditional term) was never built.
M8_CONDITIONAL_PRIOR_TRANSFER = 1.0575

#: [M8] PRE-REGISTERED BEFORE THE FIRST M8 TRAINING RUN. Full text: insights 69-75.
#:
#: **THE ASYMMETRIC PRIOR IS DECLARED HERE AS A REAL G3 CANDIDATE, NOT A CONTROL
#: AND NOT A BY-PRODUCT.** ADR-034 (2) corrected my own axis for me: the working
#: mechanism is SYMMETRIC vs ASYMMETRIC, not global vs spatial. M6's symmetric
#: widening of the global log-multiplier reached 0.58 dispersion with a 2.6x-worse
#: Brier; the activity gate widens THE SAME PHYSICAL CHANNEL with an asymmetric
#: prior and reaches 0.75-0.80 with a BETTER Brier, CRPS and calibration. At this
#: base rate spread bought DOWNWARD is cheap and spread bought UPWARD is
#: unboundedly expensive, so a symmetric latent pays the expensive side to buy the
#: cheap one and a proper score correctly declines it.
#:
#: **TWO CHANGES, AND THE MATRIX IS A 2x2 SO EACH IS ATTRIBUTABLE.** Neither is a
#: knob and neither references a gate quantity:
#:  1. ``gate_mean_preserving`` — extend the parameter-free log-normal correction
#:     to the gate. REVERSES an M6 exemption whose stated premise (protect the OFF
#:     state) ADR-034 (4) falsified: `dormant_off_rate` is 0.0000 on every gate arm
#:     and the ceiling from C1's covariates is 0.143. The exemption protects
#:     nothing measured and costs the mean. **The asymmetry SURVIVES** — the
#:     multiplier stays log-normal with sigma ~1.3, median 0.43 against mean 1.
#:  2. ``spatial_modes = 2`` — the M7 component, composed with the gate for the
#:     first time. M7 held them apart under ADR-029 (7)'s one-variable rule and
#:     both single effects are now MEASURED AND ATTRIBUTED: the gate buys
#:     dispersion (0.2331 -> 0.7552) and costs G2's criterion (0.1555 -> 0.1461);
#:     the spatial modes buy G2's criterion (0.1555 -> **0.1614, the highest of
#:     any arm ever trained**) and are worth ~nothing for area. **G2 and G3 are
#:     CUMULATIVE, so a candidate needs both, and these are the only two
#:     interventions measured to move them in the directions required.**
#:
#: `m7_offstate`'s four recorded seeds are the (no-correction, no-spatial) CELL of
#: the 2x2 and are NOT retrained — the control cannot drift under the candidate.
M8_MATRIX: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "m8_asym",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": True,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "gate_mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "innovation_encoder": True,
            "spatial_modes": 2,
        },
        "THE M8 G3 CANDIDATE, named before the first run: the ASYMMETRIC gate with "
        "its mean put back at 1, composed with the two low-rank spatial gradient "
        "modes. Declared as a CANDIDATE, not a control and not a by-product.",
    ),
    (
        "m8_asym_nospatial",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": True,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "gate_mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "innovation_encoder": True,
        },
        "ATTRIBUTION: the gate mean-correction ALONE, one variable against the "
        "reused `m7_offstate`. This is where P37/P38/P40 are adjudicated. May not "
        "be promoted even if it scores best.",
    ),
    (
        "m8_asym_nocorr",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": True,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "innovation_encoder": True,
            "spatial_modes": 2,
        },
        "ATTRIBUTION: the spatial modes ALONE under the gate, one variable against "
        "the reused `m7_offstate`. This is where P39's mechanism is isolated. May "
        "not be promoted.",
    ),
    (
        "m8_offstate_repro",
        {
            "latent_dim": 4,
            "gate_prior_mean": -1.5,
            "conditional_prior": True,
            "w_kl": 1.0,
            "n_prior_samples": 4,
            "w_growth": 0.0,
            "fair_brier": True,
            "mean_preserving_latent": True,
            "latent_rho": M6B_RHO,
            "innovation_encoder": True,
        },
        "CONTAMINATION CONTROL (M6B's `m5_zt_repro` pattern): `m7_offstate`'s exact "
        "config under M8 code. It must reproduce `m7_offstate_s1`'s fitted sigma "
        "BITWISE, or the gate_mean_preserving=False path is not M7 and every "
        "comparison in this milestone is contaminated.",
    ),
)

#: Declared BEFORE the M8 matrix ran.
M8_SELECTION_RULE = (
    "PRE-DECLARED, NOT FITTED. Candidate = 'm8_asym' (asymmetric gate, mean-corrected, "
    "composed with the two spatial gradient modes), NAMED BEFORE THE FIRST RUN and declared a "
    "REAL G3 CANDIDATE rather than a control or a by-product. 'm8_asym_nospatial' and "
    "'m8_asym_nocorr' are ATTRIBUTION arms completing a 2x2 whose fourth cell is the REUSED, "
    "NOT-RETRAINED 'm7_offstate'; neither may be promoted, even if one scores better than the "
    "candidate. 'm8_offstate_repro' is the contamination control. **G2 AND G3 ARE REPORTED "
    "TOGETHER FOR THE CANDIDATE IN ONE TABLE — gates are cumulative and a G3 number that "
    "surrenders `best_member_iou_shape_masked` is not a pass.** Sanity, TRAIN only: 3 h growth "
    "ratio in [0.8, 1.25]; sigma not collapsed. Failing is reported, never replaced. No arm's "
    "loss references area, dispersion, calibration or any G3 quantity, and nothing in the "
    "matrix was chosen after seeing a held-out number."
)

M8_DECLARED_ENTRIES: tuple[str, ...] = ("m8_asym",)

#: [M8] The six predictions, WRITTEN BEFORE THE FIRST ARM RAN, each with the
#: number that falsifies it. Running total before M8: **14 of 21 falsified across
#: M5/M6/M7**, including predictions in my own favour.
M8_PREDICTIONS: tuple[tuple[str, str], ...] = (
    (
        "P37",
        "UNDER-PREDICTION. `m8_asym_nospatial`'s held-out 3 h growth ratio (4-seed mean) lands "
        "in [0.95, 1.10], against `m7_offstate`'s 0.840. MECHANISM: E_z[e^gate] = 1 makes "
        "sigma_gate a pure spread parameter, so the ensemble mean is set by the mean-fitting "
        "terms alone — as it already is for the fully mean-preserving `m6_fair` (0.9967). "
        "ARITHMETIC: the gate's EXTRA transfer loss is 0.858, and 0.840 / 0.858 = 0.979. "
        "FALSIFIED IF outside [0.95, 1.10].",
    ),
    (
        "P38",
        "THE CORRECTION IS NEARLY ORTHOGONAL TO DISPERSION. `m8_asym_nospatial`'s equal-block "
        "dispersion (4-seed mean) stays within +/-0.15 of `m7_offstate`'s 0.7552, i.e. in "
        "[0.605, 0.905]. MECHANISM: the MEASURED `band_area_error_bias_fraction` of the "
        "dispersion denominator is 0.0046-0.0200, so removing the bias cannot move the ratio "
        "much; the correction is close to a reparameterisation. FALSIFIED IF outside that "
        "interval — and a large RISE would be as informative as a fall, because it would mean "
        "the mean and the spread were entangled far more than the decomposition says.",
    ),
    (
        "P39",
        "G2 RECOVERY. `m8_asym`'s `best_member_iou_shape_masked` (4-seed mean, equal-block) is "
        ">= 0.1520, closing at least 63% of the 0.1461 -> 0.1555 gap the gate opened. "
        "MECHANISM: `m7_spatial` is the HIGHEST-scoring arm on this criterion ever trained "
        "(0.1614 vs 0.1555) and the spatial modes are the only intervention that has ever "
        "raised it. FALSIFIED IF < 0.1520.",
    ),
    (
        "P40",
        "SEED ROBUSTNESS. `m8_asym_nospatial`'s across-seed SD of equal-block dispersion is "
        "BELOW `m7_offstate`'s 0.1704. MECHANISM: once both global log-multiplier dimensions "
        "are mean-preserving, only the TOTAL variance sigma_0^2 + sigma_3^2 is identified and "
        "dispersion depends on the total; uncorrected, the split ALSO moves the mean, so seeds "
        "land on materially different models — s1 (sigma_0 0.281, sigma_gate 1.376) scores "
        "0.906 while s2 (0.664, 0.801) scores 0.512. FALSIFIED IF SD >= 0.1704.",
    ),
    (
        "P41",
        "AGAINST MY OWN CANDIDATE, stated before running. `m8_asym`'s 4-seed mean equal-block "
        "dispersion lands in [0.70, 1.05] and AT MOST 3 of 4 seeds are inside [0.8, 1.2]. "
        "Nothing in M8 targets seed variance directly and M7's SD was 0.15-0.17, so I do not "
        "expect a robust 4/4. FALSIFIED IF 4/4 land inside, or if the mean is outside "
        "[0.70, 1.05].",
    ),
    (
        "P42",
        "THE TRANSFER LOSS IS NOT A WIDTH EFFECT. Across all 20 gate seeds (M7's 8 + M8's 12), "
        "|Spearman(sigma_gate, train->held-out growth transfer)| < 0.5. MECHANISM: M7's 8 gate "
        "seeds already show no monotone relation (sigma_gate 0.801 gives transfer 0.739 while "
        "1.606 gives 0.707 and 1.376 gives 0.686). FALSIFIED IF |rho| >= 0.5, which would make "
        "the extra loss an ensemble-width artifact and put it back inside the kernel.",
    ),
    (
        "P43",
        "**MY CANDIDATE CANNOT FIX BLOCK 5, AND I AM SAYING SO BEFORE SCORING.** ADDED "
        "MID-TRAINING, after the maintainer's S5 redirect and BEFORE any M8 held-out number "
        "existed; it changes no arm's config and no arm had been scored. simviz measured that "
        "our predicted growth rate spans 1.7x across blocks where truth spans 5.7x — we "
        "predict least where truth is fastest. `gate_mean_preserving` is a GLOBAL log-"
        "multiplier: it lifts every block by the SAME factor, so it cannot change a RATIO of "
        "ranges. PREDICTION: on `m8_asym`, block 5 is STILL the lowest-dispersion held-out "
        "block, and the across-block spread of the predicted/truth growth ratio moves by less "
        "than 20% against `m7_offstate`. FALSIFIED IF block 5 is no longer lowest, or the "
        "range moves more than 20%. If this holds, the conditional-mean compression is a "
        "SEPARATE defect from anything M8 touches and needs its own milestone.",
    ),
)

#: [M8] STATED AS A NON-PREDICTION, so that reporting it later is not a surprise:
#: I do NOT predict the OFF state improves. `dormant_off_rate` is expected to stay
#: 0.0000 on every M8 arm. ADR-034 (4) measured the ceiling at 0.143 from the
#: covariates C1 carries, and my own insight 51 ("this needs a diurnal feature")
#: is falsified — a diurnal feature makes LOFO AUC WORSE. That is a FEATURE GAP,
#: not an unfitted model, and M8 deliberately does not spend a run on it.
M8_OFF_STATE_IS_NOT_A_TARGET = True

#: The M2 experiment matrix, pre-registered in docs/decisions.md
#: BEFORE it was run. Each entry names the prediction it tests. Keeping it in
#: source rather than in a shell history is the difference between a
#: pre-registration and a recollection.
M2_MATRIX: tuple[tuple[str, dict[str, Any], str], ...] = (
    ("init_only", {"steps": 0}, "CONTROL: the growth-calibrated elliptical-Gaussian, untrained"),
    (
        "nll_only",
        {"w_growth": 0.0},
        "P1/P2: proper score alone — does it stay silent or attenuate?",
    ),
    ("growth_005", {"w_growth": 0.05}, "P3: aggregate growth moment, weak"),
    ("growth_03", {"w_growth": 0.3}, "P3: aggregate growth moment, medium"),
    ("growth_10", {"w_growth": 1.0}, "P3: aggregate growth moment, strong"),
    (
        "class_weight",
        {"w_growth": 0.0, "growth_windows_only": True},
        "P4 / R14 DIRECT TEST: growth-stratum training instead of a moment constraint",
    ),
    (
        "growth_window_form",
        {"w_growth": 0.3, "growth_moment": "window"},
        "FALSIFIED VARIANT: per-window log-growth regression, kept so it reproduces",
    ),
)

#: Declared BEFORE the M2 matrix ran. A selection rule invented after seeing the
#: numbers is not a selection rule.
SELECTION_RULE = (
    "Among configs whose TRAIN 3 h growth ratio lies in [0.8, 1.25], take the lowest TRAIN "
    "band NLL. If none qualify, take the ratio closest to 1 and SAY that none qualified. "
    "Selection uses TRAIN diagnostics only; no held-out fire is read until selection is final."
)

#: M3 runs the M3 matrix by default; M2's is reachable with ``--matrix m2``.
PREREGISTERED_MATRIX = M3_MATRIX


def run_matrix(
    base: TrainConfig | None = None,
    *,
    entries: Sequence[tuple[str, dict[str, Any], str]] = M3_MATRIX,
    write_run: bool = True,
    verbose: bool = True,
    declared: Sequence[str] = M3_DECLARED_ENTRIES,
    selection_rule: str = M3_SELECTION_RULE,
) -> dict[str, Any]:
    """Run a pre-registered matrix. ``declared`` entries are named, never fitted."""
    cfg0 = base or TrainConfig()
    from wildfire_nowcast.eval.reporting import split_fingerprint

    frozen = split_fingerprint()
    results: dict[str, Any] = {}
    for name, overrides, purpose in entries:
        cfg = TrainConfig(**{**cfg0.to_dict(), **overrides})
        if verbose:
            print(f"\n--- {name}: {purpose}")
        payload = train_kernel(
            cfg,
            train_fire_ids=frozen["train_fire_ids"],
            write_run=write_run,
            run_prefix=f"kernel-{name}",
            verbose=verbose,
            log_every=100,
        )
        final = payload["train_diagnostics"]["final"]
        results[name] = {
            "purpose": purpose,
            "overrides": overrides,
            "run_dir": payload.get("run_dir"),
            "train_band_nll": final["scores"]["band_nll"],
            "train_band_brier": final["scores"]["band_brier"],
            "train_growth_ratio_1h": final["growth"]["1h"]["growth_ratio"],
            f"train_growth_ratio_{cfg.horizon_h}h": final["growth"][f"{cfg.horizon_h}h"][
                "growth_ratio"
            ],
            "parameters": payload["parameters"]["learned"],
            "offset_anisotropy": payload["offset_anisotropy"]["final"],
            "barrier_multiplier": payload["parameters"]["learned"]["barrier_multiplier"],
            "susceptibility_mode": cfg.susceptibility_mode,
            "label_perturbation": cfg.label_perturbation,
            "label_perturbation_scale": cfg.label_perturbation_scale,
            # [M5] the latent, and the two TRAIN-only sanity conditions it has
            # to satisfy before any held-out fire is read.
            "latent": payload["latent"],
            "train_ensemble_collapse_check": payload["train_ensemble_collapse_check"],
            "kl_nats_last": (payload["history"][-1].get("kl") if payload["history"] else None),
            "kl_per_dim_last": (
                payload["history"][-1].get("kl_per_dim") if payload["history"] else None
            ),
        }
    key = f"train_growth_ratio_{cfg0.horizon_h}h"

    named = [n for n in declared if n in results]
    if named:
        # PRE-DECLARED path: the entries were fixed by ADR-015 before the run and
        # are reported whether or not they pass their own sanity condition.
        failing = [
            n for n in named if not (results[n][key] is not None and 0.8 <= results[n][key] <= 1.25)
        ]
        selected = named[0]
        note = (
            f"PRE-DECLARED entries {list(named)}; none chosen by score. "
            + (
                f"SANITY CONDITION FAILED for {failing} (TRAIN {cfg0.horizon_h} h growth ratio "
                "outside [0.8, 1.25]) — reported as failing, NOT replaced."
                if failing
                else "All declared entries meet the TRAIN growth-ratio sanity condition."
            )
        )
        return {
            "kind": "contagion_kernel_matrix",
            "split": frozen,
            "selection_rule": selection_rule,
            "declared_entries": list(named),
            "sanity_condition_failed": failing,
            "selected": selected,
            "selection_note": note,
            "results": results,
        }

    qualified = [
        n for n, r in results.items() if r[key] is not None and 0.8 <= r[key] <= 1.25
    ]
    if qualified:
        selected = min(qualified, key=lambda n: results[n]["train_band_nll"])
        note = f"{len(qualified)} config(s) met the growth-ratio band; lowest TRAIN band NLL wins"
    else:
        selected = min(
            results, key=lambda n: abs(math.log(max(results[n][key] or 1e-9, 1e-9)))
        )
        note = "NO config met the growth-ratio band — selected the closest ratio to 1"
    return {
        "kind": "contagion_kernel_matrix",
        "split": frozen,
        "selection_rule": selection_rule,
        "selected": selected,
        "selection_note": note,
        "results": results,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    d = payload["train_diagnostics"]
    print()
    print("=" * 96)
    print("CONTAGION KERNEL — TRAIN DIAGNOSTICS (no held-out fire was read)")
    print("=" * 96)
    for phase in ("init", "final"):
        g = d[phase]["growth"]
        s = d[phase]["scores"]
        horizons = ", ".join(
            f"{k} ratio {v['growth_ratio']:.3f} ({v['predicted_new_cells']:.0f}/"
            f"{v['observed_new_cells']:.0f})"
            for k, v in g.items()
            if v["growth_ratio"] is not None
        )
        print(
            f"{phase:<6} band_nll {s['band_nll']:.5f}  "
            f"band_brier {s['band_brier']:.6f}  {horizons}"
        )
    s = d["final"]["scores"]
    print(
        f"spot budget: {s['unreachable_new_cells']:.0f} of {s['truth_new_cells']:.0f} new cells "
        f"({s['unreachable_fraction']:.1%}) lie beyond the kernel's {s['reach_cells']}-cell reach"
    )
    p = payload["parameters"]
    print("learned vs init:")
    for key in ("r0_ms", "u_ref_ms", "wind_exponent", "k_slope", "lb_gain", "moisture_gain"):
        print(f"    {key:<16} {p['init'][key]:>10.4g} -> {p['learned'][key]:>10.4g}")
    print(f"    {'alpha':<16} {'--':>10} -> {p['learned']['alpha']:>10.4g}")
    print(f"    {'gamma':<16} {1.0:>10.4g} -> {p['learned']['gamma']:>10.4g}")
    print(
        f"    {'barrier_mult':<16} {p['init']['barrier_multiplier']:>10.4g} -> "
        f"{p['learned']['barrier_multiplier']:>10.4g}"
    )
    for name in ("GR", "SH", "TL"):
        print(
            f"    fuel[{name}]{'':<8} {p['init']['fuel_multipliers'][name]:>10.4g} -> "
            f"{p['learned']['fuel_multipliers'][name]:>10.4g}"
        )
    grad = payload["susceptibility_gradient"]
    print(
        f"gradient check (ADR-015 6a): d loss/d barrier_log_multiplier = "
        f"{grad['amplitude']['d_loss_d_barrier_log_multiplier']:.3e} in AMPLITUDE mode vs "
        f"{grad['reach']['d_loss_d_barrier_log_multiplier']:.3e} in M2's REACH mode"
    )
    aniso = payload["offset_anisotropy"]["final"]
    bearing = "--" if aniso["bearing_deg"] is None else f"{aniso['bearing_deg']:.0f} deg"
    print(
        f"wind-INDEPENDENT offset anisotropy (ADR-015 6b): magnitude "
        f"{aniso['magnitude']:.4f}, bearing {bearing}   "
        f"(label perturbation {'ON' if payload['label_perturbation']['enabled'] else 'off'})"
    )
    if "run_dir" in payload:
        print(f"written: {payload['run_dir']}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.model.train",
        description="Train the deterministic contagion kernel on TRAIN fires only.",
    )
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--horizon", type=int, default=TrainConfig.horizon_h)
    parser.add_argument("--radius", type=int, default=TrainConfig.radius)
    parser.add_argument("--w-growth", type=float, default=TrainConfig.w_growth)
    parser.add_argument("--growth-moment", choices=("batch", "window"), default="batch")
    parser.add_argument("--w-brier", type=float, default=TrainConfig.w_brier)
    parser.add_argument("--w-nll", type=float, default=TrainConfig.w_nll)
    parser.add_argument("--growth-only", action="store_true")
    parser.add_argument("--growth-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--prefix", type=str, default="kernel")
    parser.add_argument("--no-run-dir", action="store_true")
    parser.add_argument(
        "--matrix",
        nargs="?",
        const="m3",
        choices=("m2", "m3"),
        help="run a pre-registered matrix (default m3)",
    )
    parser.add_argument("--susceptibility-mode", choices=SUSCEPTIBILITY_MODES, default="amplitude")
    parser.add_argument("--label-perturbation", action="store_true")
    parser.add_argument("--label-perturbation-scale", type=float, default=1.0)
    parser.add_argument(
        "--mirror-ns",
        action="store_true",
        help="ABLATION: reflect every train fire north-south in memory (no tensor "
        "is modified). Diagnostic for the learned S/SW anisotropy ONLY — the "
        "resulting model must not be scored on real held-out fires.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = TrainConfig(
        horizon_h=args.horizon,
        radius=args.radius,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        w_nll=args.w_nll,
        w_brier=args.w_brier,
        w_growth=args.w_growth,
        growth_moment=args.growth_moment,
        growth_windows_only=args.growth_only,
        growth_window_weight=args.growth_weight,
        susceptibility_mode=args.susceptibility_mode,
        label_perturbation=args.label_perturbation,
        label_perturbation_scale=args.label_perturbation_scale,
        seed=args.seed,
        mirror_ns=args.mirror_ns,
    )
    if args.matrix:
        is_m3 = args.matrix == "m3"
        out = run_matrix(
            cfg,
            entries=M3_MATRIX if is_m3 else M2_MATRIX,
            write_run=not args.no_run_dir,
            declared=M3_DECLARED_ENTRIES if is_m3 else (),
            selection_rule=M3_SELECTION_RULE if is_m3 else SELECTION_RULE,
        )
        print()
        print("=" * 118)
        print(f"{args.matrix.upper()} PRE-REGISTERED MATRIX — TRAIN diagnostics only")
        print("=" * 118)
        print(
            f"{'config':<24}{'band_nll':>10}{'band_brier':>12}{'ratio_1h':>10}"
            f"{'ratio_3h':>10}{'aniso':>9}{'bearing':>9}{'barrier':>10}"
        )
        key3 = f"train_growth_ratio_{cfg.horizon_h}h"
        for name, r in out["results"].items():
            a = r.get("offset_anisotropy", {})
            bearing = a.get("bearing_deg")
            print(
                f"{name:<24}{r['train_band_nll']:>10.5f}{r['train_band_brier']:>12.6f}"
                f"{(r['train_growth_ratio_1h'] or float('nan')):>10.3f}"
                f"{(r[key3] or float('nan')):>10.3f}"
                f"{a.get('magnitude', float('nan')):>9.4f}"
                f"{('--' if bearing is None else f'{bearing:.0f}'):>9}"
                f"{r.get('barrier_multiplier', float('nan')):>10.2e}"
            )
        for name, _, purpose in (M3_MATRIX if is_m3 else M2_MATRIX):
            print(f"    {name:<24}{purpose}")
        print(f"\nSELECTION RULE: {out['selection_rule']}")
        if out.get("declared_entries"):
            print(f"DECLARED ENTRIES: {out['declared_entries']}")
            print(f"  {out['selection_note']}")
            for name in out["declared_entries"]:
                print(f"  -> {name}: {out['results'][name]['run_dir']}")
        else:
            print(f"SELECTED: {out['selected']}  ({out['selection_note']})")
            print(f"  -> {out['results'][out['selected']]['run_dir']}")
        import json as _json

        from wildfire_nowcast.common.paths import runs_dir

        path = runs_dir() / f"{args.matrix}_matrix.json"
        path.write_text(_json.dumps(out, indent=2, default=float) + "\n")
        print(f"written: {path}")
        return 0
    payload = train_kernel(cfg, write_run=not args.no_run_dir, run_prefix=args.prefix)
    _print_summary(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
