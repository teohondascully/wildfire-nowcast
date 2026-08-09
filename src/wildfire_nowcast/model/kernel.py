"""The anisotropic contagion kernel — component (a) of the C1 transition kernel.

CLAUDE.md, ground truth, not re-litigated here::

    (a) short-range anisotropic contagion (CNN over burning neighbourhood,
        initialized elliptical-Gaussian, stretched by wind/slope), plus
    (b) explicit long-range spot component (learned rate + downwind dispersal).
    Pixels are conditionally independent Bernoulli ONLY given a shared per-step
    latent z_t.

**[M5] The shared per-step latent ``z_t`` is now IMPLEMENTED, and it is
OPTIONAL.** ``latent_config=None`` (or ``dim=0``) reproduces the deterministic
G2 kernel BIT-IDENTICALLY — asserted by :func:`check_latent_off_is_bit_identical`,
not by inspection — so the P1/G2 result is not silently re-defined by the P2
model. The latent itself lives in :mod:`wildfire_nowcast.model.latent`; this
module is where it enters the physics. The spot component (b) is still ABSENT
(P3), and :meth:`ContagionKernel.to_spec` says so in the artifact, so a run
directory can never be mistaken for the full model.

The transition
--------------
``b_t(x)`` is the probability that cell ``x`` has burned by the end of hour
``t``. Fire is absorbing, so::

    lambda_t(x) = sum_{d in N} w_d(x, t) * b_{t-1}(x + d)      # contagion
    p_t(x)      = 1 - exp(-lambda_t(x))                        # survival form
    b_t(x)      = b_{t-1}(x) + (1 - b_{t-1}(x)) * p_t(x)       # absorbing

The survival form is not cosmetic: it makes independent contagion sources
*additive in the hazard*, which is what lets the spot component be added later
as another term in ``lambda`` rather than as a competing probability that has to
be merged by hand. It also bounds ``p`` in ``[0, 1)`` without a clamp.

``b_{t-1}`` under the expectation is a MEAN FIELD: it propagates probabilities,
not sampled states. With a latent present, ``z_t`` enters ``lambda`` in three
places — a global log-multiplier on the hazard, a rotation of the effective head
direction and a log-scale on the effective wind speed — so ONE draw moves the
whole field coherently and the pixels are conditionally independent Bernoulli
ONLY GIVEN ``(x_t, z_t)``, which is CLAUDE.md's model. With
``latent_config=None`` the only randomness left is the per-pixel draw: that is
the known-broken independent-noise sampler, permitted ONLY as the G3 ablation,
and the spec names it so nobody can read a dispersion number off it by accident.

The kernel weights
------------------
For offset ``d`` at distance ``|d|`` cells, arriving at ``x`` from direction
``theta`` relative to the local head direction::

    r_d(x)     = R_reach(x) * ellipse_factor(LB(x), cos theta_d)   # cells / hour
    log w_d(x) = log alpha + c_d + log s(x) - 0.5 * (|d| / (gamma * r_d(x)))^2

``r_d`` is the ELLIPSE's own directional rate of spread — reach from the cell's
moisture and effective (wind + slope) forcing, reduced by the exact polar factor
of an ellipse generated from its rear focus. So ``w`` is an elliptical Gaussian
whose axes are stretched by wind and slope, which is the initialisation
CLAUDE.md specifies, taken from the *same functions the ellipse baseline is
scored with* (:mod:`wildfire_nowcast.model.spread`) rather than from a second
implementation of them. "The CNN beat the ellipse" therefore cannot quietly mean
"the CNN beat a different ellipse".

``c_d`` is the learned part: one free log-weight per offset, initialised at 0.
At initialisation the kernel is EXACTLY the elliptical Gaussian; every departure
from it is something the fires taught it, and is inspectable offset by offset.

``s(x)`` is SUSCEPTIBILITY — the target cell's fuel-group multiplier times
``exp(barrier_log_multiplier)`` on barrier cells. **Where it enters is a defect
fix, not a style choice** (ADR-015 (6a), M2 insight 15).

M2 multiplied ``s`` into the head rate, so it landed inside the Gaussian
exponent::

    log w = ... - 0.5 * (|d| / (gamma * s * r))^2      # M2, mode="reach"

A barrier cell then had ``s = e^-6``, an exponent of ``-0.5 * (1/(0.0025 r))^2``
of order ``-1e4``, and ``exp`` of that is a HARD ZERO in float64 (the smallest
representable is ``e^-745``). Zero weight, and therefore **exactly zero
gradient**: ``barrier_log_multiplier`` and the non-burnable fuel multiplier did
not move in any of M2's seven configs, and could not have. Barrier crossing —
the P3/G4 mechanism — was structurally unlearnable, and every burnable fuel
multiplier flattening to ~1.0 was equally uninterpretable, because it could have
been either a finding or the same artefact.

In ``mode="amplitude"`` (the default, and the fix) susceptibility multiplies the
hazard AMPLITUDE, so ``d log w / d barrier_log_multiplier = barrier(x)``
EXACTLY: every barrier cell adjacent to fire supplies gradient at every step.
The physical consequence is that a barrier is now *leaky by design* — its hazard
is ``e^-6`` of open ground rather than identically zero — and that leak IS the
gradient. ``mode="reach"`` is retained so the defect reproduces on demand, the
same way the falsified per-window growth moment is retained in
:mod:`wildfire_nowcast.model.train`.

The cost of the fix, stated: in amplitude mode the fuel group no longer stretches
the kernel's reach, only its amplitude. That is a deliberate narrowing — fuel
enters once, identifiably, instead of twice in a way that made one of its two
routes unidentifiable.

Why the physics is re-implemented in torch, and how that is kept honest
----------------------------------------------------------------------
The forward pass must be differentiable, and :mod:`spread` is numpy. Two
implementations of one physics is precisely the failure mode C0 exists to
prevent, so :func:`check_torch_matches_numpy` asserts the torch path reproduces
the numpy path to 1e-9 on random inputs, and it is part of the eval self-test.
The duplication is admitted and then continuously tested, rather than assumed
away.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, UNBURNED
from wildfire_nowcast.model.api import validate_predict_inputs
from wildfire_nowcast.model.inputs import static_index, weather_index
from wildfire_nowcast.model.latent import (
    LatentConfig,
    LatentEffect,
    LatentHead,
    LatentSampler,
    latent_report,
    spatial_basis,
)
from wildfire_nowcast.model.spread import FUEL_GROUPS, EllipseParams, fuel_group

__all__ = [
    "DTYPE",
    "SUSCEPTIBILITY_MODES",
    "KernelConfig",
    "ContagionKernel",
    "StaticFields",
    "static_fields_from_array",
    "neighbour_offsets",
    "shift_field",
    "check_torch_matches_numpy",
    "check_latent_off_is_bit_identical",
    "step_covariates",
    "spatial_log_intensity_field",
    "susceptibility_gradient_report",
    "offset_anisotropy",
    "parameter_report",
    "offset_kernel_table",
]

#: float64 everywhere. The radial term is ``exp(-0.5 (d / (gamma r))^2)`` and a
#: non-burnable cell drives it to ``exp(-2e2)`` or below; in float32 that
#: underflows to a hard zero and takes the gradient with it, in float64 it stays
#: representable. The grids are ~3,000 cells, so the cost of double precision
#: here is nothing and the cost of underflow is a silently untrainable model.
DTYPE = torch.float64

_GROUP_ORDER: tuple[str, ...] = ("NB", "GR", "GS", "SH", "TU", "TL", "SB")
_EPS = 1e-12

#: Where susceptibility (fuel group + barrier) enters the log-weight.
#:
#: ``"amplitude"`` — an additive term in ``log w``. ``d log w / d theta`` is the
#: indicator of the cell, so the parameter is identified by every adjacent-to-
#: fire cell. THIS IS THE DEFAULT AND THE CORRECT ONE (ADR-015 (6a)).
#: ``"reach"`` — M2's form, inside the Gaussian exponent, where the weight
#: underflows to a hard zero and the gradient is EXACTLY zero. Retained so the
#: defect reproduces and so the fix can be measured against it, never as a
#: candidate.
SUSCEPTIBILITY_MODES: tuple[str, ...] = ("amplitude", "reach")


def neighbour_offsets(radius: int) -> tuple[tuple[int, int, float], ...]:
    """``(dr, dc, distance)`` for every offset within EUCLIDEAN ``radius``.

    Circular rather than square: a square window would give the corners a
    1.41-cell reach and the edges a 1-cell reach *as a property of the indexing*,
    which is an anisotropy the model would then have to learn its way out of.

    Row index increases SOUTHWARD (C1.4 north-up), so the (east, north) travel
    direction of an offset is ``(dc, -dr)`` — the same convention as
    :mod:`wildfire_nowcast.model.baselines.ellipse`, deliberately.
    """
    out = []
    r = int(radius)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr == 0 and dc == 0:
                continue
            d = math.hypot(dr, dc)
            if d <= r + 1e-9:
                out.append((dr, dc, d))
    return tuple(out)


@dataclass(frozen=True)
class KernelConfig:
    """Structure of the kernel. Everything here is fixed; weights are learned."""

    radius: int = 3
    cell_size_m: float = 1000.0
    lb_cap: float = 8.0
    #: Hours a cell stays in state 1 before state 2. Affects only the 1-vs-2
    #: split, never the burned SET, so it cannot move a C6 score under
    #: ``event="burned"``; it exists so the rendered state field is plausible.
    burnout_hours: int = 4
    #: Non-burnable fuel starts at ``exp(-9)`` rather than 0 so the gradient
    #: exists. A model that must learn "parking lots do not burn" from data it
    #: cannot receive a gradient about would keep whatever it was initialised
    #: with and look like it had learned it.
    nb_log_multiplier: float = -9.0
    #: Barrier cells start at ``exp(-6)`` (~blocked) and are LEARNABLE, because
    #: the barrier-crossing episodes are the thing P3 exists to explain and a
    #: hard zero would make crossings unlearnable rather than merely rare.
    #: That sentence was FALSE under ``susceptibility_mode="reach"``: the weight
    #: underflowed to a hard zero anyway and the gradient was exactly 0. It is
    #: true under ``"amplitude"``, which is why the default moved.
    barrier_log_multiplier: float = -6.0
    #: See :data:`SUSCEPTIBILITY_MODES`. Default "amplitude" (ADR-015 (6a)).
    susceptibility_mode: str = "amplitude"
    #: SOFT FLOOR on the Simard moisture damping, applied in the KERNEL ONLY:
    #: ``eta_eff = floor + (1 - floor) * eta``. Fully-healthy fuel (``eta = 1``)
    #: is UNCHANGED, so this is not a general speed-up; it only stops a
    #: fully-damped cell from having EXACTLY zero reach.
    #:
    #: WHY IT EXISTS. ``spread.FUEL_GROUPS["NB"]`` carries
    #: ``moisture_of_extinction = 1.0%``, so for any realistic dead fuel moisture
    #: the Simard ratio clamps to 1 and ``1 - 2.59r + 5.11r^2 - 3.52r^3``
    #: evaluates to **0.0 exactly**. Reach is then 0, the radial term is -inf and
    #: the weight is a HARD ZERO with EXACTLY ZERO GRADIENT — i.e. a hard
    #: non-burnable mask, and an unlearnable one. That is the SAME defect class as
    #: ADR-015 (6a) one level down: a susceptibility-like quantity sitting in the
    #: denominator of an exponent.
    #:
    #: It is a modelling error against our own labels, not merely inelegant:
    #: sim measured that cells this project's ``fuel_model_id`` channel
    #: calls NON-BURNABLE burn in the GOFER labels **66-84% as often as burnable
    #: ones** (2026-08-08 coordination). No hard mask can fit those labels. With
    #: the floor the suppression is carried by the LEARNABLE
    #: ``fuel_log_multiplier["NB"]`` instead, which is where it belongs.
    #:
    #: Deliberately NOT fixed in ``spread.FUEL_GROUPS``: that constant is also the
    #: ELLIPSE BASELINE's physics, and the ellipse is the G2 opponent. Changing
    #: the opponent mid-gate is not allowed. Kernel-only keeps the comparison
    #: honest and leaves the baseline bit-identical.
    moisture_damping_floor: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.moisture_damping_floor < 1.0:
            raise ValueError(
                f"moisture_damping_floor={self.moisture_damping_floor}; expected [0, 1). "
                "0 reproduces the hard non-burnable mask (unlearnable) on purpose."
            )
        if self.susceptibility_mode not in SUSCEPTIBILITY_MODES:
            raise ValueError(
                f"susceptibility_mode={self.susceptibility_mode!r}; "
                f"expected one of {SUSCEPTIBILITY_MODES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "radius": self.radius,
            "cell_size_m": self.cell_size_m,
            "lb_cap": self.lb_cap,
            "burnout_hours": self.burnout_hours,
            "nb_log_multiplier": self.nb_log_multiplier,
            "barrier_log_multiplier": self.barrier_log_multiplier,
            "susceptibility_mode": self.susceptibility_mode,
        }


@dataclass
class StaticFields:
    """Per-fire static tensors, derived once and reused for every window.

    Built from the C5 ``static`` array via :func:`static_fields_from_array`, so
    the channel order is :data:`~wildfire_nowcast.model.inputs.STATIC_INPUT_CHANNELS`
    and never a literal index.
    """

    fuel_onehot: Tensor  # [G, H, W]
    moisture_of_extinction: Tensor  # [H, W] %
    barrier: Tensor  # [H, W] {0,1}
    canopy_fraction: Tensor  # [H, W] in [0,1]
    tan2_slope: Tensor  # [H, W]
    aspect_sin: Tensor  # [H, W]
    aspect_cos: Tensor  # [H, W]
    shape: tuple[int, int] = field(default=(0, 0))


def static_fields_from_array(static: np.ndarray, *, device: Any = None) -> StaticFields:
    """C5 ``static`` ``f32[C_s,H,W]`` (RAW physical units) -> :class:`StaticFields`."""
    arr = np.asarray(static, dtype=np.float64)

    def get(name: str) -> np.ndarray:
        return arr[static_index(name)]

    codes = np.rint(get("fuel_model_id")).astype(np.int64)
    onehot = np.zeros((len(_GROUP_ORDER), *codes.shape), dtype=np.float64)
    mx = np.zeros(codes.shape, dtype=np.float64)
    flat = codes.ravel()
    groups = np.array([_GROUP_ORDER.index(fuel_group(int(c))) for c in flat]).reshape(codes.shape)
    for g, name in enumerate(_GROUP_ORDER):
        member = groups == g
        onehot[g][member] = 1.0
        mx[member] = FUEL_GROUPS[name][1]

    slope = np.clip(get("slope"), 0.0, 89.0)
    t = torch.as_tensor
    return StaticFields(
        fuel_onehot=t(onehot, dtype=DTYPE, device=device),
        moisture_of_extinction=t(mx, dtype=DTYPE, device=device),
        barrier=t((get("water_barrier_mask") > 0.5).astype(np.float64), dtype=DTYPE, device=device),
        canopy_fraction=t(
            np.clip(get("canopy_cover") / 100.0, 0.0, 1.0), dtype=DTYPE, device=device
        ),
        tan2_slope=t(np.tan(np.radians(slope)) ** 2, dtype=DTYPE, device=device),
        aspect_sin=t(get("aspect_sin"), dtype=DTYPE, device=device),
        aspect_cos=t(get("aspect_cos"), dtype=DTYPE, device=device),
        shape=(int(codes.shape[0]), int(codes.shape[1])),
    )


def shift_field(b: Tensor, dr: int, dc: int) -> Tensor:
    """``out[..., r, c] = b[..., r - dr, c - dc]``; off-grid is 0, never wrapped.

    A wrapped fire spreads off the north edge and reappears in the south, which
    looks exactly like long-range spotting and would be attributed to the spot
    component at P3.

    Public because :mod:`wildfire_nowcast.model.labelnoise` translates labels
    with it. One implementation, per C0.
    """
    out = torch.zeros_like(b)
    height, width = b.shape[-2], b.shape[-1]
    dst_r = slice(max(dr, 0), height + min(dr, 0))
    src_r = slice(max(-dr, 0), height + min(-dr, 0))
    dst_c = slice(max(dc, 0), width + min(dc, 0))
    src_c = slice(max(-dc, 0), width + min(-dc, 0))
    out[..., dst_r, dst_c] = b[..., src_r, src_c]
    return out


_shift = shift_field


def step_covariates(weather_step: Tensor) -> Tensor:
    """[M6] Global covariates the CONDITIONAL PRIOR sees for one step. ``[..., 3]``.

    Domain means of RH, temperature and wind SPEED, standardised by fixed
    constants rather than by ``norm_stats``. That choice is deliberate: norm
    stats are computed over the TRAIN folds, so conditioning the prior on them
    would make the generative model a function of the SPLIT, and a fold change
    would silently change what the model predicts (ADR-015's hazard, one level
    down). These constants are climatological round numbers, not fitted, so
    C-3 has nothing to bind to.

    RH and temperature are chosen because they carry the DIURNAL CYCLE, and
    dormancy is diurnal: GOES cannot see new front at night, and the fire is
    quietest when RH recovers.
    """
    u = weather_step[..., weather_index("wind_u10"), :, :]
    v = weather_step[..., weather_index("wind_v10"), :, :]
    rh = weather_step[..., weather_index("rh_2m"), :, :]
    temp = weather_step[..., weather_index("temp_2m"), :, :]
    speed = torch.sqrt(u * u + v * v + _EPS)
    dims = (-2, -1)
    return torch.stack(
        [
            (rh.mean(dim=dims) - 50.0) / 25.0,
            (temp.mean(dim=dims) - 290.0) / 10.0,
            (speed.mean(dim=dims) - 4.0) / 3.0,
        ],
        dim=-1,
    )


def _broadcast_to_field(scalar: Tensor, n_trailing: int) -> Tensor:
    """Reshape a PER-DRAW latent scalar so it broadcasts over ``n_trailing`` field axes.

    ``scalar`` carries one value per draw of ``z_t`` — one per ensemble member,
    or one per training window. ``n_trailing`` is the number of trailing axes of
    the field it modulates (2 for ``[..., H, W]``, 3 for ``[..., K, H, W]``).

    Trailing singleton axes are APPENDED to the draw rather than the field being
    expanded, for two reasons. The draw stays EXACTLY constant across every cell
    it is shared by — the property "shared per-step latent" names, and the one a
    per-pixel noise field would not have. And it broadcasts correctly when the
    field carries NO member axis of its own: in :meth:`ContagionKernel.predict`
    the weather is ``[C_w, H, W]`` for all members at once, so counting axes
    relative to the field (rather than fixing the trailing count) silently
    aligned the 24 members against the 24 leading grid rows. That is the kind of
    error that produces plausible numbers, so the trailing count is now stated by
    the caller instead of inferred.
    """
    if scalar.dim() == 0:
        return scalar
    return scalar.reshape(*scalar.shape, *([1] * int(n_trailing)))


def spatial_log_intensity_field(burned: Tensor, latent: LatentEffect | None) -> Tensor | None:
    """[M7] Contract a draw's spatial coefficients against the fire-anchored basis.

    Returns ``[..., H, W]``, or ``None`` when the model has no spatial modes (in
    which case every downstream expression is untouched and M6 reproduces
    bitwise).

    The split of labour is deliberate and matches the one ``head_rotation``
    already uses: the LATENT says how much of each mode to apply, and the KERNEL
    — the only object holding ``b_{t-1}`` — knows what the modes are anchored
    to. The basis is therefore recomputed at every step of a rollout from the
    state that step actually reached, so "one fire-radius east of the fire" keeps
    meaning that as the fire grows, instead of freezing at ``t0``.

    With ``mean_preserving`` the log-normal correction is the FIELD
    ``-0.5 sum_m sigma_m^2 phi_m(x)^2``, which makes ``E_z[e^effect] = 1`` hold
    POINTWISE — not merely on average over the domain. That is the property that
    keeps ``sigma`` a pure spread parameter, and a domain-average version would
    quietly move the mean hazard around inside the field while looking correct in
    aggregate.
    """
    if latent is None or latent.spatial_intensity is None:
        return None
    coeff = latent.spatial_intensity  # [..., R]
    basis = spatial_basis(burned, coeff.shape[-1])  # [..., R, H, W]
    field = torch.einsum("...m,...mhw->...hw", coeff, basis)
    if latent.spatial_variance is not None:
        field = field - 0.5 * torch.einsum(
            "...m,...mhw->...hw", latent.spatial_variance, basis * basis
        )
    return field


class ContagionKernel(nn.Module):
    """Deterministic anisotropic contagion kernel. No ``z_t``, no spot component.

    Parameters are stored unconstrained (log / logit) so that every physical
    quantity stays in its admissible range under unconstrained optimisation: a
    negative rate of spread or a length-to-breadth below 1 is not a bad fit, it
    is a different model.
    """

    kind = "contagion_kernel"

    def __init__(
        self,
        config: KernelConfig | None = None,
        *,
        name: str = "kernel",
        ellipse_params: EllipseParams | None = None,
        latent_config: LatentConfig | None = None,
        sampler: LatentSampler | None = None,
    ) -> None:
        super().__init__()
        self.config = config or KernelConfig()
        self.name = name
        base = ellipse_params or EllipseParams()
        self.ellipse_init = base
        #: [M5] The shared per-step latent. ``None`` is the deterministic G2
        #: kernel and is BIT-IDENTICAL to the pre-M5 code path
        #: (:func:`check_latent_off_is_bit_identical`).
        self.latent: LatentHead | None = (
            None
            if latent_config is None or latent_config.dim == 0
            else LatentHead(latent_config)
        )
        #: How :meth:`predict` draws ``z_t``. An ATTRIBUTE, not an argument,
        #: because C5's six-parameter signature is fixed and the ablation must
        #: not become a seventh parameter every caller has to know about.
        self.sampler: LatentSampler = sampler or LatentSampler("latent")
        #: C8 (INTERFACES v2.8). Stamped by the trainer; travels in ``model.json``
        #: so an evaluator can HARD FAIL when the split it is scoring against is
        #: not the split this model was fitted on. An unstamped kernel is
        #: reported as unverifiable, never as matching.
        self.provenance: dict[str, Any] = {}

        offsets = neighbour_offsets(self.config.radius)
        self.register_buffer(
            "offset_dr", torch.tensor([o[0] for o in offsets], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "offset_dc", torch.tensor([o[1] for o in offsets], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "offset_dist", torch.tensor([o[2] for o in offsets], dtype=DTYPE), persistent=False
        )
        # Travel direction (source -> target) in (east, north).
        self.register_buffer(
            "offset_east",
            torch.tensor([o[1] / o[2] for o in offsets], dtype=DTYPE),
            persistent=False,
        )
        self.register_buffer(
            "offset_north",
            torch.tensor([-o[0] / o[2] for o in offsets], dtype=DTYPE),
            persistent=False,
        )
        self.n_offsets = len(offsets)

        p = lambda v: nn.Parameter(torch.tensor(float(v), dtype=DTYPE))  # noqa: E731
        # --- the physics knobs, initialised at the calibrated ellipse ---
        self.log_r0 = p(math.log(base.r0_ms))
        self.log_u_ref = p(math.log(base.u_ref_ms))
        self.log_wind_exponent = p(math.log(base.wind_exponent))
        self.log_k_slope = p(math.log(base.k_slope))
        self.logit_waf_open = p(math.log(base.waf_open / (1 - base.waf_open)))
        self.logit_waf_closed = p(math.log(base.waf_closed / (1 - base.waf_closed)))
        self.log_lb_gain = p(0.0)  # 1 = Anderson's length-to-breadth, unmodified
        self.log_moisture_gain = p(0.0)
        # --- the kernel's own shape ---
        self.log_alpha = p(0.0)  # global intensity; growth-calibrated at init
        self.log_gamma = p(0.0)  # radial reach as a multiple of the ellipse ROS
        self.offset_log_weight = nn.Parameter(torch.zeros(self.n_offsets, dtype=DTYPE))
        # --- susceptibility ---
        init_fuel = [
            self.config.nb_log_multiplier if n == "NB" else math.log(FUEL_GROUPS[n][0])
            for n in _GROUP_ORDER
        ]
        self.fuel_log_multiplier = nn.Parameter(torch.tensor(init_fuel, dtype=DTYPE))
        self.barrier_log_multiplier = p(self.config.barrier_log_multiplier)

    # -- physics, in torch -------------------------------------------------

    def _effective_wind(
        self,
        u10: Tensor,
        v10: Tensor,
        fields: StaticFields,
        latent: LatentEffect | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        waf_open = torch.sigmoid(self.logit_waf_open)
        waf_closed = torch.sigmoid(self.logit_waf_closed)
        waf = waf_open + fields.canopy_fraction * (waf_closed - waf_open)
        u_mid, v_mid = u10 * waf, v10 * waf
        magnitude = torch.exp(self.log_k_slope) * fields.tan2_slope
        u_slope = -magnitude * fields.aspect_sin
        v_slope = -magnitude * fields.aspect_cos
        ue, vn = u_mid + u_slope, v_mid + v_slope
        speed = torch.sqrt(ue * ue + vn * vn + _EPS)
        unit_e = torch.where(speed > 1e-9, ue / speed, torch.ones_like(speed))
        unit_n = torch.where(speed > 1e-9, vn / speed, torch.zeros_like(speed))
        if latent is not None:
            # z_t rotates and rescales the EFFECTIVE forcing (wind + slope), not
            # the raw wind. One draw therefore turns the whole head direction of
            # the step together — the correlated innovation — instead of adding
            # independent jitter that a mean field would absorb.
            theta = _broadcast_to_field(latent.head_rotation, 2)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            # Rotation in the (east, north) plane; positive theta is
            # counter-clockwise, i.e. E -> N, and is reported in degrees.
            unit_e, unit_n = (
                unit_e * cos_t - unit_n * sin_t,
                unit_e * sin_t + unit_n * cos_t,
            )
            speed = speed * torch.exp(_broadcast_to_field(latent.log_wind_speed, 2))
        return speed, unit_e, unit_n

    def _length_to_breadth(self, speed: Tensor) -> Tensor:
        """Anderson (1983), with a learnable gain on the departure from 1."""
        u_mph = speed * 2.236936
        lb = 0.936 * torch.exp(0.2566 * u_mph) + 0.461 * torch.exp(-0.1548 * u_mph) - 0.397
        lb = 1.0 + torch.exp(self.log_lb_gain) * (lb - 1.0)
        return torch.clamp(lb, 1.0, float(self.config.lb_cap))

    def _moisture_damping(self, fm_pct: Tensor, mx_pct: Tensor) -> Tensor:
        ratio = torch.clamp(
            torch.exp(self.log_moisture_gain) * fm_pct / torch.clamp(mx_pct, min=1e-6), 0.0, 1.0
        )
        eta = 1.0 - 2.59 * ratio + 5.11 * ratio**2 - 3.52 * ratio**3
        return torch.clamp(eta, 0.0, 1.0)

    def log_susceptibility(self, fields: StaticFields) -> Tensor:
        """``log s(x)`` — the TARGET cell's receptivity. ``[H, W]``.

        Additive in log space, so ``d log s / d barrier_log_multiplier`` is the
        barrier indicator exactly, and ``d log s / d fuel_log_multiplier[g]`` is
        the group-``g`` indicator exactly. Both are non-zero wherever such a cell
        is within reach of fire — which is what makes the barrier learnable.
        """
        log_fuel = torch.einsum("g,g...->...", self.fuel_log_multiplier, fields.fuel_onehot)
        return log_fuel + self.barrier_log_multiplier * fields.barrier

    def _reach_cells_per_hour(
        self, speed: Tensor, fields: StaticFields, fm_pct: Tensor
    ) -> Tensor:
        """Head rate in cells/h. Susceptibility is EXCLUDED in amplitude mode.

        In ``"reach"`` mode (M2's defect, kept reproducible) susceptibility is
        multiplied in here, which is what put it inside ``exp(-0.5 (d/(gamma r))^2)``
        and drove its gradient to exactly zero.
        """
        eta = self._moisture_damping(fm_pct, fields.moisture_of_extinction)
        # SOFT floor, not a clamp: eta = 1 maps to 1, so healthy fuel is exactly
        # unchanged and only the fully-damped cells stop being a hard zero.
        # See KernelConfig.moisture_damping_floor for why this is a defect fix.
        floor = self.config.moisture_damping_floor
        eta = floor + (1.0 - floor) * eta
        normalised = torch.clamp(speed, min=0.0) / torch.exp(self.log_u_ref)
        wind_response = 1.0 + normalised ** torch.exp(self.log_wind_exponent)
        ros_ms = torch.exp(self.log_r0) * eta * wind_response
        if self.config.susceptibility_mode == "reach":
            ros_ms = ros_ms * torch.exp(self.log_susceptibility(fields))
        return ros_ms * 3600.0 / self.config.cell_size_m

    # -- the kernel --------------------------------------------------------

    def log_weights(
        self,
        weather_step: Tensor,
        fields: StaticFields,
        latent: LatentEffect | None = None,
        spatial_log_intensity: Tensor | None = None,
    ) -> Tensor:
        """``log w_d(x)`` for every offset. ``[..., K, H, W]``.

        ``weather_step`` is ``[..., C_w, H, W]`` in
        :data:`~wildfire_nowcast.model.inputs.WEATHER_INPUT_CHANNELS` order and
        RAW C1 units, exactly as C5 delivers it.

        ``latent`` is ONE draw of ``z_t`` per leading batch entry. It is added to
        ``log alpha`` (a global hazard multiplier) and folded into the effective
        wind, so it shifts every offset and every cell of the step TOGETHER —
        which is the whole content of "correlated innovations". ``None``
        reproduces the deterministic path exactly.
        """
        u10 = weather_step[..., weather_index("wind_u10"), :, :]
        v10 = weather_step[..., weather_index("wind_v10"), :, :]
        fm = weather_step[..., weather_index("fuel_moisture_proxy"), :, :]

        speed, unit_e, unit_n = self._effective_wind(u10, v10, fields, latent)
        head = self._reach_cells_per_hour(speed, fields, fm)
        lb = self._length_to_breadth(speed)
        ecc = torch.sqrt(torch.clamp(1.0 - 1.0 / (lb * lb), 0.0, 1.0 - 1e-12))

        # [..., K, H, W] by broadcasting the offset axis in at position -3.
        cos_theta = unit_e.unsqueeze(-3) * self.offset_east.reshape(-1, 1, 1) + unit_n.unsqueeze(
            -3
        ) * self.offset_north.reshape(-1, 1, 1)
        factor = (1.0 - ecc.unsqueeze(-3)) / (1.0 - ecc.unsqueeze(-3) * cos_theta)
        reach = torch.exp(self.log_gamma) * head.unsqueeze(-3) * factor
        dist = self.offset_dist.reshape(-1, 1, 1)
        radial = -0.5 * (dist / torch.clamp(reach, min=1e-9)) ** 2
        log_w = self.log_alpha + self.offset_log_weight.reshape(-1, 1, 1) + radial
        if self.config.susceptibility_mode == "amplitude":
            log_w = log_w + self.log_susceptibility(fields).unsqueeze(-3)
        if latent is not None:
            log_w = log_w + _broadcast_to_field(latent.log_intensity, 3)
        if spatial_log_intensity is not None:
            # [M7] The LOW-RANK SPATIAL modulation. It enters at exactly the same
            # place as the global `log_intensity` — an additive term on the log
            # weight — because it is the SAME physical quantity with one extra
            # degree of freedom: WHERE the burst is. The offset axis is inserted
            # at -3, so the field is shared by every offset of a cell, which is
            # what makes it a modulation of the hazard rather than of the shape.
            log_w = log_w + spatial_log_intensity.unsqueeze(-3)
        return log_w

    def step_probability(
        self,
        burned: Tensor,
        weather_step: Tensor,
        fields: StaticFields,
        latent: LatentEffect | None = None,
    ) -> Tensor:
        """One-hour ignition probability for every cell. ``[..., H, W]``.

        ``burned`` is the mean field ``b_{t-1}`` in ``[0, 1]``, NOT a sampled
        state — the contagion source is the whole burned region's frontier, per
        C1.1's note that state 1 is legitimately empty in 6-37% of frames.

        Given ``latent``, the returned field is ``p(x | x_t, z_t)``: the
        Bernoulli parameter each pixel is drawn with, conditionally independent
        of every other pixel ONLY because ``z_t`` has already been fixed.
        """
        log_w = self.log_weights(
            weather_step, fields, latent, spatial_log_intensity_field(burned, latent)
        )
        neighbours = torch.stack(
            [
                _shift(burned, int(self.offset_dr[k]), int(self.offset_dc[k]))
                for k in range(self.n_offsets)
            ],
            dim=-3,
        )
        lam = (torch.exp(log_w) * neighbours).sum(dim=-3)
        return -torch.expm1(-lam)  # 1 - exp(-lambda), accurate for small lambda

    def rollout(
        self,
        burned0: Tensor,
        weather: Tensor,
        fields: StaticFields,
        horizon_h: int,
        latents: Sequence[LatentEffect | None] | None = None,
    ) -> Tensor:
        """Mean-field pushforward. Returns ``b_k`` for ``k = 1..horizon_h``.

        ``latents[k]`` is the draw of ``z_{t+k+1}`` for step ``k`` — ONE PER
        STEP, per CLAUDE.md, not one per window. Passing ``None`` (or a shorter
        sequence padded with ``None``) runs the deterministic path for that step.

        ``weather`` is ``[..., T, C_w, H, W]`` with ``weather[k]`` driving step
        ``k`` (the C1.3 end-of-hour phase, applied by
        :func:`~wildfire_nowcast.model.inputs.forecast_inputs`).

        This is a FREE-RUNNING rollout: step 2 is driven by the model's own step
        1, not by the truth. Lead 1 is therefore the one-step term and leads
        2..H are the multi-step pushforward, from one trajectory — which is what
        CLAUDE.md's "one-step NLL + multi-step pushforward" asks for, and it
        keeps the two terms from being computed under different conditioning.
        """
        b = burned0
        out = []
        for k in range(int(horizon_h)):
            z_k = None if latents is None or k >= len(latents) else latents[k]
            p = self.step_probability(b, weather[..., k, :, :, :], fields, z_k)
            b = b + (1.0 - b) * p
            out.append(b)
        return torch.stack(out, dim=-3)

    # -- numpy-facing helpers ---------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        horizon_h: int,
        *,
        n_latent_samples: int = 0,
        seed: int = 0,
    ) -> np.ndarray:
        """``float64[horizon_h,H,W]`` P(burned by lead k). The M -> inf forecast.

        Reported alongside the sampled C5 forecast because an M-member ensemble
        pays a Monte-Carlo penalty of ``E[p(1-p)]/M`` on its Brier score, which
        at M = 24 in the growth band is a real fraction of the score being
        compared. Every model in a comparison must therefore use the SAME M —
        and this function says what the sampling cost was.

        ``n_latent_samples > 0`` returns the MARGINAL ``E_z[b_k]`` by averaging
        that many prior draws. With a latent present the ``z = 0`` field is NOT
        the marginal — ``p`` is concave in the hazard and the hazard is
        log-normal in ``z`` — so a growth calibration or a Brier computed at
        ``z = 0`` is a different quantity from the one the ensemble realises.
        Defaulting to 0 keeps the deterministic reading available and makes the
        choice explicit at every call site.
        """
        fields = static_fields_from_array(static)
        b0 = torch.as_tensor((np.asarray(x0) > UNBURNED).astype(np.float64), dtype=DTYPE)
        w = torch.as_tensor(np.asarray(weather, dtype=np.float64), dtype=DTYPE)
        horizon_h = int(horizon_h)
        if n_latent_samples <= 0 or self.latent is None:
            return self.rollout(b0, w, fields, horizon_h).numpy()
        gen = torch.Generator().manual_seed(int(seed))
        total = torch.zeros((horizon_h, *b0.shape), dtype=DTYPE)
        for _ in range(int(n_latent_samples)):
            latents = self.sampler.draw_path(
                self.latent,
                (),
                horizon_h,
                generator=gen,
                covariates=(
                    [step_covariates(w[k]) for k in range(horizon_h)]
                    if self.latent.prior_net is not None
                    else None
                ),
            )
            total = total + self.rollout(b0, w, fields, horizon_h, latents)
        return (total / float(n_latent_samples)).numpy()

    def with_sampler(self, mode: str) -> ContagionKernel:
        """A VIEW of this model whose ensemble is drawn under ``mode``.

        Shares parameters (no copy, no re-training), so the ABLATION and the
        model are provably the same fit — the only difference is whether ``z_t``
        is drawn or held at its prior mean. Any other construction of the
        ablation would confound the sampler with the parameters.
        """
        view = object.__new__(type(self))
        view.__dict__ = dict(self.__dict__)
        view._parameters = self._parameters
        view._buffers = self._buffers
        view._modules = self._modules
        view.sampler = LatentSampler(mode)
        view.name = f"{self.name}__{mode}"
        return view

    def to_spec(self) -> dict[str, Any]:
        has_z = self.latent is not None
        return {
            "kind": self.kind,
            "name": self.name,
            "config": self.config.to_dict(),
            "latent_config": (None if self.latent is None else self.latent.config.to_dict()),
            "sampler_mode": self.sampler.mode,
            "latent_report": latent_report(self.latent),
            "components": {
                "contagion": "anisotropic CNN, elliptical-Gaussian init, wind/slope stretched",
                "spot": "ABSENT (P3)",
                "latent_z_t": (
                    f"PRESENT (M5/P2): {self.latent.dim}-d shared per-step latent "
                    f"{list(self.latent.config.components)}, N(0,I) prior, learned sigma, "
                    "CVAE-style ELBO"
                    if has_z
                    else "ABSENT — this model is deterministic (the G2 kernel)"
                ),
                "ensemble": (
                    self.sampler.describe()
                    if has_z
                    else (
                        "independent-per-pixel Bernoulli given x_t, with NO latent to be "
                        "conditioned on. This IS the known-broken G3 ablation; do not read a "
                        "dispersion number off it as if it were the model's ensemble."
                    )
                ),
                "susceptibility": (
                    f"mode={self.config.susceptibility_mode}. 'amplitude' is the ADR-015 (6a) "
                    "fix; 'reach' is M2's defect (zero gradient on barrier + non-burnable) "
                    "and is a control only."
                ),
            },
            "provenance": dict(self.provenance),
            "ellipse_init": self.ellipse_init.to_dict(),
            "parameters": {
                k: (v.detach().tolist() if v.dim() else float(v.detach()))
                for k, v in self.named_parameters()
            },
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> ContagionKernel:
        raw = dict(spec.get("config", {}))
        # A spec written before ADR-015 has no susceptibility_mode and was fitted
        # under the DEFECT. Defaulting it to the new value would silently change
        # what a saved M2 checkpoint means, so absence maps to "reach".
        raw.setdefault("susceptibility_mode", "reach")
        cfg = KernelConfig(**raw)
        # A spec written before M5 carries no `latent_config`, and absence must
        # map to NO LATENT — never to the current default — or every archived G2
        # checkpoint would silently become a different model when reloaded.
        raw_latent = spec.get("latent_config")
        latent_cfg = None
        if isinstance(raw_latent, dict):
            latent_cfg = LatentConfig(
                dim=int(raw_latent.get("dim", 0)),
                init_sigma=tuple(raw_latent.get("init_sigma", LatentConfig().init_sigma)),
                max_sigma=tuple(raw_latent.get("max_sigma", LatentConfig().max_sigma)),
                encoder_channels=int(raw_latent.get("encoder_channels", 8)),
                free_bits=float(raw_latent.get("free_bits", 0.0)),
                gate_prior_mean=float(raw_latent.get("gate_prior_mean", 0.0)),
                conditional_prior=bool(raw_latent.get("conditional_prior", False)),
                # [M6] Absent keys must reload as the M5 defaults, so an archived
                # M5 checkpoint stays the model it was — same rule as the pre-M5
                # `latent_config` absence above.
                mean_preserving=bool(raw_latent.get("mean_preserving", False)),
                # [M8] Same rule as every flag before it: ABSENCE IS FALSE, so
                # every archived M5/M6/M7 checkpoint reloads as the model it was
                # fitted as rather than acquiring a correction it never had.
                gate_mean_preserving=bool(raw_latent.get("gate_mean_preserving", False)),
                rho=float(raw_latent.get("rho", 0.0)),
                # [M7] Same rule again: absence is 0 (no spatial modes), so every
                # archived M5/M6 checkpoint reloads as the global-scalar model it
                # was fitted as.
                spatial_modes=int(raw_latent.get("spatial_modes", 0)),
                spatial_init_sigma=tuple(
                    raw_latent.get("spatial_init_sigma") or LatentConfig().spatial_init_sigma
                ),
                spatial_max_sigma=tuple(
                    raw_latent.get("spatial_max_sigma") or LatentConfig().spatial_max_sigma
                ),
                innovation_channels=bool(raw_latent.get("innovation_channels", False)),
                spatial_encoder_pooling=bool(raw_latent.get("spatial_encoder_pooling", True)),
            )
        model = cls(
            cfg,
            name=str(spec.get("name", "kernel")),
            ellipse_params=EllipseParams.from_dict(spec.get("ellipse_init", {})),
            latent_config=latent_cfg,
            sampler=LatentSampler(str(spec.get("sampler_mode", "latent"))),
        )
        model.provenance = dict(spec.get("provenance", {}))
        with torch.no_grad():
            for key, value in dict(spec.get("parameters", {})).items():
                param = dict(model.named_parameters())[key]
                param.copy_(torch.as_tensor(value, dtype=DTYPE).reshape(param.shape))
        return model

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        spec_path = out / "model.json"
        spec_path.write_text(json.dumps(self.to_spec(), indent=2) + "\n")
        return spec_path

    @classmethod
    def load(cls, path: str | Path) -> ContagionKernel:
        p = Path(path)
        spec_path = p / "model.json" if p.is_dir() else p
        model = cls.from_spec(json.loads(spec_path.read_text()))
        # Remembered so the C8 check can fall back to the run directory's
        # training.json for checkpoints written before the stamp existed.
        model._loaded_from = str(spec_path)
        return model

    # -- C5 ----------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        """C5 ``predict``. Monte-Carlo ensemble from the transition kernel.

        With a latent (``self.latent is not None`` and ``sampler.mode ==
        "latent"``) this is CLAUDE.md's model: **one ``z_t`` per member per
        step**, shared by every pixel of that member-step, and pixels drawn
        conditionally independently GIVEN ``(x_t, z_t)``. A member is therefore a
        coherent scenario, and the state it feeds back is its own.

        Two ways to get the KNOWN-BROKEN independent-per-pixel ensemble, and both
        are ablations rather than candidates:

        * ``self.with_sampler("independent")`` — same fit, ``z_t`` held at the
          prior mean. This is the CONTROLLED ablation: one switch, one model.
        * ``latent_config=None`` — no latent at all, i.e. the deterministic G2
          kernel, whose only randomness is the per-pixel draw.

        Dispersion metrics on either are meaningful ONLY as the expected FAILURE
        (C6.1/ADR-011); accuracy metrics (Brier, IoU, arrival CRPS) remain
        meaningful on both.

        Seed-exact: every draw, latent and pixel, comes from one seeded
        generator, so two calls with the same ``seed`` are bitwise identical.
        """
        validate_predict_inputs(x0, static, weather, n_members, horizon_h, seed)
        n_members, horizon_h = int(n_members), int(horizon_h)
        state0 = np.asarray(x0, dtype=np.uint8)
        fields = static_fields_from_array(static)
        w = torch.as_tensor(np.asarray(weather, dtype=np.float64), dtype=DTYPE)
        gen = torch.Generator().manual_seed(int(seed))

        burned = torch.as_tensor(
            np.repeat((state0 > UNBURNED)[None], n_members, axis=0).astype(np.float64), dtype=DTYPE
        )
        ignited_at = np.full((n_members, *state0.shape), -1, dtype=np.int64)
        state = np.repeat(state0[None], n_members, axis=0)
        out = np.empty((n_members, horizon_h, *state0.shape), dtype=np.uint8)

        # [M6] ONE AR(1) path per ensemble, stepped in place so the latent draw
        # and the per-pixel draw keep consuming the SAME generator in the SAME
        # order as M5. At rho = 0 this is bitwise M5 (see `LatentPath`).
        path = self.sampler.path(self.latent)
        for k in range(horizon_h):
            # ONE draw per member for this step. Shape [M] -> broadcast to
            # [M, H, W]: constant across cells by construction, which is the
            # difference between a shared latent and a noise field.
            cov = (
                step_covariates(w[k])
                if self.latent is not None and self.latent.prior_net is not None
                else None
            )
            z_k = path.step((n_members,), generator=gen, covariates=cov)
            p = self.step_probability(burned, w[k], fields, z_k)
            draw = torch.rand(p.shape, generator=gen, dtype=DTYPE)
            newly = ((draw < p) & (burned < 0.5)).numpy()
            burned = torch.clamp(burned + torch.as_tensor(newly.astype(np.float64)), max=1.0)
            state[newly] = BURNING
            ignited_at[newly] = k
            if self.config.burnout_hours > 0:
                spent = (ignited_at >= 0) & (k - ignited_at >= self.config.burnout_hours)
                state[spent] = BURNED_OUT
            out[:, k] = state
        return out


# --------------------------------------------------------------------------
# the anti-drift check: torch physics == numpy physics
# --------------------------------------------------------------------------


def check_torch_matches_numpy(seed: int = 0, tolerance: float = 1e-9) -> dict[str, float]:
    """Assert the torch physics reproduces :mod:`spread`'s numpy physics.

    C0's rule is one implementation per adjudicated quantity. The forward pass
    has to be differentiable and :mod:`spread` is numpy, so this module is a
    SECOND implementation of the same physics — the exact situation C0 exists to
    prevent. It is admitted here and then pinned by a test rather than trusted:
    if the two ever disagree, "the kernel beat the ellipse" stops meaning what
    it says.
    """
    from wildfire_nowcast.model.spread import (
        ellipse_ros_factor,
        length_to_breadth,
        moisture_damping,
    )

    rng = np.random.default_rng(seed)
    speed = rng.uniform(0.0, 20.0, (7, 5))
    cos_theta = rng.uniform(-1.0, 1.0, (7, 5))
    fm = rng.uniform(0.0, 40.0, (7, 5))
    mx = rng.choice([15.0, 20.0, 25.0, 30.0], (7, 5))

    model = ContagionKernel()
    t = lambda a: torch.as_tensor(a, dtype=DTYPE)  # noqa: E731
    with torch.no_grad():
        lb_torch = model._length_to_breadth(t(speed)).numpy()
        eta_torch = model._moisture_damping(t(fm), t(mx)).numpy()
    lb_numpy = length_to_breadth(speed, cap=model.config.lb_cap)
    eta_numpy = moisture_damping(fm, mx)

    ecc = np.sqrt(np.clip(1.0 - 1.0 / lb_numpy**2, 0.0, 1.0 - 1e-12))
    factor_torch = ((1.0 - ecc) / (1.0 - ecc * cos_theta)).astype(np.float64)
    factor_numpy = ellipse_ros_factor(lb_numpy, cos_theta)

    errors = {
        "length_to_breadth": float(np.max(np.abs(lb_torch - lb_numpy))),
        "moisture_damping": float(np.max(np.abs(eta_torch - eta_numpy))),
        "ellipse_ros_factor": float(np.max(np.abs(factor_torch - factor_numpy))),
    }
    bad = {k: v for k, v in errors.items() if not (v <= tolerance)}
    if bad:
        raise AssertionError(
            f"torch physics has DRIFTED from spread.py (tolerance {tolerance}): {bad}. "
            "Two implementations of one physics is the C0 failure mode; fix the code, "
            "do NOT widen the tolerance."
        )
    return errors


def check_latent_off_is_bit_identical(seed: int = 7) -> dict[str, Any]:
    """[M5] A latent-free kernel and a ``z=0``-forced latent kernel must agree EXACTLY.

    Two separate guarantees, both bitwise, and both are about not silently
    redefining a result that already exists:

    1. ``latent_config=None`` must reproduce the pre-M5 forward pass. The G2
       record was produced by that code path; if adding an optional argument
       changed it by one ULP, every archived number would become
       un-reproducible and the change would be invisible.
    2. A latent model sampled at ``z = 0`` (the ABLATION) must equal the
       deterministic path. That is what makes the ablation a clean control: it
       differs from the model in the DRAW, and in nothing else.

    ``max |delta|`` is returned rather than a boolean so a future drift is
    reported as a size, not as a failed assertion with no magnitude.
    """
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    shape = (14, 18)
    fields = _synthetic_fields(shape, rng)
    weather = torch.as_tensor(rng.normal(size=(3, 5, *shape)) * 2.0 + 3.0, dtype=DTYPE)
    b0 = torch.as_tensor((rng.random(shape) < 0.15).astype(np.float64), dtype=DTYPE)

    plain = ContagionKernel(KernelConfig(), name="plain")
    with_z = ContagionKernel(KernelConfig(), name="with_z", latent_config=LatentConfig(dim=3))
    with torch.no_grad():
        for (name, dst), (_, src) in zip(
            with_z.named_parameters(), plain.named_parameters(), strict=False
        ):
            if name.startswith("latent."):
                continue
            dst.copy_(src)

    ref = plain.rollout(b0, weather, fields, 3)
    zero_effect = LatentSampler("independent").draw(with_z.latent, ())
    abl = with_z.rollout(b0, weather, fields, 3, [zero_effect] * 3)
    drawn = with_z.rollout(
        b0,
        weather,
        fields,
        3,
        [LatentSampler("latent").draw(with_z.latent, ()) for _ in range(3)],
    )

    delta_none = float(
        torch.max(torch.abs(ref - plain.rollout(b0, weather, fields, 3, None))).detach()
    )
    delta_ablation = float(torch.max(torch.abs(ref - abl)).detach())
    delta_drawn = float(torch.max(torch.abs(ref - drawn)).detach())
    if delta_none != 0.0 or delta_ablation != 0.0:
        raise AssertionError(
            "M5 latent changed the deterministic path: "
            f"latents=None delta {delta_none}, z=0 delta {delta_ablation}. Both MUST be "
            "exactly 0 or the G2 record is no longer reproducible by this code."
        )
    if delta_drawn == 0.0:
        raise AssertionError(
            "a DRAWN z_t left the field bit-identical to z=0, so the latent is not wired "
            "into the physics at all. A latent that changes nothing would pass every "
            "identity check and produce a collapsed ensemble that looks like a model."
        )
    return {
        "delta_latents_none": delta_none,
        "delta_z_zero_ablation": delta_ablation,
        "delta_z_drawn": delta_drawn,
    }


def _synthetic_fields(shape: tuple[int, int], rng: Any) -> StaticFields:
    """Plausible :class:`StaticFields` for a self-test scene (no tensor read)."""
    from wildfire_nowcast.model.inputs import N_STATIC

    static = np.zeros((N_STATIC, *shape), dtype=np.float64)
    static[static_index("fuel_model_id")] = 101.0
    static[static_index("canopy_cover")] = 20.0
    static[static_index("slope")] = rng.random(shape) * 10.0
    static[static_index("aspect_sin")] = 0.0
    static[static_index("aspect_cos")] = 1.0
    return static_fields_from_array(static)


def susceptibility_gradient_report(
    mode: str = "amplitude", *, horizon_h: int = 3, seed: int = 0
) -> dict[str, Any]:
    """Measure ``|d loss / d theta|`` for the two susceptibility parameters.

    The scene is deliberately one where the answer is known a priori: a lit
    patch, a strong east wind, a full column of barrier cells and a full column
    of non-burnable fuel, both directly downwind and inside the kernel's reach.
    Every one of those cells receives contagion, so both parameters MUST be
    identified. Under ``mode="reach"`` both gradients are exactly ``0.0`` —
    which is not a small number, it is the parameter being absent from the
    model. That contrast is the point of returning both.
    """
    from wildfire_nowcast.model.inputs import static_index, weather_index

    shape = (20, 26)
    x0 = np.zeros(shape, np.uint8)
    x0[8:12, 2:5] = 1
    static = np.zeros((8, *shape), np.float32)
    static[static_index("fuel_model_id")] = 102.0  # GR grass everywhere
    static[static_index("fuel_model_id")][:, 7] = 98.0  # NB open water column
    static[static_index("water_barrier_mask")][:, 6] = 1.0  # barrier column
    # One column per remaining fuel group, all inside the fire's reach, so a zero
    # gradient here means UNIDENTIFIED and never merely ABSENT FROM THE SCENE.
    for column, code in ((8, 121.0), (9, 141.0), (10, 161.0), (11, 181.0), (12, 201.0)):
        static[static_index("fuel_model_id")][:, column] = code
    weather = np.zeros((horizon_h, 5, *shape), np.float32)
    weather[:, weather_index("wind_u10")] = 12.0
    weather[:, weather_index("fuel_moisture_proxy")] = 4.0

    model = ContagionKernel(KernelConfig(susceptibility_mode=mode))
    fields = static_fields_from_array(static)
    b0 = torch.as_tensor((x0 > UNBURNED).astype(np.float64), dtype=DTYPE)
    w = torch.as_tensor(weather.astype(np.float64), dtype=DTYPE)
    # A truth that WANTS the fire across both columns: if the gradient exists at
    # all, this is the configuration that produces it.
    y = torch.as_tensor(
        (b0.numpy() > 0) | (np.broadcast_to(np.arange(shape[1])[None, :], shape) <= 14),
        dtype=DTYPE,
    ).expand(horizon_h, *shape)

    b = model.rollout(b0, w, fields, horizon_h)
    p = torch.clamp(b, 1e-7, 1.0 - 1e-7)
    loss = -(y * torch.log(p) + (1.0 - y) * torch.log1p(-p)).mean()
    loss.backward()

    def _grad(param: Tensor, index: int | None = None) -> float:
        # A parameter that receives no gradient at all has `grad is None`, which
        # is the SAME finding as a gradient of exactly 0 and must not crash.
        if param.grad is None:
            return 0.0
        return float(param.grad if index is None else param.grad[index])

    barrier_grad = _grad(model.barrier_log_multiplier)
    fuel_grads = {
        name: _grad(model.fuel_log_multiplier, i) for i, name in enumerate(_GROUP_ORDER)
    }
    return {
        "susceptibility_mode": mode,
        "d_loss_d_barrier_log_multiplier": barrier_grad,
        "d_loss_d_fuel_log_multiplier": fuel_grads,
        "barrier_gradient_is_zero": barrier_grad == 0.0,
        "nonburnable_gradient_is_zero": fuel_grads["NB"] == 0.0,
        "burnable_gradients_all_nonzero": all(
            fuel_grads[g] != 0.0 for g in _GROUP_ORDER if g != "NB" and fuel_grads[g] is not None
        ),
        "n_barrier_cells": int(static[static_index("water_barrier_mask")].sum()),
        "loss": float(loss.detach()),
        "nonburnable_second_mechanism": (
            "NB's gradient stays EXACTLY zero in BOTH modes, and it is NOT the ADR-015 (6a) "
            "defect. FUEL_GROUPS['NB'] carries moisture_of_extinction = 1.0%, so for any "
            "realistic dead fuel moisture the Simard ratio clamps to 1 and the damping "
            "polynomial 1 - 2.59r + 5.11r^2 - 3.52r^3 evaluates to 0.0 EXACTLY. The reach is "
            "then 0, the radial term is -inf, and the weight is a hard zero however the "
            "amplitude is parameterised. Found by TESTING the fix rather than assuming it. "
            "Not repaired here: moisture_of_extinction lives in spread.FUEL_GROUPS, which is "
            "also the ellipse baseline's physics, and changing the opponent mid-gate is not "
            "allowed. The BARRIER mask (C1 ch 12) is the crossing mechanism P3/G4 needs and "
            "it IS fixed; NB is water/rock/urban, where zero gradient encodes a correct prior."
        ),
    }


def offset_anisotropy(model: ContagionKernel) -> dict[str, Any]:
    """The WIND-INDEPENDENT directional preference of the learned offset weights.

    ADR-015 (6b): M2's free ``c_d`` grew a preference for travel to the S/SW that
    does not depend on wind at all, which is what GOFER's measured systematic
    centroid bias would look like if the model absorbed it as physics. Naming the
    symptom is not enough to test whether an intervention removed it, so this
    reduces ``c_d`` to a single vector.

    Within each distance ring the weights ``exp(c_d)`` are normalised to sum to
    1 and combined with the offsets' unit travel vectors. **An isotropic ring
    contributes exactly zero**, so a pure distance profile — which is what a
    kernel with no directional preference looks like — scores 0 regardless of how
    steeply it decays. Rings are then averaged by their offset count.

    ``magnitude`` is in [0, 1]. ``bearing_deg`` is a compass bearing (0 = travel
    toward N, 90 = toward E) of the preferred travel direction.
    """
    with torch.no_grad():
        w = torch.exp(model.offset_log_weight).numpy()
        dist = model.offset_dist.numpy()
        east = model.offset_east.numpy()
        north = model.offset_north.numpy()

    rings: list[dict[str, Any]] = []
    total_e = total_n = 0.0
    total_n_off = 0
    for d in sorted({round(float(v), 6) for v in dist}):
        sel = np.isclose(dist, d)
        weights = w[sel]
        weights = weights / max(weights.sum(), 1e-300)
        e = float((weights * east[sel]).sum())
        n = float((weights * north[sel]).sum())
        k = int(sel.sum())
        rings.append(
            {
                "distance_cells": d,
                "n_offsets": k,
                "east": e,
                "north": n,
                "magnitude": float(math.hypot(e, n)),
            }
        )
        total_e += e * k
        total_n += n * k
        total_n_off += k
    e_bar = total_e / max(total_n_off, 1)
    n_bar = total_n / max(total_n_off, 1)
    magnitude = float(math.hypot(e_bar, n_bar))
    bearing = float((math.degrees(math.atan2(e_bar, n_bar)) + 360.0) % 360.0)
    return {
        "magnitude": magnitude,
        "bearing_deg": bearing if magnitude > 1e-9 else None,
        "east": e_bar,
        "north": n_bar,
        "max_abs_log_weight": (
            float(np.abs(np.log(np.clip(w, 1e-300, None))).max()) if w.size else 0.0
        ),
        "per_ring": rings,
        "note": (
            "wind-INDEPENDENT preference of the free offset weights. 0 = isotropic. "
            "Compass bearing: 180 = S, 225 = SW."
        ),
    }


def parameter_report(model: ContagionKernel) -> dict[str, Any]:
    """Physical values of the learned parameters, next to their initialisation.

    Reported for every run: "the loss went down" is not a finding, "the wind
    exponent moved from 1.8 to 2.4 and the upslope coefficient collapsed" is.
    """
    base = model.ellipse_init
    with torch.no_grad():
        learned = {
            "r0_ms": float(torch.exp(model.log_r0)),
            "u_ref_ms": float(torch.exp(model.log_u_ref)),
            "wind_exponent": float(torch.exp(model.log_wind_exponent)),
            "k_slope": float(torch.exp(model.log_k_slope)),
            "waf_open": float(torch.sigmoid(model.logit_waf_open)),
            "waf_closed": float(torch.sigmoid(model.logit_waf_closed)),
            "lb_gain": float(torch.exp(model.log_lb_gain)),
            "moisture_gain": float(torch.exp(model.log_moisture_gain)),
            "alpha": float(torch.exp(model.log_alpha)),
            "gamma": float(torch.exp(model.log_gamma)),
            "barrier_multiplier": float(torch.exp(model.barrier_log_multiplier)),
            "fuel_multipliers": {
                name: float(torch.exp(model.fuel_log_multiplier[i]))
                for i, name in enumerate(_GROUP_ORDER)
            },
            "offset_log_weight_absmax": float(model.offset_log_weight.abs().max()),
        }
    init = {
        "r0_ms": base.r0_ms,
        "u_ref_ms": base.u_ref_ms,
        "wind_exponent": base.wind_exponent,
        "k_slope": base.k_slope,
        "waf_open": base.waf_open,
        "waf_closed": base.waf_closed,
        "lb_gain": 1.0,
        "moisture_gain": 1.0,
        "gamma": 1.0,
        "fuel_multipliers": {n: FUEL_GROUPS[n][0] for n in _GROUP_ORDER},
        "barrier_multiplier": math.exp(model.config.barrier_log_multiplier),
    }
    return {
        "learned": learned,
        "init": init,
        "susceptibility_mode": model.config.susceptibility_mode,
        "offset_anisotropy": offset_anisotropy(model),
    }


def offset_kernel_table(model: ContagionKernel) -> list[dict[str, float]]:
    """The learned per-offset corrections, as an inspectable table.

    This is the CNN part of the kernel and the only part with no physical prior,
    so it is the first place to look when the model is doing something the
    physics cannot explain.
    """
    with torch.no_grad():
        return [
            {
                "dr": int(model.offset_dr[k]),
                "dc": int(model.offset_dc[k]),
                "distance_cells": float(model.offset_dist[k]),
                "log_weight": float(model.offset_log_weight[k]),
            }
            for k in range(model.n_offsets)
        ]
