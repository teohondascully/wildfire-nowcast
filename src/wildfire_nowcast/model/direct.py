"""[M10] The DIRECT-HORIZON head - arm B of ADR-045's direct-vs-rollout experiment.

THE QUESTION THIS ARM EXISTS TO ANSWER
--------------------------------------
M9 named the defect (ADR-043): the kernel **cannot decelerate** - the elasticity
of log growth rate to log frontier length is ``-0.781 +/- 0.082`` for truth and
``-0.036 +/- 0.026`` for the model - and it did NOT establish WHERE the defect
enters. Two candidates:

    (a) the 1-step kernel is mis-specified and rolling it 3x COMPOUNDS the error;
    (b) the defect is already in the 1-step marginal and compounding is incidental.

If (a), a head that predicts the 3 h marginal in ONE SHOT fixes it and no
architecture change is warranted. This class is that head. It is the CHEAP
experiment that can retire the expensive one, so it runs first.

WHAT "DIRECT" MEANS HERE, PRECISELY
-----------------------------------
Every lead's hazard is computed from ``b_0`` - the OBSERVED state at ``t0`` - and
the model's own predicted state never re-enters the hazard. The only sequential
element left is absorbing bookkeeping: a cell that burned at lead 1 cannot burn
again at lead 2. That is carried by the INCREMENTAL hazard at fixed ``z``::

    lambda_h(x; z)  = total hazard accumulated from t0 to lead h, from b_0 alone
    dlambda_h(x; z) = lambda_h(x; z) - lambda_{h-1}(x; z)        >= 0
    p_h(x)          = 1 - exp(-dlambda_h)                        # conditional

Both terms of the difference use the SAME draw ``z_h``, so the increment is
non-negative by construction (``lambda_h`` grows in ``h`` through three
monotone routes: a larger amplitude, a longer reach and a strictly larger
stencil). No error compounds through the state; the arithmetic is monotone.

THE THREE CONSTRAINTS ADR-045 (3) IMPOSES, AND WHY THE OBVIOUS ARMS FAIL THEM
-----------------------------------------------------------------------------
B must be the "same architecture family, same receptive-field budget, parameter
count within +/-10% of A". Arm A's per-lead receptive field is ``3h`` cells (a
radius-3 stencil rolled out ``h`` times).

* A one-shot **radius-9** stencil matches the reach and costs **253 offset
  weights against A's 28**. B would win on capacity, which the pre-registration
  forbids outright.
* A pure **dilation-3** stencil keeps 28 weights and reaches 9 cells, but its
  nearest offset is 3 cells away, so it **cannot ignite a cell adjacent to a thin
  fire at all**. That is not a fair challenger, it is a crippled one, and a
  crippled B answers nothing about compounding.

So B uses a **multi-scale stencil with TIED weights**: at lead ``h`` the offsets
are the union of dilations ``1..h`` of A's radius-3 stencil, and the free weight
``c_d`` is SHARED across the dilations of one base offset. Then

    receptive field  3h  - identical to A's at every lead
    free offset weights  28  - identical to A's
    shifted-field evaluations at lead 3: 84 = A's 3 x 28 - matched

and the near shell is present at every lead, so a cell adjacent to the fire keeps
accumulating hazard for the whole window, which is the thing dilation alone gets
wrong.

THE TWO PARAMETERS B HAS AND A DOES NOT - DECLARED, NOT BURIED
--------------------------------------------------------------
:attr:`log_lead_reach_exponent` (``p``) and :attr:`log_lead_hazard_exponent`
(``q``): reach scales as ``h^p`` and hazard amplitude as ``h^q``, with ``p = q =
1`` at initialisation, i.e. plain linear-in-lead extrapolation. They are B's own
HORIZON RESPONSE - the degree of freedom by which a direct head could in
principle decline to extrapolate linearly - and they are the only capacity B has
that A does not. Two parameters against A's ~1,000; the measured ratio is
reported in the M10 artifact by :func:`parameter_counts` and **if it ever falls
outside +/-10% the arm is void.**

Note what these two CANNOT do, stated before the result: they are GLOBAL scalars,
so they cannot make the rate depend on frontier length. If B's frontier
elasticity moves toward truth it will not be because of them, and that is exactly
what makes ADR-045's P1 a real prediction rather than a foregone one.

WHAT IS DELIBERATELY NOT DIFFERENT
----------------------------------
The physics, the latent, the sampler, the absorbing bookkeeping and the RNG
consumption order are all A's - inherited, not restated. ``predict`` and
``rollout`` are NOT overridden: the base class routes both through
``step_probability_at``, and this class overrides only that. M10 compares a
rollout against a direct head, so any difference produced by *two copies of the
sampling loop* would be reported as a finding about horizons when it was a
finding about my code.

COST, STATED: computing ``dlambda_h`` needs both ``lambda_h`` and
``lambda_{h-1}``, and the ``h`` in ``h^p`` sits inside the Gaussian exponent so
the two do not share terms. Arm B therefore costs ~3x arm A per window at
``horizon_h = 3``. A shifted-neighbour cache would remove most of it and is NOT
implemented: a cache keyed on a tensor's identity is exactly the kind of
soundness risk this project keeps paying for, and the experiment is cheap enough
to buy correctness instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from wildfire_nowcast.model.kernel import (
    DTYPE,
    ContagionKernel,
    KernelConfig,
    StaticFields,
    shift_field,
    spatial_log_intensity_field,
)
from wildfire_nowcast.model.latent import LatentConfig, LatentEffect, LatentSampler
from wildfire_nowcast.model.spread import EllipseParams

__all__ = [
    "DirectConfig",
    "DirectHorizonKernel",
    "parameter_counts",
    "compare_parameter_counts",
    "check_direct_head_properties",
    "PARAMETER_PARITY_TOLERANCE",
]

#: ADR-045 (3): "parameter count within +/-10% of A". Not a threshold I chose -
#: it is the pre-registration's own number, restated in code so the check cannot
#: drift from the clause. C-3 has nothing to bind to: no sample was consulted.
PARAMETER_PARITY_TOLERANCE = 0.10


@dataclass(frozen=True)
class DirectConfig:
    """Structure of the direct head. Nothing here is fitted."""

    #: Largest dilation the multi-scale stencil ever uses. At lead ``h`` the
    #: dilations are ``1..min(h, max_dilation)``, so with the default and
    #: ``horizon_h = 3`` the lead-3 receptive field is 9 cells - A's rollout
    #: receptive field, exactly.
    max_dilation: int = 3
    #: The two lead-response parameters. ``False`` pins ``p = q = 1`` (linear in
    #: lead) and makes B's parameter count EXACTLY A's; retained so "B won on its
    #: two extra parameters" is answerable by a run rather than by argument.
    lead_response: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"max_dilation": int(self.max_dilation), "lead_response": bool(self.lead_response)}


class DirectHorizonKernel(ContagionKernel):
    """Arm B: the lead-``h`` marginal predicted in ONE SHOT from ``x_t``.

    Everything except :meth:`step_probability_at` is arm A's, inherited.
    """

    kind = "direct_horizon_kernel"

    #: Read by ``train._elbo_terms`` so the inference network anchors its spatial
    #: basis on the SAME field the decoder does. For a rollout that is the state
    #: each step reached; for a direct head it is the window's origin. A mismatch
    #: would not crash - it would quietly make the spatial latent dimensions
    #: describe a different fire from the one they modulate.
    anchors_on_origin = True

    def __init__(
        self,
        config: KernelConfig | None = None,
        *,
        name: str = "direct_kernel",
        ellipse_params: EllipseParams | None = None,
        latent_config: LatentConfig | None = None,
        sampler: LatentSampler | None = None,
        direct_config: DirectConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            name=name,
            ellipse_params=ellipse_params,
            latent_config=latent_config,
            sampler=sampler,
        )
        self.direct = direct_config or DirectConfig()
        # exp(0) = 1: at initialisation B extrapolates LINEARLY in the lead, which
        # is the null hypothesis about horizons, not a tuned starting point.
        self.log_lead_reach_exponent = torch.nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.log_lead_hazard_exponent = torch.nn.Parameter(torch.zeros((), dtype=DTYPE))
        if not self.direct.lead_response:
            self.log_lead_reach_exponent.requires_grad_(False)
            self.log_lead_hazard_exponent.requires_grad_(False)

    # -- the one thing that differs from arm A ----------------------------

    def lead_hazard(
        self,
        burned0: Tensor,
        weather: Tensor,
        fields: StaticFields,
        lead: int,
        latent: LatentEffect | None = None,
    ) -> Tensor:
        """``lambda_h`` - hazard accumulated from ``t0`` to lead ``h``, from ``b_0`` alone.

        ``weather`` is ``[..., T, C_w, H, W]``; the head sees the MEAN of the
        first ``h`` forecast hours, so it is driven by the same forcing over the
        same window as a rollout of A, aggregated rather than sequenced.
        """
        lead = int(lead)
        if lead <= 0:
            return torch.zeros_like(burned0)
        h = float(lead)
        weather_mean = weather[..., :lead, :, :, :].mean(dim=-4)
        spatial = spatial_log_intensity_field(burned0, latent)
        reach_scale = h ** torch.exp(self.log_lead_reach_exponent)
        log_amplitude = torch.exp(self.log_lead_hazard_exponent) * math.log(h)

        lam: Tensor | None = None
        for s in range(1, min(lead, int(self.direct.max_dilation)) + 1):
            log_w = self.log_weights(
                weather_mean,
                fields,
                latent,
                spatial,
                dilation=s,
                reach_scale=reach_scale,
                log_amplitude=log_amplitude,
            )
            neighbours = torch.stack(
                [
                    shift_field(burned0, s * int(self.offset_dr[k]), s * int(self.offset_dc[k]))
                    for k in range(self.n_offsets)
                ],
                dim=-3,
            )
            term = (torch.exp(log_w) * neighbours).sum(dim=-3)
            lam = term if lam is None else lam + term
        assert lam is not None
        return lam

    def step_probability_at(
        self,
        burned: Tensor,
        weather: Tensor,
        fields: StaticFields,
        k: int,
        latent: LatentEffect | None = None,
        burned0: Tensor | None = None,
    ) -> Tensor:
        """Conditional ignition probability for lead ``k+1``, given not yet burned.

        ``burned0`` is the window's origin and is what the hazard is computed
        from; ``burned`` (the state the caller has accumulated) is used by the
        CALLER for the absorbing update and is never fed back into the physics.
        Falling back to ``burned`` when no origin is supplied keeps the object
        usable in a one-step call, where the two coincide.
        """
        origin = burned if burned0 is None else burned0
        lam = self.lead_hazard(origin, weather, fields, k + 1, latent)
        if k > 0:
            lam = lam - self.lead_hazard(origin, weather, fields, k, latent)
        # p = exp(log_lead_*_exponent) > 0 strictly, and the stencil at lead h+1
        # contains the stencil at lead h, so the increment is non-negative by
        # construction and this clamp cannot bind. It is here because "cannot
        # bind" is an argument and a negative probability is a silent disaster;
        # `negative_increment_fraction` measures whether the argument holds.
        return -torch.expm1(-torch.clamp(lam, min=0.0))

    def negative_increment_fraction(
        self,
        burned0: Tensor,
        weather: Tensor,
        fields: StaticFields,
        horizon_h: int,
        latent: LatentEffect | None = None,
    ) -> float:
        """Fraction of cell-leads where the hazard increment came out NEGATIVE.

        Expected to be exactly 0.0. Measured rather than asserted, because "the
        clamp cannot bind" is the kind of claim this project has been wrong about
        (a hard-zero weight that made a parameter unlearnable was defended the
        same way for two milestones).
        """
        bad = total = 0
        for k in range(int(horizon_h)):
            lam = self.lead_hazard(burned0, weather, fields, k + 1, latent)
            if k > 0:
                lam = lam - self.lead_hazard(burned0, weather, fields, k, latent)
            bad += int((lam < 0.0).sum())
            total += int(lam.numel())
        return (bad / total) if total else 0.0

    # -- serialisation -----------------------------------------------------

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        spec["direct"] = self.direct.to_dict()
        spec["components"]["horizon"] = (
            "DIRECT (M10 arm B): every lead's marginal is predicted in one shot from x_t "
            "with a multi-scale tied-weight stencil (dilations 1..h). The model's own "
            "predicted state never re-enters the hazard; only absorbing bookkeeping is "
            "sequential."
        )
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> DirectHorizonKernel:
        model = super().from_spec(spec)
        assert isinstance(model, DirectHorizonKernel)
        raw = dict(spec.get("direct") or {})
        # Absence maps to the DEFAULT here (unlike `latent_config`, where absence
        # had to mean "no latent" to keep archived checkpoints honest) because no
        # checkpoint of this class predates the key: this class was born with it.
        model.direct = DirectConfig(
            max_dilation=int(raw.get("max_dilation", DirectConfig().max_dilation)),
            lead_response=bool(raw.get("lead_response", DirectConfig().lead_response)),
        )
        return model


# --------------------------------------------------------------------------
# the capacity check ADR-045 (3) requires - reported, never assumed
# --------------------------------------------------------------------------


def check_direct_head_properties(seed: int = 11, horizon_h: int = 3) -> dict[str, Any]:
    """Assert the four properties arm B's INTERPRETATION rests on. Returns the numbers.

    Every one of these is a claim the M10 write-up would otherwise be making from
    a docstring, and this project's record on claims-from-docstrings is bad (a
    parameter defended as learnable for two milestones had a gradient of exactly
    zero). Each raises with the measured magnitude rather than returning a bool.

    1. **DIRECTNESS.** The hazard at lead 3 is unchanged when the accumulated
       state handed to it is replaced by garbage. If this fails, B is a rollout
       wearing a direct head's name and M10 compares nothing.
    2. **TELESCOPING.** With no latent, ``b_h == 1 - (1 - b_0) exp(-lambda_h)``
       exactly: the increments compose back into the one-shot marginal, so the
       absorbing bookkeeping has not quietly changed the estimand.
    3. **RECEPTIVE FIELD 3h, WITH A POSITIVE CONTROL.** Nothing beyond ``3h``
       cells may ignite, AND something at more than ``3(h-1)`` cells must - an
       all-clear that has never been observed to find anything is not evidence
       (ADR-044 (3)).
    4. **THE NEAR SHELL KEEPS ACCUMULATING.** A cell adjacent to the fire must be
       strictly more likely at lead 3 than at lead 1. This is the property a
       pure dilation-3 stencil - the obvious construction - fails outright, and
       the reason B's stencil is multi-scale.
    """
    import numpy as np

    from wildfire_nowcast.model.inputs import N_STATIC, static_index, weather_index

    rng = np.random.default_rng(int(seed))
    shape = (41, 41)
    static = np.zeros((N_STATIC, *shape), dtype=np.float64)
    static[static_index("fuel_model_id")] = 102.0  # GR grass everywhere
    static[static_index("canopy_cover")] = 10.0
    static[static_index("aspect_cos")] = 1.0
    weather = np.zeros((horizon_h, 5, *shape), dtype=np.float64)
    weather[:, weather_index("wind_u10")] = 6.0
    weather[:, weather_index("temp_2m")] = 300.0
    weather[:, weather_index("rh_2m")] = 20.0
    weather[:, weather_index("fuel_moisture_proxy")] = 4.0

    x0 = np.zeros(shape, dtype=np.uint8)
    x0[20, 20] = 1  # ONE burning cell: the thinnest possible fire

    from wildfire_nowcast.model.kernel import static_fields_from_array

    fields = static_fields_from_array(static)
    model = DirectHorizonKernel(KernelConfig())
    b0 = torch.as_tensor((x0 > 0).astype(np.float64), dtype=DTYPE)
    w = torch.as_tensor(weather, dtype=DTYPE)

    # 1 - directness
    garbage = torch.as_tensor(rng.random(shape), dtype=DTYPE)
    with torch.no_grad():
        p_clean = model.step_probability_at(b0, w, fields, horizon_h - 1, None, b0)
        p_garbage = model.step_probability_at(garbage, w, fields, horizon_h - 1, None, b0)
    directness_delta = float(torch.max(torch.abs(p_clean - p_garbage)))

    # 2 - telescoping
    with torch.no_grad():
        b = model.rollout(b0, w, fields, horizon_h)
        worst_telescope = 0.0
        for h in range(1, horizon_h + 1):
            lam = model.lead_hazard(b0, w, fields, h, None)
            expected = 1.0 - (1.0 - b0) * torch.exp(-lam)
            worst_telescope = max(
                worst_telescope, float(torch.max(torch.abs(b[..., h - 1, :, :] - expected)))
            )
        monotone_violations = int(
            sum(
                int((b[..., h, :, :] < b[..., h - 1, :, :] - 1e-12).sum())
                for h in range(1, horizon_h)
            )
        )

    # 3 - receptive field, with the positive control
    rows, cols = np.indices(shape)
    distance = np.maximum(np.abs(rows - 20), np.abs(cols - 20)).astype(np.float64)
    reach: dict[str, Any] = {}
    for h in range(1, horizon_h + 1):
        field = b[..., h - 1, :, :].detach().numpy()
        lit = (field > 1e-12) & (distance > 0)
        max_lit = float(distance[lit].max()) if lit.any() else 0.0
        reach[str(h)] = {
            "max_lit_chebyshev_cells": max_lit,
            "budget_cells": 3 * h,
            "within_budget": max_lit <= 3 * h + 1e-9,
            # POSITIVE CONTROL: the stencil must actually reach further than the
            # previous lead's budget, or "within budget" is satisfied by a model
            # that reaches nowhere.
            "exceeds_previous_budget": max_lit > 3 * (h - 1),
        }

    # 4 - the near shell keeps accumulating
    near = distance == 1
    near_lead1 = float(b[..., 0, :, :].detach().numpy()[near].mean())
    near_leadH = float(b[..., horizon_h - 1, :, :].detach().numpy()[near].mean())

    negative_fraction = model.negative_increment_fraction(b0, w, fields, horizon_h, None)

    out = {
        "directness_max_abs_delta": directness_delta,
        "telescoping_max_abs_delta": worst_telescope,
        "monotone_violations": monotone_violations,
        "negative_increment_fraction": negative_fraction,
        "receptive_field": reach,
        "near_cell_probability_lead_1": near_lead1,
        "near_cell_probability_lead_H": near_leadH,
        "near_shell_accumulates": near_leadH > near_lead1,
    }
    failures = []
    if directness_delta != 0.0:
        failures.append(
            f"DIRECTNESS: hazard moved by {directness_delta:.3g} when the "
            "accumulated state was replaced by noise — this arm is not direct"
        )
    if worst_telescope > 1e-12:
        failures.append(
            f"TELESCOPING: increments do not compose to the one-shot marginal "
            f"(max |delta| {worst_telescope:.3g})"
        )
    if monotone_violations:
        failures.append(
            f"MONOTONICITY: {monotone_violations} cell-leads go BACKWARD; fire is "
            "absorbing and a non-monotone marginal is not a forecast"
        )
    if negative_fraction != 0.0:
        failures.append(
            f"the hazard increment was NEGATIVE on {negative_fraction:.3%} of "
            "cell-leads, so the clamp is binding and is hiding a real defect"
        )
    for h, row in reach.items():
        if not row["within_budget"]:
            failures.append(
                f"lead {h} reached {row['max_lit_chebyshev_cells']} cells against a "
                f"budget of {row['budget_cells']} — B is out-reaching arm A"
            )
        if not row["exceeds_previous_budget"]:
            failures.append(
                f"lead {h} reached only {row['max_lit_chebyshev_cells']} cells, not "
                f"past {3 * (int(h) - 1)}: the reach check cannot fail as written"
            )
    if near_leadH <= near_lead1:
        failures.append(
            "the NEAR SHELL does not accumulate: a cell adjacent to the fire is no "
            "likelier at lead H than at lead 1, which is the crippled dilation-only "
            "stencil ADR-045's capacity constraint was NOT supposed to force on us"
        )
    if failures:
        raise AssertionError(
            "DirectHorizonKernel property check FAILED:\n  - "
            + "\n  - ".join(failures)
            + f"\nmeasured: {out}"
        )
    return out


def parameter_counts(model: ContagionKernel) -> dict[str, Any]:
    """Every parameter tensor, its size, and the totals. Model-agnostic."""
    per_tensor = {name: int(p.numel()) for name, p in model.named_parameters()}
    trainable = {name: int(p.numel()) for name, p in model.named_parameters() if p.requires_grad}
    kernel_part = {k: v for k, v in per_tensor.items() if not k.startswith("latent.")}
    latent_part = {k: v for k, v in per_tensor.items() if k.startswith("latent.")}
    return {
        "kind": getattr(model, "kind", type(model).__name__),
        "name": getattr(model, "name", ""),
        "total": sum(per_tensor.values()),
        "trainable": sum(trainable.values()),
        "kernel_total": sum(kernel_part.values()),
        "latent_total": sum(latent_part.values()),
        "n_offsets": int(getattr(model, "n_offsets", 0)),
        "per_tensor": per_tensor,
    }


def compare_parameter_counts(
    incumbent: ContagionKernel,
    challenger: ContagionKernel,
    *,
    tolerance: float = PARAMETER_PARITY_TOLERANCE,
) -> dict[str, Any]:
    """ADR-045 (3)'s ``+/-10%`` capacity condition. Emits numbers, not a verdict.

    "B is not allowed to win on capacity" is a pre-registered CONDITION on the
    experiment, so the ratio is measured and printed beside every M10 number
    rather than argued in a docstring. If ``within_tolerance`` is False the arm is
    void and the report must say so.
    """
    a = parameter_counts(incumbent)
    b = parameter_counts(challenger)
    ratio = (b["total"] / a["total"]) if a["total"] else None
    extra = sorted(set(b["per_tensor"]) - set(a["per_tensor"]))
    missing = sorted(set(a["per_tensor"]) - set(b["per_tensor"]))
    return {
        "incumbent": a,
        "challenger": b,
        "ratio_challenger_over_incumbent": ratio,
        "tolerance": tolerance,
        "within_tolerance": (ratio is not None and abs(ratio - 1.0) <= tolerance),
        "tensors_only_in_challenger": extra,
        "tensors_only_in_incumbent": missing,
        "n_offset_weights": {
            "incumbent": a["per_tensor"].get("offset_log_weight"),
            "challenger": b["per_tensor"].get("offset_log_weight"),
        },
        "source": "ADR-045 (3): 'parameter count within +/-10% of A' — reported, not assumed",
    }
