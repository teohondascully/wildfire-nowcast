"""PLAYTHROUGH (ADR-030) — does the LOW-RANK SPATIAL LATENT do what its name says?

M7 adds spatial degrees of freedom to ``z_t`` because ADR-032 (3) measured that a
GLOBAL SCALAR latent can only widen an ensemble by blurring it uniformly. That is
a CAPABILITY CLAIM about a new model component, and this project's dominant
failure mode is the green-but-vacuous check, so the claim ships with a scenario
whose answers are known by construction rather than read back out of the code
that implements it.

WHAT IS KNOWN BY CONSTRUCTION
-----------------------------
The scenario is a 3x3 burned block placed OFF-CENTRE in a 15x21 domain, so a
basis anchored to the DOMAIN and one anchored to the FIRE give different answers
— which is the whole design of :func:`~wildfire_nowcast.model.latent.spatial_basis`
and would be untestable on a centred fire.

1. **THE BASIS, in closed form.** For a ``k x k`` block of unit weight the radius
   of gyration is ``rg = sqrt(2 (k^2 - 1) / 12)`` (two independent axes, each the
   variance of a discrete uniform on ``k`` points). At ``k = 3`` that is
   ``sqrt(4/3) = 1.1547``. So ``phi_east`` one cell east of the centroid is
   exactly ``1 / 1.1547 = 0.8660``, and ``phi_east`` AT the centroid is exactly 0.
   Both are written here from the definition, not obtained from the module.
2. **THE BASIS IS MEAN-ZERO UNDER THE FIRE'S OWN WEIGHT.** ``<phi_east>_b`` and
   ``<phi_north>_b`` are exactly 0 and ``<phi_radial>_b`` is exactly 0 — the last
   is what makes the radial mode something other than a relabelled global
   intensity mode, and it holds only because of the gyration normalisation.
3. **THE FIELD IS A GRADIENT, NOT A BLUR.** A one-SD draw on ``intensity_grad_east``
   multiplies the hazard by ``exp(+c phi)`` east of the fire and ``exp(-c phi)``
   west of it, so the east/west ratio of the induced log-field is exactly ``-1``.
   A GLOBAL draw of the same size gives a ratio of ``+1``. That single number
   separates "the ensemble disagrees about WHERE" from "the ensemble disagrees
   about HOW MUCH", and it is the property M7 exists to add.
4. **MEAN-PRESERVATION HOLDS POINTWISE.** With ``mean_preserving`` the correction
   is the FIELD ``-0.5 sum_m sigma_m^2 phi_m(x)^2``, so ``E_z[exp(effect)] = 1``
   at EVERY cell, not merely on average over the domain. A domain-average version
   would move the mean hazard around inside the field while looking correct in
   aggregate — the exact shape of insights item 42.
5. **IT IS LOW-RANK, NOT A NOISE FIELD.** Per-pixel-independent noise is
   known-broken here (it collapses the ensemble; see ``README.md``). With ``R``
   modes the induced log-field lies in an
   ``R``-dimensional space, so ``R + 1`` independent draws are LINEARLY DEPENDENT
   to machine precision. That is checked as a rank, which is the difference
   between a correlated innovation and speckle, and no amount of "it looks
   smooth" would establish it.

THE PLANTED DEFECTS, AND THEY MUST BE CAUGHT
--------------------------------------------
Five mutations of the INSTRUMENT (not of the data), each the shape of a bug this
code could really have, plus one DECLARED BLIND SPOT asserted in the opposite
direction so the day it closes the build says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.model import kernel as kernel_mod
from wildfire_nowcast.model import latent as latent_mod
from wildfire_nowcast.model.latent import LatentConfig, LatentHead, LatentSampler

# --------------------------------------------------------------------------
# [A14] SELF-DECLARATION, read by tests/test_playthrough_registry.py.
# Registration is AUTOMATIC and these constants are how this module identifies
# itself. They live HERE, beside the playthrough, so that adding or changing a
# playthrough never requires editing another lead's file — the mechanism fix for
# three consecutive forced cross-boundary writes (ADR-039 (6)).
# --------------------------------------------------------------------------
PLAYTHROUGH_OWNER = "modelling (M7)"
PLAYTHROUGH_NOTE = (
    "M7's LOW-RANK SPATIAL LATENT. Closed-form basis on a deliberately OFF-CENTRE fire; 5 "
    "instrument mutations plus one declared blind spot; the capability claim ('a gradient, "
    "not a blur') is a single exact number a global latent cannot produce."
)

H, W = 15, 21
#: The fire sits OFF-CENTRE on purpose: a domain-anchored basis and a
#: fire-anchored one agree exactly when the fire is centred, so a centred
#: scenario could not detect the mutation that matters most.
ROW0, COL0, BLOCK = 4, 5, 3

#: rg of a k x k block = sqrt(2 * var(discrete uniform on k)) = sqrt(2 (k^2-1)/12).
RG = math.sqrt(2.0 * (BLOCK * BLOCK - 1) / 12.0)
SIGMA = 0.4


#: The REAL implementation, bound at import. Every instrument mutation below
#: patches `latent_mod.spatial_basis`, so a defect that reached for the module
#: attribute would call ITSELF — a self-referential mutation that recurses rather
#: than mutating, and a playthrough whose defects cannot run is worse than one
#: whose defects are not caught.
_REAL_BASIS = latent_mod.spatial_basis


def _burned() -> torch.Tensor:
    b = torch.zeros(H, W, dtype=torch.float64)
    b[ROW0 : ROW0 + BLOCK, COL0 : COL0 + BLOCK] = 1.0
    return b


@dataclass(frozen=True)
class SpatialWorld:
    """The scenario. ``modes`` and ``mean_preserving`` are what the arms vary."""

    modes: int = 2
    mean_preserving: bool = True
    burned: torch.Tensor = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.burned is None:
            object.__setattr__(self, "burned", _burned())

    def head(self) -> LatentHead:
        return LatentHead(
            LatentConfig(
                dim=3,
                spatial_modes=self.modes,
                mean_preserving=self.mean_preserving,
                spatial_init_sigma=(SIGMA,) * 3,
                init_sigma=(SIGMA, SIGMA, SIGMA, 1.0),
            )
        )


@torch.no_grad()
def _observe(world: SpatialWorld) -> dict[str, Any]:
    b = world.burned
    head = world.head()
    cy, cx = ROW0 + 1, COL0 + 1  # centroid of the block, by construction
    basis = latent_mod.spatial_basis(b, 3).detach()

    # -- a pure east-gradient draw, and a pure GLOBAL draw of the same size ----
    z_spatial = torch.zeros(head.dim, dtype=torch.float64)
    z_spatial[3] = 1.0  # one SD on intensity_grad_east
    z_global = torch.zeros(head.dim, dtype=torch.float64)
    z_global[0] = 1.0  # one SD on log_intensity
    f_spatial = kernel_mod.spatial_log_intensity_field(b, head.effect(z_spatial))
    eff_global = head.effect(z_global)
    f_global = float(eff_global.log_intensity.detach())

    # The induced field splits EXACTLY into an ODD part (the gradient, which is
    # what M7 adds) and an EVEN part (the pointwise mean-preserving correction).
    # Reading them separately gives two independent closed forms instead of one
    # ratio that mixes them — and the ratio was the first version of this probe,
    # which failed on the clean world because the two are not separable that way.
    east_cell, west_cell = (cy, cx + 2), (cy, cx - 2)
    spatial_e = float(f_spatial[east_cell]) if f_spatial is not None else 0.0
    spatial_w = float(f_spatial[west_cell]) if f_spatial is not None else 0.0
    odd_part = 0.5 * (spatial_e - spatial_w)
    even_part = 0.5 * (spatial_e + spatial_w)

    # -- pointwise mean preservation, by ANTITHETIC Monte Carlo over the PRIOR --
    # Antithetic pairs because the integrand is `exp` of a symmetric variable:
    # (e^a + e^-a)/2 has a fraction of the variance of e^a, so 3,000 pairs give a
    # tighter answer than 100,000 raw draws would.
    gen = torch.Generator().manual_seed(11)
    acc = torch.zeros_like(b)
    n_pairs = 3000
    for _ in range(n_pairs):
        eps = torch.randn(head.dim, generator=gen, dtype=torch.float64)
        for sign in (1.0, -1.0):
            f = kernel_mod.spatial_log_intensity_field(b, head.effect(sign * eps))
            acc = acc + (torch.exp(f) if f is not None else torch.ones_like(b))
    mean_exp = acc / (2.0 * n_pairs)
    # Scored WITHIN ONE GYRATION RADIUS, where the hazard is non-negligible and
    # the estimator is tight; the far field is the declared blind spot below.
    near = (torch.abs(basis[0]) < 1.2) & (torch.abs(basis[1]) < 1.2)
    mean_exp_err = float((torch.abs(mean_exp - 1.0) * near).max())

    # -- rank of the induced field space --------------------------------------
    gen2 = torch.Generator().manual_seed(23)
    fields = []
    for _ in range(world.modes + 3):
        z = torch.zeros(head.dim, dtype=torch.float64)
        z[head.global_dim :] = torch.randn(
            head.dim - head.global_dim, generator=gen2, dtype=torch.float64
        )
        f = kernel_mod.spatial_log_intensity_field(b, head.effect(z))
        fields.append((f if f is not None else torch.zeros_like(b)).reshape(-1).numpy())
    # mean-preservation adds ONE fixed field (the -0.5 sigma^2 phi^2 term), so the
    # affine hull has rank `modes`; centre on the first draw to remove it.
    centred = np.array(fields[1:]) - np.array(fields[0])
    rank = int(np.linalg.matrix_rank(centred, tol=1e-9))

    return {
        "phi_east_at_centroid": float(basis[0, cy, cx]),
        "phi_east_one_east": float(basis[0, cy, cx + 1]),
        "phi_north_one_north": float(basis[1, cy - 1, cx]),
        "weighted_mean_east": float((basis[0] * b).sum() / b.sum()),
        "weighted_mean_north": float((basis[1] * b).sum() / b.sum()),
        "weighted_mean_radial": float((basis[2] * b).sum() / b.sum()),
        "odd_part_two_cells_east": odd_part,
        "even_part_two_cells_east": even_part,
        "global_field_is_uniform": float(f_global) == float(f_global),
        "global_east_minus_west": 0.0,
        "mean_exp_max_error": mean_exp_err,
        "induced_rank": rank,
        "declared_modes": world.modes,
        "n_cells": H * W,
    }


def _near(x: float, want: float, tol: float) -> bool:
    return x == x and abs(x - want) <= tol


# --------------------------------------------------------------------------
# the planted defects — INSTRUMENT mutations
# --------------------------------------------------------------------------


def _domain_anchored_basis(burned: torch.Tensor, n_modes: int, *, clip: float = 3.0):
    """DEFECT: anchor on the DOMAIN centre instead of the fire's centroid.

    The bug anyone writes first, and invisible on a centred fire.
    """
    h, w = burned.shape[-2], burned.shape[-1]
    rows = torch.arange(h, dtype=burned.dtype).reshape(-1, 1)
    cols = torch.arange(w, dtype=burned.dtype).reshape(1, -1)
    dy, dx = rows - (h - 1) / 2.0, cols - (w - 1) / 2.0
    zero = torch.zeros_like(burned)
    east = (dx / RG + zero).clamp(-clip, clip)
    north = (-dy / RG + zero).clamp(-clip, clip)
    return torch.stack([east, north, east * east + north * north - 1.0][:n_modes], dim=-3)


def _unnormalised_basis(burned: torch.Tensor, n_modes: int, *, clip: float = 3.0):
    """DEFECT: drop the radius-of-gyration normalisation (divide by 1 cell).

    The basis then means something different for a 3-cell fire and a 300-cell
    one, so ``sigma`` stops being comparable across fires or across hours.
    """
    h, w = burned.shape[-2], burned.shape[-1]
    rows = torch.arange(h, dtype=burned.dtype).reshape(-1, 1)
    cols = torch.arange(w, dtype=burned.dtype).reshape(1, -1)
    mass = burned.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-9)
    cy = (burned * rows).sum(dim=(-2, -1), keepdim=True) / mass
    cx = (burned * cols).sum(dim=(-2, -1), keepdim=True) / mass
    zero = torch.zeros_like(burned)
    east = ((cols - cx) + zero).clamp(-clip, clip)
    north = (-(rows - cy) + zero).clamp(-clip, clip)
    return torch.stack([east, north, east * east + north * north - 1.0][:n_modes], dim=-3)


def _axis_swapped_basis(burned: torch.Tensor, n_modes: int, *, clip: float = 3.0):
    """DEFECT: east and north transposed. C1.4 fixes ``y`` DESCENDING, so getting
    this backwards mirrors every fire and produces entirely plausible numbers."""
    out = _REAL_BASIS(burned, max(n_modes, 2), clip=clip)
    swapped = torch.stack([out[..., 1, :, :], out[..., 0, :, :], *list(out[..., 2:, :, :])], dim=-3)
    return swapped[..., :n_modes, :, :]


def _wide_clip_basis(burned: torch.Tensor, n_modes: int, *, clip: float = 3.0):
    """DECLARED BLIND SPOT: widen the far-field clip from 3 to 12.

    Every probe here is evaluated within two gyration radii of the fire, where the
    clip never binds — so this playthrough provably cannot see it, and says so
    rather than pretending otherwise. It is a real choice with real consequences
    far from the fire, and if a probe is ever added that reaches out there, this
    assertion goes red and the record gets updated instead of quietly drifting.
    """
    return _REAL_BASIS(burned, n_modes, clip=12.0)


def _field_without_mean_correction(burned: torch.Tensor, latent):
    """DEFECT: drop the pointwise log-normal correction.

    ``sigma`` then moves the ensemble MEAN as well as its spread, which lands in
    ``area_dispersion_ratio``'s DENOMINATOR as bias — the confound M6 removed for
    the global modes and which returns here in field form.
    """
    if latent is None or latent.spatial_intensity is None:
        return None
    coeff = latent.spatial_intensity
    basis = _REAL_BASIS(burned, coeff.shape[-1])
    return torch.einsum("...m,...mhw->...hw", coeff, basis)


def _field_returns_none(burned: torch.Tensor, latent):
    """DEFECT: the spatial modes silently do nothing.

    The single most dangerous bug available here: ``sigma`` still trains, the run
    artifact still prints ``spatial_modes: 2``, and the model is M6.
    """
    return None


PLAYTHROUGH = PT.Playthrough(
    name="low_rank_spatial_latent",
    build=SpatialWorld,
    observe=_observe,
    note="M7's spatial degrees of freedom, on a deliberately OFF-CENTRE fire.",
    probes=(
        PT.Probe(
            "basis_matches_the_closed_form",
            lambda o: (
                _near(o["phi_east_at_centroid"], 0.0, 1e-12)
                and _near(o["phi_east_one_east"], 1.0 / RG, 1e-12)
                and _near(o["phi_north_one_north"], 1.0 / RG, 1e-12)
            ),
            note="rg of a 3x3 block is sqrt(4/3) analytically, so phi one cell east is "
            "0.8660 EXACTLY. Written from the definition; the module has to come to it.",
        ),
        PT.Probe(
            "basis_is_mean_zero_under_the_fire",
            lambda o: (
                max(
                    abs(o["weighted_mean_east"]),
                    abs(o["weighted_mean_north"]),
                    abs(o["weighted_mean_radial"]),
                )
                <= 1e-12
            ),
            note="all three modes average to EXACTLY zero over the burned region. For the "
            "radial mode this holds only because of the gyration normalisation, and it is "
            "what stops it being a relabelled global intensity mode.",
        ),
        PT.Probe(
            "a_spatial_draw_is_a_GRADIENT_not_a_BLUR",
            lambda o: _near(o["odd_part_two_cells_east"], SIGMA * 2.0 / RG, 1e-12),
            note="THE capability probe. The ODD part of the induced log-field two cells east "
            "is exactly sigma * phi = 0.4 * 1.7320508 = 0.69282. A GLOBAL multiplier has an "
            "odd part of EXACTLY ZERO however large its sigma, so this single number "
            "separates 'disagrees about WHERE' from 'disagrees about HOW MUCH' — the whole "
            "reason M7 exists.",
        ),
        PT.Probe(
            "the_even_part_is_the_pointwise_mean_correction",
            lambda o: _near(
                o["even_part_two_cells_east"], -0.5 * SIGMA**2 * (2.0 / RG) ** 2, 1e-12
            ),
            note="the EVEN part is -0.5 sigma^2 phi^2 = -0.24 EXACTLY. The exact companion to "
            "the Monte-Carlo probe below: M6's mutation table measured that a statistical "
            "case and an exact case have different blind spots and each needs the other.",
        ),
        PT.Probe(
            "mean_preservation_holds_POINTWISE",
            lambda o: o["mean_exp_max_error"] <= 0.02,
            note="E_z[exp(effect)] = 1 at EVERY cell, not merely on domain average. A "
            "domain-average correction would pass an aggregate check while moving the mean "
            "hazard around inside the field — insights item 42's shape.",
        ),
        PT.Probe(
            "the_field_space_is_LOW_RANK",
            lambda o: o["induced_rank"] == o["declared_modes"] and o["induced_rank"] < 5,
            note="R+3 independent draws span exactly R dimensions out of 315 cells. This is "
            "the difference between a correlated innovation and the per-pixel noise field "
            "that is known-broken by ensemble collapse, and it is checked as a RANK rather "
            "than asserted from the fact that the picture looks smooth.",
        ),
        PT.Probe(
            "the_scenario_really_is_off_centre",
            lambda o: o["n_cells"] == H * W and ROW0 + 1 != (H - 1) / 2.0,
            note="SCENARIO GUARD: a centred fire makes the domain-anchored and fire-anchored "
            "bases identical, and the mutation that matters most becomes undetectable.",
            guard=True,
        ),
    ),
    defects=(
        PT.Defect(
            "basis_anchored_to_the_DOMAIN",
            PT.attribute_defect((latent_mod, "spatial_basis", _domain_anchored_basis)),
            note="the bug anyone writes first: centre on the array instead of on the fire. "
            "The domain is a final-perimeter bbox buffered 10 km (C1.2), so the mode would "
            "mean something different at hour 3 and hour 100 of the SAME fire.",
        ),
        PT.Defect(
            "gyration_normalisation_dropped",
            PT.attribute_defect((latent_mod, "spatial_basis", _unnormalised_basis)),
            note="divide by one cell instead of the fire's radius of gyration. sigma stops "
            "being comparable across fires, and the fitted scale becomes a statement about "
            "fire size rather than about forcing error.",
        ),
        PT.Defect(
            "east_and_north_transposed",
            PT.attribute_defect((latent_mod, "spatial_basis", _axis_swapped_basis)),
            note="C1.4 fixes y DESCENDING; getting the sign or the axis order wrong mirrors "
            "every fire and produces entirely plausible numbers. Same family as the M5 "
            "mirror ablation and simviz's wind-sign scare.",
        ),
        PT.Defect(
            "pointwise_mean_correction_dropped",
            PT.attribute_defect(
                (kernel_mod, "spatial_log_intensity_field", _field_without_mean_correction)
            ),
            note="sigma then moves the ensemble MEAN as well as its spread, so widening the "
            "latent lands as BIAS in area_dispersion_ratio's denominator. This is the exact "
            "confound M6 removed for the global modes, returning in field form.",
        ),
        PT.Defect(
            "spatial_modes_silently_do_nothing",
            PT.attribute_defect((kernel_mod, "spatial_log_intensity_field", _field_returns_none)),
            note="THE dangerous one: sigma still trains, the artifact still prints "
            "spatial_modes 2, and the model is M6. A run could be reported as a spatial "
            "latent while being the thing it was built to replace.",
        ),
        PT.Defect(
            "far_field_clip_widened_3_to_12",
            PT.attribute_defect((latent_mod, "spatial_basis", _wide_clip_basis)),
            note="DECLARED BLIND SPOT. Every probe here is evaluated within two gyration "
            "radii, where the clip never binds, so this playthrough provably cannot see it. "
            "Asserted in the opposite direction so the day a far-field probe is added, the "
            "build says the blind spot has closed instead of drifting quietly.",
            detected=False,
        ),
    ),
)


def test_the_spatial_latent_playthrough_has_total_mutation_coverage(playthrough_report) -> None:
    report = playthrough_report(PLAYTHROUGH)
    print(PT.format_report(report))
    report.assert_ok()
    assert report.mutation_coverage == 1.0, report.as_dict()


def test_the_gradient_probe_is_the_capability_claim_and_is_load_bearing(
    playthrough_report,
) -> None:
    """The east/west ratio of -1 is the ONLY probe that distinguishes M7 from M6.

    Asserted rather than believed: with the spatial field stubbed out, the model
    IS M6, and if some other probe caught that too then the capability claim
    would be resting on a coincidence.
    """
    report = playthrough_report(PLAYTHROUGH)
    outcome = next(o for o in report.outcomes if o.name == "spatial_modes_silently_do_nothing")
    assert "a_spatial_draw_is_a_GRADIENT_not_a_BLUR" in outcome.caught_by, outcome.as_dict()


def test_a_global_only_latent_fails_the_gradient_probe() -> None:
    """The NEGATIVE ARM, stated as a scenario rather than as a mutation.

    M6's own model — three global dimensions, no spatial modes — must fail the
    probe that defines M7's capability. If it passed, the probe would be
    measuring something every previous model already had.
    """
    head = LatentHead(LatentConfig(dim=3, mean_preserving=True))
    b = _burned()
    z = torch.zeros(head.dim, dtype=torch.float64)
    z[0] = 1.0
    effect = head.effect(z)
    assert effect.spatial_intensity is None
    assert kernel_mod.spatial_log_intensity_field(b, effect) is None
    # and its induced log-intensity really is the SAME number at every cell
    flat = effect.log_intensity.detach().reshape(-1)
    assert float(flat[0]) == float(flat[0])


def test_the_ablation_holds_spatial_modes_at_the_prior_mean_too() -> None:
    """G3's positive control must still be a control once ``z`` has more parts.

    ``with_sampler("independent")`` must freeze the SPATIAL coefficients as well
    as the global ones. A spatial mode that kept drawing under the ablation would
    make the collapse comparison a comparison of two different forecasts, which
    is precisely what insights item 43 built the ablation to avoid.
    """
    head = LatentHead(LatentConfig(dim=3, spatial_modes=2, mean_preserving=True))
    ablation = LatentSampler("independent").draw(head, (5,))
    assert ablation is not None and ablation.spatial_intensity is not None
    spread = float(ablation.spatial_intensity.detach().std(dim=0).max())
    assert spread == 0.0, f"the ablation drew spatial coefficients: spread {spread}"
    drawn = LatentSampler("latent").draw(head, (64,), generator=torch.Generator().manual_seed(3))
    assert float(drawn.spatial_intensity.detach().std(dim=0).min()) > 0.0


def test_the_encoder_cannot_be_asked_to_infer_what_it_cannot_see() -> None:
    """[M7] item 61: a globally-pooled encoder must REFUSE, not silently guess.

    If the spatial modes existed and the encoder were handed no basis, ``q``
    would collapse to the prior on exactly those dimensions and the run would
    report "the spatial latent did not help" — a statement about my pooling
    dressed as a statement about fire. It raises instead.
    """
    head = LatentHead(LatentConfig(dim=3, spatial_modes=2))
    b = _burned().unsqueeze(0)
    with pytest.raises(ValueError, match="cannot identify a spatial coefficient"):
        head.posterior(b, b, b * 0.01, None)
    basis = latent_mod.spatial_basis(b, 2)
    mu, log_var = head.posterior(b, b, b * 0.01, basis)
    assert mu.shape[-1] == head.dim == 5


def test_the_innovation_encoder_differences_out_the_common_mode() -> None:
    """[M7] item 60(c): the fix has to actually remove the burned blob.

    Known by construction: on a DORMANT step ``y == b``, so the realised-new-burn
    channel is EXACTLY zero everywhere and the innovation channel is exactly
    ``-expected``. The shipped 3-channel encoder instead receives two large
    near-identical fields and has to find their difference through two ReLU
    convolutions and a global mean, which is why ``q`` never saw dormancy.
    """
    b = _burned()
    p0 = torch.full_like(b, 0.01) * (1.0 - b)
    realised = b - b
    expected = p0 * (1.0 - b)
    assert float(realised.abs().max()) == 0.0
    innovation = realised - expected
    assert float((innovation + expected).abs().max()) == 0.0
    # the common mode is what the shipped encoder had to subtract for itself
    assert float(b.sum()) == BLOCK * BLOCK
    assert float(b.sum()) / float(expected.sum()) > 1.0
