"""PLAYTHROUGH (ADR-030) — the MEAN-PRESERVING ASYMMETRIC ACTIVITY GATE (M8).

M8 reverses a decision I documented at M6: the activity gate was deliberately
EXEMPTED from the log-normal mean correction, on the argument that correcting it
"would compensate every OFF draw with a hotter ON draw, i.e. would undo the thing
it exists to do". ADR-034 (4) falsified that premise — ``dormant_off_rate`` is
0.0000 on every gate arm ever trained and the ceiling from C1's covariates is
0.143 — so the exemption protects nothing measured while costing the ensemble
mean.

**REVERSING MY OWN DESIGN DECISION IS EXACTLY WHEN A KNOWN-ANSWER SCENARIO IS
WORTH MOST**, because the failure mode is not "the code is wrong", it is "the
change quietly destroyed the mechanism it was supposed to preserve". ADR-034 (2)
names ASYMMETRY as the working mechanism. A correction that symmetrised the
channel would fix the growth number and delete the reason the arm scores at all,
and every aggregate in the run table would still look plausible.

WHAT IS KNOWN BY CONSTRUCTION
-----------------------------
With ``z ~ N(mu, 1)`` and gate effect ``s z + c``:

1. **THE CORRECTION.** ``E[e^(s z)] = e^(s mu + s^2 / 2)``, so unit mean requires
   ``c = -(s mu + s^2 / 2)``. At ``s = 1.3, mu = -1.5`` that is ``+1.105``
   EXACTLY. Written here from the definition; the module has to come to it.
2. **THE MEAN.** ``E[e^(s z + c)] = 1`` to machine epsilon when corrected, and
   ``e^-1.105 = 0.33121`` when not. The uncorrected value is the BIAS the kernel
   has been silently undoing by inflating its base rate.
3. **THE ASYMMETRY IS A LOCATION-FREE PROPERTY, AND THE CORRECTION MOVES ONLY THE
   LOCATION.** ``m / E[m]`` has EXACTLY the same distribution with and without
   the correction, because ``c`` is a constant in log space. This is THE
   capability claim: *the bias is removed and the asymmetry is untouched*, and it
   is checked as a bitwise agreement of quantiles rather than as a picture.
4. **HOW ASYMMETRIC, IN ONE NUMBER.** ``P(m < E[m]) = Phi(s / 2)``: at ``s = 1.3``
   that is **0.7422**, so 74% of members sit below the ensemble mean and the mean
   is carried by a thin upper tail. A symmetric innovation gives 0.5, and
   ``log_intensity`` at its fitted 0.28 gives 0.5557. **That gap is the
   quantified sense in which the gate is the asymmetric channel** — and it is why
   ADR-034 (2) reads "spread bought downward is cheap, upward is unboundedly
   expensive".
5. **THE CONDITIONAL ROUTE IS DELIBERATELY LEFT UNCORRECTED.** Only the
   UNCONDITIONAL part of the prior mean is removed, so a dormant-looking hour can
   still be predicted quieter than an active one. That is a design decision, so
   it is planted as a defect (D5) and asserted, not left as a docstring claim.

THE PLANTED DEFECTS
-------------------
Seven mutations of the INSTRUMENT, each a bug this five-line change could really
have, plus one DECLARED BLIND SPOT asserted in the opposite direction. The sixth
(shrink sigma instead of shifting the location) is the one no mean-based check
can see, because under it the mean is CORRECT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.model.latent import LatentConfig, LatentHead

# --------------------------------------------------------------------------
# [A14] SELF-DECLARATION, read by tests/test_playthrough_registry.py.
# Registration is AUTOMATIC and these constants are how this module identifies
# itself. They live HERE, beside the playthrough, so that adding or changing a
# playthrough never requires editing another lead's file — the mechanism fix for
# three consecutive forced cross-boundary writes (ADR-039 (6)).
# --------------------------------------------------------------------------
PLAYTHROUGH_OWNER = "modelling (M8)"
PLAYTHROUGH_NOTE = (
    "M8's MEAN-PRESERVING ASYMMETRIC GATE, written because M8 REVERSES a design decision "
    "modelling documented at M6. 7 instrument mutations plus the harness's no-op control; "
    "the capability claim is that the correction is a pure LOCATION shift in log space, so "
    "the bias goes and the asymmetry ADR-034 (2) identifies as the working mechanism does "
    "not. Its sharpest defect ('fix the mean by SHRINKING the gate') is invisible to every "
    "mean-based check, because under it the mean is right."
)

#: The scenario's fixed scales. `SIGMA_GATE` is close to the value four M7 seeds
#: actually fitted (1.215-1.606), so the numbers here are the regime the model
#: really occupies rather than a convenient corner.
SIGMA_GATE = 1.3
SIGMA_INTENSITY = 0.28
GATE_PRIOR_MEAN = -1.5
N_DRAWS = 200_000
DRAW_SEED = 8

#: Bound at import: every mutation below patches `LatentHead.mean_correction`, so
#: a defect reaching for the class attribute at call time would call ITSELF. The
#: M7 playthrough learned this the hard way — an infinitely recursing mutation is
#: a defect that cannot run, which is worse than one that is not caught.
_REAL_CORRECTION = LatentHead.mean_correction


def _phi(x: float) -> float:
    """Standard normal CDF, from the definition. No scipy (C-4.3: no installs)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


#: Closed forms, written from the definition above.
CORRECTION = -(SIGMA_GATE * GATE_PRIOR_MEAN + 0.5 * SIGMA_GATE**2)
UNCORRECTED_MEAN = math.exp(SIGMA_GATE * GATE_PRIOR_MEAN + 0.5 * SIGMA_GATE**2)
MEDIAN_OVER_MEAN = math.exp(-0.5 * SIGMA_GATE**2)
P_BELOW_MEAN_GATE = _phi(SIGMA_GATE / 2.0)
P_BELOW_MEAN_INTENSITY = _phi(SIGMA_INTENSITY / 2.0)


@dataclass(frozen=True)
class GateWorld:
    """The scenario. Nothing here is fitted; the arms vary only the correction."""

    corrected: bool = True

    def head(self) -> LatentHead:
        return LatentHead(
            LatentConfig(
                dim=4,
                init_sigma=(SIGMA_INTENSITY, 0.2, 0.15, SIGMA_GATE),
                max_sigma=(2.0, 1.5, 1.0, 6.0),
                gate_prior_mean=GATE_PRIOR_MEAN,
                conditional_prior=True,
                mean_preserving=True,
                gate_mean_preserving=self.corrected,
            )
        )


@torch.no_grad()
def _observe(world: GateWorld) -> dict[str, Any]:
    head = world.head()
    # The CONDITIONAL route is configured FIRST, before any correction is read.
    # My first draft set it afterwards, so `mean_correction()` had already been
    # evaluated against a zero-initialised prior net and the defect that folds the
    # conditional term into the correction became invisible — a mutation that
    # cannot bite is a mutation that is not tested.
    with torch.no_grad():
        head.prior_net.weight.zero_()
        head.prior_net.bias.zero_()
        head.prior_net.weight[3, 2] = 1.0  # wind speed drives the gate
    uncorrected = LatentHead(
        LatentConfig(
            dim=4,
            init_sigma=(SIGMA_INTENSITY, 0.2, 0.15, SIGMA_GATE),
            max_sigma=(2.0, 1.5, 1.0, 6.0),
            gate_prior_mean=GATE_PRIOR_MEAN,
            conditional_prior=True,
            mean_preserving=True,
            gate_mean_preserving=False,
        )
    )
    sigma = head.sigma()
    corr = head.mean_correction()

    # The gate's log-multiplier under the prior, by Monte Carlo, from the SAME
    # standardised draws for both arms — so the two differ by the correction and
    # by nothing else, which is what makes probe 3 a comparison and not a race.
    gen = torch.Generator().manual_seed(DRAW_SEED)
    eps = torch.randn(N_DRAWS, generator=gen, dtype=torch.float64)
    z = eps + GATE_PRIOR_MEAN
    m_on = torch.exp(z * sigma[3] + corr[3])
    m_off = torch.exp(z * uncorrected.sigma()[3] + uncorrected.mean_correction()[3])

    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    std_on = torch.quantile(m_on / m_on.mean(), qs)
    std_off = torch.quantile(m_off / m_off.mean(), qs)

    # Dim 0's own correction must stay the plain -sigma^2/2: the gate's extra
    # linear term must not leak onto a dimension whose prior mean is zero.
    dim0_correction = float(corr[0])

    # Two covariate vectors standing for a quiet hour and an active one (high RH /
    # cool / still vs low RH / hot / windy), through the prior net configured above.
    quiet = torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64)
    active = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    mu_quiet = float(head.prior_mean(quiet)[3])
    mu_active = float(head.prior_mean(active)[3])
    s3 = float(sigma[3])
    c3 = float(corr[3])
    e_quiet = math.exp(s3 * mu_quiet + 0.5 * s3 * s3 + c3)
    e_active = math.exp(s3 * mu_active + 0.5 * s3 * s3 + c3)

    # NOTE the mask must be the UNCORRECTED head's own: my first draft used the
    # corrected head's (which carries dim 3), so the guard failed on the clean
    # world. Caught by the harness before any model ran.
    su = uncorrected.sigma()
    legacy = -0.5 * su * su * uncorrected.log_multiplier_mask
    return {
        "corrected": world.corrected,
        "sigma_gate": float(sigma[3]),
        "correction_gate": float(corr[3]),
        "dim0_correction": dim0_correction,
        "mean_multiplier_on": float(m_on.mean()),
        "mean_multiplier_off": float(m_off.mean()),
        "median_over_mean_on": float(m_on.median() / m_on.mean()),
        "p_below_mean_on": float((m_on < m_on.mean()).to(torch.float64).mean()),
        "p_below_mean_intensity": P_BELOW_MEAN_INTENSITY,
        "standardised_quantiles_on": std_on.tolist(),
        "standardised_quantiles_off": std_off.tolist(),
        "max_standardised_quantile_gap": float((std_on - std_off).abs().max()),
        "e_multiplier_quiet_hour": e_quiet,
        "e_multiplier_active_hour": e_active,
        "off_path_equals_legacy_expression": bool(
            torch.equal(uncorrected.mean_correction(), legacy)
        ),
        "n_draws": int(N_DRAWS),
    }


def _near(x: float, want: float, tol: float) -> bool:
    return abs(float(x) - float(want)) <= tol


# --------------------------------------------------------------------------
# the planted defects — instrument mutations of `LatentHead.mean_correction`
# --------------------------------------------------------------------------


def _only_quadratic(self: LatentHead) -> torch.Tensor:
    """Forget the ``sigma * mu`` term. THE one-keystroke bug: it is what you get
    by adding dim 3 to the existing mask and stopping there."""
    sigma = self.sigma()
    return -0.5 * sigma * sigma * self.log_multiplier_mask


def _only_linear(self: LatentHead) -> torch.Tensor:
    """Forget the ``sigma^2 / 2`` term — the other half of the same slip."""
    sigma = self.sigma()
    return -sigma * self.gate_prior_shift


def _sign_flipped(self: LatentHead) -> torch.Tensor:
    """Add the prior mean back instead of removing it."""
    sigma = self.sigma()
    return -0.5 * sigma * sigma * self.log_multiplier_mask + sigma * self.gate_prior_shift


def _wrong_dimension(self: LatentHead) -> torch.Tensor:
    """Put the gate's linear term on dimension 0 instead of dimension 3."""
    sigma = self.sigma()
    shift = torch.zeros_like(self.gate_prior_shift)
    shift[0] = self.gate_prior_shift[3] if self.gate_prior_shift.numel() > 3 else 0.0
    return -0.5 * sigma * sigma * self.log_multiplier_mask - sigma * shift


def _also_corrects_the_conditional_term(self: LatentHead) -> torch.Tensor:
    """The variant I deliberately did NOT build: cancel the CONDITIONAL mean too.

    Modelled here by folding the prior net's bias into the correction. It leaves
    ``E[m] = 1`` for every hour, which is arithmetically tidier and deletes the
    OFF-state route entirely — the gate would no longer be able to say that a
    quiet hour is quieter than an active one.
    """
    sigma = self.sigma()
    base = -0.5 * sigma * sigma * self.log_multiplier_mask - sigma * self.gate_prior_shift
    if self.prior_net is None or not bool(self.gate_prior_shift.abs().sum() > 0):
        return base
    extra = torch.zeros_like(base)
    extra[3] = -sigma[3] * self.prior_net.weight[3].sum()
    return base + extra


#: Bound at import for the same reason as the correction: the sigma mutation below
#: must be able to call the real implementation without calling itself.
_REAL_SIGMA = LatentHead.sigma


def _mean_fixed_by_shrinking_sigma(self: LatentHead) -> torch.Tensor:
    """ "Fix" the mean by NARROWING the gate instead of RELOCATING it.

    The most dangerous defect available to this change, and the one ADR-034 (2)
    is a warning about: shrink ``sigma_gate`` until ``E[m]`` comes back to ~1.
    The growth number is repaired, every artifact still says the gate is on, and
    the ASYMMETRY — the thing that makes the arm score at all — is gone.
    Applied only to the corrected arm, which is exactly how a real "fix" would
    have been written.
    """
    sigma = _REAL_SIGMA(self).clone()
    if bool(self.gate_prior_shift.abs().sum() > 0):
        sigma[3] = 0.30
    return sigma


#: Bound at import, same rule again.
_REAL_PRIOR_MEAN = LatentHead.prior_mean


def _prior_is_silently_unconditional(
    self: LatentHead, covariates: torch.Tensor | None = None
) -> torch.Tensor:
    """The conditional prior stops being conditional and says nothing about it.

    `M7`'s `spatial_modes_silently_do_nothing` in a new place: `prior_net` still
    exists, still trains, still serialises, and the artifact still prints
    `conditional_prior: true` — but every hour gets the same prior mean, so the
    OFF-state route is dead while looking alive. No mean-based probe can see it,
    because the UNCONDITIONAL mean is exactly what the correction pins to 1.
    """
    return _REAL_PRIOR_MEAN(self, None)


PLAYTHROUGH = PT.Playthrough(
    name="mean_preserving_asymmetric_gate",
    build=GateWorld,
    observe=_observe,
    note="M8's reversal of the M6 gate exemption: remove the BIAS, keep the ASYMMETRY.",
    probes=(
        PT.Probe(
            "correction_matches_the_closed_form",
            lambda o: _near(o["correction_gate"], CORRECTION, 1e-12),
            note="c = -(s mu + s^2/2) = +1.105 EXACTLY at s = 1.3, mu = -1.5. Written from "
            "the definition; the module has to come to it.",
        ),
        PT.Probe(
            "the_mean_multiplier_is_exactly_one",
            lambda o: (
                _near(o["mean_multiplier_on"], 1.0, 5e-3)
                and _near(o["mean_multiplier_off"], UNCORRECTED_MEAN, 5e-3)
            ),
            note="E[e^gate] = 1 corrected and 0.33121 uncorrected. The uncorrected value is "
            "the BIAS the kernel was silently undoing by inflating its base rate, and it is "
            "the whole reason held-out growth read 0.802-0.840.",
        ),
        PT.Probe(
            "THE_ASYMMETRY_SURVIVES_THE_CORRECTION",
            lambda o: o["max_standardised_quantile_gap"] <= 1e-9,
            note="THE capability probe, and the one this change could plausibly break. The "
            "correction is a CONSTANT in log space, so m / E[m] has an IDENTICAL "
            "distribution with and without it: five quantiles agree to 1e-9. ADR-034 (2) "
            "makes asymmetry the working mechanism, so a fix that symmetrised the channel "
            "would repair the mean and delete the reason the arm scores at all.",
        ),
        PT.Probe(
            "the_channel_is_ASYMMETRIC_not_merely_WIDE",
            lambda o: (
                _near(o["p_below_mean_on"], P_BELOW_MEAN_GATE, 5e-3)
                and o["p_below_mean_on"] - o["p_below_mean_intensity"] > 0.15
            ),
            note="P(m < E[m]) = Phi(s/2) = 0.7422 at the gate's fitted scale, against 0.5 for "
            "a symmetric innovation and 0.5557 for `log_intensity` at its fitted 0.28. That "
            "18-point gap is the quantified sense in which the gate is the asymmetric "
            "channel, and it is why downward spread is cheap here and upward is not.",
        ),
        PT.Probe(
            "the_conditional_route_is_deliberately_left_open",
            lambda o: o["e_multiplier_quiet_hour"] < 0.5 * o["e_multiplier_active_hour"],
            note="ONLY the unconditional part of the prior mean is removed, so a quiet hour "
            "is still predicted quieter than an active one. That is a DESIGN DECISION about "
            "the OFF-state route, so it is asserted here rather than left in a docstring — "
            "D5 plants the tidier variant that deletes it.",
        ),
        PT.Probe(
            "the_correction_does_not_leak_onto_dimension_zero",
            lambda o: _near(o["dim0_correction"], -0.5 * SIGMA_INTENSITY**2, 1e-12),
            note="dim 0's prior mean is zero, so its correction must stay the plain "
            "-sigma^2/2 = -0.0392. A linear term leaking there would bias `log_intensity` "
            "in the opposite direction and partially hide the gate's own bug.",
        ),
        PT.Probe(
            "the_scenario_sits_where_the_model_really_fitted",
            lambda o: (
                1.2 <= o["sigma_gate"] <= 1.7
                and o["off_path_equals_legacy_expression"]
                and o["n_draws"] >= 100_000
            ),
            note="SCENARIO GUARD. Four M7 seeds fitted sigma_gate 1.215-1.606, so 1.3 is the "
            "regime the model occupies, not a convenient corner; the OFF path is the "
            "verbatim pre-M8 expression; and the Monte-Carlo probes have enough draws that "
            "their 5e-3 tolerances are not noise.",
            guard=True,
        ),
    ),
    defects=(
        PT.Defect(
            "only_the_quadratic_term_removed",
            PT.attribute_defect((LatentHead, "mean_correction", _only_quadratic)),
            note="THE one-keystroke bug: add dim 3 to the existing mask and stop. Leaves "
            "E[m] = e^(s mu) = 0.1423, a 7x bias, while every artifact still prints "
            "`gate_mean_preserving: true`.",
        ),
        PT.Defect(
            "only_the_linear_term_removed",
            PT.attribute_defect((LatentHead, "mean_correction", _only_linear)),
            note="the other half of the same slip. E[m] = e^(s^2/2) = 2.31, i.e. the change "
            "would OVER-correct and turn under-prediction into over-prediction.",
        ),
        PT.Defect(
            "correction_sign_flipped",
            PT.attribute_defect((LatentHead, "mean_correction", _sign_flipped)),
            note="add the prior mean back instead of removing it. Doubles the bias rather "
            "than removing it, and is invisible in any aggregate that only reports sigma.",
        ),
        PT.Defect(
            "correction_applied_to_the_wrong_dimension",
            PT.attribute_defect((LatentHead, "mean_correction", _wrong_dimension)),
            note="the gate's linear term lands on `log_intensity`. Both dimensions are then "
            "wrong in opposite directions, which is exactly the shape that produces a "
            "plausible ensemble mean out of two cancelling errors.",
        ),
        PT.Defect(
            "the_conditional_term_is_corrected_too",
            PT.attribute_defect(
                (LatentHead, "mean_correction", _also_corrects_the_conditional_term)
            ),
            note="the tidier variant I declined: E[m] = 1 for EVERY hour, which deletes the "
            "OFF-state route rather than merely failing to reach it. Planted so that a "
            "design decision stated in a docstring is enforced by a test.",
        ),
        PT.Defect(
            "mean_fixed_by_SHRINKING_the_gate_instead_of_SHIFTING_it",
            PT.attribute_defect((LatentHead, "sigma", _mean_fixed_by_shrinking_sigma)),
            note="THE dangerous one, and the reason this playthrough exists. Narrow "
            "sigma_gate to 0.30 and E[m] returns to ~1 on its own: the growth number is "
            "repaired, `gate_mean_preserving: true` still prints, and the ASYMMETRY that "
            "ADR-034 (2) identifies as the working mechanism is gone. Only the two asymmetry "
            "probes can see it — no mean-based check can, because the mean is CORRECT.",
        ),
        PT.Defect(
            "the_conditional_prior_silently_stops_being_conditional",
            PT.attribute_defect((LatentHead, "prior_mean", _prior_is_silently_unconditional)),
            note="`prior_net` still exists, still trains, still serialises, and the artifact "
            "still prints `conditional_prior: true` — but every hour gets the same prior "
            "mean. THE ONLY PROBE THAT CAN SEE THIS IS THE CONDITIONAL-ROUTE ONE: the "
            "correction pins the UNCONDITIONAL mean to 1, so every mean-based check passes. "
            "Planted because without it that probe caught nothing on its own and the design "
            "decision it enforces would have been a docstring claim wearing a test's name.",
        ),
        PT.Defect(
            "nothing_is_changed_at_all",
            PT.no_defect(),
            note="DECLARED BLIND SPOT and the harness's own control: a defect that mutates "
            "nothing must be caught by nothing, or the coverage map is hallucinating. It "
            "also stands for what this playthrough provably CANNOT see — everything "
            "DOWNSTREAM of the multiplier distribution, i.e. whether the corrected effect "
            "actually reaches the hazard field. That wiring is covered separately by "
            "`eval/selftest.check_latent_off_reproduces_the_g2_kernel_bitwise` and "
            "`check_gate_mean_preserving_is_off_by_default_and_exact_when_on`, and the "
            "division is stated here rather than assumed.",
            detected=False,
        ),
    ),
)


def test_the_asymmetric_gate_playthrough_has_total_mutation_coverage(
    playthrough_report,
) -> None:
    report = playthrough_report(PLAYTHROUGH)
    print(PT.format_report(report))
    assert report.passed, PT.format_report(report)
    assert report.mutation_coverage == 1.0, PT.format_report(report)
    assert not report.dead_probes, PT.format_report(report)


def test_the_asymmetry_probe_is_load_bearing_and_not_a_tautology(
    playthrough_report,
) -> None:
    """A SYMMETRIC innovation must FAIL the asymmetry probe.

    Without this, "the channel is asymmetric" could be a property every latent
    dimension already had, and the probe would be measuring the existence of a
    log-normal rather than the gate's regime-switch scale. `log_intensity` at its
    fitted 0.28 is the negative control: same family, one twentieth the skew.
    """
    obs = _observe(GateWorld(corrected=True))
    name = "the_channel_is_ASYMMETRIC_not_merely_WIDE"
    probe = next(p for p in PLAYTHROUGH.probes if p.name == name)
    assert probe.check(obs)
    symmetric = dict(obs)
    symmetric["p_below_mean_on"] = P_BELOW_MEAN_INTENSITY
    assert not probe.check(symmetric)


def test_the_correction_is_exactly_a_location_shift_in_log_space() -> None:
    """The claim behind probe 3, stated on the LOG multiplier and independently.

    ``log m`` differs between the arms by the CONSTANT ``c`` at every draw, so the
    variance, the skewness and every centred moment are unchanged. Checked on the
    raw draws rather than on quantiles, so the two probes cannot fail together for
    a shared reason.
    """
    on = _observe(GateWorld(corrected=True))
    off = _observe(GateWorld(corrected=False))
    shift = on["correction_gate"] - off["correction_gate"]
    assert _near(shift, CORRECTION, 1e-12)
    assert _near(
        math.log(on["mean_multiplier_on"]) - math.log(off["mean_multiplier_off"]),
        CORRECTION,
        5e-3,
    )
    # ...and the standardised spread is identical, to the draw.
    assert np.allclose(
        on["standardised_quantiles_on"], off["standardised_quantiles_off"], atol=1e-9
    )


def test_a_config_asking_for_the_gate_correction_without_the_gate_raises() -> None:
    """A silent no-op is the green-but-vacuous shape; this refuses up front."""
    import pytest

    with pytest.raises(ValueError, match="requires dim >= 4"):
        LatentConfig(dim=3, gate_mean_preserving=True)
