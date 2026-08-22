"""PLAYTHROUGH (ADR-030) - CAN the kernel say "nothing will happen this hour"?

This is the question ADR-027 (7) calls untouched and the maintainer calls the
most valuable unsolved thing in the project: ``dormant_off_rate`` is **0.0000** on
every ``zt``, ``nozt``, ``elbo_only`` and ``brier`` arm we have ever trained. We
ignite in **953 of 953** dormant windows. Persistence scores 1.0 and
``ellipse_cal3h`` 0.0714.

Two explanations have very different consequences and nothing so far separated
them:

* **ARCHITECTURE** - the model class cannot represent a step with zero expected
  ignition at all, in which case no amount of training will find one; or
* **TRAINING/CONDITIONING** - it can, and the fitted prior simply never learned
  WHICH hours are dormant.

M6's activity gate already produced the evidence that these are different things:
the train member-area variance ratio went 1.19 -> 10.04 at no NLL cost - the
ensemble CONTAINS dormant members - while ``dormant_off_rate`` stayed at 0.0000.
**Capacity is not knowledge.**

THIS PLAYTHROUGH DECIDES IT, WITH THE ANSWER KNOWN BY CONSTRUCTION
------------------------------------------------------------------
A synthetic sequence alternates DORMANT hours (RH 92%, 278 K, 0.3 m/s) with
ACTIVE hours (RH 12%, 308 K, 10 m/s), and the truth grows in the active hours and
in no others. Three models, each of whose verdict is known before it is run:

``always_on``    the M5-shaped 3-d latent with no gate. **Must FAIL on capacity**
                 - there is no route to zero expected ignition, so
                 ``dormant_off_rate`` is 0.
``always_off``   a gate held far in the OFF state with NO conditioning.
                 **Must FAIL on conditioning.** This is the PLANTED DEFECT and it
                 is not a strawman: it is persistence in a costume, and
                 persistence scores ``dormant_off_rate`` = **1.0** on the real
                 corpus. A one-rate off-state test would rank it top.
``oracle_gate``  the same gate with its conditional prior SET BY HAND from the
                 step's own RH, i.e. training solved perfectly. **Must PASS.**

The decisive reading is ``oracle_gate``: it uses the SHIPPED architecture with no
change to the kernel's spatial structure, so if it passes, the architecture can
represent dormancy and the blocker is the conditional prior's fit. If it fails,
the architecture is the blocker - a legitimate and more consequential outcome,
and one this file is built to report rather than to avoid.

Nothing here reads a held-out fire, a tensor, or a gate metric.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.eval.validity import off_state_verdict, window_ignition_counts
from wildfire_nowcast.model.inputs import N_STATIC, N_WEATHER, static_index, weather_index
from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
from wildfire_nowcast.model.latent import ACTIVITY_GATE, LATENT_COMPONENTS, LatentConfig

# --------------------------------------------------------------------------
# [A14] SELF-DECLARATION, read by tests/test_playthrough_registry.py.
# Registration is AUTOMATIC and these constants are how this module identifies
# itself. They live HERE, beside the playthrough, so that adding or changing a
# playthrough never requires editing another lead's file - the mechanism fix for
# three consecutive forced cross-boundary writes (ADR-039 (6)).
# --------------------------------------------------------------------------
PLAYTHROUGH_OWNER = "infra (adopted from modelling)"
PLAYTHROUGH_NOTE = (
    "Can the kernel say 'nothing will happen this hour'? Three models with verdicts known in "
    "advance; both halves of the verdict shown load-bearing."
)

SHAPE = (20, 20)
N_MEMBERS = 16
SEED = 20260808
#: Hours 0..N-1; True = ACTIVE. Alternating in blocks of two so the scenario has
#: consecutive dormant hours (a fire that is quiet for one hour and quiet for six
#: are different problems) and a balanced count of each regime.
ACTIVE_HOURS = (False, False, True, True, False, False, True, True, False, False, True, True)

#: **THE DORMANT HOURS ARE DELIBERATELY MILD, AND THAT IS THE WHOLE TEST.**
#: Writing this file found it: at RH 92% / 278 K / 0.3 m/s / FMC 28 the KERNEL'S
#: OWN PHYSICS shuts the fire off - the moisture damping and the near-zero wind
#: drive the rate of spread to zero, and even the `always_on` model scores
#: `dormant_off_rate` = 1.0. A scenario like that proves nothing: it tests
#: whether Rothermel damping reaches zero, not whether the model can learn that
#: an hour is dormant.
#: The real corpus is the opposite case, and that is exactly why it is hard:
#: 953 of 1,399 held-out windows have BITWISE zero growth while the weather is
#: only moderately quiet, so the physics still predicts ignition and every kernel
#: we have trained ignites in 953 of 953. These constants reproduce THAT regime -
#: at RH 62% / 290 K / 4 m/s the `always_on` model expects 0.5-3.5 new cells in a
#: window where truth grows by exactly zero.
DORMANT_WEATHER = {"rh_2m": 62.0, "temp_2m": 290.0, "wind": 4.0, "fuel_moisture_proxy": 9.0}
ACTIVE_WEATHER = {"rh_2m": 12.0, "temp_2m": 308.0, "wind": 10.0, "fuel_moisture_proxy": 3.0}

#: Oracle prior coefficients on the STANDARDISED RH covariate ``(RH - 50)/25``,
#: solved in closed form so the gate's prior mean is -5 in a dormant hour and +1
#: in an active one: ``-w(0.48) + b = -5`` and ``-w(-1.52) + b = +1`` give
#: ``w = 3.0``, ``b = -3.56``. Nothing is fitted and nothing is searched.
ORACLE_RH_WEIGHT = -3.0
ORACLE_BIAS = -3.56

#: Index of the ACTIVITY GATE inside `LATENT_COMPONENTS`, read from the module
#: rather than written as a literal - a hard-coded 3 here would silently point at
#: a different physical quantity the day a dimension is inserted.
GATE_INDEX = LATENT_COMPONENTS.index(ACTIVITY_GATE)


def build_static() -> np.ndarray:
    static = np.zeros((N_STATIC, *SHAPE), dtype=np.float64)
    static[static_index("fuel_model_id")] = 102.0  # GR grass everywhere
    static[static_index("canopy_cover")] = 20.0
    static[static_index("aspect_cos")] = 1.0
    return static


def build_weather() -> np.ndarray:
    weather = np.zeros((len(ACTIVE_HOURS), N_WEATHER, *SHAPE), dtype=np.float64)
    for t, active in enumerate(ACTIVE_HOURS):
        spec = ACTIVE_WEATHER if active else DORMANT_WEATHER
        weather[t, weather_index("wind_u10")] = spec["wind"]
        weather[t, weather_index("rh_2m")] = spec["rh_2m"]
        weather[t, weather_index("temp_2m")] = spec["temp_2m"]
        weather[t, weather_index("fuel_moisture_proxy")] = spec["fuel_moisture_proxy"]
    return weather


def build_truth() -> np.ndarray:
    """A fire that grows ONLY in active hours. Dormancy is exact, not approximate."""
    burned = np.zeros(SHAPE, dtype=bool)
    burned[8:12, 8:12] = True
    states = []
    for active in ACTIVE_HOURS:
        if active:
            grown = burned.copy()
            grown[1:, :] |= burned[:-1, :]
            grown[:-1, :] |= burned[1:, :]
            grown[:, 1:] |= burned[:, :-1]
            grown[:, :-1] |= burned[:, 1:]
            burned = grown
        states.append(burned.copy())
    return np.stack(states).astype(np.uint8)


def _model(kind: str) -> ContagionKernel:
    """The three declared models. Only `latent_config` differs; the kernel does not."""
    if kind == "always_on":
        cfg = LatentConfig(dim=3)
    elif kind == "always_off":
        # Gate pinned deep in the OFF state, NO conditioning. Persistence in a
        # costume: it will score a perfect dormant_off_rate for the wrong reason.
        cfg = LatentConfig(dim=4, gate_prior_mean=-8.0, conditional_prior=False)
    elif kind == "oracle_gate":
        cfg = LatentConfig(dim=4, gate_prior_mean=0.0, conditional_prior=True)
    else:  # pragma: no cover - guard
        raise ValueError(kind)
    model = ContagionKernel(KernelConfig(), name=kind, latent_config=cfg)
    if kind == "oracle_gate":
        _set_oracle_prior(model)
    return model


def _set_oracle_prior(model: ContagionKernel) -> None:
    """Hand-set the conditional prior on the ACTIVITY GATE from RH alone.

    This is the answer training is trying to find, installed by hand. RH is
    covariate 0 (see `kernel.step_covariates`) and enters as ``(RH - 50)/25``.
    The two coefficients are SOLVED (see `ORACLE_RH_WEIGHT`), not tuned: they put
    the gate's prior mean at -5 in a dormant hour and +1 in an active one.
    NOTHING about the kernel's spatial structure, its offsets or its physics is
    touched - which is what makes this a test of the ARCHITECTURE and not of a
    hand-built forecaster.
    """
    assert model.latent is not None and model.latent.prior_net is not None
    with torch.no_grad():
        model.latent.prior_net.weight.zero_()
        model.latent.prior_net.bias.zero_()
        model.latent.prior_net.weight[GATE_INDEX, 0] = ORACLE_RH_WEIGHT
        model.latent.prior_net.bias[GATE_INDEX] = ORACLE_BIAS


def run_scenario(kind: str, *, n_members: int = N_MEMBERS, seed: int = SEED) -> dict:
    """End to end: C5 `predict` per hour -> C6.2 window counts -> the verdict."""
    static, weather, truth = build_static(), build_weather(), build_truth()
    model = _model(kind)
    rows = []
    for t0 in range(len(ACTIVE_HOURS) - 1):
        x0 = truth[t0]
        samples = model.predict(x0, static, weather[t0 + 1 : t0 + 2], n_members, 1, seed + t0)
        rows.append(window_ignition_counts(samples, truth[t0 + 1 : t0 + 2], x0))
    verdict = off_state_verdict(rows)
    verdict["model"] = kind
    verdict["rows"] = rows
    return verdict


# --------------------------------------------------------------------------
# the scenario itself must be what it claims to be
# --------------------------------------------------------------------------


def test_the_scenario_really_contains_dormant_and_growing_windows() -> None:
    """If this ever fails, every verdict below is vacuous - so it is checked first."""
    truth = build_truth()
    growth = [
        int(np.count_nonzero((truth[t + 1] > 0) & ~(truth[t] > 0)))
        for t in range(len(ACTIVE_HOURS) - 1)
    ]
    assert sum(1 for g in growth if g == 0) >= 4, growth
    assert sum(1 for g in growth if g > 0) >= 4, growth
    # Dormancy is BITWISE zero, not "small": a scenario with a trickle would make
    # `dormant_off_rate` untestable, because the model would be right to ignite.
    assert set(g for g in growth if g == 0) == {0}


# --------------------------------------------------------------------------
# the three declared models, each with its verdict known in advance
# --------------------------------------------------------------------------


def test_a_model_with_no_off_state_is_FLAGGED() -> None:
    """`always_on` is the M5 candidate's latent structure. It must fail on CAPACITY."""
    v = run_scenario("always_on")
    assert not v["passed"], v
    assert v["dormant_off_rate"] == 0.0, (
        "a 3-d symmetric latent around 'always on' scored a non-zero off rate; "
        "either the scenario is not dormant or the instrument is not measuring what "
        f"it claims: {v}"
    )
    assert any("cannot represent an OFF state" in r for r in v["reasons"]), v["reasons"]


def test_the_planted_defect_an_ALWAYS_OFF_model_is_FLAGGED() -> None:
    """THE NEGATIVE CONTROL. A model that never ignites must NOT pass this test.

    It scores a PERFECT `dormant_off_rate` - exactly as persistence does on the
    real corpus (1.0, ADR-027 (7)) - and is useless. If this assertion ever
    passes, the off-state playthrough has become a metric that rewards silence,
    which is the pathology C6.0 was written for and the fourth time this project
    would have shipped it.
    """
    v = run_scenario("always_off")
    assert v["dormant_off_rate"] == 1.0, v
    assert not v["passed"], (
        "an ALWAYS-OFF model passed the off-state playthrough: the harness rewards "
        f"silence and is vacuous. {v}"
    )
    assert any("WRONG hours" in r for r in v["reasons"]), v["reasons"]


def test_the_architecture_CAN_represent_an_off_state_when_conditioned() -> None:
    """THE DECISIVE READING. Architecture or training?

    The activity gate with an ORACLE conditional prior - the shipped model class,
    with the answer training is looking for installed by hand and no change to the
    kernel's spatial structure.

    PASS  -> the architecture CAN represent dormancy; the blocker is the fit of
             `p(z_t | weather)`, which is a training problem.
    FAIL  -> the architecture is the blocker. That is a legitimate outcome and a
             more consequential one; do not weaken this test to avoid it, record
             it in insights and change the model class.
    """
    v = run_scenario("oracle_gate")
    assert v["passed"], (
        "THE ARCHITECTURE CANNOT REPRESENT AN OFF STATE even with a perfectly "
        f"conditioned prior. This is a finding about the model class, not the fit: {v}"
    )
    assert v["dormant_off_rate"] >= 0.5, v
    assert v["false_off_rate"] <= 0.2, v


def test_the_oracle_beats_both_controls_on_BOTH_rates_together() -> None:
    """Neither rate alone orders the three models correctly, and that is the point."""
    on = run_scenario("always_on")
    off = run_scenario("always_off")
    oracle = run_scenario("oracle_gate")
    # `dormant_off_rate` alone ranks the useless model FIRST or joint-first ...
    assert off["dormant_off_rate"] >= oracle["dormant_off_rate"] > on["dormant_off_rate"]
    # ... and `false_off_rate` alone ranks the always-on model first.
    assert on["false_off_rate"] <= oracle["false_off_rate"] < off["false_off_rate"]
    # Only the joint verdict separates them.
    assert [on["passed"], off["passed"], oracle["passed"]] == [False, False, True]


def test_MUTATION_a_ONE_RATE_off_state_verdict_would_pass_the_planted_defect() -> None:
    """The instrument mutation: drop `false_off_rate` and the harness goes vacuous.

    A defect planted in the SCORER rather than in a model, because a playthrough
    whose negative control is caught for an obvious reason has not tested
    anything. `max_false_off_rate = 1.0` disables the conditioning half of the
    verdict - the single-line change anyone would make if they wanted a simpler
    metric - and the ALWAYS-OFF model, which ignites nothing ever, immediately
    PASSES. That is what the second rate is buying, measured rather than argued.
    """
    from wildfire_nowcast.eval.validity import off_state_verdict

    rows = run_scenario("always_off")["rows"]
    one_rate = off_state_verdict(rows, max_false_off_rate=1.0)
    assert one_rate["passed"], (
        "the mutation did not change the verdict, so `false_off_rate` is not "
        f"load-bearing and this playthrough is weaker than it claims: {one_rate}"
    )
    assert not off_state_verdict(rows)["passed"]


def test_MUTATION_a_scenario_with_no_dormant_hours_is_UNDEFINED_not_passed() -> None:
    """A degenerate scenario must not be a free pass - `eval/validity`'s own precedent.

    The M4 fix to `eval/validity.py` found an UNDEFINED case falling through to
    the verdict that says "usable as a gate floor". Same trap, one gate later.
    """
    from wildfire_nowcast.eval.validity import off_state_verdict

    growing_only = [
        {
            "mean_new_cells": 0.0,
            "max_new_cells": 0.0,
            "min_new_cells": 0.0,
            "truth_new_cells": 5.0,
            "any_member_ignited": False,
        }
    ]
    v = off_state_verdict(growing_only)
    assert not v["passed"], v
    assert any("UNDEFINED" in r for r in v["reasons"]), v["reasons"]


@pytest.mark.parametrize("n_members", [4, 16])
def test_the_verdict_does_not_depend_on_the_member_count(n_members: int) -> None:
    """`dormant_off_rate` requires EVERY member to be silent, so it could have been
    a member-count artifact. Measured across a 4x range instead of assumed."""
    assert run_scenario("oracle_gate", n_members=n_members)["passed"]
    assert not run_scenario("always_on", n_members=n_members)["passed"]


# --------------------------------------------------------------------------
# MUTATION COVERAGE (A13, ADR-030) - the declarations above, made enforceable.
#
# This file already does the hard part: three models whose verdicts are known in
# advance, a planted defect that is a real previously-shipped behaviour
# (persistence in a costume), and a mutation planted in the SCORER. What it
# cannot do by itself is guarantee that those stay load-bearing. Declaring them
# to `common.playthrough` means a future edit that makes `always_off` pass, or
# that removes the second rate, fails the build with the reason printed rather
# than leaving a green suite behind.
#
# One BLIND SPOT is declared on purpose and asserted in the opposite direction:
# the verdict must NOT change when the member count is quartered. The
# parametrised test above checks that for two arms; here it is stated as a
# property of the whole three-model comparison, so the day the verdict becomes
# member-count-dependent the record is corrected instead of quietly drifting.
# --------------------------------------------------------------------------

KINDS = ("always_on", "always_off", "oracle_gate")


@dataclass(frozen=True)
class OffStateWorld:
    n_members: int = N_MEMBERS
    verdict_kwargs: tuple[tuple[str, float], ...] = ()


def _off_state_observe(world: OffStateWorld) -> dict[str, dict]:
    """Score all three declared models under one world. One observation."""
    kwargs = dict(world.verdict_kwargs)
    out: dict[str, dict] = {}
    for kind in KINDS:
        base = run_scenario(kind, n_members=world.n_members)
        verdict = off_state_verdict(base["rows"], **kwargs) if kwargs else base
        out[kind] = {**verdict, "model": kind}
    return out


def _joint_ordering(obs: dict[str, dict]) -> bool:
    """Neither rate alone orders the three models; only the pair does."""
    on, off, oracle = obs["always_on"], obs["always_off"], obs["oracle_gate"]
    return (
        off["dormant_off_rate"] >= oracle["dormant_off_rate"] > on["dormant_off_rate"]
        and on["false_off_rate"] <= oracle["false_off_rate"] < off["false_off_rate"]
    )


PLAYTHROUGH = PT.Playthrough(
    name="off_state",
    build=OffStateWorld,
    observe=_off_state_observe,
    probes=(
        PT.Probe(
            "a_model_with_no_off_state_is_flagged",
            lambda obs: (
                not obs["always_on"]["passed"] and obs["always_on"]["dormant_off_rate"] == 0.0
            ),
            note="`always_on` is the M5 candidate's latent structure; it must fail on CAPACITY.",
        ),
        PT.Probe(
            "the_planted_always_off_model_is_flagged",
            lambda obs: (
                not obs["always_off"]["passed"] and obs["always_off"]["dormant_off_rate"] == 1.0
            ),
            note="THE NEGATIVE CONTROL. It scores a PERFECT dormant_off_rate, exactly as "
            "persistence does on the real corpus, and must still be rejected.",
        ),
        PT.Probe(
            "the_conditioned_architecture_passes",
            lambda obs: (
                bool(obs["oracle_gate"]["passed"]) and obs["oracle_gate"]["false_off_rate"] <= 0.2
            ),
            note="THE DECISIVE READING (ADR-032 (5)): the shipped architecture with the answer "
            "installed by hand. If this cannot pass, the blocker is the model class.",
        ),
        PT.Probe(
            "only_the_joint_verdict_orders_the_three",
            _joint_ordering,
            note="each rate alone ranks a useless model first; the pair is what separates them.",
        ),
    ),
    defects=(
        PT.Defect(
            "verdict_drops_the_false_off_rate",
            PT.data_defect(lambda w: replace(w, verdict_kwargs=(("max_false_off_rate", 1.0),))),
            note="the single-line change anyone would make to simplify the metric. It disables "
            "the CONDITIONING half, and the model that ignites nothing ever immediately passes "
            "-- a metric that rewards silence, for the fourth time in this project.",
        ),
        PT.Defect(
            "verdict_drops_the_dormant_off_rate",
            PT.data_defect(lambda w: replace(w, verdict_kwargs=(("min_dormant_off_rate", 0.0),))),
            note="the mirror image: disabling the CAPACITY half lets `always_on` -- which has "
            "no route to zero expected ignition at all -- pass. Both rates have to be shown "
            "load-bearing, not just the one that was interesting to write about.",
        ),
        PT.Defect(
            "a_scenario_with_no_dormant_hours",
            PT.attribute_defect((sys.modules[__name__], "ACTIVE_HOURS", (True,) * 12)),
            note="a DEGENERATE SCENARIO must not be a free pass. `eval/validity.py`'s own M4 "
            "precedent: an UNDEFINED case fell through a verdict ladder into the branch that "
            "says 'usable as a gate floor'. Same trap, one gate later.",
        ),
        PT.Defect(
            "member_count_quartered",
            PT.data_defect(lambda w: replace(w, n_members=4)),
            note="A DECLARED BLIND SPOT, asserted in the opposite direction. `dormant_off_rate` "
            "requires EVERY member to be silent, so it could have been a member-count artifact. "
            "Measured across a 4x range: no verdict moves. If this ever IS caught, the verdict "
            "has become member-count dependent and the record must change, not this test.",
            detected=False,
        ),
    ),
    note="ADR-032 (5)'s question -- architecture or training? -- with all three answers known "
    "before the run.",
)


@pytest.fixture(scope="module")
def off_state_coverage(playthrough_report) -> PT.PlaythroughReport:
    """Scored once per session; see `tests/conftest.playthrough_report`."""
    return playthrough_report(PLAYTHROUGH)


def test_MUTATION_COVERAGE_every_planted_defect_in_this_file_is_detected(
    off_state_coverage: PT.PlaythroughReport,
) -> None:
    print(PT.format_report(off_state_coverage))
    off_state_coverage.assert_ok()
    assert off_state_coverage.mutation_coverage == 1.0


def test_BOTH_rates_are_load_bearing_and_the_harness_says_which(
    off_state_coverage: PT.PlaythroughReport,
) -> None:
    """Removing either half of the verdict must break a DIFFERENT probe.

    That asymmetry is the argument for carrying two rates, and it is now measured
    rather than argued: dropping `false_off_rate` lets the always-off model pass,
    dropping `min_dormant_off_rate` lets the always-on model pass.
    """
    caught = {o.name: set(o.caught_by) for o in off_state_coverage.outcomes}
    assert "the_planted_always_off_model_is_flagged" in caught["verdict_drops_the_false_off_rate"]
    assert "a_model_with_no_off_state_is_flagged" in caught["verdict_drops_the_dormant_off_rate"]
    assert caught["member_count_quartered"] == set()
