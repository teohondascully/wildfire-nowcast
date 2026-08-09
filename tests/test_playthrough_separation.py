"""PLAYTHROUGH (ADR-030) — does the G3 separation criterion recover a KNOWN margin?

``common/separation.py`` implements ADR-032 (7): G3's calibration half becomes a
SEPARATION TEST in the G2 style rather than a minted margin. ADR-030's standing
requirement then binds immediately — **a metric that adjudicates a gate ships with
a playthrough or it does not adjudicate** — so the criterion is not allowed to
exist before this file does.

WHAT IS KNOWN BY CONSTRUCTION
-----------------------------
The statistic is a paired equal-block effect size, so for block margins
``m_1..m_n`` it is exactly ``mean(m) / sd(m, ddof=1)``, computed here from the
definition and never read back out of the module::

    margins 1, 2, 3, 4  ->  mean 2.5, sd sqrt(5/3),  separation 1.936492...
    a null candidate    ->  every margin exactly 0  ->  separation EXACTLY 0.0
    a worse candidate   ->  every margin negative   ->  separation negative
    one block, any margin -> UNDEFINED (C-3: no pass/fail on one observation)
    a constant positive margin with zero SD -> UNDEFINED, deliberately

That last one is the criterion's own trap and it is asserted as an answer, not as
a tolerance: an SD estimated as exactly 0.0 from four blocks is not an SD, and
dividing by it is exactly how a 0.001 margin acquires infinite significance —
i.e. the hazard this criterion was created to remove, sneaking back in through
the denominator.

THE CASE THE RULING WAS ABOUT, WITH ITS REAL NUMBERS
----------------------------------------------------
``the_barred_degenerate_ellipse_fails_the_SD_bar_itself`` is not a constructed
strawman. It is
``ellipse_brier_fit_all`` at 3 h, per block, copied verbatim out of
``runs/baselines-20260809-035037/results.json`` — the arm modelling disclosed
would pass ``calibration_error / silent_floor < 1`` by **0.001**. Under this
criterion it scores **+0.697 SD against the arithmetic floor and 0.000 against
the degenerate envelope**, against a bar of 2.0. The ruling's claim is therefore
checked against the exact observation that motivated it, with no run directory
required at test time.

THE PLANTED DEFECTS
-------------------
Two are planted in the CRITERION'S OWN CONSTANTS, which is the only way to show a
bar is load-bearing: lower it to 0.5 SD and the barred ellipse is admitted; lower
the block minimum to 1 and a one-fire comparison is admitted, which is C-3
verbatim. Two are planted in the data. One is a DECLARED BLIND SPOT.

AND THE RESIDUAL HAZARD, MEASURED RATHER THAN ARGUED
----------------------------------------------------
``margins_shrunk_1000x`` shrinks every margin by three orders of magnitude while
leaving the block-to-block SD ratio identical. **``separation_sd`` does not move
at all** — the statistic is scale-invariant, so statistical separation is not
magnitude, and that is the honest weakness of any separation test. The harness
prints which probe caught it, and the answer is ``relative_margin_is_material``
and nothing else. That single line of the coverage map is the argument for
reporting the relative margin beside every verdict, in a form that cannot rot.
"""


from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest

from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.common import separation as S
from wildfire_nowcast.common.separation import BlockPair, conditions, separation

# --------------------------------------------------------------------------
# [A14] SELF-DECLARATION, read by tests/test_playthrough_registry.py.
# Registration is AUTOMATIC and these constants are how this module identifies
# itself. They live HERE, beside the playthrough, so that adding or changing a
# playthrough never requires editing another lead's file — the mechanism fix for
# three consecutive forced cross-boundary writes (ADR-039 (6)).
# --------------------------------------------------------------------------
PLAYTHROUGH_OWNER = 'infra'
PLAYTHROUGH_NOTE = (
    "G3's calibration half as a SEPARATION test (ADR-032 (7)): equal-block SD over a paired "
    "margin, with the degenerate arms' measured scores as the negative controls."
)

#: A reference level in the criterion's own units (a probability). Chosen near the
#: measured growth-band base rates (0.0017-0.0125) so the relative margins below
#: are the size of real ones rather than a convenient fiction.
REFERENCE = 0.006

#: Block margins whose separation is exact: mean 2.5, sd sqrt(5/3).
KNOWN_MARGINS = (1.0, 2.0, 3.0, 4.0)
KNOWN_SCALE = 1e-4
#: SEVEN blocks, six wins and one loss: separation 2.027 (clears the 2.0 bar) with
#: unanimity 6/7. Seven and not four because of the arithmetic fact pinned in
#: `test_unanimity_is_REDUNDANT_at_four_blocks_and_binding_at_six`.
SPLIT_MARGINS = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -1.0)
WANT_KNOWN_SEPARATION = float(
    np.mean(KNOWN_MARGINS) / np.std(KNOWN_MARGINS, ddof=1)
)  # 1.9364916731037085

#: ``ellipse_brier_fit_all`` at 3 h, verbatim from the ADR-032 run of record.
#: The arm that would have passed the ratio criterion by 0.001. Blocks 3/4/5/6 =
#: bobcat / creek / czu / dolan.
BARRED_ELLIPSE_3H: tuple[tuple[int, float, float], ...] = (
    (3, 0.005323476307446831, 0.005334760974995276),
    (4, 0.004106301708500826, 0.0041072639189236),
    (5, 0.012465583492178753, 0.012465583492178753),
    (6, 0.005246249586494294, 0.005248469751363736),
)


def _pairs(margins, *, scale: float = KNOWN_SCALE, reference: float = REFERENCE):
    """Blocks whose candidate is ``reference - scale * margin`` (lower is better)."""
    return [
        BlockPair(block_id=10 + i, candidate=reference - scale * m, reference=reference)
        for i, m in enumerate(margins)
    ]


# --------------------------------------------------------------------------
# 1. known answers, asserted directly
# --------------------------------------------------------------------------


def test_the_separation_matches_the_definition_to_machine_precision() -> None:
    """No Monte Carlo anywhere: the statistic is arithmetic and is pinned as such."""
    sep = separation(_pairs(KNOWN_MARGINS))
    assert sep.separation_sd == pytest.approx(WANT_KNOWN_SEPARATION, rel=1e-12, abs=1e-12)
    assert sep.n_blocks == 4
    assert sep.blocks_favouring == 4
    assert sep.unanimous


def test_a_null_candidate_scores_exactly_zero_and_is_refused() -> None:
    """The reference-free assertion this project has learned to prefer.

    A candidate identical to its reference has mean margin 0 and SD 0 — a genuine
    0/0 — and the answer that is arithmetically right is 0.0, not ``inf`` and not
    ``nan``. ``persistence`` reproduces this EXACTLY on the real corpus, where its
    ``calibration_error`` equals its ``silent_floor`` bitwise at every block.
    """
    sep = separation(_pairs((0.0, 0.0, 0.0, 0.0)))
    assert sep.separation_sd == 0.0
    assert sep.blocks_favouring == 0
    assert not conditions(sep)["all_conditions_met"]


def test_a_worse_candidate_scores_NEGATIVE_so_the_sign_convention_is_pinned() -> None:
    """A sign error here would silently invert a gate. ADR-032 (4) already caught
    one units error in a criterion by running a known answer through it."""
    sep = separation(_pairs((-1.0, -2.0, -3.0, -4.0)))
    assert sep.separation_sd == pytest.approx(-WANT_KNOWN_SEPARATION, rel=1e-12)
    assert sep.blocks_favouring == 0


def test_one_block_is_UNDEFINED_not_a_pass() -> None:
    """C-3, verbatim: no pass/fail may be calibrated on one observation."""
    sep = separation(_pairs((3.0,)))
    assert sep.separation_sd is None
    assert "C-3" in sep.undefined_reason or "one observation" in sep.undefined_reason
    assert not conditions(sep)["all_conditions_met"]


def test_a_zero_block_to_block_sd_is_REFUSED_not_blessed_as_infinite() -> None:
    """THE trap. A constant margin over four blocks has SD exactly 0.

    If that were reported as ``inf``, an arm beating the floor by 0.001 in every
    block would score infinite separation — the 0.999 case returning through the
    denominator of the statistic built to remove it.
    """
    sep = separation(_pairs((2.0, 2.0, 2.0, 2.0)))
    assert sep.sd_margin == 0.0
    assert sep.separation_sd is None
    assert "EXACTLY 0" in sep.undefined_reason
    assert not conditions(sep)["all_conditions_met"]


def test_the_barred_degenerate_ellipse_does_not_clear_the_bar() -> None:
    """ADR-032 (7)'s motivating observation, with its own numbers.

    `ellipse_brier_fit_all` would have passed `calibration_error / silent_floor
    < 1` by 0.001. Here it scores +0.697 SD against the arithmetic floor — the bar
    is 2.0 — and it wins 3 of 4 blocks, so it fails unanimity as well.
    """
    pairs = [BlockPair(block_id=b, candidate=c, reference=f) for b, c, f in BARRED_ELLIPSE_3H]
    sep = separation(pairs)
    assert sep.separation_sd == pytest.approx(0.697, abs=0.001)
    assert sep.blocks_favouring == 3
    assert not sep.unanimous
    result = conditions(sep)
    assert not result["all_conditions_met"]
    assert result["conditions"] == {"separation": False, "unanimity": False, "block_count": True}
    # ...and its relative margin is 0.05% of the floor, which is the magnitude the
    # ratio form could not express.
    assert 0.0 < sep.relative_margin < 0.001


def test_the_criterion_is_scale_invariant_and_that_is_why_magnitude_is_REPORTED() -> None:
    """Shrinking every margin 1000x leaves the separation BITWISE unchanged.

    Stated as a property rather than discovered later: a separation test cannot
    see magnitude, by construction. ``relative_margin`` is what carries it, and it
    is reported beside every verdict for exactly this reason.
    """
    big = separation(_pairs(KNOWN_MARGINS, scale=KNOWN_SCALE))
    small = separation(_pairs(KNOWN_MARGINS, scale=KNOWN_SCALE / 1000.0))
    # Equal to 1e-9 rather than bitwise: the margin is RECONSTRUCTED as
    # `reference - candidate` from two floats, so shrinking it 1000x costs a few
    # ulps of cancellation. The statistic is exactly scale-invariant; the inputs
    # are not exactly representable, and saying so is more useful than a
    # tolerance nobody explains.
    assert small.separation_sd == pytest.approx(big.separation_sd, rel=1e-9)
    assert small.relative_margin == pytest.approx(big.relative_margin / 1000.0, rel=1e-9)


def test_unanimity_is_REDUNDANT_at_four_blocks_and_binding_at_six() -> None:
    """A derived fact worth writing down before the corpus grows.

    For ``n`` blocks where one does not favour the candidate, the largest possible
    separation is ``(n-1)/sqrt(n)`` — 1.5 at n=4, 2.04 at n=6, 2.27 at n=7. So at
    the corpus's CURRENT 4 held-out blocks, ``separation >= 2.0`` already IMPLIES
    unanimity and the second condition can never bite. **It starts biting at six
    blocks**, which is exactly what the queued corpus extension produces.

    Recorded here rather than in prose because it changes what the criterion means
    the day the corpus swaps, and because a condition that cannot bind is the kind
    of thing that gets deleted as redundant right before it becomes necessary.
    """
    for n in range(3, 10):
        margins = [10.0] * (n - 1) + [0.0]
        sep = separation(_pairs(margins))
        assert sep.blocks_favouring == n - 1
        assert sep.separation_sd == pytest.approx((n - 1) / np.sqrt(n), rel=1e-12)
    assert (4 - 1) / np.sqrt(4) < S.MIN_SEPARATION_SD, "at 4 blocks unanimity is implied"
    assert (6 - 1) / np.sqrt(6) > S.MIN_SEPARATION_SD, "at 6 blocks it becomes binding"


def test_the_fitting_sample_is_registered_and_states_its_blocks() -> None:
    """C-3's mechanical half: the constant names the sample it was fitted on."""
    assert "4 DISTINCT held-out spatial blocks" in S.FITTING_SAMPLE
    assert "4848f491e8d588fa" in S.FITTING_SAMPLE
    assert "STATED LIMITATION" in S.FITTING_SAMPLE
    assert S.MIN_SEPARATION_SD == 2.0


# --------------------------------------------------------------------------
# 2. the wiring reads C6's payload, and only C6's payload
# --------------------------------------------------------------------------


def _payload(values: dict[str, float], floors: dict[str, float], model: str = "cand") -> dict:
    """A minimal C6-shaped ``per_fire`` payload: four fires, four blocks."""
    out = {}
    for i, (fire_id, value) in enumerate(values.items()):
        floor = floors[fire_id]
        out[fire_id] = {
            "spatial_block_id": 3 + i,
            "models": {
                model: {
                    "growth_windows": {
                        "band_calibration_error_by_horizon": {"3": value},
                        "band_calibration_error_silent_floor_by_horizon": {"3": floor},
                    }
                },
                "persistence": {
                    "growth_windows": {
                        "band_calibration_error_by_horizon": {"3": floor},
                        "band_calibration_error_silent_floor_by_horizon": {"3": floor},
                    }
                },
            },
        }
    return out


def test_the_wiring_averages_blocks_and_not_windows() -> None:
    """Two fires in ONE block count once, which is ADR-021 (4)'s whole point."""
    floors = {f"f{i}": 0.006 for i in range(4)}
    values = {f"f{i}": 0.006 - 1e-4 * (i + 1) for i in range(4)}
    payload = _payload(values, floors)
    sep = S.calibration_separation(payload, "cand", 3, reference="floor")
    assert sep.n_blocks == 4
    assert sep.separation_sd == pytest.approx(WANT_KNOWN_SEPARATION, rel=1e-12)

    # Now put every fire in the SAME block: one independent unit, not four.
    for fire in payload.values():
        fire["spatial_block_id"] = 3
    collapsed = S.calibration_separation(payload, "cand", 3, reference="floor")
    assert collapsed.n_blocks == 1
    assert collapsed.separation_sd is None


def test_the_envelope_reference_is_never_easier_than_the_arithmetic_floor() -> None:
    """A missing or absent degenerate arm must not soften the opponent.

    The envelope is ``min(floor, best degenerate arm)``, so the reference can only
    ever get HARDER. Without that floor, deleting an arm from the declared list
    would make the criterion easier to pass, which is a free parameter pointing
    the flattering way.
    """
    floors = {f"f{i}": 0.006 for i in range(4)}
    values = {f"f{i}": 0.006 - 1e-4 * (i + 1) for i in range(4)}
    payload = _payload(values, floors)
    for fire in payload.values():
        fire["models"]["persistence"]["growth_windows"][
            "band_calibration_error_by_horizon"
        ]["3"] = 0.99  # a terrible degenerate arm
    env = S.calibration_separation(payload, "cand", 3, reference="envelope")
    floor = S.calibration_separation(payload, "cand", 3, reference="floor")
    assert env.mean_margin == pytest.approx(floor.mean_margin, rel=1e-12)


def test_an_unknown_reference_name_raises_rather_than_defaulting() -> None:
    with pytest.raises(ValueError, match="floor.*envelope"):
        S.calibration_separation({}, "cand", 3, reference="whatever")


# --------------------------------------------------------------------------
# 3. MUTATION COVERAGE
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SepWorld:
    """Block sets for one observation, plus the planted data transform."""

    transform: Callable[[list[BlockPair]], list[BlockPair]] = lambda pairs: pairs


def _identity(pairs: list[BlockPair]) -> list[BlockPair]:
    return pairs


def _shrink_1000x(pairs: list[BlockPair]) -> list[BlockPair]:
    return [
        replace(p, candidate=p.reference - (p.reference - p.candidate) / 1000.0) for p in pairs
    ]


def _swap_candidate_and_reference(pairs: list[BlockPair]) -> list[BlockPair]:
    return [replace(p, candidate=p.reference, reference=p.candidate) for p in pairs]


def _relabel_blocks(pairs: list[BlockPair]) -> list[BlockPair]:
    return [replace(p, block_id=900 - p.block_id) for p in pairs]


#: The UNMUTATED estimator, captured at import so a planted defect is a
#: transformation OF it rather than a re-implementation.
_REAL_SEPARATION = S.separation


def _no_degenerate_guards(pairs, *, lower_is_better: bool = True) -> S.Separation:
    """PLANTED: the two degenerate guards removed, i.e. a bare ``mean / sd``.

    This is the realistic single-line version of this module: drop the ladder and
    let the division speak. ``0/0`` becomes ``nan`` and ``positive/0`` becomes
    ``inf``, which is precisely how a 0.001 margin acquires infinite
    significance. NOTE, because it matters for reading the coverage map:
    ``Separation.check()`` is a SECOND line of defence and would also refuse this
    -- so this planted version bypasses it deliberately, in order to measure the
    PROBES rather than the invariant.
    """
    sep = _REAL_SEPARATION(pairs, lower_is_better=lower_is_better)
    if sep.sd_margin is None:
        return replace(sep, separation_sd=sep.mean_margin, undefined_reason="")
    if sep.sd_margin == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = float(np.divide(sep.mean_margin, sep.sd_margin))
        return replace(sep, separation_sd=raw, undefined_reason="")
    return sep


def _sep_observe(world: SepWorld) -> dict[str, dict]:
    """Score every case. Calls go through ``S.`` so an INSTRUMENT mutation can
    reach them -- a defect planted only in the data would never test the guards."""
    cases = {
        "known": _pairs(KNOWN_MARGINS),
        "null": _pairs((0.0, 0.0, 0.0, 0.0)),
        "worse": _pairs((-1.0, -2.0, -3.0, -4.0)),
        "split": _pairs(SPLIT_MARGINS),
        "three_blocks": _pairs((2.0, 3.0, 4.0)),
        "single_block": _pairs((3.0,)),
        "zero_sd": _pairs((2.0, 2.0, 2.0, 2.0)),
        "barred_ellipse": [
            BlockPair(block_id=b, candidate=c, reference=f) for b, c, f in BARRED_ELLIPSE_3H
        ],
    }
    return {
        name: S.conditions(S.separation(world.transform(list(pairs))))
        for name, pairs in cases.items()
    }


PLAYTHROUGH = PT.Playthrough(
    name="g3_calibration_separation",
    build=SepWorld,
    observe=_sep_observe,
    probes=(
        PT.Probe(
            "known_separation_is_exact",
            lambda obs: PT.approximately(
                obs["known"]["separation_sd"], WANT_KNOWN_SEPARATION, tol=1e-12
            ),
            note="mean/sd(ddof=1) of four known margins, to 1e-12. THE probe that sees an "
            "estimator change: ddof, a standard-error form, a dropped pairing.",
        ),
        PT.Probe(
            "a_null_candidate_scores_exactly_zero",
            lambda obs: obs["null"]["separation_sd"] == 0.0
            and not obs["null"]["all_conditions_met"],
            note="the reference-free assertion: a forecast identical to the floor separates by "
            "exactly 0. persistence reproduces it bitwise on the real corpus.",
        ),
        PT.Probe(
            "a_worse_candidate_is_negative",
            lambda obs: (obs["worse"]["separation_sd"] or 0.0) < 0.0,
            note="pins the sign convention. An inverted gate is the failure mode a units error "
            "produces, and this project has already had one.",
        ),
        PT.Probe(
            "a_split_result_fails_unanimity",
            lambda obs: obs["split"]["conditions"]["separation"]
            and not obs["split"]["conditions"]["unanimity"],
            note="6 wins and 1 loss over SEVEN blocks clears the SD bar and must still be "
            "refused. Seven and not four on purpose -- see "
            "test_unanimity_is_REDUNDANT_at_four_blocks_and_binding_at_six.",
        ),
        PT.Probe(
            "the_barred_degenerate_ellipse_fails_the_SD_bar_itself",
            lambda obs: not obs["barred_ellipse"]["conditions"]["separation"]
            and not obs["barred_ellipse"]["all_conditions_met"],
            note="ADR-032 (7)'s motivating case with its real numbers: +0.697 SD against a bar "
            "of 2.0, where the ratio form passed it by 0.001. Interrogates the SEPARATION "
            "condition and not only the conjunction, so the bar itself is what is under test.",
        ),
        PT.Probe(
            "a_single_block_comparison_is_refused",
            lambda obs: not obs["single_block"]["all_conditions_met"]
            and obs["single_block"]["separation_sd"] is None,
            note="C-3 verbatim, at the level of the statistic rather than of the reviewer.",
        ),
        PT.Probe(
            "three_blocks_is_refused_on_BLOCK_COUNT_alone",
            lambda obs: obs["three_blocks"]["conditions"]["separation"]
            and obs["three_blocks"]["conditions"]["unanimity"]
            and not obs["three_blocks"]["conditions"]["block_count"],
            note="3.0 SD, unanimous, and REFUSED -- because C6.3 requires >= 4 distinct held-out "
            "blocks. THE probe that makes the block minimum load-bearing rather than decorative.",
        ),
        PT.Probe(
            "a_zero_sd_is_refused_not_infinite",
            lambda obs: obs["zero_sd"]["separation_sd"] is None
            and not obs["zero_sd"]["all_conditions_met"],
            note="the denominator trap: a constant margin must not become infinite significance.",
        ),
        PT.Probe(
            "relative_margin_is_material",
            lambda obs: (obs["known"]["relative_margin"] or 0.0) > 0.01,
            note="THE magnitude probe. A separation test is scale-invariant, so this is the only "
            "thing standing between a real improvement and a statistically-certain trivial one.",
        ),
    ),
    defects=(
        PT.Defect(
            "bar_lowered_to_half_an_sd",
            PT.attribute_defect((S, "MIN_SEPARATION_SD", 0.5)),
            note="the ONE fitted constant, moved. At 0.5 SD the barred degenerate ellipse is "
            "admitted on separation -- which is what makes 2.0 load-bearing rather than "
            "decorative. If nothing caught this, the constant would be a comment.",
        ),
        PT.Defect(
            "block_minimum_lowered_to_one",
            PT.attribute_defect((S, "MIN_BLOCKS_FOR_SEPARATION", 1)),
            note="C6.3's >=4 held-out blocks removed. A gate decided on one spatial block is "
            "ADR-015's failure and C-3's, together.",
        ),
        PT.Defect(
            "the_degenerate_sd_guards_removed",
            PT.attribute_defect((S, "separation", _no_degenerate_guards)),
            note="the ladder that refuses a zero block-to-block SD is deleted, leaving a bare "
            "mean/sd. A one-block comparison starts reporting its raw margin (C-3 gone), a "
            "constant margin becomes INFINITE separation, and a null candidate becomes nan. "
            "This is the single most dangerous edit anyone could make to this file.",
        ),
        PT.Defect(
            "candidate_and_reference_swapped",
            PT.data_defect(lambda w: replace(w, transform=_swap_candidate_and_reference)),
            note="a sign-convention bug -- the exact class that made an insight wrong by a "
            "square root at ADR-032 (4). The winner and the opponent change places.",
        ),
        PT.Defect(
            "margins_shrunk_1000x",
            PT.data_defect(lambda w: replace(w, transform=_shrink_1000x)),
            note="THE RESIDUAL HAZARD OF ANY SEPARATION TEST, planted deliberately. Every margin "
            "becomes three orders of magnitude smaller while the block-to-block SD ratio is "
            "unchanged, so separation_sd does not move AT ALL. Only relative_margin catches it, "
            "and the coverage map is the evidence that reporting magnitude is not optional.",
        ),
        PT.Defect(
            "block_ids_relabelled",
            PT.data_defect(lambda w: replace(w, transform=_relabel_blocks)),
            note="A DECLARED BLIND SPOT, and the right one: the statistic depends on the "
            "PARTITION, not on what the blocks are called. If a probe ever catches this, the "
            "criterion has become sensitive to block numbering and something is badly wrong.",
            detected=False,
        ),
    ),
    note="ADR-032 (7)'s separation criterion. Every answer is arithmetic, and the degenerate "
    "case carries the real numbers from the run that motivated the ruling.",
)


@pytest.fixture(scope="module")
def separation_coverage(playthrough_report) -> PT.PlaythroughReport:
    """Scored once per session; see `tests/conftest.playthrough_report`."""
    return playthrough_report(PLAYTHROUGH)


def test_MUTATION_COVERAGE_every_planted_defect_in_this_file_is_detected(
    separation_coverage: PT.PlaythroughReport,
) -> None:
    print(PT.format_report(separation_coverage))
    separation_coverage.assert_ok()
    assert separation_coverage.mutation_coverage == 1.0


def test_ONLY_the_magnitude_probe_catches_a_trivially_small_margin(
    separation_coverage: PT.PlaythroughReport,
) -> None:
    """The residual hazard, measured. This is the line of the coverage map that
    justifies reporting ``relative_margin`` beside every verdict, and it will go
    red the day someone decides the magnitude column is redundant."""
    caught = {o.name: set(o.caught_by) for o in separation_coverage.outcomes}
    assert caught["margins_shrunk_1000x"] == {"relative_margin_is_material"}
    assert caught["bar_lowered_to_half_an_sd"] == {
        "the_barred_degenerate_ellipse_fails_the_SD_bar_itself"
    }
    assert caught["block_minimum_lowered_to_one"] == {
        "three_blocks_is_refused_on_BLOCK_COUNT_alone"
    }
    assert caught["block_ids_relabelled"] == set()
