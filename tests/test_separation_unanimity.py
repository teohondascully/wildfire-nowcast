"""The unanimity conjunct of the G3 separation criterion is VACUOUS at n=5.

Routed to infra by modelling, which found it and did not patch it because
`common/` is not its file. Reproduced here by SEARCH rather than relayed: a
brute-force claim that is only ever re-read is a claim, and this repository has
spent forty decisions on the difference.

The statement. `separation` requires `|mean|/sd >= MIN_SEPARATION_SD` across
blocks AND that every block favour the candidate. But a block set that is NOT
unanimous cannot push `|mean|/sd` above `(n-1)/sqrt(n)`, so whenever the bar sits
above that ceiling the second conjunct excludes nothing the first admits. At the
five held-out blocks the corpus holds, the ceiling is 1.788854 against a bar of
2.0.

NO VERDICT MOVES. That is asserted below rather than reasoned about, because "a
vacuous conjunct cannot change a conjunction" is exactly the kind of sentence
that is obviously true and occasionally wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.separation import (
    MIN_BLOCKS_FOR_SEPARATION,
    MIN_SEPARATION_SD,
    BlockPair,
    conditions,
    max_separation_without_unanimity,
    separation,
    unanimity_binds,
    unanimity_first_binds_at,
)

#: The bar ADR-133 (ii) examined, kept beside the shipped one so the general
#: statement - bar and block count decide this JOINTLY - is exercised at two bars
#: rather than asserted from one.
ADR_133_BAR = 2.536


def _stat(values: np.ndarray) -> float:
    sd = float(np.std(values, ddof=1))
    return float(abs(np.mean(values)) / sd) if sd > 0 else float("inf")


def _search_max_non_unanimous(n: int, trials: int, seed: int) -> tuple[float, np.ndarray]:
    """Largest ``|mean|/sd`` found over samples that are NOT strictly one-signed."""
    rng = np.random.default_rng(seed)
    best, arg = 0.0, np.zeros(n)
    magnitudes = np.abs(rng.normal(size=(trials, n)))
    flips = rng.integers(1, n, size=trials)
    for i in range(trials):
        candidate = magnitudes[i].copy()
        candidate[: flips[i]] *= -1.0
        value = _stat(candidate)
        if value > best:
            best, arg = value, candidate.copy()
    return best, arg


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 9, 11])
def test_the_ceiling_on_a_non_unanimous_block_set_is_found_by_SEARCH(n: int) -> None:
    """Random search must never beat the closed form, and must approach it.

    Both directions matter. If search EXCEEDED the formula the formula is wrong;
    if search fell far short of it, the formula might be a loose bound and the
    vacuity argument would not follow. It is neither: the extremal configuration
    is one block at zero with the rest equal, and search converges onto it.
    """
    closed = max_separation_without_unanimity(n)
    found, _ = _search_max_non_unanimous(n, trials=20_000, seed=n)

    assert found <= closed + 1e-9, (
        f"search found a non-unanimous block set scoring {found:.6f} against a claimed "
        f"ceiling of {closed:.6f} at n={n}. The ceiling is wrong and every vacuity "
        "statement resting on it must be withdrawn."
    )
    exact = np.array([0.0] + [1.0] * (n - 1))
    assert _stat(exact) == pytest.approx(closed), (
        "the extremal configuration named in the docstring does not reproduce the closed form"
    )

    # TIGHTNESS, and its honest scope. Attainment is settled ABOVE by exhibiting
    # the extremal configuration at every n; what search adds is independent
    # evidence that the bound is not merely a loose over-estimate. Uniform random
    # sampling reaches it only at small n - measured ratios are 1.000 / 0.993 /
    # 0.975 / 0.973 at n = 3/4/5/6 and fall to 0.688 by n = 11, because the
    # extremal region shrinks as the dimension grows. That is a property of the
    # SAMPLER, not of the bound, so the assertion is made only where the sampler
    # is capable - which is also exactly where the operational claim lives, since
    # this gate runs at 4 and 5 blocks and the conjunct goes live at 6.
    if n <= 6:
        assert found > 0.95 * closed, (
            f"search only reached {found:.6f} of a {closed:.6f} ceiling at n={n}, where it "
            "has been measured to converge. The ceiling may be loose rather than attained, "
            "and 'cannot exceed' would be doing work the measurement does not support."
        )


def test_the_conjunct_is_vacuous_at_the_block_counts_this_gate_ACTUALLY_RUNS_AT() -> None:
    """4 is the contractual minimum and 5 is what the corpus holds. Both vacuous."""
    assert not unanimity_binds(MIN_BLOCKS_FOR_SEPARATION)
    assert not unanimity_binds(5)
    assert max_separation_without_unanimity(4) == pytest.approx(1.5)
    assert max_separation_without_unanimity(5) == pytest.approx(1.788854, abs=1e-6)
    assert MIN_SEPARATION_SD == 2.0, (
        "the bar moved; the vacuity statement is about a bar/block-count PAIR and must be "
        "recomputed rather than inherited"
    )


def test_ONE_MORE_BLOCK_makes_it_live_and_that_is_the_actionable_number() -> None:
    """The whole practical content: we hold 5, and 6 would make the conjunct real."""
    assert unanimity_first_binds_at(MIN_SEPARATION_SD) == 6
    assert unanimity_binds(6) and not unanimity_binds(5)
    assert unanimity_first_binds_at(ADR_133_BAR) == 9, (
        "ADR-133 (ii) recorded the same defect at the 2.536 bar; if this stops reading 9 the "
        "two records disagree and one of them is wrong"
    )


def test_a_VACUOUS_conjunct_cannot_change_the_conjunction_and_a_LIVE_one_can() -> None:
    """The claim that licenses 'no verdict moves', with its own positive control.

    At n=5 no sample may clear the bar while failing unanimity - searched, not
    assumed. The control is the SAME search at n=6, which must FIND one: without
    that half, an empty result at n=5 would be indistinguishable from a search
    that cannot find anything anywhere.
    """
    rng = np.random.default_rng(20260825)

    def counterexamples(n: int, trials: int) -> int:
        found = 0
        for _ in range(trials):
            values = rng.normal(loc=rng.uniform(0.0, 4.0), scale=1.0, size=n)
            if _stat(values) >= MIN_SEPARATION_SD and not (
                np.all(values > 0) or np.all(values < 0)
            ):
                found += 1
        return found

    assert counterexamples(5, 200_000) == 0, (
        "a 5-block sample cleared 2.0 SD while not being unanimous, so the conjunct is NOT "
        "vacuous at n=5 and the ceiling argument is refuted"
    )
    assert counterexamples(6, 200_000) > 0, (
        "the search found no counterexample at n=6 either, where one provably exists "
        "(ceiling 2.0412 > 2.0). The n=5 result above is therefore not evidence of "
        "emptiness - it is evidence the search cannot find anything."
    )


def test_conditions_REPORTS_the_vacuity_instead_of_leaving_it_to_be_inferred() -> None:
    """A passing `unanimity: true` at n=5 carried no information and did not say so."""
    pairs = [BlockPair(block_id=i, candidate=0.10, reference=0.40 + 0.01 * i) for i in range(5)]
    out = conditions(separation(pairs))

    assert out["conditions"]["unanimity"] is True
    assert out["unanimity_binds"] is False
    assert out["max_separation_without_unanimity"] == pytest.approx(1.788854, abs=1e-6)
    assert "VACUOUS at 5 block(s)" in out["unanimity_note"]
    assert "first binds at 6 blocks" in out["unanimity_note"]

    six = [BlockPair(block_id=i, candidate=0.10, reference=0.40 + 0.01 * i) for i in range(6)]
    assert conditions(separation(six))["unanimity_binds"] is True


def test_the_report_is_ADDITIVE_and_moves_no_verdict() -> None:
    """The new keys may not touch `conditions` or `all_conditions_met`.

    Checked against a hand-computed expectation rather than against the function's
    own output, so this cannot pass by comparing a value with itself.
    """
    pairs = [BlockPair(block_id=i, candidate=0.10, reference=0.40 + 0.01 * i) for i in range(5)]
    out = conditions(separation(pairs))
    assert set(out["conditions"]) == {"separation", "unanimity", "block_count"}
    assert out["all_conditions_met"] == all(out["conditions"].values())
    assert out["all_conditions_met"] is True

    weak = [BlockPair(block_id=i, candidate=0.39, reference=0.40) for i in range(5)]
    weak_out = conditions(separation(weak))
    assert weak_out["conditions"]["unanimity"] is True
    assert weak_out["conditions"]["separation"] is False
    assert weak_out["all_conditions_met"] is False


# --------------------------------------------------------------------------
# THE DIFFERENTIAL TEST: two implementations of one closed form, held together
#
# `eval.attainable.unanimity_bound_sd` landed the same expression independently
# and within the hour of this module, derived from Cauchy-Schwarz where
# `common.separation` derives it from the extremal configuration. Two routes to
# one formula is corroboration; two COPIES of one formula is the defect C0 names.
# Until the duplicate is collapsed - a PROPOSAL, because the other copy is in
# another lead's package - these assertions are what stop them becoming two
# different definitions of one fact.
# --------------------------------------------------------------------------


def test_the_two_implementations_of_the_ceiling_AGREE_at_every_block_count() -> None:
    """Same number from both modules, 2 through 40, to floating-point equality."""
    from wildfire_nowcast.eval.attainable import unanimity_bound_sd

    for n in range(2, 41):
        assert max_separation_without_unanimity(n) == pytest.approx(
            unanimity_bound_sd(n), rel=0, abs=1e-12
        ), (
            f"common.separation and eval.attainable disagree about the unanimity ceiling at "
            f"n={n}. One of them is wrong and every vacuity statement in the tree is quoting "
            "one of them."
        )


@pytest.mark.parametrize("bar", [MIN_SEPARATION_SD, ADR_133_BAR])
def test_the_two_implementations_AGREE_ON_THE_TIE_which_they_describe_oppositely(
    bar: float,
) -> None:
    """The one case where wording could hide a real disagreement.

    `common.separation.unanimity_binds` asks `ceiling >= bar` (binds).
    `eval.attainable.unanimity_range` asks `bar > ceiling + eps` (implied, i.e.
    vacuous). Those are complementary rather than equal, and at `bar == ceiling`
    EXACTLY they must still land on the same side: the extremal configuration puts
    one margin at exactly 0, `blocks_favouring` counts `margin > 0` STRICTLY, so
    the conjunct can still fire and the honest answer is BINDS. A `>=` in the
    other module would have been a one-point vacuity claim that is false, and its
    own comment says so. Asserted here rather than trusted to two comments.
    """
    from wildfire_nowcast.eval.attainable import unanimity_range

    for n in range(2, 41):
        binds_here = unanimity_binds(n, bar)
        implied_there = unanimity_range(n_blocks=n, min_separation_sd=bar).lower.value == float(n)
        assert binds_here is not implied_there, (
            f"at n={n}, bar={bar}: common says binds={binds_here} while eval says the "
            f"unanimity edge is dead={implied_there}. These must be exact complements."
        )

    tie_n = 5
    tie_bar = max_separation_without_unanimity(tie_n)
    assert unanimity_binds(tie_n, tie_bar) is True, (
        "at a bar EQUAL to the ceiling the conjunct must still be reported as binding: the "
        "extremal set puts one margin at exactly zero and unanimity counts strictly"
    )
    assert unanimity_range(n_blocks=tie_n, min_separation_sd=tie_bar).lower.value == 0.0, (
        "the other module must reach the same conclusion at the tie by its own route"
    )
