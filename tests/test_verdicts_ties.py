"""``_best_skill_reference`` must not treat a TIE as an advantage. C6.0's own tie-break.

``common/null_check/verdicts.py:304`` reads ``if _advantage(...) > 0``. The audit
mutated it to ``>=`` and 745 tests passed. That is not a style defect, and it is
worth spelling out why before the assertions:

C6.0 decides whether a metric is SILENCE_FAVOURING by comparing every degenerate
forecaster against *the best forecaster with genuine skill*, and this function
picks which one that is. With ``>=`` the incumbent is displaced by anything that
merely equals it, so the choice among tied references falls to
``sorted(SKILL_REFERENCES)`` - which is alphabetical order, a property of two
strings and of nothing measured. A verdict that quarantines a metric, or declines
to, would then be decided by a name.

It is also the direction that flatters. The reference is the bar a degenerate has
to clear; letting ties move it moves it for reasons unrelated to skill, and the
one thing this harness exists to refuse is a comparison that can be won without
being better. ADR-047 (3) already recorded a control that should have read 0
reading -0.090; a tie-break that is not a tie-break is the same failure one level
down.

Both directions are asserted, because a function that always keeps the incumbent
passes half of this file and is just as wrong.
"""

from __future__ import annotations

import pytest

from wildfire_nowcast.common.null_check.forecasters import SKILL_REFERENCES, SKILLFUL
from wildfire_nowcast.common.null_check.registry import HIGHER, LOWER, TARGET, MetricSpec
from wildfire_nowcast.common.null_check.verdicts import _advantage, _best_skill_reference

#: Sorted, because the tie-break under test is exactly ``sorted(...)`` order: the
#: first name is the incumbent and the second is the challenger a tie must not
#: promote.
_FIRST, _SECOND = sorted(SKILL_REFERENCES)


def test_the_two_skill_references_are_still_two_and_still_ordered() -> None:
    """A one-element reference set would make every assertion below vacuous."""
    assert len(SKILL_REFERENCES) == 2, SKILL_REFERENCES
    assert _FIRST < _SECOND and _FIRST == SKILLFUL


@pytest.mark.parametrize(
    ("direction", "target"),
    [(HIGHER, None), (LOWER, None), (TARGET, 1.0)],
)
def test_a_tie_leaves_the_incumbent_in_place(direction: str, target: float | None) -> None:
    """The mutant. Equal scores are not an advantage, under every orientation.

    ``TARGET`` is included because its advantage is a distance difference, where a
    tie is the easiest thing in the world to produce: two references either side of
    the target at the same distance.
    """
    spec = MetricSpec(direction, False, target=target)
    tied = {_FIRST: 0.5, _SECOND: 0.5}
    assert _advantage(tied[_SECOND], tied[_FIRST], spec) == 0.0
    assert _best_skill_reference(tied, spec) == _FIRST, (
        "a tie promoted the alphabetically later reference. The strongest reference C6.0 "
        "scores every degenerate against would then be chosen by a name rather than by a "
        "measurement, and it would move the bar in the flattering direction."
    )


def test_a_real_advantage_still_moves_the_choice() -> None:
    """The other direction: keeping the incumbent unconditionally is equally wrong."""
    assert _best_skill_reference({_FIRST: 0.4, _SECOND: 0.6}, MetricSpec(HIGHER, False)) == _SECOND
    assert _best_skill_reference({_FIRST: 0.6, _SECOND: 0.4}, MetricSpec(HIGHER, False)) == _FIRST
    assert _best_skill_reference({_FIRST: 0.6, _SECOND: 0.4}, MetricSpec(LOWER, False)) == _SECOND
    assert _best_skill_reference({_FIRST: 0.4, _SECOND: 0.6}, MetricSpec(LOWER, False)) == _FIRST


def test_a_tie_within_floating_point_noise_is_still_a_tie() -> None:
    """The realistic form. Two seeds rarely land on the same bit pattern.

    Recorded rather than repaired: the comparison is exact, so a difference of one
    ULP DOES move the choice, and this test states which side of that line the code
    is on instead of leaving a reader to guess.
    """
    spec = MetricSpec(HIGHER, False)
    assert _best_skill_reference({_FIRST: 0.5, _SECOND: 0.5 + 1e-18}, spec) == _FIRST
    assert _best_skill_reference({_FIRST: 0.5, _SECOND: 0.5000001}, spec) == _SECOND


def test_an_unscored_reference_is_skipped_and_an_empty_set_falls_back() -> None:
    """``None`` is not a score, and it may not become one by being compared."""
    spec = MetricSpec(HIGHER, False)
    assert _best_skill_reference({_FIRST: None, _SECOND: 0.1}, spec) == _SECOND
    assert _best_skill_reference({_FIRST: 0.1, _SECOND: None}, spec) == _FIRST
    assert _best_skill_reference({_FIRST: None, _SECOND: None}, spec) == SKILLFUL
    assert _best_skill_reference({}, spec) == SKILLFUL
