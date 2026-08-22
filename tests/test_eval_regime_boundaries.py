"""The boundaries that define a REGIME, asserted at the exact value they split on.

Every gate in this project reports regime-stratified, so the dormant/growth split
is upstream of most published numbers. The three ``eval/validity.py`` survivors
below all live on that split or on the guard beside it, and none of them changes
a number by a little: each one empties a stratum or inverts a verdict.

* ``validity.py:133`` ``<= -> <``. Dormancy is defined by ``truth_new_cells == 0``,
  and counts are non-negative, so ``< 0`` empties the dormant stratum entirely.
  ``dormant_off_rate`` then goes None and the OFF-state measurement disappears
  rather than failing.
* ``validity.py:207`` ``0 -> 1``. A window that grows by exactly ONE cell stops
  being a growth window. At 1 km cells that is a real hour of a real fire, and the
  stratum it silently leaves is the one G3's calibration criterion is computed on.
* ``validity.py:274`` ``not -> ``. The non-finite guard inverts, so every FINITE
  baseline is declared VOID. A gate resting on it is then voided by a healthy
  opponent, which reads as caution and is the opposite of it.

Plus the two ``eval/masks.py`` survivors, which define what is being scored at all:

* ``masks.py:77`` ``== -> !=``. ``event_field(state, "burned_out")`` raises instead
  of returning the indicator, so the burned-out event becomes unaskable.
* ``masks.py:97`` ``4.0 -> 6.001``. The growth band silently widens by ~1.75x at
  every horizon. A wider band adds unburned cells that no model was ever going to
  ignite, which moves every band-restricted score without moving any forecast.

The rows here are hand-built rather than sampled, so the expected counts are
arithmetic rather than an appeal to a fixture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from wildfire_nowcast.eval.masks import default_band_radius, event_field
from wildfire_nowcast.eval.validity import baseline_validity, off_state_verdict


def _row(truth: float, mean: float, *, ignited: bool | None = None) -> dict[str, Any]:
    """One :func:`window_ignition_counts`-shaped row, with nothing inferred."""
    return {
        "mean_new_cells": mean,
        "max_new_cells": mean,
        "min_new_cells": mean,
        "truth_new_cells": truth,
        "any_member_ignited": bool(mean > 0) if ignited is None else ignited,
    }


def test_a_window_with_exactly_zero_truth_growth_is_DORMANT_and_not_dropped() -> None:
    """Zero is the definition of dormant, so the comparison must include it.

    ``truth_new_cells`` is a count of cells and cannot be negative, so a strict
    ``< 0`` does not narrow the stratum, it deletes it. The failure is silent:
    the key stays present and reads None.
    """
    rows = [_row(0.0, 0.0), _row(0.0, 3.0), _row(20.0, 18.0)]

    off_state = baseline_validity(rows, name="probe")["off_state"]

    assert off_state["n_dormant_windows"] == 2, (
        "windows with truth_new_cells == 0.0 were not counted as dormant. Counts are "
        "non-negative, so a strict comparison empties the stratum instead of narrowing it."
    )
    assert off_state["n_growth_windows"] == 1
    assert off_state["dormant_off_rate"] == pytest.approx(0.5)
    assert off_state["predicted_cells_in_dormant_windows"] == pytest.approx(3.0)


def test_a_window_that_grows_by_exactly_one_cell_is_a_GROWTH_window() -> None:
    """The smallest observable growth must be on the growth side of the split.

    One cell at 1 km is the finest increment the labels can express. If the
    threshold moves to ``> 1`` that hour joins the dormant stratum, where a model
    predicting nothing is rewarded for it.
    """
    rows = [_row(0.0, 0.0), _row(0.0, 0.0), _row(1.0, 2.0)]

    verdict = off_state_verdict(rows)

    assert verdict["n_growth_windows"] == 1, (
        "a window whose truth grew by exactly one cell was not counted as growing. "
        "The growth stratum now starts at two cells and nothing says so."
    )
    assert verdict["n_dormant_windows"] == 2
    assert verdict["false_off_rate"] == pytest.approx(0.0)
    assert verdict["dormant_off_rate"] == pytest.approx(1.0)


def test_a_finite_baseline_is_JUDGED_and_only_a_non_finite_one_is_VOID() -> None:
    """Both sides of the finiteness guard, because one side alone passes it inverted.

    Inverted, the guard voids every baseline whose ignition count is a number,
    which is all of them. A gate resting on a voided baseline reports VOID, and
    VOID reads like conservatism, so nobody goes looking.
    """
    healthy = [_row(10.0, 9.0), _row(0.0, 0.0), _row(20.0, 21.0)]

    verdict = baseline_validity(healthy, name="healthy")
    assert verdict["verdict"] == "OK", (
        f"a finite, well-calibrated baseline was judged {verdict['verdict']!r}. The "
        "non-finite guard is inverted: it fires on exactly the inputs it should pass."
    )
    assert verdict["gate_voided"] is False

    broken = [_row(10.0, float("inf")), _row(20.0, 1.0)]
    voided = baseline_validity(broken, name="broken")
    assert voided["verdict"] == "VOID", (
        "a non-finite ignition count was not voided, so the guard is not doing "
        "anything and the assertion above would pass against a deleted branch"
    )
    assert voided["gate_voided"] is True


def test_the_burned_out_event_is_answerable_and_only_an_unknown_event_raises() -> None:
    """``burned_out`` is one of the three declared events, so it must return a mask.

    With the comparison flipped it falls through to the ValueError meant for
    unknown names, and the state-2 event becomes unaskable while the error message
    still lists it as one of the expected values.
    """
    state = np.array([[0, 1, 2], [2, 0, 1]])

    burned_out = event_field(state, "burned_out")

    assert burned_out.dtype == np.dtype(bool)
    np.testing.assert_array_equal(burned_out, state == 2)
    np.testing.assert_array_equal(event_field(state, "burning"), state == 1)
    np.testing.assert_array_equal(event_field(state, "burned"), state > 0)

    with pytest.raises(ValueError, match="unknown event"):
        event_field(state, "smouldering")


def test_the_growth_band_is_four_cells_per_hour_and_scales_with_the_horizon() -> None:
    """Pin the documented head-rate ceiling to the number the docstring argues for.

    The band is deliberately generous, and its width is a scoring decision rather
    than a tuning knob: widening it adds unburned cells no model would reach,
    which flatters or punishes every band-restricted score at once without any
    forecast changing.
    """
    assert default_band_radius(1) == 4, "the 1 h band is no longer 4 cells, i.e. 4 km"
    assert default_band_radius(2) == 8
    assert default_band_radius(3) == 12

    assert default_band_radius(1, cells_per_hour=6.0) == 6, (
        "the keyword no longer overrides the default, so the two cannot be compared"
    )
    assert default_band_radius(0) == 4, "a non-positive horizon is floored at one hour"
