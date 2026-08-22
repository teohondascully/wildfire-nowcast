"""Four ``eval/`` survivors in the instruments themselves, not in the science.

An instrument defect is worse than a model defect, because the instrument is what
would have told you. All four of these survived a suite of 892 tests.

* ``power.py:378`` ``< -> <=``. A FLAT step between two rungs becomes an
  inversion, so a channel that is merely insensitive is reported as non-monotone.
  Those are different findings with different remedies, and this project has
  measured plateaus on real channels, so the confusion is not hypothetical.
* ``labelfloor.py:502`` ``* -> /``. The published analytic cell count stops
  scaling with the radius. It survives because ``radius=1`` is the only radius
  anything calls, and there ``2 * r`` and ``2 / r`` are both 2. A closed-form
  control that is only ever evaluated at the fixed point of its own defect is not
  a control.
* ``regime_calibration.py:140`` ``+ -> -``. The reconstructed predicted area
  becomes truth MINUS the signed bias, so a model over-predicting by b is
  reported as under-predicting by b. The reconstruction is what fills the cells no
  run emitted, and it stays finite and plausible while reversed.
* ``selftest.py:1865`` ``False -> True``. The flag recording "the oracle raised on
  an unknown window" is initialised to the answer it is trying to measure, so the
  check passes whether or not the oracle raises. It is an anti-vacuity guard made
  vacuous, which is the defect this project has now catalogued in nine places.

The last one cannot be killed by inspecting a healthy run, because a healthy run
sets the flag anyway. It is killed by breaking the thing the flag is about and
demanding the check notice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from wildfire_nowcast.eval.labelfloor import square_dilation_identity
from wildfire_nowcast.eval.power import monotonicity
from wildfire_nowcast.eval.regime_calibration import growth_calibration_of


def _rungs(levels: list[float]) -> list[dict[str, Any]]:
    return [{"truth_distance": float(i), "level": float(v)} for i, v in enumerate(levels)]


def test_a_flat_step_between_rungs_is_not_an_inversion() -> None:
    """A tie is insensitivity, not misordering, and the two must not be merged.

    ``monotonicity`` exists to separate "this channel cannot resolve the
    difference" from "this channel ranks a worse forecast above a better one".
    Counting a zero-width step as wrong-signed collapses the first into the
    second, and the first is the finding this project actually has.
    """
    flat = monotonicity(_rungs([1.0, 2.0, 2.0, 3.0]))

    assert flat["n_inversions"] == 0, (
        f"a flat step was counted as an inversion: {flat['inversions']}. A channel that "
        "plateaus is now indistinguishable from one that orders the rungs backwards."
    )
    assert flat["monotone"] is True
    assert flat["n_points"] == 4

    inverted = monotonicity(_rungs([1.0, 2.0, 1.5, 3.0]))
    assert inverted["n_inversions"] == 1, (
        "a genuinely decreasing step was not counted, so the assertion above would "
        "pass against a detector that never fires"
    )
    assert inverted["monotone"] is False


def test_the_analytic_dilation_count_scales_with_the_radius_and_stays_a_count() -> None:
    """Evaluate the closed form somewhere other than the fixed point of its defect.

    8-connected dilation by ``r`` takes an ``S x S`` square to ``(S+2r) x (S+2r)``,
    so the analytic count is ``(S+2r)^2`` for every r, not only r=1. Asserting it at
    r=2 is what makes the identity an identity rather than a coincidence, and the
    integer check catches the same defect at the default radius.
    """
    at_two = square_dilation_identity(side=21, radius=2)
    assert at_two["dilate_cells_analytic"] == 625, (
        f"analytic dilate count is {at_two['dilate_cells_analytic']} at side=21, r=2; "
        "(21 + 2*2)^2 is 625. The closed form has stopped scaling with the radius."
    )
    assert at_two["dilate_cells"] == 625
    assert at_two["erode_cells"] == 289
    assert at_two["exact"] is True

    at_one = square_dilation_identity(side=21, radius=1)
    assert at_one["dilate_cells_analytic"] == 529
    assert isinstance(at_one["dilate_cells_analytic"], int), (
        "the analytic dilate count is a number of CELLS and came back "
        f"{type(at_one['dilate_cells_analytic']).__name__}, which means it was not "
        "computed by integer arithmetic even where its value happens to be right"
    )
    assert at_one["exact"] is True


def test_the_reconstructed_predicted_area_is_truth_PLUS_the_signed_bias() -> None:
    """Sign the reconstruction, because a reversed one stays finite and plausible.

    The null model ignites nothing, so its signed band-area error IS minus the
    truth. A scored model's predicted area is then truth plus ITS signed error.
    Subtracting instead turns a 1.4x over-prediction into a 0.6x under-prediction
    with no term going negative and nothing to notice downstream.
    """
    entry = {
        "models": {
            "persistence": {"growth_windows": {"band_area_error_bias": -10.0}},
            "kernel": {"growth_windows": {"band_area_error_bias": 4.0}},
        }
    }

    cell = growth_calibration_of(entry, "kernel", "growth_windows")

    assert cell["truth_area_mean"] == pytest.approx(10.0)
    assert cell["pred_area_mean"] == pytest.approx(14.0), (
        f"predicted area reconstructed as {cell['pred_area_mean']}, not 10 + 4. A model "
        "over-predicting by 4 cells is being reported as under-predicting by 4."
    )
    assert cell["reconstructed"] == pytest.approx(1.4)
    assert cell["source"] == "reconstructed"


def test_the_unknown_window_guard_is_itself_asserted_and_not_assumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break the oracle and demand the check say so.

    ``check_window_table_refuses_a_key_collision`` reports three facts, one of
    which is "an unknown window RAISED". Its flag is set inside an except block,
    so on a healthy oracle the flag is True either way and no healthy-path
    assertion can distinguish an initialised-False from an initialised-True.

    So the oracle is replaced with one that serves ANY window happily, which is
    precisely the silent cross-window scoring the check exists to forbid. The
    check must then report ok=False. If it still reports True, the flag was never
    a measurement.
    """
    from wildfire_nowcast.model import noiseoracle

    def _never_raises(
        self: Any,
        x0: np.ndarray,
        static: Any,
        weather: Any,
        n_members: int,
        horizon_h: int,
        seed: int,
        **_: Any,
    ) -> np.ndarray:
        return np.zeros((n_members, horizon_h, *np.asarray(x0).shape), dtype=np.int8)

    monkeypatch.setattr(noiseoracle.NoisyTruthOracle, "predict", _never_raises)

    from wildfire_nowcast.eval.selftest import check_window_table_refuses_a_key_collision

    check = check_window_table_refuses_a_key_collision()

    assert check.values["miss_on_unknown_window_raised"] is False, (
        "the oracle was replaced with one that serves every window, and the check "
        "still recorded that an unknown window raised. The flag is initialised to its "
        "own answer, so it reports True whatever the oracle does."
    )
    assert check.passed is False, (
        "an oracle that happily serves a window it was never given passed the "
        "collision check, so the check cannot fail and is not evidence"
    )
    assert check.values["duplicate_raised"] is True, (
        "the duplicate-key half must still fire, or this test is passing for the wrong reason"
    )
