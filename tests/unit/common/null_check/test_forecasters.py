"""``common/null_check/forecasters.py`` - the reference models C6.0 compares against.

The one checked closely here is ``CalibratedSkillful``, because it is the
reference that decides whether a metric is flagged. C6.2 [v2.8] fits it PER LEAD
for a measured reason: the truth's own band rate runs 0.00055 / 0.00184 / 0.00540
at 1/2/3 h, a 10x ratio where linear phasing would give 3x, so a reference fitted
in total is miscalibrated at every individual lead. That argument is only worth
anything if lead ``k`` actually reads rate ``k``.
"""

from __future__ import annotations

import numpy as np

from wildfire_nowcast.common.null_check.forecasters import (
    CalibratedSkillful,
    fit_calibrated_skillful,
    null_empty,
    null_zero_ignition,
)
from wildfire_nowcast.common.null_check.windows import Window, synthetic_windows


def _window(horizon: int = 3, size: int = 11) -> Window:
    """One window whose truth adds nothing, so only false alarms can claim a cell."""
    x0 = np.zeros((size, size), dtype=np.uint8)
    x0[4:7, 4:7] = 1
    truth = np.repeat(x0[None], horizon, axis=0)
    return Window(t0=0, x0=x0, truth=truth)


def test_the_false_alarm_rate_for_lead_k_is_the_rate_declared_for_lead_k() -> None:
    """Rates that shift by one lead leave every shape, dtype and total intact.

    The reference then over-claims at one horizon and under-claims at another,
    which is read as the METRIC preferring silence rather than as the reference
    being wrong at that lead. A per-lead calibration that does not reach the last
    lead is a total calibration wearing its name.
    """
    window = _window(horizon=3)
    model = CalibratedSkillful(
        recall=0.0,
        false_alarm_by_lead=(0.0, 0.0, 1.0),
        n_windows_fitted=1,
        n_hits_by_lead=(0, 0, 0),
        n_misses_by_lead=(0, 0, 0),
    )
    samples = model(window, 8, np.random.default_rng(0))
    claimed_beyond_x0 = [
        int(np.count_nonzero((samples[:, k] > 0) & ~(window.x0 > 0)[None])) for k in range(3)
    ]

    assert claimed_beyond_x0[0] == 0, "lead 0 claimed cells at a declared rate of zero"
    assert claimed_beyond_x0[2] > 0, (
        f"lead 2 claimed {claimed_beyond_x0[2]} cells at a declared rate of 1.0, so it did not "
        f"read its own entry: per-lead counts were {claimed_beyond_x0}"
    )
    # Claims are absorbing in time, so lead 1 inherits nothing from lead 2 and
    # the middle entry is the control on the two ends.
    assert claimed_beyond_x0[1] == 0


def test_a_lead_beyond_the_fitted_horizon_reuses_the_last_rate() -> None:
    """The clamp is deliberate: a longer window must not index off the end."""
    window = _window(horizon=3)
    model = CalibratedSkillful(
        recall=0.0,
        false_alarm_by_lead=(0.0,),
        n_windows_fitted=1,
        n_hits_by_lead=(0,),
        n_misses_by_lead=(0,),
    )
    samples = model(window, 4, np.random.default_rng(1))
    assert int(np.count_nonzero((samples > 0) & ~(window.x0 > 0)[None, None])) == 0


def test_the_fit_reproduces_the_observed_number_of_new_cells() -> None:
    """C6.2: an uncalibrated baseline is not a distinct baseline, it is a broken one."""
    windows, _ = synthetic_windows(n_hours=16)
    model = fit_calibrated_skillful(windows, recall=0.55)

    assert len(model.false_alarm_by_lead) == max(int(w.truth.shape[0]) for w in windows)
    assert all(0.0 <= fa <= 1.0 for fa in model.false_alarm_by_lead)
    assert model.n_windows_fitted == len(windows)

    # recall*H_k + fa_k*M_k == H_k is the fitting identity; check it holds per lead.
    for hits, misses, fa in zip(
        model.n_hits_by_lead, model.n_misses_by_lead, model.false_alarm_by_lead, strict=True
    ):
        if misses > 0 and hits > 0:
            assert 0.55 * hits + fa * misses == float(hits) or fa == 1.0


def test_the_two_zero_ignition_nulls_ignite_nothing() -> None:
    """ADR-017's null of record: persistence, and the empty forecast."""
    window = _window()
    burned = window.x0 > 0

    persistence = null_zero_ignition(window, 4, np.random.default_rng(0))
    assert np.array_equal(persistence > 0, np.broadcast_to(burned, persistence.shape))

    empty = null_empty(window, 4, np.random.default_rng(0))
    assert not np.any(empty > 0), "the empty forecast claimed a cell"
