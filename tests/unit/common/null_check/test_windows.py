"""``common/null_check/windows.py`` - the evaluation windows C6.0 scores against.

Two things are checked. The lead alignment, which is C1.3's end-of-hour
convention made concrete: lead ``k`` of a window at ``t0`` is the label at
``t0 + 1 + k``, and a window that quietly starts one hour late scores every
forecaster against the wrong frame while every shape stays plausible. And the
generated scenario itself, which is a reference fixture rather than test data:
its zero-growth rate is a DECLARED input, and the C6.0 verdicts on the record
were produced against these exact numbers.
"""

from __future__ import annotations

import numpy as np

from wildfire_nowcast.common.null_check.windows import synthetic_windows, windows_from_tensor
from wildfire_nowcast.common.synthetic import SyntheticFire
from wildfire_nowcast.common.zarr_io import open_tensor


def test_lead_k_of_a_window_is_the_label_at_t0_plus_one_plus_k(
    default_synthetic: SyntheticFire,
) -> None:
    """The end-of-hour convention, asserted against the store the window came from.

    ``truth`` starting at ``t0 + 2``, or running one frame long, leaves every
    array 3-D and every dtype right. The forecaster is then scored against a
    future it was never shown, and the resulting number is a real number about
    the wrong question.
    """
    horizon = 3
    state = np.asarray(open_tensor(default_synthetic.tensor_path)["fire_state"].values)
    windows, stats = windows_from_tensor(default_synthetic.tensor_path, horizon_h=horizon)

    assert windows, "no windows were built, so nothing below is being checked"
    assert stats["horizon_h"] == horizon
    for window in windows:
        assert window.truth.shape[0] == horizon, (
            f"window at t0={window.t0} carries {window.truth.shape[0]} leads, not {horizon}"
        )
        assert np.array_equal(window.x0, state[window.t0])
        for k in range(horizon):
            assert np.array_equal(window.truth[k], state[window.t0 + 1 + k]), (
                f"lead {k} of the window at t0={window.t0} is not the label at "
                f"t0+1+{k}={window.t0 + 1 + k}"
            )

    # The control: consecutive frames are not all identical here, so the
    # comparison above can distinguish one offset from another.
    assert any(not np.array_equal(w.x0, w.truth[-1]) for w in windows)


def test_a_stride_skips_start_times_without_moving_the_leads(
    default_synthetic: SyntheticFire,
) -> None:
    """The stride is over ``t0`` only; the horizon after each ``t0`` is unchanged."""
    every, _ = windows_from_tensor(default_synthetic.tensor_path, horizon_h=2)
    strided, _ = windows_from_tensor(default_synthetic.tensor_path, horizon_h=2, stride=3)

    starts = {w.t0 for w in strided}
    assert starts <= {w.t0 for w in every}
    assert len(strided) < len(every), "stride=3 returned as many windows as stride=1"
    assert all(w.truth.shape[0] == 2 for w in strided)


def test_the_generated_scenario_is_the_fixture_the_published_verdicts_used() -> None:
    """A reference fixture that moves silently invalidates every number quoted against it.

    The zero-growth rate is the pathology's magnitude and is an INPUT here, not
    an accident of whichever fire was on disk, so it is pinned together with the
    domain and the window count that produced it. Any of these moving is a
    legitimate change; making it in silence is not.
    """
    windows, stats = synthetic_windows()
    assert stats["grid_shape"] == [40, 40], "the reference domain changed shape"
    assert stats["seed"] == 20260808
    assert (stats["n_windows"], stats["n_leads"]) == (37, 111)
    assert stats["zero_growth_lead_fraction"] == 1.0 / 3.0
    assert windows[0].x0.shape == (40, 40)
    assert all(w.truth.shape == (3, 40, 40) for w in windows)


def test_the_generated_labels_are_absorbing_so_the_fixture_is_legal_state_data() -> None:
    """C1.1 reproduced in the fixture: a null check on illegal labels checks nothing."""
    windows, _ = synthetic_windows(n_hours=12)
    for window in windows:
        frames = np.concatenate([window.x0[None], window.truth], axis=0)
        burned = frames > 0
        assert np.all(burned[1:] >= burned[:-1]), f"window at t0={window.t0} un-burned a cell"
        assert set(np.unique(frames)).issubset({0, 1, 2})
