"""``common/zarr_io.py`` - the C1 ``(time, channel, y, x)`` view every model reads.

The channel ORDER is a contract, not a convenience: the model indexes axis 1 by
position, so a reordering is silent everywhere and wrong everywhere. What is
checked here is that the materialised stack agrees with the declared index, and
that the one option which changes the numbers a caller gets back cannot be set
by accident.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wildfire_nowcast.common import zarr_io as zio


def test_the_stacked_view_is_indexed_by_the_declared_channel_order(
    synthetic_ds: xr.Dataset,
) -> None:
    """Position in ``CHANNELS`` IS the channel index; nothing else defines it."""
    stacked = zio.stack_channels(synthetic_ds)
    n_time = int(synthetic_ds.sizes["time"])
    assert stacked.shape[:2] == (n_time, len(zio.CHANNELS))

    for name, index in zio.CHANNEL_INDEX.items():
        assert np.array_equal(
            stacked[:, index], zio.channel_values(synthetic_ds, name, dtype=np.float32)
        ), f"channel {name!r} is not at index {index} of the stacked view"

    labelled = zio.to_stacked_dataarray(synthetic_ds)
    assert list(labelled.coords["channel"].values) == list(zio.CHANNELS)
    assert labelled.dims == ("time", "channel", "y", "x")


def test_a_missing_channel_is_named_rather_than_silently_dropped(
    synthetic_ds: xr.Dataset,
) -> None:
    """A short stack would shift every index above the gap."""
    with pytest.raises(KeyError, match="not_a_channel"):
        zio.stack_channels(synthetic_ds, ["fire_state", "not_a_channel"])


def test_the_stack_dtype_cannot_be_set_by_position(synthetic_ds: xr.Dataset) -> None:
    """``stack_channels(ds, channels, np.float64)`` reads like it names a dtype.

    Under a signature that accepts it positionally it does exactly that, and the
    caller silently gets a tensor of a different precision than the one the rest
    of the pipeline is stamped for. Keeping ``dtype`` keyword-only is what turns
    that into an error at the call site instead of a difference in the numbers.
    """
    with pytest.raises(TypeError):
        zio.stack_channels(synthetic_ds, ["fire_state"], np.float64)  # type: ignore[misc]

    # The control: by keyword it works and it does change the dtype, so the
    # assertion above is about the calling convention and not about the argument
    # being rejected on its merits.
    assert zio.stack_channels(synthetic_ds, ["fire_state"], dtype=np.float64).dtype == np.float64
