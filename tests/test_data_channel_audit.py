"""The plausibility audit: the instrument for the defect class the contract
deliberately does not police.

WHAT STANDS BEHIND THIS. The contract checks STRUCTURE. A canopy-cover channel
filled entirely with the NoData sentinel -9999 once scored a clean pass on
forty-two contract checks and was reporting a mean canopy cover of -3085%,
because -9999 is finite, integral and static and therefore satisfies every
structural clause. That fire also carried a third of the train cell-hours, so
one fire's sentinel would have moved the shared normalisation statistics under
two held-out fires that had no defect at all.

``data/audit.py`` is the module written in response to that, and it was at
**zero** line coverage. An audit that never runs against a corrupt array is
indistinguishable from an audit that returns "ok" unconditionally, which is the
same shape as the defect it exists to catch.

THE FIVE FINDINGS EXERCISED BELOW, each against an array that actually carries
the defect rather than a clean one:

1. an exact NoData sentinel, at any magnitude in the known list;
2. a value outside the channel's plausible range;
3. a class value that is not integral, which is what a class raster resampled by
   interpolation looks like;
4. a non-finite cell, which must be a finding and never a pass by default;
5. a dynamic channel that is constant, whose standard deviation is zero and
   which therefore turns every normalised copy of itself into NaN.

The sixth assertion is the one that keeps the other five honest: a clean channel
must score ``ok``. Without it, an audit that flags everything would pass all five.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.data.audit import (
    PLAUSIBLE_RANGES,
    SENTINEL_VALUES,
    audit_channels,
    channel_audit,
)


def _finding_text(report: dict) -> str:
    return " | ".join(report["findings"])


# --------------------------------------------------------------------------
# the control that keeps every other test in this file meaningful
# --------------------------------------------------------------------------


def test_a_clean_channel_scores_ok_with_no_findings() -> None:
    """The negative control. Without it an audit that flags everything passes.

    FAILS WHEN: a plausible range is tightened past real data, which is how a
    check starts firing on correct work and gets silenced rather than
    investigated.
    """
    canopy = channel_audit("canopy_cover", np.full((4, 4), 30.0, dtype=np.float32))
    assert canopy["verdict"] == "ok"
    assert canopy["findings"] == []

    wind = channel_audit("wind_u10", np.linspace(-8.0, 8.0, 32).reshape(2, 4, 4).astype(np.float32))
    assert wind["verdict"] == "ok"

    fuel = channel_audit("fuel_model_id", np.full((4, 4), 101, dtype=np.int32))
    assert fuel["verdict"] == "ok"
    assert fuel["values_present"] == [101]


# --------------------------------------------------------------------------
# 1. sentinels
# --------------------------------------------------------------------------


def test_the_nodata_sentinel_that_once_passed_every_structural_check_is_caught() -> None:
    """The exact defect, reproduced: a canopy channel that is entirely -9999.

    It is finite, it is integral, it is static and it never changes, so it
    satisfies the structural clauses completely. Only an exact-equality test
    against the known sentinel list, or the range check beside it, has anything
    to say about it.

    FAILS WHEN: the sentinel comparison is loosened to a tolerance or an
    inequality, which would start matching legitimate values, or the sentinel
    list stops being consulted, which returns this channel to the state where
    forty-two checks passed on it.
    """
    corrupt = np.full((4, 4), -9999.0, dtype=np.float32)
    report = channel_audit("canopy_cover", corrupt)

    assert report["verdict"] == "suspect"
    assert report["sentinel_hits"] == {"-9999.0": 16}
    assert "sentinel" in _finding_text(report)
    assert report["stats"]["mean"] == -9999.0, "the audit reports the value it saw"


@pytest.mark.parametrize("sentinel", [-9999.0, -32768.0, 255.0, 1e20])
def test_every_listed_sentinel_is_detected_at_its_own_magnitude(sentinel: float) -> None:
    """Four orders of magnitude apart, so a single scaled comparison cannot cover
    them and a partial list is visible.

    FAILS WHEN: the sentinel scan is written as a range rather than a set of
    exact values, at which point the float32 GeoTIFF NoData and the GRIB missing
    value at 1e20 stop being distinguishable from a large legitimate reading.
    """
    assert sentinel in SENTINEL_VALUES
    values = np.full((3, 3), 0.5, dtype=np.float64)
    values[0, 0] = sentinel

    report = channel_audit("canopy_cover", values)
    assert report["sentinel_hits"][repr(sentinel)] == 1
    assert report["verdict"] == "suspect"


def test_a_channel_free_of_sentinels_carries_no_sentinel_block_at_all() -> None:
    """FAILS WHEN: an empty hits dict is attached anyway, so every channel in the
    corpus carries a sentinel key and the presence of one stops being a signal."""
    report = channel_audit("canopy_cover", np.full((4, 4), 42.0))
    assert "sentinel_hits" not in report


# --------------------------------------------------------------------------
# 2. plausible ranges
# --------------------------------------------------------------------------


def test_a_wind_speed_no_surface_analysis_can_produce_is_flagged_with_its_bounds() -> None:
    """The advisory half of the audit, which covers the channels the contract
    does not adjudicate.

    FAILS WHEN: the out-of-range count is computed without the finite mask, so a
    NaN-bearing channel raises or reports a nonsense count, or the reported
    bounds stop matching the ones the count was taken against.
    """
    report = channel_audit("wind_u10", np.full((2, 4, 4), 1000.0, dtype=np.float32))
    assert report["verdict"] == "suspect"
    assert report["range"] == list(PLAUSIBLE_RANGES["wind_u10"])
    assert report["range_authority"].startswith("data-side plausibility")
    assert "32 cells outside" in _finding_text(report)


def test_a_channel_with_no_finite_cell_reports_that_as_a_finding_not_as_a_pass() -> None:
    """An unverifiable range is a finding. Silence here is how an all-NaN channel
    passed dozens of clauses once already.

    FAILS WHEN: the ``stats["min"] is None`` branch falls through to the range
    comparison, where every comparison against NaN is False and the channel
    scores clean.
    """
    report = channel_audit("temp_2m", np.full((2, 2), np.nan))
    assert report["verdict"] == "suspect"
    assert report["stats"]["min"] is None
    assert report["stats"]["n_nonfinite"] == 4
    assert "range is unverifiable" in _finding_text(report)


def test_a_single_non_finite_cell_is_counted_beside_otherwise_valid_statistics() -> None:
    """FAILS WHEN: the statistics are taken over the raw array rather than its
    finite subset, which makes the min and max NaN and hides both the good data
    and the bad cell behind one unreadable summary."""
    values = np.full((2, 2), 300.0)
    values[0, 0] = np.inf
    report = channel_audit("temp_2m", values)
    assert report["stats"]["n_nonfinite"] == 1
    assert report["stats"]["min"] == 300.0 and report["stats"]["max"] == 300.0
    assert "NaN/inf" in _finding_text(report)


# --------------------------------------------------------------------------
# 3. enumerated channels
# --------------------------------------------------------------------------


def test_a_non_integral_class_value_is_flagged_as_an_interpolated_class_raster() -> None:
    """The signature of a categorical channel resampled with the wrong method.

    A fuel class of 101.5 does not exist. Rounding it produces a class that does
    exist, which is why this must be caught on the raw value: the illegal-value
    check alone sees 101 and passes.

    FAILS WHEN: the integrality test is applied after a cast to int, which is
    exactly the operation that destroys the evidence, or is dropped in favour of
    the class-membership check, which this input satisfies.
    """
    values = np.array([[101.5, 102.0], [91.0, 93.0]], dtype=np.float32)
    report = channel_audit("fuel_model_id", values)

    assert report["verdict"] == "suspect"
    assert "non-integral" in _finding_text(report)
    assert report["values_present"] == [91, 93, 101, 102], "the truncated view still passes"


def test_a_class_outside_the_legal_set_is_named_rather_than_merely_counted() -> None:
    """FAILS WHEN: the illegal classes are counted and not listed, which leaves a
    reader unable to tell a single stray 0 from a whole channel written in the
    wrong enumeration."""
    values = np.array([[101, 12345], [91, 101]], dtype=np.int32)
    report = channel_audit("fuel_model_id", values)
    assert report["verdict"] == "suspect"
    assert "12345" in _finding_text(report)


def test_a_binary_mask_carrying_a_third_value_is_flagged() -> None:
    """FAILS WHEN: binary channels lose their {0,1} domain, so a mask written as
    0/255 by an upstream product reads as legal and every barrier cell in the
    fire is 255 times its intended weight after normalisation."""
    report = channel_audit("water_barrier_mask", np.array([[0, 1], [1, 255]], dtype=np.int32))
    assert report["verdict"] == "suspect"
    assert "255" in _finding_text(report)


# --------------------------------------------------------------------------
# 4. degenerate dynamics
# --------------------------------------------------------------------------


def test_a_constant_DYNAMIC_channel_is_flagged_and_a_constant_STATIC_one_is_not() -> None:
    """A zero standard deviation NaNs every normalisation that divides by it,
    but terrain is constant in time on purpose.

    FAILS WHEN: the ``not static`` term is dropped, which makes every flat
    elevation domain suspect and trains people to ignore the finding, or the
    check is dropped entirely, which lets a stuck weather channel through to
    poison the shared normalisation statistics.
    """
    flat_wind = channel_audit("wind_v10", np.full((2, 4, 4), 3.0, dtype=np.float32))
    assert flat_wind["verdict"] == "suspect"
    assert "constant" in _finding_text(flat_wind)

    flat_slope = channel_audit("slope", np.zeros((4, 4), dtype=np.float32))
    assert flat_slope["static"] is True
    assert flat_slope["verdict"] == "ok"


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_one_suspect_channel_makes_the_fire_suspect_and_is_named() -> None:
    """Per-fire aggregation must not average a defect away.

    FAILS WHEN: the aggregate verdict is computed from a majority or a fraction
    of channels rather than from any, which is how thirteen clean channels come
    to outvote the one that is entirely NoData.
    """
    report = audit_channels(
        {
            "canopy_cover": np.full((4, 4), -9999.0, dtype=np.float32),
            "slope": np.full((4, 4), 10.0, dtype=np.float32),
            "elevation": np.full((4, 4), 500.0, dtype=np.float32),
        },
        fire_id="synthetic_fire",
    )
    assert report["verdict"] == "suspect"
    assert report["suspect_channels"] == ["canopy_cover"]
    assert report["n_channels"] == 3
    assert report["fire_id"] == "synthetic_fire"


def test_an_all_clean_fire_aggregates_to_ok_with_an_empty_suspect_list() -> None:
    """FAILS WHEN: the aggregate hardcodes a verdict, which would make the test
    above pass for the wrong reason."""
    report = audit_channels(
        {
            "slope": np.full((4, 4), 10.0, dtype=np.float32),
            "canopy_cover": np.full((4, 4), 30.0, dtype=np.float32),
        }
    )
    assert report["verdict"] == "ok"
    assert report["suspect_channels"] == []
