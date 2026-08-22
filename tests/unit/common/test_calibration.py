"""``common/calibration.py`` - G3's calibration criterion assembled from pooled statistics.

The criterion is the WORST subgroup family, and the counts that travel beside it
say how much of the domain it was computed over. Those counts are what a reader
uses to decide whether a small number is calibration or is one occupied bin, so
a count that over-reports is a claim about sample size that nothing else checks.
"""

from __future__ import annotations

import pytest

from wildfire_nowcast.common.calibration import Stratum, terms_from_strata


def test_the_occupied_counts_count_occupied_strata_and_nothing_else() -> None:
    """An empty stratum contributes no cells, so it is not an occupied one.

    ``calibration_n_occupied_bins`` is published per lead and is how a reader
    tells a criterion computed over ten subgroups from one computed over two. An
    inflated count reads as more support than the number has, and every other
    field in the payload stays perfectly consistent while it does.
    """
    bins = [
        Stratum(key=0, n=0, sum_p=0.0, sum_y=0.0),
        Stratum(key=1, n=4, sum_p=1.0, sum_y=1.0),
        Stratum(key=2, n=0, sum_p=0.0, sum_y=0.0),
    ]
    rings = [Stratum(key=0, n=6, sum_p=3.0, sum_y=3.0), Stratum(key=1, n=0, sum_p=0.0, sum_y=0.0)]

    terms = terms_from_strata(bins, rings)
    assert terms.n_occupied_bins == 1, (
        f"n_occupied_bins={terms.n_occupied_bins} against exactly one bin holding cells"
    )
    assert terms.n_occupied_rings == 1
    assert terms.n_scored == 4, "n_scored counts CELLS, not strata"

    payload = terms.as_dict()
    assert payload["calibration_n_occupied_bins"] == 1
    assert payload["calibration_n_occupied_rings"] == 1

    # The control: an occupied stratum does move the count, so the assertions
    # above are reading the filter and not a constant.
    more = terms_from_strata([*bins, Stratum(key=3, n=2, sum_p=1.0, sum_y=1.0)], rings)
    assert more.n_occupied_bins == 2


def test_the_criterion_is_the_worst_of_the_two_families() -> None:
    """ADR-020: the gate criterion is the worst subgroup family by definition."""
    bins = [Stratum(key=0, n=10, sum_p=1.0, sum_y=1.0)]
    rings = [Stratum(key=0, n=10, sum_p=1.0, sum_y=6.0)]

    terms = terms_from_strata(bins, rings)
    assert terms.bins == pytest.approx(0.0)
    assert terms.frontier == pytest.approx(0.5)
    assert terms.error == pytest.approx(0.5), "the criterion took the better family"


def test_a_missing_frontier_term_is_reported_rather_than_left_blank() -> None:
    """Without x0 only the forecast's own bins are checked, which climatology passes."""
    terms = terms_from_strata([Stratum(key=0, n=10, sum_p=1.0, sum_y=1.0)], None)
    assert terms.frontier is None
    assert "frontier" in terms.unavailable_reason
    assert terms.block_dict()["calibration_unavailable_reason"] == terms.unavailable_reason
