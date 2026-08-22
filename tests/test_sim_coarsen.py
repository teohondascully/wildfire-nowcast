"""Unit tests for :mod:`wildfire_nowcast.sim.coarsen`.

Named so that a reader looking for the tests of ``sim/coarsen.py`` finds them by
name. ``sim/`` has 26 modules and, before this file, no name-based route from any
of them to a test, which is the structural reason the mutation sweep left
survivors clustered here.

WHAT THIS FILE PROTECTS
-----------------------
``resolution_limit_probe()`` is published verbatim as the
``resolution_limit_measured`` block of ``reports/figures/playthrough_coarsening.json``,
the artifact that states what a 1 km output cannot represent. That statement is
the declared limit of the ELMFIRE comparison: ELMFIRE is run at native 30 m and
only its OUTPUT is coarsened, so the width at which a finger stops surviving the
coarsening is the width at which a G5 table stops being able to see one.

The probe emits, per finger width, both a MEASUREMENT (area retained, connected
component count) and a CLAIM (``representable``). Nothing in the report's verdict
reads the claim, so before this file a wrong claim shipped silently beside a
right measurement. These tests bind the claim to the measurement printed next to
it, which is a stronger property than pinning either number on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.contract import CELL_SIZE_M
from wildfire_nowcast.sim.coarsen import (
    DEFAULT_REFINE,
    OCCUPANCY_THRESHOLD,
    coarsen_occupancy,
    coverage_fraction,
    diagonal_finger_mask,
    n_components,
    resolution_limit_probe,
)

#: The coarse cell edge in km, taken from the contract rather than spelled as a
#: literal. C1 is a 1 km lattice and ``CELL_SIZE_M`` is where that lives, so a
#: change to the grid moves this test instead of leaving it quietly wrong.
COARSE_CELL_KM = CELL_SIZE_M / 1000.0


def _measured_survival(row: dict[str, object]) -> bool:
    """Did the finger survive the coarsening, according to the row's OWN numbers?

    A finger is one connected shape. The failure mode the probe exists to expose
    is fragmentation: a sub-cell-width finger straddles cell boundaries, some
    blocks clear the 0.5 occupancy threshold and some do not, and one shape
    coarsens into several. So "survived" is "still exactly one component", read
    from the measurement rather than restated as a threshold on the width.
    """
    return int(row["components"]) == 1  # type: ignore[call-overload]


def test_the_probe_reports_a_sweep_with_both_outcomes_in_it() -> None:
    """Anti-vacuity. An agreement test over a one-sided sweep agrees with nothing.

    If every row were representable, or none were, the agreement assertion below
    would hold for a probe that had stopped measuring anything. This is the check
    that the corpus under test still contains both answers.
    """
    rows = resolution_limit_probe()
    assert len(rows) >= 3, f"the resolution sweep collapsed to {len(rows)} widths"
    flags = {bool(r["representable"]) for r in rows}
    assert flags == {True, False}, (
        f"the sweep is one-sided ({flags}); an agreement assertion over it cannot fail"
    )
    survivals = {_measured_survival(r) for r in rows}
    assert survivals == {True, False}, (
        f"every width now behaves the same ({survivals}); the probe has stopped "
        "straddling the resolution limit it exists to locate"
    )


def test_representable_agrees_with_the_measurement_printed_beside_it() -> None:
    """The published claim must match the published evidence, row by row.

    Failure condition, in one sentence: any finger width at which the coarsened
    mask is still a single connected component while ``representable`` says
    false, or is fragmented while ``representable`` says true. A width exactly
    equal to the 1 km coarse cell is that input today, because it coarsens to one
    component with 100 percent of its area retained.
    """
    rows = resolution_limit_probe()
    disagreements = [
        (r["finger_width_km"], r["representable"], r["components"], r["area_retained"])
        for r in rows
        if bool(r["representable"]) is not _measured_survival(r)
    ]
    assert not disagreements, (
        "resolution_limit_probe publishes a representability claim that contradicts "
        f"the measurement on the same row: {disagreements}. Each tuple is "
        "(width_km, representable, components, area_retained)."
    )


def test_the_representability_boundary_sits_at_the_COARSE_CELL_SIZE() -> None:
    """The limit is a property of the lattice, not a tuned constant.

    The module's claim is that a finger narrower than a coarse cell is not
    recoverable by any binary rule at that resolution. That fixes the boundary at
    the cell edge exactly, so the narrowest representable width in the sweep must
    equal the C1 cell size, and the widest non-representable one must be below
    it. An off-by-one comparison moves the boundary to the next width in the
    sweep and is caught here even if the sweep's widths change.
    """
    rows = resolution_limit_probe()
    representable = [float(r["finger_width_km"]) for r in rows if r["representable"]]
    lost = [float(r["finger_width_km"]) for r in rows if not r["representable"]]
    assert representable and lost, "one side of the boundary is empty"
    assert min(representable) == pytest.approx(COARSE_CELL_KM), (
        f"the narrowest representable finger is {min(representable)} km against a "
        f"{COARSE_CELL_KM} km coarse cell; the declared limit and the measured one disagree"
    )
    assert max(lost) < COARSE_CELL_KM, (
        f"a finger of {max(lost)} km is reported as lost at a {COARSE_CELL_KM} km cell size"
    )


def test_a_finger_exactly_one_cell_wide_really_does_survive() -> None:
    """The boundary case, measured directly rather than read out of the report.

    This is the input the two tests above turn on, so it is worth building it
    from the geometry helpers instead of trusting the probe to have built it. If
    a future change to ``coarsen_occupancy`` fragments a one-cell finger, the
    agreement tests would still pass by moving the claim; this one would not.
    """
    fine = diagonal_finger_mask(
        26,
        26,
        DEFAULT_REFINE,
        cx_km=13.0,
        cy_km=13.0,
        length_km=18.0,
        width_km=COARSE_CELL_KM,
        angle_deg=20.0,
    )
    coarse = coarsen_occupancy(fine, DEFAULT_REFINE)
    assert n_components(coarse) == 1, (
        "a finger exactly one coarse cell wide fragmented under the occupancy rule"
    )
    assert float(np.count_nonzero(coarse)) == pytest.approx(18.0 * COARSE_CELL_KM, rel=0.05)


def test_the_occupancy_rule_is_the_threshold_it_declares() -> None:
    """``coarsen_occupancy`` marks a cell burned iff coverage is at or above 0.5.

    The threshold is truth's own (``data/rasterize.COVER_THRESHOLD``). A cell
    filled to exactly the threshold must burn: scoring a baseline under a rule
    one hair stricter than the rule that made the labels is a systematic area
    bias with no visible cause in any table.
    """
    refine = 4
    fine = np.zeros((refine, 2 * refine), dtype=bool)
    # left cell: exactly half its sub-cells covered. right cell: one short of half.
    fine[:2, :refine] = True
    flat = fine[:, refine:].reshape(-1)
    flat[: (refine * refine) // 2 - 1] = True
    fine[:, refine:] = flat.reshape(refine, refine)

    frac = coverage_fraction(fine, refine)
    assert frac[0, 0] == pytest.approx(OCCUPANCY_THRESHOLD)
    assert frac[0, 1] < OCCUPANCY_THRESHOLD

    coarse = coarsen_occupancy(fine, refine)
    assert bool(coarse[0, 0]), "a cell covered to exactly the threshold was dropped"
    assert not bool(coarse[0, 1]), "a cell below the threshold was burned"
