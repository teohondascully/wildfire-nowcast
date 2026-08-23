"""Unit tests for :mod:`wildfire_nowcast.sim.components`.

WHAT THIS FILE PROTECTS
-----------------------
``ignition_components()`` is the instrument behind two published things.

1. **C2 ``n_ignition_components``**, a contract key since v2.7, which
   ``docs/interfaces.md`` requires to be DERIVED rather than defaulted and which
   is now stamped into all 21 fire manifests. This module does NOT define that
   integer and since S13 no longer pretends to: it DELEGATES to
   ``data.ignitions.count_ignition_components`` (ADR-019) and publishes the
   result under ``c2_n_ignition_components_derived``. Its own
   ``candidate_separate_ignitions`` is a SECOND, differently-scoped estimate -
   it was the evidence that corrected ``2020_scu``, and it reads 2 where C2
   reads 3 on the fixture below.
2. **The exclusion rule that binds crossings mining.** A jump between ignition
   components must be excluded from spot-event mining, or the long-range spot
   component of the kernel trains on the label provider's filing convention. The
   sharp half is that the rule is right for the 14 km and 46 km jumps and WRONG
   for the 4 km to 6 km ones, which are the spotting events that component
   exists to learn.

Both rest on the classification ladder, and the raw topology count is NOT the
published number: the module deliberately over-counts and then classifies. At
the time this file was written the module measured 11 percent line coverage and
the whole ladder was unexecuted by any test.

THE FIXTURE
-----------
A synthetic fire with an analytically known answer. No tensor store is read, so
the test runs anywhere and its expected values are arithmetic rather than
recorded.

**AND IT ANSWERS FOUR DIFFERENT QUESTIONS WITH FOUR DIFFERENT NUMBERS** (I22,
on the accepted S13 proposal). This file used to say "four detected
components, of which exactly two are candidate separate ignitions", which named
one number wrongly and left the others unnamed:

===========================================  ======  ===================================
quantity                                     value   who computes it
===========================================  ======  ===================================
UNCONNECTED BIRTHS - bodies appearing where     5     ``sim.components``, raw topology,
fire could not have spread to, over all               BEFORE any classification
hours, first-frame seeds included
this page's CANDIDATE IGNITIONS - births        2     ``sim.components``, its own
still detached after                                  ``FRAGMENT_MERGE_WINDOW_H``
``FRAGMENT_MERGE_WINDOW_H``, plus the
primary
**C2's ``n_ignition_components``**              3     ``data.ignitions``, the RATIFIED
                                                      rule (ADR-019); merging
                                                      disqualifies a body however long
                                                      it takes
FINAL-FOOTPRINT components                      4     connected components of the last
                                                      frame; cannot see a merge at all
===========================================  ======  ===================================

The corpus shows the same spread (czu 3 births vs 1 ignition; scu 5 vs 2 vs 3),
so this is not a fixture artefact. ADR-137's sentence, with numbers attached:
**an unlabelled component count is not a number, it is a question.** Every
assertion below names which question it is pinning.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.common.components import label_components
from wildfire_nowcast.sim.components import (
    FRAGMENT_KM,
    SEPARATE_IGNITION_KM,
    ignition_components,
)
from wildfire_nowcast.sim.reader import FireFrames
from wildfire_nowcast.sim.style import PlotGeometry

NY, NX, NT = 40, 60, 16
CELL_M = 1000.0

#: The four seeds, as (first hour, row, first column, last column + 1).
#: Column arithmetic at a 1 km cell IS the separation in km, which is why the
#: expected classifications below are readable off this table.
PRIMARY = (0, 20, 5, 7)
FRAGMENT_MERGED = (0, 20, 9, 11)  # 3 km away, bridged at hour 4, caught by the merge clock
BRIDGE = (4, 20, 7, 9)
FRAGMENT_MUTUAL = (0, 22, 5, 7)  # 2 km away, NEVER merges, caught only by mutual distance
SPOT = (6, 20, 16, 18)  # 6 km from the merged body, never merges
SEPARATE = (10, 20, 45, 47)  # 28 km from anything, never merges


def _synthetic_fire() -> FireFrames:
    """One primary fire, two fragments of it, one spot, and one unrelated fire."""
    state = np.zeros((NT, NY, NX), dtype=np.uint8)
    for t0, row, c0, c1 in (PRIMARY, FRAGMENT_MERGED, BRIDGE, FRAGMENT_MUTUAL, SPOT, SEPARATE):
        state[t0:, row, c0:c1] = 1
    x = np.arange(NX) * CELL_M
    y = np.arange(NY)[::-1] * CELL_M  # north up, C1.4
    geom = PlotGeometry(
        extent=(x[0] - CELL_M / 2, x[-1] + CELL_M / 2, y[-1] - CELL_M / 2, y[0] + CELL_M / 2),
        x_centres=x,
        y_centres=y,
        cell_size_m=CELL_M,
    )
    times = np.array(
        [np.datetime64("2020-08-16T00", "h") + np.timedelta64(t, "h") for t in range(NT)]
    ).astype("datetime64[ns]")
    zeros_t = np.zeros((NT, NY, NX), dtype=np.float32)
    return FireFrames(
        fire_id="synthetic_four_component",
        state=state,
        times=times,
        wind_u=zeros_t,
        wind_v=zeros_t,
        barrier=np.zeros((NY, NX), dtype=bool),
        elevation=np.zeros((NY, NX), dtype=np.float32),
        geom=geom,
        source="synthetic, built in tests/test_sim_components.py",
        attrs={},
    )


#: Spellings this file will accept for the UNCONNECTED-BIRTH count, most
#: recent last. S13 renamed the key to something with no "components" in it -
#: the module's own prose calls them unconnected births - hit the pin below,
#: and REVERTED its rename rather than write in ``tests/``, which is not its
#: package. The rename is accepted (I22) and the pin is what is fixed here: a
#: test may pin a QUANTITY, but a test that pins a SPELLING makes another
#: lead's vocabulary fix cost a cross-package edit, which is how a misleading
#: name survives. Exactly one of these must be present, so a rename cannot
#: quietly become an ADDITION and leave two names for one integer - which is
#: the S13 defect itself, and the reason `n_ignition_components` was deleted
#: from that dict. A third spelling fails here, loudly, and is one line to add.
BIRTH_COUNT_KEYS = ("n_components_detected", "n_unconnected_births")


def _birth_count(result: dict[str, object]) -> int:
    """The unconnected-birth count, under whichever accepted name it carries."""
    present = [k for k in BIRTH_COUNT_KEYS if k in result]
    assert len(present) == 1, (
        f"expected exactly one of {list(BIRTH_COUNT_KEYS)} in the result, found {present}. "
        "Two names for one integer is the S13 defect; zero means the key was renamed to "
        "something this file does not know - add the new spelling to BIRTH_COUNT_KEYS"
    )
    return int(result[present[0]])  # type: ignore[call-overload]


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return ignition_components(_synthetic_fire())


def test_the_fixture_actually_exercises_every_branch_of_the_ladder(
    result: dict[str, object],
) -> None:
    """Anti-vacuity. A ladder test on a fire with one component tests nothing.

    Every one of the four classifications must be reached by this fixture,
    otherwise the assertions below are green because a branch was never entered.
    """
    counts = result["classification_counts"]
    assert counts == {
        "primary": 1,
        "first_frame_fragment": 2,
        "spot_candidate": 1,
        "separate_ignition": 1,
    }, f"the fixture stopped covering the ladder: {counts}"


def test_the_birth_count_the_ignition_counts_and_the_footprint_are_FOUR_NUMBERS(
    result: dict[str, object],
) -> None:
    """Pins WHICH quantity each number is, because they differ on this fixture.

    The old name of this test said "the published count is the CLASSIFIED one
    not the topology count", and its message called
    ``candidate_separate_ignitions`` "the derived C2 count". It is not: C2's
    count on this same fire is **3**, from the ratified ``data/`` rule, and this
    page's own estimate is **2** because its 12 h merge window disqualifies a
    body the genealogy rule keeps. Both are right about different questions and
    the S13 defect was one of them wearing the other's name.

    Failure condition, in one sentence: any fire whose separation distances stay
    the same but whose classification changes, for instance a first-frame
    fragment 3 km from the primary being read as a separate ignition because the
    separation is taken from the distance to already-burning cells (infinite in
    the ignition frame) rather than from the smaller of that and the distance to
    the other blobs in the same frame.
    """
    assert _birth_count(result) == 5, (
        "UNCONNECTED BIRTHS moved. This is the raw topology count - first-frame bodies "
        "plus later detached ones, before any classification - and it is neither C2's "
        f"ignition count nor the footprint count: {result}"
    )
    assert result["candidate_separate_ignitions"] == 2, (
        "THIS PAGE's candidate-ignition count moved. It is 1 (the primary) plus the "
        "components classified separate_ignition under this module's own "
        f"FRAGMENT_MERGE_WINDOW_H, and this fire has exactly one of those: {result}"
    )
    assert result["c2_n_ignition_components_derived"] == 3, (
        "C2's n_ignition_components moved on this fixture. It is DELEGATED to "
        "data.ignitions.count_ignition_components (ADR-019), so this reads 3 where the "
        "line above reads 2: the 3 km first-frame pair is two seeds under "
        "SEED_MERGE_KM = 2.25 while the 6 km spot is inside SPOT_RANGE_MAX_KM and is not "
        "counted. If this fails and nothing in sim/ changed, a threshold in data/ moved "
        "and that belongs to that package to declare, not to this file to absorb"
    )
    footprint, n_footprint = label_components(_synthetic_fire().state[-1] != 0)
    assert n_footprint == 4, (
        f"the FINAL-FOOTPRINT component count moved to {n_footprint}. It is the estimand "
        "ADR-019 retired for C2 - it cannot see a merge - and it is pinned here only so "
        f"that all four numbers stay visibly different: {footprint.max()}"
    )


def test_a_four_to_six_km_jump_is_a_SPOT_CANDIDATE_and_is_not_excluded(
    result: dict[str, object],
) -> None:
    """The half of the exclusion rule that is easy to get wrong in the costly direction.

    A 6 km detached body must classify as ``spot_candidate``. If it classified as
    ``separate_ignition`` it would be excluded from crossings mining, and the
    events excluded would be exactly the long-range spotting the kernel's spot
    component exists to learn, which makes the spot gate meaningless while every
    check still reads green.
    """
    comps = {c["index"]: c for c in result["components"]}  # type: ignore[union-attr]
    spot = comps[4]
    assert spot["ignition_hour"] == SPOT[0]
    assert spot["km_to_nearest_burning"] == pytest.approx(6.0)
    assert spot["classification"] == "spot_candidate", (
        f"a {spot['km_to_nearest_burning']} km detached body classified as "
        f"{spot['classification']}, between thresholds {FRAGMENT_KM} and {SEPARATE_IGNITION_KM}"
    )
    assert comps[5]["classification"] == "separate_ignition"
    assert comps[5]["km_to_nearest_burning"] == pytest.approx(28.0)


def test_a_first_frame_fragment_is_separated_by_MUTUAL_distance_not_by_infinity(
    result: dict[str, object],
) -> None:
    """Nothing is burning in the ignition frame, so the obvious distance is useless.

    ``km_to_nearest_burning`` is infinite for everything that appears at hour 0,
    which made two adjacent first-frame pieces of one fire indistinguishable from
    a 46 km separate fire. ``mutual_km_at_ignition`` is the field that fixed it,
    and the classifier must take the smaller of the two.
    """
    comps = {c["index"]: c for c in result["components"]}  # type: ignore[union-attr]
    frag = comps[2]
    assert frag["ignition_hour"] == 0
    assert not np.isfinite(frag["km_to_nearest_burning"]), (
        "something was burning before hour 0, so this fixture no longer probes the case"
    )
    assert frag["mutual_km_at_ignition"] == pytest.approx(3.0)
    assert frag["hours_until_merge"] == BRIDGE[0]
    assert frag["classification"] == "first_frame_fragment", (
        "a piece of the primary fire, 3 km away and merged after 4 h, is being "
        f"published as {frag['classification']}"
    )


def test_the_merge_clock_is_measured_and_a_body_that_never_merges_says_so(
    result: dict[str, object],
) -> None:
    """``hours_until_merge`` is the discriminator, so ``None`` must mean never.

    A default of 0 or a silent coercion here would classify every detached body
    as a fragment of the fire it never touched.
    """
    comps = {c["index"]: c for c in result["components"]}  # type: ignore[union-attr]
    assert comps[2]["hours_until_merge"] == BRIDGE[0]
    assert comps[3]["hours_until_merge"] is None
    assert comps[4]["hours_until_merge"] is None
    assert comps[5]["hours_until_merge"] is None
    assert result["merged_hour"] is None, (
        "the per-hour component count never returns to 1 on this fire, because one "
        "fragment never rejoins, so there is no merged hour to report"
    )
    assert result["max_components_at_once"] == 4


def test_a_never_merging_fragment_is_saved_by_the_MUTUAL_distance_alone(
    result: dict[str, object],
) -> None:
    """The case the merge clock cannot reach, and the reason the field was added.

    Failure condition, in one sentence: a first-frame piece of the primary fire
    that never rejoins it, whose distance to already-burning cells is infinite
    because nothing was burning yet, and which therefore reads as an infinitely
    separated fire unless the classifier takes the SMALLER of that distance and
    the distance to the other blobs in the same frame.
    """
    comps = {c["index"]: c for c in result["components"]}  # type: ignore[union-attr]
    stranded = comps[3]
    assert stranded["ignition_hour"] == 0
    assert not np.isfinite(stranded["km_to_nearest_burning"])
    assert stranded["hours_until_merge"] is None, (
        "this fragment is meant to never merge, so the merge clause cannot rescue it "
        "and only the mutual distance can"
    )
    assert stranded["mutual_km_at_ignition"] == pytest.approx(FRAGMENT_KM), (
        "the fixture no longer sits at the declared label-noise scale, so it stops "
        "probing the threshold it was built for"
    )
    assert stranded["classification"] == "first_frame_fragment", (
        f"a blob {stranded['mutual_km_at_ignition']} km from the primary in the "
        f"ignition frame was published as {stranded['classification']}"
    )
