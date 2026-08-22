"""C2 [v2.7] ``n_ignition_components`` - negative controls, adopted from data.

ADOPTION NOTE. These six controls were written and passing in data's A11
transcript and lived nowhere else; its proposal P16 asked infra to adopt them
because it does not own ``tests/``, and ADR-022 (4) accepted. **They are adopted
by REBUILDING them against the module's public API, not by copying an
implementation** - the same standard A7 used for the two self-test modules - and
the classifications asserted here are data's, unchanged. Where its
transcript recorded a control that FAILED and concluded the code was right, that
control is adopted in the form the code says is correct, with the disagreement
kept in the docstring rather than erased.

WHY THIS ONE MATTERS MORE THAN ITS SIZE SUGGESTS. `n_ignition_components` decides
whether a 46 km jump is a FILING ARTIFACT (GOFER separating two lightning
ignitions under one fire id - excluded) or a SPOT EVENT (real long-range signal -
retained). The whole 12-fire corpus holds TWO never-merging spot candidates
against hundreds of 2.0-2.24 km rasterisation holes (R17), so the branch that
tells those apart is the branch G4's only signal survives or dies on. Excluding
all inter-component jumps would delete the signal; including them all would feed
a 46 km artifact into the spot model as if a fire had thrown embers 46 km.

The decisive control is **the 46 km birth that MERGES must return 1** - genealogy
overriding distance. It is the only one of the six where the rule's three tiers
(time, then genealogy, then distance) can disagree with each other, so it is the
only one that can distinguish the ratified rule from "threshold on distance".

A derivation that returns a plausible number on every input is worth nothing;
this file is the evidence it does not (the ADR-012 lesson, in data's words).
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_nowcast.data.ignitions import (
    SEED_MERGE_KM,
    SPOT_RANGE_MAX_KM,
    count_ignition_components,
    label_components,
)

CELL_M = 1000.0
SHAPE = (24, 96)


def _blob(mask: np.ndarray, cy: int, cx: int, r: int = 1) -> np.ndarray:
    out = mask.copy()
    out[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1] = True
    return out


def _states(frames: list[np.ndarray]) -> np.ndarray:
    """Turn a monotone sequence of ever-burned masks into legal C1.1 states.

    1 where a cell is new at this frame, 2 where it burned earlier, 0 otherwise.
    Absorbing by construction, which is the only property
    :func:`count_ignition_components` relies on.
    """
    state = np.zeros((len(frames), *frames[0].shape), dtype=np.uint8)
    for t, ever in enumerate(frames):
        older = frames[t - 1] if t else np.zeros_like(ever)
        assert np.all(ever >= older), "controls must be absorbing (C1.1)"
        state[t] = np.where(ever & ~older, 1, np.where(ever, 2, 0))
    return state


def _grow(ever: np.ndarray, steps: int = 1) -> np.ndarray:
    from wildfire_nowcast.common.states import dilate

    out = ever.copy()
    for _ in range(steps):
        out = dilate(out, 1)
    return out


def _count(frames: list[np.ndarray]):
    return count_ignition_components(_states(frames), cell_size_m=CELL_M)


# --------------------------------------------------------------------------
# the labeller itself - everything below rests on it
# --------------------------------------------------------------------------


def test_label_components_is_8_connected() -> None:
    """Diagonal touching is ONE body: the rasterisation-hole case depends on it."""
    m = np.zeros((5, 5), dtype=bool)
    m[1, 1] = m[2, 2] = True
    _, n = label_components(m)
    assert n == 1

    m2 = np.zeros((5, 5), dtype=bool)
    m2[1, 1] = m2[3, 3] = True
    assert label_components(m2)[1] == 2

    assert label_components(np.zeros((4, 4), dtype=bool))[1] == 0
    with pytest.raises(ValueError, match="2-D"):
        label_components(np.zeros((2, 2, 2), dtype=bool))


# --------------------------------------------------------------------------
# THE SIX CONTROLS (data A11 item 3), in its order
# --------------------------------------------------------------------------


def test_control_1_two_far_first_frame_seeds_are_two_ignitions() -> None:
    """SCU's shape: 29.27 km apart in the FIRST burned frame, later merging.

    Rule (a), TIME: a body in the first burned frame has no antecedent anywhere
    in the record, so it is an ignition regardless of what happens afterwards.
    This is the case a final-footprint component count gets wrong - the two
    merge, so the footprint has one body where there were two fires.
    """
    base = np.zeros(SHAPE, dtype=bool)
    f0 = _blob(_blob(base, 12, 10), 12, 40)  # nearest cells 28 km apart
    frames = [f0]
    for _ in range(16):  # enough dilations to close the 28 km gap
        frames.append(_grow(frames[-1]))

    rep = _count(frames)
    assert rep.n_ignition_components == 2
    assert rep.n_first_frame_seeds == 2
    assert rep.first_frame_seed_separations_km[0] > SEED_MERGE_KM

    # ...and they DO merge, so a footprint count would have said 1.
    assert label_components(frames[-1])[1] == 1


def test_control_2_a_one_cell_hole_is_one_ignition() -> None:
    """July Complex's shape: two first-frame bodies 2.24 km apart = ONE ignition.

    At GOFER's ~2 km effective resolution a one-cell diagonal hole in an
    advancing front is rasterisation noise. The corpus has HUNDREDS of these
    against two real spot candidates, so a rule that counted them would report
    hundreds of ignitions and bury the signal.
    """
    base = np.zeros(SHAPE, dtype=bool)
    f0 = base.copy()
    f0[12, 10] = True
    f0[13, 12] = True  # dy=1, dx=2 -> hypot = 2.236 km <= SEED_MERGE_KM
    frames = [f0, _grow(f0)]

    rep = _count(frames)
    assert rep.n_ignition_components == 1
    assert rep.n_first_frame_seeds == 1
    assert rep.first_frame_seed_separations_km[0] == pytest.approx(2.236, abs=1e-3)
    assert rep.first_frame_seed_separations_km[0] <= SEED_MERGE_KM


def test_control_3_a_46km_never_merging_birth_is_a_second_ignition() -> None:
    """``2020_july_complex``'s real second ignition: 46.10 km at h22, never merges."""
    base = np.zeros(SHAPE, dtype=bool)
    frames = [_blob(base, 12, 8)]
    for _ in range(3):
        frames.append(_grow(frames[-1]))
    frames.append(_blob(frames[-1], 12, 60))  # ~46 cells away
    for _ in range(3):
        frames.append(_grow(frames[-1]))

    rep = _count(frames)
    assert rep.n_ignition_components == 2
    assert rep.n_first_frame_seeds == 1
    births = rep.separate_ignition_births
    assert len(births) == 1
    assert births[0].gap_km > SPOT_RANGE_MAX_KM
    assert not births[0].merges_later
    assert births[0].as_dict()["classified"] == "separate_ignition"
    assert label_components(frames[-1])[1] == 2, "the control must really never merge"


def test_control_4_the_SAME_46km_birth_that_MERGES_is_ONE_ignition() -> None:
    """**THE ONE THAT MATTERS.** Genealogy overrides distance (rule (b)).

    Identical geometry to control 3 - same birth, same hour, same 46 km gap - and
    the ONLY difference is that the gap closes before the end, demonstrating a
    link to the region that preceded it. CZU's 14.14 km body at h25 is our widest
    directly observed instance of this.

    This is the control that separates the ratified rule from a distance
    threshold, and therefore the one that separates a filing artifact from G4's
    only signal. Asserted against control 3 in the same test so the two cannot
    drift apart: if this ever returns 2, distance has silently become the
    decision and every long-range spot in the corpus is about to be reclassified
    as a separate fire.
    """
    base = np.zeros(SHAPE, dtype=bool)
    frames = [_blob(base, 12, 8)]
    for _ in range(3):
        frames.append(_grow(frames[-1]))
    detached_at = len(frames)
    frames.append(_blob(frames[-1], 12, 60))
    # Close the gap: grow both bodies until they touch, so the birth has a
    # demonstrated genealogical link to its predecessor by the final frame.
    for _ in range(24):
        frames.append(_grow(frames[-1]))

    rep = _count(frames)
    assert label_components(frames[-1])[1] == 1, "the control must really merge"
    assert rep.n_ignition_components == 1
    assert rep.n_first_frame_seeds == 1
    assert rep.separate_ignition_births == []

    born = [b for b in rep.detached_births if b.hour == detached_at]
    assert born, "the 46 km birth must still be DETECTED, only reclassified"
    assert born[0].gap_km > SPOT_RANGE_MAX_KM
    assert born[0].merges_later
    assert not born[0].is_separate_ignition


def test_control_5_a_6km_never_merging_birth_is_a_SPOT_not_an_ignition() -> None:
    """SCU's two 5-6 km bodies (h27, h89): reported, never counted.

    Inside the observed genealogy range (14.14 km demonstrated), so it is a spot
    candidate. Counting it would inflate the ignition count; DELETING it would
    remove the signal G4 depends on - the failure mode ADR-017 §7 exists to
    prevent. It must be neither: retained and reported.
    """
    base = np.zeros(SHAPE, dtype=bool)
    frames = [_blob(base, 12, 10)]
    for _ in range(3):
        frames.append(_grow(frames[-1]))
    frames.append(_blob(frames[-1], 12, 22))  # ~6-9 cells clear of the front
    for _ in range(2):
        frames.append(_grow(frames[-1]))

    rep = _count(frames)
    assert label_components(frames[-1])[1] == 2, "the control must really never merge"
    assert rep.n_ignition_components == 1, "a spot is not an ignition"
    spots = rep.spot_candidates
    assert len(spots) == 1
    assert 0 < spots[0].gap_km <= SPOT_RANGE_MAX_KM
    assert spots[0].as_dict()["classified"] == "spot_candidate"
    assert spots[0].as_dict() in rep.to_provenance()["spot_candidates_reported_not_counted"]


def test_control_6_plain_growth_is_one_ignition() -> None:
    """The base case. If this ever returns >1, everything above is meaningless."""
    base = np.zeros(SHAPE, dtype=bool)
    frames = [_blob(base, 12, 20)]
    for _ in range(8):
        frames.append(_grow(frames[-1]))

    rep = _count(frames)
    assert rep.n_ignition_components == 1
    assert rep.n_first_frame_seeds == 1
    assert rep.separate_ignition_births == []
    assert rep.spot_candidates == []


# --------------------------------------------------------------------------
# the seventh: the control data wrote that FAILED, kept as the code's answer
# --------------------------------------------------------------------------


def test_a_two_cell_gap_does_NOT_merge_and_that_is_the_stated_rule() -> None:
    """data expected 2.83 km to merge; the rule says it does not. Rule wins.

    Recorded in its status entry rather than quietly edited: ``SEED_MERGE_KM`` is
    2.25 km, and a two-cell diagonal gap is ``hypot(2, 2) = 2.83 km``, outside it.
    Adopting the control in the form the CODE is right about - with the
    disagreement on the record - is the point of adopting it at all. It also pins
    the boundary from the far side, which none of the six do: control 2 pins
    2.236 <= 2.25, this pins 2.83 > 2.25, and the corpus has no observation
    anywhere in [2.3, 29] km, so nothing real sits between them.
    """
    base = np.zeros(SHAPE, dtype=bool)
    f0 = base.copy()
    f0[12, 10] = True
    f0[14, 12] = True  # hypot(2, 2) = 2.828 km
    rep = _count([f0, _grow(f0)])

    assert rep.first_frame_seed_separations_km[0] == pytest.approx(2.828, abs=1e-3)
    assert rep.first_frame_seed_separations_km[0] > SEED_MERGE_KM
    assert rep.n_ignition_components == 2


# --------------------------------------------------------------------------
# the guards around the count
# --------------------------------------------------------------------------


def test_a_fire_that_never_burns_raises_rather_than_returning_zero() -> None:
    """C2 requires an int >= 1; 0 would be a legal-looking wrong answer."""
    with pytest.raises(ValueError, match="never burns"):
        count_ignition_components(np.zeros((4, 8, 8), dtype=np.uint8), cell_size_m=CELL_M)
    with pytest.raises(ValueError, match=r"\(T, H, W\)"):
        count_ignition_components(np.zeros((8, 8), dtype=np.uint8), cell_size_m=CELL_M)


def test_the_count_is_never_below_one_and_provenance_states_its_own_rule() -> None:
    """C-3: the two thresholds must carry the sample they were fitted on."""
    base = np.zeros(SHAPE, dtype=bool)
    frames = [_blob(base, 12, 20), _grow(_blob(base, 12, 20))]
    prov = _count(frames).to_provenance()

    assert prov["n_ignition_components"] >= 1
    assert "not hand-entered" in prov["method"]
    assert "spatial blocks" in prov["thresholds_fitted_on"]
    assert str(SEED_MERGE_KM) in prov["rule"]
    assert str(SPOT_RANGE_MAX_KM) in prov["rule"]
    assert "distance alone does not" in prov["rule"]


def test_the_derivation_reproduces_the_twelve_shipped_manifests() -> None:
    """The adopted controls are synthetic; this is the same rule on real tensors.

    Skips when the corpus is absent, per A10 PROPOSAL 3 - a suite that depends on
    another lead's live artifacts must skip rather than fail, or it teaches people
    to re-run instead of to read. When the corpus IS present this is the only
    check here that would catch the derivation and the shipped manifests
    disagreeing, which is a C0 property no synthetic case can see.
    """
    import json

    from wildfire_nowcast.common.paths import fires_dir
    from wildfire_nowcast.common.zarr_io import open_tensor

    root = fires_dir()
    fires = sorted(p for p in root.glob("*/manifest.json")) if root.is_dir() else []
    if len(fires) < 2:
        pytest.skip("built fire corpus not present")

    disagreements = {}
    for manifest_path in fires:
        manifest = json.loads(manifest_path.read_text())
        declared = manifest.get(
            "n_ignition_components",
            manifest.get("provenance", {}).get("n_ignition_components"),
        )
        if declared is None:
            continue
        ds = open_tensor(manifest_path.parent / "tensor.zarr")
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
        cell_m = float(ds.attrs.get("cell_size_m", 1000.0))
        derived = count_ignition_components(state, cell_size_m=cell_m).n_ignition_components
        if derived != declared:
            disagreements[manifest_path.parent.name] = (declared, derived)

    assert not disagreements, (
        f"manifest vs re-derivation disagree (declared, derived): {disagreements}. C0: the "
        "producer and the verifier must compute this through the same code."
    )
