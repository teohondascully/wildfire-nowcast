"""Self-tests for :mod:`wildfire_nowcast.data.crossings`, importable by infra.

``tests/`` is infra's directory and infra is mid-flight, so this ships as plain
zero-argument ``test_*`` functions inside the package that owns the code — the
same pattern ``sim.selftest`` and ``eval.selftest`` already use and that
``tests/test_adopted_selftests.py`` collects by introspection. Adoption is two
lines in that file (import this module, add it to the collected list); until
then ``python -m wildfire_nowcast.data.crossings_selftest`` runs the whole set.

**Every check here can fail.** Each one plants a defect that it catches:
a jitter-sized gap that must be rejected, a barrier that must be seen, an
oblique corridor a Bresenham line would slip through, a merging far body that
must NOT be called an ignition, and a distance the ratified ignition code must
agree with cell-for-cell (C0 — the producer and the verifier may not compute
geometry through different code).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from wildfire_nowcast.data.crossings import (
    LABEL_RESOLUTION_ELEMENT_CELLS,
    MIN_EVENT_CELLS,
    MIN_GAP_KM,
    SEPARATE_IGNITION_KM,
    barrier_evidence,
    classify_body,
    crossings_path,
    detect_detached_bodies,
    nearest_pair,
    positive_control,
    supercover_line,
)
from wildfire_nowcast.data.ignitions import _min_gap_km

__all__ = ["run_all"]


def _planted(gap_cells: int, n_cells: int, hours: int = 6) -> np.ndarray:
    """One stationary front cell and one body ``gap_cells`` away from hour 3."""
    h, w = 9, 9 + gap_cells + n_cells
    st = np.zeros((hours, h, w), dtype=np.uint8)
    st[:, 4, 2] = 1
    for t in range(3, hours):
        for k in range(n_cells):
            st[t, 4, 2 + gap_cells + k] = 1
    return np.maximum.accumulate(st, axis=0)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_supercover_does_not_slip_between_two_diagonal_barrier_cells() -> None:
    """A Bresenham line crosses a diagonal barrier without touching it.

    Planted defect: the barrier is the anti-diagonal pair (4,4),(5,5) and the
    segment runs (4,2)->(6,7). A thin line reports a clean corridor across a
    river; the supercover must not.
    """
    line = supercover_line(4, 2, 6, 7)
    assert (4, 4) in line or (5, 5) in line, "supercover missed a diagonal obstacle"
    assert line[0] == (4, 2) and line[-1] == (6, 7)


def test_nearest_pair_returns_the_argmin_and_not_merely_the_distance() -> None:
    ays = np.array([4, 4])
    axs = np.array([7, 9])
    bys = np.array([4])
    bxs = np.array([2])
    dist, a_cell, b_cell = nearest_pair(ays, axs, bys, bxs)
    assert dist == 5.0
    assert a_cell == (4, 7), "the landing cell must be the CLOSEST body cell"
    assert b_cell == (4, 2)


def test_nearest_pair_on_an_empty_set_is_infinite_not_zero() -> None:
    """A missing source must not read as 'adjacent'. That direction is dangerous."""
    dist, a_cell, b_cell = nearest_pair(
        np.array([], dtype=int), np.array([], dtype=int), np.array([1]), np.array([1])
    )
    assert dist == float("inf") and a_cell == (-1, -1) and b_cell == (-1, -1)


def test_gap_agrees_with_the_ratified_ignition_distance_cell_for_cell() -> None:
    """C0: two implementations of 'closest approach' may not disagree.

    ``nearest_pair`` is new; ``ignitions._min_gap_km`` is the ratified one that
    sets C2's ``n_ignition_components``. If these ever diverge, the crossing
    count and the ignition count are measuring different geometry.
    """
    rng = np.random.default_rng(11)
    for _ in range(40):
        a = rng.integers(0, 20, size=(6, 2))
        b = rng.integers(0, 20, size=(7, 2))
        mine, _, _ = nearest_pair(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        theirs = _min_gap_km(a[:, 0], a[:, 1], b[:, 0], b[:, 1], 1.0)
        assert abs(mine - theirs) < 1e-9, "nearest_pair disagrees with ignitions"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_a_planted_crossing_is_found_at_exactly_the_planted_distance() -> None:
    bodies = detect_detached_bodies(_planted(5, 3), cell_size_m=1000.0)
    assert len(bodies) == 1
    assert bodies[0].gap_km == 5.0
    assert bodies[0].n_cells == 3
    assert bodies[0].hour == 3
    assert bodies[0].anchor == (4, 2)
    assert bodies[0].landing == (4, 7)


def test_contiguous_growth_produces_no_detached_body() -> None:
    """The check that stops a detector which fires on everything."""
    st = np.zeros((6, 9, 9), dtype=np.uint8)
    for t in range(6):
        st[t, 4, 2 : 3 + t] = 1
    assert detect_detached_bodies(np.maximum.accumulate(st, axis=0), cell_size_m=1000.0) == []


def test_a_jitter_sized_gap_is_rejected_at_the_threshold_of_record() -> None:
    """Planted defect: a 2-cell hole, the exact shape of a rasterisation miss."""
    bodies = detect_detached_bodies(_planted(2, 1), cell_size_m=1000.0)
    assert len(bodies) == 1 and bodies[0].gap_km == 2.0
    verdict = classify_body(bodies[0], min_gap_km=MIN_GAP_KM, min_cells=MIN_EVENT_CELLS)
    assert verdict == "rasterisation_jitter", "a 2 km gap is label noise, not a crossing"


def test_the_threshold_of_record_survives_gofer_label_noise() -> None:
    """2.236 km is the largest gap ONE omitted cell can manufacture. 3.0 > that."""
    assert MIN_GAP_KM > np.hypot(2, 1), "min_gap_km is inside single-cell jitter"


def test_a_far_body_that_merges_is_a_crossing_and_not_an_ignition() -> None:
    """ADR-028 (1): merging is the NORMAL FATE of a real spot fire.

    Planted defect: a body 20 km out — beyond SEPARATE_IGNITION_KM — that later
    merges. A distance-only rule deletes it and takes the G4 signal with it.
    """
    st = _planted(20, 4, hours=8)
    st[6:, 4, 2:27] = 1  # the front overruns the gap: genealogy established
    st = np.maximum.accumulate(st, axis=0)
    bodies = detect_detached_bodies(st, cell_size_m=1000.0)
    far = [b for b in bodies if b.gap_km > SEPARATE_IGNITION_KM]
    assert far and far[0].merges_later
    assert classify_body(far[0], min_gap_km=MIN_GAP_KM, min_cells=1) == "crossing"


def test_a_far_body_that_never_merges_is_a_separate_ignition() -> None:
    bodies = detect_detached_bodies(_planted(20, 4, hours=8), cell_size_m=1000.0)
    far = [b for b in bodies if b.gap_km > SEPARATE_IGNITION_KM]
    assert far and not far[0].merges_later
    assert classify_body(far[0], min_gap_km=MIN_GAP_KM, min_cells=1) == "separate_ignition"


def test_min_event_cells_actually_filters() -> None:
    """A knob reported as sensitive must be able to move the verdict."""
    body = detect_detached_bodies(_planted(5, 1), cell_size_m=1000.0)[0]
    assert classify_body(body, min_gap_km=MIN_GAP_KM, min_cells=1) == "crossing"
    assert classify_body(body, min_gap_km=MIN_GAP_KM, min_cells=2) == "too_small"


def test_gap_scales_with_cell_size_and_is_not_hardcoded_to_one_km() -> None:
    body = detect_detached_bodies(_planted(5, 2), cell_size_m=2000.0)[0]
    assert body.gap_km == 10.0


def test_dormancy_and_reignition_at_the_same_site_is_not_a_crossing() -> None:
    """Blind spot (c), asserted rather than described.

    ``fire_state`` is absorbing, so a cell that goes 1 -> 2 stays in the
    ever-burned set. Re-detection at the same site must produce nothing.
    """
    st = np.zeros((6, 9, 9), dtype=np.uint8)
    st[:, 4, 2] = 1
    st[2:, 4, 2] = 2
    st[4:, 4, 3] = 1
    assert detect_detached_bodies(st, cell_size_m=1000.0) == []


# --------------------------------------------------------------------------- #
# Barrier attribution
# --------------------------------------------------------------------------- #


def _flat(shape: tuple[int, int], value: float) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def test_a_planted_water_cell_in_the_corridor_is_reported_as_water() -> None:
    barrier = np.zeros((9, 20), dtype=np.float32)
    barrier[4, 5] = 1.0
    fuel = _flat((9, 20), 8.0)
    fuel[4, 5] = 98.0
    ev = barrier_evidence((4, 2), (4, 7), barrier, fuel, _flat((9, 20), 0.0))
    assert ev.kind == "water" and ev.n_barrier_cells == 1 and ev.n_open_water_cells == 1


def test_a_barrier_cell_that_landfire_does_not_call_water_is_not_called_water() -> None:
    """Channel 12 merges roads and water; the split must not over-claim."""
    barrier = np.zeros((9, 20), dtype=np.float32)
    barrier[4, 5] = 1.0
    ev = barrier_evidence((4, 2), (4, 7), barrier, _flat((9, 20), 8.0), _flat((9, 20), 0.0))
    assert ev.kind == "road_or_narrow_water"


def test_an_empty_corridor_reports_none_mapped_rather_than_inventing_a_barrier() -> None:
    ev = barrier_evidence(
        (4, 2),
        (4, 7),
        np.zeros((9, 20), np.float32),
        _flat((9, 20), 8.0),
        _flat((9, 20), 0.0),
    )
    assert ev.kind == "none_mapped" and ev.n_barrier_cells == 0


def test_a_ridge_is_detected_only_above_the_endpoints() -> None:
    elev = _flat((9, 20), 100.0)
    elev[4, 4] = 400.0
    ev = barrier_evidence((4, 2), (4, 7), np.zeros((9, 20), np.float32), _flat((9, 20), 8.0), elev)
    assert ev.kind == "ridge" and ev.ridge_relief_m == 300.0


def test_a_high_but_uniform_slope_is_not_a_ridge() -> None:
    """Planted defect: relief measured against sea level would fire here."""
    elev = _flat((9, 20), 0.0)
    for c in range(20):
        elev[:, c] = 100.0 * c
    ev = barrier_evidence((4, 2), (4, 7), np.zeros((9, 20), np.float32), _flat((9, 20), 8.0), elev)
    assert ev.ridge_relief_m <= 0.0 and ev.kind == "none_mapped"


def test_barrier_beats_nonburnable_beats_ridge_in_precedence() -> None:
    barrier = np.zeros((9, 20), dtype=np.float32)
    barrier[4, 5] = 1.0
    fuel = _flat((9, 20), 8.0)
    fuel[4, 4] = 93.0
    elev = _flat((9, 20), 0.0)
    elev[4, 3] = 900.0
    ev = barrier_evidence((4, 2), (4, 7), barrier, fuel, elev)
    assert ev.kind == "road_or_narrow_water"
    assert 93 in ev.nonburnable_classes and ev.ridge_relief_m == 900.0


def test_the_corridor_excludes_its_own_endpoints() -> None:
    """A burning anchor sitting on a road must not count as crossing that road."""
    barrier = np.zeros((9, 20), dtype=np.float32)
    barrier[4, 2] = 1.0
    barrier[4, 7] = 1.0
    ev = barrier_evidence((4, 2), (4, 7), barrier, _flat((9, 20), 8.0), _flat((9, 20), 0.0))
    assert ev.n_barrier_cells == 0 and ev.kind == "none_mapped"


# --------------------------------------------------------------------------- #
# Controls and the artifact itself
# --------------------------------------------------------------------------- #


def test_the_positive_control_passes_and_is_not_vacuous() -> None:
    pc = positive_control()
    assert pc["passed"] is True
    assert pc["must_be_nonzero_n_crossings_on_planted_data"] > 0
    assert pc["negative_control_contiguous_growth_n_detached"] == 0


def test_label_resolution_flag_matches_one_gofer_resolution_element() -> None:
    assert LABEL_RESOLUTION_ELEMENT_CELLS == 4


def _loaded() -> dict[str, Any] | None:
    path = crossings_path()
    return json.loads(path.read_text()) if path.exists() else None


def test_the_written_artifact_agrees_with_the_split_on_disk() -> None:
    """C8. Fold and block are properties of the SPLIT; a stale index lies quietly."""
    index = _loaded()
    if index is None:
        return
    from wildfire_nowcast.common.splits import split_fingerprint

    assert index["split_fingerprint"] == split_fingerprint()["fingerprint"]


def test_every_written_event_carries_a_block_and_a_fold() -> None:
    """An event without its block cannot be scored leave-block-out."""
    index = _loaded()
    if index is None:
        return
    assert index["events"], "the artifact exists but holds no events"
    for ev in index["events"]:
        assert isinstance(ev["spatial_block_id"], int)
        assert isinstance(ev["cv_fold"], int)
        assert ev["event_id"] and ev["fire_id"] and ev["time_utc"]
        assert ev["geometry"]["cells_rowcol"]
    ids = [e["event_id"] for e in index["events"]]
    assert len(ids) == len(set(ids)), "event ids must be unique"


def test_the_written_totals_reconcile_with_the_written_events() -> None:
    """A summary that does not add up to its own rows is the usual way this rots."""
    index = _loaded()
    if index is None:
        return
    events = index["events"]
    assert index["n_events"] == len(events)
    assert sum(index["events_per_block"].values()) == len(events)
    assert sum(index["events_per_fold"].values()) == len(events)
    assert sum(index["events_per_fire"].values()) == len(events)
    grid = index["sensitivity"]["by_min_gap_km_x_min_event_cells"]
    of_record = grid[str(index["detection_rule"]["min_gap_km"])][
        str(index["detection_rule"]["min_event_cells"])
    ]
    assert of_record == len(events), "the sensitivity grid disagrees with the event list"


def run_all() -> int:
    """Standalone runner. Returns the number of checks that ran."""
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok  {name}")
    return n


if __name__ == "__main__":  # pragma: no cover
    print(f"{run_all()} checks passed")
