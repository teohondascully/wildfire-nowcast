"""P3 - barrier-crossing / spot episode mining into ``data/events/crossings.json``.

**What this artifact is for.** G4 asks whether an explicit long-range spot
component beats a no-spot model *on crossing episodes*. That question needs an
enumerated, machine-readable set of episodes with a stated detection rule, and
every episode needs its SPATIAL BLOCK and CV FOLD, because G4 is scored
leave-block-out (ADR-028 (3)) and an event without its block cannot be scored.

**The detection rule, in one paragraph, because it has to be defensible.** A
crossing is a DETACHED BIRTH: cells that join the ever-burned set at hour ``t``
with no ever-burned 8-neighbour at ``t-1``. That is the only primitive available
from a 1 km state field - a fire that walks across a sub-cell river is not a
jump and cannot be one at this resolution. Each detached body is measured by its
``gap_km``, the closest approach between the body and the whole ever-burned
region at ``t-1``. Bodies are then classified:

* ``gap_km < MIN_GAP_KM``                        -> ``rasterisation_jitter``
* never merges AND ``gap_km > SEPARATE_IGNITION_KM`` -> ``separate_ignition``
* otherwise                                      -> ``crossing`` (counted)

**Why the gap threshold is where it is, and why that is not a fit.** GOFER's
effective resolution is ~2 km on our 1 km grid, so a one-cell perimeter omission
is indistinguishable from a real one-cell jump. The largest gap a single missing
cell can manufacture between two 8-connected bodies is 2.236 km (dy=2, dx=1).
:data:`MIN_GAP_KM` is set to 3.0 km: strictly above that construction, 1.5x the
label product's effective resolution, and at or beyond the contagion kernel's
configured radius (3 cells) so a counted event is unreachable by the short-range
component by construction. It is NOT tuned to a count - see
``sensitivity.by_min_gap_km`` in the artifact, which reports the count at every
threshold, and note that the observed gap distribution has NO mass at all
between 2.236 km and 3.606 km, so every value in [2.25, 3.6] yields the same
set. The knob that actually moves the count is :data:`MIN_EVENT_CELLS`, and that
is reported with equal prominence rather than buried.

**What is NOT counted, stated so a reader does not have to infer it.**
(a) Crossings where the barrier cell itself is labelled burned are invisible
    here - the bodies stay 8-connected and no detached birth occurs. Per-fire
    ``n_barrier_cells_ever_burned`` is reported as the size of that blind spot.
(b) Separate ignitions filed under one fire id (ADR-019). Distance alone does
    not separate those from spots; time, then genealogy, then distance does, and
    that rule lives in :mod:`wildfire_nowcast.data.ignitions` (C0 - one rule).
(c) Anything below :data:`MIN_EVENT_CELLS`.

This module is READ-ONLY with respect to every existing artifact. It opens
tensors and manifests and writes exactly one new file under ``data/events/``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.components import label_components
from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import data_dir, fires_dir
from wildfire_nowcast.common.states import dilate
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.data.ignitions import SEED_MERGE_KM, SPOT_RANGE_MAX_KM
from wildfire_nowcast.data.sources.fuels import FBFM40_NONBURNABLE

__all__ = [
    "GOFER_EFFECTIVE_RESOLUTION_KM",
    "LABEL_RESOLUTION_ELEMENT_CELLS",
    "MIN_EVENT_CELLS",
    "MIN_GAP_KM",
    "RIDGE_RELIEF_M",
    "SENSITIVITY_MIN_CELLS",
    "SENSITIVITY_MIN_GAP_KM",
    "BarrierEvidence",
    "CrossingEvent",
    "DetachedBody",
    "barrier_evidence",
    "build_index",
    "classify_body",
    "crossings_path",
    "detect_detached_bodies",
    "nearest_pair",
    "positive_control",
    "supercover_line",
    "write_crossings",
]

# --------------------------------------------------------------------------- #
# The detection rule's constants. Every one of them is stamped into the
# artifact's `detection_rule` block; none may be changed without changing that.
# --------------------------------------------------------------------------- #

#: GOFER is a ~2 km product rasterised onto a 1 km grid (ADR-006 P4).
#: Label noise is therefore at the same scale as a small crossing, which is the
#: whole reason this file states its threshold instead of assuming one.
GOFER_EFFECTIVE_RESOLUTION_KM = 2.0

#: Cells covered by ONE label-product resolution element: (2 km / 1 km)^2 = 4.
#: A body no larger than this is not resolved by the labels - flagged per event,
#: never excluded, because excluding it would delete newly-ignited spots.
LABEL_RESOLUTION_ELEMENT_CELLS = 4

#: Minimum closest-approach gap for a detached body to count as a crossing.
#: 3.0 km. See the module docstring for the derivation. Fitting sample for the
#: C-3 declaration: the observed gap distribution over 21 fires / 14 spatial
#: blocks, which is empty on (2.236, 3.606) km - the constant sits inside a flat
#: plateau rather than on a slope, and no value in [2.25, 3.6] changes the set.
MIN_GAP_KM = 3.0

#: Minimum body size, in 1 km cells, for a detached body to count.
#: Set to 1 DELIBERATELY: a spot fire is one cell at the moment it is first
#: detected, and requiring >= 2 cells silently changes the estimand to "spot
#: fires that had already grown by the time they were mapped". Single-cell
#: bodies are at or below the label product's effective resolution, so each
#: event carries `at_or_below_label_resolution` and the count at every
#: alternative is in `sensitivity.by_min_event_cells`. This is the knob the
#: count is actually sensitive to; it is reported, not chosen quietly.
MIN_EVENT_CELLS = 1

#: A never-merging body farther than this is a separate ignition, not a spot.
#: IMPORTED from `data.ignitions` (C0): the ignition count and the crossing
#: count must not disagree about what a separate ignition is.
SEPARATE_IGNITION_KM = SPOT_RANGE_MAX_KM

#: Reported for context only; nothing is classified on it here.
RASTERISATION_HOLE_KM = SEED_MERGE_KM

#: Terrain relief along the corridor, above both endpoints, that counts as a
#: ridge. 100 m at 1 km cells is a ~6 deg mean slope over the corridor - enough
#: to be a real topographic obstacle rather than roughness. Reported alongside
#: the raw `ridge_relief_m` so a reader can re-threshold it.
RIDGE_RELIEF_M = 100.0

SENSITIVITY_MIN_GAP_KM: tuple[float, ...] = (
    2.0,
    2.25,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    15.0,
)
SENSITIVITY_MIN_CELLS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Static channels consulted for barrier attribution, by C1 channel NAME.
BARRIER_CHANNEL = "water_barrier_mask"
FUEL_CHANNEL = "fuel_model_id"
ELEVATION_CHANNEL = "elevation"
WIND_U_CHANNEL = "wind_u10"
WIND_V_CHANNEL = "wind_v10"

#: FBFM40 class 98 is NB8 "Open Water". Used ONLY to split channel 12's merged
#: water-or-road flag; see `BarrierEvidence.kind` for why that split is weak.
FBFM40_OPEN_WATER = 98


def crossings_path() -> Path:
    """``data/events/crossings.json``. A NEW directory; nothing else moves.

    A DESTINATION, not a citation: ``write_crossings`` produces it and no clone
    carries it. Read it only through a guard that handles absence.
    """
    return data_dir() / "events" / "crossings.json"


# --------------------------------------------------------------------------- #
# Small pure geometry
# --------------------------------------------------------------------------- #


def supercover_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Every cell a straight segment from ``(r0,c0)`` to ``(r1,c1)`` touches.

    Supercover rather than Bresenham: a Bresenham line can slip diagonally
    between two barrier cells and report a clean corridor across a river.
    Endpoints are included; callers strip them.
    """
    dr, dc = r1 - r0, c1 - c0
    n = int(max(abs(dr), abs(dc)))
    if n == 0:
        return [(int(r0), int(c0))]
    out: list[tuple[int, int]] = []
    for k in range(4 * n + 1):
        t = k / (4 * n)
        rr = r0 + dr * t
        cc = c0 + dc * t
        for r in {int(np.floor(rr + 1e-9)), int(np.ceil(rr - 1e-9))}:
            for c in {int(np.floor(cc + 1e-9)), int(np.ceil(cc - 1e-9))}:
                if (r, c) not in out:
                    out.append((r, c))
    return out


def nearest_pair(
    ays: np.ndarray, axs: np.ndarray, bys: np.ndarray, bxs: np.ndarray
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Closest approach between two cell sets, in CELLS, with the argmin pair.

    Returns ``(distance_cells, a_cell, b_cell)``. ``inf`` and ``(-1,-1)`` when
    either set is empty, so a caller cannot mistake "no source" for "adjacent".
    """
    if not len(ays) or not len(bys):
        return float("inf"), (-1, -1), (-1, -1)
    best = float("inf")
    a_cell = (-1, -1)
    b_cell = (-1, -1)
    for y, x in zip(ays, axs, strict=True):
        d = np.hypot(bys - y, bxs - x)
        j = int(np.argmin(d))
        if float(d[j]) < best:
            best = float(d[j])
            a_cell = (int(y), int(x))
            b_cell = (int(bys[j]), int(bxs[j]))
    return best, a_cell, b_cell


# --------------------------------------------------------------------------- #
# Barrier attribution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BarrierEvidence:
    """What the static masks say lies between the front and the landing site."""

    corridor_cells: int
    n_barrier_cells: int
    n_barrier_cells_dilated: int
    n_nonburnable_cells: int
    nonburnable_classes: list[int]
    n_open_water_cells: int
    ridge_relief_m: float
    kind: str
    masks_consulted: list[str]


def barrier_evidence(
    anchor: tuple[int, int],
    landing: tuple[int, int],
    barrier: np.ndarray,
    fuel: np.ndarray,
    elevation: np.ndarray,
) -> BarrierEvidence:
    """Classify what was crossed, from the C1 static channels only.

    The corridor is the supercover segment STRICTLY BETWEEN the anchor (nearest
    already-burned cell) and the landing cell (nearest cell of the new body).
    A dilated count is reported beside it so a reader can see whether the
    verdict depends on corridor width - an oblique barrier can miss a
    one-cell-wide line.

    `kind` precedence is water/road -> non-burnable fuel -> ridge -> none. The
    water-vs-road split uses FBFM40 98 (NB8 Open Water) and is WEAK: channel 12
    merges GSW permanent water with TIGER primary/secondary roads by design
    (`data/sources/barriers.py`), and a 60 m river is below LANDFIRE's water
    class while still being flagged by GSW's line rule. Read
    `road_or_narrow_water` as "channel 12 fired and LANDFIRE does not call it
    water", never as "a highway".
    """
    line = supercover_line(*anchor, *landing)
    h, w = barrier.shape
    interior = [
        (r, c) for (r, c) in line if (r, c) not in (anchor, landing) and 0 <= r < h and 0 <= c < w
    ]
    if not interior:  # adjacent-by-construction cannot happen for a detached body
        interior = [(r, c) for (r, c) in line if 0 <= r < h and 0 <= c < w]

    rows = np.array([r for r, _ in interior], dtype=int)
    cols = np.array([c for _, c in interior], dtype=int)
    bar = barrier[rows, cols]
    fuels = fuel[rows, cols].astype(int)

    wide = np.zeros_like(barrier, dtype=bool)
    wide[rows, cols] = True
    wide = dilate(wide, 1)
    n_barrier_dilated = int((barrier.astype(bool) & wide).sum())

    nonburnable = sorted({int(v) for v in fuels if int(v) in FBFM40_NONBURNABLE})
    n_open_water = int((fuels == FBFM40_OPEN_WATER).sum())

    on_grid = [(r, c) for (r, c) in line if 0 <= r < h and 0 <= c < w]
    prof_rows = np.array([r for r, _ in on_grid], dtype=int)
    prof_cols = np.array([c for _, c in on_grid], dtype=int)
    endpoints = max(float(elevation[anchor]), float(elevation[landing]))
    relief = float(elevation[prof_rows, prof_cols].max() - endpoints) if len(on_grid) else 0.0

    n_barrier = int(bar.astype(bool).sum())
    if n_barrier > 0:
        kind = "water" if n_open_water > 0 else "road_or_narrow_water"
    elif nonburnable:
        kind = "nonburnable_fuel"
    elif relief > RIDGE_RELIEF_M:
        kind = "ridge"
    else:
        kind = "none_mapped"

    return BarrierEvidence(
        corridor_cells=len(interior),
        n_barrier_cells=n_barrier,
        n_barrier_cells_dilated=n_barrier_dilated,
        n_nonburnable_cells=int(sum(1 for v in fuels if int(v) in FBFM40_NONBURNABLE)),
        nonburnable_classes=nonburnable,
        n_open_water_cells=n_open_water,
        ridge_relief_m=round(relief, 1),
        kind=kind,
        masks_consulted=[BARRIER_CHANNEL, FUEL_CHANNEL, ELEVATION_CHANNEL],
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetachedBody:
    """New burned cells at ``hour`` with no ever-burned 8-neighbour at ``hour-1``."""

    hour: int
    n_cells: int
    gap_km: float
    merges_later: bool
    landing: tuple[int, int]
    anchor: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    n_prior_burned_cells: int
    #: Centroid of the ENTIRE ever-burned region at t-1. The anchor is chosen to
    #: minimise distance, which biases its direction; the centroid does not.
    prior_centroid: tuple[float, float]


def detect_detached_bodies(fire_state: np.ndarray, *, cell_size_m: float) -> list[DetachedBody]:
    """Every detached birth in a ``(T, H, W)`` C1 state field, unclassified.

    Deliberately returns EVERY detached body including the sub-threshold ones:
    the artifact reports what it rejected, and the sensitivity table cannot be
    built from a pre-filtered list.
    """
    st = np.asarray(fire_state)
    if st.ndim != 3:
        raise ValueError("fire_state must be (T, H, W)")
    ever = st != 0
    res_km = float(cell_size_m) / 1000.0
    any_t = ever.reshape(st.shape[0], -1).any(axis=1)
    if not any_t.any():
        raise ValueError("fire_state never burns: nothing to mine")
    t_first = int(np.argmax(any_t))

    final_labels, _ = label_components(ever[-1])
    out: list[DetachedBody] = []
    prev = ever[t_first]
    for t in range(t_first + 1, st.shape[0]):
        new = ever[t] & ~prev
        if new.any():
            detached = new & ~dilate(prev, 1)
            if detached.any():
                py, px = np.nonzero(prev)
                parent_final = set(final_labels[py, px].tolist())
                lab, n = label_components(detached)
                for k in range(1, n + 1):
                    ys, xs = np.nonzero(lab == k)
                    d_cells, landing, anchor = nearest_pair(ys, xs, py, px)
                    out.append(
                        DetachedBody(
                            hour=t,
                            n_cells=int(len(ys)),
                            gap_km=d_cells * res_km,
                            merges_later=bool(int(final_labels[ys[0], xs[0]]) in parent_final),
                            landing=landing,
                            anchor=anchor,
                            cells=tuple((int(a), int(b)) for a, b in zip(ys, xs, strict=True)),
                            n_prior_burned_cells=int(len(py)),
                            prior_centroid=(float(py.mean()), float(px.mean())),
                        )
                    )
        prev = ever[t]
    return out


def classify_body(body: DetachedBody, *, min_gap_km: float, min_cells: int) -> str:
    """``crossing`` / ``rasterisation_jitter`` / ``too_small`` / ``separate_ignition``.

    Order matters and is the ratified one (ADR-019 §7, ADR-028 (1)): time and
    genealogy decide, distance is a tiebreak. A body that MERGES is never a
    separate ignition however far it landed - merging is the normal fate of a
    real spot fire, and a definition demanding permanent separation selects for
    the rare pathological case.
    """
    if body.gap_km < min_gap_km:
        return "rasterisation_jitter"
    if (not body.merges_later) and body.gap_km > SEPARATE_IGNITION_KM:
        return "separate_ignition"
    if body.n_cells < min_cells:
        return "too_small"
    return "crossing"


# --------------------------------------------------------------------------- #
# Per-fire assembly
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CrossingEvent:
    """One counted episode, with everything a replay tool and G4 both need."""

    event_id: str
    fire_id: str
    spatial_block_id: int
    cv_fold: int
    split_role: str
    label_source: str
    #: C2 [v2.7]. A "spot" inside a multi-ignition fire id may be a co-ignition
    #: the genealogy rule kept because it merged (ADR-028 (1)). Surfaced per
    #: event so a reader can see it without opening a manifest.
    fire_n_ignition_components: int
    hour_index: int
    time_utc: str
    time_convention: str
    gap_km: float
    n_cells: int
    at_or_below_label_resolution: bool
    merges_later: bool
    barrier_crossed: str
    barrier: dict[str, Any]
    geometry: dict[str, Any]
    wind: dict[str, Any]


def _event_id(fire_id: str, hour: int, landing: tuple[int, int]) -> str:
    """Stable across re-runs: keyed on the fire, the hour and the landing cell."""
    return f"X-{fire_id}-h{hour:04d}-r{landing[0]:03d}c{landing[1]:03d}"


def _cosine(u: float, v: float, dr: float, dc: float) -> float | None:
    """Cosine between the wind vector and a row/col displacement.

    ``y`` DESCENDS on the C1 lattice (C1.4), so a +row step is SOUTH and the
    north component of the displacement is ``-dr``. Getting this sign wrong
    inverts every alignment number and reads as a physics finding.
    """
    east, north = float(dc), float(-dr)
    speed = float(np.hypot(u, v))
    norm = float(np.hypot(east, north))
    if speed <= 0 or norm <= 0:
        return None
    return float((u * east + v * north) / (speed * norm))


def _wind_at(
    ds: Any,
    hour: int,
    anchor: tuple[int, int],
    landing: tuple[int, int],
    prior_centroid: tuple[float, float],
) -> dict[str, Any]:
    """RTMA wind at the anchor and its alignment with the jump direction.

    TWO alignments are reported and neither is privileged. ``downwind_cosine``
    uses the anchor, which is chosen to MINIMISE distance and therefore has a
    direction bias of its own. ``downwind_cosine_from_prior_centroid`` uses the
    centroid of the whole ever-burned region, which is the better estimand for
    "did this land downwind of the fire" and is not chosen by any distance rule.
    """
    u = float(ds["features"].sel(channel=WIND_U_CHANNEL).values[hour, anchor[0], anchor[1]])
    v = float(ds["features"].sel(channel=WIND_V_CHANNEL).values[hour, anchor[0], anchor[1]])
    cos_anchor = _cosine(u, v, landing[0] - anchor[0], landing[1] - anchor[1])
    cos_centroid = _cosine(u, v, landing[0] - prior_centroid[0], landing[1] - prior_centroid[1])
    return {
        "wind_u10_ms": round(u, 2),
        "wind_v10_ms": round(v, 2),
        "wind_speed_ms": round(float(np.hypot(u, v)), 2),
        "displacement_east_cells": landing[1] - anchor[1],
        "displacement_north_cells": anchor[0] - landing[0],
        "downwind_cosine": None if cos_anchor is None else round(cos_anchor, 3),
        "downwind_cosine_from_prior_centroid": (
            None if cos_centroid is None else round(cos_centroid, 3)
        ),
        "prior_centroid_rowcol": [round(prior_centroid[0], 2), round(prior_centroid[1], 2)],
    }


def _fire_events(
    manifest_path: Path, *, min_gap_km: float, min_cells: int, split: dict[str, Any]
) -> tuple[list[CrossingEvent], dict[str, Any], list[DetachedBody]]:
    """Mine one fire. Returns (counted events, per-fire QA row, all bodies)."""
    manifest = json.loads(manifest_path.read_text())
    ds = open_tensor(manifest_path.parent / "tensor.zarr")
    grid = Grid.from_dataset(ds)
    state = np.asarray(ds["fire_state"].values)
    bodies = detect_detached_bodies(state, cell_size_m=grid.cell_size_m)

    barrier = np.asarray(ds["features"].sel(channel=BARRIER_CHANNEL).values[0])
    fuel = np.asarray(ds["features"].sel(channel=FUEL_CHANNEL).values[0])
    elevation = np.asarray(ds["features"].sel(channel=ELEVATION_CHANNEL).values[0])
    times = ds["time"].values

    fire_id = str(manifest["fire_id"])
    block = int(manifest["spatial_block_id"])
    fold = int(manifest["cv_fold"])
    role = "train" if fire_id in set(split.get("train_fire_ids", [])) else "heldout"
    label_source = str(manifest.get("provenance", {}).get("label_source", "gofer_published"))

    events: list[CrossingEvent] = []
    counts: Counter[str] = Counter()
    for body in bodies:
        verdict = classify_body(body, min_gap_km=min_gap_km, min_cells=min_cells)
        counts[verdict] += 1
        if verdict != "crossing":
            continue
        ev = barrier_evidence(body.anchor, body.landing, barrier, fuel, elevation)
        rows = [r for r, _ in body.cells]
        cols = [c for _, c in body.cells]
        east_land, north_land = grid.xy(*body.landing)
        east_anch, north_anch = grid.xy(*body.anchor)
        events.append(
            CrossingEvent(
                event_id=_event_id(fire_id, body.hour, body.landing),
                fire_id=fire_id,
                spatial_block_id=block,
                cv_fold=fold,
                split_role=role,
                label_source=label_source,
                fire_n_ignition_components=int(manifest.get("n_ignition_components", 1)),
                hour_index=body.hour,
                time_utc=str(np.datetime_as_string(times[body.hour], unit="s")),
                time_convention=str(ds.attrs.get("time_convention", "unknown")),
                gap_km=round(body.gap_km, 3),
                n_cells=body.n_cells,
                at_or_below_label_resolution=body.n_cells <= LABEL_RESOLUTION_ELEMENT_CELLS,
                merges_later=body.merges_later,
                barrier_crossed=ev.kind,
                barrier=asdict(ev),
                geometry={
                    "crs": str(ds.attrs.get("crs", "EPSG:5070")),
                    "cell_size_m": float(grid.cell_size_m),
                    "landing_rowcol": list(body.landing),
                    "anchor_rowcol": list(body.anchor),
                    "landing_xy_5070": [round(east_land, 1), round(north_land, 1)],
                    "anchor_xy_5070": [round(east_anch, 1), round(north_anch, 1)],
                    "centroid_rowcol": [
                        round(float(np.mean(rows)), 2),
                        round(float(np.mean(cols)), 2),
                    ],
                    "bbox_rowcol": [min(rows), min(cols), max(rows), max(cols)],
                    "cells_rowcol": [list(c) for c in body.cells],
                    "n_prior_burned_cells": body.n_prior_burned_cells,
                    "replay_window_hours": [
                        max(0, body.hour - 3),
                        min(int(state.shape[0]) - 1, body.hour + 3),
                    ],
                    "tensor": f"data/fires/{fire_id}/tensor.zarr",
                },
                wind=_wind_at(ds, body.hour, body.anchor, body.landing, body.prior_centroid),
            )
        )

    ever = state != 0
    qa = {
        "fire_id": fire_id,
        "spatial_block_id": block,
        "cv_fold": fold,
        "split_role": role,
        "label_source": label_source,
        "n_hours": int(state.shape[0]),
        "n_detached_bodies": len(bodies),
        "n_crossings": len(events),
        "max_gap_km": round(max((b.gap_km for b in bodies), default=0.0), 3),
        "verdict_counts": dict(sorted(counts.items())),
        # The size of this detector's blind spot (a): a crossing whose barrier
        # cell is itself labelled burned stays 8-connected and is never seen.
        "n_barrier_cells": int(barrier.astype(bool).sum()),
        "n_barrier_cells_ever_burned": int((barrier.astype(bool) & ever[-1]).sum()),
        "n_ignition_components": int(manifest.get("n_ignition_components", 1)),
    }
    return events, qa, bodies


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #


def _planted_fire(gap_cells: int, n_cells: int, hours: int = 6) -> np.ndarray:
    """A tiny C1-shaped state field with ONE planted crossing of known gap."""
    h, w = 9, 9 + gap_cells + n_cells
    st = np.zeros((hours, h, w), dtype=np.uint8)
    st[:, 4, 2] = 1  # a stationary one-cell front, burning throughout
    for t in range(3, hours):
        for k in range(n_cells):
            st[t, 4, 2 + gap_cells + k] = 1
    return np.maximum.accumulate(st, axis=0)


def positive_control() -> dict[str, Any]:
    """Fire the detector on planted data, on the SAME code path as the corpus.

    An all-clear scan carries a control that must return non-zero. This one
    plants a 5-cell jump and a 3-cell body, asserts the detector finds exactly
    one crossing at exactly 5.0 km, and - the half that is easy to forget -
    asserts a contiguously growing fire yields ZERO, so a detector that fires on
    everything could not pass either.
    """
    planted = detect_detached_bodies(_planted_fire(5, 3), cell_size_m=1000.0)
    hits = [
        b
        for b in planted
        if classify_body(b, min_gap_km=MIN_GAP_KM, min_cells=MIN_EVENT_CELLS) == "crossing"
    ]

    contiguous = np.zeros((6, 9, 9), dtype=np.uint8)
    for t in range(6):
        contiguous[t, 4, 2 : 3 + t] = 1
    negative = detect_detached_bodies(contiguous, cell_size_m=1000.0)

    barrier = np.zeros((9, 20), dtype=np.float32)
    barrier[4, 5] = 1.0
    fuel = np.full((9, 20), 8.0, dtype=np.float32)
    fuel[4, 5] = float(FBFM40_OPEN_WATER)
    elev = np.zeros((9, 20), dtype=np.float32)
    ev = barrier_evidence((4, 2), (4, 7), barrier, fuel, elev)

    ok = (
        len(hits) == 1
        and abs(hits[0].gap_km - 5.0) < 1e-9
        and hits[0].n_cells == 3
        and len(negative) == 0
        and ev.kind == "water"
        and ev.n_barrier_cells == 1
    )
    return {
        "must_be_nonzero_n_crossings_on_planted_data": len(hits),
        "planted_gap_km": None if not hits else round(hits[0].gap_km, 3),
        "planted_n_cells": None if not hits else hits[0].n_cells,
        "negative_control_contiguous_growth_n_detached": len(negative),
        "planted_barrier_kind": ev.kind,
        "planted_barrier_cells_in_corridor": ev.n_barrier_cells,
        "passed": bool(ok),
        "note": (
            "Runs detect_detached_bodies / classify_body / barrier_evidence — the "
            "same three functions the corpus scan uses. A fire reporting zero "
            "crossings is therefore a statement about that fire, not about a "
            "detector nobody watched fire."
        ),
    }


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


def _sensitivity(
    bodies_by_fire: dict[str, list[DetachedBody]], block_of: dict[str, int]
) -> dict[str, Any]:
    """How the count moves as the two thresholds move. Reported, never hidden."""
    flat = [(f, b) for f, bs in bodies_by_fire.items() for b in bs]

    def _hits(gap: float, cells: int) -> list[tuple[str, DetachedBody]]:
        return [
            (f, b)
            for f, b in flat
            if classify_body(b, min_gap_km=gap, min_cells=cells) == "crossing"
        ]

    by_gap: dict[str, Any] = {}
    for g in SENSITIVITY_MIN_GAP_KM:
        hits = _hits(g, MIN_EVENT_CELLS)
        by_gap[str(g)] = {
            "n_crossings": len(hits),
            "n_blocks": len({block_of[f] for f, _ in hits}),
            "n_fires": len({f for f, _ in hits}),
        }
    grid_counts = {
        str(g): {str(c): len(_hits(g, c)) for c in SENSITIVITY_MIN_CELLS}
        for g in SENSITIVITY_MIN_GAP_KM
    }
    gaps = sorted(round(b.gap_km, 3) for _, b in flat)
    return {
        "of_record": {"min_gap_km": MIN_GAP_KM, "min_event_cells": MIN_EVENT_CELLS},
        "by_min_gap_km": by_gap,
        "by_min_gap_km_x_min_event_cells": grid_counts,
        "n_detached_bodies_total": len(flat),
        "observed_gap_km_histogram": {
            str(v): sum(1 for g in gaps if g == v) for v in sorted(set(gaps))
        },
        "reading": (
            "min_gap_km sits inside an EMPTY band of the observed gap "
            "distribution, so the count is flat across it rather than on a "
            "slope. min_event_cells is the knob the count is sensitive to; its "
            "whole sweep is above and nothing is chosen quietly."
        ),
    }


def _downwind_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Are these episodes DOWNWIND? Ember transport says they should be.

    Reported because it is the cheapest available physical plausibility test on
    an event set that is about to become gate evidence, and because it does not
    come out flattering. The cosine is between the RTMA wind at the anchor at
    the event hour and the anchor->landing displacement; a 1-cell body whose
    true launch point was elsewhere on the front will have a noisy direction, so
    a single negative value proves nothing and the DISTRIBUTION is the number.
    """
    out: dict[str, Any] = {
        "note": (
            "An episode set that is not preferentially downwind is not "
            "obviously ember transport. Reported as a property of the EVIDENCE, "
            "not as a filter — nothing is excluded on this. n is small and the "
            "direction of a 1-cell body is noisy, so read the two estimands "
            "together: if they disagree, the ANCHOR CHOICE is doing the work."
        )
    }
    for key, label in (
        ("downwind_cosine", "from_anchor"),
        ("downwind_cosine_from_prior_centroid", "from_prior_centroid"),
    ):
        cos = [r["wind"].get(key) for r in rows]
        cos = [c for c in cos if c is not None]
        if not cos:
            out[label] = {"n_scored": 0}
            continue
        arr = np.asarray(cos, dtype=float)
        out[label] = {
            "n_scored": len(cos),
            "n_downwind_cos_gt_0": int((arr > 0).sum()),
            "n_upwind_cos_lt_0": int((arr < 0).sum()),
            "mean_cosine": round(float(arr.mean()), 3),
            "median_cosine": round(float(np.median(arr)), 3),
        }
    return out


def _evidence_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts, not filters. Every one of these events is IN the index.

    These are the four ways a counted episode could be something other than
    ember transport, each measured rather than argued. NO threshold here decides
    anything and none is proposed as one: they are reported so that whoever sets
    a bar can see the composition of the evidence first (ADR-040 (1)).
    """

    def _n(pred: Any) -> list[str]:
        return sorted(r["event_id"] for r in rows if pred(r))

    barely = _n(lambda r: r["geometry"]["n_prior_burned_cells"] < 10)
    multi = _n(lambda r: r["fire_n_ignition_components"] > 1)
    calm = _n(lambda r: (r["wind"]["wind_speed_ms"] or 0.0) < 3.0)
    tiny = _n(lambda r: r["n_cells"] == 1)
    return {
        "n_events": len(rows),
        "from_a_fire_smaller_than_10_cells": {
            "n": len(barely),
            "event_ids": barely,
            "why": "a jump from a barely-established fire is as consistent with a "
            "co-ignition the genealogy rule kept (because it merged) as it "
            "is with spotting",
        },
        "in_a_multi_ignition_fire": {
            "n": len(multi),
            "event_ids": multi,
            "why": "C2 n_ignition_components > 1: GOFER files separate ignitions "
            "under one fire id (ADR-019)",
        },
        "at_wind_below_3_ms": {
            "n": len(calm),
            "event_ids": calm,
            "why": "long-range ember transport at low 10 m wind is physically "
            "demanding; reported, not excluded",
        },
        "single_cell_bodies": {
            "n": len(tiny),
            "event_ids": tiny,
            "why": "1 km2 is below the label product's ~2 km resolution element",
        },
        "caveats": [
            "The wind figure is RTMA 10 m MEAN wind at 2.5 km, resampled to 1 km. "
            "Ember transport is plume- and gust-driven, so low mean wind weakens "
            "the spotting reading WITHOUT refuting it.",
            "These counts OVERLAP; do not add them.",
            "n = 12. Every proportion here has a standard error of ~0.14 at best.",
        ],
    }


def build_index(
    *, min_gap_km: float = MIN_GAP_KM, min_cells: int = MIN_EVENT_CELLS
) -> dict[str, Any]:
    """The whole artifact, as a dict. Reads only; writes nothing."""
    from wildfire_nowcast.common.splits import split_fingerprint  # noqa: PLC0415

    split = split_fingerprint()
    manifests = sorted(Path(fires_dir()).glob("*/manifest.json"))
    manifests = [m for m in manifests if (m.parent / "tensor.zarr").exists()]

    events: list[CrossingEvent] = []
    fires: list[dict[str, Any]] = []
    bodies_by_fire: dict[str, list[DetachedBody]] = {}
    block_of: dict[str, int] = {}
    for man in manifests:
        evs, qa, bodies = _fire_events(man, min_gap_km=min_gap_km, min_cells=min_cells, split=split)
        events.extend(evs)
        fires.append(qa)
        bodies_by_fire[qa["fire_id"]] = bodies
        block_of[qa["fire_id"]] = int(qa["spatial_block_id"])

    rows = [asdict(e) for e in events]
    rows.sort(key=lambda r: (r["spatial_block_id"], r["fire_id"], r["hour_index"]))

    per_block = Counter(r["spatial_block_id"] for r in rows)
    per_fold = Counter(r["cv_fold"] for r in rows)
    per_fire = Counter(r["fire_id"] for r in rows)
    per_barrier = Counter(r["barrier_crossed"] for r in rows)
    heldout = [r for r in rows if r["split_role"] == "heldout"]
    top_fire, top_n = (per_fire.most_common(1) or [(None, 0)])[0]

    return {
        "schema": "wildfire_nowcast.crossings/1",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "wildfire_nowcast.data.crossings.build_index",
        "split_fingerprint": split.get("fingerprint"),
        "split_train_folds": split.get("train_folds"),
        "split_heldout_fold_note": (
            "cv_fold and spatial_block_id are properties of the SPLIT, not of the "
            "fire. They are valid ONLY under split_fingerprint above. If it does "
            "not match the split on disk, REGENERATE — do not read the folds (C8)."
        ),
        "detection_rule": {
            "unit_of_observation": "one detached burned body at one hourly step",
            "connectivity": (
                "8-connected (common.components.label_components; the SAME "
                "implementation C2's n_ignition_components uses, per C0). A body "
                "is DETACHED at hour t when none of its cells has an ever-burned "
                "8-neighbour at hour t-1, i.e. new(t) & ~dilate(ever(t-1), 1)."
            ),
            "distance_definition": (
                "gap_km = closest approach, in kilometres, between the detached "
                "body and the ENTIRE ever-burned region at t-1 (Euclidean, cell "
                "centres, cell_size_m from the store). NOT centroid separation."
            ),
            "distance_units": "km",
            "min_gap_km": min_gap_km,
            "min_gap_km_rationale": (
                "GOFER is ~2 km effective on a 1 km grid, so km-scale label noise "
                "and a small crossing are the same size. The largest gap a single "
                "omitted cell can manufacture between two 8-connected bodies is "
                "2.236 km (dy=2,dx=1). 3.0 km is strictly above that, 1.5x the "
                "label product's effective resolution, and at/beyond the contagion "
                "kernel's configured radius (3 cells) so a counted event is "
                "unreachable by the short-range component by construction. It is "
                "NOT fitted to a count: see sensitivity.by_min_gap_km."
            ),
            "min_event_cells": min_cells,
            "min_event_cells_rationale": (
                "1, deliberately. A spot fire occupies one cell at the moment it "
                "is first mapped; requiring >=2 changes the estimand to 'spots "
                "that had already grown'. Single-cell bodies are at or below the "
                "label product's effective resolution and are flagged per event "
                "with at_or_below_label_resolution. This is the threshold the "
                "count is genuinely sensitive to and the full sweep is reported."
            ),
            "exclusions": {
                "rasterisation_jitter": f"gap_km < {min_gap_km} km",
                "separate_ignition": (
                    "never merges AND gap_km > "
                    f"{SEPARATE_IGNITION_KM} km — a filing artifact, not a spot "
                    "(ADR-019 §7 / ADR-028 (1)). A body that MERGES is never "
                    "excluded however far it landed: merging is the normal fate "
                    "of a real spot fire."
                ),
                "too_small": f"n_cells < {min_cells}",
            },
            "masks_consulted": {
                "water_barrier_mask": (
                    "C1 channel 12. JRC/GSW1_4 permanent water (occurrence >= 80%) "
                    "by area fraction >= 0.30 OR any-permanent-water line OR "
                    "TIGER/2016 primary/secondary road line. Water and road are "
                    "MERGED in this channel by design, so 'road_or_narrow_water' "
                    "cannot be read as 'a highway'."
                ),
                "fuel_model_id": (
                    f"C1 channel 9. FBFM40 non-burnable classes {sorted(FBFM40_NONBURNABLE)}; "
                    f"class {FBFM40_OPEN_WATER} (NB8 Open Water) is used only to "
                    "split channel 12's merged flag, and LANDFIRE is 3-6 years "
                    "stale, so that split is weak evidence."
                ),
                "elevation": (
                    "C1 channel 5 (3DEP). ridge_relief_m = max elevation along the "
                    f"corridor minus the higher endpoint; > {RIDGE_RELIEF_M} m is "
                    "labelled 'ridge'. Raw relief is reported for re-thresholding."
                ),
            },
            "corridor": (
                "Supercover segment between the anchor (nearest already-burned "
                "cell) and the landing cell (nearest cell of the new body), "
                "endpoints stripped. A 1-cell dilated count is reported beside it "
                "so corridor-width sensitivity is visible."
            ),
            "known_blind_spots": [
                "A crossing whose barrier cell is ITSELF labelled burned stays "
                "8-connected and produces no detached birth. Per-fire "
                "n_barrier_cells_ever_burned sizes that blind spot.",
                "Sub-cell barriers (a 60 m river inside one 1 km cell) are not "
                "jumps at this resolution and are not detectable as such.",
                "Two hours of dormancy followed by re-detection at the same site "
                "is not a crossing and is not counted: gap is measured against "
                "the cumulative ever-burned region, not against state 1.",
            ],
        },
        "n_events": len(rows),
        "n_fires_scanned": len(fires),
        "n_fires_with_events": len(per_fire),
        "n_distinct_blocks": len(per_block),
        "distinct_blocks": sorted(per_block),
        "distinct_folds": sorted(per_fold),
        "events_per_block": {str(k): per_block[k] for k in sorted(per_block)},
        "events_per_fold": {str(k): per_fold[k] for k in sorted(per_fold)},
        "events_per_fire": dict(sorted(per_fire.items())),
        "events_per_barrier_kind": dict(sorted(per_barrier.items())),
        "events_per_label_source": dict(sorted(Counter(r["label_source"] for r in rows).items())),
        "concentration": {
            "most_concentrated_fire": top_fire,
            "most_concentrated_fire_n": top_n,
            "most_concentrated_fire_share": (round(top_n / len(rows), 3) if rows else None),
            "largest_block_n": max(per_block.values()) if per_block else 0,
            "largest_block_share": (
                round(max(per_block.values()) / len(rows), 3) if rows else None
            ),
            "n_single_cell_events": sum(1 for r in rows if r["n_cells"] == 1),
            "note": (
                "Reported BEFORE any verdict, per ADR-043's 90.8%-one-fire "
                "finding: a crossing set concentrated in one fire or one block "
                "cannot support a leave-block-out verdict."
            ),
        },
        "downwind_alignment": _downwind_summary(rows),
        "evidence_quality": _evidence_quality(rows),
        "heldout": {
            "heldout_fold": [
                f for f in sorted(per_fold) if f not in set(split.get("train_folds", []))
            ],
            "n_events": len(heldout),
            "blocks": sorted({r["spatial_block_id"] for r in heldout}),
            "fires": sorted({r["fire_id"] for r in heldout}),
        },
        "sensitivity": _sensitivity(bodies_by_fire, block_of),
        "positive_control": positive_control(),
        "events": rows,
        "fires": sorted(fires, key=lambda f: f["fire_id"]),
    }


def write_crossings(path: Path | None = None) -> Path:
    """Write the index. ADDITIVE ONLY - refuses any path inside the C1 corpus."""
    out = Path(path) if path is not None else crossings_path()
    if Path(fires_dir()) in out.parents or out.name == "norm_stats.json":
        raise RuntimeError("crossings.json is additive: refusing to write into the frozen corpus")
    index = build_index()
    if not index["positive_control"]["passed"]:
        raise RuntimeError("positive control FAILED; refusing to write an unverified scan")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2) + "\n")
    return out


if __name__ == "__main__":  # pragma: no cover
    print(write_crossings())
