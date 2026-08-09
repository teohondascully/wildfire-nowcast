"""The G4 spot-event inventory, machine-readable, derived by the ratified rule.

**Why this is a separate artifact from C2's ``n_ignition_components``, which is
the whole point (ADR-019 / P19).** ``IgnitionReport.spot_candidates`` reports
only detached bodies that NEVER merge with the region preceding them. That is
CORRECT for counting ignitions — a body that merges was never a second fire —
and it is WRONG for counting spot events, because merging is the normal fate of
a real spot fire: an ember lands ahead of the front, ignites, and the front
overruns it an hour later. Read off the same field, the corpus holds 2 spot
events; read correctly, it holds 11.

Classification, all of it from ``data.ignitions`` so there is one rule
(item 30: time, then genealogy, then distance as a tiebreak):

* ``gap_km <= 2.25``           -> rasterisation hole, EXCLUDED (hundreds of them)
* ``gap_km > 15`` and never merges -> separate ignition, EXCLUDED (filing artifact)
* everything else              -> SPOT EVENT, retained, ``kind`` records whether
                                  it later merged

Output carries ``spatial_block_id`` and ``cv_fold`` per event because C6.3
counts BLOCKS, not events, and because G4 is scored leave-block-out: a table of
11 events in one block would be one observation wearing eleven hats.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import fires_dir, interim_dir
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.data.ignitions import (
    SEED_MERGE_KM,
    SPOT_RANGE_MAX_KM,
    count_ignition_components,
)

__all__ = ["SPOT_EVENTS_PATH", "collect_spot_events", "write_spot_events"]

#: Under data/interim/ deliberately: C-4 freezes data/fires/ while a lead is
#: running, and an index that points at a frozen corpus belongs beside the
#: interim corpus it was derived from.
SPOT_EVENTS_PATH = interim_dir() / "_events" / "spot_events.json"


def _tensor_dirs() -> list[Path]:
    """Every built fire, ONCE, with ``data/fires/`` winning any duplicate.

    The dedupe is not defensive tidiness — it is a defect that shipped. After
    the D6 corpus swap COPIED the nine extension fires into ``data/fires/``
    while leaving the originals in ``data/interim/``, this function returned
    both and the index reported **16 events over 30 fires** for a 21-fire corpus
    holding 12 events: every swapped fire counted twice, and a G4 verdict would
    have been computed on duplicated evidence with an inflated event count. It
    was caught by ``n_fires`` reading 30, i.e. by a number nobody could
    misread — which is the whole argument for printing denominators.
    """
    by_id: dict[str, Path] = {}
    for root in (interim_dir(), fires_dir()):  # fires_dir LAST so it overwrites
        for manifest in sorted(Path(root).glob("*/manifest.json")):
            if not (manifest.parent / "tensor.zarr").exists():
                continue
            by_id[str(json.loads(manifest.read_text())["fire_id"])] = manifest
    return [by_id[k] for k in sorted(by_id)]


def collect_spot_events() -> dict[str, Any]:
    """Every spot event in every built tensor, with its block and fold."""
    events: list[dict[str, Any]] = []
    fires: list[dict[str, Any]] = []
    for manifest_path in _tensor_dirs():
        manifest = json.loads(manifest_path.read_text())
        tensor = manifest_path.parent / "tensor.zarr"
        if not tensor.exists():
            continue
        ds = open_tensor(tensor)
        grid = Grid.from_dataset(ds)
        report = count_ignition_components(
            np.asarray(ds["fire_state"].values), cell_size_m=grid.cell_size_m
        )
        fire_id = str(manifest["fire_id"])
        block = int(manifest["spatial_block_id"])
        fold = int(manifest["cv_fold"])
        source = (
            "gofer_ext"
            if str(manifest["provenance"].get("label_source")) == "gofer_ext"
            else "gofer_published"
        )
        n_holes = 0
        for birth in report.detached_births:
            if birth.gap_km <= SEED_MERGE_KM:
                n_holes += 1
                continue
            if (not birth.merges_later) and birth.gap_km > SPOT_RANGE_MAX_KM:
                continue  # separate ignition; a filing artifact, not a spot
            events.append(
                {
                    "fire_id": fire_id,
                    "spatial_block_id": block,
                    "cv_fold": fold,
                    "label_source": source,
                    "hour": int(birth.hour),
                    "gap_km": round(float(birth.gap_km), 2),
                    "n_cells": int(birth.n_cells),
                    "kind": "merging" if birth.merges_later else "never_merging",
                }
            )
        fires.append(
            {
                "fire_id": fire_id,
                "spatial_block_id": block,
                "cv_fold": fold,
                "label_source": source,
                "n_hours": int(manifest["n_hours"]),
                "n_spot_events": sum(1 for e in events if e["fire_id"] == fire_id),
                "n_rasterisation_holes": n_holes,
                "n_ignition_components": int(manifest["n_ignition_components"]),
            }
        )

    by_block: dict[int, int] = defaultdict(int)
    for event in events:
        by_block[event["spatial_block_id"]] += 1
    blocks = sorted(by_block)
    folds = sorted({e["cv_fold"] for e in events})
    # C8. Every event carries a cv_fold and a spatial_block_id, and BOTH are
    # properties of the split, not of the fire — so this index goes stale the
    # moment the corpus changes, silently, while still looking authoritative.
    # It went stale exactly once (the D6 swap re-derived folds and six fires
    # moved). Stamping the fingerprint makes staleness a one-line comparison
    # instead of an assumption.
    from wildfire_nowcast.common.splits import split_fingerprint  # noqa: PLC0415

    fingerprint = split_fingerprint()
    return {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_fingerprint": fingerprint.get("fingerprint"),
        "split_train_folds": fingerprint.get("train_folds"),
        "split_note": (
            "cv_fold and spatial_block_id below are only valid under this "
            "split_fingerprint. If it does not match the split on disk, REGENERATE — "
            "do not read the folds"
        ),
        "rule": (
            "data.ignitions.count_ignition_components (item 30: time, then "
            f"genealogy, then distance). gap <= {SEED_MERGE_KM} km = rasterisation "
            f"hole (excluded); gap > {SPOT_RANGE_MAX_KM} km and never merging = "
            "separate ignition (excluded); everything else = spot event"
        ),
        "why_this_differs_from_c2": (
            "C2 n_ignition_components counts IGNITIONS and therefore excludes any "
            "body that later merges. Merging is the normal fate of a real spot "
            "fire, so that field undercounts spot EVENTS by design, not by error "
            "(ADR-019 / data P19)"
        ),
        "n_events": len(events),
        "n_fires": len(fires),
        "n_distinct_blocks": len(blocks),
        "distinct_blocks": blocks,
        "distinct_folds": folds,
        "events_per_block": {str(b): by_block[b] for b in blocks},
        "events_by_kind": {
            "merging": sum(1 for e in events if e["kind"] == "merging"),
            "never_merging": sum(1 for e in events if e["kind"] == "never_merging"),
        },
        "events": sorted(events, key=lambda e: (e["spatial_block_id"], e["fire_id"], e["hour"])),
        "fires": fires,
    }


def write_spot_events(path: Path | None = None) -> Path:
    """Write the inventory. ``data/interim/`` only; never near the C1 path."""
    out = Path(path) if path is not None else SPOT_EVENTS_PATH
    if fires_dir() in out.parents:
        raise RuntimeError("refusing to write the event index into the frozen C1 path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collect_spot_events(), indent=2) + "\n")
    return out


if __name__ == "__main__":  # pragma: no cover
    print(write_spot_events())
