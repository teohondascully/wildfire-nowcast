"""C2 ``cv_fold``: leave-fire-out cross-validation fold assignment.

Two leakage routes have to be closed, not one:

1. *Temporal/identity* — a fire must be wholly in one fold. That is the obvious
   one and is what "leave-fire-out" names.
2. *Spatial* — two fires whose domains overlap share terrain, fuels, barriers
   and often weather. Splitting them across folds lets the model memorise the
   landscape and score well on the held-out fire for the wrong reason. 2020 in
   California is full of these (the August/North/LNU/SCU complexes burned within
   weeks of each other, and Zogg started inside ground the Carr fire had
   touched). Overlapping fires are therefore forced into the same fold.

Assignment is deterministic (no RNG): build spatial groups by domain overlap,
then greedily pack groups into ``k`` folds largest-first, balancing total
timesteps. Same inputs always give the same folds, so a manifest written today
and one written next month agree.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "FireFoldInput",
    "assign_blocks",
    "assign_folds",
    "fold_summary",
    "fire_domain_inputs",
    "fire_assignments",
]


@dataclass(frozen=True)
class FireFoldInput:
    """What fold assignment needs to know about a fire."""

    fire_id: str
    bbox_5070: tuple[float, float, float, float]
    n_hours: int


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _spatial_groups(fires: list[FireFoldInput]) -> list[list[int]]:
    """Union-find over bbox overlap."""
    parent = list(range(len(fires)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(fires)):
        for j in range(i + 1, len(fires)):
            if _overlaps(fires[i].bbox_5070, fires[j].bbox_5070):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    groups: dict[int, list[int]] = {}
    for i in range(len(fires)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for _, v in sorted(groups.items())]


def assign_blocks(fires: Iterable[FireFoldInput]) -> dict[str, int]:
    """Map ``fire_id -> spatial_block_id`` (C2, C3.1). Deterministic.

    The id is the index of the fire's connected component among components
    ordered by their alphabetically-first member. **Always compute this over the
    full 28-fire universe**, never over the subset built so far: block ids are
    written into manifests one fire at a time and must not shift as the dataset
    grows. ``n_train_blocks`` in C3 counts DISTINCT ids over the train folds, so
    two fires sharing a block contribute one block, not two.
    """
    items = sorted(fires, key=lambda f: f.fire_id)
    groups = _spatial_groups(items)
    ordered = sorted(groups, key=lambda g: items[min(g)].fire_id)
    return {items[i].fire_id: block_id for block_id, group in enumerate(ordered) for i in group}


def assign_folds(fires: Iterable[FireFoldInput], k: int = 5) -> dict[str, int]:
    """Map ``fire_id -> cv_fold`` in ``[0, k)``. Deterministic.

    ``k = len(fires)`` gives strict leave-one-fire-out (subject to the spatial
    constraint, which may merge two fires into one fold and leave a fold empty —
    that is reported by :func:`fold_summary`, not hidden).
    """
    items = sorted(fires, key=lambda f: f.fire_id)
    if not items:
        return {}
    if k < 1:
        raise ValueError("k must be >= 1")

    groups = _spatial_groups(items)
    # largest group first so the big blocks land before the folds fill up
    ordered = sorted(groups, key=lambda g: (-sum(items[i].n_hours for i in g), items[g[0]].fire_id))
    load = [0] * k
    out: dict[str, int] = {}
    for group in ordered:
        target = min(range(k), key=lambda f: (load[f], f))
        for i in group:
            out[items[i].fire_id] = target
            load[target] += items[i].n_hours
    return out


def fire_domain_inputs(
    archive: Any = None, *, buffer_m: float | None = None
) -> list[FireFoldInput]:
    """Buffered domain bbox + hour count for every GOFER fire.

    Reads perimeter geometry only — no rasterisation, no state rule — because
    fold and block assignment depend on the domain, not on the labels.
    """
    from wildfire_nowcast.common.contract import CRS_STRING  # noqa: PLC0415
    from wildfire_nowcast.data.gofer import GoferArchive  # noqa: PLC0415
    from wildfire_nowcast.data.labels import DEFAULT_BUFFER_M, fire_domain_grid  # noqa: PLC0415

    arch = archive or GoferArchive()
    buf = DEFAULT_BUFFER_M if buffer_m is None else buffer_m
    out: list[FireFoldInput] = []
    for fire_id in arch.fire_ids():
        perims = arch.perimeters(fire_id, to_crs=CRS_STRING)
        grid = fire_domain_grid(tuple(perims.total_bounds), buffer_m=buf)
        out.append(FireFoldInput(fire_id, tuple(grid.bounds), int(len(perims))))
    return out


def fire_assignments(
    archive: Any = None,
    *,
    k: int = 5,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """``fire_id -> {cv_fold, spatial_block_id, n_hours, bbox_5070}`` for all 28 fires.

    Computed over the WHOLE fire list every time (cached on disk), so a manifest
    written for fire 2 and a manifest written for fire 27 carry consistent fold
    and block ids. Building the subset that happens to exist today would silently
    renumber blocks as the dataset grows, which would corrupt C3.3's
    ``n_train_blocks`` count after the fact.
    """
    from wildfire_nowcast.common.paths import interim_dir  # noqa: PLC0415

    path = Path(cache_path) if cache_path else interim_dir() / "_index" / "assignments.json"
    if path.is_file() and not refresh:
        cached = json.loads(path.read_text())
        if int(cached.get("k", -1)) == k:
            return cached["fires"]

    fires = fire_domain_inputs(archive)
    folds = assign_folds(fires, k=k)
    blocks = assign_blocks(fires)
    out = {
        f.fire_id: {
            "cv_fold": folds[f.fire_id],
            "spatial_block_id": blocks[f.fire_id],
            "n_hours": f.n_hours,
            "bbox_5070": list(f.bbox_5070),
        }
        for f in sorted(fires, key=lambda f: f.fire_id)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "k": k,
                "n_fires": len(out),
                "n_blocks": len(set(blocks.values())),
                "policy": "whole fires held out; spatially overlapping fires share a fold "
                "AND a block (C3.1); ids computed over all GOFER fires, not the built subset",
                "fires": out,
            },
            indent=2,
        )
        + "\n"
    )
    return out


def fold_summary(
    fires: Iterable[FireFoldInput], folds: dict[str, int], k: int = 5
) -> dict[str, Any]:
    """Human-readable balance report; goes into the norm-stats provenance."""
    items = list(fires)
    per: dict[int, dict[str, Any]] = {f: {"fires": [], "n_hours": 0} for f in range(k)}
    for fire in sorted(items, key=lambda f: f.fire_id):
        f = folds[fire.fire_id]
        per[f]["fires"].append(fire.fire_id)
        per[f]["n_hours"] += fire.n_hours
    loads = [per[f]["n_hours"] for f in range(k)]
    return {
        "k": k,
        "n_fires": len(items),
        "folds": {str(f): per[f] for f in range(k)},
        "empty_folds": [f for f in range(k) if not per[f]["fires"]],
        "hours_min": min(loads) if loads else 0,
        "hours_max": max(loads) if loads else 0,
        "imbalance_ratio": (max(loads) / min(loads)) if loads and min(loads) else None,
        "policy": "whole fires held out; spatially overlapping fires forced into one fold",
    }
