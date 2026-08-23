"""Build a 2022-2025 extension fire into ``data/interim/`` - and NOWHERE else.

Why this module exists as a separate door rather than a flag on the normal
builder: while modelling holds a running experiment, C-4 freezes tensors,
manifests, norm stats and splits. The corpus swap into ``data/fires/`` is a
separate, SERIALISED step the maintainer schedules. Every function here
therefore refuses a destination under ``data/fires/`` outright - a guard is
cheaper than remembering, and "temporarily" writing to the live path is exactly
the failure ADR-015 and ADR-021 are both instances of.

Fold and block assignment is **ADDITIVE**, which is the other half of the same
constraint:

* ``spatial_block_id`` is recomputed over the full universe, but every new fire
  id begins with ``2022_``..``2025_`` and therefore sorts *after* every existing
  one. ``assign_blocks`` orders components by their alphabetically-first member,
  so a new component can only take a NEW, higher id and an existing component
  keeps its own. :func:`check_existing_blocks_unmoved` asserts that rather than
  assuming it.
* ``cv_fold`` is **pinned** from the existing manifests for existing fires and
  only chosen for new blocks, by seeding the same least-loaded-fold rule with
  the loads already on disk. Running ``assign_folds`` over the enlarged set
  would rebalance the existing twelve, which is precisely the ADR-015 defect.
  Every interim manifest records ``fold_assignment_provisional: true``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.paths import fires_dir, interim_dir, norm_stats_path
from wildfire_nowcast.data.assemble import write_fire_tensor
from wildfire_nowcast.data.folds import (
    FireFoldInput,
    assign_blocks,
    assign_folds,
    fire_domain_inputs,
)
from wildfire_nowcast.data.gofer_ext import (
    FireSpec,
    dump_run,
    run_gofer_ext,
    spec_from_wfigs,
    to_label_build,
)
from wildfire_nowcast.data.labels import DEFAULT_BUFFER_M
from wildfire_nowcast.data.pipeline import build_fire_tensor
from wildfire_nowcast.data.sources.nifc import WfigsFire

__all__ = [
    "N_FOLDS",
    "interim_paths",
    "existing_assignments",
    "check_existing_blocks_unmoved",
    "additive_assignment",
    "build_extension_fire",
]

N_FOLDS = 5


def interim_paths(fire_id: str) -> tuple[Path, Path, Path]:
    """``(tensor, manifest, run_report)`` under ``data/interim/{fire_id}/``."""
    base = interim_dir() / fire_id
    tensor = base / "tensor.zarr"
    if fires_dir() in tensor.parents:  # pragma: no cover - structural guard
        raise RuntimeError("refusing to build into the C1 path while C-4 is in force")
    return tensor, base / "manifest.json", base / "gofer_ext_run.json"


def existing_assignments() -> dict[str, dict[str, Any]]:
    """``fire_id -> {cv_fold, spatial_block_id, n_hours, bbox_5070}`` from disk.

    Read-only over ``data/fires/*/manifest.json``. These values are AUTHORITATIVE
    and are never recomputed here.
    """
    out: dict[str, dict[str, Any]] = {}
    for manifest in sorted(fires_dir().glob("*/manifest.json")):
        man = json.loads(manifest.read_text())
        out[str(man["fire_id"])] = {
            "cv_fold": int(man["cv_fold"]),
            "spatial_block_id": int(man["spatial_block_id"]),
            "n_hours": int(man["n_hours"]),
            "bbox_5070": tuple(float(v) for v in man["bbox_5070"]),
        }
    return out


def _new_inputs(new: dict[str, tuple[Any, int]]) -> list[FireFoldInput]:
    return [
        FireFoldInput(fire_id=fid, bbox_5070=tuple(float(x) for x in bbox), n_hours=hours)
        for fid, (bbox, hours) in new.items()
    ]


def check_existing_blocks_unmoved(
    existing: dict[str, dict[str, Any]], blocks: dict[str, int]
) -> list[str]:
    """Fire ids whose ``spatial_block_id`` would change. MUST come back empty.

    A silent block shift is invisible to every per-tensor check - it is the
    ADR-015 shape exactly - so this is asserted, not trusted.
    """
    return [
        fid for fid, v in existing.items() if fid in blocks and blocks[fid] != v["spatial_block_id"]
    ]


def additive_assignment(
    new_fires: dict[str, tuple[Any, int]], *, k: int = N_FOLDS
) -> dict[str, dict[str, int]]:
    """Provisional ``{fire_id: {cv_fold, spatial_block_id}}`` for NEW fires only.

    ``new_fires`` maps ``fire_id -> (bbox_5070, n_hours)``.

    THE BASE UNIVERSE IS ALL 28 GOFER FIRES, not the 12 built so far. That is
    not a detail: ``assign_blocks`` states it explicitly, and computing over the
    built subset instead reproduces NONE of the 12 manifests - the guard below
    caught exactly that on the first run and named nine fires that would have
    silently changed block. Verified here rather than argued: over the 28-fire
    universe, ``assign_blocks`` and ``assign_folds`` reproduce all 12 built
    manifests' ``spatial_block_id`` and ``cv_fold`` with zero mismatches.
    """
    base = fire_domain_inputs()
    existing = existing_assignments()
    blocks = assign_blocks([*base, *_new_inputs(new_fires)])
    baseline_blocks = assign_blocks(base)
    moved = check_existing_blocks_unmoved(existing, blocks)
    moved += [f.fire_id for f in base if blocks[f.fire_id] != baseline_blocks[f.fire_id]]
    if moved:
        raise RuntimeError(
            "adding these fires would MOVE the spatial_block_id of already-assigned "
            f"fires {sorted(set(moved))}; that is the ADR-015 hazard and must go to "
            "the maintainer as a BLOCKER, not be worked around"
        )

    # Folds are PINNED, never rebalanced: the built 12 from their manifests, the
    # other 16 GOFER fires from the canonical 28-fire assignment they will get.
    canonical = assign_folds(base, k=k)
    load = [0] * k
    fold_of_block: dict[int, int] = {}
    for f in base:
        fold = existing.get(f.fire_id, {}).get("cv_fold", canonical[f.fire_id])
        load[fold] += f.n_hours
        fold_of_block[blocks[f.fire_id]] = fold

    # new blocks, heaviest first, into the least-loaded fold - the same rule
    # assign_folds uses, but seeded with the loads already on disk so no
    # existing fire can move.
    by_block: dict[int, list[str]] = {}
    for fid in new_fires:
        by_block.setdefault(blocks[fid], []).append(fid)
    hours_of_block = {b: sum(new_fires[f][1] for f in fids) for b, fids in by_block.items()}
    out: dict[str, dict[str, int]] = {}
    for block in sorted(by_block, key=lambda b: (-hours_of_block[b], b)):
        if block in fold_of_block:  # joined an existing block: inherit its fold (C3.1)
            fold = fold_of_block[block]
        else:
            fold = min(range(k), key=lambda f: (load[f], f))
            fold_of_block[block] = fold
        for fid in by_block[block]:
            out[fid] = {"cv_fold": fold, "spatial_block_id": block}
            load[fold] += new_fires[fid][1]
    return out


def plan_extension(fires: list[WfigsFire], *, k: int = N_FOLDS) -> dict[str, dict[str, Any]]:
    """Provisional block + fold for EVERY candidate at once, before any build.

    Assignment must see all candidates together - computing it one fire at a
    time would give two genuinely-overlapping new fires two different blocks,
    which is the landscape leakage C3.1 exists to stop.

    The domain used is the WFIGS reference perimeter bbox + 10 km, which is a
    SUPERSET of the C1 domain (the derived perimeter is inside the reference by
    construction, since stray removal clips to it). A superset can only merge
    blocks that the true domains would have left separate, never split them -
    conservative in the direction that avoids leakage, and recorded as such.
    """
    from wildfire_nowcast.data.sources.nifc import final_perimeter as _ref  # noqa: PLC0415

    new: dict[str, tuple[Any, int]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for fire in fires:
        geom = _ref(fire.irwin_id)
        minx, miny, maxx, maxy = geom.bounds
        bbox = (
            minx - DEFAULT_BUFFER_M,
            miny - DEFAULT_BUFFER_M,
            maxx + DEFAULT_BUFFER_M,
            maxy + DEFAULT_BUFFER_M,
        )
        slug = "".join(ch if ch.isalnum() else "_" for ch in fire.name.lower()).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        fire_id = f"{fire.discovery_utc.year}_{slug}"
        # Hours are unknown until the algorithm runs; fold LOAD balancing only
        # needs a monotone proxy, and area is one. Blocks do not use it at all.
        new[fire_id] = (bbox, max(int(fire.area_km2), 1))
        meta[fire_id] = {
            "name": fire.name,
            "year": fire.discovery_utc.year,
            "area_km2": round(fire.area_km2, 1),
            "irwin_id": fire.irwin_id,
            "reference_bbox_5070": [round(v, 1) for v in bbox],
        }
    assigned = additive_assignment(new, k=k)
    for fire_id, row in assigned.items():
        meta[fire_id].update(row)
    return meta


def build_extension_fire(
    fire: WfigsFire | FireSpec,
    *,
    cv_fold: int | None = None,
    spatial_block_id: int | None = None,
    wfigs_fire: WfigsFire | None = None,
) -> dict[str, Any]:
    """Labels -> 14 channels -> ``data/interim/{fire_id}/``. Two-pass by design.

    Pass 1 derives the labels (and therefore the C1 domain, which is the derived
    final perimeter's bbox + 10 km - it cannot be known before the algorithm
    runs). Pass 2 asks for the fold/block assignment for that domain and writes.
    """
    # A caller that already resolved the FireSpec must still hand over the
    # WFIGS record, or the neighbour exclusion silently becomes a no-op -
    # which is exactly what happened on the first pass and is why no shipped
    # fire has been through it.
    wfigs = (
        wfigs_fire if wfigs_fire is not None else (fire if isinstance(fire, WfigsFire) else None)
    )
    spec = fire if isinstance(fire, FireSpec) else spec_from_wfigs(fire)
    run = run_gofer_ext(spec, wfigs_fire=wfigs)
    labels = to_label_build(run)

    if cv_fold is None or spatial_block_id is None:
        bbox = tuple(
            v + s * DEFAULT_BUFFER_M
            for v, s in zip(labels.grid.bounds, (-1, -1, 1, 1), strict=True)
        )
        assigned = additive_assignment({spec.fire_id: (bbox, int(len(labels.times)))})
        cv_fold = assigned[spec.fire_id]["cv_fold"]
        spatial_block_id = assigned[spec.fire_id]["spatial_block_id"]

    bundle, timings = build_fire_tensor(
        spec.fire_id,
        cv_fold=cv_fold,
        spatial_block_id=spatial_block_id,
        label_build=labels,
    )
    bundle.provenance["fold_assignment_provisional"] = True
    bundle.provenance["fold_assignment_note"] = (
        "cv_fold and spatial_block_id are PROVISIONAL. Existing fires' assignments "
        "were pinned from data/fires/*/manifest.json and were verified unmoved; the "
        "authoritative assignment happens in the serialised corpus swap, not here "
        "(C-4 / ADR-015)."
    )
    bundle.provenance["interim_reason"] = (
        "C-4 concurrency freeze: modelling holds a running experiment against "
        "split fingerprint 4848f491e8d588fa. This tensor is contract-true and "
        "deliberately NOT in data/fires/."
    )
    bundle.provenance["built_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    tensor_path, manifest_path, report_path = interim_paths(spec.fire_id)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    written_tensor, written_manifest = write_fire_tensor(
        bundle,
        norm_stats_path=str(norm_stats_path()),
        tensor_path=tensor_path,
        manifest_path=manifest_path,
    )
    dump_run(run, report_path)
    return {
        "fire_id": spec.fire_id,
        "tensor": str(written_tensor),
        "manifest": str(written_manifest),
        "run_report": str(report_path),
        "n_hours": int(len(labels.times)),
        "grid": list(labels.grid.shape),
        "cv_fold": int(cv_fold),
        "spatial_block_id": int(spatial_block_id),
        "fuel_vintage_lag_years": bundle.fuel_vintage_lag_years,
        "n_ignition_components": bundle.n_ignition_components,
        "build_timings_s": dict(timings.stages),
        "diagnostics": run.diagnostics,
    }
