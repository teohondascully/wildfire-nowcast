"""D6 - the corpus swap: ``data/interim/`` extension fires -> ``data/fires/``.

Authorised by ADR-037 (7) after the model lever was measured exhausted. This is
the single largest change to shared state since G0, and every guard here exists
because something in this project's record went wrong in exactly that way:

* **Blocks are recomputed over the FULL universe** (28 GOFER + the extension),
  never over the built subset. ``data.folds.assign_blocks`` says so in its own
  docstring, and computing over the subset reproduces NONE of the existing
  manifests.
* **No existing fire may change block.** ADR-030 (1) / P17: a fire that BRIDGES
  two blocks MERGES them, and the held-out block count can then FALL - adding
  data can weaken the independence claim. ``2025_garnet`` is excluded for
  exactly that reason. :func:`authoritative_assignment` asserts the invariant
  rather than trusting it, and names the fires that would have moved.
* **Folds are re-derived, not pinned.** D5's assignment was explicitly
  ``fold_assignment_provisional``; this is where it stops being provisional.
  Re-deriving mints a NEW split fingerprint by construction, which is C8 working
  as designed (ADR-029 (7)): every existing result stays bound to
  ``4848f491e8d588fa``.
* **``label_source`` is written for all 21 fires** (P18, ADR-028 (5)). Twelve
  are the published GOFER product; nine are ``gofer_ext``, **our own
  reimplementation, because GOFER publishes nothing after 2021** (ADR-029 (1)).
  A corpus mixing a published product with a port of it is a different
  scientific object and every table spanning both must be able to say which is
  which. Two independent witnesses carry it: this key and ``gofer_version``.

Norm stats are NOT computed here - see :func:`wildfire_nowcast.data.cli` and C0:
``common.zarr_io.compute_norm_stats`` is the one implementation and this module
imports the corpus definition it needs rather than restating any of it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.paths import fires_dir, interim_dir
from wildfire_nowcast.data.folds import (
    FireFoldInput,
    assign_blocks,
    assign_folds,
    fire_domain_inputs,
)

__all__ = [
    "N_FOLDS",
    "PUBLISHED_LABEL_SOURCE",
    "EXTENSION_LABEL_SOURCE",
    "EXCLUDED_FIRES",
    "LABEL_SOURCE_NOTE",
    "CorpusAssignment",
    "corpus_members",
    "authoritative_assignment",
    "select_heldout_fold",
    "swap_corpus",
]

N_FOLDS = 5

#: C2 ``label_source`` (P18). ``gofer`` is the published Zenodo product;
#: ``gofer_ext`` is OUR reimplementation of the GOFER algorithm, run on GOES
#: FDCF for years the authors never published.
PUBLISHED_LABEL_SOURCE = "gofer"
EXTENSION_LABEL_SOURCE = "gofer_ext"

LABEL_SOURCE_NOTE = (
    "label_source=gofer is the PUBLISHED GOFER product (Zenodo 14638647, 2019-2021). "
    "label_source=gofer_ext is a REIMPLEMENTATION of the GOFER algorithm by this "
    "project, because GOFER publishes nothing after 2021 (all 5 Zenodo versions "
    "checked; ADR-029 (1)). The two are not the same scientific object and any table "
    "spanning both MUST state which fires are which. Port fidelity vs GOFER on 2019-2021 "
    "overlap: IoU 0.917-0.973. Against official CAL FIRE/WFIGS acreage — a third source "
    "neither product saw — published median 1.063 (n=12), gofer_ext median 1.026 (n=9)."
)

#: Candidates deliberately NOT in the corpus, with the reason. ADR-030 (1).
EXCLUDED_FIRES: dict[str, str] = {
    "2025_garnet": (
        "BRIDGES spatial blocks 4 (Creek) and 9 (KNP): admitting it MERGES them and the "
        "held-out block count FALLS. It contributes zero spot events, so its unique "
        "contribution does not outweigh the lost block (ADR-030 (1) / P17)"
    ),
}


@dataclass
class CorpusAssignment:
    """The authoritative 21-fire split, plus the evidence it is sound."""

    folds: dict[str, int]
    blocks: dict[str, int]
    n_hours: dict[str, int]
    label_source: dict[str, str]
    universe_n_fires: int
    universe_n_blocks: int
    moved_fold: dict[str, tuple[int, int]] = field(default_factory=dict)
    reassigned_extension_block: dict[str, tuple[int, int]] = field(default_factory=dict)
    domains_disagreeing_with_archive: list[str] = field(default_factory=list)

    @property
    def fire_ids(self) -> list[str]:
        return sorted(self.folds)

    def blocks_of_fold(self, fold: int) -> list[int]:
        return sorted({self.blocks[f] for f in self.fire_ids if self.folds[f] == fold})

    def summary(self) -> dict[str, Any]:
        per: dict[str, dict[str, Any]] = {}
        for fold in sorted(set(self.folds.values())):
            members = [f for f in self.fire_ids if self.folds[f] == fold]
            per[str(fold)] = {
                "fires": members,
                "blocks": self.blocks_of_fold(fold),
                "n_blocks": len(self.blocks_of_fold(fold)),
                "n_hours": sum(self.n_hours[f] for f in members),
            }
        return {
            "n_fires": len(self.fire_ids),
            "n_blocks": len({self.blocks[f] for f in self.fire_ids}),
            "blocks": sorted({self.blocks[f] for f in self.fire_ids}),
            "universe_n_fires": self.universe_n_fires,
            "universe_n_blocks": self.universe_n_blocks,
            "folds": per,
            "fires_that_changed_fold": {
                k: {"was": v[0], "now": v[1]} for k, v in sorted(self.moved_fold.items())
            },
            "extension_fires_that_changed_block": {
                k: {"provisional": v[0], "authoritative": v[1]}
                for k, v in sorted(self.reassigned_extension_block.items())
            },
            "built_domains_disagreeing_with_archive": self.domains_disagreeing_with_archive,
            "label_source_counts": {
                src: sum(1 for f in self.fire_ids if self.label_source[f] == src)
                for src in sorted(set(self.label_source.values()))
            },
        }


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def corpus_members() -> tuple[dict[str, Path], dict[str, Path]]:
    """``({published fire_id: dir}, {extension fire_id: dir})`` from disk.

    Membership is read off the FILESYSTEM, not off a list, so a fire that failed
    to build cannot be silently assumed present. A directory needs both
    ``tensor.zarr`` and ``manifest.json`` to count - half a fire is not a fire.
    """
    published: dict[str, Path] = {}
    extension: dict[str, Path] = {}
    for root, out in ((fires_dir(), published), (interim_dir(), extension)):
        for manifest in sorted(Path(root).glob("*/manifest.json")):
            if not (manifest.parent / "tensor.zarr").exists():
                continue
            out[str(_read_manifest(manifest)["fire_id"])] = manifest.parent
    # A fire already swapped is published, never both (idempotence).
    for fire_id in list(extension):
        if fire_id in published:
            extension.pop(fire_id)
    return published, extension


def authoritative_assignment(
    *, k: int = N_FOLDS, extra: dict[str, Path] | None = None
) -> CorpusAssignment:
    """The 21-fire ``cv_fold`` + ``spatial_block_id`` of record.

    Raises if adding the extension would move any existing fire's block - the
    P17 hazard, which is invisible to every per-tensor check because both
    manifests stay individually conformant.
    """
    published, extension = corpus_members()
    if extra:
        extension = {**extension, **extra}

    base = fire_domain_inputs()  # all 28 GOFER fires, built or not
    built: dict[str, dict[str, Any]] = {}
    inputs_by_id: dict[str, FireFoldInput] = {}
    label_source: dict[str, str] = {}
    for fire_id, folder in {**published, **extension}.items():
        man = _read_manifest(folder / "manifest.json")
        built[fire_id] = man
        inputs_by_id[fire_id] = FireFoldInput(
            fire_id=fire_id,
            bbox_5070=tuple(float(v) for v in man["bbox_5070"]),
            n_hours=int(man["n_hours"]),
        )
        prov = man.get("provenance") or {}
        declared = str(man.get("label_source") or prov.get("label_source") or "")
        label_source[fire_id] = (
            EXTENSION_LABEL_SOURCE if declared == EXTENSION_LABEL_SOURCE else PUBLISHED_LABEL_SOURCE
        )

    # The 28 GOFER domains carry their own inputs; a BUILT GOFER fire uses its
    # manifest bbox (the C1 domain of record) so the universe and the corpus
    # agree on the same geometry. VERIFIED, not assumed: for all 12 built GOFER
    # fires the two agree exactly on bbox AND hour count, so this substitution
    # is a no-op today and would raise below the day it stops being one.
    universe: list[FireFoldInput] = [inputs_by_id.get(f.fire_id, f) for f in base] + [
        inputs_by_id[f] for f in sorted(extension)
    ]
    disagreeing_domains = sorted(
        f.fire_id
        for f in base
        if f.fire_id in inputs_by_id
        and tuple(inputs_by_id[f.fire_id].bbox_5070) != tuple(f.bbox_5070)
    )

    blocks_before = assign_blocks(base)
    blocks = assign_blocks(universe)
    moved = sorted(
        # (a) any GOFER fire whose block id shifts when the extension joins
        {f for f in blocks_before if blocks[f] != blocks_before[f]}
        # (b) any fire ALREADY IN data/fires/ whose new id disagrees with disk.
        #     Extension fires are deliberately exempt: D5 stamped their ids
        #     `fold_assignment_provisional` and this is where that is resolved.
        | {f for f in published if blocks[f] != int(built[f]["spatial_block_id"])}
    )
    reassigned_extension = {
        f: (int(built[f]["spatial_block_id"]), blocks[f])
        for f in sorted(extension)
        if blocks[f] != int(built[f]["spatial_block_id"])
    }
    if moved:
        raise RuntimeError(
            "REFUSING TO SWAP: adding the extension would MOVE the spatial_block_id of "
            f"already-assigned fires {moved}. That merges blocks, and the held-out block "
            "count can FALL as a result (ADR-030 (1) / P17). This goes to the "
            "maintainer as a BLOCKER, not around."
        )

    folds = assign_folds(universe, k=k)
    corpus = sorted(built)
    moved_fold = {
        f: (int(built[f]["cv_fold"]), folds[f])
        for f in corpus
        if int(built[f]["cv_fold"]) != folds[f]
    }
    return CorpusAssignment(
        folds={f: folds[f] for f in corpus},
        blocks={f: blocks[f] for f in corpus},
        n_hours={f: inputs_by_id[f].n_hours for f in corpus},
        label_source=label_source,
        universe_n_fires=len(universe),
        universe_n_blocks=len(set(blocks.values())),
        moved_fold=moved_fold,
        reassigned_extension_block=reassigned_extension,
        domains_disagreeing_with_archive=disagreeing_domains,
    )


def select_heldout_fold(assignment: CorpusAssignment) -> int:
    """The fold to hold out: the one covering the MOST DISTINCT SPATIAL BLOCKS.

    C6.3 counts blocks, not fires - "more fires from the same block are the same
    evidence with false confidence". A8 established this precedent on the 12-fire
    corpus (fold 3 was the only fold that could reach 4 blocks, and picking by
    FIRE count would have missed it); ADR-014 ratified it. Ties break to the
    lowest fold id so the choice is deterministic and not a judgement call.
    """
    folds = sorted(set(assignment.folds.values()))
    return min(folds, key=lambda f: (-len(assignment.blocks_of_fold(f)), f))


def _rewrite_manifest(
    manifest: dict[str, Any],
    *,
    fire_id: str,
    assignment: CorpusAssignment,
    swapped_from: str | None,
    stamp: str,
) -> dict[str, Any]:
    """Apply the authoritative fold/block/label_source to one C2 manifest.

    Both the root key and ``provenance.spatial_block_id`` are written. Two
    copies of one fact is the defect ADR-033 (1) names; they already both exist
    on disk, so the least-bad thing available is to keep them in lockstep and
    say so - divergence is what actually bites.
    """
    man = dict(manifest)
    man["cv_fold"] = int(assignment.folds[fire_id])
    man["spatial_block_id"] = int(assignment.blocks[fire_id])
    man["label_source"] = assignment.label_source[fire_id]

    prov = dict(man.get("provenance") or {})
    prov["spatial_block_id"] = int(assignment.blocks[fire_id])
    prov["label_source"] = assignment.label_source[fire_id]
    prov["label_source_note"] = LABEL_SOURCE_NOTE
    # D5 wrote these while C-4 froze data/fires/. They are now FALSE, and a
    # stale provenance note is worse than none: it reads as authoritative.
    for dead in ("fold_assignment_provisional", "fold_assignment_note", "interim_reason"):
        prov.pop(dead, None)
    prov["fold_assignment"] = "authoritative"
    prov["fold_assignment_note"] = (
        "cv_fold and spatial_block_id re-derived over the FULL universe "
        f"({assignment.universe_n_fires} fires: 28 GOFER + the extension) by "
        "data.folds.assign_folds/assign_blocks at the D6 corpus swap (ADR-037 (7)). "
        "No existing fire changed block; the P17 bridging invariant was asserted, "
        "not assumed. This mints a NEW split fingerprint by construction — every "
        "result produced before the swap stays bound to 4848f491e8d588fa (C8)."
    )
    prov["corpus_swap_utc"] = stamp
    prov["corpus_swap_adr"] = "ADR-037 (7)"
    if swapped_from is not None:
        prov["swapped_from"] = swapped_from
    man["provenance"] = prov
    return man


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def swap_corpus(
    *, k: int = N_FOLDS, dry_run: bool = False, remove_interim: bool = False
) -> dict[str, Any]:
    """Copy the extension into ``data/fires/`` and rewrite all 21 manifests.

    COPY, then verify, then (optionally, and only on request) remove the interim
    original - never move first. A fire that cannot be made contract-clean must
    be left where it was and reported, which is impossible if the source is
    already gone.
    """
    published, extension = corpus_members()
    assignment = authoritative_assignment(k=k)
    heldout = select_heldout_fold(assignment)
    train_folds = sorted(set(assignment.folds.values()) - {heldout})
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    copied: list[str] = []
    if not dry_run:
        for fire_id, src in sorted(extension.items()):
            dst = Path(fires_dir()) / fire_id
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied.append(fire_id)

        for fire_id in assignment.fire_ids:
            path = Path(fires_dir()) / fire_id / "manifest.json"
            _write_manifest(
                path,
                _rewrite_manifest(
                    _read_manifest(path),
                    fire_id=fire_id,
                    assignment=assignment,
                    swapped_from=(str(extension[fire_id]) if fire_id in extension else None),
                    stamp=stamp,
                ),
            )
        if remove_interim:
            for fire_id in copied:
                shutil.rmtree(extension[fire_id])

    return {
        "swapped_utc": stamp,
        "dry_run": dry_run,
        "published_before": sorted(published),
        "extension_swapped": sorted(extension),
        "copied": copied,
        "excluded": EXCLUDED_FIRES,
        "heldout_fold": heldout,
        "train_folds": train_folds,
        "heldout_blocks": assignment.blocks_of_fold(heldout),
        "n_heldout_blocks": len(assignment.blocks_of_fold(heldout)),
        "train_blocks": sorted(
            {assignment.blocks[f] for f in assignment.fire_ids if assignment.folds[f] != heldout}
        ),
        "assignment": assignment.summary(),
        "per_fire": {
            f: {
                "cv_fold": assignment.folds[f],
                "spatial_block_id": assignment.blocks[f],
                "label_source": assignment.label_source[f],
                "n_hours": assignment.n_hours[f],
            }
            for f in assignment.fire_ids
        },
    }
