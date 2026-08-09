"""``python -m wildfire_nowcast.data.cli <command>`` — the data pipeline entry points.

Offline (no Earth Engine):

    fetch                     download + verify + unpack GOFER from Zenodo
    fires                     list the 28 GOFER fire ids
    labels FIRE_ID            rasterise fire_state -> data/interim/{fire_id}/
    qa FIRE_ID                print the QA report without writing anything
    folds                     print the leave-fire-out fold assignment
    assignments               print/refresh cv_fold + spatial_block_id for all fires

Needs Earth Engine auth (ADR-004):

    gee-probe                 one non-interactive readiness check
    build FIRE_ID [...]       full 14-channel C1 tensor -> data/fires/{fire_id}/
    norm-stats                recompute data/norm_stats.json over the TRAIN folds

Offline again, but needs built tensors:

    audit                     physical-plausibility scan of every built fire +
                              the shared norm stats -> data/qa_audit.json, and
                              patched into each manifest's provenance.qa
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from wildfire_nowcast.data.folds import (
    assign_folds,
    fire_assignments,
    fire_domain_inputs,
    fold_summary,
)
from wildfire_nowcast.data.gofer import GoferArchive, download_gofer
from wildfire_nowcast.data.labels import (
    DEFAULT_BUFFER_M,
    build_fire_state,
    write_interim_fire_state,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wildfire-data", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download and unpack GOFER from Zenodo")
    sub.add_parser("fires", help="list GOFER fire ids")
    sub.add_parser("gee-probe", help="one non-interactive Earth Engine readiness check")

    for name in ("labels", "qa"):
        sp = sub.add_parser(name)
        sp.add_argument("fire_id")
        sp.add_argument("--rule", default="fireline_v2", choices=["fireline_v2"])
        sp.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
        sp.add_argument("--variant", default="combined")
        if name == "labels":
            sp.add_argument("--name", default="fire_state",
                            help="output basename under data/interim/{fire_id}/")

    fp = sub.add_parser("folds")
    fp.add_argument("-k", type=int, default=5)

    ap = sub.add_parser("assignments", help="cv_fold + spatial_block_id for every fire")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--refresh", action="store_true")

    bp = sub.add_parser("build", help="build the full C1 tensor for one or more fires")
    bp.add_argument("fire_id", nargs="+")
    bp.add_argument("--quiet", action="store_true")

    np_ = sub.add_parser("norm-stats", help="recompute the C3 file over the train folds")
    np_.add_argument("--train-folds", type=int, nargs="+", default=None,
                     help="fold ids to train on (default: every fold that has a built fire)")
    np_.add_argument("--out", default=None)

    bf = sub.add_parser(
        "backfill-c2-v27",
        help="add the C2 [v2.7] keys to manifests written before the clause existed",
    )
    bf.add_argument("fire_id", nargs="*", default=None)
    bf.add_argument("--dry-run", action="store_true",
                    help="derive and print, write nothing")

    au = sub.add_parser("audit", help="physical-plausibility scan of every built fire")
    au.add_argument("fire_id", nargs="*", default=None)
    au.add_argument("--no-patch", action="store_true",
                    help="do not write the audit into each manifest's provenance.qa")
    au.add_argument("--full", action="store_true", help="print every channel, not just findings")
    return p


def _cmd_build(fire_ids: Sequence[str], *, verbose: bool) -> int:
    from wildfire_nowcast.data.assemble import write_fire_tensor  # noqa: PLC0415
    from wildfire_nowcast.data.pipeline import build_fire_tensor  # noqa: PLC0415

    arch = GoferArchive()
    assignments = fire_assignments(arch)
    rc = 0
    for fire_id in fire_ids:
        if fire_id not in assignments:
            print(f"{fire_id}: not a GOFER fire id", file=sys.stderr)
            rc = 1
            continue
        a = assignments[fire_id]
        bundle, timings = build_fire_tensor(
            fire_id,
            archive=arch,
            cv_fold=a["cv_fold"],
            spatial_block_id=a["spatial_block_id"],
            verbose=verbose,
        )
        tensor_path, manifest_path = write_fire_tensor(bundle)
        verdict = bundle.qa.get("verdict", {})
        print(
            json.dumps(
                {
                    "fire_id": fire_id,
                    "tensor": str(tensor_path),
                    "manifest": str(manifest_path),
                    "n_hours": len(bundle.times),
                    "grid_yx": [bundle.grid.ny, bundle.grid.nx],
                    "cv_fold": a["cv_fold"],
                    "spatial_block_id": a["spatial_block_id"],
                    "build_s": timings.total_s,
                    "stages_s": timings.stages,
                    "qa_pass": verdict.get("pass"),
                    "qa_failures": verdict.get("failures", []),
                }
            )
        )
        if not verdict.get("pass", False):
            rc = 1
    return rc


def _cmd_norm_stats(train_folds: Sequence[int] | None, out: str | None) -> int:
    """C3 — recompute ``data/norm_stats.json`` over the TRAIN folds only.

    **Membership comes from the MANIFESTS ON DISK, never from
    ``fire_assignments()``.** That is a corrected defect, not a style choice.
    ``fire_assignments`` enumerates the 28 GOFER fires, so after the D6 corpus
    swap it does not know the nine ``gofer_ext`` fires exist: this function would
    have opened 12 tensors instead of 21, counted 7 train blocks instead of 9,
    and printed a completely successful report while normalising the corpus on
    57% of it. Silent, green, and wrong — the exact family (all-NaN channel,
    ``-9999`` sentinel, the neighbour defence that never executed) this project
    keeps finding. The manifests are what ``common.splits.split_fingerprint``
    reads, so using them makes the norm stats and the fingerprint agree by
    construction rather than by coincidence.

    Two guards, both of which must be measurements rather than absence-of-match:
    every built tensor is accounted for as train or held-out, and no held-out
    fire's dataset can reach :func:`compute_norm_stats`.
    """
    from wildfire_nowcast.common.paths import fires_dir, norm_stats_path  # noqa: PLC0415
    from wildfire_nowcast.common.splits import split_fingerprint  # noqa: PLC0415
    from wildfire_nowcast.common.zarr_io import (  # noqa: PLC0415
        compute_norm_stats,
        open_tensor,
        write_norm_stats,
    )

    root = Path(fires_dir())
    built: dict[str, dict[str, object]] = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if not (manifest_path.parent / "tensor.zarr").exists():
            continue
        man = json.loads(manifest_path.read_text())
        built[str(man["fire_id"])] = {
            "tensor": manifest_path.parent / "tensor.zarr",
            "cv_fold": int(man["cv_fold"]),
            "spatial_block_id": int(man["spatial_block_id"]),
            "label_source": str(man.get("label_source") or "unknown"),
        }
    if not built:
        print("no built tensors under data/fires/", file=sys.stderr)
        return 1

    folds = (
        sorted({int(v["cv_fold"]) for v in built.values()})
        if train_folds is None
        else sorted(set(int(f) for f in train_folds))
    )
    train_ids = sorted(f for f, v in built.items() if int(v["cv_fold"]) in folds)
    heldout_ids = sorted(set(built) - set(train_ids))
    if not train_ids:
        print(f"no built fire lies in folds {folds}", file=sys.stderr)
        return 1
    if sorted(set(train_ids) | set(heldout_ids)) != sorted(built):
        print("train + held-out does not cover every built fire", file=sys.stderr)
        return 1

    datasets = [open_tensor(built[f]["tensor"]) for f in train_ids]  # type: ignore[arg-type]
    blocks = [int(built[f]["spatial_block_id"]) for f in train_ids]
    stats = compute_norm_stats(datasets, folds, spatial_block_ids=blocks)
    for ds in datasets:
        ds.close()

    stats["train_fire_ids"] = train_ids
    stats["train_spatial_block_ids"] = sorted(set(blocks))
    stats["heldout_fire_ids"] = heldout_ids
    stats["heldout_folds"] = sorted({int(built[f]["cv_fold"]) for f in heldout_ids})
    stats["heldout_spatial_block_ids"] = sorted(
        {int(built[f]["spatial_block_id"]) for f in heldout_ids}
    )
    stats["n_fires_in_corpus"] = len(built)
    stats["train_label_sources"] = {
        src: sorted(f for f in train_ids if built[f]["label_source"] == src)
        for src in sorted({str(built[f]["label_source"]) for f in train_ids})
    }
    stats["membership_source"] = "data/fires/*/manifest.json (C2 root cv_fold)"
    stats["policy"] = (
        "computed over TRAIN folds only; never recomputed inline by models (C3). "
        "n_train_blocks counts DISTINCT spatial_block_id, not fires (C3.3). "
        "Fire membership is read from the manifests on disk, NOT from the 28-fire "
        "GOFER assignment index — that index cannot see gofer_ext fires and would "
        "silently normalise a 21-fire corpus on 12 of them. train_fire_ids and "
        "heldout_fire_ids are both recorded so a reader can settle 'was a held-out "
        "fire's statistic in here?' with ONE read: fold INDICES are not identifying "
        "across corpus versions, fire ids are."
    )
    path = write_norm_stats(stats, Path(out) if out else norm_stats_path())
    # The fingerprint is a function of `train_folds`, which this file DEFINES, so
    # it can only be read AFTER the write. Stamping it from the pre-write state
    # would record the fingerprint of the split being replaced — the same
    # sampled-at-the-wrong-end defect C-4.2 names for code fingerprints.
    stats["split_fingerprint"] = split_fingerprint(stats_path=path).get("fingerprint")
    path = write_norm_stats(stats, path)
    print(
        json.dumps(
            {
                "path": str(path),
                "n_fires_in_corpus": len(built),
                "train_folds": folds,
                "n_train_fires": len(train_ids),
                "train_fire_ids": train_ids,
                "heldout_fire_ids": heldout_ids,
                "n_train_blocks": stats["n_train_blocks"],
                "heldout_spatial_block_ids": stats["heldout_spatial_block_ids"],
                "split_fingerprint": stats["split_fingerprint"],
                "bootstrap": stats.get("bootstrap", False),
            },
            indent=2,
        )
    )
    return 0


def _cmd_backfill_c2_v27(fire_ids: Sequence[str] | None, *, dry_run: bool) -> int:
    """Patch the two v2.7 keys into existing manifests. Idempotent; never rebuilds."""
    from wildfire_nowcast.common.paths import fires_dir  # noqa: PLC0415
    from wildfire_nowcast.data.backfill import backfill_manifest  # noqa: PLC0415

    ids = list(fire_ids) if fire_ids else sorted(
        p.name for p in Path(fires_dir()).iterdir()
        if (p / "manifest.json").exists()
    )
    if not ids:
        print("no built fires found", file=sys.stderr)
        return 1
    rc = 0
    for fid in ids:
        try:
            res = backfill_manifest(fid, dry_run=dry_run)
        except (ValueError, AssertionError, FileNotFoundError) as exc:
            print(f"{fid:32s} REFUSED: {exc}", file=sys.stderr)
            rc = 1
            continue
        mark = "would patch" if (dry_run and res.changed) else (
            "patched" if res.changed else "already conformant"
        )
        print(
            f"{fid:32s} fuel_vintage_lag_years={res.fuel_vintage_lag_years} "
            f"n_ignition_components={res.n_ignition_components}  {mark}"
        )
    return rc


def _cmd_audit(fire_ids: Sequence[str] | None, *, patch: bool, full: bool) -> int:
    """Advisory scan. Returns 0 even on `suspect` — this is not a contract.

    A finding here means *go and look*, not *the build is invalid*. Making it
    exit non-zero would put an unratified plausibility bound on the critical
    path, which ADR-010 explicitly declines to do.
    """
    from wildfire_nowcast.common.paths import data_dir  # noqa: PLC0415
    from wildfire_nowcast.data.audit import audit_built_fires  # noqa: PLC0415

    report = audit_built_fires(list(fire_ids) if fire_ids else None, patch_manifests=patch)
    out = Path(data_dir()) / "qa_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    if full:
        print(json.dumps(report, indent=2))
    else:
        for fid, rep in report["fires"].items():
            marks = rep["suspect_channels"]
            print(f"{fid:32s} {rep['verdict']:8s} {len(marks)} suspect channel(s)")
            for name in marks:
                for finding in rep["channels"][name]["findings"]:
                    print(f"    {name}: {finding}")
        ns = report.get("norm_stats")
        if ns:
            print(f"{'norm_stats.json':32s} {ns['verdict']:8s} "
                  f"n_train_blocks={ns['n_train_blocks']} folds={ns['train_folds']}")
            for finding in ns["findings"]:
                print(f"    {finding}")
    print(f"-> {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "fetch":
        print(download_gofer())
        return 0

    if args.cmd == "fires":
        arch = GoferArchive()
        tab = arch.fire_table()
        for _, row in tab.sort_values(["fyear", "fname"]).iterrows():
            print(f"{row.fire_id:34s} {row.fname:24s} {row.fyear}  "
                  f"{row.acres_official:>9,.0f} acres  ig {row.GOESIg_UTC}Z")
        return 0

    if args.cmd == "gee-probe":
        from wildfire_nowcast.data.sources.gee import probe_auth  # noqa: PLC0415

        verdict = probe_auth()
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["ok"] else 2

    if args.cmd in ("labels", "qa"):
        arch = GoferArchive(variant=args.variant)
        build = build_fire_state(
            args.fire_id, archive=arch, rule=args.rule, buffer_m=args.buffer_m
        )
        if args.cmd == "qa":
            print(json.dumps(build.qa, indent=2))
            return 0 if build.qa["verdict"]["pass"] else 1
        out = write_interim_fire_state(build, name=args.name)
        print(out)
        for w in build.qa["verdict"]["warnings"]:
            print(f"  warn: {w}", file=sys.stderr)
        return 0 if build.qa["verdict"]["pass"] else 1

    if args.cmd == "folds":
        inputs = fire_domain_inputs()
        folds = assign_folds(inputs, k=args.k)
        print(json.dumps(fold_summary(inputs, folds, k=args.k), indent=2))
        return 0

    if args.cmd == "assignments":
        print(json.dumps(fire_assignments(k=args.k, refresh=args.refresh), indent=2))
        return 0

    if args.cmd == "build":
        return _cmd_build(args.fire_id, verbose=not args.quiet)

    if args.cmd == "norm-stats":
        return _cmd_norm_stats(args.train_folds, args.out)

    if args.cmd == "backfill-c2-v27":
        return _cmd_backfill_c2_v27(args.fire_id, dry_run=args.dry_run)

    if args.cmd == "audit":
        return _cmd_audit(args.fire_id, patch=not args.no_patch, full=args.full)

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
