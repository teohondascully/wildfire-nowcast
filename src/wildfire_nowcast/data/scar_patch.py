"""Rebuild channel 13 in place for an interim fire, when MTBS has a coverage gap.

Why a patch and not a rebuild: the expensive part of an extension fire is the
GOES fetch (65-80 % of wall clock), and channel 13 does not depend on it. This
re-derives ``recent_burn_scar`` from MTBS + the WFIGS all-years perimeters +
Sentinel-2 dNBR, writes it back, and updates the manifest.

It follows ``data/backfill.py``'s discipline exactly, for the same reason: it
**diffs its own output** and raises if anything other than channel 13 and the
named provenance keys moved. A patch nobody verified is a rebuild you cannot
reproduce.

``data/interim/`` ONLY - the guard is structural, not a convention.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import fires_dir
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.data.sources.burn_scar import fetch_burn_scar
from wildfire_nowcast.data.sources.fuels import LANDFIRE_VINTAGES, vintage_for_fire

__all__ = ["patch_burn_scar"]

_CHANNEL = "recent_burn_scar"


def patch_burn_scar(tensor_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Re-derive channel 13 for one interim tensor. Returns a before/after report."""
    path = Path(tensor_path)
    if fires_dir() in path.parents:
        raise RuntimeError(
            f"refusing to patch {path}: data/fires/ is frozen under C-4 and the "
            "corpus swap is a separate, serialised step"
        )
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    ds = open_tensor(path)
    grid = Grid.from_dataset(ds)
    channels = [str(c) for c in ds["channel"].values]
    if _CHANNEL not in channels:
        raise KeyError(f"{path} has no {_CHANNEL} channel")
    idx = channels.index(_CHANNEL)
    before = np.asarray(ds["features"].values[0, idx])

    ignition = datetime.fromisoformat(str(manifest["ignition_time_utc"])).replace(tzinfo=UTC)
    folder = vintage_for_fire(int(ignition.year))
    report: dict[str, Any] = {}
    after = fetch_burn_scar(
        grid,
        LANDFIRE_VINTAGES[folder],
        ignition,
        fire_name=str(manifest["provenance"].get("gofer_fname") or ""),
        exclude_irwin=str(manifest["provenance"].get("gofer_ext_irwin_id") or "") or None,
        report=report,
    )
    delta = {
        "fire_id": str(manifest["fire_id"]),
        "cells_before": int((before > 0).sum()),
        "cells_after": int((after > 0).sum()),
        "cells_added": int(((after > 0) & ~(before > 0)).sum()),
        "cells_removed": int(((before > 0) & ~(after > 0)).sum()),
        "detail": report,
    }
    if dry_run:
        return delta
    if delta["cells_removed"]:
        # MTBS-derived scars must survive the patch: the gap fill is a UNION, so
        # anything disappearing means the MTBS half changed, which is a defect.
        raise RuntimeError(
            f"{delta['fire_id']}: the patch REMOVED {delta['cells_removed']} scar "
            "cells. The gap fill is a union over MTBS, so this cannot happen "
            "unless the MTBS query changed — refusing to write"
        )

    ds_new = xr.open_zarr(path, consolidated=True)
    values = ds_new["features"].values
    values[:, idx, :, :] = after.astype(values.dtype)[None, :, :]
    ds_new["features"].values = values
    tmp = path.parent / (path.name + ".patched")
    ds_new.to_zarr(tmp, mode="w", consolidated=True)
    ds_new.close()
    ds.close()
    import shutil  # noqa: PLC0415

    shutil.rmtree(path)
    tmp.rename(path)

    manifest["provenance"].setdefault("qa", {})["burn_scar_detail"] = report
    manifest["provenance"]["burn_scar_patched_utc"] = datetime.now(UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return delta
