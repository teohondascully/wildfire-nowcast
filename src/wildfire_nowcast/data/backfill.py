"""Backfill the C2 [v2.7] keys into manifests that predate the clause (ADR-014).

Twelve manifests were written before ``fuel_vintage_lag_years`` and
``n_ignition_components`` existed. The tensors are clean - this is a manifest-key
gap, not a data defect - so the fix is a manifest patch, NOT a rebuild. Rebuilding
would re-hit GEE/LFPS for twelve fires to change two integers, and every byte it
moved would be a byte nobody asked it to move.

Rules this module holds itself to:

- **Both keys go at the manifest ROOT**, in the same position the canonical
  builder (``common.zarr_io.build_manifest``) puts them. C2 and ADR-014 disagree
  on root-vs-provenance and the checker accepts either; the tie is broken toward
  the one implementation that already exists (C0), so a patched manifest is
  key-for-key what a rebuilt one would be. The *evidence* goes in ``provenance``,
  which is what ADR-014 actually asked for.
- **Nothing else changes.** :func:`backfill_manifest` diffs its own output and
  raises if any pre-existing key moved - including ``cv_fold`` and
  ``spatial_block_id``. C8 makes a split-fingerprint mismatch a hard fail, and
  the last time the split moved mid-flight it contaminated a trained matrix.
- **Derived, never defaulted, never overwritten.** A value already on disk that
  disagrees with the derived one raises rather than being replaced: two numbers
  for one fact is a decision, not a backfill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.contract import (
    MANIFEST_IGNITION_COMPONENTS_KEY,
    MANIFEST_VINTAGE_LAG_KEY,
    open_tensor_dataset,
)
from wildfire_nowcast.common.paths import fire_manifest_path, fire_tensor_path
from wildfire_nowcast.common.zarr_io import read_manifest, write_manifest
from wildfire_nowcast.data.ignitions import count_ignition_components

__all__ = ["BackfillResult", "fuel_vintage_lag_years", "backfill_manifest"]

#: The two v2.7 keys are inserted directly after this one, matching
#: ``common.zarr_io.build_manifest``'s ordering.
_INSERT_AFTER = "spatial_block_id"

#: Keys this module is allowed to add. Anything else moving is a bug.
_ADDED_ROOT_KEYS = (MANIFEST_VINTAGE_LAG_KEY, MANIFEST_IGNITION_COMPONENTS_KEY)
_ADDED_PROVENANCE_KEYS = ("ignition_components", "fuel_vintage_lag_derivation")


@dataclass(frozen=True)
class BackfillResult:
    fire_id: str
    fuel_vintage_lag_years: int
    n_ignition_components: int
    changed: bool
    manifest_path: Path


def fuel_vintage_lag_years(manifest: dict[str, Any]) -> int:
    """The C2 [v2.7] lag, from the manifest's own provenance.

    ``provenance.fuels_staleness_years`` already carries the right value as a
    STRING (ADR-014 asked for machine-readable, hence the ``int()``). It is
    cross-checked against ``ignition_time_utc`` year minus
    ``provenance.fuels_vintage_year`` - the same cross-check the contract runs -
    and a disagreement raises here rather than shipping two numbers for one fact.
    """
    prov = manifest.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("manifest has no provenance dict")
    raw = prov.get("fuels_staleness_years")
    if raw is None:
        raise ValueError(
            "provenance.fuels_staleness_years is absent; cannot derive "
            f"{MANIFEST_VINTAGE_LAG_KEY} without inventing it"
        )
    lag = int(raw)
    vintage = prov.get("fuels_vintage_year")
    ignition = manifest.get("ignition_time_utc")
    if vintage is not None and isinstance(ignition, str) and ignition[:4].isdigit():
        expected = int(ignition[:4]) - int(vintage)
        if lag != expected:
            raise ValueError(
                f"{MANIFEST_VINTAGE_LAG_KEY}: provenance.fuels_staleness_years="
                f"{lag} contradicts ignition year {ignition[:4]} - "
                f"fuels_vintage_year {vintage} = {expected}. Refusing to write "
                "either; the manifest disagrees with itself"
            )
    if lag < 0:
        raise ValueError(
            f"{MANIFEST_VINTAGE_LAG_KEY}={lag} < 0 would mean the LANDFIRE vintage "
            "postdates ignition, i.e. label leakage (ADR-005)"
        )
    return lag


def _insert_after(man: dict[str, Any], additions: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the dict with ``additions`` placed after :data:`_INSERT_AFTER`."""
    out: dict[str, Any] = {}
    for key, value in man.items():
        if key in additions:
            continue
        out[key] = value
        if key == _INSERT_AFTER:
            out.update(additions)
    for key, value in additions.items():
        if key not in out:  # _INSERT_AFTER absent - append rather than drop
            out[key] = value
    return out


def backfill_manifest(
    fire_id: str,
    *,
    manifest_path: Path | None = None,
    tensor_path: Path | None = None,
    dry_run: bool = False,
) -> BackfillResult:
    """Add the two C2 [v2.7] keys to one fire's manifest. Idempotent."""
    mpath = Path(manifest_path) if manifest_path else fire_manifest_path(fire_id)
    tpath = Path(tensor_path) if tensor_path else fire_tensor_path(fire_id)
    man = read_manifest(mpath)
    before = {k: v for k, v in man.items() if k not in _ADDED_ROOT_KEYS}

    lag = fuel_vintage_lag_years(man)
    with open_tensor_dataset(tpath) as ds:
        report = count_ignition_components(
            ds["fire_state"].values, cell_size_m=float(ds.attrs["cell_size_m"])
        )
    n_components = report.n_ignition_components

    for key, derived in (
        (MANIFEST_VINTAGE_LAG_KEY, lag),
        (MANIFEST_IGNITION_COMPONENTS_KEY, n_components),
    ):
        existing = man.get(key)
        if existing is not None and int(existing) != derived:
            raise ValueError(
                f"{fire_id}: manifest already records {key}={existing!r} but the "
                f"derivation gives {derived}. A backfill does not overwrite a "
                "recorded value — resolve the disagreement deliberately"
            )

    patched = _insert_after(
        man, {MANIFEST_VINTAGE_LAG_KEY: lag, MANIFEST_IGNITION_COMPONENTS_KEY: n_components}
    )
    prov = dict(patched["provenance"])
    prov["ignition_components"] = report.to_provenance()
    prov["fuel_vintage_lag_derivation"] = (
        "int(provenance.fuels_staleness_years), cross-checked against "
        "ignition year - provenance.fuels_vintage_year"
    )
    patched["provenance"] = prov

    # Guard: nothing but the declared additions may move. cv_fold and
    # spatial_block_id are in here, and C8 makes a moved split a hard fail.
    after = {k: v for k, v in patched.items() if k not in _ADDED_ROOT_KEYS}
    moved = [
        k for k in set(before) | set(after) if k != "provenance" and before.get(k) != after.get(k)
    ]
    prov_moved = [
        k
        for k in set(before.get("provenance", {})) | set(after.get("provenance", {}))
        if k not in _ADDED_PROVENANCE_KEYS
        and before.get("provenance", {}).get(k) != after.get("provenance", {}).get(k)
    ]
    if moved or prov_moved:  # pragma: no cover - defensive
        raise AssertionError(
            f"{fire_id}: backfill would change keys it does not own: "
            f"root={moved} provenance={prov_moved}"
        )

    changed = patched != man
    if changed and not dry_run:
        write_manifest(patched, mpath)
    return BackfillResult(
        fire_id=fire_id,
        fuel_vintage_lag_years=lag,
        n_ignition_components=n_components,
        changed=changed,
        manifest_path=mpath,
    )
