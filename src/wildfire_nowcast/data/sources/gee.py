"""Earth Engine session, region/time-window resolution, and pixel retrieval.

Retrieval posture is set by **ADR-004**, which supersedes the mid-assignment
Drive-export directive:

* Credentials hold the narrow scope pair ``earthengine,cloud-platform`` only.
  The default ``earthengine authenticate`` consent is rejected by Google because
  it requests the *restricted* scopes ``drive`` and ``devstorage.full_control``.
  Therefore **``Export.*.toDrive`` is not authorized** and
  ``Export.*.toCloudStorage`` is forbidden on cost grounds (GCS bills for
  storage and egress against the live billing account independently of Earth
  Engine registration).
* The default is **synchronous chunked fetch** - ``ee.data.computePixels``
  straight to numpy, chunked over time. At our scale (a fire domain is O(40-300)
  cells per side) this is sufficient and strictly simpler: no task polling, no
  Drive round-trip, no intermediate GeoTIFF.
* ``getInfo()`` on a large computation is still avoided; ``computePixels``
  is the region-sized path and is bounded by an explicit byte budget per request.

If measured volume ever defeats synchronous fetch, ADR-004 says the remedy is a
scoped re-auth adding ``drive`` and a PROPOSAL - never a silent switch to GCS.
:func:`export_image` is kept for that eventuality and refuses to run until the
scope actually exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from wildfire_nowcast.common.contract import CELL_SIZE_M, CRS_STRING
from wildfire_nowcast.common.grid import Grid

__all__ = [
    "ENV_PROJECT",
    "ENV_EXPORT_TARGET",
    "ENV_EXPORT_BUCKET",
    "ENV_EXPORT_FOLDER",
    "GeeAuthError",
    "ExportTarget",
    "ExportConfig",
    "gee_project",
    "initialize_ee",
    "probe_auth",
    "region_for_grid",
    "hourly_window",
    "pixel_grid",
    "fetch_pixels",
    "fetch_bands_chunked",
    "export_image",
]

ENV_PROJECT = "WILDFIRE_GEE_PROJECT"
ENV_EXPORT_TARGET = "WILDFIRE_GEE_EXPORT_TARGET"
ENV_EXPORT_BUCKET = "WILDFIRE_GEE_EXPORT_BUCKET"
ENV_EXPORT_FOLDER = "WILDFIRE_GEE_EXPORT_FOLDER"

#: ``sync`` = ee.data.computePixels (ADR-004 default). ``drive``/``gcs`` are
#: batch export targets that our credentials/cost posture do not currently allow.
ExportTarget = Literal["sync", "drive", "gcs"]
DEFAULT_EXPORT_TARGET: ExportTarget = "sync"
DEFAULT_EXPORT_FOLDER = "wildfire_nowcast_exports"

#: computePixels caps the response at 32 MiB; stay well under it and chunk.
MAX_FETCH_BYTES = 16 * 1024 * 1024


class GeeAuthError(RuntimeError):
    """Raised when Earth Engine is unusable. Carries the human remediation steps."""


def gee_project() -> str:
    """Cloud project id for Earth Engine, from ``$WILDFIRE_GEE_PROJECT``.

    Deliberately has no default: a wrong-but-plausible project id fails deep
    inside an export with an opaque message, whereas an unset variable fails
    here, loudly, with instructions.
    """
    project = os.environ.get(ENV_PROJECT, "").strip()
    if not project:
        raise GeeAuthError(
            f"{ENV_PROJECT} is not set. Export the Earth-Engine-registered Cloud "
            "project id before running any GEE ingestion. It is never hardcoded "
            "in this repository."
        )
    return project


@dataclass(frozen=True)
class ExportConfig:
    """How pixels come back. ``sync`` (ADR-004) by default.

    ``drive`` and ``gcs`` remain expressible so a future scoped re-auth is a
    config change rather than a rewrite, but both are gated: ``drive`` needs a
    scope we deliberately do not hold, and ``gcs`` bills against the live billing
    account and is forbidden outright.
    """

    target: ExportTarget = DEFAULT_EXPORT_TARGET
    folder: str = DEFAULT_EXPORT_FOLDER
    bucket: str | None = None

    @classmethod
    def from_env(cls) -> ExportConfig:
        target = os.environ.get(ENV_EXPORT_TARGET, DEFAULT_EXPORT_TARGET).strip().lower()
        if target not in ("sync", "drive", "gcs"):
            raise ValueError(
                f"{ENV_EXPORT_TARGET} must be 'sync', 'drive' or 'gcs', got {target!r}"
            )
        folder = os.environ.get(ENV_EXPORT_FOLDER, DEFAULT_EXPORT_FOLDER).strip()
        bucket = os.environ.get(ENV_EXPORT_BUCKET, "").strip() or None
        if target == "gcs":
            raise ValueError(
                "GCS export is forbidden (ADR-004): a bucket bills for storage and "
                "egress against the live billing account independently of Earth "
                "Engine registration. Use the default 'sync' fetch."
            )
        if target == "drive" and not bucket:
            # not an error - just make the missing precondition explicit
            pass
        return cls(target=target, folder=folder, bucket=bucket)  # type: ignore[arg-type]


def initialize_ee(project: str | None = None, *, quiet: bool = True) -> Any:
    """Initialise the ``ee`` module with the env-configured project.

    Returns the imported ``ee`` module. Raises :class:`GeeAuthError` with the
    remediation steps on any failure - this function never launches an
    interactive OAuth flow (ADR-002: auth is human-gated, one attempt, then a
    BLOCKER).
    """
    try:
        import ee  # noqa: PLC0415  (deliberately lazy: keeps the package importable unauthed)
    except ImportError as exc:  # pragma: no cover
        raise GeeAuthError("earthengine-api is not importable in this environment") from exc

    proj = project or gee_project()
    try:
        ee.Initialize(project=proj)
    except Exception as exc:
        raise GeeAuthError(
            f"Earth Engine refused to initialise for project {proj!r}: {exc}. "
            "See the GEE auth entry in docs/decisions.md for the exact "
            "commands to run. Do not retry in a loop."
        ) from exc
    if not quiet:
        print(f"Earth Engine initialised on project {proj}")
    return ee


def probe_auth() -> dict[str, Any]:
    """One non-interactive readiness probe. Returns a structured verdict.

    Never raises; the caller decides whether to proceed or file a blocker.
    """
    result: dict[str, Any] = {
        "project_env_set": bool(os.environ.get(ENV_PROJECT, "").strip()),
        "credentials_present": os.path.exists(
            os.path.expanduser("~/.config/earthengine/credentials")
        ),
        "ok": False,
        "error": None,
    }
    try:
        ee = initialize_ee()
        # Cheapest possible round-trip that proves the API is enabled AND the
        # project is EE-registered.
        _ = ee.Number(1).getInfo()
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def region_for_grid(grid: Grid) -> Any:
    """``ee.Geometry`` for a fire domain, in EPSG:5070 so it matches C1 exactly.

    Passing a projected rectangle with ``geodesic=False`` avoids the silent
    reprojection you get from a lat/lon bbox, which would put the export off the
    1 km lattice by a fraction of a cell.
    """
    import ee  # noqa: PLC0415

    minx, miny, maxx, maxy = grid.bounds
    return ee.Geometry.Rectangle(coords=[minx, miny, maxx, maxy], proj=grid.crs, geodesic=False)


def hourly_window(
    start_utc: datetime | str, end_utc: datetime | str, *, pad_h: int = 1
) -> tuple[datetime, datetime]:
    """Half-open ``[start, end)`` UTC window padded by ``pad_h`` hours each side.

    The pad exists so hourly interpolation at the endpoints has a neighbour, and
    so an RTMA hour that is published late is still inside the query window.

    ``end == start`` is LEGAL and means a single hour. It is not a degenerate
    request: the time-chunked RTMA fetch produces a one-hour trailing chunk
    whenever a fire's hour count is ``1 (mod chunk_hours)`` - 4 of the 28 GOFER
    fires (Walker, August, Bobcat, McCash) are exactly that, and rejecting it
    aborted the build after every earlier chunk had already been paid for. Only
    a REVERSED window is an error.
    """

    def _p(x: datetime | str) -> datetime:
        d = datetime.fromisoformat(str(x)) if isinstance(x, str) else x
        return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)

    s, e = _p(start_utc), _p(end_utc)
    if e < s:
        raise ValueError(f"reversed window {s} .. {e}")
    return s - timedelta(hours=pad_h), e + timedelta(hours=pad_h)


def pixel_grid(grid: Grid) -> dict[str, Any]:
    """The ``computePixels`` pixel-grid spec pinned to the C1 lattice.

    The affine is given explicitly rather than via ``scale``: ``scale`` lets
    Earth Engine choose its own origin, which silently offsets every fire by up
    to half a cell and would break cross-fire cell alignment.
    """
    return {
        "dimensions": {"width": grid.nx, "height": grid.ny},
        "affineTransform": {
            "scaleX": grid.cell_size_m,
            "shearX": 0.0,
            "translateX": grid.x_min,
            "shearY": 0.0,
            "scaleY": -grid.cell_size_m,
            "translateY": grid.y_max,
        },
        "crsCode": grid.crs,
    }


def fetch_pixels(image: Any, grid: Grid, band_names: list[str]) -> dict[str, Any]:
    """Synchronously pull ``band_names`` onto the C1 grid as ``(H, W)`` float32.

    Uses ``ee.data.computePixels`` (ADR-004). Returns ``{band: ndarray}``.
    """
    import numpy as np  # noqa: PLC0415

    ee = initialize_ee()
    payload = {
        "expression": ee.Image(image).select(band_names),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": pixel_grid(grid),
    }
    arr = ee.data.computePixels(payload)
    out: dict[str, Any] = {}
    for band in band_names:
        v = np.asarray(arr[band], dtype=np.float32)
        if v.shape != grid.shape:
            raise RuntimeError(
                f"{band}: got {v.shape}, expected {grid.shape} — the pixel grid did "
                "not pin to the C1 lattice"
            )
        out[band] = v
    return out


def fetch_bands_chunked(
    image: Any,
    grid: Grid,
    band_names: list[str],
    *,
    max_bytes: int = MAX_FETCH_BYTES,
) -> dict[str, Any]:
    """:func:`fetch_pixels` split into requests that stay under ``max_bytes``.

    ``computePixels`` caps its response at 32 MiB, which for a large fire domain
    is a few hundred hourly bands. Chunking is over *bands* (i.e. over time),
    never over space, so every returned array is a whole aligned frame and there
    is no mosaic seam to reason about.
    """
    per_band = grid.ny * grid.nx * 4
    chunk = max(1, max_bytes // max(per_band, 1))
    out: dict[str, Any] = {}
    for i in range(0, len(band_names), chunk):
        out.update(fetch_pixels(image, grid, band_names[i : i + chunk]))
    return out


@dataclass
class ExportTask:
    """A submitted batch export, enough to poll and to record in provenance."""

    name: str
    task: Any
    target: ExportTarget
    destination: str
    channels: list[str] = field(default_factory=list)

    def status(self) -> dict[str, Any]:
        return self.task.status()


def export_image(
    image: Any,
    *,
    name: str,
    grid: Grid,
    config: ExportConfig | None = None,
    channels: list[str] | None = None,
    crs: str = CRS_STRING,
    scale_m: float = CELL_SIZE_M,
    max_pixels: int = 1_000_000_000,
    start: bool = True,
) -> ExportTask:
    """Submit a batch GeoTIFF export. **Gated - not the default path (ADR-004).**

    Kept so that a future scoped re-auth is a config change rather than a
    rewrite. Refuses to run today because our credentials hold only
    ``earthengine,cloud-platform``: ``toDrive`` needs the restricted ``drive``
    scope, and ``toCloudStorage`` is forbidden on cost grounds. The default
    retrieval path is :func:`fetch_bands_chunked`.
    """
    import ee  # noqa: PLC0415

    cfg = config or ExportConfig.from_env()
    if cfg.target == "sync":
        raise ValueError(
            "export_image() is the batch path; the ADR-004 default is synchronous "
            "fetch. Call fetch_bands_chunked(), or set "
            f"{ENV_EXPORT_TARGET}=drive after a scoped re-auth that adds 'drive'."
        )
    if cfg.target == "gcs":
        raise ValueError("GCS export is forbidden (ADR-004); it bills against the project.")

    # Pin the exact affine rather than passing `scale` (see pixel_grid()).
    crs_transform = [scale_m, 0, grid.x_min, 0, -scale_m, grid.y_max]
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=cfg.folder,
        fileNamePrefix=name,
        region=region_for_grid(grid),
        crs=crs,
        crsTransform=crs_transform,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
    )
    destination = f"drive://{cfg.folder}/{name}"

    if start:
        task.start()
    return ExportTask(
        name=name,
        task=task,
        target=cfg.target,
        destination=destination,
        channels=channels or [],
    )


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
