"""GOFER hourly perimeters — Zenodo record 14638647.

GOFER is a *published file archive*, not a GEE asset: a single ``GOFER.zip``
containing shapefiles in EPSG:4326. It therefore needs no Earth Engine
authentication, which is why labels lead ingestion (ADR-003).

Archive layout (v0.12)::

    GOFER/fireData.csv                       28 fires: name, year, acres, ignition hour
    GOFER/GOFER_{Combined,East,West}/
        GOFER{C,E,W}_fireProg.shp            hourly cumulative perimeter  (Polygon/MultiPolygon)
        GOFER{C,E,W}_cfireLine.shp           concurrent active fire line (6/step, one per fconf)
        GOFER{C,E,W}_rfireLine.shp           retrospective active fire line
        GOFER{C,E,W}_fireIg.shp              ignition point(s)
        GOFER{C,E,W}_summary.csv             per-timestep scalars (farea, fperim, rates, fstate)
        GOFER{C,E,W}_scaleVal.csv            per-timestep scaling factor

``Combined`` fuses GOES-East and GOES-West and is the default label source;
``East`` and ``West`` are kept because their disagreement is a *direct*
observation of viewing-geometry (parallax) label noise — see
docs/decisions.md.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

from wildfire_nowcast.common.paths import data_dir

__all__ = [
    "ZENODO_RECORD_ID",
    "ZENODO_FILE_URL",
    "GOFER_ZIP_SHA256",
    "GOFER_VERSION",
    "Variant",
    "GoferArchive",
    "fire_id_for",
    "download_gofer",
]

ZENODO_RECORD_ID = "14638647"
ZENODO_FILE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/GOFER.zip/content"
#: sha256 of GOFER.zip as published for record 14638647 (v0.12), verified 2026-08-07.
GOFER_ZIP_SHA256 = "4366a7ce263a09346a8d16a8f1daaa67e2910e36ffb719651d4729db960a603a"
GOFER_VERSION = "zenodo-14638647-v0.12"

#: GOFER ships EPSG:4326 (WGS84 geographic); everything is reprojected to C1's 5070.
GOFER_SRC_CRS = "EPSG:4326"

#: Confidence thresholds present in ``cfireLine.fconf`` (six rows per timestep).
CFIRE_CONF_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_CFIRE_CONF = 0.50


@dataclass(frozen=True)
class Variant:
    """One GOFER algorithm variant and its filename prefix."""

    name: str
    prefix: str

    @property
    def subdir(self) -> str:
        return f"GOFER_{self.name}"


COMBINED = Variant("Combined", "GOFERC")
EAST = Variant("East", "GOFERE")
WEST = Variant("West", "GOFERW")
VARIANTS: dict[str, Variant] = {v.name.lower(): v for v in (COMBINED, EAST, WEST)}


def fire_id_for(fname: str, fyear: int | str) -> str:
    """Stable ``fire_id`` slug, e.g. ``('Kincade', 2019) -> '2019_kincade'``.

    ``fire_id`` is the primary key everywhere downstream (C1/C2 paths, fold
    assignment, crossings index), so it must be derived, not hand-typed.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", str(fname).lower()).strip("_")
    return f"{int(fyear)}_{slug}"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_gofer(dest_dir: Path | None = None, *, verify: bool = True) -> Path:
    """Fetch and unpack ``GOFER.zip`` into ``dest_dir`` (default ``data/raw/gofer``).

    Idempotent: an already-unpacked archive with a matching checksum is reused.
    Returns the path to the unpacked ``GOFER/`` directory.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else data_dir() / "raw" / "gofer"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "GOFER.zip"

    if not zip_path.exists():
        with urlopen(ZENODO_FILE_URL) as resp, zip_path.open("wb") as out:  # noqa: S310
            while block := resp.read(1 << 20):
                out.write(block)

    if verify:
        got = _sha256(zip_path)
        if got != GOFER_ZIP_SHA256:
            raise RuntimeError(
                f"GOFER.zip checksum mismatch: expected {GOFER_ZIP_SHA256}, got {got}. "
                "The Zenodo record may have been re-versioned; do not silently proceed."
            )

    root = dest_dir / "GOFER"
    if not (root / "fireData.csv").exists():
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m
                for m in zf.namelist()
                if not m.startswith("__MACOSX/") and not m.endswith(".DS_Store")
            ]
            zf.extractall(dest_dir, members=members)
    return root


class GoferArchive:
    """Read-only accessor over an unpacked GOFER directory."""

    def __init__(self, root: Path | None = None, variant: str = "combined") -> None:
        self.root = Path(root) if root is not None else data_dir() / "raw" / "gofer" / "GOFER"
        if variant.lower() not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        self.variant = VARIANTS[variant.lower()]
        if not (self.root / "fireData.csv").exists():
            raise FileNotFoundError(
                f"{self.root} is not an unpacked GOFER archive; run download_gofer() first"
            )

    # -- paths ----------------------------------------------------------------
    def _shp(self, layer: str) -> Path:
        return self.root / self.variant.subdir / f"{self.variant.prefix}_{layer}.shp"

    def _csv(self, layer: str) -> Path:
        return self.root / self.variant.subdir / f"{self.variant.prefix}_{layer}.csv"

    # -- tables ---------------------------------------------------------------
    def fire_table(self) -> pd.DataFrame:
        """``fireData.csv`` plus a derived ``fire_id`` column."""
        df = pd.read_csv(self.root / "fireData.csv")
        df["fire_id"] = [fire_id_for(n, y) for n, y in zip(df.fname, df.fyear, strict=True)]
        return df

    def fire_ids(self) -> list[str]:
        return sorted(self.fire_table().fire_id)

    def lookup(self, fire_id: str) -> pd.Series:
        tab = self.fire_table()
        hit = tab[tab.fire_id == fire_id]
        if len(hit) != 1:
            raise KeyError(f"{fire_id!r} not in GOFER (have {len(tab)} fires)")
        return hit.iloc[0]

    def summary(self, fire_id: str) -> pd.DataFrame:
        """Per-timestep scalar summary for one fire, sorted by timestep."""
        name = self.lookup(fire_id).fname
        df = pd.read_csv(self._csv("summary"))
        df = df[df.fname == name].sort_values("timestep").reset_index(drop=True)
        df["tUTC"] = pd.to_datetime(df["tUTC"])
        return df

    # -- geometry -------------------------------------------------------------
    def _read(self, layer: str, fire_id: str) -> gpd.GeoDataFrame:
        name = self.lookup(fire_id).fname
        # SQL-escape the single quote in names like "Cold Springs" is unnecessary, but
        # be defensive: GOFER names are plain, yet this is user-facing input downstream.
        where = "fname = '{}'".format(str(name).replace("'", "''"))
        gdf = gpd.read_file(self._shp(layer), where=where)
        if gdf.empty:
            raise KeyError(f"no {layer} features for {fire_id!r}")
        gdf["tUTC"] = pd.to_datetime(gdf["tUTC"])
        return gdf.sort_values("timestep").reset_index(drop=True)

    def perimeters(self, fire_id: str, *, to_crs: str | None = None) -> gpd.GeoDataFrame:
        """Hourly cumulative perimeters (``fireProg``), one row per timestep.

        ``tUTC`` is the **end** of the hour the perimeter represents.
        """
        gdf = self._read("fireProg", fire_id)
        return gdf.to_crs(to_crs) if to_crs else gdf

    def fire_lines(
        self,
        fire_id: str,
        *,
        conf: float = DEFAULT_CFIRE_CONF,
        to_crs: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Concurrent active fire line (``cfireLine``) at one confidence level.

        Six confidence levels ship per timestep; the set of them is a ready-made
        label-perturbation ensemble for the observation-noise model.
        """
        gdf = self._read("cfireLine", fire_id)
        sel = gdf[gdf.fconf.round(2) == round(conf, 2)]
        if sel.empty:
            raise KeyError(
                f"no cfireLine rows at fconf={conf}; available {sorted(gdf.fconf.unique())}"
            )
        sel = sel.sort_values("timestep").reset_index(drop=True)
        return sel.to_crs(to_crs) if to_crs else sel

    def retro_fire_lines(self, fire_id: str, *, to_crs: str | None = None) -> gpd.GeoDataFrame:
        gdf = self._read("rfireLine", fire_id)
        return gdf.to_crs(to_crs) if to_crs else gdf

    def ignition(self, fire_id: str, *, to_crs: str | None = None) -> gpd.GeoDataFrame:
        gdf = self._read("fireIg", fire_id)
        return gdf.to_crs(to_crs) if to_crs else gdf

    def provenance(self) -> dict[str, str]:
        """C2 ``provenance`` fragment for this source."""
        return {
            "gofer_zenodo_record": ZENODO_RECORD_ID,
            "gofer_version": GOFER_VERSION,
            "gofer_variant": self.variant.name,
            "gofer_zip_sha256": GOFER_ZIP_SHA256,
            "gofer_src_crs": GOFER_SRC_CRS,
        }
