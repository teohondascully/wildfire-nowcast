"""[L1] The LABEL-NOISE FLOOR — how much of our error the labels themselves own.

ADR-054 (4). GOFER is ~2 km effective on a 1 km grid and this project has never
measured what that costs. Two arms, and the second is CALIBRATED BY THE FIRST:

**(a) EMPIRICAL, on data we already hold.** Wherever an OFFICIAL perimeter exists
for a corpus fire, measure GOFER-vs-official agreement directly. That is real
label noise, not simulated. This module supplies (a).

**(b) SYNTHETIC, calibrated to (a).** Morphological degradation of the labels
through :mod:`wildfire_nowcast.model.labelnoise` — the dilate/erode observation
noise model that already exists (C0) — at a severity chosen so that
degraded-truth-vs-truth reproduces the magnitude measured in (a).
:mod:`wildfire_nowcast.model.noiseoracle` supplies (b).

WHAT THE OFFICIAL PERIMETER IS, AND WHAT IT IS NOT
-------------------------------------------------
The reference is the **WFIGS Interagency Perimeters** service — the same public
endpoint :mod:`wildfire_nowcast.data.sources.nifc` reads, queried READ-ONLY and
cached under ``runs/`` rather than under ``data/``, which it never writes to.
It is a *final* perimeter mapped by incident staff, not an hourly product; the
comparison it supports is therefore **footprint-vs-footprint at the end of the
fire**, which is the only comparison the two products can both support.

Three limitations, stated here rather than in a footnote, because the size of
the comparable sample IS part of the finding:

1. **WFIGS carries no 2019 California perimeters at all** and no record for the
   four *complexes* in the corpus. Measured, not assumed: a name query for
   ``KINCADE``/``WALKER``/``CZU``/``SCU``/``JULY``/``KNP`` returns ZERO features
   while ``CALDOR`` returns one. Six of twenty-one corpus fires therefore have
   no official comparator.
2. **The nine ``gofer_ext`` fires are NOT an independent comparison.** This
   project's own GOFER reimplementation uses the WFIGS final perimeter as its
   stray-removal reference (``nifc.final_perimeter``), so their labels were
   built partly *from* the thing they are being compared against. They are
   reported SEPARATELY and never pooled into the headline distribution.
3. A per-fire tensor domain is the final-perimeter bbox buffered 10 km, so an
   official polygon can extend past the grid. The clipped fraction is measured
   and reported per fire rather than silently absorbed into the IoU.

THE POSITIVE CONTROL ASSERTS A MAGNITUDE
----------------------------------------
:func:`square_dilation_identity` is the analytic control for the whole
degrade-and-score idea, in the style D11 set (``2*AUC-1 == a-b`` to 1.1e-16): an
``S x S`` square dilated by ``r`` cells with an 8-connected structuring element
is EXACTLY an ``(S+2r) x (S+2r)`` square, so its IoU against the original is
``S**2 / (S+2r)**2`` — a closed form with no tolerance in it. A morphology that
is off by one cell fails it. "Non-zero" would not.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import fire_tensor_path, fires_dir
from wildfire_nowcast.common.zarr_io import open_tensor

__all__ = [
    "WFIGS_PERIMETERS_URL",
    "OfficialMatch",
    "Agreement",
    "cache_dir",
    "corpus_fires",
    "wfigs_name_for",
    "match_official",
    "official_geometry",
    "gofer_footprint",
    "agreement_for_fire",
    "square_dilation_identity",
    "whole_footprint_vs_masked_increment",
]

#: The public WFIGS service. Imported rather than re-spelled would be better
#: still, but importing `data/sources/nifc.py` would drag in its cache, which
#: writes into `data/interim/_cache`; the URL is duplicated DELIBERATELY so this
#: module owns its own cache path, and the two constants are asserted equal in
#: `eval/selftest.py` so the two cannot drift silently.
WFIGS_PERIMETERS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)

_TIMEOUT_S = 180
_ATTEMPTS = 6
_BACKOFF_S = 12.0


def cache_dir() -> Path:
    """Where fetched official geometry is cached. ``runs/``, never ``data/``."""
    return Path("runs") / "_l1_wfigs"


def _query(params: dict[str, str]) -> dict[str, Any]:
    """One WFIGS query with backoff.

    The service intermittently answers a perfectly valid request with HTTP 200
    and ``{"error": {"code": 400, "message": "Invalid URL"}}`` or ``"Object
    reference not set to an instance of an object"`` — measured, on the same URL
    that succeeds on the next attempt. Retrying only on 429 (the rule in the
    catalog helper under ``data/sources/``) therefore turns a transient server
    fault into "this fire has no official perimeter", which is the single most
    dangerous failure mode for this measurement: it would silently shrink the
    comparable sample and every remaining number would still look fine.
    """
    url = WFIGS_PERIMETERS_URL + "?" + urllib.parse.urlencode(params)
    last: Any = None
    for attempt in range(_ATTEMPTS):
        try:
            req = urllib.request.Request(  # noqa: S310
                url, headers={"User-Agent": "wildfire-nowcast/eval-labelfloor"}
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
                payload = json.load(resp)
            if "error" not in payload:
                return dict(payload)
            last = payload["error"]
        except Exception as exc:  # noqa: BLE001 - transport faults are retried too
            last = repr(exc)
        if attempt < _ATTEMPTS - 1:
            time.sleep(_BACKOFF_S)
    raise RuntimeError(f"WFIGS query failed after {_ATTEMPTS} attempts: {last}")


def corpus_fires() -> list[dict[str, Any]]:
    """Every built fire with the manifest fields this measurement needs."""
    out: list[dict[str, Any]] = []
    for manifest_path in sorted(fires_dir().glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        out.append(
            {
                "fire_id": str(m["fire_id"]),
                "cv_fold": int(m["cv_fold"]),
                "spatial_block_id": int(m["spatial_block_id"]),
                "gofer_version": str(m.get("gofer_version", "")),
                "label_source": (
                    "gofer_ext" if "ext" in str(m.get("gofer_version", "")) else "gofer_published"
                ),
                "irwin_id": str(m.get("gofer_ext_irwin_id", "") or ""),
            }
        )
    return out


def wfigs_name_for(fire_id: str) -> str:
    """The incident name a corpus ``fire_id`` encodes, e.g. ``CZU LIGHTNING COMPLEX``.

    DERIVED, never a hand-written table: a hand-written map is a place to quietly
    repoint a fire at a better-agreeing perimeter, and the matching rule has to be
    auditable from the id alone.
    """
    body = fire_id.split("_", 1)[1] if "_" in fire_id else fire_id
    return body.replace("_", " ").upper().strip()


def _year_of(fire_id: str) -> int:
    return int(fire_id.split("_", 1)[0])


@dataclass(frozen=True)
class OfficialMatch:
    """One corpus fire's official comparator, with the evidence for the match."""

    fire_id: str
    matched: bool
    incident_name: str | None
    irwin_id: str | None
    gis_acres: float | None
    discovery_year: int | None
    n_revisions: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fire_id": self.fire_id,
            "matched": self.matched,
            "incident_name": self.incident_name,
            "irwin_id": self.irwin_id,
            "gis_acres": self.gis_acres,
            "discovery_year": self.discovery_year,
            "n_revisions": self.n_revisions,
            "reason": self.reason,
        }


def match_official(fire_id: str, *, known_irwin: str = "") -> OfficialMatch:
    """Find the WFIGS incident for ``fire_id`` BY NAME AND YEAR, never by fit.

    **The match is never made on agreement.** Choosing among candidate perimeters
    by which one scores the best IoU would select on the outcome and would make
    the whole distribution an upper bound of unknown tightness. Name + discovery
    year + state + incident type, and nothing else; geometry is used afterwards
    only to VERIFY that the matched polygon touches the fire at all.
    """
    name = wfigs_name_for(fire_id)
    year = _year_of(fire_id)
    where = (
        f"UPPER(attr_IncidentName)='{name}' AND attr_POOState='US-CA' "
        f"AND attr_IncidentTypeCategory='WF' "
        f"AND attr_FireDiscoveryDateTime >= TIMESTAMP '{year}-01-01 00:00:00' "
        f"AND attr_FireDiscoveryDateTime < TIMESTAMP '{year + 1}-01-01 00:00:00'"
    )
    payload = _query(
        {
            "where": where,
            "outFields": (
                "attr_IncidentName,attr_FireDiscoveryDateTime,poly_GISAcres,"
                "attr_IrwinID,poly_IRWINID"
            ),
            "returnGeometry": "false",
            "resultRecordCount": "200",
            "f": "json",
        }
    )
    feats = payload.get("features", [])
    if not feats:
        return OfficialMatch(
            fire_id=fire_id,
            matched=False,
            incident_name=None,
            irwin_id=known_irwin or None,
            gis_acres=None,
            discovery_year=year,
            n_revisions=0,
            reason=(
                f"no WFIGS WF perimeter named {name!r} in CA with discovery year {year}. "
                "WFIGS carries no 2019 CA perimeters and none of the four corpus complexes."
            ),
        )
    best = max(feats, key=lambda f: float(f["attributes"].get("poly_GISAcres") or 0.0))
    attrs = best["attributes"]
    irwin = str(attrs.get("attr_IrwinID") or attrs.get("poly_IRWINID") or "")
    disc = attrs.get("attr_FireDiscoveryDateTime")
    disc_year = (
        datetime.fromtimestamp(float(disc) / 1000.0, tz=UTC).year if disc is not None else None
    )
    if known_irwin and irwin and known_irwin.upper() != irwin.upper():
        return OfficialMatch(
            fire_id=fire_id,
            matched=False,
            incident_name=str(attrs.get("attr_IncidentName")),
            irwin_id=irwin,
            gis_acres=float(attrs.get("poly_GISAcres") or 0.0),
            discovery_year=disc_year,
            n_revisions=len(feats),
            reason=(
                f"NAME MATCH DISAGREES WITH THE MANIFEST'S OWN irwin id: manifest "
                f"{known_irwin} vs matched {irwin}. Refusing to score rather than picking one."
            ),
        )
    return OfficialMatch(
        fire_id=fire_id,
        matched=True,
        incident_name=str(attrs.get("attr_IncidentName")),
        irwin_id=irwin,
        gis_acres=float(attrs.get("poly_GISAcres") or 0.0),
        discovery_year=disc_year,
        n_revisions=len(feats),
        reason=f"name+year+state+type match; largest of {len(feats)} revision(s)",
    )


def official_geometry(irwin_id: str, *, to_crs: str = "EPSG:5070") -> tuple[Any, Any, int]:
    """``(largest_revision, union_of_revisions, n_revisions)`` in ``to_crs``.

    BOTH are returned and both are reported. The project's existing rule
    (``nifc.final_perimeter``) takes the LARGEST revision, and that stays the
    headline so this measurement is not scored under a convention minted for it;
    the union is carried beside it as the sensitivity, because a multi-part
    incident mapped in pieces is under-represented by any single revision.
    """
    # shapely ships no `py.typed` and `types-shapely` is not a declared dev
    # dependency (pyproject is infra's file, not mine). The ignore below is
    # NARROW — exactly the two untyped imports, by error code — rather than an
    # entry on the burn-down exemption list, which would silence every future
    # error in this module instead of these two.
    import geopandas as gpd  # noqa: PLC0415
    from shapely.geometry import mapping, shape  # type: ignore[import-untyped]  # noqa: PLC0415
    from shapely.ops import unary_union  # type: ignore[import-untyped]  # noqa: PLC0415

    cache = cache_dir() / f"{irwin_id.strip('{}')}_{to_crs.replace(':', '')}.geojson"
    if cache.is_file():
        blob = json.loads(cache.read_text())
        return (
            shape(blob["largest"]),
            shape(blob["union"]),
            int(blob["n_revisions"]),
        )
    payload = _query(
        {
            "where": f"attr_IrwinID='{irwin_id}' OR poly_IRWINID='{irwin_id}'",
            "outFields": "poly_GISAcres",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
    )
    geoms = [shape(f["geometry"]) for f in payload.get("features", []) if f.get("geometry")]
    if not geoms:
        raise KeyError(f"no WFIGS polygon for irwin id {irwin_id!r}")
    series = gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(to_crs)
    largest = max(series, key=lambda g: g.area)
    union = unary_union(list(series))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "irwin_id": irwin_id,
                "crs": to_crs,
                "n_revisions": len(geoms),
                "largest": mapping(largest),
                "union": mapping(union),
                "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    )
    return largest, union, len(geoms)


def gofer_footprint(fire_id: str) -> tuple[np.ndarray, Grid, dict[str, Any]]:
    """``(ever_burned_mask, grid, meta)`` for one fire's FINAL GOFER footprint.

    ``ever(t)`` is ``state > 0`` (C1.1) and is non-decreasing, so the last frame
    IS the final footprint. Read-only.
    """
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        grid = Grid.from_dataset(ds)
        state = np.asarray(ds["fire_state"].values)
        final = state[-1] > 0
        meta = {
            "n_hours": int(state.shape[0]),
            "grid_shape": [int(v) for v in final.shape],
            "cell_size_m": float(grid.cell_size_m),
            "monotone_final_is_max": bool(int(final.sum()) == int((state.max(axis=0) > 0).sum())),
        }
    finally:
        ds.close()
    return final, grid, meta


@dataclass(frozen=True)
class Agreement:
    """GOFER-vs-official agreement for one fire, with everything needed to audit it."""

    fire_id: str
    iou: float
    area_ratio: float
    gofer_km2: float
    official_km2: float
    intersection_km2: float
    official_clipped_fraction: float
    iou_union_variant: float
    label_source: str
    cv_fold: int
    spatial_block_id: int
    match: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fire_id": self.fire_id,
            "iou": self.iou,
            "area_ratio_gofer_over_official": self.area_ratio,
            "gofer_km2": self.gofer_km2,
            "official_km2": self.official_km2,
            "intersection_km2": self.intersection_km2,
            "official_clipped_fraction": self.official_clipped_fraction,
            "iou_union_variant": self.iou_union_variant,
            "label_source": self.label_source,
            "cv_fold": self.cv_fold,
            "spatial_block_id": self.spatial_block_id,
            "match": self.match,
        }


def _mask_from_geometry(geom: Any, grid: Grid) -> np.ndarray:
    """Rasterise onto the C1 grid with the SAME rule the labels used (C0).

    ``data.rasterize.polygon_mask`` is the area-fraction rule at 10x oversampling
    with a 0.5 coverage threshold. Using ``all_touched`` here instead would
    dilate the official perimeter by a cell and manufacture agreement on one side
    and disagreement on the other; using centroid-in-polygon would bias small.
    The comparison is only fair if BOTH footprints reach the grid the same way.
    """
    from wildfire_nowcast.data.rasterize import polygon_mask  # noqa: PLC0415

    return np.asarray(polygon_mask(geom, grid)) > 0


def agreement_for_fire(
    fire_id: str,
    *,
    label_source: str,
    cv_fold: int,
    spatial_block_id: int,
    known_irwin: str = "",
) -> tuple[Agreement | None, dict[str, Any]]:
    """Measure one fire. Returns ``(agreement_or_None, match_record)``."""
    match = match_official(fire_id, known_irwin=known_irwin)
    if not match.matched or not match.irwin_id:
        return None, match.to_dict()

    largest, union, n_rev = official_geometry(match.irwin_id)
    gofer, grid, meta = gofer_footprint(fire_id)
    cell_km2 = (float(grid.cell_size_m) / 1000.0) ** 2

    official = _mask_from_geometry(largest, grid)
    official_union = _mask_from_geometry(union, grid)

    inter = int((gofer & official).sum())
    uni = int((gofer | official).sum())
    if uni == 0:
        return None, {
            **match.to_dict(),
            "matched": False,
            "reason": "official and GOFER footprints are BOTH empty on this grid",
        }
    if inter == 0:
        # A name match that lands nowhere near the fire is a WRONG match, and
        # scoring it would put a 0.0 into the distribution as if it were noise.
        return None, {
            **match.to_dict(),
            "matched": False,
            "reason": (
                "matched perimeter does NOT intersect the GOFER footprint on this grid; "
                "treating as a failed match rather than as an IoU of 0.0"
            ),
        }

    # How much of the official polygon falls outside the tensor domain at all.
    on_grid_km2 = float(official.sum()) * cell_km2
    total_km2 = float(largest.area) / 1e6
    clipped = max(0.0, 1.0 - (on_grid_km2 / total_km2)) if total_km2 > 0 else 0.0

    inter_u = int((gofer & official_union).sum())
    uni_u = int((gofer | official_union).sum())

    agreement = Agreement(
        fire_id=fire_id,
        iou=float(inter) / float(uni),
        area_ratio=(float(gofer.sum()) / float(official.sum())) if official.sum() else float("nan"),
        gofer_km2=float(gofer.sum()) * cell_km2,
        official_km2=on_grid_km2,
        intersection_km2=float(inter) * cell_km2,
        official_clipped_fraction=clipped,
        iou_union_variant=(float(inter_u) / float(uni_u)) if uni_u else float("nan"),
        label_source=label_source,
        cv_fold=cv_fold,
        spatial_block_id=spatial_block_id,
        match={**match.to_dict(), "n_revisions_fetched": n_rev, "tensor": meta},
    )
    return agreement, agreement.match


# ---------------------------------------------------------------------------
# the analytic positive control
# ---------------------------------------------------------------------------


def square_dilation_identity(side: int = 21, radius: int = 1, pad: int = 12) -> dict[str, Any]:
    """Assert a MAGNITUDE, not merely a non-zero: ``IoU == S^2 / (S+2r)^2`` EXACTLY.

    8-connected dilation by ``r`` maps an ``S x S`` square to an
    ``(S+2r) x (S+2r)`` square, so every quantity below has a closed form and the
    check has no tolerance to hide in. Run against
    :func:`wildfire_nowcast.model.labelnoise.dilate_field` — the shipped
    morphology, not a local copy — so it certifies the operator the whole
    synthetic arm is built from. If someone swaps in a 4-connected structuring
    element or an off-by-one pad, ``|measured - analytic|`` stops being 0.
    """
    import torch  # noqa: PLC0415

    from wildfire_nowcast.model.labelnoise import dilate_field, erode_field  # noqa: PLC0415

    n = side + 2 * pad
    base = np.zeros((n, n), dtype=np.float32)
    base[pad : pad + side, pad : pad + side] = 1.0
    t = torch.from_numpy(base)

    dil = dilate_field(t, radius).numpy() > 0.5
    ero = erode_field(t, radius).numpy() > 0.5
    src = base > 0.5

    dil_iou = float((src & dil).sum()) / float((src | dil).sum())
    ero_iou = float((src & ero).sum()) / float((src | ero).sum())
    analytic_dil = (side**2) / float((side + 2 * radius) ** 2)
    analytic_ero = ((side - 2 * radius) ** 2) / float(side**2)
    return {
        "side": side,
        "radius": radius,
        "dilate_cells": int(dil.sum()),
        "dilate_cells_analytic": (side + 2 * radius) ** 2,
        "erode_cells": int(ero.sum()),
        "erode_cells_analytic": (side - 2 * radius) ** 2,
        "dilate_iou": dil_iou,
        "dilate_iou_analytic": analytic_dil,
        "dilate_iou_abs_error": abs(dil_iou - analytic_dil),
        "erode_iou": ero_iou,
        "erode_iou_analytic": analytic_ero,
        "erode_iou_abs_error": abs(ero_iou - analytic_ero),
        "exact": (
            int(dil.sum()) == (side + 2 * radius) ** 2
            and int(ero.sum()) == (side - 2 * radius) ** 2
            and abs(dil_iou - analytic_dil) == 0.0
            and abs(ero_iou - analytic_ero) == 0.0
        ),
        "why": (
            "8-connected dilation of an SxS square by r is exactly (S+2r)x(S+2r). The IoU "
            "therefore has a CLOSED FORM and the assertion is on the MAGNITUDE, with zero "
            "tolerance. A control that only demands 'non-zero' would pass a 4-connected "
            "operator, an off-by-one pad, and a dilation applied twice."
        ),
    }


def _rect_case(
    *,
    width: int,
    height: int,
    bar: int,
    shift: int,
    pad: int = 12,
) -> dict[str, Any]:
    """One analytic case for :func:`whole_footprint_vs_masked_increment`.

    ``x0`` is a solid ``height x width`` rectangle. Truth adds a ``1 x bar``
    increment centred on its top edge. The perturbation is a PURE eastward
    translation by ``shift`` cells, applied COHERENTLY (the same draw corrupts
    ``x0`` and the future), which is the composition
    :class:`wildfire_nowcast.model.noiseoracle.NoisyTruthOracle` uses.

    Every quantity below is an integer count with a closed form:

    ``whole``  ``|T & shift(T)| / |T | shift(T)|`` with
        ``|T & shift(T)| = (width - shift) * height + max(bar - shift, 0)``
        because the shifted bar sits in a row the rectangle does not occupy, so
        the two components cannot cross-intersect, and
        ``|T | shift(T)| = 2 * (width * height + bar) - |T & shift(T)|``.
    ``masked``  the COHERENT predicted increment is ``shift(T) \\ shift(x0) =
        shift(T \\ x0)`` — the bar, translated — against the truth bar, so
        ``max(bar - shift, 0) / (2 * bar - max(bar - shift, 0))``.
    """
    from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY  # noqa: PLC0415
    from wildfire_nowcast.eval.metrics import evaluate  # noqa: PLC0415
    from wildfire_nowcast.model.labelnoise import LabelPerturbation  # noqa: PLC0415
    from wildfire_nowcast.model.noiseoracle import perturb_burned  # noqa: PLC0415

    rows = height + 2 * pad + 2
    cols = width + 2 * pad + 2 * shift
    r0, c0 = pad + 1, pad
    x0 = np.zeros((rows, cols), dtype=np.uint8)
    x0[r0 : r0 + height, c0 : c0 + width] = 1
    burned0 = x0 > 0

    truth = burned0.copy()
    bar_c0 = c0 + (width - bar) // 2
    truth[r0 - 1, bar_c0 : bar_c0 + bar] = True

    p = LabelPerturbation(shift_r=0, shift_c=int(shift))
    noisy_truth = perturb_burned(truth, p)
    noisy_x0 = perturb_burned(burned0, p)
    pred = burned0 | (noisy_truth & ~noisy_x0)
    samples = np.maximum(x0, pred.astype(np.uint8))[None, None]

    scored = evaluate(samples, truth.astype(np.uint8)[None], x0=x0)
    masked = scored["by_mask"]["growth_band"][GATE_CRITERION_KEY]

    # The analytic value holds only if the scoring mask contains every cell of
    # both increments. Asserted, not assumed: a band that clipped one of them
    # would silently return a DIFFERENT (and still plausible) number.
    from wildfire_nowcast.eval.masks import default_band_radius, growth_band  # noqa: PLC0415

    band_mask = growth_band(x0, default_band_radius(1))
    truth_inc = truth & ~burned0
    pred_inc = pred & ~burned0
    fully_in_band = bool((truth_inc & ~band_mask).sum() == 0 and (pred_inc & ~band_mask).sum() == 0)

    inter_whole = int((truth & noisy_truth).sum())
    union_whole = int((truth | noisy_truth).sum())
    whole = inter_whole / union_whole

    overlap = max(bar - shift, 0)
    a_inter = (width - shift) * height + overlap
    a_union = 2 * (width * height + bar) - a_inter
    analytic_whole = a_inter / a_union
    analytic_masked = overlap / (2 * bar - overlap)

    return {
        "width": width,
        "height": height,
        "bar": bar,
        "shift": shift,
        "whole_footprint_iou": whole,
        "whole_footprint_iou_analytic": analytic_whole,
        "whole_footprint_abs_error": abs(whole - analytic_whole),
        "whole_intersection_cells": inter_whole,
        "whole_intersection_cells_analytic": a_inter,
        "whole_union_cells": union_whole,
        "whole_union_cells_analytic": a_union,
        "masked_increment_iou": masked,
        "masked_increment_iou_analytic": analytic_masked,
        "masked_increment_abs_error": (
            None if masked is None else abs(float(masked) - analytic_masked)
        ),
        "increments_fully_inside_band": fully_in_band,
        "truth_increment_cells": int(truth_inc.sum()),
        "pred_increment_cells": int(pred_inc.sum()),
    }


def whole_footprint_vs_masked_increment() -> dict[str, Any]:
    """ARE THE CALIBRATION TARGET AND THE GATE CRITERION THE SAME QUANTITY? No.

    ADR-056 (6). The L1 severity was calibrated so that whole-footprint
    ``IoU(noise(L), L)`` reproduced the measured GOFER-vs-official median
    (target 0.756818, achieved 0.756663), and the oracle then scored
    ``iou_shape_masked_3h = 0.7590``. Within 0.3%. If the second is a
    re-reporting of the first through the plumbing, the whole ceiling is
    circular.

    They are different functionals, and the proof here is CONSTRUCTIVE and
    asserts MAGNITUDES (the D11 standard, ``2*AUC-1 == a-b`` to 1.1e-16). Two
    cases, both exact rationals, both scored through the SHIPPED path —
    :func:`wildfire_nowcast.model.noiseoracle.perturb_burned` for the
    perturbation and :func:`wildfire_nowcast.eval.metrics.evaluate` for the
    criterion, so nothing is recomputed by a local copy:

    ``A`` 200x50 rectangle, a 4-cell increment, translated 4 cells east.
        Whole-footprint IoU ``9800/10208 = 0.96003``; masked-increment IoU
        ``0`` EXACTLY, because a 4-cell increment translated by 4 cells cannot
        overlap itself.
    ``B`` the same rectangle, a 100-cell increment, translated 10 cells east.
        Whole-footprint IoU ``9590/10610 = 0.90386``; masked-increment IoU
        ``9/11 = 0.81818``.

    **The order REVERSES**: ``whole(A) > whole(B)`` while
    ``masked(A) < masked(B)``. A monotone reparameterisation cannot do that, so
    no function can map one onto the other, and the near-agreement at scale 2.38
    is a crossing of two different curves rather than an identity.
    """
    case_a = _rect_case(width=200, height=50, bar=4, shift=4)
    case_b = _rect_case(width=200, height=50, bar=100, shift=10)
    # THE CONTROL'S OWN CONTROL. At severity zero the two estimands are equal —
    # both exactly 1.0 — so this construction is capable of reporting AGREEMENT
    # and "they diverge" is a property of cases A and B rather than of the
    # harness. A check that returns 'different' on every input is not a check.
    case_identity = _rect_case(width=200, height=50, bar=100, shift=0)
    # `err or 1.0` was the first spelling here and it is WRONG in the one case
    # that matters: an exact match is `0.0`, which is falsy, so a perfect result
    # read as a failure. Left in the record as an insight — a check that cannot
    # pass is the same defect class as a check that cannot fail (C3.5).
    errs = [
        case_a["whole_footprint_abs_error"],
        case_b["whole_footprint_abs_error"],
        case_a["masked_increment_abs_error"],
        case_b["masked_increment_abs_error"],
    ]
    exact = (
        all(e is not None and float(e) <= 1e-15 for e in errs)
        and bool(case_a["increments_fully_inside_band"])
        and bool(case_b["increments_fully_inside_band"])
    )
    reversal = bool(
        case_a["whole_footprint_iou"] > case_b["whole_footprint_iou"]
        and float(case_a["masked_increment_iou"]) < float(case_b["masked_increment_iou"])
    )
    agrees_at_identity = bool(
        case_identity["whole_footprint_iou"] == 1.0
        and float(case_identity["masked_increment_iou"]) == 1.0
    )
    return {
        "case_a": case_a,
        "case_b": case_b,
        "case_identity": case_identity,
        "magnitudes_exact": bool(exact),
        "order_reverses": reversal,
        "agrees_at_identity": agrees_at_identity,
        "verdict": (
            "INDEPENDENT ESTIMANDS"
            if (exact and reversal and agrees_at_identity)
            else "NOT DEMONSTRATED — read the per-case errors"
        ),
        "why": (
            "Whole-footprint IoU and masked-increment IoU disagree in ORDER on two "
            "constructed cases, so neither is a function of the other and the oracle's "
            "gate score cannot be re-reporting its own calibration target. Both magnitudes "
            "are closed-form rationals and are asserted with zero tolerance; 'they differ' "
            "would have been satisfied by a rounding difference."
        ),
    }
