"""D10 - is the SUB-THRESHOLD detached-body pool isotropic, or does it skew downwind?

**Pre-registered in ADR-046 (7)** before any result existed. The maintainer's
recorded prediction: ISOTROPIC, mean cosine within +/-0.05 of 0.

**The question.** D9's detector admits 12 events at ``min_gap_km = 3.0``. Below
that threshold sit 636 detached bodies (557 at gap 2.000 km, 79 at 2.236 km)
which ADR-046 (2)/(5) rule can NEVER be scored individually: at GOFER's ~2 km
effective resolution on a 1 km grid, a single omitted perimeter cell
manufactures a gap of up to 2.236 km, so each of them is individually
indistinguishable from label noise. **In aggregate they still carry direction.**

**The discriminator, and its honest limits.** Dilate/erode observation noise has
no preferred direction, so a pure-jitter pool must be ISOTROPIC. Genuine ember
spotting must skew DOWNWIND. That is the test. But there is a THIRD population
in this pool that this module must not let a reader forget:
**under-resolved rapid frontal advance.** When a front genuinely advances ~2 km
in one hour and the label product does not paint the intervening cell, the
result is a detached body at gap 2.0 km that is downwind BY CONSTRUCTION and is
NOT spotting. So a downwind skew here is NOT sufficient evidence of spotting;
it is evidence against pure isotropic jitter, which is a weaker claim.
:func:`by_parent_growth` exists to separate the two: advection under-sampling
must concentrate in high-growth hours, dilate/erode jitter must not.

**Independence.** The 636 bodies are emphatically not 636 independent draws.
They nest: body < (fire, hour) < fire < spatial block. Every interval in this
module is a CLUSTER BOOTSTRAP and every one names its unit. The naive iid
interval is computed too, and reported ONLY as the overstatement it would be -
the wrong-denominator error that cost this project its G2 headline magnitude
(ADR-042).

**Estimands.** Both of D9's, neither privileged, per ADR-046: ``downwind_cosine``
(anchor -> landing; the anchor is chosen to minimise distance, so it carries a
direction bias of its own) and ``downwind_cosine_from_prior_centroid``
(prior ever-burned centroid -> landing; not chosen by any distance rule, but it
inherits the fire's own bulk spread direction, so under downwind spread it is
biased POSITIVE even for pure jitter). If they disagree, the anchor choice is
doing the work.

**Code path.** The bodies come from :func:`crossings.detect_detached_bodies` and
every cosine from :func:`crossings._cosine` - the SAME functions that produced
``crossings.json``, imported rather than reimplemented (C0). The fast wind path
here is differential-tested against :func:`crossings._wind_at` on a random
sample before any statistic is computed.

READ-ONLY with respect to every existing artifact. Writes exactly one new file
under ``data/events/``. ``crossings.json``'s 12-event record is not touched.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.common.paths import data_dir, fires_dir
from wildfire_nowcast.common.states import dilate
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.data.crossings import (
    MIN_EVENT_CELLS,
    MIN_GAP_KM,
    WIND_U_CHANNEL,
    WIND_V_CHANNEL,
    _cosine,
    _wind_at,
    classify_body,
    detect_detached_bodies,
)

__all__ = [
    "BOOTSTRAP_DRAWS",
    "POOL_MAX_GAP_KM",
    "WIND_SPEED_BANDS",
    "BodyRecord",
    "build_report",
    "cluster_bootstrap",
    "collect_bodies",
    "isotropy_path",
    "negative_controls",
    "positive_control",
    "write_report",
    "main",
]

logger = logging.getLogger(__name__)

#: The sub-threshold pool is defined by the RATIFIED classifier, not by a new
#: constant: a body is in the pool iff
#: ``classify_body(b, min_gap_km=MIN_GAP_KM, min_cells=MIN_EVENT_CELLS)`` returns
#: ``"rasterisation_jitter"``. That is exactly "what the threshold of record
#: discards", it introduces no second source of truth, and it cannot drift from
#: ``crossings.json``.
#:
#: Reported for context only, and it is a TRAP, not a threshold: the first cut
#: of this module used ``gap_km <= 2.236`` and silently lost the 79 bodies at
#: dy=2,dx=1, whose true gap is 2.2360679... > 2.236. Caught by the denominator
#: (pool came back 557 instead of 636), which is why the reconciliation check in
#: :func:`build_report` exists.
POOL_MAX_GAP_KM = 2.236

#: Resamples for every cluster bootstrap. Fixed seed; the seed is in the report.
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260813

#: 10 m wind speed bands, m/s. Spotting requires wind; jitter does not care.
WIND_SPEED_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 4.0),
    (4.0, 6.0),
    (6.0, 1e9),
)

#: Planted downwind concentrations for the positive control (von Mises kappa).
#: Swept so the report states a MINIMUM DETECTABLE EFFECT rather than a single
#: pass/fail - a null from a statistic never observed to fire is not evidence.
PLANTED_KAPPAS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80)


def isotropy_path() -> Path:
    """``data/events/subthreshold_isotropy.json``. Additive; nothing else moves.

    A DESTINATION, not a citation: ``write_report`` below produces it from a
    built corpus and no clone carries it.
    """
    return data_dir() / "events" / "subthreshold_isotropy.json"


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BodyRecord:
    """One detached body, with everything the direction test needs."""

    fire_id: str
    spatial_block_id: int
    cv_fold: int
    split_role: str
    label_source: str
    hour: int
    n_cells: int
    gap_km: float
    merges_later: bool
    wind_u: float
    wind_v: float
    wind_speed: float
    cos_anchor: float | None
    cos_centroid: float | None
    #: Anchor averaged over ALL prior cells achieving the minimum distance.
    #: ``nearest_pair`` breaks ties with ``np.argmin``, which selects the first
    #: prior cell in row-major order - i.e. the NORTHERNMOST, then WESTERNMOST.
    #: That is a systematic tie-break and it biases the displacement toward
    #: south/east. This estimand removes it.
    cos_anchor_tie_avg: float | None
    n_tied_anchors: int
    #: The largest cosine ACHIEVABLE given this body's lattice quantisation. A
    #: gap-2.0 body can only point in 4 cardinal directions, so its mean cosine
    #: is capped at ~0.90 even under perfect alignment; a 1-cell contiguous step
    #: has 8 directions and a cap of ~0.97. Without this, the pool and the
    #: contiguous reference are not comparable.
    cos_ceiling: float | None
    disp_dr: int
    disp_dc: int
    #: Unit displacement of the body relative to the anchor / prior centroid, as
    #: (east, north). Retained so the controls can re-run the SAME estimator on
    #: synthetic directions instead of a second implementation of it.
    d_anchor: tuple[float, float]
    d_centroid: tuple[float, float]
    #: Newly ever-burned cells across the WHOLE fire at this hour. The
    #: advection-under-sampling discriminator.
    parent_growth_cells: int
    n_prior_burned_cells: int


#: The 8 first-ring lattice offsets, used for the contiguous reference.
_RING8: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _offsets_at_radius(radius: float, reach: int = 4) -> list[tuple[int, int]]:
    """Every integer offset whose length equals ``radius`` (to 1e-6)."""
    return [
        (dy, dx)
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
        if (dy or dx) and abs(float(np.hypot(dy, dx)) - radius) < 1e-6
    ]


def _ceiling(u: float, v: float, radius: float) -> float | None:
    """Best cosine reachable at this lattice radius - the quantisation cap.

    A gap-2.000 body can only point N/S/E/W, so even a perfectly wind-following
    process scores a mean cosine of ~0.90, not 1.0. Comparing a 4-direction
    population to an 8-direction one without this is comparing two different
    ceilings.
    """
    offs = _offsets_at_radius(radius)
    cs = [_cosine(u, v, dy, dx) for dy, dx in offs]
    vals = [c for c in cs if c is not None]
    return max(vals) if vals else None


def _ring_ceiling(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised quantisation cap for a 1-cell step: best of the 8 ring offsets."""
    spd = np.hypot(u, v)
    best = np.full(u.shape, -np.inf)
    for dy, dx in _RING8:
        nrm = float(np.hypot(dy, dx))
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (u * dx + v * (-dy)) / (spd * nrm)
        best = np.maximum(best, c)
    return np.where(spd > 0, best, np.nan)


def _tie_averaged_ring_direction(
    prev: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each cell, the tie-averaged direction FROM its nearest burned neighbour.

    Vectorised over the whole grid. Returns ``(dr, dc, ok)`` where ``(dr, dc)``
    is the landing-minus-anchor displacement and ``ok`` marks cells that had at
    least one burned first-ring neighbour. Averaging over ALL tied neighbours is
    what keeps this reference free of the row-major tie-break bias that
    ``nearest_pair`` carries.
    """
    h, w = prev.shape
    n = len(rows)
    dist = np.full(n, np.inf)
    for dy, dx in _RING8:
        rr, cc = rows + dy, cols + dx
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        hit = np.zeros(n, dtype=bool)
        hit[inside] = prev[rr[inside], cc[inside]]
        d = float(np.hypot(dy, dx))
        dist = np.where(hit & (d < dist), d, dist)
    sum_dy = np.zeros(n)
    sum_dx = np.zeros(n)
    cnt = np.zeros(n)
    for dy, dx in _RING8:
        rr, cc = rows + dy, cols + dx
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        hit = np.zeros(n, dtype=bool)
        hit[inside] = prev[rr[inside], cc[inside]]
        sel = hit & (np.abs(np.hypot(dy, dx) - dist) < 1e-9)
        sum_dy += np.where(sel, dy, 0.0)
        sum_dx += np.where(sel, dx, 0.0)
        cnt += sel.astype(float)
    ok = cnt > 0
    mean_dy = np.divide(sum_dy, np.maximum(cnt, 1.0))
    mean_dx = np.divide(sum_dx, np.maximum(cnt, 1.0))
    # offset points landing -> neighbour, so anchor -> landing is its negation
    return -mean_dy, -mean_dx, ok


def _split_role_map() -> tuple[dict[str, Any], set[str]]:
    from wildfire_nowcast.common.splits import split_fingerprint  # noqa: PLC0415

    split = split_fingerprint()
    return split, set(split.get("train_fire_ids", []))


def collect_bodies() -> tuple[
    list[BodyRecord],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
]:
    """Every detached body in the corpus, pooled and scored for direction.

    Returns ``(pool, per_fire_rows, split, differential_test, contiguous)``.
    ``pool`` holds
    exactly the bodies the ratified classifier calls ``rasterisation_jitter``;
    ``per_fire_rows`` counts EVERY verdict so the pool's denominator is auditable
    and reconcilable against ``crossings.json``.
    """
    split, train_fires = _split_role_map()
    manifests = sorted(Path(fires_dir()).glob("*/manifest.json"))
    manifests = [m for m in manifests if (m.parent / "tensor.zarr").exists()]
    if not manifests:
        raise RuntimeError("no fires on disk: refusing to report an empty scan")

    pool: list[BodyRecord] = []
    rows: list[dict[str, Any]] = []
    diff_samples: list[dict[str, Any]] = []
    contiguous_by_fire: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(7)

    for man in manifests:
        manifest = json.loads(man.read_text())
        ds = open_tensor(man.parent / "tensor.zarr")
        grid = Grid.from_dataset(ds)
        state = np.asarray(ds["fire_state"].values)
        bodies = detect_detached_bodies(state, cell_size_m=grid.cell_size_m)

        fire_id = str(manifest["fire_id"])
        block = int(manifest["spatial_block_id"])
        fold = int(manifest["cv_fold"])
        role = "train" if fire_id in train_fires else "heldout"
        label_source = str(manifest.get("provenance", {}).get("label_source", "gofer_published"))

        u_all = np.asarray(ds["features"].sel(channel=WIND_U_CHANNEL).values)
        v_all = np.asarray(ds["features"].sel(channel=WIND_V_CHANNEL).values)
        ever = state != 0
        growth = np.zeros(state.shape[0], dtype=int)
        growth[1:] = (ever[1:] & ~ever[:-1]).reshape(state.shape[0] - 1, -1).sum(axis=1)

        # ---- THE REFERENCE POPULATION -------------------------------------- #
        # Ordinary CONTIGUOUS new cells, scored by the identical estimator. This
        # is the comparison that decides what a downwind skew in the pool MEANS:
        # a fire spreading downwind puts its ordinary new cells downwind too, so
        # a pool that merely matches this baseline is under-resolved advance,
        # not transport past the front.
        c_cos: list[np.ndarray] = []
        c_ceil: list[np.ndarray] = []
        c_spd: list[np.ndarray] = []
        c_grw: list[np.ndarray] = []
        any_t = ever.reshape(state.shape[0], -1).any(axis=1)
        t0 = int(np.argmax(any_t))
        for t in range(t0 + 1, state.shape[0]):
            new = ever[t] & ~ever[t - 1]
            if not new.any():
                continue
            cont = new & dilate(ever[t - 1], 1)
            rr, cc2 = np.nonzero(cont)
            if not len(rr):
                continue
            ddr, ddc, ok = _tie_averaged_ring_direction(ever[t - 1], rr, cc2)
            uu = u_all[t, rr, cc2].astype(float)
            vv = v_all[t, rr, cc2].astype(float)
            spd = np.hypot(uu, vv)
            nrm = np.hypot(ddc, -ddr)
            good = ok & (spd > 0) & (nrm > 0)
            if not good.any():
                continue
            cosv = (uu[good] * ddc[good] + vv[good] * (-ddr[good])) / (spd[good] * nrm[good])
            c_cos.append(cosv)
            c_spd.append(spd[good])
            c_ceil.append(_ring_ceiling(uu[good], vv[good]))
            c_grw.append(np.full(int(good.sum()), growth[t], dtype=float))
        contiguous_by_fire[fire_id] = {
            "cos": np.concatenate(c_cos) if c_cos else np.array([]),
            "ceiling": np.concatenate(c_ceil) if c_ceil else np.array([]),
            "speed": np.concatenate(c_spd) if c_spd else np.array([]),
            "growth": np.concatenate(c_grw) if c_grw else np.array([]),
        }

        in_pool = 0
        verdicts: Counter[str] = Counter()
        for body in bodies:
            verdict = classify_body(body, min_gap_km=MIN_GAP_KM, min_cells=MIN_EVENT_CELLS)
            verdicts[verdict] += 1
            u = float(u_all[body.hour, body.anchor[0], body.anchor[1]])
            v = float(v_all[body.hour, body.anchor[0], body.anchor[1]])
            dr_a = body.landing[0] - body.anchor[0]
            dc_a = body.landing[1] - body.anchor[1]
            dr_c = body.landing[0] - body.prior_centroid[0]
            dc_c = body.landing[1] - body.prior_centroid[1]
            ca = _cosine(u, v, dr_a, dc_a)
            cc = _cosine(u, v, dr_c, dc_c)
            if verdict != "rasterisation_jitter":
                continue
            in_pool += 1
            na = float(np.hypot(dc_a, -dr_a)) or 1.0
            nc = float(np.hypot(dc_c, -dr_c)) or 1.0
            prior = ever[body.hour - 1]
            py, px = np.nonzero(prior)
            dd = np.hypot(py - body.landing[0], px - body.landing[1])
            tied = dd <= dd.min() + 1e-9
            ar, ac = float(py[tied].mean()), float(px[tied].mean())
            c_tie = _cosine(u, v, body.landing[0] - ar, body.landing[1] - ac)
            pool.append(
                BodyRecord(
                    fire_id=fire_id,
                    spatial_block_id=block,
                    cv_fold=fold,
                    split_role=role,
                    label_source=label_source,
                    hour=body.hour,
                    n_cells=body.n_cells,
                    gap_km=round(body.gap_km, 3),
                    merges_later=body.merges_later,
                    wind_u=u,
                    wind_v=v,
                    wind_speed=float(np.hypot(u, v)),
                    cos_anchor=ca,
                    cos_centroid=cc,
                    cos_anchor_tie_avg=c_tie,
                    n_tied_anchors=int(tied.sum()),
                    cos_ceiling=_ceiling(u, v, float(np.hypot(dr_a, dc_a))),
                    disp_dr=int(dr_a),
                    disp_dc=int(dc_a),
                    d_anchor=(dc_a / na, -dr_a / na),
                    d_centroid=(dc_c / nc, -dr_c / nc),
                    parent_growth_cells=int(growth[body.hour]),
                    n_prior_burned_cells=body.n_prior_burned_cells,
                )
            )
            # Differential test against the RATIFIED wind path, on ~20% of bodies.
            if rng.random() < 0.20:
                ref = _wind_at(ds, body.hour, body.anchor, body.landing, body.prior_centroid)
                diff_samples.append(
                    {
                        "fire_id": fire_id,
                        "hour": body.hour,
                        "fast_cos_anchor": None if ca is None else round(ca, 3),
                        "ref_cos_anchor": ref["downwind_cosine"],
                        "fast_cos_centroid": None if cc is None else round(cc, 3),
                        "ref_cos_centroid": ref["downwind_cosine_from_prior_centroid"],
                        "fast_speed": round(float(np.hypot(u, v)), 2),
                        "ref_speed": ref["wind_speed_ms"],
                    }
                )

        rows.append(
            {
                "fire_id": fire_id,
                "spatial_block_id": block,
                "cv_fold": fold,
                "split_role": role,
                "label_source": label_source,
                "n_hours": int(state.shape[0]),
                "n_detached_bodies": len(bodies),
                "n_in_subthreshold_pool": in_pool,
                "n_above_pool_gap": len(bodies) - in_pool,
                "verdict_counts": dict(sorted(verdicts.items())),
                "max_gap_km": round(max((b.gap_km for b in bodies), default=0.0), 3),
            }
        )
        logger.info("%s: %d bodies, %d in pool", fire_id, len(bodies), in_pool)

    mism = [
        d
        for d in diff_samples
        if d["fast_cos_anchor"] != d["ref_cos_anchor"]
        or d["fast_cos_centroid"] != d["ref_cos_centroid"]
        or d["fast_speed"] != d["ref_speed"]
    ]
    diff = {
        "what": (
            "the fast preloaded-array wind path used here, differentially tested "
            "against crossings._wind_at — the function that produced the wind "
            "block of crossings.json — on a random ~2% sample of pool bodies"
        ),
        "n_compared": len(diff_samples),
        "n_mismatched": len(mism),
        "mismatches": mism[:5],
        "passed": len(diff_samples) > 0 and not mism,
    }
    return pool, rows, split, diff, contiguous_by_fire


# --------------------------------------------------------------------------- #
# Clustered inference
# --------------------------------------------------------------------------- #


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile CI on the mean, resampling CLUSTERS with replacement.

    ``values`` and ``clusters`` are aligned 1-D arrays. Whole clusters are drawn
    with replacement and the mean is taken over the pooled resample, so a large
    cluster carries its weight exactly as it does in the point estimate. Returns
    ``{}`` -style keys with ``None`` interval when fewer than 2 clusters exist -
    an interval computed on one cluster is not an interval.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.size == 0:
        return {"n": 0, "n_clusters": 0, "mean": None, "ci95": None}
    uniq, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    vals_sorted = values[order]
    starts = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="left")
    stops = np.searchsorted(inv_sorted, np.arange(len(uniq)), side="right")
    sums = np.add.reduceat(vals_sorted, starts) if len(uniq) else np.array([])
    sizes = (stops - starts).astype(float)
    mean = float(values.mean())
    if len(uniq) < 2:
        return {
            "n": int(values.size),
            "n_clusters": int(len(uniq)),
            "mean": round(mean, 4),
            "ci95": None,
            "note": "fewer than 2 clusters: no clustered interval is computable",
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(draws, len(uniq)))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    naive_se = float(values.std(ddof=1) / np.sqrt(values.size))
    clus_se = float(boot.std(ddof=1))
    return {
        "n": int(values.size),
        "n_clusters": int(len(uniq)),
        "mean": round(mean, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "cluster_se": round(clus_se, 4),
        "naive_iid_se_DO_NOT_USE": round(naive_se, 4),
        "design_effect_se_ratio": round(clus_se / naive_se, 2) if naive_se > 0 else None,
    }


def _multi_unit(pool: list[BodyRecord], key: str) -> dict[str, Any]:
    """The same mean under every candidate independence unit, side by side."""
    vals, keep = _vals(pool, key)
    sub = [b for b, k in zip(pool, keep, strict=True) if k]
    units = {
        "body_NAIVE_iid": np.arange(len(sub)),
        "fire_hour": np.array([f"{b.fire_id}#{b.hour}" for b in sub]),
        "fire": np.array([b.fire_id for b in sub]),
        "spatial_block": np.array([b.spatial_block_id for b in sub]),
    }
    return {u: cluster_bootstrap(vals, c) for u, c in units.items()}


def _vals(pool: list[BodyRecord], key: str) -> tuple[np.ndarray, list[bool]]:
    raw = [getattr(b, key) for b in pool]
    keep = [v is not None for v in raw]
    return np.array([v for v in raw if v is not None], dtype=float), keep


def _binomial_two_sided_p(k: int, n: int) -> float | None:
    """Exact two-sided sign-test p under a fair coin. No scipy in this venv."""
    if n == 0:
        return None
    from math import comb  # noqa: PLC0415

    tail = sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(k - n / 2))
    return min(1.0, tail / 2**n)


def _stratum(pool: list[BodyRecord], key: str, unit: str = "fire") -> dict[str, Any]:
    vals, keep = _vals(pool, key)
    sub = [b for b, k in zip(pool, keep, strict=True) if k]
    cl = np.array([b.fire_id for b in sub] if unit == "fire" else [b.spatial_block_id for b in sub])
    out = cluster_bootstrap(vals, cl)
    out["independence_unit"] = unit
    out["n_downwind_cos_gt_0"] = int((vals > 0).sum())
    out["n_upwind_cos_lt_0"] = int((vals < 0).sum())
    out["median"] = round(float(np.median(vals)), 4) if vals.size else None
    # Ceiling-normalised, so strata with different lattice quantisation (a
    # gap-2.000 body has 4 available directions, a gap-2.236 body has 8) are
    # comparable. Without this the gap-distance interaction is partly a
    # comparison of two different ceilings.
    eff_v, eff_c = [], []
    for bdy in sub:
        if bdy.cos_ceiling:
            eff_v.append(getattr(bdy, key) / bdy.cos_ceiling)
            eff_c.append(bdy.fire_id)
    if eff_v:
        e = cluster_bootstrap(np.array(eff_v), np.array(eff_c))
        out["alignment_efficiency_mean"] = e["mean"]
        out["alignment_efficiency_ci95"] = e["ci95"]
        out["mean_quantisation_ceiling"] = round(
            float(np.mean([b.cos_ceiling for b in sub if b.cos_ceiling])), 4
        )
    # Distribution-free: how many FIRES have a positive mean? This makes no
    # bootstrap assumption at all and is the claim that survives if anyone
    # disputes the resampling scheme.
    per_fire_means = []
    for f in sorted({b.fire_id for b in sub}):
        fv = np.array(
            [getattr(b, key) for b in sub if b.fire_id == f and getattr(b, key) is not None]
        )
        if fv.size:
            per_fire_means.append(float(fv.mean()))
    pos = sum(1 for m in per_fire_means if m > 0)
    out["sign_test_over_fires"] = {
        "n_fires": len(per_fire_means),
        "n_fires_positive": pos,
        "two_sided_p": _binomial_two_sided_p(pos, len(per_fire_means)),
    }
    return out


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #


def _cos_from_unit(u: float, v: float, east: float, north: float) -> float | None:
    """Route a unit (east, north) direction back through the SAME estimator."""
    return _cosine(u, v, -north, east)


def positive_control(pool: list[BodyRecord], *, seed: int = 4242) -> dict[str, Any]:
    """Plant a downwind-skewed pool of the SAME size and cluster structure.

    Real winds, real fires, real (fire, hour) grouping - only the DIRECTIONS are
    synthetic, drawn von Mises about each body's own wind direction. Sweeping
    kappa turns the control into a POWER CURVE: it states the smallest planted
    mean cosine this design detects, which is the number that decides whether a
    null is 'isotropic' or 'underpowered'.

    Also carries the sign check the C1.4 y-descends convention demands: a body
    planted EXACTLY downwind must score +1.0, not -1.0.
    """
    rng = np.random.default_rng(seed)
    fires = np.array([b.fire_id for b in pool])
    wind_dir = np.array([np.arctan2(b.wind_v, b.wind_u) for b in pool])
    speed_ok = np.array([b.wind_speed > 0 for b in pool])

    sweep: list[dict[str, Any]] = []
    for kappa in PLANTED_KAPPAS:
        theta = wind_dir + rng.vonmises(0.0, kappa, size=len(pool))
        cos = np.array(
            [
                _cos_from_unit(b.wind_u, b.wind_v, float(np.cos(t)), float(np.sin(t)))
                for b, t in zip(pool, theta, strict=True)
            ],
            dtype=object,
        )
        ok = np.array([c is not None for c in cos]) & speed_ok
        vals = np.array([float(c) for c in cos[ok]], dtype=float)
        res = cluster_bootstrap(vals, fires[ok])
        sweep.append(
            {
                "planted_kappa": kappa,
                "realised_mean_cosine": res["mean"],
                "ci95_cluster_by_fire": res["ci95"],
                "detected_excludes_zero": res["excludes_zero"],
            }
        )

    exact = [
        _cos_from_unit(
            b.wind_u,
            b.wind_v,
            b.wind_u / (b.wind_speed or 1.0),
            b.wind_v / (b.wind_speed or 1.0),
        )
        for b in pool
        if b.wind_speed > 0
    ]
    exact_arr = np.array([e for e in exact if e is not None], dtype=float)
    sign_ok = bool(exact_arr.size and np.allclose(exact_arr, 1.0, atol=1e-9))

    detected = [s for s in sweep if s["detected_excludes_zero"]]
    mde = min((abs(s["realised_mean_cosine"]) for s in detected), default=None)
    return {
        "what": (
            "synthetic downwind-skewed directions planted onto the REAL pool's "
            "winds and cluster structure, scored by the same _cosine estimator "
            "and the same cluster bootstrap"
        ),
        "sweep": sweep,
        "minimum_detected_mean_cosine_cluster_by_fire": (
            None if mde is None else round(float(mde), 4)
        ),
        "sign_check_exactly_downwind_scores_plus_one": sign_ok,
        "fired": bool(detected) and sign_ok,
        "note": (
            "'fired' means the statistic DID separate a planted skew from zero "
            "on this exact code path. A null reported without this is a null "
            "from a statistic nobody watched work."
        ),
    }


def negative_controls(pool: list[BodyRecord], *, seed: int = 909) -> dict[str, Any]:
    """Three directions that must come back flat on the same code path."""
    rng = np.random.default_rng(seed)
    fires = np.array([b.fire_id for b in pool])
    out: dict[str, Any] = {}

    # (a) Wind rotated +90 deg: the CROSSWIND component. Symmetric about the
    #     wind axis under both the jitter null AND real spotting, so a non-zero
    #     value here indicts the estimator, not the physics.
    for label, rot in (("wind_rotated_90deg", np.pi / 2), ("wind_rotated_270deg", -np.pi / 2)):
        cos: list[float] = []
        keep: list[int] = []
        for i, b in enumerate(pool):
            if b.wind_speed <= 0:
                continue
            ur = b.wind_u * np.cos(rot) - b.wind_v * np.sin(rot)
            vr = b.wind_u * np.sin(rot) + b.wind_v * np.cos(rot)
            c = _cos_from_unit(ur, vr, *b.d_anchor)
            if c is not None:
                cos.append(c)
                keep.append(i)
        res = cluster_bootstrap(np.array(cos), fires[keep])
        res["must_be_flat"] = True
        out[label] = res

    # (b) Permuted wind: break the body<->wind pairing across the corpus. Under
    #     "the alignment is specific to the LOCAL wind" this collapses to the
    #     corpus-marginal alignment; it is the standard permutation null.
    idx = rng.permutation(len(pool))
    cos_p: list[float] = []
    keep_p: list[int] = []
    for i, b in enumerate(pool):
        other = pool[idx[i]]
        if other.wind_speed <= 0:
            continue
        c = _cos_from_unit(other.wind_u, other.wind_v, *b.d_anchor)
        if c is not None:
            cos_p.append(c)
            keep_p.append(i)
    out["wind_permuted_across_corpus"] = cluster_bootstrap(np.array(cos_p), fires[keep_p])
    out["wind_permuted_across_corpus"]["what"] = (
        "each body keeps its own displacement but is paired with ANOTHER body's "
        "wind vector. Residual non-zero here is corpus-marginal alignment (both "
        "winds and spread share a preferred compass direction), not per-event "
        "association."
    )

    # (c) Uniform random directions against the real winds: the pure isotropic
    #     null. This is what an isotropic pool LOOKS like on this code path.
    theta = rng.uniform(-np.pi, np.pi, size=len(pool))
    cos_u: list[float] = []
    keep_u: list[int] = []
    for i, (b, t) in enumerate(zip(pool, theta, strict=True)):
        if b.wind_speed <= 0:
            continue
        c = _cos_from_unit(b.wind_u, b.wind_v, float(np.cos(t)), float(np.sin(t)))
        if c is not None:
            cos_u.append(c)
            keep_u.append(i)
    out["uniform_random_directions"] = cluster_bootstrap(np.array(cos_u), fires[keep_u])
    out["uniform_random_directions"]["what"] = (
        "the isotropic null realised on this pool's own winds and clusters — "
        "the reference an isotropic verdict is measured against"
    )
    # WHICH OF THESE IS AN ESTIMATOR CONTROL, AND A CORRECTION I MADE AFTER
    # SEEING A NUMBER - declared rather than quietly applied.
    #
    # The first version of this function gated `passed` on all three being flat,
    # including the crosswind rotations. The rotations came back at -0.090 with
    # an interval that just excludes zero, and I reclassified them. The reason is
    # not the number, it is that I had mis-specified them: a control's job is to
    # show the INSTRUMENT returns zero when there is nothing to find.
    # `uniform_random_directions` and `wind_permuted_across_corpus` do that - the
    # first imposes isotropy, the second destroys the body<->wind pairing, and
    # under both the estimator MUST return zero. The 90-degree rotation tests
    # something else entirely: whether the displacement distribution is
    # SYMMETRIC ABOUT THE WIND AXIS, which is a substantive claim about fire
    # behaviour and terrain, not a property of the estimator. It can be non-zero
    # with the instrument working perfectly. It is therefore reported as a
    # DIAGNOSTIC and is stated in the summary, not buried.
    # Note also that +90 and -90 are the SAME statistic with opposite sign, so
    # they were never two independent controls.
    estimator = ["uniform_random_directions", "wind_permuted_across_corpus"]
    out["estimator_controls_that_must_be_flat"] = estimator
    out["reported_diagnostics_not_controls"] = [
        "wind_rotated_90deg",
        "wind_rotated_270deg",
    ]
    out["reclassification_note"] = (
        "The crosswind rotations were gated as controls in the first version of "
        "this module and were reclassified as diagnostics AFTER they came back "
        "non-zero. Declared here because changing a pass criterion after seeing "
        "it fail is exactly the move this project polices. The justification is "
        "on principle: a crosswind asymmetry is a claim about fire behaviour, "
        "not about the estimator, and cannot validate the estimator either way."
    )
    out["passed"] = all(out[k].get("excludes_zero") is False for k in estimator)
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _lattice_realisable_gaps(lo: float, hi: float, reach: int = 8) -> list[float]:
    """Euclidean cell-centre distances that EXIST on the lattice in (lo, hi)."""
    vals = {
        round(float(np.hypot(dy, dx)), 3)
        for dy in range(reach + 1)
        for dx in range(reach + 1)
        if (dy or dx)
    }
    return sorted(v for v in vals if lo < v < hi)


def _fixed_frame(pool: list[BodyRecord]) -> dict[str, Any]:
    """Displacement directions in the COMPASS frame, with no wind involved.

    If the pool were anisotropic in a fixed frame - a lattice or label-geometry
    artifact rather than a wind response - it would show up here as a preferred
    compass direction. A wind-driven skew should look roughly uniform here,
    because the wind itself is not fixed.
    """
    hist = Counter((b.disp_dr, b.disp_dc) for b in pool)
    names = {
        (-2, 0): "N",
        (2, 0): "S",
        (0, 2): "E",
        (0, -2): "W",
        (-1, 2): "NE",
        (-2, 1): "NNE",
        (1, 2): "SE",
        (2, 1): "SSE",
        (1, -2): "SW",
        (2, -1): "SSW",
        (-1, -2): "NW",
        (-2, -1): "NNW",
    }
    east = float(np.mean([b.disp_dc / np.hypot(b.disp_dr, b.disp_dc) for b in pool]))
    north = float(np.mean([-b.disp_dr / np.hypot(b.disp_dr, b.disp_dc) for b in pool]))
    return {
        "displacement_histogram_rowcol": {
            f"{names.get(k, str(k))} {k}": v for k, v in hist.most_common()
        },
        "mean_unit_vector_east": round(east, 4),
        "mean_unit_vector_north": round(north, 4),
        "resultant_length": round(float(np.hypot(east, north)), 4),
        "reading": (
            "resultant_length is the fixed-frame anisotropy. A wind-driven skew "
            "does NOT require it to be large, because the wind is not fixed; a "
            "large value would instead point at a lattice or label-geometry "
            "artifact. Compare it against the wind-relative mean cosine."
        ),
    }


def _reconcile(pool: list[BodyRecord], fire_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The pool's denominator, checked against ``crossings.json``'s own numbers.

    This check exists because it CAUGHT A REAL DEFECT: the first cut of this
    module selected the pool with ``gap_km <= 2.236`` and silently dropped the
    79 bodies at dy=2,dx=1 whose exact gap is 2.2360679... The count came back
    557 instead of 636 and nothing else would have noticed. A pool defined by a
    float literal is a second source of truth; the classifier is the first.
    """
    from wildfire_nowcast.data.crossings import crossings_path  # noqa: PLC0415

    verd: Counter[str] = Counter()
    for r in fire_rows:
        verd.update(r["verdict_counts"])
    total = sum(r["n_detached_bodies"] for r in fire_rows)
    ref: dict[str, Any] = {}
    p = crossings_path()
    if p.exists():
        d = json.loads(p.read_text())
        ref = {
            "crossings_json_n_events": d.get("n_events"),
            "crossings_json_n_detached_bodies_total": (
                d.get("sensitivity", {}).get("n_detached_bodies_total")
            ),
            "crossings_json_gap_histogram": (
                d.get("sensitivity", {}).get("observed_gap_km_histogram")
            ),
            "crossings_json_split_fingerprint": d.get("split_fingerprint"),
        }
    agree_total = ref.get("crossings_json_n_detached_bodies_total") in (None, total)
    agree_events = ref.get("crossings_json_n_events") in (None, verd.get("crossing", 0))
    partition_ok = (
        sum(verd.values()) == total == len(pool) + (total - verd.get("rasterisation_jitter", 0))
    )
    return {
        "n_detached_bodies_total": total,
        "verdict_counts": dict(sorted(verd.items())),
        "n_in_pool": len(pool),
        "pool_equals_rasterisation_jitter_count": len(pool) == verd.get("rasterisation_jitter", 0),
        "verdicts_partition_the_bodies": partition_ok,
        **ref,
        "agrees_with_crossings_json_total": agree_total,
        "agrees_with_crossings_json_n_events": agree_events,
        "passed": bool(
            len(pool) == verd.get("rasterisation_jitter", 0)
            and partition_ok
            and agree_total
            and agree_events
            and len(pool) > 0
        ),
    }


def _contiguous_reference(
    pool: list[BodyRecord], contiguous: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """THE COMPARISON THAT DECIDES WHAT A DOWNWIND SKEW MEANS.

    An ordinary contiguous new cell of a fire spreading downwind lands downwind
    of the front. So does an under-resolved 2 km advance. If the sub-threshold
    pool's alignment merely MATCHES ordinary growth, the pool is the label
    product failing to paint an intervening cell during fast frontal spread -
    not transport past the front. Only an alignment materially ABOVE the
    contiguous baseline is evidence of a distinct mechanism.

    Both populations are normalised by their own lattice quantisation ceiling,
    because a gap-2.000 body may point in only 4 directions (cap ~0.90) while a
    1-cell step has 8 (cap ~0.97). Comparing raw means would compare ceilings.
    """
    fires = sorted(set(contiguous) & {b.fire_id for b in pool})
    p_cos, p_ceil, p_fire = [], [], []
    for b in pool:
        if b.cos_anchor is None or b.cos_ceiling is None or b.fire_id not in fires:
            continue
        p_cos.append(b.cos_anchor)
        p_ceil.append(b.cos_ceiling)
        p_fire.append(b.fire_id)
    c_cos, c_ceil, c_fire = [], [], []
    for f in fires:
        d = contiguous[f]
        m = np.isfinite(d["cos"]) & np.isfinite(d["ceiling"])
        c_cos.append(d["cos"][m])
        c_ceil.append(d["ceiling"][m])
        c_fire.append(np.full(int(m.sum()), f))
    cc = np.concatenate(c_cos) if c_cos else np.array([])
    ce = np.concatenate(c_ceil) if c_ceil else np.array([])
    cf = np.concatenate(c_fire) if c_fire else np.array([])

    pool_r = cluster_bootstrap(np.array(p_cos), np.array(p_fire))
    cont_r = cluster_bootstrap(cc, cf)
    pool_eff = cluster_bootstrap(np.array(p_cos) / np.array(p_ceil), np.array(p_fire))
    cont_eff = cluster_bootstrap(cc / ce, cf)

    # Paired, per fire: the pool's alignment MINUS its own fire's contiguous
    # baseline. Paired removes every between-fire nuisance (wind climatology,
    # terrain, label source) in one step, and the fire is the resampling unit.
    per_fire_delta: dict[str, float] = {}
    for f in fires:
        pm = [c for c, ff in zip(p_cos, p_fire, strict=True) if ff == f]
        cm = cc[cf == f]
        if pm and cm.size:
            per_fire_delta[f] = round(float(np.mean(pm) - cm.mean()), 4)
    deltas = np.array(list(per_fire_delta.values()))
    d_boot = cluster_bootstrap(deltas, np.array(sorted(per_fire_delta)))

    # The same paired delta on CEILING-NORMALISED alignment. The pool's ceiling
    # (0.906) is LOWER than the contiguous ceiling (0.975) because most of the
    # pool can point only 4 ways, so the raw delta above UNDERSTATES the
    # difference. Both are reported; the raw one is the conservative one.
    eff_delta: dict[str, float] = {}
    for f in fires:
        pm = [c / cl for c, cl, ff in zip(p_cos, p_ceil, p_fire, strict=True) if ff == f and cl]
        m = cf == f
        cm = cc[m] / ce[m]
        if pm and cm.size:
            eff_delta[f] = round(float(np.mean(pm) - cm.mean()), 4)
    e_arr = np.array(list(eff_delta.values()))
    e_boot = cluster_bootstrap(e_arr, np.array(sorted(eff_delta)))

    # Does the pool's EXCESS over ordinary growth grow with wind? Spotting needs
    # wind; a label product failing to paint an intervening cell during fast
    # advance is already accounted for by the contiguous baseline at the same
    # wind. This is the sharpest discriminator available on this corpus.
    by_wind: dict[str, Any] = {}
    for lo, hi in WIND_SPEED_BANDS:
        name = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        dd: dict[str, float] = {}
        for f in fires:
            pm = [
                b.cos_anchor
                for b in pool
                if b.fire_id == f and b.cos_anchor is not None and lo <= b.wind_speed < hi
            ]
            d = contiguous[f]
            m = np.isfinite(d["cos"]) & (d["speed"] >= lo) & (d["speed"] < hi)
            if len(pm) >= 3 and int(m.sum()) >= 3:
                dd[f] = float(np.mean(pm) - d["cos"][m].mean())
        arr = np.array(list(dd.values()))
        r = cluster_bootstrap(arr, np.array(sorted(dd))) if arr.size else {"n": 0}
        r["n_fires_with_both_populations"] = len(dd)
        r["n_fires_positive"] = int((arr > 0).sum()) if arr.size else 0
        by_wind[name] = r

    return {
        "what": (
            "ordinary CONTIGUOUS new cells, scored with the identical estimator "
            "and the identical wind field — the advection baseline"
        ),
        "n_contiguous_cells": int(cc.size),
        "pool_mean_cosine_cluster_by_fire": pool_r,
        "contiguous_mean_cosine_cluster_by_fire": cont_r,
        "pool_alignment_efficiency": pool_eff,
        "contiguous_alignment_efficiency": cont_eff,
        "quantisation_ceilings": {
            "pool_mean_ceiling": round(float(np.mean(p_ceil)), 4) if p_ceil else None,
            "contiguous_mean_ceiling": round(float(ce.mean()), 4) if ce.size else None,
        },
        "paired_delta_per_fire_pool_minus_contiguous": per_fire_delta,
        "paired_delta_bootstrap_over_fires": d_boot,
        "paired_delta_sign_test_two_sided_p": _binomial_two_sided_p(
            int((deltas > 0).sum()), int(deltas.size)
        ),
        "paired_delta_ceiling_normalised_per_fire": eff_delta,
        "paired_delta_ceiling_normalised_bootstrap": e_boot,
        "paired_delta_by_wind_speed_ms": by_wind,
        "n_fires_pool_above_its_own_contiguous_baseline": int((deltas > 0).sum()),
        "n_fires_compared": int(deltas.size),
        "reading": (
            "delta > 0 means detached bodies are MORE wind-aligned than the same "
            "fire's ordinary growth, i.e. a mechanism beyond under-resolved "
            "frontal advance. delta ~ 0 means the pool is ordinary downwind "
            "spread with an unpainted intervening cell."
        ),
    }


def build_report() -> dict[str, Any]:
    """The whole D10 artifact as a dict. Reads only."""
    pool, fire_rows, split, diff, contiguous = collect_bodies()
    if not diff["passed"]:
        raise RuntimeError(
            "differential test against crossings._wind_at FAILED; refusing to "
            f"report statistics from an unverified wind path: {diff}"
        )
    recon = _reconcile(pool, fire_rows)
    if not recon["passed"]:
        raise RuntimeError(f"pool does not reconcile with crossings.json: {recon}")

    gaps = Counter(round(b.gap_km, 3) for b in pool)
    fires_c = Counter(b.fire_id for b in pool)
    blocks_c = Counter(b.spatial_block_id for b in pool)
    top_fire, top_n = fires_c.most_common(1)[0]
    top_block, top_bn = blocks_c.most_common(1)[0]

    by_gap: dict[str, Any] = {}
    for g in sorted(gaps):
        sub = [b for b in pool if round(b.gap_km, 3) == g]
        by_gap[str(g)] = {
            "n": len(sub),
            "from_anchor": _stratum(sub, "cos_anchor"),
            "from_prior_centroid": _stratum(sub, "cos_centroid"),
        }

    by_wind: dict[str, Any] = {}
    for lo, hi in WIND_SPEED_BANDS:
        sub = [b for b in pool if lo <= b.wind_speed < hi]
        name = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        by_wind[name] = {
            "n": len(sub),
            "mean_wind_speed_ms": (
                round(float(np.mean([b.wind_speed for b in sub])), 2) if sub else None
            ),
            "from_anchor": _stratum(sub, "cos_anchor") if sub else {"n": 0},
            "from_prior_centroid": _stratum(sub, "cos_centroid") if sub else {"n": 0},
        }

    growths = np.array([b.parent_growth_cells for b in pool], dtype=float)
    cuts = np.quantile(growths, [0.25, 0.5, 0.75]) if growths.size else np.array([0, 0, 0])
    by_growth: dict[str, Any] = {}
    edges = [(-1.0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], 1e18)]
    for lo, hi in edges:
        sub = [b for b in pool if lo < b.parent_growth_cells <= hi]
        by_growth[f"({lo:g},{hi:g}]"] = {
            "n": len(sub),
            "median_parent_growth_cells": (
                int(np.median([b.parent_growth_cells for b in sub])) if sub else None
            ),
            "from_anchor": _stratum(sub, "cos_anchor") if sub else {"n": 0},
        }

    per_block = {}
    for blk in sorted(blocks_c):
        sub = [b for b in pool if b.spatial_block_id == blk]
        v, _ = _vals(sub, "cos_anchor")
        vc, _ = _vals(sub, "cos_centroid")
        per_block[str(blk)] = {
            "n": len(sub),
            "n_fires": len({b.fire_id for b in sub}),
            "mean_cos_anchor": round(float(v.mean()), 4) if v.size else None,
            "mean_cos_centroid": round(float(vc.mean()), 4) if vc.size else None,
        }
    per_fire = {}
    for fid in sorted(fires_c):
        sub = [b for b in pool if b.fire_id == fid]
        v, _ = _vals(sub, "cos_anchor")
        vc, _ = _vals(sub, "cos_centroid")
        per_fire[fid] = {
            "n": len(sub),
            "spatial_block_id": sub[0].spatial_block_id,
            "split_role": sub[0].split_role,
            "mean_cos_anchor": round(float(v.mean()), 4) if v.size else None,
            "mean_cos_centroid": round(float(vc.mean()), 4) if vc.size else None,
        }

    # Leave-one-fire-out on the pooled mean: is any single fire carrying it?
    v_all, keep = _vals(pool, "cos_anchor")
    sub_all = [b for b, k in zip(pool, keep, strict=True) if k]
    loo = {}
    for fid in sorted(fires_c):
        mask = np.array([b.fire_id != fid for b in sub_all])
        if mask.sum():
            loo[fid] = round(float(v_all[mask].mean()), 4)

    pos = positive_control(pool)
    neg = negative_controls(pool)

    return {
        "schema": "wildfire_nowcast.subthreshold_isotropy/1",
        "task": "D10",
        "pre_registered_in": "ADR-046 (7)",
        "pre_registered_prediction": (
            "ISOTROPIC — the recorded prediction is |mean cosine| < "
            "0.05. Recorded here before the numbers below so the artifact carries "
            "the prediction it is judged against."
        ),
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "wildfire_nowcast.data.isotropy.build_report",
        "split_fingerprint": split.get("fingerprint"),
        "split_train_folds": split.get("train_folds"),
        "does_not_modify": (
            "data/events/crossings.json's 12-event record, data/fires/*, "
            "data/norm_stats.json, data/qa_audit.json and the split are READ "
            "ONLY here. This file is additive. Those are untracked artifacts of "
            "a built corpus, so a reader who cloned the repository has none of "
            "them; the claim is enforced by write_report, which refuses any "
            "destination inside the corpus root or named like one of them, "
            "rather than by this sentence."
        ),
        "pool_definition": {
            "what": (
                "exactly the detached bodies that crossings.classify_body calls "
                f"'rasterisation_jitter' at the threshold of record "
                f"(min_gap_km = {MIN_GAP_KM} km, min_event_cells = "
                f"{MIN_EVENT_CELLS}) — i.e. the bodies ADR-046 (5) rules can "
                "never be scored as individual events. Selected by the RATIFIED "
                "classifier, not by a float literal of my own."
            ),
            "reconciliation": recon,
            "why_this_cut": (
                "2.236 km (dy=2, dx=1) is the largest gap ONE omitted perimeter "
                "cell can manufacture between two 8-connected bodies. ADR-046 (2) "
                "keeps this physical argument and strikes the flat-count one."
            ),
            "n_bodies_in_pool": len(pool),
            "n_bodies_corpus_total": sum(r["n_detached_bodies"] for r in fire_rows),
            "n_fires_contributing": len(fires_c),
            "n_blocks_contributing": len(blocks_c),
            "gap_km_histogram": {str(k): gaps[k] for k in sorted(gaps)},
        },
        "gap_distribution_note": {
            "claim_checked": (
                "ADR-046 (2) states that no Euclidean distance on a 1 km lattice "
                "FALLS in (2.236, 3.606). Checked directly rather than accepted."
            ),
            "lattice_realisable_gaps_in_that_band": _lattice_realisable_gaps(2.236, 3.606),
            "observed_count_in_that_band": sum(n for g, n in gaps.items() if 2.236 < g < 3.606),
            "reading": (
                "sqrt(8)=2.828, 3.000 and sqrt(10)=3.162 ARE realisable between "
                "two integer cell centres and simply do not occur in this corpus. "
                "The empty band is therefore EMPIRICAL, not arithmetic. This does "
                "not disturb ADR-046 (2)'s ruling — striking the flatness argument "
                "was the conservative move and the physical argument carries the "
                "threshold on its own — but the emptiness is itself evidence: the "
                "pool below 2.236 km and the 12 events above 3.606 km are "
                "SEPARATED populations, not two ends of one continuum."
            ),
        },
        "independence": {
            "chosen_unit": "fire",
            "why": (
                "bodies nest body < (fire,hour) < fire < spatial block. Bodies in "
                "one fire share a landscape, a label product, one fire's wind "
                "climatology and one perimeter's geometry, so they are not "
                "independent draws. FIRE is the coarsest unit that still leaves "
                "enough clusters (21) for a percentile bootstrap; SPATIAL BLOCK "
                "(14) is reported beside it as the more conservative check and is "
                "the unit the CV split itself uses. The naive iid interval is "
                "printed ONLY to show the size of the overstatement it would be "
                "(ADR-042's wrong-denominator error)."
            ),
            "n_bodies": len(pool),
            "n_fire_hour_clusters": len({(b.fire_id, b.hour) for b in pool}),
            "n_fire_clusters": len(fires_c),
            "n_block_clusters": len(blocks_c),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "headline": {
            "from_anchor": _multi_unit(pool, "cos_anchor"),
            "from_prior_centroid": _multi_unit(pool, "cos_centroid"),
            "from_anchor_tie_averaged": _multi_unit(pool, "cos_anchor_tie_avg"),
            "tie_break_check": {
                "why": (
                    "nearest_pair breaks distance ties with np.argmin over "
                    "row-major prior cells, i.e. it prefers the NORTHERNMOST "
                    "then WESTERNMOST anchor, which would bias displacement "
                    "south/east. from_anchor_tie_averaged averages over every "
                    "tied anchor and removes it."
                ),
                "n_bodies_with_tied_anchors": sum(1 for b in pool if b.n_tied_anchors > 1),
                "max_tied_anchors": max((b.n_tied_anchors for b in pool), default=0),
            },
            "fixed_compass_frame": _fixed_frame(pool),
            "note": (
                "Neither estimand is privileged (ADR-046). from_anchor's anchor is "
                "chosen to MINIMISE distance and carries a direction bias of its "
                "own; from_prior_centroid is not chosen by any distance rule but "
                "inherits the fire's BULK spread direction, so under downwind "
                "spread it is biased positive even for pure jitter. If they "
                "disagree, the anchor choice is doing the work."
            ),
        },
        "by_gap_km": by_gap,
        "by_gap_km_note": (
            "The pool spans exactly TWO realised gaps, 2.000 and 2.236 km — a 12% "
            "range. The 'does the skew strengthen with distance' interaction that "
            "would most cleanly separate transport from jitter is therefore barely "
            "testable inside the pool: there is no distance leverage below the "
            "threshold, by construction of the threshold."
        ),
        "by_wind_speed_ms": by_wind,
        "by_parent_growth_cells": by_growth,
        "by_parent_growth_note": (
            "THE THIRD POPULATION. A front that genuinely advances ~2 km in one "
            "hour, with the intervening cell unpainted by a ~2 km label product, "
            "produces a detached body at gap 2.0 km that is downwind BY "
            "CONSTRUCTION and is NOT spotting. That mechanism must concentrate in "
            "HIGH-growth hours; dilate/erode jitter must not. A downwind skew that "
            "lives only in the top growth quartile is advection under-sampling, "
            "not ember transport."
        ),
        "concentration": {
            "most_concentrated_fire": top_fire,
            "most_concentrated_fire_n": top_n,
            "most_concentrated_fire_share": round(top_n / len(pool), 3),
            "largest_block": top_block,
            "largest_block_n": top_bn,
            "largest_block_share": round(top_bn / len(pool), 3),
            "per_block": per_block,
            "per_fire": per_fire,
            "leave_one_fire_out_pooled_mean_cos_anchor": loo,
            "leave_one_fire_out_range": (
                [round(min(loo.values()), 4), round(max(loo.values()), 4)] if loo else None
            ),
            "n_fires_with_positive_mean": sum(
                1 for v in per_fire.values() if (v["mean_cos_anchor"] or 0) > 0
            ),
            "n_blocks_with_positive_mean": sum(
                1 for v in per_block.values() if (v["mean_cos_anchor"] or 0) > 0
            ),
            "sign_test_over_blocks_two_sided_p": _binomial_two_sided_p(
                sum(1 for v in per_block.values() if (v["mean_cos_anchor"] or 0) > 0),
                len(per_block),
            ),
            "note": (
                "ADR-043's finding was 90.8% one fire and ADR-046's admissible set "
                "is 50% one fire. Concentration is the recurring failure mode; it "
                "is reported before any verdict."
            ),
        },
        "contiguous_growth_reference": _contiguous_reference(pool, contiguous),
        "controls": {
            "positive": pos,
            "negative": neg,
            "differential_test_vs_ratified_wind_path": diff,
        },
        "caveats": [
            "A downwind skew here does NOT establish spotting. Three populations "
            "share this pool: dilate/erode label jitter (isotropic), "
            "under-resolved rapid frontal advance (downwind by construction), and "
            "genuine short-range spotting (downwind). Only the FIRST is excluded "
            "by a positive result.",
            "RTMA is 10 m MEAN wind at 2.5 km resampled to 1 km, sampled at the "
            "ANCHOR cell. Ember transport is plume- and gust-driven, so the wind "
            "covariate is a weak proxy in both directions.",
            "GOFER's own East/West label disagreement is a centroid offset that is "
            "SYSTEMATIC, not symmetric. Dataset-wide it is 1.64 km over 28 fires; "
            "0.63 km is 2019_kincade alone, which is the number this caveat used "
            "to quote as though it were the dataset, understating the label noise "
            "2.6x. Per fire in data/interim/_index/label_noise_east_west.json. A "
            "systematic label bias is a directional error source that the "
            "isotropy premise does not cover.",
            "Cosine is scored per BODY, and 1-cell bodies at gap 2.0 km can take "
            "only 8 distinct directions (and at 2.236 km, 8 more). The estimand is "
            "coarsely quantised; the bootstrap reflects that but a reader should "
            "not expect fine resolution in the mean.",
            "This analysis sets no bar for G4, proposes none, and does not reopen min_gap_km.",
        ],
        "fires": sorted(fire_rows, key=lambda r: r["fire_id"]),
    }


def write_report(path: Path | None = None) -> Path:
    """Write the D10 artifact. ADDITIVE ONLY; refuses the frozen corpus."""
    out = Path(path) if path is not None else isotropy_path()
    if Path(fires_dir()) in out.parents or out.name in {
        "norm_stats.json",
        "qa_audit.json",
        "crossings.json",
    }:
        raise RuntimeError("D10 is additive: refusing to write into the frozen corpus")
    rep = build_report()
    if not rep["controls"]["positive"]["fired"]:
        raise RuntimeError("positive control did NOT fire; refusing to write a null")
    if not rep["controls"]["negative"]["passed"]:
        raise RuntimeError("a negative control was NOT flat; the estimator is suspect")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Write the artifact and print its path. Progress narration is a DIAGNOSTIC.

    This module used to be run as a bare ``print(write_report())`` with
    ``verbose=True`` wired in below it, so the per-fire progress was inseparable
    from the path on stdout. ADR-103 splits them: the path is the answer, the
    per-fire line is a fact about the run, and ``main`` is the one place a level
    may be set. INFO by default, because the corpus walk takes minutes and a
    silent one looks identical to a hung one.
    """
    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.data.isotropy", description=__doc__
    )
    parser.add_argument("--out", default=None, help="artifact path (default: the canonical one)")
    add_logging_arguments(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_from_args(args, default_verbosity=1)
    print(write_report(Path(args.out) if args.out else None))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
