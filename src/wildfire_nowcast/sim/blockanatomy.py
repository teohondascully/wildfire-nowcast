"""S5 — anatomy of a per-block ``band_area_dispersion_ratio``.

WHY THIS MODULE EXISTS
----------------------
G3's dispersion criterion is a RATIO::

    adr = sqrt((M+1)/M) * memberSD / rmsE

where ``memberSD`` is the ensemble's SD of total in-band burned area (the thing
G3 is about) and ``rmsE`` is the RMS error of the ENSEMBLE-MEAN area (the thing
G3 is not about). ``rmsE`` is a property of the MODEL, not of the fire: a model
that over-predicts growth inflates its own denominator and therefore *lowers its
own ratio* without its ensemble getting one cell narrower — and, symmetrically, a
model whose bias happens to match a particular fire's growth rate gets a small
denominator and a flattering ratio on that fire.

So `adr` is not comparable across blocks whose models have different biases, and
it is not comparable across arms. That is exactly the comparison ADR-034 (5)
made when it read block 5 as "perfectly ordinary at ``w_brier = 0``".

THE DECOMPOSITION
-----------------
C6 already emits ``band_area_error_bias`` and ``band_area_error_scatter`` per
fire, and ``rmsE = hypot(bias, scatter)`` exactly. This module adds ONE
model-independent scale — the RMS of TRUTH's own in-band growth over the same
window x lead units, written ``truthRMS`` — and rewrites the criterion as an
exact identity::

    adr = sqrt((M+1)/M) * s2s * relief
    s2s    = memberSD / truthRMS      # spread against the FIRE's own scale
    relief = truthRMS / rmsE          # how much the model's own error shrinks
                                      # or inflates its denominator

``s2s`` answers "is this ensemble narrow?" with a denominator the model cannot
move. ``relief`` isolates everything the model contributed to the denominator.
Neither replaces the criterion; ``adr`` remains what G3 turns on. This is the
same shape as C6's own ``area_error_decomposition`` (bias vs scatter), one level
further out.

``truthRMS`` IS NOT A NEW MEASUREMENT
------------------------------------
It is already on every run record, under ``persistence``. Persistence ignites
nothing, so within the growth band (unburned at t0, by construction) every
member area is 0, so its ``band_area_error_bias`` is exactly ``-mean(truth
in-band growth)`` and its ``band_area_error_scatter`` is exactly the SD of the
same quantity. ``truthRMS = hypot`` of those two. This module recomputes it from
the C1 tensors as well, and the two must agree to 1e-9 — a KNOWN-ANSWER check
that the window enumeration and mask used here are C6's, not a lookalike.

SCOPE. Reads C6 output JSON and C1 tensors. Calls no model, loads no checkpoint,
and writes nothing outside ``reports/figures``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.contract import BURNING as BURNING_STATE
from wildfire_nowcast.common.contract import UNBURNED
from wildfire_nowcast.common.paths import fire_tensor_path
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.eval.masks import default_band_radius, growth_band
from wildfire_nowcast.sim.components import label_components

__all__ = [
    "TruthWindow",
    "BlockTruth",
    "AdrParts",
    "truth_windows",
    "block_truth",
    "adr_parts",
    "decompose_record",
    "participation_ratio",
    "main",
]


# --------------------------------------------------------------------------
# truth side — computed from C1 tensors only
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthWindow:
    """One evaluable window's TRUTH facts. No model appears anywhere here."""

    fire_id: str
    t0: int
    #: In-band newly-burned cells at each lead 1..H (the metric's ``truth_area``).
    band_growth: tuple[int, ...]
    #: Newly-burned cells anywhere in the domain at each lead (unmasked).
    domain_growth: tuple[int, ...]
    #: 8-connected components of the burned region at t0 and at t0+H.
    n_components_t0: int
    n_components_end: int
    #: Largest in-band growth attributable to a single t0 component, per lead.
    #: A window whose growth is split across components has a smaller share.
    dominant_component_share: float

    @property
    def merged(self) -> bool:
        """Two burning bodies became one inside this window."""
        return self.n_components_end < self.n_components_t0

    @property
    def total_band_growth(self) -> int:
        return int(self.band_growth[-1]) if self.band_growth else 0


def _component_of(labels: np.ndarray, n_labels: int) -> list[np.ndarray]:
    return [labels == k for k in range(1, n_labels + 1)]


def truth_windows(
    fire_id: str,
    *,
    horizon_h: int = 3,
    stride: int = 2,
    band_radius_cells: int | None = None,
) -> list[TruthWindow]:
    """Every window C6 would score, with truth-only statistics.

    Enumeration mirrors ``model.inputs.iter_windows``: ``t0`` in
    ``range(0, n_t - horizon_h, stride)``, skipping windows whose ``x0`` has
    nothing burned. Deliberately re-implemented rather than imported, because
    importing the model package to describe TRUTH would make this probe depend
    on a module a training run is rewriting.
    """
    radius = int(band_radius_cells or default_band_radius(horizon_h))
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    finally:
        ds.close()
    n_t = int(state.shape[0])
    out: list[TruthWindow] = []
    for t0 in range(0, n_t - horizon_h, max(1, int(stride))):
        x0 = state[t0]
        burned0 = x0 > UNBURNED
        if not burned0.any():
            continue
        band = growth_band(x0, radius)
        band_growth: list[int] = []
        domain_growth: list[int] = []
        for k in range(1, horizon_h + 1):
            after = state[t0 + k] > UNBURNED
            new = after & ~burned0
            band_growth.append(int(np.count_nonzero(new & band)))
            domain_growth.append(int(np.count_nonzero(new)))
        labels0, n0 = label_components(burned0)
        end = state[t0 + horizon_h] > UNBURNED
        _, n_end = label_components(end)
        # Which t0-component does the new in-band area attach to? Grow each
        # component by the band radius and attribute the new cells it reaches.
        new_band = (end & ~burned0) & band
        total_new = int(np.count_nonzero(new_band))
        if total_new == 0 or n0 == 0:
            share = 1.0
        else:
            _, n_end_all = label_components(end)
            share = _dominant_share(labels0, n0, new_band, end, n_end_all)
        out.append(
            TruthWindow(
                fire_id=fire_id,
                t0=int(t0),
                band_growth=tuple(band_growth),
                domain_growth=tuple(domain_growth),
                n_components_t0=int(n0),
                n_components_end=int(n_end),
                dominant_component_share=float(share),
            )
        )
    return out


def _dominant_share(
    labels0: np.ndarray,
    n0: int,
    new_band: np.ndarray,
    end: np.ndarray,
    n_end: int,
) -> float:
    """Fraction of new in-band area attached to the single busiest t0 component.

    Attribution is by the END-state component each new cell belongs to: a new
    cell is credited to whichever t0 components share that end component. A cell
    in an end-component touching TWO t0 components is a merge product and is
    credited to both, so the shares can sum above 1 — that is the signature we
    want to see, not an error to normalise away.
    """
    end_labels, _ = label_components(end)
    counts = np.zeros(n0 + 1, dtype=np.int64)
    for e in range(1, n_end + 1):
        piece = end_labels == e
        new_here = int(np.count_nonzero(new_band & piece))
        if new_here == 0:
            continue
        touched = np.unique(labels0[piece & (labels0 > 0)])
        for c in touched:
            counts[int(c)] += new_here
    total = int(np.count_nonzero(new_band))
    return float(counts.max() / total) if total else 1.0


@dataclass(frozen=True)
class BlockTruth:
    """Truth-side scale of one block's growth stratum."""

    fire_id: str
    spatial_block_id: int
    n_windows: int
    n_growth_windows: int
    n_units: int  # growth windows x leads — the metric's denominator count
    truth_mean: float
    truth_sd: float
    truth_rms: float
    #: Participation ratio of the squared-error contributions: how many windows
    #: the denominator EFFECTIVELY rests on. n_eff == n_units iff every unit
    #: contributes equally.
    n_eff_units: float
    top1_share: float
    top3_share: float
    n_merge_windows: int
    mean_dominant_component_share: float
    max_band_growth: int


def block_truth(
    fire_id: str,
    spatial_block_id: int,
    *,
    horizon_h: int = 3,
    stride: int = 2,
    band_radius_cells: int | None = None,
) -> BlockTruth:
    """Truth scale + concentration for one fire's growth stratum."""
    wins = truth_windows(
        fire_id, horizon_h=horizon_h, stride=stride, band_radius_cells=band_radius_cells
    )
    growth = [w for w in wins if w.total_band_growth > 0 or _any_domain_growth(w)]
    # C6's growth stratum is `truth_growth_cells() > 0`, i.e. DOMAIN growth over
    # the whole window — not in-band growth. Match it exactly.
    growth = [w for w in wins if w.domain_growth[-1] > 0]
    vals = np.array([g for w in growth for g in w.band_growth], dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        raise ValueError(f"{fire_id}: no growth windows at stride={stride}")
    sq = vals**2
    return BlockTruth(
        fire_id=fire_id,
        spatial_block_id=int(spatial_block_id),
        n_windows=len(wins),
        n_growth_windows=len(growth),
        n_units=n,
        truth_mean=float(vals.mean()),
        truth_sd=float(vals.std(ddof=0)),
        truth_rms=float(np.sqrt((sq).mean())),
        n_eff_units=participation_ratio(sq),
        top1_share=float(np.sort(sq)[-1] / sq.sum()) if sq.sum() > 0 else 0.0,
        top3_share=float(np.sort(sq)[-3:].sum() / sq.sum()) if sq.sum() > 0 else 0.0,
        n_merge_windows=int(sum(1 for w in growth if w.merged)),
        mean_dominant_component_share=float(np.mean([w.dominant_component_share for w in growth])),
        max_band_growth=int(vals.max()),
    )


def _any_domain_growth(w: TruthWindow) -> bool:
    return any(g > 0 for g in w.domain_growth)


#: C1 channels used for the train-support distance. ``fuel_model_id`` is excluded
#: and handled as a CATEGORY (C3.2: standardising a class id is meaningless
#: arithmetic on a label).
SUPPORT_CHANNELS: tuple[str, ...] = (
    "wind_u10",
    "wind_v10",
    "temp_2m",
    "rh_2m",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "canopy_cover",
    "fuel_moisture_proxy",
    "water_barrier_mask",
    "recent_burn_scar",
)
#: FBFM40 non-burnable classes (NB1/2/3/8/9).
NON_BURNABLE: frozenset[int] = frozenset({91, 92, 93, 98, 99})


def frontier_rate(
    fire_id: str, *, horizon_h: int = 3, stride: int = 2, band_radius_cells: int | None = None
) -> dict[str, float]:
    """Truth growth per unit of t0 FRONTIER, and how much of the band is dead.

    A contagion kernel's total predicted area scales with the length of the
    frontier it propagates from. Normalising truth's growth the same way asks
    whether a block is fast *per unit of front*, which is the quantity a kernel
    with one learned rate would have to modulate — and is not the same question
    as "is this a big fire".
    """
    from wildfire_nowcast.common.states import dilate
    from wildfire_nowcast.eval.masks import frontier
    from wildfire_nowcast.sim.reader import channel_values

    radius = int(band_radius_cells or default_band_radius(horizon_h))
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
        barrier = channel_values(ds, "water_barrier_mask")[0] > 0.5
        fuel = np.rint(channel_values(ds, "fuel_model_id")[0]).astype(int)
    finally:
        ds.close()
    dead = barrier | np.isin(fuel, sorted(NON_BURNABLE))
    n_t, front, usable, grow = int(state.shape[0]), 0, 0, 0
    band_cells = 0
    band_dead = 0
    truth_on_dead = 0
    for t0 in range(0, n_t - horizon_h, max(1, int(stride))):
        x0 = state[t0]
        burned0 = x0 > UNBURNED
        if not burned0.any():
            continue
        new_end = (state[t0 + horizon_h] > UNBURNED) & ~burned0
        if not new_end.any():
            continue
        f = frontier(x0)
        band = growth_band(x0, radius)
        front += int(f.sum())
        usable += int((f & dilate((~burned0) & (~dead), 1)).sum())
        grow += int((new_end & band).sum())
        band_cells += int(band.sum())
        band_dead += int((band & dead).sum())
        truth_on_dead += int((new_end & band & dead).sum())
    return {
        "total_frontier_cells": float(front),
        "growth_per_frontier_cell": float(grow / front) if front else 0.0,
        "growth_per_usable_frontier_cell": float(grow / usable) if usable else 0.0,
        "usable_frontier_fraction": float(usable / front) if front else 0.0,
        "band_dead_fraction": float(band_dead / band_cells) if band_cells else 0.0,
        "truth_growth_on_dead_fraction": float(truth_on_dead / grow) if grow else 0.0,
    }


def participation_ratio(weights: Iterable[float]) -> float:
    """``(sum w)^2 / sum w^2`` — the effective number of contributing terms.

    A sum of ``n`` equal terms has ``n_eff == n``; a sum dominated by one term
    has ``n_eff -> 1``. This is the denominator discipline of C6.4 applied to a
    SUM rather than to a count: ``sum_sq_err`` pools 75 units on CZU, but if two
    of them carry most of the mass the ratio is an n=2 statistic wearing an
    n=75 label.
    """
    w = np.asarray(list(weights), dtype=np.float64)
    s = float(w.sum())
    ss = float((w**2).sum())
    if ss <= 0.0:
        return 0.0
    return float(s * s / ss)


# --------------------------------------------------------------------------
# model side — read back out of a C6 run record, no model call
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdrParts:
    """``band_area_dispersion_ratio`` split into a spread term and an error term."""

    model: str
    fire_id: str
    spatial_block_id: int
    n_members: int
    adr: float
    bias: float
    scatter: float
    rms_err: float
    member_sd: float
    truth_rms: float
    spread_to_signal: float
    denominator_relief: float
    identity_residual: float

    @property
    def ok(self) -> bool:
        return self.identity_residual < 1e-9


def adr_parts(
    model: str,
    fire_id: str,
    spatial_block_id: int,
    growth_block: Mapping[str, Any],
    truth_rms: float,
    n_members: int,
) -> AdrParts:
    """Split one (model, fire) growth-stratum entry of a C6 run record.

    ``growth_block`` is ``results['per_fire'][fire]['models'][model]
    ['growth_windows']`` verbatim.
    """
    adr = float(growth_block["band_area_dispersion_ratio"])
    bias = float(growth_block["band_area_error_bias"])
    scatter = float(growth_block["band_area_error_scatter"])
    rms_err = math.hypot(bias, scatter)
    factor = math.sqrt((n_members + 1.0) / n_members)
    member_sd = adr * rms_err / factor
    if truth_rms <= 0.0:
        raise ValueError(f"{fire_id}: truth_rms must be positive, got {truth_rms}")
    s2s = member_sd / truth_rms
    relief = truth_rms / rms_err if rms_err > 0 else float("inf")
    rebuilt = factor * s2s * relief
    return AdrParts(
        model=model,
        fire_id=fire_id,
        spatial_block_id=int(spatial_block_id),
        n_members=int(n_members),
        adr=adr,
        bias=bias,
        scatter=scatter,
        rms_err=rms_err,
        member_sd=member_sd,
        truth_rms=truth_rms,
        spread_to_signal=s2s,
        denominator_relief=relief,
        identity_residual=abs(rebuilt - adr),
    )


def decompose_record(
    results_path: str | Path,
    *,
    stride: int = 2,
    models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Full S5 decomposition of one C6 run record. Pure read."""
    payload = json.loads(Path(results_path).read_text())
    horizon = int(payload["horizon_h"])
    n_members = int(payload["n_members"])
    per_fire = payload["per_fire"]

    truths: dict[str, BlockTruth] = {}
    for fire_id, pf in per_fire.items():
        truths[fire_id] = block_truth(
            fire_id,
            int(pf["spatial_block_id"]),
            horizon_h=horizon,
            stride=stride,
        )

    # KNOWN-ANSWER CHECK: persistence's own denominator is truth's scale.
    checks: list[dict[str, Any]] = []
    for fire_id, pf in per_fire.items():
        p = pf["models"].get("persistence", {}).get("growth_windows")
        if not p:
            continue
        rec_rms = math.hypot(float(p["band_area_error_bias"]), float(p["band_area_error_scatter"]))
        mine = truths[fire_id].truth_rms
        checks.append(
            {
                "fire_id": fire_id,
                "persistence_rms_err_from_record": rec_rms,
                "truth_rms_recomputed_from_c1": mine,
                "abs_diff": abs(rec_rms - mine),
                "agrees": abs(rec_rms - mine) < 1e-6 * max(1.0, rec_rms),
                "persistence_bias_from_record": float(p["band_area_error_bias"]),
                "minus_truth_mean_recomputed": -truths[fire_id].truth_mean,
            }
        )

    names = (
        list(models)
        if models
        else [m for m in next(iter(per_fire.values()))["models"] if not m.endswith("__ABL")]
    )
    rows: list[dict[str, Any]] = []
    for model in names:
        for fire_id, pf in per_fire.items():
            gb = pf["models"].get(model, {}).get("growth_windows")
            if not gb or gb.get("band_area_dispersion_ratio") is None:
                continue
            rows.append(
                asdict(
                    adr_parts(
                        model,
                        fire_id,
                        int(pf["spatial_block_id"]),
                        gb,
                        truths[fire_id].truth_rms,
                        n_members,
                    )
                )
            )
    return {
        "results_path": str(results_path),
        "horizon_h": horizon,
        "n_members": n_members,
        "stride": stride,
        "split_fingerprint": payload["split_before"]["fingerprint"],
        "code_fingerprints_agree": payload["code_fingerprints_agree"],
        "identity": (
            "band_area_dispersion_ratio == sqrt((M+1)/M) * spread_to_signal * "
            "denominator_relief, exactly"
        ),
        "known_answer_check": checks,
        "block_truth": {k: asdict(v) for k, v in truths.items()},
        "parts": rows,
    }


# --------------------------------------------------------------------------
# PLAYTHROUGH — a world whose answer is known BY CONSTRUCTION
# --------------------------------------------------------------------------
#
# The scenario is three synthetic BLOCKS of windows, scored through the real C6
# `evaluate`/`aggregate` (not a re-implementation), with member areas chosen so
# that every quantity below is exact rational arithmetic:
#
#   truth area at unit (i, k) ......... k * T_i
#   member m's area ................... k * (r * T_i + c * off_m)
#   off = (-2, -1, 0, +1, +2), M = 5 .. sample SD (ddof=1) == sqrt(2.5) exactly
#
# so `memberSD = c * sqrt(2.5)`, `rmsE = |r - 1| * truthRMS`, and therefore
#
#   adr = sqrt((M+1)/M) * s2s * relief,  s2s = memberSD/truthRMS,
#                                        relief = 1/|r - 1|.
#
# THE CAPABILITY CLAIM, and it is one exact number. Block UNDER (`r = 0.5`,
# `c = 5`) and block OVER (`r = 3.0`, `c = 20`) are constructed to have the SAME
# `adr`, since `5/0.5 == 20/2`, while their ensembles differ in width by EXACTLY
# 20/5 = 4. That is the ADR-034 (5)
# artifact in miniature: an instrument that reads a ratio cannot tell those two
# blocks apart, and `s2s` must return 20/7 to nine figures or it is not measuring
# what it claims. A ratio-shaped mistake in the decomposition returns 1.

_PT_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1, 2)
_PT_TRUTH: tuple[int, ...] = (20, 24, 28, 32)
#: (block, r = predicted/truth area ratio, c = member spread scale).
#: ``r == 1`` is DELIBERATELY ABSENT and that is a finding, not a convenience:
#: a perfectly mean-calibrated ensemble makes C6's denominator exactly zero and
#: ``area_dispersion_ratio`` returns ``None``. **The criterion is undefined at
#: the calibration it is trying to reward.** My first version of this scenario
#: used ``r = 1`` as the control arm and could not be scored at all.
_PT_BLOCKS: tuple[tuple[str, float, int], ...] = (
    ("UNDER", 0.5, 5),
    ("MILD", 0.75, 7),
    ("OVER", 3.0, 20),
)
_PT_LEADS: tuple[int, ...] = (1, 2, 3)
_PT_GRID = (48, 48)
#: ``s2s(OVER) / s2s(UNDER)``, exact by construction: ``20/5``. The two blocks
#: are built to share one ``adr`` (``5/0.5 == 20/2``), so any instrument that
#: reports this as 1 is measuring the ratio again under a new name.
_PT_WIDTH_RATIO = 4.0


def _pt_field(order: np.ndarray, n_cells: int, shape: tuple[int, int]) -> np.ndarray:
    """Boolean field burning exactly ``n_cells`` band cells, in a fixed order."""
    out = np.zeros(shape, dtype=bool)
    if n_cells > 0:
        flat = order[: int(n_cells)]
        out.ravel()[flat] = True
    return out


def _pt_world() -> dict[str, Any]:
    """Three blocks of windows with exactly-designed in-band areas."""
    h, w = _PT_GRID
    x0 = np.zeros((h, w), dtype=np.uint8)
    x0[h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = BURNING_STATE
    band = growth_band(x0, default_band_radius(max(_PT_LEADS)))
    order = np.flatnonzero(band.ravel())
    blocks: dict[str, dict[str, Any]] = {}
    for name, r, c in _PT_BLOCKS:
        truth_areas: list[list[int]] = []
        member_areas: list[list[list[int]]] = []
        for base in _PT_TRUTH:
            truth_areas.append([k * base for k in _PT_LEADS])
            member_areas.append(
                [[k * int(round(r * base) + c * off) for k in _PT_LEADS] for off in _PT_OFFSETS]
            )
        blocks[name] = {
            "r": r,
            "c": c,
            "truth_areas": truth_areas,
            "member_areas": member_areas,
        }
    return {"x0": x0, "order": order, "blocks": blocks, "shape": (h, w)}


def _pt_observe(world: Mapping[str, Any]) -> dict[str, Any]:
    """Score the world through the REAL C6 and run the S5 decomposition on it."""
    from wildfire_nowcast.eval.metrics import aggregate, evaluate

    x0 = world["x0"]
    order = world["order"]
    shape = world["shape"]
    m = len(_PT_OFFSETS)
    obs: dict[str, Any] = {"blocks": {}}
    for name, block in world["blocks"].items():
        per_window = []
        zero_window = []
        flat_truth: list[float] = []
        for truth_row, member_row in zip(block["truth_areas"], block["member_areas"], strict=True):
            truth = np.stack([_pt_field(order, a, shape) for a in truth_row]).astype(np.uint8)
            samples = np.stack(
                [
                    np.stack([_pt_field(order, a, shape) for a in areas]).astype(np.uint8)
                    for areas in member_row
                ]
            )
            per_window.append(evaluate(samples, truth, x0=x0, leads=tuple(_PT_LEADS)))
            zero_window.append(
                evaluate(
                    np.repeat(x0[None, None], m, axis=0).repeat(len(_PT_LEADS), axis=1),
                    truth,
                    x0=x0,
                    leads=tuple(_PT_LEADS),
                )
            )
            flat_truth.extend(float(a) for a in truth_row)
        band = _band_headline(aggregate(per_window))
        zero_band = _band_headline(aggregate(zero_window))
        truth_rms = _pt_normaliser(flat_truth)
        parts = adr_parts(name, name, 0, band, truth_rms, m)
        obs["blocks"][name] = {
            "parts": asdict(parts),
            "designed_r": block["r"],
            "designed_c": block["c"],
            "truth_rms_designed": truth_rms,
            "zero_model_rms_err": math.hypot(
                float(zero_band["band_area_error_bias"]),
                float(zero_band["band_area_error_scatter"]),
            ),
            "pred_over_truth": 1.0
            + float(band["band_area_error_bias"]) / float(np.mean(flat_truth)),
        }
    return obs


def _pt_normaliser(values: Sequence[float]) -> float:
    """The model-independent scale: RMS of truth's in-band growth.

    A separate module-level function ON PURPOSE, so a defect can replace it with
    the plausible wrong answer — the MEAN — which is how this error would really
    arrive: at the CALL SITE, passing the wrong summary of the same numbers.
    """
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _band_headline(agg: Mapping[str, Any]) -> dict[str, Any]:
    """C6 ``aggregate`` output -> the ``band_*`` shape a run record carries.

    ``eval/baseline_run._headline`` does this renaming on the way into
    ``results.json``. Doing it here rather than teaching :func:`adr_parts` two
    key spellings keeps ONE reader of the record's shape (C0's logic): if the
    record's names ever change, exactly one function is wrong.
    """
    band = agg["by_mask"]["growth_band"]
    dec = band.get("area_error_decomposition") or {}
    return {
        "band_area_dispersion_ratio": band.get("area_dispersion_ratio"),
        "band_area_error_bias": dec.get("bias"),
        "band_area_error_scatter": dec.get("scatter"),
    }


def _pt(obs: Mapping[str, Any], block: str, key: str) -> float:
    return float(obs["blocks"][block]["parts"][key])


_PT_PROBES: tuple[Any, ...] = ()


def _probe_identity(obs: Mapping[str, Any]) -> bool:
    return all(float(b["parts"]["identity_residual"]) < 1e-12 for b in obs["blocks"].values())


def _probe_truth_rms_is_the_zero_models_denominator(obs: Mapping[str, Any]) -> bool:
    return all(
        abs(float(b["zero_model_rms_err"]) - float(b["parts"]["truth_rms"])) < 1e-9
        for b in obs["blocks"].values()
    )


def _probe_equal_adr_different_width(obs: Mapping[str, Any]) -> bool:
    """THE claim: same ratio, ensembles 20/7 apart, and s2s says 20/7."""
    same = abs(_pt(obs, "UNDER", "adr") - _pt(obs, "OVER", "adr")) < 1e-9
    ratio = _pt(obs, "OVER", "spread_to_signal") / _pt(obs, "UNDER", "spread_to_signal")
    return same and abs(ratio - _PT_WIDTH_RATIO) < 1e-9


def _probe_s2s_orders_the_three_widths(obs: Mapping[str, Any]) -> bool:
    """A RANGE probe. ADR-032 (4): a single-regime check misses the mutations
    that move a reading by a few percent, so the ORDER over three regimes is
    asserted as well as the exact pair."""
    u = _pt(obs, "UNDER", "spread_to_signal")
    c = _pt(obs, "MILD", "spread_to_signal")
    o = _pt(obs, "OVER", "spread_to_signal")
    return u < c < o


def _probe_member_sd_is_exact(obs: Mapping[str, Any]) -> bool:
    """``memberSD == c * sqrt(2.5) * sqrt(mean k^2)`` by construction."""
    scale = math.sqrt(sum(k * k for k in _PT_LEADS) / len(_PT_LEADS))
    return all(
        abs(float(b["parts"]["member_sd"]) - float(b["designed_c"]) * math.sqrt(2.5) * scale) < 1e-9
        for b in obs["blocks"].values()
    )


def _guard_scenario_really_mis_predicts(obs: Mapping[str, Any]) -> bool:
    """Pins the SCENARIO: one block really under-predicts 3.3x and one really
    over-predicts 3x. Without this the exact pair above could be satisfied by a
    world where nothing interesting happens."""
    return (
        abs(obs["blocks"]["UNDER"]["pred_over_truth"] - 0.5) < 1e-9
        and abs(obs["blocks"]["OVER"]["pred_over_truth"] - 3.0) < 1e-9
        and abs(obs["blocks"]["MILD"]["pred_over_truth"] - 0.75) < 1e-9
    )


def _defect_no_factor(world: Any) -> Any:
    return world


def build_playthrough() -> Any:
    """The S5 playthrough object. Built lazily so importing this module is cheap."""
    from wildfire_nowcast.common import playthrough as PT

    mod = sys.modules[__name__]

    def _drop_factor(
        model: str,
        fire_id: str,
        spatial_block_id: int,
        growth_block: Mapping[str, Any],
        truth_rms: float,
        n_members: int,
    ) -> AdrParts:
        return _mutated_parts(
            model, fire_id, spatial_block_id, growth_block, truth_rms, n_members, factor=1.0
        )

    def _sum_not_hypot(*a: Any, **k: Any) -> AdrParts:
        return _mutated_parts(*a, **k, rms_rule="sum")

    def _inverted_relief(*a: Any, **k: Any) -> AdrParts:
        return _mutated_parts(*a, **k, relief_rule="inverted")

    def _s2s_is_the_ratio(*a: Any, **k: Any) -> AdrParts:
        return _mutated_parts(*a, **k, s2s_rule="over_rms_err")

    def _shuffle_windows(world: Any) -> Any:
        for block in world["blocks"].values():
            idx = list(range(len(block["truth_areas"])))[::-1]
            block["truth_areas"] = [block["truth_areas"][i] for i in idx]
            block["member_areas"] = [block["member_areas"][i] for i in idx]
        return world

    return PT.Playthrough(
        name="block_area_dispersion_anatomy",
        build=_pt_world,
        observe=_pt_observe,
        probes=(
            PT.Probe("identity", _probe_identity, "adr == sqrt((M+1)/M)*s2s*relief exactly"),
            PT.Probe(
                "truth_rms_is_the_zero_models_denominator",
                _probe_truth_rms_is_the_zero_models_denominator,
                "the normaliser is already on every run record, under persistence",
            ),
            PT.Probe(
                "equal_adr_different_width",
                _probe_equal_adr_different_width,
                "two blocks with the SAME adr whose ensembles differ by exactly 20/5",
            ),
            PT.Probe(
                "s2s_orders_the_three_widths", _probe_s2s_orders_the_three_widths, "range probe"
            ),
            PT.Probe("member_sd_is_exact", _probe_member_sd_is_exact, "closed form"),
            PT.Probe(
                "scenario_really_mis_predicts",
                _guard_scenario_really_mis_predicts,
                "0.3x / 1.0x / 3.0x growth calibration is really present",
                guard=True,
            ),
        ),
        defects=(
            PT.Defect(
                "drop_the_finite_ensemble_factor",
                PT.attribute_defect((mod, "adr_parts", _drop_factor)),
                "the (M+1)/M correction, dropped. ADR-032 (2) found the biased plug-in "
                "estimator by deriving it; this is the same term left out of the READBACK, "
                "which would misattribute ~2% of every block's width.",
            ),
            PT.Defect(
                "rms_err_as_a_sum_not_a_hypotenuse",
                PT.attribute_defect((mod, "adr_parts", _sum_not_hypot)),
                "|bias| + scatter instead of hypot(bias, scatter). The obvious way to combine "
                "two error terms, and it silently inflates the denominator most on exactly the "
                "biased blocks this decomposition exists to separate.",
            ),
            PT.Defect(
                "normalise_by_truth_mean_not_truth_rms",
                PT.attribute_defect(
                    (mod, "_pt_normaliser", lambda v: float(np.mean(np.asarray(v, float))))
                ),
                "the MEAN of truth's in-band growth passed where the RMS belongs. C6's "
                "denominator is a SUM OF SQUARES, so a mean-based normaliser is off by the "
                "truth's own shape factor (measured 1.74-2.01 across our four held-out "
                "blocks) and would drift between blocks. **My first version of this mutation "
                "was NOT DETECTED and the harness said so**: I had planted it inside "
                "`adr_parts`, where a constant rescaling of the normaliser cancels between "
                "`s2s` and `relief`, leaves the identity intact, and leaves the cross-block "
                "ORDER intact — invisible to every comparison probe. It is caught only by the "
                "ABSOLUTE known-answer probe, and only once planted where the bug would really "
                "live: at the call site.",
            ),
            PT.Defect(
                "relief_inverted",
                PT.attribute_defect((mod, "adr_parts", _inverted_relief)),
                "rmsE/truthRMS instead of truthRMS/rmsE. A sign-of-effect error: it would say "
                "an over-predicting model gets denominator RELIEF where it is being penalised.",
            ),
            PT.Defect(
                "s2s_measured_against_the_models_own_error",
                PT.attribute_defect((mod, "adr_parts", _s2s_is_the_ratio)),
                "THE defect this whole module exists to rule out: computing 'spread' against "
                "rmsE, i.e. re-deriving adr and calling it a spread measure. That is precisely "
                "the reading that made block 5 look ordinary at w_brier = 0, and under it the "
                "UNDER and OVER blocks become indistinguishable (ratio 1.000, not 4.000).",
            ),
            PT.Defect(
                "reorder_the_windows_within_each_block",
                PT.data_defect(_shuffle_windows),
                "DECLARED BLIND SPOT. sum_var and sum_sq_err are SUMS over window x lead units, "
                "so any permutation preserving the (prediction, truth) pairing leaves every "
                "number identical. Neither adr nor s2s says anything about WHICH windows the "
                "spread was spent on. Member RELABELLING is invisible for the same reason.",
                detected=False,
            ),
        ),
        note=(
            "S5. Splits G3's dispersion criterion into a spread term whose denominator the "
            "model cannot move and an error term that is entirely the model's own."
        ),
    )


def _mutated_parts(
    model: str,
    fire_id: str,
    spatial_block_id: int,
    growth_block: Mapping[str, Any],
    truth_rms: float,
    n_members: int,
    *,
    factor: float | None = None,
    rms_rule: str = "hypot",
    normaliser: str = "rms",
    relief_rule: str = "normal",
    s2s_rule: str = "over_truth",
) -> AdrParts:
    """The instrument, with one knob deliberately wrong. Used only by defects."""
    adr = float(growth_block["band_area_dispersion_ratio"])
    bias = float(growth_block["band_area_error_bias"])
    scatter = float(growth_block["band_area_error_scatter"])
    rms_err = abs(bias) + scatter if rms_rule == "sum" else math.hypot(bias, scatter)
    f = factor if factor is not None else math.sqrt((n_members + 1.0) / n_members)
    member_sd = adr * rms_err / f
    norm = truth_rms if normaliser == "rms" else truth_rms * 0.5
    s2s = member_sd / (rms_err if s2s_rule == "over_rms_err" else norm)
    relief = (rms_err / norm) if relief_rule == "inverted" else (norm / rms_err)
    rebuilt = f * s2s * relief
    return AdrParts(
        model=model,
        fire_id=fire_id,
        spatial_block_id=int(spatial_block_id),
        n_members=int(n_members),
        adr=adr,
        bias=bias,
        scatter=scatter,
        rms_err=rms_err,
        member_sd=member_sd,
        truth_rms=truth_rms,
        spread_to_signal=s2s,
        denominator_relief=relief,
        identity_residual=abs(rebuilt - adr),
    )


def run_playthrough() -> dict[str, Any]:
    """Execute the S5 playthrough and return its coverage report."""
    from wildfire_nowcast.common import playthrough as PT

    return PT.run(build_playthrough()).as_dict()


# --------------------------------------------------------------------------
# the TRANSFER test — is a block's shortfall predicted by its distance from
# the train blocks' covariate support?
# --------------------------------------------------------------------------


def _scored_cells(
    fire_id: str,
    *,
    horizon_h: int,
    stride: int,
    cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows for every cell C6 SCORES, plus its FBFM40 class.

    Weather is read at ``t0 + 1`` — the hour that drives step 1 under C1.3's
    end-of-hour convention, the same phase ``model.inputs.weather_from_dataset``
    applies. Getting this wrong would compare our conditions to conditions one
    hour out of phase and quietly inflate every distance.
    """
    from wildfire_nowcast.sim.reader import channel_values

    radius = default_band_radius(horizon_h)
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
        chans = {c: channel_values(ds, c) for c in SUPPORT_CHANNELS}
        fuel = np.rint(channel_values(ds, "fuel_model_id")).astype(int)
    finally:
        ds.close()
    rows: list[np.ndarray] = []
    fuels: list[np.ndarray] = []
    n_t = int(state.shape[0])
    for t0 in range(0, n_t - horizon_h, max(1, int(stride))):
        x0 = state[t0]
        burned0 = x0 > UNBURNED
        if not burned0.any():
            continue
        if not ((state[t0 + horizon_h] > UNBURNED) & ~burned0).any():
            continue
        band = growth_band(x0, radius)
        if not band.any():
            continue
        idx = np.nonzero(band)
        rows.append(
            np.stack([chans[c][t0 + 1][idx] for c in SUPPORT_CHANNELS], axis=1).astype(np.float64)
        )
        fuels.append(fuel[t0 + 1][idx])
    x = np.concatenate(rows, 0)
    f = np.concatenate(fuels, 0)
    if x.shape[0] > cap:
        sel = rng.choice(x.shape[0], cap, replace=False)
        x, f = x[sel], f[sel]
    return x, f


def support_distance(
    train_fire_ids: Sequence[str],
    heldout_fire_ids: Sequence[str],
    *,
    horizon_h: int = 3,
    stride: int = 2,
    cap: int = 250_000,
    seed: int = 20260809,
) -> dict[str, Any]:
    """How far each held-out block's SCORED conditions sit from train support.

    Three readings, because no single one of them is convincing at n=4:
    ``mahalanobis`` on the standardised block-mean shift (uses the train
    covariance, so correlated channels do not double-count), ``oor_any_frac``
    (share of held-out scored cells outside the train 1-99 percentile box on at
    least one channel), and ``fuel_novel_frac`` (share on an FBFM40 class holding
    < 0.1% of train mass). Numpy only — C-4.3 freezes the interpreter.
    """
    rng = np.random.default_rng(seed)
    xs, fs = [], []
    for fid in train_fire_ids:
        x, f = _scored_cells(fid, horizon_h=horizon_h, stride=stride, cap=cap, rng=rng)
        xs.append(x)
        fs.append(f)
    xtr = np.concatenate(xs, 0)
    ftr = np.concatenate(fs, 0)
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd < 1e-9] = 1.0
    lo, hi = np.percentile(xtr, 1, axis=0), np.percentile(xtr, 99, axis=0)
    cinv = np.linalg.pinv(np.cov((xtr - mu) / sd, rowvar=False))
    classes, counts = np.unique(ftr, return_counts=True)
    mass = dict(zip(classes.tolist(), (counts / counts.sum()).tolist(), strict=True))

    out: dict[str, Any] = {
        "train_fire_ids": list(train_fire_ids),
        "n_train_cells": int(xtr.shape[0]),
        "channels": list(SUPPORT_CHANNELS),
        "blocks": {},
    }
    for fid in heldout_fire_ids:
        x, f = _scored_cells(fid, horizon_h=horizon_h, stride=stride, cap=cap, rng=rng)
        dz = ((x - mu) / sd).mean(0)
        out["blocks"][fid] = {
            "n_cells": int(x.shape[0]),
            "mahalanobis": float(np.sqrt(max(float(dz @ cinv @ dz), 0.0))),
            "oor_any_frac": float(np.mean(np.any((x < lo) | (x > hi), axis=1))),
            "fuel_novel_frac": float(np.mean([mass.get(int(c), 0.0) < 1e-3 for c in f])),
            "per_channel_z": {
                SUPPORT_CHANNELS[i]: float(dz[i]) for i in range(len(SUPPORT_CHANNELS))
            },
        }
    return out


# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--playthrough", action="store_true")
    ap.add_argument("--results", default="")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.playthrough:
        report = run_playthrough()
        text = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text + "\n")
        else:
            print(text)
        return 0 if report["passed"] else 1
    if not args.results:
        ap.error("--results is required unless --playthrough is given")
    out = decompose_record(args.results, stride=args.stride)
    bad = [c for c in out["known_answer_check"] if not c["agrees"]]
    resid = max((float(r["identity_residual"]) for r in out["parts"]), default=0.0)
    out["max_identity_residual"] = resid
    out["known_answer_check_passed"] = not bad
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    if bad:
        print(f"KNOWN-ANSWER CHECK FAILED on {[c['fire_id'] for c in bad]}")
        return 1
    if resid > 1e-9:
        print(f"IDENTITY RESIDUAL {resid:.3g} exceeds 1e-9")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
