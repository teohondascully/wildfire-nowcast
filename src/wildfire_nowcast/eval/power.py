"""[M11] What can each headline channel DETECT at n = 5 held-out blocks?

THE QUESTION, AND WHY IT OUTRANKS THE NEXT MODEL
------------------------------------------------
M10 (ADR-049 (6)) reported that three of five headline channels returned
``not_a_verdict`` because they could not separate their own control: 3 h band
Brier **+0.80** block-SD, arrival-time CRPS **+0.91**, ``calibration_error``
**+1.20**, against a 2.0 bar - while the control predicted **3.60x less area**.
That is a statement about the INSTRUMENT, not about that experiment, and it
applies to every verdict those channels have ever carried.

This module turns that observation into a measurement. Given a set of arms whose
TRUE degradation is known and ordered, it reports, per channel:

* the separation curve - block-SD as a function of true severity;
* the **minimum detectable effect** at the 2.0 bar with 5 blocks, read off it;
* whether the channel is MONOTONE in true severity.

**A non-monotone channel is a finding, not noise to smooth.** Insensitivity means
a gate cannot resolve a difference; non-monotonicity means the metric orders
hypotheses wrongly, which is worse, because a wrong order still produces a
confident verdict.

THE STATISTIC IS NOT NEW AND IS NOT MINE
----------------------------------------
Equal-block paired separation, **SD across held-out BLOCKS, never seeds**
(ADR-042), pooled by :func:`wildfire_nowcast.common.pooling.equal_block_mean`
(C6.3) and computed by :func:`wildfire_nowcast.common.separation.separation`
against the pre-existing, registered bar
:data:`wildfire_nowcast.common.separation.MIN_SEPARATION_SD` = 2.0. Ratios are
differenced in LOG space (C6.5), because pooling or differencing a ratio
arithmetically reintroduces the asymmetry that clause removes. **Nothing here
mints a constant**, and the procedure is deliberately identical to the one M10
licensed its channels with, so the numbers are comparable to M10's by
construction rather than by claim.

SEVERITY MUST NOT BE MEASURED BY THE CHANNEL UNDER TEST
-------------------------------------------------------
Every rung declares its severity in a unit the scored channels do not compute:
the DESIGNED area ratio for the area family, and the measured increment IoU
against the undegraded arm for the shape family. Otherwise the curve is circular
- a channel would be being characterised against itself. The one place that
caveat still bites is ``growth_calibration``, which IS an area ratio by
construction; it is reported with that stated, and it is the reason the ladder
carries a SHAPE family at exactly fixed area.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.common.pooling import (
    IncompleteBlockCoverageError,
    equal_block_mean,
    equal_block_mean_of,
)
from wildfire_nowcast.common.separation import (
    MIN_SEPARATION_SD,
    BlockPair,
    Separation,
    separation,
)

__all__ = [
    "Channel",
    "HEADLINE_CHANNELS",
    "block_values",
    "channel_separation",
    "detection_curve",
    "minimum_detectable_effect",
    "monotonicity",
]


@dataclass(frozen=True)
class Channel:
    """One headline channel, named with the key it is actually read from."""

    name: str
    key: str
    lower_is_better: bool
    is_ratio: bool
    note: str
    quarantined: bool = False

    @property
    def target(self) -> float | None:
        """The value a perfect forecast scores, where the channel has one."""
        return 1.0 if self.is_ratio else None


#: The channels a gate in this project can turn on, plus the two quarantined ones
#: that a reader will still find in a table. Declared as a table so a channel
#: cannot be characterised without naming the key it was read from - three
#: separate defects in this repo were a gate criterion missing from the artifact
#: its gate is adjudicated from.
HEADLINE_CHANNELS: tuple[Channel, ...] = (
    Channel(
        "brier_3h",
        "band_brier_by_horizon.3",
        True,
        False,
        "3 h band Brier — the accuracy criterion that returned +0.80 block-SD "
        "against a 3.60x area error at M10.",
    ),
    Channel(
        "arrival_crps",
        "arrival_crps",
        True,
        False,
        "Arrival-time CRPS. G2's supporting leg, and the channel ADR-049 (6) left open.",
    ),
    Channel(
        "calibration_error_3h",
        "band_calibration_error_by_horizon.3",
        True,
        False,
        "G3's calibration criterion (ADR-020 (4b)), growth-masked.",
    ),
    Channel(
        "dispersion",
        "band_area_dispersion_ratio",
        False,
        True,
        "G3's dispersion criterion (C6.1). A ratio: 1.0 is calibrated, so it is "
        "differenced in log space.",
    ),
    Channel(
        "growth_calibration",
        "band_growth_calibration",
        False,
        True,
        "G3's first-moment condition (ADR-039 (5)). CIRCULARITY WARNING: this "
        "channel IS a predicted-area ratio, so on the AREA family it is being "
        "measured against a near-copy of itself and its sensitivity there is a "
        "floor on what any channel could reach, not a property of its design.",
    ),
    Channel(
        "iou_shape_masked_3h",
        "band_best_member_iou_shape_masked_by_horizon.3",
        False,
        False,
        "G2's gate criterion (C6.4). Higher is better; its null floor is exactly 0.",
    ),
    Channel(
        "reliability_3h",
        "band_reliability_by_horizon.3",
        True,
        False,
        "QUARANTINED by ADR-020 (silence is trivially calibrated). Characterised "
        "because a reader will find it in a table, never as capability.",
        quarantined=True,
    ),
)


def _get(row: Mapping[str, Any] | None, key: str) -> Any:
    """Read a headline key, including the dotted ``by_horizon`` form."""
    if row is None:
        return None
    if "." not in key:
        return row.get(key)
    base, index = key.split(".", 1)
    return (row.get(base) or {}).get(index)


def block_values(per_fire: Mapping[str, Any], model: str, key: str, stratum: str) -> dict[str, Any]:
    """Equal-block pooling of one headline key. STRICT first (C6.3).

    ``allow_missing_blocks=False`` is tried first because a block that
    contributes nothing is a hard failure, not a silently smaller sample. A gap
    is still REPORTED rather than raised out, because ``area_dispersion_ratio``
    is legitimately UNDEFINED at perfect mean calibration (C6.5) - so an arm can
    lose a block for a reason that belongs to the metric - but it travels into
    the artifact, so partial coverage cannot read as full coverage.
    """
    if "." in key:
        base, index = key.split(".", 1)
        values: dict[Any, list[Any]] = {}
        for fire in per_fire.values():
            row = ((fire.get("models", {}).get(model) or {}).get(stratum)) or {}
            values.setdefault(int(fire["spatial_block_id"]), []).append(
                (row.get(base) or {}).get(index)
            )
        what = f"{model}.{stratum}.{key}"
        try:
            return {**equal_block_mean_of(values, what=what), "strict_coverage_error": None}
        except IncompleteBlockCoverageError as exc:
            out = equal_block_mean_of(values, allow_missing_blocks=True, what=what)
            return {**out, "strict_coverage_error": str(exc)}
    try:
        return {**equal_block_mean(per_fire, model, key, stratum), "strict_coverage_error": None}
    except IncompleteBlockCoverageError as exc:
        out = equal_block_mean(per_fire, model, key, stratum, allow_missing_blocks=True)
        return {**out, "strict_coverage_error": str(exc)}


def channel_separation(
    per_fire: Mapping[str, Any],
    candidate: str,
    reference: str,
    channel: Channel,
    *,
    stratum: str = "growth_windows",
    absolute_ratio: bool = False,
) -> Separation:
    """Paired equal-block separation of ``candidate`` from ``reference``.

    Ratio channels are differenced in LOG space (C6.5). For a POWER question the
    sign is not the point - the question is whether the channel registers a
    difference at all - so callers read ``abs(separation_sd)``; the signed value
    travels beside it so a direction is never lost.

    ``absolute_ratio`` switches a ratio channel from *"did it MOVE"*
    (``|log c - log r|``, the power question) to *"did it get WORSE"*
    (``|log c| - |log r|``, distance from the calibrated value 1.0). M10's
    licences used the second; the two are different questions and conflating
    them is how a channel that moved a long way in the RIGHT direction reads as
    an unmoved channel.
    """
    cand = block_values(per_fire, candidate, channel.key, stratum)["per_block"]
    ref = block_values(per_fire, reference, channel.key, stratum)["per_block"]
    pairs: list[BlockPair] = []
    lower_is_better = channel.lower_is_better
    for block in sorted(set(cand) & set(ref)):
        c, r = cand[block], ref[block]
        if c is None or r is None:
            continue
        if channel.is_ratio:
            if c <= 0 or r <= 0:
                continue
            c, r = math.log(c), math.log(r)
            if absolute_ratio:
                c, r = abs(c), abs(r)
                lower_is_better = True
        pairs.append(
            BlockPair(
                block_id=int(block), candidate=float(c), reference=float(r), label=f"block {block}"
            )
        )
    return separation(pairs, lower_is_better=lower_is_better)


def detection_curve(
    per_fire: Mapping[str, Any],
    reference: str,
    rungs: Sequence[Mapping[str, Any]],
    channel: Channel,
    *,
    stratum: str = "growth_windows",
) -> list[dict[str, Any]]:
    """One row per rung: declared severity, level, separation, detected?"""
    ref_level = block_values(per_fire, reference, channel.key, stratum).get("equal_block_mean")
    rows: list[dict[str, Any]] = []
    for rung in rungs:
        name = str(rung["model"])
        sep = channel_separation(per_fire, name, reference, channel, stratum=stratum)
        level = block_values(per_fire, name, channel.key, stratum)
        sd = sep.separation_sd
        rows.append(
            {
                "model": name,
                "family": rung.get("family"),
                "severity": float(rung["severity"]),
                "severity_unit": rung.get("severity_unit"),
                "truth_distance": rung.get("truth_distance"),
                "level": level.get("equal_block_mean"),
                "reference_level": ref_level,
                "n_blocks": sep.n_blocks,
                "separation_sd": sd,
                "abs_separation_sd": None if sd is None else abs(float(sd)),
                "blocks_favouring": sep.blocks_favouring,
                "mean_margin": sep.mean_margin,
                "detected": bool(sd is not None and abs(float(sd)) >= MIN_SEPARATION_SD),
                "undefined_reason": sep.undefined_reason,
            }
        )
    rows.sort(key=lambda r: r["severity"])
    return rows


def minimum_detectable_effect(
    rows: Sequence[Mapping[str, Any]], *, bar: float = MIN_SEPARATION_SD
) -> dict[str, Any]:
    """The smallest severity from which EVERY larger rung clears ``bar``.

    "From which every larger rung clears" rather than "the first rung that
    clears": one rung crossing a bar and the next falling back under it is not a
    detection threshold, it is noise, and reporting the lucky rung would be the
    optimistic read of exactly the instrument this module exists to distrust. The
    crossing is then linearly interpolated between the last undetected rung and
    the first sustained detection, so the answer is not quantised to the ladder.
    """
    ordered = sorted(rows, key=lambda r: r["severity"])
    scored = [r for r in ordered if r["abs_separation_sd"] is not None]
    if not scored:
        return {"mde": None, "bound": "undefined", "reason": "no rung produced a separation"}

    sustained: int | None = None
    for i in range(len(scored)):
        if all(float(r["abs_separation_sd"]) >= bar for r in scored[i:]):
            sustained = i
            break
    if sustained is None:
        biggest = scored[-1]
        return {
            "mde": None,
            "bound": "greater_than",
            "largest_severity_tested": biggest["severity"],
            "largest_separation_sd": biggest["abs_separation_sd"],
            "reason": (
                f"no rung reached {bar} block-SD; the channel did not detect a severity of "
                f"{biggest['severity']:.4g} in this unit"
            ),
        }
    if sustained == 0:
        first = scored[0]
        return {
            "mde": first["severity"],
            "bound": "at_or_below",
            "reason": (
                f"the smallest rung tested ({first['severity']:.4g}) already clears {bar} "
                "block-SD; the MDE is at or below the bottom of the ladder"
            ),
            "smallest_severity_tested": first["severity"],
        }
    lo, hi = scored[sustained - 1], scored[sustained]
    x0, y0 = float(lo["severity"]), float(lo["abs_separation_sd"])
    x1, y1 = float(hi["severity"]), float(hi["abs_separation_sd"])
    mde = x1 if y1 == y0 else x0 + (bar - y0) * (x1 - x0) / (y1 - y0)
    return {
        "mde": float(min(max(mde, x0), x1)),
        "bound": "interpolated",
        "bracket": [x0, x1],
        "bracket_separation_sd": [y0, y1],
        "bracket_models": [lo.get("model"), hi.get("model")],
        "reason": "linear interpolation between the last undetected rung and the first "
        "SUSTAINED detection",
    }


def monotonicity(
    rows: Sequence[Mapping[str, Any]],
    *,
    by: str = "truth_distance",
    value: str = "level",
    expect_increasing: bool = True,
) -> dict[str, Any]:
    """Is the channel ORDERING the rungs correctly?

    Reported loudly and never smoothed. A channel that is merely insensitive
    cannot resolve a difference; a channel that is non-monotone in true severity
    ranks a worse forecast above a better one, and will do so confidently.

    ``by='truth_distance'`` is the honest axis for a proper score: severity is
    ``|log(predicted area / truth area)|``, so the score should worsen as the
    forecast moves away from truth in EITHER direction, and a proper score
    scoring a 2x-too-small forecast the same as a 1.4x-too-large one is correct
    behaviour rather than a defect. Ordering by distance from the REFERENCE
    instead would manufacture a non-monotonicity out of that correctness.
    """
    pts = [
        (float(r[by]), float(r[value]))
        for r in rows
        if r.get(by) is not None and r.get(value) is not None
    ]
    pts.sort(key=lambda p: p[0])
    if len(pts) < 3:
        return {"monotone": None, "n_points": len(pts), "reason": "fewer than 3 usable rungs"}
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    diffs = np.diff(ys)
    wrong_sign = diffs < 0 if expect_increasing else diffs > 0
    inversions = [
        {
            "from": {by: float(xs[i]), value: float(ys[i])},
            "to": {by: float(xs[i + 1]), value: float(ys[i + 1])},
            "step": float(diffs[i]),
        }
        for i in range(len(diffs))
        if wrong_sign[i]
    ]
    spearman = _spearman(xs, ys)
    return {
        "monotone": not inversions,
        "expected_direction": "increasing" if expect_increasing else "decreasing",
        "n_points": len(pts),
        "n_inversions": len(inversions),
        "inversions": inversions,
        "spearman": spearman,
        "axis": by,
        "value": value,
    }


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Rank correlation, written out because scipy is not a dependency (C-4.3)."""
    if xs.size < 2:
        return None
    rx = np.argsort(np.argsort(xs)).astype(np.float64)
    ry = np.argsort(np.argsort(ys)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return None if denom == 0.0 else float((rx * ry).sum() / denom)
