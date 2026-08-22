"""Physical-plausibility diagnostics that a rendering suggests but cannot prove.

Everything here is computed from a C1 tensor alone - no model, no metrics - and
exists to answer questions a movie raises but the eye cannot settle:

* Does the fire actually run DOWNWIND? A sign error on ``wind_v10``, or a y-axis
  flip in the weather rasterisation, produces a wind field that still looks like
  a wind field and a fire that still looks like a fire. The two are only
  inconsistent when you compare them, which no single frame does.
* Does the fire run UPHILL? Same argument for ``elevation``/``slope``.
* Does the barrier mask block anything? A decorative barrier has already been
  found once in this project. A mask that correlates with nothing is worse than
  no mask, because a model will learn to ignore the channel and a reviewer will
  assume barriers were handled.

These are DIAGNOSTICS, not metrics: they triage a tensor, and they are not C6
and do not pretend to be. Each returns its own evidence so a verdict can be
argued with rather than believed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.sim.reader import FireFrames, load_fire

__all__ = [
    "Finding",
    "wind_alignment",
    "time_lag_consistency",
    "slope_alignment",
    "barrier_effect",
    "run_diagnostics",
    "main",
]


@dataclass
class Finding:
    """One diagnostic result. ``status`` is ok | suspect | fail | undetermined."""

    name: str
    status: str
    headline: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "headline": self.headline,
            "evidence": self.evidence,
        }


def _growth_steps(fire: FireFrames, min_cells: int = 2) -> list[int]:
    return [
        t
        for t in range(1, fire.n_hours)
        if int((fire.ever[t] & ~fire.ever[t - 1]).sum()) >= min_cells
    ]


def _centroid_m(mask: np.ndarray, fire: FireFrames) -> tuple[float, float] | None:
    """Centroid in EPSG:5070 METRES, or ``None`` for an EMPTY mask.

    In metres, +y is north, which is the same convention as ``wind_v10``. In row
    indices, +row is SOUTH, and every alignment statistic silently flips sign.

    Returning ``None`` rather than ``nan`` is deliberate and was a real defect:
    ``ever[t-1]`` is empty on a fire's first-appearance step, ``mean()`` of an
    empty slice is ``nan``, ``nan < 1e-9`` is False so the guard let it through,
    one ``nan`` poisoned the area-weighted mean, and the verdict ladder then fell
    through every ``<=``/``<`` comparison to the ``else`` branch and reported
    **ok**. A single unusable step silently converted the check into a PASS.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(fire.geom.x_centres[xs].mean()), float(fire.geom.y_centres[ys].mean())


def wind_alignment(fire: FireFrames, *, min_cells: int = 2) -> Finding:
    """Is the direction of spread aligned with the wind vector?

    Per growth hour: the unit vector from the previous burned centroid to the
    newly-burned centroid, dotted with the mean 10 m wind unit vector over the
    new cells. ``+1`` = the fire ran exactly downwind, ``-1`` = exactly upwind.
    Area-weighted across hours, so the documented wind runs dominate rather than
    a long tail of one-cell hours.
    """
    cosines: list[float] = []
    weights: list[float] = []
    per_step: list[dict[str, Any]] = []
    n_skipped = 0
    for t in _growth_steps(fire, min_cells):
        new = fire.ever[t] & ~fire.ever[t - 1]
        new_c = _centroid_m(new, fire)
        prev_c = _centroid_m(fire.ever[t - 1], fire)
        if new_c is None or prev_c is None:
            # No prior burned region => no direction of spread to speak of.
            # This is the fire's first-appearance step, not a defect.
            n_skipped += 1
            continue
        dx, dy = new_c[0] - prev_c[0], new_c[1] - prev_c[1]
        norm = float(np.hypot(dx, dy))
        if not np.isfinite(norm) or norm < 1e-9:
            n_skipped += 1
            continue
        u = float(fire.wind_u[t][new].mean())
        v = float(fire.wind_v[t][new].mean())
        speed = float(np.hypot(u, v))
        if not np.isfinite(speed) or speed < 1e-6:
            n_skipped += 1
            continue
        cos = (dx * u + dy * v) / (norm * speed)
        if not np.isfinite(cos):
            n_skipped += 1
            continue
        w = float(new.sum())
        cosines.append(cos)
        weights.append(w)
        per_step.append(
            {"t": t, "cos": round(cos, 3), "new_cells": int(w), "wind_ms": round(speed, 2)}
        )

    if not cosines:
        return Finding("wind_alignment", "undetermined", "no growth step had a usable wind vector")

    c = np.asarray(cosines)
    w = np.asarray(weights)
    weighted = float((c * w).sum() / w.sum())
    frac_downwind = float((c > 0).mean())
    n = int(c.size)
    top = sorted(per_step, key=lambda d: -d["new_cells"])[:8]
    # z for "more than half the growth hours ran downwind" against a fair coin.
    z_downwind = float((frac_downwind - 0.5) / np.sqrt(0.25 / n)) if n else 0.0

    evidence = {
        "area_weighted_cos": weighted,
        "unweighted_mean_cos": float(c.mean()),
        "frac_growth_hours_downwind": frac_downwind,
        "z_frac_downwind_vs_coinflip": z_downwind,
        "n_growth_hours": n,
        "n_steps_skipped_unusable": n_skipped,
        "largest_growth_hours": top,
    }

    if not np.isfinite(weighted):
        # Never let a non-finite statistic fall through a `<=` ladder into `ok`.
        # Per INTERFACES C-1, UNVERIFIABLE is a fail, not a pass.
        status, head = (
            "fail",
            "area-weighted cosine is not finite, so downwind alignment is UNVERIFIABLE "
            f"({n_skipped} steps unusable). Not a pass.",
        )
    elif weighted <= -0.30:
        status, head = (
            "fail",
            f"fire runs UPWIND on average (area-weighted cos {weighted:+.2f}). Suspect a sign "
            "error on wind_u10/wind_v10 or a y-axis flip in the weather rasterisation.",
        )
    elif weighted < 0.05:
        status, head = (
            "suspect",
            f"no downwind preference (area-weighted cos {weighted:+.2f}). Either the weather is "
            "misaligned with the labels, or spread here is terrain/fuel-driven.",
        )
    elif z_downwind < 2.0:
        # The area-weighted mean can be carried by two or three big hours while
        # the typical hour is a coin flip. That is not a verified alignment.
        status, head = (
            "suspect",
            f"area-weighted cos is {weighted:+.2f} but only {frac_downwind:.0%} of "
            f"{n} growth hours ran downwind (z={z_downwind:+.1f} vs a coin flip) — the mean is "
            "carried by a few large hours, not by a consistent preference.",
        )
    else:
        status, head = (
            "ok",
            f"fire runs downwind (area-weighted cos {weighted:+.2f}, "
            f"{frac_downwind:.0%} of {n} growth hours positive, z={z_downwind:+.1f}).",
        )
    return Finding("wind_alignment", status, head, evidence)


def _alignment_at_lag(fire: FireFrames, lag: int, min_cells: int) -> tuple[float, int]:
    """Area-weighted downwind cosine when the weather is shifted by ``lag`` hours."""
    cs: list[float] = []
    ws: list[float] = []
    for t in _growth_steps(fire, min_cells):
        tw = t + lag
        if not 0 <= tw < fire.n_hours:
            continue
        new = fire.ever[t] & ~fire.ever[t - 1]
        new_c = _centroid_m(new, fire)
        prev_c = _centroid_m(fire.ever[t - 1], fire)
        if new_c is None or prev_c is None:
            continue
        dx, dy = new_c[0] - prev_c[0], new_c[1] - prev_c[1]
        norm = float(np.hypot(dx, dy))
        u = float(fire.wind_u[tw][new].mean())
        v = float(fire.wind_v[tw][new].mean())
        speed = float(np.hypot(u, v))
        if not np.isfinite(norm) or not np.isfinite(speed) or norm < 1e-9 or speed < 1e-6:
            continue
        cos = (dx * u + dy * v) / (norm * speed)
        if not np.isfinite(cos):
            continue
        cs.append(cos)
        ws.append(float(new.sum()))
    if not cs:
        return float("nan"), 0
    c, w = np.asarray(cs), np.asarray(ws)
    return float((c * w).sum() / w.sum()), int(c.size)


def time_lag_consistency(fire: FireFrames, *, span: int = 3, min_cells: int = 2) -> Finding:
    """Empirical check on C1.3: is the weather in phase with the labels?

    C1.3 says GOFER ``tUTC`` is END-OF-HOUR and RTMA must be lagged an hour to
    match; ADR-006 records that getting it wrong "presents as a mediocre model,
    not as a bug". Nothing in the project verifies it - the contract can only
    check that an attribute *says* the right thing, which is not the same claim.

    This tests it from the data. Re-scoring the downwind alignment with the
    weather shifted by -span..+span hours should peak at 0 if the store's weather
    is already correctly aligned with its labels. A peak elsewhere means the
    tensor is out of phase by that many hours.

    Honest about its own power: on one fire the peak is shallow, so a peak at 0
    is *consistent with* the right lag rather than proof of it. The test sharpens
    as fires are added, and a peak at ±1 would be a loud finding at any n.
    """
    scores = {lag: _alignment_at_lag(fire, lag, min_cells) for lag in range(-span, span + 1)}
    usable = {k: v[0] for k, v in scores.items() if np.isfinite(v[0])}
    if len(usable) < 3:
        return Finding("time_lag_consistency", "undetermined", "too few growth hours to test lag")

    best = max(usable, key=lambda k: usable[k])
    margin = usable[best] - usable.get(0, float("-inf"))
    evidence = {
        "cos_by_lag": {str(k): round(v, 4) for k, v in sorted(usable.items())},
        "best_lag_h": int(best),
        "margin_over_lag0": float(margin),
        "n_growth_hours": int(scores[0][1]),
        "caveat": "shallow on a single fire; a peak at a non-zero lag is the loud case",
    }
    if best == 0:
        return Finding(
            "time_lag_consistency",
            "ok",
            f"downwind alignment peaks at lag 0 (cos {usable[0]:+.3f}), consistent with C1.3's "
            "end-of-hour convention having been applied correctly.",
            evidence,
        )
    if margin < 0.05:
        return Finding(
            "time_lag_consistency",
            "undetermined",
            f"alignment peaks at lag {best:+d} h but only by {margin:.3f} — within noise for one "
            "fire. Re-run once more fires exist.",
            evidence,
        )
    return Finding(
        "time_lag_consistency",
        "suspect",
        f"downwind alignment peaks at lag {best:+d} h, beating lag 0 by {margin:.3f}. The weather "
        f"in this store may be {abs(best)} h out of phase with the labels (C1.3).",
        evidence,
    )


def slope_alignment(fire: FireFrames, *, min_cells: int = 2) -> Finding:
    """Do newly-burned cells sit HIGHER than the region they spread from?

    Fire spreads faster upslope; a consistent downhill bias suggests elevation is
    inverted or misregistered. Weak evidence on flat terrain, so the finding
    reports the terrain relief alongside its verdict rather than asserting into
    a landscape that cannot support the claim.
    """
    deltas: list[float] = []
    weights: list[float] = []
    speeds: list[float] = []
    for t in _growth_steps(fire, min_cells):
        new = fire.ever[t] & ~fire.ever[t - 1]
        prev = fire.ever[t - 1]
        if not prev.any():
            continue
        deltas.append(float(fire.elevation[new].mean() - fire.elevation[prev].mean()))
        weights.append(float(new.sum()))
        speeds.append(float(fire.wind_speed[t][new].mean()))
    if not deltas:
        return Finding("slope_alignment", "undetermined", "no usable growth step")

    d = np.asarray(deltas)
    w = np.asarray(weights)
    sp = np.asarray(speeds)
    weighted = float((d * w).sum() / w.sum())
    relief = float(np.ptp(fire.elevation))
    # Downhill spread is real when the wind is driving it (Diablo/Santa Ana
    # events are downslope). Reporting the wind correlation is what separates
    # "physics" from "elevation channel is inverted", which look identical in
    # the mean.
    corr = float(np.corrcoef(d, sp)[0, 1]) if d.size > 2 and sp.std() > 0 else float("nan")
    evidence = {
        "area_weighted_delta_m": weighted,
        "relief_m": relief,
        "n_growth_hours": int(d.size),
        "corr_delta_elev_windspeed": corr,
    }
    if relief < 50.0:
        return Finding(
            "slope_alignment",
            "undetermined",
            f"terrain relief is only {relief:.0f} m; slope preference is not measurable here",
            evidence,
        )
    if weighted < -0.10 * relief:
        wind_driven = np.isfinite(corr) and corr < -0.2
        status = "ok" if wind_driven else "suspect"
        head = (
            f"fire spreads DOWNHILL by {abs(weighted):.0f} m on average over {relief:.0f} m of "
            + (
                f"relief, and more so in stronger wind (corr {corr:+.2f}) — consistent with a "
                "wind-driven downslope run, not an inverted elevation channel."
                if wind_driven
                else "relief with no wind association. Check the sign/registration of elevation."
            )
        )
        return Finding("slope_alignment", status, head, evidence)
    return Finding(
        "slope_alignment",
        "ok",
        f"mean elevation change on spread {weighted:+.0f} m over {relief:.0f} m of relief",
        evidence,
    )


def barrier_effect(fire: FireFrames) -> Finding:
    """Does ``water_barrier_mask`` actually suppress spread?

    Compares, among cells that were EXPOSED to the front (adjacent to burned
    ground at some point), the fraction that eventually burn inside vs outside
    the mask. A ratio near 1 means the channel is decorative for this fire - the
    exact defect already found once in this project - and a model can only learn
    to ignore it.
    """
    if not fire.barrier.any():
        return Finding("barrier_effect", "undetermined", "no barrier cells in this domain")

    from wildfire_nowcast.sim.movie import _dilate  # noqa: PLC0415

    exposed = np.zeros(fire.geom.shape, dtype=bool)
    for t in range(fire.n_hours - 1):
        exposed |= _dilate(fire.ever[t], 1) & ~fire.ever[t]
    burned = fire.ever[-1]

    inside = exposed & fire.barrier
    outside = exposed & ~fire.barrier
    if inside.sum() < 5 or outside.sum() < 5:
        return Finding(
            "barrier_effect", "undetermined", "too few exposed cells on one side of the mask"
        )

    n_in, n_out = int(inside.sum()), int(outside.sum())
    p_in = float(burned[inside].mean())
    p_out = float(burned[outside].mean())
    ratio = p_in / p_out if p_out > 0 else float("inf")

    # Two-proportion z. Exposed-cell counts are small (tens), so a ratio of 0.85
    # can be pure sampling noise; quoting it without this invites over-reading.
    p_pool = float(burned[inside | outside].mean())
    se = float(np.sqrt(max(p_pool * (1 - p_pool) * (1 / n_in + 1 / n_out), 1e-12)))
    z = (p_in - p_out) / se

    evidence = {
        "p_burn_exposed_barrier": p_in,
        "p_burn_exposed_nonbarrier": p_out,
        "ratio": ratio,
        "z_two_proportion": z,
        "n_exposed_barrier": n_in,
        "n_exposed_nonbarrier": n_out,
        "barrier_frac_of_domain": float(fire.barrier.mean()),
    }
    if ratio >= 0.9 or abs(z) < 1.0:
        return Finding(
            "barrier_effect",
            "suspect",
            f"barrier cells burn at {ratio:.2f}x the exposed non-barrier rate (z={z:+.1f}, "
            f"n={n_in}) — no clear suppression on this fire. Decorative-barrier risk: a model "
            "can only learn to ignore channel 12 from data like this.",
            evidence,
        )
    if ratio >= 0.6:
        return Finding(
            "barrier_effect",
            "ok",
            f"barrier cells burn at {ratio:.2f}x the non-barrier rate (z={z:+.1f}, n={n_in}) — "
            "weak but present suppression.",
            evidence,
        )
    return Finding(
        "barrier_effect",
        "ok",
        f"barrier cells burn at {ratio:.2f}x the non-barrier rate (z={z:+.1f}, n={n_in})",
        evidence,
    )


def run_diagnostics(tensor: str | Path) -> dict[str, Any]:
    """All plausibility checks for one tensor, as a JSON-able dict."""
    fire = load_fire(tensor)
    findings = [
        wind_alignment(fire),
        time_lag_consistency(fire),
        slope_alignment(fire),
        barrier_effect(fire),
    ]
    return {
        "tensor": str(tensor),
        **fire.summary(),
        "findings": [f.as_dict() for f in findings],
        "worst_status": _worst([f.status for f in findings]),
    }


def _worst(statuses: list[str]) -> str:
    for level in ("fail", "suspect", "undetermined", "ok"):
        if level in statuses:
            return level
    return "ok"


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.diagnostics",
        description="Physical-plausibility diagnostics on a C1 tensor.",
    )
    ap.add_argument("tensors", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    reports = [run_diagnostics(t) for t in args.tensors]
    for rep in reports:
        print(f"\n=== {rep['fire_id']}  ({rep['tensor']})  -> {rep['worst_status'].upper()}")
        for f in rep["findings"]:
            print(f"  [{f['status']:>13}] {f['name']}: {f['headline']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(reports, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
