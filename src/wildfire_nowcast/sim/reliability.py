"""A reliability diagram that CANNOT be read without its own bin counts.

    python -m wildfire_nowcast.sim.reliability <a run record with reliability_bins> \
        --emphasis <arm> --out <page>.png

``--out`` is REQUIRED and has no default. Two reasons, and the second is the
one that decided it. A page that writes itself to a fixed name silently
overwrites the last one, and a reader who finds two pages cannot tell which run
either belongs to. And a default would name a destination under the generated
figures directory, which is not in the git index, so a tracked module naming it
owes another lead a declaration entry before this file can be committed at all.
Requiring the argument leaves the destination to the caller and this module
owing nobody anything.

WHY THIS EXISTS AND WHY IT IS NOT :mod:`wildfire_nowcast.sim.dashboard`
-----------------------------------------------------------------------
``sim.dashboard`` draws one panel per lead from a per-run C6 ``evaluate()``
dict. That panel is correct and it is not sufficient, for one reason that is a
property of this problem rather than of that code: **the forecast distribution
on a fire-spread grid is almost entirely one bin.** On the pooled held-out
growth windows at 3 h, 98.86% to 100.00% of every arm's cells sit in
``[0, 0.1)`` -- the bin where every arm on the board, including a forecaster
that ignites nothing, is trivially right.

A curve drawn over ten bins gives each of them the same visual weight. So the
right-hand end of a reliability curve, which is the only part of it that says
anything about whether a model can COMMIT, can rest on nine cells and look
exactly like the left-hand end resting on 1.75 million. A diagram drawn that way
agrees with itself whatever the model does. It is a chart that cannot fail, and
this module exists because we already published one.

WHAT IS DRAWN, AND WHY EACH HALF IS LOAD-BEARING
------------------------------------------------
1. **the curve against the diagonal**, with a 95% Wilson interval on every
   plotted point. At n = 9 that interval spans most of the unit square, which
   is the honest picture of a point the eye would otherwise read as a
   measurement;
2. **the occupancy of every bin on a log axis, directly beneath and on the same
   x**, including the bins with zero cells, which the curve necessarily drops.
   The concentration is a fact about the forecast, not a rendering detail, so it
   gets an axis rather than a footnote;
3. **the commitment table**: how many cells an arm puts at or above a stated
   probability, how many of them burned, and what fraction that is. This is the
   quantity the aggregate scores average away.

WHAT IS *NOT* DONE HERE
-----------------------
Nothing is recomputed. Every number on the page is read out of C6's own
``reliability_bins`` and ``reliability_summary`` blocks, which carry ``n``,
``sum_p``, ``sum_y``, ``mean_forecast`` and ``observed_frequency`` per bin per
lead. No tensor is opened, no model is loaded, no probability is re-derived.
That is what makes the same page fair for the learned kernel, for persistence,
for the wind-advected ellipse and for ELMFIRE: all of them are read through one
aperture, and a reader can check any cell of the figure against the artifact.

THE CURVE ITSELF IS NOT RE-IMPLEMENTED. ``sim.dashboard.reliability_curve`` is
the single implementation of "turn C6 bins into a curve", and
:func:`curve_agreement` asserts that this module's row table reproduces it
exactly rather than quietly holding a second copy. What this module adds is the
part that function deliberately DROPS -- the empty bins and ``sum_y`` -- because
dropping them is right for a curve and wrong for an occupancy axis.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wildfire_nowcast.common.logs import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
)
from wildfire_nowcast.sim.dashboard import reliability_curve  # noqa: E402
from wildfire_nowcast.sim.style import COL_TEXT, COL_WARN, stamp  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LEADS",
    "DEFAULT_THRESHOLD",
    "BinRow",
    "bin_rows",
    "commitment",
    "concentration",
    "curve_agreement",
    "commitment_table",
    "wilson_interval",
    "zero_success_upper_bound",
    "build_page",
    "render",
    "main",
]

#: The leads this project forecasts at. C6.7 requires a report at all three and
#: not only at the one where an instrument looks best, so the page carries three
#: columns whether or not anything interesting happens in two of them.
DEFAULT_LEADS: tuple[int, int, int] = (1, 2, 3)

#: "Commits to a probability" means at or above one half. Stated as a constant
#: and stamped into the artifact because it is a free parameter: a threshold
#: chosen after seeing which value flatters an arm is a result about the analyst.
DEFAULT_THRESHOLD: float = 0.5

#: Draw order and styling. The emphasis arm is passed in, so this is only the
#: family palette: the ellipse arms share a hue and darken with calibration, the
#: two model arms share the warning colour, persistence is grey because it is a
#: control rather than a competitor.
_ARM_STYLE: dict[str, dict[str, Any]] = {
    "persistence": {"color": "#9ca3af", "ls": ":", "lw": 1.3},
    "ellipse": {"color": "#93c5fd", "ls": "-", "lw": 1.5},
    "ellipse_cal2h": {"color": "#3b82f6", "ls": "-", "lw": 1.5},
    "ellipse_cal3h": {"color": "#1d4ed8", "ls": "-", "lw": 1.7},
    "elmfire": {"color": "#047857", "ls": "-", "lw": 1.7},
}
_FALLBACK_STYLE: dict[str, Any] = {"color": "#6b7280", "ls": "-", "lw": 1.3}
_EMPHASIS_STYLE: dict[str, Any] = {"color": COL_WARN, "ls": "-", "lw": 2.8}
_EMPHASIS_ABLATION_STYLE: dict[str, Any] = {"color": COL_WARN, "ls": "--", "lw": 1.5}


@dataclass(frozen=True)
class BinRow:
    """One forecast-probability bin at one lead, exactly as C6 emitted it.

    ``mean_forecast`` and ``observed_frequency`` are ``None`` on an empty bin.
    They are carried as ``None`` rather than coerced to 0.0 because 0.0 is a
    measurement and ``None`` is the absence of one, and a reliability diagram is
    precisely where that distinction gets lost.
    """

    lead_h: int
    bin_index: int
    lower: float
    upper: float
    n: int
    burned: int
    mean_forecast: float | None
    observed_frequency: float | None

    @property
    def occupied(self) -> bool:
        return self.n > 0


def _open_maybe_gzip(path: str | Path) -> dict[str, Any]:
    """Read a JSON artifact whether or not it is gzipped, decided by content.

    By CONTENT and not by suffix: the gzip magic number is two bytes and a
    file renamed by hand is the ordinary way an artifact stops opening.
    """
    p = Path(path)
    head = p.open("rb").read(2)
    if head == b"\x1f\x8b":
        with gzip.open(p, "rt") as fh:
            loaded = json.load(fh)
    else:
        loaded = json.loads(p.read_text())
    if not isinstance(loaded, dict):
        raise TypeError(f"{p} does not hold a JSON object at the top level")
    return loaded


def bin_rows(bins: Sequence[Mapping[str, Any]], lead_h: int) -> list[BinRow]:
    """Every bin at one lead, EMPTY ONES INCLUDED, ordered by bin lower edge.

    The empty bins are the point. ``sim.dashboard.reliability_curve`` drops them
    and is right to -- an empty bin plotted at ``(0, 0)`` draws a
    perfectly-calibrated-looking point supported by nothing -- but the same
    omission on an OCCUPANCY axis would hide that a bin exists and is unused,
    which is a different claim from the bin not existing.
    """
    rows: list[BinRow] = []
    for b in bins:
        if int(b.get("lead_h", -1)) != int(lead_h):
            continue
        mf = b.get("mean_forecast")
        of = b.get("observed_frequency")
        rows.append(
            BinRow(
                lead_h=int(lead_h),
                bin_index=int(b["bin_index"]),
                lower=float(b["bin_lower"]),
                upper=float(b["bin_upper"]),
                n=int(b["n"]),
                burned=int(round(float(b.get("sum_y", 0.0) or 0.0))),
                mean_forecast=None if mf is None else float(mf),
                observed_frequency=None if of is None else float(of),
            )
        )
    return sorted(rows, key=lambda r: r.lower)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Closed form, no scipy.

    Wilson rather than the normal approximation because the normal one is
    degenerate at exactly the counts this page is about: at ``k = 0`` it returns
    the single point 0.0, which would draw a zero-width error bar on the most
    uncertain point in the figure. Wilson stays two-sided and finite there.

    THE BOUNDARY CASES ARE RETURNED EXACTLY rather than computed. At ``k = 0``
    the Wilson lower limit is analytically 0 (``centre`` and ``half`` are the
    same quantity there and cancel), and at ``k = n`` the upper limit is
    analytically 1. In floating point that cancellation leaves a residual of
    order 1e-17 with the WRONG SIGN, so the interval failed to contain its own
    point estimate on 6 of the bins this page draws - each of them a zero-burn
    bin, i.e. exactly the bins the figure is about. Found by matplotlib
    refusing a negative error bar, not by inspection.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    lo = 0.0 if k <= 0 else max(0.0, centre - half)
    hi = 1.0 if k >= n else min(1.0, centre + half)
    return (lo, hi)


def zero_success_upper_bound(n: int, alpha: float = 0.05) -> float | None:
    """EXACT one-sided upper confidence bound on a rate observed as 0 of ``n``.

    ``1 - alpha ** (1 / n)``, which is the Clopper-Pearson upper limit at
    ``k = 0`` written in closed form, and the sharp version of the "rule of
    three" approximation ``3 / n``. It is reported separately from the Wilson
    interval, and labelled, because "0 of 162" is the headline of this page and
    the honest statement of it is an upper bound rather than a point estimate.
    """
    if n <= 0:
        return None
    return float(1.0 - alpha ** (1.0 / n))


def commitment(
    bins: Sequence[Mapping[str, Any]],
    lead_h: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """How often an arm COMMITS, and how often it is right when it does.

    A cell counts as a commitment when its bin's LOWER edge is at or above
    ``threshold``, so a cell is never counted on the strength of a bin that
    straddles the line.
    """
    rows = [r for r in bin_rows(bins, lead_h) if r.lower >= threshold - 1e-12]
    n = sum(r.n for r in rows)
    burned = sum(r.burned for r in rows)
    lo, hi = wilson_interval(burned, n) if n > 0 else (float("nan"), float("nan"))
    return {
        "lead_h": int(lead_h),
        "threshold": float(threshold),
        "n_cells": int(n),
        "n_burned": int(burned),
        "observed_frequency": (burned / n) if n > 0 else None,
        "wilson95": None if n == 0 else [lo, hi],
        "exact_one_sided_upper95": zero_success_upper_bound(n) if (n > 0 and burned == 0) else None,
        "bins_used": [[r.lower, r.upper] for r in rows],
    }


def concentration(bins: Sequence[Mapping[str, Any]], lead_h: int) -> dict[str, Any]:
    """Where the forecast mass is: the share in the lowest bin, and how many bins are used."""
    rows = bin_rows(bins, lead_h)
    total = sum(r.n for r in rows)
    lowest = rows[0].n if rows else 0
    return {
        "lead_h": int(lead_h),
        "n_cells_total": int(total),
        "n_cells_lowest_bin": int(lowest),
        "lowest_bin": [rows[0].lower, rows[0].upper] if rows else None,
        "lowest_bin_share": (lowest / total) if total else None,
        "n_occupied_bins": sum(1 for r in rows if r.occupied),
        "n_bins": len(rows),
    }


def curve_agreement(
    bins: Sequence[Mapping[str, Any]],
    lead_h: int,
    reference_bins: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """CONTROL: the rows here must reproduce ``dashboard.reliability_curve`` exactly.

    There is one implementation of the curve and it is not in this file. This
    check exists so that the statement "the same curve, plus its counts" is
    verified on the data being drawn rather than asserted in a docstring; if it
    ever disagrees, the page carries the disagreement instead of a curve.

    ``reference_bins`` EXISTS FOR THE PLANT and for nothing else. With both sides
    fed from one list, an equality assertion is only as good as the claim that it
    could have come out False, and that claim is not observable from a green run.
    A self-test passes a perturbed list here and requires ``identical`` to go
    False, which is the only way this control is known to be a control.
    """
    against = bins if reference_bins is None else reference_bins
    theirs = reliability_curve([dict(b) for b in against], lead_h)
    mine = [r for r in bin_rows(bins, lead_h) if r.occupied and r.mean_forecast is not None]
    mine.sort(key=lambda r: float(r.mean_forecast or 0.0))
    xs = np.asarray([float(r.mean_forecast or 0.0) for r in mine], dtype=float)
    ys = np.asarray([float(r.observed_frequency or 0.0) for r in mine], dtype=float)
    ns = np.asarray([float(r.n) for r in mine], dtype=float)
    same = (
        xs.shape == theirs["forecast"].shape
        and bool(np.array_equal(xs, theirs["forecast"]))
        and bool(np.array_equal(ys, theirs["observed"]))
        and bool(np.array_equal(ns, theirs["n"]))
    )
    return {
        "lead_h": int(lead_h),
        "n_points_here": int(xs.size),
        "n_points_dashboard": int(theirs["forecast"].size),
        "identical": bool(same),
    }


def _arms(block: Mapping[str, Any]) -> list[str]:
    return [k for k, v in block.items() if isinstance(v, dict) and v.get("reliability_bins")]


def build_page(
    payload: Mapping[str, Any],
    *,
    block: str = "pooled_growth_windows",
    leads: Sequence[int] = DEFAULT_LEADS,
    threshold: float = DEFAULT_THRESHOLD,
    emphasis: str = "",
) -> dict[str, Any]:
    """Every number that will appear on the figure, as JSON, before anything is drawn.

    The sidecar is not a convenience. A figure is quotable and unauditable; the
    sidecar is what lets a reader check a point on the curve against the run
    record without re-running anything, and it is what makes a disagreement
    between this page and the artifact visible instead of arguable.
    """
    if block not in payload:
        raise KeyError(f"artifact has no block {block!r}; it has {sorted(payload)}")
    section = payload[block]
    arms = _arms(section)
    if not arms:
        raise ValueError(f"block {block!r} carries no arm with reliability_bins")

    out_arms: dict[str, Any] = {}
    for arm in arms:
        bins = list(section[arm]["reliability_bins"])
        summary = section[arm].get("reliability_summary") or {}
        out_arms[arm] = {
            "bins": {
                str(h): [
                    {
                        "bin_index": r.bin_index,
                        "lower": r.lower,
                        "upper": r.upper,
                        "n": r.n,
                        "burned": r.burned,
                        "mean_forecast": r.mean_forecast,
                        "observed_frequency": r.observed_frequency,
                        "wilson95": (list(wilson_interval(r.burned, r.n)) if r.occupied else None),
                    }
                    for r in bin_rows(bins, h)
                ]
                for h in leads
            },
            "commitment": {str(h): commitment(bins, h, threshold) for h in leads},
            "concentration": {str(h): concentration(bins, h) for h in leads},
            "curve_agreement": {str(h): curve_agreement(bins, h) for h in leads},
            "summary": {
                str(h): {
                    k: summary.get(str(h), {}).get(k)
                    for k in (
                        "base_rate",
                        "ece",
                        "reliability",
                        "resolution",
                        "calibration_error",
                        "calibration_error_silent_floor",
                        "calibration_n_scored",
                    )
                }
                for h in leads
            },
        }

    disagreements = [
        f"{arm}@{h}h"
        for arm in arms
        for h in leads
        if not out_arms[arm]["curve_agreement"][str(h)]["identical"]
    ]
    return {
        "kind": "reliability_page",
        "source_block": block,
        "leads": list(leads),
        "commitment_threshold": float(threshold),
        "emphasis_arm": emphasis,
        "arms": out_arms,
        "arm_order": arms,
        "provenance": {
            k: payload.get(k)
            for k in (
                "kind",
                "seed_rule",
                "stride",
                "n_members",
                "horizon_h",
                "primary_artifact",
                "control_cells_checked",
                "control_verdict",
            )
        },
        "curve_agreement_verdict": (
            "OK - every drawn curve reproduces sim.dashboard.reliability_curve exactly"
            if not disagreements
            else "DISAGREES - " + ", ".join(disagreements)
        ),
    }


def _style_for(arm: str, emphasis: str) -> dict[str, Any]:
    if emphasis and arm == emphasis:
        return dict(_EMPHASIS_STYLE)
    if emphasis and arm.startswith(emphasis):
        return dict(_EMPHASIS_ABLATION_STYLE)
    return dict(_ARM_STYLE.get(arm, _FALLBACK_STYLE))


def _panel_curve(ax: Any, report: Mapping[str, Any], lead: int) -> None:
    emphasis = str(report.get("emphasis_arm") or "")
    ax.plot([0, 1], [0, 1], color="#111827", lw=0.9, ls="--", zorder=1, label="perfect")
    for arm in report["arm_order"]:
        rows = report["arms"][arm]["bins"][str(lead)]
        pts = [r for r in rows if r["n"] > 0 and r["mean_forecast"] is not None]
        if not pts:
            continue
        st = _style_for(arm, emphasis)
        xs = [float(r["mean_forecast"]) for r in pts]
        ys = [float(r["observed_frequency"]) for r in pts]
        lo = [ys[i] - float(r["wilson95"][0]) for i, r in enumerate(pts)]
        hi = [float(r["wilson95"][1]) - ys[i] for i, r in enumerate(pts)]
        # Marker area grows with log10(n): a point resting on 9 cells is drawn
        # smaller than one resting on 1.75 million, so the count is legible on
        # the curve itself and not only on the axis below it.
        sizes = [8.0 + 26.0 * math.log10(max(1.0, float(r["n"]))) / 6.0 for r in pts]
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=st["color"],
            ls=st["ls"],
            lw=st["lw"],
            elinewidth=0.9,
            capsize=1.8,
            alpha=0.95,
            zorder=3,
            label=arm,
        )
        ax.scatter(xs, ys, s=sizes, color=st["color"], zorder=4, edgecolors="none")
    thr = float(report["commitment_threshold"])
    ax.axvspan(thr, 1.03, color="#fde68a", alpha=0.28, zorder=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("mean forecast probability in bin")
    if lead == report["leads"][0]:
        ax.set_ylabel("observed burn frequency")
    ax.set_title(f"reliability @ +{lead} h", fontsize=10)
    ax.grid(alpha=0.22, lw=0.5)


def _panel_occupancy(ax: Any, report: Mapping[str, Any], lead: int) -> None:
    emphasis = str(report.get("emphasis_arm") or "")
    for arm in report["arm_order"]:
        rows = report["arms"][arm]["bins"][str(lead)]
        st = _style_for(arm, emphasis)
        centres = [0.5 * (float(r["lower"]) + float(r["upper"])) for r in rows]
        # 0 cells cannot be drawn on a log axis. It is plotted at 0.5 - BELOW
        # the n=1 line, which is drawn - so an unused bin reads as "less than
        # one cell" rather than being silently absent.
        counts = [max(0.5, float(r["n"])) for r in rows]
        ax.plot(
            centres,
            counts,
            color=st["color"],
            ls=st["ls"],
            lw=st["lw"],
            marker="o",
            ms=2.6,
            alpha=0.95,
        )
    ax.axvspan(float(report["commitment_threshold"]), 1.03, color="#fde68a", alpha=0.28, zorder=0)
    ax.axhline(1.0, color=COL_TEXT, lw=0.7, ls="-", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.35, 4.0e6)
    ax.set_xlabel("forecast probability bin")
    if lead == report["leads"][0]:
        ax.set_ylabel("cells in bin (log)")
    ax.grid(alpha=0.22, lw=0.5, which="both")
    if emphasis and emphasis in report["arms"]:
        conc = report["arms"][emphasis]["concentration"][str(lead)]
        share = conc["lowest_bin_share"]
        if share is not None:
            ax.set_title(
                f"{emphasis}: {100.0 * share:.4f}% of {conc['n_cells_total']:,} cells "
                f"in [{conc['lowest_bin'][0]:.1f}, {conc['lowest_bin'][1]:.1f})",
                fontsize=8.5,
                color=COL_WARN,
            )


def commitment_table(report: Mapping[str, Any]) -> str:
    """The commitment table as plain text, so the figure and the artifact agree by eye."""
    thr = float(report["commitment_threshold"])
    lines = [
        f"CELLS THE ARM PUTS AT p >= {thr:.2f}, AND HOW MANY OF THEM BURNED",
        "",
        f"{'arm':<22}"
        + "".join(f"{'  +' + str(h) + ' h: n / burned / obs':>30}" for h in report["leads"]),
    ]
    for arm in report["arm_order"]:
        cells = []
        for h in report["leads"]:
            c = report["arms"][arm]["commitment"][str(h)]
            if c["n_cells"] == 0:
                cells.append(f"{'0 /      0 /      n/a':>30}")
            else:
                obs = c["observed_frequency"]
                cells.append(f"{c['n_cells']:>10,} / {c['n_burned']:>6,} / {obs:>8.4f}")
        lines.append(f"{arm:<22}" + "".join(cells))
    return "\n".join(lines)


def _panel_table(ax: Any, report: Mapping[str, Any]) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        commitment_table(report),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.2,
        color=COL_TEXT,
    )
    emphasis = str(report.get("emphasis_arm") or "")
    lead = int(report["leads"][-1])
    if emphasis and emphasis in report["arms"]:
        c = report["arms"][emphasis]["commitment"][str(lead)]
        ub = c["exact_one_sided_upper95"]
        if c["n_cells"] > 0 and c["n_burned"] == 0 and ub is not None:
            ax.text(
                0.0,
                0.02,
                f"{emphasis} at +{lead} h commits to p >= {c['threshold']:.2f} "
                f"{c['n_cells']:,} times and is wrong every one of them.\n"
                f"0 of {c['n_cells']:,} bounds the true rate below {100.0 * ub:.2f}% "
                "(exact one-sided 95%), so this is not a small sample looking bad.",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                family="monospace",
                fontsize=8.6,
                color=COL_WARN,
            )


def _panel_aggregate(ax: Any, report: Mapping[str, Any], lead: int) -> None:
    """The aggregate that disagrees with the diagram, drawn beside it rather than omitted.

    ECE is an n-weighted average over bins. When one bin holds 99% of the cells,
    ECE is very nearly a statement about that bin alone -- so an arm can win ECE
    and lose the diagram, and this panel is where those two facts are put in the
    same frame instead of in two documents.
    """
    emphasis = str(report.get("emphasis_arm") or "")
    arms = list(report["arm_order"])
    vals = [float(report["arms"][a]["summary"][str(lead)].get("ece") or 0.0) for a in arms]
    floors = [
        report["arms"][a]["summary"][str(lead)].get("calibration_error_silent_floor") for a in arms
    ]
    ypos = np.arange(len(arms), dtype=float)
    ax.barh(
        ypos,
        vals,
        color=[_style_for(a, emphasis)["color"] for a in arms],
        height=0.62,
        zorder=3,
    )
    floor = next((float(f) for f in floors if f is not None), None)
    if floor is not None:
        ax.axvline(
            floor,
            color=COL_WARN,
            lw=1.2,
            ls="--",
            zorder=4,
            label=f"silent floor {floor:.5f}",
        )
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels(arms, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"ECE @ +{lead} h (lower is better)", fontsize=8.5)
    ax.set_title("the aggregate, on the same cells", fontsize=9)
    ax.grid(alpha=0.22, lw=0.5, axis="x")


def render(report: Mapping[str, Any], out: str | Path) -> Path:
    """Draw the page. One PNG, three rows, nothing that is not in ``report``."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    leads = [int(h) for h in report["leads"]]

    fig = plt.figure(figsize=(16.0, 12.2))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.30, 0.78, 0.80],
        hspace=0.36,
        wspace=0.24,
        left=0.055,
        right=0.985,
        top=0.870,
        bottom=0.055,
    )
    for i, lead in enumerate(leads):
        _panel_curve(fig.add_subplot(gs[0, i]), report, lead)
    for i, lead in enumerate(leads):
        _panel_occupancy(fig.add_subplot(gs[1, i]), report, lead)
    _panel_table(fig.add_subplot(gs[2, 0:2]), report)
    _panel_aggregate(fig.add_subplot(gs[2, 2]), report, leads[-1])

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        fontsize=8.5,
        frameon=False,
        ncol=len(labels),
    )

    prov = report.get("provenance") or {}
    fig.suptitle(
        "RELIABILITY WITH ITS OWN BIN COUNTS - "
        f"{report['source_block']}, held-out blocks\n"
        f"stride={prov.get('stride')}  members={prov.get('n_members')}  "
        f"horizon={prov.get('horizon_h')} h  |  "
        "top row: the curve.  middle row: how many cells each point rests on.  "
        "bottom row: what the arm commits to, and the aggregate that disagrees.",
        fontsize=11.5,
        y=0.988,
        ha="center",
    )
    # The relayed verdict is TRUNCATED with an explicit marker rather than left
    # to run off the right edge of the canvas, which is how a caption silently
    # loses its second half.
    relayed = str(prov.get("control_verdict") or "n/a")
    if len(relayed) > 60:
        relayed = relayed[:57] + "..."
    stamp(
        fig,
        "C6 reliability_bins only - nothing recomputed, no tensor, no model. "
        f"error bars 95% Wilson. curve control: {report.get('curve_agreement_verdict')}. "
        f"source artifact control: {relayed}",
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("artifact", help="a run record carrying C6 reliability_bins per arm")
    ap.add_argument(
        "--out", required=True, help="destination .png; the JSON sidecar sits beside it"
    )
    ap.add_argument("--json", dest="json_out", default="")
    ap.add_argument("--block", default="pooled_growth_windows")
    ap.add_argument("--emphasis", default="")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    add_logging_arguments(ap)
    args = ap.parse_args(argv)
    configure_from_args(args)

    payload = _open_maybe_gzip(args.artifact)
    report = build_page(
        payload,
        block=args.block,
        threshold=float(args.threshold),
        emphasis=str(args.emphasis),
    )
    png = render(report, args.out)
    json_out = Path(args.json_out) if args.json_out else png.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=1, sort_keys=True))

    print(commitment_table(report))
    print("")
    print(f"curve control: {report['curve_agreement_verdict']}")
    print(f"figure: {png}")
    print(f"numbers: {json_out}")
    return 0 if report["curve_agreement_verdict"].startswith("OK") else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
