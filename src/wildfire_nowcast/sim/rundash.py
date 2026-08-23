"""Per-run gate dashboard - the standard readout after a modelling gate attempt.

    python -m wildfire_nowcast.sim.rundash --run runs/baselines-20260808-064738 \
        --decomposition reports/figures/iou_decomposition.json \
        --out reports/figures/gate_dashboard.png

Companion to :mod:`wildfire_nowcast.sim.dashboard`, which renders a single C6
``evaluate()`` dict. This one renders a whole GATE ATTEMPT: many models, many
held-out blocks, three horizons. It reads ``runs/<id>/results.json`` and nothing
else (plus, optionally, my own IoU decomposition). That file is C6 output
aggregated by ``eval/reporting``; no model internal is opened.

**Two labelling rules are enforced here, in code, because a figure outlives its
caption and a screenshot outlives both.**

1. ``dispersion_ratio`` is drawn ONLY with a DIAGNOSTIC-ONLY badge. C6.1 (ADR-011)
   is unambiguous: it scores a COLLAPSED ensemble at 1.000 and a healthy one at
   1.051, so it is anti-correlated with the thing it appears to measure, and
   **G3 is adjudicated on ``area_dispersion_ratio``**. An unlabelled dispersion
   chart is a reassuring picture of the exact failure it cannot see. If the key
   is present it is plotted next to ``area_dispersion_ratio``, never alone.

2. ``band_best_member_iou`` is drawn ONLY with its null floor. Measured on this
   project's own numbers, a silent ensemble scores 0.200-0.262 on the growth
   stratum and **0.829 on all windows** - up to ~90% of the value the gate is
   adjudicated on - because the metric averages over leads and an empty-vs-empty
   lead scores 1.0. Where ``sim.replay``'s silence/shape decomposition is
   supplied, the bar is stacked so the two parts are separately visible.

[S3, P2/G3] BADGING IS NO LONGER MINE TO REMEMBER
-------------------------------------------------
Both rules above were hand-written captions, and a hand-written caption is a
promise about a file I might not re-read. Every key this module plots now goes
through :mod:`wildfire_nowcast.sim.quarantine`, which classifies it against
``common.null_check.C6_METRICS`` - infra's registry, the maintainer's ruling -
and supplies the badge text from ``make null-check``'s MEASURED output. A key the
registry does not know RAISES rather than rendering plain.

That immediately caught one of my own: the old panel (4) plotted
``band_ece_by_horizon`` under the title "calibration error over lead time" with
no badge at all. ECE is quarantined by ADR-020 (climatology beats genuine skill
on it), and it is NOT G3's criterion. **G3's calibration half is
``calibration_error`` on the ``growth_band`` mask** (ADR-020,
``common.calibration.GATE_CRITERION_KEY``) and G3's dispersion half is
``area_dispersion_ratio`` in [0.8, 1.2] (ADR-011). Both are drawn here; the
retired ``reliability`` and ``ece`` families are drawn beside them, badged, so a
reader can see that the demoted numbers move differently - which is the argument
for the demotion, not a decoration.

``dispersion_ratio``'s badge also changed, and the change is the point. The old
text quoted ADR-011's "collapsed 1.000, healthy 1.051". Today's harness measures
**collapsed 1.1921 vs healthy 1.1903 - a paired advantage of -0.0019 +- 0.0050
over seeds, inside its own seed noise.** Same verdict, different mechanism: the
metric does not prefer collapse, it cannot SEE it. A hardcoded badge decays into
a false claim about a true conclusion, so the badge is now read from the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wildfire_nowcast.sim.quarantine import (  # noqa: E402
    G3_KEYS,
    GATE,
    QUARANTINED,
    badge,
    classify,
    load_null_check,
)
from wildfire_nowcast.sim.style import COL_TEXT, COL_WARN, stamp  # noqa: E402

__all__ = [
    "DEFAULT_NULL_CHECK",
    "G3_DISPERSION_BAND",
    "G3_CALIBRATION_BAR",
    "CALIBRATION_HEADLINE_KEY",
    "load_run",
    "model_order",
    "g3_readiness",
    "render_run_dashboard",
    "main",
]

#: Where ``make null-check --json`` is written by convention. Optional.
DEFAULT_NULL_CHECK = "reports/figures/null_check_report.json"

#: G3's dispersion half (ADR-011 / C6.1), on ``area_dispersion_ratio``.
G3_DISPERSION_BAND = (0.8, 1.2)

#: G3's calibration half (ADR-020), in probability POINTS on the growth band.
G3_CALIBRATION_BAR = 0.10

#: The headline key a run must carry for the calibration half to be plottable.
CALIBRATION_HEADLINE_KEY = "band_calibration_error_by_horizon"

HORIZONS = ("1", "2", "3")


def g3_readiness(payload: dict[str, Any]) -> dict[str, Any]:
    """Can this run artifact adjudicate G3 at all? Answered from its own keys.

    G2 was nearly adjudicated on a table that did not contain its own gate
    criterion, because ``eval/baseline_run._headline`` is a curated allow-list
    (infra's blocker, ADR-020 §6). The same allow-list currently carries no
    ``calibration_*`` key, so this check is not hypothetical - it is that defect,
    one gate later, and a dashboard that silently drew an empty panel would hide
    it exactly as the last one did.

    **An empty payload reads NOT adjudicable, and it used to read adjudicable.**
    ``present`` was seeded with ``setdefault`` INSIDE the model loop, so a
    payload carrying no models left it ``{}``, ``missing`` was ``[]`` and
    ``adjudicable`` came back True: the readiness check pronounced a run ready
    to decide G3 without having found a single number in it. The keys are now
    seeded before the loop, so absence of evidence is reported as both criteria
    missing, and ``n_models_examined`` publishes the denominator beside the
    verdict.

    This one returns a verdict rather than raising, unlike
    :func:`wildfire_nowcast.sim.playthrough.degeneracy_verdict`. The difference
    is what the function claims: "this artifact cannot adjudicate G3" is the
    TRUE and useful answer for an empty payload, and it is the answer this
    function exists to give, so the dashboard can draw its NOT-ADJUDICABLE
    banner instead of crashing. A degeneracy verdict over zero members has no
    true answer at all, which is why that one refuses.
    """
    pooled = payload.get("pooled_heldout") or {}
    present: dict[str, bool] = {"area_dispersion_ratio": False, CALIBRATION_HEADLINE_KEY: False}
    for stratum in ("growth_windows", "all_windows"):
        for m, node in pooled.items():
            block = (node or {}).get(stratum) or {}
            if block.get("area_dispersion_ratio") is not None:
                present["area_dispersion_ratio"] = True
            if block.get(CALIBRATION_HEADLINE_KEY) or block.get("band_calibration_error"):
                present[CALIBRATION_HEADLINE_KEY] = True
            del m
    missing = [k for k, v in present.items() if not v]
    return {
        "dispersion_criterion": G3_KEYS["dispersion"],
        "calibration_criterion": G3_KEYS["calibration"],
        "calibration_mask": G3_KEYS["calibration_mask"],
        "n_models_examined": len(pooled),
        "present": present,
        "missing": missing,
        "adjudicable": not missing,
    }


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Read one gate attempt's ``results.json``. No coercion, no defaults."""
    p = Path(run_dir)
    if p.is_dir():
        p = p / "results.json"
    return json.loads(p.read_text())


def model_order(pooled: dict[str, Any]) -> list[str]:
    """Candidates first, then opponents, then the null model - reading order."""
    names = list(pooled)
    kernels = [m for m in names if m.startswith("kernel") and m != "kernel_init"]
    init = [m for m in names if m == "kernel_init"]
    ellipses = [m for m in names if m.startswith("ellipse")]
    rest = [m for m in names if m not in kernels + init + ellipses + ["persistence"]]
    return kernels + init + ellipses + rest + [m for m in names if m == "persistence"]


def _series(pooled: dict[str, Any], model: str, key: str, stratum: str) -> list[float]:
    node = (pooled.get(model) or {}).get(stratum) or {}
    by_h = node.get(key) or {}
    return [float(by_h.get(h, np.nan)) for h in HORIZONS]


def _scalar(pooled: dict[str, Any], model: str, key: str, stratum: str) -> float:
    node = (pooled.get(model) or {}).get(stratum) or {}
    v = node.get(key)
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _colour(model: str) -> str:
    if model == "persistence":
        return "#6b7280"
    if model == "kernel_init":
        return "#a78bfa"
    if model.startswith("kernel"):
        return "#0f766e"
    return "#b45309"


def _badge_box(ax, text: str, *, y: float = -0.22, colour: str = COL_WARN) -> None:
    """Draw a badge in a reserved strip BELOW the axes. One place, one style.

    Second defect found by looking at the render: badges drawn INSIDE the axes at
    the top-left overprinted the legends of panels (2) and (3), so the quarantine
    text and the model names were mutually illegible - a badge nobody can read is
    a badge that is not there, which is the whole failure mode this module exists
    to prevent. A reserved strip cannot collide with data or a legend by
    construction; the previous version could, and did, the moment the text grew.
    """
    if not text:
        return
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        fontsize=5.8,
        color=colour,
        va="top",
        ha="left",
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "alpha": 0.95,
            "edgecolor": colour,
            "lw": 0.8,
            "boxstyle": "round,pad=0.3",
        },
    )


def _wrap(text: str, width: int = 74, max_lines: int = 6) -> str:
    import textwrap  # noqa: PLC0415

    lines = textwrap.wrap(text, width=width)[:max_lines]
    return "\n".join(lines)


def render_run_dashboard(
    payload: dict[str, Any],
    out: str | Path,
    *,
    stratum: str = "growth_windows",
    decomposition: dict[str, Any] | None = None,
    null_check: dict[str, str] | None = None,
) -> Path:
    """Eight panels. Every plotted key is classified by ``sim.quarantine`` first."""
    pooled = payload.get("pooled_heldout") or {}
    order = model_order(pooled)
    evidence = null_check or {}
    drawn: dict[str, str] = {}

    fig = plt.figure(figsize=(19.6, 14.4))
    # hspace is large on purpose: every quarantined panel reserves a strip under
    # its axes for its badge (see `_badge_box`).
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.92],
        hspace=0.82,
        wspace=0.26,
        left=0.05,
        right=0.987,
        top=0.885,
        bottom=0.065,
    )

    # -- (1) band Brier over lead time ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    drawn["band_brier_by_horizon"] = ""
    for m in order:
        ys = _series(pooled, m, "band_brier_by_horizon", stratum)
        ax.plot(
            [1, 2, 3],
            ys,
            marker="o",
            ms=4.5,
            lw=1.6,
            color=_colour(m),
            label=m,
            ls="--" if m in ("persistence", "kernel_init") else "-",
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("band Brier  (lower is better)")
    ax.set_title("(1) band Brier over lead time", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.2, frameon=False, ncol=2)

    # -- (2) band best-member IoU, WITH its null floor ---------------------
    ax = fig.add_subplot(gs[0, 1])
    floor = _series(pooled, "persistence", "band_best_member_iou_by_horizon", stratum)
    for m in order:
        if m == "persistence":
            continue
        ys = _series(pooled, m, "band_best_member_iou_by_horizon", stratum)
        ax.plot(
            [1, 2, 3],
            ys,
            marker="o",
            ms=4.5,
            lw=1.6,
            color=_colour(m),
            label=m,
            ls="--" if m == "kernel_init" else "-",
        )
    ax.plot([1, 2, 3], floor, lw=2.4, color=COL_WARN, ls=":", label="NULL FLOOR (persistence)")
    ax.fill_between([1, 2, 3], 0, floor, color=COL_WARN, alpha=0.12)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("band best-member IoU  (higher is better)")
    ax.set_title("(2) band best-member IoU — shaded = available for SILENCE", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(0, None)
    ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="upper right")
    iou_status = classify("band_best_member_iou_by_horizon", evidence=evidence)
    iou_badge = _wrap(badge(iou_status), 88, 6)
    if np.isfinite(floor).any():
        below = [
            m
            for m in order
            if m != "persistence"
            and np.nanmean(_series(pooled, m, "band_best_member_iou_by_horizon", stratum))
            < np.nanmean(floor)
        ]
        msg = (
            f"a member that ignites NOTHING scores {np.nanmean(floor):.3f} here\n"
            "(a lead where truth did not grow is empty-vs-empty = IoU 1.0)"
        )
        if below:
            msg += f"\nBELOW THE NULL FLOOR: {', '.join(below)}"
        ax.text(
            0.03,
            0.04,
            msg,
            transform=ax.transAxes,
            fontsize=6.4,
            color=COL_WARN,
            va="bottom",
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": COL_WARN,
                "lw": 0.7,
                "boxstyle": "round,pad=0.28",
            },
        )
    _badge_box(ax, iou_badge)
    drawn["band_best_member_iou_by_horizon"] = iou_badge

    # -- (3) G3 HALF ONE: dispersion. The blind one beside the adjudicating one.
    ax = fig.add_subplot(gs[0, 2])
    models = [m for m in order if m != "persistence"] + ["persistence"]
    xs = np.arange(len(models))
    dr = [_scalar(pooled, m, "dispersion_ratio", stratum) for m in models]
    adr = [_scalar(pooled, m, "area_dispersion_ratio", stratum) for m in models]
    lo, hi = G3_DISPERSION_BAND
    ax.axhspan(lo, hi, color="#0f766e", alpha=0.10, zorder=0)
    ax.bar(
        xs - 0.2,
        dr,
        width=0.38,
        color="#cbd5e1",
        edgecolor="#64748b",
        lw=0.6,
        label="dispersion_ratio — QUARANTINED (C6.1)",
    )
    ax.bar(
        xs + 0.2,
        adr,
        width=0.38,
        color="#0f766e",
        edgecolor="#134e4a",
        lw=0.6,
        label=f"area_dispersion_ratio — ADJUDICATES G3, bar [{lo}, {hi}]",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=38, ha="right", fontsize=6.4)
    ax.axhline(1.0, color=COL_TEXT, lw=0.8, ls="--")
    ax.set_ylabel("ratio")
    ax.set_title("(3) G3 HALF ONE — ensemble dispersion. Read the DARK bars.", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.set_ylim(0, max(1.42, float(np.nanmax(dr + adr + [1.3])) * 1.08))
    ax.legend(fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.78))
    disp_badge = _wrap(badge(classify("dispersion_ratio", evidence=evidence)), 88, 5)
    _badge_box(ax, disp_badge)
    drawn["dispersion_ratio"] = disp_badge
    drawn["area_dispersion_ratio"] = ""

    # -- (4) G3 HALF TWO: the calibration criterion, or a loud absence -----
    ax = fig.add_subplot(gs[1, 0])
    cal_series = {m: _series(pooled, m, CALIBRATION_HEADLINE_KEY, stratum) for m in order}
    have_cal = any(np.isfinite(v).any() for v in cal_series.values())
    if have_cal:
        for m in order:
            ax.plot(
                [1, 2, 3],
                cal_series[m],
                marker="s",
                ms=4.2,
                lw=1.6,
                color=_colour(m),
                label=m,
                ls="--" if m in ("persistence", "kernel_init") else "-",
            )
        ax.axhline(
            G3_CALIBRATION_BAR,
            color=COL_WARN,
            lw=1.6,
            ls="--",
            label=f"G3 bar = {G3_CALIBRATION_BAR:.2f} (10 pts)",
        )
        floor_cal = _series(
            pooled, "persistence", "band_calibration_error_silent_floor_by_horizon", stratum
        )
        if np.isfinite(floor_cal).any():
            ax.plot(
                [1, 2, 3],
                floor_cal,
                lw=2.0,
                color="#6b7280",
                ls=":",
                label="silent floor = the base rate exactly",
            )
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(f"{G3_KEYS['calibration']}  (probability points, lower better)")
        ax.set_title(
            f"(4) G3 HALF TWO — {G3_KEYS['calibration']} on the {G3_KEYS['calibration_mask']} mask",
            fontsize=10,
        )
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6.0, frameon=False, ncol=2)
        drawn[CALIBRATION_HEADLINE_KEY] = ""
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "G3'S CALIBRATION CRITERION IS NOT IN THIS RUN ARTIFACT\n\n"
            f"`{G3_KEYS['calibration']}` (ADR-020,\n"
            "common.calibration.GATE_CRITERION_KEY) is emitted by C6's\n"
            "`aggregate()` inside by_mask[...].reliability_summary[<lead>],\n"
            "but `eval/baseline_run._headline` is a CURATED ALLOW-LIST and\n"
            "carries no calibration_* key. So a G3 run would print a table\n"
            "WITHOUT its own gate criterion — the exact defect infra\n"
            "caught for C6.4 one gate ago (ADR-020 (6)).\n\n"
            "Verified by CALLING _headline, not by reading it.\n"
            "Raised against eval/baseline_run. Nothing here is a G3 verdict.",
            ha="center",
            va="center",
            fontsize=8.0,
            color=COL_WARN,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": COL_WARN,
                "lw": 1.4,
                "boxstyle": "round,pad=0.5",
            },
        )

    # -- (4b) the RETIRED calibration family, badged, for contrast ---------
    ax = fig.add_subplot(gs[1, 1])
    for m in order:
        ys = _series(pooled, m, "band_ece_by_horizon", stratum)
        ax.plot(
            [1, 2, 3],
            ys,
            marker="s",
            ms=4.0,
            lw=1.5,
            color=_colour(m),
            label=m,
            ls="--" if m in ("persistence", "kernel_init") else "-",
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("band ECE  (lower is better)")
    ax.set_title("(4b) RETIRED — band ECE. NOT G3's criterion.", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ece_badge = _wrap(badge(classify("band_ece_by_horizon", evidence=evidence)), 88, 5)
    _badge_box(ax, ece_badge)
    drawn["band_ece_by_horizon"] = ece_badge

    # -- (4c) reliability (REL), the metric G3's old bar rested on ---------
    ax = fig.add_subplot(gs[1, 2])
    for m in order:
        ys = _series(pooled, m, "band_reliability_by_horizon", stratum)
        ax.plot(
            [1, 2, 3],
            ys,
            marker="d",
            ms=4.0,
            lw=1.5,
            color=_colour(m),
            label=m,
            ls="--" if m in ("persistence", "kernel_init") else "-",
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("band reliability (REL)")
    ax.set_title("(4c) DEMOTED — REL, G3's ORIGINAL bar. Do not quote.", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    rel_badge = _wrap(badge(classify("band_reliability_by_horizon", evidence=evidence)), 88, 5)
    _badge_box(ax, rel_badge)
    drawn["band_reliability_by_horizon"] = rel_badge

    # -- (5) resolution: does the forecast SAY anything --------------------
    ax = fig.add_subplot(gs[2, 1])
    drawn["band_resolution_by_horizon"] = ""
    for m in order:
        ys = _series(pooled, m, "band_resolution_by_horizon", stratum)
        ax.plot(
            [1, 2, 3],
            ys,
            marker="^",
            ms=4.4,
            lw=1.5,
            color=_colour(m),
            label=m,
            ls="--" if m in ("persistence", "kernel_init") else "-",
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("band resolution  (higher is better)")
    ax.set_title("(5) resolution — persistence sits at EXACTLY 0", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    # -- (6) silence / shape split, when supplied --------------------------
    # DEFECT FOUND BY LOOKING AT THE RENDER, not by a test: this shared gs[1, 2]
    # with panel (4c) and overprinted it, so two panels' axes, ticks and titles
    # were drawn on top of each other and BOTH were unreadable. matplotlib does
    # not complain about a reused grid cell. Same family as the argparse
    # `--out`/`--outdir` prefix match and the NaN verdict ladder: the failure is
    # a plausible-looking figure, not an exception.
    ax = fig.add_subplot(gs[2, 2])
    if decomposition:
        per_fire = decomposition.get("per_fire") or {}
        names = sorted({m for f in per_fire.values() for m in f})
        sil = [
            float(
                np.nanmean([per_fire[f].get(m, {}).get("silence_term", np.nan) for f in per_fire])
            )
            for m in names
        ]
        shp = [
            float(np.nanmean([per_fire[f].get(m, {}).get("shape_term", np.nan) for f in per_fire]))
            for m in names
        ]
        xs = np.arange(len(names))
        ax.bar(xs, sil, color="#cbd5e1", edgecolor="#64748b", lw=0.6, label="silence term")
        ax.bar(
            xs,
            shp,
            bottom=sil,
            color="#0f766e",
            edgecolor="#134e4a",
            lw=0.6,
            label="shape term (mode capture)",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=38, ha="right", fontsize=6.4)
        ax.set_ylabel("band best-member IoU")
        ax.set_title("(6) WHERE the IoU comes from", fontsize=10)
        ax.legend(fontsize=6.4, frameon=False)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "silence/shape decomposition not supplied\n(python -m wildfire_nowcast.sim.replay ...)",
            ha="center",
            va="center",
            fontsize=8,
            color=COL_TEXT,
        )

    # -- (7) G3 READINESS + the badge audit, in words ----------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    ready = g3_readiness(payload)
    lines = [
        "G3 READINESS OF THIS ARTIFACT",
        "",
        f"dispersion criterion : {ready['dispersion_criterion']}"
        f"   {'PRESENT' if ready['present'].get('area_dispersion_ratio') else 'ABSENT'}",
        f"calibration criterion: {ready['calibration_criterion']}"
        f" on '{ready['calibration_mask']}'"
        f"   {'PRESENT' if ready['present'].get(CALIBRATION_HEADLINE_KEY) else 'ABSENT'}",
        "",
        "ADJUDICABLE: " + ("YES" if ready["adjudicable"] else "NO — see panel (4)"),
        "",
        "QUARANTINED and drawn on this figure (all badged):",
    ]
    q = [k for k in drawn if classify(k, evidence=evidence).state == QUARANTINED]
    lines += [f"  - {k}" for k in q] or ["  (none)"]
    lines += ["", "GATE-ELIGIBLE and drawn:"]
    g = [k for k in drawn if classify(k, evidence=evidence).state == GATE]
    lines += [f"  - {k}" for k in g] or ["  (none)"]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7.2,
        va="top",
        ha="left",
        family="monospace",
        color=COL_TEXT if ready["adjudicable"] else COL_WARN,
    )

    # -- banner ------------------------------------------------------------
    split = payload.get("split_after") or payload.get("split_before") or {}
    fp = split.get("fingerprint", "UNKNOWN")
    n_blocks = split.get("n_heldout_blocks", "?")
    # c8_split_match is a per-model DICT. Interpolating it into the title dumped
    # ~600 characters of JSON across the figure and off both edges. Reduce it to
    # the one bit that matters, and say how many models it covers.
    c8_raw = payload.get("c8_split_match")
    if isinstance(c8_raw, dict) and c8_raw:
        ok = sum(1 for v in c8_raw.values() if isinstance(v, dict) and v.get("match"))
        c8 = f"{ok}/{len(c8_raw)} models match" + ("" if ok == len(c8_raw) else "  ** HARD FAIL **")
    else:
        c8 = str(c8_raw)
    bits = [
        f"split_fingerprint {fp}",
        f"{n_blocks} held-out blocks (C6.3 requires >= 4)",
        f"C8: {c8}",
        f"stratum: {stratum}",
    ]
    fig.suptitle(
        "GATE-ATTEMPT DASHBOARD — every number here is C6 output; no model internal is read\n"
        + "   |   ".join(str(b) for b in bits),
        fontsize=10.5,
    )
    warn = [n for n in (payload.get("interpretation") or []) if "DEGENERATE" in n or "VOID" in n]
    if warn:
        # Truncate hard. These strings run to ~300 chars and ran off the canvas.
        short = [w.split(" — ")[0][:96] + (" …" if len(w) > 96 else "") for w in warn[:3]]
        fig.text(
            0.05,
            0.030,
            "C6.2 VALIDITY (a degenerate baseline VOIDS its gate):  " + "  |  ".join(short),
            fontsize=6.2,
            color=COL_WARN,
            va="top",
        )
    stamp(
        fig,
        "G3 = area_dispersion_ratio in [0.8,1.2] (ADR-011) AND calibration_error on the "
        "growth_band (ADR-020). dispersion_ratio, ECE and REL are QUARANTINED and badged; "
        "badges are looked up from common.null_check.C6_METRICS, not typed here.",
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wildfire_nowcast.sim.rundash", allow_abbrev=False)
    ap.add_argument("--run", required=True, help="runs/<id> or a results.json path")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--stratum", default="growth_windows", choices=("growth_windows", "all_windows")
    )
    ap.add_argument(
        "--decomposition",
        default=None,
        help="reports/figures/iou_decomposition.json from sim.replay",
    )
    ap.add_argument(
        "--null-check",
        default=DEFAULT_NULL_CHECK,
        help="make null-check --json output; supplies MEASURED badge evidence",
    )
    args = ap.parse_args(argv)

    payload = load_run(args.run)
    dec = json.loads(Path(args.decomposition).read_text()) if args.decomposition else None
    evidence = load_null_check(args.null_check)
    fig = render_run_dashboard(
        payload, args.out, stratum=args.stratum, decomposition=dec, null_check=evidence
    )

    pooled = payload.get("pooled_heldout") or {}
    floor = _series(pooled, "persistence", "band_best_member_iou_by_horizon", args.stratum)
    ready = g3_readiness(payload)
    print(f"[rundash] {fig}")
    print(
        f"[rundash] badge evidence: {len(evidence)} metrics from "
        f"{args.null_check if evidence else 'REGISTRY ONLY (null-check json absent)'}"
    )
    print(
        f"[rundash] G3 adjudicable from this artifact: {ready['adjudicable']}"
        + ("" if ready["adjudicable"] else f"  MISSING {ready['missing']}")
    )
    print(
        f"[rundash] NULL FLOOR (persistence band best-member IoU) by horizon: "
        f"{', '.join(f'{v:.3f}' for v in floor)}"
    )
    lo, hi = G3_DISPERSION_BAND
    for m in model_order(pooled):
        dr = _scalar(pooled, m, "dispersion_ratio", args.stratum)
        adr = _scalar(pooled, m, "area_dispersion_ratio", args.stratum)
        cal = _series(pooled, m, CALIBRATION_HEADLINE_KEY, args.stratum)
        iou = _series(pooled, m, "band_best_member_iou_by_horizon", args.stratum)
        in_band = (
            ""
            if not np.isfinite(adr)
            else (" [G3 dispersion IN BAND]" if lo <= adr <= hi else " [G3 dispersion OUT OF BAND]")
        )
        print(
            f"    {m:26s} IoU(quarantined) {', '.join(f'{v:.3f}' for v in iou)}"
            f"   dispersion_ratio {dr:.3f} (QUARANTINED)"
            f"   area_dispersion_ratio {adr:.3f} (G3){in_band}"
        )
        print(
            f"    {'':26s} calibration_error(G3) "
            f"{', '.join('n/a' if not np.isfinite(v) else f'{v:.4f}' for v in cal)}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
