"""Per-run diagnostics dashboard. Reads C6 ``evaluate()`` output JSON and NOTHING else.

    python -m wildfire_nowcast.sim.dashboard runs/*/metrics.json \
        --out reports/figures/dashboard.png

Panels: reliability at 1/2/3 h, error-and-spread over lead time, and
member-vs-truth growth small multiples.

Four things here are deliberate, and three of them are refusals.

**1. C6 JSON only - no model, no tensor, no samples.** This module imports
nothing from ``wildfire_nowcast.model``. If a quantity is not in the C6 dict it
does not appear on the figure; it is not recomputed from a side channel. That is
what makes the same dashboard fair for the learned kernel, for persistence, for
the ellipse and (at G5) for ELMFIRE: every one of them is being read through the
identical aperture. The moment a dashboard reaches past C6 for "just one more
number", the comparison stops being apples-to-apples and nobody can see it in
the picture afterwards.

**2. C6's own ``notes`` are printed on the figure, verbatim.** C6 currently
emits two, and both are load-bearing warnings about the very numbers this
dashboard plots - that the headline mask is dominated by far-field cells no
model was ever uncertain about, and that ``dispersion_ratio`` on a binary field
is algebraically a calibration statistic and *cannot detect ensemble collapse*.
A dashboard that plotted ``dispersion_ratio`` as "the" spread metric without
that caveat would be actively misleading, and a figure outlives its caption.

**3. Both masks are drawn, never just the headline.** C6 reports ``domain`` and
``growth_band``; the growth band is where a G2/G5 verdict actually lives. They
are drawn as a pair so the gap between them is visible rather than a footnote.

**4. Reportability is stamped, not assumed.** ``eval.reporting.reporting_status``
answers whether the norm stats behind these numbers satisfy C3.3
(``n_train_blocks >= 2``). If they do not, the figure gets a SMOKE TEST banner,
because a screenshot of a dashboard is exactly how a plumbing-only number gets
quoted in a gate. (This reads a JSON status, not model internals.)

**Not here: raster member-vs-truth small multiples.** Those need the member
rasters, which C6 does not emit and - per the rule above - this module will not
reach for. They live in :mod:`wildfire_nowcast.sim.ensemble`, which receives
``samples`` legitimately as the return value of a C5 ``predict()``. What C6
*does* expose is ``diagnostics.member_growth_cells`` against
``diagnostics.truth_growth_cells``, which is the same comparison reduced to the
quantity a nowcast is judged on, and that is what panel 3 draws.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wildfire_nowcast.sim.style import COL_MEMBER, COL_TRUTH, stamp  # noqa: E402

__all__ = [
    "C6Run",
    "load_c6",
    "reliability_curve",
    "render_dashboard",
    "main",
]

#: C6 keys that INTERFACES C6 requires. Absence is a contract violation, not a
#: rendering problem, so it is reported as such rather than silently skipped.
C6_REQUIRED = (
    "brier_1h",
    "brier_2h",
    "brier_3h",
    "reliability_bins",
    "arrival_crps",
    "dispersion_ratio",
    "best_member_iou",
)

_MASKS = ("domain", "growth_band")
_MASK_STYLE = {
    "domain": {"color": "#6b7280", "label": "domain"},
    "growth_band": {"color": "#b91c1c", "label": "growth band"},
}


@dataclass(frozen=True)
class C6Run:
    """One C6 ``evaluate()`` dict plus where it came from.

    ``missing`` records required C6 keys that were absent. It is carried onto
    the figure rather than raising: a dashboard whose job is to expose problems
    should be able to render a run that has one.
    """

    label: str
    source: Path
    payload: dict[str, Any]
    missing: tuple[str, ...]

    @property
    def masks(self) -> dict[str, Any]:
        by = self.payload.get("by_mask")
        return by if isinstance(by, dict) else {}

    def mask(self, name: str) -> dict[str, Any] | None:
        m = self.masks.get(name)
        return m if isinstance(m, dict) else None

    @property
    def notes(self) -> list[str]:
        return [str(n) for n in self.payload.get("notes", []) or []]


def load_c6(path: str | Path, *, label: str | None = None) -> list[C6Run]:
    """Load one file of C6 output. Accepts a bare dict or ``{"results": [...]}``.

    Nothing is normalised, coerced or defaulted beyond that: the point of this
    module is to display what C6 said.
    """
    p = Path(path)
    raw = json.loads(p.read_text())
    payloads = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    if isinstance(payloads, dict):
        payloads = [payloads]

    runs: list[C6Run] = []
    for i, payload in enumerate(payloads):
        meta = payload.get("meta") or {}
        name = label or meta.get("model") or meta.get("name") or p.stem
        if len(payloads) > 1:
            name = f"{name}[{i}]"
        missing = tuple(k for k in C6_REQUIRED if k not in payload)
        runs.append(C6Run(str(name), p, payload, missing))
    return runs


def reliability_curve(bins: list[dict[str, Any]], lead_h: int) -> dict[str, np.ndarray]:
    """Extract one lead's reliability curve from C6 ``reliability_bins``.

    C6 emits one flat list tagged with ``lead_h``. Empty bins carry
    ``n == 0`` and ``mean_forecast is None``; they are DROPPED from the curve
    and their absence is visible in the accompanying histogram. Plotting an
    empty bin at ``(0, 0)`` would draw a perfectly-calibrated-looking point
    supported by no data at all, which is the standard way a reliability diagram
    lies.
    """
    xs: list[float] = []
    ys: list[float] = []
    ns: list[float] = []
    for b in bins:
        if int(b.get("lead_h", -1)) != lead_h:
            continue
        n = float(b.get("n", 0) or 0)
        mf, of = b.get("mean_forecast"), b.get("observed_frequency")
        if n <= 0 or mf is None or of is None:
            continue
        xs.append(float(mf))
        ys.append(float(of))
        ns.append(n)
    order = np.argsort(xs) if xs else np.array([], dtype=int)
    return {
        "forecast": np.asarray(xs, dtype=float)[order],
        "observed": np.asarray(ys, dtype=float)[order],
        "n": np.asarray(ns, dtype=float)[order],
    }


def _brier_by_lead(run: C6Run, mask_name: str) -> tuple[list[int], list[float]]:
    m = run.mask(mask_name) or {}
    bbl = m.get("brier_by_lead") or {}
    leads = sorted(int(k) for k in bbl)
    return leads, [float(bbl[str(k)]) for k in leads]


def _panel_reliability(ax: Any, runs: list[C6Run], lead_h: int) -> None:
    ax.plot([0, 1], [0, 1], color="#111827", lw=0.9, ls="--", zorder=1)
    drew = False
    for run in runs:
        for mask_name in _MASKS:
            m = run.mask(mask_name)
            bins = (m or {}).get("reliability_bins") or (
                run.payload.get("reliability_bins") if mask_name == _MASKS[0] else None
            )
            if not bins:
                continue
            c = reliability_curve(list(bins), lead_h)
            if c["forecast"].size == 0:
                continue
            st = _MASK_STYLE[mask_name]
            ax.plot(
                c["forecast"],
                c["observed"],
                marker="o",
                ms=4,
                lw=1.4,
                color=st["color"],
                ls="-" if mask_name == "growth_band" else ":",
                label=f"{run.label} · {st['label']}",
                zorder=3,
            )
            drew = True
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"reliability @ +{lead_h} h", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    if not drew:
        ax.text(0.5, 0.5, "no populated bins", ha="center", va="center", color="#b91c1c")


def _panel_spread(ax: Any, runs: list[C6Run]) -> None:
    """Brier over lead time, with the spread statistics annotated beside it.

    Brier genuinely varies with lead. ``dispersion_ratio`` as C6 emits it is a
    single number per run, so it is written as text rather than drawn as a line
    over lead - a flat line across three leads would imply a measurement per
    lead that was never made.
    """
    for run in runs:
        for mask_name in _MASKS:
            leads, vals = _brier_by_lead(run, mask_name)
            if not leads:
                continue
            st = _MASK_STYLE[mask_name]
            ax.plot(
                leads,
                vals,
                marker="s",
                ms=4,
                lw=1.4,
                color=st["color"],
                ls="-" if mask_name == "growth_band" else ":",
                label=f"{run.label} · {st['label']}",
            )
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("Brier score")
    ax.set_title("Brier over lead time", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, loc="upper left", frameon=False)

    lines = []
    for run in runs:
        gb = run.mask("growth_band") or {}
        dom = run.mask("domain") or {}
        dr = gb.get("dispersion_ratio", run.payload.get("dispersion_ratio"))
        adr = gb.get("area_dispersion_ratio")
        piou = (run.payload.get("diagnostics") or {}).get("mean_pairwise_member_iou")
        lines.append(
            f"{run.label}:  dispersion {_fmt(dr)}   area-dispersion {_fmt(adr)}\n"
            f"   pairwise member IoU {_fmt(piou)}   arrival CRPS "
            f"{_fmt(gb.get('arrival_crps', dom.get('arrival_crps')))}"
        )
    ax.text(
        0.985,
        0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=6.5,
        va="bottom",
        ha="right",
        family="monospace",
        color="#374151",
        bbox={
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "boxstyle": "round,pad=0.35",
            "lw": 0.6,
        },
    )


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _panel_members(ax: Any, runs: list[C6Run]) -> None:
    """Member-vs-truth growth: the small-multiple C6 JSON can actually support.

    A nowcast is judged on how much NEW area burns, so this draws every member's
    growth-cell count against truth's. An ensemble whose members all sit on one
    side of the truth line is biased; one whose members are indistinguishable
    from each other has collapsed. Both are visible here without any raster.
    """
    ymax = 0.0
    for row, run in enumerate(runs):
        d = run.payload.get("diagnostics") or {}
        members = d.get("member_growth_cells")
        truth = d.get("truth_growth_cells")
        if not members:
            continue
        m = np.asarray(members, dtype=float)
        jitter = (np.arange(m.size) % 2) * 0.12 - 0.06
        ax.scatter(
            np.full(m.size, row, dtype=float) + jitter,
            m,
            s=22,
            color=COL_MEMBER,
            alpha=0.8,
            zorder=3,
            label="member" if row == 0 else None,
        )
        if truth is not None:
            ax.hlines(
                float(truth),
                row - 0.34,
                row + 0.34,
                color=COL_TRUTH,
                lw=2.2,
                zorder=4,
                label="truth" if row == 0 else None,
            )
            ymax = max(ymax, float(truth))
            n_above = int((m > float(truth)).sum())
            ax.annotate(
                f"{n_above}/{m.size} members ≥ truth",
                (row, float(truth)),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=6.5,
                color=COL_TRUTH,
            )
        ymax = max(ymax, float(m.max()))
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r.label for r in runs], fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("new burned cells over horizon")
    ax.set_title("member vs truth growth", fontsize=10)
    ax.set_ylim(-0.05 * max(ymax, 1.0), 1.20 * max(ymax, 1.0))
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def _reporting_banner(fig: Any) -> None:
    """Stamp C3.3 reportability. Reads a JSON status, never a model."""
    try:
        from wildfire_nowcast.eval.reporting import reporting_status  # noqa: PLC0415

        st = reporting_status()
    except Exception as exc:  # pragma: no cover
        st = {"reportable": False, "reason": f"reporting_status unavailable: {exc}"}

    if st.get("reportable"):
        msg = (
            f"C3.3 OK — norm stats span {st.get('n_train_blocks')} train blocks "
            f"(folds {st.get('train_folds')})"
        )
        color, bg = "#166534", "#dcfce7"
    else:
        msg = (
            f"SMOKE TEST ONLY — C3.3 not satisfied (n_train_blocks="
            f"{st.get('n_train_blocks')}): {st.get('reason')}. Do not quote these numbers."
        )
        color, bg = "#7f1d1d", "#fee2e2"
    fig.text(
        0.5,
        0.885,
        msg,
        ha="center",
        fontsize=8,
        color=color,
        bbox={"facecolor": bg, "edgecolor": color, "boxstyle": "round,pad=0.35", "lw": 0.8},
    )


def render_dashboard(runs: list[C6Run], out: str | Path, *, dpi: int = 130) -> Path:
    """Render the dashboard for one or more C6 runs."""
    if not runs:
        raise ValueError("no C6 runs to render")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.0, 9.6))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.05],
        hspace=0.34,
        wspace=0.30,
        left=0.06,
        right=0.98,
        top=0.80,
        bottom=0.20,
    )
    leads = sorted(
        {
            int(b.get("lead_h", 0))
            for r in runs
            for b in (r.payload.get("reliability_bins") or [])
            if b.get("lead_h") is not None
        }
    ) or [1, 2, 3]
    for i, lead in enumerate(leads[:3]):
        _panel_reliability(fig.add_subplot(gs[0, i]), runs, lead)
    _panel_spread(fig.add_subplot(gs[1, 0:2]), runs)
    _panel_members(fig.add_subplot(gs[1, 2]), runs)

    head = runs[0].payload
    fig.suptitle(
        f"C6 diagnostics — {', '.join(r.label for r in runs)}\n"
        f"event={head.get('event')}  n_members={head.get('n_members')}  "
        f"horizon={head.get('horizon_h')} h  headline mask={head.get('primary_mask')}  "
        f"CRPS={head.get('crps_estimator')}",
        fontsize=11,
        y=0.975,
    )
    _reporting_banner(fig)

    # C6's own warnings, verbatim, on the figure.
    notes: list[str] = []
    for r in runs:
        for n in r.notes:
            if n not in notes:
                notes.append(n)
    for r in runs:
        if r.missing:
            notes.append(
                f"CONTRACT: {r.label} is missing required C6 keys {list(r.missing)} "
                "— report to @model, do not interpret the panels above."
            )
    if notes:
        wrapped: list[str] = []
        for n in notes:
            words, line, first = n.split(), "", True
            for w in words:
                if len(line) + len(w) + 1 > 150:
                    wrapped.append(("  • " if first else "    ") + line)
                    first = False
                    line = ""
                line = f"{line} {w}".strip()
            wrapped.append(("  • " if first else "    ") + line)
        fig.text(
            0.06,
            0.145,
            "C6 notes (verbatim):\n" + "\n".join(wrapped),
            fontsize=7.0,
            color="#7c2d12",
            va="top",
        )
    stamp(fig, "reads C6 evaluate() JSON only — no model internals, no tensor")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.dashboard",
        description="Render the per-run diagnostics dashboard from C6 evaluate() JSON.",
    )
    ap.add_argument("metrics", nargs="+", help="C6 output JSON file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", action="append", default=None, help="label per metrics file")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args(argv)

    labels = args.label or [None] * len(args.metrics)
    if len(labels) != len(args.metrics):
        ap.error("--label must be given once per metrics file, or not at all")

    runs: list[C6Run] = []
    for path, label in zip(args.metrics, labels, strict=True):
        runs.extend(load_c6(path, label=label))

    out = render_dashboard(runs, args.out, dpi=args.dpi)
    print(f"[dashboard] {out}")
    for r in runs:
        if r.missing:
            print(f"[dashboard] CONTRACT: {r.label} missing C6 keys {list(r.missing)}")
    return 1 if any(r.missing for r in runs) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
