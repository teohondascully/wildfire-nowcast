"""[E1] One page: does the PHYSICS baseline decelerate? (ADR-064)

Reads ``runs/e1.json`` - the artifact ``sim.elmfire_stage.score`` writes - and
nothing else. No model is loaded, no metric is recomputed, and ``eval/stage.py``
is not touched: if a number is not in the artifact this page draws a MISSING
panel rather than inventing one.

WHAT THE PAGE HAS TO MAKE VISIBLE
---------------------------------
``stage_decay`` is ``log(late-half mean growth) - log(early-half mean growth)``,
so a single number hides the two means it came from. A model can land on the same
``stage_decay`` as truth while growing 30x too fast throughout, and the reader of
a bar chart would never know. Panel 2 therefore draws the EARLY and LATE means
that produce each bar, on a log axis, so a level error cannot hide inside a
ratio.

Panel 1 is the comparison ADR-064 asks for: per block, truth against arm A,
arm S and ELMFIRE, on one axis, with the sign boundary drawn. Panel 3 is the
member-count sensitivity, because the ensemble size is OUR construction and the
honest question "would a bigger ensemble have changed the sign?" should be
answerable off the page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.sim.style import COL_TEXT, COL_WARN, stamp

__all__ = ["render", "main"]

_BLOCK_NAMES = {
    4: "Creek (4)",
    5: "CZU (5)",
    6: "Dolan (6)",
    7: "July (7)",
    12: "Borel (12)",
}

_SERIES = (
    ("truth_reference", "truth (GOFER)", "#111111", "o"),
    ("arm_a_reference", "arm A (our kernel)", "#c2410c", "s"),
    ("arm_s_reference", "arm S (+4 params)", "#7c3aed", "^"),
    ("elmfire", "ELMFIRE (Rothermel)", "#0369a1", "D"),
)


def render(payload: dict[str, Any], path: Path) -> Path:
    """Draw the E1 page. Every number is substituted from ``payload``."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    per_block = payload.get("per_block") or {}
    blocks = sorted(int(b) for b in per_block)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), width_ratios=[1.25, 1.0, 0.85])

    # -- panel 1: stage_decay per block, four series ----------------------
    ax = axes[0]
    xs = np.arange(len(blocks), dtype=float)
    for key, label, colour, marker in _SERIES:
        ys = [per_block[str(b)].get(key) for b in blocks]
        drawn = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
        if not drawn:
            continue
        ax.plot(
            [d[0] for d in drawn],
            [d[1] for d in drawn],
            marker=marker,
            ms=9,
            lw=1.1,
            ls="--",
            color=colour,
            label=label,
            zorder=3,
        )
    ax.axhline(0.0, color=COL_WARN, lw=1.4, zorder=2)
    ax.text(
        0.012,
        0.0,
        " 0 = neither speeds up nor slows down",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="left",
        color=COL_WARN,
        fontsize=8.5,
    )
    # Right-aligned: the left edge is where block 0's markers sit, and a caption
    # sitting on top of a data point is a rendering defect, not a caption.
    plate = {"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.6}
    ax.text(
        0.985,
        0.975,
        "ACCELERATES with age",
        transform=ax.transAxes,
        fontsize=9,
        color=COL_WARN,
        va="top",
        ha="right",
        bbox=plate,
        zorder=4,
    )
    ax.text(
        0.985,
        0.025,
        "DECELERATES with age",
        transform=ax.transAxes,
        fontsize=9,
        color="#166534",
        va="bottom",
        ha="right",
        bbox=plate,
        zorder=4,
    )
    # Room on the right for the two captions. Without it a caption sits on the
    # last block's marker, which is the same defect class as insight 45 (iv).
    ax.set_xlim(-0.35, (len(blocks) - 1) + 0.85 if blocks else 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([_BLOCK_NAMES.get(b, str(b)) for b in blocks], fontsize=9)
    ax.set_ylabel("stage_decay  =  log(late-half mean growth / early-half mean growth)")
    ax.set_title(
        "1. ELMFIRE is Rothermel, so it is perimeter-proportional too.\n"
        "   Does it slow down as a fire ages?",
        fontsize=11,
        loc="left",
    )
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.18)

    # -- panel 2: the two means the ratio came from -----------------------
    ax = axes[1]
    width = 0.19
    for k, (who, colour) in enumerate((("truth_detail", "#111111"), ("elmfire_detail", "#0369a1"))):
        early = [(per_block[str(b)].get(who) or {}).get("early_mean_growth") for b in blocks]
        late = [(per_block[str(b)].get(who) or {}).get("late_mean_growth") for b in blocks]
        halves = ((early, "", "early half"), (late, "///", "late half"))
        for j, (vals, hatch, tag) in enumerate(halves):
            xs2 = xs + (k * 2 + j - 1.5) * width
            ok = [(x, v) for x, v in zip(xs2, vals, strict=True) if v is not None and v > 0]
            if not ok:
                continue
            ax.bar(
                [o[0] for o in ok],
                [o[1] for o in ok],
                width=width,
                color=colour,
                alpha=0.95 if j else 0.45,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.6,
                label=f"{'ELMFIRE' if k else 'truth'} {tag}",
            )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([_BLOCK_NAMES.get(b, str(b)) for b in blocks], fontsize=9)
    ax.set_ylabel("mean new burned cells per 3 h window (log)")
    ax.set_title(
        "2. The LEVELS behind the ratio.\n   Same stage_decay can hide a 30x growth error.",
        fontsize=11,
        loc="left",
    )
    ax.legend(fontsize=7.6, ncol=2, framealpha=0.92)
    ax.grid(alpha=0.18, axis="y")

    # -- panel 3: member-count sensitivity --------------------------------
    ax = axes[2]
    sens = payload.get("member_prefix_sensitivity") or {}
    for b in blocks:
        ms = sorted(int(m) for m in sens if str(b) in sens[m] and sens[m][str(b)] is not None)
        if not ms:
            continue
        ax.plot(
            ms,
            [sens[str(m)][str(b)] for m in ms],
            marker="o",
            ms=5,
            lw=1.2,
            label=_BLOCK_NAMES.get(b, str(b)),
        )
    ax.axhline(0.0, color=COL_WARN, lw=1.2)
    ax.set_xlabel("ensemble members used")
    ax.set_ylabel("stage_decay")
    ax.set_title(
        "3. The ensemble is OUR construction.\n   Does the member count set the sign?",
        fontsize=11,
        loc="left",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))

    verdict = str(payload.get("verdict", "MISSING"))
    reason = str(payload.get("verdict_reason", ""))
    fig.suptitle(
        f"E1 (ADR-064) - {payload.get('n_blocks_positive', '?')} of "
        f"{payload.get('n_blocks_scored', '?')} held-out blocks ACCELERATE under ELMFIRE   |   "
        f"verdict: {verdict}",
        fontsize=13,
        x=0.008,
        ha="left",
        color=COL_TEXT,
    )
    stamp(
        fig,
        "  ".join(
            (
                "estimator: eval/stage.py UNCHANGED (judged truth, arm A and arm S too)",
                f"split {payload.get('split_fingerprint')}",
                f"members {payload.get('headline_member_prefix') or 'MISSING'}",
                f"stride {payload.get('stride')}  horizon {payload.get('horizon_h')} h",
                "stage_decay is NOT LICENSED to decide a gate - G5 is NOT attempted here",
            )
        ),
    )
    fig.text(0.008, 0.925, reason, fontsize=8.6, color=COL_WARN, ha="left", va="top", wrap=True)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(prog="python -m wildfire_nowcast.sim.e1_report")
    ap.add_argument("--run", default="runs/e1.json")
    ap.add_argument("--out", default="reports/figures/e1_stage_decay.png")
    args = ap.parse_args(argv)
    payload = json.loads(Path(args.run).read_text())
    out = render(payload, Path(args.out))
    print(f"[e1] {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
