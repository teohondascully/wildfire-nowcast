"""S14 - the per-horizon collapse verdict on the REAL model, assembled into one page.

    python -m wildfire_nowcast.sim.s14_report --glob 'reports/figures/s14/*.json' \
        --out reports/figures/s14_collapse_real_model.png

Reads only the artifacts :mod:`wildfire_nowcast.sim.collapse` writes. It loads no
checkpoint, calls no model and re-runs no instrument, so a number on the page is
a number some earlier invocation published with its control and its horizon
beside it, and the page cannot disagree with the record it summarises.

WHAT THIS PAGE IS FOR
---------------------
Until `1791e05` the latent-off ablation arm was not loadable BY NAME, so every
collapse number this package had produced was measured on the visualisation
fixture, whose own docstring forbids its use in a gate (ADR-118, ADR-119). The
`eval/` side ran on the real model and had power only at leads 5-6 (ADR-123).
Right horizon, wrong subject; right subject, wrong horizon. This page is the
missing cell: the repaired one-step instrument, on the real model, at 1/2/3 h.

THE TWO NUMBERS AND WHICH ONE MEANS SOMETHING
---------------------------------------------
``power`` is the share of admissible verdicts reading ``collapsed`` on the
latent-off arm. On the one-step increment that arm is conditionally independent
given ``x_t``, so its index is ``1.0`` BY ALGEBRA and its power is near 1 BY
CONSTRUCTION. A page that led with it would be publishing a tautology.
``separation`` - power minus the same share on the latent-ON model - is what the
instrument can actually distinguish at that lead, and it is what this page leads
with.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.sim.absent import EXIT_NOTHING_EXAMINED, refuse_if_empty

__all__ = ["build_report", "render", "main"]

#: The bar every verdict on the page was taken against. Read from the artifacts
#: rather than restated here; this is only the key it is read from.
_THRESHOLD_KEY = "threshold"


def _load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text())
    return payload


def build_report(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Assemble published verdicts and power profiles into one record.

    Every artifact is either a VERDICT record (three per-horizon verdicts, each
    with its own control reading and its own ``lead_h``) or a REFUSAL record
    (``verdict_withheld``, no verdicts). Both belong on the page: a refusal is
    the instrument declining a degenerate subject, which is the check working,
    and hiding it would leave a reader wondering which arms were even tried.
    """
    verdict_files: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for raw in paths:
        payload = _load(Path(raw))
        payload["_artifact"] = str(raw)
        if payload.get("verdict_withheld"):
            refusals.append(payload)
        else:
            verdict_files.append(payload)

    refuse_if_empty(
        "s14_report",
        {"artifacts": len(verdict_files) + len(refusals)},
        because="a page assembled from no artifacts states nothing about any model.",
    )

    scenes: list[dict[str, Any]] = []
    controls: list[float] = []
    admissible_controls: list[float] = []
    for payload in verdict_files:
        meta = payload["meta"]
        rows = []
        for record in payload["per_horizon_verdicts"]:
            # ADR-114 (c)(d): a verdict without its horizon or without a control
            # reading beside it is not a verdict this page may repeat.
            assert "lead_h" in record, record
            assert "controls" in record, record
            controls.append(float(record["controls"]["independent_index"]))
            if bool(record["is_a_verdict"]):
                admissible_controls.append(float(record["controls"]["independent_index"]))
            rows.append(record)
        scenes.append(
            {
                "tensor": payload["tensor"],
                "fire_id": Path(str(payload["tensor"])).parent.name,
                "model": payload.get("model"),
                "subject": payload.get("subject", {}),
                "t0": meta["t0"],
                "n_members": meta["n_members"],
                "conditioning": meta["conditioning"],
                "seed": meta["seed"],
                "verdicts": rows,
                "cumulative_index_description": payload["cumulative_index_description"],
                "power": payload.get("lead_power_profile", {}),
            }
        )

    pooled = _pool_power([s["power"] for s in scenes])
    verdict_counts: dict[str, int] = {}
    for scene in scenes:
        for record in scene["verdicts"]:
            head = str(record["verdict"]).split(":")[0]
            verdict_counts[head] = verdict_counts.get(head, 0) + 1

    return {
        "n_scenes": len(scenes),
        "n_refusals": len(refusals),
        "scenes": scenes,
        "refusals": [
            {
                "artifact": r["_artifact"],
                "model": r.get("model"),
                "subject": r.get("subject", {}),
                "verdict_withheld": r["verdict_withheld"],
            }
            for r in refusals
        ],
        "verdict_counts": verdict_counts,
        "instrument_control": {
            **_control_stats(admissible_controls),
            "all_readings": _control_stats(controls),
            "note": (
                "the instrument's own positive control, independent BY CONSTRUCTION, read in "
                "the SAME invocation as the verdict beside it (ADR-114 (c)). Its null is 1.0. "
                "The headline is taken over ADMISSIBLE verdicts only: a scene the controls "
                "refused reads 0.0 by construction - nothing was uncertain, so nothing was "
                "drawn - and averaging that in would report a refusal as estimator bias. "
                "`all_readings` is the same series with the refused scenes included."
            ),
        },
        "pooled_power_profile": pooled,
    }


def _short(fire_id: str) -> str:
    """Fire id short enough to sit under a bar without colliding with its neighbour."""
    return fire_id.replace("_lightning_complex", "").replace("_complex", "")


def _control_stats(values: Sequence[float]) -> dict[str, Any]:
    """Mean / sd / range of a control series, with its own denominator as a field."""
    arr = np.asarray(list(values), dtype=float)
    return {
        "n_readings": int(arr.size),
        "independent_control_mean": float(arr.mean()) if arr.size else None,
        "independent_control_sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "independent_control_min": float(arr.min()) if arr.size else None,
        "independent_control_max": float(arr.max()) if arr.size else None,
    }


def _pool_power(profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum the per-scene tallies, then take the rate. Never a mean of rates.

    A mean of per-scene rates weights a scene the controls refused 197 times out
    of 200 exactly as heavily as one they never refused, which is how a scene
    with no power comes to dominate a power profile.
    """
    by_lead: dict[int, dict[str, int]] = {}
    for profile in profiles:
        for row in profile.get("by_lead", []):
            cell = by_lead.setdefault(
                int(row["lead_h"]),
                {
                    "treatment_collapsed": 0,
                    "treatment_not_collapsed": 0,
                    "treatment_refused": 0,
                    "null_collapsed": 0,
                    "null_not_collapsed": 0,
                    "null_refused": 0,
                },
            )
            for key in cell:
                cell[key] += int(row[key])
    rows: list[dict[str, Any]] = []
    for lead in sorted(by_lead):
        cell = by_lead[lead]
        t_n = cell["treatment_collapsed"] + cell["treatment_not_collapsed"]
        n_n = cell["null_collapsed"] + cell["null_not_collapsed"]
        power = None if t_n == 0 else cell["treatment_collapsed"] / t_n
        false_fire = None if n_n == 0 else cell["null_collapsed"] / n_n
        rows.append(
            {
                "lead_h": lead,
                **cell,
                "treatment_admissible": t_n,
                "null_admissible": n_n,
                "power": power,
                "false_fire": false_fire,
                "separation": None if power is None or false_fire is None else power - false_fire,
            }
        )
    return {
        "clause": "C6.7 [v2.18] (ADR-123)",
        "pooling": "counts summed across scenes, then the rate; never a mean of per-scene rates",
        "by_lead": rows,
    }


def render(report: dict[str, Any], out_png: str | Path) -> Path:
    """One page: the verdict grid, the per-lead profile, and what was refused."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from wildfire_nowcast.sim.style import COL_TEXT, COL_WARN, stamp  # noqa: PLC0415

    scenes = report["scenes"]
    pooled = report["pooled_power_profile"]["by_lead"]
    fig = plt.figure(figsize=(13.6, 8.4))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.35, 1.0], hspace=0.42, wspace=0.24
    )

    # -- A. the verdict grid ------------------------------------------------
    ax = fig.add_subplot(grid[0, :])
    threshold = float(scenes[0]["verdicts"][0][_THRESHOLD_KEY]) if scenes else 1.5
    labels: list[str] = []
    values: list[float] = []
    colours: list[str] = []
    ctl: list[float] = []
    for scene in scenes:
        for record in scene["verdicts"]:
            labels.append(f"{_short(scene['fire_id'])}\n{record['lead_h']} h")
            values.append(float(record["independence_dispersion_index"]))
            ctl.append(float(record["controls"]["independent_index"]))
            head = str(record["verdict"]).split(":")[0]
            colours.append(
                "#0f766e"
                if head == "collapsed"
                else ("#9ca3af" if head != "not_collapsed" else COL_WARN)
            )
    xs = np.arange(len(values))
    ax.bar(xs, values, color=colours, width=0.62, zorder=3)
    ax.plot(xs, ctl, "o", color="#111111", ms=4.2, zorder=4, label="instrument control (null 1.0)")
    ax.axhline(threshold, color=COL_WARN, lw=1.4, ls="--", zorder=2)
    ax.axhline(1.0, color="#6b7280", lw=0.9, ls=":", zorder=2)
    ax.text(
        len(values) - 0.4,
        threshold + 0.03,
        f"COLLAPSE_INDEX_THRESHOLD = {threshold:g}",
        color=COL_WARN,
        fontsize=8.4,
        ha="right",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("one-step independence dispersion index")
    ax.set_ylim(0.0, max(2.0, max(values + ctl) * 1.15))
    ax.set_title(
        "G3 (d) on the REAL model, one verdict per lead hour  "
        "(teal = collapsed, grey = NO VERDICT, the controls refused the scene)",
        fontsize=10.5,
        color=COL_TEXT,
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # -- B. power / false fire per lead ------------------------------------
    ax2 = fig.add_subplot(grid[1, 0])
    leads = [row["lead_h"] for row in pooled]
    for key, colour, name in (
        ("power", "#0f766e", "power: latent-OFF arm reads collapsed"),
        ("false_fire", COL_WARN, "false fire: latent-ON model reads collapsed"),
        ("separation", "#1f4e5f", "SEPARATION (the informative number)"),
    ):
        ys = [100.0 * row[key] if row[key] is not None else np.nan for row in pooled]
        ax2.plot(leads, ys, "-o", color=colour, lw=1.8, ms=5, label=name)
        for x, y in zip(leads, ys, strict=True):
            if np.isfinite(y):
                ax2.annotate(
                    f"{y:.0f}%",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 6),
                    fontsize=8,
                    color=colour,
                )
    ax2.set_xticks(leads)
    ax2.set_xlabel("lead hour")
    ax2.set_ylabel("% of admissible verdicts")
    ax2.set_ylim(-5, 108)
    ax2.set_title("C6.7 [v2.18]: power AT 1/2/3 h, pooled over the held-out fires", fontsize=10)
    ax2.legend(loc="center left", fontsize=7.8, frameon=False)

    # -- C. what was refused, and the subject ------------------------------
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.axis("off")
    lines = ["SUBJECT, and what the instrument refused"]
    if scenes:
        subject = scenes[0]["subject"]
        lines += [
            f"  treatment  {subject.get('treatment_address', '?')}",
            f"  null       {subject.get('null_address', '?')}",
            f"  C5 check   {str(subject.get('contract_check', '?'))[:58]}",
            f"  same draw  bit-identical={subject.get('measured_bit_identical')}",
        ]
    for refusal in report["refusals"]:
        subject = refusal.get("subject", {})
        lines += [
            "",
            f"  REFUSED    {subject.get('treatment_address', refusal.get('model'))}",
            f"             bit-identical={subject.get('measured_bit_identical')} "
            f"({subject.get('n_identical_members')} members)",
            "             a control that cannot discriminate may be LOADED,",
            "             and may not be SCORED (C5 [v2.18]).",
        ]
    counts = report["verdict_counts"]
    control = report["instrument_control"]
    lines += [
        "",
        "  published verdicts  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"  instrument control  {control['independent_control_mean']:.4f}"
        f" +/- {control['independent_control_sd']:.4f} over {control['n_readings']} readings",
    ]
    ax3.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.6,
        family="monospace",
        color=COL_TEXT,
        transform=ax3.transAxes,
    )

    stamp(
        fig,
        "power is near 1 BY ALGEBRA on the one-step increment - the latent-off arm IS the "
        "independent sampler. Read the separation.",
    )
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.s14_report", description=__doc__
    )
    ap.add_argument("--glob", action="append", default=[], help="artifacts written by sim.collapse")
    ap.add_argument("--out", required=True, help="the one-page png")
    ap.add_argument("--json", default=None, help="write the assembled record here")
    args = ap.parse_args(argv)

    paths = sorted({p for pattern in args.glob for p in globmod.glob(pattern)})
    try:
        report = build_report(paths)
    except Exception as exc:  # noqa: BLE001 - the absent case is a refusal, not a crash
        from wildfire_nowcast.sim.absent import AbsentMeasurementError  # noqa: PLC0415

        if not isinstance(exc, AbsentMeasurementError):
            raise
        print(str(exc))
        return EXIT_NOTHING_EXAMINED
    png = render(report, args.out)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {png}")
    print(json.dumps(report["pooled_power_profile"], indent=2, default=str))
    print(json.dumps(report["instrument_control"], indent=2, default=str))
    print(json.dumps(report["verdict_counts"], indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
