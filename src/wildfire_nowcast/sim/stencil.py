"""Measure a predictor's one-step spread stencil on a controlled synthetic input.

    python -m wildfire_nowcast.sim.stencil --kernel kernel=runs/kernel-nll_only-... \
        --out reports/figures/stencil.png

A rollout on a real fire cannot separate what the model believes from what the
landscape imposed. This module removes the landscape: it builds a C5 input that
is perfectly uniform and flat — one burning cell, zero slope, one fuel class, no
barrier, no scar — and asks ``predict()`` for one hour. The resulting
member-fraction field IS the transition kernel's stencil, measured entirely
through the C5 signature. Nothing here reads a weight.

**The estimator for a wind-independent directional preference.** On this domain
every source of anisotropy except the wind has been removed by construction, so
the stencil's probability-weighted centroid ``c(θ)`` at wind bearing ``θ``
decomposes into a wind-driven part plus a constant::

    c(θ) = A(θ) ŵ(θ) + c_0

Averaging ``c(θ)`` over wind bearings sampled uniformly on the circle cancels the
first term exactly when ``A`` does not depend on ``θ`` — which it cannot here,
because nothing in the domain distinguishes one bearing from another. What
survives is ``c_0``: the part of the model's preferred direction that does not
track the wind. It is reported in metres, with the still-air stencil measured
independently as a second, assumption-free estimate of the same quantity.

Probe constants are taken from the MEDIAN of a real tensor rather than invented,
so the stencil is measured somewhere the model was actually trained, and the
choice is auditable instead of arbitrary.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wildfire_nowcast.common.contract import FBFM40_CLASSES, FBFM40_NONBURNABLE  # noqa: E402
from wildfire_nowcast.common.zarr_io import channel_values, open_tensor  # noqa: E402
from wildfire_nowcast.sim.c5 import STATIC_C5, WEATHER_C5  # noqa: E402
from wildfire_nowcast.sim.style import BURN_PROB_CMAP, COL_TEXT, COL_WARN, stamp  # noqa: E402

__all__ = ["ProbeConstants", "probe_from_tensor", "build_probe", "measure_stencil",
           "wind_independent_offset", "render_stencils", "main"]

#: The burnable half of the C1.7 FBFM40 enumeration, taken from ``common`` so the
#: probe cannot disagree with the contract about what "burnable" means.
FBFM40_BURNABLE: frozenset[int] = frozenset(FBFM40_CLASSES) - FBFM40_NONBURNABLE

#: Compass bearings the wind sweep uses, in degrees FROM which... no: TOWARD
#: which the wind blows, matching the C1 ``wind_u10``/``wind_v10`` convention
#: (u eastward, v northward), so 90 deg = blowing toward the east.
SWEEP_BEARINGS: tuple[float, ...] = tuple(float(d) for d in range(0, 360, 45))


@dataclass(frozen=True)
class ProbeConstants:
    """Uniform landscape + weather for the controlled probe."""

    elevation: float
    slope: float
    aspect_sin: float
    aspect_cos: float
    fuel_model_id: float
    canopy_cover: float
    water_barrier_mask: float
    recent_burn_scar: float
    temp_2m: float
    rh_2m: float
    fuel_moisture_proxy: float
    source: str

    def static_value(self, channel: str) -> float:
        return float(getattr(self, channel))

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def probe_from_tensor(tensor_path: str | Path, *, wind_speed_ms: float = 8.0) -> ProbeConstants:
    """Median weather and modal burnable fuel from a real C1 store.

    Slope is forced to ZERO and aspect to a valid unit encoding: the probe must
    be isotropic, and a median slope would tilt every stencil in one direction
    and be mistaken for the very bias we are testing for.
    """
    ds = open_tensor(Path(tensor_path))
    try:
        fuel = np.rint(channel_values(ds, "fuel_model_id", dtype=np.float32)[0]).astype(int)
        burnable = fuel[np.isin(fuel, list(FBFM40_BURNABLE))]
        if burnable.size == 0:
            raise ValueError(f"{tensor_path} has no burnable FBFM40 cells to probe with")
        modal = int(np.bincount(burnable.ravel()).argmax())
        med = {
            name: float(np.median(channel_values(ds, name, dtype=np.float32)))
            for name in ("elevation", "canopy_cover", "temp_2m", "rh_2m", "fuel_moisture_proxy")
        }
        return ProbeConstants(
            elevation=med["elevation"],
            slope=0.0,
            aspect_sin=0.0,
            aspect_cos=1.0,
            fuel_model_id=float(modal),
            canopy_cover=med["canopy_cover"],
            water_barrier_mask=0.0,
            recent_burn_scar=0.0,
            temp_2m=med["temp_2m"],
            rh_2m=med["rh_2m"],
            fuel_moisture_proxy=med["fuel_moisture_proxy"],
            source=f"medians of {tensor_path} (slope forced to 0, aspect to a unit encoding)",
        )
    finally:
        ds.close()


def build_probe(
    const: ProbeConstants, *, size: int, wind_speed_ms: float, bearing_deg: float, horizon_h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x0, static, weather)`` for one burning cell in a uniform world.

    ``bearing_deg`` is the direction the wind blows TOWARD, clockwise from north,
    so 225 deg is a wind pushing the fire to the south-west.
    """
    if size % 2 == 0:
        raise ValueError("probe size must be odd so there is a single centre cell")
    x0 = np.zeros((size, size), dtype=np.uint8)
    x0[size // 2, size // 2] = 1

    static = np.stack(
        [np.full((size, size), const.static_value(c), dtype=np.float32) for c in STATIC_C5]
    )
    theta = np.radians(float(bearing_deg))
    wind_u = float(wind_speed_ms * np.sin(theta))  # eastward
    wind_v = float(wind_speed_ms * np.cos(theta))  # northward
    values = {
        "wind_u10": wind_u,
        "wind_v10": wind_v,
        "temp_2m": const.temp_2m,
        "rh_2m": const.rh_2m,
        "fuel_moisture_proxy": const.fuel_moisture_proxy,
    }
    weather = np.stack(
        [
            np.stack([np.full((size, size), values[c], dtype=np.float32) for c in WEATHER_C5])
            for _ in range(horizon_h)
        ]
    )
    return x0, static.astype(np.float32), weather.astype(np.float32)


def measure_stencil(
    model: Any,
    const: ProbeConstants,
    *,
    size: int = 25,
    wind_speed_ms: float = 8.0,
    bearing_deg: float = 0.0,
    n_members: int = 512,
    horizon_h: int = 1,
    seed: int = 7,
) -> dict[str, Any]:
    """Member-fraction field after ``horizon_h`` hours, plus its centroid in metres.

    The centroid is taken over NEW cells only (the ignition cell is excluded), in
    grid metres with +y NORTH — the same orientation rule the renderers enforce,
    because a row-index centroid would report every stencil as pointing south.
    """
    x0, static, weather = build_probe(
        const, size=size, wind_speed_ms=wind_speed_ms, bearing_deg=bearing_deg,
        horizon_h=horizon_h,
    )
    samples = model.predict(x0, static, weather, int(n_members), int(horizon_h), int(seed))
    burned = (np.asarray(samples)[:, -1] > 0)
    prob = burned.mean(axis=0).astype(np.float64)
    prob_new = prob.copy()
    prob_new[x0 > 0] = 0.0

    c = size // 2
    rows = (c - np.arange(size)).astype(float)  # +north
    cols = (np.arange(size) - c).astype(float)  # +east
    total = float(prob_new.sum())
    if total <= 0:
        centroid = (float("nan"), float("nan"))
        se = (float("nan"), float("nan"))
    else:
        centroid = (
            float((prob_new * cols[None, :]).sum() / total),
            float((prob_new * rows[:, None]).sum() / total),
        )
        # Sampling error on the centroid. The stencil is estimated from
        # ``n_members * total`` independent ignition EVENTS, not from
        # ``n_members`` draws, so a still-air stencil that ignites 0.02 cells per
        # member is a 10-event estimate at M=512 and its bearing means nothing.
        # Reporting the SE next to the offset is the difference between finding a
        # bias and finding Monte-Carlo noise pointing somewhere.
        n_events = float(n_members) * total
        var_x = float((prob_new * (cols[None, :] - centroid[0]) ** 2).sum() / total)
        var_y = float((prob_new * (rows[:, None] - centroid[1]) ** 2).sum() / total)
        se = (
            float(np.sqrt(var_x / n_events)) if n_events > 0 else float("nan"),
            float(np.sqrt(var_y / n_events)) if n_events > 0 else float("nan"),
        )
    return {
        "bearing_deg": float(bearing_deg),
        "wind_speed_ms": float(wind_speed_ms),
        "prob": prob,
        "expected_new_cells": total,
        "n_ignition_events": float(n_members) * total,
        "centroid_cells": centroid,
        "centroid_se_cells": se,
        "max_prob": float(prob_new.max()),
    }


def wind_independent_offset(
    model: Any,
    const: ProbeConstants,
    *,
    bearings: Sequence[float] = SWEEP_BEARINGS,
    wind_speed_ms: float = 8.0,
    still_member_boost: int = 16,
    **kw: Any,
) -> dict[str, Any]:
    """Sweep the wind around the compass and average the stencil centroids.

    The average of a wind-aligned response over uniformly-spaced bearings is
    zero, so the residual is the model's wind-INDEPENDENT preference. Reported
    with the still-air stencil beside it as an independent check: two estimators
    that disagree mean the probe is not isotropic and the number must not be used.
    """
    sweep = [measure_stencil(model, const, wind_speed_ms=wind_speed_ms, bearing_deg=b, **kw)
             for b in bearings]
    still_kw = dict(kw)
    # A still-air stencil ignites far fewer cells than a windy one, so it needs
    # proportionally more members to reach the same event count. Scaling here
    # rather than leaving the caller to notice is the difference between an
    # honest second estimator and a decorative one.
    still_kw["n_members"] = int(kw.get("n_members", 512)) * int(still_member_boost)
    still = measure_stencil(model, const, wind_speed_ms=0.0, bearing_deg=0.0, **still_kw)
    cs = np.asarray([s["centroid_cells"] for s in sweep], dtype=float)
    ok = np.isfinite(cs).all(axis=1)
    mean = cs[ok].mean(axis=0) if ok.any() else np.array([np.nan, np.nan])
    n_ok = int(ok.sum())
    se = (
        np.sqrt(
            (
                np.asarray([s["centroid_se_cells"] for s in sweep], dtype=float)[ok] ** 2
            ).sum(axis=0)
        )
        / max(n_ok, 1)
        if n_ok
        else np.array([np.nan, np.nan])
    )
    still_c = np.asarray(still["centroid_cells"], dtype=float)
    still_se = np.asarray(still["centroid_se_cells"], dtype=float)

    def bearing_of(v: np.ndarray) -> float:
        if not np.all(np.isfinite(v)):
            return float("nan")
        return float((np.degrees(np.arctan2(v[0], v[1])) + 360.0) % 360.0)

    def z(v: np.ndarray, s: np.ndarray) -> float:
        mag = float(np.hypot(*v))
        pooled = float(np.hypot(*s)) / np.sqrt(2.0)
        return mag / pooled if pooled > 0 else float("nan")

    return {
        "sweep": sweep,
        "still": still,
        "sweep_mean_centroid_cells": [float(mean[0]), float(mean[1])],
        "sweep_mean_se_cells": [float(se[0]), float(se[1])],
        "sweep_mean_magnitude_cells": float(np.hypot(*mean)),
        "sweep_mean_bearing_deg": bearing_of(mean),
        "sweep_mean_z": z(mean, se),
        "still_centroid_cells": [float(still_c[0]), float(still_c[1])],
        "still_se_cells": [float(still_se[0]), float(still_se[1])],
        "still_magnitude_cells": float(np.hypot(*still_c)),
        "still_bearing_deg": bearing_of(still_c),
        "still_z": z(still_c, still_se),
        "still_n_ignition_events": still["n_ignition_events"],
        "n_bearings": len(bearings),
        "n_bearings_with_ignition": n_ok,
        "wind_speed_ms": float(wind_speed_ms),
    }


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def _compass(bearing: float) -> str:
    if not np.isfinite(bearing):
        return "n/a"  # a model that ignites nothing has no bearing, and 0 deg is a lie
    names = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW",
             "W", "WNW", "NW", "NNW")
    return names[int((bearing % 360.0) / 22.5 + 0.5) % 16]


def render_stencils(results: dict[str, dict[str, Any]], const: ProbeConstants,
                    out: str | Path) -> Path:
    """One row per model: still air, four wind bearings, and the offset summary."""
    models = list(results)
    show = [0.0, 90.0, 180.0, 270.0]
    n_col = 1 + len(show) + 1
    fig, axes = plt.subplots(len(models), n_col, figsize=(2.45 * n_col + 1.0, 2.85 * len(models)),
                             squeeze=False)

    for r, name in enumerate(models):
        res = results[name]
        panels = [("still air (0 m/s)", res["still"])]
        by_bearing = {s["bearing_deg"]: s for s in res["sweep"]}
        panels += [
            (f"wind → {_compass(b)} ({b:.0f}°)", by_bearing[b])
            for b in show
            if b in by_bearing
        ]

        for c, (title, s) in enumerate(panels):
            ax = axes[r][c]
            prob = s["prob"]
            size = prob.shape[0]
            half = size / 2.0
            ax.imshow(prob, cmap=BURN_PROB_CMAP, vmin=0.0, vmax=max(s["max_prob"], 1e-6),
                      extent=(-half, half, -half, half), origin="upper", interpolation="nearest")
            ax.plot(0, 0, marker="+", ms=9, color="#111827", mew=1.4)
            cx, cy = s["centroid_cells"]
            if np.isfinite(cx) and np.isfinite(cy):
                ax.arrow(0, 0, cx, cy, color="#0f766e", width=0.06, length_includes_head=True,
                         zorder=5)
            span = min(9.0, half)
            ax.set_xlim(-span, span)
            ax.set_ylim(-span, span)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(
                f"E[new cells] {s['expected_new_cells']:.2f}\n"
                f"centroid ({cx:+.3f}, {cy:+.3f}) cells",
                fontsize=6.4,
            )
            if r == 0:
                ax.set_title(title, fontsize=8.5)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)

        ax = axes[r][-1]
        ax.set_aspect("equal")
        sweep_xy = np.asarray([s["centroid_cells"] for s in res["sweep"]], dtype=float)
        ax.plot(sweep_xy[:, 0], sweep_xy[:, 1], "o", ms=4.5, color="#94a3b8",
                label="centroid per wind bearing")
        mx, my = res["sweep_mean_centroid_cells"]
        ax.arrow(0, 0, mx, my, color=COL_WARN, width=0.012, length_includes_head=True, zorder=5)
        sx, sy = res["still_centroid_cells"]
        ax.plot([sx], [sy], marker="*", ms=13, color="#0f766e", ls="",
                label="still-air centroid")
        lim = max(0.35, float(np.abs(sweep_xy).max()) * 1.25)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#cbd5e1", lw=0.7)
        ax.axvline(0, color="#cbd5e1", lw=0.7)
        ax.grid(alpha=0.2, lw=0.4)
        ax.set_title("wind-independent offset", fontsize=8.5)
        ax.set_xlabel(
            f"mean over {res['n_bearings']} bearings:\n"
            f"{res['sweep_mean_magnitude_cells']:.3f} cells toward "
            f"{_compass(res['sweep_mean_bearing_deg'])} ({res['sweep_mean_bearing_deg']:.0f}°), "
            f"z={res['sweep_mean_z']:.1f}\n"
            f"still air: {res['still_magnitude_cells']:.3f} cells toward "
            f"{_compass(res['still_bearing_deg'])} "
            f"({res['still_bearing_deg']:.0f}°), z={res['still_z']:.1f}\n"
            f"({res['still_n_ignition_events']:.0f} still-air ignition events)",
            fontsize=6.4,
            color=COL_WARN if res["sweep_mean_z"] > 3.0 else COL_TEXT,
        )
        if r == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper left")

    fig.suptitle(
        "One-step spread stencil on a UNIFORM, FLAT, single-fuel probe — "
        "measured through C5 predict() only\n"
        "every source of anisotropy except wind is removed by construction, "
        "so a non-zero mean centroid "
        "over a full wind sweep is a directional preference that does not track the wind",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    stamp(fig, f"probe constants: {const.source}; +y is NORTH")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wildfire_nowcast.sim.stencil")
    ap.add_argument("--kernel", action="append", default=[], help="name=run_dir")
    ap.add_argument("--baseline", action="append", default=[], help="registered C5 baseline name")
    ap.add_argument("--run", default=None, help="gate run dir, to include its calibrated ellipse")
    ap.add_argument("--probe-tensor", required=True)
    ap.add_argument("--size", type=int, default=25)
    ap.add_argument("--members", type=int, default=512)
    ap.add_argument("--wind", type=float, default=8.0)
    ap.add_argument("--out", default="reports/figures/stencil.png")
    args = ap.parse_args(argv)

    from wildfire_nowcast.model.api import load_model  # noqa: PLC0415

    const = probe_from_tensor(args.probe_tensor)
    models: dict[str, Any] = {}
    for spec in args.kernel:
        name, _, run_dir = spec.partition("=")
        models[name] = load_model(run_dir)
    for name in args.baseline:
        models[name] = load_model(name)
    if args.run:
        from wildfire_nowcast.sim.replay import load_gate_models  # noqa: PLC0415

        gate = load_gate_models(Path(args.run) / "results.json", include_persistence=False)
        for name, m in gate.models.items():
            models.setdefault(name, m)

    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        res = wind_independent_offset(
            model, const, wind_speed_ms=args.wind, size=args.size, n_members=args.members
        )
        results[name] = res
        print(
            f"[stencil] {name:16s} sweep-mean {res['sweep_mean_magnitude_cells']:.4f} cells "
            f"toward {_compass(res['sweep_mean_bearing_deg']):>3s} "
            f"({res['sweep_mean_bearing_deg']:5.1f} deg) z={res['sweep_mean_z']:5.1f}  |  "
            f"still air {res['still_magnitude_cells']:.4f} cells toward "
            f"{_compass(res['still_bearing_deg']):>3s} "
            f"({res['still_bearing_deg']:5.1f} deg) z={res['still_z']:5.1f} "
            f"on {res['still_n_ignition_events']:.0f} events"
        )

    out = render_stencils(results, const, args.out)
    summary = {
        "kind": "simviz_stencil",
        "probe": const.to_dict(),
        "wind_speed_ms": args.wind,
        "n_members": args.members,
        "models": {
            name: {k: v for k, v in res.items() if k not in ("sweep", "still")}
            | {
                "still_expected_new_cells": res["still"]["expected_new_cells"],
                "sweep_expected_new_cells": [s["expected_new_cells"] for s in res["sweep"]],
            }
            for name, res in results.items()
        },
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[stencil] {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
