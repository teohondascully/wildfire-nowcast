"""Two structural-defect detectors, both measured through C5 ``predict()`` only.

    python -m wildfire_nowcast.sim.drift --run runs/baselines-20260808-052918 \
        --kernel kernel=runs/kernel-nll_only-20260808-044220 --outdir reports/figures

modelling reported two defects it found by inspecting its own gradients
(ADR-015 §6). Neither claim is checkable from a coordination file, and both are
claims about *behaviour*, so both are visible from outside if they are real.
This module tries to see them without reading a single model internal.

**(a) Barrier / non-burnable response.** Claim: ``barrier_log_multiplier`` and
non-burnable fuel have exactly zero gradient, so barrier crossing is unlearnable.
The observable consequence is not "the kernel ignores barriers" — the multiplier
still has whatever value it was initialised with. The observable is that the
kernel's barrier response is *whatever was assumed*, and cannot match the
labels except by luck. So we measure the response and compare it to the labels'.

The measurement controls for distance. Barrier cells are not randomly placed
relative to a fire front — water sits in valleys, roads on ridges — so a raw
"probability on barrier vs off barrier" ratio conflates suppression with
geometry. Rates are therefore computed inside each Chebyshev distance ring
around the ``t0`` burned region and pooled ring by ring.

**(b) Wind-independent drift.** Claim: the free offset weights grew a S/SW
preference matching GOFER's centroid bias (STATE R6), i.e. the model is fitting a
measurement artifact as physics. The discriminator is not "does the fire go
SW" — in California it often does, and the wind often blows that way. It is
whether the bias is fixed in the EARTH frame or in the WIND frame:

* rotate each window's displacement residual into the wind frame and average →
  a wind-frame bias (e.g. "always right of the wind") survives, an earth-fixed
  one cancels;
* average the same residuals in the earth frame → an earth-fixed bias survives,
  a wind-frame one cancels.

``R_earth >> R_wind`` is the signature of a preference that does not track the
wind. The growth-calibrated ellipse is the control: it consumes the identical
wind, slope and fuel fields through the identical C5 call and has no free
directional parameter, so whatever ``R_earth`` it shows is the share
attributable to terrain, barriers and domain geometry rather than to a learned
offset.
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
import xarray as xr  # noqa: E402

from wildfire_nowcast.common.zarr_io import channel_values, open_tensor  # noqa: E402
from wildfire_nowcast.sim.c5 import C5Inputs  # noqa: E402
from wildfire_nowcast.sim.replay import (  # noqa: E402
    GateModels,
    band_of,
    iter_eval_windows,
    load_gate_models,
)
from wildfire_nowcast.sim.style import COL_WARN, plot_extent, stamp  # noqa: E402

__all__ = [
    "NB_FUEL_CODES",
    "ring_index",
    "barrier_response",
    "displacement",
    "drift_statistics",
    "render_drift",
    "main",
]

#: FBFM40 non-burnable classes (NB1..NB9 = 91..99). A kernel that ignites these
#: is contradicting its own fuel channel, not merely mis-calibrating a rate.
NB_FUEL_CODES: tuple[int, ...] = (91, 92, 93, 98, 99)

_OCTANTS = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")


def ring_index(burned0: np.ndarray, max_ring: int) -> np.ndarray:
    """Chebyshev distance in cells from the burned region; 0 inside it.

    Written with repeated dilation rather than a distance transform so the
    package keeps its dependency surface, and because ``max_ring`` is small.
    """
    from wildfire_nowcast.common.states import dilate  # noqa: PLC0415

    out = np.zeros(burned0.shape, dtype=np.int16)
    prev = np.asarray(burned0, dtype=bool)
    for r in range(1, int(max_ring) + 1):
        grown = dilate(prev, 1)
        out[grown & ~prev] = r
        prev = grown
    return out


# --------------------------------------------------------------------------
# (a) barrier / non-burnable response
# --------------------------------------------------------------------------


@dataclass
class RingCounts:
    """Sufficient statistics for one (ring, class) cell population.

    Kept as sums rather than rates so pooling across windows is exact. A mean of
    per-window rates would let a window with three eligible cells outvote one
    with three hundred — the same weighting error that makes a pooled Brier read
    as skill when 62% of windows are motionless.
    """

    n_cells: float = 0.0
    prob_sum: float = 0.0

    def add(self, mask: np.ndarray, prob: np.ndarray) -> None:
        k = int(np.count_nonzero(mask))
        if k:
            self.n_cells += k
            self.prob_sum += float(prob[mask].sum())

    @property
    def rate(self) -> float:
        return self.prob_sum / self.n_cells if self.n_cells else float("nan")


def barrier_response(
    prob_new: np.ndarray,
    *,
    band: np.ndarray,
    rings: np.ndarray,
    barrier: np.ndarray,
    nonburnable: np.ndarray,
    acc: dict[tuple[str, int, bool], RingCounts],
    max_ring: int,
) -> None:
    """Accumulate ignition rate by (class, ring, flag) for one window in place."""
    for r in range(1, max_ring + 1):
        ring = band & (rings == r)
        if not ring.any():
            continue
        for label, flag in (("barrier", barrier), ("nonburnable", nonburnable)):
            for value in (True, False):
                sel = ring & (flag if value else ~flag)
                acc.setdefault((label, r, value), RingCounts()).add(sel, prob_new)


def suppression(
    acc: dict[tuple[str, int, bool], RingCounts], label: str, max_ring: int
) -> dict[str, Any]:
    """Pooled ignition-rate ratio ``on-class / off-class``, ring-matched.

    Ring matching is done by pooling the per-ring rate ratios weighted by the
    number of ON-class cells in the ring, which is a direct-standardised rate
    ratio: it answers "at the same distance from the fire, how much less likely
    is a barrier cell to ignite".
    """
    num = den = 0.0
    per_ring = {}
    for r in range(1, max_ring + 1):
        on = acc.get((label, r, True))
        off = acc.get((label, r, False))
        if not on or not off or not on.n_cells or not off.n_cells or not np.isfinite(off.rate):
            continue
        per_ring[r] = {"on_rate": on.rate, "off_rate": off.rate, "n_on": on.n_cells,
                       "n_off": off.n_cells}
        num += on.rate * on.n_cells
        den += off.rate * on.n_cells
    return {
        "ratio": (num / den) if den > 0 else float("nan"),
        "per_ring": per_ring,
        "n_on_cells": float(sum(v["n_on"] for v in per_ring.values())),
    }


# --------------------------------------------------------------------------
# (b) directional drift
# --------------------------------------------------------------------------


def _centroid(weight: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    total = float(weight.sum())
    if total <= 0:
        return None
    return (
        float((weight * xs[None, :]).sum() / total),
        float((weight * ys[:, None]).sum() / total),
    )


def displacement(
    prob_new: np.ndarray, burned0: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> tuple[float, float] | None:
    """Vector from the ``t0`` frontier centroid to the new-burn centroid, in metres.

    ``ys`` are NORTHINGS from the C1 coordinate, not row indices, so +y is north
    — the same rule that keeps the wind quiver honest (``sim.style``). Using row
    indices here would point every drift statistic due south.
    """
    from wildfire_nowcast.sim.reader import frontier_of  # noqa: PLC0415

    front = frontier_of(np.asarray(burned0, dtype=bool))
    a = _centroid(front.astype(np.float64), xs, ys)
    b = _centroid(np.asarray(prob_new, dtype=np.float64), xs, ys)
    if a is None or b is None:
        return None
    return (b[0] - a[0], b[1] - a[1])


def _unit(v: tuple[float, float]) -> tuple[float, float] | None:
    n = float(np.hypot(*v))
    return None if n <= 0 else (v[0] / n, v[1] / n)


@dataclass
class DriftAccumulator:
    """Resultant vectors of the displacement residual in both frames."""

    earth: list[tuple[float, float]]
    wind: list[tuple[float, float]]
    cos_wind: list[float]
    magnitude_m: list[float]
    bearings: list[tuple[float, float]]  # (wind bearing rad, displacement bearing rad)

    @staticmethod
    def empty() -> DriftAccumulator:
        return DriftAccumulator([], [], [], [], [])

    def add(self, d: tuple[float, float], w: tuple[float, float]) -> None:
        du, wu = _unit(d), _unit(w)
        if du is None or wu is None:
            return
        proj = du[0] * wu[0] + du[1] * wu[1]
        resid = (du[0] - proj * wu[0], du[1] - proj * wu[1])
        # Rotate the residual into the wind frame: x' along wind, y' left of wind.
        wx, wy = wu
        self.earth.append(resid)
        self.wind.append((resid[0] * wx + resid[1] * wy, -resid[0] * wy + resid[1] * wx))
        self.cos_wind.append(proj)
        self.magnitude_m.append(float(np.hypot(*d)))
        self.bearings.append((float(np.arctan2(wy, wx)), float(np.arctan2(du[1], du[0]))))

    def summary(self) -> dict[str, Any]:
        def resultant(vs: list[tuple[float, float]]) -> dict[str, float]:
            if not vs:
                return {"r": float("nan"), "bearing_deg": float("nan"), "x": float("nan"),
                        "y": float("nan")}
            arr = np.asarray(vs, dtype=float)
            mx, my = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            # Compass bearing of the resultant: 0 = toward N, 90 = toward E.
            return {
                "r": float(np.hypot(mx, my)),
                "bearing_deg": float((np.degrees(np.arctan2(mx, my)) + 360.0) % 360.0),
                "x": mx,
                "y": my,
            }

        return {
            "n_windows": len(self.earth),
            "earth_frame": resultant(self.earth),
            "wind_frame": resultant(self.wind),
            "mean_cos_wind": float(np.mean(self.cos_wind)) if self.cos_wind else float("nan"),
            "mean_displacement_m": (
                float(np.mean(self.magnitude_m)) if self.magnitude_m else float("nan")
            ),
        }


def _octant(bearing_deg: float) -> str:
    deg = np.degrees(bearing_deg) if bearing_deg <= 2 * np.pi else bearing_deg
    return _OCTANTS[int(deg // 45) % 8]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def drift_statistics(
    fire_id: str,
    tensor_path: str | Path,
    gate: GateModels,
    *,
    horizon_h: int = 3,
    stride: int = 2,
    n_members: int = 24,
    seed: int = 20260807,
    band_radius_cells: int = 12,
    growth_only: bool = True,
    max_ring: int = 6,
) -> dict[str, Any]:
    """Barrier response and drift for every model on one fire."""
    ds: xr.Dataset = open_tensor(Path(tensor_path))
    geom = plot_extent(
        ds["x"].values, ds["y"].values, cell_size_m=float(ds.attrs.get("cell_size_m", 1000.0))
    )
    barrier = channel_values(ds, "water_barrier_mask", dtype=np.float32)[0] > 0.5
    fuel = channel_values(ds, "fuel_model_id", dtype=np.float32)[0]
    nonburnable = np.isin(np.rint(fuel).astype(int), NB_FUEL_CODES)
    wind_u = channel_values(ds, "wind_u10", dtype=np.float32)
    wind_v = channel_values(ds, "wind_v10", dtype=np.float32)

    kept: list[tuple[int, C5Inputs, np.ndarray, np.ndarray, tuple[float, float]]] = []
    for i, w in iter_eval_windows(ds, horizon_h, stride=stride):
        if growth_only and int((w.truth[-1] > 0).sum()) <= int((w.x0 > 0).sum()):
            continue
        band = band_of(w.x0, band_radius_cells)
        rings = ring_index(w.x0 > 0, max_ring)
        sl = slice(w.t0 + 1, w.t0 + 1 + horizon_h)
        active = (w.x0 > 0) | band
        wind = (
            float(wind_u[sl][:, active].mean()),
            float(wind_v[sl][:, active].mean()),
        )
        kept.append((i, w, band, rings, wind))

    out: dict[str, Any] = {
        "fire_id": fire_id,
        "n_windows": len(kept),
        "band_radius_cells": band_radius_cells,
        "max_ring": max_ring,
        "models": {},
    }

    sources: list[tuple[str, Any]] = [("truth", None)] + list(gate.models.items())
    for name, model in sources:
        acc: dict[tuple[str, int, bool], RingCounts] = {}
        drift = DriftAccumulator.empty()
        for i, w, band, rings, wind in kept:
            burned0 = w.x0 > 0
            if name == "truth":
                prob_new = ((w.truth[-1] > 0) & ~burned0).astype(np.float64)
            else:
                samples = model.predict(w.x0, w.static, w.weather, n_members, horizon_h, seed + i)
                prob_new = ((samples[:, -1] > 0) & ~burned0[None]).mean(axis=0).astype(np.float64)
            barrier_response(
                prob_new, band=band, rings=rings, barrier=barrier, nonburnable=nonburnable,
                acc=acc, max_ring=max_ring,
            )
            d = displacement(prob_new * band, burned0, geom.x_centres, geom.y_centres)
            if d is not None:
                drift.add(d, wind)
        out["models"][name] = {
            "barrier": suppression(acc, "barrier", max_ring),
            "nonburnable": suppression(acc, "nonburnable", max_ring),
            "drift": drift.summary(),
            "_bearings": drift.bearings,
        }
    out["_geom_cell_m"] = geom.cell_size_m
    ds.close()
    return out


def render_drift(per_fire: dict[str, dict[str, Any]], out: str | Path) -> Path:
    """Two panels of evidence: barrier suppression, and earth-vs-wind-frame bias."""
    models: list[str] = []
    for entry in per_fire.values():
        for m in entry["models"]:
            if m not in models:
                models.append(m)
    preferred = ("truth", "kernel", "ellipse_cal3h", "ellipse", "kernel_init")
    order = [m for m in preferred if m in models]
    order += [m for m in models if m not in order and m != "persistence"]

    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.28,
                          left=0.06, right=0.98, top=0.86, bottom=0.09)

    # -- (a) barrier + non-burnable suppression ---------------------------
    for col, label in enumerate(("barrier", "nonburnable")):
        ax = fig.add_subplot(gs[0, col])
        fires = list(per_fire)
        width = 0.8 / max(len(order), 1)
        for k, model in enumerate(order):
            vals = [per_fire[f]["models"].get(model, {}).get(label, {}).get("ratio", np.nan)
                    for f in fires]
            xs = np.arange(len(fires)) + k * width - 0.4 + width / 2
            ax.bar(xs, vals, width=width * 0.92,
                   color="#111827" if model == "truth" else None,
                   label=model, edgecolor="#374151", lw=0.4)
        ax.axhline(1.0, color=COL_WARN, ls="--", lw=1.2)
        ax.text(len(fires) - 0.5, 1.02, "1.0 = NO suppression", color=COL_WARN, fontsize=7,
                ha="right", va="bottom")
        ax.set_xticks(range(len(fires)))
        ax.set_xticklabels([f.replace("2020_", "") for f in fires], fontsize=7.5, rotation=12)
        ax.set_yscale("log")
        ax.set_ylabel("ignition-rate ratio  on-class / off-class\n(ring-matched, pooled)")
        ax.set_title(
            f"(a) {label} response — lower = more suppression", fontsize=9.5
        )
        ax.grid(alpha=0.25, lw=0.5, axis="y")
        if col == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)

    # -- (b) earth-frame vs wind-frame residual ---------------------------
    ax = fig.add_subplot(gs[0, 2])
    fires = list(per_fire)
    for model in order:
        re = [per_fire[f]["models"][model]["drift"]["earth_frame"]["r"] for f in fires]
        rw = [per_fire[f]["models"][model]["drift"]["wind_frame"]["r"] for f in fires]
        ax.scatter(rw, re, s=46, label=model, edgecolors="#374151", lw=0.5, zorder=3)
    lim = 0.62
    ax.plot([0, lim], [0, lim], color="#111827", ls="--", lw=0.9)
    ax.text(lim * 0.52, lim * 0.46, "equal — no frame preference", rotation=41, fontsize=6.5,
            color="#374151")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("wind-frame resultant |R| (bias that tracks the wind)")
    ax.set_ylabel("earth-frame resultant |R|\n(bias fixed to the compass)")
    ax.set_title("(b) is the residual drift earth-fixed or wind-fixed?", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False)

    # -- (c) drift rose ----------------------------------------------------
    rose_models = [m for m in order if m in ("truth", "kernel", "ellipse_cal3h")][:3]
    for col, model in enumerate(rose_models):
        ax = fig.add_subplot(gs[1, col], projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        allb: list[tuple[float, float]] = []
        for f in fires:
            allb.extend(per_fire[f]["models"][model]["_bearings"])
        if allb:
            arr = np.asarray(allb)
            # compass bearing (0 = toward north) of the displacement
            comp = (np.degrees(np.arctan2(np.cos(arr[:, 1]), np.sin(arr[:, 1]))) + 360.0) % 360.0
            counts, edges = np.histogram(comp, bins=16, range=(0, 360))
            centres = np.radians(0.5 * (edges[:-1] + edges[1:]))
            ax.bar(centres, counts, width=np.radians(21), color="#0f766e", alpha=0.85,
                   edgecolor="#134e4a", lw=0.4)
            wcomp = (np.degrees(np.arctan2(np.cos(arr[:, 0]), np.sin(arr[:, 0]))) + 360.0) % 360.0
            wc, we = np.histogram(wcomp, bins=16, range=(0, 360))
            ax.plot(np.radians(0.5 * (we[:-1] + we[1:])), wc * counts.max() / max(wc.max(), 1),
                    color=COL_WARN, lw=1.4, label="wind (scaled)")
        ax.set_title(f"(c) {model}: where the front moved", fontsize=9.5, pad=14)
        ax.tick_params(labelsize=6.5)
        if col == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(-0.15, -0.12))

    fig.suptitle(
        "Can the two reported kernel defects be SEEN from outside? (C5 predict() only)\n"
        "(a) barrier + non-burnable suppression against the labels' own    "
        "(b) earth-fixed vs wind-tracking directional bias    (c) drift rose vs wind rose",
        fontsize=11,
    )
    stamp(fig, "no model internals read; the growth-calibrated ellipse is the same-inputs, "
               "no-free-direction control")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wildfire_nowcast.sim.drift")
    ap.add_argument("--run", required=True)
    ap.add_argument("--kernel", action="append", default=[])
    ap.add_argument("--fires", nargs="*", default=None)
    ap.add_argument("--fires-dir", default="data/fires")
    ap.add_argument("--outdir", default="reports/figures")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args(argv)

    run = Path(args.run)
    payload = json.loads((run / "results.json").read_text())
    horizon = int(payload.get("horizon_h", 3))
    members = int(payload.get("n_members", 24))
    seed = int(payload.get("seed", 20260807))
    fires = args.fires or list(payload.get("per_fire") or {})
    gate = load_gate_models(run / "results.json", kernels=args.kernel, include_persistence=False)

    from wildfire_nowcast.eval.masks import default_band_radius  # noqa: PLC0415

    radius = default_band_radius(horizon)
    per_fire: dict[str, dict[str, Any]] = {}
    for fire in fires:
        per_fire[fire] = drift_statistics(
            fire, Path(args.fires_dir) / fire / "tensor.zarr", gate,
            horizon_h=horizon, stride=args.stride, n_members=members, seed=seed,
            band_radius_cells=radius,
        )
        print(f"[drift] {fire} ({per_fire[fire]['n_windows']} growth windows)")
        for name, m in per_fire[fire]["models"].items():
            d = m["drift"]
            print(
                f"    {name:14s} barrier x{m['barrier']['ratio']:.3f}"
                f"  nonburnable x{m['nonburnable']['ratio']:.3f}"
                f"  | cos(wind) {d['mean_cos_wind']:+.3f}"
                f"  R_earth {d['earth_frame']['r']:.3f} @ {d['earth_frame']['bearing_deg']:5.1f}deg"
                f"  R_wind {d['wind_frame']['r']:.3f}"
            )

    outdir = Path(args.outdir)
    fig = render_drift(per_fire, outdir / "drift_and_barrier.png")
    serialisable = {
        f: {
            **{k: v for k, v in e.items() if k != "models"},
            "models": {
                m: {k: v for k, v in mv.items() if not k.startswith("_")}
                for m, mv in e["models"].items()
            },
        }
        for f, e in per_fire.items()
    }
    (outdir / "drift_and_barrier.json").write_text(
        json.dumps({"kind": "simviz_drift", "run": str(run), "per_fire": serialisable}, indent=1)
        + "\n"
    )
    print(f"[drift] {fig}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
