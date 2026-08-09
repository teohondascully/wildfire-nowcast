"""Detect and render the ignition components of a fire — C2 ``n_ignition_components``.

    python -m wildfire_nowcast.sim.components \
        --tensor data/fires/2020_july_complex/tensor.zarr \
        --outdir reports/figures --manifest-check

GOFER files separate lightning ignitions under ONE fire id (ADR-014 §7, C2
[v2.7]). ``2020_july_complex`` is declared as 2 components and SCU as 3, and the
consequence is a 47 km apparent "teleport" that is not spotting and that no
contagion kernel can or should reproduce. This module makes that visible, because
a number in a manifest is easy to forget and a picture of two fires is not.

**Detection rule.** Walk the cumulative burned region hour by hour. A new
*ignition component* is a connected component of ``ever(t)`` containing no cell of
``ever(t-1)`` — i.e. fire appears somewhere it could not have spread to. For each
one the module records the hour, the location, and the distance to the nearest
already-burning cell. That distance is the whole diagnostic: contagion at 1 km/h
cannot produce 47 km, so the number itself separates "filing artifact" from
"spot fire" without anyone having to remember which fire is which.

**Why this is not the same as the movie's teleport mark.** The movie flags a
large front gap in a single frame, which also fires on genuine long-range
spotting and on the ordinary ignition hour. This looks at the topology of the
burned set and answers a different question: how many INDEPENDENT fires are in
this store. The two are complementary and disagreeing is informative.

**Binding on P3.** Crossings mining must exclude inter-component jumps, or the
spot component trains on GOFER's filing convention and G4 becomes meaningless.
``ignition_components()`` returns the exact hours and cells to exclude.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wildfire_nowcast.common.components import label_components  # noqa: E402
from wildfire_nowcast.sim.reader import FireFrames, load_fire  # noqa: E402
from wildfire_nowcast.sim.style import (  # noqa: E402
    COL_BARRIER,
    COL_TEXT,
    COL_WARN,
    add_north_arrow,
    add_scale_bar,
    stamp,
)

__all__ = [
    "label_components",
    "IgnitionComponent",
    "ignition_components",
    "render_components",
    "main",
]

#: Separation at or below which two blobs in the SAME frame are one fire that
#: GOFER rasterised into two pieces. Anchored to the label noise scale CLAUDE.md
#: already states (~2 km effective GOES resolution), NOT fitted to any fire.
FRAGMENT_KM: float = 2.0

#: Separation at or above which a new component is confidently a DIFFERENT fire
#: rather than long-range spotting. Provisional, declared not fitted: an order of
#: magnitude above the label noise scale. Observed on 4 fires the gaps are
#: strongly bimodal — 4.0/4.1/5.0/6.0 km versus 14.1/46.1 km — with nothing in
#: between, so any cut in (6, 14) gives the same answer. C-3 applies: this is NOT
#: a pass/fail threshold and MUST NOT be pasted into a C2 manifest. @data owns
#: that number; this module reports the evidence for it.
SEPARATE_IGNITION_KM: float = 10.0

#: Distinct, print-safe colours for up to six components. Deliberately NOT the
#: fire-state palette: these are different FIRES, not different states, and
#: reusing the palette would invite reading component 2 as "more burned".
COMPONENT_COLORS: tuple[str, ...] = (
    "#b45309", "#1d4ed8", "#15803d", "#a21caf", "#0e7490", "#b91c1c",
)


# [A14, C0] `label_components` was HOISTED to `common/components.py` and is
# imported above. It is re-exported here (it stays in `__all__`) so every
# existing caller — this module, `sim/blockanatomy.py`, `sim/coarsen.py` — is
# unchanged. The local union-find implementation was byte-for-byte identical in
# behaviour to `data/ignitions.py`'s BFS flood fill on 417 masks (0 disagreements
# on partition structure AND exact label ids), measured BEFORE the hoist and
# pinned by `tests/test_components_differential.py`, which archives both
# originals verbatim. ADR-036 (2): two owners who cannot see each other's code
# were both computing C2's `n_ignition_components` and G4's spot-event count.


@dataclass
class IgnitionComponent:
    """One independent ignition: when it appeared, where, and how far from any fire."""

    index: int
    ignition_hour: int
    ignition_time_utc: str
    seed_cells: int
    seed_centroid_xy_m: tuple[float, float]
    km_to_nearest_burning: float
    final_cells: int
    final_area_km2: float
    merged_hour: int | None
    is_primary: bool
    #: Distance to the nearest cell of ANY OTHER component present in the SAME
    #: frame. ``km_to_nearest_burning`` is ``inf`` for everything in the ignition
    #: frame (nothing was burning YET), which made two adjacent first-frame blobs
    #: indistinguishable from a 46 km separate fire. This field is finite there.
    mutual_km_at_ignition: float = float("inf")
    #: Hours until this component becomes 8-connected to another component.
    #: ``None`` = never merged. Two blobs that merge in 6 h were always one fire.
    hours_until_merge: int | None = None
    classification: str = "primary"


def ignition_components(fire: FireFrames) -> dict[str, Any]:
    """Every independent ignition in the store, with its separation distance.

    ``km_to_nearest_burning`` for the FIRST component is ``inf`` by definition
    (nothing was burning), and that is reported rather than coerced to 0 — the
    same discipline that stopped the movie calling every ignition hour a 24 km
    teleport. A 0 there would read as "spread from an adjacent cell".
    """
    ever = fire.ever
    n_t = fire.n_hours
    cell_m = fire.geom.cell_size_m
    xs, ys = fire.geom.x_centres, fire.geom.y_centres

    components: list[IgnitionComponent] = []
    seed_rc: list[tuple[int, int]] = []
    per_hour_counts: list[int] = []
    assignment = np.zeros(ever.shape[1:], dtype=np.int32)

    prev = np.zeros(ever.shape[1:], dtype=bool)
    for t in range(n_t):
        cur = ever[t]
        labels, n = label_components(cur)
        per_hour_counts.append(n)
        for lab in range(1, n + 1):
            blob = labels == lab
            if np.any(blob & prev):
                continue  # grew from something already burning
            seed = blob & ~prev
            yy, xx = np.nonzero(seed)
            cx = float(xs[xx].mean())
            cy = float(ys[yy].mean())
            if prev.any():
                py, px = np.nonzero(prev)
                d = np.hypot(xs[px][None, :] - xs[xx][:, None], ys[py][None, :] - ys[yy][:, None])
                gap = float(d.min()) / 1000.0
            else:
                gap = float("inf")
            # Separation from any OTHER blob in the SAME frame. Without this,
            # every component in the ignition frame reports `inf` and two
            # adjacent first-frame fragments look exactly like a 46 km jump.
            other = cur & ~blob
            if other.any():
                oy, ox = np.nonzero(other)
                d2 = np.hypot(xs[ox][None, :] - xs[xx][:, None], ys[oy][None, :] - ys[yy][:, None])
                mutual = float(d2.min()) / 1000.0
            else:
                mutual = float("inf")
            components.append(
                IgnitionComponent(
                    index=len(components) + 1,
                    ignition_hour=int(t),
                    ignition_time_utc=fire.label(t),
                    seed_cells=int(seed.sum()),
                    seed_centroid_xy_m=(cx, cy),
                    km_to_nearest_burning=gap,
                    final_cells=0,
                    final_area_km2=0.0,
                    merged_hour=None,
                    is_primary=not prev.any(),
                    mutual_km_at_ignition=mutual,
                )
            )
            assignment[seed] = len(components)
            seed_rc.append((int(yy[0]), int(xx[0])))
        prev = cur.copy()

    # Merge time. Two components that become 8-connected were one fire whose
    # perimeter GOFER filed in pieces; one that never merges is a distinct fire.
    for i, comp in enumerate(components):
        for t in range(comp.ignition_hour + 1, n_t):
            labs, _ = label_components(ever[t])
            mine = labs[seed_rc[i]]
            if mine == 0:
                continue
            if any(
                labs[seed_rc[j]] == mine
                for j, other_c in enumerate(components)
                if j != i and other_c.ignition_hour <= t
            ):
                comp.hours_until_merge = int(t - comp.ignition_hour)
                break

    for comp in components:
        sep = min(comp.km_to_nearest_burning, comp.mutual_km_at_ignition)
        if comp.index == 1:
            comp.classification = "primary"
        elif sep <= FRAGMENT_KM or (comp.hours_until_merge is not None
                                    and comp.hours_until_merge <= 12):
            comp.classification = "first_frame_fragment"
        elif sep >= SEPARATE_IGNITION_KM:
            comp.classification = "separate_ignition"
        else:
            comp.classification = "spot_candidate"

    # Grow each label outward through the final footprint so every burned cell is
    # attributed to the ignition it descends from. Nearest-seed attribution over
    # the ARRIVAL ORDER, not Euclidean distance, so a cell belongs to whichever
    # fire actually reached it first.
    order = np.argsort(np.where(np.isnan(a := _arrival(ever)), np.inf, a), axis=None)
    flat = assignment.ravel()
    arr = a.ravel()
    h, w = assignment.shape
    for idx in order:
        if flat[idx] or not np.isfinite(arr[idx]):
            continue
        y, x = divmod(int(idx), w)
        best = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and flat[ny * w + nx]:
                    if np.isfinite(arr[ny * w + nx]) and arr[ny * w + nx] <= arr[idx]:
                        best = int(flat[ny * w + nx])
                        break
            if best:
                break
        flat[idx] = best

    cell_km2 = (cell_m / 1000.0) ** 2
    for comp in components:
        n_cells = int((assignment == comp.index).sum())
        comp.final_cells = n_cells
        comp.final_area_km2 = n_cells * cell_km2

    merged_hour = None
    for t, n in enumerate(per_hour_counts):
        if n == 1 and t > 0 and per_hour_counts[t - 1] > 1:
            merged_hour = t
            break

    by_class: dict[str, int] = {}
    for c in components:
        by_class[c.classification] = by_class.get(c.classification, 0) + 1

    return {
        "fire_id": fire.fire_id,
        # Raw topology: connected regions that appeared where fire could not have
        # spread to. This is NOT C2's n_ignition_components — see the breakdown.
        "n_components_detected": len(components),
        "n_ignition_components": len(components),
        # What the topology count is actually made of. A first_frame_fragment is
        # ONE fire filed in two pieces; a spot_candidate is the phenomenon the P2
        # spot component must LEARN and must NOT be excluded from P3 mining.
        "classification_counts": by_class,
        "candidate_separate_ignitions": 1 + by_class.get("separate_ignition", 0),
        "fragment_km_threshold": FRAGMENT_KM,
        "separate_ignition_km_threshold": SEPARATE_IGNITION_KM,
        "components": [asdict(c) for c in components],
        "components_per_hour": per_hour_counts,
        "max_components_at_once": int(max(per_hour_counts)) if per_hour_counts else 0,
        "merged_hour": merged_hour,
        "max_ignition_gap_km": max(
            (c.km_to_nearest_burning for c in components if np.isfinite(c.km_to_nearest_burning)),
            default=0.0,
        ),
        "assignment": assignment,
    }


def _arrival(ever: np.ndarray) -> np.ndarray:
    any_ever = ever.any(axis=0)
    first = np.argmax(ever, axis=0).astype(np.float64)
    return np.where(any_ever, first, np.nan)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def render_components(fire: FireFrames, result: dict[str, Any], out: str | Path) -> Path:
    """Left: the fire as two (or three) fires. Right: the frame the jump happens in."""
    geom = fire.geom
    assignment = result["assignment"]
    comps = result["components"]
    n = result["n_ignition_components"]

    fig = plt.figure(figsize=(15.4, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.0, 0.42],
                          hspace=0.30, wspace=0.22, left=0.05, right=0.98, top=0.83, bottom=0.09)

    # -- (1) final footprint, coloured by ignition component --------------
    ax = fig.add_subplot(gs[:, 0])
    rgba = np.zeros((*assignment.shape, 4), dtype=float)
    rgba[..., :3] = 0.91
    rgba[..., 3] = np.where(assignment > 0, 1.0, 0.10)
    for c in comps:
        col = matplotlib.colors.to_rgb(COMPONENT_COLORS[(c["index"] - 1) % len(COMPONENT_COLORS)])
        rgba[assignment == c["index"], :3] = col
    ax.imshow(rgba, **geom.imshow_kwargs)
    barrier = np.zeros((*fire.barrier.shape, 4), dtype=float)
    barrier[fire.barrier] = (*matplotlib.colors.to_rgb(COL_BARRIER), 0.28)
    ax.imshow(barrier, **geom.imshow_kwargs)

    _SHORT = {
        "primary": "primary",
        "first_frame_fragment": "same fire, filed in 2 pieces",
        "separate_ignition": "SEPARATE FIRE",
        "spot_candidate": "spot candidate — KEEP for P3",
    }
    # Alternate the label offset so co-located first-frame fragments do not
    # print on top of each other, which is exactly what they did on
    # 2020_july_complex (two blobs 1 km apart, three overlapping labels).
    for k, c in enumerate(comps):
        x, y = c["seed_centroid_xy_m"]
        col = COMPONENT_COLORS[(c["index"] - 1) % len(COMPONENT_COLORS)]
        ax.plot([x], [y], marker="*", ms=17, color=col, mec="white", mew=1.1, zorder=6)
        dy = 11 if k % 2 == 0 else -30
        ax.annotate(
            f"ignition {c['index']}  h{c['ignition_hour']}  {c['final_area_km2']:.0f} km²\n"
            f"{_SHORT.get(c.get('classification', ''), '')}",
            (x, y), textcoords="offset points", xytext=(11, dy), fontsize=8, color=col,
            fontweight="bold", zorder=7,
        )

    # Annotate the component with the LARGEST finite separation, not comps[1].
    # comps[1] on 2020_july_complex is an adjacent first-frame fragment whose
    # gap is `inf`, so the old code printed "inf km ... NOT spotting" across the
    # ignition labels and said it of the one component it is NOT true of.
    jumps = [c for c in comps[1:] if np.isfinite(c["km_to_nearest_burning"])]
    if jumps:
        far = max(jumps, key=lambda c: c["km_to_nearest_burning"])
        src = comps[0]["seed_centroid_xy_m"]
        b = far["seed_centroid_xy_m"]
        ax.annotate(
            "", xy=b, xytext=src,
            arrowprops={"arrowstyle": "<->", "color": COL_WARN, "lw": 1.6, "ls": "--"},
        )
        verdict = (
            "NOT spotting — a separate fire"
            if far.get("classification") == "separate_ignition"
            else "too far for 1 h of contagion — SPOT CANDIDATE, keep it"
        )
        ax.text(
            (src[0] + b[0]) / 2, (src[1] + b[1]) / 2,
            f"  {far['km_to_nearest_burning']:.1f} km to the nearest burning cell\n"
            f"  at hour {far['ignition_hour']} — {verdict}",
            color=COL_WARN, fontsize=9, fontweight="bold", va="center",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": COL_WARN, "lw": 0.8,
                  "boxstyle": "round,pad=0.3"},
        )
    ax.set_xticks([])
    ax.set_yticks([])
    add_north_arrow(ax)
    add_scale_bar(ax, geom)
    counts = result.get("classification_counts", {})
    bits = [f"{n} components detected"]
    if counts.get("separate_ignition"):
        bits.append(f"{counts['separate_ignition']} SEPARATE FIRE(S)")
    if counts.get("spot_candidate"):
        bits.append(f"{counts['spot_candidate']} spot candidate(s)")
    if counts.get("first_frame_fragment"):
        bits.append(f"{counts['first_frame_fragment']} same-fire fragment(s)")
    ax.set_title(f"{fire.fire_id}: " + ",  ".join(bits), fontsize=11)

    # -- (2)(3) the frames either side of the FIRST GENUINELY DETACHED
    # component. `not is_primary` picked component 2, which on july_complex is a
    # first-frame fragment 1 km away -- the panels then showed two frames in
    # which nothing visible happens. Prefer a separate fire, then a spot
    # candidate, and fall back to any non-primary component.
    ranked = sorted(
        (c for c in comps if not c["is_primary"]),
        key=lambda c: (
            {"separate_ignition": 0, "spot_candidate": 1}.get(c.get("classification"), 2),
            -c["km_to_nearest_burning"] if np.isfinite(c["km_to_nearest_burning"]) else 0.0,
        ),
    )
    second = ranked[0] if ranked else None
    for k, ax_pos in enumerate((gs[0, 1], gs[0, 2])):
        ax = fig.add_subplot(ax_pos)
        if second is None:
            ax.text(0.5, 0.5, "single ignition — nothing to show", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        t = max(second["ignition_hour"] - 1 + k, 0)
        rgb = np.zeros((*assignment.shape, 4), dtype=float)
        rgb[..., :3] = 0.93
        rgb[..., 3] = 0.10
        burned = fire.ever[t]
        rgb[burned, 3] = 1.0
        for c in comps:
            col = matplotlib.colors.to_rgb(
                COMPONENT_COLORS[(c["index"] - 1) % len(COMPONENT_COLORS)]
            )
            rgb[burned & (assignment == c["index"]), :3] = col
        ax.imshow(rgb, **geom.imshow_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"hour {t} — {fire.label(t)}\n{int(burned.sum())} cells burned, "
            f"{result['components_per_hour'][t]} component(s)",
            fontsize=9,
            color=COL_WARN if k == 1 else COL_TEXT,
        )

    # -- (4) component count over time ------------------------------------
    ax = fig.add_subplot(gs[1, 1:])
    counts = result["components_per_hour"]
    ax.step(range(len(counts)), counts, where="post", color="#0f766e", lw=1.6)
    ax.fill_between(range(len(counts)), counts, step="post", alpha=0.16, color="#0f766e")
    for c in comps:
        col = COMPONENT_COLORS[(c["index"] - 1) % len(COMPONENT_COLORS)]
        ax.axvline(c["ignition_hour"], color=col, lw=1.2, ls="--")
    if result["merged_hour"] is not None:
        ax.axvline(result["merged_hour"], color=COL_WARN, lw=1.2)
        ax.text(result["merged_hour"], max(counts) * 0.92, " components merge",
                fontsize=7, color=COL_WARN)
    ax.set_xlabel("hour")
    ax.set_ylabel("connected\ncomponents")
    ax.set_ylim(0, max(counts) + 0.6)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        "C2 [v2.7] n_ignition_components — separate lightning ignitions filed under one fire id\n"
        "BINDING ON P3: crossings mining must EXCLUDE inter-component jumps, or the spot model "
        "trains on GOFER's filing convention and G4 becomes meaningless",
        fontsize=11,
    )
    stamp(fig, "C1 tensor only, channels by name; component = connected region of ever-burned "
               "with no cell in the previous hour")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False deliberately. This CLI takes --outdir, but the module
    # docstring and every other CLI in sim/ take --out, and argparse's default
    # prefix matching silently accepted `--out figures/x.png` as `--outdir`,
    # creating a DIRECTORY named `x.png` and writing the figure inside it. A
    # mistyped flag must fail loudly, not quietly relocate the evidence.
    ap = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.components", allow_abbrev=False
    )
    ap.add_argument("--tensor", action="append", required=True)
    ap.add_argument("--outdir", default="reports/figures",
                    help="directory for the figures + ignition_components.json")
    ap.add_argument("--manifest-check", action="store_true",
                    help="compare the detected count with C2 n_ignition_components")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"kind": "simviz_ignition_components", "fires": {}}
    rc = 0
    for tensor in args.tensor:
        fire = load_fire(tensor)
        result = ignition_components(fire)
        fig = render_components(fire, result, outdir / f"components_{fire.fire_id}.png")
        declared: Any = None
        manifest = Path(tensor).parent / "manifest.json"
        if manifest.is_file():
            m = json.loads(manifest.read_text())
            declared = m.get("n_ignition_components", (m.get("provenance") or {}).get(
                "n_ignition_components"))
        entry = {k: v for k, v in result.items() if k != "assignment"}
        entry["c2_declared_n_ignition_components"] = declared
        entry["figure"] = str(fig)
        summary["fires"][fire.fire_id] = entry
        gaps = ", ".join(
            f"#{c['index']}@h{c['ignition_hour']} {c['km_to_nearest_burning']:.1f} km"
            for c in result["components"]
        )
        print(f"[components] {fire.fire_id}: detected {result['n_ignition_components']} "
              f"(C2 declares {declared})  [{gaps}]  -> {fig}")
        if args.manifest_check and declared != result["n_ignition_components"]:
            print(
                f"[components] CONTRACT: {fire.fire_id} manifest declares "
                f"{declared!r} but the tensor contains {result['n_ignition_components']}. "
                "C2 [v2.7] requires this key; P3 crossings mining reads it."
            )
            rc = 1
    (outdir / "ignition_components.json").write_text(json.dumps(summary, indent=1) + "\n")
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
