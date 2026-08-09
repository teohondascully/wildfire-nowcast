"""Shared rendering conventions: palette, geometry, and the north-up guard.

Everything visual in :mod:`wildfire_nowcast.sim` goes through this module so
that a convention is decided once rather than re-derived per figure. Two of
these conventions are load-bearing and are enforced, not documented:

**C1.4 orientation.** ``y`` DESCENDS and ``x`` ASCENDS in a C1 store, so array
row 0 is the NORTHERNMOST row. Get this backwards and every fire renders
mirrored — which looks plausible and survives review. :func:`plot_extent`
therefore *derives* the extent from the coordinate values and raises on a store
whose axes do not descend/ascend, instead of hardcoding an ``origin`` and
hoping. The pairing ``origin="upper"`` + ``extent=(x0, x1, y0, y1)`` with
``y1 > y0`` places ``arr[0, 0]`` at the north-west corner, which is what we
want; any other pairing is a bug.

**Data coordinates, not pixel indices.** Every axis is drawn in EPSG:5070
metres. That is not cosmetic: the wind quiver plots ``v10`` (northward) as the
vertical component, and matplotlib draws +y upward. In metre coordinates that
is north. In row-index coordinates it would be *south*, and the whole wind
field would silently point the wrong way while still looking like a wind field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

__all__ = [
    "STATE_COLORS",
    "STATE_LABELS",
    "state_cmap",
    "BURN_PROB_CMAP",
    "ARRIVAL_CMAP",
    "COL_FRONTIER",
    "COL_BARRIER",
    "COL_TRUTH",
    "COL_MEMBER",
    "COL_WIND",
    "COL_TEXT",
    "COL_WARN",
    "PlotGeometry",
    "plot_extent",
    "assert_north_up",
    "quiver_grid",
    "add_north_arrow",
    "add_scale_bar",
    "stamp",
]

# -- palette ---------------------------------------------------------------
# Chosen so that (a) burned-out stays visible under the burning layer, which is
# what makes a dormant frame legible rather than blank, and (b) the three states
# are distinguishable in greyscale print: light / bright / dark.
STATE_COLORS: tuple[str, str, str] = (
    "#e8e2d6",  # 0 unburned  — pale, recedes
    "#ff5a1f",  # 1 burning   — the only saturated warm colour in the figure
    "#3a3128",  # 2 burned-out — dark char, deliberately NOT background-coloured
)
STATE_LABELS: tuple[str, str, str] = ("unburned", "burning", "burned-out")

COL_FRONTIER = "#ffd23f"  # frontier of the burned region (C1.1 contagion source)
COL_BARRIER = "#2b6cb0"  # water / barrier mask
COL_TRUTH = "#111111"
COL_MEMBER = "#c2410c"
COL_WIND = "#1f4e5f"
COL_TEXT = "#1a1a1a"
COL_WARN = "#b91c1c"

#: Burn probability. White at 0 so "never burns in any member" reads as absence,
#: not as a low-but-present value.
BURN_PROB_CMAP = LinearSegmentedColormap.from_list(
    "burn_prob",
    ["#ffffff", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"],
)

#: Arrival time. Perceptually ordered so "early" reads as urgent.
ARRIVAL_CMAP = LinearSegmentedColormap.from_list(
    "arrival",
    ["#7f0000", "#e34a33", "#fdbb84", "#c7e9b4", "#2c7fb8"],
)


def state_cmap() -> ListedColormap:
    """Discrete 3-colour map for ``fire_state`` values ``{0, 1, 2}``."""
    return ListedColormap(list(STATE_COLORS), name="fire_state")


# -- geometry --------------------------------------------------------------


@dataclass(frozen=True)
class PlotGeometry:
    """Axis extent in CRS metres plus the coordinate arrays that produced it.

    ``extent`` is matplotlib's ``(left, right, bottom, top)`` in EPSG:5070
    metres, i.e. cell OUTER EDGES, and is only ever valid together with
    ``origin="upper"``.
    """

    extent: tuple[float, float, float, float]
    x_centres: np.ndarray
    y_centres: np.ndarray
    cell_size_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.y_centres.size), int(self.x_centres.size))

    @property
    def imshow_kwargs(self) -> dict[str, Any]:
        """The only correct way to hand a ``(y, x)`` array to ``imshow`` here."""
        return {"extent": self.extent, "origin": "upper", "interpolation": "nearest"}

    @property
    def width_km(self) -> float:
        return (self.extent[1] - self.extent[0]) / 1000.0

    @property
    def height_km(self) -> float:
        return (self.extent[3] - self.extent[2]) / 1000.0


def assert_north_up(x: np.ndarray, y: np.ndarray) -> None:
    """Raise unless ``x`` strictly ascends and ``y`` strictly descends (C1.4).

    Refusing to render is the right behaviour: a mirrored fire is worse than no
    fire, because it is reviewable and wrong rather than obviously absent.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size > 1 and not np.all(np.diff(x) > 0):
        raise ValueError(
            "C1.4 violated: `x` must strictly ASCEND (west -> east). Refusing to render — "
            "this would mirror the fire east/west."
        )
    if y.size > 1 and not np.all(np.diff(y) < 0):
        raise ValueError(
            "C1.4 violated: `y` must strictly DESCEND (north -> south, row 0 = north). "
            "Refusing to render — this would mirror the fire north/south."
        )


def plot_extent(x: np.ndarray, y: np.ndarray, cell_size_m: float = 1000.0) -> PlotGeometry:
    """Build a :class:`PlotGeometry` from C1 cell-centre coordinates.

    The extent is derived from the coordinates themselves (never assumed), and
    the C1.4 orientation is checked first, so it is not possible to get a
    silently mirrored figure out of this function.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert_north_up(x, y)
    half = 0.5 * float(cell_size_m)
    return PlotGeometry(
        extent=(
            float(x[0] - half),
            float(x[-1] + half),
            float(y[-1] - half),  # bottom = SOUTHERNMOST edge (y descends -> last)
            float(y[0] + half),  # top    = NORTHERNMOST edge (y descends -> first)
        ),
        x_centres=x,
        y_centres=y,
        cell_size_m=float(cell_size_m),
    )


def quiver_grid(
    geom: PlotGeometry, target: int = 14
) -> tuple[np.ndarray, np.ndarray, slice, slice, int]:
    """Subsampled ``(X, Y, row_slice, col_slice, step)`` for a wind quiver.

    Returns coordinates in METRES (see the module docstring: this is what makes
    the northward wind component render upward). ``step`` is the sample spacing
    in cells; callers size arrows from it so arrows never overrun their spacing.
    """
    ny, nx = geom.shape
    step = max(1, int(round(max(ny, nx) / max(target, 1))))
    rows = slice(step // 2, ny, step)
    cols = slice(step // 2, nx, step)
    xs = geom.x_centres[cols]
    ys = geom.y_centres[rows]
    return (*np.meshgrid(xs, ys), rows, cols, step)


# -- annotations -----------------------------------------------------------


def add_north_arrow(ax: Any, xy: tuple[float, float] = (0.955, 0.90)) -> None:
    """A north arrow. Cheap insurance against an unnoticed vertical flip."""
    ax.annotate(
        "N",
        xy=xy,
        xytext=(xy[0], xy[1] - 0.085),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        color=COL_TEXT,
        arrowprops={"arrowstyle": "-|>", "color": COL_TEXT, "lw": 1.1},
    )


_NICE_KM = (1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0)


def add_scale_bar(ax: Any, geom: PlotGeometry, km: float | None = None) -> None:
    """Horizontal scale bar in km, drawn in data (metre) coordinates.

    Snapped to a round number — "26 km" is a bar nobody can read a distance off.
    """
    if km is None:
        target = geom.width_km / 5.0
        km = min(_NICE_KM, key=lambda v: abs(v - target))
    x0, x1, y0, y1 = geom.extent
    pad_y = 0.045 * (y1 - y0)
    # Bottom-RIGHT: bottom-left is where the per-frame status block lives.
    bx0 = x1 - 0.04 * (x1 - x0) - km * 1000.0
    by = y0 + pad_y
    ax.plot([bx0, bx0 + km * 1000.0], [by, by], color=COL_TEXT, lw=2.0, solid_capstyle="butt")
    ax.text(
        bx0 + km * 500.0,
        by + 0.012 * (y1 - y0),
        f"{km:g} km",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COL_TEXT,
    )


def stamp(fig: Any, text: str, *, color: str = "#6b7280") -> None:
    """Provenance stamp in the figure footer.

    Every figure this package emits carries what it was made from and under what
    pacing, because a screenshot outlives its caption.
    """
    fig.text(0.005, 0.004, text, fontsize=6, color=color, ha="left", va="bottom")
