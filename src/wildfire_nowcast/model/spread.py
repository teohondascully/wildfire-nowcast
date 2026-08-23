"""Elliptical spread geometry for the C5 wind-advected-ellipse baseline.

This module holds the *physics caricature* the ellipse baseline is built from,
separated out because the learned kernel is required to be initialised to the
same shape: CLAUDE.md specifies "short-range anisotropic contagion (CNN over
burning neighbourhood, **initialized elliptical-Gaussian, stretched by
wind/slope**)". Keeping the ellipse's geometry in one importable place means the
kernel is initialised from the *same* function the baseline is scored with, so
"the CNN beat the ellipse" cannot quietly mean "the CNN beat a different
ellipse".

What is standard here and what is ours
--------------------------------------
* **Length-to-breadth from wind speed** follows Anderson (1983), the relation
  used by FARSITE/BehavePlus. Standard.
* **Polar rate of spread from the rear focus** is the textbook ellipse:
  ``r(theta) = a(1-e^2)/(1 - e cos theta)``, exact, not an approximation.
* **Dead fuel moisture damping** is Rothermel's cubic in ``FM / M_x``. Standard.
* **Wind/slope combination** as a vector sum of the midflame wind and an
  upslope "slope-equivalent wind" is the usual FARSITE device. Standard in
  form; the coefficient is ours.
* **The head rate of spread law and the per-fuel-group multipliers are OURS and
  are CARICATURES.** They are a 4-parameter monotone law, not Rothermel. This
  is deliberate: ELMFIRE with default Rothermel parameters is a separate,
  non-negotiable baseline that sim runs (CLAUDE.md), and re-deriving a
  worse Rothermel here would just be a third thing to disagree with. What
  matters is that these parameters are FIT on training fires
  (:meth:`EllipseParams.fitted`), so the ellipse is a calibrated floor rather
  than a strawman. A baseline nobody tuned is not a floor, it is a courtesy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = [
    "EllipseParams",
    "FUEL_GROUPS",
    "fuel_group",
    "fuel_group_fields",
    "length_to_breadth",
    "ellipse_ros_factor",
    "moisture_damping",
    "midflame_wind",
    "slope_equivalent_wind",
    "effective_wind",
    "head_rate_of_spread",
]

# --------------------------------------------------------------------------
# FBFM40 fuel groups (Scott & Burgan 2005 numbering)
# --------------------------------------------------------------------------

#: ``group -> (relative head ROS multiplier, moisture of extinction %)``.
#: Ordinal, not Rothermel outputs: grass carries fire fastest and driest,
#: timber litter slowest and wettest. The multipliers are fit parameters with
#: these as priors.
FUEL_GROUPS: dict[str, tuple[float, float]] = {
    "NB": (0.00, 1.0),  # 91-99 non-burnable: urban, snow, agriculture, water, bare
    "GR": (1.00, 15.0),  # 101-109 grass
    "GS": (0.80, 20.0),  # 121-124 grass-shrub
    "SH": (0.70, 25.0),  # 141-149 shrub (chaparral: the Kincade/Glass fuel)
    "TU": (0.45, 25.0),  # 161-165 timber-understory
    "TL": (0.25, 30.0),  # 181-189 timber litter
    "SB": (0.50, 25.0),  # 201-204 slash-blowdown
}

_GROUP_ORDER: tuple[str, ...] = ("NB", "GR", "GS", "SH", "TU", "TL", "SB")


def fuel_group(code: int) -> str:
    """FBFM40 class id -> group name. Unknown / out-of-range codes are NB.

    Treating an unrecognised code as non-burnable is the conservative choice:
    it makes an ingestion bug show up as a fire that refuses to spread, which is
    obvious, instead of as a fire that spreads through a parking lot, which is
    not.
    """
    c = int(code)
    if 91 <= c <= 99:
        return "NB"
    if 101 <= c <= 109:
        return "GR"
    if 121 <= c <= 124:
        return "GS"
    if 141 <= c <= 149:
        return "SH"
    if 161 <= c <= 165:
        return "TU"
    if 181 <= c <= 189:
        return "TL"
    if 201 <= c <= 204:
        return "SB"
    return "NB"


def fuel_group_fields(
    fuel_model_id: np.ndarray,
    multipliers: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(relative_ros, moisture_of_extinction_pct, burnable_mask)`` per cell.

    ``fuel_model_id`` is C1 channel 9, "int-as-float". C1.5 enforces that it is
    integral, so rounding here is a formality rather than a repair.
    """
    codes = np.rint(np.asarray(fuel_model_id, dtype=np.float64)).astype(np.int64)
    mults = dict(multipliers or {})
    rel = np.zeros(codes.shape, dtype=np.float64)
    mx = np.full(codes.shape, 25.0, dtype=np.float64)
    for name in _GROUP_ORDER:
        default_rel, default_mx = FUEL_GROUPS[name]
        member = _group_mask(codes, name)
        rel[member] = float(mults.get(name, default_rel))
        mx[member] = default_mx
    return rel, mx, rel > 0.0


def _group_mask(codes: np.ndarray, name: str) -> np.ndarray:
    ranges = {
        "NB": [(91, 99)],
        "GR": [(101, 109)],
        "GS": [(121, 124)],
        "SH": [(141, 149)],
        "TU": [(161, 165)],
        "TL": [(181, 189)],
        "SB": [(201, 204)],
    }[name]
    mask = np.zeros(codes.shape, dtype=bool)
    for lo, hi in ranges:
        mask |= (codes >= lo) & (codes <= hi)
    if name == "NB":
        # Everything unrecognised falls through to non-burnable.
        known = np.zeros(codes.shape, dtype=bool)
        for other in _GROUP_ORDER:
            if other != "NB":
                known |= _group_mask(codes, other)
        mask |= ~known
    return mask


# --------------------------------------------------------------------------
# ellipse geometry
# --------------------------------------------------------------------------


def length_to_breadth(u_mid_ms: np.ndarray | float, *, cap: float = 8.0) -> np.ndarray:
    """Length-to-breadth ratio of the fire ellipse from midflame wind speed.

    Anderson (1983), as used by FARSITE and BehavePlus, with the wind in mi/h::

        LB = 0.936 exp(0.2566 U) + 0.461 exp(-0.1548 U) - 0.397

    ``cap`` bounds it (Anderson's own fit is not intended far beyond ~8), which
    also keeps the eccentricity away from 1 where the polar form stiffens.

    Reference: Anderson, H.E. (1983), *Predicting wind-driven wildland fire size
    and shape*, USDA Forest Service Research Paper INT-305.
    """
    u_mph = np.asarray(u_mid_ms, dtype=np.float64) * 2.236936
    lb = 0.936 * np.exp(0.2566 * u_mph) + 0.461 * np.exp(-0.1548 * u_mph) - 0.397
    return np.clip(lb, 1.0, float(cap))


def ellipse_ros_factor(lb: np.ndarray | float, cos_theta: np.ndarray | float) -> np.ndarray:
    """Rate of spread in direction ``theta`` as a fraction of the HEAD rate.

    The fire ellipse is generated from its rear focus (the ignition point), so
    the polar form from that focus is exact::

        r(theta) = a (1 - e^2) / (1 - e cos theta)

    with ``theta`` measured from the head direction. Normalising by
    ``r(0) = a(1 + e)`` gives a factor of 1 at the head and
    ``(1 - e)/(1 + e)`` at the back. Eccentricity follows from the
    length-to-breadth ratio alone: ``e = sqrt(1 - LB^-2)``.
    """
    lb_arr = np.clip(np.asarray(lb, dtype=np.float64), 1.0, None)
    ecc = np.sqrt(np.clip(1.0 - 1.0 / (lb_arr * lb_arr), 0.0, 1.0 - 1e-12))
    cos_t = np.clip(np.asarray(cos_theta, dtype=np.float64), -1.0, 1.0)
    return (1.0 - ecc) / (1.0 - ecc * cos_t)


def moisture_damping(fm_pct: np.ndarray | float, mx_pct: np.ndarray | float) -> np.ndarray:
    """Rothermel's dead-fuel moisture damping coefficient.

    ``eta_M = 1 - 2.59 r + 5.11 r^2 - 3.52 r^3`` with ``r = min(FM/M_x, 1)``,
    clipped to ``[0, 1]``: at the moisture of extinction the fuel stops carrying
    fire.

    Reference: Rothermel, R.C. (1972), *A mathematical model for predicting fire
    spread in wildland fuels*, USDA Forest Service Research Paper INT-115.
    """
    ratio = np.clip(
        np.asarray(fm_pct, dtype=np.float64)
        / np.maximum(np.asarray(mx_pct, dtype=np.float64), 1e-6),
        0.0,
        1.0,
    )
    eta = 1.0 - 2.59 * ratio + 5.11 * ratio**2 - 3.52 * ratio**3
    return np.clip(eta, 0.0, 1.0)


def midflame_wind(
    u10: np.ndarray,
    v10: np.ndarray,
    canopy_cover_pct: np.ndarray | None = None,
    *,
    waf_open: float = 0.4,
    waf_closed: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """10 m wind -> midflame wind vector ``(u, v)`` in m/s.

    Fire spread responds to wind at flame height, not at 10 m. The wind
    adjustment factor is interpolated linearly from ``waf_open`` at 0% canopy to
    ``waf_closed`` under full canopy - the standard sheltered/unsheltered split,
    coarsened to a ramp because C1 gives canopy cover as a percentage and not
    the crown-fill/height triple the full WAF formula needs.
    """
    u = np.asarray(u10, dtype=np.float64)
    v = np.asarray(v10, dtype=np.float64)
    if canopy_cover_pct is None:
        waf = np.full(u.shape, float(waf_open))
    else:
        cc = np.clip(np.asarray(canopy_cover_pct, dtype=np.float64) / 100.0, 0.0, 1.0)
        waf = float(waf_open) + cc * (float(waf_closed) - float(waf_open))
    return u * waf, v * waf


def slope_equivalent_wind(
    slope_deg: np.ndarray,
    aspect_sin: np.ndarray,
    aspect_cos: np.ndarray,
    *,
    k_slope: float = 9.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Upslope "wind" vector ``(u, v)`` in m/s equivalent, from terrain alone.

    Rothermel's slope factor goes as ``tan^2(phi)``, so the equivalent wind is
    taken as ``k_slope * tan^2(phi)`` pointing UPSLOPE. C1 channels 7/8 encode
    the DOWNSLOPE azimuth (the direction the slope faces) as
    ``(sin, cos) = (east, north)`` components, with ``(0, 0)`` reserved for flat
    cells - so upslope is simply the negation, and a flat cell contributes zero
    without a special case.
    """
    tan_phi = np.tan(np.radians(np.clip(np.asarray(slope_deg, dtype=np.float64), 0.0, 89.0)))
    magnitude = float(k_slope) * tan_phi**2
    # Downslope unit vector is (aspect_sin, aspect_cos) in (east, north).
    return -magnitude * np.asarray(aspect_sin, dtype=np.float64), -magnitude * np.asarray(
        aspect_cos, dtype=np.float64
    )


def effective_wind(
    u_mid: np.ndarray,
    v_mid: np.ndarray,
    u_slope: np.ndarray,
    v_slope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vector-sum wind and slope -> ``(speed, unit_east, unit_north)``.

    The FARSITE device: slope is expressed as an equivalent wind and added as a
    vector, so a cross-slope wind produces a head direction between the two
    rather than either one winning. Flat, calm cells get a well-defined but
    arbitrary head direction (east) with zero speed, which the ellipse then
    renders isotropic (LB = 1) regardless.
    """
    ue = np.asarray(u_mid, dtype=np.float64) + np.asarray(u_slope, dtype=np.float64)
    vn = np.asarray(v_mid, dtype=np.float64) + np.asarray(v_slope, dtype=np.float64)
    speed = np.hypot(ue, vn)
    safe = np.maximum(speed, 1e-9)
    unit_e = np.where(speed > 1e-9, ue / safe, 1.0)
    unit_n = np.where(speed > 1e-9, vn / safe, 0.0)
    return speed, unit_e, unit_n


# --------------------------------------------------------------------------
# head rate of spread + the parameter object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EllipseParams:
    """The wind-advected ellipse's free parameters.

    Defaults are physically-scaled PRIORS, not measurements; :meth:`fit`
    replaces them from training fires only. They are anchored on two published
    reference points for shrub fuels (Scott & Burgan 2005 SH group, dry), with
    the wind speed taken at MIDFLAME height, not at 10 m:

    ===================  ====================  ==========================
    midflame wind        head ROS these give   published order of magnitude
    ===================  ====================  ==========================
    0 m/s (calm)         ~0.06 m/s = 215 m/h   no-wind shrub ~5-6 m/min
    4.5 m/s (10 mi/h)    ~1.0 m/s = 3.6 km/h   ~60-100 m/min
    6 m/s (Diablo)       ~1.6 m/s = 5.6 km/h   consistent with the observed
                                               Kincade +31.9 km^2 in
                                               the single hour ending
                                               2019-10-27T13:00Z
    ===================  ====================  ==========================

    The 10 m -> midflame conversion is where this is easy to get wrong by an
    order of magnitude: RTMA's 15 m/s Diablo wind is ~6 m/s at midflame after
    ``waf_open``, and calibrating ``u_ref_ms`` against the 10 m value instead
    silently makes the baseline ~10x too slow.

    ``r0_ms``
        No-wind, no-slope head rate of spread (m/s) in the reference fuel.
    ``u_ref_ms``
        Wind scale of the wind response.
    ``wind_exponent``
        Power of the wind response. ~1.5-2 is the usual Rothermel-fit range.
    ``k_slope``
        Slope-to-equivalent-wind coefficient (m/s per unit ``tan^2 phi``).
    ``waf_open`` / ``waf_closed``
        Wind adjustment factors (10 m -> midflame) at 0% and 100% canopy.
    ``lb_cap``
        Maximum length-to-breadth; bounds Anderson's extrapolation.
    ``fuel_multipliers``
        Per-FBFM40-group relative ROS. Empty means "use :data:`FUEL_GROUPS`".
    """

    r0_ms: float = 0.095
    u_ref_ms: float = 1.0
    wind_exponent: float = 1.8
    k_slope: float = 9.0
    waf_open: float = 0.40
    waf_closed: float = 0.15
    lb_cap: float = 8.0
    fuel_multipliers: tuple[tuple[str, float], ...] = ()

    @property
    def multipliers(self) -> dict[str, float]:
        return {name: value for name, value in self.fuel_multipliers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "r0_ms": self.r0_ms,
            "u_ref_ms": self.u_ref_ms,
            "wind_exponent": self.wind_exponent,
            "k_slope": self.k_slope,
            "waf_open": self.waf_open,
            "waf_closed": self.waf_closed,
            "lb_cap": self.lb_cap,
            "fuel_multipliers": {k: v for k, v in self.fuel_multipliers},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EllipseParams:
        data = dict(payload)
        mults = data.pop("fuel_multipliers", {}) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(fuel_multipliers=tuple(sorted(mults.items())), **known)

    def scaled(self, ros_multiplier: float) -> EllipseParams:
        """Same shape, head rate scaled - the ensemble's shared innovation."""
        return replace(self, r0_ms=self.r0_ms * float(ros_multiplier))

    def max_head_ros_ms(self, max_wind_ms: float = 30.0) -> float:
        """Upper bound on head ROS, used to size the integrator's substeps."""
        return float(
            self.r0_ms * (1.0 + (max_wind_ms / max(self.u_ref_ms, 1e-6)) ** self.wind_exponent)
        )


def head_rate_of_spread(
    u_eff_ms: np.ndarray,
    fuel_relative: np.ndarray,
    eta_moisture: np.ndarray,
    params: EllipseParams,
) -> np.ndarray:
    """Head rate of spread in m/s.

    ``R_h = r0 * fuel * eta_M * (1 + (U / u_ref) ** p)``

    Monotone increasing in wind, increasing in fuel loading proxy, decreasing in
    moisture, and exactly zero in non-burnable fuel or at the moisture of
    extinction. That is the whole scientific content: it is a shape with four
    knobs, and the knobs are fit. It is NOT Rothermel and must never be
    described as such - ELMFIRE is the Rothermel baseline (CLAUDE.md).
    """
    u = np.clip(np.asarray(u_eff_ms, dtype=np.float64), 0.0, None)
    wind_response = 1.0 + (u / max(params.u_ref_ms, 1e-6)) ** params.wind_exponent
    return params.r0_ms * np.asarray(fuel_relative) * np.asarray(eta_moisture) * wind_response


def substeps_for(max_ros_ms: float, cell_size_m: float, *, cap: int = 16) -> int:
    """Sub-hourly steps needed so the front is never speed-limited by the grid.

    A one-cell-per-step accumulator caps spread at one cell per step. Kincade's
    Diablo hour moved the head several km, so a single step per hour would
    silently truncate exactly the event the model exists to predict. This sizes
    the substep count to the fastest head rate present.
    """
    per_hour_cells = max_ros_ms * 3600.0 / max(cell_size_m, 1e-6)
    return int(np.clip(math.ceil(per_hour_cells), 1, cap))
