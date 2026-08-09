"""C5 baseline 2 of 2 — the wind-advected ellipse. The physics floor.

CLAUDE.md lists this as non-negotiable, and G2 is defined against it: the
contagion kernel must beat the ellipse on held-out fires or the project stops
and explains why. That makes one design decision more important than all the
others: **the ellipse must be calibrated, not a strawman.** A baseline nobody
tuned is not a floor, it is a courtesy, and beating it proves nothing.

**The calibration rule is C6.2 / ADR-011 and is not negotiable inside this
file: the head-rate scale reproduces the observed mean hourly growth of the
TRAIN fires** (:meth:`calibrate_to_growth`). It is NEVER fitted by a pixelwise
score — :meth:`fit_by_brier` is kept only as the evidence for that clause, and
its own docstring records why using it voids the gate. The chosen scale is
recorded with the fire ids it was calibrated on, so a leave-fire-out violation
is auditable after the fact rather than trusted at the time.

How it spreads
--------------
Each hour, every unburned cell accumulates ignition progress from its eight
neighbours. The rate into a cell is the elliptical rate of spread evaluated at
that cell — head rate from its own fuel, moisture and effective wind
(:mod:`wildfire_nowcast.model.spread`), reduced by the ellipse's polar factor
according to the angle between the head direction and the direction the fire is
arriving from. A cell ignites when accumulated progress reaches 1.

Two properties of that scheme matter and are not accidental:

* **It can produce exactly zero growth.** At 1 km cells and hourly steps, a
  realistic rate of spread is usually a small fraction of a cell per hour, so
  the accumulator sits below 1 and nothing ignites. A dilation-based baseline
  cannot do this — it advances at least one cell per hour, roughly 1 km/h, and
  would over-predict growth in the ~79% of hours where GOFER records bitwise
  zero (insights/data item 1). Getting the zero-growth majority right is most of
  what a baseline has to do here.
* **It seeds from the FRONTIER OF THE BURNED REGION, not from state 1.** C1.1 is
  explicit that state 1 is legitimately empty in 6-37% of frames after a long
  dormancy (Kincade: 43 of 134 frames, all of them while the fire was still
  active). A baseline conditioned on state 1 would predict nothing at all in a
  third of Kincade's frames. Here ``burned = state > 0``, so a dormant fire
  still has a front to advance.

Ensemble
--------
Members differ by a **shared per-step innovation** — one rate-of-spread
multiplier and one wind-bearing offset per (member, step), correlated in time by
an AR(1) — applied to the entire domain at once. This is deliberately the same
structural choice CLAUDE.md requires of the learned model (correlated
innovations via a shared per-step latent ``z_t``; pixels conditionally
independent ONLY given that latent). Per-pixel independent noise is the
known-broken alternative that collapses in aggregate, so the *baseline* is not
built that way either — otherwise the G3 ensemble comparison would be against a
strawman ensemble, and beating it would say nothing about dispersion.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.common.contract import BURNED_OUT, BURNING, UNBURNED
from wildfire_nowcast.model.api import validate_predict_inputs
from wildfire_nowcast.model.inputs import static_index, weather_index
from wildfire_nowcast.model.spread import (
    EllipseParams,
    effective_wind,
    ellipse_ros_factor,
    fuel_group_fields,
    head_rate_of_spread,
    length_to_breadth,
    midflame_wind,
    moisture_damping,
    slope_equivalent_wind,
    substeps_for,
)

__all__ = ["EllipseBaseline", "EllipseFitResult", "GrowthCalibration"]

#: 8-connected neighbour offsets ``(dr, dc)`` with their centre-to-centre
#: distance in cell widths. Row index increases SOUTHWARD (C1.4 north-up), so
#: the (east, north) direction of an offset is ``(dc, -dr)``.
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = tuple(
    (dr, dc, float(np.hypot(dr, dc)))
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _shift(a: np.ndarray, dr: int, dc: int, fill: float = 0.0) -> np.ndarray:
    """``out[r, c] = a[r - dr, c - dc]``, off-grid filled with ``fill``.

    i.e. move the contents of ``a`` by ``(dr, dc)``. Used to ask "is my
    ``(-dr, -dc)`` neighbour burned", so the domain edge answers "no" rather
    than wrapping — a wrapped fire would spread off the north edge and reappear
    in the south, which is the kind of bug that looks like spotting.
    """
    out = np.full_like(a, fill)
    height, width = a.shape
    dst_r = slice(max(dr, 0), height + min(dr, 0))
    src_r = slice(max(-dr, 0), height + min(-dr, 0))
    dst_c = slice(max(dc, 0), width + min(dc, 0))
    src_c = slice(max(-dc, 0), width + min(-dc, 0))
    out[dst_r, dst_c] = a[src_r, src_c]
    return out


@dataclass(frozen=True)
class GrowthCalibration:
    """Outcome of :meth:`EllipseBaseline.calibrate_to_growth` (ADR-011 / C6.2).

    The RULE OF RECORD for this baseline's one free scale. It is a physics
    baseline, so its scale reproduces observed mean hourly growth on TRAIN
    fires; it is never fitted by a pixelwise score. ``growth_ratio`` is the
    quantity being driven to 1 and is the one number to read.
    """

    params: EllipseParams
    scale: float
    predicted_new_cells: float
    observed_new_cells: float
    growth_ratio: float
    step_hours: int
    scale_grid: tuple[float, ...]
    ratio_grid: tuple[float, ...]
    monotone: bool
    bracketed: bool
    n_iterations: int
    train_fire_ids: tuple[str, ...]
    n_windows: int
    n_growth_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": (
                "ADR-011 / C6.2: scale calibrated so the deterministic ellipse reproduces the "
                "observed mean hourly growth of the TRAIN fires. NOT fitted by pixelwise Brier."
            ),
            "scale": self.scale,
            "step_hours": self.step_hours,
            "predicted_new_cells": self.predicted_new_cells,
            "observed_new_cells": self.observed_new_cells,
            "growth_ratio": self.growth_ratio,
            "mean_observed_new_cells_per_hour": (
                self.observed_new_cells / self.n_windows if self.n_windows else None
            ),
            "mean_predicted_new_cells_per_hour": (
                self.predicted_new_cells / self.n_windows if self.n_windows else None
            ),
            "scale_grid": list(self.scale_grid),
            "ratio_grid": list(self.ratio_grid),
            "growth_monotone_in_scale": self.monotone,
            "bracketed": self.bracketed,
            "n_bisection_iterations": self.n_iterations,
            "train_fire_ids": list(self.train_fire_ids),
            "n_windows": self.n_windows,
            "n_growth_windows": self.n_growth_windows,
        }


@dataclass(frozen=True)
class EllipseFitResult:
    """Outcome of :meth:`EllipseBaseline.fit_by_brier`, with its own audit trail.

    **Retained as evidence, BARRED as a gate baseline by C6.2.** See that method.
    """

    params: EllipseParams
    scale: float
    objective: float
    grid: tuple[float, ...]
    objectives: tuple[float, ...]
    train_fire_ids: tuple[str, ...]
    n_windows: int
    n_growth_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "objective_brier": self.objective,
            "grid": list(self.grid),
            "objectives": list(self.objectives),
            "train_fire_ids": list(self.train_fire_ids),
            "n_windows": self.n_windows,
            "n_growth_windows": self.n_growth_windows,
        }


class EllipseBaseline:
    """Wind- and slope-advected elliptical spread, with a shared-innovation ensemble.

    Parameters
    ----------
    params
        Physics knobs; see :class:`~wildfire_nowcast.model.spread.EllipseParams`.
    ros_sigma
        Log-normal sigma of the shared per-step rate-of-spread multiplier.
    bearing_sigma_deg
        Standard deviation of the shared per-step wind-bearing offset.
    innovation_rho
        AR(1) correlation of both innovations across steps. Step-to-step
        independent perturbations average out over a 3 h window and would
        produce an ensemble that looks dispersed per step and is degenerate over
        the horizon — the exact failure G3 exists to catch.
    burnout_hours
        Hours a newly-ignited cell stays in state 1 before going to state 2.
        Affects only the 1-vs-2 distinction, never the burned SET, so it cannot
        change any C6 score under ``event="burned"``; it exists so the state
        field sim renders is plausible. Default 4 h sits inside GOFER's
        measured residence p50 of 3-5 h (ADR-006).
    barrier_ros_factor
        Multiplier applied to the rate of spread INTO a ``water_barrier_mask``
        cell. 0.0 means a hard block. Note the mask is 1 km, so a "barrier" cell
        is usually only partly water or road — Kincade's truth burns 44 of them
        — which is precisely what the spot/crossing component has to explain.
    force_isotropic
        Pin the length-to-breadth ratio to 1. The anisotropy ablation: if the
        ellipse does not beat its own isotropic version, the wind field is not
        informative at this resolution and the kernel's wind modulation is
        unlikely to help either.
    """

    kind = "ellipse"

    def __init__(
        self,
        params: EllipseParams | None = None,
        *,
        name: str = "ellipse",
        ros_sigma: float = 0.35,
        bearing_sigma_deg: float = 20.0,
        innovation_rho: float = 0.7,
        burnout_hours: int = 4,
        barrier_ros_factor: float = 0.0,
        force_isotropic: bool = False,
        cell_size_m: float = 1000.0,
        fit_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.params = params or EllipseParams()
        self.ros_sigma = float(ros_sigma)
        self.bearing_sigma_deg = float(bearing_sigma_deg)
        self.innovation_rho = float(np.clip(innovation_rho, 0.0, 0.999))
        self.burnout_hours = int(burnout_hours)
        self.barrier_ros_factor = float(barrier_ros_factor)
        self.force_isotropic = bool(force_isotropic)
        self.cell_size_m = float(cell_size_m)
        self.fit_provenance = dict(fit_provenance or {})

    # -- static field preparation -----------------------------------------

    def _static_fields(self, static: np.ndarray) -> dict[str, np.ndarray]:
        get = lambda n: np.asarray(static[static_index(n)], dtype=np.float64)  # noqa: E731
        fuel_rel, mx_pct, _ = fuel_group_fields(get("fuel_model_id"), self.params.multipliers)
        barrier = get("water_barrier_mask") > 0.5
        barrier_factor = np.where(barrier, self.barrier_ros_factor, 1.0)
        u_slope, v_slope = slope_equivalent_wind(
            get("slope"), get("aspect_sin"), get("aspect_cos"), k_slope=self.params.k_slope
        )
        return {
            "fuel_rel": fuel_rel * barrier_factor,
            "mx_pct": mx_pct,
            "canopy": get("canopy_cover"),
            "u_slope": u_slope,
            "v_slope": v_slope,
        }

    def _step_rates(
        self,
        fields: dict[str, np.ndarray],
        weather_step: np.ndarray,
        params: EllipseParams,
        bearing_offset_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(head_ros_ms, lb, unit_east, unit_north)`` for one hour."""
        u10 = np.asarray(weather_step[weather_index("wind_u10")], dtype=np.float64)
        v10 = np.asarray(weather_step[weather_index("wind_v10")], dtype=np.float64)
        fm = np.asarray(weather_step[weather_index("fuel_moisture_proxy")], dtype=np.float64)

        u_mid, v_mid = midflame_wind(
            u10,
            v10,
            fields["canopy"],
            waf_open=params.waf_open,
            waf_closed=params.waf_closed,
        )
        speed, unit_e, unit_n = effective_wind(
            u_mid, v_mid, fields["u_slope"], fields["v_slope"]
        )
        if bearing_offset_deg:
            delta = np.radians(bearing_offset_deg)
            cos_d, sin_d = float(np.cos(delta)), float(np.sin(delta))
            unit_e, unit_n = (
                unit_e * cos_d + unit_n * sin_d,
                unit_n * cos_d - unit_e * sin_d,
            )
        eta = moisture_damping(fm, fields["mx_pct"])
        head = head_rate_of_spread(speed, fields["fuel_rel"], eta, params)
        lb = (
            np.ones_like(head)
            if self.force_isotropic
            else length_to_breadth(speed, cap=params.lb_cap)
        )
        return head, lb, unit_e, unit_n

    # -- the integrator ----------------------------------------------------

    def _advance_hour(
        self,
        burned: np.ndarray,
        progress: np.ndarray,
        head: np.ndarray,
        lb: np.ndarray,
        unit_e: np.ndarray,
        unit_n: np.ndarray,
    ) -> np.ndarray:
        """One hour of spread, in place on ``burned``/``progress``; returns new cells.

        Sub-hourly stepping is sized to the fastest head rate present, so the
        grid never silently caps the front: a 4 km/h Diablo head must be able to
        move 4 cells in an hour, and a one-shift-per-hour scheme would truncate
        exactly the event the model exists to predict.
        """
        ignitable = head > 0.0
        max_ros = float(head.max()) if head.size else 0.0
        n_sub = substeps_for(max_ros, self.cell_size_m)
        dt = 3600.0 / n_sub
        newly_total = np.zeros_like(burned)

        # Direction-dependent rate into each cell: fixed across substeps.
        rates = []
        for dr, dc, dist in _NEIGHBOURS:
            # Direction of travel (source -> target) in (east, north).
            de, dn = dc / dist, -dr / dist
            cos_theta = unit_e * de + unit_n * dn
            rate = head * ellipse_ros_factor(lb, cos_theta) * dt / (dist * self.cell_size_m)
            rates.append((dr, dc, np.where(ignitable, rate, 0.0)))

        for _ in range(n_sub):
            gain = np.zeros_like(progress)
            for dr, dc, rate in rates:
                # Fire arrives from the (-dr, -dc) neighbour, so bring that
                # neighbour's burned flag to this cell.
                np.maximum(gain, rate * _shift(burned.astype(np.float64), dr, dc), out=gain)
            progress += np.where(burned, 0.0, gain)
            newly = (progress >= 1.0) & ~burned & ignitable
            if newly.any():
                burned |= newly
                newly_total |= newly
                progress[newly] = 0.0
        return newly_total

    # -- C5 ----------------------------------------------------------------

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        validate_predict_inputs(x0, static, weather, n_members, horizon_h, seed)
        n_members, horizon_h = int(n_members), int(horizon_h)
        state0 = np.asarray(x0, dtype=np.uint8)
        fields = self._static_fields(np.asarray(static, dtype=np.float64))
        rng = np.random.default_rng(int(seed))
        ros_mult, bearing = self._innovations(rng, n_members, horizon_h)

        out = np.empty((n_members, horizon_h, *state0.shape), dtype=np.uint8)
        for member in range(n_members):
            out[member] = self._run_member(
                state0, fields, weather, horizon_h, ros_mult[member], bearing[member]
            )
        return out

    def _innovations(
        self, rng: np.random.Generator, n_members: int, horizon_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared per-(member, step) innovations, AR(1)-correlated in time.

        One scalar pair per member per step, applied to the WHOLE domain. This
        is the ellipse's stand-in for the model's ``z_t``: spread comes from a
        low-dimensional shared forcing perturbation, never from independent
        per-pixel noise (which averages out and collapses the ensemble in every
        aggregate quantity — the known-broken mode CLAUDE.md rules out).
        """
        rho = self.innovation_rho
        innov_scale = float(np.sqrt(max(1.0 - rho * rho, 1e-12)))
        log_ros = np.zeros((n_members, horizon_h))
        bearing = np.zeros((n_members, horizon_h))
        prev_r = rng.normal(0.0, 1.0, n_members)
        prev_b = rng.normal(0.0, 1.0, n_members)
        for k in range(horizon_h):
            if k:
                prev_r = rho * prev_r + innov_scale * rng.normal(0.0, 1.0, n_members)
                prev_b = rho * prev_b + innov_scale * rng.normal(0.0, 1.0, n_members)
            log_ros[:, k] = prev_r * self.ros_sigma
            bearing[:, k] = prev_b * self.bearing_sigma_deg
        # Median-preserving: exp(x - sigma^2/2) keeps the ensemble MEAN rate at
        # the deterministic rate, so adding spread does not also add bias.
        return np.exp(log_ros - 0.5 * self.ros_sigma**2), bearing

    def _run_member(
        self,
        state0: np.ndarray,
        fields: dict[str, np.ndarray],
        weather: np.ndarray,
        horizon_h: int,
        ros_mult: np.ndarray,
        bearing: np.ndarray,
    ) -> np.ndarray:
        burned = state0 > UNBURNED
        progress = np.zeros(state0.shape, dtype=np.float64)
        state = state0.copy()
        # -1 marks "already burning at t0, age unknown" — never burned out by us,
        # because guessing its age would be inventing information.
        ignited_at = np.full(state0.shape, -1, dtype=np.int64)
        traj = np.empty((horizon_h, *state0.shape), dtype=np.uint8)

        for k in range(horizon_h):
            params = self.params.scaled(float(ros_mult[k]))
            head, lb, unit_e, unit_n = self._step_rates(
                fields, np.asarray(weather[k], dtype=np.float64), params, float(bearing[k])
            )
            newly = self._advance_hour(burned, progress, head, lb, unit_e, unit_n)
            if newly.any():
                state[newly] = BURNING
                ignited_at[newly] = k
            if self.burnout_hours > 0:
                spent = (ignited_at >= 0) & (k - ignited_at >= self.burnout_hours)
                state[spent] = BURNED_OUT
            traj[k] = state
        return traj

    # -- calibration (TRAIN fires only) — THE RULE OF RECORD ---------------

    def predicted_new_cells(self, windows: Sequence[Any], scale: float) -> float:
        """Total cells the DETERMINISTIC ellipse ignites over ``windows`` at ``scale``.

        Teacher-forced: every window starts from its own observed ``x0``, so
        this is a sum of one-window increments and is directly comparable to the
        same sum taken over the labels. Deterministic (1 member, no
        innovations), because the calibration is about the rate law and not
        about ensemble spread.
        """
        twin = self._deterministic_twin()
        twin.params = self.params.scaled(float(scale))
        total = 0.0
        for window in windows:
            pred = twin.predict(window.x0, window.static, window.weather, 1, window.horizon_h, 0)
            before = np.asarray(window.x0) > UNBURNED
            after = pred[0, -1] > UNBURNED
            total += float(np.count_nonzero(after & ~before))
        return total

    def calibrate_to_growth(
        self,
        windows: Iterable[Any],
        *,
        train_fire_ids: Sequence[str] = (),
        scale_grid: Sequence[float] | None = None,
        tolerance: float = 0.02,
        max_iterations: int = 24,
    ) -> GrowthCalibration:
        """**C6.2 / ADR-011.** Scale the head rate to reproduce observed growth.

        The rule, stated so it cannot drift: choose the single head-rate scale
        at which the deterministic ellipse's TOTAL newly-burned cells over the
        TRAIN windows equals the labels' total newly-burned cells over the same
        windows. With one-hour windows that is exactly "reproduce the observed
        mean hourly growth on train fires".

        **Why not a pixelwise score.** Measured (insights item 3): the ellipse's
        precision on newly-ignited cells is 0.09-0.43, always below 0.5, so
        under a hard 0/1 Brier a predictor worse than a coin flip about *where*
        is optimally SILENT. The Brier fit picked 0.501 — one grid point off a
        dead zone where sub-half-cell/hour spread ignites nothing — and ignited
        zero cells against 782 of truth. That does not produce a weak baseline,
        it produces persistence, which is already a separate baseline. Two
        baselines that are secretly the same baseline are a mirror, not a floor.
        Growth calibration asks the baseline the question it can answer (*how
        much* burns) and leaves the question it answers badly (*where*) to be
        scored, which is what a floor is for.

        Method: total predicted growth is monotone non-decreasing in the scale,
        so the grid is swept once (and its monotonicity RECORDED, not assumed),
        the unit-ratio crossing is bracketed, and the bracket is bisected in log
        space until ``|log growth_ratio| <= tolerance``.

        **The caller passes TRAIN windows only.** Nothing here can tell a train
        fire from a held-out one; ``train_fire_ids`` exists so a violation is
        auditable after the fact.
        """
        window_list = list(windows)
        if not window_list:
            raise ValueError("calibrate_to_growth needs at least one training window")
        step_hours = {int(w.horizon_h) for w in window_list}
        if len(step_hours) != 1:
            raise ValueError(f"windows must share one horizon; got {sorted(step_hours)}")

        observed = float(sum(int(w.truth_growth_cells()) for w in window_list))
        if observed <= 0:
            raise ValueError("no observed growth in the train windows; nothing to calibrate to")

        grid = tuple(
            float(s)
            for s in (scale_grid if scale_grid is not None else np.geomspace(0.25, 16.0, 13))
        )
        cache: dict[float, float] = {}

        def ratio(scale: float) -> float:
            if scale not in cache:
                cache[scale] = self.predicted_new_cells(window_list, scale) / observed
            return cache[scale]

        ratios = tuple(ratio(s) for s in grid)
        monotone = all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:], strict=False))

        lo = hi = None
        for (s_a, r_a), (s_b, r_b) in zip(
            zip(grid, ratios, strict=True), zip(grid[1:], ratios[1:], strict=True), strict=False
        ):
            if r_a <= 1.0 <= r_b:
                lo, hi = (s_a, r_a), (s_b, r_b)
                break
        bracketed = lo is not None

        iterations = 0
        if lo is None or hi is None:
            # No crossing on the grid: take the closest ratio in log space and
            # say so, rather than silently returning an endpoint as if it were a
            # solution.
            best = min(grid, key=lambda s: abs(np.log(max(ratio(s), 1e-9))))
            chosen = best
        else:
            s_lo, s_hi = lo[0], hi[0]
            chosen = s_hi
            for _ in range(int(max_iterations)):
                iterations += 1
                mid = float(np.sqrt(s_lo * s_hi))
                r_mid = ratio(mid)
                chosen = mid
                if abs(np.log(max(r_mid, 1e-9))) <= tolerance:
                    break
                if r_mid < 1.0:
                    s_lo = mid
                else:
                    s_hi = mid

        return GrowthCalibration(
            params=self.params.scaled(chosen),
            scale=float(chosen),
            predicted_new_cells=float(ratio(chosen) * observed),
            observed_new_cells=observed,
            growth_ratio=float(ratio(chosen)),
            step_hours=int(next(iter(step_hours))),
            scale_grid=grid,
            ratio_grid=ratios,
            monotone=bool(monotone),
            bracketed=bool(bracketed),
            n_iterations=iterations,
            train_fire_ids=tuple(
                sorted({getattr(w, "fire_id", "") for w in window_list} - {""})
                or sorted(train_fire_ids)
            ),
            n_windows=len(window_list),
            n_growth_windows=sum(1 for w in window_list if w.truth_growth_cells() > 0),
        )

    def with_calibration(self, result: GrowthCalibration) -> EllipseBaseline:
        """A copy carrying growth-calibrated parameters and their audit trail."""
        clone = self.with_fit(
            EllipseFitResult(
                params=result.params,
                scale=result.scale,
                objective=float("nan"),
                grid=(),
                objectives=(),
                train_fire_ids=result.train_fire_ids,
                n_windows=result.n_windows,
                n_growth_windows=result.n_growth_windows,
            )
        )
        clone.fit_provenance = result.to_dict()
        return clone

    # -- the BARRED Brier fit, kept as evidence ---------------------------

    def fit_by_brier(
        self,
        windows: Iterable[Any],
        *,
        scales: Sequence[float] | None = None,
        train_fire_ids: Sequence[str] = (),
    ) -> EllipseFitResult:
        """Fit the head-rate scale on TRAIN windows by pooled Brier. **BARRED.**

        C6.2 forbids this as the rule for a gate baseline; it is retained
        because it is the EVIDENCE for that clause, and deleting the failed
        method would delete the reason the rule exists. Any G2 verdict that
        cites an ellipse fitted this way is void.

        One free parameter (a multiplier on ``r0_ms``) searched over a log grid.
        One parameter, not four, on purpose: with 11 effective spatial blocks
        (C3.1) there is not enough independent landscape to fit a wind exponent
        and a slope coefficient and per-fuel multipliers without overfitting the
        blocks, and an overfit "baseline" would make G2 easy for the wrong
        reason.

        **The caller is responsible for passing TRAIN windows only.** This is
        leave-fire-out CV (CLAUDE.md); nothing here can tell a train fire from a
        test one. ``train_fire_ids`` is recorded into the fitted model's spec so
        a violation is at least auditable after the fact.

        Scored deterministically (1 member, no innovations) so the fit is about
        the rate law and not about the ensemble spread.
        """
        window_list = list(windows)
        if not window_list:
            raise ValueError("fit needs at least one training window")
        default_grid = np.geomspace(0.1, 10.0, 21)
        grid = tuple(float(s) for s in (scales if scales is not None else default_grid))
        deterministic = self._deterministic_twin()

        observed_ids = {getattr(w, "fire_id", "") for w in window_list} - {""}
        objectives: list[float] = []
        for scale in grid:
            deterministic.params = self.params.scaled(scale)
            sq_err, count = 0.0, 0
            for window in window_list:
                pred = deterministic.predict(
                    window.x0, window.static, window.weather, 1, window.horizon_h, 0
                )
                p = (pred[0] > UNBURNED).astype(np.float64)
                y = (np.asarray(window.truth) > UNBURNED).astype(np.float64)
                sq_err += float(np.sum((p - y) ** 2))
                count += int(y.size)
            objectives.append(sq_err / max(count, 1))

        best = int(np.argmin(objectives))
        return EllipseFitResult(
            params=self.params.scaled(grid[best]),
            scale=grid[best],
            objective=objectives[best],
            grid=grid,
            objectives=tuple(objectives),
            train_fire_ids=tuple(sorted(observed_ids or set(train_fire_ids))),
            n_windows=len(window_list),
            n_growth_windows=sum(1 for w in window_list if w.truth_growth_cells() > 0),
        )

    def _deterministic_twin(self) -> EllipseBaseline:
        return EllipseBaseline(
            self.params,
            name=f"{self.name}-deterministic",
            ros_sigma=0.0,
            bearing_sigma_deg=0.0,
            innovation_rho=0.0,
            burnout_hours=self.burnout_hours,
            barrier_ros_factor=self.barrier_ros_factor,
            force_isotropic=self.force_isotropic,
            cell_size_m=self.cell_size_m,
        )

    def with_fit(self, result: EllipseFitResult) -> EllipseBaseline:
        """A copy carrying fitted parameters and the fit's audit trail."""
        return EllipseBaseline(
            result.params,
            name=self.name,
            ros_sigma=self.ros_sigma,
            bearing_sigma_deg=self.bearing_sigma_deg,
            innovation_rho=self.innovation_rho,
            burnout_hours=self.burnout_hours,
            barrier_ros_factor=self.barrier_ros_factor,
            force_isotropic=self.force_isotropic,
            cell_size_m=self.cell_size_m,
            fit_provenance=result.to_dict(),
        )

    # -- serialisation -----------------------------------------------------

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "params": self.params.to_dict(),
            "ensemble": {
                "ros_sigma": self.ros_sigma,
                "bearing_sigma_deg": self.bearing_sigma_deg,
                "innovation_rho": self.innovation_rho,
                "innovation_structure": "shared per-(member, step) scalars over the whole domain",
            },
            "burnout_hours": self.burnout_hours,
            "barrier_ros_factor": self.barrier_ros_factor,
            "force_isotropic": self.force_isotropic,
            "cell_size_m": self.cell_size_m,
            "fit": self.fit_provenance,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> EllipseBaseline:
        ensemble = dict(spec.get("ensemble", {}))
        return cls(
            EllipseParams.from_dict(spec.get("params", {})),
            name=str(spec.get("name", "ellipse")),
            ros_sigma=float(ensemble.get("ros_sigma", 0.35)),
            bearing_sigma_deg=float(ensemble.get("bearing_sigma_deg", 20.0)),
            innovation_rho=float(ensemble.get("innovation_rho", 0.7)),
            burnout_hours=int(spec.get("burnout_hours", 4)),
            barrier_ros_factor=float(spec.get("barrier_ros_factor", 0.0)),
            force_isotropic=bool(spec.get("force_isotropic", False)),
            cell_size_m=float(spec.get("cell_size_m", 1000.0)),
            fit_provenance=dict(spec.get("fit", {})),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"EllipseBaseline(r0_ms={self.params.r0_ms:.4g}, isotropic={self.force_isotropic})"
