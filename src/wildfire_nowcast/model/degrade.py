"""[M11] A DEGRADATION LADDER of known, ordered severity, applied to any C5 predictor.

WHY THIS EXISTS
---------------
M10 (ADR-049 (6)) found that three of five headline channels could not separate
their own control: 3 h band Brier moved **+0.80 block-SD** against an arm that
predicted **3.60x less area**, arrival-time CRPS +0.91, ``calibration_error``
+1.20, all under the 2.0 bar. A metric that cannot see a 3.6x area error is not
measuring what we think it measures at n = 5 held-out blocks, and that lands on
every past verdict scored with those channels.

The instrument question cannot be answered by another model, because a model's
true degradation is unknown. It is answered by degrading a forecast by a KNOWN,
ORDERED amount and measuring what each channel reports. This module supplies the
degradation; :mod:`wildfire_nowcast.eval.power` reads the curve off the result.

THE CONSTRUCTION, AND WHY IT IS THE ONE THAT MAKES THE LADDER INTERPRETABLE
--------------------------------------------------------------------------
Every rung is a **total order over the cells that are not burned at t0**, plus a
**size schedule**. The degraded sample at lead ``h`` is the first ``n_h`` cells of
that order. Three properties fall out for free, and each of them is a defect this
project has paid for elsewhere:

``exactness``
    The base order is *"the first lead at which the base sample burns this cell,
    then distance from the t0 burn, then a fixed permutation"*. Its first
    ``|I_h|`` cells ARE ``I_h``. So the ``k = 1`` / ``f = 0`` rung reproduces the
    wrapped model BITWISE - a null rung that must return zero separation, which
    is the ladder's negative control and is asserted, not assumed.
``nesting``
    ``n_h`` is non-decreasing and the order is fixed within a window, so
    ``D_h`` is nested in ``D_{h+1}`` by construction. Fire is absorbing
    (C1.1); a degradation that un-burns a cell would be rejected by
    ``validate_samples`` and, worse, would be a different intervention at each
    lead.
``area is exact where it is claimed to be``
    The SHAPE rungs use ``n_h`` unchanged, so total predicted area is preserved
    EXACTLY, not approximately. A channel blind to shape at fixed area is a
    different failure from one blind to area, and separating them needs the area
    held to zero error rather than to a tolerance.

TWO FAMILIES
------------
``area``  (:data:`MODE_AREA`)
    Same order as the base, size schedule ``round(k * n_h)``. ``k < 1`` keeps the
    cells the base burns EARLIEST - a temporal erosion of the front. ``k > 1``
    runs past the base prediction into the nearest unburned cells, which the base
    order ranks by distance from the t0 burn. Severity unit: the area ratio,
    which is measured on the samples and is independent of every channel scored.
``shape`` (:data:`MODE_SHAPE`)
    Size schedule ``n_h`` EXACTLY, order blended between the base order and a
    lobe pointing 180 degrees away from the base increment's own bearing.
    ``f`` is the blend weight. Severity unit: IoU of the degraded increment
    against the base increment, measured per window.

WHAT THIS MODULE IS NOT
-----------------------
It is not a model, not a baseline and not a candidate for any gate. It has no
parameters fitted to anything. It exists to be scored so that the SCORER can be
characterised, and every rung's severity is declared before it is scored.
"""

from __future__ import annotations

import hashlib
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.model.api import validate_predict_inputs, validate_samples

__all__ = [
    "MODE_AREA",
    "MODE_SHAPE",
    "MODE_SHIFT",
    "shift_offset_cells",
    "realised_displacement_km",
    "DEFINED",
    "UNDEFINED",
    "BasePredictionCache",
    "CellOrder",
    "IncrementOverlap",
    "degrade_samples",
    "increment_overlap",
    "increment_iou",
    "DegradedModel",
]

#: Outcome labels for an overlap measurement. THREE-VALUED on purpose (C6.5's
#: precedent): a comparison that cannot be made is its own answer, and it is
#: never silently turned into a number that could be ordered against the others.
DEFINED = "DEFINED"
UNDEFINED = "UNDEFINED"

#: Scale the predicted increment's AREA, keeping the base's own cell order.
MODE_AREA = "area"

#: Move the predicted increment's SHAPE at EXACTLY the base's area.
MODE_SHAPE = "shape"

#: TRANSLATE the predicted increment by ``level`` cells at EXACTLY the base's
#: area. [ADR-128 (4)] A metric may only be adopted if it orders BOTH an area
#: ladder and a DISPLACEMENT ladder, and the two must be separable; ``MODE_SHAPE``
#: cannot serve as the displacement family because its severity unit is an
#: increment IoU that saturates at 0 long before the two lobes stop moving apart.
#: A translation has a severity in KILOMETRES that keeps increasing, and its
#: ``level = 0`` rung is the identity by the same construction as the other two.
MODE_SHIFT = "shift"

#: How strongly the opposing lobe prefers direction over proximity, in cells. At
#: 3.0 a cell four rings out in the target direction outranks a cell one ring out
#: in the opposite one, which is what makes ``f = 1`` a genuinely different shape
#: rather than a slightly re-weighted version of the same ring.
_LOBE_LAMBDA = 3.0

#: Fixed tie-break permutation seed. A row-major tie-break inside a ring would
#: bias every rung toward the top-left of the grid, which is a spatial artefact
#: that the shape channels would then be measuring instead of the degradation.
_PERM_SEED = 20260814

#: Cap on the number of dilation rings when building the distance field. Beyond
#: this a cell is "far"; it still gets an order, it is just ranked after
#: everything reachable.
_MAX_RINGS = 64


def _shift_or(mask: np.ndarray) -> np.ndarray:
    """8-connected dilation by one cell, zero-filled at the grid edge."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[yd, xd] = mask[ys, xs]
            out |= shifted
    return out


def _geodesic_rings(seed_mask: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Ring index (1-based) of each ``free`` cell from ``seed_mask``; far = large."""
    dist = np.full(seed_mask.shape, _MAX_RINGS + 1, dtype=np.int32)
    reached = seed_mask.copy()
    frontier = seed_mask.copy()
    for ring in range(1, _MAX_RINGS + 1):
        nxt = _shift_or(frontier) & free & ~reached
        if not nxt.any():
            break
        dist[nxt] = ring
        reached |= nxt
        frontier = nxt
    dist[~free] = 0
    return dist


class CellOrder:
    """A total order over the not-burned-at-t0 cells, plus the base size schedule.

    Built once per (window, member-set) and reused by every rung, which is what
    makes the ladder a family of orders over one base prediction rather than a
    family of independently-noised forecasts. The rungs therefore differ from each
    other by exactly the declared parameter and by nothing else.
    """

    def __init__(self, base_samples: np.ndarray, x0: np.ndarray) -> None:
        arr = np.asarray(base_samples)
        self.n_members, self.n_leads, self.height, self.width = arr.shape
        self.n_cells = self.height * self.width
        burned0 = np.asarray(x0) > 0
        self.free = ~burned0
        self.free_flat = self.free.reshape(-1)

        # increments, [M, T, H*W] bool, nested in T because samples are absorbing
        inc = (arr > 0) & self.free[None, None]
        self.inc_flat = inc.reshape(self.n_members, self.n_leads, -1)
        self.base_sizes = self.inc_flat.sum(axis=2).astype(np.int64)
        self.base_state_flat = arr.reshape(self.n_members, self.n_leads, -1)

        seed_mask = burned0 if burned0.any() else inc.any(axis=(0, 1))
        self.rings = _geodesic_rings(seed_mask, self.free).reshape(-1)

        rng = np.random.default_rng(_PERM_SEED)
        self.perm = rng.permutation(self.n_cells)

        # bearing of every cell from the centre of the region the fire spreads FROM
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        anchor = seed_mask if seed_mask.any() else np.ones_like(self.free)
        cy = float(ys[anchor].mean())
        cx = float(xs[anchor].mean())
        self.bearing = np.arctan2(
            (ys - cy).astype(np.float64), (xs - cx).astype(np.float64)
        ).reshape(-1)

        pooled = self.inc_flat.any(axis=(0, 1))
        if pooled.any():
            self.phi_base = float(
                np.arctan2(
                    (ys.reshape(-1)[pooled] - cy).mean(),
                    (xs.reshape(-1)[pooled] - cx).mean(),
                )
            )
        else:
            self.phi_base = 0.0

        self._base_rank = self._build_base_rank()
        self._opp_rank = self._build_opposing_rank()

    # -- the two orders ----------------------------------------------------

    def _build_base_rank(self) -> np.ndarray:
        """``[M, H*W]`` rank. First ``|I_h|`` entries are exactly ``I_h``, per lead.

        Lexicographic, never an arithmetic packing of several keys into one
        number: a packed key is correct only while every field stays inside the
        stride chosen for it, and a grid small enough to break that assumption
        would silently reorder rather than fail.
        """
        # first lead at which the base burns the cell; never -> n_leads + 1
        first = np.full((self.n_members, self.n_cells), self.n_leads + 1, dtype=np.int32)
        for lead in range(self.n_leads - 1, -1, -1):
            first = np.where(self.inc_flat[:, lead], lead + 1, first)
        not_free = np.broadcast_to(~self.free_flat, first.shape)
        rings = np.broadcast_to(self.rings, first.shape)
        perm = np.broadcast_to(self.perm, first.shape)
        return _ranks_from(np.lexsort((perm, rings, first, not_free), axis=-1))

    def _build_opposing_rank(self) -> np.ndarray:
        """``[H*W]`` rank of a lobe pointing 180 degrees from the base increment."""
        target = self.phi_base + np.pi
        score = np.minimum(self.rings, _MAX_RINGS + 1).astype(np.float64) - _LOBE_LAMBDA * np.cos(
            self.bearing - target
        )
        keys = (self.perm[None, :], score[None, :], (~self.free_flat)[None, :])
        return _ranks_from(np.lexsort(keys, axis=-1))[0]

    def shift_offset(self, level: float) -> tuple[int, int]:
        """Integer ``(dy, dx)`` for a ``level``-cell translation. See :func:`shift_offset_cells`."""
        return shift_offset_cells(self.phi_base, level)

    def _build_shifted_rank(self, level: float) -> np.ndarray:
        """``[M, H*W]`` rank of the base order TRANSLATED by ``level`` cells.

        The rank of a cell is the base rank of the cell it was translated FROM,
        so the first ``n_h`` cells of this order are the base's own first ``n_h``
        cells moved bodily by the offset. Area is therefore preserved EXACTLY,
        integer-for-integer, and ``level = 0`` returns the base rank itself -
        the identity rung, by construction rather than by a branch.

        A translated cell that lands off the grid or on ground already burned at
        t0 cannot be selected (it would un-burn nothing and would break C1.1), so
        those cells are ranked LAST and the shortfall is filled by the next cells
        of the same translated order. That keeps the area exact; it does mean the
        REALISED displacement is at most the requested one near a boundary, which
        is why the realised centroid displacement is measured on the samples
        rather than assumed from the level (:func:`realised_displacement_km`).
        """
        dy, dx = self.shift_offset(level)
        if (dy, dx) == (0, 0):
            return self._base_rank
        grid = self._base_rank.reshape(self.n_members, self.height, self.width)
        far = np.int32(self.n_cells + 1)
        moved = np.full_like(grid, far)
        ys_dst = slice(max(dy, 0), self.height + min(dy, 0))
        ys_src = slice(max(-dy, 0), self.height + min(-dy, 0))
        xs_dst = slice(max(dx, 0), self.width + min(dx, 0))
        xs_src = slice(max(-dx, 0), self.width + min(-dx, 0))
        moved[:, ys_dst, xs_dst] = grid[:, ys_src, xs_src]
        score = moved.reshape(self.n_members, -1).astype(np.float64)
        not_free = np.broadcast_to(~self.free_flat, score.shape)
        perm = np.broadcast_to(self.perm, score.shape)
        return _ranks_from(np.lexsort((perm, score, not_free), axis=-1))

    # -- the rungs ---------------------------------------------------------

    def sizes_for(self, mode: str, level: float) -> np.ndarray:
        """Target ``|D_h|`` per member and lead, non-decreasing in lead."""
        if mode in (MODE_SHAPE, MODE_SHIFT):
            return self.base_sizes.copy()
        if mode != MODE_AREA:
            raise ValueError(f"unknown degradation mode {mode!r}")
        n_free = int(self.free_flat.sum())
        scaled = np.rint(self.base_sizes.astype(np.float64) * float(level)).astype(np.int64)
        scaled = np.clip(scaled, 0, n_free)
        return np.maximum.accumulate(scaled, axis=1)

    def order_for(self, mode: str, level: float) -> np.ndarray:
        """``[M, H*W]`` rank to take the first ``n_h`` of."""
        if mode == MODE_AREA:
            return self._base_rank
        if mode == MODE_SHIFT:
            return self._build_shifted_rank(float(level))
        blend = float(level)
        if blend == 0.0:
            return self._base_rank
        mixed = (1.0 - blend) * self._base_rank.astype(np.float64) + blend * self._opp_rank[
            None, :
        ].astype(np.float64)
        not_free = np.broadcast_to(~self.free_flat, mixed.shape)
        perm = np.broadcast_to(self.perm, mixed.shape)
        return _ranks_from(np.lexsort((perm, mixed, not_free), axis=-1))


def _ranks_from(order: np.ndarray) -> np.ndarray:
    """Invert a row-wise argsort/lexsort into a rank array (0 = first)."""
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(order.shape[1], dtype=np.int32)[None, :]
    return ranks


def shift_offset_cells(bearing_rad: float, level: float) -> tuple[int, int]:
    """Integer ``(dy, dx)`` displacing by ``level`` cells along ``bearing_rad``.

    ``bearing_rad`` is measured in ARRAY coordinates - ``arctan2(y - cy, x - cx)``
    with ``y`` the row index - which is the frame ``CellOrder`` already builds its
    bearings in, so no compass conversion happens here and none is owed.

    OUTWARD, along the base increment's OWN bearing, rather than along a fixed
    compass direction. A fixed direction would push half the windows' forecasts
    back INTO ground that is already burned at t0, where the translation cannot
    be realised, so the ladder's realised severity would depend on the accident of
    which way each fire happened to be running. Outward is also the operationally
    interesting error: the right shape of front, arriving too far ahead.

    Rounded to whole cells because the grid is whole cells. The requested and the
    realised magnitudes therefore differ by up to half a cell, which is why the
    realised one is measured and reported rather than the requested one.
    """
    step = float(level)
    return (int(round(step * np.sin(bearing_rad))), int(round(step * np.cos(bearing_rad))))


def realised_displacement_km(
    degraded: np.ndarray, base: np.ndarray, x0: np.ndarray, *, cell_size_km: float = 1.0
) -> float | None:
    """Centroid displacement of a rung's increment from the base's, in km.

    The ladder's severity unit, MEASURED on the samples the way the area family's
    ratio is, never read back off the level that produced it. ``None`` when
    either increment is empty - UNDEFINED, not 0.0, for the reason
    :class:`IncrementOverlap` gives.
    """
    free = np.asarray(x0) == 0
    inc_a = (np.asarray(degraded) > 0) & free[None, None]
    inc_b = (np.asarray(base) > 0) & free[None, None]
    if not inc_a.any() or not inc_b.any():
        return None
    ys, xs = np.mgrid[0 : free.shape[0], 0 : free.shape[1]]
    pooled_a = inc_a.any(axis=(0, 1))
    pooled_b = inc_b.any(axis=(0, 1))
    dy = float(ys[pooled_a].mean() - ys[pooled_b].mean())
    dx = float(xs[pooled_a].mean() - xs[pooled_b].mean())
    return float(np.hypot(dy, dx) * float(cell_size_km))


def degrade_samples(
    base_samples: np.ndarray,
    x0: np.ndarray,
    *,
    mode: str,
    level: float,
    order: CellOrder | None = None,
) -> np.ndarray:
    """Return the rung's C5 samples. ``mode='area', level=1.0`` is the identity."""
    arr = np.asarray(base_samples)
    co = order if order is not None else CellOrder(arr, x0)
    ranks = co.order_for(mode, level)
    sizes = co.sizes_for(mode, level)

    out = np.zeros_like(co.base_state_flat)
    burned0_state = np.asarray(x0).reshape(-1) > 0
    for lead in range(co.n_leads):
        keep = ranks < sizes[:, lead][:, None]
        base_state = co.base_state_flat[:, lead]
        # a kept cell the base also burns keeps its base state (1 vs 2 matters to
        # nothing downstream today, but inventing a state would be a silent change
        # of the thing being degraded); a kept cell the base does not burn is
        # BURNING, which is the weakest claim consistent with "it burned".
        added = keep & ~co.inc_flat[:, lead]
        state = np.where(keep & co.inc_flat[:, lead], base_state, 0)
        state = np.where(added, 1, state)
        out[:, lead] = np.where(burned0_state[None, :], base_state, state)
    shaped = out.reshape(arr.shape).astype(np.uint8)
    return validate_samples(shaped, x0, arr.shape[0], arr.shape[1])


@dataclass(frozen=True)
class IncrementOverlap:
    """Two forecasts' increment overlap, WITH its denominator, three-valued.

    ``UNDEFINED`` is its own outcome and is NEVER 0.0 and NEVER a pass - the same
    shape as ``ConditionResult`` in ``common/dispersion.py``, and for the same
    reason. An empty union means NEITHER forecast added a cell, which is total
    AGREEMENT; scoring it as IoU 0.0 would record perfect disagreement and would
    make a degenerate rung look like the most severe rung on the ladder. Silently
    dropping it instead would let a curve be fitted to a subset of rungs nobody
    declared.
    """

    intersection: int
    union: int
    iou: float | None

    @property
    def defined(self) -> bool:
        return self.union > 0

    @property
    def outcome(self) -> str:
        return DEFINED if self.defined else UNDEFINED

    def check(self) -> IncrementOverlap:
        if self.defined != (self.iou is not None):
            raise AssertionError(
                f"outcome {self.outcome} disagrees with iou={self.iou}; a defined overlap must "
                "carry a number and an undefined one must carry None"
            )
        if self.intersection > self.union:
            raise AssertionError(f"intersection {self.intersection} > union {self.union}")
        return self


def increment_overlap(a: np.ndarray, b: np.ndarray, x0: np.ndarray) -> IncrementOverlap:
    """Overlap of two forecasts' INCREMENTS over the t0-unburned cells, pooled.

    Over the increment rather than the burned region: the burned region is
    dominated by cells that were already burned at t0 and are identical in both,
    which would report ~1.0 for two forecasts that share nothing they predicted.

    Returns the TERMS, not just the ratio, so a caller can see the denominator
    rather than discovering it is zero by getting a ``None`` back and comparing it.
    """
    free = np.asarray(x0) == 0
    ia = (np.asarray(a) > 0) & free[None, None]
    ib = (np.asarray(b) > 0) & free[None, None]
    union = int((ia | ib).sum())
    intersection = int((ia & ib).sum())
    return IncrementOverlap(
        intersection=intersection,
        union=union,
        iou=(float(intersection) / float(union)) if union else None,
    ).check()


def increment_iou(a: np.ndarray, b: np.ndarray, x0: np.ndarray) -> float | None:
    """The ratio alone. ``None`` means **UNDEFINED**, never 0.0 and never a pass.

    A ``None`` returned from here is NOT orderable and NOT comparable: ``None >
    0.5`` raises, by design and not by accident. Callers that need to rank rungs
    must branch on :meth:`IncrementOverlap.defined` - use
    :func:`increment_overlap` - rather than defaulting the value, because the two
    plausible defaults are both wrong in the same direction. Substituting 0.0
    turns "both forecasts agreed that nothing burns" into "the two forecasts
    share nothing", which is the maximum severity this ladder can express, and
    substituting 1.0 hides a rung that produced no forecast at all.
    """
    return increment_overlap(a, b, x0).iou


class BasePredictionCache:
    """LRU of a wrapped model's OUTPUT, so 15 rungs cost one forward pass.

    Keyed by a digest of every C5 argument, not by call order: the runner scores
    one model over every window of a fire before moving to the next model, so a
    key that is not fully identifying would return another window's samples and
    the whole ladder would be a ladder over the wrong forecast. Compressed
    because a fire's worth of ``uint8`` trajectories is hundreds of MB raw and
    fire state is almost all zeros.
    """

    def __init__(self, capacity: int = 1200) -> None:
        self.capacity = int(capacity)
        self._store: OrderedDict[str, tuple[tuple[int, ...], bytes]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for arr in (np.asarray(x0), np.asarray(static), np.asarray(weather)):
            digest.update(np.ascontiguousarray(arr).tobytes())
            digest.update(repr(arr.shape).encode())
        digest.update(f"|{int(n_members)}|{int(horizon_h)}|{int(seed)}".encode())
        return digest.hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        shape, blob = entry
        return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(shape).copy()

    def put(self, key: str, samples: np.ndarray) -> None:
        arr = np.ascontiguousarray(np.asarray(samples, dtype=np.uint8))
        self._store[key] = (arr.shape, zlib.compress(arr.tobytes(), 1))
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._store),
            "capacity": self.capacity,
            "bytes": int(sum(len(b) for _, b in self._store.values())),
        }


class DegradedModel:
    """A C5 predictor that degrades a wrapped predictor by a DECLARED amount.

    Carries the wrapped model's ``provenance`` for the same reason the M10
    controls do: C8 must read the real checkpoint's split fingerprint rather than
    exempting the arm, and C-1 makes unverifiable a failure rather than a skip.
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str,
        mode: str,
        level: float,
        cache: BasePredictionCache | None = None,
    ) -> None:
        if mode not in (MODE_AREA, MODE_SHAPE, MODE_SHIFT):
            raise ValueError(f"unknown degradation mode {mode!r}")
        self.model = model
        self.name = name
        self.mode = mode
        self.level = float(level)
        self.cache = cache
        self.provenance: dict[str, Any] = dict(getattr(model, "provenance", {}) or {})
        self.provenance["degradation"] = (
            f"M11 ladder rung — mode={mode}, level={level}. Same fit and same base "
            "prediction as the arm it wraps; the ONLY difference is the declared "
            "degradation applied to the sampled trajectories."
        )

    @property
    def kind(self) -> str:
        return f"degraded({getattr(self.model, 'kind', '?')})"

    def is_identity(self) -> bool:
        """True when this rung must reproduce the wrapped model BITWISE."""
        return (self.mode == MODE_AREA and self.level == 1.0) or (
            self.mode == MODE_SHAPE and self.level == 0.0
        )

    def base_predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        if self.cache is None:
            return self.model.predict(x0, static, weather, n_members, horizon_h, seed)
        key = self.cache.key(x0, static, weather, n_members, horizon_h, seed)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        samples = np.asarray(self.model.predict(x0, static, weather, n_members, horizon_h, seed))
        self.cache.put(key, samples)
        return samples

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        """C5 ``predict``, degraded. Seed-exact, like the wrapped model."""
        validate_predict_inputs(x0, static, weather, n_members, horizon_h, seed)
        base = self.base_predict(x0, static, weather, n_members, horizon_h, seed)
        # DELIBERATELY NOT SHORT-CIRCUITED AT THE IDENTITY RUNG. Returning the
        # base directly would make "the null rung reproduces the wrapped model
        # bitwise" a property of an `if`, not of the construction - a check that
        # cannot fail. The identity rung runs the whole machinery and is asserted
        # bitwise-equal in `eval/selftest.py`.
        return degrade_samples(base, x0, mode=self.mode, level=self.level)

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "mode": self.mode,
            "level": self.level,
            "wrapped": self.model.to_spec() if hasattr(self.model, "to_spec") else None,
        }
