"""[I21, I23] Is a shape rung actually WORSE than the reference it degrades?

WHY THIS FILE EXISTS
--------------------
``model/degrade.py`` builds a severity ladder, and every acceptance test written
on it reads a positive paired score difference as "the scorer saw the
degradation" and a null as "the scorer is blind". Both readings rest on something
that was stated and not checked: that each rung is STRICTLY WORSE than the
reference it degrades. That cannot be measured on a real fire, where the model's
own error is unknown. It can be measured here, on states whose right answer is
fixed by construction.

WHAT THE MEASUREMENT FOUND, AND WHAT HAPPENED TO IT
--------------------------------------------------
[I21] On a single-body state the ladder behaved as documented. On a state with a
SECOND burned body it did not: the paired difference changed SIGN while the
realised relocation kept growing, so a nominally harsher rung produced a
genuinely BETTER forecast. Three tests in this file asserted that inversion -
they asserted a DEFECT, on purpose, as a tripwire.

[M28] The ladder was then repaired: the anchor, the bearing field and the
geodesic rings are built per connected component of the t0 burn, so each body
relocates its own share of the increment about its own centroid. The three
tripwires fired, which is what a tripwire is for, and this file is re-pointed:
the same states are now asserted to climb the ladder the way a single-body state
does.

[I23] THE OLD ASSERTIONS ARE NOT DELETED, THEY ARE MOVED ONTO A PLANT. A test
deleted because it started failing is a defect restored. What the old assertions
knew - that a two-body state used to score BETTER under a harsher rung - is kept
in executable form by :func:`single_anchor_construction`, which restores the
pre-repair construction IN MEMORY for the duration of one assertion. Under it the
inversion comes back to the digit (``-0.069 / -0.281 / -0.675 / -1.028``, 0 of 6
windows worse, which are I21's own published numbers), and the shipped module is
required to be positive and unanimous on the same states in the same test. So the
repair cannot be reverted while this suite stays green, whichever half of it is
reverted: revert the construction and the shipped-module half fails; delete the
patch point and the plant half fails to attach at all.

WHAT IS ASSERTED HERE, AND WHAT IS NOT
--------------------------------------
Nothing here is a contract clause and nothing here carries a C number. The
monotonicity property below is a PROPOSAL, held in executable form so it can be
read and adopted or rejected on its merits. ADR-128 (7) freezes new contract
clauses, and a test that acquires the authority of a clause by being green is the
failure that freeze exists to prevent.

The property is a PURE function of the ladder's rows, and it is planted against
directly: a check that fires on a negative and is silent on a zero is half blind,
and an inert rung - one that relocates nothing and moves no score - is exactly
how a defect of this shape could be "repaired" while looking green.

This file imports ``model/`` READ ONLY. It fits nothing and loads nothing, and it
scores through ``eval.metrics.evaluate``, the entry point ``runs/_m24_ladder.py``
scores its own rungs through.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from wildfire_nowcast.common.components import label_components
from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY
from wildfire_nowcast.eval.metrics import evaluate
from wildfire_nowcast.model import degrade
from wildfire_nowcast.model.degrade import (
    MODE_SHAPE,
    MODE_SHIFT,
    CellOrder,
    degrade_samples,
    realised_displacement_km,
)

#: The four shape levels the real ladder uses (``runs/_m24_ladder.py``).
SHAPE_LEVELS = (0.05, 0.15, 0.40, 1.00)

#: The displacement levels the real ladder uses, in cells. Quoted here because a
#: comparison between the two families is only about the ladder if it stays
#: inside the rungs the ladder actually has. Past this range the shift family
#: stops being monotone on a two-body state too - measured in I23, at 17 and 20
#: cells, where the second body's spurious mass dominates the reverse term - and
#: a matched pair drawn from out there would be comparing two extrapolations.
LADDER_SHIFT_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

HORIZON = 3

#: Small on purpose. The effect under test is geometric, so it does not need the
#: 24 members the real ladder uses; every claim below is unanimous across the
#: independent windows at this size, which the assertions check rather than
#: assume.
MEMBERS = 8

#: Independent draws per configuration. A ladder is never read one window at a
#: time: the acceptance test takes a paired mean over every growth window of a
#: held-out block and reports a sign count beside it, so the property is asserted
#: on the same shape of statistic.
N_WINDOWS = 6

#: [I23] MEASURED, AND THE REVERSE OF WHAT THIS FILE CALLED THEM AT I21.
#: :func:`_grow` dilates a mask in the direction OPPOSITE to the neighbour offset
#: it is handed - offset ``dx = -1`` copies column ``j`` to column ``j + 1`` - so
#: a head built from the offsets nearest bearing ``pi`` runs toward ``+x``, which
#: is where the second body sits. Every number I21 published is unaffected; the
#: two NAMES on them were swapped, and the state that inverts is the one whose
#: head runs INTO the second body, not away from it. Pinned by a test below so
#: the names cannot drift off the geometry again.
HEAD_INTO_SECOND_BODY = float(np.pi)
HEAD_AWAY_FROM_SECOND_BODY = 0.0

#: The four bearings the repair was swept at, in ARRAY coordinates.
HEAD_BEARINGS = (0.0, float(np.pi / 2), float(np.pi), float(3 * np.pi / 2))

#: The state I21 built. Same identifier, because it is the same state and I21's
#: numbers are quoted against this name; it is no longer a plant against the
#: shipped module and it is still the plant against the pre-repair construction.
TWO_BODY_STATE = "PLANT-I21-B2 (a second burned body 47 km away, in x0)"

#: The regime, not the state: the shipped module with the pre-repair single
#: anchor restored in memory. Named in the failure text so a reader of a red
#: suite learns which REGIME produced it.
SINGLE_ANCHOR_REGIME = "PLANT-I23-A1 (the pre-repair single anchor, restored in memory)"

#: A score difference smaller than this is treated as NO MOVEMENT rather than as
#: a movement of the right sign. A rung that changes the score by a rounding
#: error is not a degradation, and "positive" has to exclude zero or the property
#: passes on an inert rung.
NULL_TOLERANCE = 1e-12

_NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


# --------------------------------------------------------------------------
# the states, built so the right answer is known without running anything
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One scoring window: ``x0`` ``[H,W]``, ``truth`` ``[L,H,W]``, samples ``[M,L,H,W]``."""

    x0: np.ndarray
    truth: np.ndarray
    samples: np.ndarray


def _disc(shape: tuple[int, int], row: int, col: int, radius: int) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (ys - row) ** 2 + (xs - col) ** 2 <= radius * radius


def _head_offsets(bearing: float, half_width_deg: float = 50.0) -> list[tuple[int, int]]:
    """The 8-neighbour steps within ``half_width_deg`` of ``bearing``.

    Growth has to be ANISOTROPIC or the whole exercise measures the wrong thing.
    An isotropic dilation makes the predicted increment a complete annulus, and a
    lobe drawn out of the same annulus overlaps it whichever way it points, so no
    shape rung can express much severity. Real spread is a wind-driven head, and
    the head bearing turns out to be one of the variables the rung's level fails
    to name.

    The offsets are the neighbours nearest ``bearing``; :func:`_grow` then
    dilates OPPOSITE to them, so the head runs at ``bearing + pi``. That is
    measured, not assumed - see the test that pins the two head constants.
    """
    chosen = []
    for dy, dx in _NEIGHBOURS:
        angle = np.arctan2(dy, dx)
        delta = abs(np.arctan2(np.sin(angle - bearing), np.cos(angle - bearing)))
        if np.degrees(delta) <= half_width_deg:
            chosen.append((dy, dx))
    return chosen


def _grow(mask: np.ndarray, rings: int, offsets: Sequence[tuple[int, int]]) -> np.ndarray:
    out = mask.copy()
    for _ in range(int(rings)):
        nxt = out.copy()
        for dy, dx in offsets:
            shifted = np.zeros_like(out)
            src_y = slice(max(dy, 0), out.shape[0] + min(dy, 0))
            dst_y = slice(max(-dy, 0), out.shape[0] + min(-dy, 0))
            src_x = slice(max(dx, 0), out.shape[1] + min(dx, 0))
            dst_x = slice(max(-dx, 0), out.shape[1] + min(-dx, 0))
            shifted[dst_y, dst_x] = out[src_y, src_x]
            nxt |= shifted
        out = nxt
    return out


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """A SPATIALLY CORRELATED field in ``[0,1]``, three box blurs of white noise.

    Per-member spread has to come from coherent blobs rather than independent
    per-cell coins. Salt-and-pepper members are ragged, and a mild reorder of a
    ragged member COMPACTS it, which is a way of scoring better that has nothing
    to do with the geometry under test. Correlated innovations are also what this
    project's own kernel produces, since its noise is shared through a per-step
    latent.
    """
    field = rng.random(shape)
    for _ in range(3):
        padded = np.pad(field, 1, mode="edge")
        field = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] + field
        ) / 5.0
    low, high = float(field.min()), float(field.max())
    return (field - low) / (high - low) if high > low else field


def two_body_window(
    *,
    gap_km: int,
    second_body: bool,
    head_bearing: float = HEAD_INTO_SECOND_BODY,
    r_near: int = 6,
    r_far: int = 8,
    grow_far: int = 4,
    head_rate: int = 3,
    members: int = MEMBERS,
    seed: int = 0,
    margin: int = 34,
) -> Window:
    """A fire with an optional SECOND burned body ``gap_km`` away, and a forecast of it.

    The near body runs a head at ``head_rate`` cells per hour and that head IS the
    truth. The second body does not grow in truth, which is the configuration
    this was built to reproduce: on the multi-ignition held-out fire, 98.8 percent
    of the truth increment sits on the largest body while a third of the scored
    growth band sits around the other one.

    Every member advances BOTH bodies, because both carry burning cells at ``t0``
    and any spread model advances from all of them. The reference forecast is
    therefore right about the near body and carries spurious mass on the far one,
    which is the error the pre-repair shape rung used to remove. Each member also
    carries its own correlated susceptibility field, so no two members are
    bit-identical and a best-member statistic cannot be decided by a tie.

    ``second_body=False`` yields the SAME near body, the same truth and the same
    member draws on a grid of the same shape, so the two states differ in the
    second body and in nothing else.
    """
    offsets = _head_offsets(head_bearing)
    gap = int(gap_km)
    height = 2 * margin + 2 * max(r_near, r_far)
    row = height // 2
    col_near = margin + r_near
    col_far = col_near + r_near + gap + r_far
    width = col_far + r_far + margin
    shape = (height, width)

    near = _disc(shape, row, col_near, r_near)
    far = _disc(shape, row, col_far, r_far)

    x0 = np.zeros(shape, dtype=np.uint8)
    x0[near] = 2
    x0[near & ~_disc(shape, row, col_near, r_near - 1)] = 1
    if second_body:
        x0[far] = 2
        x0[far & ~_disc(shape, row, col_far, r_far - 1)] = 1
    burned0 = x0 > 0

    truth = np.zeros((HORIZON, *shape), dtype=np.uint8)
    acc = near.copy()
    for lead in range(HORIZON):
        acc = _grow(acc, head_rate, offsets)
        frame = np.where(burned0, x0, 0).astype(np.uint8)
        frame[acc & ~burned0] = 1
        truth[lead] = frame

    rng = np.random.default_rng(20260823 + seed)
    samples = np.zeros((members, HORIZON, *shape), dtype=np.uint8)
    for member in range(members):
        near_rate = max(head_rate + int(rng.integers(-1, 2)), 0)
        far_rate = grow_far + int(rng.integers(-1, 2))
        keep = _smooth_field(rng, shape) > 0.42
        a = near.copy()
        b = far.copy()
        for lead in range(HORIZON):
            a = (_grow(a, near_rate, offsets) & keep) | a
            if second_body and far_rate > 0:
                b = (_grow(b, far_rate, offsets) & keep) | b
            grown = (a | b) if second_body else a
            frame = np.where(burned0, x0, 0).astype(np.uint8)
            frame[grown & ~burned0] = 1
            samples[member, lead] = frame
    return Window(x0=x0, truth=truth, samples=samples)


def windows_at(**kwargs: object) -> list[Window]:
    """``N_WINDOWS`` independent draws of one configuration."""
    return [two_body_window(seed=seed, **kwargs) for seed in range(N_WINDOWS)]  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the pre-repair construction, restored in memory
# --------------------------------------------------------------------------


def _one_body(seed_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label every burned cell as ONE body, whatever its connectivity."""
    return (np.asarray(seed_mask) > 0).astype(np.int32), 1


@contextmanager
def single_anchor_construction() -> Iterator[None]:
    """Run the block with the PRE-REPAIR construction: one anchor for the scene.

    The repair acts by labelling the t0 burn into bodies; with the labelling
    forced to report a single body the shipped module builds exactly what it
    built before it - one centroid, one bearing field, one ring field seeded from
    every burned cell at once, and one global competition for the lobe - because
    with one component the grouping key IS ``not_free``. It is the module's own
    documented patch point, and it is the only line this file patches.

    This restores the CONSTRUCTION, not the source. If the repair is ever removed
    by hand this context manager will not attach - ``_seed_components`` will not
    be there - and the tests that use it turn red rather than quietly measuring
    the same thing twice.
    """
    assert hasattr(degrade, "_seed_components"), (
        "the patch point this plant attaches to is gone from model/degrade.py, so the "
        "pre-repair construction can no longer be restored in memory and the inversion "
        "guard below is not running. Re-point the plant at whatever replaced it."
    )
    original = degrade._seed_components
    degrade._seed_components = _one_body  # type: ignore[assignment]
    try:
        yield
    finally:
        degrade._seed_components = original  # type: ignore[assignment]


# --------------------------------------------------------------------------
# scoring, through the entry point the acceptance test uses
# --------------------------------------------------------------------------


def _score(window: Window, samples: np.ndarray, lead: int = HORIZON) -> tuple[float, float]:
    """``(front_distance_crps.combined_cond, best_member_iou_shape_masked)`` at LEAD.

    Both channels, because the acceptance test reads both and they are supposed
    to disagree about POWER, not about SIGN.
    """
    res = evaluate(samples, window.truth, x0=window.x0, leads=(1, 2, 3))
    band = res["by_mask"]["growth_band"]
    front = band["front_distance_crps"]["by_horizon"][str(lead)]["combined_cond"]
    iou = band[f"{GATE_CRITERION_KEY}_by_horizon"][lead - 1]
    assert front is not None and iou is not None, "the window must be admissible to both channels"
    return float(front), float(iou)


def _window_ladder(
    window: Window, mode: str = MODE_SHAPE, levels: Sequence[float] = SHAPE_LEVELS
) -> list[dict[str, float]]:
    order = CellOrder(window.samples, window.x0)
    ref_front, ref_iou = _score(window, window.samples)
    rows: list[dict[str, float]] = []
    for level in levels:
        deg = degrade_samples(window.samples, window.x0, mode=mode, level=level, order=order)
        front, iou = _score(window, deg)
        relocation = realised_displacement_km(deg, window.samples, window.x0)
        rows.append(
            {
                "level": float(level),
                "relocation_km": 0.0 if relocation is None else float(relocation),
                "d_front": front - ref_front,
                "d_iou": iou - ref_iou,
            }
        )
    return rows


def ladder(
    windows: Sequence[Window], mode: str = MODE_SHAPE, levels: Sequence[float] = SHAPE_LEVELS
) -> list[dict[str, float]]:
    """PAIRED MEAN per rung over WINDOWS, with the sign count, as the ladder is read."""
    per_window = [_window_ladder(window, mode, levels) for window in windows]
    rows: list[dict[str, float]] = []
    for index, level in enumerate(levels):
        cells = [rung[index] for rung in per_window]
        rows.append(
            {
                "level": float(level),
                "n_windows": float(len(cells)),
                "relocation_km": float(np.mean([c["relocation_km"] for c in cells])),
                "d_front": float(np.mean([c["d_front"] for c in cells])),
                "d_iou": float(np.mean([c["d_iou"] for c in cells])),
                "n_front_worse": float(sum(c["d_front"] > NULL_TOLERANCE for c in cells)),
                "n_iou_worse": float(sum(c["d_iou"] < -NULL_TOLERANCE for c in cells)),
            }
        )
    return rows


# --------------------------------------------------------------------------
# THE PROPOSED PROPERTY. Written, not wired: no C number, ADR-128 (7).
# --------------------------------------------------------------------------


def severity_violations(rows: Sequence[Mapping[str, float]], subject: str) -> list[str]:
    """EVERY way ``rows`` fails to be a severity ladder. Empty means it is one.

    PROPOSED, NOT RATIFIED. The property an acceptance test built on this ladder
    already relies on:

    0. every rung actually relocates something, and moves the score - a rung that
       changes nothing is not a mild degradation, it is not a degradation;
    1. realised relocation is non-decreasing in the level;
    2. the paired change is POSITIVE at every level on ``front_distance_crps``,
       where larger is worse, and NEGATIVE at every level on the gate criterion,
       where larger is better;
    3. every window agrees on the sign of the front channel, so the mean is not
       carried by a minority;
    4. the magnitude is non-decreasing in the level.

    Pure: it reads rows and returns strings, so it can be planted against
    directly, including on the zero that no state in this file produces by
    accident. Unanimity is asserted on the front channel only - measured, the
    gate criterion is unanimous on a single-body state at every level and only at
    the two strong levels on a two-body one, and a property should assert what
    holds rather than what would be tidy.

    Every message names ``subject``, so a caller that plants an input or a regime
    learns which from the failure text rather than from a diff.
    """
    out: list[str] = []
    relocations = [float(row["relocation_km"]) for row in rows]
    if not all(b >= a - 1e-9 for a, b in zip(relocations, relocations[1:], strict=False)):
        out.append(
            f"{subject}: realised relocation is not non-decreasing in the level: {relocations}"
        )
    for row in rows:
        head = f"{subject}: shape_f{row['level']:.3f} relocated the forecast by "
        if row["relocation_km"] <= 0.0:
            out.append(
                f"{head}{row['relocation_km']:.3f} km, so it RELOCATES NOTHING. An inert rung "
                "cannot be a rung of a severity ladder, whatever the scores do beside it."
            )
        if row["d_front"] < -NULL_TOLERANCE:
            out.append(
                f"{head}{row['relocation_km']:.3f} km and front_distance_crps got BETTER by "
                f"{-row['d_front']:.4f} km in {int(row['n_windows'] - row['n_front_worse'])} of "
                f"{int(row['n_windows'])} windows. A rung that is nominally more severe produced "
                "a genuinely better forecast, so the ladder's severity variable is not monotone "
                "in badness and no null read off it is interpretable."
            )
        elif abs(row["d_front"]) <= NULL_TOLERANCE:
            out.append(
                f"{head}{row['relocation_km']:.3f} km and DID NOT MOVE front_distance_crps AT "
                f"ALL ({row['d_front']:+.3e} km). A rung the scorer cannot distinguish from the "
                "reference is not a degradation of known severity; it is a null rung wearing a "
                "level, and reading it as either seen or blind is unsupported."
            )
        if row["d_iou"] > NULL_TOLERANCE:
            out.append(
                f"{head}{row['relocation_km']:.3f} km and {GATE_CRITERION_KEY} got BETTER by "
                f"{row['d_iou']:.4f}. Same defect, on the incumbent location criterion."
            )
        elif abs(row["d_iou"]) <= NULL_TOLERANCE:
            out.append(
                f"{head}{row['relocation_km']:.3f} km and DID NOT MOVE {GATE_CRITERION_KEY} AT "
                f"ALL ({row['d_iou']:+.3e}). Same null, on the incumbent location criterion."
            )
        if row["n_front_worse"] != row["n_windows"]:
            out.append(
                f"{subject}: shape_f{row['level']:.3f} is worse in only "
                f"{int(row['n_front_worse'])} of {int(row['n_windows'])} windows, so the mean is "
                "carried by a subset and the rung does not order what it claims to order."
            )
    fronts = [float(row["d_front"]) for row in rows]
    if not all(b >= a - 1e-9 for a, b in zip(fronts, fronts[1:], strict=False)):
        out.append(f"{subject}: the score change is not ordered by level: {fronts}")
    return out


def assert_rows_are_severity_monotone(rows: Sequence[Mapping[str, float]], subject: str) -> None:
    """Raise :class:`AssertionError` carrying EVERY violation, not only the first."""
    violations = severity_violations(rows, subject)
    if violations:
        raise AssertionError("\n".join(violations))


def assert_shape_ladder_is_severity_monotone(windows: Sequence[Window], subject: str) -> None:
    """Score the shape ladder over ``windows`` and require the property of it."""
    assert_rows_are_severity_monotone(ladder(windows), subject)


# --------------------------------------------------------------------------
# THE THREE OBSERVATIONS (ADR-093)
# --------------------------------------------------------------------------


def test_control_a_single_body_state_climbs_the_ladder_the_way_it_is_documented_to() -> None:
    """CONTROL. Must NOT fire. If it does, nothing else in this file means anything.

    One burned body, one anchor inside it, and the ladder behaves exactly as
    ``model/degrade.py`` describes: relocation grows with the level, both channels
    get worse, every window agrees, and the magnitude is ordered. Unchanged by the
    repair, which is a no-op on a single-body state - and identical to the digit
    under the pre-repair construction, which the guard below measures rather than
    assumes.
    """
    assert_shape_ladder_is_severity_monotone(
        windows_at(gap_km=47, second_body=False), "CONTROL (single body)"
    )


def test_a_second_burned_body_now_climbs_the_ladder_LIKE_a_single_body_does() -> None:
    """RE-POINTED [I23]. This test asserted the INVERSION until the repair landed.

    The only difference from the control is the second burned body in ``x0``. It
    used to take the paired difference negative at every level and 0 of 6 windows
    worse; it is now positive at every level and unanimous, which is the property
    the control satisfies.

    The inversion it used to assert is not gone from this file - it is asserted
    under :func:`single_anchor_construction` in the guard below, and this test
    would go red if the repair were reverted.
    """
    rows = ladder(windows_at(gap_km=47, second_body=True))
    assert_rows_are_severity_monotone(rows, TWO_BODY_STATE)
    for row in rows:
        assert row["d_front"] > NULL_TOLERANCE, (
            f"{TWO_BODY_STATE}: shape_f{row['level']:.3f} must make the forecast WORSE: "
            f"{row['d_front']:+.4f} km"
        )
        assert row["n_front_worse"] == row["n_windows"], (
            f"{TWO_BODY_STATE}: shape_f{row['level']:.3f} is worse in only "
            f"{int(row['n_front_worse'])} of {int(row['n_windows'])} windows"
        )


def test_restored_removing_the_second_body_returns_the_control_state_bit_for_bit() -> None:
    """RESTORED. Must not fire, and must be the control's state and not a lookalike.

    Rebuilt from scratch rather than mutated back, and compared to the control by
    identity on all three arrays, so "restored" is a measurement rather than an
    assumption about what the builder does.
    """
    control = windows_at(gap_km=47, second_body=False)
    restored = windows_at(gap_km=47, second_body=False)
    for before, after in zip(control, restored, strict=True):
        assert np.array_equal(before.x0, after.x0)
        assert np.array_equal(before.truth, after.truth)
        assert np.array_equal(before.samples, after.samples)
    assert_shape_ladder_is_severity_monotone(restored, "RESTORED (single body)")


# --------------------------------------------------------------------------
# THE GUARD: the inversion must not be able to come back unnoticed
# --------------------------------------------------------------------------


def test_GUARD_the_inversion_returns_the_moment_the_single_anchor_is_restored() -> None:
    """The old knowledge, kept executable, as a DIFFERENTIAL on one line of code.

    Same states, same truth, same member draws, same rung set, same scorer. The
    only thing that changes between the three observations is whether the t0 burn
    is labelled into bodies:

    CONTROL   shipped module      positive at every level, 6 of 6 windows worse
    PLANT     single anchor       negative at every level, 0 of 6, and the
                                  property fires SAYING the forecast got better
    RESTORED  shipped module      positive again, and equal to the control

    So the repair cannot be reverted while this suite is green. Revert the
    construction and CONTROL fails; remove the patch point and the PLANT cannot
    attach and says so; weaken the property and the row-level plant below fails.
    """

    def rows() -> list[dict[str, float]]:
        return ladder(windows_at(gap_km=47, second_body=True))

    control = rows()
    assert not severity_violations(control, TWO_BODY_STATE), (
        "the shipped module must climb the ladder on the two-body state before this test can "
        f"say anything about the plant: {severity_violations(control, TWO_BODY_STATE)}"
    )

    with single_anchor_construction():
        reverted = rows()
    violations = severity_violations(reverted, SINGLE_ANCHOR_REGIME)
    assert violations, (
        f"{SINGLE_ANCHOR_REGIME} did not fire. Either the pre-repair construction no longer "
        "inverts this state - in which case the repair is not what fixed it - or the patch went "
        "blind, and a plant that cannot fail is a check that cannot fail."
    )
    assert any("got BETTER" in message for message in violations), (
        f"the plant fired for the wrong reason: {violations}"
    )
    for row in reverted:
        assert row["d_front"] < -NULL_TOLERANCE and row["n_front_worse"] == 0.0, (
            "the pre-repair construction is supposed to invert at EVERY level and in EVERY "
            f"window: level {row['level']:.3f} reads {row['d_front']:+.4f} km in "
            f"{int(row['n_front_worse'])} of {int(row['n_windows'])} windows"
        )
        assert row["d_iou"] > NULL_TOLERANCE, (
            "and to invert on the incumbent criterion too: level "
            f"{row['level']:.3f} reads {row['d_iou']:+.4f}"
        )
    assert reverted[-1]["relocation_km"] > control[-1]["relocation_km"], (
        "the inverted ladder must also be the one that relocated FURTHER, which is what made it "
        f"a defect rather than a weak rung: {reverted[-1]['relocation_km']:.3f} km against "
        f"{control[-1]['relocation_km']:.3f} km"
    )

    restored = rows()
    assert not severity_violations(restored, TWO_BODY_STATE), (
        "the patch did not come back off: the module still fails the property after the plant"
    )
    for before, after in zip(control, restored, strict=True):
        assert before["d_front"] == after["d_front"] and before["d_iou"] == after["d_iou"], (
            "restored is not the control to the digit, so the plant left something behind: "
            f"{before} against {after}"
        )


def test_the_guard_REFUSES_if_the_patch_point_it_attaches_to_is_renamed_away() -> None:
    """The plant's own attachment is checked, because a plant that does not attach passes.

    If the repair is ever rewritten and the labelling call is renamed, the guard
    above must stop rather than run both of its observations against the shipped
    module and report agreement with itself. Nothing is scored inside the window
    where the name is missing.
    """
    original = degrade._seed_components
    delattr(degrade, "_seed_components")
    try:
        with single_anchor_construction():
            raise AssertionError("the guard attached to a patch point that is not there")
    except AssertionError as exc:
        assert "patch point" in str(exc), f"it refused for the wrong reason: {exc}"
    finally:
        degrade._seed_components = original  # type: ignore[assignment]
    with single_anchor_construction():
        assert degrade._seed_components is _one_body, "restored, and the plant attaches again"


def test_the_severity_property_fires_on_a_NEGATIVE_and_on_a_ZERO() -> None:
    """The property is planted DIRECTLY, one clause at a time, both signs and zero.

    A check that fires on a negative and is silent on a zero is half blind, and
    this file has already found one control that was exactly that. No state built
    here produces a zero by accident, so the zero is planted here and again, live,
    on the ladder's own null rung in the next test.

    The rows are the shape ladder's own measured shape, so the control is a
    ladder the property accepts and every plant is one field away from it.
    """

    def base() -> list[dict[str, float]]:
        return [
            {
                "level": level,
                "n_windows": 6.0,
                "relocation_km": km,
                "d_front": front,
                "d_iou": iou,
                "n_front_worse": 6.0,
                "n_iou_worse": 6.0,
            }
            for level, km, front, iou in zip(
                SHAPE_LEVELS,
                (1.8, 5.0, 10.4, 17.5),
                (0.065, 0.240, 0.954, 2.832),
                (-0.010, -0.025, -0.108, -0.311),
                strict=True,
            )
        ]

    assert severity_violations(base(), "CONTROL rows") == [], "the control rows must be accepted"

    def fires(mutate: dict[str, float], expect: str) -> None:
        rows = base()
        rows[-1].update(mutate)
        messages = severity_violations(rows, "PLANTED rows")
        assert any(expect in message for message in messages), (
            f"planting {mutate} must be caught by the phrase {expect!r}, got {messages}"
        )
        assert all("PLANTED rows" in message for message in messages)

    fires({"d_front": -2.832, "n_front_worse": 0.0}, "got BETTER")
    fires({"d_front": 0.0}, "DID NOT MOVE front_distance_crps")
    fires({"d_front": -1e-15}, "DID NOT MOVE front_distance_crps")
    fires({"d_iou": +0.311, "n_iou_worse": 0.0}, "got BETTER")
    fires({"d_iou": 0.0}, f"DID NOT MOVE {GATE_CRITERION_KEY}")
    fires({"relocation_km": 0.0}, "RELOCATES NOTHING")
    fires({"n_front_worse": 5.0}, "carried by a subset")
    fires({"relocation_km": 1.0}, "not non-decreasing")
    fires({"d_front": 0.5}, "not ordered by level")


def test_the_ladders_own_null_rung_is_a_LIVE_zero_and_the_property_fires_on_it() -> None:
    """The zero, measured rather than typed: the identity rung of the shift family.

    ``MODE_SHIFT`` at level 0 returns the base order itself, so the degraded
    sample is the reference BITWISE and both channels move by exactly nothing.
    That is the ladder's own documented negative control, and it is the one input
    that proves the "positive" clauses of the property are not reading a sign off
    a number that never moved. Two windows: the claim is per rung, not a
    statistic over windows.
    """
    rows = ladder(windows_at(gap_km=47, second_body=True)[:2], mode=MODE_SHIFT, levels=(0.0,))
    assert rows[0]["d_front"] == 0.0 and rows[0]["d_iou"] == 0.0, (
        f"the identity rung is supposed to be bit-identical to the reference: {rows[0]}"
    )
    messages = severity_violations(rows, "NULL RUNG (shift_d00.0)")
    assert any("DID NOT MOVE front_distance_crps" in message for message in messages), messages
    assert any(f"DID NOT MOVE {GATE_CRITERION_KEY}" in message for message in messages), messages
    assert any("RELOCATES NOTHING" in message for message in messages), messages


# --------------------------------------------------------------------------
# the measurements that name the mechanism and rule the alternatives out
# --------------------------------------------------------------------------


def test_the_head_bearing_no_longer_flips_the_sign_at_ANY_of_the_four_bearings() -> None:
    """RE-POINTED [I23]. This test asserted the sign flip until the repair landed.

    The second body, the gap, the masses and the share of the reference forecast
    that sits on the second body are all held FIXED. Only the head bearing turns.
    It used to take the paired difference from clearly positive to clearly
    negative between one bearing and another, so the severity of a rung depended
    on the angle between the wind and the axis joining the two bodies - a variable
    the level does not name. Every bearing now climbs the ladder, and the
    single-body state still does.
    """
    for bearing in HEAD_BEARINGS:
        subject = f"{TWO_BODY_STATE}, head bearing {np.degrees(bearing):.0f} deg"
        assert_shape_ladder_is_severity_monotone(
            windows_at(gap_km=47, second_body=True, head_bearing=bearing), subject
        )
    assert_shape_ladder_is_severity_monotone(
        windows_at(gap_km=47, second_body=False, head_bearing=HEAD_AWAY_FROM_SECOND_BODY),
        "CONTROL (single body), the other bearing",
    )


def test_the_head_bearing_constants_name_the_direction_the_head_actually_runs() -> None:
    """[I23] The two constants were SWAPPED in this file until today.

    ``_grow`` dilates opposite to the offset it is given, so the head runs at
    ``bearing + pi``. Every number I21 published stands; two of its labels did
    not. Both constants are asserted, in both directions, so a swap fails and a
    helper that always answered the same way fails too.
    """

    def head_runs_toward_the_second_body(bearing: float) -> bool:
        window = two_body_window(gap_km=47, second_body=True, head_bearing=bearing)
        burned0 = window.x0 > 0
        labels, n_bodies = label_components(burned0)
        assert n_bodies == 2
        _, xs = np.mgrid[0 : burned0.shape[0], 0 : burned0.shape[1]]
        centres = [float(xs[labels == body].mean()) for body in (1, 2)]
        near, far = min(centres), max(centres)
        truth_increment = ((window.truth > 0) & ~burned0[None]).any(axis=0)
        toward = float(xs[truth_increment].mean()) > near
        assert far > near
        return toward

    assert head_runs_toward_the_second_body(HEAD_INTO_SECOND_BODY), (
        "HEAD_INTO_SECOND_BODY must run the head at the second body"
    )
    assert not head_runs_toward_the_second_body(HEAD_AWAY_FROM_SECOND_BODY), (
        "HEAD_AWAY_FROM_SECOND_BODY must run the head away from the second body"
    )


def test_a_translation_of_the_same_realised_km_now_moves_the_score_the_SAME_way() -> None:
    """RE-POINTED [I23]. Two families, one window, severity matched in kilometres.

    This ruled out "the reference was bad, so any relocation helps": ``MODE_SHIFT``
    is the family whose severity unit IS kilometres, and it degraded where a shape
    rung of the same realised relocation improved. The two families now AGREE in
    sign at every matched pair, and under the pre-repair construction the same
    search on the same window still finds a pair that disagrees - which is the
    I21 measurement, kept.

    The shift levels are the ladder's own. Beyond them the shift family is not
    monotone on this state either, so a matched pair taken from out there would
    compare two extrapolations rather than two rungs.
    """
    window = two_body_window(gap_km=47, second_body=True)

    def matched_pairs() -> list[tuple[float, tuple[float, ...], tuple[float, ...]]]:
        order = CellOrder(window.samples, window.x0)
        ref_front, ref_iou = _score(window, window.samples)

        def rungs(mode: str, levels: Sequence[float]) -> list[tuple[float, float, float, float]]:
            out = []
            for level in levels:
                deg = degrade_samples(
                    window.samples, window.x0, mode=mode, level=level, order=order
                )
                km = realised_displacement_km(deg, window.samples, window.x0)
                if km is None or km <= 0.0:
                    continue
                front, iou = _score(window, deg)
                out.append((float(level), float(km), front - ref_front, iou - ref_iou))
            return out

        shapes = rungs(MODE_SHAPE, SHAPE_LEVELS)
        shifts = rungs(MODE_SHIFT, LADDER_SHIFT_LEVELS)
        assert shapes and shifts
        return [
            (abs(a[1] - b[1]) / a[1], a, b)
            for a in shapes
            for b in shifts
            if abs(a[1] - b[1]) / a[1] < 0.12
        ]

    pairs = matched_pairs()
    assert pairs, "no shape rung and shift rung agree on realised severity to within twelve percent"
    for gap, shape_rung, shift_rung in pairs:
        assert shape_rung[2] > 0.0 and shift_rung[2] > 0.0, (
            f"a shape rung at {shape_rung[1]:.3f} km and a translation at {shift_rung[1]:.3f} km "
            f"({gap * 100:.1f} percent apart) must degrade the SAME way: front "
            f"{shape_rung[2]:+.4f} against {shift_rung[2]:+.4f}"
        )
        assert shape_rung[3] < 0.0 and shift_rung[3] < 0.0, (
            f"and on the incumbent criterion: {GATE_CRITERION_KEY} {shape_rung[3]:+.4f} against "
            f"{shift_rung[3]:+.4f}"
        )

    with single_anchor_construction():
        pre_pairs = matched_pairs()
    assert any(shape_rung[2] < 0.0 < shift_rung[2] for _, shape_rung, shift_rung in pre_pairs), (
        f"{SINGLE_ANCHOR_REGIME}: the two families are supposed to DISAGREE in sign at a matched "
        f"severity before the repair, and no matched pair does: {pre_pairs}"
    )


def test_the_censoring_cap_is_not_the_mechanism() -> None:
    """Rules out saturation against the 12 km cap, on both sides of the repair.

    If the inversion had been the scorer running out of range, the censored
    fraction would be large on at least one side of it. It is near zero on the
    reference, on the strongest shape rung as shipped, and on the same rung with
    the pre-repair construction restored - so whatever was happening was happening
    well inside the scorer's dynamic range, and the repair did not move it there.
    """
    window = two_body_window(gap_km=47, second_body=True)

    def censored(samples: np.ndarray) -> float:
        res = evaluate(samples, window.truth, x0=window.x0, leads=(1, 2, 3))
        terms = res["by_mask"]["growth_band"]["front_distance_crps"]["by_horizon"]["3"]
        return float(terms["censored_fraction_cond"])

    def strongest_shape() -> np.ndarray:
        order = CellOrder(window.samples, window.x0)
        return degrade_samples(window.samples, window.x0, mode=MODE_SHAPE, level=1.0, order=order)

    cases = {"reference": window.samples, "shape_f1.000": strongest_shape()}
    with single_anchor_construction():
        cases["shape_f1.000 (single anchor)"] = strongest_shape()
    for name, samples in cases.items():
        fraction = censored(samples)
        assert fraction < 0.05, (
            f"{name} is censored at {fraction:.3f}, so this state cannot separate the inversion "
            "from saturation and the plant needs a different geometry"
        )
    # The third case has to BE a different array, or a restoration that quietly
    # stopped restoring would leave this test measuring the shipped rung twice
    # and reporting it as evidence about two regimes.
    shipped_km = realised_displacement_km(cases["shape_f1.000"], window.samples, window.x0)
    anchored_km = realised_displacement_km(
        cases["shape_f1.000 (single anchor)"], window.samples, window.x0
    )
    assert shipped_km is not None and anchored_km is not None
    assert anchored_km > shipped_km + 1.0, (
        "the single-anchor rung relocated no further than the shipped one, so the two cases above "
        f"are not two regimes: {anchored_km:.3f} km against {shipped_km:.3f} km"
    )


def test_each_body_is_ranked_from_its_OWN_centroid_and_one_anchor_corrupts_the_field() -> None:
    """[I23] The anchor claim, measured on the FIELD and not only on its summary.

    The lobe is ranked by ``cos(bearing - target)``, so what has to be right is
    the bearing of every cell, not the one summary bearing per body. Measured
    against an independent reference - each body's own share of the pooled
    increment, assigned by Euclidean nearness to the body centres this module
    placed, which is not the geodesic rule the module uses:

    CONTROL   shipped module   per-body summary error 0.0 deg, field error 0.0 deg
    PLANT     single anchor    summary error still under 12 deg on BOTH bodies
                               while the field is wrong by up to 180
    RESTORED  shipped module   back to 0.0

    That gap is why an error metric on the summary alone was never the diagnostic:
    a scene-wide anchor can produce a summary bearing right to a tenth of a degree
    while every cell it ranks is measured from a point in empty space.
    """
    window = two_body_window(gap_km=47, second_body=True)
    burned0 = window.x0 > 0
    labels, n_bodies = label_components(burned0)
    assert n_bodies == 2
    ys, xs = np.mgrid[0 : burned0.shape[0], 0 : burned0.shape[1]]
    centres = [
        (float(ys[labels == body].mean()), float(xs[labels == body].mean()))
        for body in range(1, n_bodies + 1)
    ]
    pooled = ((window.samples > 0) & ~burned0[None, None]).any(axis=(0, 1))
    nearest = np.argmin(
        np.stack([np.hypot(ys - cy, xs - cx) for cy, cx in centres], axis=0), axis=0
    )

    def wrapped_deg(delta: np.ndarray) -> np.ndarray:
        radians = np.radians(delta)
        return np.abs(np.degrees(np.arctan2(np.sin(radians), np.cos(radians))))

    def errors() -> list[tuple[float, float]]:
        order = CellOrder(window.samples, window.x0)
        phi = np.atleast_1d(np.asarray(order.phi_base, dtype=np.float64))
        used = np.degrees(np.asarray(order.bearing, dtype=np.float64)).reshape(burned0.shape)
        out = []
        for body, (cy, cx) in enumerate(centres):
            share = pooled & (nearest == body)
            assert share.any()
            own = np.degrees(np.arctan2(ys[share].mean() - cy, xs[share].mean() - cx))
            summary = float(np.degrees(phi[body if phi.size > 1 else 0]))
            field = wrapped_deg(
                used[share] - np.degrees(np.arctan2(ys[share] - cy, xs[share] - cx))
            )
            out.append((float(wrapped_deg(np.array(summary - own))), float(field.max())))
        return out

    control = errors()
    for body, (summary, field) in enumerate(control):
        assert summary < 1e-6 and field < 1e-6, (
            f"body {body}: the shipped module must measure both the summary bearing and every "
            f"cell's bearing from that body's own centroid: {summary:.3f} / {field:.3f} deg"
        )

    with single_anchor_construction():
        planted = errors()
    assert all(summary < 12.0 for summary, _ in planted), (
        "the point of this measurement is that the SUMMARY stays plausible under one anchor: "
        f"{planted}"
    )
    assert max(field for _, field in planted) > 90.0, (
        f"the single anchor must corrupt the bearing FIELD, and it did not: {planted}"
    )

    assert errors() == control, "the patch did not come back off"
