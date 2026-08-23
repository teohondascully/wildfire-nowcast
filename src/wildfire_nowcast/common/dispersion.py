"""G3's dispersion half: a **GEOMETRIC** bar, plus an explicit FIRST-MOMENT condition.

This module is the single implementation (C0) of the two conditions G3 is
adjudicated on, and of the one outcome type they are allowed to return. Nothing
here decides G3 - that verdict is the maintainer's, exactly as in ``eval``'s
``g3_summary``. What is decided here is *what a pass is allowed to mean*.

THREE DEFECTS IN THE OLD BAR, ALL MEASURED, ALL CLOSED HERE
-----------------------------------------------------------

**1. ``[0.8, 1.2]`` IS NOT SYMMETRIC, AND IT WAS ASYMMETRIC IN OUR FAVOUR.**
``adr`` is a RATIO, so "equally wrong in both directions" means equally far in
LOG space. ``1 / 0.8 = 1.25 > 1.2``: the old interval reached ~4% further into
under-dispersion than into over-dispersion (in log units, ``|log 0.8| = 0.2231``
against ``|log 1.2| = 0.1823`` - **22% more tolerance**, which is the number
ADR-037 (6) quotes). **We have under-dispersed on every arm ever run** - 6 of 56
arms inside the bar across four G3 attempts, all of them from below. So the
looser side of the bar was the side we live on, and the bar was flattering the
candidate.

The replacement is ``|log(adr)| <= log(BAR_RATIO)`` with ``BAR_RATIO = 1.2``,
i.e. ``[1/1.2, 1.2] = [0.8333, 1.2]``. **Strictly harder, and harder only on the
side we fail.** Bars only ever get harder (ADR-020's rule), and this one is
tightened while no arm has been scored on the current corpus - nobody can see
which way it cuts.

**2. A DISPERSION RATIO IS NOT INTERPRETABLE WITHOUT ITS FIRST MOMENT.**
ADR-035 established, at a residual of 2.2e-16 (machine epsilon - an identity, not
a fit)::

    adr = sqrt((M+1)/M) x ensemble_CV x growth_calibration x truth_shape x relief

``ensemble_CV`` is near-constant across blocks (1.24-1.43x). ``growth_calibration``
varies 3-7x and **dominates**. The denominator of ``adr`` is the model's own mean
error, so a model that is wrong about the MEAN can land inside a dispersion bar by
being wrong about the SPREAD in a compensating direction. **Under-dispersion, on
our own numbers, IS the growth mis-calibration re-expressed.** A gate that reads
only ``adr`` can therefore be passed by two errors cancelling.

:func:`first_moment_condition` closes that. It is defined **relative to the best
physics baseline and never as an absolute number**: the candidate's
``|log(growth_calibration)|`` must be no worse than the wind-advected ellipse's,
on the same held-out blocks under the same equal-block weighting. Reference-based
and threshold-free - there is no constant here fitted to anything, which is what
C-3 requires and what makes this admissible mid-gate.

Stated in advance (ADR-039 (5)), because it is already known to FAIL: we
over-predict mean growth 2.66-3.06x and the ellipse over-predicts 1.79x. That is
the point. The condition was invisible before, and G3 could have been passed on a
compensating error.

**3. ``area_dispersion_ratio`` IS UNDEFINED AT PERFECT MEAN CALIBRATION.**
Found by sim by building a playthrough whose control arm was a perfectly
calibrated ensemble and being unable to score it: the denominator is the model's
own mean-area error, which is exactly 0 when the mean is perfect, so the metric
returns ``None``. **A ``None`` must never be read as a pass.** That is not a
hypothetical - ``None`` is falsy, ``None <= x`` raises, ``low <= None <= high``
raises, and a ``dict.get`` of a missing key is also ``None``, so the three ways to
get here are indistinguishable at the call site.

The type system is used to make it impossible rather than merely discouraged:
:class:`ConditionResult` has three outcomes (``PASS`` / ``FAIL`` / ``UNDEFINED``)
and **raises on ``bool()``**. ``if result:`` does not silently take a branch; it
stops. This is the same move ``environments_agree`` made for missing fingerprints
(``None`` never compares equal to ``None``) and the same lesson as C1.5's rule
that an unevaluable comparison must be unpassable at the verdict choke point.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
A G3 verdict. :func:`g3_conditions` returns both conditions, always together and
always separately, plus a combined outcome that is ``PASS`` only if BOTH pass and
``UNDEFINED`` if either is unmeasurable. "Not adjudicable" is a distinct state
from "failed" and this project has already had to use it once (G4 at n=2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PASS",
    "FAIL",
    "UNDEFINED",
    "BAR_RATIO",
    "BAR_INTERVAL",
    "GATE_CRITERION_KEY",
    "GATE_MASK",
    "FIRST_MOMENT_KEY",
    "FIRST_MOMENT_REFERENCE_ROLE",
    "ConditionResult",
    "log_distance",
    "growth_calibration",
    "dispersion_condition",
    "first_moment_condition",
    "first_moment_condition_from_blocks",
    "g3_conditions",
    "combine",
    "UndefinedConditionError",
]

# --------------------------------------------------------------------------
# outcomes
# --------------------------------------------------------------------------

#: The three outcomes a G3 condition may have. ``UNDEFINED`` is a first-class
#: outcome, not a missing ``PASS``: the metric can be genuinely unmeasurable
#: (``area_dispersion_ratio`` at perfect mean calibration) and that is a fact
#: about the instrument, not a verdict about the model.
PASS = "pass"
FAIL = "fail"
UNDEFINED = "undefined"

#: The dispersion bar, as a RATIO. ``|log(adr)| <= log(BAR_RATIO)``.
#:
#: C-3 (no threshold calibrated on n=1) - FITTING SAMPLE: **none, and that is the
#: point.** 1.2 is inherited unchanged from ADR-020's pre-fixed bar, which was
#: itself set before any ensemble existed. A14 changed only the GEOMETRY of the
#: interval, not its magnitude: the endpoint we were already held to (1.2) is
#: kept, and the other endpoint is derived from it as its reciprocal instead of
#: being a second free number. There is one constant here where there were two,
#: and no arm's score participated in choosing it.
BAR_RATIO = 1.2

#: The bar as an interval, DERIVED - ``(1/1.2, 1.2) == (0.8333.., 1.2)``. Exposed
#: for printing only. Never hand-write the low endpoint: a second literal is how
#: the asymmetry got in, and ``0.8333`` rounded into a table is not ``1/1.2``.
BAR_INTERVAL: tuple[float, float] = (1.0 / BAR_RATIO, BAR_RATIO)

#: The key G3's dispersion half is adjudicated on (C6.1 / ADR-011). Named once,
#: in code, for the reason C6.4 and the calibration module give: ``dispersion_ratio``
#: scores a COLLAPSED ensemble at 1.000 and a healthy one at 1.051, so a table
#: that picks the wrong key passes the exact thing G3 exists to reject.
GATE_CRITERION_KEY = "area_dispersion_ratio"

#: ...under the growth mask. The domain-wide value is diluted by cells nobody was
#: ever uncertain about.
GATE_MASK = "growth_band"

#: The first moment: ``mean predicted area / mean truth area``. 1.0 is perfect,
#: >1 over-predicts, <1 under-predicts. It is a RATIO, so it is compared in log
#: space like the bar above.
FIRST_MOMENT_KEY = "growth_calibration"

#: WHOSE first moment the candidate is measured against. A physics baseline, and
#: specifically the one C6.2 already requires to be growth-calibrated on train
#: fires - so the reference is a number somebody else's method produced under a
#: rule written before ours. Using the best physics baseline rather than a
#: constant is what keeps this condition free of a fitted threshold.
FIRST_MOMENT_REFERENCE_ROLE = "wind_ellipse"


class UndefinedConditionError(TypeError):
    """Raised when a condition's outcome is coerced to a bool."""


@dataclass(frozen=True)
class ConditionResult:
    """One G3 condition: an outcome, the numbers behind it, and why.

    **This type raises on ``bool()``.** That is not defensive styling; it is the
    only reliable way to stop ``UNDEFINED`` being read as a pass. ``None`` is
    falsy, so ``in_interval = low <= value <= high`` explodes while
    ``if in_interval:`` quietly takes the "not in bar" branch and
    ``all([...])`` over a list containing ``None`` returns ``False`` - three
    different silent behaviours for the same missing measurement. Making the
    truth value an error forces every consumer to name which outcome it means.
    """

    name: str
    outcome: str
    value: float | None = None
    reference: float | None = None
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in (PASS, FAIL, UNDEFINED):
            raise ValueError(f"illegal outcome {self.outcome!r} for {self.name}")

    def __bool__(self) -> bool:  # pragma: no cover - the message IS the behaviour
        raise UndefinedConditionError(
            f"{self.name} has outcome {self.outcome!r}; refusing to coerce it to a bool. "
            "Compare against dispersion.PASS / FAIL / UNDEFINED explicitly. An UNDEFINED "
            "condition is not a pass, and `if result:` cannot express that difference."
        )

    @property
    def passed(self) -> bool:
        """``True`` ONLY for :data:`PASS`. ``UNDEFINED`` is never a pass."""
        return self.outcome == PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.name,
            "outcome": self.outcome,
            "value": self.value,
            "reference": self.reference,
            "detail": self.detail,
            **self.extra,
        }


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------


def _finite_positive(value: Any) -> float | None:
    """A ratio we may take a log of, or ``None``. Rejects bool, str, NaN, inf, <= 0.

    ``str`` is rejected rather than coerced ON PURPOSE, and it was caught by the
    test that plants it: ``float("0.9")`` succeeds, so a stringified table cell -
    the shape a value arrives in when it has been round-tripped through a report,
    a CSV or a JSON field someone quoted - would be silently accepted as a
    measurement. ``bool`` is rejected for the same reason (``float(True) == 1.0``,
    a perfect dispersion score conjured out of a flag). Numpy scalars and anything
    else with ``__float__`` are accepted, because ``np.float32`` is not a
    ``float`` subclass and refusing it would reject real measurements.
    """
    if value is None or isinstance(value, bool | str | bytes):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def log_distance(ratio: Any) -> float | None:
    """``|log(ratio)|`` - how wrong a ratio is, symmetrically. ``None`` if undefined.

    ``log_distance(r) == log_distance(1/r)`` exactly (up to float rounding), which
    is the property the old interval lacked and the whole reason for this module.
    """
    r = _finite_positive(ratio)
    return None if r is None else abs(math.log(r))


def growth_calibration(pred_mean_area: Any, truth_mean_area: Any) -> float | None:
    """``mean predicted area / mean truth area``, or ``None`` when undefined.

    Homed here under C0 so the first-moment condition and any diagnostic that
    reports the same quantity cannot drift apart. Undefined - not 0, not inf -
    when truth grew nothing: a calibration ratio against a zero denominator is
    the degenerate case, and the correct answer is that it cannot be measured.
    """
    pred = pred_mean_area if isinstance(pred_mean_area, int | float) else None
    truth = _finite_positive(truth_mean_area)
    if truth is None or pred is None or isinstance(pred_mean_area, bool):
        return None
    pred = float(pred)
    if not math.isfinite(pred) or pred < 0.0:
        return None
    return pred / truth


# --------------------------------------------------------------------------
# condition 1 - dispersion, on a GEOMETRIC bar
# --------------------------------------------------------------------------


def dispersion_condition(adr: Any, *, name: str = "dispersion") -> ConditionResult:
    """G3 (a): ``|log(area_dispersion_ratio)| <= log(1.2)``.

    ``adr = None`` (the metric's own signal that it is undefined) and ``adr <= 0``
    both yield :data:`UNDEFINED`, never :data:`PASS`.
    """
    d = log_distance(adr)
    bar = math.log(BAR_RATIO)
    if d is None:
        return ConditionResult(
            name=name,
            outcome=UNDEFINED,
            value=None,
            reference=BAR_RATIO,
            detail=(
                f"{GATE_CRITERION_KEY} is undefined ({adr!r}). Its denominator is the model's "
                "OWN mean-area error, which is exactly 0 at perfect mean calibration - so this "
                "metric goes undefined precisely as a model gets the first moment right. "
                "UNDEFINED is reported as its own outcome and is NOT a pass."
            ),
            extra={"bar_interval": list(BAR_INTERVAL), "log_bar": bar, "log_distance": None},
        )
    value = float(adr)
    inside = d <= bar
    side = "under-dispersed" if value < 1.0 else "over-dispersed"
    return ConditionResult(
        name=name,
        outcome=PASS if inside else FAIL,
        value=value,
        reference=BAR_RATIO,
        detail=(
            f"|log({value:.4f})| = {d:.4f} {'<=' if inside else '>'} log({BAR_RATIO}) = {bar:.4f} "
            f"({side}; geometric bar {BAR_INTERVAL[0]:.4f}..{BAR_INTERVAL[1]:.4f}). The bar is "
            "symmetric in LOG space: the old [0.8, 1.2] gave 22% more tolerance to "
            "under-dispersion, which is the only side we have ever failed on."
        ),
        extra={"bar_interval": list(BAR_INTERVAL), "log_bar": bar, "log_distance": d},
    )


# --------------------------------------------------------------------------
# condition 2 - the first moment, RELATIVE TO A PHYSICS BASELINE
# --------------------------------------------------------------------------


def first_moment_condition(
    candidate: Any,
    reference: Any,
    *,
    name: str = "first_moment",
    reference_name: str = FIRST_MOMENT_REFERENCE_ROLE,
) -> ConditionResult:
    """G3 (b): ``|log(candidate gc)| <= |log(reference gc)|``.

    Both arguments are ``growth_calibration`` RATIOS pooled the same way, over the
    same held-out blocks. There is no constant in this comparison: the bar is
    whatever the physics baseline achieved, so it cannot be tuned toward any arm's
    result (C-3) and it moves only when the opponent improves.

    Either side missing is :data:`UNDEFINED`. In particular a missing REFERENCE is
    not a free pass - "we could not score the ellipse" must never read as "we beat
    the ellipse", which is the C6.2 VOID-not-passed rule applied one level up.
    """
    dc, dr = log_distance(candidate), log_distance(reference)
    if dc is None or dr is None:
        which = "candidate" if dc is None else f"reference ({reference_name})"
        return ConditionResult(
            name=name,
            outcome=UNDEFINED,
            value=None if dc is None else float(candidate),
            reference=None if dr is None else float(reference),
            detail=(
                f"{FIRST_MOMENT_KEY} is undefined for the {which} "
                f"(candidate={candidate!r}, reference={reference!r}). A reference that could not "
                "be scored is NOT a bar that was cleared - C6.2's rule: a baseline that "
                "degenerates VOIDS its gate rather than passing it."
            ),
            extra={"reference_role": reference_name},
        )
    ok = dc <= dr
    return ConditionResult(
        name=name,
        outcome=PASS if ok else FAIL,
        value=float(candidate),
        reference=float(reference),
        detail=(
            f"|log({float(candidate):.4f})| = {dc:.4f} {'<=' if ok else '>'} "
            f"|log({float(reference):.4f})| = {dr:.4f} ({reference_name}). Mean-growth "
            "mis-calibration must be no worse than the best physics baseline's, on the same "
            "held-out blocks under equal-block weighting. Reference-based, so no threshold is "
            "fitted to any arm's result (C-3)."
        ),
        extra={
            "reference_role": reference_name,
            "candidate_log_distance": dc,
            "reference_log_distance": dr,
            "margin_log": dr - dc,
        },
    )


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def first_moment_condition_from_blocks(
    candidate_by_block: Mapping[Any, Any],
    reference_by_block: Mapping[Any, Any],
    *,
    name: str = "first_moment",
    reference_name: str = FIRST_MOMENT_REFERENCE_ROLE,
) -> ConditionResult:
    """:func:`first_moment_condition` over per-block ratios, EQUAL-BLOCK weighted.

    Requires the two mappings to cover **exactly the same blocks**. A comparison
    against a reference scored on a different set of blocks is not a comparison -
    ADR-021 (4) adopted equal-block weighting precisely because one block (Creek)
    was 47% of the window-pooled held-out mass, so a differing block set can move
    the answer more than the models do. A mismatch is :data:`UNDEFINED`.

    TWO POOLINGS ARE REPORTED, and the choice between them is stated rather than
    made silently:

    * ``arithmetic`` (**the one adjudicated**) - mean of the RATIO over blocks,
      then ``|log(mean)|``. This is what ``equal_block_mean`` already does to every
      other criterion, so the first-moment condition pools identically to the
      dispersion condition beside it.
    * ``log`` (reported) - mean of ``|log(ratio)|`` over blocks. Strictly the more
      natural pooling for a ratio, and it is NOT used, because adopting it would
      change the pooling rule for one criterion only, on infra's own
      judgement, mid-gate. Emitted so the maintainer can see whether the choice
      is outcome-determinative before ruling on it.
    """
    blocks_c = {str(k) for k in candidate_by_block}
    blocks_r = {str(k) for k in reference_by_block}
    if not blocks_c or blocks_c != blocks_r:
        return ConditionResult(
            name=name,
            outcome=UNDEFINED,
            detail=(
                "candidate and reference are not scored on the same blocks "
                f"(candidate {sorted(blocks_c)} vs {reference_name} {sorted(blocks_r)}). "
                "Equal-block weighting over different block sets is not a comparison."
            ),
            extra={"reference_role": reference_name},
        )

    cand_vals, ref_vals, undefined = [], [], []
    for key in sorted(blocks_c):
        c = _finite_positive({str(k): v for k, v in candidate_by_block.items()}[key])
        r = _finite_positive({str(k): v for k, v in reference_by_block.items()}[key])
        if c is None or r is None:
            undefined.append(key)
            continue
        cand_vals.append(c)
        ref_vals.append(r)
    if undefined:
        return ConditionResult(
            name=name,
            outcome=UNDEFINED,
            detail=(
                f"{FIRST_MOMENT_KEY} is undefined on block(s) {undefined}. Dropping them would "
                "shrink the sample silently, which is the defect equal_block_mean was just "
                "hardened against."
            ),
            extra={"reference_role": reference_name, "undefined_blocks": undefined},
        )

    result = first_moment_condition(
        _mean(cand_vals), _mean(ref_vals), name=name, reference_name=reference_name
    )
    log_candidate = _mean([abs(math.log(v)) for v in cand_vals])
    log_reference = _mean([abs(math.log(v)) for v in ref_vals])
    return ConditionResult(
        name=result.name,
        outcome=result.outcome,
        value=result.value,
        reference=result.reference,
        detail=result.detail,
        extra={
            **result.extra,
            "n_blocks": len(cand_vals),
            "blocks": sorted(blocks_c),
            "pooling": "arithmetic mean of the ratio over blocks, then |log| (adjudicated)",
            "alt_pooling_log": {
                "candidate_mean_abs_log": log_candidate,
                "reference_mean_abs_log": log_reference,
                "would_pass": (
                    None
                    if log_candidate is None or log_reference is None
                    else log_candidate <= log_reference
                ),
                "note": "REPORTED, NOT ADJUDICATED. Emitted so the maintainer can see "
                "whether the pooling choice is outcome-determinative before ruling on it.",
            },
        },
    )


# --------------------------------------------------------------------------
# both conditions, always together
# --------------------------------------------------------------------------


def g3_conditions(
    adr: Any,
    candidate_growth_calibration: Any,
    reference_growth_calibration: Any,
    *,
    reference_name: str = FIRST_MOMENT_REFERENCE_ROLE,
) -> dict[str, Any]:
    """Both G3 conditions, reported SEPARATELY and returned TOGETHER.

    ``G3 passes only if BOTH hold.`` The combined outcome is:

    * :data:`PASS` - both conditions PASS.
    * :data:`UNDEFINED` - either condition is UNDEFINED. "Not adjudicable" is a
      distinct state from "failed", and reporting an unmeasurable gate as a
      failure is as much a misstatement as reporting it as a pass. G4 has already
      had to use this state once, at n=2 spot events.
    * :data:`FAIL` - otherwise.

    No verdict on G3 itself: that is the maintainer's, as it was for G2.
    """
    dispersion = dispersion_condition(adr)
    first_moment = first_moment_condition(
        candidate_growth_calibration,
        reference_growth_calibration,
        reference_name=reference_name,
    )
    return combine(dispersion, first_moment)


def combine(*conditions: ConditionResult) -> dict[str, Any]:
    """Combine conditions under "PASS only if ALL pass, UNDEFINED dominates FAIL"."""
    outcomes = [c.outcome for c in conditions]
    if UNDEFINED in outcomes:
        combined = UNDEFINED
    elif all(o == PASS for o in outcomes):
        combined = PASS
    else:
        combined = FAIL
    return {
        "outcome": combined,
        "conditions": {c.name: c.as_dict() for c in conditions},
        "rule": (
            "G3 requires BOTH the geometric dispersion bar and the first-moment condition. "
            "An UNDEFINED condition makes the gate NOT ADJUDICABLE — it is never a pass, and "
            "it is not a failure either."
        ),
        "not_a_verdict": (
            "The gate verdict is the maintainer's. This block reports the pre-registered "
            "conditions (ADR-039 (4), (5)) and where a candidate falls."
        ),
    }
