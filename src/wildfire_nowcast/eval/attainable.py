"""[M35] Print every bar-shaped criterion beside the range its statistic can attain.

WHY THIS MODULE EXISTS
----------------------
ADR-170 (2) established that G3's calibration conjunct compares
``calibration_error`` to ``0.10`` while the statistic **cannot exceed 0.025309
anywhere in this corpus** - the bar sits ``3.95x`` above the largest value its
own statistic can take, so no forecaster predicting anything near the right
burned area can fail it. Establishing that took ADR-151 (3), then M30, then a
dedicated M34 grid. **It should have been one line of output the first time
anybody ran the gate.**

This module is that line, generalised. A criterion that compares a statistic to
a threshold reports, beside its verdict, the range the statistic could have
attained on the data it was computed from, and a per-edge margin saying how far
the bar sits outside or inside that range. *"A check that runs, passes, and
could not have failed"* is the defect class this project keeps rediscovering by
hand; here it is mechanically detectable.

THE ONE RULE THAT MAKES THIS HONEST
-----------------------------------
**AN OBSERVED MIN AND MAX OVER THE ARMS WE HAPPEN TO HOLD IS NOT AN ACHIEVABLE
RANGE.** Presenting one as a bound would reproduce the exact defect this
instrument exists to catch - it would say "nothing could have failed" when the
truthful statement is "nothing we ran did". The type system is used to make that
impossible rather than merely discouraged:

* :class:`Bound` is the ONLY thing that can enter an :class:`AttainableRange`,
  and every ``closed_form`` bound must carry a non-empty ``derivation``. There is
  no constructor path from observations to a bound - :meth:`Bound.closed_form`
  takes a derivation string and :class:`AttainableRange.check` raises without one.
* Observations enter as :class:`Certificate` objects, which prove attainability
  of ONE POINT and can only ever *widen* what is known to be reachable. A
  certificate can upgrade an edge from ``UNDETERMINED`` to ``BINDING``; **no
  certificate can ever produce ``CANNOT_FIRE``**, which is the alarming verdict.
  That asymmetry is the whole safety property and it is asserted in
  :func:`place_bar`.
* Where no bound is known, :meth:`Bound.unknown` says so and the edge reports
  ``UNDETERMINED``. **"No bound is known, so this cannot be checked" is
  information a reader needs** and is printed rather than filled in.

WHAT A BOUND IS RELATIVE TO, AND WHY IT MUST BE PRINTED
-------------------------------------------------------
A bound is meaningless without saying what is held fixed. ``calibration_error <=
mean(p) + base_rate`` holds the OUTCOMES (hence the base rate) and the
FORECAST'S OWN MEAN PROBABILITY fixed, and lets the arrangement of probability
across cells vary; that is why ADR-170's statement is *"a forecaster would have
to over-forecast mean burn probability by >= 34.7x before this conjunct could
even in principle read 0.10"* rather than a flat impossibility. A different
counterfactual class gives a different range. :class:`AttainableRange` therefore
carries ``held_fixed`` and ``varying`` as required fields, and they are printed.

TIGHTNESS IS LOAD-BEARING, NOT DECORATION
-----------------------------------------
A SOUND bound on the passing side of an edge proves the edge ``CANNOT_FIRE``.
It does NOT prove the opposite: a loose bound on the failing side proves
nothing, because the true reach of the statistic may stop short of it. So
``BINDING`` is only ever concluded from a bound marked ``tight`` (attained or
approached), from a ``Certificate``, or from ``unbounded``. Getting this
backwards would let a loose bound manufacture a reassuring "this criterion
discriminates".

WHAT IS DELIBERATELY NOT HERE
-----------------------------
**No verdict is re-adjudicated and no replacement bar is proposed.** A threshold
re-derived after seeing every arm clear it is fitted to the result (ADR-020 (4);
the refusal is ADR-170 (8)). This module reports where a bar sits relative to
what its statistic can reach and stops there. If applying it changes a verdict
anywhere, that is the maintainer's ruling and goes through DECISIONS.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ABSENT",
    "ALWAYS_FIRES",
    "BINDING",
    "CANNOT_FIRE",
    "CLOSED_FORM",
    "CONSTRUCTION",
    "DISCRIMINATING",
    "OBSERVED",
    "UNBOUNDED",
    "UNDETERMINED",
    "UNKNOWN",
    "UNSATISFIABLE",
    "VACUOUS",
    "AttainableRange",
    "BarCriterion",
    "BarPlacement",
    "Bound",
    "Certificate",
    "EdgePlacement",
    "area_dispersion_ratio_range",
    "beats_reference_criterion",
    "brier_range",
    "calibration_error_range",
    "equal_block_mean_range",
    "growth_calibration_range",
    "iou_range",
    "place_bar",
    "reference_ratio_criterion",
    "separation_sd_range",
    "unanimity_bound_sd",
    "unanimity_range",
]

# ---------------------------------------------------------------------------
# how a bound was established
# ---------------------------------------------------------------------------

#: A derivation exists and is cited. ``value`` is a real number.
CLOSED_FORM = "closed_form"
#: PROVED to have no finite bound in this direction (sup = +inf / inf = -inf).
#: This is a POSITIVE result: it proves the edge on that side can fire.
UNBOUNDED = "unbounded"
#: Nothing is established. The edge is UNDETERMINED and says so.
UNKNOWN = "unknown"

#: How a :class:`Certificate` knows its value is reachable.
OBSERVED = "observed"
CONSTRUCTION = "construction"

# ---------------------------------------------------------------------------
# per-edge outcomes
# ---------------------------------------------------------------------------

#: PROVED: no attainable value violates this edge. **The defect.**
CANNOT_FIRE = "CANNOT_FIRE"
#: PROVED: every attainable value violates this edge.
ALWAYS_FIRES = "ALWAYS_FIRES"
#: PROVED: values on both sides of this edge are reachable, so it discriminates.
BINDING = "BINDING"
#: Nothing proved either way. Not a pass for the criterion and not an alarm.
UNDETERMINED = "UNDETERMINED"
#: The criterion has no edge on this side (a one-sided bar).
ABSENT = "ABSENT"

# ---------------------------------------------------------------------------
# criterion-level verdicts
# ---------------------------------------------------------------------------

#: Every edge CANNOT_FIRE: the criterion could not have failed. **The alarm.**
VACUOUS = "VACUOUS"
#: Some edge ALWAYS_FIRES: the criterion could not have passed.
UNSATISFIABLE = "UNSATISFIABLE"
#: At least one edge is BINDING and none ALWAYS_FIRES.
DISCRIMINATING = "DISCRIMINATING"

_EPS = 1e-15


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-4:
        return f"{value:.4g}"
    return f"{value:.6g}"


@dataclass(frozen=True)
class Bound:
    """One end of an attainable range, and HOW it was established.

    Never constructed from observations. See the module docstring: the only
    constructors are :meth:`closed_form`, :meth:`unbounded` and :meth:`unknown`,
    and the first two require a derivation the reader can check.
    """

    status: str
    value: float | None = None
    tight: bool | None = None
    derivation: str = ""

    @classmethod
    def closed_form(cls, value: float, derivation: str, *, tight: bool) -> Bound:
        """A derived, citable bound. ``tight`` says whether it is ATTAINED.

        ``tight=False`` is not a lesser bound - it is a sound one whose reach is
        not known to be realised, and this module refuses to conclude ``BINDING``
        from it. That refusal is the point: a loose bound on the failing side of
        an edge is compatible with the statistic never getting there.
        """
        if not derivation.strip():
            raise ValueError(
                "a closed-form bound without a derivation is an assertion, not a bound. "
                "Cite where it comes from, in the code, beside the number"
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"closed_form bound value {value!r} is not finite. Use Bound.unbounded() to "
                "state that no finite bound exists - the two are different claims and only "
                "one of them proves an edge can fire"
            )
        return cls(status=CLOSED_FORM, value=float(value), tight=bool(tight), derivation=derivation)

    @classmethod
    def unbounded(cls, derivation: str) -> Bound:
        """PROVED unbounded in this direction. Proves the edge on this side fires."""
        if not derivation.strip():
            raise ValueError("'unbounded' is a claim with a proof; state the proof")
        return cls(status=UNBOUNDED, value=None, tight=None, derivation=derivation)

    @classmethod
    def unknown(cls, reason: str) -> Bound:
        """No bound established. Prints as such; concludes nothing either way."""
        if not reason.strip():
            raise ValueError("say WHY no bound is known - 'unknown' with no reason is noise")
        return cls(status=UNKNOWN, value=None, tight=None, derivation=reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "tight": self.tight,
            "derivation": self.derivation,
        }


@dataclass(frozen=True)
class Certificate:
    """PROOF that ONE value is reachable. Never a range, never a bound.

    ``source=OBSERVED`` means the statistic actually took this value on this
    data; ``source=CONSTRUCTION`` means an explicit admissible configuration
    producing it is written down in ``detail``. Either way a certificate can only
    WIDEN what is known to be reachable, so it can prove an edge ``BINDING`` and
    can never prove one ``CANNOT_FIRE``.
    """

    value: float
    source: str
    detail: str

    def __post_init__(self) -> None:
        if self.source not in (OBSERVED, CONSTRUCTION):
            raise ValueError(f"certificate source {self.source!r} is neither observed nor derived")
        if not self.detail.strip():
            raise ValueError("a certificate must say what configuration realises its value")
        if not math.isfinite(float(self.value)):
            raise ValueError("a certificate value must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source, "detail": self.detail}


@dataclass(frozen=True)
class AttainableRange:
    """What a statistic could take, given a STATED counterfactual class."""

    statistic: str
    lower: Bound
    upper: Bound
    held_fixed: str
    varying: str
    inputs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.held_fixed.strip() or not self.varying.strip():
            raise ValueError(
                "an attainable range is relative to what is held fixed and what varies. "
                "Both must be stated: the same statistic has different ranges under "
                "different counterfactual classes, and ADR-170's ceiling holds the "
                "forecast's own mean probability fixed"
            )
        lo, hi = self.lower.value, self.upper.value
        if lo is not None and hi is not None and lo > hi + _EPS:
            raise ValueError(f"lower bound {lo} exceeds upper bound {hi}")

    @property
    def known(self) -> bool:
        """True when SOMETHING is established at either end."""
        return self.lower.status != UNKNOWN or self.upper.status != UNKNOWN

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "lower": self.lower.as_dict(),
            "upper": self.upper.as_dict(),
            "held_fixed": self.held_fixed,
            "varying": self.varying,
            "inputs": dict(self.inputs),
            "printable": self.printable(),
        }

    def printable(self) -> str:
        left = {
            CLOSED_FORM: f"[{_fmt(self.lower.value)}",
            UNBOUNDED: "(-inf",
            UNKNOWN: "(? unknown",
        }[self.lower.status]
        right = {
            CLOSED_FORM: f"{_fmt(self.upper.value)}]",
            UNBOUNDED: "+inf)",
            UNKNOWN: "unknown ?)",
        }[self.upper.status]
        return f"{left}, {right}"


@dataclass(frozen=True)
class BarCriterion:
    """A criterion of the shape ``low <= statistic <= high``.

    Either endpoint may be ``None`` for a one-sided bar. ``*_inclusive`` records
    whether the endpoint itself passes, because ``> r`` and ``>= r`` differ
    exactly at the boundary and that is where a vacuity argument lives.
    """

    key: str
    low: float | None
    high: float | None
    source: str
    low_inclusive: bool = True
    high_inclusive: bool = True

    def __post_init__(self) -> None:
        if self.low is None and self.high is None:
            raise ValueError(f"{self.key}: a criterion with no edge is not a criterion")
        if not self.source.strip():
            raise ValueError(f"{self.key}: state where this bar comes from")

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "low": self.low,
            "high": self.high,
            "low_inclusive": self.low_inclusive,
            "high_inclusive": self.high_inclusive,
            "source": self.source,
        }


@dataclass(frozen=True)
class EdgePlacement:
    """One edge of a bar, placed against the statistic's reach."""

    side: str
    bar: float | None
    status: str
    bound: Bound
    outside_by: float | None
    ratio: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "bar": self.bar,
            "status": self.status,
            "bound": self.bound.as_dict(),
            "outside_by": self.outside_by,
            "ratio": self.ratio,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BarPlacement:
    """The instrument's output for ONE criterion. Contains no gate verdict."""

    criterion: BarCriterion
    range: AttainableRange
    verdict: str
    low_edge: EdgePlacement
    high_edge: EdgePlacement
    dead_edges: tuple[str, ...]
    certificates: tuple[Certificate, ...]
    value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion.as_dict(),
            "attainable_range": self.range.as_dict(),
            "verdict": self.verdict,
            "edges": {"low": self.low_edge.as_dict(), "high": self.high_edge.as_dict()},
            "dead_edges": list(self.dead_edges),
            "certificates": [c.as_dict() for c in self.certificates],
            "observed_value": self.value,
            "not_a_verdict": (
                "This says where a BAR sits relative to what its STATISTIC can reach. It is "
                "not a pass, a fail, or a re-adjudication of any gate, and it proposes no "
                "replacement bar (ADR-170 (8))."
            ),
            "lines": self.lines(),
        }

    def lines(self) -> list[str]:
        head = (
            f"{self.criterion.key}  value={_fmt(self.value)}  "
            f"bar=[{_fmt(self.criterion.low)}, {_fmt(self.criterion.high)}]  "
            f"attainable={self.range.printable()}  -> {self.verdict}"
        )
        out = [head]
        for edge in (self.low_edge, self.high_edge):
            if edge.status == ABSENT:
                continue
            out.append(f"    {edge.side:<4} edge {_fmt(edge.bar):<12} {edge.status}: {edge.reason}")
        out.append(f"    held fixed: {self.range.held_fixed}")
        out.append(f"    varying:    {self.range.varying}")
        return out


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= _EPS:
        return None
    return numerator / denominator


def _certificate_violating(
    certificates: Sequence[Certificate], bar: float, *, side: str, inclusive: bool
) -> Certificate | None:
    """A certificate on the FAILING side of this edge. Proves the edge can fire."""
    for cert in certificates:
        if side == "low":
            if cert.value < bar or (not inclusive and cert.value <= bar):
                return cert
        elif cert.value > bar or (not inclusive and cert.value >= bar):
            return cert
    return None


def _place_low(
    bar: float | None,
    bound: Bound,
    certificates: Sequence[Certificate],
    *,
    inclusive: bool,
) -> EdgePlacement:
    """The ``statistic >= low`` edge. It FIRES when the statistic falls below."""
    if bar is None:
        return EdgePlacement("low", None, ABSENT, bound, None, None, "one-sided bar, no low edge")
    outside_by = None if bound.value is None else bound.value - bar
    ratio = _ratio(bound.value, bar)
    # A SOUND lower bound at or above the bar proves nothing can fall below it.
    if bound.status == CLOSED_FORM and bound.value is not None:
        if bound.value > bar + _EPS or (inclusive and abs(bound.value - bar) <= _EPS):
            return EdgePlacement(
                "low",
                bar,
                CANNOT_FIRE,
                bound,
                outside_by,
                ratio,
                f"the smallest attainable value is {_fmt(bound.value)}, at or above the bar "
                f"{_fmt(bar)}: no attainable value can violate this edge",
            )
    cert = _certificate_violating(certificates, bar, side="low", inclusive=inclusive)
    if bound.status == UNBOUNDED:
        return EdgePlacement(
            "low", bar, BINDING, bound, None, None, f"unbounded below: {bound.derivation}"
        )
    if bound.status == CLOSED_FORM and bound.tight and bound.value is not None:
        return EdgePlacement(
            "low",
            bar,
            BINDING,
            bound,
            outside_by,
            ratio,
            f"the minimum {_fmt(bound.value)} is ATTAINED and lies below the bar {_fmt(bar)}, "
            "so a value that fails this edge is reachable",
        )
    if cert is not None:
        return EdgePlacement(
            "low",
            bar,
            BINDING,
            bound,
            outside_by,
            ratio,
            f"reachable value {_fmt(cert.value)} fails this edge ({cert.source}: {cert.detail})",
        )
    return EdgePlacement(
        "low",
        bar,
        UNDETERMINED,
        bound,
        outside_by,
        ratio,
        "no sound bound places this edge and nothing reachable is known to fail it: "
        f"WHETHER THIS EDGE COULD FIRE IS UNKNOWN ({bound.derivation})",
    )


def _place_high(
    bar: float | None,
    bound: Bound,
    certificates: Sequence[Certificate],
    *,
    inclusive: bool,
) -> EdgePlacement:
    """The ``statistic <= high`` edge. It FIRES when the statistic rises above."""
    if bar is None:
        return EdgePlacement("high", None, ABSENT, bound, None, None, "one-sided bar, no high edge")
    outside_by = None if bound.value is None else bar - bound.value
    ratio = _ratio(bar, bound.value)
    if bound.status == CLOSED_FORM and bound.value is not None:
        if bound.value < bar - _EPS or (inclusive and abs(bound.value - bar) <= _EPS):
            return EdgePlacement(
                "high",
                bar,
                CANNOT_FIRE,
                bound,
                outside_by,
                ratio,
                f"the bar {_fmt(bar)} sits {_fmt(outside_by)} ABOVE the largest attainable "
                f"value {_fmt(bound.value)}"
                + (f" ({_fmt(ratio)}x)" if ratio is not None else "")
                + ": no attainable value can violate this edge",
            )
    cert = _certificate_violating(certificates, bar, side="high", inclusive=inclusive)
    if bound.status == UNBOUNDED:
        return EdgePlacement(
            "high", bar, BINDING, bound, None, None, f"unbounded above: {bound.derivation}"
        )
    if bound.status == CLOSED_FORM and bound.tight and bound.value is not None:
        return EdgePlacement(
            "high",
            bar,
            BINDING,
            bound,
            outside_by,
            ratio,
            f"the maximum {_fmt(bound.value)} is ATTAINED and lies above the bar {_fmt(bar)}, "
            "so a value that fails this edge is reachable",
        )
    if cert is not None:
        return EdgePlacement(
            "high",
            bar,
            BINDING,
            bound,
            outside_by,
            ratio,
            f"reachable value {_fmt(cert.value)} fails this edge ({cert.source}: {cert.detail})",
        )
    return EdgePlacement(
        "high",
        bar,
        UNDETERMINED,
        bound,
        outside_by,
        ratio,
        "no sound bound places this edge and nothing reachable is known to fail it: "
        f"WHETHER THIS EDGE COULD FIRE IS UNKNOWN ({bound.derivation})",
    )


def _always_fires(edge: EdgePlacement, other: Bound, *, side: str) -> bool:
    """Does EVERY attainable value violate this edge?

    Low edge: the whole range sits below the bar, i.e. the UPPER bound is below it.
    High edge: the whole range sits above the bar, i.e. the LOWER bound is above it.
    """
    if edge.bar is None or other.status != CLOSED_FORM or other.value is None:
        return False
    if side == "low":
        return other.value < edge.bar - _EPS
    return other.value > edge.bar + _EPS


def place_bar(
    criterion: BarCriterion,
    attainable: AttainableRange,
    *,
    certificates: Sequence[Certificate] = (),
    value: float | None = None,
) -> BarPlacement:
    """Place a bar against what its statistic can reach. **No gate verdict.**

    The safety property, asserted rather than assumed: a :class:`Certificate` can
    only ever move an edge TOWARDS ``BINDING``. If observations could produce
    ``CANNOT_FIRE`` this instrument would launder "nothing we ran failed" into
    "nothing could fail", which is the defect it exists to catch.
    """
    low = _place_low(
        criterion.low, attainable.lower, certificates, inclusive=criterion.low_inclusive
    )
    high = _place_high(
        criterion.high, attainable.upper, certificates, inclusive=criterion.high_inclusive
    )
    # THE SAFETY PROPERTY. Recomputing with no certificates must not turn a
    # CANNOT_FIRE into anything else, i.e. certificates never create one.
    if certificates:
        bare = place_bar(criterion, attainable, certificates=(), value=value)
        for got, want in ((low, bare.low_edge), (high, bare.high_edge)):
            if got.status == CANNOT_FIRE and want.status != CANNOT_FIRE:
                raise AssertionError(
                    "a certificate produced CANNOT_FIRE. Observations may only widen what is "
                    "known to be reachable; if they can narrow it, an observed min/max has "
                    "become a bound and this instrument has acquired the defect it detects"
                )
    if _always_fires(low, attainable.upper, side="low") or _always_fires(
        high, attainable.lower, side="high"
    ):
        verdict = UNSATISFIABLE
    elif all(e.status in (CANNOT_FIRE, ABSENT) for e in (low, high)):
        verdict = VACUOUS
    elif any(e.status == BINDING for e in (low, high)):
        verdict = DISCRIMINATING
    else:
        verdict = UNDETERMINED
    dead = tuple(e.side for e in (low, high) if e.status == CANNOT_FIRE)
    return BarPlacement(
        criterion=criterion,
        range=attainable,
        verdict=verdict,
        low_edge=low,
        high_edge=high,
        dead_edges=dead,
        certificates=tuple(certificates),
        value=value,
    )


# ---------------------------------------------------------------------------
# the ranges for the criteria this project actually adjudicates
#
# The registry (`common/null_check/registry.adjudicating_metrics`) holds 37
# metrics and exactly THREE may adjudicate: `area_dispersion_ratio`,
# `best_member_iou_shape_masked`, `growth_calibration`. Those three, plus the
# quarantined `calibration_error` that ADR-170 was found on, are below.
# ---------------------------------------------------------------------------


def calibration_error_range(*, mean_forecast: float, base_rate: float) -> AttainableRange:
    """``calibration_error`` in ``[0, mean(p) + base_rate]``. ADR-170 (2).

    DERIVATION. ``calibration_error = max(ECE_bins, ECE_frontier)`` and both
    terms have the form ``sum_s w_s |p_s - o_s|`` with ``w_s`` summing to 1,
    ``sum_s w_s p_s = mean(p)`` and ``sum_s w_s o_s = base_rate``. Then
    ``|p_s - o_s| <= p_s + o_s`` termwise, so the weighted sum is at most
    ``mean(p) + base_rate`` for EACH term and therefore for their max. Verified
    numerically at M34: 0 ceiling violations over 3,672 cells.

    BOTH ENDS ARE TIGHT. Zero is attained by a perfectly calibrated forecast.
    The ceiling is attained by a forecast that is positive exactly where nothing
    burns: with weight ``w`` on a stratum forecasting ``c`` that never burns and
    ``1-w`` on a stratum forecasting 0 that carries all the burning, the
    statistic is ``wc + base_rate = mean(p) + base_rate`` exactly.

    NOT A STATEMENT ABOUT ALL FORECASTERS. ``mean(p)`` is a property of the
    forecast, so this is the reach of the statistic AT THIS MEAN PREDICTED
    PROBABILITY - which is exactly why ADR-170 phrases the consequence as a
    required over-forecasting multiple rather than as a flat impossibility.
    """
    if not (0.0 <= base_rate <= 1.0) or not (0.0 <= mean_forecast <= 1.0):
        raise ValueError("mean forecast and base rate are probabilities")
    return AttainableRange(
        statistic="calibration_error",
        lower=Bound.closed_form(
            0.0,
            "a weighted mean of absolute deviations is non-negative; 0 is attained by a "
            "perfectly calibrated forecast (null panel: oracle 0.0000)",
            tight=True,
        ),
        upper=Bound.closed_form(
            mean_forecast + base_rate,
            "|p_s - o_s| <= p_s + o_s termwise; weights sum to 1, so the weighted mean is at "
            "most mean(p) + base_rate. Holds for ECE_bins and ECE_frontier separately, hence "
            "for their max. ADR-170 (2); verified 0 violations over 3,672 cells at M34",
            tight=True,
        ),
        held_fixed="the outcomes (hence the base rate) and the forecast's own mean probability",
        varying="how forecast probability and outcomes are arranged across bins/rings",
        inputs={"mean_forecast": mean_forecast, "base_rate": base_rate},
    )


def growth_calibration_range(
    *, base_rate: float, n_scored_cells: float | None = None
) -> AttainableRange:
    """``growth_calibration`` in ``[0, 1 / base_rate]``.

    DERIVATION. ``growth_calibration = sum(ensemble-mean event area) /
    sum(truth event area)`` over the scored mask (``common.dispersion``). The
    ensemble-mean event area of a cell IS its forecast probability, so the
    numerator is ``sum_cells p <= N`` and the denominator is the truth event
    count ``T``; ``base_rate = T / N`` over the same cells, hence the ratio is at
    most ``N / T = 1 / base_rate``.

    BOTH ENDS ARE TIGHT AND BOTH ARE REALISABLE BY A DEGENERATE FORECASTER:
    0 by one that ignites nothing (``persistence`` reads exactly this), and
    ``1/base_rate`` by one that burns every masked cell with probability 1.

    The identity ``growth_calibration = mean(p) / base_rate`` follows from the
    same two sums and is what lets a run compute BOTH this range and
    :func:`calibration_error_range` from keys the artifact already carries
    (``band_base_rate``, ``band_growth_calibration``). Reproduced against the
    shipped per-block values to 1e-12 at M35.
    """
    if not 0.0 < base_rate <= 1.0:
        raise ValueError(
            "growth_calibration is undefined when truth grew nothing (base rate 0): the "
            "correct answer for a dormant stratum is that a ratio cannot be measured"
        )
    inputs = {"base_rate": base_rate}
    if n_scored_cells is not None:
        inputs["n_scored_cells"] = float(n_scored_cells)
        inputs["truth_event_cells"] = float(n_scored_cells) * base_rate
    return AttainableRange(
        statistic="growth_calibration",
        lower=Bound.closed_form(
            0.0,
            "the numerator is a sum of probabilities, so it is non-negative; 0 is attained by "
            "a forecaster that ignites nothing",
            tight=True,
        ),
        upper=Bound.closed_form(
            1.0 / base_rate,
            "sum_cells p <= N and the denominator is T = base_rate * N, so the ratio is at "
            "most 1/base_rate; attained by a forecaster that burns every masked cell",
            tight=True,
        ),
        held_fixed="the truth event field on the scored mask (hence T and the base rate)",
        varying="the ensemble-mean event probability of each masked cell, freely in [0,1]",
        inputs=inputs,
    )


def area_dispersion_ratio_range(
    *, n_members: int, n_scored_cells: float, n_units: float | None = None
) -> AttainableRange:
    """``area_dispersion_ratio``: ``[0, U]``, with ``U`` sound but VERY loose.

    ``adr = sqrt( ((M+1)/M) * sum_u Var(member areas) / sum_u (mean area - truth
    area)^2 )`` over scored units ``u`` (window x lead).

    LOWER BOUND 0, TIGHT: any ensemble whose members agree on area has numerator
    exactly 0 with a non-zero denominator. ``persistence`` reads exactly 0.0 on
    this corpus, which is a *degenerate ensemble with no spread at all* and not a
    measurement of dispersion (ADR-172 (3)).

    UPPER BOUND, SOUND AND LOOSE. Areas are integer cell counts in ``[0, N_u]``,
    so ``Var <= N_u^2/4 * M/(M-1)``; and the ensemble mean is a multiple of
    ``1/M`` while the truth area is an integer, so a NON-ZERO denominator is at
    least ``1/M^2`` (a zero denominator makes the statistic UNDEFINED, never
    passing). With ``sum_u N_u^2 <= (sum_u N_u)^2 = N^2`` this gives
    ``U = (N*M/2) * sqrt((M+1)/(M-1))``. **It is astronomically above the bar,
    which is the honest answer: the upper edge is nowhere near dead.** It is
    marked NOT TIGHT, so this module refuses to conclude ``BINDING`` from it -
    that conclusion needs a certificate, and a two-point ensemble supplies one at
    a realistic value.
    """
    if n_members < 2:
        raise ValueError("a one-member ensemble has no area spread and no ddof=1 variance")
    if n_scored_cells <= 0:
        raise ValueError("no scored cells")
    m = float(n_members)
    ceiling = (float(n_scored_cells) * m / 2.0) * math.sqrt((m + 1.0) / (m - 1.0))
    inputs = {"n_members": m, "n_scored_cells": float(n_scored_cells)}
    if n_units is not None:
        inputs["n_scored_units"] = float(n_units)
    return AttainableRange(
        statistic="area_dispersion_ratio",
        lower=Bound.closed_form(
            0.0,
            "sqrt of a ratio of non-negative sums; 0 is ATTAINED by any ensemble whose "
            "members agree on area while the mean is wrong (persistence reads exactly 0.0)",
            tight=True,
        ),
        upper=Bound.closed_form(
            ceiling,
            "Var(areas) <= N_u^2/4 * M/(M-1) for areas in [0, N_u]; a non-zero denominator is "
            ">= 1/M^2 by integrality of cell counts (a zero denominator is UNDEFINED, never a "
            "pass); sum N_u^2 <= (sum N_u)^2. LOOSE by construction - see the docstring",
            tight=False,
        ),
        held_fixed="the truth areas on the scored mask, the mask sizes, and the member count M",
        varying="each member's burned-area count, freely in [0, N_u]",
        inputs=inputs,
    )


def brier_range(statistic: str = "brier") -> AttainableRange:
    """``mean (p - y)^2`` with ``p`` in ``[0,1]`` and ``y`` in ``{0,1}``: ``[0, 1]``."""
    return AttainableRange(
        statistic=statistic,
        lower=Bound.closed_form(
            0.0, "a mean of squares is non-negative; attained by a perfect forecast", tight=True
        ),
        upper=Bound.closed_form(
            1.0,
            "(p - y)^2 <= 1 for p in [0,1] and y in {0,1}; attained by forecasting 1 on every "
            "cell that does not burn and 0 on every cell that does",
            tight=True,
        ),
        held_fixed="the outcome field on the scored mask",
        varying="the forecast probability of each scored cell, freely in [0,1]",
        inputs={},
    )


def equal_block_mean_range(ranges: Sequence[AttainableRange]) -> AttainableRange:
    """The reach of an EQUAL-BLOCK MEAN, from the per-block reaches.

    A criterion adjudicated on the equal-block mean must be placed against the
    reach of THAT quantity, not of one block's. The per-block extremes are
    SIMULTANEOUSLY attainable - blocks are scored independently, so a
    configuration realising each block's extreme in its own block is admissible -
    so the mean of tight per-block bounds is itself tight. A mean of merely SOUND
    bounds is sound and stays marked loose.

    Refuses to average ranges of different statistics or different counterfactual
    classes: that would produce a bound for a quantity nobody computed.
    """
    if not ranges:
        raise ValueError("no per-block ranges to average")
    first = ranges[0]
    for other in ranges[1:]:
        if (other.statistic, other.held_fixed, other.varying) != (
            first.statistic,
            first.held_fixed,
            first.varying,
        ):
            raise ValueError(
                "refusing to average ranges from different statistics or different "
                "counterfactual classes - the result would bound nothing that was computed"
            )

    def _side(name: str) -> Bound:
        bounds = [getattr(r, name) for r in ranges]
        if any(b.status == UNKNOWN for b in bounds):
            return Bound.unknown(
                f"at least one block has no known {name} bound, so their mean has none either"
            )
        if any(b.status == UNBOUNDED for b in bounds):
            return Bound.unbounded(f"at least one block is unbounded {name}, so the mean is")
        values = [b.value for b in bounds if b.value is not None]
        return Bound.closed_form(
            sum(values) / len(ranges),
            f"equal-block mean of {len(ranges)} per-block {name} bounds. Per-block extremes are "
            "simultaneously attainable (blocks are scored independently), so this inherits "
            "their tightness. Component derivation: " + bounds[0].derivation,
            tight=all(bool(b.tight) for b in bounds),
        )

    merged: dict[str, float] = {}
    for r in ranges:
        for k, v in r.inputs.items():
            merged[k] = merged.get(k, 0.0) + float(v) / len(ranges)
    return AttainableRange(
        statistic=first.statistic,
        lower=_side("lower"),
        upper=_side("upper"),
        held_fixed=first.held_fixed + f", over {len(ranges)} held-out blocks",
        varying=first.varying,
        inputs={f"mean_{k}": v for k, v in merged.items()},
    )


def iou_range(statistic: str = "best_member_iou_shape_masked") -> AttainableRange:
    """A Jaccard index: ``[0, 1]``, both ends attained.

    C6.4's masked variant DROPS empty-truth leads, so the empty-vs-empty
    convention that inflates ``best_member_iou`` to 1.0 is not in play and the
    null floor is exactly 0 (the registry says so in its own note). 0 is attained
    by a member disjoint from truth; 1 by a member equal to it.
    """
    return AttainableRange(
        statistic=statistic,
        lower=Bound.closed_form(
            0.0,
            "|A n B| >= 0; attained by a member disjoint from truth. Empty-truth leads are "
            "dropped by C6.4, so the empty-vs-empty 1.0 convention cannot arise here",
            tight=True,
        ),
        upper=Bound.closed_form(
            1.0,
            "|A n B| <= |A u B|; attained by a member equal to truth (selftest: "
            "check_perfect_forecast reads best_member_iou 1.0)",
            tight=True,
        ),
        held_fixed="the truth increment on each scored lead",
        varying="which cells each ensemble member burns",
        inputs={},
    )


def separation_sd_range(n_blocks: int) -> AttainableRange:
    """``separation_sd = mean(margin)/sd(margin)`` across blocks: unbounded BOTH ways.

    Not a bounded statistic and not an error: as the block-to-block SD shrinks at
    a fixed mean the ratio diverges, in either sign. ``common.separation``
    already refuses the exactly-zero-SD case rather than reporting an infinity
    (*"a negligible margin acquiring infinite significance"*), so the reach is an
    open range, not an attained one. **Unboundedness PROVES both edges of any
    finite bar can fire** - which is a positive result about the separation bar
    and says nothing about its height.
    """
    if n_blocks < 2:
        raise ValueError("a separation needs at least 2 blocks (common.separation._MIN_BLOCKS)")
    proof = (
        f"at n={n_blocks} blocks, scaling every margin's deviation from a fixed non-zero mean "
        "towards 0 drives |mean|/sd to +inf continuously; the sign of the mean is free, so "
        "both directions are unbounded"
    )
    return AttainableRange(
        statistic="separation_sd",
        lower=Bound.unbounded(proof),
        upper=Bound.unbounded(proof),
        held_fixed="the number of held-out blocks",
        varying="the per-block margins (candidate minus reference), freely in R^n",
        inputs={"n_blocks": float(n_blocks)},
    )


def unanimity_bound_sd(n_blocks: int) -> float:
    """``(n-1)/sqrt(n)`` - the separation above which unanimity is IMPLIED.

    For ``n`` numbers with sample mean ``m`` and sample SD ``s`` (ddof=1),
    ``sum_i (x_i - m)^2 = (n-1)s^2`` and, for any single index,
    ``sum_{j != i}(x_j - m)^2 >= (x_i - m)^2/(n-1)`` by Cauchy-Schwarz (the
    deviations sum to zero). Hence ``(x_i - m)^2 * n/(n-1) <= (n-1)s^2``, i.e.

        max_i |x_i - m| <= s * (n-1)/sqrt(n).

    So if ``m/s >= B`` with ``B >= (n-1)/sqrt(n)`` then ``min_i x_i >= m -
    s(n-1)/sqrt(n) >= 0``: every block already favours the candidate. This is
    ADR-133 (ii)'s "the unanimity conjunct is ALGEBRAICALLY VACUOUS at n <= 8",
    re-derived here from the inequality rather than quoted.
    """
    if n_blocks < 2:
        raise ValueError("needs at least 2 blocks")
    return (n_blocks - 1) / math.sqrt(n_blocks)


def unanimity_range(*, n_blocks: int, min_separation_sd: float) -> AttainableRange:
    """Blocks favouring the candidate, GIVEN the separation conjunct already holds.

    The counterfactual class is the point: ``conditions()`` requires
    ``separation_sd >= min_sd`` AND unanimity, so the question a reader needs
    answered is whether the SECOND conjunct can ever exclude anything the first
    admits. Under that conditioning the reach of ``blocks_favouring`` is
    ``[n, n]`` whenever ``min_sd >= (n-1)/sqrt(n)``, and the unanimity edge is
    dead. Below that threshold it is ``[0, n]``: a large positive mean can still
    coexist with one dissenting block.
    """
    if n_blocks < 2:
        raise ValueError("needs at least 2 blocks")
    threshold = unanimity_bound_sd(n_blocks)
    # STRICTLY greater. At `min_sd == threshold` exactly, the extremal
    # configuration puts one margin at EXACTLY 0, and `blocks_favouring` counts
    # `margin > 0` strictly - so the conjunct can still fire on that boundary
    # set. `>=` here would be a one-point vacuity claim that is false.
    implied = min_separation_sd > threshold + _EPS
    n = float(n_blocks)
    if implied:
        lower = Bound.closed_form(
            n,
            f"separation_sd >= {min_separation_sd:g} > (n-1)/sqrt(n) = {threshold:.6g} forces "
            f"min_i margin_i >= sd*({min_separation_sd:g} - {threshold:.6g}) > 0, so every "
            "block favours the candidate. ADR-133 (ii), re-derived in unanimity_bound_sd",
            tight=True,
        )
    else:
        lower = Bound.closed_form(
            0.0,
            f"separation_sd >= {min_separation_sd:g} <= (n-1)/sqrt(n) = {threshold:.6g} does not "
            "force any single margin's sign; a configuration with a dissenting block and a "
            "separation above the bar exists",
            tight=True,
        )
    return AttainableRange(
        statistic="blocks_favouring",
        lower=lower,
        upper=Bound.closed_form(
            n,
            "there are only n blocks and each either favours the candidate or does not",
            tight=True,
        ),
        held_fixed=(
            f"n = {n_blocks} blocks AND the other conjunct of the same criterion "
            f"(separation_sd >= {min_separation_sd:g})"
        ),
        varying="the per-block margins, subject to that separation constraint",
        inputs={
            "n_blocks": n,
            "min_separation_sd": float(min_separation_sd),
            "unanimity_implied_above": threshold,
        },
    )


def reference_ratio_criterion(key: str, reference_value: float, *, source: str) -> BarCriterion:
    """``|log(x)| <= |log(ref)|`` restated as the interval ``[1/r, r]``.

    G3's first-moment condition (ADR-039 (5)) is reference-relative and carries
    no fitted constant, but it is still a bar: with ``r = exp(|log ref|) >= 1``
    the passing set is exactly ``[1/r, r]``. Restating it as an interval is what
    lets the same instrument place it against ``growth_calibration``'s reach.
    """
    if reference_value <= 0 or not math.isfinite(reference_value):
        raise ValueError("a reference-relative bar needs a positive finite reference value")
    r = math.exp(abs(math.log(reference_value)))
    return BarCriterion(key=key, low=1.0 / r, high=r, source=source)


def beats_reference_criterion(
    key: str, reference_value: float, *, higher_is_better: bool, source: str
) -> BarCriterion:
    """ "Beat the opponent's score" as a one-sided, STRICT bar.

    G2 is *"beat the wind-advected ellipse"*, which is a threshold like any
    other: the opponent's own value. Worth placing, because a reference sitting
    at the end of its statistic's range makes the criterion UNSATISFIABLE (an IoU
    opponent at 1.0) or trivial (one at 0.0), and neither is visible from the
    verdict alone.
    """
    if higher_is_better:
        return BarCriterion(
            key=key, low=reference_value, high=None, source=source, low_inclusive=False
        )
    return BarCriterion(
        key=key, low=None, high=reference_value, source=source, high_inclusive=False
    )
