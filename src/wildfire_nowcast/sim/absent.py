"""One name for the distinction every detector in this package needs.

    an empty input is an ABSENT MEASUREMENT, not a benign result

`sim/blockanatomy.py` already learned this the expensive way: over zero rows it
published ``max_identity_residual: 0.0`` and ``known_answer_check_passed: true``
and exited 0, so a perfect agreement and a scan that examined nothing were the
same artifact, the same field, the same value and the same exit code.

The worse form is a POSITIVE CONTROL, and this package shipped one. A positive
control is the single instrument whose entire purpose is to prove that a
detector CAN fire; everything else in a suite is trusted because the controls
are trusted. `sim/playthrough.py`'s lobotomised arm declares
``expect_degenerate=True``, and :func:`~wildfire_nowcast.sim.playthrough.degeneracy_verdict`
returned ``degenerate=True`` over a ZERO-MEMBER ensemble: all three criteria are
satisfied vacuously, the arm agrees with its declared expectation, and the
playthrough passes. The control reads as fired when nothing ran, which does not
merely fail to detect. It certifies the detector as working while proving
nothing, and every result downstream inherits that certification.

The rule, therefore:

**"the control fired" and "there was nothing to fire on" must never produce the
same value.**

Two mechanisms are used here and they are not interchangeable.

REFUSAL, via :class:`AbsentMeasurementError`
    For anything that would otherwise publish a verdict. Compute nothing,
    publish nothing, exit non-zero. A number a module did not measure must not
    reach disk under a name implying it did. CLIs spend
    :data:`EXIT_NOTHING_EXAMINED` on it, which is 3 rather than 1 so that
    "nothing to examine" is distinguishable from "examined and disagreed" by a
    caller reading only the exit code, matching `tools/ci_status.py`, which
    already spends 3 on "there is no run to report on".

THE DENOMINATOR AS A FIELD
    Belt and braces for artifacts that ARE written: the count of things
    examined is always present, and the summary verdicts are present ONLY when
    something was examined, so a consumer reaching for a missing key gets a
    ``KeyError`` rather than a flattering default.

What this module is NOT for: a legitimately empty *result*. An ensemble that
ran six members and burned zero new cells has been measured, and D1 flagging it
is the criterion working. Zero MEMBERS is the absent case; zero CELLS is a
finding.
"""

from __future__ import annotations

__all__ = [
    "AbsentMeasurementError",
    "EXIT_NOTHING_EXAMINED",
    "refuse_if_empty",
]

#: Exit code for "there was nothing to examine". Deliberately NOT 1: a caller
#: that sees 1 knows a check disagreed, and a caller that sees 3 knows no check
#: was performed. Collapsing the two is how a vacuous pass becomes invisible.
EXIT_NOTHING_EXAMINED = 3


class AbsentMeasurementError(ValueError):
    """An instrument was asked to publish a verdict over an empty input.

    A ``ValueError`` rather than an ``AssertionError`` because it is a statement
    about the argument, and rather than a bare ``RuntimeError`` because callers
    that legitimately probe the empty case need to catch exactly this and not,
    say, a numpy broadcast failure that happens to travel the same path.
    """


def refuse_if_empty(what: str, counts: dict[str, int], *, because: str) -> None:
    """Raise :class:`AbsentMeasurementError` naming every empty denominator.

    ``counts`` maps the name of each thing that must be non-empty to how many of
    it there are. The message lists the zeros by name, because the failure a
    reader has to diagnose is "which denominator was zero", and a bare
    "empty input" sends them back to the source to find out.
    """
    empty = [name for name, n in counts.items() if int(n) == 0]
    if not empty:
        return
    have = ", ".join(f"{k}={v}" for k, v in counts.items())
    raise AbsentMeasurementError(
        f"NOTHING EXAMINED: {what} was asked for a verdict with {' and '.join(empty)} "
        f"empty ({have}). {because} This is NOT a pass and NOT a failure; it is an "
        "absent measurement, and no verdict is returned."
    )
