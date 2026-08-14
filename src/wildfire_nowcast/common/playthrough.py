"""ADR-030 made MECHANICAL: a playthrough that cannot fail must not ship.

ADR-030 defines a *playthrough test* as an end-to-end scenario whose correct
answer is known BY CONSTRUCTION, plus a scoring function returning a pass/fail
verdict, plus **a planted defect the harness actually detects** — and makes it
standing policy that *no gate is adjudicated on a metric whose implementation
lacks one*. That policy has paid three times in a single day. All three payoffs
are about the same thing, and none of them was caught by a rule:

1. **simviz found its own first playthrough COULD NOT HAVE FAILED** (ADR-031 (5)).
   Nearest-neighbour and area-fraction coarsening agree within 1% on smooth
   shapes, so the planted defect was invisible until the input was given
   sub-cell texture. It found this only because it went looking; nothing in the
   repo would have told it.
2. **modelling planted mutations in `eval/metrics.py` itself** (ADR-032 (4)).
   Dropping the square root is invisible at ``c = 1`` — only scoring a RANGE
   catches it. Dropping ``(M+1)/M``, or using ``ddof=0``, moves the reading 3% —
   only an exact 1e-12 case catches it. **A single-regime playthrough would have
   been GREEN for 2 of 4 mutations.**
3. A playthrough **corrected G3's own units before a model was touched**:
   ``_ratio`` takes a square root, so the criterion is in SD units, and an
   insight claiming "~2.1x too narrow" was wrong by a square root (true: ~4.7x in
   SD, 4.4% of the required variance).

Read together those say something narrow: **the dangerous playthrough is not the
one that fails, it is the one whose defect nothing detects, and you cannot see
that by reading it.** The only way to know a planted defect is detected is to
plant it and watch a probe go red. This module makes that a build step instead of
a discipline.

THE PROTOCOL
------------
A :class:`Playthrough` declares four things and the harness does the rest:

``build``    a fresh CLEAN world (rebuilt per defect, so nothing leaks sideways)
``observe``  world -> observation. The expensive part, run ONCE per world.
``probes``   named pure judgements on an observation. The known answers.
``defects``  named mutations of the world *or of the instrument*, each stating
             whether it is expected to be DETECTED.

:func:`run` then enforces, in order:

* **every probe passes on the clean world.** A defect planted into an
  already-failing scenario proves nothing — "the scenario must pass BEFORE the
  defect is planted" is the sentence both existing playthroughs wrote by hand.
* **every defect declared ``detected=True`` is caught by at least one probe.**
  This is the ADR-030 requirement and the reason this module exists. An
  undetected planted defect is a HARD FAILURE of the playthrough, not a note.
* **every defect declared ``detected=False`` is NOT caught.** A documented blind
  spot that closes is also a failure — *update the record, do not delete the
  test* (the sentence `test_playthrough_dispersion` already carries about member
  duplication). A blind spot nobody re-measures is folklore.

and REPORTS, at the C-1 reporting tier rather than as a failure:

* **which probe is the SOLE catcher of a defect** — mechanising finding (2).
  ``sole_catchers`` is the map that says *only the RANGE catches a missing square
  root*. Delete that probe and the coverage requirement goes red, which is the
  guarantee the comment in the file could not give.
* **probes that catch nothing** and are not declared ``guard=True``. A guard pins
  the SCENARIO ("this scenario really does contain dormant hours"); anything else
  that catches nothing is an assertion earning its keep only by hope.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not decide any gate, score any model, or import anything from ``model/``
or ``eval/``. It takes callables and returns a report. The verdict a playthrough
reaches is the playthrough's; the verdict a GATE reaches is the maintainer's.

It also does not attempt to invent defects for you. A generated mutation would be
a different tool (and a good one), but the three payoffs above all came from a
human choosing a mutation that a REAL bug would look like — ``nearest``
coarsening, an ensemble loop reusing one draw, ``ddof=0``. The harness's job is
to make sure the chosen mutation is actually caught, not to choose it.

C0: this is the ONE implementation of the protocol. ``tests/`` and other leads'
packages import it; nobody re-implements "did the planted defect fail".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Probe",
    "Defect",
    "Playthrough",
    "DefectOutcome",
    "PlaythroughReport",
    "PlaythroughError",
    "data_defect",
    "attribute_defect",
    "no_defect",
    "run",
    "format_report",
    "coverage_from_caught_map",
    "approximately",
]


class PlaythroughError(AssertionError):
    """A playthrough failed its own protocol. An ``AssertionError`` on purpose:
    ``pytest`` renders it as a plain test failure, which is where it belongs."""


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One known-answer judgement on an observation.

    ``check`` returns ``True`` when the observation matches the answer known by
    construction. It must be a PURE function of the observation: the harness runs
    it once per world and attributes catches to it, so a probe with side effects
    would make the coverage map a fiction.

    ``guard=True`` marks a probe that pins the SCENARIO rather than the
    instrument — "this scenario really does contain four dormant windows". Guards
    are expected to catch nothing and are exempt from the dead-probe report.
    """

    name: str
    check: Callable[[Any], bool]
    note: str = ""
    guard: bool = False


@dataclass(frozen=True)
class Defect:
    """A planted defect, and whether the harness is expected to catch it.

    ``plant`` is a context manager factory: ``plant(world)`` yields the world the
    observation should be taken from. That one shape covers both kinds of defect
    this project has actually needed —

    * a DATA defect (collapse the ensemble, shift every area) returns a mutated
      world, via :func:`data_defect`;
    * an INSTRUMENT defect (drop the square root in ``_ratio``, use ``ddof=0``)
      patches a module attribute for the duration and yields the world unchanged,
      via :func:`attribute_defect`.

    ``detected=False`` declares a DOCUMENTED BLIND SPOT: a defect this instrument
    provably cannot see. It is not an excuse — it is asserted in the opposite
    direction, so the day the blind spot closes, the build says so.
    """

    name: str
    plant: Callable[[Any], Any]
    note: str
    detected: bool = True


def data_defect(mutate: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a pure ``world -> world`` mutation as a :class:`Defect` planter."""

    @contextmanager
    def plant(world: Any) -> Iterator[Any]:
        yield mutate(world)

    return plant


def attribute_defect(*targets: tuple[Any, str, Any]) -> Callable[[Any], Any]:
    """Temporarily ``setattr(obj, name, value)`` — an INSTRUMENT mutation.

    Deliberately not ``pytest.monkeypatch``: ``common/`` must import cleanly
    outside a test session, and a playthrough should be runnable from a CLI. The
    original values are restored in a ``finally``, and
    ``test_playthrough_harness`` asserts that restoration actually happens —
    a mutation that leaked into the next test would corrupt every result after it
    and look like a flake.
    """

    @contextmanager
    def plant(world: Any) -> Iterator[Any]:
        saved = [(obj, name, getattr(obj, name)) for obj, name, _ in targets]
        try:
            for obj, name, value in targets:
                setattr(obj, name, value)
            yield world
        finally:
            for obj, name, value in saved:
                setattr(obj, name, value)

    return plant


def no_defect() -> Callable[[Any], Any]:
    """A defect that changes nothing. Only ever legitimate with ``detected=False``
    — it is the control that proves the harness does not hallucinate catches."""
    return data_defect(lambda world: world)


@dataclass(frozen=True)
class Playthrough:
    """A scenario with a known answer, its probes, and its planted defects."""

    name: str
    build: Callable[[], Any]
    observe: Callable[[Any], Any]
    probes: tuple[Probe, ...]
    defects: tuple[Defect, ...]
    note: str = ""

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not self.probes:
            problems.append("no probes: a playthrough with no known answer tests nothing")
        if not self.defects:
            problems.append(
                "no defects declared. ADR-030 requires a planted defect the harness actually "
                "detects; a playthrough without one is exactly the green-but-vacuous check the "
                "policy exists to prevent"
            )
        if not any(d.detected for d in self.defects):
            problems.append(
                "every declared defect is a blind spot, so nothing here can ever go red. At "
                "least one defect must be expected to be DETECTED"
            )
        if not any(not p.guard for p in self.probes):
            problems.append("every probe is a scenario guard; nothing judges the instrument")
        for label, names in (
            ("probe", [p.name for p in self.probes]),
            ("defect", [d.name for d in self.defects]),
        ):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                problems.append(f"duplicate {label} names {duplicates}: the coverage map is keyed "
                                "by name and would silently merge them")
        for defect in self.defects:
            if not defect.note.strip():
                problems.append(
                    f"defect {defect.name!r} states no reason. A planted defect has to say what "
                    "REAL bug it stands in for, or the next reader cannot tell a mutation that "
                    "matters from one that was easy to write"
                )
        if problems:
            joined = "\n  - ".join(problems)
            raise PlaythroughError(f"playthrough {self.name!r} is malformed:\n  - {joined}")


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DefectOutcome:
    """What happened when one declared defect was planted."""

    name: str
    expected_detected: bool
    detected: bool
    caught_by: tuple[str, ...]
    errors: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_detected": self.expected_detected,
            "detected": self.detected,
            "caught_by": list(self.caught_by),
            "errors": dict(self.errors),
            "note": self.note,
        }


@dataclass(frozen=True)
class PlaythroughReport:
    """The coverage verdict. ``failures`` is the hard tier, ``reporting`` the soft
    tier — the same two-tier severity C-1 gives the contract checker, for the same
    reason: a declared weakness is a gate, an omitted one is a failure."""

    name: str
    clean: dict[str, bool]
    clean_errors: dict[str, str]
    outcomes: tuple[DefectOutcome, ...]
    failures: tuple[str, ...]
    reporting: tuple[str, ...]
    sole_catchers: dict[str, str]
    dead_probes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def mutation_coverage(self) -> float:
        """Detected / expected-to-be-detected. ``1.0`` is the only passing value."""
        wanted = [o for o in self.outcomes if o.expected_detected]
        if not wanted:
            return 0.0
        return sum(1 for o in wanted if o.detected) / len(wanted)

    def as_dict(self) -> dict[str, Any]:
        return {
            "playthrough": self.name,
            "clean": dict(self.clean),
            "clean_errors": dict(self.clean_errors),
            "defects": {o.name: o.as_dict() for o in self.outcomes},
            "mutation_coverage": self.mutation_coverage,
            "sole_catchers": dict(self.sole_catchers),
            "dead_probes": list(self.dead_probes),
            "failures": list(self.failures),
            "reporting": list(self.reporting),
            "passed": self.passed,
        }

    def assert_ok(self, *, strict: bool = False) -> PlaythroughReport:
        """Raise :class:`PlaythroughError` on any hard failure. ``strict`` also
        promotes the reporting tier, exactly as ``--for-reporting`` does for the
        contract checker."""
        problems = list(self.failures) + (list(self.reporting) if strict else [])
        if problems:
            raise PlaythroughError(format_report(self))
        return self


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


def _judge(probes: Sequence[Probe], observation: Any) -> tuple[dict[str, bool], dict[str, str]]:
    """Run every probe, converting a raised exception into a recorded failure.

    An exception is treated as a FAILED probe rather than swallowed or re-raised.
    Under a mutated world that is usually a legitimate detection — a metric
    returning ``None`` makes ``abs(got - want)`` a ``TypeError`` — but it is
    ambiguous enough that the message is kept and printed, so nobody reads a
    ``TypeError`` in the harness as evidence about the instrument.
    """
    passed: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for probe in probes:
        try:
            passed[probe.name] = bool(probe.check(observation))
        except Exception as exc:  # noqa: BLE001 - the message IS the result
            passed[probe.name] = False
            errors[probe.name] = f"{type(exc).__name__}: {exc}"
    return passed, errors


def run(playthrough: Playthrough) -> PlaythroughReport:
    """Execute the protocol. Cost is ``(1 + n_defects)`` observations, no more.

    The world is REBUILT for every defect rather than deep-copied: a mutation
    that leaks into the next arm produces a coverage map that is confidently
    wrong, and "rebuild" is the only version of that guarantee which does not
    depend on the author's world being copyable.
    """
    clean_obs = playthrough.observe(playthrough.build())
    clean, clean_errors = _judge(playthrough.probes, clean_obs)

    failures: list[str] = []
    for name, ok in sorted(clean.items()):
        if ok:
            continue
        why = clean_errors.get(name, "returned False")
        failures.append(
            f"probe {name!r} FAILS on the CLEAN world ({why}). The scenario must pass BEFORE "
            "any defect is planted — otherwise every catch below is attributable to the "
            "scenario rather than to the defect, and the coverage map is meaningless"
        )

    outcomes: list[DefectOutcome] = []
    for defect in playthrough.defects:
        world = playthrough.build()
        with defect.plant(world) as mutated:
            observation = playthrough.observe(mutated)
            mutated_pass, mutated_errors = _judge(playthrough.probes, observation)
        caught = tuple(
            probe.name
            for probe in playthrough.probes
            if clean.get(probe.name) and not mutated_pass.get(probe.name)
        )
        outcomes.append(
            DefectOutcome(
                name=defect.name,
                expected_detected=defect.detected,
                detected=bool(caught),
                caught_by=caught,
                errors={k: v for k, v in mutated_errors.items() if k in caught},
                note=defect.note,
            )
        )

    for outcome in outcomes:
        if outcome.expected_detected and not outcome.detected:
            failures.append(
                f"PLANTED DEFECT {outcome.name!r} WAS NOT DETECTED by any probe. ADR-030: a "
                "playthrough that cannot fail is the exact thing the policy exists to prevent. "
                f"What it stands in for: {outcome.note.strip()}"
            )
        if not outcome.expected_detected and outcome.detected:
            failures.append(
                f"declared BLIND SPOT {outcome.name!r} was caught by {list(outcome.caught_by)}. "
                "The instrument has become sensitive to a defect the record says it cannot see: "
                "UPDATE THE RECORD, do not delete the test. A blind spot nobody re-measures is "
                "folklore, and this is the assertion that re-measures it"
            )

    sole_catchers = {
        o.name: o.caught_by[0] for o in outcomes if o.expected_detected and len(o.caught_by) == 1
    }
    catching = {name for o in outcomes for name in o.caught_by}
    dead = tuple(p.name for p in playthrough.probes if not p.guard and p.name not in catching)

    reporting: list[str] = []
    if dead:
        reporting.append(
            f"probes catching NO declared defect and not marked guard=True: {list(dead)}. Either "
            "they pin the scenario (declare guard=True) or no planted defect exercises them, "
            "which is where a vacuous assertion hides"
        )
    for defect_name, probe_name in sorted(sole_catchers.items()):
        reporting.append(
            f"probe {probe_name!r} is the SOLE catcher of {defect_name!r} — load-bearing. "
            "Removing or weakening it silently removes this defect from coverage"
        )

    return PlaythroughReport(
        name=playthrough.name,
        clean=clean,
        clean_errors=clean_errors,
        outcomes=tuple(outcomes),
        failures=tuple(failures),
        reporting=tuple(reporting),
        sole_catchers=sole_catchers,
        dead_probes=dead,
    )


def format_report(report: PlaythroughReport) -> str:
    """A human-readable coverage table. Printed by ``make playthrough``."""
    lines = [
        f"playthrough [{report.name}] "
        f"mutation coverage {report.mutation_coverage:.0%} "
        f"({sum(1 for o in report.outcomes if o.detected)}/"
        f"{sum(1 for o in report.outcomes if o.expected_detected)} declared defects detected)",
        f"  probes : {len(report.clean)} "
        f"({sum(1 for v in report.clean.values() if v)} pass on the clean world)",
    ]
    for outcome in report.outcomes:
        mark = "ok " if outcome.detected == outcome.expected_detected else "BAD"
        kind = "detected" if outcome.expected_detected else "BLIND SPOT (must NOT be caught)"
        lines.append(
            f"  [{mark}] {outcome.name:<34} {kind:<30} "
            f"caught_by={list(outcome.caught_by) or '-'}"
        )
    for note in report.reporting:
        lines.append(f"  [report] {note}")
    for note in report.failures:
        lines.append(f"  [FAIL] {note}")
    lines.append(f"  verdict: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# adopting a playthrough that already exists in another lead's package
# --------------------------------------------------------------------------


def coverage_from_caught_map(
    name: str,
    caught: dict[str, Iterable[str]],
    *,
    clean_passes: bool,
    notes: dict[str, str] | None = None,
    blind_spots: Iterable[str] = (),
) -> PlaythroughReport:
    """Adopt a FOREIGN playthrough that already reports which criterion caught what.

    ``sim/coarsen.py`` and ``sim/playthrough.py`` belong to the simulation and
    figures area and are not this module's to restructure (C-4 ownership rules).
    They already
    emit ``defects_caught_by: {defect: [criteria]}`` plus a "the rule passes every
    scenario" flag, which is this protocol's shape written independently — so the
    same coverage requirement can be enforced over them without touching a line
    of another lead's code.

    That is the point: **the requirement lives in one place and reaches every
    playthrough in the repo, including the ones infra does not own.**
    """
    notes = dict(notes or {})
    blind = set(blind_spots)
    outcomes = tuple(
        DefectOutcome(
            name=defect,
            expected_detected=defect not in blind,
            detected=bool(list(criteria)),
            caught_by=tuple(str(c) for c in criteria),
            note=notes.get(defect, "adopted from a foreign playthrough report"),
        )
        for defect, criteria in sorted(caught.items())
    )
    failures: list[str] = []
    if not clean_passes:
        failures.append(
            f"{name}: the RULE does not pass every scenario on the clean world, so a caught "
            "defect cannot be attributed to the defect"
        )
    for outcome in outcomes:
        if outcome.expected_detected and not outcome.detected:
            failures.append(
                f"{name}: PLANTED DEFECT {outcome.name!r} WAS NOT DETECTED by any criterion "
                "(ADR-030)"
            )
        if not outcome.expected_detected and outcome.detected:
            failures.append(f"{name}: declared blind spot {outcome.name!r} was caught — update "
                            "the record, do not delete the test")
    if not outcomes:
        failures.append(f"{name}: the foreign report declares NO planted defects at all")
    return PlaythroughReport(
        name=name,
        clean={"foreign_rule_passes_every_scenario": clean_passes},
        clean_errors={},
        outcomes=outcomes,
        failures=tuple(failures),
        reporting=(),
        sole_catchers={
            o.name: o.caught_by[0]
            for o in outcomes
            if o.expected_detected and len(o.caught_by) == 1
        },
        dead_probes=(),
    )


def approximately(value: float | None, want: float, *, tol: float) -> bool:
    """``abs(value - want) <= tol``, with ``None`` and non-finite counted as FAIL.

    Hoisted here because every probe in this repo needs it and because ``None``
    must not compare its way to a pass: the C1.5 lesson is that non-finite and
    ``None`` have to be unpassable at the choke point, not handled per caller.
    """
    if value is None or not math.isfinite(value):
        return False
    return abs(float(value) - float(want)) <= tol
