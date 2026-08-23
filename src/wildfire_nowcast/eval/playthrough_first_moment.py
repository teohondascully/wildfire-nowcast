"""PLAYTHROUGH - G3's FIRST-MOMENT condition and the GEOMETRIC bar, end to end.

ADR-039 (4) and (5) pre-registered two changes to G3's criterion: a bar that is
symmetric in LOG space, and an explicit first-moment condition defined relative
to the wind ellipse. infra built both in ``common/dispersion.py`` and
deliberately did NOT wire them, because *a clause is ratified when the gate path
runs it, not when the function exists* (C-2). This playthrough is the evidence
that the gate path now runs them.

WHAT IS KNOWN BY CONSTRUCTION
-----------------------------
Four fires in four distinct spatial blocks, one 3-lead window each, scored
through the REAL :func:`~wildfire_nowcast.eval.metrics.evaluate` /
:func:`~wildfire_nowcast.eval.metrics.aggregate` / ``_headline`` /
:func:`~wildfire_nowcast.eval.baseline_run.g3_summary` path - not a
re-implementation of any of them. Truth grows 10/20/30 cells cumulatively inside
the growth band, so ``sum(truth) = 60`` on every block. The candidate's five
members are built to make ``sum(ensemble-mean) = 66`` exactly, and the ellipse's
to make it ``107.4`` exactly. Therefore

    growth_calibration(candidate) = 66 / 60   = 1.10   on every block
    growth_calibration(ellipse)   = 107.4 / 60 = 1.79  on every block

1.79 is not an arbitrary number: it is the calibrated ellipse's measured
over-prediction on real held-out fires (ADR-021 (3b)), so the scenario's
reference is the reference the gate actually uses, at the value it actually has.
The candidate therefore PASSES the first-moment condition
(``|log 1.10| = 0.0953 <= |log 1.79| = 0.5822``) while FAILING the dispersion
bar, which is the combination the combined outcome has to get right.

WHY THE DISPERSION RATIO IS SET RATHER THAN SCORED
--------------------------------------------------
``band_area_dispersion_ratio`` is planted at **0.82** on every block. That value
is chosen because it is the one place the two bars disagree: ``0.8 <= 0.82 <=
1.2`` is TRUE under the old interval and ``|log 0.82| = 0.1985 > log 1.2 =
0.1823`` is a FAILURE under the geometric one. **The bar change is
outcome-determinative exactly there, and this scenario sits on it.**
It is SET and not scored on purpose: whether ``area_dispersion_ratio`` is
computed correctly is ``tests/test_playthrough_dispersion.py``'s known answer,
and recovering someone else's known answer a second time here would be a second
implementation of it (C0's argument, applied to tests).

THE DECLARED BLIND SPOT, AND IT IS THE INTERESTING ONE
-------------------------------------------------------
The adjudicated pooling is the arithmetic mean of the RATIO over blocks, then
``|log|`` - the same pooling ``equal_block_mean`` applies to every other
criterion. A candidate that is **1.65x on two blocks and 0.55x on two others**
has an arithmetic mean of exactly 1.10 and is therefore INDISTINGUISHABLE from a
candidate that is 1.10x everywhere. The alternative log pooling emitted beside it
sees the difference in its VALUE (0.0953 -> 0.5493) but not in its VERDICT, which
still passes against the ellipse's 0.5822. So **neither pooling's pass/fail can
see compensating per-block errors**, and this milestone's whole subject is that
our block-to-block spread is 3-7x. That is planted as a defect with
``detected=False`` and asserted in the opposite direction, so the day it is
closed the build says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from wildfire_nowcast.common import dispersion as g3
from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.eval import baseline_run, metrics

PLAYTHROUGH_OWNER = "modelling (M9)"
PLAYTHROUGH_NOTE = (
    "G3's FIRST-MOMENT condition (ADR-039 (5)) and the GEOMETRIC bar (ADR-039 (4)), wired into "
    "eval/baseline_run.g3_summary and exercised through the real evaluate/aggregate/_headline "
    "path. Known answer: growth_calibration 1.10 for the candidate and 1.79 for the ellipse, "
    "exact by construction. The dispersion ratio is planted at 0.82 — the single value where "
    "the old [0.8, 1.2] interval and the geometric [1/1.2, 1.2] one DISAGREE — so the bar "
    "change is outcome-determinative in this scenario rather than merely present. Eight "
    "mutations: a smoothed denominator, the domain mask read in place of the growth band, a "
    "DEGENERATE reference (C6.2's void-not-passed rule one level up), a candidate silently "
    "scored on three blocks, and the asymmetric bar restored in both of the two places it "
    "could re-enter. Declared blind spot: neither pooling's VERDICT can see per-block errors "
    "that cancel."
)

#: Domain, initial blob and band radius. Small on purpose - the scenario has to
#: be readable, and every quantity below is countable by hand.
_H = _W = 25
_BLOB = slice(11, 14)
_BAND_RADIUS = 4
_LEADS = (1, 2, 3)

#: Truth's CUMULATIVE new-cell count in the band at leads 1, 2, 3.
_TRUTH_COUNTS: tuple[int, int, int] = (10, 20, 30)
_TRUTH_SUM = float(sum(_TRUTH_COUNTS))

#: Five members per model. Their cumulative counts are chosen so the ensemble
#: mean's SUM OVER LEADS is exactly 66 (candidate) and exactly 107.4 (ellipse),
#: which is 1.10x and 1.79x of truth's 60.
_CANDIDATE_MEMBERS: tuple[tuple[int, int, int], ...] = (
    (10, 20, 30),
    (11, 21, 31),
    (11, 22, 33),
    (12, 23, 34),
    (12, 24, 36),
)
_ELLIPSE_MEMBERS: tuple[tuple[int, int, int], ...] = (
    (18, 36, 53),
    (18, 36, 53),
    (18, 36, 53),
    (18, 36, 54),
    (18, 36, 54),
)
_SILENT_MEMBERS: tuple[tuple[int, int, int], ...] = ((0, 0, 0),) * 5

CANDIDATE = "kernel_m9"
REFERENCE = "ellipse"
DEGENERATE = "ellipse_brier_fit_all"
NULL_MODEL = "persistence"

#: Known answers, stated as constants so a probe cannot be quietly re-fitted to
#: whatever the code produced.
WANT_CANDIDATE_GC = 1.10
WANT_REFERENCE_GC = 1.79
WANT_DOMAIN_GC = (9 * 3 + 66) / (9 * 3 + 60)  # 1.0690: the mask the gate must NOT read
PLANTED_ADR = 0.82
BLOCKS: tuple[int, ...] = (10, 11, 12, 13)


def _ordered_cells() -> list[tuple[int, int]]:
    """Unburned cells, nearest-first by Chebyshev distance from the blob.

    Nearest-first guarantees every cell any member or the truth ever ignites lies
    inside ``growth_band(x0, 4)``, so the band mask is not silently clipping the
    scenario. There are 112 such cells and the largest count used is 54.
    """
    centre = (_BLOB.start + _BLOB.stop - 1) / 2.0
    cells = [
        (i, j)
        for i in range(_H)
        for j in range(_W)
        if not (_BLOB.start <= i < _BLOB.stop and _BLOB.start <= j < _BLOB.stop)
    ]
    cells.sort(key=lambda c: (max(abs(c[0] - centre), abs(c[1] - centre)), c))
    return cells


def _state(counts: Sequence[int], x0: np.ndarray) -> np.ndarray:
    """``[3, H, W]`` absorbing states whose cumulative new-cell counts are ``counts``."""
    order = _ordered_cells()
    out = np.stack([x0.copy() for _ in counts])
    for lead, k in enumerate(counts):
        for i, j in order[:k]:
            out[lead, i, j] = 1
    return out


def _samples(members: Sequence[Sequence[int]], x0: np.ndarray) -> np.ndarray:
    return np.stack([_state(m, x0) for m in members]).astype(np.uint8)


def _row(members: Sequence[Sequence[int]], x0: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """One model on one fire, through the REAL C6 path."""
    result = metrics.evaluate(
        _samples(members, x0),
        truth,
        x0=x0,
        leads=_LEADS,
        band_radius_cells=_BAND_RADIUS,
        meta={"fire_id": "playthrough", "t0": 0},
    )
    return baseline_run._headline(metrics.aggregate([result]), len(_LEADS))


def build() -> dict[str, Any]:
    """The INGREDIENTS, not the scores.

    **Scoring happens in :func:`_observe`, deliberately.** The harness rebuilds
    the world BEFORE entering a defect's context, so anything computed in
    ``build`` is computed OUTSIDE an instrument mutation - and an instrument
    mutation that cannot reach the code it mutates reads as a clean pass. This
    playthrough caught exactly that in its own first run: the smoothed-denominator
    defect went undetected because ``growth_calibration`` was called at build
    time. Same family as ADR-035 (7)'s "the mutation was not detected until it
    moved to the call site", and the third time in this project that a
    playthrough has caught its own instrument.
    """
    x0 = np.zeros((_H, _W), dtype=np.uint8)
    x0[_BLOB, _BLOB] = 1
    return {
        "x0": x0,
        "truth": _state(_TRUTH_COUNTS, x0).astype(np.uint8),
        "members": {
            CANDIDATE: _CANDIDATE_MEMBERS,
            REFERENCE: _ELLIPSE_MEMBERS,
            DEGENERATE: _SILENT_MEMBERS,
            NULL_MODEL: _SILENT_MEMBERS,
        },
        # DATA defects replace this hook; it is the identity on the clean world.
        "per_fire_hook": lambda per_fire: per_fire,
    }


def _score(world: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Four fires, four blocks, four models, through the REAL C6 path."""
    x0, truth = world["x0"], world["truth"]
    rows = {name: _row(m, x0, truth) for name, m in world["members"].items()}
    # The dispersion ratio is PLANTED, not scored - see the module docstring.
    for name in rows:
        rows[name] = dict(rows[name])
        rows[name]["band_area_dispersion_ratio"] = PLANTED_ADR

    per_fire: dict[str, Any] = {
        f"fire_{block}": {
            "fire_id": f"fire_{block}",
            "cv_fold": 3,
            "spatial_block_id": block,
            "n_windows": 1,
            "n_growth_windows": 1,
            "models": {
                name: {"growth_windows": dict(row), "all_windows": dict(row)}
                for name, row in rows.items()
            },
        }
        for block in BLOCKS
    }
    pooled = {name: {"growth_windows": dict(row)} for name, row in rows.items()}
    return per_fire, pooled


class _AsymmetricBar:
    """The OLD ``[0.8, 1.2]`` reading, restored. Delegates everything else.

    Stands in for the most likely regression: someone re-spelling the bar at the
    call site instead of importing it. ``__getattr__`` forwards to the real
    module so the mutation is exactly one behaviour wide.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(g3, name)

    def dispersion_condition(self, adr: Any, *, name: str = "dispersion") -> g3.ConditionResult:
        if adr is None:
            return g3.dispersion_condition(adr, name=name)
        inside = 0.8 <= float(adr) <= 1.2
        return g3.ConditionResult(
            name=name,
            outcome=g3.PASS if inside else g3.FAIL,
            value=float(adr),
            reference=1.2,
            detail="MUTATION: the pre-ADR-039 asymmetric interval.",
        )


def _observe(world: dict[str, Any]) -> dict[str, Any]:
    """Score, then run the LIVE G3 reporting path and read both conditions out of it."""
    per_fire, pooled = _score(world)
    per_fire = world["per_fire_hook"](per_fire)
    summary = baseline_run.g3_summary(
        pooled,
        per_fire,
        models=(CANDIDATE, REFERENCE),
        ablations={},
        stratum="growth_windows",
    )
    # C6.0's own question, asked of THIS metric: what does a DO-NOTHING null
    # score on the first-moment condition? `make null-check` cannot answer it -
    # `growth_calibration` is nested under a non-numeric key so that
    # `common/null_check.C6_METRICS` (which C-4 freezes to me) does not hard-fail
    # on an unregistered key, and a nested value is invisible to `_flatten`. So
    # the null is scored HERE, where it is legible, rather than argued in prose.
    null_summary = baseline_run.g3_summary(
        pooled,
        per_fire,
        models=(NULL_MODEL,),
        ablations={},
        stratum="growth_windows",
    )
    entry = summary["models"][CANDIDATE]
    dispersion_cell = next(
        c for c in entry["criteria"].values() if c["key"] == "band_area_dispersion_ratio"
    )
    first_moment = entry["first_moment_condition"]
    alt = first_moment.get("alt_pooling_log") or {}
    return {
        "candidate_gc": first_moment["candidate_equal_block"],
        "reference_gc": first_moment["reference_equal_block"],
        "first_moment_outcome": first_moment["outcome"],
        "first_moment_n_blocks": first_moment["n_blocks"],
        "dispersion_outcome": dispersion_cell["condition"]["outcome"],
        "dispersion_interval": list(dispersion_cell["interval"]),
        "in_interval_boolean": dispersion_cell["in_interval_equal_block"],
        "combined_outcome": entry["g3_conditions"]["outcome"],
        "alt_pooling_would_pass": alt.get("would_pass"),
        "null_first_moment_outcome": null_summary["models"][NULL_MODEL]["first_moment_condition"][
            "outcome"
        ],
        "null_growth_calibration": null_summary["models"][NULL_MODEL]["first_moment_condition"][
            "candidate_equal_block"
        ],
        # Not from the artifact: the two facts that make the scenario the right
        # scenario. Guards, below.
        "old_interval_would_have_accepted": 0.8 <= PLANTED_ADR <= 1.2,
        "undefined_is_not_a_pass": g3.dispersion_condition(None).passed is False,
    }


def _at(obs: dict[str, Any], key: str, want: float, tol: float = 1e-9) -> bool:
    return PT.approximately(obs.get(key), want, tol=tol)


def _compensating_blocks(per_fire: dict[str, Any]) -> dict[str, Any]:
    """Make the candidate 1.65x on two blocks and 0.55x on two - mean still 1.10."""
    for value, block in zip((1.65, 0.55, 1.65, 0.55), BLOCKS, strict=True):
        row = per_fire[f"fire_{block}"]["models"][CANDIDATE]["growth_windows"]
        row[baseline_run.FIRST_MOMENT_HEADLINE_KEY] = value
    return per_fire


def _drop_one_block(per_fire: dict[str, Any]) -> dict[str, Any]:
    per_fire[f"fire_{BLOCKS[0]}"]["models"][CANDIDATE]["growth_windows"] = None
    return per_fire


def _with_hook(hook: Any) -> Any:
    """A DATA defect: swap the post-scoring hook, leaving the instrument alone."""
    return lambda world: {**world, "per_fire_hook": hook}


def _smoothed(pred: Any, truth: Any) -> float | None:
    """``(pred + 1) / (truth + 1)`` - 'just avoid the zero denominator'."""
    if pred is None or truth is None:
        return None
    return (float(pred) + 1.0) / (float(truth) + 1.0)


PLAYTHROUGH = PT.Playthrough(
    name="g3_first_moment_condition",
    build=build,
    observe=_observe,
    note=PLAYTHROUGH_NOTE,
    probes=(
        PT.Probe(
            name="growth_calibration_recovers_the_constructed_ratio",
            check=lambda o: _at(o, "candidate_gc", WANT_CANDIDATE_GC, tol=1e-9),
            note="66 predicted / 60 truth cells in the growth band, exactly, on every block.",
        ),
        PT.Probe(
            name="the_reference_is_the_calibrated_ellipse_at_1_79",
            check=lambda o: _at(o, "reference_gc", WANT_REFERENCE_GC, tol=1e-9),
            note="107.4 / 60. A reference read off the wrong model is not a bar that was cleared.",
        ),
        PT.Probe(
            name="first_moment_passes_against_a_worse_reference",
            check=lambda o: o["first_moment_outcome"] == g3.PASS,
            note="|log 1.10| = 0.0953 <= |log 1.79| = 0.5822. Threshold-free by construction.",
        ),
        PT.Probe(
            name="the_first_moment_is_pooled_over_all_four_blocks",
            check=lambda o: o["first_moment_n_blocks"] == len(BLOCKS),
            note="C6.3: a mean that silently shrinks below 4 blocks reads as full coverage.",
        ),
        PT.Probe(
            name="the_geometric_bar_FAILS_the_candidate_at_0_82",
            check=lambda o: o["dispersion_outcome"] == g3.FAIL,
            note=(
                "|log 0.82| = 0.1985 > log 1.2 = 0.1823. This is the whole content of "
                "ADR-039 (4): the old interval accepted this exact value."
            ),
        ),
        PT.Probe(
            name="the_tables_interval_is_the_geometric_one",
            check=lambda o: (
                PT.approximately(o["dispersion_interval"][0], 1.0 / 1.2, tol=1e-12)
                and PT.approximately(o["dispersion_interval"][1], 1.2, tol=1e-12)
            ),
            note="A hand-written 0.8333 is not 1/1.2, and a second literal is how the "
            "asymmetry got in the first time.",
        ),
        PT.Probe(
            name="the_boolean_beside_it_agrees_with_the_condition",
            check=lambda o: o["in_interval_boolean"] is False,
            note="Both readings are emitted; they must not disagree on the same artifact.",
        ),
        PT.Probe(
            name="both_conditions_are_required",
            check=lambda o: o["combined_outcome"] == g3.FAIL,
            note="First moment PASS + dispersion FAIL must combine to FAIL, never to a pass.",
        ),
        PT.Probe(
            name="alt_pooling_is_reported_beside_the_adjudicated_one",
            check=lambda o: o["alt_pooling_would_pass"] is True,
            note="Emitted so the maintainer can see whether the pooling choice is ever "
            "outcome-determinative before ruling on it.",
        ),
        PT.Probe(
            name="C6_0_a_do_nothing_null_cannot_PASS_the_first_moment",
            check=lambda o: (
                o["null_first_moment_outcome"] == g3.UNDEFINED
                and PT.approximately(o["null_growth_calibration"], 0.0, tol=1e-12)
            ),
            note=(
                "C6.0 for this metric, MEASURED not argued. persistence ignites zero cells, so "
                "its growth_calibration is exactly 0, `log_distance(0)` is None, and the "
                "condition is UNDEFINED — which `ConditionResult.passed` reports as False and "
                "`bool()` REFUSES to answer. Five of this project's metrics were found paying "
                "a positive score to saying nothing; this one structurally cannot."
            ),
        ),
        PT.Probe(
            name="an_undefined_condition_is_not_a_pass",
            check=lambda o: o["undefined_is_not_a_pass"] is True,
            guard=True,
            note="area_dispersion_ratio goes UNDEFINED at PERFECT mean calibration, so this "
            "case arrives precisely as a model gets the first moment right.",
        ),
        PT.Probe(
            name="the_old_interval_would_have_accepted_this_scenario",
            check=lambda o: o["old_interval_would_have_accepted"] is True,
            guard=True,
            note="Pins the scenario onto the disagreement between the two bars. Without this "
            "the geometric-bar probe could pass for a reason unrelated to the change.",
        ),
    ),
    defects=(
        PT.Defect(
            name="smoothed_denominator",
            plant=PT.attribute_defect((metrics, "growth_calibration", _smoothed)),
            note="`(pred+1)/(truth+1)`, the 'just avoid dividing by zero' fix. It silently "
            "moves EVERY calibration toward 1 and is worst on the smallest, fastest blocks — "
            "which is where this project's block spread lives.",
        ),
        PT.Defect(
            name="domain_mask_instead_of_growth_band",
            plant=PT.attribute_defect(
                (baseline_run, "FIRST_MOMENT_HEADLINE_KEY", g3.FIRST_MOMENT_KEY)
            ),
            note=f"The domain value is diluted by the already-burned blob: {WANT_DOMAIN_GC:.4f} "
            "against the band's 1.10. Dropping a `band_` prefix has already produced two "
            "wrong headline numbers in this repo.",
        ),
        PT.Defect(
            name="degenerate_reference",
            plant=PT.attribute_defect((baseline_run, "FIRST_MOMENT_REFERENCE_MODEL", DEGENERATE)),
            note="The Brier-fitted ellipse ignites ZERO cells, so its growth_calibration is 0 "
            "and undefined in log space. C6.2's rule one level up: a reference that could not "
            "be scored must never read as a reference that was beaten.",
        ),
        PT.Defect(
            name="candidate_scored_on_three_blocks",
            plant=PT.data_defect(_with_hook(_drop_one_block)),
            note="The C6.3 hole `common/pooling.py` was written against: a mean over 3 blocks "
            "reported as if it were over 4. Here it also breaks the candidate/reference block "
            "correspondence, which is not a comparison at all.",
        ),
        PT.Defect(
            name="asymmetric_bar_in_the_condition",
            plant=PT.attribute_defect((baseline_run, "g3", _AsymmetricBar())),
            note="The bar re-spelled at the call site as `0.8 <= x <= 1.2`. This is how the "
            "22%-too-tolerant lower endpoint entered, and 0.82 is a value it accepts.",
        ),
        PT.Defect(
            name="asymmetric_interval_in_the_table",
            plant=PT.attribute_defect(
                (
                    baseline_run,
                    "G3_CRITERIA",
                    tuple(
                        (key, label, 0.8, high, mask, source)
                        if key == "band_area_dispersion_ratio"
                        else (key, label, low, high, mask, source)
                        for key, label, low, high, mask, source in baseline_run.G3_CRITERIA
                    ),
                )
            ),
            note="The OTHER place the old bar can come back: the printed interval and the "
            "boolean beside the condition. A table that narrates a different bar from the one "
            "adjudicated is the defect this data-driven table was built to prevent.",
        ),
        PT.Defect(
            name="compensating_per_block_errors",
            plant=PT.data_defect(_with_hook(_compensating_blocks)),
            detected=False,
            note="DECLARED BLIND SPOT. 1.65x on two blocks and 0.55x on two has an arithmetic "
            "mean of exactly 1.10, so the adjudicated pooling cannot distinguish it from a "
            "uniformly well-calibrated candidate; the alternative log pooling sees it in its "
            "VALUE (0.0953 -> 0.5493) but still passes against the ellipse's 0.5822, so "
            "neither VERDICT sees it. Our measured block spread is 3-7x, which is why this "
            "is stated rather than hidden.",
        ),
        PT.Defect(
            name="no_op_control",
            plant=PT.no_defect(),
            detected=False,
            note="The harness must not hallucinate a catch. If this is ever reported as "
            "detected, every catch above is suspect.",
        ),
    ),
)


def run_playthrough() -> dict[str, Any]:
    """Run the protocol and return the report as a dict.

    **THIS FUNCTION IS WHAT MAKES THE MODULE VISIBLE TO THE AUDIT, and that is a
    gap worth stating.** ``tests/test_playthrough_registry.discover_playthroughs``
    scans ``src/`` for modules that define a function named ``run_playthrough``,
    while ``declared_metadata`` accepts any of ``PLAYTHROUGH`` /
    ``build_playthrough`` / ``run_playthrough`` as the entry point. So a ``src/``
    module that declares its owner and note correctly and exposes ``PLAYTHROUGH``
    is **never discovered at all** - not registered, not red, just absent.
    Auto-discovery becomes auto-omission, which is the exact failure mode A14's
    "auto-discovery is not auto-forgiveness" paragraph was written against.
    Measured: without this function the registry collected 18 tests and none of
    them was this file. ``sim/blockanatomy.py`` satisfies the same requirement the
    same way (it defines both ``run_playthrough`` and ``build_playthrough``), so
    the convention exists - it is just not stated anywhere, and a lead who
    follows the documented declaration protocol and stops there gets silence.
    Escalated rather than worked around; this function works WITH the registry.
    """
    return PT.run(PLAYTHROUGH).as_dict()


def main() -> int:  # pragma: no cover - CLI convenience, mirrors the other playthroughs
    report = PT.run(PLAYTHROUGH)
    print(PT.format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
