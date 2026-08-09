"""G3's calibration criterion as a SEPARATION TEST (ADR-032 (7)).

THE RULING, AND WHY IT IS RIGHT — MEASURED, NOT ASSUMED
-------------------------------------------------------
modelling proposed replacing G3's absolute calibration bar with
``calibration_error / silent_floor < 1`` and **disclosed against itself** that
the barred degenerate Brier-fit ellipse scores 0.999 and would pass by 0.001.
A margin was needed and C-3 forbids inventing one on a single observation.

The maintainer ruled: *do not mint a constant; convert the criterion to a
SEPARATION TEST in the G2 style — the candidate must beat the best
degenerate/null arm by a stated number of equal-block SD.* That ruling is
implemented here, and the first thing this module did was check it. Measured on
the 30 arms of ``runs/baselines-20260809-035037/results.json`` (see
:data:`FITTING_SAMPLE`), scored against the degenerate ENVELOPE:

    every degenerate/null arm      -0.70 .. 0.00 SD, 0 of 4 blocks won
    every arm that is merely worse -0.99 .. -0.23 SD, <= 3 of 4 blocks won
    every real candidate (24 arms) +3.95 .. +11.97 SD, 4 of 4 blocks won

**The ratio form separated those populations by 0.001. The separation form
separates them by a factor of 5.6 with an empty region in between.** The ruling
is vindicated by measurement, not by preference.

TWO REFERENCES, BOTH REQUIRED — THE G2 SHAPE, REUSED
-----------------------------------------------------
G2 was adjudicated against a ``rule`` opponent AND an ``envelope`` (the best
score ANY calibrated ellipse reached at that horizon), because *the rule is what
the ruling says; the envelope is what makes the answer insensitive to how the
rule is read*. The same two references exist here and both must be cleared:

``rule``
    the **silent floor**, which is an ARITHMETIC IDENTITY rather than an
    opponent: ``calibration_error_silent_floor`` is the base rate of the scored
    set exactly, i.e. what a forecast that claims nothing scores. Verified on the
    fitting sample: ``persistence``'s ``calibration_error`` equals its
    ``silent_floor`` BITWISE at every block and every horizon. Using it removes
    the free parameter "which arm counts as the degenerate one" — and a ranking's
    opponent is a free parameter someone will eventually turn.
``envelope``
    the best (lowest) ``calibration_error`` any DECLARED degenerate/null arm
    reaches in that block. This is ADR-032 (7)'s literal text, and it is strictly
    harder than the floor: on the fitting sample the Brier-fit ellipse beats the
    arithmetic floor by 4e-6 at 3 h, which is the 0.999-vs-1.000 case in its
    natural units.

WHAT IS AND IS NOT A CONSTANT HERE
----------------------------------
* :data:`MIN_SEPARATION_SD` = 2.0 — the ONE fitted constant. Its sample is stated
  in :data:`FITTING_SAMPLE` and registered in ``contract.THRESHOLD_PROVENANCE``
  so ``test_every_threshold_states_its_fitting_sample`` covers it (C-3).
* UNANIMITY across blocks is threshold-free and separates the fitting sample
  perfectly on its own (every candidate 4/4, every non-candidate <= 3/4). It is
  required IN ADDITION, exactly as G2's record cited "4/4 blocks" beside its SD.
* ``relative_margin`` is REPORTED and never gated. It is modelling's own ratio
  in its honest form (``1 - error/floor``), it has an arithmetically known null
  answer of exactly 0.0, and putting a bar on it would re-mint precisely the
  margin C-3 forbade.

THE RESIDUAL HAZARD, STATED BECAUSE IT IS REAL
----------------------------------------------
**Statistical separation is not magnitude.** A forecast that beats the floor by
0.001 in every block with almost no block-to-block variation would score a large
``separation_sd`` while being useless — the 0.999 case, wearing a different hat.
Three things answer it and none of them is a constant: the ENVELOPE reference
(against which such an arm scores <= 0, because it IS the envelope), the
``relative_margin`` reported beside every verdict, and the pre-fixed absolute bar
ADR-020 already set. If a future arm ever shows large separation with a
negligible relative margin, that is a finding to escalate, not a pass to bank.

A degenerate SD is refused rather than blessed: if the block-to-block SD comes
out exactly zero with a non-zero mean, ``separation_sd`` is ``None`` and the
conditions are NOT met. An SD estimated as exactly 0.0 from four blocks is not an
SD, and dividing by it is how a 0.001 margin becomes infinite significance.

THIS MODULE DOES NOT ADJUDICATE G3
----------------------------------
It returns numbers and per-condition booleans. The gate verdict is the
maintainer's, on the same division of labour as ``eval/baseline_run.g3_summary``
— *a lead's own code should never contain the word that closes a gate*.

C0: one implementation, model-agnostic. Nothing here imports ``model/`` or
``eval/``, reads a checkpoint, or knows which arm is ours.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.common.calibration import GATE_CRITERION_KEY, GATE_MASK

__all__ = [
    "MIN_SEPARATION_SD",
    "MIN_BLOCKS_FOR_SEPARATION",
    "FITTING_SAMPLE",
    "BlockPair",
    "Separation",
    "separation",
    "calibration_separation",
    "conditions",
]

#: **The one fitted constant.** Equal-block SD the candidate must clear.
#:
#: 2.0 sits almost exactly in the middle of a MEASURED empty region (see
#: :data:`FITTING_SAMPLE`): the highest non-candidate scores -0.23 and the lowest
#: candidate +3.95, so any threshold in that interval separates the sample
#: perfectly and 2.0 is 2.2 SD above the one and 2.0 below the other.
#: It also reuses a precedent already in this repo — ``null_check.NOISE_FLOOR_SD``
#: is 2 SD, for the related reason that a small-sample SD carries large relative
#: error (at n = 4 blocks, ~41%).
MIN_SEPARATION_SD = 2.0

#: C6.3 already requires >= 4 distinct held-out spatial blocks for a gate. Stated
#: here too because an effect size computed on 2 blocks is a number about 2 fires.
MIN_BLOCKS_FOR_SEPARATION = 4

#: C-3: the sample :data:`MIN_SEPARATION_SD` was fitted on, in full.
FITTING_SAMPLE = (
    "runs/baselines-20260809-035037/results.json (the ADR-032 G3 run of record), split "
    "fingerprint 4848f491e8d588fa: 30 arms x 4 DISTINCT held-out spatial blocks {3,4,5,6} "
    "(bobcat/creek/czu/dolan) x 3 horizons = 360 arm-block-horizon cells. C-3 satisfied: 4 "
    "blocks, not 1. Statistic = equal-block paired margin on band_calibration_error against the "
    "degenerate ENVELOPE, mean/sd over blocks. MEASURED: degenerate/null arms (persistence, "
    "ellipse_brier_fit_all, ellipse_brier_fit_growth) score -0.70..0.00 SD and win 0/4 blocks; "
    "arms that are merely worse (ellipse, ellipse_cal2h, ellipse_cal3h, m6_fair_brier0) score "
    "-0.99..-0.23 and win <=3/4; all 24 real candidate arms score +3.95..+11.97 and win 4/4, at "
    "EVERY horizon. The threshold is placed in the measured empty region (-0.23, +3.95), whose "
    "width is 5.6x the largest non-candidate score. STATED LIMITATION: this sample contains no "
    "arm that is genuinely-but-marginally better, so it locates an empty region and does NOT "
    "locate the true boundary inside it; 2.0 is the midpoint of what was measured, not a fitted "
    "decision boundary. Cross-check, not the justification: at 4 blocks, 2.0 SD is a paired t of "
    "4.0 on 3 df."
)

_MIN_BLOCKS = 2


@dataclass(frozen=True)
class BlockPair:
    """One spatial block's candidate score and the reference it must beat."""

    block_id: int
    candidate: float
    reference: float
    label: str = ""


@dataclass(frozen=True)
class Separation:
    """A paired, equal-block separation. Never a verdict — see the module doc."""

    n_blocks: int
    mean_margin: float | None
    sd_margin: float | None
    separation_sd: float | None
    blocks_favouring: int
    relative_margin: float | None
    per_block: tuple[dict[str, Any], ...]
    undefined_reason: str = ""

    @property
    def unanimous(self) -> bool:
        return self.n_blocks > 0 and self.blocks_favouring == self.n_blocks

    def check(self) -> Separation:
        """Invariants, raised rather than warned — ``CalibrationTerms.check``'s
        standard: if these stop holding the statistic has changed shape and every
        caller must be re-derived rather than quietly reading something else."""
        if self.n_blocks != len(self.per_block):
            raise AssertionError(f"n_blocks={self.n_blocks} but {len(self.per_block)} rows")
        if not 0 <= self.blocks_favouring <= self.n_blocks:
            raise AssertionError(f"blocks_favouring={self.blocks_favouring} of {self.n_blocks}")
        for name in ("mean_margin", "sd_margin", "separation_sd", "relative_margin"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise AssertionError(
                    f"{name}={value} is not finite. A non-finite separation is how a zero "
                    "block-to-block SD turns a negligible margin into infinite significance; "
                    "this module refuses it rather than reporting it"
                )
        if self.sd_margin is not None and self.sd_margin < 0:
            raise AssertionError(f"sd_margin={self.sd_margin} < 0")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_blocks": self.n_blocks,
            "mean_margin": self.mean_margin,
            "sd_margin": self.sd_margin,
            "separation_sd": self.separation_sd,
            "blocks_favouring": self.blocks_favouring,
            "unanimous": self.unanimous,
            "relative_margin": self.relative_margin,
            "per_block": list(self.per_block),
            "undefined_reason": self.undefined_reason,
        }


def separation(pairs: Sequence[BlockPair], *, lower_is_better: bool = True) -> Separation:
    """Equal-block paired separation of ``candidate`` from ``reference``.

    One value per BLOCK, never per window: window pooling weights a block by how
    many hours it burned (Creek alone is 47% of the held-out growth pool), and
    C6.3's own logic says spatial blocks are the independent units. ADR-021 (4)
    adopted equal-block weighting prospectively from G3 onward; this is that rule
    applied to the difference rather than to the level.

    The margin is ``reference - candidate`` when lower is better, so a POSITIVE
    margin always means the candidate wins, whichever direction the metric runs.
    """
    rows = [p for p in pairs if _finite(p.candidate) and _finite(p.reference)]
    margins = np.array(
        [
            (p.reference - p.candidate) if lower_is_better else (p.candidate - p.reference)
            for p in rows
        ],
        dtype=np.float64,
    )
    per_block = tuple(
        {
            "spatial_block_id": p.block_id,
            "label": p.label,
            "candidate": p.candidate,
            "reference": p.reference,
            "margin": float(m),
            "candidate_wins": bool(m > 0),
        }
        for p, m in zip(rows, margins, strict=True)
    )
    n = int(margins.size)
    if n < _MIN_BLOCKS:
        return Separation(
            n_blocks=n,
            mean_margin=float(margins.mean()) if n else None,
            sd_margin=None,
            separation_sd=None,
            blocks_favouring=int((margins > 0).sum()),
            relative_margin=None,
            per_block=per_block,
            undefined_reason=(
                f"{n} block(s) with a value: an SD needs at least {_MIN_BLOCKS}, and C-3 forbids "
                "a pass/fail calibrated on one observation"
            ),
        ).check()

    mean = float(margins.mean())
    sd = float(margins.std(ddof=1))
    reason = ""
    if sd > 0.0:
        sep: float | None = mean / sd
    elif mean == 0.0:
        # Every block identical AND identical to the reference: no separation at
        # all, which is arithmetically 0 and not 0/0. `persistence` scores exactly
        # this against the silent floor, bitwise, on the fitting sample.
        sep = 0.0
    else:
        sep = None
        reason = (
            f"block-to-block SD is EXACTLY 0 with a non-zero mean margin ({mean:.3g}). An SD "
            "estimated as zero from a handful of blocks is not an SD; dividing by it is how a "
            "negligible margin acquires infinite significance, which is the 0.001-margin hazard "
            "this criterion replaced. Refused rather than blessed"
        )

    ref_scale = float(np.mean([p.reference for p in rows])) if rows else 0.0
    relative = (mean / ref_scale) if ref_scale > 0 else None
    return Separation(
        n_blocks=n,
        mean_margin=mean,
        sd_margin=sd,
        separation_sd=sep,
        blocks_favouring=int((margins > 0).sum()),
        relative_margin=relative,
        per_block=per_block,
        undefined_reason=reason,
    ).check()


def conditions(
    sep: Separation,
    *,
    min_sd: float | None = None,
    min_blocks: int | None = None,
) -> dict[str, Any]:
    """Per-condition booleans. **NOT a gate verdict** — see the module docstring.

    The two bars default to ``None`` and are resolved from the module constants
    AT CALL TIME rather than being bound as default arguments. That is not a
    style preference: a default argument freezes the constant at import, so the
    bar could not be moved by a planted defect and
    ``tests/test_playthrough_separation.py`` could not demonstrate that lowering
    it lets the barred degenerate arm through. A constant a mutation test cannot
    reach is a constant nothing verifies.

    Three conditions, and the conjunction is reported without being named a pass:

    ``separation``  the equal-block separation clears ``min_sd``
    ``unanimity``   EVERY block favours the candidate (threshold-free, and on the
                    fitting sample it separates the arms perfectly by itself)
    ``block_count`` at least ``min_blocks`` distinct held-out blocks (C6.3)
    """
    min_sd = MIN_SEPARATION_SD if min_sd is None else min_sd
    min_blocks = MIN_BLOCKS_FOR_SEPARATION if min_blocks is None else min_blocks
    met = {
        "separation": sep.separation_sd is not None and sep.separation_sd >= min_sd,
        "unanimity": sep.unanimous,
        "block_count": sep.n_blocks >= min_blocks,
    }
    return {
        **sep.as_dict(),
        "min_separation_sd": min_sd,
        "min_blocks": min_blocks,
        "conditions": met,
        "all_conditions_met": all(met.values()),
        "fitting_sample": FITTING_SAMPLE,
        "not_a_verdict": (
            "G3 is adjudicated by the maintainer. This reports a pre-fixed criterion and "
            "where a model falls; it contains no pass/fail for the gate."
        ),
    }


# --------------------------------------------------------------------------
# the G3 wiring: calibration_error against the degenerate reference
# --------------------------------------------------------------------------

#: Arms that are DEGENERATE/NULL by construction and therefore form the envelope.
#: Named rather than inferred: "which arm is degenerate" must be a declaration a
#: reader can audit, not a heuristic on the numbers it is about to judge.
#: ``persistence`` ignites nothing; the two ``ellipse_brier_fit_*`` arms are the
#: C6.2-barred Brier fits retained as controls (8 cells against truth's 4,557).
DEFAULT_DEGENERATE_ARMS: tuple[str, ...] = (
    "persistence",
    "ellipse_brier_fit_all",
    "ellipse_brier_fit_growth",
)

_ERROR_KEY = f"band_{GATE_CRITERION_KEY}_by_horizon"
_FLOOR_KEY = f"band_{GATE_CRITERION_KEY}_silent_floor_by_horizon"


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value)


def _horizon_value(fire: Mapping[str, Any], model: str, key: str, horizon: int, stratum: str):
    row = ((fire.get("models", {}).get(model) or {}).get(stratum)) or {}
    return (row.get(key) or {}).get(str(horizon))


def calibration_separation(
    per_fire: Mapping[str, Mapping[str, Any]],
    model: str,
    horizon: int,
    *,
    reference: str = "envelope",
    degenerate_arms: Sequence[str] = DEFAULT_DEGENERATE_ARMS,
    stratum: str = "growth_windows",
) -> Separation:
    """G3's calibration half for one model at one horizon, per spatial block.

    ``reference``
        ``"floor"``    — the arithmetic silent floor (a forecast claiming nothing).
        ``"envelope"`` — the best score any ``degenerate_arms`` member reaches in
                         that block, floored by the arithmetic value so a missing
                         arm can never make the reference EASIER. Default,
                         because it is ADR-032 (7)'s literal text and strictly
                         harder.

    Reads only the C6 per-fire payload: no model internals, no checkpoint. A fire
    contributes to its ``spatial_block_id``, and blocks are averaged, never
    windows (ADR-021 (4)).
    """
    if reference not in ("floor", "envelope"):
        raise ValueError(f"reference must be 'floor' or 'envelope', got {reference!r}")

    by_block: dict[int, list[tuple[float, float, str]]] = {}
    for fire_id, fire in per_fire.items():
        candidate = _horizon_value(fire, model, _ERROR_KEY, horizon, stratum)
        floor = None
        for arm in degenerate_arms:
            floor = _horizon_value(fire, arm, _FLOOR_KEY, horizon, stratum)
            if _finite(floor):
                break
        if not (_finite(candidate) and _finite(floor)):
            continue
        ref = float(floor)
        if reference == "envelope":
            arms = [
                _horizon_value(fire, arm, _ERROR_KEY, horizon, stratum)
                for arm in degenerate_arms
            ]
            scored = [float(v) for v in arms if _finite(v)]
            if scored:
                ref = min(ref, min(scored))
        block_id = int(fire.get("spatial_block_id", -1))
        by_block.setdefault(block_id, []).append((float(candidate), ref, str(fire_id)))

    pairs = [
        BlockPair(
            block_id=block_id,
            candidate=float(np.mean([c for c, _, _ in rows])),
            reference=float(np.mean([r for _, r, _ in rows])),
            label=",".join(sorted(label for _, _, label in rows)),
        )
        for block_id, rows in sorted(by_block.items())
    ]
    return separation(pairs, lower_is_better=True)


def calibration_separation_summary(
    per_fire: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    horizon_h: int,
    *,
    degenerate_arms: Sequence[str] = DEFAULT_DEGENERATE_ARMS,
    stratum: str = "growth_windows",
) -> dict[str, Any]:
    """Every model, every horizon, both references. Emits numbers, not a verdict."""
    out: dict[str, Any] = {
        "criterion": GATE_CRITERION_KEY,
        "mask": GATE_MASK,
        "stratum": stratum,
        "degenerate_arms": list(degenerate_arms),
        "min_separation_sd": MIN_SEPARATION_SD,
        "fitting_sample": FITTING_SAMPLE,
        "source": "ADR-032 (7) — separation in the G2 style, not a minted margin",
        "models": {},
    }
    for model in models:
        rows: dict[str, Any] = {}
        for horizon in range(1, horizon_h + 1):
            entry = {}
            for ref in ("floor", "envelope"):
                sep = calibration_separation(
                    per_fire,
                    model,
                    horizon,
                    reference=ref,
                    degenerate_arms=degenerate_arms,
                    stratum=stratum,
                )
                entry[ref] = conditions(sep)
            entry["all_conditions_met_both_references"] = bool(
                entry["floor"]["all_conditions_met"] and entry["envelope"]["all_conditions_met"]
            )
            rows[str(horizon)] = entry
        out["models"][model] = rows
    return out
