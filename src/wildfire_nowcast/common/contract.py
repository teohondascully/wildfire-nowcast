"""Executable form of the INTERFACES.md contracts C1, C2 and C3.

The version this module enforces is :data:`CONTRACT_VERSION`, **derived at import
from INTERFACES.md line 1** - it is deliberately not restated in this docstring.
It used to be: this line read ``**v2.5**`` while ``CONTRACT_VERSION`` said
``v2.12``, seven versions apart, in the file that adjudicates the contract
(ADR-036 (5)). A version number written down twice is a version number that will
disagree with itself; the fix is to have one place, not two accurate ones.

This module is the single source of truth for the channel list, dtypes, grid
and time semantics. Tests, the synthetic generator, the io layer and every
lead's ingestion code import their constants from here so that "the contract"
is one object, not four copies that drift (C0, ADR-007).

Design notes
------------
* **Structure is not plausibility (R11, ADR-010).** C1.5's declarations are
  satisfied by ``-9999``: it is finite, integral and static. A CZU tensor
  carrying the USGS LFPS coastline sentinel therefore scored
  ``OK - 42 checks passed (reporting-ready)`` with a mean canopy cover of
  **-3085%**. C1.7 closes that instance by asserting the *definitional* range of
  the two channels that have one. The CLASS remains open: when you add a
  channel, ask what physically impossible value would still pass, and say so.
* **A verdict must never pass by default (ADR-012).** Every outcome goes through
  :func:`_verdict` at the single choke point :meth:`ContractReport.add`, which
  maps non-finite and unusable outcomes to ``False``. ``bool(float("nan"))`` is
  ``True`` in Python, so a NaN reaching a verdict unguarded reports PASS - the
  exact defect sim found in its own ladder, where an unverifiable
  statistic printed ``[ok]`` and simultaneously hid a good result and passed a
  weak one. Unverifiable is a FAIL here, per C-1.
* The checkers **collect** violations instead of raising on the first one, so a
  failing tensor produces a full punch list in one pass. Corollary: a malformed
  *attribute* must fail its own clause rather than raise, or one bad value
  truncates the punch list and hides everything after it.
* The checkers depend only on ``xarray``/``numpy`` - deliberately *not* on
  :mod:`wildfire_nowcast.common.zarr_io`. A tensor written with plain xarray by
  another lead is judged by exactly the same yardstick as one written by our
  own writer.
* Every check is either ``fail`` or ``reporting`` severity. ``reporting``
  mirrors the contract's own two-tier language - C3.3 says a norm-stats file
  MUST carry ``n_train_blocks`` (unconditional: a ``fail``) and that any
  *REPORTED* result must assert ``n_train_blocks >= 2`` (conditional: a
  ``reporting``). A ``reporting`` gap never blocks plumbing, is printed on
  every report whether or not you asked for it, is machine-readable as
  ``reporting_ready: false``, and is a hard failure under ``--for-reporting``
  (``make contract-reporting``). It is not an advisory note in a JSON file.

  The dividing line: a clause is ``fail`` when the invariant is **violated or
  unverifiable**, and ``reporting`` when it is **verifiable and satisfied, but
  recorded somewhere non-canonical** (or when the contract itself scopes it to
  reported results). A checker that fails a substantively-correct tensor is
  worse than no checker; one that stays silent about a known gap is worse still.

Standalone use (no pytest, any path)::

    .venv/bin/python -m wildfire_nowcast.common.contract path/to/tensor.zarr
    .venv/bin/python -m wildfire_nowcast.common.contract path/to/x.zarr --labels-only
    make contract TENSOR=path/to/tensor.zarr
    make contract-reporting TENSOR=path/to/tensor.zarr
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.common.paths import repo_root

__all__ = [
    "CONTRACT_VERSION",
    "DEFERRED_CLAUSES",
    "THRESHOLD_PROVENANCE",
    "CLAUSE_IMPLEMENTATIONS",
    "ClauseImpl",
    "CLAUSE_ENFORCED",
    "CLAUSE_EXTERNAL",
    "CLAUSE_PROCESS",
    "CLAUSE_DEFERRED",
    "MIN_BUFFER_MARGIN_CELLS",
    "MANIFEST_VINTAGE_LAG_KEY",
    "MANIFEST_IGNITION_COMPONENTS_KEY",
    "CHANNELS",
    "FEATURE_CHANNELS",
    "N_CHANNELS",
    "N_FEATURE_CHANNELS",
    "CHANNEL_INDEX",
    "CHANNEL_INDEX_OFFSET",
    "CHANNEL_UNITS",
    "STATIC_CHANNELS",
    "CATEGORICAL_CHANNELS",
    "BINARY_CHANNELS",
    "INTEGER_CHANNELS",
    "FBFM40_CLASSES",
    "FBFM40_NONBURNABLE",
    "FBFM40_OPEN_WATER",
    "PHYSICAL_RANGES",
    "FIRE_STATE",
    "FEATURES",
    "UNBURNED",
    "BURNING",
    "BURNED_OUT",
    "FIRE_STATE_VALUES",
    "CRS_EPSG",
    "CRS_STRING",
    "CELL_SIZE_M",
    "LATTICE_ORIGIN_M",
    "TIME_CONVENTION",
    "TENSOR_DIMS",
    "VAR_DIMS",
    "FIRE_STATE_DIMS",
    "FEATURES_DIMS",
    "MANIFEST_KEYS",
    "NORM_STATS_KEYS",
    "SEVERITY_FAIL",
    "SEVERITY_REPORTING",
    "channel_dtype",
    "feature_index",
    "fire_state_violations",
    "is_on_lattice",
    "Check",
    "ContractReport",
    "ContractViolation",
    "open_tensor_dataset",
    "tensor_bounds",
    "check_tensor",
    "check_manifest",
    "check_norm_stats",
    "check_all",
]

# --------------------------------------------------------------------------
# C1 constants
# --------------------------------------------------------------------------

#: Where the contract's version actually lives. Line 1 of this file is the
#: authoritative statement of the contract version - the maintainer writes it
#: when ratifying, and ``tests/test_clause_registry.py`` has always read it.
INTERFACES_RELATIVE_PATH = "docs/interfaces.md"

_VERSION_RE = re.compile(r"\bINTERFACES\s+(v\d+(?:\.\d+)*)")


def _read_contract_version() -> str:
    """Parse ``vX.Y`` out of INTERFACES.md line 1. **Raises rather than guessing.**

    [A14, ADR-033 (1)] This used to be a hardcoded literal, and it drifted from
    the file FOUR times - three of them the maintainer's, once caught by
    infra's own C-2 audit catching its author's boss inside a single edit.
    Worse, the duplication made the ownership rule unenforceable by construction:
    ``test_contract_version_matches_interfaces`` reads INTERFACES line 1, so the
    lead who owns the code had to edit the file it does not own in order to make
    the build green. **Two sources of truth for one fact forces a boundary
    crossing.** The ruling was to delete the second source, not to correct it in
    both places (ADR-020 (5)'s lesson: *removing the second source of truth is
    worth more than correcting the value in both*).

    **There is deliberately NO literal fallback.** A fallback is how drift hides:
    the moment the parse fails, a stale constant would take over silently and the
    checker would print a confident version it did not read. Failing at import is
    loud, immediate, and impossible to mistake for a green run - the same
    reasoning as C-1's "unverifiable is a hard fail".
    """
    path = repo_root() / INTERFACES_RELATIVE_PATH
    try:
        first_line = path.read_text().splitlines()[0]
    except (OSError, IndexError) as exc:
        raise RuntimeError(
            f"cannot read the contract version: {path} is missing or empty ({exc}). "
            "CONTRACT_VERSION is DERIVED from INTERFACES.md line 1 and has no fallback "
            "literal, because a fallback is how a stale version hides."
        ) from exc
    match = _VERSION_RE.search(first_line)
    if not match:
        raise RuntimeError(
            f"cannot parse a contract version from {path} line 1: {first_line!r}. "
            f"Expected a match for {_VERSION_RE.pattern!r}, e.g. '# INTERFACES v2.12 ...'. "
            "Refusing to fall back to a literal: a checker printing a version it did not "
            "read is the stale-checker hazard wearing a current label."
        )
    return match.group(1)


#: The INTERFACES.md version these checks enforce, **DERIVED from
#: INTERFACES.md line 1 at import time** - never restated here. It is printed on
#: every report: a checker silently one version behind the contract is worse than
#: no checker, because it fails conformant data and passes stale data.
CONTRACT_VERSION = _read_contract_version()

#: Clauses that INTERFACES ratifies at or below :data:`CONTRACT_VERSION` and
#: this checker deliberately does **not** enforce yet, with the reason.
#:
#: This exists because the version string alone can lie. A checker printing
#: "enforcing v2.5" while silently skipping a v2.3 clause is the stale-checker
#: hazard wearing a current label - worse than an honestly old checker, because
#: nobody goes looking. It is the C-1 corollary applied to the checker itself:
#: declaring a weakness is a gate, omitting it is a failure. Printed on every
#: report and carried in ``--json``; emptying this dict is the goal.
DEFERRED_CLAUSES: dict[str, str] = {
    "C6.7": (
        "an instrument may adjudicate at a lead where it has power, and MUST report at "
        "1/2/3 h with its power at each lead beside the verdict. RATIFIED AND NOT ENFORCED: "
        "no code yet checks that an artifact carrying a collapse verdict also carries the "
        "lead profile, so today this holds only because the leads were told to. It is "
        "deferred rather than process BECAUSE AN ARTIFACT CAN CARRY IT - the enforcement is "
        "writable and simply is not written. Enforcing it means asserting, wherever a "
        "collapse verdict is emitted, that a per-lead power profile is emitted with it; that "
        "sits in eval/ and sim/, not here, so it is external work this registry cannot do. "
        "WHY IT EXISTS: [v2.8] forbids picking the horizon where the OPPONENT is weakest and "
        "nobody wrote the mirror, so our own collapse check came to adjudicate at the lead "
        "where IT is strongest - 97-100% power at leads 5-6 while carrying 6.0% at lead 1, "
        "and this project forecasts 1-3 h (ADR-123)."
    ),
    "C1.6": (
        "leakage smoke alarm (|2·AUC−1| <= 0.6 for a static channel vs the final footprint). "
        "NOT ENFORCED, and RE-MEASURED at n=12 fires / 11 spatial blocks (A10): burnable-cell "
        "masking does NOT rescue the 0.6 bar. SCU elevation scores 0.702 masked (0.768 "
        "unmasked) and July Complex aspect_sin 0.561 — and those are the two MULTI-IGNITION "
        "fires (C2 n_ignition_components 2 and 2), whose 'final footprint' is several fires, "
        "so a static channel separating them is expected rather than leaky. Distribution over "
        "all 96 channel-fire pairs, burnable-masked: median 0.103, p95 0.460, p99 0.568, max "
        "0.702; the measured channel-13 leak scored ~0.92. A single global bar therefore has "
        "at most 0.92/0.702 = 1.31x of margin, while PER-CHANNEL bars have 2.2x on the channel "
        "that actually leaked (recent_burn_scar: legitimate max 0.409). Proposal + full table "
        "in docs/decisions.md (A10). ADR required; not an infra decision, and "
        "C-3 forbids shipping either form on infra's own judgement."
    ),
}

#: [v2.6] C-3 - every pass/fail constant, and the sample it was fitted on.
#:
#: C-3 does not say "prefer larger samples"; it says a constant that decides a
#: pass/fail MUST STATE the sample it was fitted on, and that sample must span
#: >= 2 spatial blocks. A rule nobody can audit is the C-2 failure again, so the
#: statement lives in code next to the constant, is printed on request, and is
#: asserted by tests to cover every threshold this module applies. Three
#: near-misses came from skipping this (C1.6's bar from one fire, G2's bar from
#: one held-out block, norm stats from one train fire); each looked locally
#: reasonable, which is why it is mechanical now.
THRESHOLD_PROVENANCE: dict[str, str] = {
    "PHYSICAL_RANGES.canopy_cover": (
        "[0, 100] — DEFINITIONAL (a percentage), not fitted. No sample can widen or narrow it. "
        "Observed over 12 fires: [0, 100]."
    ),
    "FBFM40_CLASSES": (
        "the 45-code Scott & Burgan (2005) enumeration — DEFINITIONAL, not fitted. Membership "
        "rather than an interval, because fuel_model_id=0 (a common LANDFIRE fill) is finite, "
        "integral, static and inside every plausible interval (ADR-013 §4)."
    ),
    "MIN_TRAIN_BLOCKS_FOR_REPORTING": (
        "2 — the smallest number that is not 1 (C3.3/ADR-008). Not fitted to data: the guard "
        "exists because n=1 makes the only train fire the only landscape."
    ),
    "MIN_BUFFER_MARGIN_CELLS": (
        "10 cells = the 10 km buffer C1.2 mandates — DEFINITIONAL given the buffer, and the "
        "check is one-sided for that reason. Measured on the fitting sample of ALL 12 built "
        "fires spanning 11 spatial blocks (C-3 satisfied): observed margins 10-13 cells, "
        "min exactly 10 on 6 of 12 fires, so the bound is tight and not padded."
    ),
    "_COORD_TOL_M": (
        "1e-6 m — floating-point tolerance on a 1000 m grid, not a calibrated threshold."
    ),
    "null_check.NOISE_FLOOR_SD": (
        "2 seed SDs — a NOISE FLOOR, not a fitted constant, and deliberately so: C6.0 needs to "
        "say whether two forecasters differ, and the alternative is a magic separation threshold "
        "that C-3 would then require a sample for. The SCALE is measured from each run's own "
        "seeds (default 5), so it states no claim about any set of fires. 2 rather than 1 because "
        "a 5-seed SD carries ~35% relative error, and because flagging more is the conservative "
        "direction for a safety check. Measured consequence on the A11 fixture: at 1 SD, "
        "dispersion_ratio's verdict flipped between disjoint seed sets; at 2 SD it is BLIND on "
        "4 of 4."
    ),
}

# --------------------------------------------------------------------------
# C-2: the clause registry - ratification is not implementation
# --------------------------------------------------------------------------

#: Every clause status the registry may carry.
CLAUSE_ENFORCED = "enforced"  #: this module (or common/splits.py) asserts it
CLAUSE_EXTERNAL = "external"  #: asserted in code another lead owns, plus tests
CLAUSE_PROCESS = "process"  #: a rule about how we work; no artifact can carry it
CLAUSE_DEFERRED = "deferred"  #: ratified and deliberately not enforced yet


@dataclass(frozen=True)
class ClauseImpl:
    """Where one numbered INTERFACES clause is implemented - or why it is not.

    ``checks`` are check ids this checker emits; ``where`` are importable dotted
    symbols. Both are VERIFIED by ``tests/test_clause_registry.py`` - check ids
    must appear in a report generated from a conformant artifact, and every
    dotted symbol must import. A registry that can lie is the problem, not the
    fix.
    """

    status: str
    checks: tuple[str, ...] = ()
    where: tuple[str, ...] = ()
    note: str = ""


#: **The audit table, kept in code so it cannot rot.**
#:
#: C1.5's ``features must be finite`` was ratified at v2.3 and never
#: implemented; it sat on paper for two contract versions while an all-NaN
#: channel passed 56 checks and ``+inf`` was actively blessed by two green
#: clauses. That was found by hand. This registry makes the same audit
#: mechanical: every numbered clause in INTERFACES.md must appear here with one
#: of four statuses, and ``tests/test_clause_registry.py`` fails if INTERFACES
#: grows a clause this file has not classified. A clause that exists only in
#: INTERFACES.md is worse than no clause, because everyone downstream believes
#: they are protected (C-2).
CLAUSE_IMPLEMENTATIONS: dict[str, ClauseImpl] = {
    "C-1": ClauseImpl(
        CLAUSE_ENFORCED,
        where=(
            "wildfire_nowcast.common.contract.SEVERITY_FAIL",
            "wildfire_nowcast.common.contract.SEVERITY_REPORTING",
            "wildfire_nowcast.common.contract.ContractReport.reporting_ok",
        ),
        note="two tiers exist, are printed unconditionally, and --for-reporting promotes "
        "reporting gaps to hard failures.",
    ),
    "C-2": ClauseImpl(
        CLAUSE_ENFORCED,
        where=("wildfire_nowcast.common.contract.CLAUSE_IMPLEMENTATIONS",),
        note="this registry IS the implementation of C-2, together with the test that parses "
        "INTERFACES.md and fails on any clause missing from it.",
    ),
    "C-3": ClauseImpl(
        CLAUSE_ENFORCED,
        where=("wildfire_nowcast.common.contract.THRESHOLD_PROVENANCE",),
        note="every pass/fail constant states its fitting sample; a test asserts the registry "
        "covers each one. C-3's judgement half - whether a stated sample is adequate - is a "
        "human call and is deliberately not automated.",
    ),
    "C-4": ClauseImpl(
        CLAUSE_PROCESS,
        note="[v2.11, ADR-022] the FROZEN SET is a rule about who may edit what WHILE ANOTHER "
        "LEAD IS RUNNING. No artifact can carry it and no checker can see it: the same file, "
        "the same edit and the same diff are legal or illegal depending on whether some other "
        "session is live, which is a fact about the maintainer's schedule and not about this "
        "repo. What IS mechanical is the DETECTION of a violation after the fact, and that is "
        "C-4.2 - see also C8.code_agrees_across_run. Classified `process` deliberately rather "
        "than left unclassified: C-2's whole point is that an unenforceable clause must say so "
        "out loud instead of being silently absent.",
    ),
    "C-4.3": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("environment_agrees_across_run", "environment_sampled_both_ends"),
        where=(
            "wildfire_nowcast.common.environment.environment_fingerprint",
            "wildfire_nowcast.common.splits.ENVIRONMENT_FINGERPRINT_FAMILY",
            "wildfire_nowcast.common.runs._environment_stamp",
        ),
        note="[v2.12, ADR-024] the INTERPRETER ENVIRONMENT joins C-4's frozen set. Origin is "
        "data's, not the contract's: D5 declined to `pip install scipy` mid-training and "
        "wrote pure-numpy filters instead, reading C-4's INTENT over its text - C-4 enumerated "
        "FILES and named no environment, so a lead could have installed anything mid-run and "
        "violated no clause. Like C-4, WHO may install WHAT WHILE is `process` and cannot be "
        "checked from inside the repo; what IS mechanical is DETECTION AFTER THE FACT, which is "
        "how C-4.2 was implemented and is why this is `enforced`: the environment is fingerprinted "
        "(interpreter, every installed distribution, lockfile digests, a named set of system "
        "tools), stamped structurally into every run dir by `runs.create_run_dir`, and "
        "disagreement between the two ends is a HARD FAIL under C8. Sampling one end is a "
        "REPORTING gap on C-4.2's precedent and for its reason: every artifact in runs/ predates "
        "this clause and stamps no environment at all, so a hard clause would fail the whole "
        "archive on bookkeeping rather than on a measurement. SCOPE IS DECLARED IN THE PAYLOAD "
        "(`covers`): arbitrary system packages are NOT enumerable portably and are not claimed.",
    ),
    "C-4.1": ClauseImpl(
        CLAUSE_PROCESS,
        where=("wildfire_nowcast.common.iou_terms.GATE_CRITERION_KEY",),
        note="[v2.11, ADR-022] OWNERSHIP of eval/. A process clause with one testable "
        "consequence: infra's edits to eval/metrics.py must be ADDITIVE, so C6's "
        "pre-existing keys must keep their values. tests/test_iou_terms.py already asserts the "
        "additive half (best_member_iou is bit-identical to a from-definition recomputation, "
        "and common.iou_terms.jaccard is bit-identical to eval.metrics.fuzzy_iou). WHO may edit "
        "WHICH file cannot be checked from inside the repo; the DECLARATION requirement is met "
        "by this lead's status entry.",
    ),
    "C-4.2": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("code_agrees_across_run", "code_sampled_both_ends"),
        where=(
            "wildfire_nowcast.common.splits.code_fingerprint_ends",
            "wildfire_nowcast.common.splits.CODE_FINGERPRINT_FAMILIES",
        ),
        note="[v2.11, ADR-022] a code fingerprint must be sampled BEFORE and AFTER. Enforced "
        "under C8, where the contract puts it: disagreement between the two ends is a HARD "
        "FAIL; sampling only one end is a REPORTING gap, because every artifact in runs/ "
        "predates this clause and the two that stamp scoring_code stamp it once. NOTE for the "
        "next bump: C-4.2's text says to hard-fail 'as check_common_code_unchanged already does "
        "for common/', but that function does NOT raise - its own docstring says 'Never raises' "
        "and its status field says 'PROPOSAL - reported, not enforced'. The hard fail is this "
        "clause, here; the premise it cites does not exist yet.",
    ),
    "C0": ClauseImpl(
        CLAUSE_ENFORCED,
        where=(
            "wildfire_nowcast.common.states.apply_state_rule",
            "wildfire_nowcast.common.zarr_io.build_manifest",
            "wildfire_nowcast.common.splits.split_fingerprint",
        ),
        note="single implementations live in common/; tests assert the retired data/ duplicates "
        "stay deleted and that eval/'s split fingerprint agrees with common/'s byte for byte.",
    ),
    "C1": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "open",
            "variables_present",
            "no_unknown_variables",
            "channel_coord",
            "channel_index_offset",
            "dtypes",
            "dims",
            "n_channels",
            "shape_consistency",
            "crs",
        ),
    ),
    "C1.1": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("fire_state_domain", "fire_state_absorbing", "fire_state_no_skip"),
        where=(
            "wildfire_nowcast.common.states.apply_state_rule",
            "wildfire_nowcast.common.contract.fire_state_violations",
        ),
    ),
    "C1.2": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("cell_size_attr", "lattice_snap", "buffer_margin"),
        where=("wildfire_nowcast.common.contract.is_on_lattice",),
    ),
    "C1.3": ClauseImpl(CLAUSE_ENFORCED, checks=("time_convention",)),
    "C1.4": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "x_coord",
            "y_coord",
            "time_coord",
            "time_monotone",
            "time_hourly",
            "attr_time_start_utc",
            "attr_time_end_utc",
            "fire_state_absorbing",
        ),
    ),
    "C1.5": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "features_finite",
            "static_channels_constant",
            "mask_channels_binary",
            "class_channels_integral",
        ),
    ),
    "C1.6": ClauseImpl(
        CLAUSE_DEFERRED,
        note="see DEFERRED_CLAUSES['C1.6']; re-measured at n=12 and still not shippable as "
        "written.",
    ),
    "C1.7": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("range_canopy_cover", "range_fuel_model_id"),
        where=("wildfire_nowcast.common.contract.PHYSICAL_RANGES",),
    ),
    "C2": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "exists",
            "parses",
            "keys",
            "type_fire_id",
            "type_cv_fold",
            "type_spatial_block_id",
            "type_provenance",
            "n_hours_matches_tensor",
            "bbox_matches_tensor",
            "fuel_vintage_lag_years",
            "n_ignition_components",
            "provenance_declares_sources",
        ),
    ),
    "C3": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "channel_order",
            "mean_shape",
            "std_shape",
            "std_positive",
            "train_folds",
            # [A14, ADR-038 (6)] membership declared BY FIRE ID, and disjoint.
            "norm_stats_declares_fire_ids",
            "train_heldout_disjoint",
            "declared_membership_matches_manifests",
        ),
        where=(
            "wildfire_nowcast.common.zarr_io.compute_norm_stats",
            "wildfire_nowcast.common.splits.declared_split_membership",
        ),
        note="the three id-level checks are cross-artifact (norm_stats.json vs every "
        "manifest) and live in splits.check_split_assignment, because a clause about a "
        "RELATION between artifacts needs a home that sees more than one (C-2's structural "
        "rule). They replaced a `train_heldout_disjoint` that intersected two lists "
        "split_fingerprint builds as a PARTITION - green by construction, on any input.",
    ),
    "C3.1": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "block_maps_to_one_fold",
            "train_heldout_blocks_disjoint",
        ),
        where=("wildfire_nowcast.common.splits.check_split_assignment",),
        note="cross-fire: invisible to every per-tensor check, like C8.",
    ),
    "C3.2": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("mean_shape", "std_shape", "categorical_identity", "categorical_note"),
    ),
    "C3.3": ClauseImpl(CLAUSE_ENFORCED, checks=("n_train_blocks", "bootstrap_guard")),
    "C3.4": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("mean_finite", "std_finite", "mean_within_physical_range"),
        note="the norm-stats file is checked as its own artifact, not as an aggregate of "
        "per-fire reports - a train mean outside a C1.7 definitional range is the measured "
        "CZU case (-492% canopy). The 'necessary but not sufficient' half is a working rule.",
    ),
    # [v2.14] Maintainer edit, declared (ADR-044). SECOND clause-authoring
    # crossing in one session: A14's auto-discovery covers metrics and
    # playthroughs but NOT clause classification, so writing a clause still
    # compels an edit to this file. Assigned to infra as a mechanism fix.
    "C6.5": ClauseImpl(
        CLAUSE_EXTERNAL,
        where=(
            "wildfire_nowcast.common.dispersion.g3_conditions",
            "wildfire_nowcast.common.dispersion.first_moment_condition_from_blocks",
            "wildfire_nowcast.eval.baseline_run.g3_summary",
        ),
        note="the geometric bar plus the reference-defined first-moment condition. Ratified at "
        "v2.14 and NOT at v2.13, because at v2.13 the code existed in `common/dispersion.py` and "
        "NOTHING CALLED IT - the only importer was `common/pooling.py`. A clause is ratified when "
        "the GATE PATH runs it. Verified before the bump: `eval/baseline_run.py` imports "
        "`common.dispersion as g3`, calls `g3.first_moment_condition_from_blocks`, and emits "
        "`g3_conditions`. UNDEFINED is a third outcome and is never a pass.",
    ),
    # [v2.13] Classified by the MAINTAINER rather than by the author of the
    # checks - a declared ownership crossing (ADR-040). C3.5 was authored here,
    # the registry correctly went red because the clause was unclassified, and
    # nobody who owned the classification was available to fix it. The crossing
    # is one dict entry and no logic; the checks it names were written and
    # verified at A14. Recorded rather than quietly done.
    "C3.5": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("norm_stats_declares_fire_ids", "train_heldout_disjoint"),
        note="replaces a `train_heldout_disjoint` that COULD NOT FAIL - it intersected two "
        "lists `split_fingerprint` builds as a partition, so it had been green since it was "
        "written and could never be anything else. The leak it existed to catch "
        "(`2020_july_complex`, 9.76% of train mass) was found by a human reading a file. "
        "Now reads the DECLARED ids from norm_stats.json, a source the fingerprint does not "
        "itself construct. Verified by planting `2020_creek` in both lists (ok -> False), "
        "planting a missing key (ok -> False), and restoring (ok -> True).",
    ),
    "C4": ClauseImpl(
        CLAUSE_EXTERNAL,
        where=("wildfire_nowcast.common.synthetic.make_synthetic_fire",),
        note="asserted end-to-end by tests/test_synthetic.py (C1-C3 conformance, all three "
        "states, scripted barrier crossing, scripted dormancy, < 5 s budget).",
    ),
    "C5": ClauseImpl(
        CLAUSE_EXTERNAL,
        where=("wildfire_nowcast.model.api.predict", "wildfire_nowcast.model.api.load_model"),
        note="modelling owns it; signature conformance is asserted by the adopted self-tests "
        "in tests/test_adopted_selftests.py.",
    ),
    "C6": ClauseImpl(
        CLAUSE_EXTERNAL,
        where=("wildfire_nowcast.eval.metrics.evaluate",),
        note="modelling owns it; the seven required keys are asserted by the adopted self-tests.",
    ),
    "C6.0": ClauseImpl(
        CLAUSE_ENFORCED,
        where=(
            "wildfire_nowcast.common.null_check.run_null_check",
            "wildfire_nowcast.common.null_check.C6_METRICS",
            "wildfire_nowcast.common.null_check.FORECASTERS",
        ),
        note="[v2.10, A11] every C6 metric is scored against DO-NOTHING nulls and a collapse "
        "ablation, all synthesised from labels alone. `make null-check`; tests/test_null_check.py "
        "asserts it reproduces the two KNOWN pathologies (dispersion_ratio, best_member_iou) and "
        "clears the corrected criterion. This is a clause about a RELATION BETWEEN MODELS, so per "
        "ADR-016 it could never have lived in check_tensor(one_store).",
    ),
    "C6.4": ClauseImpl(
        CLAUSE_ENFORCED,
        where=(
            "wildfire_nowcast.common.iou_terms.decompose_best_member_iou",
            "wildfire_nowcast.common.iou_terms.GATE_CRITERION_KEY",
            "wildfire_nowcast.eval.metrics.evaluate",
        ),
        note="[v2.10, A11] the shape/silence split lives in common/ (C0: one implementation of "
        "anything the contract adjudicates) and eval/metrics.py wires it in additively - the "
        "REPORTED best_member_iou is unchanged. Gate criterion = GATE_CRITERION_KEY. Known-answer "
        "cases in tests/test_iou_terms.py, incl. empty-vs-empty and a model that predicts "
        "nothing; agreement with sim/replay.py's independent split is asserted, not assumed.",
    ),
    "C6.1": ClauseImpl(
        CLAUSE_EXTERNAL,
        where=("wildfire_nowcast.eval.metrics.evaluate",),
        note="area_dispersion_ratio is emitted; WHICH metric adjudicates G3 is an maintainer "
        "ruling that no artifact check can enforce.",
    ),
    "C6.2": ClauseImpl(
        CLAUSE_EXTERNAL,
        where=("wildfire_nowcast.eval.validity.baseline_validity",),
        note="the zero-ignition VOID condition is code. The v2.8 per-horizon calibration "
        "amendment is NOT yet the reported form (baseline_run calibrates at one horizon per "
        "run and records others as `alternative_horizons`) - routed to the model owner, A10.",
    ),
    "C6.3": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "heldout_block_coverage",
            # [v2.16] ADR-062 (7) - an expected false is stamped, not discovered.
            "c6_3_expected_false_did_not_flip",
            "c6_3_expected_false_declared",
        ),
        where=(
            "wildfire_nowcast.common.splits.check_split_assignment",
            "wildfire_nowcast.common.splits.stamp_c6_3_expected_false",
            "wildfire_nowcast.common.splits.folds_expected_to_fail_c6_3",
        ),
        note="[v2.16] the expected-false stamp is a SIBLING of `c6_3_satisfied` and never a "
        "substitute: `stamp_c6_3_expected_false` copies the value through unchanged and RAISES "
        "if asked to stamp a true or a missing one, and `c6_3_expected_false_did_not_flip` is a "
        "HARD FAIL on any declaration sitting beside a non-false. The expected-false fold set "
        "is DERIVED from LEAVE_FOLD_OUT_BLOCKS, which is how infra found that ADR-062 (7)'s "
        "'folds 0 and 1' is THREE folds - fold 2 holds out 3 blocks, also below the minimum "
        "of 4. Ruling unaffected; raised as a PROPOSAL, not corrected in DECISIONS.md.",
    ),
    # [v2.15] Authored by infra under the explicit I1 directive (the v2.12
    # precedent). THIRD clause-authoring crossing; A14's auto-discovery still
    # does not cover clause classification, so writing a clause still compels an
    # edit to this file. The mechanism fix remains owed.
    "C6.7": ClauseImpl(
        CLAUSE_DEFERRED,
        note="see DEFERRED_CLAUSES['C6.7']; ratified at v2.18 and deliberately not enforced "
        "yet. Classified DEFERRED and not PROCESS because an artifact CAN carry the lead "
        "profile - the check is writable and unwritten, which is a different and more "
        "fixable thing than a rule no artifact could ever hold.",
    ),
    "C6.6": ClauseImpl(
        CLAUSE_ENFORCED,
        where=(
            "wildfire_nowcast.common.null_check.C6_METRICS",
            "wildfire_nowcast.common.null_check.adjudicating_metrics",
            "wildfire_nowcast.common.null_check.assert_may_adjudicate",
            "wildfire_nowcast.common.null_check.verdicts.MetricVerdict.is_failure",
        ),
        note="Brier / arrival_crps / calibration_error / reliability are NON-ADJUDICATING "
        "(ADR-053 (1)(2)): Spearman -0.45 / -0.34 / -0.14 / -0.80 against |log(area error)| on "
        "M11's 0.053x-8.0x ladder at n=5 blocks, i.e. the WRONG SIGN, and no MDE anywhere. "
        "ENFORCED rather than external: `gate_eligible` is not documentation - "
        "`MetricVerdict.is_failure` and `is_reporting_gap` both read it, so C6.0's harness "
        "(`make null-check`, inside `make ci`) changes tier for these four at this bump. "
        "`assert_may_adjudicate` RAISES rather than warning, because the failure being "
        "repaired is someone forgetting a ruling, and a warning is read by that same someone. "
        "The permitted trio is DERIVED from the flags by `adjudicating_metrics()` and pinned "
        "in tests/test_null_check.py, so it cannot drift from the flags in either direction.",
    ),
    "C7": ClauseImpl(
        CLAUSE_ENFORCED,
        where=(
            "wildfire_nowcast.common.config.load_config",
            "wildfire_nowcast.common.runs.create_run_dir",
            "wildfire_nowcast.common.paths.repo_root",
        ),
        note="run dirs carry the resolved config + git SHA + dirty flag; tests/test_hygiene.py "
        "asserts no absolute path, no notebook import and no hardcoded GCP project in src/.",
    ),
    "C8": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=("stamped", "internally_consistent", "matches_current"),
        where=(
            "wildfire_nowcast.common.splits.split_fingerprint",
            "wildfire_nowcast.common.splits.check_run_split",
            "wildfire_nowcast.common.splits.check_split_chain",
            "wildfire_nowcast.common.runs.create_run_dir",
        ),
        note="cross-fire and cross-run: no per-tensor check can see a split move (ADR-015).",
    ),
    # [v2.16] FOURTH clause-authoring crossing (cf. C6.6 at v2.15, C6.5 at
    # ADR-044, C3.5 at ADR-040). A14's auto-discovery still does not cover clause
    # classification, so writing a clause STILL compels an edit to this file.
    # Four instances; the mechanism fix is owed and is not this task.
    "C8.1": ClauseImpl(
        CLAUSE_ENFORCED,
        checks=(
            "cv_matrix_well_formed",
            "cv_matrix_member_count",
            "cv_matrix_member_stamps",
            "cv_matrix_members_distinct",
        ),
        where=(
            "wildfire_nowcast.common.splits.CV_MATRIX_KEY",
            "wildfire_nowcast.common.splits.declared_cv_matrix",
            "wildfire_nowcast.common.splits.check_run_split",
        ),
        note="[ADR-062 (6)] a leave-fold-out matrix has FIVE split fingerprints by "
        "construction and C8 hard-fails on more than one per artifact. This is an EXTENSION of "
        "the checker, NOT an exemption from it: an artifact declaring `cv_matrix` buys its "
        "member stamps out of C8.internally_consistent and pays three hard clauses no other "
        "artifact faces - the declaration must parse, the declared member count must equal the "
        "member runs PRESENT, and every member run's own stamp must equal the matrix's claim "
        "about it. Today such an artifact cannot be checked at all, so this is strictly harder. "
        "Verified by planting all three defects in tests/test_splits.py.",
    ),
    "C8.2": ClauseImpl(
        CLAUSE_ENFORCED,
        where=(
            "wildfire_nowcast.common.splits.SplitContext",
            "wildfire_nowcast.common.splits.resolve_split_context",
            "wildfire_nowcast.common.splits.assert_fit_and_stamp_agree",
            "wildfire_nowcast.common.splits.SplitFitStampMismatchError",
        ),
        note="[ADR-062 (5)] the approved `stats_path` parameter must reach read_norm_stats, "
        "split_fingerprint AND assert_split_unchanged ATOMICALLY, because a caller that sets "
        "the FIT from one fold-stats file while the STAMPS come from the default recreates the "
        "leak the parameter was approved to avoid. ENFORCED BY SHAPE, not by a report check: "
        "SplitContext resolves the path ONCE and its three operations take no path argument at "
        "all, so the desynchronised call cannot be written; resolve_split_context additionally "
        "runs assert_fit_and_stamp_agree, so an inconsistent context cannot be constructed. No "
        "`checks=` because there is no artifact to inspect - the clause is about a call shape, "
        "and tests/test_splits.py plants the desynchronised caller and shows it cannot pass.",
    ),
}


#: Channel order is fixed; the index of a name in this tuple IS its v1 channel
#: index. v2 moved channel 0 into its own variable but renumbered nothing.
CHANNELS: tuple[str, ...] = (
    "fire_state",  # 0  {0,1,2}   -> separate uint8 variable (v2)
    "wind_u10",  # 1  m/s   RTMA
    "wind_v10",  # 2  m/s   RTMA
    "temp_2m",  # 3  K     RTMA
    "rh_2m",  # 4  %     RTMA
    "elevation",  # 5  m     3DEP, static
    "slope",  # 6  deg   static
    "aspect_sin",  # 7  -     static
    "aspect_cos",  # 8  -     static
    "fuel_model_id",  # 9  FBFM40 class as float (USGS LFPS, ADR-005)
    "canopy_cover",  # 10 %     static (USGS LFPS)
    "fuel_moisture_proxy",  # 11 -     derived from RTMA
    "water_barrier_mask",  # 12 {0,1} static
    "recent_burn_scar",  # 13 {0,1} static
)

N_CHANNELS = len(CHANNELS)
CHANNEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(CHANNELS)}

#: [v2] The 13 channels carried by the `features` array, in v1 index order.
FEATURE_CHANNELS: tuple[str, ...] = CHANNELS[1:]
N_FEATURE_CHANNELS = len(FEATURE_CHANNELS)

#: [v2] position along `features`' channel axis + this == the v1 channel index.
CHANNEL_INDEX_OFFSET = 1

CHANNEL_UNITS: dict[str, str] = {
    "fire_state": "1",
    "wind_u10": "m s-1",
    "wind_v10": "m s-1",
    "temp_2m": "K",
    "rh_2m": "%",
    "elevation": "m",
    "slope": "degree",
    "aspect_sin": "1",
    "aspect_cos": "1",
    "fuel_model_id": "1",
    "canopy_cover": "%",
    "fuel_moisture_proxy": "%",
    "water_barrier_mask": "1",
    "recent_burn_scar": "1",
}

#: Channels C1 marks "static" / "static per fire". They still carry a leading
#: time axis ("repeated over time") so the stacked view is uniform.
STATIC_CHANNELS = frozenset(
    {
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "fuel_model_id",
        "canopy_cover",
        "water_barrier_mask",
        "recent_burn_scar",
    }
)

#: [v2.2] C3.2 - class labels, not quantities. Identity transform only.
CATEGORICAL_CHANNELS = frozenset({"fire_state", "fuel_model_id"})

#: C1 declares these as ``{0,1}`` masks.
BINARY_CHANNELS = frozenset({"water_barrier_mask", "recent_burn_scar"})

#: C1 declares channel 9 as "int-as-float, FBFM40 class".
INTEGER_CHANNELS = frozenset({"fuel_model_id"})

#: [v2.4] C1.7 - the FBFM40 enumeration (Scott & Burgan 2005, LANDFIRE codes).
#: 40 burnable models in six groups, plus the five NB (non-burnable) codes.
#: This is the WHOLE legal domain of channel 9: an FBFM40 raster cannot hold a
#: value outside it, so anything else is a sentinel, a fill, or a resampling
#: artefact - never data.
FBFM40_NONBURNABLE: frozenset[int] = frozenset(
    {
        91,  # NB1 Urban / developed
        92,  # NB2 Snow / ice
        93,  # NB3 Agricultural
        98,  # NB8 Open water
        99,  # NB9 Bare ground
    }
)
FBFM40_OPEN_WATER = 98  #: ADR-010's fill policy of record for off-coast NoData.

FBFM40_CLASSES: frozenset[int] = (
    FBFM40_NONBURNABLE
    | frozenset(range(101, 110))  # GR1-GR9   grass
    | frozenset(range(121, 125))  # GS1-GS4   grass-shrub
    | frozenset(range(141, 150))  # SH1-SH9   shrub
    | frozenset(range(161, 166))  # TU1-TU5   timber-understory
    | frozenset(range(181, 190))  # TL1-TL9   timber litter
    | frozenset(range(201, 205))  # SB1-SB4   slash-blowdown
)

#: [v2.4] C1.7 - ``channel -> (low, high)``, INCLUSIVE, both definitional.
#:
#: Deliberately SHORT. Every entry here is a range outside which no legitimate
#: value exists, so a violation is always a bug and a hard fail carries no
#: false-positive mode (this is the distinction from C1.6's heuristic, ADR-010).
#: Ranges that are merely *implausible* - a 500 m/s wind, a 900% humidity - are
#: NOT here: they are proposed in docs/decisions.md and stay out
#: until an ADR ratifies them. Adding an unratified clause here is the same
#: mistake as tolerating a sentinel, pointed the other way.
PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "canopy_cover": (0.0, 100.0),  # it is a percentage
}

FIRE_STATE = "fire_state"
FEATURES = "features"
UNBURNED, BURNING, BURNED_OUT = 0, 1, 2
FIRE_STATE_VALUES = (UNBURNED, BURNING, BURNED_OUT)

CRS_EPSG = 5070
CRS_STRING = f"EPSG:{CRS_EPSG}"
CELL_SIZE_M = 1000.0

#: [v2] C1.2 - the single continental lattice. Cell EDGES fall on integer
#: multiples of the cell size measured from this origin, so cell (i, j) denotes
#: the same ground in every fire and two buffered domains can be compared
#: cell-for-cell (which is also what makes the C3.1 spatial blocking meaningful).
LATTICE_ORIGIN_M = (0.0, 0.0)

#: [v2] C1.2 - "final-perimeter bbox buffered 10 km" in CELLS. One-sided: the
#: burned footprint must be at least this far inside the domain edge.
#:
#: This is the definitional half of C1.2 and the only half with a failure mode
#: worth a hard clause: a smaller margin means the domain was NOT built from the
#: final perimeter (or the fire was clipped), so spread ran off-grid and the
#: tensor silently truncates a fire. A LARGER margin is merely generous - the C4
#: synthetic legitimately uses a fixed 128x128 lattice domain - so the upper
#: side is not enforced. Fitting sample (C-3): all 12 built fires, 11 spatial
#: blocks; observed 10-13 cells, minimum exactly 10.
MIN_BUFFER_MARGIN_CELLS = 10

#: [v2] C1.3 - GOFER ``tUTC`` is end-of-hour; RTMA is lagged 1 h to match.
TIME_CONVENTION = "end_of_hour"

#: The stacked, model-facing view. On disk this is two variables (v2).
TENSOR_DIMS = ("time", "channel", "y", "x")
FEATURES_DIMS = ("time", "channel", "y", "x")
FIRE_STATE_DIMS = ("time", "y", "x")
VAR_DIMS = FIRE_STATE_DIMS  # kept: the per-channel dims of a 3-D field

#: [v2] Exactly these two data variables live in a C1 store.
DATA_VARS = (FIRE_STATE, FEATURES)

#: Non-channel variables tolerated in the store (CF grid-mapping carriers).
ALLOWED_EXTRA_VARS = frozenset({"spatial_ref", "crs"})

ATTR_CRS = "crs"
ATTR_CELL_SIZE = "cell_size_m"
ATTR_TIME_START = "time_start_utc"
ATTR_TIME_END = "time_end_utc"
ATTR_TIME_CONVENTION = "time_convention"
ATTR_CHANNEL_INDEX_OFFSET = "channel_index_offset"
ATTR_CHANNEL_ORDER = "channel_order"

# C2
#: [v2.7] ADR-014 says these live in ``provenance``; INTERFACES C2 lists them
#: under "Keys". The contract is genuinely ambiguous about LOCATION, so the
#: checker accepts either and enforces the INVARIANT - an int, machine-readable,
#: present. Legislating a location the contract does not fix would be inventing
#: a clause; see the PROPOSAL in docs/decisions.md (A10).
MANIFEST_VINTAGE_LAG_KEY = "fuel_vintage_lag_years"
MANIFEST_IGNITION_COMPONENTS_KEY = "n_ignition_components"

MANIFEST_KEYS: tuple[str, ...] = (
    "fire_id",
    "gofer_version",
    "bbox_5070",
    "ignition_time_utc",
    "n_hours",
    "cv_fold",
    "spatial_block_id",  # [v2] C3.1 - folds are blocked, not per-fire
    "created_utc",
    "provenance",
    "norm_stats_path",
)

# C3
NORM_STATS_KEYS: tuple[str, ...] = (
    "channel_order",
    "mean",
    "std",
    "train_folds",
    "n_train_blocks",  # [v2.2] C3.3 bootstrap guard
    "created_utc",
)

#: [v2.2] C3.2 - the nested per-channel block data emitted before the file
#: shape was ratified. Canonical shape is TOP-LEVEL dicts; this is dropped at A5.
NORM_STATS_LEGACY_BLOCK = "channels"
NORM_STATS_CATEGORICAL_NOTE = "categorical_identity_note"

#: C3.3 - a norm-stats file is reporting-grade only at or above this many blocks.
MIN_TRAIN_BLOCKS_FOR_REPORTING = 2

SEVERITY_FAIL = "fail"
SEVERITY_REPORTING = "reporting"

_COORD_TOL_M = 1e-6
_ONE_HOUR = np.timedelta64(1, "h")


def channel_dtype(name: str) -> np.dtype:
    """dtype required by C1 for ``name``: uint8 for fire_state, else float32."""
    if name not in CHANNEL_INDEX:
        raise KeyError(f"{name!r} is not a C1 channel; expected one of {CHANNELS}")
    return np.dtype(np.uint8) if name == FIRE_STATE else np.dtype(np.float32)


def feature_index(name: str) -> int:
    """Position of ``name`` along ``features``' channel axis (v1 index - 1)."""
    if name not in CHANNEL_INDEX:
        raise KeyError(f"{name!r} is not a C1 channel; expected one of {CHANNELS}")
    if name == FIRE_STATE:
        raise KeyError(
            "fire_state is a separate uint8 variable (C1 v2); it has no position in `features`"
        )
    return CHANNEL_INDEX[name] - CHANNEL_INDEX_OFFSET


def is_on_lattice(
    coords: np.ndarray,
    *,
    cell_size_m: float = CELL_SIZE_M,
    origin_m: float = 0.0,
) -> bool:
    """True when cell-CENTRE ``coords`` sit on the continental lattice (C1.2).

    Centres are offset half a cell from the edges, so the test is on
    ``coord + cell/2`` being an integer number of cells from the origin.
    """
    values = np.asarray(coords, dtype=np.float64)
    offset = (values + cell_size_m / 2.0 - origin_m) / cell_size_m
    return bool(np.all(np.abs(offset - np.round(offset)) * cell_size_m <= 1e-3))


def fire_state_violations(state: np.ndarray) -> list[str]:
    """The C1.1 guarantees, as a list of violated-clause messages (empty = ok).

    One implementation, used by the checker *and* by
    :mod:`wildfire_nowcast.common.states` when it validates its own output.
    """
    arr = np.asarray(state)
    out: list[str] = []
    if arr.size == 0:
        return out
    bad = np.setdiff1d(np.unique(arr), np.asarray(FIRE_STATE_VALUES))
    if bad.size:
        out.append(f"values outside {FIRE_STATE_VALUES}: {bad.tolist()}")
    if arr.ndim == 3 and arr.shape[0] >= 2:
        if not bool(np.all(arr[1:] >= arr[:-1])):
            out.append("fire_state decreases in time; fire must be absorbing (0 -> 1 -> 2 only)")
        skips = int(np.count_nonzero((arr[:-1] == UNBURNED) & (arr[1:] == BURNED_OUT)))
        if skips:
            out.append(f"{skips} cell-steps skip 0 -> 2 without passing through 1 (C1.1)")
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


class ContractViolation(AssertionError):
    """Raised by :meth:`ContractReport.raise_for_status` when checks fail."""


def _verdict(ok: object) -> bool:
    """Coerce a check outcome to a strict boolean. **Non-finite is False.**

    Project policy (ADR-012): *a diagnostic that fails to ``ok`` is worse than
    no diagnostic.* Any verdict ladder must treat a non-finite or unusable
    outcome as FAIL, never as pass-by-default.

    This is the one place it can be enforced for the whole checker, because
    every clause becomes a verdict here and nowhere else. It is needed because
    Python's own truthiness disagrees::

        bool(float("nan"))      -> True     # a NaN would report PASS
        bool(np.float64("nan")) -> True
        float("nan") <= 1e-3    -> False    # ...but every comparison is False

    That pair is the whole defect: a statistic that goes NaN makes every
    ``<``/``<=`` guard False, so an ``if/elif`` ladder hands it to the trailing
    ``else``, and if the outcome is then passed on as a raw float it is *also*
    truthy. sim hit exactly this - ``cos: +nan [ok]`` on 2 of 5 fires,
    hiding a good CZU result and passing a weak Zogg one.

    Accepted: real booleans (and ``numpy.bool_``) pass through unchanged, so a
    correctly-written clause is unaffected. Rejected as ``False``: ``None``,
    NaN, +/-inf, and anything that cannot be evaluated. An array is True only if
    every element is - ``np.all`` of an empty array is vacuously True, so an
    empty comparison is treated as unverifiable and fails.
    """
    if isinstance(ok, bool | np.bool_):
        return bool(ok)
    if ok is None:
        return False
    if isinstance(ok, np.ndarray):
        if ok.size == 0:
            return False  # nothing was compared; vacuous truth is not evidence
        if np.issubdtype(ok.dtype, np.number) and not bool(np.all(np.isfinite(ok))):
            return False
        return bool(np.all(ok))
    if isinstance(ok, int | float | np.number):
        value = float(ok)
        return bool(np.isfinite(value)) and value != 0.0
    return bool(ok)


def _as_float(value: object) -> float | None:
    """``float(value)`` or ``None`` - never raises, never returns a NaN.

    A malformed attribute must fail *its own* clause. If it raises instead, the
    checker aborts and every clause after it goes unreported, which turns one
    bad attribute into an invisible tensor (the punch-list property in the
    module docstring).
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


@dataclass(frozen=True)
class Check:
    contract: str
    check_id: str
    ok: bool
    message: str
    severity: str = SEVERITY_FAIL

    def __str__(self) -> str:
        if self.ok:
            tag = "PASS"
        else:
            tag = "FAIL" if self.severity == SEVERITY_FAIL else "REPORTING"
        return f"[{tag}] {self.contract}.{self.check_id}: {self.message}"


@dataclass
class ContractReport:
    """Result of running a set of contract checks against one target."""

    target: str
    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        contract: str,
        check_id: str,
        ok: object,
        message: str,
        *,
        severity: str = SEVERITY_FAIL,
    ) -> bool:
        """Record one clause. ``ok`` passes through :func:`_verdict`, so a
        non-finite or unverifiable outcome is a FAIL and never a silent PASS."""
        verdict = _verdict(ok)
        self.checks.append(Check(contract, check_id, verdict, message, severity))
        return verdict

    def extend(self, other: ContractReport) -> None:
        self.checks.extend(other.checks)

    @property
    def ok(self) -> bool:
        """Conformant for *use*: every hard clause passes."""
        return not self.failures

    @property
    def reporting_ok(self) -> bool:
        """Conformant for a number that appears in a gate: every clause passes."""
        return not self.failures and not self.reporting_gaps

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == SEVERITY_FAIL]

    @property
    def reporting_gaps(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == SEVERITY_REPORTING]

    def format(self, verbose: bool = False, *, for_reporting: bool = False) -> str:
        shown = self.checks if verbose else self.failures
        head = f"contract check [enforcing INTERFACES {CONTRACT_VERSION}]: {self.target}"
        parts = [head]
        if shown:
            parts.append("\n".join(f"  {c}" for c in shown))
        gaps = self.reporting_gaps
        if gaps and not verbose:
            parts.append("\n".join(f"  {c}" for c in gaps))
        if gaps:
            parts.append(
                f"  REPORTING GATE: {len(gaps)} clause(s) above are contract-mandatory for any\n"
                f"  number that appears in a gate. Plumbing-valid, NOT reporting-valid.\n"
                f"  Enforced as hard failures by: make contract-reporting TENSOR={self.target}"
            )
        for clause, why in sorted(DEFERRED_CLAUSES.items()):
            parts.append(f"  NOT ENFORCED — {clause}: {why}")
        if self.failures:
            parts.append(f"  {len(self.failures)}/{len(self.checks)} checks FAILED")
        elif gaps:
            parts.append(
                f"  {'FAILED (reporting mode)' if for_reporting else 'OK for use'} — "
                f"{len(self.checks) - len(gaps)}/{len(self.checks)} checks passed, "
                f"{len(gaps)} reporting gap(s)"
            )
        else:
            parts.append(f"  OK — {len(self.checks)} checks passed (reporting-ready)")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "enforces_interfaces_version": CONTRACT_VERSION,
            "deferred_clauses": dict(DEFERRED_CLAUSES),
            "ok": self.ok,
            "reporting_ready": self.reporting_ok,
            "n_checks": len(self.checks),
            "n_failures": len(self.failures),
            "n_reporting_gaps": len(self.reporting_gaps),
            "reporting_gaps": [c.check_id for c in self.reporting_gaps],
            "checks": [
                {
                    "contract": c.contract,
                    "id": c.check_id,
                    "ok": c.ok,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }

    def raise_for_status(self, *, for_reporting: bool = False) -> None:
        good = self.reporting_ok if for_reporting else self.ok
        if not good:
            raise ContractViolation(self.format(verbose=False, for_reporting=for_reporting))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def open_tensor_dataset(path: str | Path) -> xr.Dataset:
    """Open a tensor store read-only, decoding time to datetime64.

    Works with or without consolidated zarr metadata so a store written by any
    lead, with any writer, opens the same way.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no tensor store at {p}")
    try:
        return cast("xr.Dataset", xr.open_zarr(p, consolidated=True, decode_timedelta=False))
    except Exception:
        return cast("xr.Dataset", xr.open_zarr(p, consolidated=False, decode_timedelta=False))


def _is_naive_iso(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value.endswith("Z") or "+" in value:
        return False
    try:
        np.datetime64(value)
    except Exception:
        return False
    return True


def _as_seconds(value: np.datetime64) -> np.datetime64:
    """Truncate a datetime64 to second resolution, keeping its static type.

    NOT ``np.datetime64(value, "s")``, and the reason is a seven-day CI outage.
    That spelling needs ``# type: ignore[call-overload, no-any-return]`` under
    numpy 2.5.1 and is an UNUSED ignore under numpy 2.5.2, which added the
    ``(datetime64, unit)`` overload to its stubs. With `strict` (so
    `warn_unused_ignores`) the same source line was therefore green on a machine
    holding 2.5.1 and red in a CI that resolved 2.5.2 -- opposite verdicts, no
    diff between them. Suppressing it either way just picks a numpy to be right
    on; `requirements.lock` now binds the resolution so the two cannot diverge
    again, and this spelling needs no suppression under EITHER.

    ``.astype(np.dtype("datetime64[s]"))`` is typed by both stub versions:
    ``dtype("datetime64[s]")`` resolves to ``dtype[datetime64[datetime]]`` and
    the ``astype(dtype[_SCT]) -> _SCT`` overload carries that through, so the
    result is ``datetime64[datetime]`` rather than ``Any``. A bare string dtype
    (``.astype("datetime64[s]")``) returns ``Any`` and would trade the ignore for
    a silent hole. Measured identical to the old call -- 53 values across ten
    units, NaT, pre-epoch and the ns bounds, equal in repr, dtype and bytes,
    against a negative control that separates.
    """
    return value.astype(np.dtype("datetime64[s]"))


def _epsg_of(value: object) -> int | None:
    """Best-effort EPSG resolution; falls back to string match without pyproj."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    try:
        from pyproj import CRS as _CRS

        return _CRS.from_user_input(value).to_epsg()
    except Exception:
        stripped = value.strip().upper().replace(" ", "")
        if stripped == CRS_STRING.upper():
            return CRS_EPSG
        return None


def _as_str_list(value: object) -> list[str] | None:
    """Coerce a list-or-JSON-string attribute to a list of str, else None.

    zarr attrs are JSON, but leads have written channel lists both as real
    arrays and as ``json.dumps`` strings. Both are readable; neither is worth a
    contract failure on its own.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [str(v) for v in value]
    return None


def _attr(ds: xr.Dataset, name: str, *, var: str | None = None) -> Any:
    """Look an attribute up on the dataset root, then on a variable.

    C1 says the *store* records these; it does not legislate root-vs-variable,
    and data reasonably hung ``channel_index_offset`` on ``features``.
    """
    if name in ds.attrs:
        return ds.attrs[name]
    if var and var in ds.variables and name in ds[var].attrs:
        return ds[var].attrs[name]
    return None


def _feature_slice(ds: xr.Dataset, name: str) -> np.ndarray:
    """``(time, y, x)`` values of one feature channel, read one channel at a time."""
    return np.asarray(ds[FEATURES].isel(channel=feature_index(name)).values)


def tensor_bounds(ds: xr.Dataset) -> tuple[float, float, float, float]:
    """Outer edge bounds ``(xmin, ymin, xmax, ymax)`` of a tensor's grid."""
    half = CELL_SIZE_M / 2.0
    x = np.asarray(ds["x"].values, dtype=np.float64)
    y = np.asarray(ds["y"].values, dtype=np.float64)
    return (float(x[0] - half), float(y[-1] - half), float(x[-1] + half), float(y[0] + half))


# --------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------


def check_tensor(
    target: str | Path | xr.Dataset,
    *,
    required_channels: Sequence[str] = CHANNELS,
    require_channel_coord: bool = True,
    check_absorbing: bool = True,
    manifest_path: str | Path | None = None,
) -> ContractReport:
    """Assert C1 against any tensor store.

    Parameters
    ----------
    target
        Path to a ``.zarr`` store, or an already-open :class:`xarray.Dataset`.
    required_channels
        Channels that must be present. Defaults to all 14 (full C1). Pass
        ``["fire_state"]`` to check a labels-only interim store (ADR-003) - the
        grid, time and state checks are identical, only completeness is relaxed.
    require_channel_coord
        Require the on-disk ``channel`` coordinate that records C1 order.
    check_absorbing
        Assert the C1.1 state guarantees (absorbing, no 0 -> 2 skip).
    manifest_path
        Sibling C2 manifest, consulted only as a fallback for C1.3's
        ``time_convention``. Defaults to ``manifest.json`` next to the store.
    """
    label = target if isinstance(target, str | Path) else "<xr.Dataset>"
    rep = ContractReport(target=str(label))

    if isinstance(target, xr.Dataset):
        ds = target
    else:
        try:
            ds = open_tensor_dataset(target)
        except Exception as exc:  # fatal: nothing else is checkable
            rep.add("C1", "open", False, f"cannot open store: {type(exc).__name__}: {exc}")
            return rep
        if manifest_path is None:
            manifest_path = Path(target).parent / "manifest.json"
    rep.add("C1", "open", True, "store opens as an xarray Dataset")

    require_features = any(c != FIRE_STATE for c in required_channels)
    _check_variables(rep, ds, required_channels, require_features, require_channel_coord)
    _check_dtypes_and_dims(rep, ds, required_channels, require_features)
    _check_crs(rep, ds)
    _check_spatial_coords(rep, ds)
    _check_time(rep, ds, manifest_path)
    _check_fire_state_values(rep, ds, required_channels, check_absorbing)
    _check_buffer_margin(rep, ds, required_channels)
    if require_features and FEATURES in ds.data_vars:
        _check_feature_domains(rep, ds)
    return rep


def _check_buffer_margin(
    rep: ContractReport, ds: xr.Dataset, required_channels: Sequence[str]
) -> None:
    """C1.2 - the domain is the final-perimeter bbox **buffered 10 km**.

    Ratified at v2 and never implemented until A10: the checker asserted the
    lattice snap and the cell size, which are the *other* two sentences of
    C1.2, and nothing looked at the buffer at all. A domain built without it
    (or a tensor cropped after the fact) clips the fire at the edge, and every
    other clause passes: the grid is snapped, the states are absorbing, the
    features are finite, and the fire simply stops growing when it reaches the
    boundary. That reads as a model failure, not a data failure.

    One-sided by design (see :data:`MIN_BUFFER_MARGIN_CELLS`). The statistic is
    the gap in cells between the FINAL burned footprint and each domain edge;
    on the 12 built fires it is 10-13 cells, minimum exactly 10 on half of them.
    """
    if FIRE_STATE not in required_channels or FIRE_STATE not in ds.data_vars:
        return
    arr = np.asarray(ds[FIRE_STATE].values)
    if arr.ndim != 3 or arr.size == 0:
        return
    burned = arr[-1] > UNBURNED
    if not bool(np.any(burned)):
        rep.add(
            "C1",
            "buffer_margin",
            False,
            "no cell is burned in the final frame, so C1.2's 10 km buffer around the FINAL "
            "PERIMETER cannot be evaluated. C-1: unverifiable is a failure, not a pass",
        )
        return
    rows = np.flatnonzero(np.any(burned, axis=1))
    cols = np.flatnonzero(np.any(burned, axis=0))
    ny, nx = burned.shape
    margins = {
        "north": int(rows[0]),
        "south": int(ny - 1 - rows[-1]),
        "west": int(cols[0]),
        "east": int(nx - 1 - cols[-1]),
    }
    tight = {k: v for k, v in margins.items() if v < MIN_BUFFER_MARGIN_CELLS}
    rep.add(
        "C1",
        "buffer_margin",
        not tight,
        f"final burned footprint sits >= {MIN_BUFFER_MARGIN_CELLS} cells inside every edge "
        f"(C1.2 10 km buffer; observed N/S/W/E = "
        f"{margins['north']}/{margins['south']}/{margins['west']}/{margins['east']})"
        if not tight
        else (
            "C1.2: the final burned footprint comes within "
            + ", ".join(f"{v} cells of the {k} edge" for k, v in sorted(tight.items()))
            + f" — the domain must be the final-perimeter bbox buffered 10 km, i.e. >= "
            f"{MIN_BUFFER_MARGIN_CELLS} cells of context on every side. A clipped domain "
            "passes every other clause and presents as a fire that mysteriously stops "
            "growing at the boundary"
        ),
    )


def _check_variables(
    rep: ContractReport,
    ds: xr.Dataset,
    required_channels: Sequence[str],
    require_features: bool,
    require_channel_coord: bool,
) -> None:
    present = set(ds.data_vars)
    expected = {FIRE_STATE} | ({FEATURES} if require_features else set())
    missing = sorted(expected - present)
    rep.add(
        "C1",
        "variables_present",
        not missing,
        f"store carries {sorted(expected)} (v2 two-variable layout)"
        if not missing
        else f"missing data variables: {missing}",
    )

    # xarray keys are `Hashable`; every store this checker admits uses `str`.
    unexpected = sorted(cast("set[str]", present - set(DATA_VARS) - ALLOWED_EXTRA_VARS))
    hint = ""
    if any(v in CHANNEL_INDEX for v in unexpected):
        hint = (
            " — this looks like the v1 one-variable-per-channel layout; v2 stacks channels "
            "1-13 into a single `features` array (ADR-006 P2)"
        )
    rep.add(
        "C1",
        "no_unknown_variables",
        not unexpected,
        "no data variables outside the C1 v2 layout"
        if not unexpected
        else f"unexpected data variables: {unexpected}{hint}",
    )

    if not require_features:
        return

    if require_channel_coord:
        if "channel" not in ds.coords and "channel" not in ds.variables:
            rep.add(
                "C1",
                "channel_coord",
                False,
                "missing 'channel' coordinate carrying the 13 feature channel NAMES",
            )
        else:
            order = [str(c) for c in np.atleast_1d(ds["channel"].values)]
            ok = order == list(FEATURE_CHANNELS)
            detail = f"channel coord order/content wrong: {order}"
            if order == list(CHANNELS):
                detail = (
                    "channel coord lists all 14 names including fire_state; v2 `features` "
                    "carries channels 1-13 only (fire_state is its own uint8 variable)"
                )
            rep.add(
                "C1",
                "channel_coord",
                ok,
                f"channel coord == C1 channels 1-13 ({N_FEATURE_CHANNELS} names)" if ok else detail,
            )

    raw_offset = _attr(ds, ATTR_CHANNEL_INDEX_OFFSET, var=FEATURES)
    ok = isinstance(raw_offset, int | float | np.integer) and (
        int(raw_offset) == CHANNEL_INDEX_OFFSET
    )
    rep.add(
        "C1",
        "channel_index_offset",
        ok,
        f"attrs[{ATTR_CHANNEL_INDEX_OFFSET!r}] == {CHANNEL_INDEX_OFFSET} "
        "(features position + offset == v1 channel index)"
        if ok
        else f"attrs[{ATTR_CHANNEL_INDEX_OFFSET!r}]={raw_offset!r}; C1 v2 requires "
        f"{CHANNEL_INDEX_OFFSET} on the store root or on `features`",
    )

    declared = _as_str_list(ds.attrs.get(ATTR_CHANNEL_ORDER))
    if declared is not None:
        ok = declared in (list(CHANNELS), list(FEATURE_CHANNELS))
        rep.add(
            "C1",
            "channel_order_attr",
            ok,
            f"attrs[{ATTR_CHANNEL_ORDER!r}] agrees with the C1 order"
            if ok
            else f"attrs[{ATTR_CHANNEL_ORDER!r}] disagrees with C1: {declared}",
        )


def _check_dtypes_and_dims(
    rep: ContractReport,
    ds: xr.Dataset,
    required_channels: Sequence[str],
    require_features: bool,
) -> None:
    bad_dtype: list[str] = []
    bad_dims: list[str] = []

    if FIRE_STATE in ds.data_vars:
        var = ds[FIRE_STATE]
        if var.dtype != np.uint8:
            bad_dtype.append(f"{FIRE_STATE}: {var.dtype} != uint8")
        if tuple(var.dims) != FIRE_STATE_DIMS:
            bad_dims.append(f"{FIRE_STATE}: {tuple(var.dims)} != {FIRE_STATE_DIMS}")
    if require_features and FEATURES in ds.data_vars:
        var = ds[FEATURES]
        if var.dtype != np.float32:
            bad_dtype.append(f"{FEATURES}: {var.dtype} != float32")
        if tuple(var.dims) != FEATURES_DIMS:
            bad_dims.append(f"{FEATURES}: {tuple(var.dims)} != {FEATURES_DIMS}")

    rep.add(
        "C1",
        "dtypes",
        not bad_dtype,
        "fire_state uint8, features float32 (one dtype per array — the v1 "
        "'float32 except fire_state' single array was unsatisfiable)"
        if not bad_dtype
        else "; ".join(bad_dtype),
    )
    rep.add(
        "C1",
        "dims",
        not bad_dims,
        f"fire_state {FIRE_STATE_DIMS}, features {FEATURES_DIMS}"
        if not bad_dims
        else "; ".join(bad_dims),
    )

    if not (require_features and FEATURES in ds.data_vars and FIRE_STATE in ds.data_vars):
        return
    n_ch = int(ds.sizes.get("channel", -1))
    rep.add(
        "C1",
        "n_channels",
        n_ch == N_FEATURE_CHANNELS,
        f"features carries all {N_FEATURE_CHANNELS} feature channels"
        if n_ch == N_FEATURE_CHANNELS
        else f"features has {n_ch} channels, expected {N_FEATURE_CHANNELS} (C1 channels 1-13)",
    )
    fs, ft = ds[FIRE_STATE].shape, ds[FEATURES].shape
    ok = len(ft) == 4 and len(fs) == 3 and (ft[0], ft[2], ft[3]) == fs
    rep.add(
        "C1",
        "shape_consistency",
        ok,
        f"fire_state {fs} and features {ft} share (time, y, x)"
        if ok
        else f"fire_state {fs} and features {ft} disagree on (time, y, x)",
    )


def _check_crs(rep: ContractReport, ds: xr.Dataset) -> None:
    raw = ds.attrs.get(ATTR_CRS)
    if raw is None and "spatial_ref" in ds.variables:
        raw = ds["spatial_ref"].attrs.get("spatial_ref") or ds["spatial_ref"].attrs.get("crs_wkt")
    epsg = _epsg_of(raw)
    rep.add(
        "C1",
        "crs",
        epsg == CRS_EPSG,
        f"CRS is {CRS_STRING}"
        if epsg == CRS_EPSG
        else f"attrs[{ATTR_CRS!r}]={raw!r} resolves to EPSG:{epsg}, expected {CRS_EPSG}",
    )
    raw_cell = ds.attrs.get(ATTR_CELL_SIZE)
    cell = _as_float(raw_cell)
    ok = cell is not None and abs(cell - CELL_SIZE_M) <= _COORD_TOL_M
    rep.add(
        "C1",
        "cell_size_attr",
        ok,
        f"attrs[{ATTR_CELL_SIZE!r}] == {CELL_SIZE_M} (C1.2 canonical name)"
        if ok
        else f"attrs[{ATTR_CELL_SIZE!r}]={raw_cell!r}, expected {CELL_SIZE_M} "
        "(must be a finite number)",
    )


def _check_axis(
    rep: ContractReport, ds: xr.Dataset, axis: str, expected_step: float, direction: str
) -> None:
    if axis not in ds.coords:
        rep.add("C1", f"{axis}_coord", False, f"missing '{axis}' coordinate")
        return
    vals = np.asarray(ds[axis].values)
    if vals.ndim != 1 or vals.size < 2:
        rep.add(
            "C1",
            f"{axis}_coord",
            False,
            f"'{axis}' must be 1-D with >= 2 cells, got shape {vals.shape}",
        )
        return
    if not np.issubdtype(vals.dtype, np.floating):
        rep.add("C1", f"{axis}_coord", False, f"'{axis}' dtype {vals.dtype} is not floating")
        return
    diffs = np.diff(vals.astype(np.float64))
    ok = bool(np.all(np.abs(diffs - expected_step) <= _COORD_TOL_M))
    rep.add(
        "C1",
        f"{axis}_coord",
        ok,
        f"'{axis}' is {direction} with uniform {abs(expected_step):.0f} m cells"
        if ok
        else (
            f"'{axis}' spacing must be exactly {expected_step:+.0f} m ({direction}); "
            f"observed min={diffs.min():.6g} max={diffs.max():.6g}"
        ),
    )


def _check_spatial_coords(rep: ContractReport, ds: xr.Dataset) -> None:
    # North-up raster convention: x ascends east, y descends south.
    _check_axis(rep, ds, "x", CELL_SIZE_M, "ascending (east-positive)")
    _check_axis(rep, ds, "y", -CELL_SIZE_M, "descending (north-up)")

    # C1.2 continental lattice: cell (i, j) must mean the same ground in every
    # fire, or overlapping buffered domains cannot be compared cell-for-cell and
    # the C3.1 spatial blocking loses its meaning.
    off = []
    for axis, origin in (("x", LATTICE_ORIGIN_M[0]), ("y", LATTICE_ORIGIN_M[1])):
        if axis not in ds.coords:
            continue
        if not is_on_lattice(ds[axis].values, origin_m=origin):
            residual = (np.asarray(ds[axis].values, dtype=np.float64)[0] + CELL_SIZE_M / 2.0) % (
                CELL_SIZE_M
            )
            off.append(f"{axis} (edge offset {residual:.3f} m from the lattice)")
    rep.add(
        "C1",
        "lattice_snap",
        not off,
        "grid is snapped to the continental EPSG:5070 1 km lattice (C1.2)"
        if not off
        else f"grid is NOT on the continental lattice: {', '.join(off)}; "
        "cell (i,j) would denote different ground in different fires",
    )


def _check_time(rep: ContractReport, ds: xr.Dataset, manifest_path: str | Path | None) -> None:
    if "time" not in ds.coords:
        rep.add("C1", "time_coord", False, "missing 'time' coordinate")
        return
    t = np.asarray(ds["time"].values)
    if not np.issubdtype(t.dtype, np.datetime64):
        rep.add(
            "C1",
            "time_coord",
            False,
            f"'time' dtype {t.dtype} is not datetime64 (must decode to naive UTC datetimes)",
        )
        return
    rep.add("C1", "time_coord", True, f"'time' is datetime64 (naive UTC), {t.size} steps")

    if t.size >= 2:
        diffs = np.diff(t)
        hourly = bool(np.all(diffs == _ONE_HOUR))
        monotone = bool(np.all(diffs > np.timedelta64(0, "s")))
        rep.add(
            "C1",
            "time_monotone",
            monotone,
            "time strictly increasing" if monotone else "time is not strictly increasing",
        )
        rep.add(
            "C1",
            "time_hourly",
            hourly,
            "every step is exactly 1 h"
            if hourly
            else f"non-hourly steps present: unique diffs {np.unique(diffs)}",
        )
    else:
        rep.add("C1", "time_monotone", True, "single time step (trivially monotone)")
        rep.add("C1", "time_hourly", True, "single time step (no step to check)")

    for attr, expected in ((ATTR_TIME_START, t[0]), (ATTR_TIME_END, t[-1])):
        raw = ds.attrs.get(attr)
        ok = _is_naive_iso(raw) and np.datetime64(raw) == _as_seconds(expected)
        rep.add(
            "C1",
            f"attr_{attr}",
            ok,
            f"attrs[{attr!r}] is a naive-UTC ISO string matching the time coord"
            if ok
            else (
                f"attrs[{attr!r}]={raw!r}; expected naive ISO (no 'Z'/offset) "
                f"== {_as_seconds(expected)}"
            ),
        )

    _check_time_convention(rep, ds, manifest_path)


def _check_time_convention(
    rep: ContractReport, ds: xr.Dataset, manifest_path: str | Path | None
) -> None:
    """C1.3 - the store records ``time_convention: "end_of_hour"``.

    Severity note. A *wrong* or *absent-everywhere* convention is a hard failure:
    it is the silently-catastrophic case, where the whole fire trains an hour out
    of phase with its weather and presents as a mediocre model rather than a bug.
    A convention that is correct and machine-readable in the sibling C2 manifest
    but missing from the store's own attrs is a reporting gap: the invariant is
    verified, only its location is non-canonical.
    """
    raw = ds.attrs.get(ATTR_TIME_CONVENTION)
    if isinstance(raw, str) and raw:
        ok = raw == TIME_CONVENTION
        rep.add(
            "C1",
            "time_convention",
            ok,
            f"attrs[{ATTR_TIME_CONVENTION!r}] == {TIME_CONVENTION!r} (C1.3; RTMA lagged 1 h)"
            if ok
            else f"attrs[{ATTR_TIME_CONVENTION!r}]={raw!r}, expected {TIME_CONVENTION!r}. "
            "GOFER tUTC is end-of-hour; getting this wrong trains every fire one hour "
            "out of phase with its weather (C1.3)",
        )
        return

    fallback = None
    if manifest_path is not None and Path(manifest_path).is_file():
        try:
            man = json.loads(Path(manifest_path).read_text())
            prov = man.get("provenance")
            if isinstance(prov, dict):
                fallback = prov.get(ATTR_TIME_CONVENTION)
            fallback = fallback or man.get(ATTR_TIME_CONVENTION)
        except Exception:
            fallback = None

    if fallback == TIME_CONVENTION:
        rep.add(
            "C1",
            "time_convention",
            False,
            f"C1.3 satisfied via the sibling manifest provenance, but the STORE does not "
            f"self-describe: add attrs[{ATTR_TIME_CONVENTION!r}] = {TIME_CONVENTION!r}. A tensor "
            "handed to a model without its manifest carries no time convention at all",
            severity=SEVERITY_REPORTING,
        )
        return

    rep.add(
        "C1",
        "time_convention",
        False,
        f"attrs[{ATTR_TIME_CONVENTION!r}] is absent from the store and from the sibling "
        f"manifest provenance; C1.3 requires {TIME_CONVENTION!r} (GOFER tUTC is end-of-hour, "
        "RTMA must be lagged 1 h to match)",
    )


def _check_fire_state_values(
    rep: ContractReport,
    ds: xr.Dataset,
    required_channels: Sequence[str],
    check_absorbing: bool,
) -> None:
    if FIRE_STATE not in required_channels or FIRE_STATE not in ds.data_vars:
        return
    arr = np.asarray(ds[FIRE_STATE].values)
    violations = fire_state_violations(arr)

    domain = [v for v in violations if v.startswith("values outside")]
    rep.add(
        "C1",
        "fire_state_domain",
        not domain,
        f"fire_state values ⊆ {FIRE_STATE_VALUES}" if not domain else f"fire_state {domain[0]}",
    )
    if not check_absorbing:
        return

    absorbing = [v for v in violations if "decreases" in v]
    rep.add(
        "C1",
        "fire_state_absorbing",
        not absorbing,
        "fire_state never decreases in time (fire is absorbing; C1.1)"
        if not absorbing
        else absorbing[0] + " — the cheapest detector of a broken perimeter rasterisation",
    )
    skips = [v for v in violations if "skip" in v]
    rep.add(
        "C1",
        "fire_state_no_skip",
        not skips,
        "no cell jumps 0 -> 2 without a burning hour (C1.1 guarantee)" if not skips else skips[0],
    )


def _check_feature_domains(rep: ContractReport, ds: xr.Dataset) -> None:
    """C1.5 declarations (finite, static, ``{0,1}``, int-as-float) and C1.7 range."""
    if int(ds.sizes.get("channel", 0)) != N_FEATURE_CHANNELS:
        return  # n_channels already failed; per-channel indexing is meaningless

    _check_features_finite(rep, ds)
    _check_physical_ranges(rep, ds)

    drifting: list[str] = []
    for name in sorted(STATIC_CHANNELS):
        arr = _feature_slice(ds, name)
        if arr.shape[0] > 1 and not bool(np.array_equal(arr, np.broadcast_to(arr[0], arr.shape))):
            n = int(np.count_nonzero(np.any(arr != arr[0], axis=0)))
            drifting.append(f"{name} ({n} cells vary)")
    rep.add(
        "C1",
        "static_channels_constant",
        not drifting,
        f"all {len(STATIC_CHANNELS)} static channels are repeated unchanged over time"
        if not drifting
        else f"channels C1 declares static vary in time: {', '.join(drifting)}",
    )

    bad_mask: list[str] = []
    for name in sorted(BINARY_CHANNELS):
        values = np.unique(_feature_slice(ds, name))
        if not set(np.asarray(values).tolist()) <= {0.0, 1.0}:
            bad_mask.append(f"{name}: {values[:6].tolist()}")
    rep.add(
        "C1",
        "mask_channels_binary",
        not bad_mask,
        f"{sorted(BINARY_CHANNELS)} are {{0,1}} masks"
        if not bad_mask
        else f"mask channels carry non-{{0,1}} values: {'; '.join(bad_mask)}",
    )

    bad_int: list[str] = []
    for name in sorted(INTEGER_CHANNELS):
        arr = _feature_slice(ds, name)
        if not bool(np.all(arr == np.round(arr))):
            bad_int.append(name)
    rep.add(
        "C1",
        "class_channels_integral",
        not bad_int,
        f"{sorted(INTEGER_CHANNELS)} hold integral class ids stored as float"
        if not bad_int
        else f"class-id channels hold non-integral values (resampled by interpolation?): {bad_int}",
    )


def _check_features_finite(rep: ContractReport, ds: xr.Dataset) -> None:
    """C1.5 [v2.3] - ``features`` must be finite (no NaN/inf). HARD FAIL.

    This is the widest hole the checker has ever had, and it is the same shape
    as the defect ADR-012 made project policy. Measured on the v2.2 checker: a
    ``features`` array with ``rh_2m`` set to NaN everywhere passed **all 56
    checks**, because no clause looked. The three C1.5 declarations that did
    exist are worse than silent on ``+inf`` - they actively bless it:
    ``inf == round(inf)`` is True, so an infinite ``fuel_model_id`` satisfied
    ``class_channels_integral``, and ``array_equal`` of an all-inf slab against
    itself is True, so it satisfied ``static_channels_constant``.

    Downstream, one NaN NaNs its channel's C3 mean and therefore every
    normalisation of that channel for every fire (C3.4: a per-fire defect
    propagates globally). That is why this is a hard fail and not a warning.
    """
    arr = np.asarray(ds[FEATURES].values)
    finite = np.isfinite(arr)
    if bool(np.all(finite)):
        rep.add("C1", "features_finite", True, "every value in `features` is finite (C1.5)")
        return

    detail: list[str] = []
    for pos, name in enumerate(FEATURE_CHANNELS):
        bad = int(arr[:, pos].size - np.count_nonzero(finite[:, pos]))
        if not bad:
            continue
        values = arr[:, pos][~finite[:, pos]]
        kinds = sorted({"NaN" if np.isnan(v) else f"{v:+.0f}" for v in np.unique(values)[:4]})
        detail.append(f"{name} ({bad} cell-steps: {', '.join(kinds)})")
    rep.add(
        "C1",
        "features_finite",
        False,
        "C1.5: `features` carries non-finite values in "
        + "; ".join(detail)
        + ". A single NaN NaNs that channel's C3 mean and therefore every fire's normalisation "
        "of it (C3.4). Note `inf` also SATISFIES the integral and static declarations, so this "
        "clause is the only thing standing between an infinite channel and a clean report",
    )


def _check_physical_ranges(rep: ContractReport, ds: xr.Dataset) -> None:
    """C1.7 [v2.4] - definitional ranges on the static physical channels. HARD FAIL.

    ADR-010, in one line: the USGS LFPS returns ``-9999`` off the LANDFIRE
    coastline, and ``-9999`` is finite, integral and static - so it satisfied
    every C1.5 declaration and a CZU tensor carrying it reported
    ``OK - 42 checks passed (reporting-ready)`` at a mean canopy cover of
    **-3085%**. Structure was checked; plausibility was not.

    Hard rather than ``reporting`` because these ranges are DEFINITIONAL, not
    heuristic: ``canopy_cover`` is a percentage and ``fuel_model_id`` is an
    enumeration, so no legitimate value lies outside them and there is no
    false-positive mode to protect (contrast C1.6, kept soft precisely because
    it has one). Where a domain genuinely lacks source coverage the answer is a
    documented FILL POLICY, never a tolerated sentinel.
    """
    for name, (low, high) in sorted(PHYSICAL_RANGES.items()):
        values = _feature_slice(ds, name)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            rep.add(
                "C1",
                f"range_{name}",
                False,
                f"{name} has no finite values, so its C1.7 range [{low:g}, {high:g}] cannot be "
                "evaluated. C-1: unverifiable is a failure, not a pass",
            )
            continue
        lo, hi = float(finite.min()), float(finite.max())
        n_bad = int(np.count_nonzero((values < low) | (values > high)))
        unit = CHANNEL_UNITS.get(name, "")
        rep.add(
            "C1",
            f"range_{name}",
            n_bad == 0,
            f"{name} ∈ [{low:g}, {high:g}] {unit} (observed [{lo:g}, {hi:g}]) — C1.7"
            if n_bad == 0
            else (
                f"C1.7: {name} leaves its definitional range [{low:g}, {high:g}] {unit} in "
                f"{n_bad} cell-steps; observed [{lo:g}, {hi:g}], mean {float(finite.mean()):.2f}. "
                + _sentinel_hint(finite, name)
            ),
        )
    _check_fuel_model_enumeration(rep, ds)


def _check_fuel_model_enumeration(rep: ContractReport, ds: xr.Dataset) -> None:
    """C1.7 - ``fuel_model_id`` must be an FBFM40 class, not merely an integer."""
    name = "fuel_model_id"
    values = _feature_slice(ds, name)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        rep.add(
            "C1",
            f"range_{name}",
            False,
            f"{name} has no finite values, so its FBFM40 membership cannot be evaluated. "
            "C-1: unverifiable is a failure, not a pass",
        )
        return
    present = np.unique(finite)
    illegal = [v for v in present if float(v) != round(float(v)) or int(v) not in FBFM40_CLASSES]
    if not illegal:
        rep.add(
            "C1",
            f"range_{name}",
            True,
            f"{name}: all {present.size} distinct values are FBFM40 classes "
            f"(Scott & Burgan 2005; {len(FBFM40_CLASSES)} legal codes) — C1.7",
        )
        return
    n_bad = int(np.count_nonzero(np.isin(values, np.asarray(illegal))))
    shown = ", ".join(f"{float(v):g}" for v in illegal[:6]) + ("…" if len(illegal) > 6 else "")
    rep.add(
        "C1",
        f"range_{name}",
        False,
        f"C1.7: {name} carries {len(illegal)} value(s) outside the FBFM40 class set "
        f"({n_bad} cell-steps): {shown}. FBFM40 is an ENUMERATION — 91/92/93/98/99 plus "
        "101-109, 121-124, 141-149, 161-165, 181-189, 201-204 — so anything else is a "
        "sentinel, a fill or a resampling artefact, never data. " + _sentinel_hint(finite, name),
    )


def _sentinel_hint(values: np.ndarray, name: str) -> str:
    """Name the LFPS sentinel when it is the cause, and state the fill policy."""
    if not bool(np.any(values == -9999.0)):
        return (
            "C1.7 is definitional: fix the SOURCE or document a fill policy; do not widen "
            "the range."
        )
    n = int(np.count_nonzero(values == -9999.0))
    fill = f"{FBFM40_OPEN_WATER} (NB8 Open Water)" if name == "fuel_model_id" else "0"
    return (
        f"{n} cell-steps are exactly -9999 — this is the USGS LFPS NoData sentinel off the "
        f"LANDFIRE coastline (ADR-010), not a measurement. Fill policy of record: map NoData "
        f"to {fill}, and VALIDATE the fill against an independent source (the JRC water mask, "
        "not LFPS itself — validate a hole against a different source than the one that made "
        "it). Never tolerate the sentinel: it is 33.1% of train cell-hours on CZU and would "
        "move the C3 train-mean canopy to -492% from 27.94%, corrupting held-out fires that "
        "contain no NoData at all (C3.4)."
    )


# --------------------------------------------------------------------------
# C2
# --------------------------------------------------------------------------


def check_manifest(manifest_path: str | Path, ds: xr.Dataset | None = None) -> ContractReport:
    """Assert C2 for a manifest.json, optionally cross-checked against a tensor.

    Extra keys are PERMITTED (C2 is a superset contract, v2.1).
    """
    p = Path(manifest_path)
    rep = ContractReport(target=str(p))
    if not p.is_file():
        rep.add("C2", "exists", False, f"no manifest at {p}")
        return rep
    rep.add("C2", "exists", True, "manifest.json exists")

    try:
        man = json.loads(p.read_text())
    except Exception as exc:
        rep.add("C2", "parses", False, f"not valid JSON: {exc}")
        return rep
    if not isinstance(man, dict):
        rep.add("C2", "parses", False, f"manifest must be a JSON object, got {type(man).__name__}")
        return rep
    rep.add("C2", "parses", True, "manifest parses as a JSON object")

    missing = [k for k in MANIFEST_KEYS if k not in man]
    rep.add(
        "C2",
        "keys",
        not missing,
        f"all {len(MANIFEST_KEYS)} C2 keys present (extra keys are permitted)"
        if not missing
        else f"missing keys: {missing}",
    )

    def _is_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    checks: list[tuple[str, bool, str]] = [
        (
            "fire_id",
            isinstance(man.get("fire_id"), str) and bool(man.get("fire_id")),
            "fire_id is a non-empty string",
        ),
        (
            "gofer_version",
            isinstance(man.get("gofer_version"), str) and bool(man.get("gofer_version")),
            "gofer_version is a non-empty string",
        ),
        (
            "bbox_5070",
            isinstance(man.get("bbox_5070"), list | tuple)
            and len(man.get("bbox_5070", ())) == 4
            and all(isinstance(v, int | float) for v in man.get("bbox_5070", ())),
            "bbox_5070 is [xmin, ymin, xmax, ymax] in EPSG:5070 metres",
        ),
        (
            "ignition_time_utc",
            _is_naive_iso(man.get("ignition_time_utc")),
            "ignition_time_utc is a naive-UTC ISO string",
        ),
        ("n_hours", _is_int(man.get("n_hours")), "n_hours is an int"),
        ("cv_fold", _is_int(man.get("cv_fold")), "cv_fold is an int (fold id)"),
        (
            "spatial_block_id",
            _is_int(man.get("spatial_block_id")),
            "spatial_block_id is an int — folds are blocked on connected buffered domains, "
            "so effective n is 11, not 28 (C3.1)",
        ),
        (
            "created_utc",
            _is_naive_iso(man.get("created_utc")),
            "created_utc is a naive-UTC ISO string",
        ),
        (
            "provenance",
            isinstance(man.get("provenance"), dict) and bool(man.get("provenance")),
            "provenance is a non-empty {source: pull-date} dict",
        ),
        (
            "norm_stats_path",
            isinstance(man.get("norm_stats_path"), str) and bool(man.get("norm_stats_path")),
            "norm_stats_path is a string path",
        ),
    ]
    for key, ok, msg in checks:
        rep.add("C2", f"type_{key}", ok, msg if ok else f"{key}={man.get(key)!r} — expected {msg}")

    _check_manifest_v27_keys(rep, man)
    _check_provenance_declarations(rep, man)

    if ds is not None:
        n_time = int(ds.sizes.get("time", -1))
        ok = man.get("n_hours") == n_time
        rep.add(
            "C2",
            "n_hours_matches_tensor",
            ok,
            f"n_hours == len(time) == {n_time}"
            if ok
            else f"n_hours={man.get('n_hours')} but tensor has {n_time} time steps",
        )
        bbox = man.get("bbox_5070")
        if isinstance(bbox, list | tuple) and len(bbox) == 4 and {"x", "y"} <= set(ds.coords):
            expected = tensor_bounds(ds)
            # `_as_float` rather than `float`: a non-numeric or NaN bbox entry must
            # fail THIS clause, not raise and truncate the rest of the punch list.
            got = [_as_float(v) for v in bbox]
            ok = all(
                a is not None and abs(a - float(b)) <= 1e-3
                for a, b in zip(got, expected, strict=True)
            )
            rep.add(
                "C2",
                "bbox_matches_tensor",
                ok,
                "bbox_5070 == tensor outer bounds"
                if ok
                else f"bbox_5070={list(bbox)} but tensor bounds are {list(expected)} "
                "(entries must be finite numbers)",
            )
    return rep


def _manifest_lookup(man: Mapping[str, Any], key: str) -> Any:
    """Value of ``key`` at the manifest root or inside ``provenance``.

    INTERFACES C2 lists the v2.7 keys under "Keys" (root); ADR-014 says
    ``provenance`` must carry them. Both readings are defensible, so both
    locations satisfy the clause and neither is invented by the checker. Same
    precedent as ``channel_index_offset``, which C1 requires the STORE to record
    without legislating root-vs-variable.
    """
    if key in man:
        return man[key]
    prov = man.get("provenance")
    if isinstance(prov, Mapping) and key in prov:
        return prov[key]
    return None


def _as_int(value: object) -> int | None:
    """Strict int coercion. A NUMERIC STRING IS NOT AN INT.

    ``fuels_staleness_years: "3"`` is what the manifests carry today, and
    ADR-014's whole point was *machine-readable*: a consumer that has to guess
    whether a field is a string or a number has to write the guess, and the
    guess is where the next silent defect lives.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _check_manifest_v27_keys(rep: ContractReport, man: Mapping[str, Any]) -> None:
    """C2 [v2.7] - ``fuel_vintage_lag_years`` and ``n_ignition_components`` (ADR-014)."""
    lag = _as_int(_manifest_lookup(man, MANIFEST_VINTAGE_LAG_KEY))
    ok = lag is not None and lag >= 0
    rep.add(
        "C2",
        MANIFEST_VINTAGE_LAG_KEY,
        ok,
        f"{MANIFEST_VINTAGE_LAG_KEY} = {lag} (int; LANDFIRE vintage precedes ignition by this "
        "many years)"
        if ok
        else f"C2 [v2.7] requires {MANIFEST_VINTAGE_LAG_KEY} as a non-negative INT (root or "
        f"provenance); got {_manifest_lookup(man, MANIFEST_VINTAGE_LAG_KEY)!r}. LFPS publishes "
        "no LF2020 product, so 2021 fires carry 5-year and 2022 fires 6-year stale fuels, not "
        "the 1-2 years the MTBS correction is sized for (ADR-014 §3). This is staleness, not "
        "leakage — but the MTBS "
        "correction is far past its design load and any result spanning 2021+ fires MUST state "
        "the lag, so it cannot be left as a string for a reader to parse",
    )

    # Cross-check against the vintage the same manifest already records. A lag
    # that contradicts its own vintage is worse than a missing lag: it is a
    # number a reader would trust.
    prov = man.get("provenance")
    vintage = None
    if isinstance(prov, Mapping):
        vintage = _as_float(prov.get("fuels_vintage_year"))
    fire_year = None
    ignition = man.get("ignition_time_utc")
    if isinstance(ignition, str) and len(ignition) >= 4 and ignition[:4].isdigit():
        fire_year = int(ignition[:4])
    if ok and vintage is not None and fire_year is not None:
        expected = fire_year - int(vintage)
        rep.add(
            "C2",
            "fuel_vintage_lag_consistent",
            lag == expected,
            f"{MANIFEST_VINTAGE_LAG_KEY} = {lag} == ignition year {fire_year} - LANDFIRE "
            f"vintage {int(vintage)}"
            if lag == expected
            else f"{MANIFEST_VINTAGE_LAG_KEY} = {lag} contradicts this manifest's own "
            f"provenance: ignition year {fire_year} - fuels_vintage_year {int(vintage)} = "
            f"{expected}. Two numbers for one fact is worse than one missing number",
        )

    components = _as_int(_manifest_lookup(man, MANIFEST_IGNITION_COMPONENTS_KEY))
    ok = components is not None and components >= 1
    rep.add(
        "C2",
        MANIFEST_IGNITION_COMPONENTS_KEY,
        ok,
        f"{MANIFEST_IGNITION_COMPONENTS_KEY} = {components} (int >= 1)"
        if ok
        else f"C2 [v2.7] requires {MANIFEST_IGNITION_COMPONENTS_KEY} as an INT >= 1 (root or "
        f"provenance); got {_manifest_lookup(man, MANIFEST_IGNITION_COMPONENTS_KEY)!r}. GOFER "
        "FILES SEPARATE LIGHTNING IGNITIONS UNDER ONE FIRE ID: 2020_july_complex has 2 "
        "ignitions (its real second one is the 46.10 km body born at h22; its two 2.24 km "
        "first-frame bodies are ONE ignition split by a rasterisation hole) and SCU has 2 "
        "(29.27 km apart in the first burned frame, later MERGING) plus 2 spot candidates. "
        "[v2.10 CORRECTION, ADR-019: this message said 'SCU has 3' for two contract versions. "
        "That count used FINAL-FOOTPRINT components, which cannot see a merge — the ESTIMAND "
        "was wrong, not just the value.] A separate ignition is a filing artifact and no "
        "contagion kernel can or should reproduce it; a SPOT is real signal and must be kept. "
        "Distance alone does not separate them — use time, then genealogy, then distance. "
        "Undeclared, this corrupts P3 crossings mining and makes G4 meaningless",
    )


#: Facts C2 [v2] says ``provenance`` MUST record, and the keys that satisfy each.
#: Alternatives, not synonyms: leads name things differently and the contract
#: names the FACT, so the checker asserts the fact is recoverable rather than
#: dictating a spelling it never ratified.
PROVENANCE_REQUIRED_FACTS: dict[str, tuple[str, ...]] = {
    "LANDFIRE vintage (ADR-005: vintage must precede ignition)": (
        "fuels_vintage_year",
        "fuels_vintage_folder",
        "landfire_vintage",
    ),
    "state rule (C1.1 fireline_v2)": ("state_rule",),
    "fconf used (C1.1: the six levels ARE the label-perturbation ensemble)": (
        "cfire_conf",
        "fconf",
    ),
}


def _check_provenance_declarations(rep: ContractReport, man: Mapping[str, Any]) -> None:
    """C2 [v2] - ``provenance`` MUST record the LANDFIRE vintage, state rule and ``fconf``.

    Ratified at v2 and never implemented until A10; the checker asserted only
    that ``provenance`` was a non-empty dict. All three facts are load-bearing
    downstream and none is recoverable from the tensor: the vintage because R15
    made staleness a stated limitation of every 2021+ result, the state rule
    because a tensor built under the RETIRED provisional rule is
    indistinguishable from a conformant one by inspection, and ``fconf`` because
    C1.1 designates its six levels as the observation-noise ensemble that
    modelling needs for the ADR-015 §6b label-noise defect.

    Presence, not plausibility: an honest ``"n/a (synthetic)"`` passes. That is
    the C-1 corollary - declaring is a gate, omitting is a failure - and it is
    why the C4 fixture can satisfy a clause about LANDFIRE without pretending to
    have used LANDFIRE.
    """
    prov = man.get("provenance")
    if not isinstance(prov, Mapping):
        return  # type_provenance already failed; do not double-report
    missing = [
        f"{fact} (any of {list(keys)})"
        for fact, keys in PROVENANCE_REQUIRED_FACTS.items()
        if not any(str(prov.get(k, "")).strip() for k in keys)
    ]
    rep.add(
        "C2",
        "provenance_declares_sources",
        not missing,
        "provenance records the LANDFIRE vintage, the state rule and the fconf used (C2 v2)"
        if not missing
        else "C2 [v2]: provenance MUST record the per-fire LANDFIRE vintage (ADR-005) and the "
        "state rule + fconf used. Missing: " + "; ".join(missing) + ". An honest "
        "'n/a (synthetic)' satisfies this — omission does not, because a tensor whose label "
        "rule and fuels vintage are unrecorded cannot be attributed to any pipeline version",
    )


# --------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------


def check_norm_stats(path: str | Path) -> ContractReport:
    """Assert C3 (incl. v2.2's C3.2 file shape and C3.3 bootstrap guard)."""
    p = Path(path)
    rep = ContractReport(target=str(p))
    if not p.is_file():
        rep.add("C3", "exists", False, f"no norm stats at {p}")
        return rep
    rep.add("C3", "exists", True, "norm_stats.json exists")

    try:
        stats = json.loads(p.read_text())
    except Exception as exc:
        rep.add("C3", "parses", False, f"not valid JSON: {exc}")
        return rep
    if not isinstance(stats, dict):
        rep.add("C3", "parses", False, "norm stats must be a JSON object")
        return rep
    rep.add("C3", "parses", True, "norm stats parse as a JSON object")

    missing = [k for k in NORM_STATS_KEYS if k not in stats]
    rep.add(
        "C3",
        "keys",
        not missing,
        f"all {len(NORM_STATS_KEYS)} C3 keys present"
        if not missing
        else f"missing keys: {missing}",
    )

    order = stats.get("channel_order")
    ok = isinstance(order, list) and [str(c) for c in order] == list(CHANNELS)
    rep.add(
        "C3",
        "channel_order",
        ok,
        f"channel_order == C1 order ({N_CHANNELS} names)"
        if ok
        else f"channel_order must equal the C1 channel order, got {order!r}",
    )

    _check_norm_stats_blocks(rep, stats)
    _check_norm_stats_categorical(rep, stats)

    folds = stats.get("train_folds")
    ok = isinstance(folds, list) and all(
        isinstance(f, int) and not isinstance(f, bool) for f in folds
    )
    rep.add(
        "C3",
        "train_folds",
        ok,
        "train_folds records which folds the stats were computed over"
        if ok
        else f"train_folds must be a list of int fold ids, got {folds!r} "
        "(C3: stats come from TRAIN folds only, and that must be auditable)",
    )

    _check_bootstrap_guard(rep, stats)
    _check_legacy_nested_block(rep, stats)
    return rep


def _check_norm_stats_blocks(rep: ContractReport, stats: dict[str, Any]) -> None:
    """C3.2 - the canonical file shape is TOP-LEVEL ``mean``/``std`` dicts."""
    for key in ("mean", "std"):
        block = stats.get(key)
        if not isinstance(block, dict):
            rep.add(
                "C3",
                f"{key}_shape",
                False,
                f"{key} must be a TOP-LEVEL dict keyed by channel name (C3.2 canonical shape)",
            )
            continue
        missing_ch = [c for c in CHANNELS if c not in block]
        extra_ch = [c for c in block if c not in CHANNEL_INDEX]
        shape_ok = not missing_ch and not extra_ch
        rep.add(
            "C3",
            f"{key}_shape",
            shape_ok,
            f"{key} has exactly one entry per channel ({N_CHANNELS})"
            if shape_ok
            else f"{key}: missing={missing_ch} unexpected={extra_ch}",
        )
        if not shape_ok:
            continue
        finite = [c for c in CHANNELS if not _is_finite(block[c])]
        rep.add(
            "C3",
            f"{key}_finite",
            not finite,
            f"all {key} values finite"
            if not finite
            else f"{key} non-finite or non-numeric: {finite}",
        )
        if finite:
            continue
        if key == "std":
            nonpos = [c for c in CHANNELS if float(block[c]) <= 0.0]
            rep.add(
                "C3",
                "std_positive",
                not nonpos,
                "all std > 0 (no divide-by-zero at normalization time)"
                if not nonpos
                else f"std <= 0 for {nonpos}; normalization would divide by zero",
            )
    _check_norm_stats_physical(rep, stats)


def _check_norm_stats_physical(rep: ContractReport, stats: dict[str, Any]) -> None:
    """C3.4 - the norm-stats file is checked as its OWN artifact, not as a summary.

    ADR-010's measured case: CZU's ``-9999`` sentinel is 33.1% of train
    cell-hours and would have moved the TRAIN mean ``canopy_cover`` to
    **-492.13%** from 27.94%, corrupting the normalisation of two held-out fires
    containing no NoData at all. Per-fire QA is necessary and NOT sufficient,
    because a fire can be poisoned by a bug it does not contain - so the shared
    file needs a check that no per-fire report can supply.

    Scoped to channels with a C1.7 DEFINITIONAL range, and skipping the
    categorical channels (whose mean is the identity 0 by C3.2, not a physical
    quantity). A mean outside a definitional range is impossible for any
    weighting of legitimate values.
    """
    mean = stats.get("mean")
    if not isinstance(mean, dict):
        return
    bad: list[str] = []
    checked: list[str] = []
    for name, (low, high) in sorted(PHYSICAL_RANGES.items()):
        if name in CATEGORICAL_CHANNELS or name not in mean or not _is_finite(mean[name]):
            continue
        checked.append(name)
        value = float(mean[name])
        if value < low or value > high:
            bad.append(f"{name} mean={value:.2f} outside [{low:g}, {high:g}]")
    if not checked:
        return
    rep.add(
        "C3",
        "mean_within_physical_range",
        not bad,
        f"train means of {checked} lie inside their C1.7 definitional ranges (C3.4)"
        if not bad
        else "C3.4: " + "; ".join(bad) + ". A mean outside a DEFINITIONAL range cannot come "
        "from any weighting of legitimate values, so one fire's sentinel has poisoned the "
        "shared stats — the measured case is CZU's -9999 moving train-mean canopy to -492% "
        "and corrupting two held-out fires that contain no NoData at all (ADR-010)",
    )


def _check_norm_stats_categorical(rep: ContractReport, stats: dict[str, Any]) -> None:
    """C3.2 - categorical channels take the identity transform, plus a note."""
    mean, std = stats.get("mean"), stats.get("std")
    if not isinstance(mean, dict) or not isinstance(std, dict):
        return
    bad: list[str] = []
    for name in sorted(CATEGORICAL_CHANNELS):
        if name not in mean or name not in std:
            continue
        if not (_is_finite(mean[name]) and _is_finite(std[name])):
            bad.append(f"{name}: non-numeric")
            continue
        if float(mean[name]) != 0.0 or float(std[name]) != 1.0:
            bad.append(f"{name}: mean={mean[name]} std={std[name]}")
    rep.add(
        "C3",
        "categorical_identity",
        not bad,
        f"{sorted(CATEGORICAL_CHANNELS)} carry the identity transform (mean=0, std=1)"
        if not bad
        else f"categorical channels must take the identity transform, got {'; '.join(bad)}. "
        "Standardising an FBFM40 class id is meaningless arithmetic on a label (C3.2)",
    )
    note = stats.get(NORM_STATS_CATEGORICAL_NOTE)
    ok = isinstance(note, str) and bool(note.strip())
    rep.add(
        "C3",
        "categorical_note",
        ok,
        f"{NORM_STATS_CATEGORICAL_NOTE!r} explains why channels 0 and 9 are identity-mapped"
        if ok
        else f"C3.2 requires a {NORM_STATS_CATEGORICAL_NOTE!r} string; a bare mean=0/std=1 is "
        "indistinguishable from a bug",
    )


def _check_bootstrap_guard(rep: ContractReport, stats: dict[str, Any]) -> None:
    """C3.3 - ``n_train_blocks`` must exist (hard); ``>= 2`` gates reporting."""
    raw = stats.get("n_train_blocks")
    ok = isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1
    rep.add(
        "C3",
        "n_train_blocks",
        ok,
        f"n_train_blocks = {raw} (distinct spatial_block_id over the train folds)"
        if ok
        else f"n_train_blocks={raw!r}; C3.3 REQUIRES a positive int. Without it the bootstrap "
        "guard cannot be evaluated at all, which is strictly worse than declaring 1",
    )
    if not ok:
        return
    # `ok` above already established `isinstance(raw, int)`; the cast restates
    # that for the type checker without adding a second runtime guard.
    enough = int(cast("int", raw)) >= MIN_TRAIN_BLOCKS_FOR_REPORTING
    rep.add(
        "C3",
        "bootstrap_guard",
        enough,
        f"n_train_blocks = {raw} >= {MIN_TRAIN_BLOCKS_FOR_REPORTING}: stats span more than one "
        "landscape"
        if enough
        else f"n_train_blocks = {raw} < {MIN_TRAIN_BLOCKS_FOR_REPORTING}: BOOTSTRAP stats. The "
        "only train fire is also the only landscape, which satisfies C3's letter and violates "
        "its spirit. Valid for plumbing only, never for a number that appears in a gate (C3.3)",
        severity=SEVERITY_REPORTING,
    )
    marked = stats.get("bootstrap")
    if not enough and marked is not True:
        rep.add(
            "C3",
            "bootstrap_marked",
            False,
            "C3.3 says a bootstrap file is marked `bootstrap: true`; this one is not, so a "
            "consumer reading only the JSON cannot tell",
            severity=SEVERITY_REPORTING,
        )


def _check_legacy_nested_block(rep: ContractReport, stats: dict[str, Any]) -> None:
    """C3.2 - the pre-v2.2 nested ``channels`` block is retired.

    Present-but-agreeing is a reporting gap (data drops it at A5, ADR-008).
    Present-and-DISAGREEING is a hard failure: two consumers reading two keys
    would then normalise differently, which is the ambiguity C3.2 exists to end.

    Non-finite handling, corrected under ADR-012. An explicit ``null`` IS
    agreement (that is how a categorical channel is written). A NaN or inf is
    NOT: it cannot be compared, and the previous code ``continue``-d past it
    straight into "does not contradict" - an unverifiable value landing in the
    pass branch, the exact shape this policy exists to forbid.
    """
    nested = stats.get(NORM_STATS_LEGACY_BLOCK)
    if not isinstance(nested, dict):
        return
    mean, std = stats.get("mean"), stats.get("std")
    conflicts: list[str] = []
    if isinstance(mean, dict) and isinstance(std, dict):
        for name, entry in nested.items():
            if not isinstance(entry, dict) or name not in mean:
                continue
            for key, top in (("mean", mean), ("std", std)):
                inner = entry.get(key)
                if inner is None:
                    continue  # `null` for a categorical channel is agreement
                if not _is_finite(inner) or not _is_finite(top.get(name)):
                    conflicts.append(
                        f"{name}.{key}: nested={inner!r} top-level={top.get(name)!r} "
                        "— not comparable, so agreement is UNVERIFIABLE (C-1)"
                    )
                    continue
                if abs(float(inner) - float(top[name])) > 1e-9 * max(1.0, abs(float(top[name]))):
                    conflicts.append(f"{name}.{key}: nested={inner} top-level={top[name]}")
    rep.add(
        "C3",
        "nested_block_agrees",
        not conflicts,
        "the legacy nested block does not contradict the canonical top-level stats"
        if not conflicts
        else "legacy nested `channels` block CONTRADICTS the canonical top-level stats: "
        + "; ".join(conflicts[:4]),
    )
    rep.add(
        "C3",
        "no_legacy_nested_block",
        False,
        f"a nested {NORM_STATS_LEGACY_BLOCK!r} block is present; C3.2 fixed the canonical shape "
        "as TOP-LEVEL channel_order/mean/std and this duplicate must be dropped (ADR-008, A5)",
        severity=SEVERITY_REPORTING,
    )


def _is_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return bool(np.isfinite(float(value)))


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------


def check_all(
    tensor_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    norm_stats_path: str | Path | None = None,
    require_manifest: bool = True,
    require_norm_stats: bool = True,
    required_channels: Sequence[str] = CHANNELS,
) -> ContractReport:
    """Run C1 (+C2, +C3) for a tensor store and its siblings.

    ``manifest_path`` defaults to ``manifest.json`` beside the store;
    ``norm_stats_path`` defaults to whatever the manifest points at, resolved
    relative to the manifest directory.
    """
    tensor_path = Path(tensor_path)
    mpath = Path(manifest_path) if manifest_path else tensor_path.parent / "manifest.json"

    rep = check_tensor(tensor_path, required_channels=required_channels, manifest_path=mpath)
    rep.target = str(tensor_path)

    ds: xr.Dataset | None = None
    try:
        ds = open_tensor_dataset(tensor_path)
    except Exception:
        ds = None

    if require_manifest:
        rep.extend(check_manifest(mpath, ds=ds))

        if require_norm_stats and norm_stats_path is None and mpath.is_file():
            try:
                declared = json.loads(mpath.read_text()).get("norm_stats_path")
            except Exception:
                declared = None
            if isinstance(declared, str) and declared:
                candidate = Path(declared)
                norm_stats_path = (
                    candidate if candidate.is_absolute() else (mpath.parent / candidate).resolve()
                )

    if require_norm_stats:
        if norm_stats_path is None:
            rep.add(
                "C3",
                "located",
                False,
                "could not locate norm stats: pass --norm-stats or set manifest.norm_stats_path",
            )
        else:
            rep.extend(check_norm_stats(norm_stats_path))
    return rep


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.common.contract",
        description=(
            "Check any tensor.zarr against the INTERFACES.md contracts C1/C2/C3. "
            "Exit code 0 = conformant, 1 = violation."
        ),
    )
    p.add_argument("tensor", type=Path, help="path to a tensor.zarr store")
    p.add_argument("--manifest", type=Path, default=None, help="default: manifest.json alongside")
    p.add_argument("--norm-stats", type=Path, default=None, help="default: from the manifest")
    p.add_argument("--skip-manifest", action="store_true", help="skip C2")
    p.add_argument("--skip-norm-stats", action="store_true", help="skip C3")
    p.add_argument(
        "--labels-only",
        action="store_true",
        help=(
            "check only fire_state against C1 (grid/time/state rules still enforced). "
            "For ADR-003 interim label stores, which are NOT C1-complete."
        ),
    )
    p.add_argument(
        "--for-reporting",
        action="store_true",
        help=(
            "promote reporting-gate clauses to hard failures. Required before any number "
            "from this artifact appears in a gate (e.g. C3.3 n_train_blocks >= 2)."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true", help="print passing checks too")
    p.add_argument("--json", action="store_true", help="emit a JSON report on stdout")
    add_logging_arguments(p)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    # ADR-103: the ONE place this program is allowed to configure logging. It is
    # also why the handler is pinned to stderr - `make contract-all-fires` greps
    # the table this main prints to stdout.
    configure_from_args(args)
    if args.labels_only:
        rep = check_tensor(args.tensor, required_channels=[FIRE_STATE], require_channel_coord=False)
        rep.target = str(args.tensor)
    else:
        rep = check_all(
            args.tensor,
            manifest_path=args.manifest,
            norm_stats_path=args.norm_stats,
            require_manifest=not args.skip_manifest,
            require_norm_stats=not args.skip_norm_stats,
        )
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(rep.format(verbose=args.verbose, for_reporting=args.for_reporting))
    good = rep.reporting_ok if args.for_reporting else rep.ok
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
