"""The C3.3 reporting guard, made mechanical.

C3.3 (ADR-008 P8, enforced at v2.3): a norm-stats file built when the only train
fire is also the only landscape satisfies C3's letter and violates its spirit. It
carries ``n_train_blocks`` and any REPORTED result must assert
``n_train_blocks >= 2``.

``make contract-reporting`` enforces that on a TENSOR. This module enforces it on
a NUMBER, at the moment the number is about to be written down, because that is
the moment it matters and the moment it is most likely to be skipped. Everything
this project emits goes through one of two functions:

* :func:`stamp_smoke_test` - marks a result as plumbing-only. Always allowed.
* :func:`assert_reportable` - raises unless the norm stats span >= 2 blocks.

There is no third option and no ``force=True``. The trap this guards against is
the highest-consequence trap for the first model numbers:
``data/norm_stats.json`` is currently a BOOTSTRAP with
``n_train_blocks = 1`` and ``train_folds = [4]``, and Kincade's own ``cv_fold``
is 4 - so the only fire is simultaneously the only train fire and the only
possible test fire. Any number produced against it is a smoke test, and every
artifact this package writes says so in its own payload rather than only in a
document that a reader may not have.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.contract import MIN_TRAIN_BLOCKS_FOR_REPORTING
from wildfire_nowcast.common.paths import norm_stats_path

__all__ = [
    "NotReportableError",
    "SplitChangedError",
    "reporting_status",
    "assert_reportable",
    "stamp_smoke_test",
    "split_fingerprint",
    "assert_split_unchanged",
    "assert_model_split_matches",
    "common_code_fingerprint",
    "scoring_code_fingerprint",
    "check_common_code_unchanged",
    "SMOKE_TEST_BANNER",
]

SMOKE_TEST_BANNER = (
    "SMOKE TEST -- NOT A RESULT. Produced under C3.3 bootstrap norm stats "
    "(n_train_blocks < 2). Numbers here verify plumbing only and must not be "
    "quoted in a gate, a report or a figure caption."
)


class NotReportableError(RuntimeError):
    """Raised when a number would be reported under bootstrap norm stats."""


def reporting_status(path: str | Path | None = None) -> dict[str, Any]:
    """Machine-readable C3.3 state of a norm-stats file. Never raises on content.

    A MISSING file is reported as not-reportable rather than as an error, so
    this can be embedded in any artifact; a missing ``n_train_blocks`` key is
    treated as a hard defect (ADR-009: an unevaluable guard is strictly worse
    than a declared-weak one).
    """
    target = Path(path) if path is not None else norm_stats_path()
    if not target.is_file():
        return {
            "path": str(target),
            "exists": False,
            "reportable": False,
            "reason": f"no norm-stats file at {target}",
        }
    try:
        stats: Mapping[str, Any] = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        return {
            "path": str(target),
            "exists": True,
            "reportable": False,
            "reason": f"norm-stats file is not valid JSON: {exc}",
        }
    blocks = stats.get("n_train_blocks")
    if not isinstance(blocks, int) or isinstance(blocks, bool):
        return {
            "path": str(target),
            "exists": True,
            "reportable": False,
            "n_train_blocks": blocks,
            "reason": (
                "n_train_blocks is missing or not an int; C3.3 requires a positive int. "
                "The guard cannot be evaluated at all, which is worse than declaring 1."
            ),
        }
    reportable = blocks >= MIN_TRAIN_BLOCKS_FOR_REPORTING
    return {
        "path": str(target),
        "exists": True,
        "reportable": reportable,
        "n_train_blocks": blocks,
        "min_required": MIN_TRAIN_BLOCKS_FOR_REPORTING,
        "bootstrap": bool(stats.get("bootstrap", False)),
        "train_folds": list(stats.get("train_folds", [])),
        "train_fire_ids": list(stats.get("train_fire_ids", [])),
        "reason": (
            "ok"
            if reportable
            else (
                f"n_train_blocks={blocks} < {MIN_TRAIN_BLOCKS_FOR_REPORTING}: the train set "
                "spans one landscape, so held-out numbers measure one landscape (C3.3)."
            )
        ),
    }


def assert_reportable(path: str | Path | None = None) -> dict[str, Any]:
    """Raise :class:`NotReportableError` unless C3.3 is satisfied.

    Call this immediately before any number leaves this package for a gate, a
    figure or a document. Returns the status dict so it can be embedded as
    provenance in the same artifact it just authorised.
    """
    status = reporting_status(path)
    if not status["reportable"]:
        raise NotReportableError(
            f"REFUSING to report a number: {status['reason']} "
            f"(file: {status['path']}). Build more fires and recompute the stats; "
            "do not relax the guard. Use stamp_smoke_test() for plumbing runs."
        )
    return status


# --------------------------------------------------------------------------
# C-4.2 - the code fingerprints. **THE MODULE SETS ARE NO LONGER ENUMERATED
# HERE (ADR-057).**
#
# Two hand-written tuples used to live at this spot: ``_COMMON_CODE_MODULES``
# (7 files out of `common/`) and ``_SCORING_CODE_MODULES`` (10 files out of
# `eval/` + `model/`). The second one OMITTED ``model/noiseoracle.py`` and
# ``model/direct.py``, so a run stamped "one version of the scoring code" while
# an uncovered module was edited after it finished. A list that must be updated
# by hand is not a check on a growing tree; it is a claim that decays silently.
#
# Both sets are now WALKED from the imported package by
# ``common/codefingerprint.py``: whole subtrees, recursive, so a module added
# tomorrow and a module that becomes a package are both covered without anyone
# remembering. That module is in `common/` and not here on purpose - C0 puts the
# one implementation of an adjudicated quantity in `common/`, the same argument
# that re-homed C8 just below.
#
# CONSEQUENCE, STATED RATHER THAN DISCOVERED: coverage widened, so both
# fingerprint VALUES change, and `per_file` keys are now package-relative
# (``common/contract.py``, not ``contract.py``). Fingerprints recorded by runs
# before this change are not comparable with fingerprints recorded after it.
#
# C8 - RE-HOMED to common/splits.py (ADR-015 (4) -> infra A10).
#
# This module originated `split_fingerprint` / `assert_split_unchanged` during
# M2, after the CV split moved under a running experiment. infra has since
# re-homed that logic into `common/splits.py` as contract, deliberately
# reproducing `4848f491e8d588fa` byte-identically so no artifact already citing
# it is orphaned. C0 says an adjudicated quantity has exactly ONE
# implementation, so this module now DELEGATES rather than keeping a second
# copy that could drift. The names stay importable from here because callers
# (mine and simviz's) already import them from here, and moving an import is
# not worth breaking a consumer over.
#
# `SplitChangedError` is aliased, not redefined: two exception classes with one
# name is how an `except` clause in another module silently stops catching.
# --------------------------------------------------------------------------
from wildfire_nowcast.common.codefingerprint import (  # noqa: E402
    COMMON_SUBTREES,
    SCORING_SUBTREES,
    code_fingerprint,
)
from wildfire_nowcast.common.splits import (  # noqa: E402
    SplitChangedError,
    assert_split_unchanged,
    split_fingerprint,
)


def common_code_fingerprint() -> dict[str, Any]:
    """Hash the ``common/`` modules a reported number is computed through.

    **This is the second instance of the hazard C8 was written for, and it is not
    yet contract - it is a PROPOSAL, so it warns rather than fails.** ADR-015 (4)
    made the CV split's version auditable after a fold change landed under a
    running experiment. The same shape recurs one level down: ``common/`` is
    shared, infra owns it, and it can be rewritten while a training run is
    in flight. Measured live during M3 - ``common/{contract,zarr_io,runs,
    splits,synthetic}.py`` were all rewritten inside a ten-minute window, and
    ``synthetic.py`` was momentarily un-importable - so this is an observed
    hazard, not a hypothetical one.

    A hash change does NOT invalidate a result by itself (a docstring edit is
    harmless), which is exactly why this reports instead of raising: an
    over-strict check on shared code would fire constantly and be disabled. What
    it buys is that "which version of the state rule produced this number" stops
    being unanswerable after the fact.

    [ADR-057] Covers ALL of ``common/`` now, recursively, rather than seven named
    files. That is not a widening for its own sake: C-4 freezes the WHOLE of
    ``common/`` while a lead runs, so a fingerprint over part of it answered a
    narrower question than the clause it serves asks.
    """
    return code_fingerprint(
        COMMON_SUBTREES,
        status="PROPOSAL: reported, not enforced.",
    )


def scoring_code_fingerprint() -> dict[str, Any]:
    """Hash the ``eval/`` + ``model/`` modules a reported number is computed IN.

    WHY THIS EXISTS, measured rather than assumed: ADR-016 replaced ``git_sha
    "unknown"`` with a real SHA, and the SHA is real - but **every run in this
    repo also carries ``git_dirty: true``, because the entire ``src/`` tree is
    still untracked against the scaffold commit.** So the SHA identifies the
    scaffold, not the code that produced the number. A reader who checks out
    ``f5e5857`` gets a repo with no model in it. The fingerprint below is what
    actually distinguishes two runs of the same nominal commit.

    Reported, never enforced, for the same reason as
    :func:`common_code_fingerprint`: a docstring edit changes the hash and does
    not change the result, so raising here would train people to bypass it.

    [ADR-057] The ten enumerated modules are gone. This walks ``eval/`` and
    ``model/`` whole, which is what makes the answer to "was the scoring code
    edited during this run" mean the same thing next month as it does today.
    """
    return code_fingerprint(
        SCORING_SUBTREES,
        status=(
            "PROPOSAL — reported, not enforced. `git_sha` is real but NOT identifying "
            "while `git_dirty` is true for every run; this is."
        ),
    )


def check_common_code_unchanged(before: Mapping[str, Any]) -> dict[str, Any]:
    """Report whether ``common/`` moved during a run. Never raises (see above)."""
    now = common_code_fingerprint()
    changed = [
        name
        for name, digest in now["per_file"].items()
        if digest != dict(before.get("per_file", {})).get(name)
    ]
    return {
        **now,
        "changed_during_run": changed,
        "warning": (
            None
            if not changed
            else (
                f"SHARED CODE MOVED DURING THIS RUN: {changed}. The numbers in this artifact "
                "were computed partly before and partly after that edit. Re-run before "
                "quoting them in a gate."
            )
        ),
    }


def assert_model_split_matches(
    model: Any, evaluation_split: Mapping[str, Any], *, name: str = "model"
) -> dict[str, Any]:
    """C8 (INTERFACES v2.8): HARD FAIL if a model's TRAINING split != this one.

    ``assert_split_unchanged`` catches the split moving *during* a run. It cannot
    catch the other half of the same hazard: a checkpoint trained under an
    OLDER split being scored under a newer one. That mismatch is exactly how a
    trained-on fire ends up in a held-out table, and no per-tensor check can see
    it because every tensor stays individually conformant.

    Resolution order, all of it verification and none of it assumption:

    1. ``model.provenance['split_fingerprint']`` - stamped by the trainer.
    2. ``<run_dir>/training.json`` ``split_before.fingerprint`` - where runs
       predating the stamp recorded it. Reading it there is still reading it off
       disk; it is not inferring it.
    3. Nothing found -> **HARD FAIL**. Per C-1, "invariant violated OR
       unverifiable" is a failure. An unstamped checkpoint is not presumed
       innocent, because the presumption is precisely what the clause exists to
       remove.

    A baseline with no training split (persistence, ellipse) is exempt: it is
    calibrated inside this run, on this split, by this function's caller.
    """
    expected = str(evaluation_split.get("fingerprint"))
    provenance = dict(getattr(model, "provenance", {}) or {})
    found = provenance.get("split_fingerprint")
    source = "model.provenance"
    if not found:
        run_dir = provenance.get("run_dir") or getattr(model, "run_dir", None)
        candidates = [Path(run_dir)] if run_dir else []
        loaded_from = getattr(model, "_loaded_from", None)
        if loaded_from:
            p = Path(loaded_from)
            candidates.append(p if p.is_dir() else p.parent)
        for candidate in candidates:
            training = candidate / "training.json"
            if training.is_file():
                payload = json.loads(training.read_text())
                found = (payload.get("split_before") or {}).get("fingerprint")
                source = f"{training}"
                if found:
                    break
    if not found:
        raise SplitChangedError(
            f"C8 HARD FAIL: model {name!r} carries NO split_fingerprint, in its spec or in a "
            f"training.json beside it. INTERFACES v2.8 requires every run to stamp the split "
            f"it was fitted on, and C-1 makes 'unverifiable' a failure rather than a pass. "
            f"The evaluation split is {expected}. Retrain, or point at the run directory that "
            f"recorded the fingerprint; do NOT score it unstamped."
        )
    if str(found) != expected:
        raise SplitChangedError(
            f"C8 HARD FAIL: model {name!r} was TRAINED on split {found} (source: {source}) but "
            f"is being EVALUATED on split {expected}. Fires that were TRAIN under {found} may "
            f"be HELD OUT under {expected}, which makes every held-out number here trained-on. "
            "This is the failure ADR-015 (4) made contract. Retrain on the current split."
        )
    return {
        "model": name,
        "training_split_fingerprint": str(found),
        "evaluation_split_fingerprint": expected,
        "match": True,
        "source": source,
    }


def stamp_smoke_test(payload: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """Return ``payload`` marked as a smoke test, with the C3.3 status embedded.

    Idempotent and total: a smoke-test artifact carries its own disclaimer, so a
    JSON file that outlives the notes written beside it cannot be mistaken for a
    result by whoever finds it next.
    """
    out = dict(payload)
    out["smoke_test"] = True
    out["reporting_ready"] = False
    out["smoke_test_banner"] = SMOKE_TEST_BANNER
    out["c3_3_status"] = reporting_status(path)
    return out
