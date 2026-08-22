"""C8 - the split fingerprint, and the cross-fire clauses no per-tensor check can see.

**Why this module exists (ADR-015 §4).** The CV split moved *mid-task*:
``train_folds`` went from ``[0, 1, 3]`` to ``[0, 1, 2, 4]`` while modelling was
training, and four fires silently crossed from TRAIN to HELD-OUT. Every tensor
was individually conformant the entire time - **no per-tensor check could ever
have seen it**, because the defect is not in any tensor, it is in the RELATION
between them and a running experiment. modelling detected it, discarded the
contaminated matrix rather than shipping it, and wrote ``split_fingerprint`` /
``assert_split_unchanged`` in ``eval/``. Per C0 the contract-adjudicated
implementation lives here, in ``common/``; the logic is modelling's and is
unchanged, because it was sound.

Three kinds of clause live here, all of them cross-fire:

* **C8** - a run stamps a fingerprint; the fingerprints inside one artifact must
  agree (train-time vs eval-time), and an evaluation must not consume a
  checkpoint trained under a different split. Mismatch is a **HARD FAIL**.
* **C3.1** - overlapping fires MUST share a fold. Two fires in one spatial block
  landing in different folds is landscape leakage, and it is invisible to both
  fires' own manifests.
* **C3.3 / C6.3** - how many distinct *blocks* the split holds out (not fires).

Compatibility note, deliberately load-bearing: :func:`split_fingerprint` must
reproduce the fingerprint already on record (``4848f491e8d588fa``, ADR-015), or
re-homing it would silently invalidate every artifact stamped so far. The
payload construction is therefore byte-identical to ``eval.reporting``'s and
:func:`tests.test_splits` asserts the two agree.

Standalone use::

    .venv/bin/python -m wildfire_nowcast.common.splits            # current split
    .venv/bin/python -m wildfire_nowcast.common.splits --run runs/kernel-.../
    make contract-split RUN=runs/baselines-20260808-052918
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.contract import (
    SEVERITY_FAIL,
    SEVERITY_REPORTING,
    ContractReport,
)
from wildfire_nowcast.common.paths import fires_dir, norm_stats_path, repo_relative, runs_dir

__all__ = [
    "SplitChangedError",
    "FINGERPRINT_KEYS",
    "MIN_HELDOUT_BLOCKS_FOR_G2",
    "DECLARED_MEMBERSHIP_KEYS",
    "declared_split_membership",
    "split_fingerprint",
    "assert_split_unchanged",
    "fingerprints_in",
    "read_json",
    "check_split_assignment",
    "check_run_split",
    "check_split_chain",
    "CODE_FINGERPRINT_FAMILIES",
    "ENVIRONMENT_FINGERPRINT_FAMILY",
    "RUN_FINGERPRINT_FAMILIES",
    "code_fingerprint_ends",
    # [v2.16] C8.1 - the CV-matrix artifact class (ADR-062 (6))
    "CV_MATRIX_KEY",
    "CV_MATRIX_MEMBER_KEYS",
    "CvMatrixMember",
    "DeclaredCvMatrix",
    "declared_cv_matrix",
    # [v2.16] C8.2 - fit/stamp atomicity (ADR-062 (5))
    "SPLIT_CONTEXT_KEY",
    "SplitContext",
    "SplitFitStampMismatchError",
    "assert_fit_and_stamp_agree",
    "resolve_split_context",
    # [v2.16] C6.3 (addition) - the expected-false stamp (ADR-062 (7))
    "C6_3_EXPECTED_FALSE_KEY",
    "LEAVE_FOLD_OUT_BLOCKS",
    "folds_expected_to_fail_c6_3",
    "stamp_c6_3_expected_false",
]


class SplitChangedError(RuntimeError):
    """Raised when the CV split moved between training and reporting."""


#: Keys under which a fingerprint may be stamped in a run artifact. Several
#: shapes exist in ``runs/`` already (``split_before``/``split_after`` blocks,
#: ``scope.split_fingerprint``, a bare string); C8 says every run stamps one, it
#: does not legislate the key. Reading all of them is the point: a checker that
#: only understands its own writer's shape is not a checker.
FINGERPRINT_KEYS: tuple[str, ...] = (
    "split_fingerprint",
    "split_before",
    "split_after",
    "split",
    "fingerprint",
)

#: C6.3 - G2 requires >= 4 DISTINCT held-out spatial blocks (ADR-011).
MIN_HELDOUT_BLOCKS_FOR_G2 = 4

#: [A14] C3 - the two keys ``data/norm_stats.json`` MUST declare, by fire id.
#:
#: ADR-038 (6) ruled these into the contract after the maintainer queried
#: ``train_fires`` (a key that does not exist), got ``None``, and published "the
#: file records no train list" as a fact. The keys were there the whole time and
#: **nothing read them** - so the misread was undetectable by anything except
#: someone happening to look at the right key name.
#:
#: They are checked BY ID and not by fold index on purpose: ``cv_fold`` is not
#: identifying across corpus versions. ``train_folds [0, 1, 2, 4]`` names 12
#: fires under the pre-D6 corpus and 16 under the current one, so two artifacts
#: can agree on every fold number and disagree about which fires were trained on.
#: This is the same reason C8 fingerprints the split rather than trusting a fold
#: list, and the same "second place to forget" that left ``CONTRACT_VERSION``
#: stale for seven versions.
DECLARED_MEMBERSHIP_KEYS: tuple[str, str] = ("train_fire_ids", "heldout_fire_ids")

#: [v2.11] C-4.2 - code-fingerprint families a run artifact may stamp. Each must
#: be sampled at BOTH ends of the run.
#:
#: ``common_code``  - the ``common/`` modules a number is computed THROUGH.
#: ``scoring_code`` - the ``eval/`` + ``model/`` modules it is computed IN.
CODE_FINGERPRINT_FAMILIES: tuple[str, ...] = ("common_code", "scoring_code")

#: [v2.12] C-4.3 - the INTERPRETER ENVIRONMENT joins C-4's frozen set (ADR-024).
#: A separate family, and deliberately NOT appended to
#: :data:`CODE_FINGERPRINT_FAMILIES`: it gets its own check ids so a report says
#: WHICH thing moved. Giving two different quantities one key name is exactly the
#: defect that made ``C8.internally_consistent`` false on every artifact in this
#: repo (A12), and adding a third fingerprint family under the ``code_`` name
#: would have recreated it on every future run.
ENVIRONMENT_FINGERPRINT_FAMILY = "environment"

#: Every family a run artifact may stamp at both ends. Used by
#: :func:`code_fingerprint_ends` and by :data:`_NON_SPLIT_FINGERPRINT_BLOCKS` -
#: the latter matters more than it looks: an ``environment`` block carries its own
#: ``fingerprint`` key, so without this the split walker would have counted it as
#: a SECOND split stamp and hard-failed ``C8.internally_consistent`` on every run
#: that stamps one. That is the A12 false positive, pre-empted rather than
#: rediscovered.
RUN_FINGERPRINT_FAMILIES: tuple[str, ...] = (
    *CODE_FINGERPRINT_FAMILIES,
    ENVIRONMENT_FINGERPRINT_FAMILY,
)


def code_fingerprint_ends(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """[v2.11] C-4.2 - every code fingerprint in one artifact, keyed by END.

    Returns ``{family: {"before": fp, "after": fp, "unpaired": fp}}``, omitting
    ends that are absent. ``unpaired`` is a bare ``{family}`` key with no
    ``_before``/``_after`` suffix - the exact shape C-4.2 was written about, and
    the reason this returns a structure rather than a boolean: the difference
    between "sampled twice and agreed" and "sampled once" is the whole clause.

    Pure and payload-only, so it is testable without a run directory.
    """
    out: dict[str, dict[str, str]] = {}

    def digest(node: object) -> str | None:
        if isinstance(node, str):
            return node
        if isinstance(node, Mapping):
            value = node.get("fingerprint")
            return str(value) if isinstance(value, (str, int)) else None
        return None

    for family in RUN_FINGERPRINT_FAMILIES:
        ends: dict[str, str] = {}
        for suffix, end in (("_before", "before"), ("_after", "after"), ("", "unpaired")):
            found = digest(payload.get(f"{family}{suffix}"))
            if found is not None:
                ends[end] = found
        if ends:
            out[family] = ends
    return out


def read_json(path: str | Path) -> dict[str, Any] | None:
    """``json.loads`` of a file, or ``None``. Never raises - a malformed
    artifact must fail its own clause rather than abort the punch list."""
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------
# The fingerprint itself (modelling's logic, re-homed under C0)
# --------------------------------------------------------------------------


def _train_folds_from_norm_stats(path: str | Path | None = None) -> list[int]:
    stats = read_json(path or norm_stats_path()) or {}
    folds = stats.get("train_folds")
    if not isinstance(folds, list):
        return []
    out: list[int] = []
    for f in folds:
        if isinstance(f, bool) or not isinstance(f, int | float):
            continue
        out.append(int(f))
    return sorted(out)


def declared_split_membership(path: str | Path | None = None) -> dict[str, Any]:
    """Read the fire ids ``norm_stats.json`` DECLARES, and say what is wrong with them.

    This function READS A VALUE. That is the whole design: the thing it replaced
    (``set(fp["train_fire_ids"]) & set(fp["heldout_fire_ids"])`` on the
    fingerprint's own output) proved a set was empty using two lists that
    :func:`split_fingerprint` builds as ``[r for r in rows if r[1] in train_set]``
    and ``[r for r in rows if r[1] not in train_set]`` - **a partition by
    construction, whose intersection is empty as a matter of set algebra, on any
    input, forever.** It printed green for months and could not have printed
    anything else. Reading the recorded lists is the only version of this check
    that can fail.

    Returns ``{"train": [...], "heldout": [...], "problems": [...], "present":
    bool}``. Problems are strings, never exceptions: a malformed artifact must
    fail its own clause rather than abort the punch list.
    """
    stats = read_json(path or norm_stats_path())
    problems: list[str] = []
    if stats is None:
        return {
            "train": [],
            "heldout": [],
            "present": False,
            "problems": [
                f"norm_stats.json is missing or unreadable at {path or norm_stats_path()}"
            ],
        }

    out: dict[str, list[str]] = {}
    for key, role in zip(DECLARED_MEMBERSHIP_KEYS, ("train", "heldout"), strict=True):
        value = stats.get(key)
        if value is None:
            problems.append(
                f"`{key}` is ABSENT. C3 requires it: fold indices are not identifying across "
                "corpus versions, so a file that records only `train_folds` cannot be audited "
                "for leakage by anyone reading it later"
            )
            out[role] = []
            continue
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            problems.append(f"`{key}` is not a list of fire-id strings: {value!r}")
            out[role] = []
            continue
        if not value:
            problems.append(
                f"`{key}` is an EMPTY list; a split with no {role} fires is not a split"
            )
        duplicates = sorted({v for v in value if value.count(v) > 1})
        if duplicates:
            problems.append(f"`{key}` lists the same fire more than once: {duplicates}")
        out[role] = [str(v) for v in value]

    return {
        "train": out.get("train", []),
        "heldout": out.get("heldout", []),
        "present": not problems,
        "problems": problems,
    }


def split_fingerprint(
    *,
    fires_root: str | Path | None = None,
    stats_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the CURRENT leave-fire-out split: fires, folds, blocks, train folds.

    Total by construction: a missing ``data/fires/`` or a missing norm-stats file
    yields a fingerprint over zero fires rather than an exception, because this
    is called from :func:`wildfire_nowcast.common.runs.create_run_dir` and a
    provenance stamp must never be the thing that kills a training run.

    A fire whose manifest is unreadable or lacks ``cv_fold`` /
    ``spatial_block_id`` is recorded in ``unreadable`` and excluded from the
    hash; :func:`check_split_assignment` fails on it. Silently hashing a
    partial split would make the fingerprint agree across a real change.
    """
    root = Path(fires_root) if fires_root else fires_dir()
    rows: list[list[Any]] = []
    unreadable: list[str] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        man = read_json(manifest_path)
        try:
            rows.append(
                [
                    str(man["fire_id"]),  # type: ignore[index]
                    int(man["cv_fold"]),  # type: ignore[index]
                    int(man["spatial_block_id"]),  # type: ignore[index]
                    int(man["n_hours"]),  # type: ignore[index]
                ]
            )
        except Exception:
            unreadable.append(str(manifest_path.parent.name))

    train_folds = _train_folds_from_norm_stats(stats_path)
    train_set = set(train_folds)
    train = [r for r in rows if r[1] in train_set]
    heldout = [r for r in rows if r[1] not in train_set]
    heldout_blocks = {int(r[2]) for r in heldout}

    # Byte-identical to eval.reporting.split_fingerprint: json.dumps of
    # {"fires": rows, "train_folds": train_folds} with sort_keys, sha256[:16].
    # Do not "improve" this construction - it would invalidate every artifact
    # already stamped (the fingerprint of record is 4848f491e8d588fa).
    payload = json.dumps({"fires": rows, "train_folds": train_folds}, sort_keys=True)
    return {
        "fingerprint": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "n_fires": len(rows),
        "train_folds": train_folds,
        "train_fire_ids": [r[0] for r in train],
        "heldout_fire_ids": [r[0] for r in heldout],
        "train_blocks": sorted({int(r[2]) for r in train}),
        "heldout_blocks": sorted(heldout_blocks),
        "n_heldout_blocks": len(heldout_blocks),
        "unreadable_fires": unreadable,
        "c6_3_satisfied": len(heldout_blocks) >= MIN_HELDOUT_BLOCKS_FOR_G2,
        "c6_3_note": (
            f"C6.3: G2 requires >= {MIN_HELDOUT_BLOCKS_FOR_G2} DISTINCT held-out spatial "
            "blocks. More fires from the same block are the same evidence with false "
            "confidence, not more evidence."
        ),
    }


def assert_split_unchanged(before: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Raise :class:`SplitChangedError` if the split moved since ``before``.

    Call at the END of any run whose result depends on the split - which is
    every leave-fire-out run. A result produced across a split change is not a
    weak result, it is a leaked one.
    """
    now = split_fingerprint(**kwargs)
    if now["fingerprint"] != before.get("fingerprint"):
        raise SplitChangedError(
            "THE CV SPLIT CHANGED DURING THIS RUN (C8). "
            f"before={before.get('fingerprint')} ({len(before.get('train_fire_ids', []))} train "
            f"fires, folds {before.get('train_folds')}), "
            f"now={now['fingerprint']} ({len(now['train_fire_ids'])} train fires, folds "
            f"{now['train_folds']}). Fires that were TRAIN may now be HELD OUT, which makes "
            "every held-out number here trained-on. Re-run against a frozen split; do not "
            "report these numbers."
        )
    return now


# --------------------------------------------------------------------------
# [v2.16] C8.2 - ATOMICITY OF THE FIT AND THE STAMPS (ADR-062 (5))
# --------------------------------------------------------------------------


class SplitFitStampMismatchError(RuntimeError):
    """The normalisation FIT and the split STAMP came from different objects."""


#: The provenance block a :class:`SplitContext` writes into every stamp it
#: produces, so a later reader can tell WHICH of the five fold-stats files a
#: number was produced under without re-deriving it.
SPLIT_CONTEXT_KEY = "split_context"


def assert_fit_and_stamp_agree(
    stats: Mapping[str, Any], stamp: Mapping[str, Any], *, where: str = ""
) -> None:
    """[v2.16] C8.2 - raise unless the FIT and the STAMP describe the same partition.

    ``stats`` is the normalisation actually consumed (a ``norm_stats.json``
    payload); ``stamp`` is a :func:`split_fingerprint` result. The danger ADR-062
    (5) names is a caller setting the FIT from one of the five fold-stats files
    while the STAMPS still come from the default - which recreates, silently, the
    exact leak the ``stats_path`` parameter was approved to avoid.

    **This READS VALUES.** It compares ``train_folds`` and the declared fire ids,
    which is the whole reason it can fail: the check it replaces would have been
    "did the caller pass the same path twice", a proposition about a call site
    rather than about the objects, and unfalsifiable from the artifact.
    """
    fit_folds = stats.get("train_folds")
    stamp_folds = stamp.get("train_folds")
    site = f" ({where})" if where else ""
    if fit_folds != stamp_folds:
        raise SplitFitStampMismatchError(
            f"C8.2 — THE FIT AND THE SPLIT STAMP DISAGREE{site}. The normalisation was fitted "
            f"on train_folds {fit_folds}; the split stamp says train_folds {stamp_folds}. The "
            "fold partition and the normalisation are THE SAME OBJECT (ADR-062 (5)): a run that "
            "normalises with one and stamps the other reports a split it did not train under, "
            "and every held-out number in it may be measured on a fire the normalisation saw. "
            "Resolve both through ONE SplitContext."
        )
    for key in DECLARED_MEMBERSHIP_KEYS:
        declared = stats.get(key)
        if declared is None:
            continue
        if sorted(str(v) for v in declared) != sorted(str(v) for v in stamp.get(key, [])):
            raise SplitFitStampMismatchError(
                f"C8.2 — THE FIT AND THE SPLIT STAMP DISAGREE{site} on `{key}`. The "
                f"normalisation declares {sorted(str(v) for v in declared)}; the split stamp "
                f"derives {sorted(str(v) for v in stamp.get(key, []))} from the manifests "
                "on disk. Same defect as the fold mismatch above, one level finer: fold indices "
                "are not identifying across corpus versions, so two files can agree on every "
                "fold number and disagree about which fires those folds contain."
            )


def _same_stats_path(a: str | Path, b: str | Path) -> bool:
    """Do two recorded ``stats_path`` strings name the same file?

    C8.2 asks whether a run OPENED and CLOSED on one partition. That question is
    about a FILE, and two spellings of one file are not two partitions. Since
    :meth:`SplitContext.provenance` now records the repo-relative form, a run
    whose ``before`` stamp predates that change carries the absolute spelling and
    its ``after`` stamp carries the relative one; comparing the raw strings would
    report a fold rotation that never happened.

    Both sides are normalised through :func:`repo_relative`, which is the exact
    inverse of the representation change and nothing more. It does NOT compare
    contents, and it does NOT resolve symlinks: two genuinely different files
    still differ, which is the failure this guard exists to catch.
    """
    return repo_relative(a) == repo_relative(b)


@dataclass(frozen=True)
class SplitContext:
    """[v2.16] C8.2 - the fit and both stamps, resolved ONCE and inseparable.

    **The shape, and why this shape.** ADR-062 (5) approved ``stats_path`` so a
    caller can train against one of five internally-consistent fold-stats files,
    and required that the one parameter reach :func:`read_norm_stats
    <wildfire_nowcast.common.zarr_io.read_norm_stats>`, :func:`split_fingerprint`
    and :func:`assert_split_unchanged` **atomically** - *make that failure
    impossible, not merely discouraged.*

    A parameter threaded to three call sites cannot be made impossible to
    desynchronise; it can only be made easy to synchronise, and "easy" is what
    the leak was already. So the parameter is REMOVED FROM THE CALLS INSTEAD.
    This object resolves the path once, at construction, and its three
    operations - :meth:`norm_stats`, :meth:`fingerprint`, :meth:`assert_unchanged`
    - **take no path argument at all.** You cannot pass a different path to a
    method that has no path parameter, and
    ``tests/test_splits.py::test_no_split_context_operation_accepts_a_path``
    asserts that property by introspection rather than by convention.

    The second half is the belt: :func:`resolve_split_context` calls
    :func:`assert_fit_and_stamp_agree` before returning, so an inconsistent
    context cannot be CONSTRUCTED either - which covers the case the shape
    cannot, namely a stats file whose declared membership has drifted from the
    manifests it names.

    Frozen, because the whole guarantee is that the resolution point is one
    point: a context you can re-point is a parameter again.
    """

    stats_path: Path
    fires_root: Path

    # -- the three operations. NONE of them takes a path. ------------------
    def norm_stats(self) -> dict[str, Any]:
        """THE FIT: the normalisation this context's runs consume."""
        # Imported here rather than at module scope on purpose: `zarr_io` pulls
        # in xarray/zarr, and `splits` is imported by `runs.create_run_dir`,
        # i.e. by every run. C0 still holds - this is THE `read_norm_stats`,
        # not a second copy.
        from wildfire_nowcast.common.zarr_io import read_norm_stats

        return read_norm_stats(self.stats_path)

    def fingerprint(self) -> dict[str, Any]:
        """STAMP 1: the split fingerprint, plus this context's provenance."""
        stamp = split_fingerprint(fires_root=self.fires_root, stats_path=self.stats_path)
        stamp[SPLIT_CONTEXT_KEY] = self.provenance()
        return stamp

    def assert_unchanged(self, before: Mapping[str, Any]) -> dict[str, Any]:
        """STAMP 2: the end-of-run stamp, against ``before`` from THIS context."""
        prior = before.get(SPLIT_CONTEXT_KEY)
        if isinstance(prior, Mapping):
            was = str(prior.get("stats_path", ""))
            if was and not _same_stats_path(was, self.stats_path):
                raise SplitFitStampMismatchError(
                    f"C8.2 — this run OPENED under {was} and is CLOSING under "
                    f"{self.stats_path}. Two different fold partitions bracket one run, so the "
                    "'unchanged' this would assert is between two things that were never the "
                    "same. Close a run with the context that opened it."
                )
        now = assert_split_unchanged(before, fires_root=self.fires_root, stats_path=self.stats_path)
        now[SPLIT_CONTEXT_KEY] = self.provenance()
        return now

    def check_assignment(self) -> ContractReport:
        """C3/C3.1 for THIS partition - same atomicity, same reason."""
        return check_split_assignment(fires_root=self.fires_root, stats_path=self.stats_path)

    def provenance(self) -> dict[str, str]:
        """What a reader needs to know which of the five folds a number came from.

        Both paths are rendered REPO-RELATIVE (``repo_relative``). This block is
        copied verbatim into every published run artifact, and ``fires_root``
        defaults to ``fires_dir()``, which is absolute - so the default path
        through this function used to stamp the operator's home directory into
        the evidence. The identity a reader needs is *which partition*, and
        ``data/fires`` says that exactly as well as a machine-specific prefix.

        Comparisons against this block go through :func:`_same_stats_path`, so an
        artifact stamped before this change still compares equal to one stamped
        after it.
        """
        return {
            "stats_path": repo_relative(self.stats_path),
            "fires_root": repo_relative(self.fires_root),
            "clause": "C8.2 [v2.16] (ADR-062 (5))",
        }


def resolve_split_context(
    *,
    stats_path: str | Path | None = None,
    fires_root: str | Path | None = None,
) -> SplitContext:
    """THE single resolution point for ``stats_path`` (C8.2, ADR-062 (5)).

    ``None`` means the repo default, so ``resolve_split_context()`` is today's
    behaviour exactly. Pass one of the five leave-fold-out artifacts to move the
    fit and both stamps together - there is no way to move only one.

    Raises :class:`SplitFitStampMismatchError` if the resolved fit and stamp do
    not describe the same partition, so the inconsistent context never exists.
    """
    resolved_stats = Path(stats_path) if stats_path is not None else norm_stats_path()
    resolved_fires = Path(fires_root) if fires_root is not None else fires_dir()
    context = SplitContext(stats_path=resolved_stats, fires_root=resolved_fires)
    stats = read_json(resolved_stats)
    if stats is None:
        raise SplitFitStampMismatchError(
            f"C8.2 — no readable norm_stats at {resolved_stats}. A context anchored on a stats "
            "file that does not exist would stamp a partition nothing was fitted on; C-1: "
            "unverifiable is a failure, not a pass."
        )
    assert_fit_and_stamp_agree(stats, context.fingerprint(), where=f"resolving {resolved_stats}")
    return context


# --------------------------------------------------------------------------
# [v2.16] C6.3 (addition) - AN EXPECTED FALSE IS STAMPED, NOT DISCOVERED
# (ADR-062 (7))
# --------------------------------------------------------------------------


#: The key under which a run DECLARES that its ``c6_3_satisfied: false`` is
#: expected. It sits BESIDE the value and never replaces it.
C6_3_EXPECTED_FALSE_KEY = "c6_3_expected_false"

#: [ADR-062 (7)] the leave-fold-out partition of the 14 spatial blocks. Pinned
#: here so the claim "these folds are expected to report false" is DERIVED from
#: the partition rather than restated next to it - the second copy of a fact is
#: how ``CONTRACT_VERSION`` stayed stale for seven versions.
LEAVE_FOLD_OUT_BLOCKS: dict[int, tuple[int, ...]] = {
    0: (2,),
    1: (1,),
    2: (3, 9, 13),
    3: (4, 5, 6, 7, 12),
    4: (0, 8, 10, 11),
}


def folds_expected_to_fail_c6_3(
    partition: Mapping[int, Sequence[int]] | None = None,
) -> tuple[int, ...]:
    """Which folds hold out too few blocks for C6.3, DERIVED from the partition.

    **THIS RETURNS THREE FOLDS, NOT TWO.** ADR-062 (7) names folds 0 and 1 as the
    expected-false pair because they hold out one block each. Fold 2 holds out
    ``{3, 9, 13}`` - **three** blocks, which is also below
    :data:`MIN_HELDOUT_BLOCKS_FOR_G2` (4), so it reports ``c6_3_satisfied: false``
    as well. The ADR's arithmetic, not its ruling, is what differs; the ruling
    ("an expected false must be stamped, not discovered") applies unchanged and
    to one more fold than it names. Raised to the maintainer as a PROPOSAL rather
    than corrected in DECISIONS.md, which is not this lead's file.

    Deriving it is the point: had this been a hand-written ``(0, 1)`` it would
    have reproduced the slip and then outlived it.
    """
    blocks = LEAVE_FOLD_OUT_BLOCKS if partition is None else partition
    return tuple(
        sorted(fold for fold, held in blocks.items() if len(set(held)) < MIN_HELDOUT_BLOCKS_FOR_G2)
    )


def stamp_c6_3_expected_false(
    fingerprint: Mapping[str, Any], *, citation: str, why: str
) -> dict[str, Any]:
    """[v2.16] Mark a ``c6_3_satisfied: false`` as EXPECTED. **The value does not move.**

    ADR-062 (7): folds that hold out fewer than
    :data:`MIN_HELDOUT_BLOCKS_FOR_G2` blocks report ``c6_3_satisfied: false``,
    which is correct and expected for a CV matrix that pools all 14 blocks and
    adjudicates nothing gate-shaped - *but it must be STAMPED as expected, citing
    the ADR, so a future reader does not discover it as a fault.*

    **An expected-false must still BE false.** This returns a copy whose
    ``c6_3_satisfied`` is byte-identical to the input's; the declaration is a
    sibling key, never a substitute. Stamping a SATISFIED split as
    "expected-false" raises, because that is the failure this mechanism would
    otherwise create: a way to write "expected" next to a value and have the
    reader stop looking at the value.

    :raises ValueError: if ``c6_3_satisfied`` is not exactly ``False``, if the
        citation names no ADR, or if no reason is given.
    """
    value = fingerprint.get("c6_3_satisfied")
    if value is not False:
        raise ValueError(
            f"refusing to stamp `{C6_3_EXPECTED_FALSE_KEY}` on a split whose c6_3_satisfied is "
            f"{value!r}. An EXPECTED FALSE must BE false: a stamp that can be applied to a true "
            "or a missing value is a stamp that tells a reader to stop looking at the value, "
            "which is the opposite of what ADR-062 (7) asked for."
        )
    if not re.search(r"ADR-\d+", citation):
        raise ValueError(
            f"`citation` must name an ADR (e.g. 'ADR-062 (7)'); got {citation!r}. The whole "
            "purpose of the stamp is that a future reader can find the ruling that made this "
            "expected — an expectation with no provenance is an assertion."
        )
    if not why.strip():
        raise ValueError("`why` must say what makes this false EXPECTED, in this run's terms")

    stamped = dict(fingerprint)
    stamped[C6_3_EXPECTED_FALSE_KEY] = {
        "citation": citation,
        "why": why.strip(),
        "clause": "C6.3 (addition) [v2.16]",
        "min_heldout_blocks_for_g2": MIN_HELDOUT_BLOCKS_FOR_G2,
        "n_heldout_blocks": fingerprint.get("n_heldout_blocks"),
    }
    return stamped


def _c6_3_sites(payload: object, *, prefix: str = "") -> dict[str, tuple[object, object]]:
    """Every mapping carrying a ``c6_3_satisfied``, as ``{where: (value, declaration)}``."""
    found: dict[str, tuple[object, object]] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            if "c6_3_satisfied" in node:
                found[path or "<root>"] = (
                    node.get("c6_3_satisfied"),
                    node.get(C6_3_EXPECTED_FALSE_KEY),
                )
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list | tuple):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(payload, prefix)
    return found


def _add_c6_3_expectation_clauses(
    rep: ContractReport, payloads: Mapping[str, Mapping[str, Any]]
) -> None:
    """[v2.16] C6.3 (addition) - an expected false is declared, and stays false."""
    sites: dict[str, tuple[object, object]] = {}
    for name, payload in payloads.items():
        for where, pair in _c6_3_sites(payload).items():
            sites[f"{name}:{where}"] = pair
    if not sites:
        return

    flipped: list[str] = []
    undeclared: list[str] = []
    declared = 0
    for where, (value, declaration) in sorted(sites.items()):
        if declaration is not None:
            if value is not False:
                flipped.append(f"{where} declares an expected-false but c6_3_satisfied={value!r}")
                continue
            citation = (
                declaration.get("citation") if isinstance(declaration, Mapping) else declaration
            )
            if not (isinstance(citation, str) and re.search(r"ADR-\d+", citation)):
                flipped.append(f"{where} declares an expected-false citing nothing: {citation!r}")
                continue
            declared += 1
        elif value is False:
            undeclared.append(where)

    rep.add(
        "C6.3",
        "c6_3_expected_false_did_not_flip",
        not flipped,
        f"all {declared} expected-false declaration(s) sit beside a c6_3_satisfied that is "
        "still False, and each cites an ADR"
        if not flipped
        else "C6.3 HARD FAIL — an expected-false declaration is not describing a false: "
        + "; ".join(flipped[:6])
        + ". ADR-062 (7) asked that an expected false be STAMPED, not that it be resolved. A "
        "stamp that can sit beside a TRUE or a missing value tells a reader to stop looking at "
        "the value, which is strictly worse than the surprise it was meant to prevent",
        severity=SEVERITY_FAIL,
    )
    rep.add(
        "C6.3",
        "c6_3_expected_false_declared",
        not undeclared,
        f"every c6_3_satisfied=false in this artifact is declared expected ({declared})"
        if not undeclared
        else "C6.3 — c6_3_satisfied is FALSE and undeclared at: "
        + "; ".join(undeclared[:6])
        + f". A fold holding out fewer than {MIN_HELDOUT_BLOCKS_FOR_G2} blocks legitimately "
        "reports false (ADR-062 (7): folds 0, 1 and 2 of the leave-fold-out partition do), but "
        "it must be stamped with the ruling that makes it expected — `stamp_c6_3_expected_false`"
        " — so a future reader does not discover it as a fault. REPORTING tier: the value is "
        "already true of the split, and every archived artifact predates this clause",
        severity=SEVERITY_REPORTING,
    )


# --------------------------------------------------------------------------
# C8 - fingerprint extraction and checking
# --------------------------------------------------------------------------


#: Subtrees :func:`fingerprints_in` must NOT descend into, because they carry a
#: ``fingerprint`` key that is not a SPLIT fingerprint.
#:
#: THIS IS A FIXED FALSE POSITIVE IN A HARD CLAUSE, found by A12 and mine. When
#: modelling began stamping ``common_code_before``/``_after`` and
#: ``scoring_code`` - each a dict with its own ``fingerprint`` key - this walker
#: collected all three as split stamps, so ``C8.internally_consistent`` reported
#: "this artifact carries MORE THAN ONE split fingerprint" and HARD FAILED on
#: every run in the repo, including the G2 record of ADR-021, whose seven stamps
#: were in fact one split fingerprint (``4848f491e8d588fa``, agreeing in all four
#: places) plus two code fingerprints doing their job. ``C8.chain`` failed the
#: same way, for the same reason.
#:
#: The lesson is narrower than "watch your walkers": **two different quantities
#: were given the same key name in different blocks, and a syntactic scan cannot
#: tell them apart.** The scan is still right to be syntactic (C8 exists because
#: a convention was not enough); what it needs is to know which blocks are not
#: about splits. A hard clause that fires on every artifact is worse than no
#: clause, because the first thing anyone does with it is stop reading it.
_NON_SPLIT_FINGERPRINT_BLOCKS: frozenset[str] = frozenset(
    f"{family}{suffix}"
    for family in RUN_FINGERPRINT_FAMILIES
    for suffix in ("", "_before", "_after")
)


# --------------------------------------------------------------------------
# [v2.16] C8.1 - the CV-MATRIX artifact class (ADR-062 (6))
# --------------------------------------------------------------------------


#: The key an AGGREGATE artifact uses to declare the fold runs it summarises.
#:
#: **Why this key exists at all.** A full leave-fold-out matrix has FIVE split
#: fingerprints by construction - one per fold - and
#: :func:`check_run_split` hard-fails ``C8.internally_consistent`` on more than
#: one fingerprint per artifact. modelling proposed recording the five under a
#: name the checker does not read, *and flagged that doing so is exactly the move
#: C8 exists to prevent*. That refusal was upheld and generalised (ADR-062 (6)):
#: **the answer to "the checker cannot express this" is to extend the checker,
#: never to rename the field.**
#:
#: So this is NOT an exemption and the trade is deliberately unfavourable. An
#: artifact that declares ``cv_matrix`` buys its member stamps out of the
#: one-fingerprint rule and pays for it with THREE new hard clauses that do not
#: apply to anything else: the declaration must parse, the declared member count
#: must match the member runs actually present, and every member run's own stamp
#: must equal what the matrix claims about it. Today a matrix cannot be checked
#: at all; under C8.1 it is checked against the run dirs on disk.
CV_MATRIX_KEY = "cv_matrix"

#: The keys a declared matrix member MUST carry. ``run`` is what makes the claim
#: falsifiable: without a path to the fold's run dir, the declared fingerprint is
#: an unverifiable assertion about a run nobody can find, which is C-1's
#: "unverifiable is a failure" one level up.
CV_MATRIX_MEMBER_KEYS: tuple[str, str] = ("run", "split_fingerprint")

#: The key carrying the declared member count. Checked against the member runs
#: PRESENT rather than against the member entries alone: a matrix that declares
#: five folds, lists five, and has four on disk is a partial run being read as a
#: whole one - ADR-063 (3)'s expensive failure mode, made mechanical.
CV_MATRIX_COUNT_KEY = "n_members"


@dataclass(frozen=True)
class CvMatrixMember:
    """One fold of a CV matrix, as the AGGREGATE declares it.

    ``fingerprint`` is a CLAIM about ``run``, never a substitute for it. C8.1
    reads the run dir and fails if the two disagree.
    """

    label: str
    run: str
    fingerprint: str


@dataclass(frozen=True)
class DeclaredCvMatrix:
    """A parsed ``cv_matrix`` declaration, with its own defects listed.

    ``problems`` are strings and never exceptions, for the same reason
    :func:`read_json` never raises: a malformed artifact must fail its own clause
    rather than abort the punch list.
    """

    n_members: int | None
    members: tuple[CvMatrixMember, ...]
    problems: tuple[str, ...]

    @property
    def fingerprints(self) -> tuple[str, ...]:
        return tuple(m.fingerprint for m in self.members)


def _member_from(label: str, node: object) -> tuple[CvMatrixMember | None, list[str]]:
    problems: list[str] = []
    if not isinstance(node, Mapping):
        return None, [f"member {label!r} is not an object: {node!r}"]
    values: dict[str, str] = {}
    for key in CV_MATRIX_MEMBER_KEYS:
        value = node.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(
                f"member {label!r} does not declare a non-empty `{key}`: {value!r}. C8.1 "
                "requires both, because a fingerprint with no run dir cannot be checked "
                "against anything"
            )
            continue
        values[key] = value.strip()
    if problems:
        return None, problems
    member = CvMatrixMember(label=label, run=values["run"], fingerprint=values["split_fingerprint"])
    return member, []


def declared_cv_matrix(payload: Mapping[str, Any]) -> DeclaredCvMatrix | None:
    """Parse a ``cv_matrix`` declaration, or return ``None`` if there is none.

    Accepts ``members`` as a mapping ``{label: member}`` (the natural shape for
    folds) or as a list of member objects, in which case a member's own
    ``label``/``fold`` key names it and the index is the fallback. Reading both
    shapes is the same reasoning as :data:`FINGERPRINT_KEYS`: a checker that only
    understands its own writer's shape is not a checker.
    """
    node = payload.get(CV_MATRIX_KEY)
    if node is None:
        return None
    if not isinstance(node, Mapping):
        return DeclaredCvMatrix(
            None, (), (f"`{CV_MATRIX_KEY}` is not an object: {type(node).__name__}",)
        )

    problems: list[str] = []
    raw_count = node.get(CV_MATRIX_COUNT_KEY)
    n_members: int | None = None
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        problems.append(
            f"`{CV_MATRIX_KEY}.{CV_MATRIX_COUNT_KEY}` is absent or not an int: {raw_count!r}. "
            "The count is the whole point of declaring it — it is what makes a MISSING fold "
            "detectable rather than merely absent"
        )
    else:
        n_members = int(raw_count)

    raw_members = node.get("members")
    members: list[CvMatrixMember] = []
    if isinstance(raw_members, Mapping):
        entries: list[tuple[str, object]] = [(str(k), v) for k, v in raw_members.items()]
    elif isinstance(raw_members, list):
        entries = []
        for i, item in enumerate(raw_members):
            label = str(i)
            if isinstance(item, Mapping):
                for key in ("label", "fold", "name"):
                    if isinstance(item.get(key), (str, int)):
                        label = str(item[key])
                        break
            entries.append((label, item))
    else:
        entries = []
        problems.append(
            f"`{CV_MATRIX_KEY}.members` is absent or is neither an object nor a list: "
            f"{type(raw_members).__name__}"
        )

    seen: set[str] = set()
    for label, item in entries:
        if label in seen:
            problems.append(f"member label {label!r} is declared more than once")
        seen.add(label)
        member, member_problems = _member_from(label, item)
        problems.extend(member_problems)
        if member is not None:
            members.append(member)

    return DeclaredCvMatrix(n_members, tuple(members), tuple(problems))


def _resolve_member_run(run: str, root: Path) -> Path:
    """Where a declared member's run dir is, given the runs root.

    Absolute paths are honoured; anything containing ``runs/`` is taken relative
    to the runs root from that segment on; a bare name is a directory under it.
    """
    path = Path(run)
    if path.is_absolute():
        return path
    if "runs/" in run:
        return root / run.split("runs/", 1)[1]
    return root / run


def fingerprints_in(payload: object, *, prefix: str = "") -> dict[str, str]:
    """Every SPLIT fingerprint in a run artifact, keyed by its dotted location.

    Walks nested dicts/lists so a fingerprint stamped in ``scope``,
    ``split_before``, ``split_after`` or at the root is all found by one pass.
    Code-fingerprint blocks are skipped - see
    :data:`_NON_SPLIT_FINGERPRINT_BLOCKS`.

    [v2.16] ``cv_matrix`` is skipped here and read by :func:`declared_cv_matrix`
    instead. This is the one place the distinction matters, so it is stated
    plainly: the member stamps are NOT hidden from the checker, they are moved to
    a STRICTER checker. Collecting them here would make every leave-fold-out
    matrix hard-fail ``C8.internally_consistent`` for having the five
    fingerprints it is defined to have, while checking none of them against the
    fold runs on disk.
    """
    found: dict[str, str] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                here = f"{path}.{key}" if path else str(key)
                if key in _NON_SPLIT_FINGERPRINT_BLOCKS or key == CV_MATRIX_KEY:
                    continue
                if key in FINGERPRINT_KEYS:
                    if isinstance(value, str) and value:
                        found[here] = value
                        continue
                    if isinstance(value, Mapping):
                        fp = value.get("fingerprint")
                        if isinstance(fp, str) and fp:
                            # Recorded once, at the block. Descending would also
                            # match its own `fingerprint` key and report one
                            # stamp as two, inflating every count in the report.
                            found[here] = fp
                            continue
                walk(value, here)
        elif isinstance(node, list | tuple):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(payload, prefix)
    return found


def _referenced_run_dirs(payload: Mapping[str, Any], root: Path) -> list[Path]:
    """Run directories this artifact says it consumed (checkpoints, matrices).

    Found by scanning every string value for a path under ``runs/``. Deliberately
    syntactic: the alternative is a convention every lead must remember, and C8
    exists precisely because a convention was not enough.
    """
    hits: list[Path] = []

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            for token in node.replace("=", " ").replace(",", " ").split():
                if "runs/" not in token:
                    continue
                tail = token.split("runs/", 1)[1].strip("'\"").split("/")[0]
                candidate = root / tail
                if tail and candidate.is_dir() and candidate not in hits:
                    hits.append(candidate)

    walk(payload)
    return hits


def _artifact_fingerprints(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in sorted(p.name for p in run_dir.glob("*.json")):
        payload = read_json(run_dir / name)
        if payload is None:
            continue
        for where, fp in fingerprints_in(payload).items():
            out[f"{run_dir.name}/{name}:{where}"] = fp
    return out


def check_run_split(
    target: str | Path | Mapping[str, Any],
    *,
    current: Mapping[str, Any] | None = None,
    runs_root: str | Path | None = None,
    follow_references: bool = True,
) -> ContractReport:
    """Assert C8 for one run directory or one result artifact.

    Clauses
    -------
    ``C8.stamped``            (fail)      the artifact carries a fingerprint.
    ``C8.internally_consistent`` (fail)   every fingerprint inside it agrees -
                                          this is the literal train-vs-eval
                                          mismatch, e.g. ``split_before`` vs
                                          ``split_after``.
    ``C8.chain``              (fail)      every run directory this artifact says
                                          it consumed carries the same
                                          fingerprint. A model trained under
                                          split A and evaluated under split B is
                                          the ADR-015 defect exactly.
    ``C8.matches_current``    (reporting) the fingerprint equals the split on
                                          disk NOW.
    ``C8.code_agrees_across_run`` (fail)  [v2.11, C-4.2] where a code fingerprint
                                          IS sampled at both ends, the two ends
                                          must agree. Disagreement means the
                                          numbers were computed partly before and
                                          partly after an edit.
    ``C8.code_sampled_both_ends`` (reporting) [v2.11, C-4.2] every code
                                          fingerprint family is sampled BEFORE as
                                          well as after - see below.
    ``C8.1 cv_matrix_*``      (fail)      [v2.16, ADR-062 (6)] if this artifact
                                          declares a CV matrix, the declaration
                                          must parse, the declared member count
                                          must match the member runs present, and
                                          every member run's own stamp must equal
                                          what the matrix claims about it. See
                                          :func:`_add_cv_matrix_clauses`.

    Why the last one is ``reporting`` and not ``fail``: an archived result was
    internally consistent when it was produced, and ADR-014 approved a fold
    rebalance that will legitimately move the fingerprint. What must never
    happen is *quoting* a stale-split number next to a current one, and that is
    exactly what the reporting tier is for (C-1). The train-vs-eval mismatch,
    which the contract names as a HARD FAIL, is ``internally_consistent`` and
    ``chain``.

    Why ``code_sampled_both_ends`` is ``reporting`` and not ``fail``, stated
    because C-1's own corollary points the other way (an unevaluable guard is
    normally a hard fail). Every artifact in ``runs/`` today predates C-4.2, and
    the two that stamp ``scoring_code`` stamp it ONCE, at payload construction.
    Making this hard would retroactively fail the G2 record of ADR-021 on a
    bookkeeping property rather than on any measurement - and whether that record
    must be re-run is an maintainer ruling, not a checker's. Two things narrow
    the exposure and are worth stating rather than leaving to be inferred: that
    record's ``common_code`` IS sampled at both ends and agrees, and
    ``common/iou_terms.py`` - where the G2 gate criterion is actually computed -
    is inside ``common_code``. So the un-verified span is ``eval/`` and
    ``model/``, not the criterion. PROPOSAL for the next bump: promote to ``fail``
    once a run stamps both ends, at which point the reporting tier has served its
    purpose and the clause can bite.
    """
    root = Path(runs_root) if runs_root else runs_dir()

    if isinstance(target, Mapping):
        label, payloads = "<payload>", {"<payload>": dict(target)}
    else:
        path = Path(target)
        label = str(path)
        if path.is_dir():
            payloads = {}
            for candidate in sorted(path.glob("*.json")):
                payload = read_json(candidate)
                if payload is not None:
                    payloads[candidate.name] = payload
        else:
            payload = read_json(path)
            payloads = {} if payload is None else {path.name: payload}

    rep = ContractReport(target=label)
    if not payloads:
        rep.add("C8", "readable", False, f"no readable JSON artifact at {label}")
        return rep
    rep.add("C8", "readable", True, f"{len(payloads)} JSON artifact(s) read")

    found: dict[str, str] = {}
    for name, payload in payloads.items():
        for where, fp in fingerprints_in(payload).items():
            found[f"{name}:{where}"] = fp

    # [v2.16] C8.1 - a declared CV matrix, if this artifact is one.
    matrices: dict[str, DeclaredCvMatrix] = {}
    for name, payload in payloads.items():
        matrix = declared_cv_matrix(payload)
        if matrix is not None:
            matrices[name] = matrix
    member_fingerprints = sorted({fp for m in matrices.values() for fp in m.fingerprints})
    member_dirs = {
        _resolve_member_run(member.run, root)
        for matrix in matrices.values()
        for member in matrix.members
    }

    rep.add(
        "C8",
        "stamped",
        bool(found) or bool(matrices),
        f"split fingerprint stamped in {len(found)} location(s): {sorted(set(found.values()))}"
        if found
        else f"a CV matrix declaring {len(member_fingerprints)} member fingerprint(s) "
        f"{member_fingerprints} (C8.1)"
        if matrices
        else "NO split fingerprint anywhere in this run. C8: every run stamps one and every "
        "reported number carries it. Without it, a number cannot be shown to have been "
        "produced under the split it claims — and the split HAS moved under a running "
        "experiment before (ADR-015)",
    )
    if not (found or matrices):
        return rep

    distinct = sorted(set(found.values()))
    if matrices and not found:
        # A pure aggregate stamps nothing of its own; there is no train-vs-eval
        # pair inside it to disagree. Its members' agreement with the fold runs
        # on disk is C8.1's job and is strictly harder than this clause. Stated
        # rather than skipped, because a clause that silently stops applying is
        # how a hard gate becomes decoration.
        rep.add(
            "C8",
            "internally_consistent",
            True,
            f"CV matrix: this aggregate carries no split stamp of its own, so there is no "
            f"train-vs-eval pair here to disagree. Its {len(member_fingerprints)} member "
            "fingerprint(s) are checked against the fold run dirs by C8.1, which is stricter "
            "than this clause (ADR-062 (6))",
            severity=SEVERITY_FAIL,
        )
    else:
        rep.add(
            "C8",
            "internally_consistent",
            len(distinct) == 1,
            f"all {len(found)} stamps agree ({distinct[0]})"
            if len(distinct) == 1
            else "C8 HARD FAIL — this artifact carries MORE THAN ONE split fingerprint: "
            + "; ".join(f"{k}={v}" for k, v in sorted(found.items()))
            + ". The split used for TRAINING differs from the split used for EVALUATION, so "
            "fires that were trained on are being scored as held-out. Discard the numbers and "
            "re-run against a frozen split",
            severity=SEVERITY_FAIL,
        )

    _add_cv_matrix_clauses(rep, matrices, root)
    _add_c6_3_expectation_clauses(rep, payloads)

    if follow_references:
        chain: dict[str, str] = {}
        for payload in payloads.values():
            for ref in _referenced_run_dirs(payload, root):
                # Declared members are EXCLUDED here and checked by C8.1 against
                # what the matrix claims about each one individually. Left in,
                # every matrix would hard-fail C8.chain for holding the five
                # different fingerprints it exists to hold - and would do so
                # WITHOUT ever comparing a member's stamp to its own claim.
                if ref in member_dirs:
                    continue
                chain.update(_artifact_fingerprints(ref))
        accepted = set(distinct) | set(member_fingerprints)
        mismatched = {k: v for k, v in chain.items() if v not in accepted}
        if chain:
            rep.add(
                "C8",
                "chain",
                not mismatched,
                f"all {len(chain)} stamp(s) in the {len({k.split('/')[0] for k in chain})} "
                "referenced run(s) agree with this artifact"
                if not mismatched
                else "C8 HARD FAIL — this run consumed artifacts stamped with a DIFFERENT "
                "split: "
                + "; ".join(f"{k}={v}" for k, v in sorted(mismatched.items())[:6])
                + f" vs this artifact's {sorted(accepted)}. A checkpoint trained under one split "
                "and evaluated under another produces held-out numbers on trained-on fires",
                severity=SEVERITY_FAIL,
            )

    _add_code_fingerprint_clauses(rep, payloads)

    now = dict(current) if current is not None else split_fingerprint()
    ok = now.get("fingerprint") in set(distinct) | set(member_fingerprints)
    rep.add(
        "C8",
        "matches_current",
        ok,
        f"fingerprint matches the split on disk now ({now.get('fingerprint')}; "
        f"{now.get('n_fires')} fires, train folds {now.get('train_folds')})"
        if ok
        else "this artifact was produced under split "
        f"{sorted(set(distinct) | set(member_fingerprints))}, but the split on disk is now "
        f"{now.get('fingerprint')} ({now.get('n_fires')} fires, train folds "
        f"{now.get('train_folds')}, {now.get('n_heldout_blocks')} held-out blocks). The numbers "
        "were internally consistent when produced and are NOT invalidated by this, but they "
        "measure a different split from any number produced today: re-run before quoting them "
        "beside a current result (C-1)",
        severity=SEVERITY_REPORTING,
    )
    return rep


def _add_cv_matrix_clauses(
    rep: ContractReport, matrices: Mapping[str, DeclaredCvMatrix], root: Path
) -> None:
    """[v2.16] C8.1 - the three clauses a CV matrix pays for its exemption with.

    ``cv_matrix_well_formed``      (fail) the declaration parses: an int
                                          ``n_members`` and members each carrying
                                          a ``run`` and a ``split_fingerprint``.
    ``cv_matrix_member_count``     (fail) the declared count equals the number of
                                          member entries AND the number of member
                                          run dirs actually on disk.
    ``cv_matrix_member_stamps``    (fail) every member run's OWN stamp equals what
                                          the matrix claims about it.
    ``cv_matrix_members_distinct`` (reporting) two members under one fingerprint.

    Why the last one is REPORTING and not FAIL, stated because the other three
    are hard and the asymmetry should not have to be inferred: a leave-fold-out
    matrix's five folds have five different splits, so duplicates there are a
    real defect - but the mandatory **null rung** (ADR-062 (4): retrain the same
    arm at a second seed on every fold) is by design the SAME split twice, and a
    hard clause would forbid the control that makes the matrix readable. So it is
    surfaced, loudly, and does not block.
    """
    if not matrices:
        return

    problems: list[str] = []
    count_problems: list[str] = []
    stamp_problems: list[str] = []
    checked = 0
    present_total = 0
    declared_total = 0
    by_fingerprint: dict[str, list[str]] = {}

    for name, matrix in sorted(matrices.items()):
        problems.extend(f"{name}: {p}" for p in matrix.problems)
        declared_total += len(matrix.members)
        present: list[CvMatrixMember] = []
        for member in matrix.members:
            by_fingerprint.setdefault(member.fingerprint, []).append(f"{name}:{member.label}")
            run_dir = _resolve_member_run(member.run, root)
            if not run_dir.is_dir():
                count_problems.append(
                    f"{name}: member {member.label!r} declares run {member.run!r}, and there is "
                    f"no such run directory at {run_dir}"
                )
                continue
            present.append(member)
            stamps = _artifact_fingerprints(run_dir)
            if not stamps:
                stamp_problems.append(
                    f"{name}: member {member.label!r} run {run_dir.name} carries NO split "
                    f"fingerprint at all, so the matrix's claim of {member.fingerprint} about it "
                    "is unverifiable (C-1: unverifiable is a failure, not a pass)"
                )
                continue
            checked += 1
            disagreeing = {k: v for k, v in stamps.items() if v != member.fingerprint}
            if disagreeing:
                stamp_problems.append(
                    f"{name}: member {member.label!r} is CLAIMED to be {member.fingerprint} but "
                    f"{run_dir.name} stamps "
                    + "; ".join(f"{k}={v}" for k, v in sorted(disagreeing.items())[:4])
                )
        present_total += len(present)
        if matrix.n_members is not None:
            if matrix.n_members != len(matrix.members):
                count_problems.append(
                    f"{name}: declares {CV_MATRIX_COUNT_KEY}={matrix.n_members} but lists "
                    f"{len(matrix.members)} usable member(s)"
                )
            if matrix.n_members != len(present):
                count_problems.append(
                    f"{name}: declares {CV_MATRIX_COUNT_KEY}={matrix.n_members} but "
                    f"{len(present)} member run(s) are present on disk"
                )

    rep.add(
        "C8.1",
        "cv_matrix_well_formed",
        not problems,
        f"the CV matrix declaration in {len(matrices)} artifact(s) parses: "
        f"{declared_total} member(s), each with a run and a split fingerprint"
        if not problems
        else "C8.1 HARD FAIL — the CV matrix declaration does not parse: "
        + "; ".join(problems[:6])
        + ". `cv_matrix` is the key that buys a matrix out of C8's one-fingerprint rule; a "
        "declaration the checker cannot read buys the exemption and pays nothing, which is the "
        "renamed-field move ADR-062 (6) refused",
        severity=SEVERITY_FAIL,
    )
    rep.add(
        "C8.1",
        "cv_matrix_member_count",
        not count_problems,
        f"every declared member run is present: {present_total} of {declared_total}"
        if not count_problems
        else "C8.1 HARD FAIL — the declared member count does not match the member runs "
        "present: "
        + "; ".join(count_problems[:6])
        + ". A matrix missing a fold is a PARTIAL result that reads as a whole one, and the "
        "criterion it feeds is a count over folds (>= 11/14): a fold that silently does not "
        "exist changes the denominator without changing the report",
        severity=SEVERITY_FAIL,
    )
    rep.add(
        "C8.1",
        "cv_matrix_member_stamps",
        not stamp_problems,
        f"all {checked} member run(s) stamp exactly the fingerprint the matrix claims for them"
        if not stamp_problems
        else "C8.1 HARD FAIL — a member run's OWN stamp differs from what this matrix claims "
        "about it: "
        + "; ".join(stamp_problems[:6])
        + ". The claim is what a reader trusts and the run dir is what was actually trained, so "
        "a disagreement means the matrix is describing a split that produced none of its "
        "numbers — the ADR-015 defect with the aggregate as its subject",
        severity=SEVERITY_FAIL,
    )
    duplicated = {fp: where for fp, where in by_fingerprint.items() if len(where) > 1}
    rep.add(
        "C8.1",
        "cv_matrix_members_distinct",
        not duplicated,
        f"the {len(by_fingerprint)} member fingerprint(s) are distinct — every fold ran under "
        "its own split"
        if not duplicated
        else "C8.1 — two or more members share one split fingerprint: "
        + "; ".join(f"{fp} <- {sorted(w)}" for fp, w in sorted(duplicated.items())[:4])
        + ". In a leave-fold-out matrix that means two folds trained on the same partition, so "
        "one of them held out nothing new. LEGITIMATE for a null rung (the same arm at a second "
        "seed IS the same split by design, ADR-062 (4)) — which is why this reports rather than "
        "blocks. Say which it is in the artifact",
        severity=SEVERITY_REPORTING,
    )


def _add_code_fingerprint_clauses(
    rep: ContractReport, payloads: Mapping[str, Mapping[str, Any]]
) -> None:
    """[v2.11] C-4.2 - a fingerprint must be sampled BEFORE *and* AFTER the run.

    The defect, in the clause's own words: *a fingerprint sampled after the fact
    records the wrong code precisely in the case it was built to catch.* If
    ``eval/metrics.py`` is rewritten nine minutes into an 876-second run, a hash
    taken at payload construction is the hash of the NEW file - so the artifact
    confidently records code that produced none of its numbers, and the one
    stamp that exists reads as reassurance.
    """
    ends_by_artifact: dict[str, dict[str, dict[str, str]]] = {
        name: code_fingerprint_ends(payload)
        for name, payload in payloads.items()
        if code_fingerprint_ends(payload)
    }
    if not ends_by_artifact:
        return

    def _tally(wanted: tuple[str, ...]) -> tuple[dict[str, str], dict[str, str], int]:
        disagreed: dict[str, str] = {}
        unpaired: dict[str, str] = {}
        paired = 0
        for name, families in sorted(ends_by_artifact.items()):
            for family, ends in sorted(families.items()):
                if family not in wanted:
                    continue
                before, after = ends.get("before"), ends.get("after")
                if before is not None and after is not None:
                    paired += 1
                    if before != after:
                        disagreed[f"{name}:{family}"] = f"before={before} after={after}"
                else:
                    seen = ends.get("unpaired") or before or after
                    which = "before-only" if before else ("after-only" if after else "unpaired")
                    unpaired[f"{name}:{family}"] = f"{which} ({seen})"
        return disagreed, unpaired, paired

    disagreed, unpaired, paired = _tally(CODE_FINGERPRINT_FAMILIES)
    _add_environment_clauses(rep, *_tally((ENVIRONMENT_FINGERPRINT_FAMILY,)))
    if not (disagreed or unpaired or paired):
        return

    rep.add(
        "C8",
        "code_agrees_across_run",
        not disagreed,
        f"all {paired} both-ended code fingerprint(s) agree across the run"
        if not disagreed
        else "C-4.2 HARD FAIL — code MOVED DURING THIS RUN: "
        + "; ".join(f"{k} {v}" for k, v in sorted(disagreed.items()))
        + ". The numbers in this artifact were computed partly before and partly after that "
        "edit, so they are not all products of one implementation. Discard and re-run against "
        "frozen code (C-4)",
        severity=SEVERITY_FAIL,
    )
    rep.add(
        "C8",
        "code_sampled_both_ends",
        not unpaired,
        f"every code fingerprint family is sampled at BOTH ends ({paired} pair(s))"
        if not unpaired
        else "C-4.2 — sampled at ONE end only: "
        + "; ".join(f"{k} {v}" for k, v in sorted(unpaired.items()))
        + ". A fingerprint taken at payload construction is the hash of the code as it stands "
        "AFTER the run, so it records the wrong version precisely in the case it exists to "
        "catch, and it reads as reassurance while doing so. Sample both ends and compare",
        severity=SEVERITY_REPORTING,
    )


def _add_environment_clauses(
    rep: ContractReport,
    disagreed: Mapping[str, str],
    unpaired: Mapping[str, str],
    paired: int,
) -> None:
    """[v2.12] C-4.3 - the interpreter environment must not move across a run.

    Its own check ids, not the ``code_*`` ones. What this catches, concretely:
    a ``pip install`` or a ``uv sync`` landing mid-run, an editable install being
    rebuilt, or the ELMFIRE binary being recompiled between the model's numbers
    and the baseline's. C-4 could not name any of those because it enumerated
    FILES; data read its intent correctly anyway (ADR-024) and this is that
    reading made mechanical.

    Tiering follows C-4.2 exactly, and for the same reason rather than by
    analogy: **every artifact in ``runs/`` predates this clause and stamps no
    environment at all**, so a hard "sampled both ends" would fail the whole
    archive on a bookkeeping property rather than on a measurement. Disagreement
    between two stamps that DO exist is a hard fail, because that is a
    measurement.
    """
    if not (disagreed or unpaired or paired):
        return
    rep.add(
        "C8",
        "environment_agrees_across_run",
        not disagreed,
        f"the interpreter environment is unchanged across the run ({paired} pair(s))"
        if not disagreed
        else "C-4.3 HARD FAIL — the ENVIRONMENT MOVED DURING THIS RUN: "
        + "; ".join(f"{k} {v}" for k, v in sorted(disagreed.items()))
        + ". A package, lockfile or system tool changed while these numbers were being "
        "produced, so they are not all products of one environment. This is the shared-state "
        "change C-4 exists to stop; the enumerated file list simply never named it (ADR-024). "
        "Discard and re-run against a frozen environment",
        severity=SEVERITY_FAIL,
    )
    rep.add(
        "C8",
        "environment_sampled_both_ends",
        not unpaired,
        f"the environment is sampled at BOTH ends ({paired} pair(s))"
        if not unpaired
        else "C-4.3 — sampled at ONE end only: "
        + "; ".join(f"{k} {v}" for k, v in sorted(unpaired.items()))
        + ". Same defect C-4.2 names for code: a stamp taken at payload construction records "
        "the environment as it stands AFTER the run, which is the wrong one precisely in the "
        "case it exists to catch",
        severity=SEVERITY_REPORTING,
    )


def check_split_chain(
    train_artifact: str | Path | Mapping[str, Any],
    eval_artifact: str | Path | Mapping[str, Any],
) -> ContractReport:
    """C8 across two artifacts: the TRAIN run and the EVAL run must agree.

    This is the contract's own sentence, made executable: *a mismatch between
    the split used for TRAINING and the split used for EVALUATION is a HARD
    FAIL.*
    """
    rep = ContractReport(target=f"{train_artifact} -> {eval_artifact}")
    sides: dict[str, dict[str, str]] = {}
    for role, target in (("train", train_artifact), ("eval", eval_artifact)):
        if isinstance(target, Mapping):
            sides[role] = fingerprints_in(dict(target))
            continue
        path = Path(target)
        if path.is_dir():
            sides[role] = _artifact_fingerprints(path)
        else:
            payload = read_json(path)
            sides[role] = fingerprints_in(payload) if payload else {}

    for role in ("train", "eval"):
        rep.add(
            "C8",
            f"stamped_{role}",
            bool(sides[role]),
            f"{role} artifact carries a fingerprint"
            if sides[role]
            else f"{role} artifact carries NO split fingerprint, so the comparison C8 requires "
            "cannot be made at all. C-1: unverifiable is a failure, not a pass",
        )
    if not (sides["train"] and sides["eval"]):
        return rep

    train_fps = sorted(set(sides["train"].values()))
    eval_fps = sorted(set(sides["eval"].values()))
    ok = train_fps == eval_fps and len(train_fps) == 1
    rep.add(
        "C8",
        "train_eval_match",
        ok,
        f"training and evaluation ran under the same split ({train_fps[0]})"
        if ok
        else f"C8 HARD FAIL — training ran under {train_fps} and evaluation under {eval_fps}. "
        "Four fires crossed from TRAIN to HELD-OUT exactly this way while a matrix was "
        "training (ADR-015); every tensor was individually conformant throughout, so no "
        "per-tensor check could see it. Discard the evaluation",
    )
    return rep


# --------------------------------------------------------------------------
# C3.1 - the cross-fire fold clauses
# --------------------------------------------------------------------------


def check_split_assignment(
    *,
    fires_root: str | Path | None = None,
    stats_path: str | Path | None = None,
    manifests: Sequence[Mapping[str, Any]] | None = None,
) -> ContractReport:
    """Assert C3.1 across ALL fires: one block never straddles two folds.

    C3.1: *buffered domains overlap ... overlapping fires MUST share a fold -
    leave-one-fire-out across an overlapping pair is landscape leakage.* Like
    C8, this is invisible to every per-fire check: both manifests are perfectly
    conformant, and only their JOIN is wrong.
    """
    root = Path(fires_root) if fires_root else fires_dir()
    rep = ContractReport(target=str(root))

    if manifests is None:
        loaded: list[Mapping[str, Any]] = []
        bad: list[str] = []
        for path in sorted(root.glob("*/manifest.json")):
            man = read_json(path)
            if man is None:
                bad.append(path.parent.name)
            else:
                loaded.append(man)
        rep.add(
            "C3.1",
            "manifests_readable",
            not bad,
            f"{len(loaded)} manifest(s) read"
            if not bad
            else f"unreadable manifests: {bad} — a fire missing from the split is a fire whose "
            "fold nobody can audit",
        )
        manifests = loaded
    if not manifests:
        rep.add("C3.1", "any_fires", False, f"no fire manifests under {root}")
        return rep

    typed: list[tuple[str, int, int]] = []
    malformed: list[str] = []
    # Named `entry` rather than reusing `man`: the earlier `man` is the
    # Optional result of `read_json`, and rebinding it here would have this
    # loop's element inherit that `| None` for no reason.
    for entry in manifests:
        try:
            typed.append(
                (str(entry["fire_id"]), int(entry["cv_fold"]), int(entry["spatial_block_id"]))
            )
        except Exception:
            malformed.append(str(entry.get("fire_id", "<no fire_id>")))
    rep.add(
        "C3.1",
        "fold_and_block_typed",
        not malformed,
        f"all {len(typed)} fires declare an int cv_fold and spatial_block_id"
        if not malformed
        else f"fires without a usable cv_fold/spatial_block_id: {malformed}",
    )

    by_block: dict[int, set[int]] = {}
    fires_by_block: dict[int, list[str]] = {}
    for fire_id, fold, block in typed:
        by_block.setdefault(block, set()).add(fold)
        fires_by_block.setdefault(block, []).append(fire_id)
    straddling = {b: sorted(f) for b, f in by_block.items() if len(f) > 1}
    rep.add(
        "C3.1",
        "block_maps_to_one_fold",
        not straddling,
        f"each of the {len(by_block)} spatial blocks sits entirely in one fold "
        "(no landscape straddles the train/held-out boundary)"
        if not straddling
        else "C3.1 VIOLATED — these spatial blocks straddle more than one fold: "
        + "; ".join(
            f"block {b}: folds {folds} ({', '.join(fires_by_block[b])})"
            for b, folds in sorted(straddling.items())
        )
        + ". Buffered domains overlap, so two fires in one block share landscape: scoring one "
        "while training on the other is leakage that every per-fire check passes",
    )

    fp = split_fingerprint(fires_root=root, stats_path=stats_path)

    # ------------------------------------------------------------------
    # [A14] C3 - the DECLARED membership, read from norm_stats.json (ADR-038 (6))
    # ------------------------------------------------------------------
    declared = declared_split_membership(stats_path or norm_stats_path())
    rep.add(
        "C3",
        "norm_stats_declares_fire_ids",
        declared["present"],
        f"norm_stats.json declares {len(declared['train'])} train and "
        f"{len(declared['heldout'])} held-out fires BY ID"
        if declared["present"]
        else "C3 VIOLATED — "
        + "; ".join(declared["problems"])
        + ". A MISSING key is a FAILURE and not a skip: an absent check that reports green is "
        "the failure mode this project has hit most often (an all-NaN channel passed 56 checks; "
        "C1.5 sat unimplemented for two contract versions; the train/held-out intersection was "
        "checked against a partition that could not overlap)",
    )
    overlap = sorted(set(declared["train"]) & set(declared["heldout"]))
    rep.add(
        "C3",
        "train_heldout_disjoint",
        declared["present"] and not overlap,
        f"the DECLARED train ({len(declared['train'])} fires) and held-out "
        f"({len(declared['heldout'])} fires) sets are disjoint"
        if declared["present"] and not overlap
        else (
            f"C3 VIOLATED — {len(overlap)} fire(s) are declared in BOTH train and held-out: "
            f"{overlap}. Every held-out number covering "
            f"{'these fires' if len(overlap) > 1 else 'this fire'} is measured on a fire the "
            "normalisation was fitted on. Leakage invalidates every gate, so this is HARD"
            if overlap
            else "membership is undeclared, so disjointness is UNVERIFIABLE — C-1: "
            "unverifiable is a hard fail, because an unevaluable guard is strictly worse "
            "than a declared-weak one"
        ),
    )
    # The check that can actually catch a STALE file: the declared ids against
    # the fold assignment on disk. ADR-038 (1) measured this exact hazard -
    # `2020_july_complex` carried 9.76% of train mass and moved train -> held-out,
    # and a norm_stats.json still listing it as train would have baked its
    # statistics into the normalisation of a fire it was then scored on. Silent,
    # and undetectable in any downstream artifact.
    if declared["present"]:
        drift = {
            "train_only_in_norm_stats": sorted(set(declared["train"]) - set(fp["train_fire_ids"])),
            "train_only_in_manifests": sorted(set(fp["train_fire_ids"]) - set(declared["train"])),
            "heldout_only_in_norm_stats": sorted(
                set(declared["heldout"]) - set(fp["heldout_fire_ids"])
            ),
            "heldout_only_in_manifests": sorted(
                set(fp["heldout_fire_ids"]) - set(declared["heldout"])
            ),
        }
        moved = {k: v for k, v in drift.items() if v}
        rep.add(
            "C3",
            "declared_membership_matches_manifests",
            not moved,
            f"norm_stats.json's declared membership matches all {fp['n_fires']} manifests' "
            "cv_fold on disk"
            if not moved
            else "C3 VIOLATED — norm_stats.json's declared membership DISAGREES with the "
            f"cv_fold on disk: {moved}. The stats were computed over a different set of fires "
            "than the split now says are train. This is ADR-038 (1) exactly, where one fire "
            "carrying 9.76% of train mass moved train -> held-out",
        )

    block_overlap = sorted(set(fp["train_blocks"]) & set(fp["heldout_blocks"]))
    rep.add(
        "C3.1",
        "train_heldout_blocks_disjoint",
        not block_overlap,
        f"train blocks {fp['train_blocks']} and held-out blocks {fp['heldout_blocks']} "
        "share no landscape"
        if not block_overlap
        else f"C3.1 VIOLATED — blocks {block_overlap} appear in BOTH train and held-out. The "
        "held-out score is measured on a landscape the model trained on",
    )
    rep.add(
        "C3.1",
        "heldout_block_coverage",
        fp["n_heldout_blocks"] >= MIN_HELDOUT_BLOCKS_FOR_G2,
        f"{fp['n_heldout_blocks']} distinct held-out spatial blocks "
        f"{fp['heldout_blocks']} — C6.3 satisfied for G2"
        if fp["n_heldout_blocks"] >= MIN_HELDOUT_BLOCKS_FOR_G2
        else f"only {fp['n_heldout_blocks']} distinct held-out block(s) "
        f"{fp['heldout_blocks']}; C6.3 requires >= {MIN_HELDOUT_BLOCKS_FOR_G2} for G2. More "
        "fires from the same block are the same evidence with false confidence",
        severity=SEVERITY_REPORTING,
    )
    return rep


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.common.splits",
        description=(
            "C8 split fingerprint + C3.1 cross-fire fold checks. With no arguments, print "
            "the current split. Exit code 0 = conformant, 1 = violation."
        ),
    )
    p.add_argument("--run", type=Path, default=None, help="a run directory or results.json")
    p.add_argument("--train-run", type=Path, default=None, help="TRAIN side of a C8 comparison")
    p.add_argument("--eval-run", type=Path, default=None, help="EVAL side of a C8 comparison")
    p.add_argument("--json", action="store_true", help="emit JSON on stdout")
    p.add_argument("-v", "--verbose", action="store_true", help="print passing checks too")
    p.add_argument(
        "--for-reporting",
        action="store_true",
        help="promote reporting-gate clauses (e.g. C8.matches_current) to hard failures",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    current = split_fingerprint()

    rep = check_split_assignment()
    if args.run is not None:
        rep.extend(check_run_split(args.run, current=current))
        rep.target = str(args.run)
    if args.train_run is not None and args.eval_run is not None:
        rep.extend(check_split_chain(args.train_run, args.eval_run))

    if args.json:
        print(json.dumps({"current_split": current, "report": rep.to_dict()}, indent=2))
    else:
        print(f"current split: {current['fingerprint']}  ({current['n_fires']} fires, ")
        print(
            f"  train folds {current['train_folds']} -> {len(current['train_fire_ids'])} fires / "
            f"{len(current['train_blocks'])} blocks; held out "
            f"{len(current['heldout_fire_ids'])} fires / {current['n_heldout_blocks']} blocks)"
        )
        print(rep.format(verbose=args.verbose, for_reporting=args.for_reporting))
    good = rep.reporting_ok if args.for_reporting else rep.ok
    return 0 if good else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())
