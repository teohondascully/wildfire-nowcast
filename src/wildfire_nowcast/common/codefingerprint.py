"""WHICH CODE produced a number — DISCOVERED from the tree, never enumerated.

C-4.2 requires a code fingerprint at both ends of a run so that "was the scoring
code edited while this ran" is answerable. The fingerprint is only as good as its
COVERAGE, and coverage used to be a hand-written tuple in ``eval/reporting.py``.

**THE FAILURE THIS MODULE EXISTS TO REMOVE (ADR-057).** That tuple omitted
``model/noiseoracle.py`` and ``model/direct.py``. A run therefore stamped "ONE
version of the scoring code", the omitted module was edited AFTER the run
finished, and an artifact was read as though produced by code that no longer
existed. Nothing was red at any point. **The defect is not a check that cannot
FAIL — it is a check that cannot DISCRIMINATE**, because what it covered was a
list that silently omitted new files. This is the THIRD member of that family
(C8 cannot distinguish the 1 km and 2 km corpora; the public-tell allowlist; the
scoring-module list), so the repair is structural: the set is derived by walking
the package, and a module that lands tomorrow is covered the moment it exists.

Three properties follow, and each one is a bug that is now unreachable:

1. **A NEW MODULE IS COVERED WITHOUT A LIST EDIT.** Nobody has to remember.
2. **A MODULE THAT BECOMES A PACKAGE STAYS COVERED**, because the walk is
   recursive. ``common/null_check.py`` became ``common/null_check/`` at A15 and
   silently left the old list; that same hazard is what stopped ``contract.py``
   from being split (ADR-047 (6)(7)).
3. **A SCAN THAT MATCHES NOTHING IS A FAILURE, NOT A CLEAN TREE.** Four of this
   project's false negatives were empty scans reporting success, so
   :func:`discover_modules` raises rather than returning an empty set.

The tree is located through the IMPORTED package (``wildfire_nowcast.__file__``)
rather than through ``repo_root()``: what a reader wants to know is which code
RAN, and the code that ran is the code that was imported. ``repo_root()`` is
redirectable by ``$WILDFIRE_REPO_ROOT``, so deriving the path from it can hash a
tree nobody executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "COMMON_SUBTREES",
    "SCORING_SUBTREES",
    "DIGEST_CHARS",
    "EmptyFingerprintScanError",
    "package_root",
    "discover_modules",
    "fingerprint_modules",
    "code_fingerprint",
]

#: Characters of hex kept per digest. Truncation is fine here: this identifies a
#: version of a file among the handful a project produces, it is not a security
#: boundary. Stated so the number is a decision rather than an accident.
DIGEST_CHARS = 16

#: Subtrees of the package whose code a REPORTED NUMBER is computed through.
#: ``common/`` is the whole of C-4's frozen shared surface; ``eval/`` + ``model/``
#: are the scoring code. Whole subtrees, not curated files — curation is the
#: defect this module removes.
COMMON_SUBTREES: tuple[str, ...] = ("common",)
SCORING_SUBTREES: tuple[str, ...] = ("eval", "model")


class EmptyFingerprintScanError(RuntimeError):
    """Raised when a scan covers nothing. An empty all-clear is not an all-clear."""


def package_root() -> Path:
    """Directory of the ``wildfire_nowcast`` package **that is imported**."""
    import wildfire_nowcast

    return Path(str(wildfire_nowcast.__file__)).resolve().parent


def discover_modules(subtrees: Sequence[str], *, root: Path | None = None) -> tuple[str, ...]:
    """Every ``*.py`` under each subtree, recursively, as POSIX paths from ``root``.

    Sorted and de-duplicated so the result is an order-independent identity.
    Raises :class:`EmptyFingerprintScanError` if a subtree is absent or if the
    whole walk finds nothing — a fingerprint over zero files hashes the empty
    dict and looks exactly like agreement.
    """
    base = package_root() if root is None else Path(root)
    found: set[str] = set()
    for subtree in subtrees:
        directory = base / subtree
        if not directory.is_dir():
            raise EmptyFingerprintScanError(
                f"cannot fingerprint {subtree!r}: {directory} is not a directory. Refusing to "
                "report a fingerprint over a subtree that was not scanned — an absent subtree "
                "and an unchanged subtree must never produce the same payload."
            )
        found.update(
            path.relative_to(base).as_posix()
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    if not found:
        raise EmptyFingerprintScanError(
            f"scanned {list(subtrees)} under {base} and found no Python modules. A scan that "
            "matches nothing is this project's most repeated false negative; it is a failure, "
            "not a clean tree."
        )
    return tuple(sorted(found))


def fingerprint_modules(modules: Sequence[str], *, root: Path | None = None) -> dict[str, str]:
    """Per-module content digests, keyed by path relative to the package root.

    A target that does not resolve to a file records ``"MISSING"`` — **carried
    over verbatim from the enumerated implementation and repaired separately**,
    because a sentinel that a reader can skim past is its own defect
    (ADR-047 (7)) and it deserves its own commit and its own planted test.
    """
    base = package_root() if root is None else Path(root)
    out: dict[str, str] = {}
    for name in modules:
        path = base / name
        out[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest()[:DIGEST_CHARS]
            if path.is_file()
            else "MISSING"
        )
    return out


def code_fingerprint(
    subtrees: Sequence[str], *, status: str, root: Path | None = None
) -> dict[str, Any]:
    """The C-4.2 payload for a set of subtrees: combined hash + per-module hashes.

    Shape is unchanged from the enumerated version (``fingerprint``, ``per_file``,
    ``modules``, ``status``) so every existing reader keeps working. What changed
    is that ``modules`` is now an OBSERVATION of the tree rather than a claim
    about it.
    """
    modules = discover_modules(subtrees, root=root)
    per_file = fingerprint_modules(modules, root=root)
    combined = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode()).hexdigest()[
        :DIGEST_CHARS
    ]
    return {
        "fingerprint": combined,
        "per_file": per_file,
        "modules": list(modules),
        "subtrees": list(subtrees),
        "status": status,
    }
