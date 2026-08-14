"""C-4.3 [v2.12] — the interpreter ENVIRONMENT is in C-4's frozen set.

WHERE THIS CLAUSE CAME FROM, AND IT WAS NOT THIS MODULE'S AUTHOR
----------------------------------------------------------------
The ingestion work needed ``scipy`` mid-burst and **declined to install it**, on
the reasoning that adding a package to the shared virtualenv while training is
in flight is a
shared-state change of exactly the class C-4 exists to stop — and wrote pure-numpy
equivalents instead. **C-4 as written enumerated FILES and said nothing about the
environment**, so a lead could have installed anything mid-run and violated no
clause. A lead extended a rule to a case it did not name, conservatively and with
a stated reason, and ADR-024 ratified that as a directive.

It stayed a directive rather than a clause for a reason worth repeating here: the
maintainer bumped INTERFACES to v2.12, then **reverted its own bump**, because
infra was barred from running and C-2 forbids a ratified-but-unimplemented
clause. *Do not ratify a clause into INTERFACES when the lead who must implement
it is barred from running.* This module is the implementation that unblocks it.

WHAT IS MECHANICALLY CHECKABLE, AND WHAT IS NOT
-----------------------------------------------
C-4 is classified ``process`` because the same edit is legal or illegal depending
on whether another session is live — a fact about the maintainer's schedule,
not about this repo. The same is true of ``pip install``. What IS mechanical is
the **detection of a violation after the fact**, and that is exactly how C-4.2
was implemented: sample a fingerprint at BOTH ends of a run and hard-fail on
disagreement. C-4.3 gets the same treatment and its own check ids —
``C8.environment_agrees_across_run`` and ``C8.environment_sampled_both_ends`` —
deliberately NOT the ``code_*`` ones, because giving two different quantities one
key name is precisely the defect that made ``C8.internally_consistent`` false on
every artifact in the repo (A12).

WHAT THE FINGERPRINT COVERS, AND THE LIMIT, STATED
--------------------------------------------------
COVERED: the interpreter (version, implementation, platform), **every installed
distribution's name and version**, the lockfile digest if one exists, and a small
declared set of SYSTEM TOOLS resolved on ``PATH`` (their real path, size and
mtime) — which is how ``pip install``, ``uv sync``, an editable-install change and
a rebuilt ELMFIRE binary all become visible.

NOT COVERED: arbitrary system packages. There is no portable way to enumerate
what ``brew upgrade`` did, and a fingerprint that pretends otherwise is worse
than one that states its scope — the C1.6 mistake, pointed the other way. The
scope is therefore recorded IN the fingerprint payload (``covers``), so a reader
of an artifact can see what the hash did and did not see, rather than inferring
it. A check that cannot fail is not a check; a check whose reach is undocumented
is the same thing one level up.

C0: one implementation. ``runs.create_run_dir`` stamps it structurally, so a run
directory carries it no matter who wrote the run — the same reasoning that put
the split stamp there after 10 of 20 run dirs turned out to carry none.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.paths import repo_root

__all__ = [
    "ENVIRONMENT_COVERS",
    "LOCKFILE_CANDIDATES",
    "SYSTEM_TOOLS",
    "installed_distributions",
    "lockfile_digest",
    "system_tools",
    "environment_fingerprint",
    "environments_agree",
]

#: Files that, if present, pin the environment. Checked in order; all that exist
#: are hashed, because a repo carrying two of them and honouring one is a hazard
#: this would otherwise hide.
LOCKFILE_CANDIDATES: tuple[str, ...] = (
    "uv.lock",
    "poetry.lock",
    "requirements.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "constraints.txt",
)

#: System executables whose identity can change a result without any Python
#: package moving. Small and NAMED rather than inferred: an open-ended scan of
#: ``PATH`` would make the fingerprint differ between two developers' machines
#: for reasons that have nothing to do with a run.
SYSTEM_TOOLS: tuple[str, ...] = ("mpifort", "mpif90", "gfortran", "gdalinfo", "git")

#: The scope statement carried INSIDE every fingerprint payload. See module doc.
ENVIRONMENT_COVERS = (
    "interpreter version/implementation/platform; every installed distribution "
    "name+version; the digest of any lockfile present; and the resolved path, size and "
    "mtime of a NAMED set of system tools. It does NOT cover arbitrary system packages "
    "(no portable enumeration exists), so an unrelated `brew upgrade` is invisible here. "
    "Scope is recorded in the payload rather than inferred, because a check whose reach is "
    "undocumented reads as stronger than it is."
)


def installed_distributions() -> list[list[str]]:
    """``[[name, version], ...]`` for every distribution, sorted and deduplicated.

    Duplicates happen for real (two ``.dist-info`` directories for one package
    after a botched upgrade), so they are kept rather than collapsed: that state
    is itself an environment change worth a different hash.
    """
    rows: list[list[str]] = []
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # noqa: BLE001 - a broken dist-info must not abort a stamp
            name = None
        rows.append([str(name or "<unknown>"), str(dist.version or "<unknown>")])
    return sorted(rows)


def lockfile_digest(root: Path | None = None) -> dict[str, str]:
    """``{filename: sha256}`` for every lockfile that exists. Empty if none do."""
    base = Path(root) if root is not None else repo_root()
    out: dict[str, str] = {}
    for name in LOCKFILE_CANDIDATES:
        path = base / name
        try:
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except OSError:
            continue
    return out


def system_tools() -> dict[str, str]:
    """Resolved path + size + mtime for each declared tool, or ``"absent"``.

    ``absent`` is recorded rather than omitted: a tool DISAPPEARING is an
    environment change, and a key that vanishes from a dict changes the hash in a
    way nobody can read afterwards.
    """
    out: dict[str, str] = {}
    names = list(SYSTEM_TOOLS)
    explicit = os.environ.get("WILDFIRE_ELMFIRE_BIN")
    if explicit:
        names.append(explicit)
    for name in names:
        resolved = shutil.which(name) or (name if Path(name).is_file() else None)
        if resolved is None:
            out[name] = "absent"
            continue
        try:
            stat = Path(resolved).stat()
            out[name] = f"{resolved}:{stat.st_size}:{int(stat.st_mtime)}"
        except OSError:
            out[name] = f"{resolved}:unreadable"
    return out


def environment_fingerprint(*, root: Path | None = None) -> dict[str, Any]:
    """[C-4.3, v2.12] Hash the interpreter environment. **Never raises.**

    Total by construction, on the same reasoning as ``splits.split_fingerprint``:
    this is called from ``runs.create_run_dir`` and *a provenance stamp that can
    kill a training run is a worse defect than the one it prevents.* A failure is
    recorded as an ``error`` string inside the payload — visible, not fatal.
    """
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "covers": ENVIRONMENT_COVERS,
    }
    try:
        dists = installed_distributions()
        payload["n_distributions"] = len(dists)
        payload["lockfiles"] = lockfile_digest(root)
        payload["system_tools"] = system_tools()
        material = json.dumps(
            {
                "python": payload["python"],
                "implementation": payload["implementation"],
                "platform": payload["platform"],
                "distributions": dists,
                "lockfiles": payload["lockfiles"],
                "system_tools": payload["system_tools"],
            },
            sort_keys=True,
        )
        payload["fingerprint"] = hashlib.sha256(material.encode()).hexdigest()[:16]
    except Exception as exc:  # pragma: no cover - defensive by design
        payload["fingerprint"] = None
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def environments_agree(before: Any, after: Any) -> bool:
    """Do two stamps describe the same environment?

    ``None``/missing on either side is NOT agreement. The C1.5 lesson at the
    choke point: an unevaluable comparison must be unpassable, or a run with no
    stamp reads as a run that did not change.
    """

    def digest(node: Any) -> str | None:
        if isinstance(node, str) and node:
            return node
        if isinstance(node, dict):
            value = node.get("fingerprint")
            return value if isinstance(value, str) and value else None
        return None

    left, right = digest(before), digest(after)
    return left is not None and right is not None and left == right
