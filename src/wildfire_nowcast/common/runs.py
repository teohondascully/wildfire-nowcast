"""Run directories: ``runs/{run_id}/`` with the resolved config and a git SHA.

C7: *every training run logs its resolved config + git SHA into
``runs/{run_id}/config.yaml``*. This module is the only implementation of that,
so a run directory always means the same thing no matter which lead produced it.

A run directory contains:

* ``config.yaml``   - the fully resolved config, plus a ``_run`` provenance block
* ``run_meta.json`` - the same provenance block, for machine readers
* whatever the run itself writes (checkpoints, metrics, figures)

The provenance block records the git SHA **and whether the tree was dirty**. A
SHA alone is a comfortable lie when there are uncommitted changes, and results
that cannot be traced to an exact tree state are not results.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.config import dump_yaml, load_yaml
from wildfire_nowcast.common.paths import repo_root, runs_dir

__all__ = ["RunDir", "git_sha", "git_is_dirty", "new_run_id", "create_run_dir", "read_run"]

_RUN_KEY = "_run"
_META_NAME = "run_meta.json"
_CONFIG_NAME = "config.yaml"


@dataclass(frozen=True)
class RunDir:
    """A created run directory."""

    run_id: str
    path: Path
    config_path: Path
    meta_path: Path

    def sub(self, *parts: str) -> Path:
        """Create and return a subdirectory of the run (e.g. ``run.sub("figures")``)."""
        p = self.path.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd or repo_root()),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_sha(short: bool = False) -> str | None:
    """Current commit SHA, or ``None`` outside a git work tree.

    ``git rev-parse --short`` needs the ref as well; omitting it made every run
    in ``runs/`` record ``git_sha_short: "unknown"`` while the full SHA was
    fine. A provenance field that silently degrades to a placeholder is the
    C-2 failure in miniature - it looks recorded and is not.
    """
    args = ("rev-parse", "--short", "HEAD") if short else ("rev-parse", "HEAD")
    return _git(*args)


def git_is_dirty() -> bool | None:
    """Whether the work tree has uncommitted changes (``None`` if unknown)."""
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def new_run_id(prefix: str = "run") -> str:
    """A sortable run id: ``{prefix}-YYYYmmdd-HHMMSS``."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def _provenance(run_id: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    sha = git_sha()
    meta: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": sha or "unknown",
        "git_sha_short": git_sha(short=True) or "unknown",
        "git_dirty": git_is_dirty(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "created_utc": datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(sys.argv),
    }
    for name in ("numpy", "xarray", "zarr", "torch"):
        try:
            module = __import__(name)
            # str() matters: torch.__version__ is a str *subclass*.
            meta.setdefault("versions", {})[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional at run time
            meta.setdefault("versions", {})[name] = "not installed"
    meta["split_fingerprint"] = _split_stamp()
    meta["environment_before"] = _environment_stamp()
    if extra:
        meta.update(dict(extra))
    return meta


def _split_stamp() -> dict[str, Any]:
    """C8 - the split every run was created under, stamped automatically.

    Structural on purpose. C8 says *every run stamps ``split_fingerprint``*, and
    the way that clause fails is not refusal, it is a lead writing a new entry
    point and forgetting: 10 of the 20 run directories in ``runs/`` today carry
    no fingerprint at all, which is why the first kernel matrix cannot be shown
    to belong to any split. Putting the stamp in ``create_run_dir`` means a run
    directory carries it no matter who wrote the run.

    Never raises. A provenance stamp that can kill a training run is a worse
    defect than the one it prevents, so a failure is recorded as an error string
    inside the stamp - visible, and not fatal.
    """
    try:
        from wildfire_nowcast.common.splits import split_fingerprint

        return split_fingerprint()
    except Exception as exc:  # pragma: no cover - defensive by design
        return {"fingerprint": None, "error": f"{type(exc).__name__}: {exc}"}


def _environment_stamp() -> dict[str, Any]:
    """[v2.12] C-4.3 - the interpreter environment, stamped automatically.

    Structural for the same reason ``_split_stamp`` is: C-4.3 says a run must be
    attributable to ONE environment, and the way that clause fails is not refusal,
    it is a lead writing a new entry point and forgetting. Ten of twenty run
    directories carry no split fingerprint at all for exactly that reason.

    Stamped as ``environment_before`` because it is taken at run CREATION.
    C-4.2's finding applies verbatim: *a fingerprint sampled after the fact
    records the wrong state precisely in the case it was built to catch.* The
    caller stamps ``environment_after`` into its own payload at the end;
    ``common.splits`` hard-fails on disagreement and reports a one-ended stamp as
    a C-1 gap.

    Never raises - a provenance stamp that can kill a training run is a worse
    defect than the one it prevents.
    """
    try:
        from wildfire_nowcast.common.environment import environment_fingerprint

        return environment_fingerprint()
    except Exception as exc:  # pragma: no cover - defensive by design
        return {"fingerprint": None, "error": f"{type(exc).__name__}: {exc}"}


def create_run_dir(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    prefix: str = "run",
    runs_root: str | Path | None = None,
    extra_meta: Mapping[str, Any] | None = None,
    exist_ok: bool = False,
) -> RunDir:
    """Create ``runs/{run_id}/`` and write the resolved config + provenance.

    Parameters
    ----------
    config
        The *resolved* config (see :func:`wildfire_nowcast.common.config.load_config`).
        It is written verbatim so the run reproduces without ``configs/``.
    run_id
        Defaults to a UTC timestamp. Collisions raise unless ``exist_ok``.
    """
    run_id = run_id or new_run_id(prefix)
    root = Path(runs_root) if runs_root else runs_dir()
    path = root / run_id
    if path.exists() and not exist_ok:
        raise FileExistsError(f"run directory {path} already exists")
    path.mkdir(parents=True, exist_ok=True)

    meta = _provenance(run_id, extra_meta)
    payload = {k: v for k, v in dict(config).items() if k != _RUN_KEY}
    payload[_RUN_KEY] = meta

    config_path = dump_yaml(payload, path / _CONFIG_NAME)
    meta_path = path / _META_NAME
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return RunDir(run_id=run_id, path=path, config_path=config_path, meta_path=meta_path)


def read_run(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a run directory back as ``(config_without_meta, provenance)``."""
    p = Path(path)
    payload = load_yaml(p / _CONFIG_NAME)
    meta = payload.pop(_RUN_KEY, {})
    return payload, meta
