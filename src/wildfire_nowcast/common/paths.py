"""Repository path resolution.

No module in ``src/`` may hardcode an absolute path (C7). Everything that needs
a location on disk resolves it from here, and every default is overridable by
an environment variable so tests and other leads can redirect output.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "data_dir",
    "fires_dir",
    "interim_dir",
    "runs_dir",
    "outputs_dir",
    "configs_dir",
    "norm_stats_path",
    "fire_tensor_path",
    "fire_manifest_path",
    "repo_relative",
]

_ENV_ROOT = "WILDFIRE_REPO_ROOT"
_ENV_DATA = "WILDFIRE_DATA_DIR"
_ENV_RUNS = "WILDFIRE_RUNS_DIR"
_ENV_OUTPUTS = "WILDFIRE_OUTPUTS_DIR"


def repo_root() -> Path:
    """Absolute path to the repository root.

    Resolution order: ``$WILDFIRE_REPO_ROOT``, then the nearest ancestor of this
    file containing a ``pyproject.toml``, then a three-level walk up from
    ``src/wildfire_nowcast/common/``.
    """
    env = os.environ.get(_ENV_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


def _env_dir(var: str, default: Path) -> Path:
    env = os.environ.get(var)
    return Path(env).expanduser().resolve() if env else default


def data_dir() -> Path:
    """Root of on-disk data artifacts (``data/``). Owned by data."""
    return _env_dir(_ENV_DATA, repo_root() / "data")


def fires_dir() -> Path:
    """``data/fires/`` - the C1 path root for complete 14-channel tensors."""
    return data_dir() / "fires"


def interim_dir() -> Path:
    """``data/interim/`` - partial artifacts. Per ADR-003 a partial tensor is
    NEVER written to the C1 path; it lands here until all 14 channels exist."""
    return data_dir() / "interim"


def runs_dir() -> Path:
    """``runs/`` - one subdirectory per experiment run (C7)."""
    return _env_dir(_ENV_RUNS, repo_root() / "runs")


def outputs_dir() -> Path:
    """``outputs/`` - scratch artifacts (synthetic fires, movies)."""
    return _env_dir(_ENV_OUTPUTS, repo_root() / "outputs")


def configs_dir() -> Path:
    """``configs/`` - one yaml per experiment (C7)."""
    return repo_root() / "configs"


def norm_stats_path() -> Path:
    """``data/norm_stats.json`` - the C3 path."""
    return data_dir() / "norm_stats.json"


def fire_tensor_path(fire_id: str) -> Path:
    """The C1 path for a fire: ``data/fires/{fire_id}/tensor.zarr``."""
    return fires_dir() / fire_id / "tensor.zarr"


def fire_manifest_path(fire_id: str) -> Path:
    """The C2 path for a fire: ``data/fires/{fire_id}/manifest.json``."""
    return fires_dir() / fire_id / "manifest.json"


def repo_relative(path: str | Path) -> str:
    """How a path is written INTO an artifact: relative to the repo, POSIX form.

    A result artifact records where its inputs were, and this repository
    publishes result artifacts. An absolute path publishes the account name and
    the directory layout of whoever ran it, which is metadata about a machine and
    not evidence about a fire. Worse, it is metadata that nobody rereads, so it
    survives every review that looks at the numbers.

    The rule is narrow on purpose:

    * inside the repo, return the POSIX relative path (``runs/x``, ``data/fires``);
    * anywhere else, return ``str(path)`` UNCHANGED.

    The second half is the important one. A checkpoint on another volume is
    genuinely outside this tree, and rendering it as ``../../..`` would turn a
    readable location into a puzzle while still leaking the layout. A path this
    function cannot make repo-relative is reported as it is, and the tracked-tree
    scan in ``tests/test_hygiene.py`` is what refuses to publish it.

    Not resolved through symlinks: ``resolve()`` on both sides would make the
    output depend on how the caller happened to reach the file, and two runs of
    the same command would then record different strings.
    """
    candidate = Path(path)
    root = repo_root()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return candidate.absolute().relative_to(root).as_posix()
    except ValueError:
        return str(path)
