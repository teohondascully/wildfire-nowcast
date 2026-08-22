"""``common/paths.py`` - C7 path resolution, including the branch a checkout never takes.

``repo_root`` has three arms and only two of them are reachable from a test run
inside this repository: the environment override and the ``pyproject.toml``
search. The third, a fixed walk up from ``src/wildfire_nowcast/common/``, is the
one an installed copy without a ``pyproject.toml`` would use, and it decides
where ``runs/``, ``configs/`` and ``data/`` are for that copy. It is reached here
by executing the module from a directory tree that has no ``pyproject.toml``
above it, which is the only way to run the arm rather than to restate it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from wildfire_nowcast.common import paths


def _load_from(module_path: Path) -> object:
    spec = importlib.util.spec_from_file_location("wildfire_paths_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fallback_root_is_the_directory_that_holds_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three levels up from ``common/paths.py`` is the root, and nothing else is.

    Copied and executed rather than imported: the live module always finds this
    repository's ``pyproject.toml`` first, so the fallback is dead code from the
    point of view of every other test in the suite. Off by one here and an
    installed copy resolves ``runs/`` and ``configs/`` one directory too high,
    silently, in a place where nobody is watching a test suite.
    """
    monkeypatch.delenv("WILDFIRE_REPO_ROOT", raising=False)
    root = (tmp_path / "installed").resolve()
    package = root / "src" / "wildfire_nowcast" / "common"
    package.mkdir(parents=True)
    (package / "paths.py").write_text(Path(str(paths.__file__)).read_text(), encoding="utf-8")

    # Anti-vacuity: if any ancestor carries a pyproject.toml the search arm wins
    # and this test measures nothing. Assert the precondition rather than hope.
    for parent in [package, *package.parents]:
        assert not (parent / "pyproject.toml").exists(), (
            f"{parent} carries a pyproject.toml, so the fallback arm is never reached and "
            "this test would pass without exercising it"
        )

    module = _load_from(package / "paths.py")
    assert module.repo_root() == root, (  # type: ignore[attr-defined]
        "the fallback did not land on the directory that holds src/"
    )
    assert module.configs_dir() == root / "configs"  # type: ignore[attr-defined]
    assert module.runs_dir() == root / "runs"  # type: ignore[attr-defined]


def test_the_environment_override_outranks_both_other_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lead redirecting output must not be overruled by a file on disk."""
    monkeypatch.setenv("WILDFIRE_REPO_ROOT", str(tmp_path))
    assert paths.repo_root() == tmp_path.resolve()
    monkeypatch.delenv("WILDFIRE_REPO_ROOT")
    assert (paths.repo_root() / "pyproject.toml").is_file()


def test_repo_relative_leaves_an_outside_path_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside the repo it is a POSIX relative path; outside it is returned unchanged.

    Rendering an outside path as ``../../..`` would still leak the layout while
    making the location unreadable, so the rule is narrow on purpose.
    """
    monkeypatch.delenv("WILDFIRE_REPO_ROOT", raising=False)
    root = paths.repo_root()
    assert paths.repo_relative(root / "runs" / "x") == "runs/x"
    assert paths.repo_relative(Path("/somewhere/else/ckpt.pt")) == "/somewhere/else/ckpt.pt"
