"""A built wheel must carry every non-Python file ``src/`` needs at run time.

WHY THIS FILE EXISTS
--------------------
``src/wildfire_nowcast/model/reference_fit/model.json`` was tracked so that a
reader could load the archived fit that a published result rests on, rather than
only re-run it. The sentence recorded at the time was "a reader can now reach a
latent-bearing address". That sentence is **channel dependent** and nothing said
so: ``[tool.setuptools.package-data]`` listed only ``py.typed``, so

* ``git clone`` gives you ``reference.py`` **and** the fit, and
* ``pip install .`` gives you ``reference.py`` and **not** the fit.

The failure is silent in the direction that matters. ``load_model`` resolves the
checkpoint directory relative to the installed package, so on an installed copy
it raises a missing-file error at call time, in someone else's session, long
after the claim was published.

WHY THIS BUILDS A WHEEL INSTEAD OF READING ``pyproject.toml``
------------------------------------------------------------
A test that reads the configuration and asserts a string cannot fail for the
reason that matters. It asserts that somebody wrote a pattern down; it does not
assert that the pattern MATCHES, that the build backend honours it for a file
living inside a sub-package, or that the bytes that arrive are the bytes that
were tracked. All three have their own failure modes and only the artifact
answers them. So this stages the tracked tree, drives the declared PEP 517
backend, and opens the resulting zip.

The build takes well under a second because it stages source only. It is not
marked slow.

WHY THE STAGED TREE COMES FROM THE GIT INDEX
--------------------------------------------
The subject is "what does a person who clones this repository and installs it
get", and a clone carries tracked files. Copying the working directory instead
would package whatever untracked work happens to be in flight, which is how one
lead's scratch file becomes another lead's green test. Paths come from the
index; CONTENTS come from the working tree, so an uncommitted fix to
``pyproject.toml`` is still the thing under test.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from wildfire_nowcast.common.paths import repo_root

#: The distribution's import package. Members inside the wheel are prefixed with
#: this, so a tracked path ``src/<PKG>/a/b`` arrives as ``<PKG>/a/b``.
PKG = "wildfire_nowcast"

#: Stated as an expectation rather than derived, on purpose. The derived set
#: below is what the assertion runs on; this exists so that a staging step which
#: silently copies nothing produces a RED test instead of an empty universal
#: quantifier. A check that passes when its input is empty is not a check.
MUST_BE_IN_THE_DERIVED_SET = (
    f"{PKG}/py.typed",
    f"{PKG}/model/reference_fit/model.json",
)

#: The Python module that reads the data file above. The defect this file closes
#: was precisely that the wheel carried this and not its data, so both halves are
#: named in one assertion.
READER_MODULE = f"{PKG}/model/reference.py"

_BUILD_DRIVER = """
import contextlib
import io
import os
import sys

tree, outdir = sys.argv[1], sys.argv[2]
os.chdir(tree)
from setuptools import build_meta

# build_meta rewrites sys.argv for the underlying command, so `outdir` is read
# into a local first. Losing this line produced a `--dist-dir/...whl` path.
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    name = build_meta.build_wheel(outdir)
sys.stdout.write(name)
"""


def _tracked(prefix: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", prefix],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return []
    return [p for p in out.stdout.split("\0") if p]


def _stage(dest: Path) -> None:
    """Copy the tracked build inputs into ``dest``: ``src/`` plus the root files."""
    root = repo_root()
    paths = _tracked("src")
    for extra in ("pyproject.toml", "README.md", "LICENSE"):
        if (root / extra).is_file():
            paths.append(extra)
    for rel in paths:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage the tracked tree, build one wheel from it, return the wheel path.

    ``tmp_path_factory`` is used rather than ``mkdtemp`` so the staged tree and
    the build directory are removed by pytest. This repository has already had
    its disk filled twice by temporary directories that were never collected.
    """
    if not _tracked("src"):
        pytest.skip("not a git checkout with a populated index, so there is no clone to model")
    base = tmp_path_factory.mktemp("wheelprobe")
    tree, outdir = base / "tree", base / "dist"
    outdir.mkdir()
    _stage(tree)
    proc = subprocess.run(
        [sys.executable, "-c", _BUILD_DRIVER, str(tree), str(outdir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        "the declared PEP 517 backend could not build a wheel from the tracked tree, "
        "so this file proves nothing about an installed copy.\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    wheel = outdir / proc.stdout.strip()
    assert wheel.is_file(), f"backend reported {proc.stdout.strip()!r} but no such file exists"
    return wheel


def _members(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _expected_data_members() -> dict[str, Path]:
    """Every tracked NON-Python file under ``src/``, keyed by its wheel member name."""
    root = repo_root()
    out: dict[str, Path] = {}
    for rel in _tracked("src"):
        if rel.endswith(".py"):
            continue
        out[rel[len("src/") :]] = root / rel
    return out


def test_the_derived_set_is_not_empty_and_names_the_files_it_is_here_for() -> None:
    """Guard the quantifier before trusting it (the check that cannot fail).

    ``test_the_wheel_carries_every_tracked_data_file`` iterates a derived set. If
    the derivation ever returns nothing, that test passes while asserting
    nothing at all, which is the shape of defect this repository has shipped
    three times.
    """
    derived = set(_expected_data_members())
    assert derived, "no tracked non-Python file was found under src/, so the assertion is vacuous"
    missing = [name for name in MUST_BE_IN_THE_DERIVED_SET if name not in derived]
    assert not missing, (
        f"the derivation no longer sees {missing}. Either the file moved, or "
        "`git ls-files src` is being read wrongly. Do not delete the expectation "
        "to make this pass."
    )


def test_the_wheel_carries_every_tracked_data_file(built_wheel: Path) -> None:
    """The general clause. A new data file added without a `package-data` entry is RED.

    Byte identity, not presence: a member that is present and truncated satisfies
    "a reader can reach it" and fails "a reader can check it", and the second is
    the claim the tracked fit was added to support.
    """
    members = _members(built_wheel)
    absent, differing = [], []
    for name, source in _expected_data_members().items():
        if name not in members:
            absent.append(name)
            continue
        got = hashlib.sha256(members[name]).hexdigest()
        want = hashlib.sha256(source.read_bytes()).hexdigest()
        if got != want:
            differing.append(f"{name}: wheel {got[:16]} vs tree {want[:16]}")
    assert not absent, (
        f"{len(absent)} tracked data file(s) under src/ do not reach a built wheel: {absent}. "
        "`git clone` carries them and `pip install .` does not, so any claim that a "
        "reader can reach them is true of one channel and false of the other. "
        "Add the pattern to [tool.setuptools.package-data] in pyproject.toml; do not "
        "delete the file from the tree to make this green."
    )
    assert not differing, f"member bytes differ from the tracked source: {differing}"


def test_the_wheel_carries_the_reader_and_its_data_together(built_wheel: Path) -> None:
    """The specific sentence, stated as one assertion so neither half can pass alone.

    The recorded form of this defect was "a wheel carries ``reference.py`` and
    NOT the fit". Asserting the pair keeps that sentence testable even if the
    general clause above is later narrowed.
    """
    members = _members(built_wheel)
    fit = f"{PKG}/model/reference_fit/model.json"
    assert READER_MODULE in members, (
        f"{READER_MODULE} is missing from the wheel, so this test is measuring a "
        "broken build rather than the packaging of data files"
    )
    assert fit in members, (
        f"the wheel carries {READER_MODULE} and NOT {fit}. The reader ships without "
        "the artifact it reads, so an installed copy resolves the checkpoint "
        "directory to a path that does not exist."
    )
