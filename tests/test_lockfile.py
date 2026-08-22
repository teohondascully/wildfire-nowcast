"""``requirements.lock`` must BIND. A lock nobody installs from is worse than none.

WHAT WENT WRONG, MEASURED
-------------------------
Until I8 this file was referenced by no install path. ``Makefile`` ran
``uv pip install -e ".[dev]"``; the lock was named only in two ``pyproject.toml``
comments, in one ruff-pin assertion, and in ``common/environment.py``'s list of
files worth hashing. None of those install anything. So every clone re-resolved,
and a clean clone taken on 2026-08-21 landed **15 of the lock's 73 packages on
different versions and dropped a 16th** (numpy 2.5.1 -> 2.5.2, affine 2.4.0 ->
3.0.0, rasterio 1.5.0 -> 1.5.1, cligj gone, and twelve more).

That is not a tidiness problem. numpy 2.5.2 added a ``(datetime64, unit)``
overload to its stubs, so one type-ignore comment in ``common/contract.py`` was
REQUIRED under the version on the author's machine and UNUSED under the version
CI resolved. ``make typecheck`` exited 0 locally five times while the public
badge had been red for seven days and thirteen commits, and no diff existed
between the two verdicts because there was nothing to diff: the source was
identical and the environments were not.

The file also carried zero hashes, so even the versions it did name were pinned
by string rather than by artifact.

WHAT IS ENFORCED HERE
---------------------
1. The Makefile installs FROM the lock, with ``--require-hashes``. The specific
   regression guarded against is the exact line that was there before.
2. Every row is pinned with ``==`` and carries at least one hash.
3. Every dependency ``pyproject.toml`` declares appears in the lock, so a new
   dependency cannot be added and silently not installed (``make install``
   installs the project with ``--no-deps``, which is what makes that possible).
4. **The interpreter you are running in matches the lock.** This is the one that
   would have caught the outage on the day it started, and it is deliberately
   two-sided: a missing package, a wrong version, AND an extra package all fail.
   An extra package is a failure because C-4.3 puts the environment in C-4's
   frozen set -- ``pip install`` into the shared venv mid-experiment is the thing
   that clause forbids, and this is the first mechanical detector for it.

ON (4) FIRING WHEN SOMEONE HAS NOT RUN ``make install``
-------------------------------------------------------
ADR-073 (6) is the standing rule that a check firing on correct work gets
disabled, and a disabled check is worse than an absent one because it still
reads as coverage. This check is inside that rule rather than an exception to it:
drifting from the lock is not correct work here, the failure names the one
command that repairs it, and the repair is ``make install``. It fires on stale
environments, which is the state it exists to name.

ON WHY IT IS NOT ALSO A PIN ON THE HASH OF THE FILE
----------------------------------------------------
``common/environment.py`` already hashes the lock into every run's environment
fingerprint, so a change to this file is already visible in the artifacts. A
second, hand-maintained pin here would go red on every legitimate relock and
teach people to edit the pin -- the same reasoning that kept the live-tree
fingerprint assertion out of the suite at I7.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from wildfire_nowcast.common.paths import repo_root

LOCK = "requirements.lock"
MAKEFILE = "Makefile"
PYPROJECT = "pyproject.toml"

#: The project itself is installed separately, by path, and is not in the lock.
PROJECT_NAME = canonicalize_name("wildfire-nowcast")

#: `name==version[ ; marker] \` -- the head of a `uv pip compile` block. The
#: trailing backslash and the `--hash=` lines that follow are matched off.
_ROW_RE = re.compile(r"^([A-Za-z0-9._-]+)==([^ ;\\]+)(?:\s*;\s*(.+?))?\s*\\?$")


def _lock_text() -> str:
    return (repo_root() / LOCK).read_text()


def _makefile_text() -> str:
    return (repo_root() / MAKEFILE).read_text()


def lock_rows() -> dict[str, tuple[str, str | None]]:
    """``{canonical name: (version, marker or None)}`` for every pinned row."""
    rows: dict[str, tuple[str, str | None]] = {}
    for line in _lock_text().splitlines():
        if not line or line[0].isspace() or line.startswith("#"):
            continue
        match = _ROW_RE.match(line)
        if match is None:
            continue
        rows[canonicalize_name(match.group(1))] = (match.group(2), match.group(3))
    return rows


def lock_blocks() -> dict[str, list[str]]:
    """``{canonical name: [hash, ...]}``, reading the indented lines under each row."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in _lock_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if not line[0].isspace():
            match = _ROW_RE.match(line)
            current = canonicalize_name(match.group(1)) if match else None
            if current is not None:
                blocks.setdefault(current, [])
            continue
        if current is not None:
            blocks[current].extend(re.findall(r"--hash=(\S+)", line))
    return blocks


def applicable_rows() -> dict[str, str]:
    """The lock's rows whose environment markers hold on THIS interpreter."""
    out: dict[str, str] = {}
    for name, (version, marker) in lock_rows().items():
        if marker is None or Marker(marker).evaluate():
            out[name] = version
    return out


def installed_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            out[canonicalize_name(name)] = dist.version
    out.pop(PROJECT_NAME, None)
    return out


def declared_dependencies() -> set[str]:
    config = tomllib.loads((repo_root() / PYPROJECT).read_text())
    project = config["project"]
    specs: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    return {canonicalize_name(Requirement(s).name) for s in specs}


# --------------------------------------------------------------------------
# 0. The parses, before anything is concluded from them
# --------------------------------------------------------------------------


def test_the_parses_are_not_silently_empty() -> None:
    """Positive control. Four confident false negatives here came from empty scans."""
    assert (repo_root() / LOCK).is_file(), f"{LOCK} does not exist"
    rows = lock_rows()
    assert len(rows) >= 50, f"only {len(rows)} rows parsed out of {LOCK}: the parser is broken"
    assert len(applicable_rows()) >= 50, (
        "no row's marker evaluated true: the marker parse is broken"
    )
    assert len(installed_versions()) >= 50, "importlib.metadata found almost nothing"
    assert len(declared_dependencies()) >= 10, "pyproject.toml parse found almost no dependencies"
    # Negative control on the row regex: an indented hash line is NOT a row.
    assert _ROW_RE.match("    --hash=sha256:abc") is None
    assert _ROW_RE.match("numpy==2.5.1 \\") is not None
    assert _ROW_RE.match("triton==3.7.1 ; sys_platform == 'linux' \\") is not None


# --------------------------------------------------------------------------
# 1. The lock is on the install path at all
# --------------------------------------------------------------------------


def test_make_install_installs_from_the_lock() -> None:
    """The regression is named literally, because it is what was there for weeks."""
    makefile = _makefile_text()
    install = re.search(r"^install:.*?\n((?:\t.*\n)+)", makefile, flags=re.MULTILINE)
    assert install, "the Makefile has no `install:` recipe"
    recipe = install.group(1)

    assert LOCK in recipe, (
        f"`make install` does not name {LOCK}. That is exactly the state this repo was in "
        "until I8: a lock file referenced by two comments and installed by nothing, while "
        "every clone re-resolved and 15 of 73 packages moved."
    )
    assert "pip sync" in recipe, (
        "`make install` no longer uses `uv pip sync`. `pip install -r` leaves packages that "
        "are NOT in the lock in place, so the venv becomes a superset nobody wrote down."
    )
    assert "--require-hashes" in recipe, (
        "the lock is installed without `--require-hashes`, so the pins bind the version string "
        "and not the artifact."
    )
    assert '-e ".[dev]"' not in recipe, (
        "`make install` is resolving from pyproject again. That re-introduces the seven-day "
        "outage of 2026-08-14..21 exactly."
    )


def test_the_workflow_installs_the_same_way_a_developer_does() -> None:
    """CI must not have its own install path, or the lock binds only one of them."""
    workflow = (repo_root() / ".github/workflows/ci.yml").read_text()
    runs = re.findall(r"^\s*run:\s*(.+)$", workflow, flags=re.MULTILINE)
    assert runs, "the CI workflow has no `run:` steps to audit"
    assert any(r.strip() == "make install" for r in runs), (
        "the CI workflow does not install via `make install`, so it can install something "
        "other than what a developer installs"
    )
    assert not any("pip install" in r or "pip sync" in r for r in runs), (
        "the CI workflow installs packages directly instead of through `make install`: "
        f"{[r for r in runs if 'pip' in r]}"
    )


# --------------------------------------------------------------------------
# 2. The lock is a lock
# --------------------------------------------------------------------------


def test_every_row_is_pinned_and_hashed() -> None:
    blocks = lock_blocks()
    unhashed = sorted(name for name, hashes in blocks.items() if not hashes)
    assert not unhashed, (
        f"{len(unhashed)} row(s) in {LOCK} carry no hash: {unhashed[:10]}. "
        "`--require-hashes` would reject the file, and a version pin without a hash trusts "
        "whatever the index serves under that name today."
    )
    assert all(h.startswith("sha256:") for hs in blocks.values() for h in hs), (
        "a hash in the lock is not a sha256"
    )


def test_the_lock_covers_every_declared_dependency() -> None:
    """`make install` uses `--no-deps` for the project, so an absent row is an ImportError."""
    missing = sorted(declared_dependencies() - set(lock_rows()))
    assert not missing, (
        f"pyproject.toml declares {missing} but {LOCK} does not pin them. `make install` "
        "installs the project with `--no-deps`, so these would simply be absent. Run "
        "`make relock` and commit the result."
    )


def test_the_lock_still_covers_the_linux_runner() -> None:
    """A macOS-only relock would silently unpin CI.

    ``--universal`` is what keeps one file usable on both. If someone regenerates
    without it, every marker-gated row disappears, the file still looks fine on a
    laptop, and the runner goes back to resolving torch's CUDA stack on its own.
    """
    linux_only = [
        name
        for name, (_, marker) in lock_rows().items()
        if marker and not Marker(marker).evaluate({"sys_platform": "darwin"})
    ]
    assert len(linux_only) >= 10, (
        f"only {len(linux_only)} rows are gated off macOS. The lock was probably regenerated "
        "without `--universal`, which unpins the CI runner while looking correct here. "
        "Use `make relock`."
    )


# --------------------------------------------------------------------------
# 3. The interpreter running this test IS the lock
# --------------------------------------------------------------------------


def test_this_environment_is_exactly_the_lock() -> None:
    """The check that would have caught the outage on day one.

    A green gate on a machine whose packages are not the repository's packages is
    a statement about that machine. Two-sided on purpose: extra packages fail too,
    because C-4.3 freezes the environment and this is its first detector.
    """
    expected = applicable_rows()
    installed = installed_versions()

    missing = sorted(set(expected) - set(installed))
    extra = sorted(set(installed) - set(expected))
    wrong = sorted(
        (name, expected[name], installed[name])
        for name in set(expected) & set(installed)
        if expected[name] != installed[name]
    )

    assert not (missing or extra or wrong), (
        "this interpreter is not the environment the repository pins.\n"
        f"  missing (in {LOCK}, not installed): {missing}\n"
        f"  extra (installed, not in {LOCK}):   {extra}\n"
        f"  wrong version (lock, installed):    {wrong}\n"
        "Run `make install`. If you added a package on purpose, add it to pyproject.toml and "
        "run `make relock` -- an undeclared package in the shared venv is a C-4.3 violation, "
        "and it is how a gate comes to mean something different on two machines."
    )
