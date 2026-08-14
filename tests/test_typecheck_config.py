"""The type check must be able to fail, and its exemption list must be able to shrink.

[A17] This project's stated code style is typed Python (see README.md) and
nothing enforced it until now. `pyproject.toml` even carries a
`Typing :: Typed` classifier. The obvious way to close that gap — turn mypy on and relax it until
it is green — produces a check that cannot fail, which is this project's most
expensive and most frequently repeated defect: an all-NaN channel passed 56
contract clauses, `train_heldout_disjoint` intersected a partition it had
constructed, and `common_code_fingerprint` recorded a package as ``MISSING``
without saying so.

So the configuration is `strict = true` for everything plus an ENUMERATED list
of exempt modules, and this module pins that list. The two directions are
guarded in two different places, on purpose:

* a module ADDED to the list -> **this file** goes red, because the pin below
  no longer matches `pyproject.toml`. Growing the debt takes a visible edit.
* a module on the list that has become CLEAN -> ``make typecheck`` goes red,
  because ``tools/typecheck.py`` re-runs mypy with the exemptions removed and
  requires every listed module to still fail. Debt that was paid cannot sit
  there being reported as outstanding.

Nothing here runs mypy. These assertions hold on any machine, with or without a
type checker installed, which is deliberate: a test that skips when its tool is
missing is a check that cannot fail, one level up.
"""

from __future__ import annotations

import re

import typecheck  # tools/typecheck.py, via `pythonpath` in pyproject.toml
from wildfire_nowcast.common.paths import repo_root

#: The burn-down list, pinned. Measured 2026-08-14. Two groups, both exempt by
#: OWNERSHIP rather than by difficulty: `model/`, `eval/` and `data/` were being
#: written in by other leads (M11, D11 — ADR-050) and infra may not edit them,
#: and `sim/` belongs to simviz. Error counts are recorded in `pyproject.toml`
#: as documentation and are deliberately not asserted: pinning a count inside a
#: directory somebody else is working in goes red on their ordinary work, which
#: teaches people to edit the pin instead of reading it.
PINNED_BURN_DOWN = frozenset(
    {
        "wildfire_nowcast.data.cli",
        "wildfire_nowcast.data.folds",
        "wildfire_nowcast.data.gofer_ext",
        "wildfire_nowcast.data.ignitions",
        "wildfire_nowcast.data.interim_build",
        "wildfire_nowcast.data.isotropy",
        "wildfire_nowcast.data.isotropy_selftest",
        "wildfire_nowcast.data.leakage",
        "wildfire_nowcast.data.pipeline",
        "wildfire_nowcast.data.rasterize",
        "wildfire_nowcast.data.sources.burn_scar",
        "wildfire_nowcast.data.sources.gee",
        "wildfire_nowcast.data.sources.nifc",
        "wildfire_nowcast.data.swap",
        "wildfire_nowcast.eval.baseline_run",
        "wildfire_nowcast.eval.masks",
        "wildfire_nowcast.eval.regime_calibration",
        "wildfire_nowcast.eval.selftest",
        "wildfire_nowcast.model.api",
        "wildfire_nowcast.model.baselines.ellipse",
        "wildfire_nowcast.model.controls",
        "wildfire_nowcast.model.degrade",
        "wildfire_nowcast.model.direct",
        "wildfire_nowcast.model.inputs",
        "wildfire_nowcast.model.kernel",
        "wildfire_nowcast.model.latent",
        "wildfire_nowcast.model.spread",
        "wildfire_nowcast.model.train",
        "wildfire_nowcast.sim.blockanatomy",
        "wildfire_nowcast.sim.coarsen",
        "wildfire_nowcast.sim.diagnostics",
        "wildfire_nowcast.sim.drift",
        "wildfire_nowcast.sim.elmfire",
        "wildfire_nowcast.sim.landfire",
        "wildfire_nowcast.sim.movie",
        "wildfire_nowcast.sim.playthrough",
        "wildfire_nowcast.sim.reader",
        "wildfire_nowcast.sim.replay",
        "wildfire_nowcast.sim.review",
        "wildfire_nowcast.sim.rundash",
        "wildfire_nowcast.sim.selftest",
    }
)

#: `common/` is infra's own surface and is the one package with NO exemption.
#: Asserted structurally below rather than stated in a comment somewhere.
STRICT_PACKAGE_PREFIX = "wildfire_nowcast.common."

# The two ways a checker gets silenced by hand. Both patterns are spelled in
# HALVES and joined here, so this file contains no literal specimen and needs no
# exemption from its own scan — the file most likely to acquire one must not be
# the one file that can never report it (ADR-048 (2)).
_BARE_IGNORE = re.compile(r"#\s*" + "ty" + r"pe:\s*" + "igno" + r"re(?!\[)")
_MODULE_DIRECTIVE = re.compile(r"^\s*#\s*" + "my" + r"py:\s")
_CODED_IGNORE = re.compile(r"#\s*" + "ty" + r"pe:\s*" + "igno" + r"re\[")

_SPECIMEN_BARE = "x = f()  # " + "ty" + "pe: " + "igno" + "re"
_SPECIMEN_DIRECTIVE = "# " + "my" + "py: ignore-errors"
_SPECIMEN_CODED = "x = f()  # " + "ty" + "pe: " + "igno" + "re[arg-type]"


def _config() -> dict[str, object]:
    return typecheck.load_pyproject()


def _scanned_files() -> list:
    root = repo_root()
    return sorted(
        p
        for base in ("src", "tests", "tools")
        for p in (root / base).rglob("*.py")
        if "__pycache__" not in p.parts
    )


# --------------------------------------------------------------------------
# positive controls first: every scan below is worthless until it has been
# watched matching something.
# --------------------------------------------------------------------------


def test_the_parses_are_not_silently_empty() -> None:
    """A configuration scan that matches nothing passes vacuously."""
    table = typecheck.mypy_table(_config())
    assert table, "pyproject.toml has no [tool.mypy] settings"
    assert len(typecheck.burn_down_modules(_config())) >= 30
    assert len(_scanned_files()) >= 100, "the source scan found almost no files"


def test_the_patterns_still_match_what_they_hunt_and_still_spare_what_they_must() -> None:
    """Pattern control. Narrowing these into silence turns other tests red too."""
    assert _BARE_IGNORE.search(_SPECIMEN_BARE)
    assert _MODULE_DIRECTIVE.search(_SPECIMEN_DIRECTIVE)
    # An ignore that NAMES its error code is exactly what the configuration
    # asks for (`ignore-without-code`), so it must survive the scan.
    assert not _BARE_IGNORE.search(_SPECIMEN_CODED)
    assert _CODED_IGNORE.search(_SPECIMEN_CODED)


def test_the_exemption_derivation_reads_the_shape_and_not_a_marker() -> None:
    """`burn_down_modules` is structural; prove it on a fixture, not on the repo."""
    fixture = {
        "tool": {
            "mypy": {
                "overrides": [
                    {"module": ["a.b"], "ignore_missing_imports": True},
                    {"module": ["c.d", "e.f"], "ignore_errors": True},
                    {"module": ["g.h"], "ignore_errors": False},
                ]
            }
        }
    }
    assert typecheck.burn_down_modules(fixture) == ["c.d", "e.f"]
    assert typecheck.burn_down_modules({"tool": {"mypy": {}}}) == []


# --------------------------------------------------------------------------
# the configuration itself
# --------------------------------------------------------------------------


def test_the_checker_is_strict_and_covers_the_package() -> None:
    table = typecheck.mypy_table(_config())
    assert table.get("strict") is True, (
        "strict is off. A lenient global setting is green everywhere and can therefore never "
        "say which modules are actually typed."
    )
    assert table.get("warn_unused_configs") is True, (
        "without warn_unused_configs a per-module section that matches nothing sits there "
        "looking like coverage"
    )
    files = table.get("files") or []
    assert any("src/wildfire_nowcast" in str(f) for f in files), files
    assert "ignore-without-code" in (table.get("enable_error_code") or []), (
        "a bare ignore is the one-line version of the blanket suppression this config exists "
        "to prevent"
    )
    assert table.get("ignore_errors") is not True, "the GLOBAL setting must never be ignore_errors"


def test_the_burn_down_list_matches_its_pin() -> None:
    """Adding a module to the exemption list must be a visible, deliberate edit."""
    listed = set(typecheck.burn_down_modules(_config()))
    added = sorted(listed - PINNED_BURN_DOWN)
    removed = sorted(PINNED_BURN_DOWN - listed)
    assert listed == PINNED_BURN_DOWN, (
        f"the type-check exemption list has moved. Newly exempt: {added}; retired without "
        f"updating this pin: {removed}. Growing it is allowed and is supposed to be visible; "
        "shrinking it is the goal and belongs in the same change as the pyproject edit."
    )


def test_no_exemption_is_a_wildcard() -> None:
    """A glob is a blanket exemption wearing a list's clothes."""
    globs = [m for m in typecheck.burn_down_modules(_config()) if "*" in m]
    assert not globs, f"exemptions must name modules, not patterns: {globs}"


def test_common_is_strict_with_no_exemptions() -> None:
    """infra's own surface carries no debt, and that is checked, not claimed."""
    offenders = [
        m for m in typecheck.burn_down_modules(_config()) if m.startswith(STRICT_PACKAGE_PREFIX)
    ]
    assert not offenders, (
        f"{offenders} are exempt from type checking inside `common/`. `common/` is the package "
        "the contract adjudicates through (C0) and it is infra's to fix, so an exemption here is "
        "debt, not a fence."
    )


def test_every_exemption_names_a_module_that_exists() -> None:
    """A stale entry describes debt that no longer has a file. That is a lie."""
    src = repo_root() / "src"
    missing = []
    for module in typecheck.burn_down_modules(_config()):
        rel = module.replace(".", "/")
        if not (src / f"{rel}.py").is_file() and not (src / rel / "__init__.py").is_file():
            missing.append(module)
    assert not missing, f"exempted modules that do not exist: {missing}"


# --------------------------------------------------------------------------
# hand-silencing
# --------------------------------------------------------------------------


def test_nothing_silences_the_checker_by_hand() -> None:
    """No bare ignore and no module-level `mypy:` directive anywhere.

    A bare ignore hides an unknown number of errors of unknown kind; a
    module-level directive hides a whole file while the configuration still
    claims the module is covered. Both defeat the burn-down list by routing
    around it, which is why they are banned outright rather than counted.
    """
    root = repo_root()
    offenders = []
    for path in _scanned_files():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _BARE_IGNORE.search(line) or _MODULE_DIRECTIVE.search(line):
                offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()[:80]}")
    assert not offenders, (
        "these silence the type checker without naming what they silence:\n  "
        + "\n  ".join(offenders)
        + "\nUse `# ty" + "pe: igno" + "re[<code>]`, or put the module on the burn-down list "
        "where it is counted."
    )


def test_the_coded_ignores_that_do_exist_are_visible() -> None:
    """Positive control on the same corpus: coded ignores exist and are found.

    Without this, a scan that had gone blind to the whole comment family would
    report the tree clean and look identical to a tree that is clean.
    """
    found = sum(len(_CODED_IGNORE.findall(p.read_text())) for p in _scanned_files())
    assert found > 0, (
        "the scan found no ignore comments of ANY form in 100+ files, which means it is no "
        "longer matching the comment family it audits"
    )


def test_the_scan_reads_its_own_source_and_is_not_self_exempt() -> None:
    """This file is scanned like every other, and it holds no literal specimen.

    Excluding it would make the file most likely to acquire a hand-silencing
    comment the one file that could never report one (ADR-048 (2)).
    """
    me = repo_root() / "tests" / "test_typecheck_config.py"
    assert me in _scanned_files(), "this file is not in its own scan"
    text = me.read_text()
    assert not _BARE_IGNORE.search(text) and not _MODULE_DIRECTIVE.search(text)
    # ... and the halves are really still here, so the patterns above are real.
    assert "igno" in text and "my" + "py" in text
