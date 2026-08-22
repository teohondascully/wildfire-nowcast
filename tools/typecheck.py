#!/usr/bin/env python
"""Run mypy, and audit the burn-down list that lets it be green.

`pyproject.toml` sets `strict = true` for the whole package and then exempts a
list of modules with `ignore_errors = true`. That list is the only reason the
type check passes today, so it is also the only thing that can quietly turn the
check into decoration. A burn-down list that is only ever read by the tool it
excuses will rot into a permanent exemption.

So this runs mypy TWICE:

**the gate** - mypy with the project configuration. Red if any module that is
NOT exempt fails. That is the direction everybody expects.

**the audit** - mypy again with the exemptions REMOVED, and then:

* every exempt module must STILL fail. One that has become clean is reported as
  ``RETIRE``, and the audit exits non-zero. This is the direction that rots: a
  list which only checks its ceiling records debt that was paid years ago and
  makes the coverage number a lie in the flattering direction.
* the audit refuses to report success if it found no errors anywhere at all.
  A scan that matches nothing passes vacuously, and this project has produced
  four confident false negatives that way.

The exemption list is not spelled here. It is DERIVED from `pyproject.toml`, by
the structural rule "an override block whose `ignore_errors` is true", so the
file mypy reads and the file this audits cannot be different files.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "src"

#: How to invoke mypy. The Makefile passes a pinned, isolated `uv tool run`
#: invocation; the default keeps the script usable on its own.
MYPY_COMMAND = os.environ.get("MYPY", "mypy")


def load_pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def mypy_table(config: dict[str, Any]) -> dict[str, Any]:
    table = config.get("tool", {}).get("mypy")
    if not isinstance(table, dict):
        raise SystemExit("pyproject.toml has no [tool.mypy] section: nothing is type-checked")
    return table


def overrides(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(mypy_table(config).get("overrides", []))


def burn_down_modules(config: dict[str, Any]) -> list[str]:
    """Modules exempted by `ignore_errors = true`, in configuration order.

    Structural, not by a marker comment: whatever else an override block does,
    it is an exemption exactly when it turns errors off.
    """
    out: list[str] = []
    for block in overrides(config):
        if block.get("ignore_errors") is True:
            out.extend(block.get("module", []))
    return out


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot serialise {value!r} into a mypy config")


def audit_config_text(config: dict[str, Any]) -> str:
    """The project's mypy settings with every `ignore_errors` block dropped."""
    table = mypy_table(config)
    lines = ["[tool.mypy]"]
    for key, value in table.items():
        if key == "overrides":
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append('cache_dir = ".mypy_cache/audit"')
    for block in overrides(config):
        if block.get("ignore_errors") is True:
            continue
        lines.append("")
        lines.append("[[tool.mypy.overrides]]")
        for key, value in block.items():
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def module_of(path: Path) -> str | None:
    """`src/wildfire_nowcast/a/b.py` -> `wildfire_nowcast.a.b`."""
    try:
        rel = path.resolve().relative_to(PACKAGE_ROOT)
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts or not parts[-1].endswith(".py"):
        return None
    parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def failing_modules(mypy_output: str) -> set[str]:
    found: set[str] = set()
    for line in mypy_output.splitlines():
        if ": error:" not in line:
            continue
        located = line.split(":", 1)[0]
        module = module_of(REPO_ROOT / located)
        if module is not None:
            found.add(module)
    return found


def run_mypy(extra: list[str]) -> subprocess.CompletedProcess[str]:
    command = shlex.split(MYPY_COMMAND) + extra
    return subprocess.run(  # noqa: S603 - the command is repo configuration
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def gate(python_executable: str | None) -> int:
    extra = ["--python-executable", python_executable] if python_executable else []
    result = run_mypy(extra)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def audit(config: dict[str, Any], python_executable: str | None, show_errors: bool = False) -> int:
    listed = burn_down_modules(config)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "mypy-audit.toml"
        config_path.write_text(audit_config_text(config))
        extra = ["--config-file", str(config_path), "--no-error-summary"]
        if python_executable:
            extra += ["--python-executable", python_executable]
        result = run_mypy(extra)

    if show_errors:
        sys.stdout.write(result.stdout)

    failing = failing_modules(result.stdout)

    # Positive control. Without the exemptions this tree is known to fail; an
    # empty result means the scan is broken, not that the debt is paid.
    if not failing:
        sys.stderr.write(
            "TYPECHECK AUDIT ABORTED: with every exemption removed, mypy reported no failing "
            "module at all. That is either a broken invocation or 41 simultaneous retirements, "
            "and neither may be reported as a pass.\n"
        )
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        return 2

    retire = [m for m in listed if m not in failing]
    unlisted = sorted(failing - set(listed))

    print(f"burn-down: {len(listed)} exempt module(s); {len(failing)} still fail under --strict")
    if unlisted:
        print(
            "\nNOT EXEMPT AND FAILING (`make typecheck` is red on these, which is the point):\n  "
            + "\n  ".join(unlisted)
        )
    if retire:
        print(
            "\nRETIRE THESE — they are exempt in pyproject.toml and they now PASS:\n  "
            + "\n  ".join(retire)
            + "\n\nDelete each from its [[tool.mypy.overrides]] block and from the pinned set in "
            "tests/test_typecheck_config.py, in the same change. An exemption for a module that "
            "does not need one is debt that was already paid being reported as outstanding."
        )
        return 1
    print("no exemption is stale: every listed module still fails without its exemption.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-only", action="store_true", help="run mypy, skip the audit")
    parser.add_argument("--audit-only", action="store_true", help="audit the list, skip mypy")
    parser.add_argument(
        "--python-executable",
        default=None,
        help="interpreter whose site-packages mypy resolves third-party types from",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="print the audit run's errors — what each exempt module still owes",
    )
    args = parser.parse_args(argv)

    config = load_pyproject()
    status = 0
    if not args.audit_only:
        status |= gate(args.python_executable)
    if not args.gate_only:
        status |= audit(config, args.python_executable, show_errors=args.show_errors)
    return status


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
