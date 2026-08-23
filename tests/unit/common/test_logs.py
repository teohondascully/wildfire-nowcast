"""`common/logs.py` - the ADR-103 convention, and the properties it is worth having.

The interesting assertions here are not "the helper sets a level". They are the
three things the two hand-rolled ``_log`` shims could not do and the one thing
they must never start doing:

* a diagnostic reaches stderr and **never** stdout, because stdout is what
  `make contract-all-fires` greps and what `--json` callers parse;
* ONE subsystem can be made chatty without making every subsystem chatty;
* importing the package configures nothing, and a warning still reaches a reader
  in a program that configured nothing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from wildfire_nowcast.common.logs import (
    DEFAULT_FORMAT,
    LEVEL_ENV_VAR,
    PACKAGE_LOGGER_NAME,
    PER_LOGGER_ENV_VAR,
    LogStreamError,
    add_logging_arguments,
    configure_logging,
    installed_handler,
    level_for_verbosity,
    levels_of,
    parse_level,
    parse_per_logger_levels,
)

PYTHON = sys.executable


@pytest.fixture(autouse=True)
def _restore_logging_state() -> object:
    """Every test here mutates global logging state; put it back afterwards.

    pytest's own logging plugin owns handlers on the root logger. Removing ours
    by tag rather than clearing the list is the same discipline the helper uses,
    and it is why these tests can run inside a suite that is itself logging.
    """
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in before_handlers:
            root.removeHandler(handler)
    for handler in before_handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(before_level)


def _run(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """A CLEAN interpreter. In-process assertions about handlers are worthless.

    pytest installs its own root handlers, so "the root logger has no handlers"
    can only be asked of a process pytest is not running in.
    """
    base = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    if env:
        base.update(env)
    return subprocess.run(
        [PYTHON, "-c", code], capture_output=True, text=True, check=False, env=base
    )


# --------------------------------------------------------------------------
# rule 3: no handler at import
# --------------------------------------------------------------------------


def test_importing_every_module_in_the_package_configures_no_handler() -> None:
    """A library that installs a handler decides the format for a program it is not.

    Walks the whole installed package rather than a hand-listed sample: the
    module that breaks this rule will be the one nobody put on the list.
    """
    code = (
        "import importlib, logging, pkgutil, json, sys\n"
        "import wildfire_nowcast\n"
        "names = []\n"
        "for m in pkgutil.walk_packages(wildfire_nowcast.__path__, 'wildfire_nowcast.'):\n"
        "    try:\n"
        "        importlib.import_module(m.name)\n"
        "        names.append(m.name)\n"
        "    except Exception as exc:\n"
        "        names.append(m.name + ' !' + type(exc).__name__)\n"
        "root = [type(h).__name__ for h in logging.getLogger().handlers]\n"
        "pkg = [type(h).__name__ for h in logging.getLogger('wildfire_nowcast').handlers]\n"
        "print(json.dumps({'imported': len(names), 'root_handlers': root,\n"
        "                  'pkg_handlers': pkg,\n"
        "                  'failed': [n for n in names if '!' in n]}))\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["imported"] > 40, (
        f"only {payload['imported']} modules imported; the walk is wrong"
    )
    assert payload["root_handlers"] == [], (
        f"importing the package installed {payload['root_handlers']} on the ROOT logger. "
        "ADR-103: no handler is configured at import; configuration happens in main."
    )
    assert payload["pkg_handlers"] == [], payload["pkg_handlers"]


def test_an_unconfigured_program_still_gets_the_warning_on_stderr() -> None:
    """The property that makes "no handler at import" safe to insist on.

    `logging`'s own lastResort handler emits WARNING and above to stderr. Without
    this, rule 3 would mean a library warning in an unconfigured program went
    nowhere, which is exactly the state ADR-103 (4) was written to end.
    """
    code = (
        "from wildfire_nowcast.common.config import to_plain\n"
        "class Bad:\n"
        "    def item(self):\n"
        "        raise RuntimeError('no')\n"
        "print('OUTPUT', to_plain(Bad()).startswith('<'))\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("OUTPUT"), proc.stdout
    assert "to_plain" in proc.stderr and "item() raised" in proc.stderr, (
        "the swallowed exception left no trace. It is the concrete payoff ADR-103 (4) "
        f"names, and stderr read: {proc.stderr!r}"
    )


# --------------------------------------------------------------------------
# stdout is the program's answer and diagnostics may not touch it
# --------------------------------------------------------------------------


def test_configure_logging_refuses_stdout() -> None:
    with pytest.raises(LogStreamError) as excinfo:
        configure_logging(stream=sys.stdout)
    assert "stdout" in str(excinfo.value)


def test_a_diagnostic_and_the_programs_output_land_on_different_streams() -> None:
    """End to end, in a real process: the JSON on stdout stays parseable.

    `common/contract.py --json` prints a document a caller parses. A timestamped
    WARNING interleaved into it is a parse error in whatever reads it, so this
    asserts the separation on the real streams rather than on the handler object.
    """
    code = (
        "import json, logging\n"
        "from wildfire_nowcast.common.logs import configure_logging\n"
        "configure_logging(level='DEBUG')\n"
        "logging.getLogger('wildfire_nowcast.demo').warning('a diagnostic %s', 1)\n"
        "print(json.dumps({'ok': True}))\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"ok": True}, proc.stdout
    assert "a diagnostic 1" in proc.stderr, proc.stderr
    assert "a diagnostic" not in proc.stdout


# --------------------------------------------------------------------------
# the capability the shims could not express
# --------------------------------------------------------------------------


def test_one_subsystem_can_be_raised_without_raising_the_others() -> None:
    """`def _log(msg): if verbose: print(msg)` has exactly one switch for everything.

    This is the argument for the module, stated as a measurement: after asking
    for DEBUG on one logger, that logger is at DEBUG and its sibling is not.
    """
    configure_logging(per_logger={f"{PACKAGE_LOGGER_NAME}.data.pipeline": "DEBUG"})
    levels = levels_of(
        [
            f"{PACKAGE_LOGGER_NAME}.data.pipeline",
            f"{PACKAGE_LOGGER_NAME}.model.train",
            PACKAGE_LOGGER_NAME,
        ]
    )
    assert levels[f"{PACKAGE_LOGGER_NAME}.data.pipeline"] == logging.DEBUG, levels
    assert levels[f"{PACKAGE_LOGGER_NAME}.model.train"] == logging.WARNING, levels


def test_the_same_control_is_reachable_from_the_environment() -> None:
    """A lead debugging someone else's CLI must not have to edit that lead's argv."""
    spec = f"{PACKAGE_LOGGER_NAME}.sim=DEBUG,{PACKAGE_LOGGER_NAME}.eval=ERROR"
    configure_logging(env={PER_LOGGER_ENV_VAR: spec})
    levels = levels_of([f"{PACKAGE_LOGGER_NAME}.sim", f"{PACKAGE_LOGGER_NAME}.eval"])
    assert levels == {
        f"{PACKAGE_LOGGER_NAME}.sim": logging.DEBUG,
        f"{PACKAGE_LOGGER_NAME}.eval": logging.ERROR,
    }, levels


def test_an_explicit_level_beats_the_environment_which_beats_the_ladder() -> None:
    env = {LEVEL_ENV_VAR: "ERROR"}
    configure_logging(verbosity=2, env=env)
    assert logging.getLogger().level == logging.ERROR
    configure_logging(verbosity=2, level="DEBUG", env=env)
    assert logging.getLogger().level == logging.DEBUG
    configure_logging(verbosity=2, env={})
    assert logging.getLogger().level == logging.DEBUG
    configure_logging(verbosity=0, env={})
    assert logging.getLogger().level == logging.WARNING


# --------------------------------------------------------------------------
# the helper's own arithmetic
# --------------------------------------------------------------------------


def test_the_verbosity_ladder_is_three_rungs_and_clamps() -> None:
    assert level_for_verbosity(0) == logging.WARNING
    assert level_for_verbosity(1) == logging.INFO
    assert level_for_verbosity(2) == logging.DEBUG
    assert level_for_verbosity(9) == logging.DEBUG
    assert level_for_verbosity(-3) == logging.WARNING


@pytest.mark.parametrize(
    ("text", "want"),
    [("DEBUG", logging.DEBUG), ("debug", logging.DEBUG), (" info ", logging.INFO), ("30", 30)],
)
def test_parse_level_accepts_the_four_spellings_a_person_will_type(text: str, want: int) -> None:
    assert parse_level(text) == want


@pytest.mark.parametrize("text", ["", "  ", "LOUD", "verbose"])
def test_parse_level_REFUSES_what_it_does_not_know(text: str) -> None:
    """A level string that silently became WARNING would be configuration that lies."""
    with pytest.raises(ValueError):
        parse_level(text)


def test_parse_per_logger_levels_round_trips_and_refuses_a_bare_name() -> None:
    assert parse_per_logger_levels("") == {}
    assert parse_per_logger_levels(" a=DEBUG , b=30 ") == {"a": logging.DEBUG, "b": 30}
    with pytest.raises(ValueError):
        parse_per_logger_levels("a=DEBUG,justaname")
    with pytest.raises(ValueError):
        parse_per_logger_levels("=DEBUG")


# --------------------------------------------------------------------------
# idempotence, and not trampling a host application
# --------------------------------------------------------------------------


def test_configuring_twice_leaves_exactly_one_handler_of_ours() -> None:
    configure_logging()
    first = installed_handler()
    configure_logging(level="DEBUG")
    second = installed_handler()
    ours = [h for h in logging.getLogger().handlers if h is first or h is second]
    assert second is not None and second is not first
    assert ours == [second], f"{len(ours)} handlers of ours are installed, not 1"


def test_a_handler_this_module_did_not_install_is_left_alone() -> None:
    """Configuring us must not silently unconfigure the program that imported us."""
    foreign = logging.StreamHandler(sys.stderr)
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        configure_logging()
        assert foreign in root.handlers, "we removed a handler we did not install"
    finally:
        root.removeHandler(foreign)


def test_installed_handler_is_None_before_anything_configures() -> None:
    """The honest answer to "has main run yet", asked where pytest is not logging."""
    code = (
        "from wildfire_nowcast.common.logs import installed_handler, configure_logging\n"
        "print(installed_handler() is None, end=' ')\n"
        "configure_logging()\n"
        "print(installed_handler() is not None)\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True True", proc.stdout


def test_the_format_carries_the_module_tag_and_a_timestamp() -> None:
    """The two things the shims threw away, asserted on the format rather than assumed."""
    assert "%(name)s" in DEFAULT_FORMAT and "%(asctime)s" in DEFAULT_FORMAT
    assert "%(levelname)" in DEFAULT_FORMAT


def test_the_cli_flags_are_spelled_once_and_do_not_collide_with_verbose() -> None:
    """`--verbose` already means "print the passing checks too" in two C1/C8 CLIs.

    Reusing it for diagnostics would fuse OUTPUT and DIAGNOSTICS in the two CLIs
    that most need them apart, so the helper deliberately does not offer `-v`.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    add_logging_arguments(parser)
    args = parser.parse_args(["-v", "--log-level", "DEBUG", "--log-levels", "a=INFO"])
    assert args.verbose is True
    assert args.log_level == "DEBUG"
    assert parse_per_logger_levels(args.log_levels) == {"a": logging.INFO}


# --------------------------------------------------------------------------
# the convention is written where another lead will look
# --------------------------------------------------------------------------


def test_the_module_docstring_states_the_rule_the_other_leads_follow() -> None:
    """A convention that lives only in a status file is one only its author read."""
    from wildfire_nowcast.common import logs

    doc = " ".join((logs.__doc__ or "").split())
    for phrase in (
        "Program output stays `print`",
        "getLogger( __name__)",
        "No handler is configured at import",
        "Configuration happens in `main`",
    ):
        assert phrase in doc, (
            f"the module docstring no longer states {phrase!r}. It is the only place "
            "another lead is told what the convention is."
        )
    assert "data/gofer_ext.py:449" in doc and "data/pipeline.py:117" in doc, (
        "the docstring no longer names the two shims that motivated it; the reason "
        "for the convention is what stops it being cargo cult"
    )


def test_the_helper_lives_in_common_so_four_leads_cannot_each_invent_one() -> None:
    from wildfire_nowcast.common import logs

    assert Path(logs.__file__).parent.name == "common"
