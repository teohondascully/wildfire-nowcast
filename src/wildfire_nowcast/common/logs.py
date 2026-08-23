"""ADR-103 made mechanical: OUTPUT is printed, DIAGNOSTICS are logged.

THE RULE, in the form the other three leads follow
--------------------------------------------------
1. **Program output stays `print` and stays on stdout.** CLI results, report
   bodies, contract tables, `_print_summary`. These ARE the deliverable. They do
   not acquire a timestamp, a level or a suppression switch, because something
   downstream reads them: `make contract-all-fires` greps the contract table and
   `common/contract.py --json` emits JSON a caller parses.
2. **Diagnostics become stdlib `logging`.** One `logger = logging.getLogger(
   __name__)` per module, at module scope, spelled with `__name__` exactly.
   Progress narration, "I fell back to X", "I skipped this file", and every
   handler that used to swallow an exception silently.
3. **No handler is configured at import.** A library that installs a handler
   decides the format for a program it is not. Every module here is a library to
   somebody, so the only thing a module does at import is ask for its logger.
4. **Configuration happens in `main` and nowhere else**, through
   :func:`configure_logging`. One helper, so that four leads do not each invent
   the same six lines of `basicConfig`.

WHY THIS EXISTS RATHER THAN A `verbose` FLAG
--------------------------------------------
The codebase had already reinvented a logger twice, identically, nested inside a
function and closing over a boolean::

    def _log(msg: str) -> None:      # data/gofer_ext.py:449
        if verbose:                  # data/pipeline.py:117
            print(msg, flush=True)

Exactly one level, no timestamp, no module tag, and no way to raise verbosity for
one subsystem without raising it for every subsystem. The last of those is the
one that costs time: a fire build that is slow in the label stage cannot be made
chatty in the label stage alone. :func:`configure_logging` takes per-logger
levels, from an argument or from ``WILDFIRE_LOG_LEVELS``, so
``wildfire_nowcast.data.pipeline=DEBUG`` raises one subsystem and leaves the rest
where they were. That capability is what the shims could not express, and the
test that demonstrates it is the argument for this module.

DIAGNOSTICS GO TO STDERR AND MAY NOT GO TO STDOUT
-------------------------------------------------
Enforced, not documented: :func:`configure_logging` raises if it is handed
``sys.stdout``. A timestamped line interleaved into the JSON that
`common/contract.py --json` prints is a parse error in whatever reads it, and a
diagnostic interleaved into the contract table is a line the `grep` in
`make contract-all-fires` may match. stdout belongs to the program's answer.

WHAT AN UNCONFIGURED PROGRAM DOES
---------------------------------
Nothing is lost when nobody calls :func:`configure_logging`: `logging`'s own
``lastResort`` handler emits WARNING and above to stderr with no formatting. So a
library warning still reaches a reader in a program that never configured
anything, which is the property that makes rule 3 safe to insist on.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable, Mapping
from typing import Final, TextIO

__all__ = [
    "DEFAULT_DATE_FORMAT",
    "DEFAULT_FORMAT",
    "LEVEL_ENV_VAR",
    "PACKAGE_LOGGER_NAME",
    "PER_LOGGER_ENV_VAR",
    "VERBOSITY_LEVELS",
    "LogStreamError",
    "add_logging_arguments",
    "configure_from_args",
    "configure_logging",
    "installed_handler",
    "level_for_verbosity",
    "levels_of",
    "parse_level",
    "parse_per_logger_levels",
]

#: The package every module's ``__name__`` logger hangs off. Named here so a
#: caller can raise the whole package in one line without spelling the string.
PACKAGE_LOGGER_NAME: Final = "wildfire_nowcast"

#: ``WILDFIRE_LOG_LEVEL=DEBUG`` beats the verbosity ladder; an explicit ``level=``
#: argument beats both. Environment sits in the middle deliberately: a lead
#: debugging someone else's CLI can raise it without editing that lead's argv.
LEVEL_ENV_VAR: Final = "WILDFIRE_LOG_LEVEL"

#: ``WILDFIRE_LOG_LEVELS="wildfire_nowcast.data=DEBUG,matplotlib=ERROR"`` - the
#: per-subsystem control the two ``_log`` shims could not express.
PER_LOGGER_ENV_VAR: Final = "WILDFIRE_LOG_LEVELS"

#: ISO-ish, sortable, and no locale in it. A log line that cannot be sorted is a
#: log line that cannot be joined to a run directory.
DEFAULT_DATE_FORMAT: Final = "%Y-%m-%dT%H:%M:%S"

#: The module tag is the point: ``wildfire_nowcast.data.pipeline`` says which
#: subsystem is talking, which is the first thing the ``_log`` shims threw away.
DEFAULT_FORMAT: Final = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

#: ``-v`` -> INFO, ``-vv`` -> DEBUG, nothing -> WARNING. Three rungs, because a
#: fourth would be a level nobody could describe.
VERBOSITY_LEVELS: Final[tuple[int, ...]] = (logging.WARNING, logging.INFO, logging.DEBUG)

#: Marks the handler this module installed, so a second call replaces OUR handler
#: and leaves a host application's alone. A name-based check would match anything
#: that happened to be a ``StreamHandler`` on stderr.
_OWNED = "_wildfire_nowcast_handler"

#: Third-party loggers held at WARNING unless the caller asks for otherwise.
#: NOT a correctness list and it cannot cause a false all-clear: the worst a
#: stale entry does is leave matplotlib quiet. Without it, ``-vv`` on any figure
#: path buries our own DEBUG lines under font-manager chatter.
NOISY_THIRD_PARTY: Final[tuple[str, ...]] = (
    "asyncio",
    "botocore",
    "fiona",
    "matplotlib",
    "numexpr",
    "PIL",
    "rasterio",
    "urllib3",
)


class LogStreamError(ValueError):
    """Raised when diagnostics are pointed at stdout.

    stdout carries the program's answer - a contract table that
    `make contract-all-fires` greps, JSON that a caller parses. A log line
    interleaved into either is a defect in whatever reads it, so this is a
    refusal rather than a docstring.
    """


def level_for_verbosity(verbosity: int) -> int:
    """``0 -> WARNING``, ``1 -> INFO``, ``2+ -> DEBUG``. Clamped, never raising."""
    if verbosity < 0:
        verbosity = 0
    if verbosity >= len(VERBOSITY_LEVELS):
        verbosity = len(VERBOSITY_LEVELS) - 1
    return VERBOSITY_LEVELS[verbosity]


def parse_level(text: str | int) -> int:
    """``"DEBUG"``, ``"debug"``, ``"10"`` and ``10`` all mean ``logging.DEBUG``.

    Raises ``ValueError`` on anything else. An unrecognised level that silently
    became WARNING would be a configuration string that does nothing and says
    nothing, which is the failure mode this repository keeps paying for.
    """
    if isinstance(text, int):
        return text
    stripped = text.strip()
    if not stripped:
        raise ValueError("an empty logging level is not a level")
    if stripped.isdigit():
        return int(stripped)
    named = logging.getLevelNamesMapping().get(stripped.upper())
    if named is None:
        raise ValueError(f"{text!r} is not a logging level name or number")
    return named


def parse_per_logger_levels(spec: str) -> dict[str, int]:
    """``"a=DEBUG,b=30"`` -> ``{"a": 10, "b": 30}``. Empty string -> ``{}``."""
    out: dict[str, int] = {}
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        name, sep, level = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"{item!r} is not a `logger=LEVEL` pair")
        out[name.strip()] = parse_level(level)
    return out


def installed_handler() -> logging.Handler | None:
    """The handler this module installed on the root logger, or ``None``.

    The honest answer to "has anybody configured logging yet", used by the tests
    that assert importing the package configures nothing.
    """
    for handler in logging.getLogger().handlers:
        if getattr(handler, _OWNED, False):
            return handler
    return None


def _resolve_level(
    verbosity: int,
    level: int | str | None,
    env: Mapping[str, str],
) -> int:
    if level is not None:
        return parse_level(level)
    from_env = env.get(LEVEL_ENV_VAR)
    if from_env:
        return parse_level(from_env)
    return level_for_verbosity(verbosity)


def _resolve_per_logger(
    per_logger: Mapping[str, int | str] | None,
    env: Mapping[str, str],
) -> dict[str, int]:
    out = parse_per_logger_levels(env.get(PER_LOGGER_ENV_VAR, ""))
    for name, value in (per_logger or {}).items():
        out[name] = parse_level(value)
    return out


def configure_logging(
    *,
    verbosity: int = 0,
    level: int | str | None = None,
    stream: TextIO | None = None,
    per_logger: Mapping[str, int | str] | None = None,
    env: Mapping[str, str] | None = None,
    quiet_third_party: bool = True,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> logging.Handler:
    """Install exactly one stderr handler and set levels. **Call this from `main`.**

    Precedence for the global level: explicit ``level`` beats ``WILDFIRE_LOG_LEVEL``
    beats the ``verbosity`` ladder. Per-logger overrides from ``per_logger`` beat
    the ones parsed out of ``WILDFIRE_LOG_LEVELS``.

    Idempotent by construction: the handler it installs is tagged, and a second
    call removes the tagged one before installing its replacement. Handlers this
    module did not install are left alone, so importing us into a host
    application that already configured logging does not silently unconfigure it.

    Returns the installed handler, so a caller that wants to assert on it can,
    and so a test does not have to go looking through ``root.handlers``.

    :raises LogStreamError: if ``stream`` is ``sys.stdout``.
    """
    resolved_env: Mapping[str, str] = os.environ if env is None else env
    target: TextIO = sys.stderr if stream is None else stream
    if target is sys.stdout:
        raise LogStreamError(
            "diagnostics may not be written to stdout: stdout carries the program's "
            "output (contract tables, JSON, report bodies) and a caller parses it. "
            "Pass sys.stderr, or leave `stream` unset."
        )

    global_level = _resolve_level(verbosity, level, resolved_env)
    overrides = _resolve_per_logger(per_logger, resolved_env)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _OWNED, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(target)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handler.setLevel(logging.NOTSET)
    setattr(handler, _OWNED, True)
    root.addHandler(handler)
    root.setLevel(global_level)

    if quiet_third_party:
        for name in NOISY_THIRD_PARTY:
            if name not in overrides:
                logging.getLogger(name).setLevel(max(global_level, logging.WARNING))
    for name, value in overrides.items():
        logging.getLogger(name).setLevel(value)
    return handler


def add_logging_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the two diagnostic flags every CLI here carries, spelled once.

    ``--log-level`` and ``--log-levels``, deliberately NOT ``-v/--verbose``:
    ``common/contract.py`` and ``common/splits.py`` already spell ``--verbose``
    and it means "print the passing checks too", which is a statement about
    program OUTPUT. Reusing it for diagnostics would fuse the two things this
    convention exists to separate, in the two CLIs that most need them apart.

    Returns the parser so a builder can use it inline.
    """
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        metavar="LEVEL",
        help=(
            "DIAGNOSTIC level on stderr: DEBUG, INFO, WARNING, ERROR. Default WARNING, "
            f"or ${LEVEL_ENV_VAR}. Program output on stdout is unaffected."
        ),
    )
    parser.add_argument(
        "--log-levels",
        dest="log_levels",
        default="",
        metavar="NAME=LEVEL,...",
        help=(
            "per-subsystem diagnostic levels, e.g. wildfire_nowcast.data=DEBUG. "
            f"Same syntax as ${PER_LOGGER_ENV_VAR}."
        ),
    )
    return parser


def configure_from_args(args: argparse.Namespace, *, default_verbosity: int = 0) -> logging.Handler:
    """The one line a ``main`` runs after :func:`add_logging_arguments`.

    ``default_verbosity`` is what this program does when the user asks for
    nothing: 0 (WARNING) for a CLI whose answer is on stdout, 1 (INFO) for a
    long-running wrapper whose progress narration is the point. ``--log-level``
    still overrides it either way.

    Tolerates a namespace built by a parser that did NOT take those flags, so a
    CLI can configure logging before it has adopted them rather than being
    unable to configure it at all.
    """
    return configure_logging(
        verbosity=default_verbosity,
        level=getattr(args, "log_level", None),
        per_logger=parse_per_logger_levels(getattr(args, "log_levels", "") or ""),
    )


def levels_of(names: Iterable[str]) -> dict[str, int]:
    """``{name: effective level}``, for reporting what a configuration actually did."""
    return {name: logging.getLogger(name).getEffectiveLevel() for name in names}
