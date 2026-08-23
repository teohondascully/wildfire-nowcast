"""ADR-103, enforced on the tree rather than remembered: ONE logging convention.

ADR-103 (6) says no lead may introduce a second logging convention. A ruling that
lives in a decisions file is a ruling three leads have to remember; these are the
scans that make the second convention fail instead.

Four rules, four scans, each with the defect it must catch planted in a specimen:

1. every ``getLogger`` call passes ``__name__`` - a literal name is how two
   modules end up sharing one logger and neither can be raised alone;
2. nothing configures logging at import - the library discipline;
3. configuration goes through ``common/logs.configure_logging`` and nowhere else,
   so ``basicConfig`` appearing anywhere is a second convention starting;
4. the hand-rolled ``def _log(msg): if verbose: print(msg)`` shim is a BURN-DOWN.
   It held exactly the two instances ADR-103 (2) named, both in ``data/``. **Both
   were converted at D14 and the list is now EMPTY**, having first been observed
   failing in the retirement direction. A third instance appearing anywhere still
   fails this file; what an empty list can no longer do is notice a REVERT, so
   :func:`test_the_two_retired_shims_stayed_retired` asserts the positive fact
   instead - the two modules carry a module logger and use it.

The shim scan is STRUCTURAL - a nested function whose whole body is
``if <flag>: print(...)`` - not a search for the name ``_log``. Renaming it to
``_say`` must not buy an exemption; seven previous instances in this repository
survived precisely because the check knew a name rather than a shape.

Scope, stated rather than left to be discovered: rules 1-3 run over ``src/`` and
``tools/``, the code that ships. ``tests/`` is excluded from rule 3 only, because
a test that configures logging is a test harness, not a program. Rule 4 runs over
the whole tracked tree.
"""

from __future__ import annotations

import ast
import subprocess
import tomllib
from dataclasses import dataclass

import pytest

from wildfire_nowcast.common.paths import repo_root

#: Attribute names that are a logging emission, for the "is this a logging call"
#: question. ``warn`` is here because ruff's G010 exists to remove it and the
#: scan has to see it before ruff can complain about it.
_LEVEL_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)

#: Every way this repository is allowed to install logging configuration. Anything
#: else - `basicConfig`, `dictConfig`, a hand-rolled handler at module scope - is
#: a second convention.
_CONFIGURATION_CALLS = frozenset({"configure_logging", "configure_from_args"})
_FORBIDDEN_CONFIGURATION_CALLS = frozenset({"basicConfig", "dictConfig", "fileConfig"})

#: The shims ADR-103 (2) names. **RETIRED AT D14 and now EMPTY**: their owner converted
#: both, and the list is emptied in the same commit because a burn-down entry that
#: outlives its reason is an allow-list. It fired in the retirement direction
#: before it was emptied, which is the observation infra could not make itself.
#: Empty is the correct end state and NOT a weaker check: `new` still names a
#: third instance the moment one appears anywhere in the tree, and
#: :func:`test_the_two_retired_shims_stayed_retired` supplies the positive half
#: that an empty set cannot.
PRINT_LOGGER_SHIMS: frozenset[str] = frozenset()

#: The two modules the retired entries named. Kept as data rather than deleted so
#: the retirement is PINNED rather than merely absent: reverting either module to
#: a `_log` shim has to go red somewhere, and with the burn-down empty this is the
#: only place left that can do it.
RETIRED_PRINT_LOGGER_SHIMS: frozenset[str] = frozenset(
    {
        "src/wildfire_nowcast/data/gofer_ext.py",
        "src/wildfire_nowcast/data/pipeline.py",
    }
)


def _tracked(pattern: str) -> list[str]:
    """``git ls-files``, so the corpus is the tree that ships and not this disk.

    From the git INDEX rather than a filesystem walk: ADR-102 (5) - a control
    whose corpus reads the filesystem is a control about the checkout.
    """
    out = subprocess.run(
        ["git", "-C", str(repo_root()), "ls-files", pattern],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line.endswith(".py")]


def _source_corpus() -> list[str]:
    return [*_tracked("src/*.py"), *_tracked("tools/*.py")]


def _parse(rel: str) -> ast.Module:
    return ast.parse((repo_root() / rel).read_text(encoding="utf-8"), filename=rel)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.detail}"


# --------------------------------------------------------------------------
# rule 1: getLogger(__name__), always
# --------------------------------------------------------------------------


def bad_get_logger_calls(rel: str, tree: ast.Module) -> list[Finding]:
    """Two rules, split by where the call is, because the two places differ.

    * **At module scope** a ``getLogger`` call IS the module's own logger and must
      be passed ``__name__``. A literal there is how two modules end up sharing a
      logger and neither can be raised alone.
    * **Inside a function** a computed name is the legitimate shape - it is how
      ``common/logs`` applies a per-subsystem level and how any helper works. What
      is never legitimate anywhere is a string LITERAL or ``__file__``.

    Deciding by position rather than exempting the implementing module by path:
    a path exemption would make the one module that defines the convention the
    one module the convention is not checked against.
    """
    out: list[Finding] = []

    def walk(node: ast.AST, *, module_scope: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner_scope = module_scope and not isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef
            )
            if isinstance(child, ast.Call):
                func = child.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "getLogger":
                    arg = child.args[0] if child.args else None
                    if isinstance(arg, ast.Constant):
                        out.append(
                            Finding(rel, child.lineno, "getLogger() is passed a string LITERAL")
                        )
                    elif isinstance(arg, ast.Name) and arg.id == "__file__":
                        out.append(Finding(rel, child.lineno, "getLogger() is passed __file__"))
                    elif module_scope and not (isinstance(arg, ast.Name) and arg.id == "__name__"):
                        out.append(
                            Finding(
                                rel,
                                child.lineno,
                                "a module-scope getLogger() must be passed __name__",
                            )
                        )
            walk(child, module_scope=inner_scope)

    walk(tree, module_scope=True)
    return sorted(out, key=lambda f: f.line)


def test_every_logger_in_the_shipped_tree_is_named_after_its_module() -> None:
    findings = [f for rel in _source_corpus() for f in bad_get_logger_calls(rel, _parse(rel))]
    assert not findings, (
        "a logger is named by something other than __name__:\n  "
        + "\n  ".join(str(f) for f in findings)
        + "\nA literal name puts two modules on one logger, and per-subsystem level "
        "control is the whole reason ADR-103 chose logging over the `_log` shims."
    )


def test_the_getLogger_scan_has_a_corpus_and_is_not_passing_on_nothing() -> None:
    """The anti-vacuity control. An empty scan reporting clean is this repo's oldest bug."""
    seen = 0
    for rel in _source_corpus():
        for node in ast.walk(_parse(rel)):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "getLogger":
                    seen += 1
    assert seen >= 6, (
        f"only {seen} getLogger call(s) in src/ + tools/. The rule above is checking an "
        "almost empty set, which is how LOG and G sat in the ruff config passing "
        "unconditionally (ADR-103 (5))."
    )


# --------------------------------------------------------------------------
# rules 2 and 3: configuration happens in main, through the one helper
# --------------------------------------------------------------------------


def configuration_outside_main(rel: str, tree: ast.Module) -> list[Finding]:
    """Calls that CONFIGURE logging anywhere other than inside a ``main`` function.

    The module that DEFINES ``configure_logging`` may call it - that is the
    implementation, not a second configuration site - and this is decided by
    reading the module's own definitions rather than by exempting a path.
    """
    defined_here = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    allowed_enclosing = {"main"} | (_CONFIGURATION_CALLS & defined_here)

    out: list[Finding] = []

    def visit(node: ast.AST, enclosing: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, (*enclosing, child.name))
                continue
            if isinstance(child, ast.Call):
                func = child.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in _FORBIDDEN_CONFIGURATION_CALLS:
                    out.append(
                        Finding(
                            rel,
                            child.lineno,
                            f"{name}() is a SECOND logging convention; use "
                            "common/logs.configure_logging from main",
                        )
                    )
                elif name in _CONFIGURATION_CALLS and not (set(enclosing) & allowed_enclosing):
                    where = "module scope" if not enclosing else f"{'.'.join(enclosing)}()"
                    out.append(
                        Finding(
                            rel,
                            child.lineno,
                            f"{name}() is called from {where}; ADR-103 permits configuration "
                            "only in main",
                        )
                    )
            visit(child, enclosing)

    visit(tree, ())
    return out


def test_nothing_in_the_shipped_tree_configures_logging_outside_main() -> None:
    findings = [f for rel in _source_corpus() for f in configuration_outside_main(rel, _parse(rel))]
    assert not findings, (
        "logging is configured outside a main():\n  "
        + "\n  ".join(str(f) for f in findings)
        + "\nA module that configures at import decides the format for a program it is not."
    )


def test_at_least_one_main_actually_configures_logging() -> None:
    """The other half of the rule above, which would otherwise pass over zero CLIs."""
    configuring = sorted(
        {
            rel
            for rel in _source_corpus()
            for node in ast.walk(_parse(rel))
            if isinstance(node, ast.Call)
            and (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            in _CONFIGURATION_CALLS
        }
    )
    assert len(configuring) >= 4, (
        f"only {len(configuring)} module(s) call the configuration helper: {configuring}. "
        "The 'only in main' rule above is then a rule about nothing."
    )


# --------------------------------------------------------------------------
# rule 4: the hand-rolled shim is a burn-down, and it belongs to the `data/` package
# --------------------------------------------------------------------------


def print_logger_shims(rel: str, tree: ast.AST) -> list[Finding]:
    """Nested functions whose entire body is ``if <flag>: print(...)``.

    Structural on purpose. ``def _log`` renamed to ``def _say`` is the same
    defect, and a check that knows the name is how seven previous instances of
    this class survived a sweep.
    """
    out: list[Finding] = []
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in outer.body:
            if not isinstance(inner, ast.FunctionDef):
                continue
            body = [n for n in inner.body if not _is_docstring(n)]
            if len(body) != 1 or not isinstance(body[0], ast.If):
                continue
            guarded = body[0].body
            if body[0].orelse or len(guarded) != 1 or not isinstance(guarded[0], ast.Expr):
                continue
            call = guarded[0].value
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "print":
                out.append(
                    Finding(
                        rel,
                        inner.lineno,
                        f"def {inner.name}() is a hand-rolled logger: one level, no "
                        "timestamp, no module tag (ADR-103 (2))",
                    )
                )
    return out


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


def burn_down_drift(
    found: frozenset[str] | set[str], declared: frozenset[str]
) -> tuple[list[str], list[str]]:
    """``(appeared, retired)`` - the two directions a burn-down list can rot in.

    Hoisted out of the test so that BOTH directions can be demonstrated on
    synthetic sets. The second direction is otherwise unprovable by infra: it
    fires when a shim is deleted by the owner of `data/`, which this lead may not edit.
    """
    return sorted(set(found) - declared), sorted(declared - set(found))


def test_the_print_logger_shims_are_exactly_the_two_that_are_declared() -> None:
    """A burn-down in both directions: a third fails, and deleting one fails too."""
    found = {f.path for rel in _tracked("*.py") for f in print_logger_shims(rel, _parse(rel))}
    new, gone = burn_down_drift(found, PRINT_LOGGER_SHIMS)
    assert not new, (
        f"a THIRD hand-rolled print logger appeared in {new}. ADR-103 (6): no lead may "
        "introduce a second logging convention. Use logging.getLogger(__name__)."
    )
    assert not gone, (
        f"{gone} no longer contains the shim it is listed for. It has been converted - "
        "remove it from PRINT_LOGGER_SHIMS in the same commit, because a burn-down entry "
        "that outlives its reason is an allow-list."
    )


def test_the_burn_down_fires_in_BOTH_directions_on_a_synthetic_pair() -> None:
    """C3.5 for the direction infra cannot plant: a shim its owner has already deleted."""
    declared = frozenset({"a.py", "b.py"})
    assert burn_down_drift({"a.py", "b.py"}, declared) == ([], [])
    assert burn_down_drift({"a.py", "b.py", "c.py"}, declared) == (["c.py"], [])
    assert burn_down_drift({"a.py"}, declared) == ([], ["b.py"])
    assert burn_down_drift(set(), declared) == ([], ["a.py", "b.py"])


def test_the_two_retired_shims_stayed_retired() -> None:
    """The positive half of an EMPTY burn-down, which `all(...)` over `frozenset()` is not.

    When ``PRINT_LOGGER_SHIMS`` was emptied at D14, the ownership assertion that
    used to live here became ``all(... for path in frozenset())``, i.e. ``True``
    unconditionally: a check that cannot fail, which is the class this repository
    has now found four times. What replaces it is a statement that can:
    both named modules ask for a logger at MODULE scope with ``__name__``, and
    both actually emit. Reverting either to ``def _log(msg): if verbose: print``
    removes the emission and turns this red, so the retirement is pinned rather
    than merely unrecorded.
    """
    for rel in sorted(RETIRED_PRINT_LOGGER_SHIMS):
        tree = _parse(rel)
        module_scope_loggers = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_attr(node.value) == "getLogger"
            and [a for a in node.value.args if isinstance(a, ast.Name) and a.id == "__name__"]
        ]
        assert module_scope_loggers, (
            f"{rel} carries no module-scope logging.getLogger(__name__). It was one of the "
            "two ADR-103 (2) `_log` shims; retiring the burn-down entry without the logger "
            "would leave the conversion unpinned in both directions at once."
        )
        emissions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LEVEL_METHODS
            and getattr(node.func.value, "id", "") == "logger"
        ]
        assert emissions, f"{rel} has a logger and never uses it"


def _call_attr(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


# --------------------------------------------------------------------------
# ADR-103 (5): the two ruff rule families were passing on an empty set
# --------------------------------------------------------------------------


def test_the_logging_rule_families_are_selected_AND_have_something_to_check() -> None:
    """`LOG` and `G` were selected with zero logging calls in the tree, so they could
    not fail. Selection alone is therefore not evidence that they are working: this
    asserts both halves, the configuration and a live corpus for it.
    """
    config = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    select = config["tool"]["ruff"]["lint"]["select"]
    assert "LOG" in select and "G" in select, select

    calls = 0
    for rel in _source_corpus():
        for node in ast.walk(_parse(rel)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LEVEL_METHODS
                and getattr(node.func.value, "id", "") in {"logger", "logging", "LOGGER"}
            ):
                calls += 1
    assert calls >= 6, (
        f"only {calls} logging emission(s) in src/ + tools/, so ruff's LOG and G families "
        "are back to reporting clean on an empty set (ADR-103 (5))."
    )


# --------------------------------------------------------------------------
# C3.5: each scan ships with the defect it must catch
# --------------------------------------------------------------------------

_SPECIMEN_BAD_LOGGER = """
import logging

logger = logging.getLogger("wildfire")
other = logging.getLogger(__file__)
fine = logging.getLogger(__name__)


def per_subsystem(name: str) -> object:
    return logging.getLogger(name)


def the_root() -> object:
    return logging.getLogger()


def hard_coded() -> object:
    return logging.getLogger("wildfire.again")
"""

_SPECIMEN_IMPORT_TIME_CONFIG = """
from wildfire_nowcast.common.logs import configure_logging

configure_logging(level="DEBUG")


def main() -> int:
    configure_logging(level="INFO")
    return 0
"""

_SPECIMEN_BASIC_CONFIG = """
import logging


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    return 0
"""

_SPECIMEN_SHIM = """
def build(verbose: bool = True) -> None:
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    _log("hello")


def renamed(verbose: bool = True) -> None:
    def _say(msg: str) -> None:
        if verbose:
            print(msg)

    _say("hello")


def not_a_shim(verbose: bool = True) -> None:
    def helper(msg: str) -> str:
        if verbose:
            return msg.upper()
        return msg

    helper("hello")
"""


def test_the_getLogger_scan_catches_a_literal_name_and_dunder_file() -> None:
    """Three violations, three clean calls, and the split is by POSITION not by path.

    Line 4 a module-scope literal, line 5 ``__file__``, line 18 a literal inside a
    function. Line 6 (``__name__``), line 10 (a computed name inside a helper) and
    line 14 (the root logger inside a function) are the legitimate shapes.
    """
    findings = bad_get_logger_calls("specimen.py", ast.parse(_SPECIMEN_BAD_LOGGER))
    assert [f.line for f in findings] == [4, 5, 18], findings


def test_the_configuration_scan_catches_an_import_time_call_and_not_the_one_in_main() -> None:
    findings = configuration_outside_main("specimen.py", ast.parse(_SPECIMEN_IMPORT_TIME_CONFIG))
    assert len(findings) == 1 and findings[0].line == 4, findings
    assert "module scope" in findings[0].detail


def test_the_configuration_scan_catches_basicConfig_even_inside_main() -> None:
    """`basicConfig` in main is still a second convention: four leads, four formats."""
    findings = configuration_outside_main("specimen.py", ast.parse(_SPECIMEN_BASIC_CONFIG))
    assert len(findings) == 1 and "SECOND logging convention" in findings[0].detail, findings


def test_the_shim_scan_catches_a_RENAMED_shim_and_leaves_a_real_helper_alone() -> None:
    findings = print_logger_shims("specimen.py", ast.parse(_SPECIMEN_SHIM))
    assert [f.line for f in findings] == [3, 11], findings


@pytest.mark.parametrize(
    "specimen",
    [_SPECIMEN_BAD_LOGGER, _SPECIMEN_IMPORT_TIME_CONFIG, _SPECIMEN_BASIC_CONFIG, _SPECIMEN_SHIM],
)
def test_every_specimen_parses_so_a_scan_is_never_silently_skipped(specimen: str) -> None:
    """A specimen that stopped parsing would make its scan return nothing and pass."""
    assert isinstance(ast.parse(specimen), ast.Module)


def test_the_live_corpus_is_not_empty() -> None:
    """`git ls-files` returning nothing would make every scan above vacuous."""
    corpus = _source_corpus()
    assert len(corpus) > 80, f"{len(corpus)} python files under src/ + tools/; the listing is wrong"
    assert any(rel.startswith("tools/") for rel in corpus), corpus[:5]
    assert any(rel.endswith("common/logs.py") for rel in corpus), (
        "common/logs.py is not in the corpus, so the module that DEFINES the convention "
        "is the one module the convention is not checked against"
    )
