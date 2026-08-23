"""THE registry: every playthrough in this repo, and its mutation coverage.

``common/playthrough.py`` makes a playthrough *able* to prove it can fail. This
module makes it *have to*. It is the same move ``tests/test_clause_registry.py``
made for C-2 - an audit that reads the repo rather than a hand-kept list, so it
cannot drift from the thing it audits:

1. Every playthrough in the repo is registered with an owner and a note.
   **A new playthrough fails this suite until it is registered**, which is the
   point: adding one now breaks the build until its planted defects are declared,
   exactly as adding an INTERFACES clause breaks the build until it is classified.
2. DISCOVERY IS BY SCANNING THE FILESYSTEM, not by importing a list. A file named
   ``tests/test_playthrough_*.py``, or any module in ``src/`` that defines
   ``run_playthrough``, must be registered. A wiring that needs infra's
   attention to stay complete rots the moment another lead is mid-flight -
   ``tests/test_adopted_selftests.py`` learned that already and this copies it.
3. **[A14] REGISTRATION IS AUTOMATIC: a playthrough declares itself IN ITS OWN
   MODULE.** See :func:`declared_metadata`. Nothing below has to be edited to add
   one.
4. Every registered playthrough must reach **100% mutation coverage**: every
   defect it declares is caught, and every blind spot it declares is not.

WHY REGISTRATION HAD TO BECOME AUTOMATIC [A14, ADR-039 (6)]
------------------------------------------------------------
Discovery was already automatic; **registration was not**, and a hand-written
entry in THIS FILE - which infra owns - was required before a playthrough
could pass. That forced **three cross-boundary writes in three consecutive
tasks**: simviz added 11 lines here (ADR-035 (7)), modelling 9 lines (ADR-034
(6)), and the same root cause put infra into INTERFACES.md (ADR-033 (1)).
Each was disclosed, offered for reversion and ratified - and each happened
because *the design made compliance impossible*. C-4 exists to prevent exactly
the edit the mechanism was compelling.

**When a boundary is crossed under mechanical compulsion, fix the mechanism, not
the lead.** A playthrough now declares its own owner, note and (optionally)
slowness as module-level constants in the file that contains it, and this module
reads them. The two entries that remain hand-written are FROZEN and named
(:data:`_GRANDFATHERED`), and a test asserts that set cannot grow.

**AUTO-DISCOVERY IS NOT AUTO-FORGIVENESS.** A discovered module that declares
nothing is a RED BUILD, not a skip - with a message naming the four lines to add.
The failure moved from "you forgot to edit someone else's file" to "you forgot to
describe your own", which is a failure the author can fix inside their own
ownership boundary.

WHY THIS AND NOT JUST THE POLICY
--------------------------------
ADR-030 says *no gate is adjudicated on a metric whose implementation lacks a
playthrough test that recovers a known answer and detects a planted defect*. As
written that binds the REVIEWER at adjudication time, on a judgement about a
file they did not write. Three of this project's six green-but-vacuous defects were
found by the person who wrote the check, going back to look; the other three were
found much later by someone else. **A policy enforced by inspection is a policy
enforced by whoever happens to look.** After this module, a playthrough that
cannot fail is a red build.

WHAT IS DELIBERATELY NOT ENFORCED HERE
--------------------------------------
That a gate HAS a playthrough. Which metric adjudicates which gate is C6.1's
business and the maintainer's ruling, and a test asserting "G5 has a
playthrough" would be infra legislating gate scope from ``tests/``. This
module enforces the property ADR-030 actually names - *the planted defect is
detected* - over every playthrough that exists.
"""

from __future__ import annotations

import ast
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from wildfire_nowcast.common import playthrough as PT
from wildfire_nowcast.common.paths import repo_root

TESTS_DIR = Path(__file__).resolve().parent


def _sibling(name: str) -> ModuleType:
    """Import a sibling test module by name.

    ``tests/`` is deliberately NOT a package (adding ``__init__.py`` would change
    how every other test file is collected), so this goes through the directory
    pytest already puts on ``sys.path`` and makes that explicit rather than
    relying on collection order.
    """
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    return importlib.import_module(name)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

#: A module in ``src/`` that defines this name is a playthrough by construction -
#: it is the entry point sim chose, independently, before this registry
#: existed. Matching on the function rather than on the filename means a
#: playthrough cannot escape the audit by being called something else.
PLAYTHROUGH_ENTRY_POINT = "run_playthrough"


def _defines(path: Path, name: str) -> bool:
    """Does this module define ``name`` at module level? Parsed, not imported.

    ``ast`` rather than ``importlib`` on purpose: discovery must not depend on a
    module's imports being satisfiable. ``sim/playthrough.py`` pulls in the
    ELMFIRE adapter, and a discovery pass that crashed on a missing binary would
    silently stop discovering things - which is the failure mode this module is
    about.
    """
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):  # pragma: no cover - a file we cannot read
        return False
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
        for node in tree.body
    )


#: THIS module matches the discovery pattern and is not a playthrough. Excluded
#: by exact path rather than by loosening the pattern: a discovery rule with a
#: wildcard exception is a discovery rule someone can slip through.
_NOT_A_PLAYTHROUGH = frozenset({"tests/test_playthrough_registry.py"})


def discover_playthroughs() -> set[str]:
    """Every playthrough-shaped module in the repo, as a repo-relative path."""
    root = repo_root()
    found = {
        str(p.relative_to(root)) for p in sorted((root / "tests").glob("test_playthrough_*.py"))
    }
    found |= {
        str(p.relative_to(root))
        for p in sorted((root / "src").rglob("*.py"))
        if _defines(p, PLAYTHROUGH_ENTRY_POINT)
    }
    return found - _NOT_A_PLAYTHROUGH


# --------------------------------------------------------------------------
# [A14] SELF-DECLARATION - the mechanism fix
# --------------------------------------------------------------------------

#: Module-level constants a playthrough declares ABOUT ITSELF, in its own file.
#: ``OWNER`` and ``NOTE`` are required; ``SLOW`` is a reason string, present only
#: if the playthrough is slow enough to deselect from ``make test``.
OWNER_ATTR, NOTE_ATTR, SLOW_ATTR = (
    "PLAYTHROUGH_OWNER",
    "PLAYTHROUGH_NOTE",
    "PLAYTHROUGH_SLOW",
)

#: How a module hands over the thing to be run, in the order they are tried.
#: ``PLAYTHROUGH`` (an object) · ``build_playthrough()`` (a factory, for modules
#: whose construction is expensive) · ``run_playthrough()`` (a foreign entry point
#: that returns its own report - simviz invented this one independently, before
#: the protocol existed, which is why it is honoured rather than replaced).
ENTRY_POINTS = ("PLAYTHROUGH", "build_playthrough", PLAYTHROUGH_ENTRY_POINT)


def _module_name(rel_path: str) -> str:
    parts = Path(rel_path).with_suffix("").parts
    return parts[-1] if parts[0] == "tests" else ".".join(parts[1:])


def _literal(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def declared_metadata(rel_path: str) -> dict[str, str]:
    """What a module declares about itself. **Parsed, never imported.**

    AST rather than ``importlib`` for the same reason discovery uses it:
    ``sim/playthrough.py`` pulls in the ELMFIRE adapter, and a registration pass
    that crashed on a missing binary would silently stop registering things -
    which is the failure mode this module is about. Registration must be
    answerable from the source text alone; only RUNNING one needs an import.

    Returns the declared strings plus ``entry_point``: which of
    :data:`ENTRY_POINTS` the module provides, or ``""``.
    """
    path = repo_root() / rel_path
    out: dict[str, str] = {"owner": "", "note": "", "slow": "", "entry_point": ""}
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):  # pragma: no cover - a file we cannot read
        return out

    assigned: dict[str, ast.AST] = {}
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assigned[node.target.id] = node.value
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defined.add(node.name)

    for key, attr in (("owner", OWNER_ATTR), ("note", NOTE_ATTR), ("slow", SLOW_ATTR)):
        if attr in assigned:
            out[key] = _literal(assigned[attr]) or ""
    for candidate in ENTRY_POINTS:
        if candidate in assigned or candidate in defined:
            out["entry_point"] = candidate
            break
    return out


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Registered:
    """One playthrough: who owns it, and how its coverage is obtained.

    ``report`` returns a :class:`~wildfire_nowcast.common.playthrough.PlaythroughReport`.
    Native playthroughs build one by running the protocol; foreign ones adapt the
    report their own module already emits (``coverage_from_caught_map``), so a
    file infra does not own gets the same requirement with nobody crossing an
    ownership line.

    ``unavailable`` returns a REASON STRING when the playthrough cannot run in
    this environment - currently only the ELMFIRE arm, which needs a compiled
    binary. It becomes a REPORTING gap (C-1) rather than a silent skip: an
    unrunnable playthrough is a declared weakness, and omitting it is the failure.
    """

    owner: str
    note: str
    #: FOREIGN playthroughs supply a report adapter; NATIVE ones supply ``obj``
    #: instead, so the session-scoped factory can serve one report to this file
    #: and to the owning one. Exactly one of the two must be set.
    report: Callable[..., PT.PlaythroughReport] | None = None
    unavailable: Callable[[], str] = lambda: ""
    #: ``slow`` entries are deselected by ``make test`` and run by
    #: ``make playthrough``. Only ever justified by SECONDS, and the reason is
    #: printed, so the gap is declared rather than ambient (C-1).
    slow: str = ""
    #: Native playthroughs expose their `Playthrough` object so the session-scoped
    #: report factory can serve one report to this file AND to the owning one.
    obj: Callable[[], object] | None = None


def _load(rel_path: str) -> ModuleType:
    """Import a discovered module, whether it lives in ``tests/`` or ``src/``."""
    name = _module_name(rel_path)
    return _sibling(name) if rel_path.startswith("tests/") else importlib.import_module(name)


def _auto_obj(rel_path: str, entry_point: str) -> Callable[[], object]:
    """Return the declared playthrough object, resolved LAZILY at run time."""

    def get() -> object:
        module = _load(rel_path)
        attr = getattr(module, entry_point)
        return attr() if entry_point == "build_playthrough" else attr

    return get


def _auto_caught_map_report(rel_path: str) -> Callable[[], PT.PlaythroughReport]:
    """Adapt a foreign ``run_playthrough()`` that already emits ``defects_caught_by``.

    sim invented that shape independently, before this protocol existed.
    Honouring it generically - rather than translating each module by hand here -
    is what lets a NEW foreign playthrough register with no edit to this file.
    """

    def report() -> PT.PlaythroughReport:
        module = _load(rel_path)
        payload = module.run_playthrough()
        caught = payload["defects_caught_by"]
        declared = getattr(module, "DEFECTIVE_COARSENERS", None) or list(caught)
        return PT.coverage_from_caught_map(
            Path(rel_path).stem,
            {name: list(caught.get(name, [])) for name in declared},
            clean_passes=bool(
                payload.get("rule_passes_every_scenario", payload.get("clean_passes", True))
            ),
            notes=dict(payload.get("defect_notes", {})),
        )

    return report


def _coarsening_report() -> PT.PlaythroughReport:
    """ADOPTED from sim's ``sim/coarsen.py``, which already emits exactly
    this protocol's shape (``defects_caught_by``) having invented it independently."""
    from wildfire_nowcast.sim.coarsen import DEFECTIVE_COARSENERS, run_playthrough

    report = run_playthrough()
    return PT.coverage_from_caught_map(
        "coarsening_correctness",
        {name: report["defects_caught_by"].get(name, []) for name in DEFECTIVE_COARSENERS},
        clean_passes=bool(report["rule_passes_every_scenario"]),
        notes={
            "nearest": "centroid sampling. Area error is small and UNBIASED on blobs, so an "
            "area check alone passes it - simviz's own ADR-031 (5) finding, and the reason "
            "this file exists.",
            "all": "integer-cell erosion; loses the whole boundary band.",
            "any": "any-touch dilation.",
        },
    )


def _elmfire_binary_missing() -> str:
    try:
        from wildfire_nowcast.sim.elmfire import find_binary

        find_binary()
    except Exception as exc:  # noqa: BLE001 - the reason IS the result
        return f"{type(exc).__name__}: {exc}"
    return ""


def _non_degeneracy_report() -> PT.PlaythroughReport:
    """ADOPTED from ``sim/playthrough.py``: arms whose expected verdict is declared."""
    from wildfire_nowcast.sim.playthrough import ARMS, run_playthrough

    report = run_playthrough()
    rows = {r["arm"]: r for r in report["arms"]}
    caught = {
        arm.name: (["degeneracy_verdict"] if rows[arm.name]["agrees_with_expectation"] else [])
        for arm in ARMS
        if arm.expect_degenerate
    }
    return PT.coverage_from_caught_map(
        "baseline_non_degeneracy",
        caught,
        clean_passes=all(
            r["agrees_with_expectation"] for r in report["arms"] if not r["expect_degenerate"]
        ),
        notes={
            "lobotomised": "NOT a strawman: it is the S3 input mapping C1's channel set forces, "
            "measured at exactly zero growth (ADR-031 (2)).",
        },
    )


#: **THE FROZEN GRANDFATHER LIST.** Two modules in ``src/wildfire_nowcast/sim/``
#: predate the self-declaration convention. They belong to sim, so
#: migrating their metadata into their own files would be a cross-boundary write
#: to fix a rule about cross-boundary writes. Their entries therefore stay here -
#: **named, closed, and asserted un-growable** by
#: :func:`test_the_grandfather_list_cannot_grow`.
#:
#: A declaration in the module itself WINS over an entry here, so this is a
#: migration path and not a permanent carve-out: whenever simviz next touches
#: either file, four constants retire its entry. Any NEW playthrough, anywhere,
#: must declare itself.
#:
#: ``sim/playthrough.py`` additionally needs a bespoke report adapter because its
#: payload shape is ``arms``/``agrees_with_expectation`` rather than the
#: ``defects_caught_by`` map :func:`_auto_caught_map_report` understands.
_GRANDFATHERED: dict[str, Registered] = {
    "src/wildfire_nowcast/sim/blockanatomy.py": Registered(
        owner="simviz (S5)",
        note="PLAYTHROUGH 3 - the ANATOMY of G3's dispersion criterion. Splits "
        "`band_area_dispersion_ratio` into a spread term whose denominator the model cannot "
        "move and an error term that is entirely the model's own. Its capability claim is one "
        "exact number: two blocks constructed to SHARE an `adr` whose ensembles differ in "
        "width by exactly 4, which a ratio-shaped instrument reports as 1. Scored through the "
        "real C6 `evaluate`/`aggregate`, not a re-implementation.",
        obj=_auto_obj("src/wildfire_nowcast/sim/blockanatomy.py", "build_playthrough"),
    ),
    "src/wildfire_nowcast/sim/coarsen.py": Registered(
        owner="simviz",
        note="PLAYTHROUGH 1 - the 30 m -> 1 km coarsening rule. ADOPTED, not edited: its own "
        "report already carries defects_caught_by, so the requirement reaches it untouched.",
        report=_coarsening_report,
    ),
    "src/wildfire_nowcast/sim/playthrough.py": Registered(
        owner="simviz",
        note="PLAYTHROUGH 2 - ELMFIRE baseline non-degeneracy. Needs a compiled binary, so it "
        "is a DECLARED reporting gap here rather than a silent skip.",
        report=_non_degeneracy_report,
        unavailable=_elmfire_binary_missing,
        slow="33 s: it compiles nothing but RUNS a real Fortran simulator six times "
        "(four arms plus a two-run determinism check). Deselected from `make test` so the "
        "inner loop stays under a minute; `make playthrough` and `make check` run it.",
    ),
}


class UndeclaredPlaythroughError(AssertionError):
    """A discovered playthrough that declares nothing about itself."""


def _register(rel_path: str) -> Registered:
    """Build a registry entry from what the module declares. Raises if it declares nothing."""
    if (declared := declared_metadata(rel_path))["owner"] and declared["note"]:
        entry_point = declared["entry_point"]
        if entry_point in ("PLAYTHROUGH", "build_playthrough"):
            return Registered(
                owner=declared["owner"],
                note=declared["note"],
                slow=declared["slow"],
                obj=_auto_obj(rel_path, entry_point),
            )
        if entry_point == PLAYTHROUGH_ENTRY_POINT:
            return Registered(
                owner=declared["owner"],
                note=declared["note"],
                slow=declared["slow"],
                report=_auto_caught_map_report(rel_path),
            )
    if rel_path in _GRANDFATHERED:
        return _GRANDFATHERED[rel_path]
    raise UndeclaredPlaythroughError(
        f"{rel_path} looks like a playthrough and declares nothing about itself. "
        "ADR-030 makes a playthrough a load-bearing artifact, so it needs an owner, a reason, "
        "and a way to obtain its mutation-coverage report. **Add these to YOUR OWN FILE** — "
        "nothing in tests/ has to change, which is the entire point of the A14 mechanism fix:\n"
        f'    {OWNER_ATTR} = "<your lead name + task>"\n'
        f'    {NOTE_ATTR} = "<what known answer it recovers and what defect it plants, '
        '40+ chars>"\n'
        f'    {SLOW_ATTR} = "<only if slow: the reason, in SECONDS>"\n'
        f"...and expose ONE of {ENTRY_POINTS}. "
        f"Currently declared: {declared}."
    )


def build_registry() -> dict[str, Registered]:
    """The registry, DERIVED from the repo. Never hand-maintained."""
    return {path: _register(path) for path in sorted(discover_playthroughs())}


#: **The audit table - computed, not written.** A playthrough that declares
#: nothing fails :func:`test_every_playthrough_in_the_repo_is_registered` at
#: import, which is a RED BUILD and not a skip: auto-discovery must never become
#: auto-forgiveness.
PLAYTHROUGHS: dict[str, Registered] = build_registry()


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------


def test_the_discovery_scan_is_not_silently_empty() -> None:
    """Guard the scanner itself: if it matched nothing, every audit below passes.

    Same failure this repo has now seen four times - an all-NaN channel passing
    56 checks, a `find -newermt` that matched nothing, a clause parser blind to a
    bold heading, a hard C8 clause false on every artifact. **A verification that
    cannot fail is not a verification.**
    """
    found = discover_playthroughs()
    assert len(found) >= 5, found
    assert "src/wildfire_nowcast/sim/coarsen.py" in found, (
        "the src/ scan stopped finding run_playthrough, so a foreign playthrough could be "
        "added with no registry entry and no failure"
    )
    assert "tests/test_playthrough_dispersion.py" in found


def test_every_playthrough_in_the_repo_is_registered() -> None:
    """THE audit. An unregistered playthrough fails here, before it can be cited."""
    found = discover_playthroughs()
    unregistered = sorted(found - set(PLAYTHROUGHS))
    assert not unregistered, (
        f"{unregistered} look like playthroughs but are not in PLAYTHROUGHS. ADR-030 makes a "
        "playthrough a load-bearing artifact: register it with an owner, a note, and a way to "
        "obtain its mutation-coverage report, so 'the planted defect is detected' is a build "
        "step rather than a claim in a docstring."
    )
    stale = sorted(set(PLAYTHROUGHS) - found)
    assert not stale, f"the registry claims playthroughs that no longer exist: {stale}"


# --------------------------------------------------------------------------
# [A14] the auto-registration mechanism, and its refusal to forgive
# --------------------------------------------------------------------------


def test_MOST_playthroughs_are_registered_WITHOUT_any_entry_in_this_file() -> None:
    """The mechanism fix, asserted as a property rather than described.

    Every module that declares itself must be registered by the AUTO path - i.e.
    its entry could be deleted from this file and nothing would change, because
    there is no entry. If this ever drops to zero the auto path has stopped
    working and the hand-written table is silently back.
    """
    auto = sorted(set(PLAYTHROUGHS) - set(_GRANDFATHERED))
    assert len(auto) >= 6, auto
    for path in auto:
        declared = declared_metadata(path)
        assert declared["owner"], path
        assert len(declared["note"]) >= 40, path
        assert declared["entry_point"] in ENTRY_POINTS, (path, declared)


def test_PLANTED_an_UNDECLARED_playthrough_is_a_RED_BUILD_not_a_skip(tmp_path) -> None:  # noqa: ANN001
    """PLANTED DEFECT: a new playthrough that says nothing about itself.

    **Auto-discovery must not become auto-forgiveness.** The whole hazard of
    making registration automatic is that "nobody has to do anything" slides into
    "nothing is checked". A module matching the discovery pattern and declaring
    nothing must still turn the build RED - the failure simply moved from "you
    forgot to edit infra's file" to "you forgot to describe your own".
    """
    orphan = tmp_path / "test_playthrough_orphan.py"
    orphan.write_text(
        '"""A playthrough that declares nothing."""\n\n\ndef run_playthrough():\n    return {}\n'
    )
    rel = str(orphan.relative_to(tmp_path))

    real = build_registry.__globals__["repo_root"]
    try:
        build_registry.__globals__["repo_root"] = lambda: tmp_path
        with pytest.raises(UndeclaredPlaythroughError) as exc:
            _register(rel)
    finally:
        build_registry.__globals__["repo_root"] = real

    message = str(exc.value)
    assert OWNER_ATTR in message and NOTE_ATTR in message
    assert "YOUR OWN FILE" in message, (
        "the failure must tell the author what to add IN THEIR OWN DIRECTORY. A message that "
        "sends them to tests/ recreates the cross-boundary write this replaced"
    )


def test_a_declaration_in_the_module_OVERRIDES_the_grandfather_entry() -> None:
    """The frozen list is a MIGRATION PATH, not a permanent carve-out.

    Asserted by giving a grandfathered path a declaration and watching the auto
    path win - so the day sim adds four constants to ``sim/coarsen.py``,
    its entry here becomes dead and the next test says so.
    """
    path = "src/wildfire_nowcast/sim/coarsen.py"
    assert path in _GRANDFATHERED
    assert PLAYTHROUGHS[path].report is not None, "today: adapted by hand"

    import test_playthrough_registry as R

    real = R.declared_metadata
    try:
        R.declared_metadata = lambda p: (
            {"owner": "simviz", "note": "x" * 50, "slow": "", "entry_point": "run_playthrough"}
            if p == path
            else real(p)
        )
        assert R._register(path).owner == "simviz"
        assert R._register(path).report is not None
        assert R._register(path) is not _GRANDFATHERED[path], "the module's own words win"
    finally:
        R.declared_metadata = real


def test_the_grandfather_list_cannot_grow() -> None:
    """A closed exception list. Anything new declares itself, no exceptions.

    An open-ended "legacy" table is just the hand-written registry with an
    apology attached; pinning the exact key set is what makes it a migration.
    """
    assert set(_GRANDFATHERED) == {
        "src/wildfire_nowcast/sim/blockanatomy.py",
        "src/wildfire_nowcast/sim/coarsen.py",
        "src/wildfire_nowcast/sim/playthrough.py",
    }, (
        "these three predate self-declaration and live in sim's directory, so moving "
        "their metadata would be a cross-boundary write to fix a rule about cross-boundary "
        "writes. NOTHING ELSE may be added: a new playthrough declares PLAYTHROUGH_OWNER / "
        "PLAYTHROUGH_NOTE in its own file."
    )


def test_the_metadata_parser_never_imports_the_module() -> None:
    """Registration must be answerable from source text alone.

    ``sim/playthrough.py`` pulls in the ELMFIRE adapter, and a registration pass
    that crashed on a missing binary would silently stop registering things -
    which is the failure mode this whole file is about. Verified by parsing a
    module whose import would raise, not by reading the implementation.
    """
    import test_playthrough_registry as R

    bad = Path(repo_root()) / "tests" / "_registry_probe_not_importable.py"
    bad.write_text(
        '"""Unimportable on purpose."""\n'
        "raise RuntimeError('importing me is the defect')\n\n"
        "PLAYTHROUGH_OWNER = 'infra'\n"
        "PLAYTHROUGH_NOTE = 'a declaration that must be readable without executing the module'\n"
        "def run_playthrough():\n    return {}\n"
    )
    try:
        declared = R.declared_metadata(str(bad.relative_to(repo_root())))
        assert declared["owner"] == "infra"
        assert declared["entry_point"] == "run_playthrough"
    finally:
        bad.unlink()


def test_every_registry_entry_states_an_owner_and_a_reason() -> None:
    missing = [n for n, e in PLAYTHROUGHS.items() if (e.report is None) == (e.obj is None)]
    assert not missing, (
        f"{missing} supply neither a Playthrough object nor a report adapter (or both). "
        "Exactly one is required, so there is one way to obtain a coverage report."
    )
    thin = {
        name: entry
        for name, entry in PLAYTHROUGHS.items()
        if not entry.owner.strip() or len(entry.note.strip()) < 40
    }
    assert not thin, sorted(thin)
    # A `slow` entry must say WHY, and most of the registry must NOT be slow: a
    # coverage audit that `make test` mostly skips is an audit nobody runs.
    unexplained = [n for n, e in PLAYTHROUGHS.items() if e.slow and len(e.slow) < 40]
    assert not unexplained, unexplained
    slow = [n for n, e in PLAYTHROUGHS.items() if e.slow]
    assert len(slow) * 2 < len(PLAYTHROUGHS), (
        f"{len(slow)} of {len(PLAYTHROUGHS)} playthroughs are marked slow and therefore "
        "deselected by `make test`. Past half, the default suite stops enforcing ADR-030."
    )


#: Parametrised at COLLECTION so `-m "not slow"` can actually deselect. A marker
#: added inside the test body arrives after selection has happened, which is a
#: check that looks applied and is not - the shape of defect this file is about.
_CASES = [
    pytest.param(name, marks=[pytest.mark.slow] if PLAYTHROUGHS[name].slow else [], id=name)
    for name in sorted(PLAYTHROUGHS)
]


@pytest.mark.parametrize("name", _CASES)
def test_MUTATION_COVERAGE_is_total(name: str, playthrough_report) -> None:
    """Every declared defect is detected, and every declared blind spot is not.

    This is ADR-030's standing requirement, applied uniformly, including to the
    two playthroughs infra does not own - via their own reports, with no
    edit to another lead's file.
    """
    entry = PLAYTHROUGHS[name]
    reason = entry.unavailable()
    if reason:
        pytest.skip(
            f"{name} cannot run in this environment ({reason}). DECLARED, not silent: see "
            "test_unavailable_playthroughs_are_declared_not_forgotten"
        )
    report = playthrough_report(entry.obj()) if entry.obj is not None else entry.report()
    print(PT.format_report(report))
    report.assert_ok()
    assert report.mutation_coverage == 1.0, report.as_dict()


# --------------------------------------------------------------------------
# THE EMPTY REGISTRY (ADR-030, simviz's S9 report against infra)
# --------------------------------------------------------------------------


def test_an_EMPTIED_defect_registry_turns_this_gate_RED_and_not_GREEN() -> None:
    """Proved by FORCING the registry empty, not by reading the source.

    ADR-030 makes `make playthrough` a gate on the grounds that a playthrough
    that cannot fail turns it red. With `DEFECTIVE_COARSENERS` empty, the
    coarsening playthrough plants nothing, catches all zero of its defects, and
    the natural reading of `all(v for v in caught.values())` is vacuously True -
    so the gate's own premise inverts and it turns GREEN.

    Two layers have to hold and this exercises the first: `sim/coarsen.py`
    refuses to BUILD the report. The second is exercised by the test below, so
    that the guarantee does not depend on a guard in a package infra does not own.
    """
    from wildfire_nowcast.sim import coarsen as CO
    from wildfire_nowcast.sim.absent import AbsentMeasurementError

    kept = dict(CO.DEFECTIVE_COARSENERS)
    assert kept, "the registry is ALREADY empty, so emptying it below proves nothing"

    healthy = _coarsening_report()
    assert healthy.passed and healthy.mutation_coverage == 1.0, healthy.as_dict()

    try:
        CO.DEFECTIVE_COARSENERS.clear()
        assert not CO.DEFECTIVE_COARSENERS, "the plant did not take; the observation is void"
        with pytest.raises((AbsentMeasurementError, PT.VacuousPlaythroughError)) as excinfo:
            _coarsening_report()
        message = str(excinfo.value)
    finally:
        CO.DEFECTIVE_COARSENERS.clear()
        CO.DEFECTIVE_COARSENERS.update(kept)

    assert CO.DEFECTIVE_COARSENERS == kept, "the registry was not restored"
    assert "planted_defects" in message or "NO planted defects" in message, message

    restored = _coarsening_report()
    assert restored.passed and restored.mutation_coverage == 1.0, restored.as_dict()


def test_the_ADAPTER_refuses_a_vacuous_map_even_with_no_guard_upstream() -> None:
    """The second layer. `common/` may not rely on a guard in `sim/` staying put.

    Both shapes of a zero denominator, refused where the coverage number is
    COMPUTED rather than where the defects are declared. The second shape is the
    one that survived until now: a report with defect rows in it, every one a
    declared blind spot, printing `mutation coverage 0%` beside `verdict: PASS`.
    """
    with pytest.raises(PT.VacuousPlaythroughError) as empty:
        PT.coverage_from_caught_map("coarsening_correctness", {}, clean_passes=True)
    assert "NO planted defects at all" in str(empty.value)

    with pytest.raises(PT.VacuousPlaythroughError) as blind:
        PT.coverage_from_caught_map(
            "coarsening_correctness",
            {"nearest": [], "all": []},
            clean_passes=True,
            blind_spots=["nearest", "all"],
        )
    assert "BLIND SPOTS" in str(blind.value)


def test_the_NATIVE_path_has_refused_both_shapes_since_ADR_030() -> None:
    """The asymmetry that let the adopted path drift: stated, and now closed.

    `Playthrough.__post_init__` refuses zero defects AND an all-blind-spot set.
    `coverage_from_caught_map` refused only the first. That difference is
    invisible from either report, which is exactly what simviz found between
    `sim/playthrough.py` and `sim/coarsen.py`.
    """
    probe = PT.Probe("always", lambda _: True)
    blind_only = PT.Defect("noop", PT.no_defect(), detected=False, note="changes nothing")

    with pytest.raises(PT.PlaythroughError) as no_defects:
        PT.Playthrough(name="x", build=dict, observe=lambda o: o, probes=(probe,), defects=())
    assert "no defects declared" in str(no_defects.value)

    with pytest.raises(PT.PlaythroughError) as all_blind:
        PT.Playthrough(
            name="x", build=dict, observe=lambda o: o, probes=(probe,), defects=(blind_only,)
        )
    assert "every declared defect is a blind spot" in str(all_blind.value)


def test_every_REGISTERED_playthrough_declares_at_least_one_catchable_defect() -> None:
    """The minimum simviz asked for, asserted over the live registry.

    Not "at least one outcome": at least one outcome that is EXPECTED TO BE
    DETECTED. A registry of blind spots has the same zero denominator as an
    empty one and reads PASS just as convincingly.
    """
    checked: list[str] = []
    for name in sorted(PLAYTHROUGHS):
        entry = PLAYTHROUGHS[name]
        if entry.unavailable():
            continue
        if entry.obj is None:
            continue
        playthrough = entry.obj()
        defects = getattr(playthrough, "defects", ())
        assert any(getattr(d, "detected", False) for d in defects), (
            f"{name} declares {len(defects)} defect(s) and none of them is expected to be "
            "detected, so its coverage denominator is zero and it cannot go red"
        )
        checked.append(name)
    assert len(checked) >= 3, f"only {checked} were checked; the registry walk is wrong"


def test_unavailable_playthroughs_are_declared_not_forgotten() -> None:
    """A skipped playthrough is a REPORTING gap with a named reason (C-1).

    C-1's corollary is the whole point: declaring a weakness is a gate, omitting
    it is a hard fail, because an unevaluable guard is strictly worse than a
    declared-weak one. This test is what makes the ELMFIRE skip visible instead
    of ambient.
    """
    unavailable = {name: PLAYTHROUGHS[name].unavailable() for name in sorted(PLAYTHROUGHS)}
    blocked = {k: v for k, v in unavailable.items() if v}
    # The registry must never be ENTIRELY unavailable: that would be a green suite
    # proving nothing, which is the shape of every defect this file guards.
    assert len(blocked) < len(PLAYTHROUGHS), (
        f"every registered playthrough is unavailable: {blocked}. A coverage audit in which "
        "nothing runs is the green-but-vacuous failure with the auditor as its subject"
    )
    print(
        f"[registry] {len(PLAYTHROUGHS) - len(blocked)}/{len(PLAYTHROUGHS)} runnable here; "
        f"declared unavailable: {blocked or 'none'}"
    )
