#!/usr/bin/env python3
"""Every repository-relative path the tracked tree names, and whether a CLONER can open it.

THIS VERIFIES REACHABILITY. IT DOES NOT VERIFY SUPPORT.
-------------------------------------------------------
A path RESOLVES here when it is in the git index, which means a reader who
clones this repository has the file. Whether that file CONTAINS the thing the
citing sentence claims is a different property and this module says nothing
about it. The distinction is not academic: four citations of ``docs/decisions.md``
once satisfied a reachability rule while pointing at 87 lines of prose with no
ADR numbers in it (ADR-105 (3)). A green verdict here means "the reader can open
what you named", never "what you named backs what you said".

WHY THIS EXISTS WHEN THREE OTHER SWEEPS ALREADY RAN
---------------------------------------------------
Each of the three was scoped to a different subset of one idea, and the idea is
"provenance that evaporates at the repository boundary, which is exactly where a
portfolio reader stands":

* ``tests/test_hygiene.py`` catches references to the private instruction file;
* ``tools/cited_runs.py`` catches citations of ``runs/`` artifacts;
* the coordination sweeps caught ``coordination/`` paths.

A citation of an untracked ``reports/figures/*.json`` was in NONE of them, and
there were 26 of those across 12 files when this module was written. So the
question asked here is not "does this path match a known-bad prefix" but "does
every path-like token in tracked source resolve to something a cloner has". The
answer is derived from ``git ls-files``, never from the filesystem: this machine
holds ``runs/``, ``reports/`` and ``data/`` and a clone holds none of them, so a
filesystem answer would pass here and fail in CI, which is how three earlier
controls in this repository came to be believed while wrong.

WHAT IS AN OBLIGATION AND WHAT IS DECLARED
-------------------------------------------
An obligation is an unresolved path-like token cited by a tracked file. Two
escapes exist and both are enumerated rather than inferred:

* ``DECLARED`` - one entry per (citing file, token) pair, grouped under a
  category that carries the reason. The categories are listed below.
* ``DEBT`` - one entry per citing file, with a count and an owner. These are
  real instances of the class, in packages this lead may not edit. A pin, not an
  acceptance: it fails when it rises, when it falls, and when it goes stale.

CATEGORIES OF ``DECLARED``, checked against the code in both directions:

``specimen``   a path INVENTED by a test or a tool to exercise a scanner; it is
               never opened and it names nothing.
``output``     a DESTINATION this program writes, named in the module that
               writes it or in a test of that module.
``corpus``     the untracked fire corpus and the artifacts derived from it.
``proposal``   a path named as a DESIGN that does not exist yet, in a passage
               that says so in the sentence above it.
``evidence``   a cited run artifact already declared, with its measured reason,
               in ``tools/cited_runs.py``.

That list was free prose until ADR-116 and it was wrong in BOTH directions: it
promised a category for "a path in another system entirely" that the code has
never implemented, and ``proposal`` was implemented and undocumented, so a real
zip member could not be declared while a phantom category invited a declaration
that could not be made. ``tests/test_documented_categories.py`` now reads the
block above by its header and compares it with ``DECLARED`` itself, so a
category added to one and not the other fails. What that check CANNOT see is a
category described in English without naming its key, which is the form the
phantom took, and the convention above - name the key, then describe it - is
what buys the check.

WHAT THIS SCANNER CANNOT SEE, PRINTED BESIDE ITS VERDICT
---------------------------------------------------------
A path ASSEMBLED at run time is invisible to any token scan.
``tools/claimaudit.py`` builds ``root / "coordination" / "DECISIONS.md"`` from
three fragments and no walk over string literals will ever find it.

So is a BARE filename with no directory component, and that exclusion has a
measured reason: resolving it would make ordinary code (``np.log``, ``a.py``,
``good.json``) an obligation, and the tier is dominated by artifact NAMES
(``tensor.zarr``, ``manifest.json``) that this scan already sees in their
directory form. The tier is COUNTED on every run rather than described.

Two subclasses of it are checked elsewhere, and the hand-off is written as data
in ``DELEGATED``: each entry names the module, the reader inside it, and a PROBE
token that reader must catch. Until ADR-116 this docstring printed instead that
the whole class was "left to ``tests/test_hygiene.py``, whose pattern set covers
that class" - and it was not; that pattern set covered exactly one token of it
and the rest was checked by nothing. Neither module was wrong about itself. The
HAND-OFF was the unowned surface, so it is the thing that now carries a test.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cited_runs import EXEMPT as RUNS_EXEMPT  # noqa: E402

#: The only shape assumption: a run of characters a POSIX path may use, holding
#: at least one ``/``. Deliberately wider than any directory this repository
#: has, so a citation of a path nobody anticipated lands in a bucket rather than
#: falling out of the scan.
PATH_TOKEN: Final = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./-]*")

#: Extensions that make a token a FILE citation, i.e. something a reader opens.
FILE_SUFFIXES: Final[tuple[str, ...]] = (
    ".json",
    ".py",
    ".csv",
    ".md",
    ".png",
    ".pt",
    ".txt",
    ".yaml",
    ".yml",
    ".zarr",
    ".nc",
    ".tif",
    ".html",
    ".svg",
    ".cfg",
    ".toml",
    ".ini",
    ".sh",
    ".lock",
    ".npz",
    ".parquet",
    ".geojson",
    ".f90",
)

#: Characters that, immediately before a match, mean the match is not a
#: repository-relative path: a shell or make VARIABLE expansion (``$$d/...``,
#: ``$root/...``) and the continuation of an ABSOLUTE path (``/Volumes/...``).
#: Structural, so no list of known-bad strings is needed and none is kept.
_NOT_A_START: Final = "$/\\%{"


@dataclass(frozen=True)
class Reference:
    """One path-like token, in one file, at the lines where it occurs."""

    citer: str
    token: str
    lines: tuple[int, ...]
    resolution: str

    @property
    def resolved(self) -> bool:
        return self.resolution != "unresolved"

    def __str__(self) -> str:
        where = ",".join(str(n) for n in self.lines)
        return f"{self.citer}:{where} -> {self.token} [{self.resolution}]"


# --------------------------------------------------------------------------
# the declared escapes, each with the reason its category exists
# --------------------------------------------------------------------------

#: (category -> (reason, the (citer, token) pairs it covers)).
#:
#: KEYED BY THE PAIR, NOT BY THE TOKEN, and that is load-bearing. The same path
#: can be a destination in the module that writes it and a provenance claim in a
#: module that reads it: ``reports/figures/s5_block5_anatomy.json`` is both, in
#: two different packages. A token-keyed exemption would silently bless the
#: second because of the first, across a fence, which is exactly the judgement
#: this lead may not make for another lead's file.
DECLARED: Final[dict[str, tuple[str, tuple[tuple[str, str], ...]]]] = {
    "specimen": (
        "A path INVENTED by a test or a tool to exercise a scanner. It is never "
        "opened, it names nothing, and it exists so that a check can be shown to "
        "fire. Spelling these in halves is the alternative and it is used where "
        "the file is read by the very scan it tests; doing it everywhere would "
        "make the tests unreadable for no gain in truth.",
        (
            ("tests/test_cited_runs.py", "deep/er/still-here_v2.json"),
            ("tests/test_cited_runs.py", "kernel-nll_only-20260808-044220/model.json"),
            ("tests/test_code_fingerprint.py", "eval/elasticity.py"),
            ("tests/test_code_fingerprint.py", "eval/gone.py"),
            ("tests/test_hygiene.py", "reports/figures/x.png"),
            ("tests/test_hygiene.py", "tests/test_a.py"),
            ("tests/test_hygiene.py", "tests/test_b.py"),
            ("tests/test_hygiene.py", "tests/test_x.py"),
            ("tests/test_hygiene.py", "umes/scratch2/fires/tensor.zarr"),
            ("tests/test_large_file_guard.py", "vendored/runs/m19_collapse_curve.json"),
            ("tests/test_prose_output_literals.py", "eval/r.py"),
            ("tests/test_prose_output_literals.py", "sim/new.py"),
            ("tests/test_prose_output_literals.py", "sim/old.py"),
            ("tests/test_prose_output_literals.py", "src/wildfire_nowcast/eval/y.py"),
            ("tests/test_prose_output_literals.py", "src/wildfire_nowcast/sim/y.py"),
            ("tests/test_prose_output_literals.py", "tools/x.py"),
            ("tests/test_typecheck_config.py", "tools/no_such_guard.py"),
            ("tests/unit/common/test_codefingerprint.py", "sub/a.py"),
            ("tests/unit/common/test_codefingerprint.py", "sub/b.py"),
            ("tools/claimaudit.py", "src/wildfire_nowcast/__clean__.py"),
            ("tools/claimaudit.py", "src/wildfire_nowcast/__planted__.py"),
            ("tools/typecheck.py", "src/wildfire_nowcast/a/b.py"),
            ("src/wildfire_nowcast/common/contract.py", "path/to/tensor.zarr"),
            ("src/wildfire_nowcast/common/contract.py", "path/to/x.zarr"),
        ),
    ),
    "output": (
        "A DESTINATION this program writes, named in the module that writes it "
        "or in a test of that module. A cloner is not expected to have the file; "
        "they are expected to be able to produce it, and the code that does is "
        "in the tree. This is a claim about ROLE and it is made only for files "
        "this lead owns.",
        (
            ("Makefile", "outputs/synthetic_fire/tensor.zarr"),
            ("configs/synthetic_smoke.yaml", "outputs/synthetic_fire/tensor.zarr"),
            ("src/wildfire_nowcast/common/synthetic.py", "outputs/synthetic_fire/tensor.zarr"),
            ("tests/test_sim_coarsen.py", "reports/figures/playthrough_coarsening.json"),
            ("tests/test_sim_dashboard.py", "reports/figures/gate_dashboard_g2_record.png"),
            ("tests/test_sim_review.py", "reports/figures/elmfire_degeneracy_verdict.json"),
            ("tests/test_sim_review.py", "reports/review.html"),
            ("tests/test_sim_s5_report.py", "reports/figures/s5_block5_anatomy.json"),
            ("src/wildfire_nowcast/data/cli.py", "data/qa_audit.json"),
            ("src/wildfire_nowcast/data/crossings.py", "data/events/crossings.json"),
            ("src/wildfire_nowcast/data/isotropy.py", "data/events/subthreshold_isotropy.json"),
            (
                "src/wildfire_nowcast/data/leakage.py",
                "data/leakage/c1_6_channel_leakage.json",
            ),
            # Declared at S12 by the owner of `sim/`, for that package only. The
            # test applied is ROLE and it is stated so the next reader can apply it:
            # PLUMBING - where a program of this tree PUTS an artifact or GETS
            # one - is declared here; PROVENANCE - "this file backs the number
            # printed beside it" - is NOT, and the four provenance citations this
            # package held were repaired in source instead. The producing command
            # is named at every citation site below, so the reason above is
            # checkable where the path is written and not only in this table.
            ("src/wildfire_nowcast/sim/coarsen.py", "reports/figures/playthrough_coarsening.json"),
            # S13, same lead and same ROLE test as the S12 block above: both are
            # in the usage block of the module that writes them, and the command
            # that produces each is the line they sit on.
            ("src/wildfire_nowcast/sim/collapse.py", "outputs/synthetic_fire/tensor.zarr"),
            ("src/wildfire_nowcast/sim/collapse.py", "reports/figures/collapse_per_horizon.json"),
            # S14, same lead and same ROLE test again: the page is written by the
            # command on the line that names it, in this module's usage block.
            (
                "src/wildfire_nowcast/sim/s14_report.py",
                "reports/figures/s14_collapse_real_model.png",
            ),
            ("src/wildfire_nowcast/sim/dashboard.py", "reports/figures/dashboard.png"),
            ("src/wildfire_nowcast/sim/e1_report.py", "reports/figures/e1_stage_decay.png"),
            ("src/wildfire_nowcast/sim/elmfire.py", "reports/figures/elmfire_native_smoke.json"),
            ("src/wildfire_nowcast/sim/ensemble.py", "outputs/synthetic_fire/tensor.zarr"),
            ("src/wildfire_nowcast/sim/ensemble.py", "reports/figures/synthetic_ensemble.png"),
            ("src/wildfire_nowcast/sim/growth.py", "reports/figures/growth_anatomy.png"),
            (
                "src/wildfire_nowcast/sim/playthrough.py",
                "reports/figures/elmfire_degeneracy_verdict.json",
            ),
            (
                "src/wildfire_nowcast/sim/playthrough.py",
                "reports/figures/playthrough_coarsening.json",
            ),
            (
                "src/wildfire_nowcast/sim/playthrough.py",
                "reports/figures/playthrough_nondegeneracy.json",
            ),
            ("src/wildfire_nowcast/sim/review.py", "reports/review.html"),
            ("src/wildfire_nowcast/sim/rundash.py", "reports/figures/gate_dashboard.png"),
            ("src/wildfire_nowcast/sim/rundash.py", "reports/figures/iou_decomposition.json"),
            ("src/wildfire_nowcast/sim/rundash.py", "reports/figures/null_check_report.json"),
            (
                "src/wildfire_nowcast/sim/selftest.py",
                "reports/figures/playthrough_nondegeneracy.json",
            ),
            ("src/wildfire_nowcast/sim/stencil.py", "reports/figures/stencil.png"),
        ),
    ),
    "corpus": (
        "The untracked fire corpus and its derived artifacts, ~700 MB, "
        "deliberately outside the repository. Every citer below either SKIPS "
        "when the path is absent or is a command line a reader types after "
        "building the corpus, so the absence is handled rather than assumed "
        "away.",
        (
            ("Makefile", "data/fires/2019_kincade/tensor.zarr"),
            ("tests/conftest.py", "data/fires/2019_kincade/tensor.zarr"),
            ("tests/test_contracts.py", "data/fires/2019_kincade/tensor.zarr"),
            ("tests/test_large_file_guard.py", "data/fires/2019_kincade/tensor.zarr"),
            ("tests/test_sim_review.py", "data/fires/2019_kincade/tensor.zarr"),
            ("tests/test_adopted_selftests.py", "data/events/crossings.json"),
            ("tests/test_data_crossings_rules.py", "data/events/crossings.json"),
            ("tests/test_data_leakage_stats.py", "data/leakage/c1_6_channel_leakage.json"),
            (
                "src/wildfire_nowcast/data/crossings_selftest.py",
                "data/events/crossings.json",
            ),
            ("src/wildfire_nowcast/data/isotropy.py", "data/events/crossings.json"),
            ("src/wildfire_nowcast/data/isotropy.py", "data/qa_audit.json"),
            # S12: both are command lines a reader types after building
            # the corpus, in the usage block at the top of the module.
            (
                "src/wildfire_nowcast/sim/components.py",
                "data/fires/2020_july_complex/tensor.zarr",
            ),
            ("src/wildfire_nowcast/sim/movie.py", "data/fires/2019_kincade/tensor.zarr"),
        ),
    ),
    "proposal": (
        "A path named as a DESIGN that does not exist yet, in a passage that "
        "says so in the sentence above it. These are not broken citations: no "
        "reader is invited to open them. The category is deliberately narrow and "
        "each pair is a deliberate edit, because 'it is only a proposal' is the "
        "sentence an escape hatch would be built out of.",
        (
            ("tests/README.md", "tests/contract/test_c1_tensor.py"),
            ("tests/README.md", "tests/contract/test_c2_manifest.py"),
            ("tests/README.md", "tests/contract/test_c3_norm_stats.py"),
            ("tests/README.md", "tests/contract/test_c6_metrics.py"),
            ("tests/README.md", "tests/contract/test_c8_splits.py"),
        ),
    ),
    "evidence": (
        "Cited run artifacts that stay out of the tree, already declared with "
        "their measured reason in tools/cited_runs.py::EXEMPT. Re-declared here "
        "rather than re-argued: the membership is CHECKED against that dict, so "
        "the two cannot drift.",
        (
            ("src/wildfire_nowcast/common/separation.py", "runs/baselines-20260809-035037"),
            ("tests/test_cited_runs.py", "runs/baselines-20260808-095003"),
            ("tests/test_playthrough_separation.py", "runs/baselines-20260809-035037"),
            ("tests/test_published_constants.py", "runs/baselines-20260809-073414"),
            ("tools/cited_runs.py", "runs/baselines-20260808-095003"),
            ("tools/cited_runs.py", "runs/baselines-20260809-035037"),
            ("tools/cited_runs.py", "runs/baselines-20260809-073414"),
            ("tools/cited_runs.py", "runs/baselines-20260809-102243"),
            # S12: the two modules that READ the records these two
            # figures are built from. Membership is checked against
            # tools/cited_runs.py::EXEMPT, so this borrows that reason and does
            # not restate it.
            ("src/wildfire_nowcast/sim/review.py", "runs/baselines-20260808-095003"),
            ("src/wildfire_nowcast/sim/review.py", "runs/baselines-20260809-073414"),
            ("src/wildfire_nowcast/sim/review.py", "runs/baselines-20260809-102243"),
            ("src/wildfire_nowcast/sim/s5_report.py", "runs/baselines-20260809-073414"),
        ),
    ),
}

#: The tail every ``evidence`` entry above is missing, so the pair table stays
#: readable and the full token is still what gets matched.
_EVIDENCE_TAIL: Final = "/results.json"

#: The two files that CARRY the declarations, and are therefore citers of every
#: path they declare.
#:
#: WITHOUT THIS THEY WOULD BE THEIR OWN LARGEST OFFENDER, and the tempting fix -
#: skip these two paths - would make the files guaranteed to contain invented
#: paths the only files that can never report one. That is the shape of a check
#: that cannot fail, and this repository has shipped three of them. So the rule
#: is narrow and it is checked: inside these two files, a token resolves ONLY if
#: it is a token this module already declares somewhere. Anything else is an
#: ordinary unresolved citation and fails, which
#: ``test_the_declaration_files_hide_nothing`` asserts by construction rather
#: than by inspection.
DECLARATION_FILES: Final[tuple[str, ...]] = (
    "tools/cited_paths.py",
    "tests/test_cited_paths.py",
)

#: Unresolved citations in packages this lead may NOT edit, per citing file.
#:
#: A PIN, NOT AN ACCEPTANCE, and it fails in four directions exactly as
#: ``prose_scan.OUTPUT_LITERAL_DEBT`` does: an undeclared file fails, a count
#: that RISES fails, a count that FALLS fails so a sweep is recorded when it
#: happens, and an entry that reaches zero is STALE and fails.
#:
#: OWNERS. Every entry is in a package the author of this module does not write
#: to, and each was reported to that package's owner WITH the enumeration rather
#: than swept across the boundary. That is why ``docs/interfaces.md`` is no
#: longer here: this module reported the dead ``null_check`` single-file path to
#: the contract's owner instead of editing the contract, and the owner
#: discharged it at v2.17 (ADR-113). The enumeration did the work; crossing the
#: fence would have produced the same diff with none of the review.
DEBT: Final[dict[str, int]] = {
    "src/wildfire_nowcast/data/gofer.py": 1,
    "src/wildfire_nowcast/eval/regime_calibration.py": 1,
    # S12: 26 of the 27 discharged - 22 declared above by ROLE and 4
    # repaired in source, where the citation was PROVENANCE for a published
    # number and the target can never be made reachable. The one that remains is
    # not a judgement call: `sim/review.py` publishes
    # `Provenance for the three bullets: <the artifact>` on the reviewer page,
    # which is the exact class this module exists to catch, and its value is
    # pinned by `tests/test_sim_review.py:566`, which is in a package this
    # package's owner does not write to. Reported to that owner rather than
    # swept across the boundary.
    "src/wildfire_nowcast/sim/review.py": 1,
}

UNSEEN_BY_CONSTRUCTION: Final = (
    "a path ASSEMBLED at run time from fragments (tools/claimaudit.py builds one "
    "from three), and a BARE filename with no directory component, which is COUNTED "
    "below and not resolved: resolving it would make ordinary code an obligation."
)

#: The bare-filename tier is not resolved here, and these are the subclasses of it
#: that ARE checked, WHERE, and with WHAT PROBE.
#:
#: A DELEGATION IS A CLAIM ABOUT ANOTHER CHECKER, so it is written as data rather
#: than as a sentence. Each entry is (module, reader, probe): the module a reader
#: can open, the callable inside it that does the work, and a token that callable
#: MUST return. ``tests/test_cited_paths.py`` imports the module BY THIS PATH and
#: runs the probe through the named reader, in both directions - the probe must be
#: caught and ordinary text must not be. Two things follow, and both are the point.
#: A delegation to a check that does not perform it fails HERE, which is the defect
#: ADR-116 found after this module had printed the claim on every run for weeks.
#: And a delegation whose class has emptied fails too, because the probe stops
#: being caught, so a caveat cannot outlive the thing it describes.
DELEGATED: Final[dict[str, tuple[str, str, str]]] = {
    # The tell scan. This is the ONE bare filename the pattern set covered when the
    # sentence claimed it covered the class, and it is the probe for that reason.
    # Spelled in halves because this file is scanned by that same reader.
    "a bare filename naming PRIVATE TOOLING": (
        "tests/test_hygiene.py",
        "tells_in",
        "CLAUD" + "E" + ".md",
    ),
    # I18. Derived from git history, not from a list: a name this repository USED
    # to carry and no longer does. The probe is written literally, and is pinned in
    # that module's DELETED_FILE_CITATIONS, because a probe spelled in halves would
    # only prove that the reader can catch a token nobody writes.
    "a bare filename this repository DELETED": (
        "tests/test_hygiene.py",
        "deleted_file_names_in",
        "null_check.py",
    ),
}


def _declared_pairs() -> dict[tuple[str, str], str]:
    """``{(citer, token): category}``, with the evidence tails restored."""
    out: dict[tuple[str, str], str] = {}
    for category, (_reason, pairs) in DECLARED.items():
        for citer, token in pairs:
            full = token + _EVIDENCE_TAIL if category == "evidence" else token
            out[(citer, full)] = category
    return out


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------


def tracked_files(root: Path) -> list[str]:
    """``git ls-files``: the published tree, which is exactly the tracked tree."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(p for p in out.stdout.splitlines() if p)


def _is_path_shaped(token: str) -> bool:
    """A token that names a file: it has a directory part and a real extension."""
    if "/" not in token or not token.endswith(FILE_SUFFIXES):
        return False
    segments = token.split("/")
    if any(segment == "" for segment in segments):
        return False
    # ``.md/.rst/.txt`` in a help string is three extensions, not a path. A final
    # segment that is nothing but an extension is never a filename.
    return not segments[-1].startswith(".")


def tokens_in(text: str) -> Iterator[tuple[str, int]]:
    """``(token, line)`` for every path-shaped token, with the structural skips applied."""
    for match in PATH_TOKEN.finditer(text):
        token = match.group(0).rstrip("./-")
        start = match.start()
        if start > 0 and text[start - 1] in _NOT_A_START:
            continue
        if "://" in text[max(0, start - 10) : start + 4]:
            continue
        if not _is_path_shaped(token):
            continue
        yield token, text[:start].count("\n") + 1


def bare_tokens_in(text: str) -> Iterator[tuple[str, int]]:
    """``(token, line)`` for every FILENAME with no directory component.

    THE TIER THIS MODULE DOES NOT RESOLVE, exported so the checks that DO cover
    parts of it read the same definition of "bare filename" that this scanner
    excludes. Two definitions of one boundary is how a hand-off acquires a gap
    neither side can see: `tests/test_hygiene.py` calls this rather than writing
    a second pattern, and the same structural skips apply on both sides of the
    boundary because there is only one implementation of them.
    """
    for match in PATH_TOKEN.finditer(text):
        token = match.group(0).rstrip("./-")
        start = match.start()
        if start > 0 and text[start - 1] in _NOT_A_START:
            continue
        if "/" in token or token.startswith("."):
            continue
        if not token.endswith(FILE_SUFFIXES):
            continue
        yield token, text[:start].count("\n") + 1


def _suffix_index(files: Iterable[str]) -> dict[str, int]:
    """``{trailing path fragment: how many tracked files end with it}``."""
    index: dict[str, int] = {}
    for rel in files:
        parts = rel.split("/")
        for i in range(len(parts)):
            fragment = "/".join(parts[i:])
            index[fragment] = index.get(fragment, 0) + 1
    return index


@dataclass
class Enumeration:
    """Every path-like reference in the tracked tree, classified."""

    references: tuple[Reference, ...] = ()
    problems: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    artifact_tier: int = 0
    bare_tier: int = 0

    def of_resolution(self, resolution: str) -> list[Reference]:
        return [r for r in self.references if r.resolution == resolution]


def enumerate_references(
    root: Path,
    *,
    declared: dict[tuple[str, str], str] | None = None,
    debt: dict[str, int] | None = None,
) -> Enumeration:
    """Walk the tracked tree, resolve every path-like token against the git index.

    ``declared`` and ``debt`` are injectable so the tests can run the REAL walk
    over a throwaway repository. Reading the module-level tables unconditionally
    would make every synthetic corpus report this repository's debt as stale,
    which is a check answering a question about a tree it was not given.
    """
    pairs = _declared_pairs() if declared is None else declared
    pinned = DEBT if debt is None else debt
    files = tracked_files(root)
    tracked = set(files)
    suffixes = _suffix_index(files)

    found: dict[tuple[str, str], list[int]] = {}
    artifact_tier = 0
    bare_tier = 0
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        if rel.startswith("runs/"):
            # ARTIFACT TIER, excluded and counted. A tracked run record is
            # EVIDENCE: the paths inside it are what that run read and wrote,
            # written by the code, and they may not be edited to look better.
            # `tools/cited_runs.py` draws the same line for the same reason. It
            # is the public READING surface that carries citation obligations,
            # and a machine-written record is not one.
            artifact_tier += sum(1 for _ in tokens_in(path.read_bytes().decode("utf-8", "replace")))
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        bare_tier += sum(1 for _ in bare_tokens_in(text))
        for token, line in tokens_in(text):
            found.setdefault((rel, token), []).append(line)

    references: list[Reference] = []
    undeclared: dict[str, int] = {}
    declared_tokens = {token for _citer, token in pairs}
    for (rel, token), lines in sorted(found.items()):
        if token in tracked:
            resolution = "tracked"
        elif suffixes.get(token, 0) > 0:
            resolution = "suffix"
        elif (rel, token) in pairs:
            resolution = pairs[(rel, token)]
        elif rel in DECLARATION_FILES and token in declared_tokens:
            resolution = "declaration"
        else:
            resolution = "unresolved"
            undeclared[rel] = undeclared.get(rel, 0) + 1
        references.append(Reference(rel, token, tuple(sorted(set(lines))), resolution))

    enum = Enumeration(references=tuple(references))
    for resolution in [r.resolution for r in references]:
        enum.counts[resolution] = enum.counts.get(resolution, 0) + 1
    enum.artifact_tier = artifact_tier
    enum.bare_tier = bare_tier

    enum.problems.extend(_audit_debt(undeclared, pinned))
    enum.problems.extend(_audit_declarations(pairs, set(found)))
    return enum


def _audit_debt(undeclared: dict[str, int], debt: dict[str, int] | None = None) -> list[str]:
    """The four directions, in the order a reader needs them."""
    pinned = DEBT if debt is None else debt
    problems: list[str] = []
    for rel, count in sorted(undeclared.items()):
        if rel not in pinned:
            problems.append(
                f"UNRESOLVABLE: {rel} cites {count} path(s) a cloner cannot open, "
                "and is not declared. Name the thing inline, cite something tracked, "
                "or declare the pair with its reason."
            )
        elif count > pinned[rel]:
            problems.append(f"RISEN: {rel} carried {pinned[rel]}, now cites {count}")
        elif count < pinned[rel]:
            problems.append(
                f"FALLEN: {rel} carried {pinned[rel]}, now cites {count}; set it to {count}. "
                "A pin that tolerates a fall leaves slack a later commit can refill."
            )
    for rel, expected in sorted(pinned.items()):
        if undeclared.get(rel, 0) == 0:
            problems.append(f"STALE: {rel} is clean (declared {expected}); remove it from DEBT")
    return problems


def _audit_declarations(pairs: dict[tuple[str, str], str], seen: set[tuple[str, str]]) -> list[str]:
    """A declaration that no longer describes anything is an allow-list forming."""
    problems = [
        f"STALE DECLARATION: {citer} no longer cites {token}; remove the pair"
        for (citer, token) in sorted(pairs)
        if (citer, token) not in seen
    ]
    # The `evidence` category borrows another module's reasoning, so it is held
    # to that module's list rather than to a copy of it. Two dicts saying the
    # same thing is one dict and one thing that will disagree with it later.
    problems.extend(
        f"UNBACKED EVIDENCE DECLARATION: {token} is declared evidence by {citer} and is "
        "not in tools/cited_runs.py::EXEMPT, which is where that reason lives"
        for (citer, token), category in sorted(pairs.items())
        if category == "evidence" and token not in RUNS_EXEMPT
    )
    return problems


def report(enum: Enumeration) -> str:
    """The verdict, with every category it excluded printed beside it."""
    lines = ["REACHABILITY of path-like tokens in the tracked tree (NOT support):"]
    for resolution in sorted(enum.counts):
        lines.append(f"  {resolution:>10}  {enum.counts[resolution]:>4}")
    lines.append(f"  {'TOTAL':>10}  {len(enum.references):>4}")
    lines.append(
        f"declared debt: {sum(DEBT.values())} reference(s) across {len(DEBT)} file(s) "
        "owned by other leads. A PIN: it fails when it rises AND when it falls."
    )
    lines.append(
        f"EXCLUDED, ARTIFACT TIER: {enum.artifact_tier} path token(s) inside tracked "
        "runs/ records. Machine-written evidence of what a run read and wrote, not a "
        "reading surface, and not editable to look better."
    )
    lines.append(
        f"EXCLUDED, BARE FILENAME TIER: {enum.bare_tier} token(s) with no directory "
        "component, MEASURED here rather than described. Two subclasses of it are "
        "delegated, and each delegation carries a probe another test executes:"
    )
    for klass, (module, reader, _probe) in sorted(DELEGATED.items()):
        lines.append(f"  DELEGATED  {klass}: {module} -> {reader}()")
    lines.append(
        "  The REST of the tier - artifact names and specimens a test invents - is "
        "UNCHECKED by anything, and saying so is the whole of what this line claims."
    )
    lines.append(f"NOT SEEN BY THIS SCANNER: {UNSEEN_BY_CONSTRUCTION}")
    lines.append(
        "WHAT A PASS MEANS: every path named here is in the git index, so a cloner "
        "can open it. It does NOT mean the file supports the claim beside it."
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root (default: cwd)")
    parser.add_argument("--show", default="", help="list references with this resolution")
    args = parser.parse_args(list(argv) if argv is not None else None)

    enum = enumerate_references(Path(args.repo).resolve())
    print(report(enum))
    if args.show:
        for ref in enum.of_resolution(args.show):
            print(f"    {ref}")
    for problem in enum.problems:
        print(f"  [FAIL] {problem}")
    print(f"verdict: {'PASS' if not enum.problems else 'FAIL'}")
    return 0 if not enum.problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
