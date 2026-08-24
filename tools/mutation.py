"""Mutation sweep over ``common/`` and ``eval/``, pinned to the SET of survivors.

A green suite says the tests ran. It does not say they would have noticed. The
external audit measured that directly: a sweep over the best-covered 22 modules
killed 40 of 66 single-site mutants and left **26 survivors**, including a
4-connectivity table that could be made asymmetric, a dilation slice that could be
moved by one, an inverted ``not`` inside C-4.2's own clause, and a ``>`` that could
become ``>=`` so that ties count as advantages. Every one of those left 745 tests
passing.

This is that measurement made repeatable and turned into a gate. THE PIN IS A SET
OF SURVIVOR DESCRIPTORS, NOT A COUNT OF SURVIVORS (ADR-154): the gate compares the
measured set against :data:`PINNED_SURVIVORS` and reports both directions, the
survivors that APPEARED and the survivors that DISAPPEARED. It fails on either. A
count was tried first and drifted from 21 to 25 unseen; worse, it could not have
seen the event it existed for, because one old survivor killed while one new one
appears leaves the total where it was. **A count pin is blind to a swap.**

**It runs in a git worktree, never in the working tree.** Two of this project's
recorded process failures were a lead editing a file another lead was running
against (C-4 breaches, ADR-052 (5), ADR-053). A mutation sweep edits `eval/` by
construction, so running it in place would breach that fence a hundred times per
invocation. The workspace is built from ``HEAD``, then every tracked file that
differs in the working tree is copied over it and the number of carried files is
PRINTED, so the sweep measures the tree the developer is looking at and says how
it got there.

**Five controls, because a sweep that reports zero survivors is exactly what a
broken sweep reports.**

1. The workspace must import ITS OWN copy of ``wildfire_nowcast``. The editable
   install points ``sys.path`` at the real ``src/``, so without this the mutants
   would never be loaded and all of them would read SURVIVED.
2. Every mutant is read back through the interpreter that is about to run the
   tests, from ``module.__file__``, and the run is abandoned if the mutated token
   is not there. A mutation that failed to apply is not a survivor.
3. THE BYTECODE IS CHECKED, NOT ONLY THE SOURCE. CPython invalidates a ``.pyc``
   on ``(source mtime in WHOLE SECONDS, source size)``, so a same-length edit made
   and reverted inside one second is invisible and the stale bytecode runs
   instead: the file verifiably says ``max`` and the program returns the ``min``
   answer. Reproduced here before this guard was written. Every run purges
   ``__pycache__``, sets ``PYTHONDONTWRITEBYTECODE``, and compares
   ``marshal.dumps`` of the code object the LOADER would hand the interpreter
   against a fresh ``compile`` of the file on disk. A mismatch is
   ``STALE_BYTECODE``: a refusal to measure, which is a different claim from
   "nothing caught it" and is never counted as one.
4. A MUTANT THAT DOES NOT PARSE IS NOT A MUTANT. The token rules are blind to
   grammar, so ``*`` becomes ``/`` inside ``run(["git", *args])`` too, and the
   result does not compile. Every test then fails on a collection error and the
   mutant reads KILLED: a test is credited with noticing something it never saw,
   which is the exact mirror of the false survivor. The first baseline contained
   three. Such sites are dropped at enumeration.
5. The unmutated workspace must exit 0 first. Any test that fails there is an
   environment artifact of the sandbox (no ``.venv`` of its own, no installed
   hooks), is deselected for the sweep, and is NAMED in the output rather than
   quietly dropped, because a suite that is already red kills every mutant.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import io
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

# ADR-103: a logger, and NO handler configured at import. This module
# deliberately imports NOTHING from `wildfire_nowcast`, because `common/` is one
# of the packages it mutates and a sweeper must not run through code it is
# breaking. It therefore does not call `common/logs.configure_logging` either:
# unconfigured, `logging`'s lastResort handler still puts WARNING and above on
# stderr, which is all this module emits.
logger = logging.getLogger(__name__)

#: The packages under sweep. ``common/`` is infra's and ``eval/`` is the scoring
#: code every published number came out of; the plan names exactly these two.
#:
#: ``model/`` IS NOT SWEPT, AND THE REASON THIS SCOPE WAS CHOSEN HAS EXPIRED.
#: The original argument was comparability with the external audit's 26 survivors
#: of 66. ADR-085 (2) established that figure was wrong three separate ways,
#: including three of its 55 "kills" being mutants that do not parse, so there is
#: nothing left to be comparable with. What is left out is 16 modules and 9,208
#: lines holding the transition kernel, the latent and both baselines, with 2,624
#: mutable sites and zero mutation measurement.
#: PRICED, so the decision is a decision and not an omission: adding ``"model"``
#: here is the ONLY code change required, because ``target_modules`` derives the
#: file list from ``git ls-files``. It would add 48 rows, 9 of them with no
#: mutable site, so 39 mutants, at roughly 100 s each: about 17 minutes on four
#: workers against the 43 measured here. The real cost is not the sweep, it is
#: the burn-down: at the survivor rate the other two packages showed before this
#: burst, 39 mutants would put something like 15 to 25 new survivors on the
#: pin, all of them new debt against this gate. Widening is therefore a
#: one-line config change plus a package's worth of work, and it is not done here.
TARGET_PACKAGES: Final = ("common", "eval")

#: Where in each module's site list the mutants are taken. Three fractions, so a
#: module contributes a site near its top, its middle and its end rather than one
#: site that could be anywhere. The figure is arbitrary in the way a sample size
#: is arbitrary: it is not derived from anything, and changing it RE-SAMPLES every
#: module - which now shows up as a wholesale APPEARED/DISAPPEARED diff rather than
#: as a number moving - so it is pinned here and moved deliberately or not at all.
SAMPLE_FRACTIONS: Final = (0.2, 0.5, 0.8)

#: Single-token substitutions. A boundary becomes its neighbour, an inequality
#: flips, a conjunction weakens: the defects that survive review and change a
#: verdict, not the ones that raise on the first call.
OPERATOR_MUTATIONS: Final = {
    ">=": ">",
    "<=": "<",
    ">": ">=",
    "<": "<=",
    "==": "!=",
    "!=": "==",
    "+": "-",
    "-": "+",
    "*": "/",
}
KEYWORD_MUTATIONS: Final = {
    "and": "or",
    "or": "and",
    "True": "False",
    "False": "True",
    "not": "",
}

#: THE IDENTITY OF ONE MUTANT, and the only identity in this file. Six fields:
#: ``(module, stripped source line, WHICH OCCURRENCE of that line in the module,
#: index of the site on that line, old token, new token)``.
#:
#: Line NUMBER is deliberately absent: it moves under any edit above the site, and
#: a descriptor that renames itself when an unrelated line is inserted is noise
#: rather than a pin. The two counters are what make the identity INJECTIVE, and
#: both are load-bearing on the live corpus rather than defensive:
#:
#: * the site index separates two sites on ONE line - ``max(a, 0) + min(b, 0)``
#:   has two ``0`` sites and one exemption must not cover both;
#: * the occurrence ordinal separates ONE site form on two IDENTICAL lines.
#:   MEASURED, not supposed: ``common/pooling.py`` carries the keyword-only marker
#:   ``*,`` at lines 86 and 147, BOTH are sampled, and without this field the two
#:   mutants share one descriptor. Three pinned survivors need it today
#:   (``eval/power.py``'s eighth bare ``False,``, ``eval/stage.py``'s eighth
#:   ``*,``, and the second of two identical ``independent_floor_sd=`` lines in
#:   ``eval/collapse_curve.py``). A SET pinned on a descriptor that is not
#:   injective is blind to a swap between the two sites it conflates - which is
#:   the defect ADR-154 replaced the count to close, one level down.
MutantKey = tuple[str, str, int, int, str, str]

#: A NOT-KILLED MUTANT IS ONE OF THREE THINGS, AND LUMPING THEM MANUFACTURES THE
#: DEFECT THIS GATE EXISTS TO PREVENT.
#:
#: * ``SURVIVED`` - real debt. A test could kill it and none does. This is the
#:   only state :data:`PINNED_SURVIVORS` holds.
#: * ``EQUIVALENT`` - provably unkillable. No input distinguishes the mutant from
#:   the original, so demanding a test for it is demanding a test that asserts
#:   something false. Declared below, WITH the proof, and the declaration fails if
#:   the proof stops existing.
#: * ``STALE_BYTECODE`` / ``NOT_APPLIED`` - never executed. A refusal to measure,
#:   which is not the same claim as "nothing caught it" and must never be counted
#:   as one.
#:
#: A pin that lumped the second into the first would push a lead to write an
#: untestable test to reach zero. A pin that lumped the third into the first would
#: count a measurement that did not happen.
#:
#: Keyed by :data:`MutantKey`, the one identity this file uses for a mutant.
#: NOT by line number, which moves under any edit above it, and not by file name,
#: which would make this an allow-list. Reformat or edit the line and the entry
#: stops matching, which is correct: the proof was about that line.
EQUIVALENT_MUTANTS: Final = {
    (
        "src/wildfire_nowcast/common/states.py",
        "ys_dst = slice(max(dy, 0), h + min(dy, 0))",
        0,
        2,
        "0",
        "1",
    ): (
        "dy takes only -1, 0 and 1, so `h + min(dy, 1)` differs from `h + min(dy, 0)` at "
        "dy == 1 alone, where the end index h+1 CLIPS to h on a length-h axis. The two forms "
        "therefore agree on every input the offset table can produce.",
        "tests/test_states_geometry.py::"
        "test_the_dilation_slice_survivor_is_an_EQUIVALENT_MUTANT_and_this_is_the_proof",
    ),
    (
        "src/wildfire_nowcast/common/states.py",
        "prev_ever = np.zeros(masks.shape[1:], dtype=bool)",
        0,
        0,
        "1",
        "2",
    ): (
        "The seed is read exactly once, at i == 0, as `cur_ever & ~prev_ever`, and is then "
        "rebound to `cur_ever`, which is always full shape. So the question is whether an "
        "all-false (W,) field and an all-false (H, W) field are distinguishable under `&` and "
        "`~` against an (H, W) operand: they are not, because numpy broadcasts the trailing "
        "axis and false is the identity of the operation either way. Proved by arithmetic "
        "rather than by running the mutant, because ADR-084 showed that byte-identical output "
        "with a mutant on disk is exactly what a stale .pyc also produces.",
        "tests/unit/common/test_states.py::"
        "test_the_prev_ever_seed_survivor_is_an_EQUIVALENT_MUTANT_and_this_is_the_proof",
    ),
}

#: THE PIN, AND IT IS A SET OF SURVIVORS RATHER THAN A COUNT OF THEM (ADR-154).
#:
#: A COUNT PIN IS BLIND TO A SWAP. The integer this replaces read 21 while the
#: sweep read 25, and the reconciliation offered - "4 of the 25 sit in three
#: ``eval/`` modules that did not exist when the pin was taken, 21 + 4 = 25" - is
#: CONSISTENT WITH but does not PROVE that no old survivor moved: one old survivor
#: killed while one new one appeared gives the identical total. 25 == 25 whether
#: the population is stable or has churned entirely, so the one event the gate
#: exists to catch is the one arithmetic cannot see.
#:
#: The set is compared in BOTH directions and both are reported: survivors that
#: APPEARED, and survivors that DISAPPEARED. Corpus growth then arrives as an
#: explicit diff naming its modules, and a regression can no longer hide behind a
#: coincidentally-killed mutant. A count is still printed BESIDE this set as a
#: convenience readout; it does not adjudicate anything.
#:
#: WHAT AN ENTRY IS, and why it is not the obvious cheaper thing. Each entry is a
#: :data:`MutantKey`: content-addressed, so inserting a line above a survivor does
#: not rename it, and editing the survivor's own line DOES - which is correct,
#: because the measurement was about that line. The obvious alternative is
#: ``(module, fraction)``, which is what the sweep is actually indexed by and is
#: two fields instead of six. It is REJECTED: ``select_site`` re-samples when the
#: module's site list changes length, so ``(module, 0.5)`` is stable in NAME while
#: silently changing WHICH mutant it names. A descriptor that cannot go stale
#: cannot report drift either.
#:
#: THE SAMPLE IS NOT THE DESCRIPTOR, and this pin cannot hide that either: adding
#: one mutable token to a swept module shifts that module's three sampled sites,
#: so an unrelated edit inside ``common/`` or ``eval/`` can retire up to three
#: entries and mint three others. That churn was always happening; a count could
#: not see it, and this reports it as paired appear/disappear rows in one module.
PINNED_SURVIVORS: Final[frozenset[MutantKey]] = frozenset(
    {
        (
            "src/wildfire_nowcast/common/environment.py",
            "out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]",
            0,
            0,
            "16",
            "17",
        ),
        (
            "src/wildfire_nowcast/common/null_check/cli.py",
            "Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2))",
            0,
            0,
            "2",
            "3",
        ),
        (
            "src/wildfire_nowcast/common/null_check/cli.py",
            'lines.append(f"--- mask: {mask} " + "-" * max(0, 60 - len(mask)))',
            0,
            2,
            "0",
            "1",
        ),
        (
            "src/wildfire_nowcast/common/synthetic.py",
            "spot_col = min(nx - 1 - EDGE_RESERVE_CELLS, river_hi + max(5, nx // 20))",
            0,
            1,
            "1",
            "2",
        ),
        (
            "src/wildfire_nowcast/eval/baseline_run.py",
            # Spelled in two pieces because the whole line is 102 characters and the
            # limit is 100. The concatenation is the pinned text EXACTLY, and
            # `--check-pin` verifies every entry against the sampler, so a typo
            # here is not silent.
            '"band_ece": (band.get("reliability_summary") or {})'
            '.get(str(horizon_h), {}).get("ece"),',
            0,
            0,
            "or",
            "and",
        ),
        (
            "src/wildfire_nowcast/eval/baseline_run.py",
            "else (value < rule_value if lower_better else value > rule_value)",
            0,
            0,
            "<",
            "<=",
        ),
        (
            "src/wildfire_nowcast/eval/baseline_run.py",
            "return lines + [",
            0,
            0,
            "+",
            "-",
        ),
        (
            "src/wildfire_nowcast/eval/blocktest.py",
            "return 0.0 if t > 0 else 1.0",
            0,
            3,
            "1.0",
            "1.501",
        ),
        (
            "src/wildfire_nowcast/eval/collapse_bars.py",
            "(fine.conditional_variance * scale + fine.latent_variance * scale**2)",
            0,
            0,
            "*",
            "/",
        ),
        (
            "src/wildfire_nowcast/eval/collapse_curve.py",
            "independent_floor_sd=float(np.sqrt(np.sum(marginal * (1.0 - marginal)))),",
            1,
            1,
            "1.0",
            "1.501",
        ),
        (
            "src/wildfire_nowcast/eval/collapse_curve.py",
            'lines.append("=" * 100)',
            0,
            1,
            "100",
            "101",
        ),
        (
            "src/wildfire_nowcast/eval/labelfloor.py",
            "cache.parent.mkdir(parents=True, exist_ok=True)",
            0,
            0,
            "True",
            "False",
        ),
        (
            "src/wildfire_nowcast/eval/meanfield.py",
            "if not bins",
            0,
            0,
            "not",
            "",
        ),
        (
            "src/wildfire_nowcast/eval/meanfield.py",
            "if not mask.any():",
            0,
            0,
            "not",
            "",
        ),
        (
            "src/wildfire_nowcast/eval/meanfield.py",
            'w.fire_id, {"predicted_new_cells": 0.0, "observed_new_cells": 0.0, "n_windows": 0.0}',
            0,
            1,
            "0.0",
            "1",
        ),
        (
            "src/wildfire_nowcast/eval/metrics.py",
            "n_all_silent = sum(1 for t in defined if t.cond_outcome == FRONT_ALL_SILENT)",
            0,
            0,
            "1",
            "2",
        ),
        (
            "src/wildfire_nowcast/eval/playthrough_first_moment.py",
            "if not (_BLOB.start <= i < _BLOB.stop and _BLOB.start <= j < _BLOB.stop)",
            0,
            3,
            "and",
            "or",
        ),
        (
            "src/wildfire_nowcast/eval/power.py",
            "False,",
            7,
            0,
            "False",
            "True",
        ),
        (
            "src/wildfire_nowcast/eval/regime_calibration.py",
            '"split_fingerprint": (results.get("split_before") or {}).get("fingerprint"),',
            0,
            0,
            "or",
            "and",
        ),
        (
            "src/wildfire_nowcast/eval/regime_calibration.py",
            "if value is None or not math.isfinite(float(value)) or float(value) <= 0:",
            0,
            2,
            "or",
            "and",
        ),
        (
            "src/wildfire_nowcast/eval/response.py",
            '"insufficient": False,',
            0,
            0,
            "False",
            "True",
        ),
        (
            "src/wildfire_nowcast/eval/response.py",
            "and float(r[model_key]) > 0",
            1,
            2,
            "0",
            "1",
        ),
        (
            "src/wildfire_nowcast/eval/response.py",
            "if r.get(target) is not None",
            0,
            0,
            "not",
            "",
        ),
        (
            "src/wildfire_nowcast/eval/selftest.py",
            "for f in (0.01, 0.5, 1.0)",
            0,
            1,
            "0.5",
            "0.751",
        ),
        (
            "src/wildfire_nowcast/eval/stage.py",
            "*,",
            7,
            0,
            "*",
            "/",
        ),
    }
)

#: How the set above was obtained, so that a reader can reproduce it rather than
#: believe it. The two artifacts the older paragraphs were "recounted from" are
#: not on disk or in the index, which is half of why the set is written out here
#: site by site: a narrative cannot be diffed.
MEASURED_AT: Final = (
    "CURRENT, and it is the SET above: `python tools/mutation.py --pristine "
    "--workers 6 --no-pin --json <path>` at "
    "d7874ff20a2759bbcf336bd3b6bd688cb76cf076, carrying 0 working-tree files "
    "(run as `--no-budget`, which is what that flag was called until this commit "
    "renamed it). 138 rows, 9 with no mutable site, 129 mutants attempted, 102 "
    "KILLED, 25 SURVIVED, 2 EQUIVALENT, 0 unmeasured, 93.8 min on 6 workers. "
    "25 survivor ROWS over 25 DISTINCT SITES - no site was sampled twice today, "
    "which is the coarseness the row-counting pin had to warn about and a set "
    "cannot have. One test was deselected as an environment artifact and is named "
    "in the output: `tests/test_code_fingerprint_pins.py::"
    "test_the_transition_is_formatting_only`, which does not run in a bare "
    "worktree. The 25 descriptors are UNCHANGED from the sweep at 68ebd2f, and "
    "that was PRE-REGISTERED rather than observed: `common/` and `eval/` are "
    "byte-identical across those two commits, the sampler is deterministic, and "
    "the only suite change between them was tests being ADDED - which can kill a "
    "mutant and cannot revive one. The prediction was written down before this "
    "sweep reported and it held on all 25. "
    "Each SURVIVED row is turned into a `MutantKey` by re-deriving the site from "
    "`(module, fraction)`, which is deterministic, rather than by parsing the "
    "printed descriptor - four of the 25 descriptors printed at 68ebd2f are "
    "AMBIGUOUS at the line level (`eval/meanfield.py:91` alone holds three "
    "`0.0`->`1` sites), so the human string cannot identify a site and was never "
    "able to. "
    "PRIOR, the last COUNT pin, kept because the burn-down it records is real: "
    "`python tools/mutation.py --pristine --workers 3 --no-budget`, suite "
    "`pytest -x -m 'not slow'`, 42 modules x 3 fractions over common/ and eval/. "
    "AT 1a7c480 (this paragraph said CURRENT until the set replaced the count): "
    "126 rows, 9 with no mutable site, 117 mutants attempted, "
    "94 KILLED, 21 SURVIVED, 2 EQUIVALENT, 0 unmeasured, 43.3 min. "
    "The 21 survivors are 3 in common/ and 18 in eval/, and the drop from 58 is two "
    "packages moving at once: in common/ 20 distinct sites killed and a twenty-first "
    "proved unkillable, in eval/ 12 killed. (An earlier revision of this note said 19 "
    "and a twentieth; recounted from the two artifacts, it is 20 and a twenty-first. "
    "The budget is unaffected: it counts survivors, not kills.) THE ROWS RECONCILE "
    "EXACTLY, which is the check that the drop is work and not re-measurement: common/ "
    "held 28 survivor rows over 24 distinct sites and now holds 3, so 24 rows were "
    "killed and 1 became EQUIVALENT; eval/ went 30 rows to 18, so 12 were killed; "
    "24 + 1 + 12 = 37 = 58 - 21, leaving nothing to attribute to a corpus change. "
    "The 24 killed rows are 20 sites because paths.py:47 and null_check/__main__.py:9 "
    "were each selected by all three fractions. A cross-check that could have disagreed "
    "and did not: a separate `--only /eval/` sweep at 12e8dfb reports 18 survivors, the "
    "same 18 this run finds. "
    "PRIOR, at 4071f6c: 58 KILLED, 58 SURVIVED, 1 EQUIVALENT, 45.9 min. Before that, at "
    "cc82876 and before this gate existed, it was reported as 126 mutants / 55 killed / 62 "
    "survived, and that report was wrong twice: THREE of its kills were mutants that do not "
    "parse (see mutable_sites), so the honest figure is 114 real mutants / 52 killed / 62 "
    "survived, and EQUIVALENT was unreachable because the registry's own integrity check "
    "failed on the line it pins. Both defects were fixed before the 58 was taken, so the "
    "three numbers are comparable in the survivor column: 62 -> 58 -> 21. "
    "THE ROW-VERSUS-SITE COARSENESS HAS GONE, and it went by being killed rather than by "
    "being redefined. The budget counts ROWS, and on a module with few sites all three "
    "fractions can select the same one; at 4071f6c two sites were selected three times each "
    "(common/paths.py:47, common/null_check/__main__.py:9), so 58 rows were 54 DISTINCT "
    "survivors. Both of those sites are now dead, and the 21 rows are 21 distinct sites."
)

_PYTEST_ARGS: Final = ("-x", "-q", "-m", "not slow", "-p", "no:randomly", "-p", "no:cacheprovider")

#: Directories whose untracked files are carried into the workspace. A new test
#: file is the usual reason a sweep is being re-run, and a sweep that could not
#: see it would report the old number with total confidence.
_CARRIED_UNTRACKED: Final = ("src/", "tests/", "tools/", "configs/")


@dataclass(frozen=True)
class Site:
    """One mutable token: where it is, what it says, what it would say."""

    line: int
    col_start: int
    col_end: int
    old: str
    new: str


@dataclass
class Result:
    """The verdict on one mutant."""

    module: str
    fraction: float
    verdict: str
    descriptor: str
    n_sites: int
    exit_code: int | None = None
    seconds: float = 0.0
    detail: str = ""
    #: The content-addressed identity of this mutant, or ``None`` for a module
    #: with no mutable site. ``descriptor`` above is the READABLE form and cannot
    #: do this job: `eval/meanfield.py:91 '0.0'->'1'` names three different sites,
    #: and four of the 25 descriptors printed at 68ebd2f are ambiguous that way.
    key: MutantKey | None = None


@dataclass
class Sweep:
    """Everything one invocation measured."""

    results: list[Result] = field(default_factory=list)
    deselected: list[str] = field(default_factory=list)
    #: Temp entries present after the sweep that were not there before it. A gate
    #: that re-runs the whole suite once per mutant multiplies any per-run leak by
    #: the mutant count: a 4.7 MB temp dir left by one suite run became 70 GB and
    #: 15,462 directories across one day. Cost per invocation is not the number
    #: that matters; cost times invocations is.
    leaked: list[str] = field(default_factory=list)
    #: The sha actually swept, read out of the WORKSPACE rather than the repo.
    #: Without it a pinned number is attributable only by comparing the process
    #: start time against commit timestamps - which this session had to do, with a
    #: five-second margin, to find out which commit a pin of 21 was measured at.
    head: str = ""
    carried: int = 0
    seconds: float = 0.0

    @property
    def survivors(self) -> list[Result]:
        return [r for r in self.results if r.verdict == "SURVIVED"]

    @property
    def killed(self) -> list[Result]:
        return [r for r in self.results if r.verdict == "KILLED"]

    @property
    def equivalent(self) -> list[Result]:
        return [r for r in self.results if r.verdict == "EQUIVALENT"]

    @property
    def survivor_keys(self) -> set[MutantKey]:
        """The measured survivor SET - the field the gate adjudicates on.

        A set, not a list, so the same site sampled by two fractions is ONE
        survivor. The count could not do that: ``MEASURED_AT`` records 58 rows
        that were 54 distinct sites, and a pin over rows moves when the sampling
        repeats itself rather than when the debt does.
        """
        return {r.key for r in self.survivors if r.key is not None}

    @property
    def enumerated_keys(self) -> set[MutantKey]:
        """Every mutant this sweep gave a verdict on, whatever the verdict.

        Used to split a DISAPPEARED survivor into "a test now kills it" (the key
        is still in here) and "no such mutant exists any more" (it is not).
        """
        return {r.key for r in self.results if r.key is not None}

    @property
    def unmeasured(self) -> list[Result]:
        """Mutants that never executed. NOT survivors, and not quietly dropped."""
        return [
            r
            for r in self.results
            if r.verdict in ("STALE_BYTECODE", "NOT_APPLIED", "PROBE_FAILED")
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            # The SET first and the counts after it, in the artifact as in the
            # verdict: a reader who takes the first number they find must land on
            # the adjudicating field, not on the convenience readout.
            "pinned_survivors": sorted(PINNED_SURVIVORS),
            "survivor_keys": sorted(self.survivor_keys),
            "n_pinned": len(PINNED_SURVIVORS),
            "measured_at": MEASURED_AT,
            "head": self.head,
            "leaked_temp_entries": self.leaked,
            "n_mutants": len(self.results),
            "n_killed": len(self.killed),
            "n_survived": len(self.survivors),
            "n_equivalent": len(self.equivalent),
            "n_unmeasured": len(self.unmeasured),
            "deselected": self.deselected,
            "carried_working_tree_files": self.carried,
            "seconds": round(self.seconds, 1),
            "results": [asdict(r) for r in self.results],
        }


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------


def apply_site(source: str, site: Site) -> str:
    """``source`` with one token replaced. The only place a mutant is constructed."""
    lines = source.splitlines(keepends=True)
    line = lines[site.line - 1]
    lines[site.line - 1] = line[: site.col_start] + site.new + line[site.col_end :]
    return "".join(lines)


def mutable_sites(source: str) -> list[Site]:
    """Every single-token mutation site in ``source`` THAT STILL PARSES, in file order.

    Token-level rather than AST-level on purpose: the mutant must be byte-identical
    to the original everywhere else, and an AST round-trip reformats the file, which
    would make the diff unreadable and would move the code fingerprint for reasons
    that have nothing to do with the mutation.

    THE PARSE FILTER IS NOT TIDINESS; IT REMOVES FAKE KILLS. The token rules are
    blind to grammar, so ``*`` is mutated to ``/`` wherever it appears - including
    the unpack in ``subprocess.run(["git", *args])``, which yields ``[..., /args]``
    and does not compile. Every test then fails on a collection error, the mutant
    reads KILLED, and the sweep credits a test that noticed nothing. That is the
    exact mirror of the false survivor this tool was written to prevent, and the
    first baseline had **three** of them (``common/runs.py`` twice,
    ``common/seeds.py`` once), all three counted as kills.

    Stated as a general rule rather than a special case for ``*args``: a mutation
    that cannot be compiled is not a mutation, because a suite cannot fail to
    notice a file the parser rejects. Costs one parse per site, about 24 s over the
    whole corpus, against a sweep measured in tens of minutes.
    """
    out: list[Site] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.start[0] != token.end[0]:
            continue
        new: str | None = None
        if token.type == tokenize.OP:
            new = OPERATOR_MUTATIONS.get(token.string)
        elif token.type == tokenize.NAME:
            new = KEYWORD_MUTATIONS.get(token.string)
        elif token.type == tokenize.NUMBER:
            new = _mutate_number(token.string)
        if new is None:
            continue
        site = Site(token.start[0], token.start[1], token.end[1], token.string, new)
        try:
            ast.parse(apply_site(source, site))
        except SyntaxError:
            continue
        out.append(site)
    return out


def _mutate_number(literal: str) -> str | None:
    """Move a numeric literal off its value: a boundary is where off-by-one lives."""
    try:
        value = float(literal)
    except ValueError:
        return None
    if value == 0:
        return "1"
    if literal.isdigit():
        return str(int(literal) + 1)
    return repr(round(value * 1.5 + 0.001, 6))


def select_site(sites: Sequence[Site], fraction: float) -> Site:
    """The site at ``fraction`` of the way through the list. Deterministic."""
    return sites[int(len(sites) * fraction) % len(sites)]


def line_occurrence(source: str, line_number: int) -> int:
    """How many EARLIER lines of ``source`` carry the same stripped text.

    The field that makes :data:`MutantKey` injective across identical lines. Zero
    for a line that is unique in its file, which is most of them.
    """
    lines = source.splitlines()
    text = lines[line_number - 1].strip()
    return sum(1 for earlier in lines[: line_number - 1] if earlier.strip() == text)


def mutant_key(
    rel: str, source: str, site: Site, *, sites: Sequence[Site] | None = None
) -> MutantKey:
    """The content-addressed identity of one mutant. See :data:`MutantKey`.

    Used by BOTH registries. They ask different questions - one exempts a mutant
    from the debt, the other pins the debt - and answering them with two identity
    schemes is how the two would come to disagree about what a mutant IS.

    ``sites`` is the module's already-enumerated site list. Passing it is worth a
    keyword because enumeration re-tokenises and re-parses the whole file: over the
    corpus, keying every mutant without it costs minutes rather than seconds, and
    :func:`corpus_keys` is a gate step.
    """
    line = source.splitlines()[site.line - 1]
    enumerated = mutable_sites(source) if sites is None else sites
    on_this_line = [s for s in enumerated if s.line == site.line]
    return (
        rel,
        line.strip(),
        line_occurrence(source, site.line),
        on_this_line.index(site),
        site.old,
        site.new,
    )


def render_key(key: MutantKey) -> str:
    """One reviewable line for one mutant. For humans; never parsed back."""
    rel, line_text, occurrence, index, old, new = key
    where = rel.split("wildfire_nowcast/")[-1]
    nth = f" (occurrence {occurrence})" if occurrence else ""
    return f"{where}  {old!r}->{new!r}  site {index} of {line_text!r}{nth}"


def equivalence_note(rel: str, source: str, site: Site) -> tuple[str, str] | None:
    """``(reason, proving test node id)`` if this mutant is declared unkillable."""
    return EQUIVALENT_MUTANTS.get(mutant_key(rel, source, site))


def equivalence_line_as_mutated(key: MutantKey) -> str:
    """The pinned line WITH its own declared mutation applied.

    Needed because the registry's integrity check would otherwise kill the mutant
    it exempts. That check asserts the pinned line is still in the file, so while
    the sweep has that very line mutated it fails, the suite goes red, and the
    mutant reads KILLED - by a test about the registry, not about behaviour. The
    first corrected sweep reported ``equivalent 0`` with a proven-equivalent
    mutant in the corpus, so the third state of the burn-down was unreachable.

    A sweep applying the DECLARED mutation is not the pinned line moving on; it is
    the sweep doing its job. Tolerating exactly this one variant is therefore the
    smallest tolerance that makes EQUIVALENT reachable, and any other edit to the
    line still breaks the key.
    """
    _rel, line_text, _occurrence, index, old, new = key
    on_the_line = mutable_sites(line_text)
    if index >= len(on_the_line):
        raise ValueError(f"site {index} does not exist on {line_text!r}")
    site = on_the_line[index]
    if (site.old, site.new) != (old, new):
        raise ValueError(
            f"site {index} of {line_text!r} is {site.old!r}->{site.new!r}, not {old!r}->{new!r}"
        )
    return apply_site(line_text, site)


def target_modules(repo: Path) -> list[str]:
    """Tracked modules under the swept packages, as repo-relative paths."""
    args = [
        "git",
        "-C",
        str(repo),
        "ls-files",
        *[f"src/wildfire_nowcast/{p}" for p in TARGET_PACKAGES],
    ]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    return sorted(line for line in out.splitlines() if line.endswith(".py"))


def corpus_keys(repo: Path, modules: Sequence[str] | None = None) -> set[MutantKey]:
    """Every mutant identity the sampler WOULD produce at this tree.

    No suite, no worktree, no subprocess but ``git ls-files``: this is the half of
    the pin that is checkable in seconds. It answers "does each pinned survivor
    still NAME a live mutant", never "does it still survive" - that needs the
    tests, and the tests are the 110 minutes.

    ``modules`` narrows the enumeration, and it is exact rather than approximate:
    a key carries its own module path, so no other module can produce it. The
    saving is real and SMALL, which is worth stating because the obvious guess is
    wrong - on an idle machine, 37.0 s of CPU for all 46 modules against 33.4 s
    for the sixteen the pin names. `eval/selftest.py` alone is 25.0 s of that:
    `mutable_sites` parses the WHOLE file once per site to drop mutants that do
    not compile, and that file has 2,161 sites. The cost is quadratic inside ONE
    module rather than spread over the corpus, so dropping thirty files off the
    list barely touches it.
    """
    out: set[MutantKey] = set()
    for rel in target_modules(repo) if modules is None else modules:
        source = (repo / rel).read_text(encoding="utf-8")
        sites = mutable_sites(source)
        if not sites:
            continue
        for fraction in SAMPLE_FRACTIONS:
            site = select_site(sites, fraction)
            out.add(mutant_key(rel, source, site, sites=sites))
    return out


def pinned_modules() -> list[str]:
    """The modules :data:`PINNED_SURVIVORS` names, in path order."""
    return sorted({key[0] for key in PINNED_SURVIVORS})


def pin_check_verdict(corpus: Collection[MutantKey]) -> tuple[int, str]:
    """``(exit code, message)`` for the STALE half of the pin. Pure.

    A pinned survivor whose identity is not in the corpus is not debt any more; it
    is a line in a registry that has stopped referring to anything, and it will sit
    there until someone spends 110 minutes to find out. This says so in seconds.

    It deliberately says NOTHING about the other direction. A survivor that appeared
    is invisible without running the tests, and a check that reported "pin OK" while
    blind to half of the question would be worse than no check: this project has
    forty decisions about gates that pass because they cannot fail.

    IT DOES NOT RUN UNDER PYTEST, AND THAT IS NOT AN OVERSIGHT. Its input is the
    text of the swept packages, and during a sweep that text is MUTATED by
    construction: a mutated token changes its own key, the check would go red, the
    suite would exit non-zero and the mutant would read KILLED - by the pin's own
    bookkeeping rather than by a test about behaviour. `tools/mutation.py` already
    carries one scar of exactly this (`equivalence_line_as_mutated`, which exists so
    the registry's integrity test does not kill the mutant it exempts). So this runs
    as its own gate step, against the real tree, where nothing is mutated.
    """
    stale = sorted(key for key in PINNED_SURVIVORS if key not in corpus)
    n_pinned, n_corpus = len(PINNED_SURVIVORS), len(corpus)
    if not stale:
        return 0, (
            f"OK: all {n_pinned} pinned survivors still name a live mutant "
            f"({n_corpus} in the corpus). This says nothing about whether they still "
            "SURVIVE - only `make mutation` measures that."
        )
    lines = [
        f"FAIL: {len(stale)} of {n_pinned} pinned survivors name no mutant in the corpus "
        f"({n_corpus} enumerated). Their line was edited, or their module was re-sampled, so "
        "the pin has stopped checking them and a sweep would report them as VANISHED:"
    ]
    lines.extend(f"  ? {render_key(key)}" for key in stale)
    return 1, "\n".join(lines)


# --------------------------------------------------------------------------
# workspace
# --------------------------------------------------------------------------


def build_workspace(repo: Path, workspace: Path, *, pristine: bool = False) -> int:
    """A git worktree at HEAD, overlaid with the working tree. Returns files carried.

    ``pristine`` carries nothing, so the sweep measures the COMMIT and the number it
    reports can be quoted against a sha. That is the mode a reported figure must use:
    three leads share this tree, and a working-tree overlay silently pulls another
    lead's half-written test file into a number attributed to this one. It happened
    while this tool was being written - six untracked test files appeared under
    ``tests/`` mid-task - which is why the flag exists rather than the convention.
    """
    if workspace.exists():
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(workspace)],
            capture_output=True,
            text=True,
            check=False,
        )
        shutil.rmtree(workspace, ignore_errors=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(workspace), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    purge_bytecode(workspace)
    if pristine:
        return 0
    changed = subprocess.run(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    untracked = [
        line
        for line in subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if line.startswith(_CARRIED_UNTRACKED)
    ]
    carried = 0
    for rel in sorted(set(changed) | set(untracked)):
        source, destination = repo / rel, workspace / rel
        if not source.exists():
            destination.unlink(missing_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        carried += 1
    purge_bytecode(workspace)
    return carried


def _child_env(workspace: Path, python: Path) -> dict[str, str]:
    """A minimal, explicit environment. The workspace's own ``src`` comes FIRST.

    TMPDIR IS LOAD-BEARING AND ITS ABSENCE MADE THE LEAK DETECTOR BLIND. This
    environment is built from nothing rather than copied, so anything not listed
    here is simply gone from the child. TMPDIR was not listed, so every child
    pytest fell back to Python's default of ``/tmp`` while the detector compared
    before and after in the PARENT's ``tempfile.gettempdir()``. Two different
    directories: the difference was taken over a location the suite never wrote
    to, and it reported an empty leak list no matter what the children left
    behind. Measured, not reasoned about - parent ``/private/tmp/probe-XXXX``,
    child ``/tmp``.

    Pinning it to the parent's value does two things. The detector now watches
    where the children actually write, and a private TMPDIR isolates the whole
    process tree instead of only its root.
    """
    return {
        "PATH": f"{python.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(workspace / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": tempfile.gettempdir(),
        "_ZO_DOCTOR": "0",
    }


def _run_pytest(workspace: Path, python: Path, extra: Sequence[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [str(python), "-m", "pytest", *_PYTEST_ARGS, *extra],
        cwd=workspace,
        env=_child_env(workspace, python),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def assert_workspace_is_self_contained(workspace: Path, python: Path) -> None:
    """Control 1: the workspace must import its OWN package, not the editable one."""
    proc = subprocess.run(
        [str(python), "-c", "import wildfire_nowcast; print(wildfire_nowcast.__file__)"],
        cwd=workspace,
        env=_child_env(workspace, python),
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = Path(proc.stdout.strip()).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise RuntimeError(
            f"the workspace imports {resolved}, which is OUTSIDE {workspace}. Every mutant "
            "would read SURVIVED because none of them would ever be loaded."
        )


def failing_tests(output: str) -> list[str]:
    """Node ids pytest reported as FAILED or ERROR, from its own summary lines."""
    out: list[str] = []
    for line in output.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            out.append(line.split(" ", 1)[1].split(" ")[0])
    return sorted(set(out))


#: How many times the baseline may find a new sandbox artifact before giving up.
#: A cap rather than a while-loop, because "keep deselecting until green" is also
#: how a genuinely broken suite gets swept under the rug one test at a time.
MAX_BASELINE_ROUNDS: Final = 10


def baseline(workspace: Path, python: Path) -> list[str]:
    """Control 3: run unmutated, deselect whatever the sandbox itself breaks.

    ITERATES, because the suite runs under ``-x`` and therefore names ONE failure
    per attempt. The single-pass version worked only by luck: this repository's
    own checkout has exactly one sandbox artifact, so one round was always enough.
    Pointed at a clone it found a second (uninstalled git hooks) and aborted the
    whole sweep, which is how the limitation surfaced.

    Every deselected node is returned and printed, never silently dropped, because
    a suite that is already red kills every mutant and means nothing.
    """
    broken: list[str] = []
    for _ in range(MAX_BASELINE_ROUNDS):
        deselect = [arg for node in broken for arg in ("--deselect", node)]
        code, output = _run_pytest(workspace, python, deselect)
        if code == 0:
            return broken
        found = [node for node in failing_tests(output) if node not in broken]
        if not found:
            raise RuntimeError(
                f"unmutated workspace exited {code} and named no NEW test. Already "
                f"deselected {broken}.\n{output[-3000:]}"
            )
        broken.extend(found)
    raise RuntimeError(
        f"unmutated workspace still red after {MAX_BASELINE_ROUNDS} rounds, having deselected "
        f"{broken}. A sweep against a red suite kills every mutant and means nothing."
    )


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def module_name(rel: str) -> str:
    """``src/wildfire_nowcast/common/states.py`` -> ``wildfire_nowcast.common.states``."""
    return rel.removeprefix("src/").removesuffix(".py").replace("/", ".")


#: Control 2, in two halves. The first reads the mutated LINE back through the
#: interpreter that is about to run the tests. The second is the one that matters
#: and was added once the hazard was demonstrated: it compares the BYTECODE the
#: loader would hand the interpreter against a fresh compile of the file on disk.
#:
#: CPython invalidates a ``.pyc`` on ``(source mtime in WHOLE SECONDS, source
#: size)``. A same-length edit made and reverted inside one second is therefore
#: invisible and the stale bytecode runs instead, so a mutant whose source is
#: verifiably present can fail to execute and read as a SURVIVOR. Reading the
#: source proves nothing about that; ``marshal.dumps`` of the loaded code object
#: against a fresh ``compile`` proves it exactly.
_PROBE = """
import importlib.util as u, marshal, pathlib, sys
spec = u.find_spec({name!r})
origin = pathlib.Path(spec.origin)
loaded = spec.loader.get_code({name!r})
fresh = compile(origin.read_text(), str(origin), "exec")
print(origin.read_text().splitlines()[{index}])
print("BYTECODE_MATCHES_SOURCE" if marshal.dumps(loaded) == marshal.dumps(fresh) else "STALE")
"""


class ProbeFailed(Exception):
    """The read-back could not be performed, so nothing about this mutant is known."""


def _read_back(workspace: Path, python: Path, rel: str, site: Site) -> tuple[str, bool]:
    """Return ``(the line the interpreter reads, whether its bytecode is that source)``.

    ``check=False`` DELIBERATELY. The first version raised, and a single probe
    failure aborted a 42-minute sweep at the 39th minute with a traceback and no
    partial result. One unmeasurable mutant is one unmeasured verdict, not the loss
    of every measurement taken beside it; the caller turns this into
    ``PROBE_FAILED``, which the gate counts as unmeasured and never as a pass.
    """
    proc = subprocess.run(
        [str(python), "-c", _PROBE.format(name=module_name(rel), index=site.line - 1)],
        cwd=workspace,
        env=_child_env(workspace, python),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = proc.stdout.rstrip("\n").splitlines()
    if proc.returncode != 0 or len(lines) < 2:
        raise ProbeFailed(
            f"exit {proc.returncode}: {(proc.stderr.strip().splitlines() or [''])[-1]}"
        )
    return lines[0], lines[-1] == "BYTECODE_MATCHES_SOURCE"


def purge_bytecode(root: Path) -> int:
    """Delete every ``__pycache__`` under ``root``. Returns how many were removed.

    ``PYTHONDONTWRITEBYTECODE`` stops a cache being WRITTEN; it does not stop one
    already on disk being READ. Both halves are needed, and neither is a substitute
    for the probe above, which is what actually reads a value.
    """
    removed = 0
    for cache in sorted(root.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    return removed


def run_one(
    workspace: Path, python: Path, rel: str, fraction: float, deselect: Sequence[str]
) -> Result:
    """Apply one mutant, run the suite, restore the file whatever happens."""
    path = workspace / rel
    original = path.read_text(encoding="utf-8")
    sites = mutable_sites(original)
    if not sites:
        return Result(rel, fraction, "NO_SITES", "", 0)
    site = select_site(sites, fraction)
    mutated = apply_site(original, site)
    expected = mutated.splitlines()[site.line - 1]
    # `line:col`, not `line`, and the column is not decoration. The printed form
    # used to be `module:line 'old'->'new'`, which does not identify a site: four
    # of the 25 survivors filed at 68ebd2f are ambiguous under it, and
    # `eval/meanfield.py:91` alone holds three `0.0`->`1` sites. `line:col` is
    # also what an editor jumps to.
    where = rel.split("wildfire_nowcast/")[-1]
    descriptor = f"{where}:{site.line}:{site.col_start} {site.old!r}->{site.new!r}"
    key = mutant_key(rel, original, site)
    started = time.monotonic()
    try:
        path.write_text(mutated, encoding="utf-8")
        purge_bytecode(workspace / "src")
        try:
            seen, bytecode_is_fresh = _read_back(workspace, python, rel, site)
        except ProbeFailed as exc:
            return Result(
                rel,
                fraction,
                "PROBE_FAILED",
                descriptor,
                len(sites),
                key=key,
                detail=f"the read-back could not run, so this mutant was not measured: {exc}",
            )
        if seen != expected:
            return Result(
                rel,
                fraction,
                "NOT_APPLIED",
                descriptor,
                len(sites),
                key=key,
                detail=f"the interpreter reads {seen!r}, not the mutant",
            )
        if not bytecode_is_fresh:
            return Result(
                rel,
                fraction,
                "STALE_BYTECODE",
                descriptor,
                len(sites),
                key=key,
                detail=(
                    "the source is mutated and the loader would still run the OLD bytecode. "
                    "This is a REFUSAL to measure, never a survivor: a mutant that does not "
                    "execute cannot be said to have survived anything."
                ),
            )
        code, output = _run_pytest(workspace, python, deselect)
    finally:
        path.write_text(original, encoding="utf-8")
        purge_bytecode(workspace / "src")
    verdict = "SURVIVED" if code == 0 else "KILLED"
    if verdict == "SURVIVED":
        equivalent = equivalence_note(rel, original, site)
        if equivalent is not None:
            return Result(
                rel,
                fraction,
                "EQUIVALENT",
                descriptor,
                len(sites),
                key=key,
                exit_code=code,
                seconds=round(time.monotonic() - started, 1),
                detail=equivalent[0],
            )
    # `failing_tests` and NOT a grep for `FAILED `, which is what this line used to
    # be. Two of the 58 kills in the I12 sweep were recorded with no attributable
    # node, and neither was unattributable: `common/playthrough.py:240` is killed by
    # a COLLECTION ERROR in `tests/test_playthrough_asymmetric_gate.py`, and
    # `common/null_check/forecasters.py:82` by a fixture error in
    # `tests/test_null_check.py::test_best_member_iou_is_flagged_in_the_growth_band`
    # (a C1.3 phase guard raising `truth covers 3 steps but samples cover 40`).
    # Both print an `ERROR ` line, the deselect path already read those, and this
    # one threw the information away. Reporting it changes nothing about the
    # verdict and everything about whether the verdict can be checked.
    nodes = failing_tests(output)
    tail = ["; ".join(nodes[:3])] if nodes else []
    return Result(
        rel,
        fraction,
        verdict,
        descriptor,
        len(sites),
        key=key,
        exit_code=code,
        seconds=round(time.monotonic() - started, 1),
        detail=tail[0] if tail else "",
    )


def sweep(
    repo: Path,
    python: Path,
    root: Path,
    *,
    workers: int,
    only: str = "",
    pristine: bool = False,
    max_mutants: int = 0,
) -> Sweep:
    """Build ``workers`` workspaces and run every mutant exactly once.

    ``max_mutants`` truncates the job list, deterministically and from the front.
    It exists so the gate can be MEASURED cheaply before it is run in full: three
    mutants take minutes and reveal a per-run temp leak just as well as 117 do,
    and the leak is what multiplies.
    """
    modules = [m for m in target_modules(repo) if only in m]
    jobs = [(m, f) for m in modules for f in SAMPLE_FRACTIONS]
    if max_mutants > 0:
        jobs = jobs[:max_mutants]
    spaces = [root / f"ws{i}" for i in range(workers)]
    out = Sweep()
    started = time.monotonic()

    cleanup_from = _remove_worktrees  # named so the finally below cannot drift from it
    before = temp_entries()

    try:
        for space in spaces:
            out.carried = build_workspace(repo, space, pristine=pristine)
            assert_workspace_is_self_contained(space, python)
        return _sweep_inner(repo, python, spaces, out, jobs, workers, started)
    finally:
        out.seconds = time.monotonic() - started
        cleanup_from(repo, spaces)
        shutil.rmtree(root, ignore_errors=True)
        # Measured AFTER the cleanup, so what is listed is what genuinely survived
        # this sweep rather than what it was still using.
        out.leaked = sorted(temp_entries() - before)


#: Names the detector must NOT call a leak. Exactly one entry, and it is narrow on
#: purpose: ``pytest-of-<user>`` is pytest's own ``tmp_path`` base, which pytest
#: creates once and garbage-collects down to the last three runs. It is BOUNDED and
#: self-managing, which is the opposite of the per-call ``mkdtemp`` this detector
#: exists to catch. Without it the unplanted control fires on every run, and a
#: detector that fires on everything is as useless as one that fires on nothing.
IGNORED_TEMP_PREFIXES: Final = ("pytest-of-",)


def temp_entries() -> set[str]:
    """Every name directly under the temp directory, for before/after comparison.

    Excludes :data:`IGNORED_TEMP_PREFIXES`. That exclusion was added because the
    unplanted control FIRED: a clean 3-mutant sweep reported one leaked entry,
    ``pytest-of-thondascully``. Reporting it would have made the gate exit 3 on
    every green run, and it would have made the planted capture unfalsifiable,
    since residue would be present either way.
    """
    root = Path(tempfile.gettempdir())
    try:
        return {
            entry.name
            for entry in root.iterdir()
            if not entry.name.startswith(IGNORED_TEMP_PREFIXES)
        }
    except OSError:  # pragma: no cover - unreadable temp dir
        # An unreadable temp directory makes the leak detector return the SAME
        # answer as a clean one. ADR-092 is the instance where a leak detector
        # was proven blind and green twice; it may not go blind silently again.
        logger.warning("temp directory %s could not be listed; leak detection is BLIND", root)
        return set()


def _remove_worktrees(repo: Path, spaces: Sequence[Path]) -> None:
    """Every worktree this sweep made, on every exit path."""
    for space in spaces:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(space)],
            capture_output=True,
            text=True,
            check=False,
        )


def _sweep_inner(
    repo: Path,
    python: Path,
    spaces: list[Path],
    out: Sweep,
    jobs: list[tuple[str, float]],
    workers: int,
    started: float,
) -> Sweep:
    """The measurement itself. Split out so the caller's finally covers SETUP too.

    The first version of the cleanup wrapped only this part, and a failure in the
    build loop above - which is where the self-containment control raises - left a
    worktree behind anyway. Proved by forcing that exact failure and counting.
    """
    out.head = subprocess.run(
        ["git", "-C", str(spaces[0]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out.deselected = baseline(spaces[0], python)
    deselect = [arg for node in out.deselected for arg in ("--deselect", node)]

    def confirm(space: Path) -> None:
        """Every workspace is proven green before it judges anything, not just ws0."""
        code, output = _run_pytest(space, python, deselect)
        if code != 0:
            raise RuntimeError(f"{space} is not green before any mutant:\n{output[-2000:]}")

    def work(index: int) -> list[Result]:
        return [
            run_one(spaces[index], python, module, fraction, deselect)
            for module, fraction in jobs[index::workers]
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(confirm, spaces[1:]))
        for batch in pool.map(work, range(workers)):
            out.results.extend(batch)
    out.results.sort(key=lambda r: (r.module, r.fraction))
    out.seconds = time.monotonic() - started
    return out


def format_pin(keys: Collection[MutantKey]) -> str:
    """``keys`` as the literal that would replace :data:`PINNED_SURVIVORS`.

    Printed on any diff so that accepting a change is a paste and a review rather
    than a transcription. The 25 descriptors this pin was built from had to be
    re-derived from `(module, fraction)` because the printed form was ambiguous;
    nobody should have to do that twice.
    """
    lines = ["PINNED_SURVIVORS: Final[frozenset[MutantKey]] = frozenset(", "    {"]
    for key in sorted(keys):
        lines.append("        (")
        lines.extend(f"            {_as_source(field)}," for field in key)
        lines.append("        ),")
    lines.extend(["    }", ")"])
    return "\n".join(lines)


def _as_source(field: str | int) -> str:
    """``field`` as `ruff format` would write it, so a paste is not a lint failure.

    Double quotes, except where the value contains one and no single quote - which
    is ruff's own rule, and matters here because half of these lines are dict keys
    out of `eval/`.

    THE BACKSLASH IS ESCAPED ON BOTH BRANCHES, and it was not on the first version.
    A pinned line holding a `"` AND a `\\` - a regex literal inside a formatted
    string, which `eval/` has plenty of room for - came back from `literal_eval`
    one backslash short, so the emitted pin would have been a DIFFERENT set from
    the measured one while looking exactly right. Caught by round-tripping every
    field rather than by reading the branch; `tests/test_hygiene.py` keeps that
    round-trip, with adversarial probes the corpus does not happen to contain
    today.
    """
    if isinstance(field, int):
        return repr(field)
    escaped = field.replace("\\", "\\\\")
    if '"' in field and "'" not in field:
        return "'" + escaped + "'"
    return '"' + escaped.replace('"', '\\"') + '"'


def exit_code(verdict_code: int, leaked: int) -> int:
    """The sweep's exit status, as a pure function of the two things that decide it.

    A LEAK WINS THE EXIT CODE AND NO LONGER SUPPRESSES THE MEASUREMENT. One 4.7 MB
    temp directory per suite run is nothing; this gate runs the suite once per
    mutant, and the same directory became 70 GB and 15,462 entries in a day. So the
    residue is a failure of the run.

    It used to be spelled as an early ``return 3`` ahead of the pin verdict, which
    made the sweep at 68ebd2f red for a leak and SILENT about its own subject - it
    never printed the budget line at all. Split out here for the same reason
    :func:`survivor_verdict` is: a decision reachable only through 110 minutes is a
    decision nobody has checked the direction of.
    """
    return 3 if leaked else verdict_code


def survivor_verdict(
    measured: Collection[MutantKey],
    enumerated: Collection[MutantKey],
    unmeasured: int,
    equivalent: int,
) -> tuple[int, str]:
    """``(exit code, message)``. Pure, so the gate's decision is testable in a millisecond.

    Extracted from ``main`` deliberately: a 110-minute sweep is not a way to find
    out whether the comparison is the right way round, and a gate whose verdict
    logic is only reachable through the slow path is a gate nobody checks.

    BOTH DIRECTIONS FAIL, and the disappeared direction is the one that needed
    arguing, because a survivor disappearing is usually GOOD news. It fails anyway,
    for three reasons:

    1. A pin left standing after its mutant died is an over-estimate of the debt,
       and an over-estimate forgives the next regression - the same argument the
       count pin already made in ITS falling direction, which was ratified.
    2. "Disappeared" is TWO events and only one of them is good news. If the key is
       still in the enumerated corpus, a test now kills the mutant: real progress.
       If it is NOT, no mutant with that identity exists any more - the line was
       edited or the module re-sampled - and NOTHING was measured about it in
       either direction. Silence there is a pin quietly ceasing to check anything,
       which is how this file's previous pin drifted four survivors unseen.
    3. The report has nowhere loud to land. This sweep is 110 minutes and runs on a
       SCHEDULE, so its output is a log a human reads on purpose. A non-failing
       "note" in a scheduled log is exactly the thing that goes unread; a red
       target with a paste-ready replacement pin is not.

    THE CASE FOR A REPORT INSTEAD, because it is a real one and it lost on a
    detail rather than on principle: failing a gate for an IMPROVEMENT puts a red
    target on the person who wrote the killing test, and a gate that punishes good
    work is a gate people route around. What answers it is the size of the remedy,
    not the direction of the news - the message prints a paste-ready
    :func:`format_pin` block, so accepting the change is one paste and a review. A
    red that costs one line is a different object from a red that costs an
    investigation, and only the second one gets routed around.

    The remedy is stated in the message: delete the dead entries in the same commit
    as the test that killed them.
    """
    if unmeasured:
        return 2, (
            f"FAIL: {unmeasured} mutant(s) never executed, so this sweep did not measure what "
            "it claims. A refusal to measure is not a pass and not a survivor."
        )
    measured, enumerated = set(measured), set(enumerated)
    appeared = sorted(measured - PINNED_SURVIVORS)
    disappeared = sorted(PINNED_SURVIVORS - measured)
    killed = [key for key in disappeared if key in enumerated]
    vanished = [key for key in disappeared if key not in enumerated]
    readout = (
        f"{len(measured)} survivors measured against {len(PINNED_SURVIVORS)} pinned "
        f"({equivalent} proved unkillable). The COUNT is a readout; the SET decides."
    )
    if not appeared and not disappeared:
        return 0, f"OK: the survivor set is exactly PINNED_SURVIVORS. {readout}"

    lines = [
        f"FAIL: the survivor SET has moved - {len(appeared)} appeared, {len(disappeared)} "
        f"disappeared ({len(killed)} killed, {len(vanished)} vanished). {readout}"
    ]
    if appeared:
        lines.append(
            "APPEARED - a mutant survives that was not pinned. Either new debt, or a module "
            "that grew or was re-sampled. Name the cause; do not extend the pin to make it "
            "quiet:"
        )
        lines.extend(f"  + {render_key(key)}" for key in appeared)
    if killed:
        lines.append(
            "DISAPPEARED, KILLED - the mutant is still in the corpus and a test now kills it. "
            "That is the good direction, and it still fails: delete these entries in the SAME "
            "commit as the test that killed them, or the pin becomes an over-estimate that "
            "forgives the next regression:"
        )
        lines.extend(f"  - {render_key(key)}" for key in killed)
    if vanished:
        lines.append(
            "DISAPPEARED, VANISHED - no mutant with this identity exists any more, so nothing "
            "was measured about it in either direction. The line was edited, or its module was "
            "re-sampled. This is the entry that must never be silent:"
        )
        lines.extend(f"  ? {render_key(key)}" for key in vanished)
    lines.append("The measured set, as a paste-ready pin:")
    lines.append(format_pin(measured))
    return 1, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", default="", help="substring filter over module paths")
    parser.add_argument(
        "--pristine",
        action="store_true",
        help="measure HEAD exactly; carry no working-tree file. Use this for a quoted number.",
    )
    parser.add_argument(
        "--workspace-root",
        default="",
        help="where worktrees are built (default: a temp dir, NEVER inside the repo)",
    )
    parser.add_argument("--json", default="", help="write the full result set here")
    parser.add_argument(
        "--max-mutants",
        type=int,
        default=0,
        help=(
            "run only the first N mutants. For MEASURING the gate cheaply - a per-run "
            "temp leak shows up in three mutants and is then multiplied by all of them."
        ),
    )
    parser.add_argument(
        "--check-pin",
        action="store_true",
        help=(
            "check PINNED_SURVIVORS against the corpus and exit. Seconds, no suite, no "
            "worktree: it can only report entries that name no mutant any more"
        ),
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help="measure and report without adjudicating PINNED_SURVIVORS (how a new pin is taken)",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    if args.check_pin:
        code, message = pin_check_verdict(corpus_keys(repo, pinned_modules()))
        print(message)
        return code

    # OUTSIDE the repository on purpose: a worktree under `repo/` would be visible
    # to `git ls-files --others`, to ruff and to the hygiene scan, and a sweep that
    # changes what the hygiene suite is looking at is measuring itself.
    # ADR-098: PER INVOCATION, never a fixed name under a shared temp directory.
    # The default was `gettempdir()/wildfire-nowcast-mutation`, so two concurrent
    # sweeps shared `ws0..ws3` - and `build_workspace` removes an existing
    # workspace with `worktree remove --force` plus `rmtree`. That is not a
    # cosmetic collision: it DELETES A RUNNING SWEEP'S WORKTREE, which is
    # precisely what happened twice on 2026-08-22 and was attributed to a manual
    # cleanup by prefix. `mkdtemp` gives each sweep its own root, and the
    # `shutil.rmtree(root)` in `sweep`'s finally then removes only its own.
    root = (
        Path(args.workspace_root).resolve()
        if args.workspace_root
        else Path(tempfile.mkdtemp(prefix="wildfire-nowcast-mutation-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    result = sweep(
        repo,
        Path(sys.executable),
        root,
        workers=args.workers,
        only=args.only,
        pristine=args.pristine,
        max_mutants=args.max_mutants,
    )

    n_survived = len(result.survivors)
    # The sha FIRST, because it is the only line that says what the rest is about.
    print(f"swept {result.head[:7]} ({'pristine' if args.pristine else 'working tree'})")
    print(
        f"mutants {len(result.results)}  killed {len(result.killed)}  survived {n_survived}  "
        f"equivalent {len(result.equivalent)}  unmeasured {len(result.unmeasured)}"
    )
    print(f"carried {result.carried} working-tree file(s); deselected {result.deselected}")
    for row in result.survivors:
        print(f"  SURVIVED    {row.descriptor}")
    for row in result.equivalent:
        print(f"  EQUIVALENT  {row.descriptor}")
    for row in result.unmeasured:
        print(f"  {row.verdict}  {row.descriptor}  {row.detail}")
    print(f"{result.seconds / 60:.1f} min")
    if result.leaked:
        print(
            f"LEAKED {len(result.leaked)} temp entr(y/ies), first few: {result.leaked[:5]}. "
            "A gate that runs the suite once per mutant multiplies this by the mutant "
            "count, so it is reported as a failure and not as a footnote."
        )
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    # THE PIN VERDICT IS PRINTED BEFORE ANY EXIT CODE IS CHOSEN. At 68ebd2f the
    # leak check returned 3 first and `make mutation` never printed the budget line
    # at all, so a sweep that was red for one reason said NOTHING about the other -
    # and the reason it was silent about is the one it exists for. The leak still
    # WINS the exit code; it no longer suppresses the measurement.
    partial = args.only or args.max_mutants
    if partial:
        print(
            f"PARTIAL SWEEP ({'--only ' + args.only if args.only else ''}"
            f"{' ' if args.only and args.max_mutants else ''}"
            f"{'--max-mutants ' + str(args.max_mutants) if args.max_mutants else ''}): "
            "PINNED_SURVIVORS is NOT adjudicated. Every pinned survivor outside this "
            "selection would read as DISAPPEARED, so a truncated run could ask for the "
            "pin to be emptied. Re-run whole to move the pin."
        )
    elif args.no_pin:
        print(
            "REPORTING ONLY (--no-pin): PINNED_SURVIVORS is NOT adjudicated. The measured "
            "set, as a paste-ready pin:\n" + format_pin(result.survivor_keys)
        )
    else:
        code, message = survivor_verdict(
            result.survivor_keys,
            result.enumerated_keys,
            len(result.unmeasured),
            len(result.equivalent),
        )
        print(message)
        return exit_code(code, len(result.leaked))
    return exit_code(0, len(result.leaked))


if __name__ == "__main__":
    raise SystemExit(main())
