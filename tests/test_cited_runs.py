"""The ``runs/`` citation class: enumerated by a walk, not by a typed pattern.

The class was first enumerated with ``runs/[A-Za-z0-9_]*\\.json`` and came back
as twelve. It was twenty-one then, twenty-four after M19's three sweep artifacts,
and is twenty-five now, the last being ``runs/_m24_ladder.py``. The two misses
were STRUCTURAL rather than unlucky: the character class excludes ``/`` and
``-``, so no subdirectory can ever match, and the literal suffix excludes every
extension but one. The six files that second miss hides are the analysis scripts
that produced published numbers; five were there at the enumeration and the sixth
arrived when a test began deriving its protocol from one of them.

These tests hold the replacement to a higher bar than "it found more": they
measure both patterns on the SAME corpus, they demonstrate the superseded
pattern's two blind spots on strings small enough to read, and they plant a
citation in a throwaway repository to prove the walk reports one it has never
seen before.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cited_runs as C  # noqa: E402
from wildfire_nowcast.common.paths import repo_root  # noqa: E402

#: What the tree is expected to hold. Both halves are pinned: a citation that
#: stops being tracked fails, and a NEW citation that nobody accounted for fails.
#:
#: 17 -> 20 when M19's three sweep artifacts were tracked, so that the finding
#: that the documented collapse ablation fires 97 times in 200 can be CHECKED by
#: a clone rather than only re-run. Both numbers here were OBSERVED FAILING
#: FIRST (``assert 20 == 17`` and, below, ``assert 15 == 12``); neither was
#: derived from the change and then confirmed by the change.
#:
#: [I24] 20 -> 21: ``runs/_m24_ladder.py``, newly cited by
#: ``tests/test_degrade_ladder_severity.py``. The citation is LOAD-BEARING and
#: was not removed to keep the count still: that file re-declares
#: ``SHAPE_LEVELS`` and ``LADDER_SHIFT_LEVELS`` from the ladder script, which is
#: their ONLY definition anywhere in the tree, and it names that script as the
#: entry point whose ``evaluate()`` call it reproduces. Strip the citation and a
#: copied rung set loses the only link to its source.
#: HOW IT REACHED ``main`` IS THE FINDING, NOT THE COUNT: ``_source_file_tokens``
#: iterates ``C.tracked_files(base)``, so THE GIT INDEX IS AN INPUT TO THIS
#: SUITE. A green run taken before ``git add`` measured a different corpus than
#: the one that was pushed. Run this file AFTER staging, or it is answering
#: about a tree nobody has.
#: Held to this pin's own standard: all three numbers were OBSERVED FAILING
#: FIRST, one at a time, each after the one above it was bumped
#: (``assert 21 == 20``, then ``assert 10 == 9``, then ``assert 6 == 5``), and
#: none was derived from the change and then confirmed by it. ``superseded``
#: did NOT move and was not touched: it stays at 15 because the new token ends
#: ``.py``, which is the blind spot this file exists to demonstrate - measured
#: here, not assumed, since ``assert len(superseded) == 15`` passed unedited
#: through all three observations.
EXPECTED_TRACKED = 21
EXPECTED_EXEMPT = 4
EXPECTED_CLASS = EXPECTED_TRACKED + EXPECTED_EXEMPT

#: The internal coordination role, spelled in halves for the same reason
#: ``tests/test_hygiene.py`` does it: this file is scanned by that check too, and
#: a scan that cannot see one file is the failure this repository keeps paying
#: for.
_ROLE = re.compile("orchestr" + "ator", re.IGNORECASE)

#: Specimen tokens, spelled in HALVES for the same reason and with the same
#: consequence. This file is read by the very scan it tests: written literally,
#: three invented paths would BE citations of files that do not exist, and this
#: module would fail its own check. The tempting fix - exempt this path - would
#: make the one file guaranteed to contain unresolvable citations the one file
#: that can never report one. That is the shape of a check that cannot fail.
_RUNS = "ru" + "ns/"
_SPECIMEN_DEEP = _RUNS + "deep/er/still-here_v2.json"
_SPECIMEN_SCRIPT = _RUNS + "tool.py"
_SPECIMEN_NESTED = _RUNS + "kernel-nll_only-20260808-044220/model.json"


def _enum() -> C.Enumeration:
    return C.enumerate_citations(repo_root())


def _gzipped_tokens_cited_by_the_tracked_tree() -> set[str]:
    """Real ``.gz`` citations, harvested from the tree rather than written here.

    [I31] These are the adversary that was not consulted when the walk was
    built. There were 34 gzipped records under ``runs/`` on the day the defect
    was found and not one of them was checked, because ``.gz`` was absent from
    :data:`cited_runs.FILE_SUFFIXES` and a token with no recognised extension is
    classified ``dir_or_prefix`` and never enforced.

    Harvested, not spelled, for two reasons. Spelling one would make this file
    cite an untracked artifact, which is the offence it exists to detect. And a
    specimen invented to match the fix would have passed the BROKEN version too:
    the only thing that told the two apart was a real citation.

    Artifact citers are INCLUDED here, unlike everywhere else in this file. The
    tracked citers of gzipped records are themselves under ``runs/``, so the
    usual skip would empty this set and the control would silently go blind.
    """
    base = repo_root()
    out: set[str] = set()
    for rel in C.tracked_files(base):
        path = base / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        out |= {tok for tok in C.tokens_in(text) if tok.endswith(".gz")}
    return out


#: The two failure modes of the superseded pattern, named so the table below
#: cannot quietly merge them. They are NOT distinguishable by ``fullmatch``,
#: which returns ``None`` for both; only what the pattern EMITS tells them apart.
BLIND = "BLIND"
PHANTOM = "PHANTOM"

#: Is the leftover rule currently idle, i.e. is ``superseded - replacement``
#: empty in THIS tree? It is, today, and that is a fact about the corpus rather
#: than about the rule. It stops being true the moment a gzipped citation is
#: staged from tracked source, which is a pending change on another lead's desk.
#:
#: THE FLAG EXISTS SO THE PROSE DERIVES FROM IT RATHER THAN RESTATING IT. A
#: number or a claim written into a docstring beside the thing it describes will
#: drift from it - this repository has paid for that three times in one file.
#: When it goes live: flip this to ``False``, and the note it governs flips with
#: it. That is the entire remedy; nothing else in the invariant changes, which is
#: the point of having built the rule before it was needed.
LEFTOVER_RULE_IS_IDLE = True

#: The phantom row is drawn from the tree, and which record it is does not
#: matter - only that it is one somebody actually cites. If the harvest is ever
#: empty the table would silently lose its fourth case, so the emptiness is
#: asserted separately rather than defaulted away.
_HARVESTED_GZ = sorted(_gzipped_tokens_cited_by_the_tracked_tree())


# --------------------------------------------------------------------------
# THE CLASS ITSELF
# --------------------------------------------------------------------------


def test_every_cited_runs_file_is_tracked_or_declared() -> None:
    """A path in published source that a public reader cannot open is a defect."""
    enum = _enum()
    assert enum.problems == [], "\n".join(enum.problems)


def test_the_class_is_the_size_it_is_declared_to_be() -> None:
    enum = _enum()
    tracked = enum.of_kind("tracked")
    exempt = enum.of_kind("exempt")
    assert len(tracked) == EXPECTED_TRACKED, [c.token for c in tracked]
    assert len(exempt) == EXPECTED_EXEMPT, [c.token for c in exempt]
    assert len(tracked) + len(exempt) == EXPECTED_CLASS


def test_the_analysis_scripts_are_in_the_tree() -> None:
    """The half the superseded pattern could not see, named individually.

    The first five are the enumeration's own discovery set. ``_m24_ladder.py``
    [I24] is the sixth and is named here for the same reason as the others: a
    count can be raised, whereas a name that stops resolving says which file
    left. Naming it is the direction of this check that a bumped pin does not
    cover.
    """
    tracked = {c.token for c in _enum().of_kind("tracked")}
    for name in (
        "runs/_m9_response.py",
        "runs/_m9_scaling.py",
        "runs/_m10_bitidentity.py",
        "runs/_s1_bitidentity.py",
        "runs/_s1_score.py",
        "runs/_m24_ladder.py",
    ):
        assert name in tracked, f"{name} is cited by tracked source and is not tracked"


# --------------------------------------------------------------------------
# THE CONTROL: BOTH PATTERNS, ONE CORPUS
# --------------------------------------------------------------------------


def _source_file_tokens(pattern: re.Pattern[str]) -> set[str]:
    """File-shaped ``runs/`` tokens cited by tracked NON-artifact files."""
    base = repo_root()
    out: set[str] = set()
    for rel in C.tracked_files(base):
        if C.is_evidence(rel):
            # [I33] Was `rel.startswith("runs/")` - a FOURTH copy of the rule,
            # in the very file that measures it. It kept the old prefix answer
            # while the scanners moved to `tools.evidence`, so this helper read
            # newly-tracked EVIDENCE as source and reported 17 where the
            # scanners read 15. A test carrying its own copy of the rule it
            # checks does not verify the rule, it shadows it.
            continue
        path = base / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        out |= {t for t in C.tokens_in(text, pattern) if t.endswith(C.FILE_SUFFIXES)}
    return out


def _resolves(token: str) -> bool:
    """Does a public reader get a file? Git answers, exactly as the module does.

    Disk is deliberately not consulted. A leftover that exists only on the
    evidence machine must count as UNRESOLVED here, or this suite would reach a
    different verdict in a clone than it reaches on the machine that wrote it -
    the precise failure ``cited_runs`` was built to avoid.
    """
    return token in set(C.tracked_files(repo_root())) or token in C.EXEMPT


def classify_leftovers(
    superseded: set[str],
    replacement: set[str],
    resolves: Callable[[str], bool],
) -> dict[str, set[str]]:
    """Split ``superseded - replacement`` into phantoms and real misses.

    A pure function, taking its corpus and its resolver as arguments, because
    the interesting case is EMPTY in this tree right now (see the vacuity note
    on the invariant below) and a rule that can only be exercised by the tree it
    guards is a rule nobody has watched work.

    ``phantom``    the leftover names nothing. The superseded pattern INVENTED
                   it by truncating a longer citation; no reader was ever sent
                   anywhere. Tolerated, and pinned by shape below.
    ``real_miss``  the leftover names something a reader can open, and the
                   replacement failed to find it. That is a regression in the
                   walk and there is no reading of it that is acceptable.
    """
    leftover = superseded - replacement
    return {
        "phantom": {t for t in leftover if not resolves(t)},
        "real_miss": {t for t in leftover if resolves(t)},
    }


def test_the_replacement_finds_the_whole_class_where_the_superseded_pattern_found_twelve() -> None:
    """Both measured here, now, on one corpus. Neither number is remembered."""
    superseded = _source_file_tokens(C.SUPERSEDED_PATTERN)
    replacement = _source_file_tokens(C.RUNS_TOKEN)

    # 12 -> 15: the three M19 artifacts are ``runs/<word>.json``, which is the one
    # shape the superseded pattern could always see. The number that carries the
    # finding is ``missed`` below; it was UNCHANGED at 9 through that move and
    # went 9 -> 10 at I24, when a ``.py`` citation - invisible to the superseded
    # pattern by construction - entered the class. ``superseded`` is unmoved at
    # 15 for exactly that reason, and this line passed unedited while the two
    # below were each observed failing.
    #
    # [I31] IF THIS FAILS AT 16 AND THE EXTRA TOKEN ENDS ``.json``: a gzipped
    # citation has just been staged from tracked source, and the extra token is
    # the PHANTOM the superseded pattern makes of it. That is expected, not a
    # defect; the pair moves to 16 and 26 together and the phantom must show up
    # in ``leftovers`` below. If only ONE of the two moves, stop - that is the
    # blind spot reopening.
    assert len(superseded) == 15, sorted(superseded)
    assert len(replacement) == EXPECTED_CLASS, sorted(replacement)

    missed = replacement - superseded
    assert len(missed) == 10, sorted(missed)
    # 5 -> 6 at I24: ``runs/_m24_ladder.py``. The ``/results.json`` half is
    # untouched at 4, so the two blind spots stay separately countable.
    assert sum(1 for t in missed if t.endswith(".py")) == 6, sorted(missed)
    assert sum(1 for t in missed if "/results.json" in t) == 4, sorted(missed)


def test_no_citation_a_reader_can_open_is_lost_by_the_replacement() -> None:
    """The superset claim, restated at I31 so that it ADDS rather than relaxes.

    IT USED TO READ ``superseded < replacement``, and that bar was WRONG in a
    way no count could show. The superseded pattern does not only MISS shapes;
    on a gzipped record it truncates and emits ``<stem>.json``, a name that has
    never existed. A phantom is not evidence of a miss, so a bare subset test
    goes red on a citation that is perfectly well resolved - and the obvious
    repair, deleting the clause, would have dropped the only check that says
    the new walk lost nothing.

    Three clauses replace the one, and every citation in ``superseded`` is
    covered by exactly one of them:

    1. anything that RESOLVES is found by the replacement. This is the original
       claim on the only tokens where it means something;
    2. every leftover resolves to NOTHING. If a leftover ever does resolve,
       ``real_miss`` is non-empty and this goes red - that is the regression
       the old clause was really there to catch;
    3. every leftover is a strict PREFIX of a token the replacement did find,
       i.e. it is demonstrably the truncation of a real citation rather than an
       unexplained name. Clause 3 is what keeps clause 2 from being a licence:
       without it, any unexplained leftover could hide behind "it resolves to
       nothing".

    Clause 3 only holds BECAUSE ``.gz`` was added to ``FILE_SUFFIXES``. Drop the
    suffix and the gzipped original falls out of ``replacement``, leaving the
    phantom with nothing to be a prefix of. The two halves of ADR-181 are
    coupled here, deliberately, so neither can be reverted quietly.
    """
    superseded = _source_file_tokens(C.SUPERSEDED_PATTERN)
    replacement = _source_file_tokens(C.RUNS_TOKEN)
    split = classify_leftovers(superseded, replacement, _resolves)

    assert {t for t in superseded if _resolves(t)} <= replacement, sorted(
        {t for t in superseded if _resolves(t)} - replacement
    )
    assert split["real_miss"] == set(), (
        "a citation the superseded pattern found, which a reader CAN open, is "
        f"missing from the replacement: {sorted(split['real_miss'])}"
    )
    for phantom in sorted(split["phantom"]):
        assert any(real.startswith(phantom) and real != phantom for real in replacement), (
            f"{phantom} resolves to nothing AND is not the truncation of any citation "
            "the replacement found. It is unexplained, which is not the same as harmless."
        )


def test_the_leftover_rule_is_exercised_even_though_this_tree_leaves_it_idle() -> None:
    """The anti-vacuity control for the test above, and it is needed TODAY.

    While :data:`LEFTOVER_RULE_IS_IDLE` holds, ``phantom`` and ``real_miss`` are
    BOTH empty in this tree, so two of the invariant's three clauses loop zero
    times and cannot fail. That is asserted here rather than left for someone to
    infer from a green run, and the flag is asserted against the corpus in both
    directions: idleness that has quietly ended is a stale claim, and a flag
    left flipped after the corpus went quiet again is equally stale.

    The rule is then driven on REAL strings rather than the live sets, so it is
    exercised whichever way the flag points. Both branches are shown taking
    opposite verdicts, which is what makes a green invariant above worth
    reading at all.
    """
    superseded = _source_file_tokens(C.SUPERSEDED_PATTERN)
    replacement = _source_file_tokens(C.RUNS_TOKEN)
    live = classify_leftovers(superseded, replacement, _resolves)
    assert (live == {"phantom": set(), "real_miss": set()}) is LEFTOVER_RULE_IS_IDLE, (
        "the idleness flag no longer describes this tree. Flip "
        "LEFTOVER_RULE_IS_IDLE and rewrite the sentence it governs; do NOT delete "
        f"this test and do not delete the flag. observed: {live}"
    )

    # A REAL gzipped record, cited by tracked source, truncated by the real
    # superseded pattern. Nothing here is invented: the token is harvested and
    # the phantom is whatever the frozen pattern actually emits for it.
    gz = _gzipped_tokens_cited_by_the_tracked_tree()
    assert gz, "no gzipped citation left in the tracked tree; this control has gone blind"
    real_gz = sorted(gz)[0]
    phantom = sorted(C.tokens_in(real_gz, C.SUPERSEDED_PATTERN))[0]
    assert phantom != real_gz and real_gz.startswith(phantom)

    split = classify_leftovers({phantom}, {real_gz}, _resolves)
    assert split["phantom"] == {phantom}, split
    assert split["real_miss"] == set(), split

    # The positive control: the same machinery on a token that DOES resolve
    # must land in the other bucket, or the split above proves nothing.
    resolving = sorted(t for t in replacement if _resolves(t))[0]
    control = classify_leftovers({resolving}, set(), _resolves)
    assert control["real_miss"] == {resolving}, control
    assert control["phantom"] == set(), control


@pytest.mark.parametrize(
    ("token", "mode", "why"),
    [
        (
            "runs/baselines-20260808-095003/results.json",
            BLIND,
            "a subdirectory: `/` is not in the class",
        ),
        ("runs/_s1_score.py", BLIND, "a non-json extension: the suffix is a literal"),
        (_SPECIMEN_NESTED, BLIND, "a hyphen AND a subdirectory"),
        (
            _HARVESTED_GZ[0] if _HARVESTED_GZ else "",
            PHANTOM,
            "a gzipped record: the literal `.json` matches INSIDE the name and stops",
        ),
    ],
)
def test_the_superseded_pattern_fails_in_two_distinguishable_ways(
    token: str, mode: str, why: str
) -> None:
    """Four strings, two failure modes, and the second is the one that lies.

    [I31] This table used to have three rows and one mode. Adding a gzipped
    fourth row on the same shape would have proved nothing, because
    ``fullmatch`` is ``None`` for it exactly as it is for the other three - the
    row would have passed while demonstrating the wrong thing.

    The modes differ in what the pattern EMITS, which is what a caller actually
    consumes:

    ``BLIND``     it emits nothing. Blindness OMITS. The citation is absent from
                  the old reading, and absence is at least honest.
    ``PHANTOM``   it emits a TRUNCATED name that has never existed. A phantom
                  ASSERTS, and because the truncation still ends ``.json`` it is
                  indistinguishable from a true finding by anything downstream -
                  including, until I31, the superset invariant in this file.
    """
    assert token, "the harvested phantom specimen is missing; see the assertion below"
    assert not C.SUPERSEDED_PATTERN.fullmatch(token), why
    assert C.RUNS_TOKEN.fullmatch(token), f"the replacement must match: {token}"

    emitted = C.tokens_in(token, C.SUPERSEDED_PATTERN)
    if mode == BLIND:
        assert emitted == set(), f"expected blindness, got an emission: {emitted}"
    else:
        assert emitted, "expected a phantom emission, got blindness"
        assert token not in emitted, "a phantom must differ from the token it came from"
        for ghost in emitted:
            assert token.startswith(ghost), (ghost, token)
            assert not _resolves(ghost), f"{ghost} was supposed to name nothing"


def test_the_phantom_row_is_a_real_citation_and_not_an_invention() -> None:
    """Guards the ``if _HARVESTED_GZ else`` above from defaulting into silence.

    An empty harvest would leave the fourth row holding the empty string, and
    the row's first assertion would fail loudly - but it would fail as a
    mystery. This says what the mystery is: the tracked tree stopped citing any
    gzipped record, so the phantom case has no live specimen left.
    """
    assert _HARVESTED_GZ, (
        "no tracked file cites a `.gz` record any more, so the PHANTOM row above "
        "has no real specimen. Do not substitute an invented string: find out why "
        "the citation left, then either restore it or retire the row deliberately."
    )
    assert all(tok.endswith(".gz") for tok in _HARVESTED_GZ), _HARVESTED_GZ
    assert ".gz" in C.FILE_SUFFIXES, (
        "`.gz` has been removed from FILE_SUFFIXES, which re-opens the I31 blind "
        "spot: gzipped citations become `dir_or_prefix` and are never enforced."
    )


def test_the_superseded_pattern_still_matches_what_it_was_built_for() -> None:
    """A negative control. If it matched nothing, the comparison would be empty."""
    assert C.SUPERSEDED_PATTERN.fullmatch("runs/s1.json")
    assert C.SUPERSEDED_PATTERN.fullmatch("runs/m6_innovation_autocorr.json")


# --------------------------------------------------------------------------
# THE POSITIVE CONTROL: A CITATION THIS SCAN HAS NEVER SEEN
# --------------------------------------------------------------------------


def _throwaway_repo(tmp_path: Path, body: str) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "module.py").write_text(body)
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    return tmp_path


def test_a_planted_citation_is_reported_by_the_real_walk(tmp_path: Path) -> None:
    """Plant one, in a repository the scan has never read, and read the verdict."""
    root = _throwaway_repo(
        tmp_path,
        f'"""Reads ``{_SPECIMEN_DEEP}`` and ``{_SPECIMEN_SCRIPT}``."""\n',
    )
    enum = C.enumerate_citations(root)
    problems = "\n".join(enum.problems)
    assert _SPECIMEN_DEEP in problems, problems
    assert _SPECIMEN_SCRIPT in problems, problems


def test_a_citation_of_a_tracked_file_is_not_reported(tmp_path: Path) -> None:
    """The negative half: tracking the cited file clears it. Same walk, same repo."""
    root = _throwaway_repo(tmp_path, f'"""Reads ``{_SPECIMEN_SCRIPT}``."""\n')
    (root / "runs").mkdir()
    (root / _SPECIMEN_SCRIPT).write_text("x = 1\n")
    subprocess.run(["git", "add", "-f", _SPECIMEN_SCRIPT], cwd=root, check=True)
    enum = C.enumerate_citations(root)
    assert enum.problems == [], "\n".join(enum.problems)


def test_a_directory_citation_is_reported_and_never_enforced(tmp_path: Path) -> None:
    """A run DIRECTORY is not an obligation: tracking one means tracking checkpoints."""
    root = _throwaway_repo(tmp_path, '"""    make replay RUN=runs/baselines-20260808-052918"""\n')
    enum = C.enumerate_citations(root)
    assert enum.problems == []
    assert [c.token for c in enum.of_kind("dir_or_prefix")] == ["runs/baselines-20260808-052918"]


# --------------------------------------------------------------------------
# THE WALK PRINTS WHAT IT CANNOT SEE (ADR-181 (7))
# --------------------------------------------------------------------------


def test_an_assembled_citation_is_invisible_to_the_walk_and_this_is_demonstrated(
    tmp_path: Path,
) -> None:
    """The blindness itself, planted and observed, before anything is said about it.

    Two forms, because they degrade differently. The first is interrupted right
    after the separator and vanishes entirely. The second has a literal stem, so
    a shorn token survives - WITHOUT its extension, which is what makes it look
    like a directory citation and so escape enforcement.
    """
    body = (
        "def a(run_id):\n"
        f'    return f"{_RUNS}{{run_id}}/model.json"\n'
        "def b(fold):\n"
        f'    return f"{_RUNS}s1-{{fold}}.json"\n'
    )
    root = _throwaway_repo(tmp_path, body)
    enum = C.enumerate_citations(root)

    # Neither assembled path is an obligation, and that is the defect being shown.
    assert enum.problems == [], "\n".join(enum.problems)
    dirs = {c.token for c in enum.of_kind("dir_or_prefix")}
    assert _RUNS.rstrip("/") + "/s1" in dirs, (
        f"the truncated stem should survive as an unenforced directory citation: {dirs}"
    )
    assert not any(c.token.endswith(".json") for c in enum.citations), (
        "neither assembled citation may appear as a file token: the walk cannot know "
        f"what the assembled name will be. {[c.token for c in enum.citations]}"
    )


def test_the_walk_names_the_places_it_is_blind(tmp_path: Path) -> None:
    """Same plant, now read through the reporter. Found, classified, and printed."""
    body = (
        "def a(run_id):\n"
        f'    return f"{_RUNS}{{run_id}}/model.json"\n'
        "def b(fold):\n"
        f'    return f"{_RUNS}s1-{{fold}}.json"\n'
    )
    root = _throwaway_repo(tmp_path, body)
    sites = C.assembled_sites(root)
    assert [s.path for s in sites] == ["module.py", "module.py"], sites
    assert [s.line for s in sites] == [2, 4], sites

    vanishing = [s for s in sites if not s.truncates]
    truncating = [s for s in sites if s.truncates]
    assert len(vanishing) == 1 and vanishing[0].stem == "", vanishing
    assert len(truncating) == 1 and truncating[0].stem == "s1-", truncating

    report = C._report(C.enumerate_citations(root), root)
    assert "SCAN LIMITS" in report, report
    assert "2 sites in 1 tracked files" in report, report
    assert "module.py:4" in report, report


def test_a_scan_limit_is_never_a_problem_and_never_sets_the_exit_status(
    tmp_path: Path,
) -> None:
    """The boundary is printed BESIDE the findings, never mixed into them.

    If assembly ever leaked into ``problems`` the check would fail on every
    repository that builds a run directory from a run id, which is all of them,
    and the pressure would then be to delete the diagnostic.
    """
    body = f'def a(run_id):\n    return f"{_RUNS}{{run_id}}"\n'
    root = _throwaway_repo(tmp_path, body)
    enum = C.enumerate_citations(root)
    assert C.assembled_sites(root), "the plant must be detected, or this proves nothing"
    assert enum.problems == [], "\n".join(enum.problems)


def test_the_limits_probe_does_not_count_itself() -> None:
    """A regression pin on a defect this probe shipped with for one revision.

    The first version classified a fragment by comparing it against a spelled
    out marker, so the tool matched its OWN source and reported 13 sites as 14.
    A detector that has to write down the thing it detects becomes its own first
    finding. The marker is a capture group now, and this holds it there.
    """
    sites = C.assembled_sites(repo_root())
    assert not any(s.path == "tools/cited_runs.py" for s in sites), [
        s for s in sites if s.path == "tools/cited_runs.py"
    ]
    assert not any(s.path == "tests/test_cited_runs.py" for s in sites), (
        "this test file now contains a literal assembled citation, so the specimens "
        "above are no longer being joined at run time"
    )
    assert sites, (
        "the probe found nothing anywhere, which means it is broken rather than that "
        "the tree is clean: this repository names every run directory from a run id"
    )


# --------------------------------------------------------------------------
# THE EXEMPTION CARRIES ITS OWN REASON, AND THE REASON IS MEASURED
# --------------------------------------------------------------------------


def test_every_exemption_states_a_reason_and_a_number() -> None:
    assert C.EXEMPT, "an empty exemption list would make the class test vacuous"
    for token, (reason, count) in C.EXEMPT.items():
        assert reason.strip(), token
        assert count > 0, f"{token}: an exemption whose measurement is 0 has no reason left"


def test_the_exemption_reason_is_still_true_where_the_file_is_present() -> None:
    """Fails in BOTH directions: a count that rises and a count that falls.

    Skipped in a clone, where the untracked evidence does not exist. That is
    stated rather than hidden: this assertion is a local one, and the assertion
    that gates CI is ``test_every_cited_runs_file_is_tracked_or_declared``, which
    is derived from git and therefore identical everywhere.
    """
    base = repo_root()
    checked = 0
    for token, (_reason, expected) in C.EXEMPT.items():
        path = base / token
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        actual = len(_ROLE.findall(text))
        assert actual == expected, (
            f"{token}: the exemption is justified by {expected} internal-role occurrences "
            f"and the file now carries {actual}. If it is 0 the file is publishable and the "
            "exemption must go; if it moved, the reason must be restated with the new number."
        )
        checked += 1
    if checked == 0:
        pytest.skip("no exempt artifact present: this is a clone, not the evidence machine")


def test_this_module_is_scanned_by_the_check_it_tests_and_is_not_exempt() -> None:
    """No self-exemption, asserted rather than trusted.

    The specimens above are joined at runtime so this file contains no literal
    unresolvable citation. That property is what buys the absence of an
    exemption, so it is checked here rather than assumed.
    """
    rel = "tests/test_cited_runs.py"
    assert rel in C.tracked_files(repo_root()), "not tracked, so the scan never reads it"
    source = (repo_root() / rel).read_bytes().decode()
    unresolved = {
        t
        for t in C.tokens_in(source)
        if t.endswith(C.FILE_SUFFIXES) and t not in C.EXEMPT and t not in set(C.tracked_files())
    }
    assert not unresolved, f"this file now contains a LITERAL unresolvable citation: {unresolved}"
    # ...and the fragments really are here, so the halves are halves.
    assert "ru" in source and _RUNS in _SPECIMEN_DEEP
