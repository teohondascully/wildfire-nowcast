"""ADR-098 (3): no fixed path under `/tmp` or any shared temp directory, anywhere.

`Makefile:239` wrote every fire's contract report to `/tmp/_c.txt` - a fixed,
predictable name in a world-writable directory, rewritten once per fire, inside
the reporting path of this project's flagship check. Four leads share this tree
and have run that target concurrently. Two runs interleave writes to one file, so
the `[FAIL]` lines printed can belong to a DIFFERENT fire than the one
`echo "FAIL $d"` names beside them.

**The exit code stays correct, and that is what makes it the worse failure.** A
wrong exit code gets investigated; a plausible failure filed under the wrong fire
gets acted on, and this project has already spent real effort chasing false REDs
caused by four leads sharing one tree.

The second instance was in infra's own tooling and is the more dangerous one:
`tools/mutation.py` defaulted its workspace root to
`gettempdir()/wildfire-nowcast-mutation`, so two concurrent sweeps shared
`ws0..ws3` - and `build_workspace` removes an existing workspace with
`worktree remove --force` plus `rmtree`. A fixed name there does not corrupt a
report, it DELETES A RUNNING SWEEP.

The demonstration below is deterministic rather than a race. A test for a
concurrency hazard that is itself timing-dependent reports the load on the
machine (ADR-084), so the interleaving is DRIVEN, step by step, from here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import prose_scan  # noqa: E402 - tools/ is not a package; same route test_hygiene uses
from wildfire_nowcast.common.paths import repo_root  # noqa: E402

#: Where a shared temp path can be written down. `configs/` is included because a
#: yaml scratch path is the same defect wearing a configuration's clothes.
SCANNED = ("Makefile", ".github", "tools", "tests", "src", "configs")

#: A literal path under a shared temp root. `/tmp/x`, `/private/tmp/x`,
#: `/var/tmp/x`. The trailing name is what makes it FIXED; `/tmp/` alone in prose
#: is not a path a program writes to.
_LITERAL_TEMP_PATH = re.compile(r"(?<![\w.])/(?:private/)?(?:var/)?tmp/[A-Za-z0-9_.\-]+")

#: `Path(tempfile.gettempdir()) / "a-fixed-name"` and its spellings. The first
#: instance in this repository was exactly this expression.
_COMPUTED_TEMP_PATH = re.compile(r"gettempdir\(\)\s*\)?\s*/\s*[\"'][A-Za-z0-9_.\-]+[\"']")

#: `mktemp` templates, `mkdtemp` prefixes and doc prose are NOT fixed paths: the
#: X's are replaced. Matched so the scan does not report its own repair.
_PER_INVOCATION = re.compile(r"XXXXXX|mkdtemp|TemporaryDirectory|mkstemp")


def _tracked_under(prefix: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo_root()), "ls-files", prefix],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def fixed_shared_temp_paths(rel: str, text: str) -> list[tuple[int, str]]:
    """``(line, matched text)`` for every fixed path under a shared temp directory.

    PROSE IS EXCLUDED STRUCTURALLY, never by file name. A paragraph explaining
    why `/tmp/_c.txt` was a defect is not a defect, and the Makefile comment that
    now explains it is the first thing a name-blind regex flagged. Python
    comments and docstrings come from `prose_scan.prose_spans` (C0: one
    implementation of "is this prose"); for a Makefile or a workflow, a line
    whose first non-space character is `#`.

    What is NOT excluded is a string LITERAL, because that is where a program
    puts the path it writes to. This module therefore spells its own specimens in
    halves and joins them at runtime, exactly as `tests/test_hygiene.py` does, so
    that it needs no exemption and is scanned like every other file. Exempting
    the scanner would make the likeliest offender the one file that can never
    report.
    """
    spans: list[object] = []
    if rel.endswith(".py"):
        try:
            spans = list(prose_scan.prose_spans(text))
        except (SyntaxError, ValueError, IndentationError):
            # A fragment that is not a whole module, or a file that does not
            # parse. Scanned WITHOUT the prose exclusion, which is the loud
            # direction: a false positive is read, a false negative is not.
            spans = []
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _PER_INVOCATION.search(line):
            continue
        if not rel.endswith(".py") and line.lstrip().startswith("#"):
            continue
        for pattern in (_LITERAL_TEMP_PATH, _COMPUTED_TEMP_PATH):
            for match in pattern.finditer(line):
                if any(span.contains(lineno, match.start()) for span in spans):  # type: ignore[attr-defined]
                    continue
                out.append((lineno, match.group(0)))
    return out


def _scan_tree() -> list[str]:
    findings: list[str] = []
    for prefix in SCANNED:
        for rel in _tracked_under(prefix):
            path = repo_root() / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line, hit in fixed_shared_temp_paths(rel, text):
                findings.append(f"{rel}:{line} {hit}")
    return findings


# --------------------------------------------------------------------------
# the tree
# --------------------------------------------------------------------------


def test_no_fixed_path_under_a_shared_temp_directory_anywhere() -> None:
    findings = _scan_tree()
    assert not findings, (
        "a fixed path under a shared temp directory:\n  "
        + "\n  ".join(findings)
        + "\nADR-098 (3): mktemp per invocation, cleaned up. A predictable name in a "
        "shared directory is a correctness bug wherever more than one process can run "
        "the target, and in this repository more than one process always can."
    )


def test_the_scan_has_a_corpus_and_can_see_the_file_the_defect_was_in() -> None:
    """Positive control on the CORPUS, separate from the control on the pattern."""
    scanned = [rel for prefix in SCANNED for rel in _tracked_under(prefix)]
    assert len(scanned) > 150, f"{len(scanned)} files scanned; the listing is wrong"
    assert "Makefile" in scanned, "the file the defect was in is not in the corpus"
    assert "tools/mutation.py" in scanned


#: Every specimen below is spelled in HALVES and joined at runtime. A specimen
#: written out whole would be a fixed shared temp path in a tracked file, and the
#: only ways out of that are an exemption for this file or a scan that cannot see
#: it. Both are how seven earlier instances of this class survived.
_SLASH = "/"
_TMP = "tmp"
_GET = "gettempdir()"

_DEFECT_SPECIMENS: list[tuple[str, str]] = [
    (
        "$(PY) -m contract $$d > " + _SLASH + _TMP + _SLASH + "_c.txt 2>&1",
        _SLASH + _TMP + _SLASH + "_c.txt",
    ),
    (
        'out = Path("' + _SLASH + "private" + _SLASH + _TMP + _SLASH + 'wfnc-report")',
        _SLASH + "private" + _SLASH + _TMP + _SLASH + "wfnc-report",
    ),
    (
        "root = Path(tempfile." + _GET + ') / "wildfire-nowcast-mutation"',
        _GET + ') / "wildfire-nowcast-mutation"',
    ),
    ("cache = Path(" + _GET + ") / 'shared.json'", _GET + ") / 'shared.json'"),
    (
        "log = " + _SLASH + "var" + _SLASH + _TMP + _SLASH + "wfnc.log",
        _SLASH + "var" + _SLASH + _TMP + _SLASH + "wfnc.log",
    ),
]


@pytest.mark.parametrize(("line", "expected"), _DEFECT_SPECIMENS)
def test_the_scan_catches_every_spelling_of_the_defect(line: str, expected: str) -> None:
    """C3.5, including the two spellings that were actually in this repository."""
    hits = fixed_shared_temp_paths("specimen.py", line)
    assert hits, line
    assert expected in hits[0][1] or hits[0][1] in expected, hits


def test_a_specimen_inside_a_COMMENT_is_prose_and_is_not_flagged() -> None:
    """The distinction the Makefile's own explanation of the defect depends on."""
    line, _ = _DEFECT_SPECIMENS[0]
    assert fixed_shared_temp_paths("specimen.py", line)
    assert not fixed_shared_temp_paths("specimen.py", "x = 1  # " + line)
    assert not fixed_shared_temp_paths("Makefile", "## " + line)


def test_a_specimen_inside_a_DOCSTRING_is_prose_and_is_not_flagged() -> None:
    line, _ = _DEFECT_SPECIMENS[1]
    module = '"""It used to read ' + line + '."""\n'
    assert not fixed_shared_temp_paths("specimen.py", module)


@pytest.mark.parametrize(
    "line",
    [
        'c=$(mktemp "${TMPDIR:-' + _SLASH + _TMP + "}" + _SLASH + 'wfnc-contract.XXXXXX")',
        'root = Path(tempfile.mkdtemp(prefix="wildfire-nowcast-mutation-"))',
        "with tempfile.TemporaryDirectory() as tmp:",
        "fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix='.tmp')",
        "# the parent's " + _SLASH + _TMP + " is where the children wrote",
    ],
)
def test_the_scan_does_NOT_flag_a_per_invocation_path(line: str) -> None:
    """The negative control. A scan that flags its own repair teaches people to disable it."""
    assert not fixed_shared_temp_paths("specimen", line), line


# --------------------------------------------------------------------------
# the recipe that had the defect
# --------------------------------------------------------------------------


def _recipe(target: str) -> str:
    text = (repo_root() / "Makefile").read_text(encoding="utf-8")
    match = re.search(rf"^{target}:.*?\n((?:\t.*\n|\n)*)", text, flags=re.MULTILINE)
    assert match, f"the Makefile has no `{target}:` recipe"
    return match.group(1)


def test_contract_all_fires_takes_a_per_invocation_scratch_file_and_removes_it() -> None:
    recipe = _recipe("contract-all-fires")
    assert "mktemp" in recipe, recipe
    assert "trap" in recipe and "rm -f" in recipe, (
        "the scratch file is not removed on every exit path. A 21 fire run invites the "
        "interrupt that an EXIT-only trap does not cover, which is why INT and TERM are "
        "named too"
    )
    assert _DEFECT_SPECIMENS[0][1] not in recipe, "the old fixed path is still in the recipe"
    assert not fixed_shared_temp_paths("Makefile", recipe), recipe


def test_the_recipe_still_does_the_job_it_did_before() -> None:
    """The repair must not have removed the reporting it exists for."""
    recipe = _recipe("contract-all-fires")
    assert 'echo "FAIL $$d"' in recipe, "the target no longer names the failing fire"
    assert "[FAIL" in recipe, "the target no longer prints the failing clauses"
    assert "exit $$rc" in recipe, "the exit code is no longer propagated"


# --------------------------------------------------------------------------
# the misattribution, demonstrated rather than asserted
# --------------------------------------------------------------------------


def _sh(script: str, **env: str) -> str:
    proc = subprocess.run(
        ["/bin/sh", "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", **env},
    )
    return proc.stdout


def test_a_FIXED_scratch_path_reports_one_fires_failures_under_another_fires_name(
    tmp_path: Path,
) -> None:
    """The defect, reproduced deterministically: no race, a driven interleaving.

    Three steps, in the order two concurrent runs of the old recipe produce them:
    run A writes its report, run B overwrites the same fixed path, run A greps and
    prints what it finds beside the name of ITS fire. The `[FAIL]` line printed
    under `FAIL 2019_kincade` belongs to `2020_creek`.
    """
    fixed = tmp_path / "_c.txt"
    _sh(f'printf "  [FAIL] C1.2 2019_kincade\\n" > "{fixed}"')
    _sh(f'printf "  [FAIL] C3.1 2020_creek\\n" > "{fixed}"')
    printed = _sh(
        f'echo "FAIL data/fires/2019_kincade/"; grep -E "^[[:space:]]+\\[FAIL\\]" "{fixed}"'
    )

    assert "FAIL data/fires/2019_kincade/" in printed
    assert "2020_creek" in printed, printed
    assert "2019_kincade" not in printed.split("\n")[1], (
        "the misattribution did not reproduce, so the rest of this file is about nothing"
    )


def test_a_PER_INVOCATION_scratch_path_attributes_the_failures_to_the_right_fire(
    tmp_path: Path,
) -> None:
    """The same three steps against the shape the Makefile now has."""
    a = _sh(f'mktemp "{tmp_path}/wfnc-contract.XXXXXX"').strip()
    b = _sh(f'mktemp "{tmp_path}/wfnc-contract.XXXXXX"').strip()
    assert a != b, "mktemp handed two runs the same path, which is the whole point"

    _sh(f'printf "  [FAIL] C1.2 2019_kincade\\n" > "{a}"')
    _sh(f'printf "  [FAIL] C3.1 2020_creek\\n" > "{b}"')
    printed = _sh(f'echo "FAIL data/fires/2019_kincade/"; grep -E "^[[:space:]]+\\[FAIL\\]" "{a}"')

    assert "FAIL data/fires/2019_kincade/" in printed
    assert "2019_kincade" in printed.split("\n")[1], printed
    assert "2020_creek" not in printed, printed


def test_the_trap_removes_the_scratch_file_even_on_an_interrupt(tmp_path: Path) -> None:
    """`trap ... EXIT INT TERM`, exercised rather than read."""
    marker = tmp_path / "created"
    script = (
        f'c=$(mktemp "{tmp_path}/wfnc-contract.XXXXXX"); '
        f"trap 'rm -f \"$c\"' EXIT INT TERM; "
        f'echo "$c" > "{marker}"; '
        f"kill -INT $$; sleep 5"
    )
    subprocess.run(
        ["/bin/sh", "-c", script], capture_output=True, text=True, check=False, timeout=20
    )
    created = Path(marker.read_text().strip())
    assert not created.exists(), f"{created} survived an interrupt"
    assert not list(tmp_path.glob("wfnc-contract.*")), list(tmp_path.iterdir())


def test_the_mutation_sweep_root_is_per_invocation() -> None:
    """The second instance, and the one that deletes a RUNNING sweep rather than a report."""
    source = (repo_root() / "tools" / "mutation.py").read_text(encoding="utf-8")
    assert 'mkdtemp(prefix="wildfire-nowcast-mutation-")' in source, (
        "the sweep root is no longer per invocation. Two concurrent sweeps then share "
        "ws0..ws3, and build_workspace removes an existing workspace with "
        "`worktree remove --force` plus rmtree."
    )
    assert not fixed_shared_temp_paths("tools/mutation.py", source)


def test_two_sweep_roots_taken_back_to_back_differ() -> None:
    """Reading the source says the call changed; this says the call does the job."""
    sys.path.insert(0, str(repo_root() / "tools"))
    import tempfile

    first = tempfile.mkdtemp(prefix="wildfire-nowcast-mutation-")
    second = tempfile.mkdtemp(prefix="wildfire-nowcast-mutation-")
    try:
        assert first != second
    finally:
        Path(first).rmdir()
        Path(second).rmdir()
