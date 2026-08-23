"""[I6] No `.pre-commit-config.yaml` hook may rewrite a file during a commit.

WHY. `eval/reporting.scoring_code_fingerprint` hashes the SOURCE BYTES of every
module under `model/` and `eval/`, and every run artifact stamps that hash at
both ends (C-4.2). A hook that edits the worktree therefore moves the identity of
the code an artifact claims to have been produced by, invisibly, inside a commit
that was about something else. Measured on the day this test was written: the
`ruff-format` hook rewrote `model/train.py` and moved the fingerprint
`85ecff60dff714b8` -> `5b44d1d308ebec0d`, while `85ecff60dff714b8` is the hash
carried by every S1 number on disk. The commit that hit it was stopped and
escalated rather than worked around, and the ruling was that a rewriting
formatter is incompatible with a code-fingerprint contract.

IT IS A CLASS, NOT AN INCIDENT. `ruff format --check src tests tools` reported
**88 files would be reformatted, 62 already formatted** when the hook was
changed. A rewriting hook discharges that debt file-by-file, silently, inside
whatever commit happens to touch each file. Formatting debt is discharged
deliberately, in its own commit, or not at all.

WHAT IS ASSERTED HERE, IN ORDER OF STRENGTH.
  1. A MEASUREMENT: the configured commands are run against a deliberately
     misformatted file and a deliberately unlinted one. They must exit non-zero
     and leave the bytes ALONE. The positive control runs the SAME commands with
     the config's args removed and asserts the bytes DO change -- so a pass here
     means "the args prevented a rewrite that would otherwise have happened",
     not "this file was never at risk".
  2. The arg lists in the config, so the measurement cannot be satisfied by a
     command nobody runs.
  3. That the hook's ruff is the ruff `make lint` runs, which is what makes (1)
     a faithful stand-in for the hook rather than an analogy.
  4. [I20] That `make lint` runs every ruff SUBCOMMAND this config enforces.
     (3) pinned the binary and left the commands unpinned, so the two agreed
     about which ruff and disagreed about what it was asked to do.

WHAT (4) IS FOR, AND IT IS THE DEFECT THIS MODULE ALMOST HAD. `make lint` was
`ruff check` alone for the whole life of this repository, while the hooks below
run `ruff` AND `ruff-format --check`. Every "lint clean" ever reported here was
therefore strictly narrower than any reader would take it, INCLUDING the people
who asked for it. Nothing false was ever said, which is why no amount of reading
reports would have found it: it surfaced when the stronger check refused a
commit seconds after the weaker one reported zero. A check that passes is not
evidence until you know what it declines to look at.

STATED LIMIT. (1) invokes ruff directly with the args this repo contributes,
rather than through `pre-commit`. Going through `pre-commit` would clone
`ruff-pre-commit` from GitHub on a cold store, and this suite is otherwise
network-free (ADR-070 (5)). The entry `ruff format` / `ruff check` is
ruff-pre-commit's, reproduced here; (3) pins the version, and
`tests/test_push_guard.py` pins which stages these hooks run at.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from wildfire_nowcast.common.paths import repo_root

#: THE MAINTAINED SURFACE, PINNED, AND DELIBERATELY SMALL. "Does this hook
#: rewrite files" is not derivable for an arbitrary hook, so the residue is
#: declared: these are ruff's mutating flags. Growing this set is a visible edit.
#: Everything else in this file is measured or read from the config.
MUTATING_FLAGS = {"--fix", "--fix-only", "--unsafe-fixes"}

#: `ruff format` rewrites unless told to report instead. Either of these turns it
#: into a reporter; `--diff` also prints what it would have done.
NON_MUTATING_FORMAT_FLAGS = {"--check", "--diff"}

BADLY_FORMATTED = "def f( x ,y ):\n    return   {  'a':x,'b':y }\n"
UNLINTED = "import os\nimport sys\n\n\ndef g() -> int:\n    return 1\n"


def _config() -> dict[str, object]:
    return yaml.safe_load((repo_root() / ".pre-commit-config.yaml").read_text())


def _ruff_repo() -> dict[str, object]:
    repos = [
        r
        for r in _config()["repos"]  # type: ignore[index]
        if isinstance(r.get("repo"), str) and r["repo"].endswith("ruff-pre-commit")
    ]
    assert len(repos) == 1, f"expected exactly one ruff-pre-commit repo, found {len(repos)}"
    return repos[0]


def _ruff_hooks() -> dict[str, dict[str, object]]:
    hooks = {h["id"]: h for h in _ruff_repo()["hooks"]}  # type: ignore[union-attr]
    assert {"ruff", "ruff-format"} <= set(hooks), (
        f"the ruff hooks are not registered under the ids this test knows: {sorted(hooks)}"
    )
    return hooks


def _args(hook_id: str) -> list[str]:
    return [str(a) for a in _ruff_hooks()[hook_id].get("args", [])]


def _ruff() -> Path:
    ruff = repo_root() / ".venv" / "bin" / "ruff"
    if not ruff.is_file():  # pragma: no cover - environment without the venv
        found = shutil.which("ruff")
        assert found, "no ruff on this machine, so nothing below could be measured"
        return Path(found)
    return ruff


def _run(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, check=False)


# --------------------------------------------------------------------------
# 1. THE MEASUREMENT, with the control that makes a pass mean something
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hook_id", "subcommand", "content"),
    [("ruff-format", "format", BADLY_FORMATTED), ("ruff", "check", UNLINTED)],
)
def test_the_configured_command_REFUSES_the_defect_and_leaves_the_BYTES_ALONE(
    tmp_path: Path, hook_id: str, subcommand: str, content: str
) -> None:
    """PLANTED DEFECT AND POSITIVE CONTROL IN ONE TEST, because separately either lies.

    The defect: a file ruff objects to. The control: the same file, the same
    command, with this repo's args removed -- which MUST rewrite it. Without the
    control, a green here is also what you would see if ruff silently did
    nothing, if the file were excluded, or if the binary were missing.
    """
    guarded = tmp_path / "planted.py"
    guarded.write_text(content)
    before = guarded.read_bytes()

    done = _run(
        [str(_ruff()), subcommand, "--force-exclude", *_args(hook_id), guarded.name], tmp_path
    )
    assert done.returncode != 0, (
        f"`ruff {subcommand}` with the configured args ACCEPTED a planted defect:\n"
        f"{done.stdout}{done.stderr}"
    )
    assert guarded.read_bytes() == before, (
        f"the {hook_id} hook REWROTE the file. That is the defect this test exists for: a "
        "rewrite during `git commit` moves `scoring_code_fingerprint` and detaches every "
        "artifact bound to it from the code that produced it."
    )

    control = tmp_path / "control.py"
    control.write_text(content)
    fixer = ["--fix"] if subcommand == "check" else []
    _run([str(_ruff()), subcommand, "--force-exclude", *fixer, control.name], tmp_path)
    assert control.read_text() != content, (
        "the control did NOT rewrite the file, so the planted defect was not something ruff "
        "would have changed and the assertion above proves nothing"
    )


# --------------------------------------------------------------------------
# 2. The config that decides which command runs
# --------------------------------------------------------------------------


def test_no_ruff_hook_carries_a_mutating_flag() -> None:
    for hook_id, hook in _ruff_hooks().items():
        offenders = MUTATING_FLAGS.intersection(str(a) for a in hook.get("args", []))
        assert not offenders, (
            f"the {hook_id!r} hook carries {sorted(offenders)}, so `git commit` rewrites source "
            "files. See the block at the top of .pre-commit-config.yaml: this repo stamps a hash "
            "of source bytes into every run artifact."
        )


def test_the_formatter_hook_reports_instead_of_reformatting() -> None:
    args = set(_args("ruff-format"))
    assert args & NON_MUTATING_FORMAT_FLAGS, (
        "`ruff-format` carries neither --check nor --diff, so it REWRITES files during a commit. "
        f"Give it one of {sorted(NON_MUTATING_FORMAT_FLAGS)}; formatting debt is discharged in "
        "its own labelled commit, never as a side effect of a science commit."
    )


def test_the_reason_is_recorded_beside_the_setting_so_it_is_not_fixed_back() -> None:
    """A bare `--check` reads like a preference and gets 'corrected' by the next reader.

    Four of this project's recurring defects are a rule that outlived the reason
    for it. The fingerprint is the reason, so the fingerprint has to be in the
    file that carries the setting.
    """
    text = (repo_root() / ".pre-commit-config.yaml").read_text()
    assert "scoring_code_fingerprint" in text, (
        "the config no longer says WHY these hooks may not rewrite files. Restore the rationale "
        "in the same edit that changes the setting."
    )


# --------------------------------------------------------------------------
# 3. What makes (1) a stand-in for the hook rather than an analogy
# --------------------------------------------------------------------------


def test_the_hooks_ruff_is_the_ruff_the_gate_and_the_lockfile_pin() -> None:
    """The config asserts this in prose and, until this test, nothing enforced it.

    A hook on a different ruff than `make lint` can pass locally and fail in CI,
    which teaches people to distrust the hook -- and it would also make the
    measurement above a statement about a version nobody runs.
    """
    hook_version = str(_ruff_repo()["rev"]).lstrip("v")

    lock = (repo_root() / "requirements.lock").read_text()
    # [I8] The lock now carries hashes, so a pinned row is `ruff==X \` with the
    # `--hash=` lines indented under it. The optional continuation is the ONLY
    # relaxation: still exactly one row, still anchored, still an exact version.
    pinned = re.findall(r"^ruff==(\S+?)(?: \\)?$", lock, flags=re.MULTILINE)
    assert len(pinned) == 1, f"requirements.lock pins ruff {pinned}"

    out = _run([str(_ruff()), "--version"], repo_root()).stdout.split()
    installed = out[1] if len(out) > 1 else ""

    assert hook_version == pinned[0] == installed, (
        f"the pre-commit hook runs ruff {hook_version}, requirements.lock pins {pinned[0]}, "
        f".venv has {installed}. Bump all three in one commit."
    )


# --------------------------------------------------------------------------
# 4. [I20] The gate must run what the hook enforces, not merely the same binary
# --------------------------------------------------------------------------

#: ruff-pre-commit's hook-id to subcommand mapping. A DECLARED residue, like
#: ``MUTATING_FLAGS`` above: it is not derivable from the config, it belongs to
#: the upstream repository, and growing it is a visible edit. A hook id absent
#: from this map is REFUSED below rather than skipped, because a new ruff hook
#: that this map has never heard of is exactly the thing that would slip past.
HOOK_SUBCOMMAND = {"ruff": "check", "ruff-format": "format"}

#: The Makefile target that every status report in this project quotes.
GATE_TARGET = "lint"


def _makefile_recipe(target: str) -> list[str]:
    """The recipe lines of ``target``, read from the Makefile.

    Tab-indented lines following ``target:`` up to the next non-indented,
    non-blank, non-comment line. Deliberately not a make invocation: running
    ``make -n`` would need the venv and would report what make WOULD do on this
    machine, and the subject here is what the file says every machine does.
    """
    lines = (repo_root() / "Makefile").read_text().splitlines()
    out: list[str] = []
    collecting = False
    for line in lines:
        if re.match(rf"^{re.escape(target)}\s*:", line):
            collecting = True
            continue
        if collecting:
            if line.startswith("\t"):
                out.append(line.strip())
            elif line.strip() == "" or line.lstrip().startswith("#"):
                continue
            else:
                break
    return out


def _ruff_subcommands_in(recipe: list[str]) -> set[str]:
    """Every ``$(RUFF) <subcommand>`` the recipe invokes."""
    found = set()
    for line in recipe:
        for match in re.finditer(r"\$\(RUFF\)\s+([a-z-]+)", line):
            found.add(match.group(1))
    return found


def test_the_makefile_parse_is_not_silently_empty() -> None:
    """POSITIVE CONTROL, first. A parser that finds nothing agrees with everything."""
    recipe = _makefile_recipe(GATE_TARGET)
    assert recipe, (
        f"the Makefile has no `{GATE_TARGET}:` target, or its recipe could not be read. "
        "Every assertion below would pass vacuously."
    )
    assert _ruff_subcommands_in(recipe), f"no `$(RUFF) <cmd>` found in `{GATE_TARGET}`: {recipe}"
    # ...and it must be able to see MORE than one, or "covers all of them" below
    # is satisfied by a parser that only ever finds the first.
    probe = ["$(RUFF) check a b", "$(RUFF) format --check a b"]
    assert _ruff_subcommands_in(probe) == {"check", "format"}


def test_every_hook_id_is_one_this_test_knows_how_to_read() -> None:
    """A ruff hook this map has never heard of must be LOUD, not silently uncovered."""
    unknown = sorted(set(_ruff_hooks()) - set(HOOK_SUBCOMMAND))
    assert not unknown, (
        f"the config runs ruff hook(s) {unknown} that this module cannot map to a "
        "subcommand, so the coverage assertion below would not see them. Add the "
        "mapping to HOOK_SUBCOMMAND in the same edit that adds the hook."
    )


def test_make_lint_runs_every_ruff_subcommand_the_commit_hook_enforces() -> None:
    """[I20] The gate a lead runs must not be narrower than the gate that decides landing.

    This is the assertion that would have failed for the whole life of the
    repository. `make lint` ran `ruff check`; the hook ran `ruff` AND
    `ruff-format --check`; both were reported as "lint clean" and neither report
    was false.
    """
    enforced = {HOOK_SUBCOMMAND[h] for h in _ruff_hooks()}
    covered = _ruff_subcommands_in(_makefile_recipe(GATE_TARGET))
    missing = sorted(enforced - covered)
    assert not missing, (
        f"`make {GATE_TARGET}` does not run `ruff {' / '.join(missing)}`, which "
        ".pre-commit-config.yaml enforces on every commit. The gate a lead runs and "
        "reports would be strictly weaker than the gate that decides whether the work "
        "can land, and both would be called lint. Add the missing command to the "
        f"`{GATE_TARGET}` recipe; do not remove the hook to make this pass."
    )


def test_the_gate_formats_in_report_mode_so_it_cannot_rewrite_source() -> None:
    """`make lint` runs in CI and locally. A bare `ruff format` there is the C-4.2 hazard.

    The whole reason the hook carries ``--check`` is that a rewrite moves
    ``scoring_code_fingerprint`` under artifacts bound to it. Putting the
    formatter into the gate without ``--check`` would reintroduce that in the one
    place nobody looks, because a gate is expected to be read-only.
    """
    for line in _makefile_recipe(GATE_TARGET):
        if re.search(r"\$\(RUFF\)\s+format", line) and not (
            set(line.split()) & NON_MUTATING_FORMAT_FLAGS
        ):
            raise AssertionError(
                f"`make {GATE_TARGET}` runs the formatter in REWRITE mode: {line!r}. "
                f"Give it one of {sorted(NON_MUTATING_FORMAT_FLAGS)}."
            )


def _path_operands(line: str) -> tuple[str, ...]:
    """The path operands of one ``$(RUFF) <subcommand> [flags] <paths...>`` line.

    Everything up to and including the subcommand is dropped, then flags. Written
    as its own function because the first version of the caller kept the
    subcommand and reported `check` and `format` as two different path sets --
    a check that fails on a correct tree, found by running the control first.
    """
    after = re.sub(r"^.*\$\(RUFF\)\s+[a-z-]+\s*", "", line)
    return tuple(a for a in after.split() if not a.startswith("-"))


def test_both_halves_of_the_gate_cover_the_same_paths() -> None:
    """Two halves with two path lists is how a gate half goes blind on a directory.

    Asserted on the recipe text rather than by running ruff twice: the failure
    being prevented is somebody adding a root to one line and not the other.
    """
    assert _path_operands("\t$(RUFF) format --check src tests") == ("src", "tests")
    scopes = {_path_operands(line) for line in _makefile_recipe(GATE_TARGET) if "$(RUFF)" in line}
    assert len(scopes) == 1, (
        f"the halves of `make {GATE_TARGET}` lint different path sets: {sorted(scopes)}. "
        "Use the shared LINT_PATHS variable so they cannot drift."
    )
