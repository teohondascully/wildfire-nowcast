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
