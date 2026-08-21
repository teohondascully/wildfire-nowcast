"""[I5, amended I6] Nothing is published that this remote does not already carry.

WHY. The public history was rebuilt from scratch before the repo went public.
The history it replaced is still on disk, now as `refs/archive/pre-public-backup`:
52 commits, **no merge base with `main`**, and files that name internal tooling in
comments and docstrings. `git push --mirror` would publish all of it. The commit
messages are clean, so `tools/commit_guard.py` -- the I4 guard -- would pass every
one of them; this is a different failure and needs a different instrument.

[I6] THE SUBJECT IS MANUFACTURED, NOT FOUND. Moving that ref out of
`refs/heads/` (ADR-070 (4.2)) made `git push --all` structurally unable to name
it -- a real fix -- and it also deleted the subject of two tests here, which
stopped adjudicating anything and reported SKIPPED. A skip is not a pass, but a
summary line does not say so. Both now build their own orphan branch in a
throwaway repository (the `hazard` fixture), so they discriminate no matter what
this repository's refs look like, and a third sweeps EVERY namespace of the real
checkout so the archived backup is still covered where it actually lives. No test
in this file can be disarmed by repository state, and none of them skips.

THREE LAYERS, AND EACH ONE CATCHES THE REMOVAL OF THE ONE BELOW IT.
  (a) `<hooks>/pre-push.legacy` -- the AUTHORITATIVE layer. It reads git's own
      pre-push protocol on stdin and therefore sees EVERY ref in the push.
  (b) the `push-guard` hook in `.pre-commit-config.yaml` -- arrives with
      `make install`. It CANNOT see the whole push (measured; see the module
      docstring of `tools/push_guard.py`) so its job is to refuse when layer (a)
      did not run, which is what makes deleting layer (a) fail CLOSED.
  (c) these tests, which assert both are wired in this checkout.

THE RULES ARE DERIVED, NOT LISTED. No branch name appears in
`tools/push_guard.py`; the publishable set is read from `refs/remotes/<remote>/`,
and "unrelated history" is `git merge-base`'s verdict. Two tests below pin that:
one asserts the derived set is exactly what this repository's remote carries
today, the other asserts no branch name has crept into the module as a literal.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

import push_guard  # tools/push_guard.py, via `pythonpath` in pyproject.toml
from wildfire_nowcast.common.paths import repo_root

ZERO = "0" * 40

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _run(args: Sequence[str], cwd: Path, **kw: object) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    return subprocess.run(
        list(args), cwd=cwd, capture_output=True, text=True, env=env, check=False, **kw
    )


def _git(args: Sequence[str], cwd: Path) -> str:
    done = _run(["git", *args], cwd)
    assert done.returncode == 0, f"git {args} failed: {done.stderr}"
    return done.stdout


def _remote_branches(bare: Path) -> set[str]:
    out = _git(["--git-dir", str(bare), "for-each-ref", "--format=%(refname)", "refs/heads/"], bare)
    return set(out.split())


def _local_branches(repo: Path) -> set[str]:
    return set(_git(["for-each-ref", "--format=%(refname:short)", "refs/heads/"], repo).split())


def _all_local_refs(repo: Path) -> set[str]:
    """EVERY local ref, not only `refs/heads/`.

    `refs/heads/` IS NOT THE SUBJECT. The pre-public backup was moved to
    `refs/archive/pre-public-backup` (ADR-070 (4.2)) so that `git push --all`,
    which matches `refs/heads/*` only, is structurally unable to name it. That is
    a good fix and it deleted this file's subject: two tests below stopped
    discriminating and reported a SKIP, which reads like a pass. `git push
    --mirror` still traverses every namespace, so the hazard is still real and
    it is still on disk -- it is just no longer where a `refs/heads/` query
    looks. Ask git for all of them.
    """
    return set(_git(["for-each-ref", "--format=%(refname)"], repo).split())


def _ref_leaf_names(repo: Path) -> set[str]:
    """The bare names a guard must not have written down, from every namespace."""
    return {ref.rsplit("/", 1)[-1] for ref in _all_local_refs(repo)}


# --------------------------------------------------------------------------
# The derivation, measured against THIS repository
# --------------------------------------------------------------------------


def test_the_publishable_set_is_derived_and_is_exactly_what_the_remote_carries() -> None:
    """POSITIVE CONTROL FIRST: an empty derived set would make every test below vacuous.

    The set is read from `refs/remotes/origin/`. Pinning the value here is the
    same move as pinning `EXPECTED_CONSTRUCTIONS` for the commit guard: the rule
    stays derived, and the day the derivation starts returning something else,
    that is a diff someone has to argue for rather than a silent widening.
    """
    published = push_guard.published_refs(repo_root(), "origin")
    assert published, "the derived publishable set is EMPTY, so nothing below proves anything"
    assert published == {"refs/heads/main"}, (
        f"this remote now carries {sorted(published)}. That is allowed, and it is meant to be "
        "visible: update this assertion in the same commit."
    )


def test_no_branch_name_is_written_down_anywhere_in_the_guard() -> None:
    """If a branch name ever appears as a literal, the rule stopped being derived.

    Docstrings are excluded on purpose -- prose may name a branch, code may not.
    """
    source = (repo_root() / "tools" / "push_guard.py").read_text()
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]
    assert literals, "no string literals found: the scan is broken, not the module clean"
    # Over EVERY namespace, not `refs/heads/`. `pre-public-backup` -- the name
    # this module must not contain, and the only one that has ever been at risk
    # of being typed in -- lives under `refs/archive/` now, so a `refs/heads/`
    # query would quietly stop checking the one name that matters.
    for branch in _ref_leaf_names(repo_root()):
        offenders = [
            lit for lit in literals if lit == branch or f"/{branch}" in lit or f"{branch}/" in lit
        ]
        assert not offenders, (
            f"{branch!r} is written into tools/push_guard.py as a literal ({offenders}), so the "
            "publishable set is no longer derived from the remote"
        )


# --------------------------------------------------------------------------
# The real hazard, adjudicated WITHOUT performing it
# --------------------------------------------------------------------------


def _push_line(repo: Path, local: str, destination: str, remote_sha: str) -> push_guard.PushRef:
    sha = _git(["rev-parse", local], repo).strip()
    return push_guard.PushRef(local, sha, destination, remote_sha)


@pytest.fixture(scope="session")
def hazard(_template: Path) -> Path:
    """A REPOSITORY THAT CARRIES THE HAZARD BY CONSTRUCTION, not by luck.

    THIS FIXTURE EXISTS BECAUSE THE TESTS BELOW USED TO READ THE REAL REPOSITORY
    AND SKIP WHEN IT WAS CLEAN. That is the sixth instance of this project's
    "check that cannot discriminate" family: the backup ref was moved out of
    `refs/heads/` (ADR-070 (4.2)) -- a good structural fix -- and two tests
    silently lost their subject and reported SKIPPED, which in a summary line
    reads like a pass. A test whose ability to fail depends on the state of the
    repository it is defending is not defending it.

    So the subject is MANUFACTURED: `_template` builds `main` (pushed to a
    throwaway bare remote, so `refs/remotes/origin/main` exists and the derived
    publishable set is non-empty) plus `local-only`, an ORPHAN branch with a
    second root and therefore no merge base with `main` -- the structure of
    `pre-public-backup`, not an imitation of it. The structure is ASSERTED here,
    so a fixture that stopped carrying the hazard fails loudly instead of making
    every test that depends on it vacuous.
    """
    work = _template / "work"
    published = push_guard.published_refs(work, "origin")
    assert published == {"refs/heads/main"}, f"the fixture remote carries {sorted(published)}"
    unpublished = sorted(_local_branches(work) - {"main"})
    assert unpublished == ["local-only"], (
        f"the fixture no longer manufactures an unpublished branch (found {unpublished}), so "
        "every test that uses it would pass without adjudicating anything"
    )
    assert not push_guard.shares_history(work, "local-only", "main"), (
        "the manufactured branch shares history with main, so it is not the hazard: rule 2 "
        "would have nothing to catch"
    )
    return work


def test_a_MANUFACTURED_unpublished_branch_is_refused_whatever_the_real_repo_holds(
    hazard: Path,
) -> None:
    """`git push --all`, adjudicated against a repository built to carry the hazard.

    Its predecessor parametrised over the real `refs/heads/` and skipped when
    there was nothing to catch. This one cannot be disarmed by anything anyone
    does to this repository's refs.
    """
    published = push_guard.published_refs(hazard, "origin")
    for branch in sorted(_local_branches(hazard) - {"main"}):
        refusals = push_guard.adjudicate(
            [_push_line(hazard, branch, f"refs/heads/{branch}", ZERO)],
            published=published,
            repo=hazard,
            remote="origin",
        )
        assert [r.rule for r in refusals] == ["unpublished-ref"], refusals
        assert branch in str(refusals[0])

    # THE OTHER HALF OF THE CONTROL, in the same repository: a guard that
    # refuses everything would satisfy the loop above.
    assert (
        push_guard.adjudicate(
            [_push_line(hazard, "main", "refs/heads/main", ZERO)],
            published=published,
            repo=hazard,
            remote="origin",
        )
        == []
    )


def test_every_ref_in_THIS_checkout_is_adjudicated_and_only_the_published_one_is_allowed() -> None:
    """THE REAL REPOSITORY, SWEPT OVER EVERY NAMESPACE -- so the archived backup is covered.

    `git push --mirror` traverses more than `refs/heads/`, and that is where
    `refs/archive/pre-public-backup` lives: 52 commits with no merge base with
    `main`. This asserts BOTH directions on live refs -- the published one is
    allowed, everything else is refused -- so it discriminates today with the
    backup archived, and it would still discriminate if the backup came back to
    `refs/heads/` or a new branch appeared. It cannot skip.
    """
    published = push_guard.published_refs(repo_root(), "origin")
    assert published, "the derived publishable set is EMPTY, so nothing below proves anything"

    local = {ref for ref in _all_local_refs(repo_root()) if not ref.startswith("refs/remotes/")}
    assert local, "no local refs at all: the sweep is broken, not the repository clean"

    allowed, refused = [], []
    for ref in sorted(local):
        refusals = push_guard.adjudicate(
            [_push_line(repo_root(), ref, ref, ZERO)],
            published=published,
            repo=repo_root(),
            remote="origin",
        )
        (allowed if not refusals else refused).append(ref)
        if ref in published:
            assert not refusals, f"{ref} is published and was refused: {refusals}"
        else:
            assert [r.rule for r in refusals] == ["unpublished-ref"], (ref, refusals)

    # THE SWEEP MUST NOT BECOME A TEST OF WHETHER A HAZARD HAPPENS TO BE ON DISK.
    # Today it sees both kinds -- `refs/heads/main` allowed, the archived backup
    # refused -- and that is recorded in the message below so a change is
    # legible. But deleting the backup is a legitimate act, and this test failing
    # (or skipping) because of it would be the same defect wearing the other
    # face. The refusal direction is owned by the MANUFACTURED case above, which
    # no repository state can disarm; what this one must always assert is that
    # the guard does not simply refuse everything on live refs.
    assert allowed, (
        f"every local ref was refused ({refused}), including ones the remote already carries. "
        "A guard that refuses everything is one that gets uninstalled."
    )


def test_a_legitimate_push_of_the_published_branch_is_allowed() -> None:
    """THE OTHER HALF OF THE CONTROL. A guard that refuses everything is not a guard."""
    head = _git(["rev-parse", "refs/heads/main"], repo_root()).strip()
    assert (
        push_guard.adjudicate(
            [push_guard.PushRef("refs/heads/main", head, "refs/heads/main", head)],
            published=push_guard.published_refs(repo_root(), "origin"),
            repo=repo_root(),
            remote="origin",
        )
        == []
    )


def test_unrelated_history_cannot_be_smuggled_onto_the_published_ref(hazard: Path) -> None:
    """RULE 1 ADJUDICATES THE DESTINATION, SO ON ITS OWN IT IS DEFEATED BY A REFSPEC.

    `git push origin <local-only>:main` has an allowed destination and publishes
    exactly the content rule 1 exists to keep back. Rule 2 is what catches it,
    and it catches it by asking git, not by naming anything.

    THE SMUGGLED BRANCH IS MANUFACTURED. This test used to look for an orphan in
    the real repository and `pytest.skip("no local-only branch on disk to
    smuggle")` when it found none -- so the day the backup ref was archived, the
    test stopped adjudicating rule 2 at all and said so in a line that reads like
    a pass. The fixture builds the orphan, and asserts it is one.
    """
    others = sorted(_local_branches(hazard) - {"main"})
    unrelated = [b for b in others if not push_guard.shares_history(hazard, b, "main")]
    assert unrelated, (
        f"the manufactured repository holds {others}, none of which is an unrelated history, "
        "so there is nothing here for rule 2 to catch and the fixture is broken"
    )
    refusals = push_guard.adjudicate(
        [_push_line(hazard, unrelated[0], "refs/heads/main", ZERO)],
        published=push_guard.published_refs(hazard, "origin"),
        repo=hazard,
        remote="origin",
    )
    assert [r.rule for r in refusals] == ["unrelated-history"], refusals

    # CONTROL, SAME REPOSITORY, SAME DESTINATION: a related history IS allowed.
    # Rule 2 refusing everything would satisfy the assertion above, and an
    # amend-and-force-push has to stay possible -- it is how the attribution
    # trailer was removed from the public history in the first place.
    assert (
        push_guard.adjudicate(
            [_push_line(hazard, "main", "refs/heads/main", ZERO)],
            published=push_guard.published_refs(hazard, "origin"),
            repo=hazard,
            remote="origin",
        )
        == []
    )


# --------------------------------------------------------------------------
# Fail closed
# --------------------------------------------------------------------------


def test_an_empty_publishable_set_refuses_instead_of_allowing() -> None:
    refusals = push_guard.adjudicate(
        [push_guard.PushRef("refs/heads/x", "a" * 40, "refs/heads/x", ZERO)],
        published=(),
        repo=repo_root(),
        remote="nowhere",
    )
    assert [r.rule for r in refusals] == ["no-published-refs"], refusals


def test_a_sha_that_is_not_in_the_object_database_refuses_instead_of_allowing() -> None:
    """ "I could not check it" must not read the same as "I checked it and it was fine"."""
    refusals = push_guard.adjudicate(
        [push_guard.PushRef("refs/heads/main", "b" * 40, "refs/heads/main", ZERO)],
        published={"refs/heads/main"},
        repo=repo_root(),
        remote="origin",
    )
    assert [r.rule for r in refusals] == ["unreadable-object"], refusals


def test_a_malformed_ref_line_raises_rather_than_parsing_to_something_plausible() -> None:
    with pytest.raises(ValueError, match="cannot read this pre-push line"):
        push_guard.parse_push_refs("refs/heads/main deadbeef\n")


def test_the_parser_reads_gits_delete_form_and_refuses_a_line_with_extra_fields() -> None:
    """The delete line is git's, captured verbatim from a real `git push --delete`.

    The second half is the fail-closed direction. A ref name cannot contain a
    space -- `git check-ref-format "refs/heads/with space"` exits 1, checked
    rather than assumed -- so a five-field line is not a ref with a space in it,
    it is a line this parser does not understand. Understanding it wrongly would
    attach a sha to the wrong ref and produce a confident verdict about something
    else, which is worse than refusing.
    """
    refs = push_guard.parse_push_refs(
        f"(delete) {ZERO} refs/heads/gone {'a' * 40}\n"
        f"refs/heads/keep {'b' * 40} refs/heads/keep {ZERO}\n"
        "\n"
    )
    assert len(refs) == 2
    assert refs[0].is_delete and refs[0].remote_ref == "refs/heads/gone"
    assert refs[1].local_ref == "refs/heads/keep"
    assert refs[1].local_sha == "b" * 40
    assert not refs[1].is_delete

    with pytest.raises(ValueError, match="cannot read this pre-push line"):
        push_guard.parse_push_refs(f"refs/heads/a b {'c' * 40} refs/heads/a b {ZERO}\n")


def test_zero_refs_is_allowed_because_it_publishes_nothing() -> None:
    """MEASURED, NOT ASSUMED. An up-to-date `git push` really does invoke the pre-push
    hook with empty stdin, so refusing an empty ref list would be a false positive on a
    command that sends nothing. "Nothing to publish" is not "cannot determine"."""
    assert push_guard.parse_push_refs("") == []
    assert push_guard.adjudicate([], published=(), repo=repo_root(), remote="origin") == []


# --------------------------------------------------------------------------
# Wiring: it arrives with `make install`, and it is installed HERE
# --------------------------------------------------------------------------


def test_the_pre_push_hook_arrives_with_make_install_rather_than_by_instruction() -> None:
    config = yaml.safe_load((repo_root() / ".pre-commit-config.yaml").read_text())
    assert "pre-push" in config.get("default_install_hook_types", []), (
        "`pre-commit install` will not wire the pre-push hook type, so the guard would arrive "
        "as an instruction"
    )
    hooks = [
        h
        for r in config["repos"]
        if r.get("repo") == "local"
        for h in r["hooks"]
        if h["id"] == "push-guard"
    ]
    assert len(hooks) == 1, "the push-guard hook is not registered exactly once"
    assert hooks[0]["stages"] == ["pre-push"], hooks[0]
    assert hooks[0]["always_run"] is True, "a skipped guard is not a guard"

    script = repo_root() / "tools" / "push_guard.py"
    assert script.is_file() and script.stat().st_mode & 0o111, "the hook script is not executable"
    assert script.read_text().startswith("#!"), "the hook script has no shebang"

    makefile = (repo_root() / "Makefile").read_text()
    body = makefile.split("\nhooks:", 1)[1].split("\n\n", 1)[0]
    assert "push_guard.py --install" in body, (
        "`make hooks` no longer installs the AUTHORITATIVE layer, so only the partial "
        "pre-commit-stage layer would be wired"
    )


def test_the_guard_is_the_ONLY_hook_that_runs_on_push_and_nothing_rewrites_files_there() -> None:
    """FOUND BY RUNNING IT, NOT BY READING IT, AND IT WAS MY OWN DEFECT.

    A pre-commit hook with no `stages` key runs at EVERY installed stage. Adding
    `pre-push` to `default_install_hook_types` therefore enlisted the formatters
    into `git push`: in the first end-to-end run, one push executed `ruff-format`
    and REWROTE 88 tracked files, across directories this repo deliberately
    fences between four leads. A guard that edits another lead's module while
    their experiment is running is a worse defect than the one it prevents.
    Every hook now declares its stage; this test is what keeps that true, because
    the failure is silent and appears only on the next hook someone adds.
    """
    config = yaml.safe_load((repo_root() / ".pre-commit-config.yaml").read_text())
    installed = set(config.get("default_install_hook_types", []))
    on_push = [
        h["id"]
        for r in config["repos"]
        for h in r["hooks"]
        if "pre-push" in set(h.get("stages", sorted(installed)))
    ]
    assert on_push == ["push-guard"], (
        f"these hooks run on `git push`: {on_push}. Only the guard may. A hook with no "
        "`stages` key runs at every installed stage, so the fix is to give it one."
    )


def test_both_layers_are_actually_installed_in_this_checkout() -> None:
    """LAYER (c). The config can be perfect while the hooks directory is empty.

    THIS USED TO SKIP OUTSIDE A GIT CHECKOUT and it no longer does. The reasoning
    for the skip was sound -- an export with no `.git` correctly has no hooks --
    but the skip was reachable from REPOSITORY STATE, and this file has now been
    bitten twice by exactly that: a skip that reads like a pass in the summary
    line while the instrument sees nothing. The precedent is `test_hygiene.py`,
    which REFUSES on a shallow clone rather than reporting clean (ADR-069 (4)).
    An unverifiable checkout is not a green one, so it is red and says why.
    """
    dot_git = repo_root() / ".git"
    assert dot_git.exists(), (
        f"{dot_git} does not exist, so this suite is running outside a git checkout and the "
        "wiring of the publication guard CANNOT BE CHECKED HERE. That is not a pass. Run the "
        "suite from a checkout."
    )
    hooks = push_guard.hooks_dir(repo_root())
    chained = hooks / "pre-push"
    authoritative = hooks / "pre-push.legacy"
    assert os.access(chained, os.X_OK), f"{chained} is missing: nothing invokes the guard"
    assert os.access(authoritative, os.X_OK), (
        f"{authoritative} is missing, so only the PARTIAL layer would run. Run `make hooks`."
    )
    assert "push_guard.py" in authoritative.read_text()


# --------------------------------------------------------------------------
# END TO END: a real `git push` against a throwaway remote.
# Never against `origin` -- the point of the guard is that origin is precious.
# --------------------------------------------------------------------------


def _build_template(root: Path) -> Path:
    """A miniature of this repository's hazard: one published branch, one orphan branch."""
    bare = root / "remote.git"
    work = root / "work"
    _git(["init", "-q", "--bare", str(bare)], root)
    _git(["init", "-q", "-b", "main", str(work)], root)
    for key, value in (
        ("user.email", "nobody@example.invalid"),
        ("user.name", "nobody"),
        ("commit.gpgsign", "false"),
    ):
        _git(["config", key, value], work)

    (work / "published.txt").write_text("published\n")
    _git(["add", "published.txt"], work)
    _git(["commit", "-q", "-m", "published history"], work)

    # The orphan: a SECOND ROOT, so `git merge-base` finds nothing between them.
    # That is the structure of `pre-public-backup`, not an imitation of it.
    _git(["checkout", "-q", "--orphan", "local-only"], work)
    _git(["rm", "-rqf", "."], work)
    (work / "secret.txt").write_text("pre-public history\n")
    _git(["add", "secret.txt"], work)
    _git(["commit", "-q", "-m", "pre-public history"], work)
    _git(["checkout", "-q", "main"], work)

    # A RELATIVE remote url, so the whole tree can be copied per test.
    _git(["remote", "add", "origin", "../remote.git"], work)
    _git(["push", "-q", "origin", "main"], work)

    tools = work / "tools"
    tools.mkdir()
    shutil.copy2(repo_root() / "tools" / "push_guard.py", tools / "push_guard.py")

    # The hook definition under test is READ FROM THE REAL CONFIG, not retyped, so
    # this fixture cannot drift from what `make install` wires. Only the remote
    # `repos:` entries are dropped -- they are irrelevant at the pre-push stage
    # and would make this test depend on the network.
    real = yaml.safe_load((repo_root() / ".pre-commit-config.yaml").read_text())
    local = [
        {"repo": "local", "hooks": [h]}
        for r in real["repos"]
        if r.get("repo") == "local"
        for h in r["hooks"]
        if h["id"] == "push-guard"
    ]
    assert local, "the push-guard hook is not in .pre-commit-config.yaml"
    (work / ".pre-commit-config.yaml").write_text(
        yaml.safe_dump({"default_install_hook_types": ["pre-push"], "repos": local})
    )
    _git(["add", ".pre-commit-config.yaml", "tools/push_guard.py"], work)
    _git(["commit", "-q", "-m", "wiring"], work)
    return root


@pytest.fixture(scope="session")
def _template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_template(tmp_path_factory.mktemp("push-guard-template"))


@pytest.fixture
def clone(_template: Path, tmp_path: Path) -> Path:
    """A throwaway copy with the hooks installed exactly as `make hooks` installs them."""
    dest = tmp_path / "sandbox"
    shutil.copytree(_template, dest)
    work = dest / "work"
    installer = [sys.executable, str(work / "tools" / "push_guard.py"), "--install"]
    assert _run([sys.executable, "-m", "pre_commit", "install"], work).returncode == 0
    done = _run(installer, work)
    assert done.returncode == 0, done.stderr
    return work


def test_END_TO_END_a_push_of_a_non_published_branch_is_REFUSED(clone: Path) -> None:
    """THE PLANTED DEFECT: the exact command named as the hazard, run for real."""
    bare = clone.parent / "remote.git"
    assert _remote_branches(bare) == {"refs/heads/main"}, "the fixture remote is already dirty"

    done = _run(["git", "push", "--all", "origin"], clone)
    assert done.returncode != 0, f"the guard ALLOWED `git push --all`:\n{done.stdout}{done.stderr}"
    assert "unpublished-ref" in done.stderr, done.stderr
    assert "local-only" in done.stderr, done.stderr
    assert _remote_branches(bare) == {"refs/heads/main"}, "the branch was published anyway"


def test_END_TO_END_a_push_of_the_published_branch_still_SUCCEEDS(clone: Path) -> None:
    """A guard that blocks the work is a guard someone uninstalls."""
    bare = clone.parent / "remote.git"
    before = _git(["--git-dir", str(bare), "rev-parse", "refs/heads/main"], bare).strip()
    (clone / "published.txt").write_text("more published work\n")
    _git(["add", "published.txt"], clone)
    _git(["commit", "-q", "-m", "ordinary work"], clone)

    done = _run(["git", "push", "origin", "main"], clone)
    assert done.returncode == 0, f"the guard REFUSED a legitimate push:\n{done.stderr}"
    after = _git(["--git-dir", str(bare), "rev-parse", "refs/heads/main"], bare).strip()
    assert after != before, "the push reported success and published nothing"


def test_END_TO_END_POSITIVE_CONTROL_the_same_push_succeeds_with_the_guard_bypassed(
    clone: Path,
) -> None:
    """A PASS MUST BE DISTINGUISHABLE FROM A DEAD HOOK.

    Same repository, same command, same remote, guard bypassed: the branch IS
    published. So the refusal in the test above is the guard's doing and not a
    push that could never have worked -- which is the failure mode that let four
    public-tree scrubs report clean while they were wrong.
    """
    bare = clone.parent / "remote.git"
    done = _run(["git", "push", "--no-verify", "--all", "origin"], clone)
    assert done.returncode == 0, f"the control push failed for an unrelated reason:\n{done.stderr}"
    assert _remote_branches(bare) == {"refs/heads/main", "refs/heads/local-only"}, (
        "the control did not publish the branch, so the refusal above proves nothing"
    )


def _commit_and_push(clone: Path, text: str) -> subprocess.CompletedProcess[str]:
    (clone / "published.txt").write_text(text)
    _git(["add", "published.txt"], clone)
    _git(["commit", "-q", "-m", "ordinary work"], clone)
    return _run(["git", "push", "origin", "main"], clone)


def test_END_TO_END_deleting_the_authoritative_layer_STOPS_pushes_rather_than_unguarding_them(
    clone: Path,
) -> None:
    """FAIL CLOSED AT THE INSTALLATION LEVEL, which is where this guard is weakest.

    `pre-commit install -f --hook-type pre-push` deletes the slot the full-ref-list
    layer lives in. Without this check the remaining layer would still run, still
    report a verdict, and be blind to `git push --all` -- a green light from an
    instrument that cannot see its subject.

    THE FIRST PUSH IS NOT SCENERY, IT IS THE REGRESSION. The first version of this
    guard accepted any breadcrumb younger than a time window, and this test passed
    because the deletion happened before the clone had ever pushed, so there was no
    breadcrumb at all. A live run then ALLOWED a real second push seconds after a
    guarded one, because the earlier push's breadcrumb was still young. The fixture
    could not see the hole; the real push could. The breadcrumb now carries the
    parent process of the push it was written in, so an earlier push cannot vouch
    for a later one, and this test performs a successful push first to prove it.
    """
    first = _commit_and_push(clone, "an ordinary first push that IS guarded\n")
    assert first.returncode == 0, f"the guarded first push failed: {first.stdout}{first.stderr}"
    assert push_guard.read_breadcrumb(clone) is not None, (
        "the authoritative layer left no breadcrumb, so the second half of this test would "
        "pass for the wrong reason"
    )

    authoritative = push_guard.hooks_dir(clone) / "pre-push.legacy"
    assert authoritative.exists()
    authoritative.unlink()

    done = _commit_and_push(clone, "work that should not go out unchecked\n")
    assert done.returncode != 0, (
        f"with the authoritative layer deleted the push was ALLOWED:\n{done.stdout}{done.stderr}"
    )
    # pre-commit CAPTURES a hook's stderr and re-emits it on ITS OWN stdout, so
    # the refusal has to be looked for in both streams. Asserting on one of them
    # would be a test that passes, or fails, on the wrong evidence -- and this
    # one failed that way first, which is how the behaviour got measured.
    assert "authoritative-layer-did-not-run" in done.stdout + done.stderr, (
        f"stdout={done.stdout!r} stderr={done.stderr!r}"
    )


def test_a_breadcrumb_from_a_DIFFERENT_push_does_not_vouch_for_this_one(tmp_path: Path) -> None:
    """The unit form of the hole above, so it is pinned without spawning a push.

    Both layers are children of one `pre-commit hook-impl` process, so they observe
    the same parent pid -- measured with a probe in a throwaway clone, not assumed
    from the documentation. That identity is what ties a breadcrumb to ONE push;
    the age bound is only there to stop a recycled pid being believed forever.
    """
    _git(["init", "-q", "-b", "main", str(tmp_path)], tmp_path)
    crumb = tmp_path / ".git" / "push-guard-ran"

    assert not push_guard.authoritative_layer_ran(tmp_path, ppid=4242), "no breadcrumb must fail"

    crumb.write_text("1000.0 4242\n")
    assert push_guard.authoritative_layer_ran(tmp_path, ppid=4242, now=1000.5)
    assert not push_guard.authoritative_layer_ran(tmp_path, ppid=9999, now=1000.5), (
        "a breadcrumb written by a DIFFERENT push was accepted for this one"
    )
    assert not push_guard.authoritative_layer_ran(
        tmp_path, ppid=4242, now=1000.0 + push_guard.BREADCRUMB_MAX_AGE_S + 1
    ), "a stale breadcrumb was accepted"

    crumb.write_text("not a timestamp\n")
    assert not push_guard.authoritative_layer_ran(tmp_path, ppid=4242), (
        "an unreadable breadcrumb must read as ABSENT, never as present"
    )
