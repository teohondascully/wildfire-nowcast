"""The 500 KB commit guard, and the one path that is named out of it.

WHY THIS FILE EXISTS
--------------------
``check-added-large-files`` is configured at ``--maxkb=500`` because the accident
this repository is exposed to is a stray ``git add data/`` of a ~700 MB corpus.
Tracking ``runs/m19_collapse_curve.json`` (820,811 bytes) required getting one
file past that bar. There were two ways to do it and they are not equivalent:
raise ``--maxkb`` for everything, or name the single path. The second was taken,
and this module is what makes that choice checkable rather than a claim in a
comment.

WHAT IT VERIFIES, AND WHAT IT DOES NOT
--------------------------------------
It verifies the CONFIGURATION: the bar is still 500 KB, the exclusion is one
anchored path, and no other tracked file is over the bar. It does NOT execute
the upstream hook, which installs its own environment from the network and has
no place in a unit test. So this asserts the rule as configured, not the
upstream implementation of it, and the end-to-end observation (a 600 KB file
staged in a fresh clone is refused, the named file is accepted) was made by hand
and recorded with the change. That gap is stated because a mirrored rule can
drift from the tool it mirrors.

The exemption carries a MEASUREMENT, not just a reason: the named file must
still be tracked and still be over the bar. If it ever falls under 500 KB the
exemption has lost its reason and this file says so, which is the property that
stops an exemption outliving the thing it was written for.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from wildfire_nowcast.common.paths import repo_root

#: The bar, in kilobytes, as configured. Read back out of the file below rather
#: than trusted from here; this is the value the assertion compares against.
EXPECTED_MAXKB = 500

#: A specimen path that does not exist, JOINED AT RUNTIME for the reason
#: ``tests/test_cited_runs.py`` gives for its own: this file is read by that
#: scan, and written literally the string below would BE a citation of a
#: ``runs/`` artifact the repository does not contain. It was written literally
#: in the first draft and the scan caught it in a fresh clone, on the control
#: run, before any plant. Exempting this path would have made the file that
#: fabricates paths the one file that can never report one.
_SPECIMEN_UNTRACKED = "ru" + "ns/some_new_sweep" + ".json"

#: The one path named out of the bar, and why the number matters: 820,811 bytes.
EXEMPT_PATH = "runs/m19_collapse_curve.json"


def _config() -> dict[str, object]:
    text = (repo_root() / ".pre-commit-config.yaml").read_text()
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def _large_file_hook() -> dict[str, object]:
    """The one hook entry, found by id rather than by position."""
    repos = _config()["repos"]
    assert isinstance(repos, list)
    for repo in repos:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "check-added-large-files":
                assert isinstance(hook, dict)
                return hook
    raise AssertionError("check-added-large-files is not configured at all")


def hook_would_refuse(rel: str, size_bytes: int, *, maxkb: int, exclude: str | None) -> bool:
    """The upstream rule, restated: over the bar AND not matched by ``exclude``.

    Pure, so the tests below can put a size in front of it instead of writing a
    600 KB file to disk to learn what the configuration does.
    """
    if exclude is not None and re.search(exclude, rel):
        return False
    return size_bytes > maxkb * 1024


def test_the_bar_is_still_five_hundred_kilobytes() -> None:
    hook = _large_file_hook()
    assert hook.get("args") == [f"--maxkb={EXPECTED_MAXKB}"], hook.get("args")


def test_the_exclusion_is_one_anchored_path_and_not_a_prefix() -> None:
    """A negation at the wrong depth fails silently; so does an unanchored one."""
    exclude = _large_file_hook().get("exclude")
    assert isinstance(exclude, str), "the exclusion was removed or is not a string"
    pattern = re.compile(exclude)

    assert pattern.search(EXEMPT_PATH), exclude
    for other in (
        "runs/m19_documented_control.json",
        "runs/m19_collapse_curve.json.bak",
        "vendored/runs/m19_collapse_curve.json",
        "data/fires/2019_kincade/tensor.zarr",
        "runs/m19_collapse_curve.jsonx",
    ):
        assert not pattern.search(other), f"the exclusion also covers {other}: {exclude}"


def test_the_named_file_is_tracked_and_is_still_over_the_bar() -> None:
    """The exemption's reason, measured. If the file shrinks, the reason is gone."""
    path = repo_root() / EXEMPT_PATH
    assert path.is_file(), f"{EXEMPT_PATH} is named out of the guard and is not on disk"
    size = path.stat().st_size
    assert size > EXPECTED_MAXKB * 1024, (
        f"{EXEMPT_PATH} is {size} bytes, under the {EXPECTED_MAXKB} KB bar. "
        "It no longer needs to be named out of the guard: delete the exclusion."
    )


#: EVERY tracked file over the 500 KB bar, with its exact size.
#:
#: THE HOOK CANNOT SEE THIS LIST, AND THAT IS THE POINT.
#: ``check-added-large-files`` inspects the files being ADDED in the commit in
#: front of it, so anything already in the index is permanently outside its
#: reach. Two of the three below were tracked before this guard was written and
#: have never been examined by it: nothing in the repository knew they were over
#: the bar until this assertion was written and failed. The third is the one
#: named in ``.pre-commit-config.yaml``.
#:
#: Sizes are pinned exactly, so an entry cannot go stale quietly, and a NEW file
#: over the bar fails here even though the hook that is supposed to stop it may
#: have been skipped for that commit.
OVER_THE_BAR: dict[str, int] = {
    # The S1 fold-rotation rows: one row per window, cited by tracked source.
    "runs/s1_rows_a_s1.json": 835640,
    # U0b's per-block elasticity record, cited by tracked source.
    "runs/u0b.json": 3283305,
    # M19's collapse curve. The only one of the three the hook has ever seen,
    # and the only one named in `.pre-commit-config.yaml`.
    EXEMPT_PATH: 820811,
}


def test_the_files_over_the_bar_are_exactly_the_declared_ones() -> None:
    """The population, both directions: a new one fails, a stale entry fails."""
    base = repo_root()
    measured = {}
    for rel in _tracked_files():
        path = base / rel
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > EXPECTED_MAXKB * 1024:
            measured[rel] = size
    assert measured == OVER_THE_BAR, (
        "the set of tracked files over the commit-size bar has moved. Measured "
        f"{measured}, declared {OVER_THE_BAR}. A new entry is a file the guard "
        "either did not see or was skipped for; a missing one is a stale entry."
    )


def test_the_rule_answers_both_ways_on_a_size_it_is_given() -> None:
    """CAPABILITY, named input, independent of what the tree happens to hold.

    A population check ("nothing is over the bar") passes just as well when the
    predicate has stopped working, so the predicate is put in front of a size it
    must refuse and a size it must accept.
    """
    exclude = _large_file_hook().get("exclude")
    assert isinstance(exclude, str)
    six_hundred_kb = 600 * 1024

    assert hook_would_refuse(_SPECIMEN_UNTRACKED, six_hundred_kb, maxkb=500, exclude=exclude)
    assert not hook_would_refuse(_SPECIMEN_UNTRACKED, 400 * 1024, maxkb=500, exclude=exclude)
    assert not hook_would_refuse(EXEMPT_PATH, six_hundred_kb, maxkb=500, exclude=exclude)
    # ...and with no exclusion configured at all, the named path is refused too,
    # so the acceptance above is attributable to the exclusion and to nothing else.
    assert hook_would_refuse(EXEMPT_PATH, six_hundred_kb, maxkb=500, exclude=None)


def _tracked_files() -> list[str]:
    import subprocess

    out = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(p for p in out.stdout.splitlines() if p)


def test_the_corpus_this_walks_is_not_empty() -> None:
    """Anti-vacuity for the population check above: an empty list passes it silently."""
    files = _tracked_files()
    assert len(files) > 100, f"only {len(files)} tracked files"
    assert EXEMPT_PATH in files, "the exempted artifact is not tracked, so nothing was gained"
    assert Path(__file__).name == "test_large_file_guard.py"
