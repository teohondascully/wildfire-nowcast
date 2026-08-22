"""``common/runs.py`` - run directories, and the provenance stamped into them.

ADR-016 cost this project every pre-A10 run: ``git_sha`` wrote ``"unknown"`` into
twenty run directories and nothing noticed, so no number produced before that
could be attributed to a commit. The stamps are therefore treated as a product
here rather than as decoration, and each assertion names the way its field has
already failed once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wildfire_nowcast.common.runs import create_run_dir, git_sha, read_run


def test_the_short_sha_is_short_and_is_a_prefix_of_the_long_one(tmp_path: Path) -> None:
    """``git_sha_short`` that holds a 40-character sha is not a defect any reader sees.

    Both fields would still be present, both would still be truthful, and every
    downstream citation would still resolve. It would simply stop being the short
    form, which is what run ids, report tables and CI status lines quote.
    """
    if git_sha() is None:
        pytest.fail(
            "git reported no sha, so this environment cannot say whether provenance works. "
            "That is a failure to measure, not a pass: ADR-016 is exactly the case where "
            "the stamp was absent and the suite was green."
        )

    run = create_run_dir({"lr": 0.1}, run_id="probe", runs_root=tmp_path)
    meta = json.loads(run.meta_path.read_text())
    long_sha, short_sha = meta["git_sha"], meta["git_sha_short"]

    assert len(long_sha) == 40, long_sha
    assert 0 < len(short_sha) < len(long_sha), (
        f"git_sha_short={short_sha!r} is not shorter than git_sha={long_sha!r}"
    )
    assert long_sha.startswith(short_sha)


def test_a_run_directory_carries_the_resolved_config_and_reads_back(tmp_path: Path) -> None:
    """The config is written verbatim so a run reproduces without ``configs/``."""
    config = {"model": {"lr": 0.1}, "seed": 7}
    run = create_run_dir(config, run_id="r1", runs_root=tmp_path)

    assert run.path == tmp_path / "r1"
    assert run.config_path.is_file() and run.meta_path.is_file()

    payload, meta = read_run(run.path)
    assert payload == config, "the resolved config did not survive the round trip"
    assert meta["run_id"] == "r1"
    assert "split_fingerprint" in meta and "environment_before" in meta


def test_creating_a_run_under_a_root_that_does_not_exist_yet_works_and_repeats(
    tmp_path: Path,
) -> None:
    """Two independent properties of one ``mkdir`` call, and they fail differently.

    ``parents=True`` is what lets a caller point ``runs_root`` at a directory it
    has not created; ``exist_ok=True`` is what makes the explicit ``exist_ok``
    parameter above mean something, since without it the second call raises from
    ``mkdir`` after the guard has already decided to allow it.
    """
    nested = tmp_path / "not" / "created" / "yet"
    first = create_run_dir({"a": 1}, run_id="same", runs_root=nested)
    assert first.path.is_dir()

    again = create_run_dir({"a": 2}, run_id="same", runs_root=nested, exist_ok=True)
    assert again.path == first.path
    payload, _ = read_run(again.path)
    assert payload == {"a": 2}, "the second write did not land"

    with pytest.raises(FileExistsError):
        create_run_dir({"a": 3}, run_id="same", runs_root=nested)
