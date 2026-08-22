"""The C3.3 reporting gate and the C8 split check, asserted where they DECIDE.

Three mutation survivors close here. None of them moves a number: each one flips
a verdict, which is the class of defect that reaches a document intact because
every figure downstream still renders.

* ``reporting.py:84`` ``False -> True``. A norm-stats file that is not valid JSON
  is published as REPORTABLE. The single input on which the guard has nothing to
  read is the input on which it would have said yes.
* ``reporting.py:128`` ``not -> ``. :func:`assert_reportable` raises on a
  reportable file and returns quietly on an unreportable one, i.e. exactly
  backwards. A test that only checks "it raises on a bad file" PASSES the
  inversion, so both directions are asserted here.
* ``reporting.py:308`` ``or -> and``. The ``training.json`` fallback stops
  resolving and a correctly stamped run is rejected as unstamped. The provenance
  path returns first and hides it, so the test drives the FALLBACK specifically.

The pattern each test follows is the one this project keeps relearning: assert
the guard's answer on BOTH sides of its own boundary, because a guard that always
says the same thing agrees with a correct guard half the time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wildfire_nowcast.eval.reporting import (
    MIN_TRAIN_BLOCKS_FOR_REPORTING,
    NotReportableError,
    SplitChangedError,
    assert_model_split_matches,
    assert_reportable,
    reporting_status,
)


def _stats_file(tmp_path: Path, blocks: int) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"n_train_blocks": blocks, "train_folds": [0, 1]}))
    return path


class _Model:
    """A checkpoint stub carrying only what the C8 resolver is allowed to read."""

    def __init__(self, provenance: dict[str, Any]) -> None:
        self.provenance = provenance


def test_a_norm_stats_file_that_is_not_valid_json_is_never_reportable(tmp_path: Path) -> None:
    """The unreadable file is the one the guard must be loudest about.

    ``reporting_status`` promises never to raise on content, so the only way the
    JSON branch can report a defect is through ``reportable``. If that key is
    True there, a truncated write during a build publishes as a result.
    """
    path = tmp_path / "norm_stats.json"
    path.write_text("{not json at all")

    status = reporting_status(path)

    assert status["exists"] is True, "the file is on disk; the defect is its contents"
    assert status["reportable"] is False, (
        "a norm-stats file that does not parse was reported as REPORTABLE. The guard "
        "cannot have read n_train_blocks, so this is a yes with nothing behind it."
    )
    assert "not valid JSON" in status["reason"]
    with pytest.raises(NotReportableError):
        assert_reportable(path)


def test_assert_reportable_refuses_the_unreportable_and_returns_on_the_reportable(
    tmp_path: Path,
) -> None:
    """Both directions, because either one alone passes the inverted guard.

    With the ``not`` dropped from the check, the refusal moves onto the healthy
    file and the weak file sails through. A one-sided test sees only half of that
    and reports green.
    """
    weak = _stats_file(tmp_path / "weak", MIN_TRAIN_BLOCKS_FOR_REPORTING - 1)
    strong = _stats_file(tmp_path / "strong", MIN_TRAIN_BLOCKS_FOR_REPORTING)

    with pytest.raises(NotReportableError) as raised:
        assert_reportable(weak)
    assert "REFUSING to report" in str(raised.value)

    status = assert_reportable(strong)
    assert status["reportable"] is True, (
        "a file satisfying C3.3 was refused. The guard is inverted: it now blocks "
        "exactly the runs it exists to authorise."
    )
    assert status["n_train_blocks"] == MIN_TRAIN_BLOCKS_FOR_REPORTING


def test_the_split_fingerprint_is_recovered_from_training_json_when_the_spec_lacks_it(
    tmp_path: Path,
) -> None:
    """Drive resolution step 2, which step 1 hides whenever it succeeds.

    Runs predating the provenance stamp recorded their split in
    ``training.json``. Reading it there is still reading it off disk, and C8
    accepts it. If that lookup silently returns nothing, every such run is
    rejected as unstamped and the operator's only route is to re-pin something.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training.json").write_text(json.dumps({"split_before": {"fingerprint": "abc123"}}))
    model = _Model({"run_dir": str(run_dir)})

    matched = assert_model_split_matches(model, {"fingerprint": "abc123"}, name="fallback")

    assert matched["match"] is True
    assert matched["training_split_fingerprint"] == "abc123"
    assert matched["source"].endswith("training.json"), (
        "the fingerprint was not resolved from training.json, so step 2 of the C8 "
        "resolution order is not doing anything"
    )


def test_a_model_with_no_recorded_split_anywhere_is_still_a_hard_fail(tmp_path: Path) -> None:
    """The control for the test above: C8 must still refuse the genuinely unstamped.

    Without this, the previous test is satisfied by a resolver that accepts
    everything, which is the failure mode C-1 names: unverifiable treated as a
    pass.
    """
    empty_dir = tmp_path / "unstamped"
    empty_dir.mkdir()
    model = _Model({"run_dir": str(empty_dir)})

    with pytest.raises(SplitChangedError) as raised:
        assert_model_split_matches(model, {"fingerprint": "abc123"}, name="unstamped")
    assert "carries NO split_fingerprint" in str(raised.value)
