"""C8 split fingerprint + C3.1 cross-fire fold clauses.

These are the clauses **no per-tensor check can ever see**. In ADR-015 the CV
split moved mid-task, four fires crossed from TRAIN to HELD-OUT under a running
training job, and every tensor was individually conformant the entire time. The
tests below therefore all construct a *valid* pair of artifacts and then move
the split between them: if a test can be made to pass by fixing a tensor, it is
testing the wrong thing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wildfire_nowcast.common import splits as S
from wildfire_nowcast.common.paths import fires_dir, norm_stats_path

# --------------------------------------------------------------------------
# a self-contained fake split on disk
# --------------------------------------------------------------------------


def _fire(root: Path, fire_id: str, fold: int, block: int, hours: int = 24) -> None:
    d = root / fire_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "fire_id": fire_id,
                "cv_fold": fold,
                "spatial_block_id": block,
                "n_hours": hours,
            }
        )
    )


def _stats(
    path: Path,
    *,
    train_folds: list[int],
    train: list[str] | None,
    heldout: list[str] | None,
    n_train_blocks: int = 2,
) -> None:
    """Write a ``norm_stats.json``. ``None`` OMITS the key - that is a defect to plant."""
    payload: dict[str, Any] = {"train_folds": train_folds, "n_train_blocks": n_train_blocks}
    if train is not None:
        payload["train_fire_ids"] = train
    if heldout is not None:
        payload["heldout_fire_ids"] = heldout
    path.write_text(json.dumps(payload))


@pytest.fixture
def fake_split(tmp_path: Path) -> tuple[Path, Path]:
    """``(fires_root, stats_path)`` for a 4-fire, 4-block, 2-train-fold split."""
    fires = tmp_path / "fires"
    _fire(fires, "a_fire", fold=0, block=0)
    _fire(fires, "b_fire", fold=1, block=1)
    _fire(fires, "c_fire", fold=2, block=2)
    _fire(fires, "d_fire", fold=3, block=3)
    stats = tmp_path / "norm_stats.json"
    _stats(
        stats,
        train_folds=[0, 1],
        train=["a_fire", "b_fire"],
        heldout=["c_fire", "d_fire"],
    )
    return fires, stats


def _fp(fake: tuple[Path, Path]) -> dict[str, Any]:
    return S.split_fingerprint(fires_root=fake[0], stats_path=fake[1])


# --------------------------------------------------------------------------
# C0 - one implementation, and it must reproduce the fingerprint of record
# --------------------------------------------------------------------------


@pytest.mark.skipif(not fires_dir().is_dir(), reason="no built fires on this machine")
def test_common_and_eval_fingerprints_agree_byte_for_byte() -> None:
    """C0: modelling's ``eval/`` copy and this one must never disagree.

    Re-homing a function under C0 is only safe if it is the SAME function. If
    this ever fails, one of the two changed and every artifact stamped by the
    other is unverifiable - which is worse than the duplication itself.
    """
    from wildfire_nowcast.eval.reporting import split_fingerprint as eval_fp

    ours = S.split_fingerprint()
    theirs = eval_fp()
    assert ours["fingerprint"] == theirs["fingerprint"]
    for key in ("train_fire_ids", "heldout_fire_ids", "train_folds", "n_heldout_blocks"):
        assert ours[key] == theirs[key], key


#: **THE C8 ARCHIVE BOUNDARY, kept as two named constants rather than one
#: overwritten literal** (ADR-039 (3), adopting data's suggestion verbatim).
#:
#: ``FINGERPRINT_PRE_D6`` - 12 fires, ``train_folds [0, 1, 2, 4]``, 4 held-out
#: blocks {3,4,5,6}. **Everything produced before the D6 corpus swap is bound to
#: this value and stays bound to it:** G2's record of adjudication
#: (``runs/baselines-20260808-095003``, ADR-021) and ALL FOUR G3 attempts
#: (ADR-027 / ADR-032 / ADR-034 / ADR-037, i.e. M5, M6, M7 and M8, including the
#: `m8_asym` candidate and the 2x2 factorial). No number stamped with it may be
#: quoted beside a number stamped with the one below - that is what C8's
#: ``matches_current`` reporting clause is for.
FINGERPRINT_PRE_D6 = "4848f491e8d588fa"

#: ``FINGERPRINT_OF_RECORD`` - the CURRENT split. 21 fires (ADR-037 (7) authorised
#: the swap, ADR-038 verified it), 14 spatial blocks, ``train_folds [0, 1, 2, 4]``
#: = 16 fires / 9 blocks, held out fold 3 = 5 fires / 5 blocks {4,5,6,7,12}.
#: **Nothing has been scored under it yet** - M9 will be the first.
FINGERPRINT_OF_RECORD = "b3e5dadad01eaef9"


@pytest.mark.skipif(
    not (fires_dir().is_dir() and norm_stats_path().is_file()),
    reason="no built fires on this machine",
)
def test_the_fingerprint_of_record_is_reproduced() -> None:
    """The split on disk must reproduce :data:`FINGERPRINT_OF_RECORD`.

    Pinned deliberately. If the split legitimately moves this test fails and the
    number in ADR-015/STATE.md must be updated in the same commit - a fingerprint
    nobody notices changing is not a fingerprint.

    **This test has now fired once in the correct direction and been updated
    deliberately** (A14, ADR-038 (9) / ADR-039 (3)). It is NOT auto-healed: the
    superseded value stays in the file as :data:`FINGERPRINT_PRE_D6`, because the
    thing C8 protects is not "the current split is X" but "the boundary between
    the two is legible". A boundary that lives only in a diff is a boundary
    nobody can read at the moment they are about to quote a stale number.
    """
    fp = S.split_fingerprint()
    assert fp["fingerprint"] == FINGERPRINT_OF_RECORD, (
        f"the CV split has moved: now {fp['fingerprint']} with train folds {fp['train_folds']} "
        f"and {fp['n_fires']} fires, against the pinned {FINGERPRINT_OF_RECORD} (21 fires, "
        "ADR-038). Update ADR-015/STATE.md in the same change, re-run anything being quoted, "
        "and KEEP the superseded value as a named constant here — do not overwrite it."
    )


def test_the_two_fingerprints_of_record_are_distinct_and_both_still_named() -> None:
    """The archive boundary is only useful while BOTH sides of it have a name.

    Trivial-looking and deliberately present: the cheapest way for this boundary
    to be lost is for a future update to overwrite ``FINGERPRINT_OF_RECORD`` and
    delete ``FINGERPRINT_PRE_D6`` as "dead", at which point every archived G2/G3
    number stops being attributable to a split anyone can name.
    """
    assert FINGERPRINT_PRE_D6 != FINGERPRINT_OF_RECORD
    assert len(FINGERPRINT_PRE_D6) == len(FINGERPRINT_OF_RECORD) == 16


def test_a_pre_d6_result_quoted_beside_a_current_one_is_a_reporting_gate() -> None:
    """C8's ``matches_current``, exercised across the REAL archive boundary.

    The two constants are put through the checker rather than merely asserted to
    differ: an artifact stamped ``FINGERPRINT_PRE_D6`` is internally consistent
    (it WAS correct when produced, so ``ok``) and must nonetheless be blocked
    from reporting beside the current split. This is the executable form of "all
    prior results stay bound to 4848f491e8d588fa".
    """
    rep = S.check_run_split(
        {
            "split_before": {"fingerprint": FINGERPRINT_PRE_D6},
            "split_after": {"fingerprint": FINGERPRINT_PRE_D6},
        },
        current={"fingerprint": FINGERPRINT_OF_RECORD},
    )
    assert rep.ok, "an archived G2/G3 result is not corrupt, it is superseded"
    assert not rep.reporting_ok
    assert "matches_current" in {c.check_id for c in rep.reporting_gaps}


def test_fingerprint_is_deterministic_and_order_independent(fake_split) -> None:  # noqa: ANN001
    assert _fp(fake_split)["fingerprint"] == _fp(fake_split)["fingerprint"]


def test_fingerprint_changes_when_a_fire_changes_fold(fake_split) -> None:  # noqa: ANN001
    """The ADR-015 event, in miniature: one fire crosses train -> held-out."""
    fires, stats = fake_split
    before = _fp(fake_split)
    _fire(fires, "a_fire", fold=3, block=0)  # was train (fold 0), now held out
    after = _fp(fake_split)
    assert before["fingerprint"] != after["fingerprint"]
    assert "a_fire" in before["train_fire_ids"]
    assert "a_fire" in after["heldout_fire_ids"]


def test_fingerprint_changes_when_a_fire_is_added(fake_split) -> None:  # noqa: ANN001
    before = _fp(fake_split)
    _fire(fake_split[0], "e_fire", fold=1, block=4)
    assert _fp(fake_split)["fingerprint"] != before["fingerprint"]


def test_fingerprint_changes_when_train_folds_change(fake_split) -> None:  # noqa: ANN001
    before = _fp(fake_split)
    _stats(
        fake_split[1],
        train_folds=[0, 1, 2],
        train=["a_fire", "b_fire", "c_fire"],
        heldout=["d_fire"],
        n_train_blocks=3,
    )
    assert _fp(fake_split)["fingerprint"] != before["fingerprint"]


def test_missing_data_yields_a_fingerprint_not_an_exception(tmp_path: Path) -> None:
    """It is stamped from ``create_run_dir``; it must never kill a training run."""
    fp = S.split_fingerprint(fires_root=tmp_path / "nope", stats_path=tmp_path / "nope.json")
    assert fp["n_fires"] == 0
    assert isinstance(fp["fingerprint"], str)


def test_an_unreadable_manifest_is_recorded_not_silently_dropped(fake_split) -> None:  # noqa: ANN001
    fires, _ = fake_split
    (fires / "b_fire" / "manifest.json").write_text("{not json")
    fp = _fp(fake_split)
    assert fp["unreadable_fires"] == ["b_fire"]
    rep = S.check_split_assignment(fires_root=fires, stats_path=fake_split[1])
    assert "manifests_readable" in {c.check_id for c in rep.failures}


def test_assert_split_unchanged_raises_when_the_split_moves(fake_split) -> None:  # noqa: ANN001
    fires, stats = fake_split
    before = _fp(fake_split)
    S.assert_split_unchanged(before, fires_root=fires, stats_path=stats)  # no move: fine
    _fire(fires, "c_fire", fold=0, block=2)
    with pytest.raises(S.SplitChangedError, match="SPLIT CHANGED"):
        S.assert_split_unchanged(before, fires_root=fires, stats_path=stats)


# --------------------------------------------------------------------------
# C8 - the hard fail
# --------------------------------------------------------------------------


def test_a_train_eval_mismatch_is_a_hard_fail() -> None:
    """The contract's own sentence, executable: *a mismatch between the split
    used for TRAINING and the split used for EVALUATION is a HARD FAIL.*"""
    rep = S.check_run_split(
        {"split_before": {"fingerprint": "aaaa"}, "split_after": {"fingerprint": "bbbb"}},
        current={"fingerprint": "aaaa"},
    )
    failed = {c.check_id for c in rep.failures}
    assert "internally_consistent" in failed
    assert not rep.ok
    assert "HARD FAIL" in rep.format()


def test_matching_stamps_pass() -> None:
    rep = S.check_run_split(
        {"scope": {"split_fingerprint": "aaaa"}, "split_after": {"fingerprint": "aaaa"}},
        current={"fingerprint": "aaaa"},
    )
    assert rep.reporting_ok, rep.format(verbose=True)


def test_an_unstamped_run_fails_because_it_cannot_be_verified() -> None:
    rep = S.check_run_split({"kind": "some_result", "brier_1h": 0.1}, current={"fingerprint": "a"})
    assert "stamped" in {c.check_id for c in rep.failures}


def test_a_stale_fingerprint_is_a_reporting_gate_not_a_failure() -> None:
    """An archived result was internally consistent when produced; quoting it
    beside a current number is what must be blocked (C-1's two tiers)."""
    rep = S.check_run_split(
        {"split_before": {"fingerprint": "old0"}, "split_after": {"fingerprint": "old0"}},
        current={"fingerprint": "new1"},
    )
    assert rep.ok, "a stale split does not make the artifact unusable"
    assert not rep.reporting_ok
    assert "matches_current" in {c.check_id for c in rep.reporting_gaps}


def test_fingerprints_are_found_wherever_a_lead_stamped_them() -> None:
    found = S.fingerprints_in(
        {
            "scope": {"split_fingerprint": "a"},
            "split_before": {"fingerprint": "a"},
            "runs": [{"split": {"fingerprint": "a"}}],
        }
    )
    assert set(found.values()) == {"a"}
    assert len(found) == 3


# --------------------------------------------------------------------------
# [v2.11] C-4.2 - a code fingerprint must be sampled BEFORE *and* AFTER
# --------------------------------------------------------------------------


def test_a_code_fingerprint_is_NOT_a_split_fingerprint() -> None:
    """The FALSE POSITIVE this clause exposed, pinned so it cannot come back.

    ``common_code_before``/``_after`` and ``scoring_code`` are each a dict with
    their own ``fingerprint`` key. Collecting those as SPLIT stamps made
    ``internally_consistent`` report "MORE THAN ONE split fingerprint" and HARD
    FAIL on every run in the repo - including the G2 record of ADR-021, whose
    stamps were one split fingerprint agreeing in four places plus two code
    fingerprints doing their job.

    A hard clause that fires on every artifact is worse than no clause: the first
    thing anyone does with it is stop reading it. Two quantities sharing one key
    name in different blocks is not something a syntactic scan can resolve on its
    own, so the scan is told which blocks are not about splits.
    """
    payload = {
        "split_before": {"fingerprint": "aaaa"},
        "split_after": {"fingerprint": "aaaa"},
        "common_code_before": {"fingerprint": "code1"},
        "common_code_after": {"fingerprint": "code1"},
        "scoring_code": {"fingerprint": "code2"},
    }
    found = S.fingerprints_in(payload)
    assert set(found.values()) == {"aaaa"}, found

    rep = S.check_run_split(payload, current={"fingerprint": "aaaa"})
    assert "internally_consistent" not in {c.check_id for c in rep.failures}
    assert rep.ok, rep.format(verbose=True)


def test_code_that_moved_during_the_run_is_a_HARD_fail() -> None:
    """C-4.2's own sentence: the numbers are partly before and partly after."""
    rep = S.check_run_split(
        {
            "split_before": {"fingerprint": "aaaa"},
            "split_after": {"fingerprint": "aaaa"},
            "scoring_code_before": {"fingerprint": "old"},
            "scoring_code_after": {"fingerprint": "new"},
        },
        current={"fingerprint": "aaaa"},
    )
    failed = {c.check_id for c in rep.failures}
    assert "code_agrees_across_run" in failed
    assert not rep.ok
    assert "MOVED DURING THIS RUN" in rep.format()


def test_a_one_ended_code_fingerprint_is_a_reporting_gap() -> None:
    """Sampling only at payload construction records the code as it stands AFTER.

    Reporting rather than hard, deliberately and on the record: every artifact in
    ``runs/`` predates C-4.2, and the two that stamp ``scoring_code`` stamp it
    once. Hard-failing today would void the G2 record on a bookkeeping property
    rather than on a measurement, and whether that record must be re-run is an
    maintainer ruling. Promote once a run stamps both ends.
    """
    rep = S.check_run_split(
        {
            "split_before": {"fingerprint": "aaaa"},
            "split_after": {"fingerprint": "aaaa"},
            "scoring_code": {"fingerprint": "only-one-end"},
        },
        current={"fingerprint": "aaaa"},
    )
    assert rep.ok, "one-ended is a gap, not a defect in the numbers"
    assert not rep.reporting_ok
    assert "code_sampled_both_ends" in {c.check_id for c in rep.reporting_gaps}


def test_both_ends_agreeing_passes_and_says_so() -> None:
    rep = S.check_run_split(
        {
            "split_before": {"fingerprint": "aaaa"},
            "split_after": {"fingerprint": "aaaa"},
            "common_code_before": {"fingerprint": "c"},
            "common_code_after": {"fingerprint": "c"},
            "scoring_code_before": {"fingerprint": "s"},
            "scoring_code_after": {"fingerprint": "s"},
        },
        current={"fingerprint": "aaaa"},
    )
    assert rep.reporting_ok, rep.format(verbose=True)
    assert "code_agrees_across_run" in {c.check_id for c in rep.checks}
    assert "code_sampled_both_ends" in {c.check_id for c in rep.checks}


def test_code_fingerprint_ends_reads_every_shape_and_omits_what_is_absent() -> None:
    ends = S.code_fingerprint_ends(
        {
            "common_code_before": {"fingerprint": "a"},
            "common_code_after": {"fingerprint": "b"},
            "scoring_code": "bare-string-is-also-a-fingerprint",
        }
    )
    assert ends["common_code"] == {"before": "a", "after": "b"}
    assert ends["scoring_code"] == {"unpaired": "bare-string-is-also-a-fingerprint"}
    assert S.code_fingerprint_ends({"split_before": {"fingerprint": "x"}}) == {}


def test_an_artifact_with_no_code_fingerprint_emits_no_c4_2_clause() -> None:
    """C-4.2 has nothing to say about a run predating the mechanism entirely.

    Emitting a passing clause there would be the absent-clause-reads-as-passing
    failure: 'sampled at both ends' would print green for an artifact that
    sampled neither end.
    """
    rep = S.check_run_split(
        {"split_before": {"fingerprint": "aaaa"}, "split_after": {"fingerprint": "aaaa"}},
        current={"fingerprint": "aaaa"},
    )
    ids = {c.check_id for c in rep.checks}
    assert "code_agrees_across_run" not in ids
    assert "code_sampled_both_ends" not in ids


def test_check_split_chain_fails_across_two_artifacts(tmp_path: Path) -> None:
    """A checkpoint trained under split A, evaluated under split B - ADR-015 exactly."""
    train = tmp_path / "train.json"
    ev = tmp_path / "eval.json"
    train.write_text(json.dumps({"split_before": {"fingerprint": "aaaa"}}))
    ev.write_text(json.dumps({"split_before": {"fingerprint": "bbbb"}}))
    rep = S.check_split_chain(train, ev)
    assert "train_eval_match" in {c.check_id for c in rep.failures}
    assert not rep.ok

    ev.write_text(json.dumps({"split_before": {"fingerprint": "aaaa"}}))
    assert S.check_split_chain(train, ev).ok


def test_an_unstamped_side_of_a_chain_is_unverifiable_not_ok(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    ev = tmp_path / "eval.json"
    train.write_text(json.dumps({"split_before": {"fingerprint": "aaaa"}}))
    ev.write_text(json.dumps({"brier_1h": 0.1}))
    rep = S.check_split_chain(train, ev)
    assert "stamped_eval" in {c.check_id for c in rep.failures}


def test_the_chain_follows_run_references(tmp_path: Path) -> None:
    """An eval artifact naming ``runs/kernel-x`` inherits that run's stamp."""
    runs = tmp_path / "runs"
    (runs / "kernel-x").mkdir(parents=True)
    (runs / "kernel-x" / "results.json").write_text(
        json.dumps({"split_before": {"fingerprint": "old0"}})
    )
    payload = {
        "split_before": {"fingerprint": "new1"},
        "argv": ["--kernel", "kernel=runs/kernel-x"],
    }
    rep = S.check_run_split(
        payload, current={"fingerprint": "new1"}, runs_root=runs, follow_references=True
    )
    assert "chain" in {c.check_id for c in rep.failures}, rep.format(verbose=True)


# --------------------------------------------------------------------------
# C3.1 - overlapping fires MUST share a fold
# --------------------------------------------------------------------------


def test_a_block_straddling_two_folds_is_a_hard_fail(fake_split) -> None:  # noqa: ANN001
    """Landscape leakage that BOTH manifests pass individually."""
    fires, stats = fake_split
    _fire(fires, "e_fire", fold=3, block=0)  # block 0 is already fold 0
    rep = S.check_split_assignment(fires_root=fires, stats_path=stats)
    failed = {c.check_id for c in rep.failures}
    assert "block_maps_to_one_fold" in failed
    assert "train_heldout_blocks_disjoint" in failed
    assert "a_fire" in rep.format() and "e_fire" in rep.format()


def test_a_clean_split_passes_c3_1(fake_split) -> None:  # noqa: ANN001
    rep = S.check_split_assignment(fires_root=fake_split[0], stats_path=fake_split[1])
    assert rep.ok, rep.format(verbose=True)


def test_block_coverage_below_four_is_a_reporting_gate(fake_split) -> None:  # noqa: ANN001
    """C6.3: G2 needs >= 4 DISTINCT held-out blocks. This split holds out 2."""
    rep = S.check_split_assignment(fires_root=fake_split[0], stats_path=fake_split[1])
    assert rep.ok and not rep.reporting_ok
    assert "heldout_block_coverage" in {c.check_id for c in rep.reporting_gaps}


@pytest.mark.skipif(not fires_dir().is_dir(), reason="no built fires on this machine")
def test_the_real_split_satisfies_c3_1_and_c6_3() -> None:
    rep = S.check_split_assignment()
    assert rep.reporting_ok, rep.format(verbose=True)


# --------------------------------------------------------------------------
# [A14] C3 - DECLARED train/held-out membership, by fire id (ADR-038 (6))
#
# Every test below PLANTS THE DEFECT AND ASSERTS THE CHECK CATCHES IT. The
# clause exists because its predecessor could not: `train_heldout_disjoint`
# intersected two lists that `split_fingerprint` builds as a partition, so it
# printed green on every input it could ever be given. A check that cannot fail
# is not a check, and the first duty of the replacement is to prove it can.
# --------------------------------------------------------------------------


def test_the_OLD_disjointness_check_could_not_have_failed(fake_split) -> None:  # noqa: ANN001
    """The vacuity, pinned as a fact rather than left as an anecdote.

    ``split_fingerprint`` partitions the same rows by ``fold in train_folds``,
    so its ``train_fire_ids`` and ``heldout_fire_ids`` cannot share a member -
    **however corrupt the artifacts are.** This test deliberately hands it the
    worst input it will ever see (a fire moved into a straddling block, plus a
    norm-stats file whose declared membership is a lie) and shows the derived
    intersection is STILL empty. That is why the clause now reads a value.
    """
    fires, stats = fake_split
    _fire(fires, "e_fire", fold=3, block=0)  # block 0 straddles folds 0 and 3
    _stats(stats, train_folds=[0, 1], train=["a_fire", "d_fire"], heldout=["d_fire"])
    fp = S.split_fingerprint(fires_root=fires, stats_path=stats)
    assert not (set(fp["train_fire_ids"]) & set(fp["heldout_fire_ids"])), (
        "if this ever becomes non-empty the derived form is no longer a partition and this "
        "test's premise needs re-reading"
    )


def test_an_overlapping_fire_id_is_a_HARD_failure(fake_split) -> None:  # noqa: ANN001
    """PLANTED DEFECT: one fire declared in BOTH train and held-out.

    This is the case ADR-038 (6) ruled into the contract. It is leakage of the
    most consequential kind - the normalisation is fitted on a fire that is then
    scored - and until now literally nothing in the repo read the two keys.
    """
    fires, stats = fake_split
    _stats(
        stats,
        train_folds=[0, 1],
        train=["a_fire", "b_fire", "c_fire"],
        heldout=["c_fire", "d_fire"],  # c_fire is in both
    )
    rep = S.check_split_assignment(fires_root=fires, stats_path=stats)
    failed = {c.check_id: c for c in rep.failures}
    assert "train_heldout_disjoint" in failed, rep.format(verbose=True)
    assert failed["train_heldout_disjoint"].severity == "fail", "leakage is HARD, never reporting"
    assert "c_fire" in failed["train_heldout_disjoint"].message, "the offending id must be NAMED"
    assert "a_fire" not in failed["train_heldout_disjoint"].message, "and only the offending id"
    assert not rep.ok


@pytest.mark.parametrize("omit", ["train_fire_ids", "heldout_fire_ids", "both"])
def test_a_MISSING_membership_key_is_a_failure_not_a_skip(fake_split, omit: str) -> None:  # noqa: ANN001
    """PLANTED DEFECT: the key is absent.

    Absence must not read as "nothing to check here". The maintainer's own
    ADR-038 (2) is the case study: a query for a key that did not exist returned
    ``None`` and was published as "the file records no train list". A check that
    silently skips is indistinguishable from a check that passes, and this repo
    has now hit that shape at least four times.
    """
    fires, stats = fake_split
    _stats(
        stats,
        train_folds=[0, 1],
        train=None if omit in ("train_fire_ids", "both") else ["a_fire", "b_fire"],
        heldout=None if omit in ("heldout_fire_ids", "both") else ["c_fire", "d_fire"],
    )
    rep = S.check_split_assignment(fires_root=fires, stats_path=stats)
    failed = {c.check_id for c in rep.failures}
    assert "norm_stats_declares_fire_ids" in failed
    assert "train_heldout_disjoint" in failed, (
        "an UNVERIFIABLE disjointness must fail too (C-1: `fail` is 'invariant violated OR "
        "unverifiable'). If only the declaration check fired, a file could delete the keys and "
        "lose the leakage guard while the leakage check itself still printed green"
    )
    assert not rep.ok


def test_a_STALE_norm_stats_disagreeing_with_the_manifests_is_caught(fake_split) -> None:  # noqa: ANN001
    """PLANTED DEFECT: the corpus moved and norm_stats.json did not.

    ADR-038 (1) measured this exact hazard at 9.76% of train mass -
    ``2020_july_complex`` moved train -> held-out during the D6 swap, and a
    stats file still listing it as train would have baked its statistics into
    the normalisation of a fire it was then scored on. Both artifacts are
    individually well-formed; only their JOIN is wrong, which is why this cannot
    live in a per-fire check.
    """
    fires, stats = fake_split
    _fire(fires, "b_fire", fold=3, block=1)  # b_fire moves train -> held out
    _stats(stats, train_folds=[0, 1], train=["a_fire", "b_fire"], heldout=["c_fire", "d_fire"])
    rep = S.check_split_assignment(fires_root=fires, stats_path=stats)
    failed = {c.check_id: c for c in rep.failures}
    assert "declared_membership_matches_manifests" in failed, rep.format(verbose=True)
    assert "b_fire" in failed["declared_membership_matches_manifests"].message
    assert "train_heldout_disjoint" not in failed, (
        "the declared sets ARE disjoint here — the defect is that they are stale. Two "
        "different failures must not collapse into one check id"
    )


def test_a_malformed_or_duplicated_membership_list_is_caught(fake_split) -> None:  # noqa: ANN001
    """PLANTED DEFECT: the right key holding the wrong kind of value."""
    fires, stats = fake_split
    stats.write_text(
        json.dumps(
            {
                "train_folds": [0, 1],
                "n_train_blocks": 2,
                "train_fire_ids": [0, 1],  # ints, not fire ids
                "heldout_fire_ids": ["c_fire", "c_fire"],  # duplicated
            }
        )
    )
    declared = S.declared_split_membership(stats)
    assert not declared["present"]
    assert any("not a list of fire-id strings" in p for p in declared["problems"])
    assert any("more than once" in p for p in declared["problems"])
    rep = S.check_split_assignment(fires_root=fires, stats_path=stats)
    assert "norm_stats_declares_fire_ids" in {c.check_id for c in rep.failures}


def test_an_unreadable_norm_stats_file_is_a_failure_not_an_exception(tmp_path: Path) -> None:
    declared = S.declared_split_membership(tmp_path / "does_not_exist.json")
    assert declared["present"] is False
    assert declared["problems"] and declared["train"] == [] and declared["heldout"] == []


@pytest.mark.skipif(
    not (fires_dir().is_dir() and norm_stats_path().is_file()),
    reason="no built fires on this machine",
)
def test_the_REAL_norm_stats_declares_a_disjoint_membership() -> None:
    """The clause READ AGAINST THE REAL ARTIFACT, with the counts asserted.

    Deliberately not phrased as "the intersection is empty": that is an
    absence-of-match claim, and this project has produced four confident false
    negatives from exactly that shape. The sizes are read and asserted, so a
    query that silently found nothing fails here instead of passing.
    """
    declared = S.declared_split_membership()
    assert declared["present"], declared["problems"]
    assert len(declared["train"]) == 16, declared["train"]
    assert len(declared["heldout"]) == 5, declared["heldout"]
    assert "2020_july_complex" in declared["heldout"], (
        "ADR-038 (1): this fire carried 9.76% of the STALE train mass and moved to held-out"
    )
    assert not (set(declared["train"]) & set(declared["heldout"]))


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


@pytest.mark.skipif(not fires_dir().is_dir(), reason="no built fires on this machine")
def test_cli_runs_and_reports_the_current_split() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "wildfire_nowcast.common.splits", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["current_split"]["fingerprint"]
    assert payload["report"]["ok"] is True


def test_create_run_dir_stamps_the_split(tmp_path: Path) -> None:
    """C8: *every run stamps* ``split_fingerprint`` - structurally, not by memory.

    10 of the 20 run directories in ``runs/`` carry no fingerprint at all, all
    of them from before ``eval/`` started stamping one. Putting it in
    ``create_run_dir`` means it no longer depends on which entry point a lead
    writes next.
    """
    from wildfire_nowcast.common.runs import create_run_dir

    run = create_run_dir({"experiment": "t"}, run_id="r1", runs_root=tmp_path)
    meta = json.loads(run.meta_path.read_text())
    assert "split_fingerprint" in meta
    assert isinstance(meta["split_fingerprint"].get("fingerprint"), str)
    assert S.fingerprints_in(meta), "the stamp must be discoverable by the C8 checker"


# --------------------------------------------------------------------------
# C-4.3 [v2.12] - the interpreter ENVIRONMENT is in C-4's frozen set (ADR-024)
# --------------------------------------------------------------------------


def test_the_environment_fingerprint_is_deterministic_within_a_process() -> None:
    """Two stamps taken back to back must agree, or every C-4.3 verdict is noise.

    This is the property the whole clause rests on: the check is
    before-vs-after, so a fingerprint that varies on its own would hard-fail
    every run and be disabled within a day (the C1.6 lesson).
    """
    from wildfire_nowcast.common.environment import environment_fingerprint

    first, second = environment_fingerprint(), environment_fingerprint()
    assert first["fingerprint"] == second["fingerprint"]
    assert isinstance(first["fingerprint"], str) and len(first["fingerprint"]) == 16
    assert first["n_distributions"] > 10, first
    assert "covers" in first, "the payload must state its own scope, not imply it"


def test_the_environment_fingerprint_MOVES_when_a_package_version_moves() -> None:
    """The positive control: if this cannot change, the clause cannot fire.

    Asserted by substituting the distribution enumeration rather than by
    installing anything - installing a package to test the no-installing clause
    would be its own joke, and C-4.3 binds this session too.
    """
    from wildfire_nowcast.common import environment as E

    baseline = E.environment_fingerprint()["fingerprint"]
    real = E.installed_distributions
    try:
        E.installed_distributions = lambda: [*real(), ["scipy", "1.14.0"]]
        moved = E.environment_fingerprint()["fingerprint"]
    finally:
        E.installed_distributions = real
    assert moved != baseline, (
        "adding a distribution did not change the environment fingerprint, so C-4.3 could "
        "never detect the `pip install` it exists to detect"
    )
    assert E.environment_fingerprint()["fingerprint"] == baseline, "not restored"


def test_an_environment_that_moved_during_the_run_is_a_HARD_fail() -> None:
    """C-4.3's own sentence, made executable - and its own check id, not code's."""
    rep = S.check_run_split(
        {
            "split_before": {"fingerprint": "aaaa"},
            "split_after": {"fingerprint": "aaaa"},
            "environment_before": {"fingerprint": "env1"},
            "environment_after": {"fingerprint": "env2"},
        },
        current={"fingerprint": "aaaa"},
    )
    failed = {c.check_id for c in rep.failures}
    assert "environment_agrees_across_run" in failed
    assert "code_agrees_across_run" not in failed, (
        "the environment must not be reported under the CODE check id. Two quantities sharing "
        "one key name is exactly what made C8.internally_consistent false on every artifact "
        "in this repo (A12); adding a third family under `code_` would have repeated it"
    )
    assert not rep.ok


def test_a_one_ended_environment_stamp_is_a_reporting_gap_not_a_failure() -> None:
    """Same tier as C-4.2, for its reason and not by analogy: every artifact in
    ``runs/`` predates this clause and stamps no environment at all."""
    rep = S.check_run_split(
        {
            "split_before": {"fingerprint": "aaaa"},
            "split_after": {"fingerprint": "aaaa"},
            "environment_after": {"fingerprint": "env1"},
        },
        current={"fingerprint": "aaaa"},
    )
    ids = {c.check_id: c for c in rep.checks}
    assert ids["environment_sampled_both_ends"].ok is False
    assert ids["environment_sampled_both_ends"].severity == "reporting"
    assert "environment_agrees_across_run" not in {c.check_id for c in rep.failures}


def test_an_environment_block_is_NOT_counted_as_a_split_fingerprint() -> None:
    """The A12 false positive, PRE-EMPTED rather than rediscovered.

    ``environment_before`` carries its own ``fingerprint`` key. If the split
    walker collected it, ``C8.internally_consistent`` would report "MORE THAN ONE
    split fingerprint" and hard-fail every run that stamps one - which, now that
    ``create_run_dir`` stamps it structurally, is every run from today onward.
    """
    payload = {
        "split_before": {"fingerprint": "aaaa"},
        "split_after": {"fingerprint": "aaaa"},
        "environment_before": {"fingerprint": "env1"},
        "environment_after": {"fingerprint": "env1"},
    }
    assert set(S.fingerprints_in(payload).values()) == {"aaaa"}
    rep = S.check_run_split(payload, current={"fingerprint": "aaaa"})
    assert rep.ok, rep.format(verbose=True)


def test_an_artifact_with_no_environment_stamp_emits_no_c4_3_clause() -> None:
    """Every archived run predates C-4.3. It must not acquire a phantom gap."""
    rep = S.check_run_split(
        {"split_before": {"fingerprint": "aaaa"}, "split_after": {"fingerprint": "aaaa"}},
        current={"fingerprint": "aaaa"},
    )
    assert not any(c.check_id.startswith("environment_") for c in rep.checks)


def test_missing_or_none_environment_stamps_never_compare_equal() -> None:
    """C1.5's rule at the choke point: an unevaluable comparison is unpassable.

    Without this, a run stamping ``None`` at both ends would read as "the
    environment did not change" - a green check standing in for a measurement
    that was never taken, which is this project's single most repeated defect.
    """
    from wildfire_nowcast.common.environment import environments_agree

    assert environments_agree({"fingerprint": "a"}, {"fingerprint": "a"})
    assert environments_agree("a", "a")
    assert not environments_agree({"fingerprint": None}, {"fingerprint": None})
    assert not environments_agree(None, None)
    assert not environments_agree({}, {})
    assert not environments_agree({"fingerprint": "a"}, None)


def test_create_run_dir_stamps_the_environment(tmp_path: Path) -> None:
    """C-4.3 structurally, on ``_split_stamp``'s precedent: a run directory carries
    it no matter who wrote the run."""
    from wildfire_nowcast.common.runs import create_run_dir

    run = create_run_dir({"experiment": "t"}, run_id="env1", runs_root=tmp_path)
    meta = json.loads(run.meta_path.read_text())
    assert isinstance(meta["environment_before"].get("fingerprint"), str)
    assert "covers" in meta["environment_before"]
    # ...and it must NOT pollute the split stamp (see the A12 pre-emption above).
    assert set(S.fingerprints_in(meta).values()) == {meta["split_fingerprint"]["fingerprint"]}


# --------------------------------------------------------------------------
# [v2.16] C8.1 - the CV-MATRIX artifact class (ADR-062 (6))
#
# Every one of these plants the defect the clause names. A guard nobody has
# watched fail is not a guard: C8.1 exists because a leave-fold-out matrix has
# five fingerprints by construction, and the tempting fix - record them under a
# name the checker does not read - is the exact move C8 exists to prevent.
# --------------------------------------------------------------------------


def _fold_run(runs_root: Path, name: str, fingerprint: str) -> None:
    """A fold run dir carrying exactly ONE stamp, as C8.1 requires of a member."""
    run_dir = runs_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps({"split_fingerprint": fingerprint}))


def _matrix_payload(members: dict[str, str], *, n_members: int | None = None) -> dict[str, Any]:
    """An aggregate declaring ``{label: fingerprint}``, run dirs named ``s1-<label>``."""
    declared = {
        label: {"run": f"runs/s1-{label}", "split_fingerprint": fp} for label, fp in members.items()
    }
    return {
        "arm": "S",
        "cv_matrix": {
            "n_members": len(declared) if n_members is None else n_members,
            "members": declared,
            "adr": "ADR-062 (6)",
        },
    }


#: Five folds, five splits - the shape ADR-062 (6) is actually about.
_FIVE_FOLDS = {
    "fold0": "0000aaaaaaaaaaaa",
    "fold1": "1111bbbbbbbbbbbb",
    "fold2": "2222cccccccccccc",
    "fold3": "3333dddddddddddd",
    "fold4": "4444eeeeeeeeeeee",
}


def _five_fold_matrix(runs_root: Path) -> dict[str, Any]:
    for label, fp in _FIVE_FOLDS.items():
        _fold_run(runs_root, f"s1-{label}", fp)
    return _matrix_payload(_FIVE_FOLDS)


def test_a_well_formed_five_fold_matrix_is_GREEN(tmp_path: Path) -> None:
    """OBSERVATION 3 (the control): five folds, five fingerprints, all verified.

    Under v2.15 this identical artifact HARD FAILS `C8.internally_consistent`
    for carrying five stamps - which is why the matrix could not be checked at
    all rather than being checked leniently.
    """
    payload = _five_fold_matrix(tmp_path)
    rep = S.check_run_split(
        payload, current={"fingerprint": _FIVE_FOLDS["fold3"]}, runs_root=tmp_path
    )
    assert rep.ok, rep.format(verbose=True)
    assert rep.reporting_ok, rep.format(verbose=True)
    emitted = {c.check_id for c in rep.checks}
    assert {
        "cv_matrix_well_formed",
        "cv_matrix_member_count",
        "cv_matrix_member_stamps",
        "cv_matrix_members_distinct",
    } <= emitted, sorted(emitted)


def test_a_member_stamp_that_DISAGREES_with_the_matrix_claim_is_a_HARD_fail(
    tmp_path: Path,
) -> None:
    """OBSERVATION 1 - PLANTED: fold 2's run dir was trained under a different split.

    This is the ADR-015 defect with the AGGREGATE as its subject: the claim is
    what a reader trusts, the run dir is what was actually trained, and a matrix
    describing a split that produced none of its numbers reads as reassurance
    while doing so.
    """
    payload = _five_fold_matrix(tmp_path)
    _fold_run(tmp_path, "s1-fold2", "9999ffffffffffff")  # PLANTED: claim says 2222cc...

    rep = S.check_run_split(
        payload, current={"fingerprint": _FIVE_FOLDS["fold3"]}, runs_root=tmp_path
    )
    failed = {c.check_id for c in rep.failures}
    assert "cv_matrix_member_stamps" in failed, rep.format(verbose=True)
    assert not rep.ok
    detail = next(c.message for c in rep.failures if c.check_id == "cv_matrix_member_stamps")
    assert "fold2" in detail and "9999ffffffffffff" in detail and "2222cccccccccccc" in detail, (
        "the failure must name WHICH member, what was CLAIMED and what was STAMPED — a hard "
        "clause whose message does not identify the offender gets read once and then ignored"
    )
    # ...and the OTHER four members must not be implicated by one bad member.
    assert "cv_matrix_member_count" not in failed


def test_five_declared_but_four_present_is_a_HARD_fail(tmp_path: Path) -> None:
    """OBSERVATION 2 - PLANTED: the matrix declares 5 folds and 4 exist on disk.

    The criterion this matrix feeds is a COUNT OVER FOLDS (>= 11/14, ADR-061
    (6)), so a fold that silently does not exist moves the denominator without
    changing anything a reader can see. ADR-063 (3) named exactly this: a partial
    run that is not self-identifying is the expensive kind of interrupted run.
    """
    payload = _five_fold_matrix(tmp_path)
    shutil.rmtree(tmp_path / "s1-fold4")  # PLANTED: 5 declared, 4 present

    rep = S.check_run_split(
        payload, current={"fingerprint": _FIVE_FOLDS["fold3"]}, runs_root=tmp_path
    )
    failed = {c.check_id for c in rep.failures}
    assert "cv_matrix_member_count" in failed, rep.format(verbose=True)
    assert not rep.ok
    detail = next(c.message for c in rep.failures if c.check_id == "cv_matrix_member_count")
    assert "fold4" in detail and "n_members=5" in detail, detail


def test_a_matrix_declaration_the_checker_cannot_read_is_a_HARD_fail(tmp_path: Path) -> None:
    """PLANTED: `cv_matrix` present, members carrying a fingerprint and NO run.

    Without this clause the exemption is free: declare the key, get the member
    stamps out of `C8.internally_consistent`, and declare nothing checkable. That
    is the renamed-field move under a different name, so it must cost more than
    the honest form, not less.
    """
    payload = {
        "cv_matrix": {
            "n_members": 5,
            "members": {label: {"split_fingerprint": fp} for label, fp in _FIVE_FOLDS.items()},
        }
    }
    rep = S.check_run_split(payload, current={"fingerprint": "zzzz"}, runs_root=tmp_path)
    failed = {c.check_id for c in rep.failures}
    assert "cv_matrix_well_formed" in failed, rep.format(verbose=True)
    assert not rep.ok


def test_a_matrix_member_run_with_NO_stamp_at_all_is_a_HARD_fail(tmp_path: Path) -> None:
    """C-1 one level up: an unverifiable claim is a failure, not a pass.

    The dir exists, so the count clause is satisfied; there is simply nothing in
    it to compare the claim against. A checker that treated "no stamp to compare"
    as agreement would pass the emptiest possible matrix.
    """
    payload = _five_fold_matrix(tmp_path)
    (tmp_path / "s1-fold1" / "results.json").unlink()

    rep = S.check_run_split(
        payload, current={"fingerprint": _FIVE_FOLDS["fold3"]}, runs_root=tmp_path
    )
    failed = {c.check_id for c in rep.failures}
    assert "cv_matrix_member_stamps" in failed, rep.format(verbose=True)
    assert "unverifiable" in next(
        c.message for c in rep.failures if c.check_id == "cv_matrix_member_stamps"
    )


def test_two_members_sharing_a_fingerprint_REPORTS_but_does_not_block(tmp_path: Path) -> None:
    """The one C8.1 clause that is reporting-tier, and the reason is the NULL RUNG.

    Two folds under one split IS a defect in a leave-fold-out matrix. But the
    mandatory null rung - the same arm retrained at a second seed on every fold
    (ADR-062 (4)) - is the same split twice BY DESIGN, and a hard clause here
    would forbid the control that makes the matrix readable. Reported loudly,
    never blocking.
    """
    duplicated = {**_FIVE_FOLDS, "fold4": _FIVE_FOLDS["fold3"]}
    for label, fp in duplicated.items():
        _fold_run(tmp_path, f"s1-{label}", fp)
    rep = S.check_run_split(
        _matrix_payload(duplicated),
        current={"fingerprint": _FIVE_FOLDS["fold3"]},
        runs_root=tmp_path,
    )
    ids = {c.check_id: c for c in rep.checks}
    assert ids["cv_matrix_members_distinct"].ok is False
    assert ids["cv_matrix_members_distinct"].severity == "reporting"
    assert rep.ok and not rep.reporting_ok, rep.format(verbose=True)


def test_the_matrix_key_does_not_buy_a_FOLD_run_out_of_anything(tmp_path: Path) -> None:
    """Each fold is its own run dir carrying ONE stamp - that part is UNCHANGED.

    The exemption is for the aggregate's declaration and nothing else. A fold run
    that carries two disagreeing stamps is the literal train-vs-eval mismatch and
    is still a hard fail, matrix or no matrix.
    """
    run_dir = tmp_path / "s1-fold0"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            {"split_before": {"fingerprint": "aaaa"}, "split_after": {"fingerprint": "bbbb"}}
        )
    )
    rep = S.check_run_split(run_dir, current={"fingerprint": "aaaa"}, runs_root=tmp_path)
    assert "internally_consistent" in {c.check_id for c in rep.failures}, rep.format(verbose=True)


def test_an_ordinary_artifact_emits_NO_cv_matrix_clause(tmp_path: Path) -> None:
    """Every archived run predates C8.1 and must not acquire a phantom failure.

    Same reasoning as C-4.3's tiering, and asserted for the same reason: a hard
    clause that fires on every artifact is worse than no clause, because the
    first thing anyone does with it is stop reading it.
    """
    rep = S.check_run_split(
        {"split_before": {"fingerprint": "aaaa"}, "split_after": {"fingerprint": "aaaa"}},
        current={"fingerprint": "aaaa"},
        runs_root=tmp_path,
    )
    assert not any(c.check_id.startswith("cv_matrix_") for c in rep.checks)
    assert rep.ok


def test_the_member_stamps_are_STILL_VISIBLE_to_the_checker(tmp_path: Path) -> None:
    """The refusal ADR-062 (6) upheld, asserted as a property rather than trusted.

    `fingerprints_in` skips `cv_matrix` so a matrix does not hard-fail
    `internally_consistent` for its own structure - but the five stamps must not
    thereby become invisible. `declared_cv_matrix` reads every one of them, and
    `check_run_split` checks each against its run dir. If a future edit moved the
    members somewhere the parser does not look, this test goes red.
    """
    payload = _five_fold_matrix(tmp_path)
    assert S.fingerprints_in(payload) == {}, "the aggregate carries no stamp of its own"
    matrix = S.declared_cv_matrix(payload)
    assert matrix is not None
    assert set(matrix.fingerprints) == set(_FIVE_FOLDS.values()), (
        "the checker must SEE all five declared fingerprints. Recording them where it cannot "
        "is precisely the move C8 exists to prevent (ADR-062 (6))"
    )
    assert matrix.n_members == 5


def test_a_matrix_still_reports_when_the_split_on_disk_is_none_of_its_members(
    tmp_path: Path,
) -> None:
    """C8.matches_current stays REPORTING-tier and stays meaningful for a matrix.

    It passes when today's split is any ONE of the members - a fold run scored
    today is legitimately current - and gaps when it is none of them.
    """
    payload = _five_fold_matrix(tmp_path)
    ok = S.check_run_split(
        payload, current={"fingerprint": _FIVE_FOLDS["fold0"]}, runs_root=tmp_path
    )
    assert {c.check_id: c for c in ok.checks}["matches_current"].ok is True
    stale = S.check_run_split(payload, current={"fingerprint": "not-a-member"}, runs_root=tmp_path)
    gap = {c.check_id: c for c in stale.checks}["matches_current"]
    assert gap.ok is False and gap.severity == "reporting"
    assert stale.ok, "a stale-split matrix is a REPORTING gap, not a hard failure"


# --------------------------------------------------------------------------
# [v2.16] C8.2 - ATOMICITY OF THE FIT AND THE STAMPS (ADR-062 (5))
#
# The approved `stats_path` parameter lets a caller train against one of five
# internally-consistent fold-stats files. The danger is a caller setting the FIT
# from one of them while the STAMPS still come from the default, which silently
# recreates the leak the parameter was approved to avoid. These tests plant that
# caller and show it cannot pass.
# --------------------------------------------------------------------------


@pytest.fixture
def two_folds(tmp_path: Path) -> tuple[Path, Path, Path]:
    """``(fires_root, stats_A, stats_B)`` - the same corpus under TWO partitions.

    This is the leave-fold-out situation in miniature: one set of manifests, two
    ``norm_stats.json`` files each fitted on its own folds and each declaring its
    own membership by id. Nothing here is malformed; the only way to go wrong is
    to mix them.
    """
    fires = tmp_path / "fires"
    _fire(fires, "a_fire", fold=0, block=0)
    _fire(fires, "b_fire", fold=1, block=1)
    _fire(fires, "c_fire", fold=2, block=2)
    _fire(fires, "d_fire", fold=3, block=3)

    stats_a = tmp_path / "norm_stats_heldout3.json"
    _stats(stats_a, train_folds=[0, 1, 2], train=["a_fire", "b_fire", "c_fire"], heldout=["d_fire"])
    stats_b = tmp_path / "norm_stats_heldout0.json"
    _stats(stats_b, train_folds=[1, 2, 3], train=["b_fire", "c_fire", "d_fire"], heldout=["a_fire"])
    return fires, stats_a, stats_b


def test_a_split_context_moves_the_fit_and_BOTH_stamps_together(
    two_folds: tuple[Path, Path, Path],
) -> None:
    """The control: one parameter, one resolution point, three products agreeing."""
    fires, _stats_a, stats_b = two_folds
    ctx = S.resolve_split_context(stats_path=stats_b, fires_root=fires)

    fit = ctx.norm_stats()
    stamp = ctx.fingerprint()
    assert fit["train_folds"] == stamp["train_folds"] == [1, 2, 3]
    assert sorted(fit["train_fire_ids"]) == sorted(stamp["train_fire_ids"])
    closing = ctx.assert_unchanged(stamp)
    assert closing["fingerprint"] == stamp["fingerprint"]
    # ...and the stamp says WHICH of the five it came from, so a later reader
    # does not have to re-derive it.
    assert stamp[S.SPLIT_CONTEXT_KEY]["stats_path"] == str(stats_b)


def test_PLANTED_a_caller_that_sets_the_FIT_without_moving_the_STAMPS_cannot_pass(
    two_folds: tuple[Path, Path, Path],
) -> None:
    """OBSERVATION - PLANTED: the exact leak ADR-062 (5) says must be impossible.

    This is the old shape, written out deliberately: `read_norm_stats(B)` for the
    fit, `split_fingerprint()` at the default for the stamp. Every individual
    call is correct; only their JOIN is wrong - the same structure as ADR-015,
    where every tensor was conformant and the relation between them was not.
    """
    from wildfire_nowcast.common.zarr_io import read_norm_stats

    fires, stats_a, stats_b = two_folds
    fit = read_norm_stats(stats_b)  # PLANTED: fit from B ...
    stamp = S.split_fingerprint(fires_root=fires, stats_path=stats_a)  # ... stamp from A

    with pytest.raises(S.SplitFitStampMismatchError) as excinfo:
        S.assert_fit_and_stamp_agree(fit, stamp)
    message = str(excinfo.value)
    assert "[1, 2, 3]" in message and "[0, 1, 2]" in message, message
    assert "SAME OBJECT" in message, "the failure must state the invariant it defends"

    # And the atomic form of the same intent is fine, which is what makes the
    # test above a statement about the MIXING rather than about either file.
    S.assert_fit_and_stamp_agree(
        read_norm_stats(stats_b),
        S.split_fingerprint(fires_root=fires, stats_path=stats_b),
    )


def test_no_split_context_operation_accepts_a_path(two_folds: tuple[Path, Path, Path]) -> None:
    """The shape claim, asserted by INTROSPECTION rather than by convention.

    "Make it impossible, not discouraged" is a claim about the API surface, so it
    is checked against the API surface. If a future edit adds a `stats_path` back
    onto any of these methods, the desynchronised call becomes expressible again
    and this test goes red at that moment rather than at the next leak.
    """
    import inspect

    for name in ("norm_stats", "fingerprint", "assert_unchanged", "check_assignment"):
        params = set(inspect.signature(getattr(S.SplitContext, name)).parameters) - {"self"}
        offending = {p for p in params if "path" in p or "root" in p or "stats" in p}
        assert not offending, (
            f"SplitContext.{name} accepts {offending}. C8.2 is enforced by SHAPE: the whole "
            "guarantee is that these operations have NO path parameter to desynchronise, so "
            "adding one back reduces the clause to a convention"
        )

    fires, _stats_a, stats_b = two_folds
    ctx = S.resolve_split_context(stats_path=stats_b, fires_root=fires)
    with pytest.raises(TypeError):
        ctx.fingerprint(stats_path=_stats_a)  # type: ignore[call-arg]


def test_PLANTED_a_stats_file_whose_membership_drifted_cannot_become_a_context(
    tmp_path: Path,
) -> None:
    """The case the SHAPE cannot cover, which is why the belt exists.

    A single path used consistently everywhere still leaks if the file's declared
    membership has drifted from the manifests it names - ADR-038 (1), where one
    fire carrying 9.76% of train mass moved train -> held-out. The shape cannot
    see this; the value-reading guard can.
    """
    fires = tmp_path / "fires"
    _fire(fires, "a_fire", fold=0, block=0)
    _fire(fires, "b_fire", fold=1, block=1)
    _fire(fires, "c_fire", fold=3, block=2)  # PLANTED: manifest says fold 3 ...
    stats = tmp_path / "norm_stats.json"
    _stats(
        stats,
        train_folds=[0, 1],
        train=["a_fire", "b_fire", "c_fire"],  # ... the stats still call it TRAIN
        heldout=[],
    )
    with pytest.raises(S.SplitFitStampMismatchError) as excinfo:
        S.resolve_split_context(stats_path=stats, fires_root=fires)
    assert "c_fire" in str(excinfo.value)


def test_a_run_cannot_be_CLOSED_under_a_different_context_than_it_OPENED(
    two_folds: tuple[Path, Path, Path],
) -> None:
    """Two partitions bracketing one run: the 'unchanged' would be between two
    things that were never the same."""
    fires, stats_a, stats_b = two_folds
    opened = S.resolve_split_context(stats_path=stats_a, fires_root=fires).fingerprint()
    closing = S.resolve_split_context(stats_path=stats_b, fires_root=fires)
    with pytest.raises(S.SplitFitStampMismatchError) as excinfo:
        closing.assert_unchanged(opened)
    assert str(stats_a) in str(excinfo.value) and str(stats_b) in str(excinfo.value)


def test_the_context_provenance_is_written_repo_relative(
    two_folds: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """[I11] The stamp copied into every run artifact must not name a home directory.

    ``fires_root`` defaults to ``fires_dir()``, which is absolute, so the DEFAULT
    path through ``provenance()`` used to publish the operator's account name and
    directory layout into every result artifact that carries this block. 64 such
    paths reached the public tree before this was measured.
    """
    fires, stats_a, _stats_b = two_folds
    monkeypatch.setenv("WILDFIRE_REPO_ROOT", str(fires.parent))
    prov = S.resolve_split_context(stats_path=stats_a, fires_root=fires).provenance()
    assert prov["fires_root"] == fires.name, prov
    assert prov["stats_path"] == str(stats_a.relative_to(fires.parent)), prov
    assert not prov["fires_root"].startswith("/"), prov


def test_a_stamp_written_before_the_relative_rewrite_still_closes_its_own_run(
    two_folds: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """[I11] The representation changed; the partition did not.

    C8.2 asks whether a run opened and closed on ONE partition. Comparing the raw
    strings after the rewrite would answer "no" for a run that opened with the
    absolute spelling and closed with the relative one, i.e. it would report a
    fold rotation that never happened. The comparison is normalised through the
    same function that produced the change and through nothing else.
    """
    fires, stats_a, _stats_b = two_folds
    monkeypatch.setenv("WILDFIRE_REPO_ROOT", str(fires.parent))
    ctx = S.resolve_split_context(stats_path=stats_a, fires_root=fires)
    before = ctx.fingerprint()
    assert not before[S.SPLIT_CONTEXT_KEY]["stats_path"].startswith("/")

    # ...now age it: put back the ABSOLUTE spelling a pre-I11 run would carry.
    aged = dict(before)
    aged[S.SPLIT_CONTEXT_KEY] = dict(before[S.SPLIT_CONTEXT_KEY])
    aged[S.SPLIT_CONTEXT_KEY]["stats_path"] = str(stats_a.resolve())
    ctx.assert_unchanged(aged)  # must not raise


def test_the_normalised_comparison_still_separates_two_different_files(
    two_folds: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the test above. Tolerating a SPELLING is not tolerating a FILE."""
    fires, stats_a, stats_b = two_folds
    monkeypatch.setenv("WILDFIRE_REPO_ROOT", str(fires.parent))
    assert S._same_stats_path(stats_a, str(stats_a.resolve()))
    assert not S._same_stats_path(stats_a, stats_b)

    opened = S.resolve_split_context(stats_path=stats_a, fires_root=fires).fingerprint()
    closing = S.resolve_split_context(stats_path=stats_b, fires_root=fires)
    with pytest.raises(S.SplitFitStampMismatchError):
        closing.assert_unchanged(opened)


def test_a_split_context_still_detects_the_ADR_015_defect(
    two_folds: tuple[Path, Path, Path],
) -> None:
    """C8's original job survives the new wrapper - the split MOVING mid-run.

    Worth a test rather than an assumption: `assert_unchanged` gained a
    precondition, and a guard that starts failing early can stop reaching the
    check it was wrapping.
    """
    fires, _stats_a, stats_b = two_folds
    ctx = S.resolve_split_context(stats_path=stats_b, fires_root=fires)
    before = ctx.fingerprint()
    _fire(fires, "e_fire", fold=1, block=4)  # a fire appears mid-run
    with pytest.raises(S.SplitChangedError):
        ctx.assert_unchanged(before)


def test_the_default_context_is_todays_behaviour_exactly(
    two_folds: tuple[Path, Path, Path],
) -> None:
    """`stats_path=None` must be byte-identical to the unparameterised call.

    ADR-062 (5) approved the parameter on the condition that the default is
    unchanged; five artifacts are being stamped through `split_fingerprint` in
    parallel and the hash of record may not move.
    """
    fires, stats_a, _stats_b = two_folds
    ctx = S.resolve_split_context(stats_path=stats_a, fires_root=fires)
    direct = S.split_fingerprint(fires_root=fires, stats_path=stats_a)
    through = ctx.fingerprint()
    assert through["fingerprint"] == direct["fingerprint"]
    assert {k: v for k, v in through.items() if k != S.SPLIT_CONTEXT_KEY} == direct, (
        "the context may ADD provenance and may not change any value split_fingerprint "
        "produced — the hash of record is being stamped through it right now"
    )


# --------------------------------------------------------------------------
# [v2.16] C6.3 (addition) - AN EXPECTED FALSE IS STAMPED, NOT DISCOVERED
# (ADR-062 (7))
#
# The whole hazard of an "expected" stamp is that it becomes a way to write
# EXPECTED next to a value and have the reader stop looking at the value. So the
# tests that matter here are the ones that try to make the stamp move it.
# --------------------------------------------------------------------------


def test_the_expected_false_stamp_does_NOT_flip_the_value() -> None:
    """THE property. `c6_3_satisfied` must be byte-identical after stamping."""
    before = {"c6_3_satisfied": False, "n_heldout_blocks": 1, "heldout_blocks": [2]}
    after = S.stamp_c6_3_expected_false(
        before,
        citation="ADR-062 (7)",
        why="fold 0 holds out block {2} alone; S1 pools 14 blocks and adjudicates nothing "
        "gate-shaped",
    )
    assert after["c6_3_satisfied"] is False, "the stamp MOVED the value it was meant to annotate"
    assert before["c6_3_satisfied"] is False, "the input was mutated"
    assert {k: v for k, v in after.items() if k != S.C6_3_EXPECTED_FALSE_KEY} == before
    assert "ADR-062" in after[S.C6_3_EXPECTED_FALSE_KEY]["citation"]


def test_PLANTED_stamping_a_SATISFIED_split_as_expected_false_raises() -> None:
    """The abuse this mechanism would otherwise create, planted.

    If "expected" could be written beside a `true`, the stamp would be a way of
    telling a reader to stop reading - the same failure as a warning nobody reads
    (C6.6's reasoning), one artifact down.
    """
    for value in (True, None, 1, "false"):
        with pytest.raises(ValueError, match="must BE false"):
            S.stamp_c6_3_expected_false(
                {"c6_3_satisfied": value}, citation="ADR-062 (7)", why="because I said so"
            )


def test_an_expectation_with_no_ADR_behind_it_is_refused() -> None:
    """An expectation with no provenance is an assertion, and this project has
    been burned by exactly that: a ruling made two weeks earlier that nobody
    could find from the artifact."""
    with pytest.raises(ValueError, match="must name an ADR"):
        S.stamp_c6_3_expected_false(
            {"c6_3_satisfied": False}, citation="the maintainer said it was fine", why="one block"
        )
    with pytest.raises(ValueError, match="what makes this false EXPECTED"):
        S.stamp_c6_3_expected_false({"c6_3_satisfied": False}, citation="ADR-062 (7)", why="  ")


def test_PLANTED_a_declaration_beside_a_TRUE_value_is_a_HARD_fail() -> None:
    """The checker's half of the same property, in case someone writes the JSON
    by hand and never goes through the stamping function."""
    rep = S.check_run_split(
        {
            "split_fingerprint": "aaaa",
            "fold": {
                "c6_3_satisfied": True,  # PLANTED: declared expected-false, but TRUE
                "c6_3_expected_false": {"citation": "ADR-062 (7)", "why": "one block"},
            },
        },
        current={"fingerprint": "aaaa"},
    )
    failed = {c.check_id for c in rep.failures}
    assert "c6_3_expected_false_did_not_flip" in failed, rep.format(verbose=True)
    assert not rep.ok


def test_PLANTED_a_declaration_citing_nothing_is_a_HARD_fail() -> None:
    rep = S.check_run_split(
        {
            "split_fingerprint": "aaaa",
            "fold": {
                "c6_3_satisfied": False,
                "c6_3_expected_false": {"citation": "trust me", "why": "one block"},
            },
        },
        current={"fingerprint": "aaaa"},
    )
    assert "c6_3_expected_false_did_not_flip" in {c.check_id for c in rep.failures}


def test_an_UNDECLARED_false_is_a_reporting_gap_and_the_declared_one_is_clean() -> None:
    """The tier, and its reason: the value is already TRUE of the split (the fold
    really does hold out one block), so it is a documentation gap and not a
    measurement failure. Every archived artifact predates the clause."""
    undeclared = S.check_run_split(
        {"split_fingerprint": "aaaa", "fold": {"c6_3_satisfied": False}},
        current={"fingerprint": "aaaa"},
    )
    gap = {c.check_id: c for c in undeclared.checks}["c6_3_expected_false_declared"]
    assert gap.ok is False and gap.severity == "reporting"
    assert undeclared.ok and not undeclared.reporting_ok

    declared = S.check_run_split(
        {
            "split_fingerprint": "aaaa",
            "fold": S.stamp_c6_3_expected_false(
                {"c6_3_satisfied": False, "n_heldout_blocks": 1},
                citation="ADR-062 (7)",
                why="fold 0 holds out one block",
            ),
        },
        current={"fingerprint": "aaaa"},
    )
    assert declared.ok and declared.reporting_ok, declared.format(verbose=True)


def test_an_artifact_with_no_c6_3_key_emits_NO_expectation_clause() -> None:
    """Same pre-emption as C-4.3 and C8.1: no phantom gap on the archive."""
    rep = S.check_run_split(
        {"split_before": {"fingerprint": "aaaa"}, "split_after": {"fingerprint": "aaaa"}},
        current={"fingerprint": "aaaa"},
    )
    assert not any(c.check_id.startswith("c6_3_expected_false") for c in rep.checks)


def test_the_expected_false_FOLD_SET_is_derived_and_it_is_THREE_folds() -> None:
    """ADR-062 (7) names folds 0 and 1. The partition it states names THREE.

    Fold 2 holds out ``{3, 9, 13}`` - three blocks, below the minimum of 4 - so
    it reports ``c6_3_satisfied: false`` exactly as folds 0 and 1 do. The ADR's
    RULING is unaffected and simply covers one more fold than it names; this is
    recorded here rather than corrected in DECISIONS.md, which is not this
    lead's file.

    The set is DERIVED from the partition. A hand-written ``(0, 1)`` would have
    reproduced the slip and then outlived it - the same shape as the enumerated
    `_SCORING_CODE_MODULES` and the public-tell allowlist.
    """
    assert S.folds_expected_to_fail_c6_3() == (0, 1, 2)
    # ...and the partition itself is a partition: 14 blocks, each exactly once.
    held = [b for blocks in S.LEAVE_FOLD_OUT_BLOCKS.values() for b in blocks]
    assert sorted(held) == list(range(14)), held
    assert len(held) == len(set(held)) == 14


def test_the_derivation_tracks_the_bar_rather_than_restating_it() -> None:
    """Positive control for the derivation: give it a partition where every fold
    clears the bar and it must return NOTHING. A function that returned (0, 1, 2)
    for any input would pass the test above and mean nothing."""
    generous = {k: tuple(range(k * 4, k * 4 + 4)) for k in range(5)}
    assert S.folds_expected_to_fail_c6_3(generous) == ()
    thin = {k: (k,) for k in range(5)}
    assert S.folds_expected_to_fail_c6_3(thin) == (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------
# C-4.2 / C-4.3: THE CLAUSE THAT SAYS "SAMPLED AT BOTH ENDS", AND ITS OWN `not`
#
# `common/splits.py` carries four `not unpaired` sites across the code and
# environment clauses. The audit removed one - the message conditional at what
# is now line 1480 - and 745 tests passed. Nothing read those messages, so a
# clause could report `ok=True` while printing "sampled at ONE end only", or
# report a failure while printing "sampled at BOTH ends".
#
# This is the clause that exists because `eval/metrics.py` was rewritten nine
# minutes into a gate-adjudicating run (ADR-022). Its own report saying the
# opposite of its own verdict is exactly the failure it was built to catch, one
# level up: a reader who trusts the sentence would draw the wrong conclusion,
# and a reader who trusts the flag would not know the sentence disagreed.
#
# Both the FLAG and the SENTENCE are asserted, in both directions, for both
# families. A test on the flag alone leaves the message free to invert.
# --------------------------------------------------------------------------


def _fingerprint_clauses(payload: dict[str, Any]) -> dict[str, tuple[bool, str]]:
    """Run C8's fingerprint clauses over one synthetic artifact, keyed by check id."""
    rep = S.ContractReport(target="synthetic")
    S._add_code_fingerprint_clauses(rep, {"run.json": payload})
    return {c.check_id: (c.ok, c.message) for c in rep.checks}


_BOTH_ENDS = {
    "common_code_before": "aaaa",
    "common_code_after": "aaaa",
    "scoring_code_before": "bbbb",
    "scoring_code_after": "bbbb",
    "environment_before": "cccc",
    "environment_after": "cccc",
}

_ONE_END = {"common_code": "aaaa", "scoring_code": "bbbb", "environment": "cccc"}


def test_a_both_ended_artifact_passes_AND_says_so() -> None:
    """The clause and its sentence agree when the artifact is well-formed."""
    clauses = _fingerprint_clauses(dict(_BOTH_ENDS))
    for check in ("code_sampled_both_ends", "environment_sampled_both_ends"):
        ok, message = clauses[check]
        assert ok, f"{check} failed on an artifact stamped at both ends: {message}"
        assert "BOTH ends" in message, (
            f"{check} passed while reporting {message!r}. A clause whose verdict and whose "
            "sentence disagree is worse than either being wrong alone: one of the two readers "
            "of this report is always misled, and neither can tell which."
        )
        assert "ONE end only" not in message, message


def test_a_one_ended_artifact_FAILS_AND_names_what_is_missing() -> None:
    """The other direction, which is the case C-4.2 was written about."""
    clauses = _fingerprint_clauses(dict(_ONE_END))
    for check in ("code_sampled_both_ends", "environment_sampled_both_ends"):
        ok, message = clauses[check]
        assert not ok, f"{check} passed on an artifact sampled at ONE end: {message}"
        assert "ONE end only" in message, (
            f"{check} failed while reporting {message!r}, which reads as a pass. The C-4.2 "
            "defect is a stamp that reassures; a message that reassures is the same defect "
            "in the report."
        )
        assert "BOTH ends" not in message, message
    assert "run.json:common_code" in clauses["code_sampled_both_ends"][1]
    assert "run.json:environment" in clauses["environment_sampled_both_ends"][1]


def test_a_disagreeing_pair_is_a_HARD_fail_and_a_missing_pair_is_only_reporting() -> None:
    """The two severities are the whole point of C-4.2's tiering, so they are pinned.

    Disagreement is a MEASUREMENT and hard-fails; absence is a bookkeeping gap that
    every pre-clause artifact in `runs/` has, and demoting it would fail the entire
    archive for a property none of it could have had.
    """
    moved = dict(_BOTH_ENDS)
    moved["common_code_after"] = "zzzz"
    moved["environment_after"] = "dddd"
    rep = S.ContractReport(target="synthetic")
    S._add_code_fingerprint_clauses(rep, {"run.json": moved})
    by_id = {c.check_id: c for c in rep.checks}
    assert not by_id["code_agrees_across_run"].ok
    assert by_id["code_agrees_across_run"].severity == S.SEVERITY_FAIL
    assert "MOVED DURING THIS RUN" in by_id["code_agrees_across_run"].message
    assert not by_id["environment_agrees_across_run"].ok
    assert by_id["environment_agrees_across_run"].severity == S.SEVERITY_FAIL

    rep = S.ContractReport(target="synthetic")
    S._add_code_fingerprint_clauses(rep, {"run.json": dict(_ONE_END)})
    by_id = {c.check_id: c for c in rep.checks}
    assert by_id["code_sampled_both_ends"].severity == S.SEVERITY_REPORTING
    assert by_id["environment_sampled_both_ends"].severity == S.SEVERITY_REPORTING


def test_an_artifact_with_no_fingerprints_at_all_emits_no_clause() -> None:
    """The control: the assertions above are about a clause that has to be REACHED.

    An artifact stamping nothing must add no check, so a green report over the
    pre-clause archive is silence rather than a pass.
    """
    rep = S.ContractReport(target="synthetic")
    S._add_code_fingerprint_clauses(rep, {"run.json": {"fingerprint": "4848f491e8d588fa"}})
    assert rep.checks == []
