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
    """Write a ``norm_stats.json``. ``None`` OMITS the key — that is a defect to plant."""
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
# C0 — one implementation, and it must reproduce the fingerprint of record
# --------------------------------------------------------------------------


@pytest.mark.skipif(not fires_dir().is_dir(), reason="no built fires on this machine")
def test_common_and_eval_fingerprints_agree_byte_for_byte() -> None:
    """C0: modelling's ``eval/`` copy and this one must never disagree.

    Re-homing a function under C0 is only safe if it is the SAME function. If
    this ever fails, one of the two changed and every artifact stamped by the
    other is unverifiable — which is worse than the duplication itself.
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
#: ``FINGERPRINT_PRE_D6`` — 12 fires, ``train_folds [0, 1, 2, 4]``, 4 held-out
#: blocks {3,4,5,6}. **Everything produced before the D6 corpus swap is bound to
#: this value and stays bound to it:** G2's record of adjudication
#: (``runs/baselines-20260808-095003``, ADR-021) and ALL FOUR G3 attempts
#: (ADR-027 / ADR-032 / ADR-034 / ADR-037, i.e. M5, M6, M7 and M8, including the
#: `m8_asym` candidate and the 2x2 factorial). No number stamped with it may be
#: quoted beside a number stamped with the one below — that is what C8's
#: ``matches_current`` reporting clause is for.
FINGERPRINT_PRE_D6 = "4848f491e8d588fa"

#: ``FINGERPRINT_OF_RECORD`` — the CURRENT split. 21 fires (ADR-037 (7) authorised
#: the swap, ADR-038 verified it), 14 spatial blocks, ``train_folds [0, 1, 2, 4]``
#: = 16 fires / 9 blocks, held out fold 3 = 5 fires / 5 blocks {4,5,6,7,12}.
#: **Nothing has been scored under it yet** — M9 will be the first.
FINGERPRINT_OF_RECORD = "b3e5dadad01eaef9"


@pytest.mark.skipif(
    not (fires_dir().is_dir() and norm_stats_path().is_file()),
    reason="no built fires on this machine",
)
def test_the_fingerprint_of_record_is_reproduced() -> None:
    """The split on disk must reproduce :data:`FINGERPRINT_OF_RECORD`.

    Pinned deliberately. If the split legitimately moves this test fails and the
    number in ADR-015/STATE.md must be updated in the same commit — a fingerprint
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
# C8 — the hard fail
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
# [v2.11] C-4.2 — a code fingerprint must be sampled BEFORE *and* AFTER
# --------------------------------------------------------------------------


def test_a_code_fingerprint_is_NOT_a_split_fingerprint() -> None:
    """The FALSE POSITIVE this clause exposed, pinned so it cannot come back.

    ``common_code_before``/``_after`` and ``scoring_code`` are each a dict with
    their own ``fingerprint`` key. Collecting those as SPLIT stamps made
    ``internally_consistent`` report "MORE THAN ONE split fingerprint" and HARD
    FAIL on every run in the repo — including the G2 record of ADR-021, whose
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
    """A checkpoint trained under split A, evaluated under split B — ADR-015 exactly."""
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
# C3.1 — overlapping fires MUST share a fold
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
# [A14] C3 — DECLARED train/held-out membership, by fire id (ADR-038 (6))
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
    so its ``train_fire_ids`` and ``heldout_fire_ids`` cannot share a member —
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
    most consequential kind — the normalisation is fitted on a fire that is then
    scored — and until now literally nothing in the repo read the two keys.
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

    ADR-038 (1) measured this exact hazard at 9.76% of train mass —
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
    """C8: *every run stamps* ``split_fingerprint`` — structurally, not by memory.

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
# C-4.3 [v2.12] — the interpreter ENVIRONMENT is in C-4's frozen set (ADR-024)
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
    installing anything — installing a package to test the no-installing clause
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
    """C-4.3's own sentence, made executable — and its own check id, not code's."""
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
    split fingerprint" and hard-fail every run that stamps one — which, now that
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
    environment did not change" — a green check standing in for a measurement
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
    assert set(S.fingerprints_in(meta).values()) == {
        meta["split_fingerprint"]["fingerprint"]
    }
