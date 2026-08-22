"""Unit tests for ``sim/elmfire_stage.check_truth_pairing``.

WHAT THIS FILE PROTECTS
-----------------------
The current headline of this project is that stock ELMFIRE, the field's standard
operational Rothermel model, ACCELERATES on 4 of 4 scored held-out blocks, more
than our kernel does, and is farther from truth than our kernel on 3 of 4. That
result is a comparison, so it is only worth as much as the claim that the two
sides were scored on the same windows against the same truth. That claim is a
single number in ``runs/e1.json``: 587 of 587 shared windows, 0 truth
disagreements, and ``paired: true``.

``check_truth_pairing`` is the only thing that computes it, its own docstring
says "the whole comparison rests on this", and no test executed it. The failure
that matters is not a wrong count. It is ``paired`` going TRUE on an empty
intersection, which reads as perfect agreement and is the check-that-cannot-fail
class this repository has now catalogued eight times.

Everything here is built from literal rows and a temporary file. No ELMFIRE
binary, no tensor store, no run record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from wildfire_nowcast.sim.elmfire_stage import check_truth_pairing


def _rows(*specs: tuple[str, int, float]) -> list[dict[str, Any]]:
    return [
        {"fire_id": f, "t0": t, "truth_growth": g, "model_growth": g * 0.5} for f, t, g in specs
    ]


def _reference(tmp_path: Path, rows: list[dict[str, Any]], name: str = "ref.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"rows": rows}))
    return p


OURS = _rows(("2020_czu", 10, 4.0), ("2020_czu", 12, 6.5), ("2020_dolan", 8, 1.25))


def test_matching_windows_and_matching_truth_pair(tmp_path: Path) -> None:
    out = check_truth_pairing(OURS, _reference(tmp_path, OURS))
    assert out["paired"] is True
    assert out["n_shared_windows"] == 3
    assert out["n_ours"] == out["n_reference_same_fires"] == 3
    assert out["n_truth_disagreements"] == 0
    assert out["n_only_in_ours"] == out["n_only_in_reference"] == 0


def test_ZERO_DISAGREEMENTS_ON_ZERO_SHARED_WINDOWS_IS_NOT_PAIRED(tmp_path: Path) -> None:
    """The whole reason this file exists.

    Failure condition, in one sentence: a reference whose windows do not overlap
    ours at all, which produces zero truth disagreements because there is nothing
    to disagree about, and which must report ``paired: false`` rather than a
    clean bill of health computed over an empty set.
    """
    disjoint = _rows(("2020_czu", 90, 4.0), ("2020_dolan", 91, 1.25))
    out = check_truth_pairing(OURS, _reference(tmp_path, disjoint))
    assert out["n_shared_windows"] == 0
    assert out["n_truth_disagreements"] == 0, "the fixture is meant to have nothing to compare"
    assert out["paired"] is False, (
        "an empty intersection reported as paired. Zero disagreements over zero "
        "shared windows is the absence of evidence, not evidence of agreement."
    )


def test_one_truth_disagreement_is_found_and_named(tmp_path: Path) -> None:
    """A silent truth mismatch would make the two sides different experiments."""
    theirs = _rows(("2020_czu", 10, 4.0), ("2020_czu", 12, 9.0), ("2020_dolan", 8, 1.25))
    out = check_truth_pairing(OURS, _reference(tmp_path, theirs))
    assert out["n_shared_windows"] == 3
    assert out["n_truth_disagreements"] == 1
    assert out["paired"] is False
    first = out["first_disagreements"][0]
    assert first["key"] == ["2020_czu", 12]
    assert (first["ours"], first["reference"]) == (6.5, 9.0)


def test_the_truth_comparison_is_EXACT_and_not_merely_close(tmp_path: Path) -> None:
    """Truth is read from the same store by both sides, so it must be bit-identical.

    Failure condition, in one sentence: a reference truth differing in the last
    representable bit, which a default ``math.isclose`` tolerance would forgive,
    and which cannot arise from reading the same labels twice. Any difference at
    all means the two sides took truth from different places, and forgiving a
    small one is how a systematically different window set gets through.
    """
    nudged = 6.5 + math.ulp(6.5)
    assert nudged != 6.5 and math.isclose(nudged, 6.5)  # the default tolerance forgives it
    theirs = _rows(("2020_czu", 10, 4.0), ("2020_czu", 12, nudged), ("2020_dolan", 8, 1.25))
    out = check_truth_pairing(OURS, _reference(tmp_path, theirs))
    assert out["n_truth_disagreements"] == 1, (
        "a one-ULP truth difference was forgiven, so the pairing check is running "
        "with a tolerance and can no longer see a different window set"
    )
    assert out["paired"] is False


def test_a_reference_covering_fires_we_did_not_run_still_pairs(tmp_path: Path) -> None:
    """The real case: the reference holds five held-out fires and ELMFIRE scored four.

    The reference is restricted to the fires present in our rows before the
    counts are compared. Without that restriction the delivered four-block result
    could never report ``paired: true``, and the honest partial delivery would be
    indistinguishable from a broken one.
    """
    theirs = OURS + _rows(("2020_creek", 3, 12.0), ("2020_creek", 5, 14.0))
    out = check_truth_pairing(OURS, _reference(tmp_path, theirs))
    assert out["n_reference_same_fires"] == 3, "the reference was not restricted to our fires"
    assert out["n_shared_windows"] == 3
    assert out["paired"] is True


def test_a_window_only_we_scored_breaks_the_pairing(tmp_path: Path) -> None:
    """Scoring a window the reference never scored is a different window set."""
    theirs = _rows(("2020_czu", 10, 4.0), ("2020_czu", 12, 6.5))
    out = check_truth_pairing(OURS, _reference(tmp_path, theirs))
    assert out["n_only_in_ours"] == 1
    assert out["n_shared_windows"] == 2
    assert out["n_truth_disagreements"] == 0
    assert out["paired"] is False, (
        "an extra window on our side left the pairing green because every SHARED "
        "window agreed; the counts are part of the claim, not decoration"
    )


def test_the_payload_counts_add_up_to_the_denominators_the_claim_is_quoted_with(
    tmp_path: Path,
) -> None:
    """ "587 of 587" is two numbers, and they must be arithmetically consistent.

    Failure condition, in one sentence: any pair of row sets that partly overlap,
    on which the shared count plus the ours-only count must equal our total and
    the shared count plus the reference-only count must equal the restricted
    reference total. A published ratio whose parts do not sum to its denominator
    is unreadable, and a reader has no other way to check it.
    """
    theirs = _rows(("2020_czu", 10, 4.0), ("2020_czu", 99, 1.0), ("2020_creek", 3, 12.0))
    out = check_truth_pairing(OURS, _reference(tmp_path, theirs))
    for field in (
        "n_ours",
        "n_reference_same_fires",
        "n_shared_windows",
        "n_only_in_ours",
        "n_only_in_reference",
        "n_truth_disagreements",
    ):
        assert isinstance(out[field], int), f"{field} is not an int"
    assert out["n_shared_windows"] + out["n_only_in_ours"] == out["n_ours"]
    assert out["n_shared_windows"] + out["n_only_in_reference"] == out["n_reference_same_fires"]
    assert (out["n_shared_windows"], out["n_only_in_ours"], out["n_only_in_reference"]) == (1, 2, 1)
    assert out["paired"] is False


def test_AN_EMPTY_RUN_PAIRS_WITH_NOTHING(tmp_path: Path) -> None:
    """The one input on which every other clause of ``paired`` is satisfied.

    Failure condition, in one sentence: a fire that produced no scored rows at
    all, for instance one whose windows were all filtered out or whose run was
    killed, against which the shared set, our set and the restricted reference
    set are all empty and therefore all equal, so every count clause agrees and
    only the emptiness guard stands between that and ``paired: true``.

    This is not hypothetical. The ELMFIRE comparison delivered four of five
    held-out blocks and the fifth was stopped deliberately, so an empty row list
    for a fire is a state this code path can actually be handed.
    """
    out = check_truth_pairing([], _reference(tmp_path, OURS))
    assert out["n_ours"] == 0
    assert out["n_reference_same_fires"] == 0, "the reference is not restricted to our fires"
    assert out["n_shared_windows"] == 0
    assert out["n_truth_disagreements"] == 0
    assert out["n_only_in_ours"] == out["n_only_in_reference"] == 0
    assert out["paired"] is False, (
        "an empty run reported as paired with the reference. Every count clause is "
        "satisfied by 0 == 0 == 0, so this is exactly the shape of check that passes "
        "because it had nothing to look at."
    )
