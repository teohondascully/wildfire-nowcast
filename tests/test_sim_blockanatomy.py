"""S5 block anatomy: three outcomes, never two.

``sim/blockanatomy.py``'s ``main`` used to publish ``max_identity_residual:
0.0`` and exit ``0`` when it had NO ROWS TO CHECK. ``max(..., default=0.0)``
and ``all([])`` each have a benign answer for "nothing", so an empty input
produced byte-for-byte the same verdict as a perfect result: same JSON field,
same value, same exit code. Neither a reader of the artifact nor a CI job could
tell "every block agrees to machine precision" from "no block was examined",
and the artifact is what a reader trusts when they will not read the code.

These tests pin the DISTINCTION rather than the happy path. Each one is a
discriminating pair or triple: the same call has to give different answers for
"rows examined and all agreed", "rows examined and something disagreed" and
"no rows examined". A test that only asserted the empty case fails would be
satisfied by a module that refuses everything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from wildfire_nowcast.sim import blockanatomy as BA

# --------------------------------------------------------------------------
# builders - a C6 run record's SHAPE, with nothing in it that needs a tensor
# --------------------------------------------------------------------------

_BIAS = -30.0
_SCATTER = 40.0
#: ``hypot(_BIAS, _SCATTER)``, exactly, on a 3-4-5 triangle. The known-answer
#: check compares persistence's recorded RMS error against truth's own RMS
#: scale recomputed from C1, so a stub block truth carrying this value makes
#: the check AGREE by construction and any disagreement in a test below is the
#: thing that test planted, not float noise.
_TRUTH_RMS = 50.0


def _decomposition(
    *,
    n_rows: int,
    n_checks: int,
    residual: float = 0.0,
    checks_agree: bool = True,
) -> dict[str, Any]:
    """A ``decompose_record`` return value with a chosen number of rows."""
    return {
        "results_path": "<in-memory>",
        "horizon_h": 3,
        "n_members": 24,
        "stride": 2,
        "split_fingerprint": "b3e5dadad01eaef9",
        "code_fingerprints_agree": {"verdict": "OK"},
        "known_answer_check": [
            {"fire_id": f"fire_{i}", "agrees": checks_agree} for i in range(n_checks)
        ],
        "block_truth": {},
        "parts": [
            {"model": "m", "fire_id": f"fire_{i}", "identity_residual": residual}
            for i in range(n_rows)
        ],
    }


def _growth_block(adr: float | None) -> dict[str, Any]:
    return {
        "band_area_dispersion_ratio": adr,
        "band_area_error_bias": _BIAS,
        "band_area_error_scatter": _SCATTER,
    }


def _record(adr: float | None) -> dict[str, Any]:
    """A minimal C6 run record. One fire, one model, one growth stratum."""
    return {
        "horizon_h": 3,
        "n_members": 24,
        "split_before": {"fingerprint": "b3e5dadad01eaef9"},
        "code_fingerprints_agree": {"verdict": "OK"},
        "per_fire": {
            "2020_dolan": {
                "spatial_block_id": 7,
                "models": {"persistence": _growth_block_holder(adr)},
            }
        },
    }


def _growth_block_holder(adr: float | None) -> dict[str, Any]:
    return {"growth_windows": _growth_block(adr)}


def _stub_block_truth(
    fire_id: str,
    spatial_block_id: int,
    **_: Any,
) -> BA.BlockTruth:
    """``block_truth`` without the C1 tensor read, and nothing else stubbed.

    The row-building loop in ``decompose_record`` is the REAL one; only the
    tensor store is replaced. That is what makes the zero-row test below a
    statement about a reachable path rather than about a mock.
    """
    return BA.BlockTruth(
        fire_id=fire_id,
        spatial_block_id=int(spatial_block_id),
        n_windows=12,
        n_growth_windows=8,
        n_units=24,
        truth_mean=30.0,
        truth_sd=40.0,
        truth_rms=_TRUTH_RMS,
        n_eff_units=20.0,
        top1_share=0.1,
        top3_share=0.3,
        n_merge_windows=0,
        mean_dominant_component_share=1.0,
        max_band_growth=90,
    )


# --------------------------------------------------------------------------
# the three outcomes
# --------------------------------------------------------------------------


def test_no_rows_examined_is_not_the_same_verdict_as_every_row_agreeing() -> None:
    agreed = BA.annotate_for_publication(_decomposition(n_rows=4, n_checks=2))
    empty = BA.annotate_for_publication(_decomposition(n_rows=0, n_checks=2))

    agreed_code, _ = BA.publication_verdict(agreed)
    empty_code, empty_why = BA.publication_verdict(empty)

    assert agreed_code == BA.EXIT_OK == 0, agreed_code
    assert empty_code == BA.EXIT_NOTHING_EXAMINED, empty_code
    assert empty_code != agreed_code, (
        "zero rows and four agreeing rows returned the same exit code, which is the "
        "defect: an empty input reads as a perfect result"
    )
    assert "max_identity_residual" not in empty, (
        "a residual computed over no rows must not appear in the artifact at all; "
        f"got {empty.get('max_identity_residual')!r}"
    )
    assert empty["n_rows"] == 0
    assert "NOT a pass" in empty_why, empty_why


def test_no_known_answer_checks_is_also_not_a_pass() -> None:
    """``all([])`` is the same defect wearing the other name."""
    checked = BA.annotate_for_publication(_decomposition(n_rows=4, n_checks=2))
    unchecked = BA.annotate_for_publication(_decomposition(n_rows=4, n_checks=0))

    assert BA.publication_verdict(checked)[0] == BA.EXIT_OK
    assert BA.publication_verdict(unchecked)[0] == BA.EXIT_NOTHING_EXAMINED
    assert "known_answer_check_passed" not in unchecked, (
        "a known-answer verdict over zero checks must not be published as True"
    )
    assert checked["known_answer_check_passed"] is True


def test_a_disagreeing_row_is_distinguishable_from_both_other_outcomes() -> None:
    """Three codes for three states. Refusing everything would fail this."""
    agreed = BA.annotate_for_publication(_decomposition(n_rows=3, n_checks=3))
    disagreed = BA.annotate_for_publication(_decomposition(n_rows=3, n_checks=3, residual=1e-3))
    failed_check = BA.annotate_for_publication(
        _decomposition(n_rows=3, n_checks=3, checks_agree=False)
    )
    empty = BA.annotate_for_publication(_decomposition(n_rows=0, n_checks=3))

    codes = {
        "agreed": BA.publication_verdict(agreed)[0],
        "residual_too_big": BA.publication_verdict(disagreed)[0],
        "check_failed": BA.publication_verdict(failed_check)[0],
        "nothing_examined": BA.publication_verdict(empty)[0],
    }
    assert codes["agreed"] == BA.EXIT_OK
    assert codes["residual_too_big"] == BA.EXIT_DISAGREED
    assert codes["check_failed"] == BA.EXIT_DISAGREED
    assert codes["nothing_examined"] == BA.EXIT_NOTHING_EXAMINED
    assert len(set(codes.values())) == 3, codes


def test_a_non_finite_residual_is_not_agreement() -> None:
    """``rms_err == 0`` makes ``relief`` infinite and the rebuilt ratio NaN.

    ``nan > tol`` is False, so a residual that could not be computed would have
    walked past the threshold as if it had agreed.
    """
    nan_row = BA.annotate_for_publication(_decomposition(n_rows=2, n_checks=2, residual=math.nan))
    assert nan_row["n_non_finite_residuals"] == 2
    code, why = BA.publication_verdict(nan_row)
    assert code == BA.EXIT_DISAGREED, code
    assert "not finite" in why, why

    finite = BA.annotate_for_publication(_decomposition(n_rows=2, n_checks=2))
    assert finite["n_non_finite_residuals"] == 0
    assert BA.publication_verdict(finite)[0] == BA.EXIT_OK


# --------------------------------------------------------------------------
# the artifact on disk, and the path that produces an empty one
# --------------------------------------------------------------------------


def test_main_writes_no_artifact_for_a_residual_it_did_not_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pair: an examined run writes a file, an unexamined run writes none."""
    good = tmp_path / "good.json"
    monkeypatch.setattr(
        BA, "decompose_record", lambda *a, **k: _decomposition(n_rows=6, n_checks=2)
    )
    assert BA.main(["--results", "ignored.json", "--out", str(good)]) == BA.EXIT_OK
    published = json.loads(good.read_text())
    assert published["n_rows"] == 6
    assert published["max_identity_residual"] == 0.0

    nothing = tmp_path / "nothing.json"
    monkeypatch.setattr(
        BA, "decompose_record", lambda *a, **k: _decomposition(n_rows=0, n_checks=2)
    )
    code = BA.main(["--results", "ignored.json", "--out", str(nothing)])
    assert code == BA.EXIT_NOTHING_EXAMINED, code
    assert not nothing.exists(), (
        "an artifact was published for a run that examined nothing; its "
        f"contents: {nothing.read_text()}"
    )


def test_the_zero_row_path_is_reachable_through_the_real_row_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a mock: ``decompose_record``'s own skip rule produces zero rows.

    A run record whose only growth stratum has ``band_area_dispersion_ratio:
    null`` - a block C6 could not compute a ratio for - is skipped by the loop
    itself, and the decomposition comes back with an empty ``parts``.
    """
    monkeypatch.setattr(BA, "block_truth", _stub_block_truth)

    with_ratio = tmp_path / "with_ratio.json"
    with_ratio.write_text(json.dumps(_record(1.25)))
    populated = BA.decompose_record(with_ratio)
    assert len(populated["parts"]) == 1, populated["parts"]
    assert populated["known_answer_check"][0]["agrees"] is True

    without = tmp_path / "without_ratio.json"
    without.write_text(json.dumps(_record(None)))
    empty = BA.decompose_record(without)
    assert empty["parts"] == [], empty["parts"]
    assert len(empty["known_answer_check"]) == 1, (
        "the known-answer check must still have run; if it did not, this test "
        "would be exercising a different emptiness than the one it names"
    )

    out = tmp_path / "refused.json"
    code = BA.main(["--results", str(without), "--out", str(out)])
    assert code == BA.EXIT_NOTHING_EXAMINED, code
    assert not out.exists()

    ok = tmp_path / "published.json"
    assert BA.main(["--results", str(with_ratio), "--out", str(ok)]) == BA.EXIT_OK
    assert json.loads(ok.read_text())["n_rows"] == 1
