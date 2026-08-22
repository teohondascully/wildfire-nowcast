"""Unit tests for :mod:`wildfire_nowcast.sim.s5_report`.

WHAT THIS FILE PROTECTS
-----------------------
``build_report`` is the sole computer of ``reports/figures/s5_block5_anatomy.json``,
the artifact behind the S5 result: the G3 dispersion criterion split into an
exact four-factor identity,

    adr = sqrt((M+1)/M) x CV x growth_calibration x truth_shape x relief

published with two validity numbers that are computed HERE and nowhere else:

* **``max_identity_residual``** -- quoted as "residual 3.3e-16 over 96 cells".
  It is the entire warrant that the decomposition is an identity rather than an
  approximation. A residual computed over the wrong set of cells, or over an
  empty set, would certify an identity that was never evaluated.
* **``known_answer_check``** -- quoted as "my window enumeration reproduces C6's
  own persistence denominator to 0.0 exactly on 2 of 4 fires and <4e-15 on the
  other two". That is the check that says the C1-side recomputation and the C6
  record are talking about the same quantity, and it is what makes the whole
  decomposition a measurement rather than an assertion.

The module measured **0 percent coverage** (118 statements, 20 branches) when
this file was written.

HOW THE TENSOR DEPENDENCY IS HANDLED
------------------------------------
``build_report`` reaches the C1 tensor store through three ``blockanatomy``
functions (``block_truth``, ``frontier_rate``, ``support_distance``). Those are
stubbed here with values chosen by hand, which is the point: the truth scale
becomes a KNOWN number, so the identity residual and the known-answer difference
both have closed forms the test states in advance. ``adr_parts`` -- the function
that actually performs the algebra -- is NOT stubbed and runs for real.

WHAT IS NOT TESTED HERE, AND WHY
--------------------------------
``main`` is not covered: it writes into the repository's own
``reports/figures`` directory with no way to redirect it, so running it in a
test would either overwrite a published artifact or require the test to reach
past the CLI it is meant to be testing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from wildfire_nowcast.sim import s5_report
from wildfire_nowcast.sim.blockanatomy import BlockTruth
from wildfire_nowcast.sim.s5_report import _factor_spread, build_report, render

N_MEMBERS = 24
HORIZON = 3
#: sqrt((M+1)/M) with M = 24.
FACTOR = math.sqrt((N_MEMBERS + 1.0) / N_MEMBERS)

#: fire id -> (spatial block id, truth_mean, truth_rms). Block 5's truth scale
#: is deliberately the largest, so an arm that under-predicts it is visible in
#: growth_calibration rather than in the ratio.
FIRES: dict[str, tuple[int, float, float]] = {
    "2020_bobcat": (3, 8.0, 10.0),
    "2020_creek": (4, 6.0, 8.0),
    "2020_czu_lightning_complex": (5, 20.0, 25.0),
    "2020_dolan": (6, 5.0, 6.25),
}

#: bias and scatter give hypot(3, 4) == 5 exactly for persistence on every fire,
#: so the known-answer difference against each fire's truth_rms is exact.
PERSISTENCE_BIAS = 3.0
PERSISTENCE_SCATTER = 4.0
PERSISTENCE_RMS = 5.0


def _growth(adr: float | None, bias: float = 2.0, scatter: float = 1.5) -> dict[str, Any]:
    return {
        "band_area_dispersion_ratio": adr,
        "band_area_error_bias": bias,
        "band_area_error_scatter": scatter,
    }


def _record(models_for: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """A minimal C6 run record with four fires and a fixed arm list."""
    default_models: dict[str, Any] = {
        "persistence": _growth(0.4, PERSISTENCE_BIAS, PERSISTENCE_SCATTER),
        "m6_fair_brier0_s1": _growth(0.55),
        "m7_offstate_s1": _growth(0.90),
        "m7_gate_nofix_s3": _growth(1.03),
        "m6_fair_s1": _growth(0.22),
        "m6_fair_s1__ABL": _growth(0.05),
        "m8_undefined_mean": _growth(None),
    }
    per_fire: dict[str, Any] = {}
    for fid, (block, _mean, _rms) in FIRES.items():
        models = dict(default_models)
        if models_for and fid in models_for:
            models.update(models_for[fid])
        per_fire[fid] = {
            "spatial_block_id": block,
            "models": {k: {"growth_windows": v} for k, v in models.items()},
        }
    return {
        "horizon_h": HORIZON,
        "n_members": N_MEMBERS,
        "split_before": {"fingerprint": "4848f491e8d588fa"},
        "code_fingerprints_agree": {"verdict": "agree"},
        "scope": {
            "train_fire_ids": ["2019_kincade", "2020_glass"],
            "heldout_fire_ids": list(FIRES),
        },
        "per_fire": per_fire,
    }


def _truth(fid: str) -> BlockTruth:
    block, mean, rms = FIRES[fid]
    return BlockTruth(
        fire_id=fid,
        spatial_block_id=block,
        n_windows=100,
        n_growth_windows=40,
        n_units=120,
        truth_mean=mean,
        truth_sd=math.sqrt(max(rms * rms - mean * mean, 0.0)),
        truth_rms=rms,
        n_eff_units=90.0,
        top1_share=0.1,
        top3_share=0.25,
        n_merge_windows=0,
        mean_dominant_component_share=1.0,
        max_band_growth=50,
    )


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the three C1-tensor readers with known answers."""
    monkeypatch.setattr(s5_report, "block_truth", lambda fid, block, **kw: _truth(fid))
    monkeypatch.setattr(
        s5_report,
        "frontier_rate",
        lambda fid, **kw: {"growth_per_frontier_cell": 0.1, "fire_id": fid},
    )
    monkeypatch.setattr(
        s5_report,
        "support_distance",
        lambda train, heldout, **kw: {
            "channels": ["elevation"],
            "blocks": {fid: {"mahalanobis": 1.0 + FIRES[fid][0]} for fid in heldout},
        },
    )


def _build(tmp_path: Path, record: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    p = tmp_path / "results.json"
    p.write_text(json.dumps(record if record is not None else _record()))
    return build_report(p, **kw)


# --------------------------------------------------------------------------
# The known-answer check
# --------------------------------------------------------------------------


def test_the_known_answer_check_compares_the_HYPOT_of_bias_and_scatter_to_truth_RMS(
    tmp_path: Path, stubbed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check that says the C1 recomputation and the C6 record mean the same thing.

    ``rmsE = hypot(bias, scatter)``, so persistence's recorded denominator here is
    exactly 5.0 on every fire. Two of the four fires are given a truth RMS of
    exactly that, and the other two are not, which reproduces the published shape
    of the result ("0.0 exactly on 2 of 4") on a fixture where the answer is known
    in advance.

    WHAT WOULD MAKE THIS FAIL: summing bias and scatter instead of taking their
    hypotenuse -- 7.0 rather than 5.0, which turns two exact agreements into two
    disagreements of 2.0 and would have been reported as a reproduction failure
    of the tensor side rather than of the arithmetic.
    """
    # Two fires match the recorded denominator exactly, two do not.
    matched = ("2020_bobcat", "2020_creek")
    for fid in matched:
        block, mean, _rms = FIRES[fid]
        monkeypatch.setitem(FIRES, fid, (block, mean, PERSISTENCE_RMS))

    out = _build(tmp_path, _record())
    by_fire = {c["fire_id"]: c for c in out["known_answer_check"]}
    assert len(by_fire) == len(FIRES)
    for fid, c in by_fire.items():
        assert c["persistence_rms_err_from_record"] == pytest.approx(PERSISTENCE_RMS)
        assert c["truth_rms_recomputed_from_c1"] == pytest.approx(FIRES[fid][2])
        assert c["abs_diff"] == pytest.approx(abs(PERSISTENCE_RMS - FIRES[fid][2]))
    exact = sorted(f for f, c in by_fire.items() if c["abs_diff"] == 0.0)
    assert exact == sorted(matched)


# --------------------------------------------------------------------------
# The identity residual and the set it is taken over
# --------------------------------------------------------------------------


def test_the_identity_holds_on_every_row_and_the_residual_is_the_MAX_over_them(
    tmp_path: Path, stubbed: None
) -> None:
    """The published warrant for calling the decomposition an identity.

    Every row is rebuilt here from its own reported factors and compared to the
    ``adr`` it came from, then ``max_identity_residual`` is checked to be the
    largest of those rebuilt residuals -- not the first, not the mean. A mean
    would hide one broken cell among 95 sound ones, which is the failure mode a
    residual over 96 cells exists to catch.

    WHAT WOULD MAKE THIS FAIL: reporting the mean or the last residual, or
    dropping ``abs`` so a positive and a negative residual could cancel into a
    reassuring maximum.
    """
    out = _build(tmp_path, _record())
    rows = out["parts"]
    assert rows, "the fixture must produce rows or the residual proves nothing"

    residuals = []
    for r in rows:
        rebuilt = (
            FACTOR
            * float(r["ensemble_cv"])
            * float(r["growth_calibration"])
            * float(r["truth_shape_factor"])
            * float(r["denominator_relief"])
        )
        assert rebuilt == pytest.approx(float(r["adr"]), rel=1e-12, abs=1e-12)
        residuals.append(float(r["identity_residual"]))
        assert residuals[-1] >= 0.0

    # EXACT equality, not approx. The residuals here span 0.0 to 1.1e-16, and
    # pytest.approx carries a default ABSOLUTE tolerance of 1e-12, which is four
    # orders of magnitude larger than the whole spread -- so an approximate
    # comparison of this quantity accepts the max, the min, the mean and zero
    # alike. It was written that way first and a planted ``min`` did not redden
    # it: a check that could not fail, on the number that certifies the identity.
    assert len(set(residuals)) > 1, "the fixture must spread the residuals or max is not a choice"
    assert out["max_identity_residual"] == max(residuals)
    assert out["max_identity_residual"] > min(residuals)
    assert out["max_identity_residual"] < 1e-9
    assert len(residuals) == len(rows)


def test_an_arm_whose_criterion_is_UNDEFINED_is_absent_from_the_residual_set(
    tmp_path: Path, stubbed: None
) -> None:
    """``band_area_dispersion_ratio`` is None at perfect mean calibration.

    That is not hypothetical: it is this lead's own finding, filed against
    ``eval/metrics.py``, that the G3 criterion goes unmeasurable exactly as a
    model gets the first moment right. Such a cell cannot enter the identity, so
    it must be absent from ``parts`` -- and therefore from the count of cells the
    published residual is quoted over.

    WHAT WOULD MAKE THIS FAIL: coercing the missing ratio to 0.0, which admits a
    row whose ``adr`` was never measured and whose residual is trivially zero,
    diluting the maximum with a cell that carries no information.
    """
    out = _build(tmp_path, _record())
    models_seen = {r["model"] for r in out["parts"]}
    assert "m8_undefined_mean" not in models_seen
    # every other non-ablation arm survives, on all four fires
    assert models_seen == {
        "persistence",
        "m6_fair_brier0_s1",
        "m7_offstate_s1",
        "m7_gate_nofix_s3",
        "m6_fair_s1",
    }
    assert len(out["parts"]) == len(models_seen) * len(FIRES)


def test_ablation_arms_are_excluded_from_the_arm_list(tmp_path: Path, stubbed: None) -> None:
    """The published cell count is arms x blocks, and ablations are not arms.

    ``__ABL`` variants are the latent-off controls. Including them would inflate
    the "over 96 cells" denominator with rows that are not members of the arm
    family the identity is quoted for.

    WHAT WOULD MAKE THIS FAIL: dropping the ``endswith("__ABL")`` filter, which
    adds one row per fire and changes the published cell count without changing
    any per-cell number, so nothing else on the page would move.
    """
    out = _build(tmp_path, _record())
    assert not [r for r in out["parts"] if str(r["model"]).endswith("__ABL")]


def test_a_report_with_NO_scorable_cell_RAISES_rather_than_publishing_a_perfect_residual(
    tmp_path: Path, stubbed: None
) -> None:
    """A residual of 0.0 over zero cells is the check that cannot fail.

    If every arm's criterion is undefined, ``parts`` is empty and there is no
    identity to verify. ``build_report`` raises on the empty maximum instead of
    returning a report whose ``max_identity_residual`` reads 0.0 -- which is what
    a perfectly verified identity also reads, and the two would be
    indistinguishable in the artifact.

    WHAT WOULD MAKE THIS FAIL: adding ``default=0.0`` to the ``max``, which
    converts a refusal to measure into the most reassuring number the field can
    hold.
    """
    rec = _record()
    for pf in rec["per_fire"].values():
        for m in pf["models"].values():
            m["growth_windows"]["band_area_dispersion_ratio"] = None
    with pytest.raises(ValueError):
        _build(tmp_path, rec)


def test_the_report_carries_the_split_fingerprint_and_refuses_to_call_itself_a_verdict(
    tmp_path: Path, stubbed: None
) -> None:
    """Provenance travels with the numbers, and the artifact disclaims adjudication.

    Every published number in this project is bound to a split fingerprint, and
    the S5 artifact is bound to the 12-fire archive boundary. The artifact also
    states in its own body that nothing in it is a gate verdict, because a
    machine-readable decomposition of a gate criterion is exactly the file
    someone would quote as one.

    WHAT WOULD MAKE THIS FAIL: dropping either field, which is invisible in every
    number on the page.
    """
    out = _build(tmp_path, _record())
    assert out["split_fingerprint"] == "4848f491e8d588fa"
    assert out["code_fingerprints_agree"] == "agree"
    assert out["n_members"] == N_MEMBERS
    assert out["horizon_h"] == HORIZON
    assert "pass/fail for any gate" in out["not_a_verdict"]
    assert out["focus_arms"] == list(s5_report.FOCUS_ARMS)


def test_with_support_False_skips_the_tensor_walk_and_leaves_the_field_empty(
    tmp_path: Path, stubbed: None
) -> None:
    """The expensive leg is optional and its absence is visible, not defaulted.

    ``support_distance`` walks the training tensors. When it is switched off the
    field is an empty dict, so a reader of the artifact can tell "not computed"
    from "computed and small".

    WHAT WOULD MAKE THIS FAIL: emitting a plausible-looking placeholder instead
    of an empty mapping.
    """
    out = _build(tmp_path, _record(), with_support=False)
    assert out["train_support_distance"] == {}
    on = _build(tmp_path, _record(), with_support=True)
    assert set(on["train_support_distance"]["blocks"]) == set(FIRES)


# --------------------------------------------------------------------------
# _factor_spread
# --------------------------------------------------------------------------


def test_factor_spread_is_a_ratio_of_ABSOLUTE_extrema_and_is_per_arm() -> None:
    """ "That factor varies 4.5x across blocks" is this function, and only this one.

    The published S5 reading is a comparison of spreads: the criterion looks flat
    while ``s2s`` varies 4.5x, and the flattener is ``relief``. That is a
    statement about max/min per arm, so the rows of one arm must not leak into
    another's, and the ratio is taken over magnitudes.

    WHAT WOULD MAKE THIS FAIL: pooling arms (this fixture would read 8.0 for
    both), or taking max-min instead of max/min (7.0 rather than 8.0).
    """
    rows = [
        {"model": "a", "s2s": 0.5},
        {"model": "a", "s2s": 4.0},
        {"model": "a", "s2s": 2.0},
        {"model": "b", "s2s": 1.0},
        {"model": "b", "s2s": 1.5},
    ]
    assert _factor_spread(rows, "a", "s2s") == pytest.approx(8.0)
    assert _factor_spread(rows, "b", "s2s") == pytest.approx(1.5)


def test_factor_spread_returns_NaN_rather_than_dividing_by_a_zero_or_absent_factor() -> None:
    """A spread over a zero minimum is infinite, and infinity is not a measurement.

    ``None`` entries are dropped and a non-positive minimum yields NaN, so a
    degenerate factor cannot be published as an enormous spread. NaN propagates
    into the figure as a gap; ``inf`` would render as the largest bar on the
    chart.

    WHAT WOULD MAKE THIS FAIL: removing the ``min(vals) > 0`` guard, which
    returns ``inf`` for the zero case and raises ``ZeroDivisionError`` never --
    numpy-free float division by 0.0 raises, so the report would abort on an arm
    that merely had a degenerate block.
    """
    zero_min = [{"model": "a", "k": 0.0}, {"model": "a", "k": 2.0}]
    assert math.isnan(_factor_spread(zero_min, "a", "k"))
    assert math.isnan(_factor_spread([{"model": "a", "k": None}], "a", "k"))
    assert math.isnan(_factor_spread([], "a", "k"))


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


def test_render_keeps_a_MISSING_cell_as_a_gap_instead_of_shifting_the_bars_left(
    tmp_path: Path, stubbed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing cell must leave a GAP, not slide the series along the block axis.

    Panel (A) draws two bars per block against block-id tick labels. One
    ``(arm, block)`` cell is deleted from the record here, so the arm has three
    values for four ticks. The placeholder keeps the series the same length as
    the axis; dropping the absent cell instead would put every remaining bar
    under the wrong block label -- a figure that is wrong in the one way a reader
    cannot detect, because it still looks like a complete chart.

    The assertion is on the artists the figure is made of: eight bar patches for
    four blocks, and the missing one drawn at a non-finite height rather than at
    zero, because a zero bar is a measurement and a gap is not.

    WHAT WOULD MAKE THIS FAIL: replacing the NaN placeholder in ``pick`` with a
    comprehension that skips absent blocks.
    """
    import matplotlib.pyplot as plt

    rec = _record()
    del rec["per_fire"]["2020_creek"]["models"]["m6_fair_brier0_s1"]  # block 4
    report = _build(tmp_path, rec)
    blocks = sorted({int(r["spatial_block_id"]) for r in report["parts"]})
    assert blocks == [3, 4, 5, 6]

    captured: list[Any] = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda f: (captured.append(f), real_close(f))[1])

    out = tmp_path / "s5.png"
    render(report, out)
    assert out.is_file() and out.stat().st_size > 0
    assert captured, "the figure was never closed, so nothing was captured"

    panel_a = captured[0].axes[0]
    series = list(panel_a.containers)
    assert len(series) == 2, "one bar series for adr and one for s2s"
    assert [len(s) for s in series] == [len(blocks), len(blocks)]
    assert [t.get_text() for t in panel_a.get_xticklabels()] == [f"block {b}" for b in blocks]
    heights = [p.get_height() for s in series for p in s]
    assert sum(1 for h in heights if not math.isfinite(h)) == 2, "adr and s2s both absent"
    assert 0.0 not in heights, "an absent cell is a gap, never a zero-height bar"


def test_render_REFUSES_a_corpus_with_more_than_four_blocks(tmp_path: Path, stubbed: None) -> None:
    """The palette is pinned to four blocks and the mismatch is loud.

    Panel (B) zips the blocks against a four-colour list with ``strict=True``.
    The corpus of record now holds FIVE held-out blocks, so this renderer cannot
    draw the current fold -- and it says so by raising rather than by recycling a
    colour, which would put two blocks on the page in the same ink. That is a
    real limitation of this module and it is pinned here so it cannot be
    "fixed" by silently truncating the block list.

    WHAT WOULD MAKE THIS FAIL: dropping ``strict=True``, after which a fifth
    block is silently omitted from panel (B) while panels (A), (C) and (D) still
    show it.
    """
    report = _build(tmp_path, _record())
    extra = dict(report["parts"][0])
    extra["spatial_block_id"] = 12
    extra["fire_id"] = "2020_fifth"
    report = {**report, "parts": [*report["parts"], extra]}
    with pytest.raises(ValueError):
        render(report, tmp_path / "five.png")
