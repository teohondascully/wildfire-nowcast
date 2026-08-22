"""Unit tests for :mod:`wildfire_nowcast.sim.dashboard`.

WHAT THIS FILE PROTECTS
-----------------------
This module renders the standard per-run diagnostics dashboard, the figure this
lead is required to produce after every gate attempt. Its output is on disk as
``reports/figures/gate_dashboard_g2_record.png``, ``gate_dashboard_m3.png`` and
``dashboard_kincade.png``, and the first of those is the picture of the run G2
was adjudicated on. It measured **0 percent coverage** (212 statements, 70
branches, none executed by any test) when this file was written.

Three of its behaviours are load-bearing rather than cosmetic:

1. **``reliability_curve`` drops empty bins.** Its own docstring names the
   failure it exists to prevent: *"Plotting an empty bin at (0, 0) would draw a
   perfectly-calibrated-looking point supported by no data at all, which is the
   standard way a reliability diagram lies."* Reliability at 1/2/3 h is a
   headline metric in CLAUDE.md, and a reliability diagram is read by eye -- an
   unsupported point at the origin is indistinguishable from a good one.
2. **The C3.3 reportability banner.** The module docstring states the reason:
   *"a screenshot of a dashboard is exactly how a plumbing-only number gets
   quoted in a gate."* If ``reporting_status`` says the norm stats behind the
   numbers do not satisfy C3.3, the figure must carry SMOKE TEST ONLY. A banner
   that said the same thing either way would be a check that cannot fail.
3. **A missing required C6 key exits non-zero.** ``C6_REQUIRED`` names the keys
   the C6 contract requires. Their absence is a contract violation, and the CLI
   reports it BOTH on the figure and in its exit code, so a pipeline cannot
   render a contract-violating run and read success off the process.

WHAT IS NOT TESTED HERE, AND WHY
--------------------------------
Nothing about the visual appearance of the PNG is asserted: colours, layout,
tick placement and DPI are judgement, not contract. The tests reach into the
panel functions and read the artists' own data and text, which is what the
figure is made of; they do not compare rendered pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402

from wildfire_nowcast.sim import dashboard  # noqa: E402
from wildfire_nowcast.sim.dashboard import (  # noqa: E402
    C6_REQUIRED,
    C6Run,
    _panel_members,
    _panel_reliability,
    _reporting_banner,
    load_c6,
    main,
    reliability_curve,
    render_dashboard,
)


def _bin(lead: int, mf: float | None, of: float | None, n: float) -> dict[str, Any]:
    return {"lead_h": lead, "mean_forecast": mf, "observed_frequency": of, "n": n}


def _payload(**over: Any) -> dict[str, Any]:
    """A C6 ``evaluate()`` payload carrying every required key."""
    base: dict[str, Any] = {
        "event": "2019_kincade",
        "n_members": 24,
        "horizon_h": 3,
        "primary_mask": "growth_band",
        "crps_estimator": "empirical",
        "brier_1h": 0.10,
        "brier_2h": 0.12,
        "brier_3h": 0.15,
        "arrival_crps": 1.5,
        "dispersion_ratio": 0.30,
        "best_member_iou": 0.23,
        "reliability_bins": [_bin(1, 0.2, 0.25, 100.0), _bin(1, 0.6, 0.55, 40.0)],
        "diagnostics": {
            "member_growth_cells": [10, 12, 14],
            "truth_growth_cells": 42,
            "mean_pairwise_member_iou": 0.8,
        },
        "by_mask": {
            "growth_band": {
                "brier_by_lead": {"1": 0.11, "2": 0.13, "3": 0.16},
                "dispersion_ratio": 0.29,
                "area_dispersion_ratio": 0.20,
                "arrival_crps": 1.6,
                "reliability_bins": [_bin(1, 0.3, 0.35, 50.0)],
            },
            "domain": {
                "brier_by_lead": {"1": 0.05, "2": 0.06, "3": 0.07},
                "arrival_crps": 1.2,
            },
        },
        "notes": ["dispersion_ratio on a binary field cannot detect ensemble collapse"],
    }
    base.update(over)
    return base


def _run(payload: dict[str, Any] | None = None, label: str = "kernel") -> C6Run:
    p = _payload() if payload is None else payload
    missing = tuple(k for k in C6_REQUIRED if k not in p)
    return C6Run(label, Path("synthetic.json"), p, missing)


# --------------------------------------------------------------------------
# reliability_curve
# --------------------------------------------------------------------------


def test_an_EMPTY_reliability_bin_is_DROPPED_and_never_drawn_at_the_origin() -> None:
    """The failure mode the function's own docstring names.

    C6 emits empty bins with ``n == 0`` and ``mean_forecast is None``. Coerced to
    numbers they become the point (0, 0), which sits exactly on the diagonal and
    reads as perfect calibration supported by nothing. Three flavours of empty
    are present here and all three must vanish; only the one populated bin
    survives.

    WHAT WOULD MAKE THIS FAIL: replacing the guard with ``float(mf or 0.0)``, or
    dropping the ``n <= 0`` clause -- either puts a point on the diagonal that no
    observation supports, and the curve still looks like a curve.
    """
    bins = [
        _bin(1, None, None, 0.0),  # empty, both fields absent
        _bin(1, 0.4, None, 12.0),  # observed frequency undefined
        _bin(1, None, 0.4, 12.0),  # forecast mean undefined
        _bin(1, 0.9, 0.9, 0.0),  # both present but zero support
        _bin(1, 0.5, 0.45, 30.0),  # the only real bin
    ]
    c = reliability_curve(bins, 1)
    assert c["forecast"].tolist() == [0.5]
    assert c["observed"].tolist() == [0.45]
    assert c["n"].tolist() == [30.0]


def test_the_curve_is_restricted_to_ONE_lead_and_is_sorted_by_forecast_probability() -> None:
    """C6 emits one flat list tagged with ``lead_h``; a curve is one lead of it.

    The bins are supplied deliberately out of order and with two leads mixed, so
    a function that ignored ``lead_h`` would draw a zig-zag across both leads and
    a function that skipped the sort would draw the right points joined in the
    wrong order. Both are visible only as a shape, which is exactly what nobody
    checks on a rendered figure.

    WHAT WOULD MAKE THIS FAIL: dropping the ``lead_h`` filter (six points), or
    dropping the ``argsort`` (the same three points in the order 0.8, 0.2, 0.5).
    """
    bins = [
        _bin(1, 0.8, 0.75, 10.0),
        _bin(2, 0.1, 0.05, 99.0),
        _bin(1, 0.2, 0.15, 20.0),
        _bin(2, 0.9, 0.95, 99.0),
        _bin(1, 0.5, 0.45, 30.0),
        _bin(3, 0.3, 0.35, 99.0),
    ]
    c = reliability_curve(bins, 1)
    assert c["forecast"].tolist() == [0.2, 0.5, 0.8]
    assert c["observed"].tolist() == [0.15, 0.45, 0.75]
    assert c["n"].tolist() == [20.0, 30.0, 10.0]
    # A lead with no bins at all yields an empty curve, not an exception.
    assert reliability_curve(bins, 9)["forecast"].size == 0


def test_a_lead_with_only_empty_bins_draws_the_MISSING_notice_and_no_line() -> None:
    """An unsupported panel says so on the figure instead of looking empty.

    A blank reliability panel and a reliability panel whose data were all dropped
    look identical to a reader. The panel writes "no populated bins" when nothing
    was plottable, and that text is the only thing distinguishing the two.

    WHAT WOULD MAKE THIS FAIL: rendering the panel without the ``drew`` flag, or
    letting an all-empty curve through as a zero-length line, which draws
    nothing and says nothing.
    """
    run = _run(_payload(reliability_bins=[_bin(1, None, None, 0.0)], by_mask={}))
    fig, ax = plt.subplots()
    try:
        _panel_reliability(ax, [run], 1)
        assert any("no populated bins" in t.get_text() for t in ax.texts)
        # the diagonal reference line only; no data line
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# The C6 contract keys and the exit code
# --------------------------------------------------------------------------


def test_load_c6_names_every_MISSING_required_key_rather_than_defaulting_it(
    tmp_path: Path,
) -> None:
    """Absence of a required C6 key is a contract violation, not a gap to fill.

    ``load_c6`` records exactly which of ``C6_REQUIRED`` were absent, in contract
    order, and carries them onto the figure rather than raising, because a
    dashboard whose job is to expose problems must be able to render a run that
    has one.

    WHAT WOULD MAKE THIS FAIL: substituting a default for a missing key, or
    reporting only the first absence -- both of which turn a violation into a
    plausible-looking panel.
    """
    payload = _payload()
    payload.pop("arrival_crps")
    payload.pop("dispersion_ratio")
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(payload))

    runs = load_c6(p, label="kernel")
    assert len(runs) == 1
    assert runs[0].missing == ("arrival_crps", "dispersion_ratio")
    assert runs[0].label == "kernel"


def test_a_LIST_of_C6_payloads_is_labelled_per_element_so_two_runs_cannot_merge(
    tmp_path: Path,
) -> None:
    """``{"results": [...]}`` and a bare dict are both accepted, and are not merged.

    When one file carries several payloads each gets its own indexed label, so a
    legend cannot show two different models under one name. This is the aperture
    the module's fairness argument depends on: every model is read through the
    identical reader, and the labels are what keep them apart.

    WHAT WOULD MAKE THIS FAIL: labelling all elements with the shared file stem,
    which draws two curves with one legend entry.
    """
    p = tmp_path / "many.json"
    p.write_text(json.dumps({"results": [_payload(), _payload()]}))
    runs = load_c6(p, label="ellipse")
    assert [r.label for r in runs] == ["ellipse[0]", "ellipse[1]"]

    q = tmp_path / "one.json"
    q.write_text(json.dumps(_payload()))
    assert [r.label for r in load_c6(q)] == ["one"]


def test_the_CLI_exits_NONZERO_when_a_rendered_run_violates_the_C6_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The figure is not the only report; the exit code is machine-readable.

    A pipeline that renders a dashboard and checks the process status must not
    read success on a run missing required C6 keys. The figure still renders --
    refusing to draw would hide the problem -- so the exit code is the part that
    a script can act on.

    WHAT WOULD MAKE THIS FAIL: ``return 0`` at the end of ``main`` regardless of
    ``r.missing``, which leaves a violation visible only to a human looking at
    the picture.
    """
    monkeypatch.setattr(dashboard, "_reporting_banner", lambda fig: None)

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_payload()))
    out = tmp_path / "ok.png"
    assert main([str(good), "--out", str(out)]) == 0
    assert out.is_file()

    broken_payload = _payload()
    broken_payload.pop("brier_3h")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(broken_payload))
    out2 = tmp_path / "bad.png"
    assert main([str(bad), "--out", str(out2)]) == 1
    assert out2.is_file(), "the figure must still be produced, only the status differs"


def test_render_dashboard_REFUSES_an_empty_run_list(tmp_path: Path) -> None:
    """An empty dashboard is a blank page that reads as a dashboard.

    WHAT WOULD MAKE THIS FAIL: returning a figure with no panels instead of
    raising, which writes a PNG that looks like a run with nothing to report.
    """
    with pytest.raises(ValueError):
        render_dashboard([], tmp_path / "empty.png")


# --------------------------------------------------------------------------
# The C3.3 reportability banner
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expect", "forbid"),
    [
        (
            {"reportable": True, "n_train_blocks": 9, "train_folds": [0, 1, 2, 4]},
            "C3.3 OK",
            "SMOKE TEST",
        ),
        (
            {"reportable": False, "n_train_blocks": 1, "reason": "one train block"},
            "SMOKE TEST ONLY",
            "C3.3 OK",
        ),
    ],
)
def test_the_banner_changes_with_C3_3_and_says_DO_NOT_QUOTE_when_it_fails(
    monkeypatch: pytest.MonkeyPatch, status: dict[str, Any], expect: str, forbid: str
) -> None:
    """A screenshot of a dashboard is how a plumbing-only number gets quoted.

    Both directions are asserted from one parametrisation, because a banner that
    printed the same string either way would pass a one-sided test while telling
    a reader nothing. The failing branch must also carry the instruction, not
    only the diagnosis: "Do not quote these numbers".

    WHAT WOULD MAKE THIS FAIL: reading ``st.get("reportable")`` with a default of
    True, which stamps C3.3 OK on a status dict that never answered.
    """
    import wildfire_nowcast.eval.reporting as reporting

    monkeypatch.setattr(reporting, "reporting_status", lambda *a, **k: status)
    fig = plt.figure()
    try:
        _reporting_banner(fig)
        text = " ".join(t.get_text() for t in fig.texts)
        assert expect in text
        assert forbid not in text
        if not status["reportable"]:
            assert "Do not quote these numbers" in text
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# The member-vs-truth panel
# --------------------------------------------------------------------------


def test_the_member_annotation_states_the_SAME_comparator_it_counts_with() -> None:
    """The ensemble-bias readout and its caption have to agree at a tie.

    The panel counts members STRICTLY above truth. With truth = 3 and members
    1, 3, 5 the strict count is 1 and the non-strict count is 2, so the two
    comparators disagree on this fixture by exactly the tied member. A caption
    reading "at or above" over a strictly-above count tells a reader that one of
    three members reached truth when two did, and the direction of that error
    flatters an under-dispersed ensemble, which is the defect this project
    already has.

    (The mismatch was live: the annotation was written with a non-strict
    comparator while the count was strict. The caption was corrected to the
    comparator actually used, so no published count moved.)

    WHAT WOULD MAKE THIS FAIL: changing either the count or the caption without
    the other.
    """
    payload = _payload(diagnostics={"member_growth_cells": [1, 3, 5], "truth_growth_cells": 3})
    fig, ax = plt.subplots()
    try:
        _panel_members(ax, [_run(payload)])
        labels = [t.get_text() for t in ax.texts]
        assert labels == ["1/3 members > truth"]
        # U+2265 GREATER-THAN OR EQUAL TO, spelled by code point rather than
        # written out, so this file does not itself carry the character class
        # the repository is burning down.
        assert chr(0x2265) not in labels[0]
    finally:
        plt.close(fig)


def test_a_COLLAPSED_ensemble_and_a_spread_one_are_distinguishable_on_the_panel() -> None:
    """The panel exists to make collapse visible without a raster.

    Persistence scores ``mean_pairwise_member_iou = 1.000`` and zero area
    dispersion -- total collapse, and a known-good positive control. Drawn, a
    collapsed ensemble is a single stack of coincident points. The assertion is
    on the scattered y values themselves, which is what the eye is being asked to
    read.

    WHAT WOULD MAKE THIS FAIL: plotting the ensemble MEAN instead of the
    members, which makes a collapsed and a spread ensemble identical on the
    figure.
    """
    collapsed = _run(
        _payload(diagnostics={"member_growth_cells": [7, 7, 7], "truth_growth_cells": 40}),
        label="persistence",
    )
    spread = _run(
        _payload(diagnostics={"member_growth_cells": [2, 20, 60], "truth_growth_cells": 40}),
        label="kernel",
    )
    fig, ax = plt.subplots()
    try:
        _panel_members(ax, [collapsed, spread])
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == 2, "one member scatter per run"
        ys = [np.asarray(c.get_offsets())[:, 1].tolist() for c in scatters]
        assert ys[0] == [7.0, 7.0, 7.0]
        assert ys[1] == [2.0, 20.0, 60.0]
        assert len(set(ys[0])) == 1 and len(set(ys[1])) == 3
    finally:
        plt.close(fig)


def test_C6s_own_notes_are_carried_onto_the_figure_verbatim_and_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6's notes are warnings ABOUT the numbers the dashboard plots.

    One of them says ``dispersion_ratio`` on a binary field cannot detect
    ensemble collapse. A figure that plotted that quantity without its caveat
    would be actively misleading, and a figure outlives its caption. Two runs
    carrying the identical note contribute it once; a contract violation adds its
    own note naming the missing keys.

    WHAT WOULD MAKE THIS FAIL: summarising a note instead of reproducing it, or
    dropping the deduplication so one warning is printed once per run and the
    panel area overflows.
    """
    captured: list[Any] = []
    real_close = plt.close
    monkeypatch.setattr(dashboard, "_reporting_banner", lambda fig: None)
    monkeypatch.setattr(dashboard.plt, "close", lambda f: (captured.append(f), real_close(f))[1])

    note = "dispersion_ratio on a binary field cannot detect ensemble collapse"
    p1 = _payload(notes=[note])
    p2 = _payload(notes=[note])
    p2.pop("best_member_iou")
    runs = [_run(p1, "a"), _run(p2, "b")]

    render_dashboard(runs, tmp_path / "d.png")
    assert captured, "the figure was never closed, so nothing was captured"
    text = "\n".join(t.get_text() for t in captured[0].texts)
    assert text.count(note) == 1
    assert "CONTRACT" in text and "best_member_iou" in text
