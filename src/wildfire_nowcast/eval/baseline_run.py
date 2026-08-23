"""Leave-fire-out evaluation of the C5 baselines through C6. The G2 floor.

This module produces the numbers that G2 has to beat. It exists as a module and
not a notebook because every one of its honesty constraints is a line of code
that must not be skippable:

* **Folds come from ``data/norm_stats.json``, never from an argument.** The
  train set is whatever the normalisation was fitted on; anything else is
  leakage through the back door, because the stats a model consumes already
  saw those fires.
* **The ellipse is fitted on TRAIN windows only** and its fitted scale is
  recorded with the fire ids it saw, so a violation is auditable after the fact.
* **Held-out SPATIAL BLOCKS are counted, not fires** (C3.1). With the five P0
  fires the held-out fold contains Kincade and Glass, which are the SAME block
  (Sonoma/Napa). Two fires, one landscape, n = 1. Every
  emitted artifact carries ``n_heldout_blocks`` next to every score, because
  "evaluated on 2 held-out fires" reads like n = 2 and it is not.
* **Every score is emitted twice**: over all windows, and over the
  growth-conditioned stratum. 51-91% of GOFER hours are at BITWISE zero growth,
  median ~0.79; :mod:`wildfire_nowcast.eval.masks` carries the provenance. On the
  pooled all-windows score
  persistence is exactly right ~4 times in 5, so a pooled Brier is mostly a
  measure of how often nothing happened. Reporting the pooled number alone
  would make the floor look like a ceiling. The growth stratum is not a
  headline either (it conditions on the outcome) -- the pair is the result.
* **The ``growth_band`` mask is carried alongside the domain mask.** Most of the
  domain is far-field cells no model was ever uncertain about.

Run it::

    python -m wildfire_nowcast.eval.baseline_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildfire_nowcast.common import dispersion as g3
from wildfire_nowcast.common.calibration import GATE_CRITERION_KEY as CALIBRATION_GATE_KEY
from wildfire_nowcast.common.calibration import GATE_MASK as CALIBRATION_GATE_MASK
from wildfire_nowcast.common.dispersion import FIRST_MOMENT_KEY
from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY, REPORTED_ONLY_KEY
from wildfire_nowcast.common.paths import fire_tensor_path, fires_dir, norm_stats_path
from wildfire_nowcast.common.pooling import equal_block_mean
from wildfire_nowcast.common.runs import create_run_dir
from wildfire_nowcast.common.zarr_io import open_tensor, read_manifest, read_norm_stats
from wildfire_nowcast.eval.metrics import aggregate, evaluate
from wildfire_nowcast.eval.reporting import (
    assert_model_split_matches,
    assert_reportable,
    assert_split_unchanged,
    check_common_code_unchanged,
    common_code_fingerprint,
    scoring_code_fingerprint,
    split_fingerprint,
)
from wildfire_nowcast.eval.validity import baseline_validity, window_ignition_counts
from wildfire_nowcast.model.baselines import EllipseBaseline, PersistenceBaseline
from wildfire_nowcast.model.inputs import iter_windows

__all__ = ["FireSplit", "load_splits", "run_baselines", "g2_per_horizon", "G2_METRICS", "main"]

DEFAULT_HORIZON = 3
DEFAULT_MEMBERS = 24
DEFAULT_SEED = 20260807

#: Baselines whose zero ignition is DEFINITIONAL, declared explicitly so the
#: C6.2 check is never talked out of firing by inference (eval/validity.py).
NULL_MODELS = frozenset({"persistence"})


@dataclass(frozen=True)
class FireSplit:
    """One fire, with the two labels that decide how it may be used."""

    fire_id: str
    cv_fold: int
    spatial_block_id: int
    n_hours: int
    is_train: bool

    @property
    def role(self) -> str:
        return "train" if self.is_train else "heldout"


def load_splits(train_folds: Sequence[int]) -> list[FireSplit]:
    """Every built fire, labelled train/held-out by its C2 ``cv_fold``."""
    splits: list[FireSplit] = []
    for manifest_path in sorted(fires_dir().glob("*/manifest.json")):
        m = read_manifest(manifest_path)
        fold = int(m["cv_fold"])
        splits.append(
            FireSplit(
                fire_id=str(m["fire_id"]),
                cv_fold=fold,
                spatial_block_id=int(m["spatial_block_id"]),
                n_hours=int(m["n_hours"]),
                is_train=fold in set(int(f) for f in train_folds),
            )
        )
    if not splits:
        raise FileNotFoundError(f"no fires under {fires_dir()}")
    return splits


def _windows_for(fire_id: str, horizon_h: int, stride: int) -> list[Any]:
    ds = open_tensor(fire_tensor_path(fire_id))
    try:
        return list(iter_windows(ds, horizon_h, stride=stride, fire_id=fire_id))
    finally:
        ds.close()


def _zero_growth_fraction(windows: Iterable[Any]) -> tuple[int, int]:
    ws = list(windows)
    zero = sum(1 for w in ws if w.truth_growth_cells() == 0)
    return zero, len(ws)


def _score(
    model: Any,
    windows: Sequence[Any],
    n_members: int,
    horizon_h: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-window C6 results and per-window C6.2 ignition counts.

    Returns the LISTS, not aggregates: :func:`aggregate` pools sufficient
    statistics and is deliberately not idempotent (it emits no ``_pool``), so
    aggregating twice is impossible rather than silently wrong. Pooling happens
    exactly once, at the level being reported.
    """
    per_window: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    for i, w in enumerate(windows):
        samples = model.predict(w.x0, w.static, w.weather, n_members, horizon_h, seed + i)
        per_window.append(
            evaluate(
                samples,
                w.truth,
                x0=w.x0,
                leads=tuple(range(1, horizon_h + 1)),
                meta={"fire_id": w.fire_id, "t0": w.t0},
            )
        )
        counts.append(window_ignition_counts(samples, w.truth, w.x0))
    return per_window, counts


#: [M5] Every key that is a GATE CRITERION, imported from the modules that DEFINE
#: the gates rather than spelled out here. :func:`assert_gate_criteria_present`
#: hard-fails the runner if any of them is missing from a headline.
#:
#: WHY THIS EXISTS AND WHY IT IS A CLASS FIX RATHER THAN A PATCH. Twice now a
#: curated allow-list has stood between a gate's criterion and the file the gate
#: is adjudicated from: C6.4's `best_member_iou_shape_masked` (found by
#: infra, A11) and ADR-020's `calibration_error` (found by sim, S3).
#: Both times the run would have exited 0 while emitting a table that did not
#: contain the number the gate turns on. A hand-maintained list that must be
#: updated every time a gate changes IS the defect, so the invariant is now
#: enforced mechanically and derived from `common/`: adding a criterion in
#: `common/` and forgetting it here now RAISES instead of going quiet.
GATE_CRITERION_HEADLINE_KEYS: tuple[str, ...] = (
    f"band_{GATE_CRITERION_KEY}_by_horizon",  # G2 shape (C6.4)
    f"band_{CALIBRATION_GATE_KEY}_by_horizon",  # G3 calibration (ADR-020 4b)
    f"band_{CALIBRATION_GATE_KEY}_n_scored_by_horizon",  # ... and its denominator
    "band_area_dispersion_ratio",  # G3 dispersion (C6.1 / ADR-011)
    # [M9] G3's FIRST MOMENT (ADR-039 (5)). Added the moment the condition became
    # live, which is the whole discipline this list encodes: a criterion added in
    # `common/` and forgotten here now RAISES. `common.dispersion` DEFINES the
    # condition, so the key is imported from there and never spelled.
    f"band_{FIRST_MOMENT_KEY}",
)


def assert_gate_criteria_present(headline: Mapping[str, Any]) -> None:
    """HARD FAIL if a headline omits any gate criterion. Never a warning.

    C-1's corollary: an unevaluable guard is strictly worse than a declared-weak
    one. A gate adjudicated from a file that does not contain its own criterion
    is exactly that, and it has happened twice.
    """
    missing = [k for k in GATE_CRITERION_HEADLINE_KEYS if k not in headline]
    if missing:
        raise AssertionError(
            f"headline is missing GATE CRITERIA {missing}. These keys are imported from "
            "common/iou_terms.py and common/calibration.py, which DEFINE the gates, so a "
            "criterion cannot be added there and silently dropped here. Add it to _headline "
            "in the same edit — do NOT remove it from this list."
        )


def _headline(result: dict[str, Any], horizon_h: int) -> dict[str, Any]:
    """The few numbers a gate should actually cite, domain AND growth_band.

    Curated, but NO LONGER curated where it matters: every GATE CRITERION is
    asserted present by :func:`assert_gate_criteria_present` against a list
    derived from the modules that define the gates. A metric C6 emits and this
    function does not name still does not exist downstream - that part is
    deliberate, since the artifact should not be every key C6 can compute - but
    a criterion a GATE turns on can no longer go missing quietly. It has twice.
    """
    band = result.get("by_mask", {}).get("growth_band", {})
    domain = result.get("by_mask", {}).get("domain", {})
    band_brier = band.get("brier_by_lead", {})
    band_iou_h = band.get("best_member_iou_by_horizon") or []
    rel = band.get("reliability_summary") or {}
    band_area = band.get("area_error_decomposition") or {}
    domain_area = domain.get("area_error_decomposition") or {}
    band_fm = band.get("first_moment") or {}
    domain_fm = domain.get("first_moment") or {}

    def _by_h(key: str) -> dict[str, Any]:
        seq = band.get(key) or []
        return {str(h): (seq[h - 1] if len(seq) >= h else None) for h in range(1, horizon_h + 1)}

    out = {
        # -- [v2.10] C6.4: the GATE CRITERION and the four terms that audit it.
        # `best_member_iou` stays REPORTED and bit-identical above; carrying the
        # split lets a reader reconstruct it (silence + shape == reported, at
        # EVERY horizon) instead of taking the correction on trust. The
        # `_n_windows_` row is the gate term's DENOMINATOR: it is undefined on a
        # window with no truth growth, so it pools over fewer windows than
        # `best_member_iou` does, and a 1 h number resting on a third of the
        # sample must be visible (C6.3's "state the count" applied to a metric).
        f"band_{GATE_CRITERION_KEY}_by_horizon": _by_h(f"{GATE_CRITERION_KEY}_by_horizon"),
        f"band_{GATE_CRITERION_KEY}_n_windows_by_horizon": _by_h(
            f"{GATE_CRITERION_KEY}_n_windows_by_horizon"
        ),
        f"band_{GATE_CRITERION_KEY}": band.get(GATE_CRITERION_KEY),
        "band_best_member_iou_silence_by_horizon": _by_h("best_member_iou_silence_by_horizon"),
        "band_best_member_iou_shape_by_horizon": _by_h("best_member_iou_shape_by_horizon"),
        "band_best_member_iou_silent_floor_by_horizon": _by_h(
            "best_member_iou_silent_floor_by_horizon"
        ),
        "band_best_member_iou_silence": band.get("best_member_iou_silence"),
        "band_best_member_iou_shape": band.get("best_member_iou_shape"),
        "band_best_member_iou_silent_floor": band.get("best_member_iou_silent_floor"),
        "band_best_member_iou_gate_criterion": band.get("best_member_iou_gate_criterion"),
        # -- ADR-015 (3): the per-horizon block. G2 is adjudicated from THESE. --
        "band_brier_by_horizon": {str(h): band_brier.get(h) for h in range(1, horizon_h + 1)},
        "band_best_member_iou_by_horizon": {
            str(h): (band_iou_h[h - 1] if len(band_iou_h) >= h else None)
            for h in range(1, horizon_h + 1)
        },
        "band_reliability_by_horizon": {
            str(h): rel.get(str(h), {}).get("reliability") for h in range(1, horizon_h + 1)
        },
        "band_resolution_by_horizon": {
            str(h): rel.get(str(h), {}).get("resolution") for h in range(1, horizon_h + 1)
        },
        "band_ece_by_horizon": {
            str(h): rel.get(str(h), {}).get("ece") for h in range(1, horizon_h + 1)
        },
        "brier_1h": result.get("brier_1h"),
        f"brier_{horizon_h}h": result.get(f"brier_{horizon_h}h"),
        "arrival_crps": result.get("arrival_crps"),
        "dispersion_ratio": result.get("dispersion_ratio"),
        "area_dispersion_ratio": domain.get("area_dispersion_ratio"),
        "best_member_iou": result.get("best_member_iou"),
        "best_member_iou_tolerant": domain.get("best_member_iou_tolerant"),
        "band_brier_1h": band_brier.get(1),
        "band_brier": band_brier.get(horizon_h),
        "band_best_member_iou": band.get("best_member_iou"),
        "band_area_dispersion_ratio": band.get("area_dispersion_ratio"),
        # reliability_summary is keyed BY LEAD ("1","2","3"); reading it as a flat
        # dict returned None for every run and printed as "--". Found by asking
        # why a headline field was empty rather than accepting the dash.
        "band_reliability": (band.get("reliability_summary") or {})
        .get(str(horizon_h), {})
        .get("reliability"),
        "band_ece": (band.get("reliability_summary") or {}).get(str(horizon_h), {}).get("ece"),
        "band_resolution": (band.get("reliability_summary") or {})
        .get(str(horizon_h), {})
        .get("resolution"),
        "band_base_rate": (band.get("reliability_summary") or {})
        .get(str(horizon_h), {})
        .get("base_rate"),
        "band_n_cells": band.get("n_cells"),
        "n_windows": result.get("n_windows"),
        # -- [M5 / ADR-020 (4)] G3's CALIBRATION half. `reliability_*` was
        # DEMOTED (null 0.0050 vs skill 0.0099 - silence is trivially
        # calibrated); `calibration_error` on the GROWTH-MASKED subset was
        # registered in its place and is named in code by
        # `common.calibration.GATE_CRITERION_KEY`, imported rather than spelled,
        # so this table cannot drift from the contract's own choice. The `band_`
        # prefix is not decoration: `common.calibration.GATE_MASK` is
        # `growth_band`, and the domain value is DILUTED by cells nobody was ever
        # uncertain about.
        f"band_{CALIBRATION_GATE_KEY}_by_horizon": {
            str(h): rel.get(str(h), {}).get(CALIBRATION_GATE_KEY) for h in range(1, horizon_h + 1)
        },
        f"band_{CALIBRATION_GATE_KEY}_bins_by_horizon": {
            str(h): rel.get(str(h), {}).get(f"{CALIBRATION_GATE_KEY}_bins")
            for h in range(1, horizon_h + 1)
        },
        f"band_{CALIBRATION_GATE_KEY}_frontier_by_horizon": {
            str(h): rel.get(str(h), {}).get(f"{CALIBRATION_GATE_KEY}_frontier")
            for h in range(1, horizon_h + 1)
        },
        f"band_{CALIBRATION_GATE_KEY}_silent_floor_by_horizon": {
            str(h): rel.get(str(h), {}).get(f"{CALIBRATION_GATE_KEY}_silent_floor")
            for h in range(1, horizon_h + 1)
        },
        f"band_{CALIBRATION_GATE_KEY}_n_scored_by_horizon": {
            str(h): rel.get(str(h), {}).get("calibration_n_scored") for h in range(1, horizon_h + 1)
        },
        f"band_{CALIBRATION_GATE_KEY}": rel.get(str(horizon_h), {}).get(CALIBRATION_GATE_KEY),
        "calibration_gate_criterion": result.get("calibration_gate_criterion"),
        "calibration_gate_mask": result.get("calibration_gate_mask"),
        # -- [M5] the DENOMINATOR of G3's dispersion half, split into bias and
        # scatter. DIAGNOSTIC ONLY; `area_dispersion_ratio` above is the
        # criterion. Carried because our kernel over-predicts held-out growth
        # 2.66-3.06x (ADR-021 (3b)) and a low dispersion ratio caused by BIAS and
        # one caused by a NARROW ensemble have opposite remedies.
        "band_area_error_bias": band_area.get("bias"),
        "band_area_error_scatter": band_area.get("scatter"),
        "band_area_error_bias_fraction": band_area.get("bias_fraction"),
        "band_area_dispersion_ratio_debiased": band_area.get("ratio_debiased"),
        "area_error_bias": domain_area.get("bias"),
        "area_error_bias_fraction": domain_area.get("bias_fraction"),
        "area_dispersion_ratio_debiased": domain_area.get("ratio_debiased"),
        # -- [M9 / ADR-039 (5)] G3's FIRST-MOMENT CONDITION, in the artifact the
        # gate is read from. `growth_calibration` was a GATE CONDITION that
        # existed nowhere in `eval/`: simviz recomputed it from the C1 tensors in
        # `sim/s5_report.py`, so nothing in `results.json` could be scored
        # against ADR-039 (5) and the clause could not be true. Third instance of
        # a criterion missing from the file its gate turns on (C6.4's shape term,
        # ADR-020's `calibration_error`), and the first one found BEFORE the
        # verdict rather than after. The two sums travel WITH the ratio because a
        # calibration ratio without its denominator is exactly the shape of
        # number this project keeps getting wrong.
        f"band_{FIRST_MOMENT_KEY}": band_fm.get(FIRST_MOMENT_KEY),
        "band_first_moment_pred_area_sum": band_fm.get("pred_area_sum"),
        "band_first_moment_truth_area_sum": band_fm.get("truth_area_sum"),
        FIRST_MOMENT_KEY: domain_fm.get(FIRST_MOMENT_KEY),
        "first_moment_pred_area_sum": domain_fm.get("pred_area_sum"),
        "first_moment_truth_area_sum": domain_fm.get("truth_area_sum"),
        # -- [M5] ensemble diversity: the OTHER independent read on collapse.
        "band_mean_pairwise_member_iou": band.get("mean_pairwise_member_iou"),
        "mean_pairwise_member_iou": result.get("diagnostics", {}).get("mean_pairwise_member_iou"),
    }
    assert_gate_criteria_present(out)
    return out


def run_baselines(
    *,
    horizon_h: int = DEFAULT_HORIZON,
    n_members: int = DEFAULT_MEMBERS,
    seed: int = DEFAULT_SEED,
    stride: int = 1,
    fit_stride: int = 1,
    write_run: bool = True,
    extra_models: Mapping[str, Any] | None = None,
    skip_barred_controls: bool = False,
    calibration_horizons: Sequence[int] = (),
    ablation_of: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Calibrate the ellipse on train fires, score every model on held-out fires.

    ``extra_models`` puts a learned predictor through the IDENTICAL path as the
    baselines - same windows, same members, same seeds, same C6 call, same C6.2
    validity check. That is not a convenience: a model evaluated by its own
    script and a baseline evaluated by this one differ by every detail nobody
    wrote down, and the difference always flatters the model whose author wrote
    the script.
    """
    # C3.3 first: refuse to produce a number under bootstrap norm stats.
    c33 = assert_reportable()
    split = split_fingerprint()
    code = common_code_fingerprint()
    # [v2.11 / C-4.2] SAMPLED BEFORE THE RUN. The M4 artifact stamped this ONCE,
    # at payload construction - i.e. AFTER - so it recorded the hash of an
    # `eval/metrics.py` that had been rewritten nine minutes into the run and had
    # produced none of its numbers. A fingerprint taken at the end records the
    # wrong code precisely in the case it exists to catch, and reads as
    # reassurance while doing so (ADR-023 (5)). Both ends, compared below.
    scoring_before = scoring_code_fingerprint()
    stats = read_norm_stats(norm_stats_path())
    train_folds = [int(f) for f in stats["train_folds"]]

    splits = load_splits(train_folds)
    train = [s for s in splits if s.is_train]
    heldout = [s for s in splits if not s.is_train]
    if not heldout:
        raise RuntimeError("no held-out fires: every built fire is in a train fold")

    heldout_blocks = sorted({s.spatial_block_id for s in heldout})
    train_blocks = sorted({s.spatial_block_id for s in train})
    overlap = set(heldout_blocks) & set(train_blocks)
    if overlap:
        raise RuntimeError(
            f"LANDSCAPE LEAKAGE: spatial block(s) {sorted(overlap)} appear in BOTH the train "
            "and held-out folds. C3.1 forbids this; refusing to score."
        )

    t_start = time.time()
    train_windows: list[Any] = []
    hourly_train_windows: list[Any] = []
    for s in train:
        train_windows.extend(_windows_for(s.fire_id, horizon_h, fit_stride))
        hourly_train_windows.extend(_windows_for(s.fire_id, 1, fit_stride))
    train_growth = [w for w in train_windows if w.truth_growth_cells() > 0]
    if not train_growth:
        raise RuntimeError("no train window has any growth; cannot fit a spread rate")

    # -- calibrate the ellipse on TRAIN ONLY -------------------------------
    # C6.2 / ADR-011: the head-rate scale REPRODUCES OBSERVED MEAN HOURLY
    # GROWTH on train fires. One-hour windows, because "hourly growth" is the
    # quantity named in the rule.
    #
    # The two Brier fits are still run, and are the CONTROL, not a candidate:
    # they are the measured evidence for C6.2 and they demonstrate the clause
    # firing. Both are expected to ignite zero cells; if they ever stop doing
    # so, the clause's premise has changed and ADR-011 needs revisiting.
    train_ids = [s.fire_id for s in train]
    calibration = EllipseBaseline().calibrate_to_growth(
        hourly_train_windows, train_fire_ids=train_ids
    )
    # ADR-011 names "observed mean HOURLY growth", so the rule of record is
    # calibrated on 1 h windows. Measured caveat, and it is a large one: the
    # ellipse's growth is strongly SUPER-LINEAR in the horizon (its accumulator
    # is a threshold crossing, so sub-threshold progress that is discarded at
    # 1 h ignites by 3 h), while the labels' growth is very nearly linear. No
    # single scale therefore calibrates it at both 1 h and 3 h. These extra
    # calibrations quantify how much of any G2 margin is that mismatch rather
    # than model skill.
    #
    # [v2.8 / ADR-015 (3)] PER-HORIZON CALIBRATION. The ellipse's growth is
    # super-linear in horizon while the labels' is linear, so the calibration
    # horizon is a free parameter worth ~4.7x in the opponent's over-prediction
    # ratio. We do not get to hold a lever on our opponent's strength: the
    # ellipse is calibrated SEPARATELY AT EACH EVALUATION HORIZON, and at horizon
    # H the model must beat `ellipse_calHh` scored at lead H.
    alt_calibrations = {}
    for horizon in calibration_horizons:
        if int(horizon) == 1:
            # Identical by construction to the ADR-011 rule of record; refitting
            # it would only add bisection noise to a number used as an opponent.
            alt_calibrations[1] = calibration
            continue
        windows_h = (
            train_windows
            if int(horizon) == horizon_h
            else [w for s in train for w in _windows_for(s.fire_id, int(horizon), fit_stride)]
        )
        alt_calibrations[int(horizon)] = EllipseBaseline().calibrate_to_growth(
            windows_h, train_fire_ids=train_ids
        )
    fit_all = EllipseBaseline().fit_by_brier(train_windows, train_fire_ids=train_ids)
    fit_growth = EllipseBaseline().fit_by_brier(train_growth, train_fire_ids=train_ids)
    for fit in (calibration, fit_all, fit_growth, *alt_calibrations.values()):
        leaked = set(fit.train_fire_ids) & {s.fire_id for s in heldout}
        if leaked:
            raise RuntimeError(f"ellipse was calibrated on held-out fires {sorted(leaked)}")

    models: dict[str, Any] = {
        "persistence": PersistenceBaseline(),
        "ellipse": EllipseBaseline(name="ellipse").with_calibration(calibration),
    }
    if not skip_barred_controls:
        models["ellipse_brier_fit_all"] = EllipseBaseline(name="ellipse_brier_fit_all").with_fit(
            fit_all
        )
        models["ellipse_brier_fit_growth"] = EllipseBaseline(
            name="ellipse_brier_fit_growth"
        ).with_fit(fit_growth)
    for horizon, cal in alt_calibrations.items():
        if horizon == 1:
            continue  # `ellipse` already IS the 1 h calibration
        models[f"ellipse_cal{horizon}h"] = EllipseBaseline(
            name=f"ellipse_cal{horizon}h"
        ).with_calibration(cal)
    c8: dict[str, Any] = {}
    for name, model in dict(extra_models or {}).items():
        if name in models:
            raise ValueError(f"extra model {name!r} would shadow a baseline")
        c8[name] = assert_model_split_matches(model, split, name=name)
        models[name] = model

    # -- score every held-out fire ----------------------------------------
    per_fire: dict[str, Any] = {}
    pooled_all: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    pooled_growth: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    # [M5, maintainer directive] REGIME-STRATIFIED REPORTING. Dormant and
    # growth windows are DIFFERENT PREDICTION PROBLEMS with opposite difficulty -
    # 953 vs 446 on this held-out set, and simviz's S3 showed our entire growth
    # over-prediction is bought in the dormant half while the growth half is
    # calibrated to within 2-13%. An aggregate blends two regimes into a number
    # that describes neither. The dormant stratum is emitted for every model at
    # the same level as the growth stratum, not as a footnote.
    pooled_dormant: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    pooled_counts: dict[str, list[dict[str, Any]]] = {k: [] for k in models}

    for s in heldout:
        windows = _windows_for(s.fire_id, horizon_h, stride)
        growth = [w for w in windows if w.truth_growth_cells() > 0]
        zero, total = _zero_growth_fraction(windows)
        entry: dict[str, Any] = {
            "fire_id": s.fire_id,
            "cv_fold": s.cv_fold,
            "spatial_block_id": s.spatial_block_id,
            "n_hours": s.n_hours,
            "n_windows": total,
            "n_zero_growth_windows": zero,
            "zero_growth_fraction": (zero / total) if total else None,
            "n_growth_windows": len(growth),
            "models": {},
        }
        for name, model in models.items():
            all_res, counts = _score(model, windows, n_members, horizon_h, seed)
            growth_res = [
                r for w, r in zip(windows, all_res, strict=True) if w.truth_growth_cells() > 0
            ]
            dormant_res = [
                r for w, r in zip(windows, all_res, strict=True) if w.truth_growth_cells() == 0
            ]
            pooled_all[name].extend(all_res)
            pooled_growth[name].extend(growth_res)
            pooled_dormant[name].extend(dormant_res)
            pooled_counts[name].extend(counts)
            entry["models"][name] = {
                "all_windows": _headline(aggregate(all_res), horizon_h) if all_res else None,
                "growth_windows": (
                    _headline(aggregate(growth_res), horizon_h) if growth_res else None
                ),
                "dormant_windows": (
                    _headline(aggregate(dormant_res), horizon_h) if dormant_res else None
                ),
                "c6_2_validity": baseline_validity(
                    counts,
                    name=name,
                    scope=f"held-out {s.fire_id}",
                    null_model=name in NULL_MODELS,
                ),
            }
        per_fire[s.fire_id] = entry

    pooled = {
        name: {
            "all_windows": (
                _headline(aggregate(pooled_all[name]), horizon_h) if pooled_all[name] else None
            ),
            "growth_windows": (
                _headline(aggregate(pooled_growth[name]), horizon_h)
                if pooled_growth[name]
                else None
            ),
            "dormant_windows": (
                _headline(aggregate(pooled_dormant[name]), horizon_h)
                if pooled_dormant[name]
                else None
            ),
        }
        for name in models
    }
    validity = {
        name: baseline_validity(
            pooled_counts[name],
            name=name,
            scope="pooled held-out",
            null_model=name in NULL_MODELS,
        )
        for name in models
    }

    # C-4.2, the AFTER end. Sampled before the payload is built so the
    # comparison is over the whole scoring span, not over the dict literal.
    scoring_after = scoring_code_fingerprint()
    common_after = check_common_code_unchanged(code)

    payload: dict[str, Any] = {
        "kind": "c5_baselines_leave_fire_out",
        "horizon_h": horizon_h,
        "n_members": n_members,
        "seed": seed,
        "elapsed_s": round(time.time() - t_start, 1),
        "c3_3": c33,
        "split_before": split,
        "split_after": assert_split_unchanged(split),
        "common_code_before": code,
        "common_code_after": common_after,
        # C-4.2: BOTH ENDS. `common.splits.code_fingerprint_ends` reads the
        # `_before`/`_after` suffixes; a bare `scoring_code` key is the
        # one-ended shape the clause was written about, and it is deliberately
        # no longer emitted.
        "scoring_code_before": scoring_before,
        "scoring_code_after": scoring_after,
        "code_fingerprints_agree": {
            "scoring_code": scoring_before["fingerprint"] == scoring_after["fingerprint"],
            "common_code": code["fingerprint"] == common_after["fingerprint"],
            "scoring_code_changed_files": [
                name
                for name, digest in scoring_after["per_file"].items()
                if digest != scoring_before["per_file"].get(name)
            ],
            "verdict": (
                "OK — every scored number in this artifact was produced by ONE version of "
                "the scoring code"
                if scoring_before["fingerprint"] == scoring_after["fingerprint"]
                else "C-4.2 HARD FAIL — scoring code MOVED DURING THIS RUN. Discard and "
                "re-run against frozen code (C-4)."
            ),
        },
        "gate_criterion": {
            "key": GATE_CRITERION_KEY,
            "source": "common.iou_terms.GATE_CRITERION_KEY (C6.4, ADR-020 (2))",
            "reported_only": REPORTED_ONLY_KEY,
            "adjudicating_metrics": [k for k, _, _, a in G2_METRICS if a],
            "reported_not_adjudicating": [k for k, _, _, a in G2_METRICS if not a],
            "null_check": (
                "`make null-check` was run against THIS metric table before this run: "
                f"{GATE_CRITERION_KEY} scores the two do-nothing nulls at exactly 0.000 on "
                "both masks (skill 0.476 in growth_band), brier_1/2/3h and arrival_crps "
                "are ok, and area_dispersion_ratio is ok. reliability_3h is flagged "
                "SILENCE_FAVOURING at REPORTING severity and is therefore NOT quoted here "
                "as capability. ADR-020 standing policy."
            ),
        },
        # -- the scoping block. Read before any number below. --------------
        "scope": {
            "n_fires_built": len(splits),
            "n_train_fires": len(train),
            "n_heldout_fires": len(heldout),
            "n_train_blocks": len(train_blocks),
            "n_heldout_blocks": len(heldout_blocks),
            "heldout_block_ids": heldout_blocks,
            "train_fire_ids": [s.fire_id for s in train],
            "heldout_fire_ids": [s.fire_id for s in heldout],
            "c6_3_satisfied": split["c6_3_satisfied"],
            "split_fingerprint": split["fingerprint"],
            "warning": (
                f"EFFECTIVE n = {len(heldout_blocks)} HELD-OUT SPATIAL BLOCK(S), not "
                f"{len(heldout)} fires and not 28. C3.1: buffered domains overlap, so "
                "fires in one block share a landscape and are not independent samples. "
                "Do NOT quote a CV spread across these fires as if they were replicates."
            ),
        },
        "ellipse_calibration": {
            "rule_of_record": calibration.to_dict(),
            "alternative_horizons": {str(k): v.to_dict() for k, v in alt_calibrations.items()},
            "barred_controls": {
                "brier_fit_all_windows": fit_all.to_dict(),
                "brier_fit_growth_windows": fit_growth.to_dict(),
            },
            "note": (
                "`rule_of_record` is C6.2/ADR-011: the scale reproduces the observed mean "
                "HOURLY growth of the train fires. The two `barred_controls` are the same "
                "one-parameter scale fitted by pooled Brier, retained as the EVIDENCE for "
                "C6.2 and expected to ignite zero cells on held-out. Read their "
                "c6_2_validity verdicts: a VOID control is the clause working, not a bug."
            ),
        },
        "c8_split_match": c8,
        "c6_2_validity": validity,
        "per_fire": per_fire,
        "pooled_heldout": pooled,
        "regime_stratification": {
            "strata": ["growth_windows", "dormant_windows", "all_windows"],
            "rule": (
                "REGIME-STRATIFIED IS PRIMARY, AGGREGATE IS SECONDARY. Dormant (truth grew "
                "ZERO cells) and growth windows are different prediction problems with "
                "opposite difficulty and roughly a 2:1 count ratio; an aggregate blends them "
                "into a number that describes neither. `all_windows` is retained because it "
                "is incorruptible, not because it is informative."
            ),
            "note_1h": (
                "Do not read the 1 h aggregate Brier as a capability comparison against "
                "persistence: at 1 km cells most fires move sub-cell in an hour, so "
                "persistence wins there by QUANTISATION, not by merit."
            ),
        },
        "g2_per_horizon": (
            g2_per_horizon(pooled, per_fire, horizon_h, candidates=list(dict(extra_models or {})))
            if extra_models
            else None
        ),
        # [M5] G3's pre-fixed bar (ADR-020 (4)). Emitted for EVERY scored model,
        # baselines included, so the model's dispersion is read next to the
        # opponent's rather than against an interval alone.
        "g3": (
            g3_summary(
                pooled,
                per_fire,
                models=list(models),
                ablations=dict(ablation_of or {}),
                horizon_h=horizon_h,
            )
            if extra_models
            else None
        ),
        "interpretation": _interpretation(per_fire, validity),
    }

    if write_run:
        run = create_run_dir(
            {
                "experiment": "c5_baselines_leave_fire_out",
                "horizon_h": horizon_h,
                "n_members": n_members,
                "seed": seed,
                "stride": stride,
                "train_folds": train_folds,
            },
            prefix="baselines",
        )
        (run.path / "results.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
        payload["run_dir"] = str(run.path)
    return payload


#: Metrics G2 is adjudicated on, their direction, and whether they ADJUDICATE.
#: Named here rather than chosen at print time so a verdict cannot be assembled
#: from whichever metric happened to win.
#:
#: [v2.10] C6.4 splits this table in two. ``adjudicates=True`` rows enter the
#: verdict count; ``adjudicates=False`` rows are scored, printed and stored
#: identically but CANNOT move the verdict. ``best_member_iou`` is the second
#: kind: ADR-017 ruled it unfit (empty-vs-empty banks IoU 1.0, so a model
#: igniting zero cells outranks every model that predicts anything), and C6.4
#: keeps it REPORTED because hiding it would hide the pathology. The shape
#: criterion that replaces it is ``common.iou_terms.GATE_CRITERION_KEY``, whose
#: null floor is exactly 0 - imported, never spelled out here, so this table
#: cannot drift from the contract's own choice.
G2_METRICS: tuple[tuple[str, str, bool, bool], ...] = (
    ("band_brier_by_horizon", "band Brier", True, True),  # lower is better
    (f"band_{GATE_CRITERION_KEY}_by_horizon", "band best-member IoU (SHAPE, masked)", False, True),
    (
        f"band_{REPORTED_ONLY_KEY}_by_horizon",
        "band best-member IoU (REPORTED, C6.4: not a gate)",
        False,
        False,
    ),
)


def g2_per_horizon(
    pooled: Mapping[str, Any],
    per_fire: Mapping[str, Any],
    horizon_h: int,
    *,
    candidates: Sequence[str],
    stratum: str = "growth_windows",
) -> dict[str, Any]:
    """ADR-015 (3): adjudicate each candidate at EACH horizon against the ellipse
    calibrated AT THAT HORIZON.

    Two opponents are reported at every horizon and the harder one is the point:

    ``rule``
        ``ellipse_cal{H}h`` - the literal reading of the ruling. At horizon H the
        opponent is the ellipse whose scale was fitted to reproduce train growth
        at horizon H.
    ``envelope``
        The BEST score any calibrated ellipse variant achieves at horizon H,
        metric by metric. This is strictly harder than the rule and it removes
        the last residue of discretion: even if some other calibration happens to
        suit the ellipse better at this horizon, the model has to beat that too.
        The rule is what ADR-015 says; the envelope is what makes the answer
        insensitive to how the rule is read.

    Persistence is reported at every horizon but is NOT the G2 opponent - G2 is
    "beat the wind-advected ellipse". It is here because its resolution is
    exactly zero and a Brier comparison against it means something different.
    """
    ellipse_names = [n for n in pooled if n.startswith("ellipse") and "brier_fit" not in n]
    horizons = [str(h) for h in range(1, horizon_h + 1)]

    def score(model: str, key: str, h: str) -> float | None:
        row = (pooled.get(model) or {}).get(stratum)
        if not row:
            return None
        return (row.get(key) or {}).get(h)

    out: dict[str, Any] = {
        "stratum": stratum,
        "rule": (
            "ADR-015 (3) / C6.2 v2.8: at horizon H the opponent is ellipse_cal{H}h, the "
            "ellipse calibrated to train growth AT H. The `envelope` column additionally "
            "requires beating the best score ANY calibrated ellipse reaches at H."
        ),
        "ellipse_variants_scored": ellipse_names,
        "by_horizon": {},
    }
    for h in horizons:
        rule_name = f"ellipse_cal{h}h" if f"ellipse_cal{h}h" in pooled else "ellipse"
        entry: dict[str, Any] = {
            "rule_opponent": rule_name,
            "metrics": {},
        }
        for key, label, lower_better, adjudicates in G2_METRICS:
            opponents = {n: score(n, key, h) for n in ellipse_names}
            valid = {n: v for n, v in opponents.items() if v is not None}
            envelope_name, envelope_value = (
                (
                    (min(valid, key=lambda n: valid[n]), min(valid.values()))
                    if lower_better
                    else (max(valid, key=lambda n: valid[n]), max(valid.values()))
                )
                if valid
                else (None, None)
            )
            rule_value = opponents.get(rule_name)
            per_model: dict[str, Any] = {}
            for candidate in candidates:
                value = score(candidate, key, h)
                if value is None:
                    per_model[candidate] = None
                    continue
                beats_rule = (
                    None
                    if rule_value is None
                    else (value < rule_value if lower_better else value > rule_value)
                )
                beats_env = (
                    None
                    if envelope_value is None
                    else (value < envelope_value if lower_better else value > envelope_value)
                )
                blocks = []
                for fid, fire in per_fire.items():
                    cand_row = ((fire["models"].get(candidate) or {}).get(stratum)) or {}
                    opp_row = ((fire["models"].get(rule_name) or {}).get(stratum)) or {}
                    a = (cand_row.get(key) or {}).get(h)
                    b = (opp_row.get(key) or {}).get(h)
                    if a is None or b is None:
                        continue
                    blocks.append(
                        {
                            "fire_id": fid,
                            "spatial_block_id": fire["spatial_block_id"],
                            "candidate": a,
                            "opponent": b,
                            "candidate_wins": (a < b) if lower_better else (a > b),
                        }
                    )
                per_model[candidate] = {
                    "value": value,
                    "beats_rule_opponent": beats_rule,
                    "beats_envelope": beats_env,
                    "blocks_won": sum(1 for b in blocks if b["candidate_wins"]),
                    "n_blocks": len(blocks),
                    "per_block": blocks,
                }
            entry["metrics"][label] = {
                "key": key,
                "lower_is_better": lower_better,
                "adjudicates": adjudicates,
                "gate_criterion": (key.endswith(f"{GATE_CRITERION_KEY}_by_horizon")),
                "n_windows_by_horizon": (
                    {
                        n: (
                            ((pooled.get(n) or {}).get(stratum) or {}).get(
                                f"band_{GATE_CRITERION_KEY}_n_windows_by_horizon", {}
                            )
                        ).get(h)
                        for n in [*ellipse_names, "persistence", *candidates]
                    }
                    if key.endswith(f"{GATE_CRITERION_KEY}_by_horizon")
                    else None
                ),
                "opponents": opponents,
                "rule_opponent_value": rule_value,
                "envelope_value": envelope_value,
                "envelope_from": envelope_name,
                "persistence": score("persistence", key, h),
                "candidates": per_model,
            }
        out["by_horizon"][h] = entry

    # -- the verdict, computed from the table rather than narrated ----------
    # [v2.10] Only `adjudicates=True` metrics enter the count. The REPORTED-only
    # row is still scored and still printed, and its outcome is carried in
    # `reported_not_adjudicated` so the reader sees it - it just cannot move the
    # verdict, which is exactly what C6.4 rules.
    verdicts: dict[str, Any] = {}
    for candidate in candidates:
        wins = losses = unknown = 0
        detail = []
        reported: list[dict[str, Any]] = []
        for h in horizons:
            for _, label, _, adjudicates in G2_METRICS:
                cell = out["by_horizon"][h]["metrics"][label]["candidates"].get(candidate)
                beat = None if cell is None else cell["beats_rule_opponent"]
                row = {"horizon_h": int(h), "metric": label, "beats_rule": beat}
                if not adjudicates:
                    reported.append(row)
                    continue
                detail.append(row)
                if beat is True:
                    wins += 1
                elif beat is False:
                    losses += 1
                else:
                    unknown += 1
        if unknown:
            verdict = "UNDECIDABLE"
        elif losses == 0:
            verdict = "BEATS THE ELLIPSE AT EVERY HORIZON ON EVERY G2 METRIC"
        elif wins == 0:
            verdict = "LOSES TO THE ELLIPSE AT EVERY HORIZON ON EVERY G2 METRIC"
        else:
            verdict = "SPLIT"
        verdicts[candidate] = {
            "verdict": verdict,
            "n_comparisons": wins + losses + unknown,
            "n_won": wins,
            "n_lost": losses,
            "detail": detail,
            "adjudicating_metrics": [k for k, _, _, a in G2_METRICS if a],
            "reported_not_adjudicated": reported,
            "note": (
                "A SPLIT IS NOT A PASS (ADR-015 (1)). This function reports the count; the "
                "gate verdict is the maintainer's. [v2.10/C6.4] `best_member_iou` is "
                "scored and shown under `reported_not_adjudicated` but is excluded from "
                "the count: its null floor exceeds every trained model's score, so a "
                "verdict containing it is a verdict about the zero-growth rate of the "
                "labels. The shape criterion "
                f"({GATE_CRITERION_KEY}) replaces it and its null floor is exactly 0."
            ),
        }
    out["verdicts"] = verdicts
    return out


#: [M5] G3's bar, ADR-020 (4), copied here as DATA so a table cannot narrate a
#: different one. Each entry is (key, label, low, high, mask, source).
#: ``reliability_*`` is DELIBERATELY ABSENT: ADR-020 demoted it (null 0.0050 vs
#: skill 0.0099 - REL/ECE are pure calibration statistics and silence is
#: trivially calibrated) and registered `calibration_error` on the GROWTH-MASKED
#: subset in its place. ``dispersion_ratio`` is absent for the same class of
#: reason (C6.1: it scores a COLLAPSED ensemble at 1.000 and a healthy one at
#: 1.051, so it would PASS the thing G3 exists to reject).
G3_CRITERIA: tuple[tuple[str, str, float, float, str, str], ...] = (
    (
        "band_area_dispersion_ratio",
        "ensemble dispersion (area spread-skill)",
        # [M9 / ADR-039 (4)] THE GEOMETRIC BAR, IMPORTED, NEVER SPELLED.
        # `[0.8, 1.2]` was not symmetric in log space - |log 0.8| = 0.2231 against
        # |log 1.2| = 0.1823, i.e. 22% MORE TOLERANCE on the under-dispersed side,
        # which is the only side we have ever failed on. Writing `0.8333` here
        # would reintroduce exactly the defect: a second free literal that is not
        # `1/1.2`. `common.dispersion.BAR_INTERVAL` is derived from one constant.
        g3.BAR_INTERVAL[0],
        g3.BAR_INTERVAL[1],
        "growth_band",
        "ADR-039 (4) geometric bar / ADR-020 (4a) / C6.1 / ADR-011",
    ),
    (
        f"band_{CALIBRATION_GATE_KEY}",
        "calibration error, GROWTH-MASKED",
        0.0,
        0.10,
        CALIBRATION_GATE_MASK,
        "ADR-020 (4b) — the '+/-10 pts' bar, on the masked subset",
    ),
)


#: [M8] C6.3's own number, named once. G2 required >= 4 distinct held-out spatial
#: blocks and G3 inherits it; spelling it inline in a comparison is how a
#: requirement drifts from the clause that states it.
_C6_3_MIN_BLOCKS = 4


#: [M9] The ellipse arm whose first moment the candidate is measured against.
#: C6.2's growth-calibrated ellipse - the PHYSICS baseline, calibrated to
#: reproduce observed mean hourly growth on TRAIN fires, never Brier-fitted.
#: Named once, in code, because "the best physics baseline" is a phrase and
#: `"ellipse"` is a key, and the two drift.
FIRST_MOMENT_REFERENCE_MODEL = "ellipse"


#: [M9] ...but at the EVALUATION HORIZON, per C6.2 [v2.8] / ADR-015 (3), and this
#: is not a detail. The ellipse's growth is strongly SUPER-LINEAR in the horizon
#: while the labels' is linear, so the calibration horizon is worth ~4.7x in the
#: opponent's over-prediction ratio. **We do not get to pick the horizon where our
#: opponent is weakest**, and on the M8 record the choice is
#: OUTCOME-DETERMINATIVE for the all-window reading: the 1 h-calibrated `ellipse`
#: sits at growth_calibration 4.98 on all windows and `ellipse_cal3h` at 1.07, so
#: a candidate at 2.24 beats one reference and loses to the other.
#: The per-horizon arm exists only when the runner was asked for it
#: (`--calibration-horizon`), so the fallback is explicit and is REPORTED in the
#: emitted row rather than being silent.
def _ellipse_arms(per_fire: Mapping[str, Any]) -> list[str]:
    """Every GROWTH-CALIBRATED ellipse arm present. The Brier fits are EXCLUDED.

    C6.2: a Brier-fitted ellipse ignites zero cells, which converts it into
    persistence. It is a barred CONTROL, never a reference - using it would be
    "we beat a baseline that predicted nothing", which C6.2 calls VOID.
    """
    models = next(iter(per_fire.values()), {}).get("models", {})
    return [
        name
        for name in models
        if name == FIRST_MOMENT_REFERENCE_MODEL
        or (name.startswith("ellipse_cal") and name.endswith("h"))
    ]


def first_moment_reference(per_fire: Mapping[str, Any], horizon_h: int) -> str:
    """The ellipse arm the first moment is measured against, at THIS horizon."""
    preferred = f"ellipse_cal{int(horizon_h)}h"
    return preferred if preferred in _ellipse_arms(per_fire) else FIRST_MOMENT_REFERENCE_MODEL


#: [M9] The HEADLINE key the first-moment condition reads, WITH ITS MASK. The
#: `band_` prefix is not decoration: `common.dispersion.GATE_MASK` is
#: `growth_band`, and the domain value is diluted by the already-burned blob and
#: by far-field cells nobody was ever uncertain about - on the first-moment
#: playthrough's own scenario the two differ by 1.10 vs 1.069. Named once so the
#: mask cannot be lost between here and the artifact.
FIRST_MOMENT_HEADLINE_KEY = f"band_{FIRST_MOMENT_KEY}"


def _first_moment_row(
    per_fire: Mapping[str, Any], model: str, stratum: str, *, reference: str
) -> dict[str, Any]:
    """[M9 / ADR-039 (5)] G3's first-moment condition for one model.

    Both sides come from the SAME `per_fire` block, the SAME stratum and the SAME
    equal-block pooling, so the only difference between candidate and reference
    is which model produced the areas. `common.dispersion` decides what a pass
    is; this function only supplies the two numbers.

    A candidate that is not scored on a block the reference IS scored on (or vice
    versa) is UNDEFINED, not a pass - `first_moment_condition_from_blocks`
    enforces that, and `allow_missing_blocks=True` is passed here deliberately so
    the MISMATCH is reported as an undefined condition rather than raised as a
    pooling error. A raised exception at this point would kill the whole run over
    a baseline that ignited nothing on one fire.
    """
    key = FIRST_MOMENT_HEADLINE_KEY
    cand = equal_block_mean(per_fire, model, key, stratum, allow_missing_blocks=True)
    ref = equal_block_mean(per_fire, reference, key, stratum, allow_missing_blocks=True)
    condition = g3.first_moment_condition_from_blocks(
        cand["per_block"], ref["per_block"], reference_name=reference
    )
    return {
        "key": key,
        "mask": g3.GATE_MASK,
        "source": "ADR-039 (5) — relative to the best physics baseline, no fitted threshold",
        "candidate_equal_block": cand["equal_block_mean"],
        "reference_equal_block": ref["equal_block_mean"],
        "candidate_per_block": cand["per_block"],
        "reference_per_block": ref["per_block"],
        "n_blocks": cand["n_blocks_contributing"],
        "n_blocks_expected": cand["n_blocks_expected"],
        "candidate_complete": cand["complete"],
        "reference_complete": ref["complete"],
        "c6_3_satisfied": cand["n_blocks_contributing"] >= _C6_3_MIN_BLOCKS,
        **condition.as_dict(),
    }


def g3_summary(
    pooled: Mapping[str, Any],
    per_fire: Mapping[str, Any],
    *,
    models: Sequence[str],
    ablations: Mapping[str, str],
    stratum: str = "growth_windows",
    horizon_h: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """[M5] Score every model against ADR-020's PRE-FIXED G3 bar. NO VERDICT.

    This function deliberately does NOT decide G3. It emits, per model, each
    criterion's value window-pooled AND equal-block, whether it lies inside the
    pre-fixed interval, and the collapse comparison against the declared
    ablation. The gate verdict is the maintainer's - the same division of
    labour as G2, and the reason a lead's own code should never contain the word
    that closes a gate.

    ``ablations`` maps a model name to the name of ITS independent-noise
    ablation. G3 (d) requires the ablation to DEMONSTRATE collapse; because the
    ablation shares the model's parameters exactly (``with_sampler``), a null
    result there is evidence about the ensemble machinery, not about the model.
    """
    out: dict[str, Any] = {
        "stratum": stratum,
        "bar": [
            {
                "key": key,
                "label": label,
                "low": low,
                "high": high,
                "mask": mask,
                "source": source,
            }
            for key, label, low, high, mask, source in G3_CRITERIA
        ],
        "pooling": (
            "EQUAL-BLOCK is the rule of record from G3 onward (ADR-021 (4)); the "
            "window-pooled value is emitted beside it and never instead of it."
        ),
        "not_a_verdict": (
            "G3 is adjudicated by the maintainer. This block reports the pre-fixed "
            "criteria and where each model falls; it contains no pass/fail for the gate."
        ),
        "models": {},
    }
    reference_model = first_moment_reference(per_fire, horizon_h)
    out["first_moment_reference"] = {
        "model": reference_model,
        "horizon_h": horizon_h,
        "rule": (
            "ADR-039 (5) measures the candidate's |log(growth_calibration)| against the best "
            "PHYSICS baseline's on the same held-out blocks. C6.2 [v2.8] fixes which ellipse "
            "that is: the one calibrated AT THE EVALUATION HORIZON. Falls back to the 1 h "
            "`ellipse` only when the per-horizon arm was not scored, and says so here."
        ),
        "fell_back": reference_model == FIRST_MOMENT_REFERENCE_MODEL,
    }
    for model in models:
        row = (pooled.get(model) or {}).get(stratum) or {}
        criteria: dict[str, Any] = {}
        for key, label, low, high, mask, source in G3_CRITERIA:
            # [M9] `allow_missing_blocks=True` is the DECLARED permissive path
            # (common/pooling.py's contract): this loop scores EVERY model in the
            # table, including baselines that are degenerate on a block by
            # construction, and a raise here would kill the run rather than report
            # the gap. The gap is not swallowed - `complete`, `dropped_blocks` and
            # `c6_3_satisfied` are emitted below, and the hard path is exercised
            # by the first-moment playthrough.
            blockwise = equal_block_mean(per_fire, model, key, stratum, allow_missing_blocks=True)
            pooled_value = row.get(key)
            eb = blockwise["equal_block_mean"]
            # [M9 / ADR-039 (4)] THE THREE-VALUED OUTCOME, from `common/`. The
            # boolean beside it is retained so the old and new readings can be
            # compared on the same artifact, but `None` in that boolean and
            # `UNDEFINED` here are NOT the same statement: `area_dispersion_ratio`
            # is undefined at PERFECT mean calibration (its denominator is the
            # model's own mean-area error), so the criterion goes unmeasurable
            # exactly as a model gets the first moment right. A `None` must never
            # be read as a pass.
            outcome = g3.dispersion_condition(eb) if key == "band_area_dispersion_ratio" else None
            criteria[label] = {
                "key": key,
                "mask": mask,
                "interval": [low, high],
                "source": source,
                "window_pooled": pooled_value,
                "equal_block": eb,
                "in_interval_equal_block": (None if eb is None else low <= eb <= high),
                "in_interval_window_pooled": (
                    None if pooled_value is None else low <= pooled_value <= high
                ),
                **({"condition": outcome.as_dict()} if outcome is not None else {}),
                "per_block": blockwise["per_block"],
                "n_blocks": blockwise["n_blocks"],
                "n_blocks_expected": blockwise["n_blocks_expected"],
                "dropped_blocks": blockwise["dropped_blocks"],
                "complete": blockwise["complete"],
                # [M8] C6.3 SAYS >= 4 DISTINCT HELD-OUT BLOCKS, AND NOTHING HERE
                # CHECKED IT. `equal_block_mean` SKIPS a block whose value is
                # None - `_ratio` returns None when the denominator is <= EPS -
                # so a criterion could be computed on 3 blocks, print
                # `n_blocks: 3`, and still report `in_interval_equal_block: true`
                # with nothing flagging that the sample had silently shrunk. The
                # denominator was emitted but never ADJUDICATED, which is the
                # same shape as quoting the gate IoU without
                # `..._n_windows_by_horizon`. Audited on the M8 artifact: 0 of
                # 100 model x criterion cells offend, so this did not bite - it
                # is a guard against the case where it would.
                "c6_3_satisfied": blockwise["n_blocks"] >= _C6_3_MIN_BLOCKS,
                "c6_3_note": (
                    "C6.3 requires >= 4 distinct held-out spatial blocks. A criterion whose "
                    "value is undefined on a block is DROPPED from the equal-block mean, so "
                    "this flag is the only thing that says the mean was taken over fewer "
                    "blocks than the gate requires."
                ),
            }
        collapse = None
        ablation = ablations.get(model)
        if ablation and ablation in pooled:
            abl_row = (pooled.get(ablation) or {}).get(stratum) or {}
            pairs = {}
            for key in (
                "band_area_dispersion_ratio",
                "band_mean_pairwise_member_iou",
                "band_area_error_scatter",
            ):
                pairs[key] = {"model": row.get(key), "ablation": abl_row.get(key)}
            model_disp = row.get("band_area_dispersion_ratio")
            abl_disp = abl_row.get("band_area_dispersion_ratio")
            # [M9, maintainer directive 2026-08-09] **AN ABLATION CANNOT
            # DEMONSTRATE COLLAPSE IN AN ARM THAT HAD NO DISPERSION TO LOSE.**
            # Measured across 25 arms: 16 separate at 3.7-7.8x, and the other 9
            # move 1.1-1.6x and were ALREADY below 0.29 - so their clause (d)
            # "evidence" is a comparison between two ensembles that both fail the
            # dispersion bar outright. This flag says so; it does NOT change
            # clause (d), which is the maintainer's.
            # C-3: the threshold is NOT a new fitted constant. It is the gate's
            # own pre-registered bar (`common.dispersion.BAR_INTERVAL[0]`), so
            # nothing here was calibrated on any arm's result.
            both = [d for d in (model_disp, abl_disp) if d is not None]
            uninformative = bool(both) and all(
                g3.dispersion_condition(d).outcome != g3.PASS for d in both
            )
            collapse = {
                "ablation_model": ablation,
                "comparison": pairs,
                "dispersion_ratio_model_over_ablation": (
                    (model_disp / abl_disp) if model_disp and abl_disp and abl_disp > 0 else None
                ),
                "ablation_uninformative_for_clause_d": uninformative,
                "ablation_uninformative_reason": (
                    "BOTH the arm and its ablation fail the geometric dispersion bar "
                    f"{list(g3.BAR_INTERVAL)}, so the separation between them is a difference "
                    "between two under-dispersed ensembles and is NOT evidence that this arm's "
                    "shared latent produces a healthy spread. Flagged, not decided: G3 (d) is "
                    "the maintainer's clause."
                    if uninformative
                    else "not flagged: at least one side clears the dispersion bar"
                ),
                "note": (
                    "The ablation shares this model's PARAMETERS exactly (kernel."
                    "with_sampler): it differs only in whether z_t is drawn. G3 (d) "
                    "requires it to DEMONSTRATE collapse, so it is a POSITIVE CONTROL for "
                    "our ensemble machinery — a failure to collapse is a finding about the "
                    "instrument, not a success for the model."
                ),
            }
        # [M9 / ADR-039 (5)] THE FIRST-MOMENT CONDITION, LIVE.
        first_moment = _first_moment_row(per_fire, model, stratum, reference=reference_model)
        # The OTHER ellipse calibration, reported and never adjudicated. The
        # reference choice moves this condition, so hiding the alternative would
        # be picking the opponent we win against - the exact thing C6.2 [v2.8]
        # forbids one level down.
        alternates = {
            name: _first_moment_row(per_fire, model, stratum, reference=name)
            for name in sorted(_ellipse_arms(per_fire))
            if name != reference_model
        }
        dispersion_condition = next(
            (
                c["condition"]
                for c in criteria.values()
                if c["key"] == "band_area_dispersion_ratio" and "condition" in c
            ),
            None,
        )
        combined = g3.combine(
            g3.dispersion_condition(
                None if dispersion_condition is None else dispersion_condition.get("value")
            ),
            g3.first_moment_condition(
                first_moment.get("candidate_equal_block"),
                first_moment.get("reference_equal_block"),
                reference_name=FIRST_MOMENT_REFERENCE_MODEL,
            ),
        )
        out["models"][model] = {
            "criteria": criteria,
            "collapse_ablation": collapse,
            "first_moment_condition": first_moment,
            "first_moment_against_other_ellipse_calibrations": {
                "adjudicated_reference": reference_model,
                "rule": (
                    "C6.2 [v2.8]: the ellipse is calibrated SEPARATELY AT EACH EVALUATION "
                    "HORIZON and the model must beat its own best-calibrated form AT THAT "
                    "HORIZON. Everything below is REPORTED, never adjudicated, so the "
                    "reference choice is visible instead of being a free parameter."
                ),
                "alternates": {
                    name: {
                        "reference_equal_block": row["reference_equal_block"],
                        "outcome": row["outcome"],
                    }
                    for name, row in alternates.items()
                },
            },
            # BOTH conditions, reported separately, combined only at the end.
            # PASS requires both; UNDEFINED dominates FAIL, because "not
            # adjudicable" is a distinct state from "failed" and this project has
            # already had to use it once (G4 at n=2).
            "g3_conditions": combined,
            "growth_calibration": {
                "band_growth_calibration": row.get(f"band_{FIRST_MOMENT_KEY}"),
                "band_first_moment_truth_area_sum": row.get("band_first_moment_truth_area_sum"),
                "band_first_moment_pred_area_sum": row.get("band_first_moment_pred_area_sum"),
                "band_area_error_bias": row.get("band_area_error_bias"),
                "band_area_error_bias_fraction": row.get("band_area_error_bias_fraction"),
                "band_area_dispersion_ratio_debiased": row.get(
                    "band_area_dispersion_ratio_debiased"
                ),
                "note": (
                    "DIAGNOSTIC. `band_area_error_bias_fraction` is how much of the "
                    "dispersion criterion's DENOMINATOR is systematic over/under-prediction "
                    "rather than scatter. It explains a failure; it never replaces the "
                    "criterion, which is `band_area_dispersion_ratio`. [M9] "
                    "`band_growth_calibration` is NOT a diagnostic — it is G3's first-moment "
                    "CONDITION (ADR-039 (5)) and is adjudicated in `first_moment_condition`. "
                    "It is repeated here window-pooled so the two poolings sit side by side."
                ),
            },
        }
    return out


def _fmt(row: dict[str, Any], key: str, nd: int = 5) -> str:
    """Table cell, or a visible dash. Never prints ``None`` as if it were a score."""
    value = row.get(key)
    return "  --  " if value is None else f"{value:.{nd}f}"


def _num(value: Any, nd: int = 5) -> str:
    return "  --   " if value is None else f"{float(value):.{nd}f}"


def _interpretation(per_fire: dict[str, Any], validity: dict[str, Any] | None = None) -> list[str]:
    """Caveats that must travel WITH the numbers, not in a separate document."""
    fracs = [
        v["zero_growth_fraction"]
        for v in per_fire.values()
        if v["zero_growth_fraction"] is not None
    ]
    mean_zero = float(np.mean(fracs)) if fracs else float("nan")
    lines = [v["statement"] for v in (validity or {}).values()]
    return lines + [
        f"{mean_zero:.0%} of held-out windows have BITWISE ZERO growth over the whole "
        "horizon (51-91% dataset-wide, median ~0.79). Persistence "
        "is EXACTLY correct on every one of them, so its all-windows score is mostly a "
        "measure of how often GOES saw no new front -- an observation artefact, not skill. "
        "Read the growth_windows stratum next to it or the floor looks like a ceiling.",
        "A learned kernel that minimises NLL on raw hourly steps will converge to "
        "persistence and post an excellent all-windows Brier. Beating persistence on "
        "all_windows is therefore NOT evidence for G2; beating it on growth_windows AND "
        "in the growth_band mask is.",
        "Domain-mask Brier is dominated by far-field cells no model was uncertain about. "
        "band_* keys restrict to the reachable band around the t0 frontier and are the "
        "numbers a gate should cite.",
        "dispersion_ratio on a binary field is algebraically a calibration statistic and "
        "CANNOT detect ensemble collapse (it scores a collapsed ensemble at exactly 1.0). "
        "area_dispersion_ratio is the G3 number.",
        "Persistence has zero ensemble spread by construction, so its dispersion_ratio "
        "and area_dispersion_ratio are 0 -- degenerate, not calibrated.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.eval.baseline_run",
        description="Leave-fire-out C5 baselines scored through C6.",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--members", type=int, default=DEFAULT_MEMBERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fit-stride", type=int, default=1)
    parser.add_argument("--no-run-dir", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--kernel",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="score a saved contagion kernel through the SAME path as the baselines",
    )
    parser.add_argument("--skip-barred-controls", action="store_true")
    parser.add_argument("--calibration-horizon", action="append", type=int, default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)

    extra: dict[str, Any] = {}
    for item in args.kernel:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--kernel expects NAME=PATH, got {item!r}")
        from wildfire_nowcast.model.kernel import ContagionKernel

        extra[name] = ContagionKernel.load(path)
    payload = run_baselines(
        horizon_h=args.horizon,
        n_members=args.members,
        seed=args.seed,
        stride=args.stride,
        fit_stride=args.fit_stride,
        write_run=not args.no_run_dir,
        extra_models=extra,
        skip_barred_controls=args.skip_barred_controls,
        calibration_horizons=tuple(args.calibration_horizon),
    )
    if args.json:
        print(json.dumps(payload, indent=2, default=float))
        return 0

    scope = payload["scope"]
    print("=" * 96)
    print("C5 BASELINES — LEAVE-FIRE-OUT, scored through C6")
    print("=" * 96)
    print(
        f"train fires   : {', '.join(scope['train_fire_ids'])}  ({scope['n_train_blocks']} blocks)"
    )
    print(f"held-out fires: {', '.join(scope['heldout_fire_ids'])}")
    print(f"!! {scope['warning']}")
    cal = payload["ellipse_calibration"]["rule_of_record"]
    barred = payload["ellipse_calibration"]["barred_controls"]
    print(
        f"ellipse head-rate scale (C6.2 GROWTH CALIBRATION) = {cal['scale']:.4g}: predicted "
        f"{cal['predicted_new_cells']:.0f} vs observed {cal['observed_new_cells']:.0f} train "
        f"new cells over {cal['n_windows']} 1 h windows (ratio {cal['growth_ratio']:.3f}, "
        f"monotone={cal['growth_monotone_in_scale']})"
    )
    print(
        f"   BARRED controls, kept as evidence: Brier fit on all windows = "
        f"{barred['brier_fit_all_windows']['scale']:.3g}, on growth windows = "
        f"{barred['brier_fit_growth_windows']['scale']:.3g}"
    )
    print()
    print("-" * 96)
    print("C6.2 BASELINE VALIDITY — a baseline that ignites nothing VOIDS its gate")
    print("-" * 96)
    for name, v in payload["c6_2_validity"].items():
        ratio = v["growth_ratio"]
        ratio_txt = "  --  " if ratio is None else f"{ratio:.3f}"
        print(
            f"{name:<26}{v['verdict']:<12}pred={v['n_new_cells_predicted']:>9.0f}  "
            f"truth={v['n_new_cells_truth']:>7.0f}  ratio={ratio_txt}  "
            f"windows with ignition={v['n_windows_with_any_ignition']}/{v['n_windows']}"
        )
    print()

    hz = payload["horizon_h"]
    model_names = tuple(payload["pooled_heldout"])
    for stratum in ("all_windows", "growth_windows"):
        print("-" * 96)
        print(f"{stratum.upper()}   (pooled over held-out; {scope['n_heldout_blocks']} block(s))")
        print("-" * 96)
        print(
            f"{'model':<20}{'brier_1h':>11}{f'brier_{hz}h':>11}{'band_brier':>12}"
            f"{'arr_crps':>11}{'GATE_shape':>11}{'band_iou*':>10}{'silence*':>10}"
            f"{'floor*':>9}{'area_disp':>11}{'n_win':>8}"
        )
        for name in model_names:
            r = payload["pooled_heldout"][name][stratum]
            if r is None:
                print(f"{name:<20}  (no windows in this stratum)")
                continue
            print(
                f"{name:<20}{_fmt(r, 'brier_1h'):>11}{_fmt(r, f'brier_{hz}h'):>11}"
                f"{_fmt(r, 'band_brier'):>12}{_fmt(r, 'arrival_crps', 4):>11}"
                f"{_fmt(r, f'band_{GATE_CRITERION_KEY}', 4):>11}"
                f"{_fmt(r, 'band_best_member_iou', 4):>10}"
                f"{_fmt(r, 'band_best_member_iou_silence', 4):>10}"
                f"{_fmt(r, 'band_best_member_iou_silent_floor', 4):>9}"
                f"{_fmt(r, 'area_dispersion_ratio', 4):>11}"
                f"{str(r.get('n_windows')):>8}"
            )
        print(
            "  GATE_shape = band_" + GATE_CRITERION_KEY + " (C6.4 gate criterion, null floor 0). "
            "* = REPORTED only, never a gate: band_iou banks empty-vs-empty at 1.0, and "
            "floor is what a PREDICTS-NOTHING forecast scores on these same windows."
        )
        print()

    print("-" * 96)
    print("PER HELD-OUT FIRE (same spatial block — NOT independent replicates)")
    print("-" * 96)
    for fid, v in payload["per_fire"].items():
        print(
            f"{fid:<30} block={v['spatial_block_id']} fold={v['cv_fold']} "
            f"windows={v['n_windows']} zero-growth={v['zero_growth_fraction']:.0%} "
            f"growth={v['n_growth_windows']}"
        )
        for name in model_names:
            a = v["models"][name]["all_windows"] or {}
            g = v["models"][name]["growth_windows"] or {}
            gb = f"{g['band_brier']:.5f}" if g.get("band_brier") is not None else "  --  "
            ab = f"{a['band_brier']:.5f}" if a.get("band_brier") is not None else "  --  "
            gi = (
                f"{g['band_best_member_iou']:.4f}"
                if g.get("band_best_member_iou") is not None
                else "  --  "
            )
            print(f"    {name:<20} band_brier all={ab}  growth={gb}  growth band_iou={gi}")
    g2 = payload.get("g2_per_horizon")
    if g2:
        print("=" * 118)
        print("G2 UNDER ADR-015 (3) — ELLIPSE CALIBRATED SEPARATELY AT EACH EVALUATION HORIZON")
        print("=" * 118)
        print(f"stratum: {g2['stratum']}   |   {g2['rule']}")
        print()
        for h, entry in g2["by_horizon"].items():
            print(f"--- horizon {h} h   (rule opponent = {entry['rule_opponent']})")
            for label, m in entry["metrics"].items():
                direction = "lower better" if m["lower_is_better"] else "higher better"
                env = m["envelope_value"]
                role = "ADJUDICATES" if m["adjudicates"] else "REPORTED ONLY (C6.4)"
                nwin = m.get("n_windows_by_horizon") or {}
                nwin_txt = (
                    f"   n_windows={nwin.get('persistence')} (gate term is undefined where "
                    "truth did not grow)"
                    if nwin
                    else ""
                )
                print(f"  {label}  [{role}] ({direction}){nwin_txt}")
                print(
                    f"      {'opponents:':<26} rule={_num(m['rule_opponent_value'])}  "
                    f"envelope={_num(env)} [{m['envelope_from']}]  "
                    f"persistence={_num(m['persistence'])}"
                )
                for name, cell in m["candidates"].items():
                    if cell is None:
                        print(f"      {name:<26}   --")
                        continue
                    mark = {True: "BEATS", False: "loses", None: "  ?  "}
                    print(
                        f"      {name:<26} {_num(cell['value'])}  "
                        f"vs rule: {mark[cell['beats_rule_opponent']]}  "
                        f"vs envelope: {mark[cell['beats_envelope']]}  "
                        f"blocks {cell['blocks_won']}/{cell['n_blocks']}"
                    )
            print()
        print("VERDICT PER CANDIDATE (count, not a gate decision):")
        for name, v in g2["verdicts"].items():
            print(
                f"  {name:<28} {v['verdict']}   ({v['n_won']} won / {v['n_lost']} lost "
                f"of {v['n_comparisons']} horizon x metric comparisons)"
            )
        print()

    if payload.get("c8_split_match"):
        print("C8 SPLIT FINGERPRINT (hard fail on mismatch):")
        for name, v in payload["c8_split_match"].items():
            print(
                f"  {name:<28} trained {v['training_split_fingerprint']} == evaluated "
                f"{v['evaluation_split_fingerprint']}  (source: {v['source']})"
            )
        print()
    print("INTERPRETATION — these travel with the numbers:")
    for line in payload["interpretation"]:
        print(f"  * {line}")
    if "run_dir" in payload:
        print(f"\nwritten: {payload['run_dir']}/results.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
