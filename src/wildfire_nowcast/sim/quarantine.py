"""Which metrics may adjudicate, which are quarantined — sourced from the CONTRACT.

A dashboard that draws a quarantined metric next to a gate metric without saying
so will get the quarantined one cited. That has already happened to this project
three times in the other direction (``dispersion_ratio`` at ADR-011,
``best_member_iou`` at ADR-017, ``reliability`` at ADR-020), and each time the
number was already on a chart before anyone knew it was blind.

So the badge is not a caption I remember to write. It is looked up, per key, from
:data:`wildfire_nowcast.common.null_check.C6_METRICS` — the registry infra
owns and the maintainer rules on — and any run-artifact key this module cannot
classify raises. **An unclassified key is an unbadged key**, which is the failure
mode, so it fails loudly rather than rendering plain.

Three states, and they are the contract's, not mine:

``GATE``
    ``gate_eligible=True`` in the registry. May decide something.
``QUARANTINED``
    ``quarantined_by`` is set. MUST carry the badge and the clause that did it.
``REPORTED``
    neither. Legal to show, illegal to quote as capability.

Live evidence
-------------
``make null-check --json`` writes a report whose ``verdicts`` carry the MEASURED
separation behind each quarantine. Where that file is supplied, the badge quotes
today's number instead of a number I typed in once. This matters more than it
sounds: my own S2 dashboard badge said ``dispersion_ratio`` scores "collapsed
1.000, healthy 1.051" — the ADR-011 measurement — while the current harness
measures **collapsed 1.1921 vs healthy 1.1903, a paired advantage of -0.0019
+- 0.0050 over seeds, i.e. inside its own seed noise**. Same verdict, different
mechanism: it is not that the metric prefers collapse, it is that it cannot tell
the two apart at all. A hardcoded badge decays into a false claim about a true
conclusion.

This module reads only ``common/`` (which it does not modify) and JSON. It
imports nothing from ``model/``, opens no checkpoint and reads no ``runs/``
payload beyond the keys it is handed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.calibration import GATE_CRITERION_KEY as CALIBRATION_GATE_KEY
from wildfire_nowcast.common.calibration import GATE_MASK as CALIBRATION_GATE_MASK
from wildfire_nowcast.common.iou_terms import GATE_CRITERION_KEY as IOU_GATE_KEY
from wildfire_nowcast.common.null_check import C6_METRICS

__all__ = [
    "GATE",
    "QUARANTINED",
    "REPORTED",
    "UNKNOWN",
    "CALIBRATION_GATE_KEY",
    "CALIBRATION_GATE_MASK",
    "IOU_GATE_KEY",
    "G3_KEYS",
    "MetricStatus",
    "headline_key_to_metric",
    "classify",
    "badge",
    "load_null_check",
    "audit_plotted_keys",
]

GATE = "GATE"
QUARANTINED = "QUARANTINED"
REPORTED = "REPORTED"
UNKNOWN = "UNKNOWN"

#: What G3 is actually adjudicated on (ADR-011 dispersion half, ADR-020
#: calibration half). Named here so a G3 panel cannot be built against the wrong
#: pair by accident, the same guard ``GATE_CRITERION_KEY`` puts on G2.
G3_KEYS: dict[str, str] = {
    "dispersion": "area_dispersion_ratio",
    "calibration": CALIBRATION_GATE_KEY,
    "calibration_mask": CALIBRATION_GATE_MASK,
}

#: ``results.json``'s ``_headline`` renames C6's keys. This is the ONLY place the
#: two vocabularies are joined; every panel goes through it, so a renamed key
#: becomes an exception rather than a silently unbadged chart.
#:
#: ``None`` means "structural, not a metric" (window counts, criterion names).
_HEADLINE_TO_C6: dict[str, str | None] = {
    # --- skill ----------------------------------------------------------
    "band_brier_by_horizon": "brier_{h}h",
    "band_brier": "brier_3h",
    "band_brier_1h": "brier_1h",
    "brier_1h": "brier_1h",
    "brier_3h": "brier_3h",
    "arrival_crps": "arrival_crps",
    # --- IoU family -----------------------------------------------------
    "band_best_member_iou": "best_member_iou",
    "band_best_member_iou_by_horizon": "best_member_iou",
    "band_best_member_iou_shape": "best_member_iou_shape",
    "band_best_member_iou_shape_by_horizon": "best_member_iou_shape",
    "band_best_member_iou_shape_masked": IOU_GATE_KEY,
    "band_best_member_iou_shape_masked_by_horizon": IOU_GATE_KEY,
    "band_best_member_iou_silence": "best_member_iou_silence",
    "band_best_member_iou_silence_by_horizon": "best_member_iou_silence",
    "band_best_member_iou_silent_floor": "best_member_iou_silent_floor",
    "band_best_member_iou_silent_floor_by_horizon": "best_member_iou_silent_floor",
    "best_member_iou": "best_member_iou",
    "best_member_iou_tolerant": "best_member_iou_tolerant",
    # --- dispersion -----------------------------------------------------
    "dispersion_ratio": "dispersion_ratio",
    "area_dispersion_ratio": "area_dispersion_ratio",
    "band_area_dispersion_ratio": "area_dispersion_ratio",
    # --- calibration ----------------------------------------------------
    "band_ece": "ece_3h",
    "band_ece_by_horizon": "ece_{h}h",
    "band_reliability": "reliability_3h",
    "band_reliability_by_horizon": "reliability_{h}h",
    "band_resolution": "resolution_3h",
    "band_resolution_by_horizon": "resolution_{h}h",
    "band_calibration_error": f"{CALIBRATION_GATE_KEY}_3h",
    "band_calibration_error_by_horizon": CALIBRATION_GATE_KEY + "_{h}h",
    "band_calibration_error_bins_by_horizon": CALIBRATION_GATE_KEY + "_bins_{h}h",
    "band_calibration_error_frontier_by_horizon": CALIBRATION_GATE_KEY + "_frontier_{h}h",
    "band_calibration_error_silent_floor_by_horizon": CALIBRATION_GATE_KEY + "_silent_floor_{h}h",
    # --- structural, deliberately not metrics ---------------------------
    "band_base_rate": None,
    "band_n_cells": None,
    "n_windows": None,
    "band_best_member_iou_gate_criterion": None,
    "band_best_member_iou_shape_masked_n_windows_by_horizon": None,
}


@dataclass(frozen=True)
class MetricStatus:
    """One key's contractual standing, plus whatever evidence we have for it."""

    headline_key: str
    c6_key: str | None
    state: str
    quarantined_by: str = ""
    note: str = ""
    evidence: str = ""

    @property
    def must_badge(self) -> bool:
        return self.state in (QUARANTINED, UNKNOWN)

    @property
    def adjudicates(self) -> bool:
        return self.state == GATE


def headline_key_to_metric(headline_key: str, horizon: int | None = None) -> str | None:
    """Map a ``results.json`` headline key onto its C6 registry name."""
    if headline_key not in _HEADLINE_TO_C6:
        raise KeyError(
            f"{headline_key!r} has no entry in sim.quarantine._HEADLINE_TO_C6. "
            "Refusing to render an unclassified key: an unclassified key is an "
            "UNBADGED key, and every metric pathology this project has found was "
            "on a chart before it was understood. Add it (with its C6 name) or "
            "map it to None if it is structural."
        )
    template = _HEADLINE_TO_C6[headline_key]
    if template is None:
        return None
    if "{h}" in template:
        if horizon is None:
            # Per-horizon family: any member has the same standing, so classify
            # on the last horizon rather than refusing.
            horizon = 3
        return template.format(h=int(horizon))
    return template


def load_null_check(path: str | Path | None) -> dict[str, str]:
    """``metric -> one-line MEASURED evidence`` from ``make null-check --json``.

    Missing file is not an error: the badge falls back to the registry note, and
    says which it used. A dashboard that refuses to render without an optional
    file is a dashboard nobody runs.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text())
    out: dict[str, str] = {}
    for entry in _iter_verdicts(payload):
        name = entry.get("metric") or entry.get("name")
        if not isinstance(name, str):
            continue
        # Prefer the FAILING evidence. `detail` carries the comparison verdict,
        # which for a quarantined metric is often the reassuring "every degenerate
        # model ranks below genuine skill" line — true, and exactly the sentence a
        # badge must not quote. `capture_detail` is the zero-capture axiom's
        # finding (ADR-023) and is the one that survives a member-count change.
        parts = [
            entry.get("capture_detail") or "",
            entry.get("detail") or entry.get("reason") or entry.get("statement") or "",
        ]
        flagged = str(entry.get("verdict", "")).lower() not in ("ok", "", "n/a")
        detail = parts[0] or (parts[1] if flagged else "")
        if isinstance(detail, list):
            detail = " ".join(str(d) for d in detail)
        if detail and name not in out:
            out[name] = str(detail)
    return out


def _iter_verdicts(node: Any) -> list[dict[str, Any]]:
    """Walk the null-check payload for verdict-shaped dicts, shape-agnostically."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "metric" in node and any(k in node for k in ("verdict", "flags", "is_flagged")):
            found.append(node)
        for v in node.values():
            found.extend(_iter_verdicts(v))
    elif isinstance(node, list):
        for v in node:
            found.extend(_iter_verdicts(v))
    return found


def classify(
    headline_key: str,
    *,
    horizon: int | None = None,
    evidence: dict[str, str] | None = None,
) -> MetricStatus:
    """Look up one key's standing. Raises on a key this module does not know."""
    c6 = headline_key_to_metric(headline_key, horizon)
    if c6 is None:
        return MetricStatus(headline_key, None, REPORTED, note="structural, not a metric")
    spec = C6_METRICS.get(c6)
    if spec is None:
        return MetricStatus(
            headline_key,
            c6,
            UNKNOWN,
            note=(
                f"{c6!r} is not in common.null_check.C6_METRICS, so it has never been "
                "null-checked. C6.0: no metric enters a gate without clearing the null "
                "check first."
            ),
        )
    state = GATE if spec.gate_eligible else (QUARANTINED if spec.quarantined_by else REPORTED)
    live = (evidence or {}).get(c6, "")
    return MetricStatus(
        headline_key,
        c6,
        state,
        quarantined_by=spec.quarantined_by,
        note=spec.note,
        evidence=live,
    )


def badge(status: MetricStatus, *, max_chars: int = 460) -> str:
    """The text that MUST appear on any panel drawing this key. '' if none is due."""
    if status.state == GATE:
        return ""
    if status.state == REPORTED and not status.evidence:
        return ""
    if status.state == UNKNOWN:
        return f"UNCHECKED METRIC — {status.note}"
    if status.state == REPORTED:
        return f"REPORTED, not a gate criterion — {status.evidence}"[:max_chars]
    src = "MEASURED" if status.evidence else "registry"
    body = status.evidence or status.note
    return (f"QUARANTINED by {status.quarantined_by} — must not adjudicate. [{src}] {body}")[
        :max_chars
    ]


def audit_plotted_keys(
    keys: dict[str, str], *, evidence: dict[str, str] | None = None
) -> list[str]:
    """``{headline_key: badge_text_actually_drawn}`` -> list of violations.

    Used by the self-tests. The rule it enforces is the one that survives a
    screenshot: **every quarantined key that appears on a figure carries its
    clause on the same figure.** Empty list means the figure is honest.
    """
    problems: list[str] = []
    for key, drawn in keys.items():
        status = classify(key, evidence=evidence)
        want = badge(status)
        if not want:
            continue
        marker = status.quarantined_by or ("UNCHECKED" if status.state == UNKNOWN else "REPORTED")
        if marker.split()[0] not in drawn and "QUARANTINED" not in drawn.upper():
            problems.append(
                f"{key}: state={status.state} requires a badge citing {marker!r}; "
                f"figure drew {drawn!r}"
            )
    return problems
