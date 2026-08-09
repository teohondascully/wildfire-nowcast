"""C5 baselines — the floor and the debugging oracle.

CLAUDE.md fixes three non-negotiable baselines: persistence, wind-advected
ellipse, and ELMFIRE Monte Carlo with default Rothermel parameters. The first
two live here. ELMFIRE is run by sim in ``sim/``; it enters the
comparison at G6 through the same C5 interface and the same C6 metrics, which is
the only thing that makes the head-to-head honest.

``BASELINES`` is the registry :func:`wildfire_nowcast.model.api.load_model`
resolves names against, so ``load_model("ellipse")`` works without a checkpoint.
"""

from __future__ import annotations

from wildfire_nowcast.model.baselines.ellipse import (
    EllipseBaseline,
    EllipseFitResult,
    GrowthCalibration,
)
from wildfire_nowcast.model.baselines.persistence import PersistenceBaseline

__all__ = [
    "PersistenceBaseline",
    "EllipseBaseline",
    "EllipseFitResult",
    "GrowthCalibration",
    "BASELINES",
]

#: name -> class. Every entry exposes ``predict``, ``to_spec`` and ``from_spec``.
BASELINES: dict[str, type] = {
    PersistenceBaseline.kind: PersistenceBaseline,
    EllipseBaseline.kind: EllipseBaseline,
}
