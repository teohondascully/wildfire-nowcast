"""C5 baseline 1 of 2 - persistence. The floor, and the debugging oracle.

Persistence predicts that nothing changes: every member, at every lead, is
``x0``. It is trivial and it is the most important number in the project,
because of insights/data item 1:

    51-91% of GOFER hours have BITWISE ZERO growth (median ~0.79; Kincade has a
    32-hour run). This is an observation artefact -- GOES cannot see new front
    at night or under cloud -- not the fire stopping.

So on raw hourly steps, persistence is *right about four times in five*. Any
model that fails to beat it is not merely weak; and any model that beats it by a
small margin on a pooled all-hours score has probably just learned to predict
"no change" slightly more cleverly. This is why every C6 report must be read
alongside its growth-conditioned stratum, and why persistence is reported first.

Its second job is as an oracle. Persistence has zero free parameters and zero
randomness, so if a metric moves when the seed or the member count changes, the
metric is wrong, not the model. :mod:`wildfire_nowcast.eval.selftest` uses it
exactly that way.

Deliberate design notes
-----------------------
* **No burnout decay.** A cell burning at ``t0`` stays in state 1 forever. Truth
  decays 1 -> 2 with a residence p50 of 3-5 h (ADR-006), so persistence is
  "wrong" about that - correctly. Persistence means persistence; smuggling a
  decay model in would make it a one-parameter model wearing the floor's name.
* **All members identical.** Ensemble spread is exactly 0. That is the point:
  it anchors the dispersion axis of G3 at the degenerate end, so a learned
  ensemble's dispersion ratio has something to be better than.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wildfire_nowcast.model.api import validate_predict_inputs

__all__ = ["PersistenceBaseline"]


class PersistenceBaseline:
    """``samples[m, k] == x0`` for every member ``m`` and lead ``k``."""

    kind = "persistence"

    def __init__(self, name: str = "persistence") -> None:
        self.name = name

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        validate_predict_inputs(x0, static, weather, n_members, horizon_h, seed)
        base = np.asarray(x0, dtype=np.uint8)
        return np.broadcast_to(base, (int(n_members), int(horizon_h), *base.shape)).copy()

    # -- serialisation ----------------------------------------------------

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "params": {}}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> PersistenceBaseline:
        return cls(name=str(spec.get("name", "persistence")))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "PersistenceBaseline()"
