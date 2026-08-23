"""C2 [v2.7] ``n_ignition_components`` - how many fires are inside one fire id.

GOFER files SEPARATE LIGHTNING IGNITIONS UNDER ONE FIRE ID. Undeclared, those
show up downstream as enormous "spot events" that no contagion kernel can or
should reproduce (ADR-014 §7). This module derives the count once per fire, from
the shipped ``fire_state`` field, and records the evidence so the next fire is
self-serve rather than hand-entered.

**The estimand is IGNITIONS, not final-footprint components.** Those are
different numbers and the difference is not cosmetic - ADR-017 §7 corrected
ADR-014 on exactly this point. Measured on SCU: the final footprint has 3
connected components, but two of them are 5-6 km bodies born mid-run (h27, h89)
that are spot CANDIDATES, while the two genuine ignitions are 29.3 km apart in
the very first burned frame and later MERGE, so they are invisible to a
component count. A footprint-component count gets SCU wrong twice over: wrong
number (3 vs 2) and wrong objects.

Rule of record - time and genealogy first, distance only as a tiebreak, per
ADR-017 §7 ("distance alone does not separate them"):

  (a) TIME. Bodies present in the FIRST burned frame have no antecedent anywhere
      in the record, so each is an ignition. Bodies in that frame closer than
      :data:`SEED_MERGE_KM` to one another are ONE body: at GOFER's ~2 km
      effective resolution a one-cell diagonal hole is rasterisation noise, not
      a second fire. Measured: ``2020_july_complex``'s two first-frame seeds are
      2.24 km apart (noise, one ignition); SCU's are 29.27 km apart (two).
  (b) GENEALOGY. A detached birth that later MERGES with the region that
      preceded it has a demonstrated link to it and is never counted as a
      separate ignition, however far it landed.
  (c) DISTANCE, tiebreak only. Among never-merging detached births, those within
      :data:`SPOT_RANGE_MAX_KM` are spot candidates (real signal, P3 keeps
      them); beyond it there is no observed genealogy at that range and they are
      counted as separate ignitions.

This module deliberately stops at the C2 integer. It does NOT mine, classify or
export crossing episodes - that is P3, and P3 owns the harder half of the same
distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wildfire_nowcast.common.components import label_components
from wildfire_nowcast.common.states import dilate

__all__ = [
    "SEED_MERGE_KM",
    "SPOT_RANGE_MAX_KM",
    "DetachedBirth",
    "IgnitionReport",
    "label_components",
    "count_ignition_components",
]

#: Two bodies in the first burned frame closer than this are one body. One cell
#: diagonal at 1 km is 1.414 km and a one-cell hole between them is 2.236 km;
#: GOFER's effective resolution is ~2 km, so a gap this size is rasterisation
#: noise. Fitted sample: 12 fires / 11 spatial blocks (C-3) - the only two
#: multi-seed fires in the corpus sit at 2.24 km and 29.27 km, three orders of
#: separation apart, so no value in [2.3, 29] changes any count.
SEED_MERGE_KM = 2.25

#: A never-merging detached birth farther than this is a separate ignition, not
#: a spot. Fitted sample: 12 fires / 11 spatial blocks (C-3). The largest
#: detached birth that DEMONSTRABLY merges with its predecessor is 14.14 km
#: (CZU, h25) - i.e. we have direct evidence of genealogy out to 14 km and none
#: beyond it. Rounded up to 15 km, which is the same bound an earlier and
#: independent read of the same corpus had already settled on, before there was
#: a rule here to enforce it. The corpus's only never-merging births are at
#: 5.0/6.0 km (SCU, spots) and 46.1 km
#: (``2020_july_complex``, an ignition), so no value in [15, 46] changes any
#: count either.
SPOT_RANGE_MAX_KM = 15.0

# [A14, C0] `label_components` was HOISTED to `common/components.py` - this
# module's BFS flood fill IS the implementation that survived - and is imported
# above. It is re-exported here (it stays in `__all__`) so every existing caller
# is unchanged. `sim/components.py` held a second, independent union-find copy of
# the same function; the two agreed on 417 masks with 0 disagreements on both
# partition structure and exact label ids, measured BEFORE the hoist and pinned
# by `tests/test_components_differential.py`, which archives both originals
# verbatim. ADR-036 (2): this function determines C2's `n_ignition_components`
# and G4's spot-event count, and C0 forbids the producer and the verifier
# computing geometry through different code.


def _min_gap_km(
    ys: np.ndarray, xs: np.ndarray, ty: np.ndarray, tx: np.ndarray, res_km: float
) -> float:
    """Closest approach, in km, between two sets of cell indices."""
    if not len(ys) or not len(ty):
        return float("inf")
    return (
        float(min(float(np.hypot(ty - y, tx - x).min()) for y, x in zip(ys, xs, strict=True)))
        * res_km
    )


@dataclass(frozen=True)
class DetachedBirth:
    """New burned cells with no burned 8-neighbour at the previous hour."""

    hour: int
    n_cells: int
    gap_km: float
    #: Does this body end up in the same final-frame component as the region
    #: that preceded it? Genealogy evidence, rule (b).
    merges_later: bool

    @property
    def is_separate_ignition(self) -> bool:
        return (not self.merges_later) and self.gap_km > SPOT_RANGE_MAX_KM

    def as_dict(self) -> dict[str, Any]:
        return {
            "hour": self.hour,
            "n_cells": self.n_cells,
            "gap_km": round(self.gap_km, 2),
            "merges_later": self.merges_later,
            "classified": "separate_ignition" if self.is_separate_ignition else "spot_candidate",
        }


@dataclass(frozen=True)
class IgnitionReport:
    """Everything behind the one integer C2 asks for."""

    n_ignition_components: int
    first_burn_hour: int
    n_first_frame_seeds: int
    first_frame_seed_separations_km: list[float]
    detached_births: list[DetachedBirth] = field(default_factory=list)

    @property
    def separate_ignition_births(self) -> list[DetachedBirth]:
        return [b for b in self.detached_births if b.is_separate_ignition]

    @property
    def spot_candidates(self) -> list[DetachedBirth]:
        """Never-merging detached bodies inside the observed genealogy range.

        Reported, NOT counted. ADR-017 §7: these are the long-range spotting
        signal G4 depends on, and deleting them was the failure mode the revised
        rule exists to prevent. P3 decides what to do with them.
        """
        return [
            b for b in self.detached_births if not b.merges_later and not b.is_separate_ignition
        ]

    def to_provenance(self) -> dict[str, Any]:
        """The C2 ``provenance`` fragment: HOW the number was derived."""
        return {
            "n_ignition_components": self.n_ignition_components,
            "method": "derived from the shipped fire_state field by "
            "wildfire_nowcast.data.ignitions.count_ignition_components "
            "(not hand-entered)",
            "rule": (
                "count = (a) 8-connected bodies in the FIRST burned frame, bodies "
                f"closer than {SEED_MERGE_KM} km merged as rasterisation noise "
                "(they have no antecedent in the record, so each is an ignition) "
                "+ (c) later detached births that NEVER merge with the region "
                f"preceding them AND land farther than {SPOT_RANGE_MAX_KM} km. "
                "Time and genealogy decide; distance is only a tiebreak among "
                "never-merging births (ADR-017 §7 — distance alone does not "
                "separate a separate ignition from a spot)"
            ),
            "thresholds_fitted_on": "12 fires / 11 spatial blocks (C-3)",
            "first_burn_hour": self.first_burn_hour,
            "n_first_frame_seeds": self.n_first_frame_seeds,
            "first_frame_seed_separations_km": [
                round(v, 2) for v in self.first_frame_seed_separations_km
            ],
            "separate_ignition_births": [b.as_dict() for b in self.separate_ignition_births],
            "spot_candidates_reported_not_counted": [b.as_dict() for b in self.spot_candidates],
            "n_detached_births_total": len(self.detached_births),
        }


def count_ignition_components(fire_state: np.ndarray, *, cell_size_m: float) -> IgnitionReport:
    """Derive C2's ``n_ignition_components`` from a ``(T, H, W)`` state field.

    Relies only on the C1.1 guarantee that ``fire_state`` is absorbing, so the
    ever-burned set is monotone and the final frame contains every body that
    ever existed.
    """
    st = np.asarray(fire_state)
    if st.ndim != 3:
        raise ValueError("fire_state must be (T, H, W)")
    ever = st != 0
    res_km = float(cell_size_m) / 1000.0
    any_t = ever.reshape(st.shape[0], -1).any(axis=1)
    if not any_t.any():
        raise ValueError("fire_state never burns: no ignition to count")
    t_first = int(np.argmax(any_t))

    # -- (a) bodies in the first burned frame, merged across rasterisation holes
    seed_labels, n_seed = label_components(ever[t_first])
    seed_cells = [np.nonzero(seed_labels == k) for k in range(1, n_seed + 1)]
    parent = list(range(n_seed))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    separations: list[float] = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            gap = _min_gap_km(*seed_cells[i], *seed_cells[j], res_km)
            separations.append(gap)
            if gap <= SEED_MERGE_KM:
                parent[_find(i)] = _find(j)
    n_first_frame_seeds = len({_find(i) for i in range(n_seed)})

    # -- (b)/(c) detached births, with their fate at the final frame
    final_labels, _ = label_components(ever[-1])
    births: list[DetachedBirth] = []
    prev = ever[t_first]
    for t in range(t_first + 1, st.shape[0]):
        new = ever[t] & ~prev
        if new.any():
            detached = new & ~dilate(prev, 1)
            if detached.any():
                py, px = np.nonzero(prev)
                parent_final = set(final_labels[py, px].tolist())
                lab, n = label_components(detached)
                for k in range(1, n + 1):
                    ys, xs = np.nonzero(lab == k)
                    births.append(
                        DetachedBirth(
                            hour=t,
                            n_cells=int(len(ys)),
                            gap_km=_min_gap_km(ys, xs, py, px, res_km),
                            merges_later=bool(int(final_labels[ys[0], xs[0]]) in parent_final),
                        )
                    )
        prev = ever[t]

    report = IgnitionReport(
        n_ignition_components=n_first_frame_seeds
        + sum(1 for b in births if b.is_separate_ignition),
        first_burn_hour=t_first,
        n_first_frame_seeds=n_first_frame_seeds,
        first_frame_seed_separations_km=separations,
        detached_births=births,
    )
    if report.n_ignition_components < 1:  # pragma: no cover - unreachable by (a)
        raise ValueError("n_ignition_components must be >= 1 (C2 v2.7)")
    return report
