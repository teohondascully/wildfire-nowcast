"""C0 - the ONE implementation of 8-connected component labelling.

WHY THIS IS IN ``common/`` (ADR-036 (2), hoisted at A14)
---------------------------------------------------------
``label_components`` existed TWICE, with the same signature and two genuinely
different algorithms: a union-find raster scan in ``sim/components.py`` and a BFS
flood fill in ``data/ignitions.py``. Two owners, neither able to see the other's
code, both computing the quantity that determines **C2's
``n_ignition_components``** and **G4's spot-event count**.

C0 says it plainly: *"the producer and the verifier computing geometry through
different code is how a tensor passes its check and is still wrong."* The two
copies agreed on every input anyone tried - 417 masks at A14, 0 disagreements on
both partition structure and exact label ids - so this was **latent risk, not a
live defect, and G4's 12-event count is safe.** Nothing enforced tomorrow, though,
and this quantity has already produced one cross-lead disagreement: ADR-019's
SCU ``3 -> 2`` correction, where the count was wrong because the ESTIMAND was
wrong (final-footprint components cannot see merges).

The equivalence is pinned by ``tests/test_components_differential.py``, which
archives BOTH originals verbatim and checks this function against them. That test
was written and landed green BEFORE the hoist, so "behaviour-preserving" is a
measurement rather than an intention.

WHICH IMPLEMENTATION SURVIVED, AND WHAT WAS ADDED
------------------------------------------------
The BFS flood fill (``data/``'s), because it visits only cells that are set, where
the union-find visits every cell of the grid twice in Python. **Its ``ndim`` guard
came with it**, and that guard existed in only one of the two copies - the same
asymmetry as ``silent_floor``, where the canonical version validated its horizon
and the ``sim/replay.py`` copy had no horizon concept at all (ADR-036 (3)). A
hoist must take the UNION of the guards: keeping only what both copies did would
quietly delete protection that one path had.
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["NEIGHBOURHOOD_8", "label_components"]

#: The 8-neighbourhood. Named because "8-connected" is a modelling choice, not a
#: detail: at 1 km cells a diagonal touch is 1.414 km, inside GOFER's ~2 km
#: effective resolution, so two diagonally adjacent burned cells are one body and
#: not two. A 4-connected rule would split a diagonal fire front into a chain of
#: separate "ignitions" and inflate every count this function feeds.
NEIGHBOURHOOD_8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labelling of a 2-D boolean mask. Labels are ``1..n``, 0 = off.

    Returns ``(labels, n)``. Labels are assigned in raster order of each
    component's first cell, so the numbering is deterministic and stable - which
    matters because callers index by label id (``data/ignitions.py`` compares
    ``final_labels`` between frames to establish genealogy).

    numpy-only: no scipy in this environment, and C-4.3 puts the interpreter
    environment in the frozen set, so a filter that needs a new dependency is a
    shared-state change rather than an implementation detail. Domains are
    10^3-10^4 cells, where a BFS flood fill is comfortably fast enough.

    Raises ``ValueError`` on anything that is not 2-D. That guard existed in only
    one of the two hoisted copies; see the module docstring.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape}")
    labels = np.zeros(m.shape, dtype=np.int32)
    h, w = m.shape
    n = 0
    for sy, sx in zip(*np.nonzero(m), strict=True):
        if labels[sy, sx]:
            continue
        n += 1
        queue: deque[tuple[int, int]] = deque([(int(sy), int(sx))])
        labels[sy, sx] = n
        while queue:
            y, x = queue.popleft()
            for dy, dx in NEIGHBOURHOOD_8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not labels[yy, xx]:
                    labels[yy, xx] = n
                    queue.append((yy, xx))
    return labels, n
