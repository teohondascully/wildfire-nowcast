"""C0 — ``label_components`` had TWO implementations. This is the proof they agree.

ADR-036 (2) found the same signature ``(mask) -> (labels, n)`` in
``sim/components.py:87`` (union-find over one raster pass) and
``data/ignitions.py:85`` (BFS flood fill) — **two owners who cannot see each
other's code**, computing the quantity that determines C2's
``n_ignition_components`` and G4's spot-event count. They are not a copy-paste:
they are two genuinely different algorithms that happen to agree.

That is exactly what C0 forbids: *"the producer and the verifier computing
geometry through different code is how a tensor passes its check and is still
wrong."* And this quantity has already produced one cross-lead disagreement —
ADR-019's SCU 3 -> 2 correction.

**THIS FILE IS WRITTEN AND LANDED GREEN BEFORE THE HOIST**, against both original
implementations, and re-run afterwards against the hoisted one. Both originals are
ARCHIVED VERBATIM below so the comparison survives the refactor: after the call
sites move, a test that compared ``common`` against ``common`` would prove
nothing. That is the ``silent_floor`` precedent (A13: 3003 cases before, 3000
after, 0 mismatches) applied to a function two leads depend on.

WHAT IS COMPARED, AND WHY BOTH
------------------------------
* **partition structure** — which cells are grouped together, independent of the
  integers used to label them. This is the property the contract actually cares
  about, and it is what the maintainer's 400-mask check compared.
* **exact array equality** — the labels themselves. Stronger than required, and
  measured rather than assumed: if it holds, the hoist cannot change any
  downstream consumer that indexes by label id (``ignitions.py`` does, at
  ``final_labels[...]``), so behaviour preservation needs no further argument.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

# --------------------------------------------------------------------------
# ARCHIVED ORIGINALS — verbatim copies, frozen at A14, never to be "improved"
# --------------------------------------------------------------------------


def _sim_original(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """VERBATIM ``sim/components.py:label_components`` as of A14. Do not edit.

    Union-find over a single raster pass. Frozen: its value here is as a fixed
    reference point, so "improving" it would delete the evidence.
    """
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for y in range(h):
        for x in range(w):
            if not m[y, x]:
                continue
            neigh = [
                labels[y + dy, x + dx]
                for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1))
                if 0 <= y + dy < h and 0 <= x + dx < w and labels[y + dy, x + dx] > 0
            ]
            if neigh:
                lab = min(neigh)
                labels[y, x] = lab
                for n in neigh:
                    union(lab, n)
            else:
                parent.append(len(parent))
                labels[y, x] = len(parent) - 1

    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    for y in range(h):
        for x in range(w):
            if labels[y, x]:
                root = find(labels[y, x])
                if root not in remap:
                    remap[root] = len(remap) + 1
                out[y, x] = remap[root]
    return out, len(remap)


_NBR8_ORIGINAL: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _data_original(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """VERBATIM ``data/ignitions.py:label_components`` as of A14. Do not edit."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("mask must be 2-D")
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
            for dy, dx in _NBR8_ORIGINAL:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not labels[yy, xx]:
                    labels[yy, xx] = n
                    queue.append((yy, xx))
    return labels, n


# --------------------------------------------------------------------------
# the corpus of masks
# --------------------------------------------------------------------------


def _partition(labels: np.ndarray) -> set[frozenset[tuple[int, int]]]:
    """Which cells are grouped together, IGNORING the integers used to name them."""
    groups: dict[int, set[tuple[int, int]]] = {}
    for (y, x), lab in np.ndenumerate(labels):
        if lab:
            groups.setdefault(int(lab), set()).add((int(y), int(x)))
    return {frozenset(g) for g in groups.values()}


def _handmade_masks() -> list[tuple[str, np.ndarray]]:
    """Structures chosen because they distinguish the two algorithms if anything does.

    Random masks at one density explore one regime. Diagonal chains test the
    8-connectivity that the raster-scan version reaches with a 4-neighbour
    backward mask and the flood fill reaches with all 8; the checkerboard is the
    maximal-8-connectivity/minimal-4-connectivity case; a ring is the case where a
    single-pass scan must UNION two runs it labelled separately, which is the one
    place a union-find can be wrong and a flood fill cannot.
    """
    out: list[tuple[str, np.ndarray]] = []
    out.append(("empty", np.zeros((5, 7), bool)))
    out.append(("full", np.ones((5, 7), bool)))
    out.append(("single_cell", np.eye(1, dtype=bool)))
    out.append(("1x1_off", np.zeros((1, 1), bool)))

    diag = np.eye(9, dtype=bool)
    out.append(("main_diagonal", diag))
    out.append(("anti_diagonal", diag[:, ::-1].copy()))
    out.append(("both_diagonals", diag | diag[:, ::-1]))

    board = np.indices((9, 9)).sum(axis=0) % 2 == 0
    out.append(("checkerboard", board))
    out.append(("checkerboard_inverse", ~board))

    ring = np.zeros((9, 9), bool)
    ring[1:8, 1] = ring[1:8, 7] = ring[1, 1:8] = ring[7, 1:8] = True
    out.append(("ring", ring))
    out.append(("ring_with_hole_filled", ring | np.pad(np.ones((3, 3), bool), 3)))

    # the U: two vertical runs joined only at the bottom. A one-pass scan labels
    # the arms separately and must merge them at the join.
    u = np.zeros((8, 6), bool)
    u[:, 0] = u[:, 5] = u[7, :] = True
    out.append(("u_shape", u))

    comb = np.zeros((8, 9), bool)
    comb[0, :] = True
    comb[:, ::2] = True
    out.append(("comb", comb))

    out.append(("row_vector", np.array([[True, False, True, True]])))
    out.append(("column_vector", np.array([[True], [False], [True], [True]])))
    out.append(("tall_thin", np.random.default_rng(1).random((40, 2)) < 0.4))
    out.append(("wide_flat", np.random.default_rng(2).random((2, 40)) < 0.4))
    return out


def _random_masks(n: int = 400) -> list[tuple[str, np.ndarray]]:
    """Random masks across shapes AND densities — the density sweep matters.

    A single density explores one connectivity regime; near the percolation
    threshold is where a labelling disagreement would actually show up.
    """
    rng = np.random.default_rng(20260809)
    out: list[tuple[str, np.ndarray]] = []
    for i in range(n):
        h = int(rng.integers(1, 18))
        w = int(rng.integers(1, 18))
        density = float(rng.uniform(0.02, 0.98))
        out.append((f"random[{i}] {h}x{w} p={density:.2f}", rng.random((h, w)) < density))
    return out


ALL_MASKS = _handmade_masks() + _random_masks()


def test_the_mask_corpus_is_not_empty_and_actually_varies() -> None:
    """Guard the corpus before trusting any agreement it reports.

    A differential test over a corpus of identical or trivial masks agrees
    perfectly and proves nothing — the same shape as every other confident empty
    result this project has produced.
    """
    assert len(ALL_MASKS) >= 400
    counts = {int(_data_original(m)[1]) for _, m in ALL_MASKS}
    assert 0 in counts and 1 in counts
    assert max(counts) >= 5, f"no mask produced 5+ components; corpus is too easy: {sorted(counts)}"
    assert any(m.size == 1 for _, m in ALL_MASKS)
    assert any(m.all() for _, m in ALL_MASKS) and any(not m.any() for _, m in ALL_MASKS)


# --------------------------------------------------------------------------
# (a) the two ORIGINALS agree — landed green BEFORE the hoist
# --------------------------------------------------------------------------


def test_the_two_ORIGINAL_implementations_agree_on_every_mask() -> None:
    """400+ masks, partition structure AND exact labels. 0 disagreements expected."""
    disagreements: list[str] = []
    for name, mask in ALL_MASKS:
        a_lab, a_n = _sim_original(mask)
        b_lab, b_n = _data_original(mask)
        if a_n != b_n:
            disagreements.append(f"{name}: count {a_n} vs {b_n}")
        elif _partition(a_lab) != _partition(b_lab):
            disagreements.append(f"{name}: same count {a_n}, DIFFERENT partition")
        elif not np.array_equal(a_lab, b_lab):
            disagreements.append(f"{name}: same partition, different label ids")
    assert not disagreements, disagreements[:10]


# --------------------------------------------------------------------------
# (d) the HOISTED implementation agrees with BOTH archived originals
# --------------------------------------------------------------------------


def test_the_HOISTED_implementation_agrees_with_BOTH_archived_originals() -> None:
    """C0's guarantee, executable. This is the test that outlives the refactor."""
    from wildfire_nowcast.common.components import label_components

    disagreements: list[str] = []
    for name, mask in ALL_MASKS:
        want_lab, want_n = _data_original(mask)
        alt_lab, alt_n = _sim_original(mask)
        got_lab, got_n = label_components(mask)
        if not (got_n == want_n == alt_n):
            disagreements.append(f"{name}: hoisted {got_n}, data {want_n}, sim {alt_n}")
        elif not (np.array_equal(got_lab, want_lab) and np.array_equal(got_lab, alt_lab)):
            disagreements.append(f"{name}: labels differ from an archived original")
        elif got_lab.dtype != want_lab.dtype:
            disagreements.append(f"{name}: dtype {got_lab.dtype} vs {want_lab.dtype}")
    assert not disagreements, disagreements[:10]


def test_both_call_sites_now_import_the_SAME_object() -> None:
    """C0: one implementation. Asserted by object identity, not by reading the source.

    ``is`` rather than ``==``: a re-export that had drifted into a wrapper would
    compare equal by behaviour on this corpus and still be a second implementation
    with its own future.
    """
    from wildfire_nowcast.common.components import label_components as canonical
    from wildfire_nowcast.data.ignitions import label_components as data_side
    from wildfire_nowcast.sim.components import label_components as sim_side

    assert sim_side is canonical
    assert data_side is canonical


def test_the_hoisted_version_KEEPS_the_guard_only_one_copy_had() -> None:
    """``data/``'s copy validated ``ndim == 2``; ``sim/``'s did not.

    Exactly the ``silent_floor`` situation (ADR-036 (3)): the canonical version
    validated and the copy had no such concept, so the guard was silently absent
    on one path. Hoisting must take the UNION of the guards, never the
    intersection — a refactor that keeps only what both copies did quietly
    deletes protection.
    """
    from wildfire_nowcast.common.components import label_components

    with pytest.raises(ValueError, match="2-D"):
        label_components(np.zeros((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="2-D"):
        label_components(np.zeros(4, dtype=bool))
    # and the guard the sim copy DID have implicitly: a non-bool input coerces
    assert label_components(np.array([[0, 2], [0, 0]]))[1] == 1


def test_8_connectivity_is_the_rule_and_a_diagonal_is_ONE_component() -> None:
    """The property both originals encode, stated once where the hoist can be read."""
    from wildfire_nowcast.common.components import label_components

    diag = np.eye(4, dtype=bool)
    assert label_components(diag)[1] == 1, "8-connected: a diagonal chain is one body"
    apart = np.zeros((4, 5), bool)
    apart[0, 0] = apart[2, 3] = True
    assert label_components(apart)[1] == 2
