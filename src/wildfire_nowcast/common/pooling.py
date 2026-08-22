"""EQUAL-BLOCK pooling - and a dropped block is now a HARD failure.

WHY POOLING IS EQUAL-BLOCK AT ALL (ADR-021 (4))
-----------------------------------------------
Window pooling weights a spatial block by how many hours it burned. Creek alone
is **47% of the held-out growth pool**, so a window-pooled held-out number is
mostly one landscape. C6.3's own logic - spatial blocks are the independent
units - says that is the same false-confidence hazard C6.3 was written against.
modelling raised this at M4 *having declined to use it*, because it would have
won two persistence comparisons; the maintainer adopted it for G3 onward and
deliberately left the G2 record window-pooled. Both numbers are emitted, always.

WHY THIS MODULE EXISTS SEPARATELY (A14, ADR-037 (5) / ADR-039 (6))
-------------------------------------------------------------------
The original implementation dropped any block whose value was ``None`` and then
reported the mean **as if it were complete**::

    for fire in per_fire.values():
        value = row.get(key)
        if value is None:
            continue                 # <- the block vanishes here
    ...
    "equal_block_mean": float(np.mean(list(per_block.values())))

``_ratio`` returns ``None`` whenever its denominator is at or below EPS, so a
criterion could be computed on 3 blocks, print ``n_blocks: 3``, and still report
``in_interval_equal_block: true`` with nothing saying the sample had silently
shrunk below the 4 distinct held-out blocks C6.3 requires. The denominator was
*emitted* but never *adjudicated* - the same shape as quoting a gate IoU without
its window count, and the same shape as the vacuous train/held-out disjointness
check A14 replaced. **Silent partial coverage reads as full coverage.**

Audited on the M8 artifact at the time: 0 of 100 model x criterion cells offended,
so no past verdict is affected. This is a guard against the case where it would
be, installed before that case exists.

THE RULE
--------
A block that contributes nothing to the mean is a :class:`IncompleteBlockCoverageError`
**by default**. Callers that legitimately expect gaps - a diagnostic sweep over
arms that were never scored everywhere, say - pass ``allow_missing_blocks=True``
and get the gap enumerated in the return instead of raised. The opt-in is
explicit and per-call, so the permissive path appears in the caller's own source
where a reviewer can see it, rather than being the ambient default.

The return carries **the block count that actually contributed**
(``n_blocks_contributing``) beside the count that was expected
(``n_blocks_expected``), because "how many blocks is this mean over" must be
answerable from the artifact itself and not from re-deriving it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "IncompleteBlockCoverageError",
    "equal_block_mean",
    "equal_block_mean_of",
]


class IncompleteBlockCoverageError(ValueError):
    """A block contributed no value, so the mean is over fewer blocks than claimed."""


def _finite(value: Any) -> float | None:
    """A real number, or ``None``. ``bool``/``str`` are NOT numbers here.

    Same rejection list as ``common.dispersion._finite_positive`` and for the same
    measured reason: ``float("0.9")`` and ``float(True)`` both succeed, so a
    stringified table cell or a flag would be pooled as if it were a score.
    """
    if value is None or isinstance(value, bool | str | bytes):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def equal_block_mean_of(
    values_by_block: Mapping[Any, Any],
    *,
    allow_missing_blocks: bool = False,
    what: str = "value",
) -> dict[str, Any]:
    """Mean over BLOCKS of the per-block mean, with dropped blocks made loud.

    ``values_by_block`` maps a block id to either one value or a sequence of
    values (one per fire in that block). A value of ``None`` - or any non-finite
    thing - does not contribute.

    Raises :class:`IncompleteBlockCoverageError` if any block ends up
    contributing nothing, unless ``allow_missing_blocks`` is set.
    """
    per_block: dict[Any, float] = {}
    dropped: list[Any] = []
    n_values = 0
    n_dropped_values = 0

    for block, raw in values_by_block.items():
        items = list(raw) if isinstance(raw, list | tuple) else [raw]
        good = [v for v in (_finite(i) for i in items) if v is not None]
        n_values += len(good)
        n_dropped_values += len(items) - len(good)
        if good:
            per_block[block] = sum(good) / len(good)
        else:
            dropped.append(block)

    expected = len(values_by_block)
    contributing = len(per_block)
    out: dict[str, Any] = {
        "equal_block_mean": (sum(per_block.values()) / contributing) if per_block else None,
        "per_block": dict(per_block),
        # kept for drop-in compatibility with the pre-A14 return shape
        "n_blocks": contributing,
        "n_fires": n_values,
        # ...and the new, unambiguous pair
        "n_blocks_contributing": contributing,
        "n_blocks_expected": expected,
        "dropped_blocks": sorted(dropped, key=repr),
        "n_dropped_values": n_dropped_values,
        "complete": not dropped and n_dropped_values == 0,
    }
    if dropped and not allow_missing_blocks:
        raise IncompleteBlockCoverageError(
            f"{what}: {len(dropped)} of {expected} block(s) contributed NOTHING to the "
            f"equal-block mean ({out['dropped_blocks']}), so the mean is over "
            f"{contributing} block(s) and would be reported as if it were over {expected}. "
            "C6.3 requires >= 4 DISTINCT held-out spatial blocks and a mean that silently "
            "shrinks below that reads as full coverage. If gaps are expected here, pass "
            "allow_missing_blocks=True at the call site so the permissive path is visible "
            "to a reviewer."
        )
    return out


def equal_block_mean(
    per_fire: Mapping[str, Any],
    model: str,
    key: str,
    stratum: str,
    *,
    allow_missing_blocks: bool = False,
) -> dict[str, Any]:
    """[ADR-021 (4)] Average a metric over BLOCKS, not over windows.

    Signature-compatible with the ``eval/baseline_run.py`` original so wiring it
    up is an import change, with one added keyword. ``per_fire`` is the C6
    per-fire block: ``{fire_id: {"spatial_block_id": int, "models": {model:
    {stratum: {key: value}}}}}``.

    A fire whose ``spatial_block_id`` is missing or unusable is NOT silently
    skipped - it lands under a ``None`` block id, which then either contributes
    or is reported as dropped like any other. Dropping a fire because its block
    label is malformed would be the same silent-shrink defect one level down.
    """
    values_by_block: dict[Any, list[Any]] = {}
    for fire in per_fire.values():
        row = ((fire.get("models", {}).get(model) or {}).get(stratum)) or {}
        try:
            block: Any = int(fire["spatial_block_id"])
        except (KeyError, TypeError, ValueError):
            block = None
        values_by_block.setdefault(block, []).append(row.get(key))
    return equal_block_mean_of(
        values_by_block,
        allow_missing_blocks=allow_missing_blocks,
        what=f"{model}.{stratum}.{key}",
    )
