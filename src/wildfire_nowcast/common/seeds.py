"""Seeds that mean the same thing in two different processes.

**WHY THIS EXISTS, AND IT INVALIDATES A CLAIM WE HAVE MADE REPEATEDLY.** The
label-noise oracle seeded its RNG on the builtin ``hash`` of the fire id, folded
into 31 bits. CPython randomises :func:`hash` for ``str`` on every start unless
``PYTHONHASHSEED`` is set, so **every "same seed" claim we have made is false
across processes** - the same fire_id produced a different draw in each run, and
measured run-to-run movement was 0.018. Small, real, and quantified; the
dangerous part is that it is invisible inside any single process, where the
value is perfectly stable and reproducible.

``PYTHONHASHSEED=0`` was pinned in one run script as a stopgap. A stopgap in the
environment is the wrong home for a property of a number: it has to be
remembered at every invocation, it is absent from the artifact unless someone
records it, and nothing goes red when it is forgotten. The next experiment
compares arms trained in SEPARATE PROCESSES, so the property moves into the code
that needs it.

:func:`stable_seed` is a truncated BLAKE2b over the parts' bytes. It is stable
across processes, machines, and interpreter versions, and it depends on nothing
but its arguments.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

__all__ = ["stable_seed", "MAX_SEED_BITS", "SEED_BITS"]

#: BLAKE2b digest width used, in bits. 64 is plenty to separate the handful of
#: identifiers a corpus has, and it fits an unsigned 64-bit integer exactly.
MAX_SEED_BITS = 64

#: Default width. 31 bits keeps the result inside a C ``int``, which is what
#: ``numpy.random.default_rng`` seed sequences and every legacy seeding API in
#: this repo expect. It also matches the ``% (2**31)`` the broken call sites used,
#: so only the DERIVATION changed, not the shape of the number.
SEED_BITS = 31

#: ASCII unit separator. Joining parts with a byte that cannot occur in a fire id
#: or a decimal integer keeps ``("a", "bc")`` and ``("ab", "c")`` distinct - an
#: ambiguous concatenation is a silent collision between two different runs.
_SEPARATOR = b"\x1f"


def _as_bytes(part: str | int | bytes) -> bytes:
    if isinstance(part, bytes):
        return part
    if isinstance(part, bool):
        # bool is an int in Python, and True/1 hashing alike is the sort of
        # collision that is only ever found after it has confused a result.
        raise TypeError("stable_seed refuses bool: pass an explicit int or str instead")
    if isinstance(part, int):
        return str(part).encode("ascii")
    if isinstance(part, str):
        return part.encode("utf-8")
    raise TypeError(f"stable_seed accepts str, int or bytes; got {type(part).__name__}")


def stable_seed(*parts: str | int | bytes, bits: int = SEED_BITS) -> int:
    """A deterministic non-negative seed derived from ``parts``.

    Deterministic across processes, machines and interpreter versions - which is
    exactly what :func:`hash` is not. Use this anywhere a seed is derived from an
    identifier (a fire id, a run id, an arm name).

    Raises on an empty call: a seed derived from nothing is a constant wearing
    the costume of a derived value.
    """
    if not parts:
        raise ValueError(
            "stable_seed() needs at least one part. A seed derived from nothing is a "
            "constant, and it should be written as one so a reader can see it."
        )
    if not 1 <= bits <= MAX_SEED_BITS:
        raise ValueError(f"bits must be in [1, {MAX_SEED_BITS}], got {bits}")
    material: Sequence[bytes] = [_as_bytes(p) for p in parts]
    digest = hashlib.blake2b(_SEPARATOR.join(material), digest_size=MAX_SEED_BITS // 8)
    return int.from_bytes(digest.digest(), "big") & ((1 << bits) - 1)
