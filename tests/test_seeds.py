"""``stable_seed`` - the property CPython's ``hash`` does not have.

``NoisyTruthOracle.predict`` seeded on the builtin ``hash`` of the fire id, folded
into 31 bits. CPython randomises ``str`` hashing per process unless
``PYTHONHASHSEED`` is set, so the
same fire drew a different stream in every run and every "same seed" claim was
false across processes. Inside one process it looked perfect, which is why it
survived so long.

The load-bearing test here is :func:`test_the_seed_is_identical_in_two_separate
_interpreters`, which spawns real subprocesses under HOSTILE hash seeds. Anything
asserted inside this process cannot see the bug at all.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from wildfire_nowcast.common.seeds import MAX_SEED_BITS, SEED_BITS, stable_seed

FIRE = "2019_kincade"

#: The seed for :data:`FIRE`, pinned. This is a VALUE other artifacts get keyed
#: to, so changing the derivation must be a visible edit rather than a quiet
#: drift - the same discipline that made ``split_fingerprint`` reproduce
#: ``4848f491e8d588fa`` byte-identically when it was re-homed.
FIRE_SEED = 217_768_104


def _seed_in_subprocess(fire_id: str, hash_seed: str) -> int:
    """``stable_seed(fire_id)`` from a FRESH interpreter with a chosen PYTHONHASHSEED."""
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from wildfire_nowcast.common.seeds import stable_seed;"
            f"print(stable_seed({fire_id!r}))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return int(out.stdout.strip())


def _builtin_hash_in_subprocess(fire_id: str, hash_seed: str) -> int:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [sys.executable, "-c", f"print(abs(hash({fire_id!r})) % (2**31))"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return int(out.stdout.strip())


def test_the_seed_is_identical_in_two_separate_interpreters() -> None:
    """THE test. Two processes, two different hash seeds, one seed value."""
    a = _seed_in_subprocess(FIRE, "0")
    b = _seed_in_subprocess(FIRE, "12345")
    assert a == b == FIRE_SEED, (a, b, FIRE_SEED)


def test_the_builtin_hash_really_does_move_between_those_same_processes() -> None:
    """POSITIVE CONTROL. Without this, the test above could be measuring nothing.

    If CPython ever stopped randomising ``str`` hashing, the test above would
    pass for a reason that has nothing to do with the fix, and this repo has
    produced four green scans that were measuring nothing.
    """
    a = _builtin_hash_in_subprocess(FIRE, "0")
    b = _builtin_hash_in_subprocess(FIRE, "12345")
    assert a != b, (
        f"builtin hash({FIRE!r}) returned {a} under both PYTHONHASHSEED values, so this "
        "control is not exercising per-process randomisation and the test above proves less "
        "than it appears to"
    )


def test_the_pinned_value_is_the_derivation_we_shipped() -> None:
    assert stable_seed(FIRE) == FIRE_SEED


def test_distinct_ids_get_distinct_seeds() -> None:
    ids = [
        "2019_kincade",
        "2020_creek",
        "2020_czu",
        "2020_july_complex",
        "2021_dixie",
        "2020_scu",
    ]
    seeds = [stable_seed(i) for i in ids]
    assert len(set(seeds)) == len(ids), dict(zip(ids, seeds, strict=True))


def test_parts_cannot_be_ambiguously_concatenated() -> None:
    """``("a", "bc")`` and ``("ab", "c")`` are different runs and must differ."""
    assert stable_seed("a", "bc") != stable_seed("ab", "c")
    assert stable_seed("fire", 1) != stable_seed("fire", 2)
    assert stable_seed("fire", 1) != stable_seed("fire1")


def test_the_seed_fits_where_the_old_one_did() -> None:
    for fire in ("2019_kincade", "2020_creek", "x", "a" * 200):
        seed = stable_seed(fire)
        assert 0 <= seed < 2**SEED_BITS
    assert 0 <= stable_seed("x", bits=MAX_SEED_BITS) < 2**MAX_SEED_BITS


def test_a_seed_derived_from_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        stable_seed()


def test_bool_is_refused_because_it_is_secretly_an_int() -> None:
    with pytest.raises(TypeError, match="refuses bool"):
        stable_seed("fire", True)


def test_an_unsupported_part_type_is_refused() -> None:
    with pytest.raises(TypeError, match="str, int or bytes"):
        stable_seed(1.5)  # type: ignore[arg-type]


def test_the_oracle_no_longer_seeds_on_the_builtin_hash() -> None:
    """Pin the call sites, not just the helper.

    A stable helper that nobody calls is the C-2 failure: ratification is not
    implementation. Spelled in halves so this file does not itself contain the
    pattern it forbids and can be scanned like any other file (A16's trick).
    """
    from pathlib import Path

    from wildfire_nowcast.common.codefingerprint import package_root

    forbidden = "abs(ha" + "sh(fire_id))"
    offenders = [
        path.relative_to(package_root()).as_posix()
        for path in Path(package_root()).rglob("*.py")
        if "__pycache__" not in path.parts and forbidden in path.read_text()
    ]
    assert not offenders, f"{offenders} still seed on the per-process builtin hash"
