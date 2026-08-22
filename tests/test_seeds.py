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

The CALL SITES are pinned the same way, by
:func:`test_the_oracle_draws_the_same_ensemble_in_two_separate_interpreters`.
They used to be pinned by a search of the package source for one spelling of the
defect, which passed against every other spelling of it, and a search for a
spelling is a test of the spelling. The behavioural form ships with the control
it needs: :func:`test_the_fire_id_really_does_reach_the_draw_at_a_severity_that
_cannot_move_it` shows the output moves with the fire id at all, since otherwise
"identical across two processes" would be satisfied by a constant.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from wildfire_nowcast.common.seeds import MAX_SEED_BITS, SEED_BITS, stable_seed
from wildfire_nowcast.model.noiseoracle import severity_for

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


#: Two fire ids that are deliberately ABSENT from the measured label-noise index,
#: so both fall back to the dataset mean and receive IDENTICAL severity. Any
#: difference between their forecasts is then the SEED and can be nothing else,
#: which is what makes the second test below a control rather than a coincidence.
PROBE_FIRE_A = "synthetic_seed_probe_a"
PROBE_FIRE_B = "synthetic_seed_probe_b"

#: Run in a FRESH interpreter, because ``PYTHONHASHSEED`` is read once at start
#: up and nothing asserted inside this process can see the defect at all. It
#: exercises BOTH call sites in one go: ``NoisyTruthOracle.predict`` seeds the
#: member draw on the fire id, and ``final_footprint_agreement`` seeds the
#: severity calibration on it.
_ORACLE_SCRIPT = """
import hashlib
import sys
from types import SimpleNamespace

import numpy as np

from wildfire_nowcast.model.inputs import N_STATIC, N_WEATHER
from wildfire_nowcast.model.noiseoracle import (
    NoisyTruthOracle,
    WindowTable,
    final_footprint_agreement,
)

fire = sys.argv[1]
h = w = 12
x0 = np.zeros((h, w), np.uint8)
x0[4:8, 4:8] = 1
truth = np.zeros((3, h, w), np.uint8)
for k in range(3):
    truth[k] = x0
    truth[k][3:9, 3:9] = 1
static = np.zeros((N_STATIC, h, w), np.float32)
weather = np.zeros((3, N_WEATHER, h, w), np.float32)
table = WindowTable()
table.add(
    SimpleNamespace(x0=x0, static=static, weather=weather, fire_id=fire, t0=7, truth=truth)
)
oracle = NoisyTruthOracle(
    table, name="seed-probe", scale=1.0, split_fingerprint="seed-probe"
)
members = oracle.predict(x0, static, weather, 8, 3, 11)
print(hashlib.sha256(members.tobytes()).hexdigest()[:16])
print("%.17g" % final_footprint_agreement(fire, x0 > 0, 1.0, n_draws=16)["mean_iou"])
"""


def _oracle_outputs(fire_id: str, hash_seed: str) -> tuple[str, str]:
    """``(ensemble digest, calibration mean IoU)`` from a fresh interpreter."""
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [sys.executable, "-c", _ORACLE_SCRIPT, fire_id],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    digest, mean_iou = out.stdout.split()
    return digest, mean_iou


def test_the_oracle_draws_the_same_ensemble_in_two_separate_interpreters() -> None:
    """The CALL SITES, behaviourally. Pinning the helper is not pinning its users.

    WHAT WOULD MAKE THIS FAIL: a fire-id-derived seed anywhere under
    ``NoisyTruthOracle.predict`` or ``final_footprint_agreement`` that changes
    value between two interpreters started with different ``PYTHONHASHSEED``.

    This replaces a test that searched the source for the literal string
    ``abs(ha`` + ``sh(fire_id))``. That check passed against every spelling of
    the same defect - ``hash(fire_id) % (2**31)`` among them - because a grep for
    one way of writing a bug is a test of the spelling and not of the behaviour.
    C-2 says ratification is not implementation; the same distinction applies one
    level down, where a stable helper existing is not a stable helper being used.
    """
    a_digest, a_iou = _oracle_outputs(PROBE_FIRE_A, "0")
    b_digest, b_iou = _oracle_outputs(PROBE_FIRE_A, "12345")
    assert a_digest == b_digest, (
        f"the oracle drew a DIFFERENT ensemble for {PROBE_FIRE_A!r} under two hash seeds "
        f"({a_digest} vs {b_digest}), so a per-process seed is reaching the draw"
    )
    assert a_iou == b_iou, f"the calibration IoU moved between hash seeds: {a_iou} vs {b_iou}"


def test_the_fire_id_really_does_reach_the_draw_at_a_severity_that_cannot_move_it() -> None:
    """POSITIVE CONTROL. Without it, "identical across processes" could mean "constant".

    WHAT WOULD MAKE THIS FAIL: an oracle whose output does not depend on the fire
    id at all, at which point the test above would be measuring nothing.

    The two probe ids are absent from the measured index, so both take the
    dataset-mean fallback and their severities are equal field by field. That is
    asserted here rather than assumed, because if the severities differed the
    difference in output would have a second explanation and this would stop
    being a control.
    """
    sev_a = severity_for(PROBE_FIRE_A, 1.0)
    sev_b = severity_for(PROBE_FIRE_B, 1.0)
    assert sev_a.source.startswith("DATASET MEAN fallback"), sev_a.source
    assert sev_b.source.startswith("DATASET MEAN fallback"), sev_b.source
    # `source` and `fire_id` carry the id as text; the fields that drive the
    # perturbation are the numeric ones and those are what must match.
    moving = {
        k: (v, getattr(sev_b, k))
        for k, v in vars(sev_a).items()
        if k not in ("fire_id", "source") and v != getattr(sev_b, k)
    }
    assert not moving, f"the two probe fires do not share a severity: {moving}"

    a_digest, a_iou = _oracle_outputs(PROBE_FIRE_A, "0")
    b_digest, b_iou = _oracle_outputs(PROBE_FIRE_B, "0")
    assert a_digest != b_digest, (
        "both probe fires drew the SAME ensemble, so the fire id does not reach the "
        "member draw and the cross-process test above cannot see a seeding defect"
    )
    assert a_iou != b_iou, (
        "both probe fires scored the same calibration IoU, so the fire id does not "
        "reach final_footprint_agreement either"
    )
