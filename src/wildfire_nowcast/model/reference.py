"""The one latent-bearing C5 address a clone of this repository can open.

WHAT THIS IS FOR
----------------
G3 (d) asks whether holding the shared latent ``z_t`` at its prior mean
collapses the ensemble. That question can only be asked of a model that HAS a
latent, and :func:`~wildfire_nowcast.model.api.assert_ablation_arm_is_demonstrative`
refuses it of one that does not. The registry model, ``contagion_kernel``, is
constructed with ``latent_config=None``: its ablation arm is bit-identical to
it, the refusal fires, and that refusal is correct.

So until this file existed, every address on which G3 (d) could be STATED was an
untracked checkpoint directory in the run archive, which is not part of the
published tree. The first statement of G3 (d) about this project's model
(ADR-124) rests on one of them, and a reader with a clone could not open it.
This module tracks ONE fit, 58,256 bytes of it, so that the address the contract
talks about resolves in a clone.

**NO CHANGE TO C5 WAS NEEDED AND NONE WAS MADE.**
:func:`~wildfire_nowcast.model.api.load_model` already accepts a directory
holding ``model.json``, and ``<address>__independent`` already derives the
latent-off arm from whatever that address resolves to. What was missing was a
tracked subject, not a mechanism. Nothing here registers a second entry: the arm
stays DERIVED, NEVER REGISTERED (C5 [v2.18]), and
:data:`REFERENCE_FIT_ADDRESS` is a path, not a registry name, so
``available_models()`` is unchanged.

WHICH FIT, AND WHAT IT MAY BE USED TO CLAIM
-------------------------------------------
It is the S1 arm-A fit at fold 3, trained under split fingerprint
``b3e5dadad01eaef9`` on the sixteen fires of folds [0, 1, 2, 4]. The directory it
was archived under on the training machine is named in
:data:`REFERENCE_FIT_PROVENANCE`, and it is named THERE and nowhere else on
purpose: a fact repeated in prose beside a machine-readable copy of itself is a
fact with two spellings and one reader, which is how the two drift.

The copy here is byte-identical to that archive: both digests were read in one
command on 2026-08-23 and are the value pinned in :data:`REFERENCE_FIT_SHA256`,
which the self-test re-reads on every run. Byte-identity was then checked
BEHAVIOURALLY as well, because a digest says the file did not move and not that
the predictor built from it behaves the same way: on a real held-out scene the
two addresses draw bit-identical samples on both arms, and the five per-fire
collapse artifacts this address produces are identical to ADR-124's field for
field. **Both of those are measurements on the training machine and neither is
checkable in a clone**, which is the limit below and is stated rather than
implied.

**MAGNITUDE DOES NOT TRAVEL BETWEEN FITS.** ADR-124 records the collapse
instrument separating the two arms by 15.1 / 20.5 / 25.0 points at 1 / 2 / 3 h
on this fit, and by 36.7 / 43.3 / 65.6 on a second fit at the same fold. The
direction survived and the magnitude did not, so a number measured here is about
THIS fit and is not a property of "the model". Tracking a fit makes it
checkable; it does not promote it.

WHAT A CLONE STILL CANNOT DO
----------------------------
Reach the VERDICT. ADR-124's verdict is scored on the five held-out fold-3
fires, and ``data/`` is untracked, so a clone can load this fit, take its arm,
verify that the arm is demonstrative and draw from both arms on a synthetic
scene, and cannot re-score the held-out cells. One of the two reproducibility
barriers is removed here and the other one is not.

TWO PROPERTIES OF THE ARCHIVE THAT ARE REPORTED, NOT REPAIRED
-------------------------------------------------------------
1. **The label does not identify the fit.** ``ContagionKernel.from_spec``
   honours the ``name`` in the spec, and 107 of the 108 archived specs on the
   training machine carry the same one: ``contagion_kernel``. So
   ``load_model(REFERENCE_FIT_ADDRESS).name`` and
   ``load_model("contagion_kernel").name`` are the same string, for a
   latent-bearing fit and for a model with no latent at all, which is exactly
   the pair whose difference decides G3 (d). A run record or a figure legend
   carrying only that label cannot say which one it drew. The address
   distinguishes them and the label does not, so records should carry the
   address; ``sim/collapse.py`` already does.
2. **The spec is not repaired to look better.** It is a copy of evidence. It
   keeps its own ``name``, its own ``components`` prose and its own escapes.

PACKAGING
---------
``model.json`` is package DATA, and ``[tool.setuptools.package-data]`` in
``pyproject.toml`` declares only ``py.typed`` today, so an installed wheel does
not carry it. **Measured rather than read off the configuration**: a wheel built
from this tree contains ``model/reference.py`` and does NOT contain
``reference_fit/model.json``. Every gate in this repository runs from the source
tree, where
:data:`REFERENCE_FIT_DIR` is resolved from ``__file__`` and is therefore
correct wherever the tree is. :func:`load_reference_fit` raises with that
sentence rather than with a bare ``FileNotFoundError`` if the file is absent, so
the packaging gap reports itself instead of arriving as a puzzle.

ADR-122 APPLIES TO THE ADDRESS ITSELF. :data:`REFERENCE_FIT_DIR` is resolved
through ``__file__``, so it names whichever tree the interpreter IMPORTED, which
in a clone with the shared editable install is the SHARED tree. Point the
interpreter at the clone and print the resolved path beside any verdict; the
self-test emits it for that reason.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from wildfire_nowcast.model.api import (
    ABLATION_ARM_SUFFIX,
    SPEC_NAME,
    Predictor,
    ablation_arm,
    load_model,
)

__all__ = [
    "REFERENCE_FIT_ADDRESS",
    "REFERENCE_FIT_ARM_ADDRESS",
    "REFERENCE_FIT_BYTES",
    "REFERENCE_FIT_DIR",
    "REFERENCE_FIT_N_PARAMETERS",
    "REFERENCE_FIT_PROVENANCE",
    "REFERENCE_FIT_SHA256",
    "REFERENCE_FIT_SPEC",
    "load_reference_fit",
    "reference_fit_arm",
    "reference_fit_pair",
    "reference_fit_sha256",
]

#: The tracked checkpoint directory, resolved from this module's own location so
#: that it does not depend on the working directory. See the ADR-122 note above:
#: it names the tree the interpreter imported, which is not always the clone.
REFERENCE_FIT_DIR: Final[Path] = Path(__file__).resolve().parent / "reference_fit"

#: The spec file inside it. ``load_model`` finds this itself from the directory.
REFERENCE_FIT_SPEC: Final[Path] = REFERENCE_FIT_DIR / SPEC_NAME

#: The same address written the way a reader types it at the repository root.
#: This is the spelling that belongs in a run record: it is portable, whereas
#: :data:`REFERENCE_FIT_DIR` is absolute and would put one machine's home
#: directory into an artifact.
REFERENCE_FIT_ADDRESS: Final[str] = "src/wildfire_nowcast/model/reference_fit"

#: The latent-off ABLATION ARM of the address above. Derived at resolution time
#: by ``load_model``; there is no second registry entry and no second fit.
REFERENCE_FIT_ARM_ADDRESS: Final[str] = REFERENCE_FIT_ADDRESS + ABLATION_ARM_SUFFIX

#: sha256 of the tracked ``model.json``. Read on 2026-08-23 in the same command
#: as the digest of the archived original named in
#: :data:`REFERENCE_FIT_PROVENANCE`, and equal to it. A clone can check this pin
#: against the file it has; it cannot check the file it does not have, which is
#: the point of tracking one.
REFERENCE_FIT_SHA256: Final[str] = (
    "82209f508eba9e568b0b0208c6b772cdf2a4330f38d255a9841f8904d96ac9d1"
)

#: Size on disk, so the cost of a clonable G3 (d) subject is a number rather
#: than an impression.
REFERENCE_FIT_BYTES: Final[int] = 58256

#: Learned scalars in the spec, latent encoder and conditional prior included.
REFERENCE_FIT_N_PARAMETERS: Final[int] = 1352

#: What the fit's own ``provenance`` block must say. Pinned here so that
#: swapping the artifact for a different fit fails a check instead of silently
#: re-pointing every sentence written about this address.
REFERENCE_FIT_PROVENANCE: Final[Mapping[str, object]] = MappingProxyType(
    {
        "split_fingerprint": "b3e5dadad01eaef9",
        "train_folds": (0, 1, 2, 4),
        "n_train_fires": 16,
        "n_heldout_blocks": 5,
        "trained_utc": "2026-08-21T17:53:46Z",
        "archived_run_directory": "runs/s1_arma_s1_f3-20260821-180258",
    }
)


def reference_fit_sha256() -> str:
    """Digest of the tracked spec as it is on disk right now."""
    return hashlib.sha256(REFERENCE_FIT_SPEC.read_bytes()).hexdigest()


def load_reference_fit() -> Predictor:
    """The tracked latent-bearing fit, through C5 and nothing else."""
    if not REFERENCE_FIT_SPEC.is_file():
        raise FileNotFoundError(
            f"{REFERENCE_FIT_SPEC} is missing. It is package DATA and "
            "[tool.setuptools.package-data] declares only py.typed, so an installed wheel "
            "does not carry it; run from the source tree, where this path is resolved from "
            "the imported module's own location."
        )
    return load_model(REFERENCE_FIT_DIR)


def reference_fit_pair() -> tuple[Predictor, Predictor]:
    """``(model, arm)`` from ONE load, which is the only way to get the pair.

    **CALL THIS RATHER THAN THE TWO LOADERS SEPARATELY, and the reason is a
    reading I got wrong here first.** Every call to :func:`load_reference_fit`
    builds a new object from the spec. Two such objects hold parameters that are
    bitwise EQUAL and are not the SAME OBJECTS, so a sharing test written across
    two loads reports ``False`` by construction, and a collapse comparison
    written across two loads would differ in the parameters as well as in the
    sampler even though every number looked right. That is the confound C5
    [v2.18] makes an identity check about, arriving through the front door.

    Inside one call the arm is a view of the model that was just loaded, so
    :func:`~wildfire_nowcast.model.api.ablation_arm` sees the objects it
    requires and the pair is provably the same fit.
    """
    model = load_reference_fit()
    return model, ablation_arm(model)


def reference_fit_arm() -> Predictor:
    """Its latent-off ablation arm, sharing that load's own parameter objects.

    Equivalent to ``load_model(f"{REFERENCE_FIT_DIR}{ABLATION_ARM_SUFFIX}")``.
    Both routes go through :func:`~wildfire_nowcast.model.api.ablation_arm`,
    which raises unless every parameter of the arm IS the model's own object, so
    neither route can return a look-alike.

    The arm returned here shares parameters with the model loaded INSIDE this
    call, which is not the model any other call returned. Use
    :func:`reference_fit_pair` when both halves are needed.
    """
    return reference_fit_pair()[1]
