"""The observation-noise model for GOFER labels, as a training-time ensemble.

ADR-015 (6b): the M2 kernel's free offset weights grew a **wind-independent
S/SW preference**, which is what GOFER's measured systematic centroid bias would
look like if the model absorbed it as physics. The instruction - correctly - is
to handle that by *marginalising over the label noise*, not by penalising the
offset weights until the symptom disappears. A penalty removes the evidence; a
marginalisation removes the incentive.

What perturbs the labels, and what provably does not
----------------------------------------------------
INTERFACES C1.1 designates ``cfireLine``'s six ``fconf`` levels as "the
label-perturbation ensemble for the observation-noise augmentation". **For a
``burned``-set target that ensemble is exactly the identity**, and this is a
proof, not an opinion. C1.1's own algebra is::

    ever(t)       = OR_{s<=t} inside perimeter(s)          # no fconf
    burning(t)    = (burning(t-1) or new(t)) and (new(t) or active(t))
    burned_out(t) = ever(t) and not burning(t)

so ``state > 0`` is ``ever(t)``, and ``ever`` never reads ``cfireLine``. ``fconf``
moves cells between states 1 and 2 and can move nothing else.

Measured rather than merely argued (:func:`fconf_burned_set_invariance`, run on
Kincade, Walker and Zogg across all six shipped levels):

    burned cells differing from fconf=0.50 : 0  /  0  /  0     (18 of 18 cases)
    state-1 cells differing from fconf=0.50 : up to 1,217 cell-frames
    state-1 totals across levels            : 1,022 - 2,744 (Kincade), a 2.7x swing

So the designated ensemble is a **null perturbation for this milestone's target**
and cannot address the S/SW artefact. That is a finding about the interface, not
a licence to skip the marginalisation: CLAUDE.md separately mandates
"dilate/erode label augmentation as observation-noise model", and the artefact
lives in the PERIMETER, so the perturbation must act on the perimeter.

The perimeter noise model, sized from measurement
-------------------------------------------------
The tracked index ``data/interim/_index/label_noise_east_west.json`` measures
GOFER-East against GOFER-West on the same fire - two independent renderings of
one perimeter, i.e. a direct read of the observation noise. Dataset-wide: IoU
0.687, centroid offset magnitude 1.64 km, equivalent-radius mismatch 0.91 km,
East larger in 90% of all timesteps.
Per-fire values (0.49 km Bobcat to 5.52 km CZU) are in
``data/interim/_index/label_noise_east_west.json`` and are used per fire, because
the measurement describes the offset as "a per-fire latent nuisance
parameter, resampled per fire, not per pixel and not per step".

Two draws per fire per gradient step, both **shared by every cell and every hour
in the batch**:

``shift``
    An integer-cell translation. Magnitude from a half-normal with
    ``sigma = centroid_offset_km / 2`` (East-vs-West is the disagreement BETWEEN
    two estimates; one estimate's departure from the consensus is about half of
    it), direction UNIFORM, magnitude capped at 2 cells. Stochastic rounding, so
    each draw is a hard label while the ensemble marginal is the continuous one.
``morph``
    Dilate or erode by one cell, with ``P(any) = radius_mismatch_km / 2``.

Why the sign of ``morph`` is symmetric even though the measured bias is not
--------------------------------------------------------------------------
East > West in 90% of timesteps identifies the sign of ``E - W``. It does **not**
identify the sign of ``Combined - truth``, which is the quantity a training
augmentation has to marginalise. The measurement's own prescription is a
per-fire scalar area bias "drawn from something like N(0, 15-25%)" - zero mean.
The thing that makes this a BIAS model rather than the mis-specified symmetric
jitter data warns against is not an asymmetric sign, it is that **one draw
covers the whole fire**: within a draw every cell and every hour is displaced
and inflated together. A per-pixel or per-step draw would average out inside a
single batch and would model nothing.

Direction is uniform for the same reason and it is the operative choice for the
artefact: if the S/SW preference is a real physical anisotropy the fires taught
the model, an isotropically-displaced label cannot remove it; if it is the
label's own displacement being fitted as physics, an isotropic displacement
destroys it. That makes this a TEST, not just a regulariser.

Everything here is read-only with respect to ``data/``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor
from torch.nn import functional as F

from wildfire_nowcast.model.kernel import shift_field

__all__ = [
    "FCONF_LEVELS",
    "LabelPerturbation",
    "FireNoiseModel",
    "IDENTITY",
    "noise_model_for",
    "sample_perturbation",
    "apply_perturbation",
    "dilate_field",
    "erode_field",
    "growth_band_field",
    "fconf_burned_set_invariance",
    "perturbation_report",
]

#: The six ``cfireLine`` confidence levels C1.1 designates as the perturbation
#: ensemble. Mirrored from :mod:`wildfire_nowcast.data.gofer` rather than
#: re-derived; imported lazily there because that module pulls in geopandas.
FCONF_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_FCONF = 0.50

#: Dataset means from the tracked index
#: ``data/interim/_index/label_noise_east_west.json``:
#: ``dataset_mean.centroid_offset_km_mean`` and
#: ``dataset_mean.equiv_radius_mismatch_km_mean``, each rounded to 2 dp. Used
#: when a fire has no row in the per-fire index. Declared rather than silently defaulting to zero: a
#: missing calibration must degrade to "the dataset's noise", never to "no noise".
DATASET_CENTROID_OFFSET_KM = 1.64
DATASET_RADIUS_MISMATCH_KM = 0.91

#: A 3-cell shift exceeds the kernel's whole 3-cell contagion radius, at which
#: point the perturbation has stopped being an observation-noise model and has
#: become a different fire. Capped, and the cap is reported.
MAX_SHIFT_CELLS = 2


@dataclass(frozen=True)
class LabelPerturbation:
    """One realisation of the observation noise, shared across a whole batch."""

    shift_r: int = 0
    shift_c: int = 0
    morph: int = 0  # +1 dilate one cell, -1 erode one cell, 0 identity
    fconf: float = DEFAULT_FCONF

    @property
    def is_identity(self) -> bool:
        return self.shift_r == 0 and self.shift_c == 0 and self.morph == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift_r": self.shift_r,
            "shift_c": self.shift_c,
            "morph": self.morph,
            "fconf": self.fconf,
            "fconf_effect_on_burned_set": "NONE — provably the identity (C1.1 `ever`)",
        }


IDENTITY = LabelPerturbation()


@dataclass(frozen=True)
class FireNoiseModel:
    """Per-fire calibration of the perimeter noise, from measured East/West."""

    fire_id: str
    offset_sigma_cells: float
    p_morph: float
    source: str
    centroid_offset_km: float
    radius_mismatch_km: float
    east_larger_fraction: float | None = None
    max_shift_cells: int = MAX_SHIFT_CELLS

    def to_dict(self) -> dict[str, Any]:
        return {
            "fire_id": self.fire_id,
            "offset_sigma_cells": self.offset_sigma_cells,
            "p_morph": self.p_morph,
            "max_shift_cells": self.max_shift_cells,
            "measured_centroid_offset_km": self.centroid_offset_km,
            "measured_radius_mismatch_km": self.radius_mismatch_km,
            "measured_east_larger_fraction": self.east_larger_fraction,
            "source": self.source,
        }


def _index_path() -> Path:
    from wildfire_nowcast.common.paths import data_dir

    return Path(data_dir()) / "interim" / "_index" / "label_noise_east_west.json"


def noise_model_for(
    fire_id: str, *, cell_size_m: float = 1000.0, scale: float = 1.0
) -> FireNoiseModel:
    """Read the measured East/West disagreement for ``fire_id``.

    ``scale`` multiplies both magnitudes so an ablation can sweep the strength of
    the perturbation without inventing a second noise model.
    """
    cell_km = float(cell_size_m) / 1000.0
    path = _index_path()
    row: dict[str, Any] = {}
    source = f"DATASET MEAN fallback (no row for {fire_id})"
    if path.is_file():
        payload = json.loads(path.read_text())
        per_fire = dict(payload.get("per_fire", {}))
        if fire_id in per_fire:
            row = dict(per_fire[fire_id])
            source = f"{path.name}: per-fire measured East-vs-West"
    offset_km = float(row.get("centroid_offset_km_mean", DATASET_CENTROID_OFFSET_KM))
    mismatch_km = float(row.get("equiv_radius_mismatch_km_mean", DATASET_RADIUS_MISMATCH_KM))
    return FireNoiseModel(
        fire_id=fire_id,
        # Half the pairwise disagreement: E-vs-W measures the gap between two
        # estimates, not either one's distance from the truth.
        offset_sigma_cells=float(scale) * 0.5 * offset_km / cell_km,
        p_morph=min(0.5, float(scale) * 0.5 * mismatch_km / cell_km),
        source=source,
        centroid_offset_km=offset_km,
        radius_mismatch_km=mismatch_km,
        east_larger_fraction=(
            float(row["east_larger_fraction"]) if "east_larger_fraction" in row else None
        ),
    )


def _stochastic_round(value: float, rng: np.random.Generator) -> int:
    """Round to an integer, preserving the mean exactly.

    Deterministic rounding would collapse the whole perturbation for any fire
    whose measured offset is under half a cell (most of them), so the ensemble
    would silently be the identity for exactly the fires that need it least
    checked.
    """
    low = math.floor(value)
    return int(low + (1 if rng.random() < (value - low) else 0))


def sample_perturbation(model: FireNoiseModel, rng: np.random.Generator) -> LabelPerturbation:
    """Draw ONE realisation for a whole fire. Direction uniform, sign symmetric."""
    magnitude = abs(rng.normal(0.0, model.offset_sigma_cells)) if model.offset_sigma_cells else 0.0
    magnitude = min(magnitude, float(model.max_shift_cells))
    angle = rng.uniform(0.0, 2.0 * math.pi)
    # Row index increases SOUTHWARD (C1.4 north-up), matching kernel.shift_field.
    shift_r = _stochastic_round(-magnitude * math.cos(angle), rng)
    shift_c = _stochastic_round(magnitude * math.sin(angle), rng)
    morph = 0
    if rng.random() < model.p_morph:
        morph = 1 if rng.random() < 0.5 else -1
    return LabelPerturbation(
        shift_r=shift_r,
        shift_c=shift_c,
        morph=morph,
        fconf=float(rng.choice(FCONF_LEVELS)),
    )


# --------------------------------------------------------------------------
# morphology on torch fields (8-connected, matching common.states.dilate)
# --------------------------------------------------------------------------


def _pooled(x: Tensor, pad_value: float) -> Tensor:
    flat = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
    padded = F.pad(flat, (1, 1, 1, 1), value=pad_value)
    out = F.max_pool2d(padded, kernel_size=3, stride=1)
    return out.reshape(x.shape)


def dilate_field(x: Tensor, iterations: int = 1) -> Tensor:
    """8-connected binary dilation on ``[..., H, W]`` in ``{0., 1.}``.

    Off-grid is UNBURNED, so a fire cannot be dilated in from outside the domain.
    """
    out = x
    for _ in range(int(iterations)):
        out = _pooled(out, pad_value=0.0)
    return out


def erode_field(x: Tensor, iterations: int = 1) -> Tensor:
    """8-connected binary erosion. Off-grid counts as UNBURNED, so edges erode."""
    out = x
    for _ in range(int(iterations)):
        out = 1.0 - _pooled(1.0 - out, pad_value=1.0)
    return out


def apply_perturbation(x: Tensor, p: LabelPerturbation) -> Tensor:
    """Apply ``p`` to a burned-indicator field ``[..., H, W]`` in ``{0., 1.}``.

    Order is morph-then-shift, and both operators are MONOTONE, which is the
    property the whole scheme rests on: if ``b0 <= truth_k`` pointwise before the
    perturbation then it still holds after it, so a perturbed window remains a
    legal absorbing-fire window and ``new cells = truth_k - b0`` stays >= 0.
    Perturbing ``b0`` and ``truth`` with DIFFERENT draws would break that and
    would manufacture negative growth.
    """
    out = x
    if p.morph > 0:
        out = dilate_field(out, p.morph)
    elif p.morph < 0:
        out = erode_field(out, -p.morph)
    if p.shift_r or p.shift_c:
        out = shift_field(out, p.shift_r, p.shift_c)
    return out


def growth_band_field(burned0: Tensor, radius_cells: int) -> Tensor:
    """The C6 ``growth_band`` mask, recomputed in torch from a perturbed ``b0``.

    Mirrors :func:`wildfire_nowcast.eval.masks.growth_band` exactly - dilate the
    frontier of the burned set, minus the burned set - because a perturbed label
    with an unperturbed scoring mask would score the model on cells the perturbed
    fire was never near.
    """
    unburned = 1.0 - burned0
    frontier = burned0 * dilate_field(unburned, 1)
    return (dilate_field(frontier, max(1, int(radius_cells))) * unburned) > 0.5


# --------------------------------------------------------------------------
# the fconf claim, measured
# --------------------------------------------------------------------------


def fconf_burned_set_invariance(
    fire_ids: Sequence[str] = ("2019_kincade", "2019_walker", "2020_zogg"),
    levels: Sequence[float] = FCONF_LEVELS,
) -> dict[str, Any]:
    """Rebuild the C1.1 labels at every ``fconf`` and diff the BURNED set.

    Read-only: :func:`wildfire_nowcast.data.labels.build_fire_state` returns the
    state array in memory and writes nothing. Expensive (it re-rasterises every
    perimeter), so it is a measurement to be run and recorded, not a per-run
    assertion.
    """
    from wildfire_nowcast.data.labels import build_fire_state

    per_fire: dict[str, Any] = {}
    worst_burned_diff = 0
    for fire_id in fire_ids:
        states = {
            float(c): build_fire_state(
                fire_id, cfire_conf=float(c), with_east_west_noise=False
            ).state
            for c in levels
        }
        ref = states[DEFAULT_FCONF]
        rows = {}
        for conf, st in states.items():
            burned_diff = int(np.count_nonzero((st > 0) != (ref > 0)))
            worst_burned_diff = max(worst_burned_diff, burned_diff)
            rows[str(conf)] = {
                "burned_cells_differing_vs_050": burned_diff,
                "state1_cells_total": int(np.count_nonzero(st == 1)),
                "state1_cells_differing_vs_050": int(np.count_nonzero((st == 1) != (ref == 1))),
            }
        per_fire[fire_id] = rows
    return {
        "levels": [float(c) for c in levels],
        "per_fire": per_fire,
        "max_burned_cells_differing": worst_burned_diff,
        "burned_set_is_invariant_to_fconf": worst_burned_diff == 0,
        "why": (
            "C1.1 defines ever(t) from perimeters alone; cfireLine only splits ever(t) into "
            "states 1 and 2. A burned-set target is therefore invariant to fconf BY "
            "CONSTRUCTION, and this measures it rather than asserting it."
        ),
    }


def perturbation_report(
    fire_ids: Sequence[str], *, scale: float = 1.0, n_draws: int = 4000, seed: int = 0
) -> dict[str, Any]:
    """What the sampler actually does, per fire. Reported with every run.

    An augmentation nobody measures is indistinguishable from an augmentation
    that is silently the identity.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for fire_id in fire_ids:
        model = noise_model_for(fire_id, scale=scale)
        draws = [sample_perturbation(model, rng) for _ in range(int(n_draws))]
        shifts = np.array([[d.shift_r, d.shift_c] for d in draws], dtype=float)
        morphs = np.array([d.morph for d in draws], dtype=float)
        out[fire_id] = {
            **model.to_dict(),
            "fraction_identity": float(np.mean([d.is_identity for d in draws])),
            "mean_shift_r": float(shifts[:, 0].mean()),
            "mean_shift_c": float(shifts[:, 1].mean()),
            "rms_shift_cells": float(np.sqrt((shifts**2).sum(axis=1)).mean()),
            "max_abs_shift_cells": int(np.abs(shifts).max()),
            "p_dilate_realised": float(np.mean(morphs > 0)),
            "p_erode_realised": float(np.mean(morphs < 0)),
        }
    return {
        "scale": scale,
        "n_draws": n_draws,
        "per_fire": out,
        "note": (
            "One draw per fire per gradient step, shared by every cell and hour in the "
            "batch. mean_shift_* near 0 is the point: the DIRECTION is uniform, so the "
            "ensemble cannot teach a directional preference, only unteach one."
        ),
    }
