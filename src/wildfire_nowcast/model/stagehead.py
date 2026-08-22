"""[S1] ARM S - the incumbent kernel plus ONE scalar covariate: log burned area.

ADR-061 (6) asks a single question: does the transition kernel need a STAGE
covariate? Truth decelerates on 12 of 14 spatial blocks (ADR-061 (1-3)); arm A's
ensemble does not. The cheapest hypothesis that could explain that is that the
hazard depends on how far along the fire is, and that the kernel has no way to
express it because every one of its inputs is LOCAL (this cell's fuel, this
cell's wind) while "how big is this fire" is GLOBAL.

**One scalar in, four parameters out.** The covariate is

    z = (log1p(sum b) - STAGE_CENTRE) / STAGE_SCALE

where ``sum b`` is the expected burned CELL COUNT of the state the step is
taken from. It is a function of ``x_t`` alone, so arm S is still Markov in
``x_t`` and still admissible under C1's absorbing-state semantics; in a rollout
it is recomputed at every step from the state that step reached, exactly as
``spatial_basis`` is.

WHERE IT ACTS, and why there are two blocks and not one
-------------------------------------------------------
``log_amplitude_coeff`` (3) is a cubic in ``z`` added to ``log alpha`` - the
global hazard scale. This is the direct expression of "an old fire ignites less
per frontier cell".

``log_reach_coeff`` (1) multiplies the directional rate of spread. This is a
DIFFERENT physical mechanism: a fire can keep its per-cell hazard and simply
stop reaching as far. Amplitude and reach are not redundant - reach enters
inside ``exp(-0.5 (d / reach)^2)`` and therefore changes the SHAPE of the
stencil, while amplitude shifts every offset together.

Both enter through ``log_weights``'s existing ``log_amplitude`` / ``reach_scale``
keyword arguments (added for M10), so arm S restates none of the physics. C0:
one implementation of the elliptical-Gaussian weight.

NO CONSTANT TERM, on purpose. A constant in ``f(z)`` is exactly ``log_alpha``,
which the kernel already has and already calibrates
(``calibrate_alpha_to_growth``). Carrying both would give the optimiser an
exactly flat direction - two parameters that can only be identified by their
sum - which is the same unidentifiability defect ADR-015 (6a) found one level
down. The consequence worth stating: the function CLASS is invariant to the
values of :data:`STAGE_CENTRE` and :data:`STAGE_SCALE` up to a constant that
``log_alpha`` absorbs, so those two numbers are a conditioning choice and not a
modelling one.

ZERO INIT IS THE POINT. Every coefficient starts at 0, so ``log_amplitude`` is
0 and ``reach_scale`` is 1 and arm S at initialisation is arm A. That is not
merely tidy: it is what makes "S beat A" a statement about the covariate rather
than about a different starting point. And because the basis is LINEAR IN THE
PARAMETERS, every gradient at init is the basis function itself and is
non-zero - the head cannot be born dead the way a ``v * tanh(w z + b)`` form
would be (``df/dw = v sech^2(.) z = 0`` at ``v = 0``, so ``w`` and ``b`` would
never move and the "4-parameter" head would really have 2).
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

DTYPE: Final = torch.float32

#: Centre of the log-area covariate, in ``log1p(cells)``. exp(7) - 1 ~ 1096 cells
#: ~ 1096 km^2 at C1's 1 km grid. Chosen as a round number near the middle of the
#: corpus's log-area range BEFORE any arm was fitted; see the module docstring for
#: why the function class does not depend on it.
STAGE_CENTRE: Final = 7.0
#: Scale of the same covariate. The corpus spans roughly z in [-2.5, +2].
STAGE_SCALE: Final = 2.0

#: Number of learned parameters. Reported next to arm A's count, never asserted.
N_STAGE_PARAMETERS: Final = 4

STAGE_BASIS: Final = ("z", "z^2", "z^3")


class StageHead(nn.Module):
    """log burned area -> (additive log hazard, multiplicative reach). 4 parameters."""

    def __init__(self) -> None:
        super().__init__()
        #: Cubic in ``z``, added to ``log alpha``. No constant term (see module docstring).
        self.log_amplitude_coeff = nn.Parameter(torch.zeros(len(STAGE_BASIS), dtype=DTYPE))
        #: Linear in ``z``, exponentiated onto the directional rate of spread.
        self.log_reach_coeff = nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def covariate(self, burned: Tensor) -> Tensor:
        """``z`` for every leading batch/member entry of ``burned``. ``[...]``.

        ``burned`` is the MEAN FIELD in [0, 1], so the sum is the EXPECTED burned
        cell count, not a sampled one - the same quantity ``step_probability``
        already treats as the contagion source. ``log1p`` rather than ``log`` so
        an empty state is 0 rather than ``-inf``: C1.1 records that 6-37% of
        frames have an empty state-1 set, and a covariate that is ``-inf`` on
        those frames would take the whole step's hazard with it.
        """
        area = burned.sum(dim=(-2, -1))
        return (torch.log1p(torch.clamp(area, min=0.0)) - STAGE_CENTRE) / STAGE_SCALE

    def forward(self, burned: Tensor) -> tuple[Tensor, Tensor]:
        """``(log_amplitude, log_reach_scale)``, each ``[...]``, from ``b_{t-1}``."""
        z = self.covariate(burned)
        basis = torch.stack([z, z * z, z * z * z], dim=-1)
        log_amplitude = (basis * self.log_amplitude_coeff).sum(dim=-1)
        log_reach = self.log_reach_coeff[0] * z
        return log_amplitude, log_reach

    def report(self) -> dict[str, object]:
        """What the head learned, in units a reader can argue with.

        ``d log hazard / d log area`` at the centre is the headline: it is the
        elasticity ADR-058 (2) quotes for TRUTH, so the two are directly
        comparable. Reported, never gated.
        """
        with torch.no_grad():
            a = [float(v) for v in self.log_amplitude_coeff]
            r = float(self.log_reach_coeff[0])
        return {
            "log_amplitude_coeff": dict(zip(STAGE_BASIS, a, strict=True)),
            "log_reach_coeff_z": r,
            # z = (log1p(A) - C) / S, so d/d log A = (1/S) d/dz, and at z = 0
            # only the linear coefficient survives.
            "d_log_hazard_d_log_area_at_centre": a[0] / STAGE_SCALE,
            "d_log_reach_d_log_area": r / STAGE_SCALE,
            "centre_log1p_cells": STAGE_CENTRE,
            "scale": STAGE_SCALE,
            "n_parameters": N_STAGE_PARAMETERS,
        }
