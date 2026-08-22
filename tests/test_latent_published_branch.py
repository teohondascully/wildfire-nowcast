"""The latent branch that PUBLISHED, under test as the branch that published.

Until this file existed the suite covered the legacy three-channel encoder while
every number this project has published from the M7, M8, M9, M10 and S1 matrices
came out of the four-channel ``innovation_channels`` branch. Measured on this
disk at the time of writing: 86 artifacts under ``runs/`` record
``innovation_channels: true`` and 9 record ``false``, and the 9 are the four
``m7_gate_nofix`` seeds plus the matrix that reports them, i.e. a deliberate
ablation. The tested path was not the published path, and both states read as a
green suite.

WHAT THIS FILE ADDS THAT THE ENCODER TESTS DO NOT
-------------------------------------------------
``tests/test_playthrough_spatial_latent.py`` checks the contents of the
innovation channel on a fire with NO spatial modes. That is one of the two
branches the published configuration takes at once. The dominant published
configuration (26 run directories, including every M10 and S1 fit) has
``innovation_channels`` AND ``spatial_modes = 2``, so the projection statistics
are computed from an innovation-derived feature map, and that combination had no
test at all. It also had no test of the two boundaries where a flag can be lost
without any arithmetic being wrong:

* the CHECKPOINT boundary, where a spec that records the flag must reload onto
  the encoder that was fitted, or every re-score of a published checkpoint
  silently runs the legacy branch while the artifact says otherwise;
* the ARTIFACT boundary, where the recorded flag must be the encoder that ran
  rather than an independently maintained copy of the intention.

WHY THE OBVIOUS FORM OF EACH ASSERTION IS NOT USED
---------------------------------------------------
``_Encoder`` zero-initialises its head so that ``q == prior`` at step one, which
means ``mu`` is identically zero for every possible input. Any end-to-end
assertion on an untouched network passes against a deleted branch. Two defences
are used here and neither is decorative: the closed-form tests rewire the network
to a single tap so the whole forward pass is one known functional of one input
channel, and the round-trip test randomises the encoder weights first so that
"the two posteriors agree" is a statement about 6 non-trivial numbers rather than
about zero.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from wildfire_nowcast.model.kernel import ContagionKernel, KernelConfig
from wildfire_nowcast.model.latent import LatentConfig, LatentHead, latent_report

# --------------------------------------------------------------------------
# the configuration that published
# --------------------------------------------------------------------------

#: The ``latent_config`` block recorded by the runs that produced the published
#: M8, M9, M10 and S1 numbers, copied from the M10 direct-head run's own
#: ``model.json`` and identical in 26 run directories. It is a literal rather
#: than a read of a run directory: those artifacts are not all tracked, so a test
#: that read one would assert on this machine and be vacuous in a clone, which is
#: the exact shape that took CI down twice this week. No path to an untracked
#: artifact is cited here either, for the same reason: a reader could not open
#: it.
PUBLISHED_LATENT_CONFIG: dict[str, Any] = {
    "dim": 4,
    "components": [
        "log_intensity",
        "head_rotation",
        "log_wind_speed",
        "activity_gate",
        "intensity_grad_east",
        "intensity_grad_north",
    ],
    "init_sigma": [0.35, 0.2, 0.15, 2.0],
    "max_sigma": [2.0, 1.5, 1.0, 6.0],
    "encoder_channels": 8,
    "free_bits": 0.02,
    "gate_prior_mean": -1.5,
    "conditional_prior": True,
    "mean_preserving": True,
    "gate_mean_preserving": True,
    "rho": 0.5144,
    "spatial_modes": 2,
    "spatial_init_sigma": [0.3, 0.3],
    "spatial_max_sigma": [2.0, 2.0],
    "innovation_channels": True,
    "spatial_encoder_pooling": True,
}

#: Exactly the flags a training config sets. Everything else in
#: :data:`PUBLISHED_LATENT_CONFIG` is a default, which is why the first test
#: below is worth having: a default that drifts moves a published model without
#: anyone editing a run.
PUBLISHED_FLAGS: dict[str, Any] = {
    "dim": 4,
    "encoder_channels": 8,
    "free_bits": 0.02,
    "gate_prior_mean": -1.5,
    "conditional_prior": True,
    "mean_preserving": True,
    "gate_mean_preserving": True,
    "rho": 0.5144,
    "spatial_modes": 2,
    "innovation_channels": True,
    "spatial_encoder_pooling": True,
}

# --------------------------------------------------------------------------
# a scenario whose every answer is arithmetic on declared geometry
# --------------------------------------------------------------------------

H, W = 9, 11
ROW0, COL0, BLOCK = 2, 3, 3
#: Cells unburned in ``b`` and burned in ``y``. Written as coordinates so the
#: expected innovation is something a reader can compute on paper.
NEW_CELLS: tuple[tuple[int, int], ...] = ((5, 4), (6, 7))
#: The step probability at ``z = 0``, constant over the domain on purpose.
P_ZERO = 0.25


def _burned() -> torch.Tensor:
    b = torch.zeros(H, W, dtype=torch.float64)
    b[ROW0 : ROW0 + BLOCK, COL0 : COL0 + BLOCK] = 1.0
    return b


def _grown() -> torch.Tensor:
    y = _burned()
    for row, col in NEW_CELLS:
        assert y[row, col] == 0.0, f"({row},{col}) is burned in b; the scenario is wrong"
        y[row, col] = 1.0
    return y


def _tapped_head(*, innovation: bool = True) -> LatentHead:
    """A one-channel encoder rewired so the posterior mean IS a chosen statistic.

    One centre-tap convolution, one identity convolution and a head that reads a
    single pooled statistic. With ``encoder_channels = 1`` the statistics vector
    is ``[mean, amax, proj_mode0, proj_mode1]``, so a head row set at index ``i``
    makes ``mu[i]`` that statistic and nothing else. Each caller sets the head
    rows it wants; the taps below are the parts common to all of them.
    """
    head = LatentHead(
        LatentConfig(
            dim=2,
            spatial_modes=2,
            encoder_channels=1,
            innovation_channels=innovation,
        )
    )
    enc = head.encoder
    with torch.no_grad():
        for layer in (enc.conv1, enc.conv2, enc.head):
            layer.weight.zero_()
            layer.bias.zero_()
        # The LAST input channel: index 3 on the innovation branch (the
        # difference) and index 2 on the legacy branch (p_zero).
        enc.conv1.weight[0, enc.conv1.in_channels - 1, 1, 1] = 1.0
        enc.conv2.weight[0, 0, 1, 1] = 1.0
    return head


def _one_hot_basis() -> torch.Tensor:
    """Two spatial modes with hand-chosen support, so a projection is a number.

    Mode 0 is one cell of unit weight, so its normalised projection is the
    feature map AT that cell. Mode 1 puts weight 1 on the same cell and weight 2
    on a cell inside the burned block, so its projection is
    ``(1 * f_new + 2 * f_burned) / (1 + 4)`` and an unnormalised inner product
    would give a visibly different answer.
    """
    basis = torch.zeros(2, H, W, dtype=torch.float64)
    row, col = NEW_CELLS[0]
    basis[0, row, col] = 1.0
    basis[1, row, col] = 1.0
    basis[1, ROW0, COL0] = 2.0
    return basis


# --------------------------------------------------------------------------
# 1. the configuration itself
# --------------------------------------------------------------------------


def test_the_published_latent_configuration_is_still_what_these_flags_build() -> None:
    """The published ``latent_config`` block, rebuilt from the flags alone.

    WHAT WOULD MAKE THIS FAIL: any change to a ``LatentConfig`` default that a
    published run relied on without setting, which would move an archived model
    without anyone editing a run or a config.

    Sixteen of the twenty entries are defaults. That is the whole point: the
    eleven flags a training config sets are visible in every driver, and the
    defaults behind them are visible nowhere.
    """
    got = LatentConfig(**PUBLISHED_FLAGS).to_dict()

    assert got == PUBLISHED_LATENT_CONFIG, {
        key: (got.get(key), PUBLISHED_LATENT_CONFIG.get(key))
        for key in sorted(set(got) | set(PUBLISHED_LATENT_CONFIG))
        if got.get(key) != PUBLISHED_LATENT_CONFIG.get(key)
    }


# --------------------------------------------------------------------------
# 2. the checkpoint boundary
# --------------------------------------------------------------------------


def test_a_published_spec_reloads_onto_the_innovation_encoder_it_records() -> None:
    """A checkpoint that says ``innovation_channels: true`` must reload as one.

    WHAT WOULD MAKE THIS FAIL: a ``from_spec`` that drops the key or defaults it
    to ``False``, which would score every published checkpoint on the legacy
    three-channel encoder while its own ``model.json`` claims otherwise.

    The spec is the published block verbatim and carries no parameters, so the
    branch is decided by the flag alone and a shape mismatch cannot stand in for
    the assertion.
    """
    model = ContagionKernel.from_spec(
        {"config": {}, "latent_config": dict(PUBLISHED_LATENT_CONFIG)}
    )

    assert model.latent is not None
    assert model.latent.config.innovation_channels is True
    assert model.latent.encoder.conv1.in_channels == 4, (
        "the reloaded encoder takes the legacy three channels, so this checkpoint "
        "would be scored as a different model than the one that was fitted"
    )
    assert model.latent.spatial_modes == 2


def test_a_published_spec_round_trips_through_json_to_the_SAME_posterior() -> None:
    """Save and reload the published configuration and get the same numbers back.

    WHAT WOULD MAKE THIS FAIL: any part of the encoder that survives ``to_spec``
    in name but not in value, on a configuration that has BOTH the innovation
    channels and the spatial modes.

    The encoder weights are randomised BEFORE the round trip. Left at their
    shipped zero initialisation the posterior is identically zero and the two
    sides would agree whatever either of them did, which is the same trap the
    assertion that used to sit in the spatial playthrough fell into.
    """
    config = LatentConfig(**PUBLISHED_FLAGS)
    model = ContagionKernel(KernelConfig(), name="round_trip_probe", latent_config=config)
    assert model.latent is not None
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for param in model.latent.encoder.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))

    spec = model.to_spec()
    assert spec["latent_config"]["innovation_channels"] is True
    reloaded = ContagionKernel.from_spec(json.loads(json.dumps(spec)))
    assert reloaded.latent is not None

    draws = torch.Generator().manual_seed(3)
    b = (torch.rand(H, W, generator=draws, dtype=torch.float64) > 0.7).double()
    y = torch.clamp(
        b + (torch.rand(H, W, generator=draws, dtype=torch.float64) > 0.9).double(), max=1.0
    )
    p_zero = torch.rand(H, W, generator=draws, dtype=torch.float64)
    basis = torch.rand(2, H, W, generator=draws, dtype=torch.float64)

    mu_before, log_var_before = model.latent.posterior(b, y, p_zero, basis)
    mu_after, log_var_after = reloaded.latent.posterior(b, y, p_zero, basis)

    # Not a vacuous comparison: state the size of what is being compared.
    assert float(mu_before.detach().abs().max()) > 1.0, (
        "the randomisation did not take, so agreement below would be agreement about zero"
    )
    assert torch.equal(mu_before, mu_after), (
        float((mu_before - mu_after).abs().max()),
        [float(v) for v in mu_before.detach()],
    )
    assert torch.equal(log_var_before, log_var_after)


# --------------------------------------------------------------------------
# 3. the artifact boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("innovation", [True, False])
def test_the_run_artifact_cannot_record_an_encoder_that_did_not_run(innovation: bool) -> None:
    """The flag in the artifact is bound to the encoder's own input width.

    WHAT WOULD MAKE THIS FAIL: a report that carries the intention rather than
    the encoder, i.e. any state in which ``latent_report`` says ``true`` and the
    first convolution takes three channels, or the reverse.

    Both branches are asserted, so a report hardcoded to either answer fails on
    one of the two parametrisations rather than on neither.
    """
    head = LatentHead(LatentConfig(**{**PUBLISHED_FLAGS, "innovation_channels": innovation}))
    report = latent_report(head)

    assert report["present"] is True
    assert report["innovation_channels"] is innovation
    assert (head.encoder.conv1.in_channels == 4) is innovation, (
        f"the report says innovation_channels={report['innovation_channels']} while the "
        f"encoder takes {head.encoder.conv1.in_channels} input channels"
    )


# --------------------------------------------------------------------------
# 4. the two branches the published runs take AT ONCE
# --------------------------------------------------------------------------


def test_the_innovation_reaches_the_SPATIAL_projection_statistics() -> None:
    """The projections are of the innovation feature map, normalised.

    WHAT WOULD MAKE THIS FAIL: a spatial projection taken from the raw realised
    burn instead of the innovation (mode 0 would read 1.0 rather than 0.75), or
    an inner product that skips the division by ``<phi, phi>`` (mode 1 would read
    0.75 rather than 0.15).

    This is the combination 26 published run directories used and no test
    exercised: ``innovation_channels`` and ``spatial_modes`` are separate
    branches inside one forward pass, and covering them one at a time leaves the
    published path uncovered.
    """
    head = _tapped_head()
    with torch.no_grad():
        head.encoder.head.weight[0, 2] = 1.0  # mu[0] = projection onto mode 0
        head.encoder.head.weight[1, 3] = 1.0  # mu[1] = projection onto mode 1

    b, y = _burned(), _grown()
    p_zero = torch.full((H, W), P_ZERO, dtype=torch.float64)
    mu, _log_var = head.posterior(b, y, p_zero, _one_hot_basis())

    innovation_at_a_new_cell = 1.0 - P_ZERO
    assert float(mu[0].detach()) == pytest.approx(innovation_at_a_new_cell, abs=1e-15)
    assert float(mu[1].detach()) == pytest.approx(innovation_at_a_new_cell / 5.0, abs=1e-15)

    # The two values the named defects would produce, stated as numbers.
    assert float(mu[0].detach()) != pytest.approx(1.0, abs=1e-9)
    assert float(mu[1].detach()) != pytest.approx(innovation_at_a_new_cell, abs=1e-9)


def test_the_encoder_is_four_channels_wide_when_it_also_has_spatial_modes() -> None:
    """The extra input channel is not quietly dropped once a basis is required.

    WHAT WOULD MAKE THIS FAIL: an encoder that takes the legacy three channels
    whenever spatial modes are configured, which would leave the published
    configuration running the branch its artifacts say it is not running.
    """
    head = LatentHead(LatentConfig(**PUBLISHED_FLAGS))
    assert head.encoder.conv1.in_channels == 4
    assert head.encoder.head.in_features == (2 + 2) * 8

    legacy = LatentHead(LatentConfig(**{**PUBLISHED_FLAGS, "innovation_channels": False}))
    assert legacy.encoder.conv1.in_channels == 3
    assert legacy.encoder.head.in_features == (2 + 2) * 8


def test_the_two_pooled_statistics_are_different_functionals_of_one_feature_map() -> None:
    """Mean and max must both be there, because they answer different questions.

    WHAT WOULD MAKE THIS FAIL: an encoder that pools the same functional twice,
    for instance a second mean where the max belongs, which is what the module's
    own docstring says would leave the direction component with nothing to key
    on: a step that burns a little everywhere and a step that makes one run have
    the same mean and different maxima.

    On this scenario the mean of ``relu(innovation)`` over the domain is
    ``2 * 0.75 / 99`` and its maximum is ``0.75``, so the two statistics differ
    by a factor of 49.5 and neither can stand in for the other.
    """
    head = _tapped_head()
    with torch.no_grad():
        head.encoder.head.weight[0, 0] = 1.0  # mu[0] = mean over H and W
        head.encoder.head.weight[1, 1] = 1.0  # mu[1] = max over H and W

    b, y = _burned(), _grown()
    p_zero = torch.full((H, W), P_ZERO, dtype=torch.float64)
    mu, _log_var = head.posterior(b, y, p_zero, _one_hot_basis())

    want_mean = len(NEW_CELLS) * (1.0 - P_ZERO) / (H * W)
    want_max = 1.0 - P_ZERO
    assert float(mu[0].detach()) == pytest.approx(want_mean, abs=1e-15)
    assert float(mu[1].detach()) == pytest.approx(want_max, abs=1e-15)
    assert float(mu[1].detach()) / float(mu[0].detach()) == pytest.approx(H * W / len(NEW_CELLS))


def test_the_pooling_is_over_the_two_SPATIAL_axes_and_nothing_else() -> None:
    """Each statistic is one number per sample per channel, not per row or batch.

    WHAT WOULD MAKE THIS FAIL: pooling that includes the channel axis or the
    batch axis, which would mix samples in a batched fit and silently make one
    fire's posterior depend on the fires beside it in the batch.

    Checked by BATCHING: two independent steps are encoded together and each
    posterior must equal the one that step gets on its own. A pool that reached
    across the batch axis could not satisfy that, and a shape assertion alone
    could not have detected it.
    """
    head = _tapped_head()
    with torch.no_grad():
        head.encoder.head.weight[0, 0] = 1.0
        head.encoder.head.weight[1, 1] = 1.0

    b, y = _burned(), _grown()
    dormant = _burned()
    p_zero = torch.full((H, W), P_ZERO, dtype=torch.float64)

    alone_growing, _ = head.posterior(b, y, p_zero, _one_hot_basis())
    alone_dormant, _ = head.posterior(dormant, dormant, p_zero, _one_hot_basis())

    batched, _ = head.posterior(
        torch.stack([b, dormant]),
        torch.stack([y, dormant]),
        torch.stack([p_zero, p_zero]),
        torch.stack([_one_hot_basis(), _one_hot_basis()]),
    )

    assert tuple(batched.shape) == (2, head.dim)
    assert torch.equal(batched[0], alone_growing), (
        [float(v) for v in batched[0].detach()],
        [float(v) for v in alone_growing.detach()],
    )
    assert torch.equal(batched[1], alone_dormant)
    # The two steps must not be the same number, or the check above is vacuous.
    assert float((batched[0] - batched[1]).detach().abs().max()) > 0.0
