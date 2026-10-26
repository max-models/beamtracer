"""A traced neutralizer channel against the global model alone; placeholder data."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

wellbalanced = pytest.importorskip("wellbalanced.neutralizer")

from beamtracer.neutralizer import NeutralizerChannel  # noqa: E402


def neutralizer(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return wellbalanced.illustrative_deuterium_neutralizer(
            gas_inflow=6.0e21, **overrides
        )


@pytest.fixture(scope="module")
def parallel():
    channel = NeutralizerChannel(neutralizer(), beam_width=0.06, num_particles=40)
    return channel, channel.solve()


def test_a_parallel_beam_reproduces_the_global_model_along_the_axis(parallel) -> None:
    _, result = parallel
    assert result.converged
    assert result.transmission == pytest.approx(1.0, abs=1e-9)
    axis = result.axis
    assert result.neutral_fraction == pytest.approx(axis.neutral_fraction, abs=2e-4)
    along_axis = np.trapezoid(axis.fractions, axis.position, axis=0) / axis.position[-1]
    assert result.path_fractions == pytest.approx(along_axis, abs=2e-4)
    assert result.plasma.electron_density == pytest.approx(
        axis.plasma.electron_density, rel=1e-3
    )
    assert sum(result.exit_fractions.values()) == pytest.approx(1.0, abs=1e-9)


def test_a_diverging_beam_loses_particles_and_neutrals_on_the_walls() -> None:
    channel = NeutralizerChannel(
        neutralizer(), beam_width=0.08, divergence=0.01, num_particles=60
    )
    result = channel.solve()
    assert result.converged
    assert 0.5 < result.transmission < 0.95
    assert result.neutral_fraction < result.axis.neutral_fraction
    # particles lost early feed the plasma less
    assert result.path_fractions.sum() < 1.0 - 1e-3
    assert result.plasma.electron_density < result.axis.plasma.electron_density
    # the wide angles are the ones lost: the neutrals that leave diverge less
    assert 0.0 < result.neutral_divergence < 0.01


def test_the_beam_case_is_checked(parallel) -> None:
    channel, _ = parallel
    config = channel.beam_config(1e19, 1e17)
    assert config.charge_changing.densities == {"gas": 1e19, "plasma": 1e17}
    assert config.compensation[0].fraction == 1.0
    with pytest.raises(ValueError, match="fit in the channel"):
        NeutralizerChannel(neutralizer(), beam_width=0.2)
    with pytest.raises(ValueError, match="divide"):
        NeutralizerChannel(neutralizer(), beam_width=0.06, cell=0.007).beam_config(
            1e19, 1e17
        )


def test_analog_trajectories_reproduce_the_parallel_beam(parallel) -> None:
    """One state per trajectory: the same neutral fraction, within the statistics."""
    num = 1000
    channel = NeutralizerChannel(
        neutralizer(), beam_width=0.06, num_particles=num, mode="analog", seed=5
    )
    result = channel.solve()
    _, reference = parallel
    assert result.converged
    assert result.transmission == pytest.approx(1.0, abs=1e-9)
    fraction = reference.neutral_fraction
    spread = np.sqrt(fraction * (1.0 - fraction) / num)
    assert result.neutral_fraction == pytest.approx(fraction, abs=4.0 * spread)
    assert result.path_fractions == pytest.approx(
        reference.path_fractions, abs=4.0 * 0.5 / np.sqrt(num)
    )
    hits = result.model.plane(channel.exit_plane).fractions
    assert set(np.unique(hits)) <= {0.0, 1.0}
