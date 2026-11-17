"""Anderson mixing and the coarse phase of the Vlasov iteration."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer import BeamOptics
from beamtracer.solver import AndersonMixer, BeamParticles
from beamtracer.solver.vlasov import thin_particles
from beamtracer.tests.unit.test_beam_optics import _extraction


def _contraction(x):
    """G(x) = M x + b with the slow modes of a Vlasov iteration (eigenvalues to 0.95)."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    m = q @ np.diag(np.linspace(-0.6, 0.95, 20)) @ q.T
    return m @ x + np.arange(20.0)


def _iterate(update, start, target, tol=1e-10, limit=500):
    """Iterations of ``x -> update(x, G(x))`` to reach the fixed point ``target``."""
    x = start
    for n in range(1, limit + 1):
        x = update(x, _contraction(x))
        if np.max(np.abs(x - target)) < tol:
            return n
    return limit


def test_anderson_beats_linear_mixing_on_a_linear_map() -> None:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    m = q @ np.diag(np.linspace(-0.6, 0.95, 20)) @ q.T
    target = np.linalg.solve(np.eye(20) - m, np.arange(20.0))
    start = np.zeros(20)
    linear = _iterate(lambda x, g: g, start, target)
    mixer = AndersonMixer(depth=5)
    anderson = _iterate(mixer.update, start, target)
    assert anderson < linear / 5


def test_anderson_without_history_is_linear_mixing() -> None:
    mixer = AndersonMixer(depth=3, relaxation=0.4)
    rho, deposit = np.ones((3, 2)), np.full((3, 2), 2.0)
    np.testing.assert_allclose(mixer.update(rho, deposit), 1.4)
    mixer.reset()
    np.testing.assert_allclose(mixer.update(rho, deposit), 1.4)
    with pytest.raises(ValueError, match="depth"):
        AndersonMixer(depth=0)


def test_thinning_keeps_the_current_of_every_species() -> None:
    n = 11
    particles = BeamParticles(
        np.zeros((3, 2 * n)), np.zeros((3, 2 * n)), np.ones(2 * n), np.ones(2 * n),
        np.concatenate([np.full(n, 1.0), np.full(n, 3.0)]),
        np.repeat(np.array([0, 1], dtype=np.int32), n),
    )  # fmt: skip
    thinned = thin_particles(particles, 4)
    assert len(thinned) == 6
    for species, total in ((0, n), (1, 3 * n)):
        assert thinned.current[thinned.species == species].sum() == pytest.approx(total)


def _converged(**settings) -> BeamOptics:
    return BeamOptics(_extraction(1.0).replace(iterations=100, **settings)).run()


def test_anderson_converges_in_fewer_iterations_to_the_same_beam() -> None:
    linear = _converged(relaxation=0.8)
    anderson = _converged(relaxation=1.0, mixing="anderson")
    assert anderson.stats["converged"] and linear.stats["converged"]
    assert len(anderson.convergence_history) < len(linear.convergence_history)
    a, b = anderson.beam_moments(7.5e-3), linear.beam_moments(7.5e-3)
    assert a.current == pytest.approx(b.current, rel=1e-6)
    assert a.rms_size == pytest.approx(b.rms_size, rel=1e-4)
    assert np.abs(anderson.potential - linear.potential).max() < 1e-3


def test_the_coarse_phase_traces_fewer_trajectories_then_all() -> None:
    model = _converged(mixing="anderson", coarse_particles=0.25, coarse_tolerance=0.5)
    counts = [entry["trajectories"] for entry in model.convergence_history]
    assert counts[0] == 25 and counts[-1] == 100
    assert counts == sorted(counts)  # coarse first, then all
    assert model.stats["converged"]
    full = _converged(mixing="anderson")
    a, b = model.beam_moments(7.5e-3), full.beam_moments(7.5e-3)
    assert a.rms_size == pytest.approx(b.rms_size, rel=1e-4)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"mixing": "broyden"}, "mixing"),
        ({"anderson_depth": 0}, "anderson_depth"),
        ({"coarse_particles": 0.0}, "coarse_particles"),
        ({"coarse_particles": 0.5}, "coarse_tolerance"),
    ],
)
def test_bad_settings_are_rejected(settings, match) -> None:
    with pytest.raises(ValueError, match=match):
        _extraction(1.0).replace(**settings)


def test_the_settings_round_trip() -> None:
    config = _extraction(1.0).replace(
        mixing="anderson", anderson_depth=3, coarse_particles=0.5, coarse_tolerance=0.1
    )
    again = type(config).from_dict(config.to_dict())
    assert (again.mixing, again.anderson_depth) == ("anderson", 3)
    assert (again.coarse_particles, again.coarse_tolerance) == (0.5, 0.1)
