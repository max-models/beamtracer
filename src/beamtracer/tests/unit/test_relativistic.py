"""The relativistic tracers: exact energy gain and flight time, gyroradius, slow-ion limit.

Electrons accelerated through 1 MV gain exactly ``q U`` and reach ``gamma`` of about 3 at
a speed below ``c`` (the classical tracers pass it), their deposited charge is the current
times the relativistic flight time, and in a magnetic field they gyrate on ``gamma m v /
(q B)``; slow protons follow the classical tracers. The compiled tracers agree with the
NumPy one, and the beams and the normalized emittance use ``beta gamma``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants

from beamtracer import BeamOptics
from beamtracer.solver import (
    BeamParticles,
    BeamTracer,
    IonOpticsMesh,
    PlaneHits,
    StaggeredElectricField,
    beam_with_energy,
    plane_moments,
)
from beamtracer.solver.fields3d import StaggeredElectricField3D
from beamtracer.solver.mesh3d import IonOpticsMesh3D
from beamtracer.solver.tracer3d import BeamTracer3D, beam_on_surface

C = constants.c
M_E = constants.m_e
E = constants.e
#: 1 MV over 2 cm: an electron from rest ends at 1 MeV, gamma = 2.96.
LENGTH, VOLTAGE = 0.02, 1.0e6


def _electrons(position: np.ndarray, velocity: np.ndarray, current=1e-6):
    n = position.shape[1]
    return BeamParticles(
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        np.full(n, -E),
        np.full(n, M_E),
        np.full(n, current),
        np.zeros(n, dtype=np.int32),
    )


def _exact(v0: float, distance: float) -> tuple[float, float]:
    """The speed after ``distance`` in the field and the flight time, hyperbolic motion."""
    a = E * VOLTAGE / LENGTH / M_E  # du/dt
    gamma0 = 1.0 / np.sqrt(1.0 - (v0 / C) ** 2)
    gamma = gamma0 + a * distance / C**2
    u0, u = gamma0 * v0, C * np.sqrt(gamma * gamma - 1.0)
    return u / gamma, (u - u0) / a


def _flight_time(result, volume: np.ndarray, n: int, current: float) -> float:
    """The time the charge density holds: the node charges over the trajectories' current."""
    return -float(np.sum(result.charge_density * volume)) / (n * current)


def _node_volumes(shape, h, dims):
    volume = np.full(shape, h**dims)
    for axis in range(len(shape)):
        index = [slice(None)] * len(shape)
        for end in (0, -1):
            index[axis] = end
            volume[tuple(index)] /= 2.0
    return volume


def _plates_2d() -> tuple[IonOpticsMesh, StaggeredElectricField]:
    mesh = IonOpticsMesh((81, 11), (0.0, 0.0), LENGTH / 80, "planar")
    potential = VOLTAGE / LENGTH * mesh.position_x(np.arange(81))[:, None] * np.ones(11)
    return mesh, StaggeredElectricField(mesh, potential)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("integrator", ["rk4", "boris"])
def test_electrons_gain_the_exact_energy_below_c(integrator, compiled) -> None:
    if compiled and not BeamTracer.compiled_available():
        pytest.skip("Pyccel is not available")
    mesh, field = _plates_2d()
    plane = LENGTH - 1e-9
    n, v0, current = 5, 1e6, 1e-6
    position = np.stack([np.zeros(n), np.linspace(1e-4, 4e-4, n), np.zeros(n)])
    velocity = np.stack([np.full(n, v0), np.zeros(n), np.zeros(n)])
    speeds = {}
    for relativistic in (False, True):
        tracer = BeamTracer(
            mesh, field, planes=[plane], integrator=integrator, compiled=compiled,
            relativistic=relativistic, step_fraction=0.25,
        )  # fmt: skip
        result = tracer.trace(_electrons(position, velocity, current))
        assert np.all(result.status == 0)
        hits = result.planes[0]
        speeds[relativistic] = np.linalg.norm(hits.velocity, axis=0)
    speed, flight = _exact(v0, plane)
    np.testing.assert_allclose(speeds[True], speed, rtol=1e-5)
    assert speeds[True].max() < C < speeds[False].min()  # the classical tracer passes c
    # 1 MeV at the plane, from the crossing velocities (Boris: second order in the step)
    moments = plane_moments(hits, cylindrical=False, mass=M_E)
    kinetic = (1.0 / np.sqrt(1.0 - (v0 / C) ** 2) - 1.0) * M_E * C**2 / E
    assert moments.mean_energy == pytest.approx(
        VOLTAGE * plane / LENGTH + kinetic, rel=2e-5
    )
    # the deposit: the current times the time of flight (to the end, one step's time more
    # at most), so the charge per step uses the time, at the speed u / gamma
    time = _flight_time(result, _node_volumes(mesh.shape, mesh.h, 2), n, current)
    end_speed, end_flight = _exact(v0, LENGTH)
    assert end_flight * (1 - 1e-4) < time < end_flight + 0.25 * mesh.h / end_speed


def test_the_relativistic_compiled_tracer_agrees_with_the_numpy_one() -> None:
    if not BeamTracer.compiled_available():
        pytest.skip("Pyccel is not available")
    mesh, field = _plates_2d()
    rng = np.random.default_rng(0)
    n = 30
    position = np.stack([np.zeros(n), rng.uniform(1e-4, 4e-4, n), np.zeros(n)])
    velocity = np.stack([np.full(n, 3e7), rng.normal(0, 1e6, n), rng.normal(0, 1e6, n)])
    for integrator in ("rk4", "boris"):
        results = [
            BeamTracer(
                mesh,
                field,
                planes=[0.01, 0.019],
                integrator=integrator,
                compiled=compiled,
                relativistic=True,
            ).trace(_electrons(position, velocity))  # fmt: skip
            for compiled in (True, False)
        ]
        a, b = results
        np.testing.assert_array_equal(a.status, b.status)
        assert a.steps == b.steps
        np.testing.assert_allclose(
            a.charge_density, b.charge_density, rtol=0,
            atol=1e-10 * np.abs(b.charge_density).max(),
        )  # fmt: skip
        for pa, pb in zip(a.planes, b.planes, strict=True):
            np.testing.assert_allclose(pa.point, pb.point, rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(pa.velocity, pb.velocity, rtol=1e-10)


@pytest.mark.parametrize("integrator", ["rk4", "boris"])
def test_the_3d_tracer_gains_the_exact_energy(integrator) -> None:
    mesh = IonOpticsMesh3D((41, 9, 9), (0.0, -2e-3, -2e-3), LENGTH / 40)
    x, _, _ = mesh.nodes()
    field = StaggeredElectricField3D(mesh, VOLTAGE / LENGTH * x)
    plane = LENGTH - 1e-9
    n, v0, current = 6, 1e6, 1e-6
    position = np.stack([np.zeros(n), np.linspace(-1e-3, 1e-3, n), np.zeros(n)])
    velocity = np.stack([np.full(n, v0), np.zeros(n), np.zeros(n)])
    tracer = BeamTracer3D(
        mesh, field, planes=[plane], integrator=integrator, relativistic=True,
        step_fraction=0.25,
    )  # fmt: skip
    result = tracer.trace(_electrons(position, velocity, current))
    assert np.all(result.status == 0)
    speed, _ = _exact(v0, plane)
    np.testing.assert_allclose(
        np.linalg.norm(result.planes[0].velocity, axis=0), speed, rtol=1e-5
    )
    time = _flight_time(result, _node_volumes(mesh.shape, mesh.h, 3), n, current)
    end_speed, end_flight = _exact(v0, LENGTH)
    assert end_flight * (1 - 1e-4) < time < end_flight + 0.25 * mesh.h / end_speed


#: A 0.9 c electron in 0.5 T: gamma = 2.29, the gyroradius gamma m v / (e B) = 7.0 mm.
SPEED, FIELD = 0.9 * C, 0.5
RADIUS = M_E * SPEED / np.sqrt(1 - 0.81) / (E * FIELD)


@pytest.mark.parametrize("integrator", ["rk4", "boris"])
def test_the_3d_tracer_gyrates_on_the_relativistic_radius(integrator) -> None:
    mesh = IonOpticsMesh3D((81, 81, 5), (0.0, -20e-3, -1e-3), 0.5e-3)
    field = StaggeredElectricField3D(mesh, np.zeros(mesh.shape))
    magnetic = np.zeros((3, *mesh.shape))
    magnetic[2] = FIELD
    tracer = BeamTracer3D(
        mesh, field, magnetic, integrator=integrator, save_trajectories=1,
        step_fraction=0.2, relativistic=True,
    )  # fmt: skip
    start = np.array([[1e-3], [0.0], [0.0]])
    result = tracer.trace(_electrons(start, np.array([[SPEED], [0.0], [0.0]])))
    path = result.trajectories[0]
    # an electron turns to +y: the centre of its circle is at (1 mm, +r)
    distance = np.hypot(path[:, 0] - 1e-3, path[:, 1] - RADIUS)
    np.testing.assert_allclose(distance, RADIUS, rtol=2e-3)


@pytest.mark.parametrize("integrator", ["rk4", "boris"])
def test_the_numpy_tracer_gyrates_on_the_relativistic_radius(integrator) -> None:
    mesh = IonOpticsMesh((81, 81), (0.0, 0.0), 0.5e-3, "planar")
    field = StaggeredElectricField(mesh, np.zeros(mesh.shape))

    def magnetic(x, y):
        return np.zeros_like(x), np.zeros_like(x), np.full_like(x, FIELD)

    tracer = BeamTracer(
        mesh, field, magnetic, integrator=integrator, save_trajectories=1,
        step_fraction=0.2, relativistic=True,
    )  # fmt: skip
    start = np.array([[1e-3], [5e-3], [0.0]])
    result = tracer.trace(_electrons(start, np.array([[SPEED], [0.0], [0.0]])))
    path = result.trajectories[0]
    distance = np.hypot(path[:, 0] - 1e-3, path[:, 1] - (5e-3 + RADIUS))
    np.testing.assert_allclose(distance, RADIUS, rtol=2e-3)
    assert result.status[0] == 1  # a half turn on it heads back and leaves at x = 0


def test_slow_protons_follow_the_classical_tracer() -> None:
    from beamtracer.tests.unit.test_compiled_tracer import _planar

    classical = BeamOptics(_planar()).run()
    relativistic = BeamOptics(_planar(relativistic=True)).run()
    assert relativistic.tracer.relativistic
    for pa, pb in zip(
        relativistic.state.trace.planes, classical.state.trace.planes, strict=True
    ):
        assert pa.velocity.shape == pb.velocity.shape
        # 3 keV protons, v / c = 2.5e-3: the paths differ by about (v / c)^2
        np.testing.assert_allclose(pa.velocity, pb.velocity, rtol=1e-4, atol=10.0)


def test_relativistic_beams_have_the_speed_of_their_energy() -> None:
    mass = M_E / constants.atomic_mass  # [u]
    gamma = 1.0 + 1e6 * E / (M_E * C**2)
    beam, j_over_v = beam_with_energy(
        64, 100.0, -1.0, mass, 1e6, 0.0, 0.0, (0.0, 0.0), (0.0, 1e-3),
        cylindrical=False, relativistic=True,
    )  # fmt: skip
    speed = C * np.sqrt(1.0 - 1.0 / gamma**2)
    np.testing.assert_allclose(np.linalg.norm(beam.velocity, axis=0), speed, rtol=1e-12)
    assert j_over_v == pytest.approx(100.0 / speed, rel=1e-12)
    # the spreads are drawn in u = gamma v: every velocity stays below c
    hot, _ = beam_with_energy(
        64, 100.0, -1.0, mass, 1e6, 1e5, 1e5, (0.0, 0.0), (0.0, 1e-3),
        cylindrical=True, relativistic=True,
    )  # fmt: skip
    assert np.linalg.norm(hot.velocity, axis=0).max() < C
    surface, jv = beam_on_surface(
        50, 100.0, -1.0, mass, 1e6, 0.0, 0.0, (0.0, 0.0, 0.0), (1e-3, 1e-3),
        relativistic=True,
    )  # fmt: skip
    np.testing.assert_allclose(surface.velocity[0], speed, rtol=1e-12)
    assert jv == pytest.approx(100.0 / speed, rel=1e-12)
    # slow ions: the classical speed to (v / c)^2
    slow, _ = beam_with_energy(
        4, 1.0, 1.0, 1.0, 3000.0, 0.0, 0.0, (0.0, 0.0), (0.0, 1e-3),
        cylindrical=False, relativistic=True,
    )  # fmt: skip
    classical = np.sqrt(2 * 3000.0 * E / constants.atomic_mass)
    np.testing.assert_allclose(slow.velocity[0], classical, rtol=1e-5)


def test_the_normalized_emittance_takes_beta_gamma() -> None:
    rng = np.random.default_rng(2)
    n = 200
    vx = 0.6 * C  # beta gamma = 0.75
    y = rng.normal(0, 1e-3, n)
    angle = 2.0 * y + rng.normal(0, 1e-3, n)
    hits = PlaneHits(0.0)
    hits.append(
        np.stack([np.zeros(n), y, np.zeros(n)]),
        np.stack([np.full(n, vx), angle * vx, np.zeros(n)]),
        np.ones(n),
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=np.int8),
    )
    moments = plane_moments(hits, cylindrical=False, mass=M_E)
    lorentz = 1.0 / np.sqrt(1.0 - (vx / C) ** 2 * (1.0 + angle**2))
    beta_gamma = float(np.mean(lorentz)) * 0.6
    assert moments.normalized_emittance == pytest.approx(
        moments.emittance * beta_gamma, rel=1e-12
    )
    assert beta_gamma == pytest.approx(0.75, rel=1e-4)
    energy = (lorentz - 1.0) * M_E * C**2 / E
    assert moments.mean_energy == pytest.approx(float(np.mean(energy)), rel=1e-10)
