"""Tests of the ion-optics solver against analytic results."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants
from scipy.constants import epsilon_0

from beamtracer.solver import (
    AxisymmetricMagneticField,
    BeamParticles,
    BeamTracer,
    BoltzmannElectrons,
    Electrode,
    FaceCondition,
    IonOpticsMesh,
    IonOpticsPoisson,
    PlaneHits,
    StaggeredElectricField,
    beam_with_energy,
    mismatch,
    plane_moments,
    solenoid_on_axis,
)


def _plates(shape=(21, 11), h=1e-3, geometry="planar", **faces) -> IonOpticsMesh:
    faces = {
        "x_min": FaceCondition("dirichlet", 0.0),
        "x_max": FaceCondition("dirichlet", 100.0),
        **faces,
    }
    return IonOpticsMesh(
        shape=shape, origin=(0.0, 0.0), h=h, geometry=geometry, faces=faces
    )


# -- mesh -------------------------------------------------------------------------------------


def test_wall_distance_is_bracketed_in_255ths_of_a_cell() -> None:
    h = 1e-3
    surface = 5.3e-3
    mesh = IonOpticsMesh(
        shape=(11, 5),
        origin=(0.0, 0.0),
        h=h,
        geometry="planar",
        electrodes=[Electrode(lambda x, y: x >= surface, 10.0)],
    )
    assert mesh.near_solid[5].all() and not mesh.near_solid[4].any()
    distance = mesh.distance[1, 5]  # node 5 towards +x
    assert np.allclose(distance, np.floor(0.3 * 255 + 1) / 255, atol=1 / 255)
    assert (mesh.fixed_value[6:] == 10.0).all()


def test_cylindrical_mesh_needs_the_axis() -> None:
    with pytest.raises(ValueError, match="axis"):
        IonOpticsMesh(shape=(5, 5), origin=(0.0, 1e-3), h=1e-3)


# -- Poisson ----------------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", ["planar", "cylindrical"])
def test_potential_between_plates_is_linear(geometry: str) -> None:
    mesh = _plates(geometry=geometry)
    phi = IonOpticsPoisson(mesh).solve()
    x = mesh.x[:, None]
    assert np.allclose(phi, 100.0 * x / x[-1], atol=1e-9)


def test_coaxial_electrode_off_the_nodes() -> None:
    """A rod of radius a at V0 inside a grounded tube: phi = V0 ln(R/r) / ln(R/a)."""
    h = 0.25e-3
    a, big_r, v0 = 1.13e-3, 10e-3, 1000.0
    mesh = IonOpticsMesh(
        shape=(9, round(big_r / h) + 1),
        origin=(0.0, 0.0),
        h=h,
        electrodes=[Electrode(lambda x, r: r <= a, v0)],
        faces={"y_max": FaceCondition("dirichlet", 0.0)},
    )
    phi = IonOpticsPoisson(mesh).solve()
    r = mesh.y
    free = r > a
    exact = v0 * np.log(big_r / r[free]) / np.log(big_r / a)
    assert np.max(np.abs(phi[4, free] - exact)) < 2e-3 * v0


def test_uniform_charge_in_a_grounded_tube() -> None:
    """phi = rho (R^2 - r^2) / (4 epsilon_0) for a uniformly charged cylinder."""
    h = 0.2e-3
    big_r, rho = 8e-3, 1e-6
    mesh = IonOpticsMesh(
        shape=(5, round(big_r / h) + 1),
        origin=(0.0, 0.0),
        h=h,
        faces={"y_max": FaceCondition("dirichlet", 0.0)},
    )
    phi = IonOpticsPoisson(mesh).solve(np.full(mesh.shape, rho))
    exact = rho * (big_r**2 - mesh.y**2) / (4 * epsilon_0)
    assert np.allclose(phi[2], exact, rtol=1e-9, atol=1e-9 * exact.max())


def test_boltzmann_electrons_satisfy_the_nonlinear_equation() -> None:
    mesh = _plates()
    plasma = BoltzmannElectrons(
        rho_e=1e-6, electron_temperature=5.0, plasma_potential=50.0
    )
    phi = IonOpticsPoisson(mesh).solve(plasma=plasma)
    # interior residual of laplace(phi) = rho_e exp((phi - Up)/Te) / epsilon_0
    lap = (
        phi[2:, 1:-1]
        + phi[:-2, 1:-1]
        + phi[1:-1, 2:]
        + phi[1:-1, :-2]
        - 4 * phi[1:-1, 1:-1]
    ) / mesh.h**2
    source = 1e-6 * np.exp((phi[1:-1, 1:-1] - 50.0) / 5.0) / epsilon_0
    assert np.allclose(lap, source, rtol=1e-6, atol=1e-6 * source.max())


# -- fields -----------------------------------------------------------------------------------


def test_staggered_field_of_a_linear_potential_is_uniform() -> None:
    mesh = _plates(geometry="cylindrical")
    phi = IonOpticsPoisson(mesh).solve()
    field = StaggeredElectricField(mesh, phi)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, mesh.upper[0], 200)
    r = rng.uniform(0, mesh.upper[1], 200)
    ex, er = field(x, r)
    assert np.allclose(ex, -100.0 / mesh.upper[0]) and np.allclose(er, 0.0, atol=1e-9)


def test_magnetic_field_is_divergence_free_to_its_order() -> None:
    # B_z is interpolated linearly between the samples while B_r uses the finite-difference
    # derivative, so div B vanishes only to O(dz^2) (here 0.2 %).
    z = np.linspace(-0.3, 0.3, 1201)
    bz = solenoid_on_axis(z, 0.0, 0.2, 0.05, 0.2)
    field = AxisymmetricMagneticField(z[0], z[1] - z[0], bz, order=6)
    assert field(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.2, rel=1e-6)
    # div B = dBz/dz + (1/r) d(r Br)/dr
    zz, rr, d = 0.08, 0.02, 1e-5

    def bz_at(z):
        return field(np.array([z]), np.array([rr]))[0]

    def rbr(r):
        return r * field(np.array([zz]), np.array([r]))[1]

    dbz = (bz_at(zz + d) - bz_at(zz - d)) / (2 * d)
    div = dbz + (rbr(rr + d) - rbr(rr - d)) / (2 * d) / rr
    assert abs(div[0]) < 1e-2 * abs(dbz[0])


# -- particles --------------------------------------------------------------------------------


def test_cylindrical_beam_carries_j_times_the_disc_area() -> None:
    particles, j_over_v = beam_with_energy(
        500,
        1000.0,
        1.0,
        1.0,
        10.0,
        0.0,
        1.0,
        (0.0, 0.0),
        (0.0, 2e-3),
        cylindrical=True,
    )
    assert particles.current.sum() == pytest.approx(1000.0 * np.pi * 4e-6, rel=1e-3)
    v = np.sqrt(2 * 10.0 * constants.e / (constants.atomic_mass))
    assert j_over_v == pytest.approx(1000.0 / v)
    assert np.all(particles.velocity[0] > 0)


def test_sobol_beam_is_the_same_every_call() -> None:
    args = (100, 10.0, 1.0, 1.0, 5.0, 0.1, 0.5, (0.0, 0.0), (0.0, 1e-3))
    a, _ = beam_with_energy(*args, cylindrical=True)
    b, _ = beam_with_energy(*args, cylindrical=True)
    assert np.array_equal(a.velocity, b.velocity)


def test_energy_gain_and_deposited_charge_in_a_uniform_field() -> None:
    mesh = _plates(shape=(41, 11), geometry="planar")
    phi = IonOpticsPoisson(mesh).solve()
    tracer = BeamTracer(
        mesh, StaggeredElectricField(mesh, phi), planes=[mesh.upper[0] - 1e-9]
    )
    n = 5
    # a negative ion (charge -e) accelerated by +100 V from rest-ish
    m = constants.atomic_mass
    v0 = 1e3
    particles = BeamParticles(
        position=np.stack([np.zeros(n), np.linspace(2e-3, 8e-3, n), np.zeros(n)]),
        velocity=np.stack([np.full(n, v0), np.zeros(n), np.zeros(n)]),
        charge=np.full(n, -constants.e),
        mass=np.full(n, m),
        current=np.full(n, 1e-6),
        species=np.zeros(n, dtype=np.int32),
    )
    result = tracer.trace(particles)
    hits = result.planes[0]
    energy = 0.5 * m * np.sum(hits.velocity**2, axis=0) / constants.e
    assert np.allclose(energy, 100.0 + 0.5 * m * v0**2 / constants.e, rtol=1e-5)
    # the deposited charge is -I (a negative ion) times the transit time t
    a = constants.e * 100.0 / (m * mesh.upper[0])
    t = (-v0 + np.sqrt(v0**2 + 2 * a * mesh.upper[0])) / a
    volume = np.full(mesh.shape, mesh.h**2)
    volume[[0, -1], :] /= 2
    volume[:, [0, -1]] /= 2
    total = np.sum(result.charge_density * volume)
    assert total == pytest.approx(-n * 1e-6 * t, rel=1e-3)


def test_busch_theorem_through_a_solenoid() -> None:
    """Busch's theorem: r^2 (omega + q B / 2m) is conserved (paraxially)."""
    h = 1e-3
    mesh = IonOpticsMesh(shape=(601, 41), origin=(0.0, 0.0), h=h)
    z = np.arange(-0.1, 0.7, 0.5e-3)
    bfield = AxisymmetricMagneticField(
        z[0], 0.5e-3, solenoid_on_axis(z, 0.3, 0.2, 0.05, 0.1)
    )
    zero = StaggeredElectricField(mesh, np.zeros(mesh.shape))
    tracer = BeamTracer(mesh, zero, bfield, planes=[0.29], step_fraction=0.25)
    m = constants.m_p
    v = np.sqrt(2 * 1e4 * constants.e / m)
    particles = BeamParticles(
        position=np.array([[0.0], [5e-3], [0.0]]),
        velocity=np.array([[v], [0.0], [0.0]]),
        charge=np.array([constants.e]),
        mass=np.array([m]),
        current=np.array([1e-9]),
        species=np.zeros(1, dtype=np.int32),
    )
    hits = tracer.trace(particles).planes[0]
    point, vel = hits.point[:, 0], hits.velocity[:, 0]
    r = np.hypot(point[1], point[2])
    omega = (point[1] * vel[2] - point[2] * vel[1]) / r**2
    b_here = bfield(np.array([0.29]), np.array([r]))[0][0]
    canonical = r**2 * (omega + constants.e * b_here / (2 * m))
    # at the start omega = 0 in the solenoid's fringe field
    b_start = bfield(np.array([0.0]), np.array([5e-3]))[0][0]
    canonical_start = 25e-6 * constants.e * b_start / (2 * m)
    # B(r) r^2 / 2 stands in for the enclosed flux: O(B'' r^4 / 16), 1e-3 of the scale here
    scale = constants.e * b_here / (2 * m) * r**2
    assert canonical == pytest.approx(canonical_start, abs=1e-3 * scale)


# -- diagnostics ------------------------------------------------------------------------------


def test_ring_moments_equal_the_moments_of_the_rotated_rings() -> None:
    rng = np.random.default_rng(1)
    n = 300
    r = rng.uniform(0, 3e-3, n)
    rp = 5.0 * r + rng.normal(0, 1e-3, n)
    ap = rng.normal(0, 2e-3, n)
    w = rng.uniform(0.5, 1.5, n)
    hits = PlaneHits(0.0)
    hits.append(
        np.stack([np.zeros(n), r, np.zeros(n)]),
        np.stack([np.ones(n), rp, ap]),
        w,
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=np.int8),
    )
    ring = plane_moments(hits, cylindrical=True, mass=1.0)
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    x = (r[:, None] * np.cos(theta)).ravel()
    xp = (rp[:, None] * np.cos(theta) - ap[:, None] * np.sin(theta)).ravel()
    ww = np.repeat(w, theta.size)
    xx, pp, xpp = (np.average(q, weights=ww) for q in (x * x, xp * xp, x * xp))
    assert ring.emittance == pytest.approx(np.sqrt(xx * pp - xpp**2), rel=1e-9)
    assert ring.alpha == pytest.approx(-xpp / ring.emittance, rel=1e-9)


def test_matched_beam_has_no_mismatch() -> None:
    assert mismatch(1.02, 0.11, 1.02, 0.11) == pytest.approx(0.0, abs=1e-12)
    assert mismatch(0.0, 0.2, 1.02, 0.11) > 0.1
