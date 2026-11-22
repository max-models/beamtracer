"""Tests for the beam's own magnetic field (Ampère's law)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants
from scipy.constants import mu_0

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    SpaceChargeCompensation,
)
from beamtracer.config import BeamSpecies3D
from beamtracer.solver.mesh import IonOpticsMesh
from beamtracer.solver.mesh3d import IonOpticsMesh3D
from beamtracer.solver.self_field import (
    NodeMagneticField,
    VectorPotential3D,
    axisymmetric_field,
    field_2d,
    planar_field,
)

#: A 25 MeV proton beam: beta^2 = 0.053 (non-relativistic), so its pinch is visible.
ENERGY = 25e6
SPEED = np.sqrt(2.0 * ENERGY * constants.e / (1.007276 * constants.atomic_mass))
BETA2 = (SPEED / constants.c) ** 2


def test_the_axisymmetric_field_is_exact_for_a_current_linear_in_r_squared() -> None:
    mesh = IonOpticsMesh((3, 11), (0.0, 0.0), 1e-3, "cylindrical")
    r = mesh.y
    j0, outer = 1e6, 10e-3
    jx = np.broadcast_to(j0 * (1.0 - r**2 / outer**2), mesh.shape)
    b = axisymmetric_field(mesh, jx)
    enclosed = np.pi * j0 * (r**2 - r**4 / (2.0 * outer**2))
    exact = np.where(r > 0, mu_0 * enclosed / (2.0 * np.pi * np.maximum(r, 1e-30)), 0.0)
    assert b[1] == pytest.approx(exact, rel=1e-12, abs=1e-15)
    assert b[:, 0] == pytest.approx(0.0)


def test_the_planar_field_of_a_uniform_current() -> None:
    j0 = 2e5
    free = IonOpticsMesh((3, 11), (0.0, -5e-3), 1e-3, "planar")
    assert planar_field(free, np.full(free.shape, j0))[0] == pytest.approx(
        mu_0 * j0 * free.y
    )
    half = IonOpticsMesh((3, 6), (0.0, 0.0), 1e-3, "planar", symmetry_plane=True)
    assert planar_field(half, np.full(half.shape, j0))[1] == pytest.approx(
        mu_0 * j0 * half.y
    )


def test_the_node_field_interpolates_bilinearly() -> None:
    mesh = IonOpticsMesh((3, 3), (0.0, 0.0), 1.0, "planar")
    values = mesh.x[:, None] + 2.0 * mesh.y[None, :]
    field = NodeMagneticField(mesh, values)
    bx, by, bz = field(np.array([0.5, 1.5, 9.0]), np.array([0.25, 1.0, 1.0]))
    assert bz == pytest.approx([1.0, 3.5, 4.0])  # constant beyond the mesh
    assert np.all(bx == 0.0) and np.all(by == 0.0)
    cylinder = IonOpticsMesh((3, 3), (0.0, 0.0), 1.0, "cylindrical")
    assert field_2d(cylinder, np.ones(cylinder.shape)).values[0, 2] == pytest.approx(
        mu_0
    )


@pytest.mark.parametrize("planes", [(), ("y_min",)])
def test_the_vector_potential_converges_to_a_manufactured_field(planes) -> None:
    # A_x = cos(pi y / 2 W) sin(pi z / W) with y in [0, W] (symmetry plane at y = 0) or
    # sin(pi y / W) sin(pi z / W) with y in [0, W]; z in [0, W]; uniform in x
    width = 1e-2

    def error(n: int) -> float:
        h = width / (n - 1)
        mesh = IonOpticsMesh3D((5, n, n), (0.0, 0.0, 0.0), h, symmetry_planes=planes)
        _, y, z = np.meshgrid(mesh.x, mesh.y, mesh.z, indexing="ij")
        ky, kz = (
            (np.pi / (2 * width), np.pi / width) if planes else (np.pi / width,) * 2
        )
        shape_y = np.cos(ky * y) if planes else np.sin(ky * y)
        dshape_y = -ky * np.sin(ky * y) if planes else ky * np.cos(ky * y)
        current = np.zeros((3, *mesh.shape))
        current[0] = (ky**2 + kz**2) * shape_y * np.sin(kz * z) / mu_0
        b = VectorPotential3D(mesh)(current)
        exact_by = shape_y * kz * np.cos(kz * z)
        exact_bz = -dshape_y * np.sin(kz * z)
        assert np.abs(b[0]).max() < 1e-9 * kz
        # relative to the largest field, kz
        return float(
            max(np.abs(b[1] - exact_by).max(), np.abs(b[2] - exact_bz).max()) / kz
        )

    coarse, fine = error(9), error(17)
    assert fine < 0.02
    assert coarse / fine > 3.0  # second order


def _drift(coordinates: str, *, fraction: float, own: bool) -> float:
    """The edge of a uniform proton beam after drifting 0.1 m: its growth [m]."""
    neumann = FieldBoundaryCondition.neumann(0.0)
    radius = 2e-3
    config = BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(101, 11),
            lower_bound=(0.0, 0.0),
            upper_bound=(0.1, 10e-3),
            coordinates=coordinates,
        ),
        bcs_field=FieldBCs(
            x_lower=neumann,
            x_upper=neumann,
            y_lower=neumann,
            y_upper=FieldBoundaryCondition.dirichlet(0.0),
        ),
        beams=[
            BeamSpecies(
                "H+",
                6e6,
                (0.0, 0.0),
                (0.0, radius),
                energy=ENERGY,
                transverse_temperature=0.0,
                num_particles=100,
            )
        ],  # fmt: skip
        compensation=[SpaceChargeCompensation(0.0, 1.0, fraction)] if fraction else [],
        iterations=6,
        tolerance=1e-6,
        diagnostic_planes=[0.0999],
        symmetry_plane=coordinates == "cartesian",
        self_magnetic_field=own,
    )
    model = BeamOptics(config)
    model.run()
    if own:
        assert model.state.self_magnetic is not None
        assert model.state.history[-1]["max_self_magnetic"] > 0.0
    point = model.state.trace.planes[0].point
    edge = point[1] if coordinates == "cartesian" else np.hypot(point[1], point[2])
    return float(edge.max()) - radius


@pytest.mark.parametrize("coordinates", ["cylindrical", "cartesian"])
@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_the_pinch_scales_the_space_charge_by_one_minus_f_minus_beta_squared(
    coordinates, fraction
) -> None:
    without = _drift(coordinates, fraction=fraction, own=False)
    with_field = _drift(coordinates, fraction=fraction, own=True)
    expected = (1.0 - fraction - BETA2) / (1.0 - fraction)
    assert with_field / without == pytest.approx(expected, rel=0.02)


def test_the_pinch_in_3d() -> None:
    neumann = FieldBoundaryCondition.neumann(0.0)
    zero = FieldBoundaryCondition.dirichlet(0.0)
    radius, h, half = 2e-3, 0.5e-3, 5e-3
    n = int(round(2 * half / h)) + 1

    def rms_growth(own: bool) -> float:
        config = BeamOpticsConfig(
            grid=GridParams(
                ndim=3,
                num_gridpoints=(51, n, n),
                lower_bound=(0.0, -half, -half),
                upper_bound=(0.025, half, half),
                coordinates="cartesian",
            ),
            bcs_field=FieldBCs(
                x_lower=neumann, x_upper=neumann, y_lower=zero, y_upper=zero,
                z_lower=zero, z_upper=zero,
            ),
            beams=[
                BeamSpecies3D(
                    "H+", 1e8, (0.0, 0.0, 0.0), (radius,), shape="disc",
                    energy=ENERGY, transverse_temperature=0.0, num_particles=400,
                )
            ],
            iterations=6,
            tolerance=1e-6,
            diagnostic_planes=[0.0249],
            self_magnetic_field=own,
        )  # fmt: skip
        model = BeamOptics(config)
        model.run()
        point = model.state.trace.planes[0].point
        return float(np.sqrt(2.0 * np.mean(point[1] ** 2 + point[2] ** 2))) - radius

    assert rms_growth(True) / rms_growth(False) == pytest.approx(1.0 - BETA2, rel=0.02)
