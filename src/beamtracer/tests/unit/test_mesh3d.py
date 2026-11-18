"""Tests for the 3D ion-optics mesh, its field solve and its electric field."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer.solver.fields3d import StaggeredElectricField3D, _staggered_problem
from beamtracer.solver.mesh import Electrode, FaceCondition, IonOpticsMesh
from beamtracer.solver.mesh3d import Electrode3D, IonOpticsMesh3D
from beamtracer.solver.poisson import BoltzmannElectrons, IonOpticsPoisson, Layout3D

INNER, OUTER = 0.2, 0.8


def _radius(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def _spheres(n: int) -> IonOpticsMesh3D:
    """Concentric spheres: the inner at 1 V, the outer grounded."""
    return IonOpticsMesh3D(
        (n + 1,) * 3,
        (-1.0, -1.0, -1.0),
        2.0 / n,
        [
            Electrode3D(lambda x, y, z: _radius(x, y, z) < INNER, 1.0, "inner"),
            Electrode3D(lambda x, y, z: _radius(x, y, z) > OUTER, 0.0, "outer"),
        ],
    )


def test_the_mesh_classifies_nodes_and_brackets_the_surface() -> None:
    mesh = _spheres(16)
    x, y, z = mesh.nodes()
    r = _radius(x, y, z)
    np.testing.assert_array_equal(mesh.solid == 0, r < INNER)
    np.testing.assert_array_equal(mesh.fixed, (r < INNER) | (r > OUTER))
    # the distance from a near node to the inner sphere along -x, against the exact one
    i, j, k = np.nonzero(mesh.near_solid & (mesh.solid < 0) & (r < 0.5))
    toward = mesh.solid[i - 1, j, k] == 0  # its -x neighbour is in the inner sphere
    i, j, k = i[toward], j[toward], k[toward]
    xs, ys, zs = x[i, j, k], y[i, j, k], z[i, j, k]
    exact = (xs - np.sqrt(INNER**2 - ys**2 - zs**2)) / mesh.h
    assert i.size > 0
    np.testing.assert_allclose(mesh.distance[0, i, j, k], exact, atol=1.0 / 255 + 1e-9)


def test_the_mesh_checks_its_settings() -> None:
    with pytest.raises(ValueError, match="3 axes"):
        IonOpticsMesh3D((5, 5), (0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="unknown faces"):
        IonOpticsMesh3D((5, 5, 5), (0, 0, 0), 1.0, faces={"w_min": FaceCondition()})
    with pytest.raises(ValueError, match="symmetry plane is one of"):
        IonOpticsMesh3D((5, 5, 5), (0, 0, 0), 1.0, symmetry_planes=("x_min",))
    with pytest.raises(ValueError, match="zero normal derivative"):
        IonOpticsMesh3D(
            (5, 5, 5), (0, 0, 0), 1.0,
            faces={"y_min": FaceCondition("dirichlet", 0.0)}, symmetry_planes=("y_min",),
        )  # fmt: skip


def test_the_potential_between_spheres_converges() -> None:
    errors = []
    for n in (16, 32):
        mesh = _spheres(n)
        phi = IonOpticsPoisson(mesh).solve()
        r = _radius(*mesh.nodes())
        between = (r > INNER) & (r < OUTER)
        exact = (1.0 / r[between] - 1.0 / OUTER) / (1.0 / INNER - 1.0 / OUTER)
        errors.append(np.max(np.abs(phi[between] - exact)))
    assert errors[0] < 0.012
    assert errors[1] < 0.6 * errors[0]


def _extruded(electrodes: bool) -> tuple[np.ndarray, np.ndarray]:
    """A planar case with a Boltzmann plasma, solved in 2D and extruded along z in 3D."""
    zero, high = FaceCondition("dirichlet", 0.0), FaceCondition("dirichlet", -500.0)
    faces = {"x_min": zero, "x_max": high}
    h, (nx, ny, nz) = 0.25e-3, (41, 21, 5)

    def plate(x, y):
        return (x > 3.03e-3) & (x < 4.03e-3) & (y > 2.13e-3)

    flat = [Electrode(plate, -300.0)] if electrodes else []
    deep = [Electrode3D(lambda x, y, z: plate(x, y), -300.0)] if electrodes else []
    mesh2 = IonOpticsMesh((nx, ny), (0.0, 0.0), h, "planar", flat, faces=faces)
    mesh3 = IonOpticsMesh3D((nx, ny, nz), (0.0, 0.0, 0.0), h, deep, faces=faces)
    plasma = BoltzmannElectrons(1e-4, 5.0, 0.0)
    rho = np.random.default_rng(1).uniform(0.0, 4e-5, (nx, ny))
    phi2 = IonOpticsPoisson(mesh2).solve(rho, plasma=plasma, initial=np.zeros((nx, ny)))
    phi3 = IonOpticsPoisson(mesh3).solve(
        np.repeat(rho[:, :, None], nz, axis=2),
        plasma=plasma,
        initial=np.zeros((nx, ny, nz)),
    )
    return phi2, phi3


def test_an_extruded_plasma_case_solves_as_in_2d() -> None:
    phi2, phi3 = _extruded(electrodes=False)
    np.testing.assert_allclose(phi3, np.repeat(phi2[:, :, None], 5, axis=2), atol=1e-8)


def test_an_extruded_electrode_differs_only_at_the_z_faces_cut_cells() -> None:
    # patchwork takes the cut edge's two-point flux at nodes on the domain's faces and
    # Shortley-Weller rows inside, so the z faces' cut nodes differ from the 2D ones
    phi2, phi3 = _extruded(electrodes=True)
    assert np.max(np.abs(phi3 - phi2[:, :, None])) < 2e-3 * 500.0


def test_the_layout_maps_patchwork_nodes_to_the_mesh() -> None:
    mesh = _spheres(8)
    poisson = IonOpticsPoisson(mesh)
    layout = poisson.layout
    assert isinstance(layout, Layout3D)
    values = np.random.default_rng(0).normal(size=mesh.shape)
    np.testing.assert_array_equal(layout.on_base(layout.from_base(values)), values)
    np.testing.assert_allclose(layout.x, layout.from_base(mesh.nodes()[0]))
    field = layout.field(layout.from_base(values))
    assert isinstance(field, StaggeredElectricField3D)
    with pytest.raises(ValueError, match="refinement"):
        IonOpticsPoisson(mesh, refinement=[object()]).solve()  # type: ignore[list-item]


def test_the_field_converges_and_vanishes_across_a_symmetry_plane() -> None:
    problem = _staggered_problem()
    errors = [problem.error(step) for step in problem.steps]
    assert errors[-1] < errors[0] / 10
    h = 0.1
    mesh = IonOpticsMesh3D(
        (11, 11, 11), (0, 0, 0), h, symmetry_planes=("y_min", "z_min")
    )
    x, y, z = mesh.nodes()
    field = StaggeredElectricField3D(mesh, np.cos(y) * np.cos(z) + x)
    _, ey, ez = field(np.full(5, 0.5), np.zeros(5), np.linspace(0.1, 0.9, 5))
    np.testing.assert_allclose(ey, 0.0, atol=1e-12)
    _, _, ez = field(np.full(5, 0.5), np.linspace(0.1, 0.9, 5), np.zeros(5))
    np.testing.assert_allclose(ez, 0.0, atol=1e-12)
