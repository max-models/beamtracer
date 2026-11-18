"""The Newton--Krylov solve preconditioned by multigrid (``backend="amg"``, ``"multigrid"``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from beamtracer import BeamOptics
from beamtracer.solver import BoltzmannElectrons, BoltzmannIons, IonOpticsPoisson
from beamtracer.solver.mesh3d import Electrode3D, IonOpticsMesh3D
from beamtracer.solver.refinement import Refinement
from beamtracer.tests.unit.test_beam_optics import _extraction
from beamtracer.tests.unit.test_beam_optics_3d import _diode
from beamtracer.tests.unit.test_newton_jacobian import _mesh

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyamg") is None, reason="needs pyamg"
)


def _amg_available() -> bool:
    from patchwork.solvers.backends import BACKENDS

    return "amg" in BACKENDS


needs_backend = pytest.mark.skipif(
    not _amg_available(), reason="needs patchwork's 'amg' backend"
)


def _multigrid_available() -> bool:
    import inspect

    from patchwork.solvers.multigrid import CompositeMultigrid

    return hasattr(CompositeMultigrid, "update_diagonal") and "smoother" in str(
        inspect.signature(CompositeMultigrid.__init__)
    )


#: The Krylov backends: algebraic multigrid (kept preconditioner) and patchwork's
#: geometric multigrid (a preconditioner of every Newton step's own Jacobian).
BACKENDS = [
    pytest.param("amg", marks=needs_backend),
    pytest.param(
        "multigrid",
        marks=pytest.mark.skipif(
            not _multigrid_available(),
            reason="needs patchwork's Gauss-Seidel multigrid with diagonal updates",
        ),
    ),
]

PLASMAS = [
    BoltzmannElectrons(rho_e=1e-5, electron_temperature=5.0, plasma_potential=50.0),
    BoltzmannIons(rho_p=1e-5, ion_temperature=2.0, plasma_potential=-5.0),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("plasma", PLASMAS)
def test_newton_krylov_reaches_the_direct_solution_and_keeps_its_preconditioner(
    plasma, backend
) -> None:
    mesh = _mesh()
    rng = np.random.default_rng(3)
    rhos = [rng.uniform(0.0, 2e-6, mesh.shape) for _ in range(4)]
    direct = IonOpticsPoisson(mesh, reuse_jacobian=False)
    krylov = IonOpticsPoisson(mesh, backend=backend)
    phi_direct = phi_krylov = None
    builds = []
    for rho in rhos:  # a sequence of solves, as in the Vlasov iteration
        phi_direct = direct.solve(rho, plasma=plasma, initial=phi_direct)
        phi_krylov = krylov.solve(rho, plasma=plasma, initial=phi_krylov)
        builds.append(krylov.factorisations)
        assert krylov.krylov_steps > 0
        np.testing.assert_allclose(phi_krylov, phi_direct, rtol=0, atol=2e-5)
    assert builds[0] >= 1
    if backend == "amg":  # the preconditioner is kept across solves
        assert sum(builds[1:]) < len(builds) - 1
    else:  # every Newton step takes its own Jacobian, by a diagonal update
        assert builds[-1] == krylov.newton_iterations


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_linear_solve_with_multigrid_is_the_direct_one(backend) -> None:
    mesh = _mesh()
    rho = np.random.default_rng(0).uniform(0.0, 1e-6, mesh.shape)
    np.testing.assert_allclose(
        IonOpticsPoisson(mesh, backend=backend).solve(rho),
        IonOpticsPoisson(mesh).solve(rho),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_newton_krylov_with_refined_boxes_takes_the_hanging_nodes_along(
    backend,
) -> None:
    mesh = _mesh()
    boxes = [Refinement(1, (5e-3, 2e-3), (12e-3, 7e-3))]
    plasma = PLASMAS[0]
    direct = IonOpticsPoisson(mesh, reuse_jacobian=False, refinement=boxes)
    krylov = IonOpticsPoisson(mesh, backend=backend, refinement=boxes)
    layout = direct.layout
    assert layout.refined
    rho = np.random.default_rng(1).uniform(0.0, 2e-6, layout.size)
    initial = np.zeros(layout.size)
    np.testing.assert_allclose(
        krylov.solve(rho, plasma=plasma, initial=initial),
        direct.solve(rho, plasma=plasma, initial=initial),
        rtol=0,
        atol=2e-5,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_newton_krylov_in_3d(backend) -> None:
    mesh = IonOpticsMesh3D(
        (21, 11, 11), (0.0, 0.0, 0.0), 5e-4,
        [Electrode3D(lambda x, y, z: (x > 4.1e-3) & (x < 5.1e-3) & (y * y + z * z > 2.1e-3**2), -100.0)],
        symmetry_planes=("y_min", "z_min"),
    )  # fmt: skip
    plasma = BoltzmannElectrons(
        rho_e=1e-5, electron_temperature=5.0, plasma_potential=5.0
    )
    rho = np.full(mesh.shape, 5e-6)
    initial = np.zeros(mesh.shape)
    np.testing.assert_allclose(
        IonOpticsPoisson(mesh, backend=backend).solve(
            rho, plasma=plasma, initial=initial
        ),
        IonOpticsPoisson(mesh, reuse_jacobian=False).solve(
            rho, plasma=plasma, initial=initial
        ),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("three", [False, True])
def test_beam_optics_with_the_multigrid_field_solves(three, backend) -> None:
    config = _diode(three=True) if three else _extraction(1.0)
    plane = 9.8e-3 if three else 7.5e-3
    lu = BeamOptics(config).run().beam_moments(plane)
    amg = BeamOptics(config.replace(field_backend=backend)).run().beam_moments(plane)
    assert amg.current == pytest.approx(lu.current, rel=1e-6)
    assert amg.rms_size == pytest.approx(lu.rms_size, rel=1e-5)
    assert amg.rms_angle == pytest.approx(lu.rms_angle, rel=1e-5)
