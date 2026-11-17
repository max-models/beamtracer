"""Newton's method with a reused (chord) Jacobian against one factorised every step."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer.solver import (
    BoltzmannElectrons,
    BoltzmannIons,
    FaceCondition,
    IonOpticsMesh,
    IonOpticsPoisson,
)


def _mesh() -> IonOpticsMesh:
    faces = {
        "x_min": FaceCondition("dirichlet", 0.0),
        "x_max": FaceCondition("dirichlet", 100.0),
    }
    return IonOpticsMesh(shape=(41, 21), origin=(0.0, 0.0), h=5e-4, faces=faces)


@pytest.mark.parametrize(
    "plasma",
    [
        BoltzmannElectrons(rho_e=1e-5, electron_temperature=5.0, plasma_potential=50.0),
        BoltzmannIons(rho_p=1e-5, ion_temperature=2.0, plasma_potential=-5.0),
    ],
)
def test_the_chord_method_reaches_newtons_solution(plasma) -> None:
    mesh = _mesh()
    rng = np.random.default_rng(3)
    rhos = [rng.uniform(0.0, 2e-6, mesh.shape) for _ in range(4)]
    fresh, reused = IonOpticsPoisson(mesh, reuse_jacobian=False), IonOpticsPoisson(mesh)
    phi_fresh = phi_reused = None
    factorisations = {"fresh": 0, "reused": 0}
    for rho in rhos:  # a sequence of solves, as in the Vlasov iteration
        phi_fresh = fresh.solve(rho, plasma=plasma, initial=phi_fresh)
        phi_reused = reused.solve(rho, plasma=plasma, initial=phi_reused)
        factorisations["fresh"] += fresh.factorisations
        factorisations["reused"] += reused.factorisations
        # Newton stops with updates below 1e-6 of the 100 V span, the chord method below
        # a tenth of that
        np.testing.assert_allclose(phi_reused, phi_fresh, rtol=0, atol=2e-5)
    assert factorisations["reused"] < factorisations["fresh"] / 2


def test_without_a_plasma_nothing_is_factorised_but_the_matrix() -> None:
    poisson = IonOpticsPoisson(_mesh())
    poisson.solve(np.zeros((41, 21)))
    assert poisson.factorisations == 0 and poisson.newton_iterations == 0
