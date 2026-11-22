"""A graded beam axis: the mesh, the field, the Poisson solve and the tracers on it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.constants import epsilon_0

from beamtracer import BeamOptics, GridParams
from beamtracer.solver import (
    BeamTracer,
    Electrode,
    FaceCondition,
    IonOpticsMesh,
    IonOpticsPoisson,
    StaggeredElectricField,
)
from beamtracer.solver.refinement import Refinement, patches_of
from beamtracer.tests.unit.test_compiled_tracer import _assert_same, _planar, _run

ZERO = FaceCondition("dirichlet", 0.0)


def _nodes(n: int, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """``n`` nodes of a smooth map: cells from 0.7 to 1.3 of the mean."""
    s = np.linspace(0.0, 1.0, n)
    return lower + (upper - lower) * (s + 0.3 * np.sin(2.0 * np.pi * s) / (2.0 * np.pi))


def _orders(errors) -> np.ndarray:
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


def test_the_mesh_maps_positions_to_node_indices() -> None:
    nodes = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    mesh = IonOpticsMesh((5, 3), (0.0, 0.0), 0.5, "planar", x_nodes=nodes)
    assert mesh.graded and mesh.upper == (1.0, 1.0)
    np.testing.assert_array_equal(mesh.x, nodes)
    np.testing.assert_allclose(mesh.dx, [0.1, 0.2, 0.3, 0.4])
    x = np.array([0.0, 0.05, 0.45, 1.0])
    np.testing.assert_allclose(mesh.index_x(x), [0.0, 0.5, 2.5, 4.0])
    np.testing.assert_allclose(mesh.position_x(mesh.index_x(x)), x)
    np.testing.assert_allclose(mesh.logical_x(x), 0.5 * mesh.index_x(x))
    assert mesh.logical_upper == (2.0, 1.0)
    np.testing.assert_allclose(mesh.spacing_at(np.array([0.05, 0.9])), [0.1, 0.4])
    uniform = IonOpticsMesh((5, 3), (0.0, 0.0), 0.25, "planar")
    assert not uniform.graded and uniform.spacing_at(0.3) == 0.25
    with pytest.raises(ValueError, match="increase"):
        IonOpticsMesh((3, 3), (0.0, 0.0), 0.5, "planar", x_nodes=[0.0, 0.2, 0.1])
    with pytest.raises(ValueError, match="origin"):
        IonOpticsMesh((3, 3), (0.1, 0.0), 0.5, "planar", x_nodes=[0.0, 0.2, 0.3])
    with pytest.raises(ValueError, match="3 positions"):
        IonOpticsMesh((3, 3), (0.0, 0.0), 0.5, "planar", x_nodes=[0.0, 1.0])


def test_surface_distances_are_fractions_of_their_edge() -> None:
    """A plate at x > 0.25: the node at 0.1 sees it 0.15 away, 0.75 of its edge of 0.2."""
    plate = Electrode(lambda x, y: x > 0.25, 1.0)
    nodes = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    mesh = IonOpticsMesh((5, 3), (0.0, 0.0), 0.5, "planar", [plate], x_nodes=nodes)
    assert mesh.near_solid[1].all()
    np.testing.assert_allclose(mesh.distance[1, 1], 0.75, atol=1 / 255)


def test_the_field_and_the_solve_converge_at_second_order() -> None:
    """The cylindrical manufactured potential of `IonOpticsPoisson`'s card, graded along x."""
    solve, field = [], []
    rng = np.random.default_rng(0)
    px, pr = rng.uniform(-0.9, 0.9, 300), rng.uniform(0.05, 0.9, 300)
    for n in (8, 16, 32):
        mesh = IonOpticsMesh(
            (2 * n + 1, n + 1),
            (-1.0, 0.0),
            1.0 / n,
            "cylindrical",
            faces={"x_min": ZERO, "x_max": ZERO, "y_max": ZERO},
            x_nodes=_nodes(2 * n + 1),
        )
        x, r = mesh.nodes()
        cx, cr = np.cos(0.5 * np.pi * x), np.cos(0.5 * np.pi * r)
        laplacian = -0.5 * np.pi**2 * cx * cr - 0.25 * np.pi**2 * cx * np.sinc(0.5 * r)
        phi = IonOpticsPoisson(mesh).solve(-epsilon_0 * laplacian)
        solve.append(np.abs(phi - cx * cr).max())
        ex, er = StaggeredElectricField(mesh, cx * cr)(px, pr)
        exact_x = 0.5 * np.pi * np.sin(0.5 * np.pi * px) * np.cos(0.5 * np.pi * pr)
        exact_r = 0.5 * np.pi * np.cos(0.5 * np.pi * px) * np.sin(0.5 * np.pi * pr)
        field.append(max(np.abs(ex - exact_x).max(), np.abs(er - exact_r).max()))
    assert np.all(_orders(solve) > 1.9), solve
    assert np.all(_orders(field) > 1.8), field


def test_refined_boxes_continue_the_grading() -> None:
    nodes = _nodes(21, 0.0, 2.0)
    mesh = IonOpticsMesh((21, 11), (0.0, 0.0), 0.1, "planar", x_nodes=nodes)
    base, patch = patches_of(mesh, [Refinement(1, (0.5, 0.2), (1.2, 0.6))])
    # every other node of the box is a base node, the others halve the base edges
    first = patch.box[0][0] // 2
    np.testing.assert_allclose(
        patch.mesh.x[::2], nodes[first : first + patch.mesh.x[::2].size]
    )
    np.testing.assert_allclose(
        patch.mesh.x[1::2], 0.5 * (patch.mesh.x[:-1:2] + patch.mesh.x[2::2])
    )
    assert patch.mesh.logical_x0 == pytest.approx(patch.box[0][0] * 0.05)
    x = np.linspace(patch.mesh.x[0], patch.mesh.x[-1], 7)
    np.testing.assert_allclose(patch.mesh.logical_x(x), mesh.logical_x(x))
    assert patch.field_mesh.x[2] == patch.mesh.x[0]


def _graded_planar(**settings):
    """The planar extraction of the compiled tracer's tests, graded along the beam axis."""
    config = _planar(**settings)
    grid = config.grid
    config.grid = GridParams(
        num_gridpoints=grid.num_gridpoints,
        upper_bound=grid.upper_bound,
        x_spacing=[(0.0, 0.5e-3), (0.04, 1.0e-3), (0.12, 2.0e-3)],
    )
    return dataclasses.replace(config)


def test_the_grid_takes_a_spacing_along_x() -> None:
    config = _graded_planar()
    grid = config.grid
    assert grid.graded and grid.num_gridpoints[1] == 51
    sizes = np.diff(grid.x_nodes)
    assert sizes[0] == pytest.approx(0.5e-3, rel=0.05)
    assert sizes[-1] == pytest.approx(2.0e-3, rel=0.05)
    assert GridParams.from_dict(grid.to_dict()).x_spacing == grid.x_spacing
    with pytest.raises(ValueError, match="2D"):
        GridParams(ndim=3, x_spacing=[(0.0, 1e-3)])


@pytest.mark.skipif(
    not BeamTracer.compiled_available(), reason="Pyccel is not available"
)
def test_the_compiled_tracer_agrees_on_a_graded_mesh() -> None:
    config = _graded_planar()
    _assert_same(_run(config, False), _run(config, True))


@pytest.mark.skipif(
    not BeamTracer.compiled_available(), reason="Pyccel is not available"
)
def test_the_compiled_tracer_agrees_with_refined_boxes_on_a_graded_mesh() -> None:
    from beamtracer import RefinementParams

    config = _graded_planar(
        refinement=[RefinementParams(1, (0.015, 0.02), (0.045, 0.045))]
    )
    _assert_same(_run(config, False), _run(config, True))


def test_a_graded_mesh_gives_the_beam_of_a_fine_uniform_one() -> None:
    """Fewer nodes graded toward the emitter: the same transmitted current and divergence."""
    fine = _planar()
    fine.grid = GridParams(num_gridpoints=(241, 101), upper_bound=(0.12, 0.05))
    graded = _graded_planar()
    results = [BeamOptics(config).run() for config in (fine, graded)]
    planes = [result.state.trace.planes[1] for result in results]
    currents = [np.abs(plane.current[plane.direction > 0]).sum() for plane in planes]
    assert currents[1] == pytest.approx(currents[0], rel=0.02)
    angles = [np.std(plane.velocity[1] / plane.velocity[0]) for plane in planes]
    assert angles[1] == pytest.approx(angles[0], rel=0.1)
