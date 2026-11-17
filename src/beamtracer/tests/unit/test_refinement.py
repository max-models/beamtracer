"""Refined boxes: the patches, the composite solve and the trace on the finest box."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0

from beamtracer import BeamOptics, RefinementParams
from beamtracer.solver import FaceCondition, IonOpticsMesh, IonOpticsPoisson
from beamtracer.solver.refinement import Refinement, patches_of
from beamtracer.tests.unit.test_beam_optics import _extraction
from beamtracer.tests.unit.test_compiled_tracer import _planar

MM = 1e-3


def test_the_setting_is_checked_and_round_trips() -> None:
    with pytest.raises(ValueError, match="level"):
        RefinementParams(0, (0, 0), (1e-3, 1e-3))
    with pytest.raises(ValueError, match="lower < upper"):
        RefinementParams(1, (2e-3, 0), (1e-3, 1e-3))
    config = _extraction(1.0).replace(
        refinement=[RefinementParams(1, ("0.5 mm", "0 mm"), ("3 mm", "1.5 mm"))]
    )
    again = type(config).from_dict(config.to_dict())
    assert again.refinement == config.refinement
    assert again.refinement[0].upper == pytest.approx((3e-3, 1.5e-3))


def _plates(geometry: str = "planar") -> IonOpticsMesh:
    faces = {
        "x_min": FaceCondition("dirichlet", 0.0),
        "x_max": FaceCondition("dirichlet", 0.0),
    }
    return IonOpticsMesh((21, 11), (0.0, 0.0), 1e-3, geometry, faces=faces)


def test_boxes_snap_to_the_level_below_and_their_fields_get_a_margin() -> None:
    mesh = _plates()
    patches = patches_of(
        mesh,
        [
            Refinement(1, (2.6e-3, 0.0), (7.2e-3, 4.1e-3)),
            Refinement(2, (3e-3, 0), (5e-3, 2e-3)),
        ],
    )
    one, two = patches[1], patches[2]
    assert one.box == ((4, 16), (0, 10))  # level-1 cells: base nodes 2..8, 0..5
    assert one.mesh.origin == pytest.approx((2e-3, 0.0))
    assert one.mesh.h == pytest.approx(0.5e-3)
    # two cells of margin, none below the face y = 0
    assert one.field_mesh.origin == pytest.approx((1e-3, 0.0))
    assert one.field_mesh.shape == (one.mesh.shape[0] + 4, one.mesh.shape[1] + 2)
    assert two.box == ((12, 20), (0, 8)) and two.mesh.h == pytest.approx(0.25e-3)
    with pytest.raises(ValueError, match="outside"):
        patches_of(mesh, [Refinement(1, (0.03, 0.0), (0.04, 0.01))])


@pytest.mark.parametrize("geometry", ["planar", "cylindrical"])
def test_the_volumes_fill_the_domain(geometry) -> None:
    mesh = _plates(geometry)
    plain = IonOpticsPoisson(mesh).layout
    refined = IonOpticsPoisson(
        mesh, refinement=[Refinement(1, (5e-3, 0.0), (12e-3, 6e-3))]
    ).layout
    assert refined.size > plain.size
    total = 20e-3 * 10e-3 if geometry == "planar" else np.pi * 10e-3**2 * 20e-3
    assert plain.volumes.sum() == pytest.approx(total)
    assert refined.volumes.sum() == pytest.approx(total)
    np.testing.assert_array_equal(plain.from_base(np.arange(231.0).reshape(21, 11))[plain.base.index], np.arange(231.0).reshape(21, 11))  # fmt: skip
    with pytest.raises(ValueError, match="composite"):
        refined.from_base(np.zeros(mesh.shape))


def test_a_uniform_charge_between_plates_on_refined_boxes() -> None:
    """-phi'' = rho / eps0 with phi = 0 at x = 0, L: a parabola.

    Exact for the stencil on a uniform mesh; on refined boxes patchwork's composite scheme
    has a first-order truncation error on the coarse–fine nodes (largest at a box's inner
    corner), so the error is small and halves with the cells.
    """
    rho = 1e-6
    errors = []
    for n in (21, 41):
        faces = {
            "x_min": FaceCondition("dirichlet", 0.0),
            "x_max": FaceCondition("dirichlet", 0.0),
        }
        mesh = IonOpticsMesh(
            (n, (n + 1) // 2), (0.0, 0.0), 20e-3 / (n - 1), faces=faces
        )
        solver = IonOpticsPoisson(
            mesh,
            refinement=[
                Refinement(1, (4e-3, 0.0), (14e-3, 6e-3)),
                Refinement(2, (6e-3, 0.0), (10e-3, 4e-3)),
            ],
        )
        layout = solver.layout
        phi = solver.solve(np.full(layout.size, rho))
        exact = rho / (2 * epsilon_0) * layout.x * (20e-3 - layout.x)
        errors.append(np.abs(phi - exact).max() / exact.max())
    assert errors[0] < 1e-2
    assert errors[1] < 0.6 * errors[0]


def _refined(config, boxes):
    return config.replace(refinement=[RefinementParams(*box) for box in boxes])


BOXES = {
    "planar": [(1, (0.0, 0.0), (0.04, 0.02))],
    "cylindrical": [
        (1, (0.0, 0.0), (2.0 * MM, 1.5 * MM)),
        (2, (0.0, 0.0), (1.5 * MM, 1.2 * MM)),
    ],
}


@pytest.mark.parametrize("case", ["planar", "cylindrical"])
def test_the_compiled_tracer_traces_refined_boxes_as_numpy_does(case) -> None:
    config = _planar() if case == "planar" else _extraction(1.0).replace(iterations=4)
    config = _refined(config, BOXES[case])
    models = []
    for compiled in (True, False):
        model = BeamOptics(config)
        model.tracer.compiled = compiled
        models.append(model.run())
    a, b = (m.state for m in models)
    np.testing.assert_array_equal(a.trace.status, b.trace.status)
    scale = np.abs(b.composite_charge_density).max()
    np.testing.assert_allclose(
        a.composite_charge_density,
        b.composite_charge_density,
        rtol=0,
        atol=1e-9 * scale,
    )
    np.testing.assert_allclose(
        a.composite_potential, b.composite_potential, rtol=0, atol=1e-6
    )
    assert a.layout.refined and a.composite_potential.size > a.potential.size


def test_a_refined_box_brings_the_beam_to_that_of_the_fine_mesh() -> None:
    """Extraction through an aperture: the box around it does what halving the cells does."""

    def divergence(config) -> float:
        model = BeamOptics(config.replace(iterations=60, tolerance=1e-5)).run()
        return model.beam_moments(7.5e-3).rms_angle

    coarse = _extraction(1.0)
    grid = coarse.grid
    fine = coarse.replace(
        grid=type(grid)(
            num_gridpoints=(161, 61), lower_bound=grid.lower_bound,
            upper_bound=grid.upper_bound, coordinates=grid.coordinates,
        )
    )  # fmt: skip
    box = _refined(coarse, [(1, (0.0, 0.0), (4.5 * MM, 2.0 * MM))])
    reference = divergence(fine)
    assert abs(divergence(box) - reference) < 0.5 * abs(divergence(coarse) - reference)
