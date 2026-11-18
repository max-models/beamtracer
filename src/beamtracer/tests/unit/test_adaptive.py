"""Adaptive refinement: tags, boxes, the transfer between meshes and the Vlasov iteration."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer import AdaptiveRefinementParams, BeamOptics
from beamtracer.solver import FaceCondition, IonOpticsMesh, IonOpticsPoisson
from beamtracer.solver.adaptive import (
    AdaptiveRefinement,
    adapted_boxes,
    field_variation,
    sample,
)
from beamtracer.solver.mesh3d import IonOpticsMesh3D
from beamtracer.solver.refinement import Refinement
from beamtracer.tests.unit.test_beam_optics import _extraction


def test_the_settings_are_checked_and_round_trip() -> None:
    for wrong in (
        {"max_level": 0},
        {"threshold": 0.0},
        {"every": 0},
        {"regrids": 0},
        {"buffer": -1},
        {"efficiency": 0.0},
    ):
        with pytest.raises(ValueError):
            AdaptiveRefinementParams(**wrong)
    config = _extraction(1.0).replace(
        adaptive_refinement={"max_level": 2, "threshold": 0.03}
    )
    assert config.adaptive_refinement == AdaptiveRefinementParams(2, 0.03)
    again = type(config).from_dict(config.to_dict())
    assert again.adaptive_refinement == config.adaptive_refinement
    assert (
        type(config).from_dict(_extraction(1.0).to_dict()).adaptive_refinement is None
    )


def _step_field(edge: float):
    """A field that jumps at ``x = edge`` (2D or 3D points)."""

    def field(x, *rest):
        strength = np.where(x < edge, 0.0, 1.0e5)
        return (strength, *(np.zeros_like(x) for _ in rest))

    return field


def test_cells_where_the_field_changes_are_tagged_and_boxed_near_the_beam() -> None:
    mesh = IonOpticsMesh((41, 21), (0.0, 0.0), 1e-4)
    field = _step_field(2.05e-3)
    variation = field_variation(field, mesh, 0)
    assert variation.shape == (40, 20)
    assert np.flatnonzero(variation.max(axis=1)).tolist() == [20]
    boxes = adapted_boxes(field, mesh, AdaptiveRefinement(buffer=2, min_size=1))
    assert len(boxes) == 1
    box = boxes[0]
    assert box.level == 1
    assert box.lower[0] == pytest.approx(1.8e-3) and box.upper[0] == pytest.approx(
        2.3e-3
    )
    # a beam only near the axis: the box keeps to it
    density = np.zeros(mesh.shape)
    density[:, :4] = 1.0
    (near,) = adapted_boxes(
        field, mesh, AdaptiveRefinement(buffer=2, min_size=1), density=density
    )
    assert near.upper[1] == pytest.approx(0.6e-3)  # the beam's cells and the buffer
    # two levels, the second inside the first; nothing where the field is uniform
    two = adapted_boxes(field, mesh, AdaptiveRefinement(max_level=2, min_size=1))
    assert [b.level for b in two] == [1, 2]
    assert two[0].lower[0] <= two[1].lower[0] and two[1].upper[0] <= two[0].upper[0]
    assert adapted_boxes(lambda x, y: (0 * x, 0 * y), mesh, AdaptiveRefinement()) == []


def test_3d_tags() -> None:
    mesh = IonOpticsMesh3D((21, 9, 9), (0.0, 0.0, 0.0), 1e-4)
    boxes = adapted_boxes(_step_field(1.05e-3), mesh, AdaptiveRefinement(min_size=1))
    assert [b.level for b in boxes] == [1]
    assert len(boxes[0].lower) == 3


@pytest.mark.parametrize("three", [False, True])
def test_a_linear_potential_moves_exactly_between_composite_meshes(three) -> None:
    if three:
        faces = {"x_min": FaceCondition("dirichlet", 0.0)}
        mesh = IonOpticsMesh3D((17, 9, 9), (0.0, 0.0, 0.0), 1e-4, faces=faces)
        old = [Refinement(1, (0.0, 0.0, 0.0), (8e-4, 4e-4, 4e-4))]
        new = [Refinement(1, (4e-4, 2e-4, 0.0), (1.2e-3, 8e-4, 8e-4))]
    else:
        faces = {"x_min": FaceCondition("dirichlet", 0.0)}
        mesh = IonOpticsMesh((17, 9), (0.0, 0.0), 1e-4, faces=faces)
        old = [Refinement(1, (0.0, 0.0), (8e-4, 4e-4))]
        new = [Refinement(1, (4e-4, 2e-4), (1.2e-3, 8e-4))]
    before = IonOpticsPoisson(mesh, refinement=old)
    after = IonOpticsPoisson(mesh, refinement=new)
    weights = np.array([3.0, -2.0, 1.0][: before.coordinates.shape[0]])
    values = weights @ before.coordinates
    moved = sample(before.layout, values, after.coordinates)
    np.testing.assert_allclose(moved, weights @ after.coordinates, rtol=0, atol=1e-15)


def test_adaptive_boxes_bring_the_beam_close_to_that_of_a_fine_mesh() -> None:
    """The extraction of the mesh refinement example: one adaptive level.

    The beam's divergence varies by about 0.1 mrad between nearby runs (a trajectory
    near an electrode's edge), so the box only has to take the coarse mesh well towards
    the fine one.
    """
    coarse = _extraction(1.0).replace(iterations=60, tolerance=1e-5)
    grid = coarse.grid
    fine = coarse.replace(
        grid=type(grid)(
            num_gridpoints=(321, 121), lower_bound=grid.lower_bound,
            upper_bound=grid.upper_bound, coordinates=grid.coordinates,
        )
    )  # fmt: skip

    def divergence(model: BeamOptics) -> float:
        return model.beam_moments(7.5e-3).rms_angle

    reference = divergence(BeamOptics(fine).run())
    plain = divergence(BeamOptics(coarse).run())
    model = BeamOptics(
        coarse.replace(adaptive_refinement=AdaptiveRefinementParams())
    ).run()
    layout = model.state.layout
    assert layout.refined
    assert layout.size < 0.2 * fine.grid.num_gridpoints[0] * fine.grid.num_gridpoints[1]
    assert {h["boxes"] for h in model.convergence_history} >= {
        0,
        len(layout.patches) - 1,
    }
    assert abs(divergence(model) - reference) < 0.6 * abs(plain - reference)
