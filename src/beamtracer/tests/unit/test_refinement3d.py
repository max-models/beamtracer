"""Refined boxes on a 3D grid: the patches, the composite solve and the beam through them."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0

from beamtracer import BeamOptics, RefinementParams
from beamtracer.solver import FaceCondition, IonOpticsPoisson
from beamtracer.solver.mesh3d import IonOpticsMesh3D
from beamtracer.solver.refinement import Refinement
from beamtracer.solver.refinement3d import PatchedElectricField3D, patches_of_3d
from beamtracer.tests.unit.test_beam_optics_3d import _diode

MM = 1e-3


def _plates(n: int = 11) -> IonOpticsMesh3D:
    """Grounded plates at x = 0 and x = 10 mm, Neumann on the other faces."""
    faces = {
        "x_min": FaceCondition("dirichlet", 0.0),
        "x_max": FaceCondition("dirichlet", 0.0),
    }
    m = (n + 1) // 2
    return IonOpticsMesh3D((n, m, m), (0.0, 0.0, 0.0), 10 * MM / (n - 1), faces=faces)


def test_the_setting_takes_3d_corners_and_round_trips() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        RefinementParams(1, (0, 0, 2e-3), (1e-3, 1e-3, 1e-3))
    config = _diode(three=True).replace(
        refinement=[
            RefinementParams(1, ("3 mm", "0 mm", "0 mm"), ("7 mm", "2 mm", "1 mm"))
        ]
    )
    again = type(config).from_dict(config.to_dict())
    assert again.refinement == config.refinement
    assert again.refinement[0].upper == pytest.approx((7e-3, 2e-3, 1e-3))


def test_boxes_snap_to_the_level_below_and_their_fields_get_a_margin() -> None:
    mesh = _plates()
    patches = patches_of_3d(
        mesh, [Refinement(1, (2.6e-3, 0.0, 0.0), (7.2e-3, 3.1e-3, 2e-3))]
    )
    one = patches[1]
    assert one.box == (
        (4, 16),
        (0, 8),
        (0, 4),
    )  # level-1 cells: base nodes 2..8, 0..4, 0..2
    assert one.mesh.origin == pytest.approx((2e-3, 0.0, 0.0))
    assert one.mesh.h == pytest.approx(0.5e-3)
    # two cells of margin, none below the faces y = 0 and z = 0
    assert one.field_mesh is not None
    assert one.field_mesh.origin == pytest.approx((1e-3, 0.0, 0.0))
    assert one.field_mesh.shape == tuple(
        s + d for s, d in zip(one.mesh.shape, (4, 2, 2), strict=True)
    )
    with pytest.raises(ValueError, match="outside"):
        patches_of_3d(mesh, [Refinement(1, (0.03, 0.0, 0.0), (0.04, 0.01, 0.01))])
    with pytest.raises(ValueError, match="corners"):
        patches_of_3d(mesh, [Refinement(1, (0.0, 0.0), (1e-3, 1e-3))])


def test_the_volumes_fill_the_domain() -> None:
    mesh = _plates()
    plain = IonOpticsPoisson(mesh).layout
    refined = IonOpticsPoisson(
        mesh, refinement=[Refinement(1, (3e-3, 0.0, 0.0), (7e-3, 3e-3, 3e-3))]
    ).layout
    assert refined.size > plain.size
    assert plain.volumes.sum() == pytest.approx(10e-3 * 5e-3 * 5e-3)
    assert refined.volumes.sum() == pytest.approx(10e-3 * 5e-3 * 5e-3)
    values = np.arange(float(np.prod(mesh.shape))).reshape(mesh.shape)
    np.testing.assert_array_equal(plain.on_base(plain.from_base(values)), values)
    with pytest.raises(ValueError, match="composite"):
        refined.from_base(np.zeros(mesh.shape))


def test_a_uniform_charge_between_plates_on_refined_boxes() -> None:
    """-phi'' = rho / eps0 with phi = 0 at x = 0, L: a parabola, also on the refined boxes."""
    rho = 1e-6
    errors = []
    for n in (11, 21):
        solver = IonOpticsPoisson(
            _plates(n),
            refinement=[
                Refinement(1, (2e-3, 0.0, 0.0), (8e-3, 3e-3, 3e-3)),
                Refinement(2, (4e-3, 0.0, 0.0), (6e-3, 2e-3, 2e-3)),
            ],
        )
        layout = solver.layout
        assert len(layout.patches) == 3
        phi = solver.solve(np.full(layout.size, rho))
        exact = rho / (2 * epsilon_0) * layout.x * (10e-3 - layout.x)
        errors.append(np.abs(phi - exact).max() / exact.max())
    assert errors[0] < 2e-2
    assert errors[1] < 0.6 * errors[0]


def test_the_patched_field_of_a_linear_potential_is_uniform() -> None:
    """The ghost nodes of a refined field take the coarse potential: exact for a linear one."""
    mesh = _plates()
    layout = IonOpticsPoisson(
        mesh, refinement=[Refinement(1, (3e-3, 0.0, 0.0), (7e-3, 3e-3, 3e-3))]
    ).layout
    field = layout.field(-2e3 * layout.x)
    assert isinstance(field, PatchedElectricField3D)
    rng = np.random.default_rng(0)
    x, y, z = (rng.uniform(0.2e-3, upper, 400) for upper in (9.8e-3, 4.8e-3, 4.8e-3))
    assert (field.patch_of(x, y, z) == 1).any()
    ex, ey, ez = field(x, y, z)
    np.testing.assert_allclose(ex, 2e3, rtol=1e-9)
    np.testing.assert_allclose(ey, 0.0, atol=1e-6)
    np.testing.assert_allclose(ez, 0.0, atol=1e-6)


def test_refined_boxes_bring_the_beam_to_that_of_the_fine_mesh() -> None:
    """The slit diode: a box over the domain is the fine mesh, a smaller one goes part way."""

    def moments(config) -> np.ndarray:
        model = BeamOptics(config.replace(iterations=6)).run()
        if config.refinement:
            assert model.state.layout.refined
        m = model.beam_moments(9.8e-3)
        return np.array([m.current, m.rms_size, m.rms_angle])

    coarse = _diode(three=True)
    grid = coarse.grid
    fine = coarse.replace(
        grid=type(grid)(
            ndim=3, num_gridpoints=(81, 33, 9),
            lower_bound=grid.lower_bound, upper_bound=grid.upper_bound,
        )
    )  # fmt: skip
    reference, rough = moments(fine), moments(coarse)
    everywhere = RefinementParams(1, (0.0, 0.0, 0.0), (10e-3, 4e-3, 1e-3))
    np.testing.assert_allclose(
        moments(coarse.replace(refinement=[everywhere])), reference, rtol=1e-9
    )
    part = moments(
        coarse.replace(
            refinement=[RefinementParams(1, (3.5e-3, 0, 0), (10e-3, 4e-3, 1e-3))]
        )
    )
    assert np.all(np.abs(part - reference)[1:] < np.abs(rough - reference)[1:])
