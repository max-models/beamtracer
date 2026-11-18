"""Tests for a planar symmetry plane: half of a mirrored geometry."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ObjectParams,
)
from beamtracer.solver import IonOpticsMesh

H, LENGTH, HALF = 0.5e-3, 40e-3, 10e-3
PLANE = LENGTH - 0.25e-3


def _plate(
    x0: float, x1: float, y0: float, y1: float, potential: float
) -> ObjectParams:
    """A plate ``[x0, x1] x [y0, y1]``.

    Its edges lie between nodes, off the ``h / 255`` points where the wall distances are
    bracketed: a surface exactly on one is found on either side by rounding, which would
    break the mirror symmetry the tests compare.
    """
    return ObjectParams(
        type="polygon",
        vertices=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        bc_type="dirichlet",
        bc_value=potential,
    )


def _lens(full: bool, **settings) -> BeamOpticsConfig:
    """A planar einzel lens focusing a proton beam across ``y = 0``, with space charge.

    ``full`` is the whole geometry, mirrored at ``y = 0``; else its upper half with a
    symmetry plane.
    """
    ny = int(round(HALF / H))
    plates = {}
    for name, (x0, x1), gap, potential in (
        ("entrance", (-1e-3, 10.23e-3), 4.13e-3, 0.0),
        ("lens", (15.23e-3, 24.77e-3), 2.17e-3, -12_000.0),
        ("exit", (29.77e-3, 41e-3), 4.13e-3, 0.0),
    ):
        plates[f"{name}_top"] = _plate(x0, x1, gap, HALF + 1e-3, potential)
        if full:
            plates[f"{name}_bottom"] = _plate(x0, x1, -HALF - 1e-3, -gap, potential)
    neumann = FieldBoundaryCondition.neumann(0.0)
    num = 400
    return BeamOpticsConfig(
        grid=GridParams(
            num_gridpoints=(int(round(LENGTH / H)) + 1, (2 * ny if full else ny) + 1),
            lower_bound=(0.0, -HALF if full else 0.0),
            upper_bound=(LENGTH, HALF),
        ),
        objects=plates,
        bcs_field=FieldBCs(
            x_lower=neumann, x_upper=neumann, y_lower=neumann, y_upper=neumann
        ),
        beams=[
            BeamSpecies(
                "H+",
                20.0,
                (0.0, -3e-3 if full else 0.0),
                (0.0, 3e-3),
                energy=5000.0,
                transverse_temperature=0.0,
                num_particles=2 * num if full else num,
            )
        ],  # fmt: skip
        iterations=3,
        tolerance=0.0,
        diagnostic_planes=[PLANE],
        symmetry_plane=not full,
        **settings,
    )


def _run(config: BeamOpticsConfig, compiled: bool | None) -> BeamOptics:
    model = BeamOptics(config)
    model.tracer.compiled = compiled
    model.run()
    return model


@pytest.mark.parametrize("compiled", [False, None])
def test_half_with_a_symmetry_plane_is_the_mirrored_whole(compiled) -> None:
    full = _run(_lens(full=True), compiled)
    half = _run(_lens(full=False), compiled)
    hits = half.plane(PLANE)
    assert (hits.point[1] < 0.0).any()  # the lens focuses the beam across the plane
    ny = half.potential.shape[1]
    np.testing.assert_allclose(
        half.potential, full.potential[:, ny - 1 :], rtol=0.0, atol=1e-6 * 12_000.0
    )
    # the half carries half of the whole's current, with the same moments
    whole = full.plane(PLANE)
    assert hits.current.sum() == pytest.approx(0.5 * whole.current.sum(), rel=1e-9)

    def rms(plane) -> float:
        return float(np.sqrt(np.average(plane.point[1] ** 2, weights=plane.current)))

    assert rms(hits) == pytest.approx(rms(whole), rel=1e-6)


def test_without_the_symmetry_plane_the_beam_leaves_at_the_plane() -> None:
    half = _run(_lens(full=False).replace(symmetry_plane=False), None)
    assert not (half.plane(PLANE).point[1] < 0.0).any()
    assert half.state.trace.status.tolist().count(2) > 0  # through the y_min face


def test_a_symmetry_plane_needs_a_planar_mesh_with_a_neumann_face() -> None:
    from beamtracer.solver.mesh import FaceCondition

    with pytest.raises(ValueError, match="planar"):
        IonOpticsMesh((5, 5), (0.0, 0.0), 1.0, "cylindrical", symmetry_plane=True)
    with pytest.raises(ValueError, match="zero"):
        IonOpticsMesh(
            (5, 5), (0.0, 0.0), 1.0, "planar",
            faces={"y_min": FaceCondition("dirichlet", 0.0)}, symmetry_plane=True,
        )  # fmt: skip
    with pytest.raises(ValueError, match="planar"):
        _lens(full=False).replace(
            grid=GridParams(num_gridpoints=(11, 11), coordinates="cylindrical")
        )


def test_the_setting_round_trips() -> None:
    config = _lens(full=False)
    assert BeamOpticsConfig.from_dict(config.to_dict()).symmetry_plane
