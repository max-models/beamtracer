"""Tests for the tabulated face condition, `FaceProfile`."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    FaceProfile,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ParticleBeam,
)


def test_profile_interpolates_along_its_coordinate() -> None:
    profile = FaceProfile("y", [0.0, 1.0e-3, 2.0e-3], [0.0, -2.0, -6.0], unit="kV")
    y = np.array([[0.0, 0.5e-3], [1.5e-3, 5.0e-3]])
    value = profile.evaluate(x=np.zeros_like(y), y=y)
    assert value.shape == y.shape
    # linear between the points, the end value beyond them, in volts
    np.testing.assert_allclose(value, [[0.0, -1000.0], [-4000.0, -6000.0]])


def test_profile_needs_a_valid_table() -> None:
    with pytest.raises(ValueError, match="x, y or z"):
        FaceProfile("r", [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="one value each"):
        FaceProfile("y", [0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="increase"):
        FaceProfile("y", [1.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="needs the y coordinate"):
        FaceProfile("y", [0.0, 1.0], [0.0, 1.0]).evaluate(x=np.zeros(2))


def test_profile_round_trips_through_the_configuration_form() -> None:
    face = FieldBoundaryCondition.profile("y", [0.0, 1.0e-3], [0.0, -50.0])
    neumann = FieldBoundaryCondition.neumann(0.0)
    bcs = FieldBCs(face, FieldBoundaryCondition.dirichlet(0.0), neumann, neumann)
    params = bcs.to_dict()
    assert params["x"]["min"] == {
        "profile": {
            "coordinate": "y",
            "points": [0.0, 1.0e-3],
            "values": [0.0, -50.0],
            "unit": "V",
        }
    }
    read = FieldBCs.from_dict(params).get_bc(dir="x", side="min")
    assert read.bc_type == "dirichlet"
    assert isinstance(read.value, FaceProfile)
    np.testing.assert_allclose(read.evaluate(x=0.0, y=0.5e-3), -25.0)


def _diode(upstream: FieldBoundaryCondition) -> BeamOpticsConfig:
    """Ions crossing a 10 mm gap whose upstream face is ``upstream``."""
    neumann = FieldBoundaryCondition.neumann(0.0)
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(21, 11),
            lower_bound=(0.0, 0.0),
            upper_bound=(10e-3, 5e-3),
            coordinates="cartesian",
        ),
        bcs_field=FieldBCs(
            upstream, FieldBoundaryCondition.dirichlet(-1000.0), neumann, neumann
        ),
        beams=[
            ParticleBeam(
                "H+",
                position=[[1e-6, y, 0.0] for y in np.linspace(0.5e-3, 4.5e-3, 9)],
                velocity=[[2.0e4, 0.0, 0.0]] * 9,
                current=[1.0e-4] * 9,
            )
        ],
        iterations=5,
    )


def test_profile_face_solves_like_the_same_expression() -> None:
    expression = FieldBoundaryCondition.dirichlet("-20000 * y", "V")
    points = np.linspace(0.0, 5e-3, 11)
    profile = FieldBoundaryCondition.profile("y", points, -20000.0 * points)
    by_expression = BeamOptics(_diode(expression)).run()
    by_profile = BeamOptics(_diode(profile)).run()
    np.testing.assert_allclose(
        by_profile.potential, by_expression.potential, rtol=0.0, atol=1e-9
    )
    # the face holds the tabulated values
    np.testing.assert_allclose(by_profile.potential[0], -20000.0 * points)
