"""Imposed fields from takeout field maps: in traces, in BeamOptics and its config."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants
from takeout import FieldMap

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    FieldMapParams,
    GridParams,
    ParticleBeam,
)
from beamtracer.beamline import FieldRegion, Markers
from beamtracer.config import AxialField

PROTON = 1.007276 * constants.atomic_mass


def test_field_region_traces_an_analytic_field_from_its_map() -> None:
    """A D+ through a Gaussian B_z(x): the trace in the map follows the analytic one."""
    mass, speed = 2.014 * constants.atomic_mass, 1e7
    width, peak = 0.05, 0.5

    def analytic(points: np.ndarray) -> np.ndarray:
        b = np.zeros_like(points)
        b[2] = peak * np.exp(-((points[0] - 0.1) ** 2) / width**2)
        return b

    field_map = FieldMap.from_function(
        lambda p: analytic(np.pad(p.T, ((0, 2), (0, 0)))).T,
        {"x": np.linspace(0.0, 0.2, 801)},
        source="a Gaussian B_z(x)",
    )
    markers = Markers(
        np.zeros((3, 1)), np.array([[speed], [0.0], [0.0]]), 1.0, np.array([2])
    )
    ends = []
    for magnetic in (analytic, field_map):
        region = FieldRegion([], exit=0.2, mass=mass, magnetic=magnetic, step=1e-3)
        ends.append(region.trace(markers).exit.position[:, 0])
    # a deflection of a few cm, the same to the interpolation error
    assert ends[0][1] < -0.01
    np.testing.assert_allclose(ends[1], ends[0], rtol=1e-4, atol=1e-7)

    cylindrical = FieldMap.from_arrays(
        {"x": [0.0, 1.0]}, {"b_field_x": [1.0, 1.0]}, frame="cylindrical"
    )
    with pytest.raises(ValueError, match="'cartesian' field map"):
        FieldRegion([], exit=1.0, mass=mass, magnetic=cylindrical)


def _planar(field_map: str | None = None, **settings) -> BeamOpticsConfig:
    """Grounded walls around 10 mm x 5 mm, one proton at y = 2.5 mm along +x."""
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(21, 11),
            lower_bound=(0.0, 0.0),
            upper_bound=(10e-3, 5e-3),
            coordinates="cartesian",
        ),
        beams=[ParticleBeam("H+", [[0.0, 2.5e-3, 0.0]], [[1e5, 0.0, 0.0]], [1e-12])],
        field_map=field_map,
        iterations=1,
        diagnostic_planes=[9e-3],
        **settings,
    )


def test_beam_optics_takes_e_and_b_from_a_map(tmp_path) -> None:
    b, e = 0.01, 1e3
    path = tmp_path / "planar.nc"
    FieldMap.from_arrays(
        {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        {"b_field_z": np.full((2, 2), b)},
    ).write(path)
    model = BeamOptics(_planar(str(path))).run()
    # a proton along +x in +B_z turns towards -y on its Larmor circle
    radius = PROTON * 1e5 / (constants.e * b)
    y = 2.5e-3 - (radius - np.sqrt(radius**2 - 9e-3**2))
    assert model.plane(9e-3).point[1, 0] == pytest.approx(y, rel=1e-5)

    FieldMap.from_arrays(
        {"x": [-1.0, 1.0]}, {"e_field_y": [e, e]}, source="uniform E_y"
    ).write(path)
    model = BeamOptics(_planar(str(path))).run()
    t = 9e-3 / 1e5
    y = 2.5e-3 + 0.5 * constants.e * e / PROTON * t**2
    assert model.plane(9e-3).point[1, 0] == pytest.approx(y, rel=1e-5)


def test_beam_optics_map_matches_the_axial_field(tmp_path) -> None:
    """A uniform B_x from a map and from a measured axial profile trace alike."""
    b = 0.05
    path = tmp_path / "cylindrical.nc"
    FieldMap.from_arrays(
        {"x": [-1.0, 1.0], "y": [0.0, 1.0]},
        {"b_field_x": np.full((2, 2), b)},
        frame="cylindrical",
    ).write(path)
    settings = {
        "grid": GridParams(
            ndim=2,
            num_gridpoints=(21, 11),
            lower_bound=(0.0, 0.0),
            upper_bound=(10e-3, 5e-3),
            coordinates="cylindrical",
        ),
        "beams": [ParticleBeam("H+", [[0.0, 1e-3, 0.0]], [[1e5, 0.0, 2e4]], [1e-12])],
        "iterations": 1,
        "diagnostic_planes": [9e-3],
    }
    hits = []
    for extra in (
        {"field_map": FieldMapParams(str(path))},
        {"axial_field": AxialField(-0.01, 1e-3, (b,) * 40)},
    ):
        model = BeamOptics(BeamOpticsConfig(**settings, **extra)).run()
        hits.append(model.plane(9e-3).point[:, 0])
    np.testing.assert_allclose(hits[0], hits[1], rtol=1e-9, atol=1e-12)
    assert abs(hits[0][1]) > 1e-6  # the field turned the trajectory

    # both: the map adds to the solenoids
    config = BeamOpticsConfig(
        **settings,
        field_map={"file": str(path)},
        axial_field=AxialField(-0.01, 1e-3, (0.0,) * 40),
    )
    model = BeamOptics(config)
    assert model.tracer.magnetic(np.array([1e-3]), np.array([1e-3]))[0] == (
        pytest.approx(b)
    )


def test_azimuthal_component_and_wrong_maps(tmp_path) -> None:
    path = tmp_path / "theta.nc"
    FieldMap.from_arrays(
        {"x": [-1.0, 1.0]}, {"b_field_z": [0.1, 0.1]}, frame="cylindrical"
    ).write(path)
    config = _planar().replace(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(21, 11),
            upper_bound=(10e-3, 5e-3),
            coordinates="cylindrical",
        ),
        field_map=str(path),
    )
    model = BeamOptics(config)
    # at y > 0 the azimuth is +z; v_x x B_theta z pushes a proton to -y (the axis)
    x = np.array([[1e-3], [1e-3], [0.0]])
    v = np.array([[1e5], [0.0], [0.0]])
    tracer = model.tracer
    tracer.electric = lambda x, y: (0.0 * x, 0.0 * x)
    a = tracer.acceleration(x, v, constants.e / PROTON)
    assert a[1, 0] == pytest.approx(-constants.e / PROTON * 1e5 * 0.1)
    assert a[0, 0] == 0.0 and a[2, 0] == 0.0

    with pytest.raises(ValueError, match="'cylindrical' frame"):
        BeamOptics(_planar(str(path)))
    FieldMap.from_arrays(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
        {"e_field_x": np.zeros((2, 2, 2))},
    ).write(tmp_path / "3d.nc")
    with pytest.raises(ValueError, match=r"needs one on \(x\[, y\]\)"):
        BeamOptics(_planar(str(tmp_path / "3d.nc")))


def test_config_round_trip_and_checks(tmp_path) -> None:
    config = _planar(str(tmp_path / "map.nc"))
    assert config.field_map == FieldMapParams(str(tmp_path / "map.nc"))
    params = config.to_dict()
    assert params["field_map"] == {"file": str(tmp_path / "map.nc"), "outside": "clip"}
    assert BeamOpticsConfig.from_dict(params).field_map == config.field_map
    assert _planar().to_dict().get("field_map") is None
    with pytest.raises(ValueError, match="outside must be"):
        FieldMapParams("map.nc", outside="nearest")
