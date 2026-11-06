"""Field maps: the shared netCDF layout, interpolation, and imposed fields in traces."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy import constants

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    FieldMap,
    FieldMapParams,
    GridParams,
    ParticleBeam,
)
from beamtracer.beamline import FieldRegion, Markers
from beamtracer.config import AxialField
from beamtracer.field_map import (
    FIELD_MAP_FORMAT,
    FieldMapError,
    parse_frame_axes,
    validate_field_map,
)

PROTON = 1.007276 * constants.atomic_mass


def picnic_style_map() -> xr.Dataset:
    """A map built with xarray alone, as picnic's ``FieldMap.from_arrays`` writes one."""
    x = np.linspace(0.0, 0.1, 11)
    y = np.linspace(-0.02, 0.02, 5)
    bz = 0.01 + 0.1 * x[:, None] + 0.5 * y[None, :]
    return xr.Dataset(
        {
            "b_field_z": (("x", "y"), bz, {"units": "T"}),
            "e_field_x": (("x", "y"), np.full((11, 5), 100.0), {"units": "V/m"}),
        },
        coords={
            "x": ("x", x, {"units": "m"}),
            "y": ("y", y, {"units": "m"}),
        },
        attrs={
            "Conventions": "max-models-field-map-1.0",
            "frame": "cartesian",
            "source": "picnic.parameters.field_map.FieldMap.from_arrays",
            "frame_axes": "x=z y=-s z=r",
        },
    )


# -- the format -------------------------------------------------------------------------------


def test_round_trip(tmp_path) -> None:
    x, y = np.linspace(0.0, 1.0, 5), np.linspace(0.0, 0.5, 3)
    field_map = FieldMap.from_function(
        lambda p: np.stack([p[:, 0], p[:, 1], p[:, 0] * p[:, 1]], axis=1),
        {"x": x, "y": y},
        kind="electric",
        frame="cylindrical",
        frame_axes="x=z y=r",
        source="test",
    )
    path = field_map.write(tmp_path / "map.nc")
    assert field_map.path == path
    again = FieldMap.read(path, outside="zero")
    assert again.path == path and again.outside == "zero"
    assert again.frame == "cylindrical" and again.dims == ("x", "y")
    assert again.dataset.attrs["Conventions"] == FIELD_MAP_FORMAT
    assert again.dataset.attrs["frame_axes"] == "x=z y=r"
    assert again.components("electric") == ("e_field_x", "e_field_y", "e_field_z")
    assert again.components("magnetic") == ()
    for name in again.components("electric"):
        np.testing.assert_array_equal(
            again.dataset[name].values, field_map.dataset[name].values
        )
        assert again.dataset[name].attrs["units"] == "V/m"
    assert again.dataset["x"].attrs["units"] == "m"


def test_a_map_written_by_picnic_is_read(tmp_path) -> None:
    path = tmp_path / "picnic.nc"
    picnic_style_map().to_netcdf(path)
    field_map = FieldMap.read(path)
    # B_z is linear in x and y, so the multilinear interpolation is exact
    points = np.array([[0.033, 0.007], [0.091, -0.013]])
    b = field_map.magnetic_field(points)
    expected = 0.01 + 0.1 * points[:, 0] + 0.5 * points[:, 1]
    np.testing.assert_allclose(b[:, 2], expected, rtol=1e-12)
    np.testing.assert_array_equal(b[:, :2], 0.0)
    np.testing.assert_allclose(field_map.electric_field(points), [[100, 0], [100, 0]])
    # and what beamtracer writes passes picnic's checks: the same attributes and names
    written = xr.open_dataset(field_map.write(tmp_path / "again.nc")).load()
    assert validate_field_map(written) == []
    assert written.attrs == picnic_style_map().attrs
    for name in ("x", "y", "b_field_z", "e_field_x"):
        assert written[name].attrs["units"] == picnic_style_map()[name].attrs["units"]
        assert written[name].dims == picnic_style_map()[name].dims


def test_validation_lists_every_problem() -> None:
    good = picnic_style_map()
    assert validate_field_map(good) == []
    bad = good.assign_attrs(Conventions="max-models-output-1.0", frame="polar")
    bad = bad.assign_attrs(frame_axes="x=z y=z")
    bad["b_field_z"].attrs["units"] = "G"
    bad["x"].attrs["units"] = "mm"
    bad["e_field_x"] = bad["e_field_x"].transpose("y", "x")
    bad["potential"] = (("x", "y"), np.zeros((11, 5)), {"units": "V"})
    problems = validate_field_map(bad)
    for text in (
        "Conventions",
        "frame is 'polar'",
        "z is mapped twice",
        "units 'G'",
        "coordinate 'x' has units 'mm'",
        "e_field_x: dims",
        "potential: not a field-map variable",
    ):
        assert any(text in p for p in problems), text
    with pytest.raises(FieldMapError, match="not a field map"):
        FieldMap(bad)

    decreasing = good.assign_coords(x=("x", good.x.values[::-1], {"units": "m"}))
    assert any("not increasing" in p for p in validate_field_map(decreasing))
    no_coordinate = good.drop_vars("y")
    assert any("no coordinate" in p for p in validate_field_map(no_coordinate))
    timed = good.expand_dims(t=[0.0])
    assert any("(x, y, z) only" in p for p in validate_field_map(timed))
    empty = xr.Dataset(
        coords={"x": ("x", [0.0, 1.0], {"units": "m"})}, attrs=good.attrs
    )
    assert any("no b_field" in p for p in validate_field_map(empty))


def test_frame_axes() -> None:
    assert parse_frame_axes("x=z y=-s z=r") == {
        "x": (1, "z"),
        "y": (-1, "theta"),
        "z": (1, "r"),
    }
    for text, match in (
        ("x=q", "cannot read"),
        ("x=z x=r", "x is mapped twice"),
        ("x=z y=s z=r", "left-handed"),
    ):
        with pytest.raises(FieldMapError, match=match):
            parse_frame_axes(text)


def test_interpolation_outside_and_missing_dims() -> None:
    x = np.linspace(0.0, 1.0, 11)
    field_map = FieldMap.from_arrays({"x": x}, {"b_field_y": 2.0 * x})
    # a 1D map is uniform along y and z
    points = np.array([[0.25, 7.0, -3.0], [0.55, 0.0, 0.0]])
    np.testing.assert_allclose(field_map.magnetic_field(points)[:, 1], [0.5, 1.1])
    outside = np.array([[-1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(field_map.magnetic_field(outside)[:, 1], [0.0, 2.0])
    zero = FieldMap(field_map.dataset, outside="zero")
    np.testing.assert_array_equal(zero.magnetic_field(outside), 0.0)
    with pytest.raises(ValueError, match="outside the field map"):
        FieldMap(field_map.dataset, outside="error").magnetic_field(outside)
    with pytest.raises(ValueError, match="outside must be"):
        FieldMap(field_map.dataset, outside="wrap")
    with pytest.raises(ValueError, match="no e_field"):
        field_map.electric_field(points)
    with pytest.raises(ValueError, match="no e_field"):
        field_map.points_function("electric")

    plane = FieldMap.from_arrays(
        {"x": x, "y": x, "z": x}, {"e_field_x": np.ones((11, 11, 11))}
    )
    with pytest.raises(ValueError, match="at least 3 coordinates"):
        plane.electric_field(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="2D plane"):
        plane.plane_function("electric")

    mesh = np.meshgrid(x[:3], x[:2], indexing="ij")
    values = field_map.evaluate(tuple(mesh))
    assert values.shape == (3, 2, 3)
    np.testing.assert_allclose(values[:, 0, 1], 2.0 * x[:3])
    assert field_map.evaluate((x[:4],)).shape == (4, 3)


# -- imposed fields in traces -----------------------------------------------------------------


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
