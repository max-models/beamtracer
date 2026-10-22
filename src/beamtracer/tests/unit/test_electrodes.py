"""Tests of electrodes made of several shapes (``ObjectParams.union``)."""

from __future__ import annotations

import json

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
    ShapeParams,
)

VOLTAGE = -800.0


def _tube_shapes() -> list[ShapeParams]:
    """A tube with rounded ends: a rectangle and a circle at each end."""
    return [
        ShapeParams("rectangle", midpoint=(5e-3, 3e-3), side_lengths=(3e-3, 1e-3)),
        ShapeParams("circle", midpoint=(3.5e-3, 3e-3), radius=0.5e-3),
        ShapeParams("circle", midpoint=(6.5e-3, 3e-3), radius=0.5e-3),
    ]


def _config(objects: dict[str, ObjectParams]) -> BeamOpticsConfig:
    """A planar gap at 0 V and -1 kV with a beam that partly hits the tube."""
    neumann = FieldBoundaryCondition.neumann(0.0)
    return BeamOpticsConfig(
        grid=GridParams(
            num_gridpoints=(41, 21),
            lower_bound=(0.0, 0.0),
            upper_bound=(10e-3, 5e-3),
        ),
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(0.0),
            x_upper=FieldBoundaryCondition.dirichlet(-1000.0),
            y_lower=neumann,
            y_upper=neumann,
        ),
        objects=objects,
        beams=[
            BeamSpecies(
                "H+",
                1.0,
                (0.0, 0.0),
                (0.0, 5e-3),
                energy=0.01,
                transverse_temperature=0.0,
                num_particles=40,
            )
        ],
        iterations=1,
    )


def _union() -> BeamOpticsConfig:
    return _config({"tube": ObjectParams.union(_tube_shapes(), VOLTAGE)})


def _separate() -> BeamOpticsConfig:
    objects = {}
    for number, shape in enumerate(_tube_shapes()):
        geometry = {key: value for key, value in vars(shape).items() if key != "type"}
        objects[f"tube.{number}"] = ObjectParams(
            type=shape.type, bc_type="dirichlet", bc_value=VOLTAGE, **geometry
        )
    return _config(objects)


def test_a_union_is_one_electrode_with_the_potential_of_its_shapes() -> None:
    union = BeamOptics(_union()).run()
    separate = BeamOptics(_separate()).run()
    assert len(union.mesh.electrodes) == 1
    np.testing.assert_array_equal(union.mesh.fixed, separate.mesh.fixed)
    np.testing.assert_array_equal(union.mesh.fixed_value, separate.mesh.fixed_value)
    np.testing.assert_array_equal(union.mesh.distance, separate.mesh.distance)
    np.testing.assert_array_equal(union.potential, separate.potential)


def test_hits_and_diagnostics_name_the_union_once() -> None:
    model = BeamOptics(_union()).run()
    assert model.stats["electrode"] > 0
    hit = model.state.trace.electrode
    assert set(hit[hit >= 0].tolist()) == {0}
    fields = model.surface_fields()
    assert list(fields) == ["tube"]
    separate = BeamOptics(_separate()).run().surface_fields()
    assert len(separate) == 3
    assert fields["tube"] == max(separate.values())


def test_a_union_round_trips_through_its_configuration_file_form() -> None:
    config = _union()
    params = json.loads(json.dumps(config.to_dict()))
    tube = params["objects"]["tube"]
    assert tube["type"] == "union"
    assert [shape["type"] for shape in tube["shapes"]] == [
        "rectangle",
        "circle",
        "circle",
    ]
    assert tube["bc"] == {
        "type": "dirichlet",
        "potential": {"value": VOLTAGE, "unit": "V"},
    }
    again = BeamOpticsConfig.from_dict(params)
    assert again.to_dict() == config.to_dict()
    assert again.objects["tube"].bc_value == VOLTAGE
    assert [shape.to_dict() for shape in again.objects["tube"].shape_list] == [
        shape.to_dict() for shape in _tube_shapes()
    ]


def test_union_shapes_read_units_and_configuration_entries() -> None:
    tube = ObjectParams(
        type="union",
        bc_type="dirichlet",
        bc_value=VOLTAGE,
        shapes=[
            {
                "type": "circle",
                "midpoint": {"values": [3.5, 3.0], "unit": "mm"},
                "radius": {"value": 0.5, "unit": "mm"},
            },
            ShapeParams("rectangle", midpoint=(5e-3, 3e-3), side_lengths=(3e-3, 1e-3)),
        ],  # type: ignore[list-item]
    )
    circle = tube.shape_list[0]
    assert circle.radius == pytest.approx(0.5e-3)
    assert tube.build().contains_points(np.array([[3.2e-3], [3e-3]]))[0]


def test_a_union_is_checked() -> None:
    with pytest.raises(ValueError, match="at least one shape"):
        ObjectParams.union([], VOLTAGE)
    with pytest.raises(ValueError, match="in its shapes"):
        ObjectParams(
            type="union",
            bc_type="dirichlet",
            bc_value=VOLTAGE,
            radius=1e-3,
            shapes=_tube_shapes(),
        )
    with pytest.raises(ValueError, match="only a union"):
        ObjectParams(
            type="circle",
            bc_type="dirichlet",
            bc_value=VOLTAGE,
            midpoint=(0.0, 0.0),
            radius=1e-3,
            shapes=_tube_shapes(),
        )
    with pytest.raises(ValueError, match="shape type"):
        ShapeParams("union")
    with pytest.raises(ValueError, match="no constructor arguments"):
        ObjectParams.union(_tube_shapes(), VOLTAGE).get_filtered_obj_params()
