"""The beamline: its power balance and the markers handed on."""

import numpy as np
import pytest
from plasmacoll import ChargeStateTransitions
from scipy import constants
from simgeometry import Cuboid, SlantedTube

from beamtracer.beamline import (
    ApertureGrid,
    Duct,
    FieldRegion,
    Markers,
    beamline_summary,
    box_field,
    gaussian_beam,
    profile_field,
)

MASS = 2.014 * constants.atomic_mass
SPEED = 1e7


def test_the_power_balance_over_the_whole_beamline():
    beamlet = gaussian_beam(500, power=1e4, speed=SPEED, width=2e-3, divergence=5e-3)
    beam = ApertureGrid.regular(3, 3, (0.02, 0.02)).markers(beamlet)
    beam.state = np.random.default_rng(1).choice(3, len(beam), p=[0.2, 0.6, 0.2])
    gap, length = 0.12, 1.5
    plates = [
        Cuboid((length / 2, s * (gap / 2 + 0.005), 0), (length, 0.01, 0.5))
        for s in (1, -1)
    ]
    dump = FieldRegion(
        plates,
        exit=length,
        mass=MASS,
        electric=box_field((0, 3e5, 0), (0, -gap / 2, -1), (length, gap / 2, 1)),
        step=2e-3,
    ).trace(beam)
    assert np.all(dump.exit.state == 1)  # the ions are dumped
    tube = SlantedTube((length, 0, 0), (4.0, 0, 0), 0.08, 0.3, (1, 0, 0))
    transitions = ChargeStateTransitions({"D2": {("0", "+"): 2e-20}})
    duct = Duct([tube], port=4.0, transitions=transitions, gas={"D2": 2e18}).trace(
        dump.exit
    )
    stray = profile_field([length, 4.0], [[0, 0], [0, 0], [0.02, 0.1]])
    bent = FieldRegion([tube], exit=4.0, mass=MASS, magnetic=stray).trace(
        duct.reionized
    )
    summary = beamline_summary(
        Markers.concatenate([duct.port, bent.exit]),
        {"dump": dump.wall, "duct": duct.wall, "duct (reionized)": bent.wall},
        power=beam.total_power,
        halo_radius=0.05,
        footprint_extent=(-0.1, 0.1, -0.1, 0.1),
    )
    total = float(summary.power_port + summary.power_on.sum())
    assert total == pytest.approx(beam.total_power, rel=1e-12)
    assert list(summary.part.values) == ["dump", "duct", "duct (reionized)"]
    assert 0.0 < float(summary.power_halo) < float(summary.power_port)
    assert summary.footprint.dims == ("y", "z")
    with pytest.raises(ValueError, match="add up"):
        beamline_summary(duct.port, {"dump": dump.wall}, power=beam.total_power)


def test_the_dataset_round_trip():
    beam = gaussian_beam(100, power=2e3, speed=SPEED, width=0.01, divergence=0.01)
    beam.state[:10] = 2
    data = beam.to_dataset(MASS)
    assert data.position.dims == ("component", "marker")
    energy = 0.5 * MASS * np.sum(beam.velocity**2, axis=0)
    assert np.allclose(data.weight.values * energy, beam.power)
    assert list(data.state.values[:11]) == ["+"] * 10 + ["0"]
    back = Markers.from_dataset(data)
    assert np.array_equal(back.position, beam.position)
    assert np.array_equal(back.state, beam.state)
    assert back.states == beam.states


def test_placing_the_beam_in_the_tokamak_frame():
    beam = gaussian_beam(50, power=1.0, speed=SPEED, width=0.01, divergence=0.01)
    beam = beam.drift_to(2.0)
    direction = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2)
    placed = beam.placed((5.0, -3.0, 0.2), direction, x=2.0)
    # the axis point goes to the origin, the axis to the direction, z stays up
    on_axis = Markers(np.array([[2.0], [0.0], [0.0]]), [[SPEED], [0], [0]], 1.0, 1)
    moved = on_axis.placed((5.0, -3.0, 0.2), direction, up=(0.0, 0.1, 1.0), x=2.0)
    assert np.allclose(moved.position[:, 0], [5.0, -3.0, 0.2])
    assert np.allclose(moved.velocity[:, 0], SPEED * direction)
    assert np.allclose(placed.position[2] - 0.2, beam.position[2])
    # distances and speeds are kept
    assert np.allclose(
        np.linalg.norm(placed.velocity, axis=0), np.linalg.norm(beam.velocity, axis=0)
    )
    with pytest.raises(ValueError, match="up"):
        beam.placed((0, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="same states"):
        Markers.concatenate(
            [beam, Markers(np.zeros((3, 1)), np.ones((3, 1)), 1.0, 0, ("a",))]
        )
