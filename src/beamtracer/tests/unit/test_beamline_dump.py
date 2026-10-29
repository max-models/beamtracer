"""The beamline: charged markers through the dump and the stray field."""

import numpy as np
import pytest
from scipy import constants
from simgeometry import Cuboid, SlantedTube

from beamtracer.beamline import (
    Duct,
    FieldRegion,
    Markers,
    box_field,
    gaussian_beam,
    profile_field,
)

MASS = 2.014 * constants.atomic_mass
SPEED = 1e7  # about 1 MeV for deuterium


def pencil(states, num: int = 1, slope: float = 0.0) -> Markers:
    """Markers on the axis at x = 0, along +x, one per state."""
    states = np.asarray(states)
    position = np.zeros((3, states.size * num))
    velocity = np.zeros((3, states.size * num))
    velocity[0] = SPEED
    velocity[1] = SPEED * slope
    return Markers(position, velocity, 1.0, np.repeat(states, num))


def test_uniform_magnetic_field_bends_on_the_larmor_circle():
    b = 0.5
    radius = MASS * SPEED / (constants.e * b)  # about 0.42 m
    exit = 0.2
    region = FieldRegion(
        [], exit=exit, mass=MASS, magnetic=box_field((0, 0, b)), step=1e-3, save=1
    )
    result = region.trace(pencil([2]))
    assert len(result.exit) == 1
    # a D+ along +x in +B_z turns towards -y
    y = -(radius - np.sqrt(radius**2 - exit**2))
    assert result.exit.position[1, 0] == pytest.approx(y, rel=1e-4)
    assert np.linalg.norm(result.exit.velocity[:, 0]) == pytest.approx(SPEED)
    path = result.trajectories[0]
    centre = np.array([0.0, -radius])
    assert np.allclose(np.hypot(*(path[:2] - centre[:, None])), radius, rtol=1e-5)


def test_uniform_electric_field_gives_the_parabola():
    e_field = 1e6
    region = FieldRegion(
        [], exit=1.0, mass=MASS, electric=box_field((0, e_field, 0)), step=1e-3
    )
    result = region.trace(pencil([0, 1, 2]))
    deflection = 0.5 * constants.e * e_field / MASS * (1.0 / SPEED) ** 2
    y = result.exit.position[1]
    order = np.argsort(result.exit.state)
    assert np.allclose(y[order], [-deflection, 0.0, deflection], rtol=1e-3, atol=1e-12)


def dump_plates(gap: float, length: float) -> list[Cuboid]:
    """Two plates at y = +-gap/2 from x = 0 to length."""
    return [
        Cuboid((length / 2, s * (gap / 2 + 0.01), 0), (length, 0.02, 1.0))
        for s in (1, -1)
    ]


def test_the_dump_sorts_the_states_onto_the_plates():
    gap, length, e_field = 0.1, 1.5, 2e5
    beam = gaussian_beam(
        3000, power=3e6, speed=SPEED, width=0.005, divergence=0.003, seed=6
    )
    beam.state = np.repeat([0, 1, 2], 1000)
    plates = dump_plates(gap, length)
    region = FieldRegion(
        plates,
        exit=length,
        mass=MASS,
        electric=box_field((0, e_field, 0), (0, -gap / 2, -1), (length, gap / 2, 1)),
        step=2e-3,
    )
    result = region.trace(beam)
    # the neutrals pass, the D+ hit the plate at +y, the D- the one at -y
    assert np.all(result.exit.state == 1)
    assert len(result.exit) == 1000
    assert np.all(result.wall_index[result.wall.state == 2] == 0)
    assert np.all(result.wall_index[result.wall.state == 0] == 1)
    assert np.allclose(np.abs(result.wall.position[1]), gap / 2)
    assert np.allclose(np.abs(result.wall_normal[1]), 1.0)
    # where on the plate: the parabola from the axis
    distance = SPEED * np.sqrt(gap / (constants.e * e_field / MASS))
    assert np.median(result.wall.position[0]) == pytest.approx(distance, rel=0.1)
    total = result.exit.total_power + result.wall.total_power
    assert total == pytest.approx(3e6)
    assert float(result.summary().power_lost) == 0.0
    assert result.wall_power(2) == pytest.approx([1e6, 1e6])


def test_stray_field_bends_reionized_ions_onto_the_duct():
    from plasmacoll import ChargeStateTransitions

    length, bore = 3.0, 0.1
    tube = SlantedTube((0, 0, 0), (length, 0, 0), bore, 0.3, (1, 0, 0))
    transitions = ChargeStateTransitions({"D2": {("0", "+"): 2e-20}})
    beam = gaussian_beam(4000, power=1.0, speed=SPEED, width=0.01, divergence=0.0)
    duct = Duct([tube], port=length, transitions=transitions, gas={"D2": 5e18})
    reionized = duct.trace(beam).reionized
    assert len(reionized) > 100
    stray = profile_field([0.0, length], [[0, 0], [0, 0], [0.05, 0.2]])
    bent = FieldRegion([tube], exit=length, mass=MASS, magnetic=stray, step=5e-3)
    result = bent.trace(reionized)
    # the ions turn towards -y, the ones born early hit the bore
    assert len(result.wall) > 0
    assert np.all(result.wall.position[1] < 0)
    assert np.allclose(np.hypot(*result.wall.position[1:]), bore)
    assert result.wall.position[0].min() > reionized.position[0].min()
    assert len(result.wall) + len(result.exit) == len(reionized)


def test_markers_inside_a_wall_or_past_the_exit_stop_at_once():
    plate = Cuboid((0.5, 0.0, 0.0), (1.0, 0.02, 0.02))
    markers = pencil([1, 1])
    markers.position[0, 1] = 2.0
    result = FieldRegion([plate], exit=1.5, mass=MASS).trace(markers)
    assert len(result.wall) == 1 and len(result.exit) == 1


def test_lost_markers_and_refused_inputs():
    # an ion turned round by a strong field never reaches the exit
    region = FieldRegion(
        [], exit=10.0, mass=MASS, magnetic=box_field((0, 0, 5.0)), max_steps=50
    )
    result = region.trace(pencil([2]))
    assert len(result.lost) == 1
    with pytest.raises(ValueError, match="no charge"):
        FieldRegion([], exit=1.0, mass=MASS, charges={"0": 0.0}).trace(pencil([2]))
    with pytest.raises(ValueError, match="positive"):
        FieldRegion([], exit=1.0, mass=MASS, step=0.0)
    with pytest.raises(ValueError, match="increasing"):
        profile_field([0.0], [[0], [0], [0]])


def test_box_and_profile_fields():
    field = box_field((1.0, 2.0, 3.0), (0, 0, 0), (1, 1, 1))
    points = np.array([[0.5, 2.0], [0.5, 0.5], [0.5, 0.5]])
    assert np.allclose(field(points), [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    profile = profile_field([0.0, 2.0], [[0, 0], [0, 0], [0.0, 1.0]])
    assert np.allclose(profile(np.array([[1.0, 5.0], [0, 0], [0, 0]]))[2], [0.5, 1.0])
