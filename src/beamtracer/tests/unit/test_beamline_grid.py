"""The beamline: the whole grid from one beamlet."""

import numpy as np
import pytest
from scipy import constants

from beamtracer.beamline import ApertureGrid, beamlet_markers, gaussian_beam
from beamtracer.solver.tracer import PlaneHits

MASS = 2.014 * constants.atomic_mass
SPEED = 1e7


def ring_hits(num: int = 50, fractions: bool = False) -> PlaneHits:
    """Crossings of a cylindrical beamlet at x = 0.1: rings out to 5 mm, slopes r / 1 m."""
    rng = np.random.default_rng(3)
    r = rng.uniform(0.0, 5e-3, num)
    hits = PlaneHits(0.1)
    point = np.vstack([np.full(num, 0.1), r, np.zeros(num)])
    velocity = np.vstack([np.full(num, SPEED), SPEED * r / 1.0, np.zeros(num)])
    share = None
    if fractions:
        share = np.vstack([np.full(num, 0.2), np.full(num, 0.5), np.full(num, 0.3)])
    hits.append(
        point,
        velocity,
        np.full(num, 1e-3),
        np.zeros(num, dtype=int),
        np.ones(num, dtype=int),
        share,
    )
    return hits


def kinetic(velocity: np.ndarray) -> np.ndarray:
    return 0.5 * MASS * np.sum(velocity**2, axis=0)


def test_cylindrical_beamlet_becomes_rings():
    hits = ring_hits()
    markers = beamlet_markers(hits, mass=MASS, cylindrical=True, copies=400)
    assert len(markers) == 50 * 400
    # the radius and the radial slope of every copy are those of its ring
    r = np.hypot(markers.position[1], markers.position[2])
    assert np.allclose(r, np.repeat(hits.point[1], 400))
    radial = (
        markers.position[1] * markers.velocity[1]
        + markers.position[2] * markers.velocity[2]
    ) / r
    assert np.allclose(radial, np.repeat(hits.velocity[1], 400))
    # the azimuths are spread round the axis
    assert abs(markers.position[1].mean()) < 2e-4
    assert abs(markers.position[2].mean()) < 2e-4
    expected = np.sum(hits.current / constants.e * kinetic(hits.velocity))
    assert markers.total_power == pytest.approx(expected)
    assert np.all(markers.state == 0)  # "-" by default


def test_species_and_backward_crossings_are_left_out():
    hits = ring_hits(10)
    hits.species[:4] = 1
    hits.direction[:2] = -1
    markers = beamlet_markers(hits, mass=MASS, cylindrical=True, species=0)
    assert len(markers) == 6
    assert len(beamlet_markers(hits, mass=MASS, cylindrical=True)) == 8


def test_drift_moves_along_straight_lines():
    beam = gaussian_beam(100, power=1.0, speed=SPEED, width=1e-3, divergence=0.01)
    moved = beam.drift_to(2.0)
    assert np.allclose(moved.position[0], 2.0)
    slope = beam.velocity[1:] / beam.velocity[0]
    assert np.allclose(moved.position[1:], beam.position[1:] + 2.0 * slope)


def test_planar_beamlet_spreads_over_the_slit():
    hits = ring_hits()
    markers = beamlet_markers(
        hits, mass=MASS, cylindrical=False, slit_length=0.02, copies=10, state="0"
    )
    assert np.all(np.abs(markers.position[2]) <= 0.01)
    assert np.allclose(markers.position[1], np.repeat(hits.point[1], 10))
    expected = 0.02 * np.sum(hits.current / constants.e * kinetic(hits.velocity))
    assert markers.total_power == pytest.approx(expected)
    assert np.all(markers.state == 1)


def test_fractions_split_the_power():
    hits = ring_hits(fractions=True)
    markers = beamlet_markers(hits, mass=MASS, cylindrical=True, copies=3)
    total = np.sum(hits.current / constants.e * kinetic(hits.velocity))
    per_state = np.bincount(markers.state, weights=markers.power, minlength=3)
    assert np.allclose(per_state / total, [0.2, 0.5, 0.3])


def test_refused_beamlets():
    with pytest.raises(ValueError, match="slit_length"):
        beamlet_markers(ring_hits(), mass=MASS, cylindrical=False)
    with pytest.raises(ValueError, match="copies"):
        beamlet_markers(ring_hits(), mass=MASS, cylindrical=True, copies=0)
    with pytest.raises(ValueError, match="one row per state"):
        beamlet_markers(
            ring_hits(fractions=True), mass=MASS, cylindrical=True, states=("0", "+")
        )


def slope_rms(markers) -> np.ndarray:
    return np.std(markers.velocity[1:] / markers.velocity[0], axis=1)


def test_flat_grid_keeps_the_beamlet_divergence():
    beamlet = gaussian_beam(
        5000, power=1.0, speed=SPEED, width=2e-3, divergence=5e-3, seed=4
    )
    grid = ApertureGrid.regular(3, 4, (0.02, 0.022))
    beam = grid.markers(beamlet)
    assert len(beam) == 12 * 5000
    assert beam.total_power == pytest.approx(12.0)
    assert np.allclose(slope_rms(beam), slope_rms(beamlet))
    # the centres
    centres = beam.position[1:].reshape(2, 12, 5000).mean(axis=2)
    assert np.allclose(centres, grid.centres, atol=2e-4)
    assert np.allclose(grid.centres[0].reshape(3, 4)[0], [-0.03, -0.01, 0.01, 0.03])


def centroid_at(markers, x: float, count: int) -> np.ndarray:
    """The centroid of each of ``count`` beamlets in the plane ``x``."""
    t = (x - markers.position[0]) / markers.velocity[0]
    at = markers.position[1:] + markers.velocity[1:] * t
    return at.reshape(2, count, -1).mean(axis=2)


def test_curved_grid_aims_at_the_focus():
    beamlet = gaussian_beam(
        2000, power=1.0, speed=SPEED, width=1e-3, divergence=0.0, seed=5
    )
    grid = ApertureGrid.regular(4, 5, (0.02, 0.02), focal_length=(7.0, 21.0))
    beam = grid.markers(beamlet)
    # in y the beamlets cross at 7 m, in z at 21 m
    own = beamlet.position[1:].mean(axis=1)[:, None]
    at_7 = centroid_at(beam, 7.0, len(grid)) - own
    at_21 = centroid_at(beam, 21.0, len(grid)) - own
    assert np.allclose(at_7[0], 0.0, atol=1e-6)
    assert np.allclose(at_21[1], 0.0, atol=1e-6)
    assert np.ptp(at_7[1]) > 0.03
    # every beamlet keeps its width
    width = beam.position[1:].reshape(2, len(grid), -1).std(axis=2)
    assert np.allclose(width, beamlet.position[1:].std(axis=1)[:, None], rtol=1e-3)


def test_steering_tilts_each_beamlet():
    beamlet = gaussian_beam(1000, power=1.0, speed=SPEED, width=1e-3, divergence=0.0)
    steering = np.array([[0.01, -0.02], [0.0, 0.005]])
    grid = ApertureGrid(np.zeros((2, 2)), steering=steering)
    beam = grid.markers(beamlet)
    own = beamlet.position[1:].mean(axis=1)[:, None]
    assert np.allclose(centroid_at(beam, 2.0, 2) - own, 2.0 * steering, atol=1e-6)
    assert np.allclose(grid.slopes, steering)
