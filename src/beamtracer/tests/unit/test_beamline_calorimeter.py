"""The beamline: the calorimeter and the beam's width and divergence from it."""

import numpy as np
import pytest

from beamtracer.beamline import (
    Calorimeter,
    divergence_from_widths,
    fit_footprint,
    gaussian_beam,
)

SPEED = 1e7


def test_the_map_holds_the_power_on_the_face():
    beam = gaussian_beam(50_000, power=5e6, speed=SPEED, width=0.02, divergence=0.01)
    calorimeter = Calorimeter(2.0, (-0.1, 0.1, -0.05, 0.05), tiles=(20, 10))
    power_map = calorimeter.measure(beam)
    assert power_map.shape == (20, 10)
    tile = 0.01 * 0.01
    landed = beam.drift_to(2.0).position[1:]
    on = (np.abs(landed[0]) <= 0.1) & (np.abs(landed[1]) <= 0.05)
    assert float(power_map.sum()) * tile == pytest.approx(5e6 * on.mean())
    assert power_map.attrs["position"] == 2.0


@pytest.mark.parametrize("distance", [1.0, 4.0])
def test_the_fit_recovers_the_width(distance):
    width, divergence = 0.01, 0.006
    beam = gaussian_beam(
        200_000, power=1.0, speed=SPEED, width=width, divergence=divergence
    )
    sigma = np.hypot(width, distance * divergence)
    half = 5 * sigma
    power_map = Calorimeter(distance, (-half, half, -half, half), (50, 50)).measure(
        beam
    )
    fit = fit_footprint(power_map)
    assert fit["width_y"] == pytest.approx(sigma, rel=0.02)
    assert fit["width_z"] == pytest.approx(sigma, rel=0.02)
    assert abs(fit["centre_y"]) < 0.02 * sigma
    assert fit["peak"] == pytest.approx(1 / (2 * np.pi * sigma**2), rel=0.03)


def test_the_divergence_from_two_calorimeters():
    width, divergence = 0.01, 0.006
    beam = gaussian_beam(
        200_000, power=1.0, speed=SPEED, width=width, divergence=divergence
    )
    distances = [1.0, 3.0, 6.0]
    widths = []
    for d in distances:
        half = 5 * np.hypot(width, d * divergence)
        fit = fit_footprint(
            Calorimeter(d, (-half, half, -half, half), (50, 50)).measure(beam)
        )
        widths.append(fit["width_y"])
    w0, theta = divergence_from_widths(distances, widths)
    assert theta == pytest.approx(divergence, rel=0.03)
    assert w0 == pytest.approx(width, rel=0.15)
    assert beam.rms_angle == pytest.approx([divergence] * 2, rel=0.01)


def test_divergence_from_exact_widths():
    distances = np.array([0.5, 2.0])
    widths = np.hypot(0.02, distances * 0.01)
    assert divergence_from_widths(distances, widths) == pytest.approx((0.02, 0.01))
    with pytest.raises(ValueError, match="two distances"):
        divergence_from_widths([1.0], [0.1])
