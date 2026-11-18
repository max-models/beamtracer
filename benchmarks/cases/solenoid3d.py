"""Protons gyrating in an axial magnetic field in 3D (after IBSimu's ``viz_solenoid_3d``)."""

from typing import Any

import numpy as np
from common3d import Case3D

PEAK, CENTRE, WIDTH = 1.0, 0.1, 0.04


def bx(x: np.ndarray) -> np.ndarray:
    """The example's axial field: a Gaussian of 1 T peak at x = 100 mm, uniform across."""
    return PEAK * np.exp(-((x - CENTRE) ** 2) / WIDTH**2)


def make(particles: int) -> Any:
    """Protons at 440 km/s along x on three rings, with a transverse velocity of 40 km/s."""
    radius = np.repeat([5e-3, 10e-3, 15e-3], 8)
    angle = np.tile(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False), 3)
    n = radius.size
    position = np.stack([np.zeros(n), radius * np.cos(angle), radius * np.sin(angle)])
    velocity = np.stack(
        [np.full(n, 440e3), -40e3 * np.sin(angle), 40e3 * np.cos(angle) * 0.5]
    )
    return position, velocity, np.full(n, 1e-12)


CASE = Case3D(
    name="solenoid3d",
    title="Axial magnetic field, 3D gyration",
    source="viz_solenoid_3d.py",
    nodes=(101, 26, 26),
    origin=(0.0, -25e-3, -25e-3),
    h=2e-3,
    particles=24,
    iterations=1,
    mass=1.0,
    symmetry_planes=(),
    electrodes=(),
    # IBSimu solves no field; grounded faces keep beamtracer's potential zero too
    faces=(0.0,) * 6,
    beam={"make": make},
    tolerance={
        "current": 1e-9,
        "mean_y": 1e-3,
        "mean_z": 1e-3,
        "rms_y": 1e-3,
        "rms_z": 1e-3,
        "rms_angle_y": 2e-3,
        "rms_angle_z": 2e-3,
        "energy": 1e-5,
    },  # fmt: skip
    solve=False,
    bx=bx,
    notes=(
        "The example's particles move along the field and feel no force; here they get a "
        "transverse velocity, so they gyrate. Both codes take the field at the nodes."
    ),
)
