"""Positive-ion extraction through round apertures in 3D, a quarter with two symmetry planes."""

from typing import Any

import numpy as np
from common3d import Case3D, Tube
from scipy.constants import atomic_mass, e

DENSITY, ENERGY = 600.0, 5.0  # A/m^2, eV
RHO = DENSITY / np.sqrt(2.0 * ENERGY * e / atomic_mass)  # the beam's J / v at its start


def make(particles: int) -> Any:
    """Protons from the quarter 1.5 mm square at x = 0, 5 eV, 0.5 eV transverse."""
    from beamtracer.solver.tracer3d import beam_on_surface

    beam, _ = beam_on_surface(
        particles, DENSITY, 1.0, 1.0, ENERGY, 0.0, 0.5,
        (0.0, 0.75e-3, 0.75e-3), (1.5e-3, 1.5e-3),
    )  # fmt: skip
    return beam.position, beam.velocity, beam.current


CASE = Case3D(
    name="extraction3d",
    title="3D extraction, Boltzmann electrons",
    source="plasmacyl.py (made 3D)",
    nodes=(81, 26, 26),
    origin=(0.0, 0.0, 0.0),
    h=1e-4,
    particles=900,
    iterations=16,
    mass=1.0,
    symmetry_planes=("y_min", "z_min"),
    electrodes=(
        Tube("plasma", (1.0e-3, 1.8e-3), 0.6e-3, 5e-3, 0.0),
        Tube("puller", (6.0e-3, 7.0e-3), 1.2e-3, 5e-3, -10_000.0),
    ),
    faces=(None, -10_000.0, None, None, None, None),
    beam={"make": make},
    tolerance={
        "current": 0.02,
        "rms_y": 0.02,
        "rms_z": 0.02,
        "rms_angle_y": 0.03,
        "rms_angle_z": 0.03,
        "energy": 0.001,
        "potential_rms": 0.005,
    },  # fmt: skip
    solver="bicgstab",
    plasma={
        "initial_x": 0.5e-3,
        "potential": 5.0,
        "electron_temperature": 5.0,
        "rho": RHO,
    },
    notes=(
        "Not an IBSimu example: a plasma electrode and a puller with round apertures, "
        "the quarter of a round beamlet, in IBSimu's MODE_3D and in beamtracer. After 8 "
        "iterations the two still differ by 5% (the meniscus is still moving); from 14 on "
        "they agree to 0.2%."
    ),
)
