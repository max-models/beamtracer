"""Planar positive-ion extraction from a plasma with Boltzmann electrons (IBSimu's ``plasma2d``)."""

from common import Case, Electrode

TOP, RIGHT = 0.01, 0.015  # beyond the mesh, 12 mm by 7.04 mm

CASE = Case(
    name="plasma2d",
    title="Planar positive-ion extraction, Boltzmann electrons",
    source="plasma2d.py",
    cylindrical=False,
    nodes=(76, 45),
    h=1.6e-4,
    particles=50_000,
    iterations=5,
    mass=1.0,
    symmetric=True,
    electrodes=(
        Electrode(
            "plasma",
            (
                (-1e-3, 1.5e-3),
                (0.5e-3, 1.5e-3),
                (0.5e-3, 0.5e-3),
                (0.75e-3, 0.5e-3),
                (2.0e-3, 3.0e-3),
                (2.0e-3, TOP),
                (-1e-3, TOP),
            ),
            0.0,
        ),  # fmt: skip
        Electrode(
            "puller",
            (
                (10.0e-3, 2.0e-3),
                (10.5e-3, 1.5e-3),
                (RIGHT, 1.5e-3),
                (RIGHT, TOP),
                (10.0e-3, TOP),
            ),
            -8000.0,
        ),
    ),
    faces=(None, -8000.0, None, None),
    beam={
        "current_density": 600.0,
        "energy": 5.0,
        "transverse_temperature": 0.5,
        "start": (0.0, 0.0),
        "end": (0.0, 1.5e-3),
    },  # fmt: skip
    tolerance={
        "current": 0.02,
        "rms": 0.02,
        "rms_angle": 0.03,
        "energy": 0.001,
        "potential_rms": 0.005,
    },  # fmt: skip
    plasma={"initial_x": 0.6e-3, "potential": 5.0, "electron_temperature": 5.0},
)
