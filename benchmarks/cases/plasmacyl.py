"""Cylindrical He+ extraction from a plasma with Boltzmann electrons (IBSimu's ``plasmacyl``)."""

from common import Case, Electrode

TOP, RIGHT = 0.01, 0.015  # beyond the mesh, 12 mm by 7 mm
KNEE = (0.54e-3 + 1.0e-3) / 2.28  # where r = 2.28 x - 1 mm reaches 0.54 mm

CASE = Case(
    name="plasmacyl",
    title="Cylindrical He+ extraction, Boltzmann electrons",
    source="plasmacyl.py",
    cylindrical=True,
    nodes=(241, 141),
    h=5.0e-5,
    particles=15_000,
    iterations=15,
    mass=4.0,
    symmetric=True,
    electrodes=(
        Electrode(
            "plasma",
            (
                (-1e-3, 0.54e-3),
                (KNEE, 0.54e-3),
                (1.87e-3, 2.28 * 1.87e-3 - 1.0e-3),
                (1.87e-3, TOP),
                (-1e-3, TOP),
            ),
            0.0,
        ),  # fmt: skip
        Electrode(
            "puller",
            (
                (9.5e-3, 3.3333e-3),
                (10.5e-3, 2.3333e-3),
                (RIGHT, 2.3333e-3),
                (RIGHT, TOP),
                (9.5e-3, TOP),
            ),
            -12_000.0,
        ),  # fmt: skip
    ),
    faces=(None, -12_000.0, None, None),
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
    solver="bicgstab",
    plasma={"initial_x": 0.55e-3, "potential": 5.0, "electron_temperature": 5.0},
)
