"""A 0.5 T solenoid focusing 10 keV protons in a grounded pipe (IBSimu's ``sim_solenoid``)."""

from typing import Any

import numpy as np
from common import Case, Electrode, box

PEAK, CENTRE, WIDTH, DZ = 0.5, 0.1, 0.03, 1.0e-3


def axial_field() -> np.ndarray:
    """Return the example's on-axis field: a Gaussian sampled every mm from 0 to 0.2 m."""
    z = np.linspace(0.0, 0.2, 201)
    return PEAK * np.exp(-((z - CENTRE) ** 2) / WIDTH**2)


def ibsimu_magnetic(nodes: tuple[int, int], h: float) -> Any:
    """Return IBSimu's field of the on-axis profile."""
    import ibsimu

    return ibsimu.AxisymmetricVectorField(
        ibsimu.MODE_CYL, 0.0, DZ, axial_field().tolist()
    )


def beamtracer_fields(nodes: tuple[int, int], h: float) -> dict[str, Any]:
    """Return the same on-axis profile for beamtracer."""
    from beamtracer import AxialField

    return {"axial_field": AxialField(z0=0.0, dz=DZ, bz=tuple(axial_field()))}


CASE = Case(
    name="solenoid",
    title="Solenoid lens, 10 keV protons, 0.5 T",
    source="sim_solenoid.py",
    cylindrical=True,
    nodes=(201, 31),
    h=1.0e-3,
    particles=2000,
    iterations=1,
    mass=1.0,
    symmetric=True,
    electrodes=(Electrode("pipe", box(-0.01, 0.21, 0.025, 0.04), 0.0),),
    faces=(None, None, None, None),
    beam={
        "current_density": 1.0e-3,
        "energy": 10_000.0,
        "transverse_temperature": 0.02,
        "start": (0.0, 0.0),
        "end": (0.0, 5.0e-3),
    },  # fmt: skip
    tolerance={"current": 0.01, "rms": 0.02, "rms_angle": 0.03, "energy": 0.001},
    solve=False,
    ibsimu_magnetic=ibsimu_magnetic,
    beamtracer_fields=beamtracer_fields,
    notes="2000 trajectories instead of the example's 100; IBSimu solves no field (the pipe is grounded).",
)
