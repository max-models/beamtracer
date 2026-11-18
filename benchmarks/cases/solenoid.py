"""A 0.5 T solenoid focusing 10 keV protons in a grounded pipe (IBSimu's ``sim_solenoid``)."""

from __future__ import annotations

import numpy as np
from common import Case, Outcome, Region, beamtracer_run, ibsimu_run, rectangle

FACES = [None, None, None, None]
BEAM = {
    "current_density": 1.0e-3, "energy": 10_000.0, "transverse_temperature": 0.02,
    "start": (0.0, 0.0), "end": (0.0, 5.0e-3),
}  # fmt: skip
PEAK, CENTRE, WIDTH, DZ = 0.5, 0.1, 0.03, 1.0e-3


def axial_field() -> np.ndarray:
    """Return the example's on-axis field: a Gaussian sampled every mm from 0 to 0.2 m."""
    z = np.linspace(0.0, 0.2, 201)
    return PEAK * np.exp(-((z - CENTRE) ** 2) / WIDTH**2)


def magnetic(geom: object, nodes: tuple[int, int], h: float) -> object:
    """Return IBSimu's field of the on-axis profile."""
    import ibsimu

    return ibsimu.AxisymmetricVectorField(
        ibsimu.MODE_CYL, 0.0, DZ, axial_field().tolist()
    )


def solids() -> list:
    """Return IBSimu's electrode, the grounded pipe: inside test and potential."""
    return [(lambda x, r, z: r > 0.025, 0.0)]


def run_ibsimu(
    nodes: tuple[int, int], h: float, particles: int, threads: int, region: Region
) -> Outcome:
    """Run the case in IBSimu: no field solve, the pipe is grounded."""
    return ibsimu_run(
        CASE, nodes, h, particles, threads, region,
        solids=solids(), faces=FACES, beam=BEAM,
        magnetic=magnetic, solve=False,
    )  # fmt: skip


def run_beamtracer(
    nodes: tuple[int, int], h: float, particles: int, integrator: str, region: Region
) -> Outcome:
    """Run the case in beamtracer."""
    from beamtracer import AxialField

    return beamtracer_run(
        CASE, nodes, h, particles, integrator, region,
        objects={"pipe": rectangle(-0.01, 0.21, 0.025, 0.04, 0.0, strict=True)}, faces=FACES, beam=BEAM,
        axial_field=AxialField(z0=0.0, dz=DZ, bz=tuple(axial_field())),
    )  # fmt: skip


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
    tolerance={"current": 0.01, "rms": 0.02, "rms_angle": 0.03, "energy": 0.001},
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
    notes="2000 trajectories instead of the example's 100.",
)
