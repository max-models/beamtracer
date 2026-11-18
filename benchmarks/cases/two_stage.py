"""A two-gap planar accelerator taking protons from 10 eV to 50 keV (IBSimu's ``sim_two_stage_accel``)."""

from __future__ import annotations

from common import Case, Outcome, Region, beamtracer_run, ibsimu_run, rectangle

FACES = [None, None, None, None]
# The example's emitting line runs from (1 mm, 17 mm) to (0, 23 mm).
BEAM = {
    "current_density": 1.0e-3,
    "energy": 10.0,
    "start": (1.0e-3, 0.017),
    "end": (0.0, 0.023),
}
CENTRE = 0.02
# name, x range, potential; every aperture is 10 mm wide
ELECTRODES = (
    ("anode", (-0.01, 0.01), 50_000.0),
    ("intermediate", (0.09, 0.11), 10_000.0),
    ("ground", (0.19, 0.21), 0.0),
)


def solids() -> list:
    """Return IBSimu's electrodes: inside tests and potentials."""
    return [
        (lambda x, y, z: x < 0.01 and abs(y - CENTRE) > 0.005, 50_000.0),
        (lambda x, y, z: 0.09 < x < 0.11 and abs(y - CENTRE) > 0.005, 10_000.0),
        (lambda x, y, z: 0.19 < x < 0.2 and abs(y - CENTRE) > 0.005, 0.0),
    ]


def electrodes() -> dict:
    """Return the same electrodes for beamtracer: each a rectangle above and one below."""
    objects = {}
    for name, (x0, x1), potential in ELECTRODES:
        objects[f"{name}_top"] = rectangle(
            x0, x1, CENTRE + 0.005, 2 * CENTRE + 0.01, potential, strict=True
        )
        objects[f"{name}_bottom"] = rectangle(
            x0, x1, -0.01, CENTRE - 0.005, potential, strict=True
        )
    return objects


def run_ibsimu(
    nodes: tuple[int, int], h: float, particles: int, threads: int, region: Region
) -> Outcome:
    """Run the case in IBSimu."""
    return ibsimu_run(
        CASE, nodes, h, particles, threads, region,
        solids=solids(), faces=FACES, beam=BEAM,
    )  # fmt: skip


def run_beamtracer(
    nodes: tuple[int, int], h: float, particles: int, integrator: str, region: Region
) -> Outcome:
    """Run the case in beamtracer."""
    return beamtracer_run(
        CASE, nodes, h, particles, integrator, region,
        objects=electrodes(), faces=FACES, beam=BEAM,
    )  # fmt: skip


CASE = Case(
    name="two_stage",
    title="Two-gap planar accelerator, 10 eV to 50 keV",
    source="sim_two_stage_accel.py",
    cylindrical=False,
    nodes=(201, 41),
    h=1.0e-3,
    particles=2000,
    iterations=5,
    mass=1.0,
    symmetric=False,
    tolerance={
        "current": 0.01,
        "mean": 0.001,
        "rms": 0.02,
        "rms_angle": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
    notes="2000 trajectories instead of the example's 200.",
)
