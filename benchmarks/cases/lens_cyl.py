"""A cylindrical einzel lens in a beam pipe, 30 keV protons (IBSimu's ``sim_cyl_lens``)."""

from __future__ import annotations

from common import Case, Outcome, Region, beamtracer_run, ibsimu_run, rectangle

FACES = [None, None, None, None]
BEAM = {
    "current_density": 100.0,
    "energy": 30_000.0,
    "start": (0.0, 0.0),
    "end": (0.0, 8.0e-3),
}


def solids() -> list:
    """Return IBSimu's electrodes: inside tests and potentials."""
    return [
        (lambda x, r, z: r > 0.015, 0.0),
        (lambda x, r, z: 0.04 < x < 0.06 and r > 0.01, -15_000.0),
    ]


def electrodes() -> dict:
    """Return the same electrodes for beamtracer."""
    return {
        "pipe": rectangle(-0.01, 0.11, 0.015, 0.03, 0.0, strict=True),
        "lens": rectangle(0.04, 0.06, 0.01, 0.03, -15_000.0, strict=True),
    }


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
    name="lens_cyl",
    title="Cylindrical einzel lens, 30 keV protons, space charge",
    source="sim_cyl_lens.py",
    cylindrical=True,
    nodes=(201, 41),
    h=0.5e-3,
    particles=2000,
    iterations=2,
    mass=1.0,
    symmetric=True,
    tolerance={
        "current": 0.01,
        "rms": 0.01,
        "rms_angle": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
    notes="2000 trajectories instead of the example's 500.",
)
