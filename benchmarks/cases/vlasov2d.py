"""Planar proton extraction with three electrodes and space charge (IBSimu's ``vlasov2d``)."""

from __future__ import annotations

from common import Case, Outcome, Region, beamtracer_run, ibsimu_run, polygon, rectangle

FACES = [-3000.0, -1000.0, None, None]
BEAM = {
    "current_density": 50.0,
    "energy": 3000.0,
    "start": (0.0, 0.0),
    "end": (0.0, 0.012),
}
TOP, RIGHT = 0.06, 0.13  # beyond the mesh, 0.12 m by 0.05 m


def solids() -> list:
    """Return IBSimu's electrodes: inside tests and potentials."""
    return [
        (lambda x, y, z: x <= 0.02 and y >= 0.018, -3000.0),
        (lambda x, y, z: 0.03 <= x <= 0.04 and y >= 0.02, -14000.0),
        (lambda x, y, z: x >= 0.06 and y >= 0.03 and y >= 0.07 - 0.5 * x, -1000.0),
    ]


def electrodes() -> dict:
    """Return the same electrodes for beamtracer."""
    return {
        "plasma": rectangle(-0.01, 0.02, 0.018, TOP, -3000.0),
        "puller": rectangle(0.03, 0.04, 0.02, TOP, -14000.0),
        "ground": polygon(
            [(0.06, 0.04), (0.08, 0.03), (RIGHT, 0.03), (RIGHT, TOP), (0.06, TOP)],
            -1000.0,
        ),
    }


def run_ibsimu(
    nodes: tuple[int, int], h: float, particles: int, threads: int, region: Region
) -> Outcome:
    """Run the case in IBSimu."""
    return ibsimu_run(
        CASE, nodes, h, particles, threads, region,
        solids=solids(), faces=FACES, beam=BEAM, solver="bicgstab",
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
    name="vlasov2d",
    title="Planar proton extraction, three electrodes, space charge",
    source="vlasov2d.py",
    cylindrical=False,
    nodes=(241, 101),
    h=0.5e-3,
    particles=10_000,
    iterations=5,
    mass=1.0,
    symmetric=True,
    tolerance={
        "current": 0.01,
        "rms": 0.01,
        "rms_angle": 0.01,
        "energy": 0.001,
        "potential_rms": 0.002,
    },
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
)
