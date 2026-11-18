"""Planar positive-ion extraction from a plasma with Boltzmann electrons (IBSimu's ``plasma2d``)."""

from __future__ import annotations

from common import Case, Outcome, Region, beamtracer_run, ibsimu_run, polygon

FACES = [None, -8000.0, None, None]
BEAM = {
    "current_density": 600.0, "energy": 5.0, "transverse_temperature": 0.5,
    "start": (0.0, 0.0), "end": (0.0, 1.5e-3),
}  # fmt: skip
PLASMA = {"initial_x": 0.6e-3, "potential": 5.0, "electron_temperature": 5.0}
TOP, RIGHT = 0.01, 0.015  # beyond the mesh, 12 mm by 7.04 mm


def solids() -> list:
    """Return IBSimu's electrodes: inside tests and potentials."""

    def plasma_electrode(x: float, y: float, z: float) -> bool:
        return (
            x <= 2.0e-3
            and y >= 0.5e-3
            and y >= 2.0 * x - 1.0e-3
            and (x >= 0.5e-3 or y >= 1.5e-3)
        )

    def puller(x: float, y: float, z: float) -> bool:
        return x >= 10.0e-3 and y >= 1.5e-3 and y >= 12.0e-3 - x

    return [(plasma_electrode, 0.0), (puller, -8000.0)]


def electrodes() -> dict:
    """Return the same electrodes for beamtracer."""
    return {
        "plasma": polygon(
            [
                (-1e-3, 1.5e-3),
                (0.5e-3, 1.5e-3),
                (0.5e-3, 0.5e-3),
                (0.75e-3, 0.5e-3),
                (2.0e-3, 3.0e-3),
                (2.0e-3, TOP),
                (-1e-3, TOP),
            ],
            0.0,
        ),  # fmt: skip
        "puller": polygon(
            [
                (10.0e-3, 2.0e-3),
                (10.5e-3, 1.5e-3),
                (RIGHT, 1.5e-3),
                (RIGHT, TOP),
                (10.0e-3, TOP),
            ],
            -8000.0,
        ),
    }


def run_ibsimu(
    nodes: tuple[int, int], h: float, particles: int, threads: int, region: Region
) -> Outcome:
    """Run the case in IBSimu."""
    return ibsimu_run(
        CASE, nodes, h, particles, threads, region,
        solids=solids(), faces=FACES, beam=BEAM, plasma=PLASMA,
    )  # fmt: skip


def run_beamtracer(
    nodes: tuple[int, int], h: float, particles: int, integrator: str, region: Region
) -> Outcome:
    """Run the case in beamtracer."""
    from beamtracer import ExtractionPlasma

    plasma = ExtractionPlasma(
        electron_temperature=PLASMA["electron_temperature"],
        plasma_potential=PLASMA["potential"],
        initial_plasma_x=PLASMA["initial_x"],
    )
    return beamtracer_run(
        CASE, nodes, h, particles, integrator, region,
        objects=electrodes(), faces=FACES, beam=BEAM, plasma=plasma,
    )  # fmt: skip


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
    tolerance={
        "current": 0.02,
        "rms": 0.02,
        "rms_angle": 0.03,
        "energy": 0.001,
        "potential_rms": 0.005,
    },
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
)
