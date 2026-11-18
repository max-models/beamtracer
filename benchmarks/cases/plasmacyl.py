"""Cylindrical He+ extraction from a plasma with Boltzmann electrons (IBSimu's ``plasmacyl``)."""

from __future__ import annotations

from common import Case, Outcome, Region, beamtracer_run, ibsimu_run, polygon

FACES = [None, -12_000.0, None, None]
BEAM = {
    "current_density": 600.0, "energy": 5.0, "transverse_temperature": 0.5,
    "start": (0.0, 0.0), "end": (0.0, 1.5e-3),
}  # fmt: skip
PLASMA = {"initial_x": 0.55e-3, "potential": 5.0, "electron_temperature": 5.0}
TOP, RIGHT = 0.01, 0.015  # beyond the mesh, 12 mm by 7 mm


def solids() -> list:
    """Return IBSimu's electrodes: inside tests and potentials."""

    def plasma_electrode(x: float, r: float, z: float) -> bool:
        return x <= 1.87e-3 and r >= 0.54e-3 and r >= 2.28 * x - 1.0e-3

    def puller(x: float, r: float, z: float) -> bool:
        return x >= 9.5e-3 and r >= 2.3333e-3 and r >= 12.8333e-3 - x

    return [(plasma_electrode, 0.0), (puller, -12_000.0)]


def electrodes() -> dict:
    """Return the same electrodes for beamtracer."""
    knee = (0.54e-3 + 1.0e-3) / 2.28  # where r = 2.28 x - 1 mm reaches 0.54 mm
    return {
        "plasma": polygon(
            [
                (-1e-3, 0.54e-3),
                (knee, 0.54e-3),
                (1.87e-3, 2.28 * 1.87e-3 - 1.0e-3),
                (1.87e-3, TOP),
                (-1e-3, TOP),
            ],
            0.0,
        ),  # fmt: skip
        "puller": polygon(
            [
                (9.5e-3, 3.3333e-3),
                (10.5e-3, 2.3333e-3),
                (RIGHT, 2.3333e-3),
                (RIGHT, TOP),
                (9.5e-3, TOP),
            ],
            -12_000.0,
        ),
    }


def run_ibsimu(
    nodes: tuple[int, int], h: float, particles: int, threads: int, region: Region
) -> Outcome:
    """Run the case in IBSimu."""
    return ibsimu_run(
        CASE, nodes, h, particles, threads, region,
        solids=solids(), faces=FACES, beam=BEAM, plasma=PLASMA, solver="bicgstab",
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
