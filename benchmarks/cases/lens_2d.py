"""A planar einzel lens focusing a 10 keV proton beam (IBSimu's ``sim_2d_lens``)."""

from __future__ import annotations

from common import Case, Outcome, Region, beamtracer_run, ibsimu_run, rectangle

FACES = [None, None, None, None]
BEAM = {
    "current_density": 1.0e-3,
    "energy": 10_000.0,
    "start": (0.0, 0.015),
    "end": (0.0, 0.025),
}
CENTRE = 0.02
# name, x range, half gap, potential
ELECTRODES = (
    ("left", (-0.01, 0.04), 0.01, 0.0),
    ("middle", (0.045, 0.055), 0.005, -5000.0),
    ("right", (0.06, 0.11), 0.01, 0.0),
)


def solids() -> list:
    """Return IBSimu's electrodes: inside tests and potentials."""
    return [
        (lambda x, y, z: x < 0.04 and abs(y - CENTRE) > 0.01, 0.0),
        (lambda x, y, z: 0.045 < x < 0.055 and abs(y - CENTRE) > 0.005, -5000.0),
        (lambda x, y, z: 0.06 < x < 0.1 and abs(y - CENTRE) > 0.01, 0.0),
    ]


def electrodes() -> dict:
    """Return the same electrodes for beamtracer: each a rectangle above and one below."""
    objects = {}
    for name, (x0, x1), gap, potential in ELECTRODES:
        objects[f"{name}_top"] = rectangle(
            x0, x1, CENTRE + gap, 2 * CENTRE + 0.01, potential, strict=True
        )
        objects[f"{name}_bottom"] = rectangle(
            x0, x1, -0.01, CENTRE - gap, potential, strict=True
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
    name="lens_2d",
    title="Planar einzel lens, 10 keV protons",
    source="sim_2d_lens.py",
    cylindrical=False,
    nodes=(101, 41),
    h=1.0e-3,
    particles=2000,
    iterations=2,
    mass=1.0,
    symmetric=False,
    tolerance={
        "current": 0.01,
        "mean": 0.001,
        "rms": 0.01,
        "rms_angle": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
    notes="2000 trajectories instead of the example's 200.",
)
