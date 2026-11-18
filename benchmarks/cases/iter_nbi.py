"""A seven-electrode cylindrical accelerator to 120 keV with chamfered apertures (IBSimu's ``sim_iter_nbi``)."""

from __future__ import annotations

import math

from common import EPS, Case, Outcome, Region, beamtracer_run, ibsimu_run, polygon

FACES = [None, None, None, None]
# The example's emitting line runs from (1 mm, 0) to (0, 6.5 mm).
BEAM = {
    "current_density": 100.0,
    "energy": 10.0,
    "start": (1.0e-3, 0.0),
    "end": (0.0, 6.5e-3),
}
CENTRES = (5e-3, 20e-3, 50e-3, 80e-3, 110e-3, 140e-3, 170e-3)
VOLTAGES = (120e3, 110e3, 85e3, 65e3, 45e3, 25e3, 0.0)
THICKNESS, APERTURE, SLOPE = 6e-3, 7e-3, math.tan(math.radians(30.0))


def inside(centre: float):
    """Return IBSimu's inside test of the electrode at ``centre``: its aperture opens downstream."""

    def test(x: float, r: float, z: float) -> bool:
        dx = x - centre
        if abs(dx) > THICKNESS / 2:
            return False
        return r > APERTURE + (dx + THICKNESS / 2) * SLOPE

    return test


def solids() -> list:
    """Return IBSimu's electrodes: inside tests and potentials."""
    return [(inside(c), v) for c, v in zip(CENTRES, VOLTAGES, strict=True)]


def electrodes() -> dict:
    """Return the same electrodes for beamtracer, reaching beyond the mesh's 10 mm radius.

    IBSimu's test includes the faces in x (``|dx| <= t/2``) but not the aperture surface
    (``r >`` it): the faces are moved out and the aperture surface up by `EPS`.
    """
    objects = {}
    for k, (c, v) in enumerate(zip(CENTRES, VOLTAGES, strict=True)):
        x0, x1 = c - THICKNESS / 2 - EPS, c + THICKNESS / 2 + EPS
        r0, r1 = (
            APERTURE - EPS * SLOPE + EPS,
            APERTURE + (THICKNESS + EPS) * SLOPE + EPS,
        )
        vertices = [(x0, r0), (x1, r1), (x1, 0.02), (x0, 0.02)]
        objects[f"electrode{k + 1}"] = polygon(vertices, v, strict=None)
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
    name="iter_nbi",
    title="Seven-electrode cylindrical accelerator to 120 keV, no space charge",
    source="sim_iter_nbi.py",
    cylindrical=True,
    nodes=(401, 21),
    h=0.5e-3,
    particles=2000,
    iterations=1,
    mass=1.0,
    symmetric=True,
    tolerance={
        "current": 0.01,
        "rms": 0.02,
        "rms_angle": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
    notes="2000 trajectories instead of the example's 500; one field solve and trace, as in the example.",
)
