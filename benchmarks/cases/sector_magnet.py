"""A 0.2 T sector field bending a 50 keV proton beam (IBSimu's ``sim_sector_magnet``)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from common import Case, Outcome, Region, beamtracer_run, ibsimu_run

FACES = [
    0.0,
    0.0,
    0.0,
    0.0,
]  # IBSimu does not solve; grounded faces keep beamtracer's potential zero
BEAM = {
    "current_density": 1.0e-3,
    "energy": 50_000.0,
    "start": (0.0, 0.095),
    "end": (0.0, 0.105),
}
FIELD, X0, X1 = 0.2, 0.05, 0.1  # Bz in 50 mm < x < 100 mm, set at the nodes


def bz(nodes: tuple[int, int], h: float) -> np.ndarray:
    """Return Bz at the nodes, ``(nx, ny)``, as the example sets it."""
    x = np.arange(nodes[0]) * h
    inside = (x > X0) & (x < X1)
    return np.repeat(np.where(inside, FIELD, 0.0)[:, None], nodes[1], axis=1)


def magnetic(geom: object, nodes: tuple[int, int], h: float) -> object:
    """Return IBSimu's mesh field with Bz at the nodes."""
    import ibsimu

    field = ibsimu.MeshVectorField(
        ibsimu.MODE_2D, [False, False, True], ibsimu.Int3D(nodes[0], nodes[1], 1),
        ibsimu.Vec3D(0, 0, 0), h,
    )  # fmt: skip
    values = bz(nodes, h)
    for i, j in zip(*np.nonzero(values), strict=True):
        field.set3(int(i), int(j), 0, ibsimu.Vec3D(0, 0, float(values[i, j])))
    return field


def run_ibsimu(
    nodes: tuple[int, int], h: float, particles: int, threads: int, region: Region
) -> Outcome:
    """Run the case in IBSimu: no electrodes and no field solve."""
    return ibsimu_run(
        CASE, nodes, h, particles, threads, region,
        solids=[], faces=FACES, beam=BEAM, magnetic=magnetic, solve=False,
    )  # fmt: skip


def run_beamtracer(
    nodes: tuple[int, int], h: float, particles: int, integrator: str, region: Region
) -> Outcome:
    """Run the case in beamtracer, the field a map at the same nodes."""
    from beamtracer import FieldMapParams
    from beamtracer.field_map import FieldMap

    x, y = np.arange(nodes[0]) * h, np.arange(nodes[1]) * h
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sector.nc"
        FieldMap.from_arrays({"x": x, "y": y}, {"b_field_z": bz(nodes, h)}).write(path)
        return beamtracer_run(
            CASE, nodes, h, particles, integrator, region,
            objects={}, faces=FACES, beam=BEAM, field_map=FieldMapParams(str(path)),
        )  # fmt: skip


CASE = Case(
    name="sector_magnet",
    title="Sector magnet, 50 keV protons, 0.2 T",
    source="sim_sector_magnet.py",
    cylindrical=False,
    nodes=(101, 101),
    h=2.0e-3,
    particles=500,
    iterations=1,
    mass=1.0,
    symmetric=False,
    tolerance={
        "current": 0.01,
        "mean": 0.001,
        "rms": 0.01,
        "rms_angle": 0.01,
        "energy": 0.001,
    },
    ibsimu=run_ibsimu,
    beamtracer=run_beamtracer,
    notes="500 trajectories instead of the example's 50.",
)
