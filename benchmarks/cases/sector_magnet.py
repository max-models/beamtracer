"""A 0.2 T sector field bending a 50 keV proton beam (IBSimu's ``sim_sector_magnet``)."""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from common import Case

FIELD, X0, X1 = 0.2, 0.05, 0.1  # Bz in 50 mm < x < 100 mm, set at the nodes


def bz(nodes: tuple[int, int], h: float) -> np.ndarray:
    """Return Bz at the nodes, ``(nx, ny)``, as the example sets it."""
    x = np.arange(nodes[0]) * h
    inside = (x > X0) & (x < X1)
    return np.repeat(np.where(inside, FIELD, 0.0)[:, None], nodes[1], axis=1)


def ibsimu_magnetic(nodes: tuple[int, int], h: float) -> Any:
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


def beamtracer_fields(nodes: tuple[int, int], h: float) -> dict[str, Any]:
    """Return the same field as a beamtracer field map at the nodes (a netCDF file)."""
    from beamtracer import FieldMapParams
    from beamtracer.field_map import FieldMap

    path = Path(tempfile.mkdtemp()) / "sector.nc"
    x, y = np.arange(nodes[0]) * h, np.arange(nodes[1]) * h
    FieldMap.from_arrays({"x": x, "y": y}, {"b_field_z": bz(nodes, h)}).write(path)
    return {"field_map": FieldMapParams(str(path))}


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
    electrodes=(),
    # IBSimu solves no field; grounded faces keep beamtracer's potential zero too
    faces=(0.0, 0.0, 0.0, 0.0),
    beam={
        "current_density": 1.0e-3,
        "energy": 50_000.0,
        "start": (0.0, 0.095),
        "end": (0.0, 0.105),
    },
    tolerance={
        "current": 0.01,
        "mean": 0.001,
        "mean_angle": 0.001,
        "rms": 0.01,
        "energy": 0.001,
    },
    solve=False,
    ibsimu_magnetic=ibsimu_magnetic,
    beamtracer_fields=beamtracer_fields,
    notes="500 trajectories instead of the example's 50.",
)
