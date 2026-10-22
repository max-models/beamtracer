"""The Metal version (Apple-silicon GPU, float32) of the ``deposit_trajectories_2d`` kernel.

``deposit_trajectories_2d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per charge; the coordinates relative to the origin are formed on
the host in float64 (``r1``, ``r2`` of the cylindrical weights as well), see
:mod:`beamtracer.kernels._metal_grid`.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray as Array

from beamtracer.kernels._metal_grid import make_scatter_kernel, scatter

_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= cell_x_shape[0]) return;
    int nx = shape[0], ny = shape[1];
    int i = (int)metal::floor(cell_x[k]);
    int j = (int)metal::floor(cell_y[k]);
    float t = frac_x[k];
    float u = frac_y[k];
    if (i < 0) { t = 0.0f; i = 0; }
    else if (i >= nx - 1) { t = 1.0f; i = nx - 2; }
    if (j < 0) { u = 0.0f; j = 0; }
    else if (j >= ny - 1) { u = 1.0f; j = ny - 2; }
    float value = q[k];
    atomic_fetch_add_explicit(&delta[i * ny + j], (1.0f - t) * (1.0f - u) * value, memory_order_relaxed);
    atomic_fetch_add_explicit(&delta[i * ny + j + 1], (1.0f - t) * u * value, memory_order_relaxed);
    atomic_fetch_add_explicit(&delta[(i + 1) * ny + j], t * (1.0f - u) * value, memory_order_relaxed);
    atomic_fetch_add_explicit(&delta[(i + 1) * ny + j + 1], t * u * value, memory_order_relaxed);
"""

_KERNEL = make_scatter_kernel(
    "beamtracer_deposit_trajectories_2d",
    _SOURCE,
    inputs=["cell_x", "cell_y", "frac_x", "frac_y", "q", "shape"],
)


def deposit_trajectories_2d(
    px: Array,
    py: Array,
    q: Array,
    origin_x: float,
    origin_y: float,
    h: float,
    cylindrical: int,
    charge: Array,
) -> None:
    """Add the charges ``q`` at ``(px, py)`` to the node charges on the GPU, area-weighted.

    Bilinear in ``x``; in ``y`` linear, or in cylindrical geometry (``y`` the radius)
    linear in ``r**2``. Charge beyond the mesh goes to its boundary nodes.
    """
    if len(px) == 0:
        return
    x = np.asarray(px, dtype=np.float64)
    y = np.asarray(py, dtype=np.float64)
    cell_x = np.floor((x - origin_x) / h)
    cell_y = np.floor((y - origin_y) / h)
    frac_x = (x - (origin_x + h * cell_x)) / h
    if cylindrical != 0:
        r1 = origin_y + h * cell_y
        r2 = r1 + h
        frac_y = (y * y - r1 * r1) / (r2 * r2 - r1 * r1)
    else:
        frac_y = (y - (origin_y + h * cell_y)) / h
    # cells far outside the mesh only need to stay outside it in float32
    cell_x = np.clip(cell_x, -2.0, charge.shape[0] + 1.0)
    cell_y = np.clip(cell_y, -2.0, charge.shape[1] + 1.0)
    scatter(
        _KERNEL,
        cell_x,
        cell_y,
        frac_x,
        frac_y,
        q,
        [charge.shape[0], charge.shape[1]],
        field=charge,
        n_threads=len(px),
    )
