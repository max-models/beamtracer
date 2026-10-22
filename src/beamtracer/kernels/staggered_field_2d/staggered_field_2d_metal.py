"""The Metal version (Apple-silicon GPU, float32) of the ``staggered_field_2d`` kernel.

``staggered_field_2d_pyccel.py`` next to this file is the CPU version with the same
arguments. One thread per point; see :mod:`beamtracer.kernels._metal`. The cell
coordinates ``(p - origin) / h`` are formed on the host in float64, then the GPU
interpolates in float32.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray as Array

from beamtracer.kernels._metal import make_kernel

_HEADER = """
float beamtracer_clamped_bilinear(
    const device float* f, uint nj, float s, float t, int imax, int jmax) {
    int i = min(max((int)metal::floor(s), 0), imax);
    int j = min(max((int)metal::floor(t), 0), jmax);
    float u = s - (float)i;
    float w = t - (float)j;
    float low = f[i * nj + j] + w * (f[i * nj + j + 1] - f[i * nj + j]);
    float high = f[(i + 1) * nj + j] + w * (f[(i + 1) * nj + j + 1] - f[(i + 1) * nj + j]);
    return low + u * (high - low);
}
"""

_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= cell_x_shape[0]) return;
    int nx = fy_shape[0];
    int ny = fx_shape[1];
    float sx = cell_x[k];
    float sy = cell_y[k];
    ex[k] = beamtracer_clamped_bilinear(fx, fx_shape[1], sx + 0.5f, sy, nx - 1, ny - 2);
    ey[k] = beamtracer_clamped_bilinear(fy, fy_shape[1], sx, sy + 0.5f, nx - 2, ny - 1);
"""

_KERNEL = make_kernel(
    "beamtracer_staggered_field_2d",
    _SOURCE,
    inputs=["cell_x", "cell_y", "fx", "fy"],
    outputs=["ex", "ey"],
    header=_HEADER,
)


def staggered_field_2d(
    px: Array,
    py: Array,
    fx: Array,
    fy: Array,
    origin_x: float,
    origin_y: float,
    h: float,
    ex: Array,
    ey: Array,
) -> None:
    """``(E_x, E_y)`` at the points from the edge fields on the GPU."""
    cell_x = (np.asarray(px) - origin_x) / h
    cell_y = (np.asarray(py) - origin_y) / h
    _KERNEL(cell_x, cell_y, fx, fy, out=(ex, ey))
