"""The NumPy version of the ``staggered_field_2d`` kernel, for machines without Pyccel.

Vectorized over the points; the same arguments and results as
``staggered_field_2d_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def _bilinear(values, s, t, imax, jmax):
    """Interpolate ``values[i, j]`` at fractional indices, clamping the cell."""
    i = np.clip(np.floor(s).astype(np.int64), 0, imax)
    j = np.clip(np.floor(t).astype(np.int64), 0, jmax)
    u = s - i
    w = t - j
    low = values[i, j] + w * (values[i, j + 1] - values[i, j])
    high = values[i + 1, j] + w * (values[i + 1, j + 1] - values[i + 1, j])
    return low + u * (high - low)


def staggered_field_2d(
    px: np.ndarray,
    py: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    origin_x: float,
    origin_y: float,
    h: float,
    ex: np.ndarray,
    ey: np.ndarray,
) -> None:
    """``(E_x, E_y)`` at the points ``(px, py)`` from the edge fields."""
    nx, ny = fy.shape[0], fx.shape[1]
    sx = (px - origin_x) / h
    sy = (py - origin_y) / h
    ex[:] = _bilinear(fx, sx + 0.5, sy, nx - 1, ny - 2)
    ey[:] = _bilinear(fy, sx, sy + 0.5, nx - 2, ny - 1)
