"""The NumPy version of the ``deposit_trajectories_2d`` kernel, for machines without Pyccel.

Vectorized over the charges; the same arguments and results as
``deposit_trajectories_2d_pyccel.py``.
"""

from __future__ import annotations

import numpy as np


def deposit_trajectories_2d(
    px: np.ndarray,
    py: np.ndarray,
    q: np.ndarray,
    origin_x: float,
    origin_y: float,
    h: float,
    cylindrical: int,
    charge: np.ndarray,
) -> None:
    """Add the charges ``q`` at ``(px, py)`` to the node charges, area-weighted."""
    nx, ny = charge.shape
    i = np.floor((px - origin_x) / h).astype(np.int64)
    j = np.floor((py - origin_y) / h).astype(np.int64)
    t = (px - (origin_x + h * i)) / h
    if cylindrical:
        r1 = origin_y + h * j
        r2 = r1 + h
        u = (py * py - r1 * r1) / (r2 * r2 - r1 * r1)
    else:
        u = (py - (origin_y + h * j)) / h
    t = np.where(i < 0, 0.0, np.where(i >= nx - 1, 1.0, t))
    i = np.clip(i, 0, nx - 2)
    u = np.where(j < 0, 0.0, np.where(j >= ny - 1, 1.0, u))
    j = np.clip(j, 0, ny - 2)
    p = i * ny + j
    nodes = np.concatenate([p, p + 1, p + ny, p + ny + 1])
    weights = np.concatenate(
        [(1.0 - t) * (1.0 - u) * q, (1.0 - t) * u * q, t * (1.0 - u) * q, t * u * q]
    )
    charge += np.bincount(nodes, weights=weights, minlength=charge.size).reshape(nx, ny)
