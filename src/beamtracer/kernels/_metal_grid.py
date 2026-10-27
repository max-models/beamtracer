"""Helpers of the Metal gather, scatter and deposit kernels.

Positions ``(ndim, N)`` are turned into cell coordinates ``(x - lower) / cell_size`` on
the host in float64; the kernels compute the cell and the weights in float32 with
:data:`CELL_HEADER`, the arithmetic of picnic's ``linear_cell.cuh``. A scatter accumulates
into a zeroed float32 grid with float atomics (no 64-bit atomics on the Apple GPU),
which the host adds to the caller's grid in float64: the summation order, hence the
last bits, differ from the CPU. See :mod:`beamtracer.kernels`.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from numpy import ndarray as Array

from beamtracer.kernels import KERNEL_PACKAGE

__all__ = ["CELL_HEADER", "make_scatter_kernel", "relative", "scatter"]

#: The cell of a coordinate along one axis (the arithmetic of picnic's ``linear_cell.cuh``).
CELL_HEADER = """
inline void beamtracer_linear_cell(
    float rel, int num, int periodic,
    thread int& left, thread int& right, thread float& fraction) {
    int l;
    if (periodic) {
        rel = rel - (float)num * metal::floor(rel / (float)num);
        l = ((int)rel) % num;
        right = (l + 1) % num;
    } else {
        l = rel >= 0.0f ? (int)rel : -1;
        if (l < 0) l = 0;
        if (l > num - 2) l = num - 2;
        right = l + 1;
    }
    left = l;
    fraction = metal::clamp(rel - (float)l, 0.0f, 1.0f);
}
"""


def relative(
    positions: Array, lower: tuple[float, ...], cell: tuple[float, ...]
) -> Array:
    """Return the cell coordinates ``(x - lower) / cell_size``, shape ``(ndim, N)``.

    ``positions`` is component-major, ``(ndim, N)``; the result has the same layout,
    so a Metal kernel reads component ``d`` of marker ``i`` as ``rel[d * n + i]``.
    """
    positions = np.asarray(positions)
    offsets = np.asarray(lower, dtype=np.float64)
    sizes = np.asarray(cell, dtype=np.float64)
    return (positions.reshape(-1, positions.shape[-1]) - offsets[:, None]) / sizes[
        :, None
    ]


def make_scatter_kernel(
    name: str, source: str, inputs: list[str], header: str = ""
) -> xp.kernels.MetalKernel:
    """A kernel adding into the zeroed float32 output ``delta`` with float atomics."""
    return KERNEL_PACKAGE.metal_kernel(
        name, source, inputs, ["delta"], header=header, atomic_outputs=True
    )


def scatter(
    kernel: xp.kernels.MetalKernel,
    *args: object,
    field: Array,
    n_threads: int,
) -> None:
    """Run a scatter ``kernel`` and add its float32 result to ``field`` (in float64)."""
    delta = np.zeros(field.shape, dtype=np.float64)
    kernel(*args, out=delta, n_threads=max(n_threads, 1))
    field += delta
