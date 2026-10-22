"""Helpers of the Metal versions (``<name>/<name>_metal.py``) of the kernels.

A Metal version is the body of a Metal Shading Language kernel run by
:class:`cunumpy.kernels.MetalKernel` on the GPU of an Apple-silicon Mac, wrapped in
a function with the arguments of the Pyccel version: NumPy arrays in, results
written into the output arrays. The GPU has no float64, so the kernels compute in
float32 (``float64="cast"``): expect a relative error of about 1e-5 against the
CPU versions. Import a version only where :func:`metal_available` is true.
"""

from __future__ import annotations

import cunumpy as xp

#: Threads per threadgroup of every Metal kernel.
THREADGROUP = 256


def metal_available() -> bool:
    """Whether the Metal versions can run here (MLX installed, Metal GPU present)."""
    return bool(xp.kernels.metal_available())


def make_kernel(
    name: str,
    source: str,
    inputs: list[str],
    outputs: list[str],
    header: str = "",
) -> xp.kernels.MetalKernel:
    """A :class:`~cunumpy.kernels.MetalKernel` computing in float32, outputs zeroed."""
    return xp.kernels.MetalKernel(
        source,
        inputs=inputs,
        outputs=outputs,
        name=name,
        header=header,
        threadgroup=THREADGROUP,
        float64="cast",
        init_value=0,
    )
