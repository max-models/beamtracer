"""The beam tracer's compiled kernels: one folder per kernel.

Every kernel ``<name>`` lives in its own folder::

    beamtracer/kernels/<name>/__init__.py        kernel = KERNEL_PACKAGE.from_folder(__name__)
    beamtracer/kernels/<name>/<name>_pyccel.py   def <name>(...)    CPU, compiled with Pyccel
    beamtracer/kernels/<name>/<name>_cuda.cu     __global__ void <name>(...)    GPU (NVRTC)
    beamtracer/kernels/<name>/<name>_numpy.py    def <name>(...)    CPU, without Pyccel
    beamtracer/kernels/<name>/<name>_metal.py    def <name>(...)    Apple-silicon GPU (MLX)

All versions take the same arguments: a call runs the CUDA version on CuPy arrays and
the compiled Pyccel version on NumPy arrays (the NumPy version if Pyccel is
unavailable; ``cunumpy.kernels.set_host_kernel_implementation`` chooses).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import cunumpy as xp
from cunumpy.kernels import metal_available

__all__ = ["KERNELS", "KERNEL_NAMES", "METAL_KERNEL_NAMES", "metal_kernel"]

#: The kernels, by name: the staggered electric field at points and the space-charge
#: deposit along the trajectories (:mod:`beamtracer.solver`).
KERNEL_NAMES = ("staggered_field_2d", "deposit_trajectories_2d")

_PACKAGE = __name__.rpartition(".")[0]

#: Every kernel of the beam tracer, by name.
KERNELS = xp.kernels.KernelCatalog(
    {
        name: importlib.import_module(f"{_PACKAGE}.{name}").kernel
        for name in KERNEL_NAMES
    }
)

#: The kernels with a Metal version (``<name>/<name>_metal.py``).
METAL_KERNEL_NAMES = KERNEL_NAMES


def metal_kernel(name: str) -> Callable[..., None]:
    """Return the Metal version of kernel ``name``: a function with the Pyccel arguments.

    Needs MLX and a Metal GPU; the arrays are NumPy arrays, computed in float32 and
    written back into the output arrays.

    Raises:
        KeyError: If the kernel has no Metal version.
        RuntimeError: If there is no usable Metal GPU.
    """
    if name not in METAL_KERNEL_NAMES:
        raise KeyError(f"{name!r} has no Metal version; have {METAL_KERNEL_NAMES}")
    if not metal_available():
        raise RuntimeError("Metal kernels need MLX and a Metal GPU")
    module = importlib.import_module(f"{_PACKAGE}.{name}.{name}_metal")
    return getattr(module, name)
