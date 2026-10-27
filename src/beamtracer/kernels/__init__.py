"""The beam tracer's compiled kernels, one folder per kernel (see :mod:`beamtracer.kernels.catalog`).

Every kernel folder builds its kernel with :data:`KERNEL_PACKAGE`::

    kernel = KERNEL_PACKAGE.from_folder(__name__)

with the defaults of :class:`cunumpy.kernels.KernelPackage`: the Pyccel version is
compiled on its first host call and kept under ``BEAMTRACER_KERNEL_CACHE`` (default
``~/.cache/beamtracer/kernels``; ``BEAMTRACER_OPENMP=0`` builds serial), the CUDA version runs
for CuPy arguments and the host version for NumPy ones, one CUDA thread per entry of
the last axis of the first array. Importing a kernel needs neither Pyccel nor CuPy.

The Metal versions (``<name>/<name>_metal.py``) wrap a kernel built by
``KERNEL_PACKAGE.metal_kernel`` in a function with the arguments of the Pyccel
version. The Apple GPU has no float64, so they compute in float32: expect a relative
error of about 1e-5 against the CPU versions. Import one only where
:func:`cunumpy.kernels.metal_available` is true.
"""

import cunumpy as xp

#: The build cache and launch options of the beam tracer's kernels.
KERNEL_PACKAGE = xp.kernels.KernelPackage("beamtracer")
