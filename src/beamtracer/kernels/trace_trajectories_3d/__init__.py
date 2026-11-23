"""The trace_trajectories_3d kernel of the 3D tracer.

Pyccel and CUDA versions with the same arguments.

Every trajectory traced from start to end in one call: the Runge--Kutta or Boris steps
through the staggered electric field (and a magnetic field on the nodes), the space-charge
deposit, the crossings of the diagnostic planes and the tests for leaving the mesh or
hitting an electrode. ``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays
and otherwise the Pyccel version (or, without Pyccel, its source as Python).
"""

import cunumpy as xp

from beamtracer.kernels import KERNEL_PACKAGE

# the CUDA version includes mpiparticles' boris.cuh (the relativistic Boris block)
kernel = KERNEL_PACKAGE.from_folder(
    __name__, include_dirs=xp.kernels.source_roots("beamtracer", "mpiparticles")
)
