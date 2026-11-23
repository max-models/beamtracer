"""The trace_trajectories_2d kernel of the tracer.

Pyccel and CUDA versions with the same arguments.

Every trajectory traced from start to end in one call: the Runge--Kutta steps through the
staggered electric field, the space-charge deposit, the crossings of the diagnostic planes
and the tests for leaving the mesh or hitting an electrode. ``kernel(...)`` runs the CUDA
version when the arrays are CuPy arrays and otherwise the Pyccel version; without Pyccel,
`beamtracer.solver.BeamTracer` uses its NumPy path instead of this kernel.
"""

import cunumpy as xp

from beamtracer.kernels import KERNEL_PACKAGE

# the CUDA version includes mpiparticles' boris.cuh (the Lorentz factor)
kernel = KERNEL_PACKAGE.from_folder(
    __name__, include_dirs=xp.kernels.source_roots("beamtracer", "mpiparticles")
)
