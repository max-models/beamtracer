"""The deposit_trajectories_2d kernel of the tracer.

Pyccel, NumPy, CUDA and Metal versions with the same arguments.

The space charge of the trajectories, deposited at the midpoint of every step with an
area-weighted linear shape: bilinear in planar geometry, linear in ``r**2`` along the radius
in cylindrical geometry, so a uniform density gives node charges proportional to the ring
volumes.

``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays, and otherwise the
Pyccel version, or the NumPy version if Pyccel is unavailable (see
``cunumpy.kernels.set_host_kernel_implementation`` to choose).
"""

from beamtracer.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
