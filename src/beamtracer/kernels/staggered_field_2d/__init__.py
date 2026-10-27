"""The staggered_field_2d kernel of the tracer.

Pyccel, NumPy, CUDA and Metal versions with the same arguments.

The electric field held on the edges between nodes, interpolated bilinearly at points.

``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays, and otherwise the
Pyccel version, or the NumPy version if Pyccel is unavailable (see
``cunumpy.kernels.set_host_kernel_implementation`` to choose).
"""

from beamtracer.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
