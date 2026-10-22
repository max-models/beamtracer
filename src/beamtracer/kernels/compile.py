"""Compile the beam tracer's Pyccel kernel modules once and reuse the builds.

The cache is :class:`cunumpy.kernels.PyccelCache`: a kernel module is built with
``pyccel compile`` on its first use and stored under ``BEAMTRACER_KERNEL_CACHE``
(default ``~/.cache/beamtracer/kernels``, or ``<tmp>/beamtracer-kernels`` if that is
not writable). Later processes import the stored extension directly.
``BEAMTRACER_OPENMP=0`` builds the kernels serial; the setting is part of the key.
"""

from __future__ import annotations

import cunumpy as xp

#: The on-disk build cache of the beam tracer's kernels.
KERNEL_CACHE = xp.kernels.PyccelCache("beamtracer")

#: Return the compiled form of a kernel module, building it on first use
#: (:meth:`cunumpy.kernels.PyccelCache.compile`).
compile_cached = KERNEL_CACHE.compile
