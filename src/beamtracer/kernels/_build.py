"""The options every kernel folder passes to :meth:`cunumpy.kernels.Kernel.from_folder`.

The ``__init__.py`` of a kernel folder ``beamtracer/kernels/<name>/`` declares its
kernel with these::

    kernel = xp.kernels.Kernel.from_folder(__name__, **KERNEL_OPTIONS)

which finds ``<name>_pyccel.py``, ``<name>_numpy.py`` and ``<name>_cuda.cu`` in the
folder. Nothing is compiled here: the Pyccel version is compiled on its first host
call (cached on disk by :data:`~beamtracer.kernels.compile.KERNEL_CACHE`), the CUDA
version on its first launch. Importing this module needs neither Pyccel nor CuPy.
"""

from __future__ import annotations

from typing import Any

#: CUDA threads per block of every kernel.
BLOCK_SIZE = 256


def _compile(module: Any) -> Any:
    from beamtracer.kernels.compile import compile_cached

    return compile_cached(module)


#: ``<name>_pyccel.py`` compiled with Pyccel on first use; CUDA for CuPy arguments,
#: the host version for NumPy ones; without ``n_threads`` one CUDA thread per entry
#: of the last axis of the first array (both kernels take per-point rows first).
KERNEL_OPTIONS: dict[str, Any] = {
    "host_suffix": "_pyccel",
    "compile_host": _compile,
    "dispatch": "arrays",
    "block_size": BLOCK_SIZE,
    "n_threads_from": "last_axis",
}
