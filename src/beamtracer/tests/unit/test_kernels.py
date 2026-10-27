"""The ``staggered_field_2d`` and ``deposit_trajectories_2d`` kernels.

Their Pyccel (or Python), NumPy and CUDA versions against each other; the CUDA version runs
on the CPU through cunumpy's emulator (skipped without a C++ compiler). The Metal versions
against the CPU on the GPU of an Apple-silicon Mac: the GPU computes in float32, so they
agree to about 1e-5 of the largest value (skipped without MLX and a Metal GPU).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler
from cunumpy.kernels import metal_available

from beamtracer.kernels.catalog import KERNEL_NAMES, KERNELS

NAMES = ("staggered_field_2d", "deposit_trajectories_2d")
METAL = pytest.mark.skipif(
    not metal_available(), reason="MLX with a Metal GPU is not available"
)


def _cases(name, rng):
    """Arguments of kernel ``name`` and the indices of its output arrays."""
    n = 257
    if name == "staggered_field_2d":
        nx, ny = 9, 7
        fx, fy = rng.normal(size=(nx + 1, ny)), rng.normal(size=(nx, ny + 1))
        # inside, on and beyond the mesh (extrapolated)
        px = rng.uniform(-0.3, 0.3 + 8 * 0.1, n)
        py = rng.uniform(-0.3, 0.3 + 6 * 0.1, n)
        return [px, py, fx, fy, 0.0, 0.0, 0.1, np.zeros(n), np.zeros(n)], (7, 8)
    nx, ny = 9, 7
    px = rng.uniform(-0.2, 1.0, n)
    py = rng.uniform(-0.1, 0.8, n)
    q = rng.normal(size=n)
    return [px, py, q, 0.0, 0.0, 0.1, 0, np.zeros((nx, ny))], (7,)


def _run(implementation, args):
    args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    implementation(*args)
    return args


def test_the_catalog_has_every_kernel():
    assert set(KERNEL_NAMES) == set(NAMES)
    assert all(KERNELS[name] is not None for name in NAMES)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("cylindrical", [0, 1])
def test_host_versions_agree(name, cylindrical):
    args, outputs = _cases(name, np.random.default_rng(4))
    if name == "deposit_trajectories_2d":
        args[6] = cylindrical
    elif cylindrical:
        pytest.skip("only the deposit has a cylindrical variant")
    host = KERNELS[name].host_kernel.kernel
    expected = _run(host.python, args)
    for implementation in (host.get("numpy"), KERNELS[name]):
        result = _run(implementation, args)
        for i in outputs:
            np.testing.assert_allclose(result[i], expected[i], rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
@pytest.mark.parametrize("name", NAMES)
def test_cuda_version_agrees(name):
    args, outputs = _cases(name, np.random.default_rng(5))
    expected = _run(KERNELS[name].host_kernel.kernel.python, args)
    result = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    emulate_cuda_kernel(KERNELS[name].cuda_kernel, *result, n_threads=args[0].shape[-1])
    for i in outputs:
        np.testing.assert_allclose(result[i], expected[i], rtol=1e-13, atol=1e-13)


def test_deposit_conserves_charge():
    args, _ = _cases("deposit_trajectories_2d", np.random.default_rng(6))
    result = _run(KERNELS["deposit_trajectories_2d"], args)
    assert result[7].sum() == pytest.approx(args[2].sum(), abs=1e-12)


def _metal(name):
    module = importlib.import_module(f"beamtracer.kernels.{name}.{name}_metal")
    return getattr(module, name)


def _close(gpu, cpu):
    scale = max(float(np.abs(cpu).max()), 1.0)
    np.testing.assert_allclose(gpu / scale, cpu / scale, rtol=0.0, atol=3e-5)


@METAL
def test_metal_staggered_field_2d():
    rng = np.random.default_rng(3)
    nx, ny, h = 40, 30, 1e-3
    fx = rng.normal(size=(nx + 1, ny))
    fy = rng.normal(size=(nx, ny + 1))
    origin = (0.0123, -0.0045)
    px = origin[0] + h * rng.uniform(-1.0, nx + 1.0, 500)  # some points outside
    py = origin[1] + h * rng.uniform(-1.0, ny + 1.0, 500)
    results = []
    for function in (KERNELS["staggered_field_2d"], _metal("staggered_field_2d")):
        ex, ey = np.zeros(len(px)), np.zeros(len(px))
        function(px, py, fx, fy, origin[0], origin[1], h, ex, ey)
        results.append((ex, ey))
    _close(results[1][0], results[0][0])
    _close(results[1][1], results[0][1])


@METAL
@pytest.mark.parametrize("cylindrical", [0, 1])
def test_metal_deposit_trajectories_2d(cylindrical):
    rng = np.random.default_rng(4)
    n, nx, ny, h = 4000, 30, 20, 1e-3
    origin = (0.02, 0.0 if cylindrical else -0.004)
    px = origin[0] + h * rng.uniform(-1.0, nx, n)  # some beyond the mesh
    py = origin[1] + h * rng.uniform(-1.0 if not cylindrical else 0.0, ny, n)
    q = rng.normal(size=n)
    results = []
    for function in (
        KERNELS["deposit_trajectories_2d"],
        _metal("deposit_trajectories_2d"),
    ):
        charge = np.full((nx, ny), 0.5)
        function(px, py, q, origin[0], origin[1], h, cylindrical, charge)
        results.append(charge)
    _close(results[1], results[0])
