"""Tests for the 3D tracer and its compiled kernel."""

from __future__ import annotations

import numpy as np
import pytest
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler
from scipy.constants import e, m_p

from beamtracer.kernels.trace_trajectories_3d import kernel
from beamtracer.solver.fields3d import StaggeredElectricField3D
from beamtracer.solver.mesh3d import Electrode3D, IonOpticsMesh3D
from beamtracer.solver.tracer import BeamParticles
from beamtracer.solver.tracer3d import BeamTracer3D


def _protons(position: np.ndarray, velocity: np.ndarray, current: float = 1e-3):
    n = position.shape[1]
    return BeamParticles(
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        np.full(n, e),
        np.full(n, m_p),
        np.full(n, current),
        np.zeros(n, dtype=np.int32),
    )


def _kernel_case(
    integrator: int, magnetic: bool, mirror: bool, refined: bool = False
) -> list:
    """Random fields on a small mesh (and, ``refined``, a box of half the spacing)."""
    rng = np.random.default_rng(3)
    n, (nx, ny, nz), h = 30, (16, 9, 8), 1e-3
    # (shape, origin, spacing, box) of every patch, finest first, the mesh last
    shapes = [((nx, ny, nz), (0.0, 0.0, 0.0), h)]
    if refined:
        shapes.insert(0, ((13, 9, 7), (4e-3, 0.0, 0.0), h / 2))
    fxs, fys, fzs, patch_int, patch_real = [], [], [], [], []
    offsets = [0, 0, 0, 0]
    for (px, py, pz), origin, ph in shapes:
        fxs.append((rng.normal(size=(px + 1, py, pz)) * 1e4 + 3e5).ravel())
        fys.append((rng.normal(size=(px, py + 1, pz)) * 1e4).ravel())
        fzs.append((rng.normal(size=(px, py, pz + 1)) * 1e4).ravel())
        patch_int.append((px, py, pz, *offsets[:3], px * py * pz, offsets[3]))
        upper = tuple(origin[a] + ((px, py, pz)[a] - 1) * ph for a in range(3))
        patch_real.append((*origin, ph, *origin, *upper))
        offsets = [offsets[0] + fxs[-1].size, offsets[1] + fys[-1].size,
                   offsets[2] + fzs[-1].size, offsets[3] + px * py * pz]  # fmt: skip
    bn = rng.normal(size=(3, nx, ny, nz)) * 0.05 if magnetic else np.zeros((3, 2, 2, 2))
    x = np.stack(
        [rng.uniform(0, 3e-3, n), rng.uniform(0, 3e-3, n), rng.uniform(0, 3e-3, n)]
    )
    v = np.stack(
        [np.full(n, 3e5), -np.abs(rng.normal(size=n)) * 3e4, rng.normal(size=n) * 3e4]
    )
    cells = np.full((nx - 1, ny - 1, nz - 1), -1, dtype=np.int64)
    cells[8, :, :] = np.arange((ny - 1) * (nz - 1)).reshape(ny - 1, nz - 1)
    subcells = rng.choice(
        np.array([-2, -1, 0], dtype=np.int64),
        size=(ny - 1) * (nz - 1) * 64,
        p=[0.1, 0.8, 0.1],
    )
    slot = np.full(n, -1, dtype=np.int64)
    slot[::3] = np.arange(len(slot[::3]))
    return [
        np.arange(n, dtype=np.int64), x, v, np.full(n, 9.6e7), np.ones(n),
        np.zeros(n, dtype=np.int64), np.full(n, -1, dtype=np.int64),
        np.full(n, -1, dtype=np.int64), np.concatenate(fxs), np.concatenate(fys),
        np.concatenate(fzs), np.array(patch_int, dtype=np.int64), np.array(patch_real),
        bn, int(magnetic),
        0.0, 0.0, 0.0, h, (nx - 1) * h, (ny - 1) * h, (nz - 1) * h,
        int(mirror), int(mirror), integrator, 0.5, 700, cells, subcells, 4,
        np.array([5e-3, 11e-3]), np.zeros((n, 2, 3, 8)), np.zeros((n, 2), dtype=np.int64),
        slot, np.zeros((slot.max() + 1, 10, 3)), np.zeros(slot.max() + 1, dtype=np.int64),
        2, np.zeros((2, offsets[3])),
    ]  # fmt: skip


def _call(function, args) -> list:
    args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    function(*args)
    return args


OUTPUTS = (1, 2, 5, 6, 7, 31, 32, 34, 35)
CHARGE = 37


@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("integrator", [0, 1])
def test_kernel_versions_agree(integrator, magnetic, mirror, refined) -> None:
    args = _kernel_case(integrator, magnetic, mirror, refined)
    expected = _call(kernel.host_kernel.kernel.python, args)
    result = _call(kernel, args)
    assert set(np.unique(expected[6])) >= {-2, 0}  # stopped at surfaces and transmitted
    assert expected[35].max() > 10  # some paths outgrow their buffer
    if not mirror:  # trajectories heading for the lower y face leave through it
        assert (expected[6] == 2).any()
    for i in OUTPUTS:
        np.testing.assert_allclose(result[i], expected[i], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result[CHARGE].sum(0), expected[CHARGE].sum(0), rtol=1e-10, atol=1e-20
    )


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("integrator", [0, 1])
def test_cuda_version_agrees(integrator, magnetic, mirror, refined) -> None:
    args = _kernel_case(integrator, magnetic, mirror, refined)
    expected = _call(kernel.host_kernel.kernel.python, args)
    result = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    result[CHARGE] = np.zeros_like(result[CHARGE][:1])  # the GPU deposits into one slab
    emulate_cuda_kernel(kernel.cuda_kernel, *result, n_threads=args[0].shape[-1])
    for i in OUTPUTS:
        np.testing.assert_allclose(result[i], expected[i], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        result[CHARGE][0], expected[CHARGE].sum(0), rtol=1e-10, atol=1e-20
    )


def _uniform(field_x: float, *, bz: float = 0.0, shape=(21, 11, 11), h=1e-3):
    mesh = IonOpticsMesh3D(shape, (0.0, -5e-3, -5e-3), h)
    x, _, _ = mesh.nodes()
    magnetic = None
    if bz:
        magnetic = np.zeros((3, *mesh.shape))
        magnetic[2] = bz
    return mesh, StaggeredElectricField3D(mesh, -field_x * x), magnetic


@pytest.mark.parametrize("integrator", ["rk4", "boris"])
def test_a_uniform_field_accelerates_and_deposits_the_flight_time(integrator) -> None:
    mesh, field, _ = _uniform(1e5)
    tracer = BeamTracer3D(mesh, field, planes=[0.015], integrator=integrator)
    n, v0, current = 40, 1e5, 1e-3
    position = np.stack([np.zeros(n), np.linspace(-2e-3, 2e-3, n), np.zeros(n)])
    velocity = np.stack([np.full(n, v0), np.zeros(n), np.zeros(n)])
    result = tracer.trace(_protons(position, velocity, current))
    assert np.all(result.status == 0)
    a = e / m_p * 1e5
    np.testing.assert_allclose(
        result.planes[0].velocity[0], np.sqrt(v0**2 + 2 * a * 0.015), rtol=1e-4
    )
    # the node charges (the densities times the node volumes, halved on faces) are the
    # current times the time of flight through the mesh, and up to one step beyond it
    # (the last step deposits all of its time)
    volume = np.full(mesh.shape, mesh.h**3)
    for axis in range(3):
        index = [slice(None)] * 3
        for end in (0, -1):
            index[axis] = end
            volume[tuple(index)] /= 2.0
    flight = (-v0 + np.sqrt(v0**2 + 2 * a * 0.02)) / a
    last_step = 0.5 * mesh.h / np.sqrt(v0**2 + 2 * a * 0.02)
    time = float(np.sum(result.charge_density * volume)) / (n * current)
    assert flight * (1 - 1e-6) < time < flight + last_step


def test_boris_keeps_the_speed_in_a_magnetic_field_and_follows_rk4() -> None:
    mesh, field, magnetic = _uniform(0.0, bz=0.5, shape=(41, 41, 11), h=1e-3)
    mesh = IonOpticsMesh3D((41, 41, 11), (0.0, -20e-3, -5e-3), 1e-3)
    field = StaggeredElectricField3D(mesh, np.zeros(mesh.shape))
    magnetic = np.zeros((3, *mesh.shape))
    magnetic[2] = 0.5
    speed = 1e6  # Larmor radius 21 mm
    position = np.array([[1e-3], [0.0], [0.0]])
    velocity = np.array([[speed], [0.0], [0.0]])
    final = {}
    for integrator in ("rk4", "boris"):
        tracer = BeamTracer3D(
            mesh, field, magnetic, integrator=integrator, save_trajectories=1,
            step_fraction=0.2,
        )  # fmt: skip
        result = tracer.trace(_protons(position, velocity))
        path = result.trajectories[0]
        # on the circle of the Larmor radius about (1 mm, -r)
        radius = m_p * speed / (e * 0.5)
        distance = np.hypot(path[:, 0] - 1e-3, path[:, 1] + radius)
        np.testing.assert_allclose(distance, radius, rtol=2e-3)
        final[integrator] = path[-1]
    np.testing.assert_allclose(final["boris"], final["rk4"], atol=2e-5)


def test_trajectories_stop_on_an_electrode() -> None:
    mesh = IonOpticsMesh3D(
        (21, 11, 11), (0.0, -5e-3, -5e-3), 1e-3,
        [Electrode3D(lambda x, y, z: (x - 10e-3) ** 2 + y**2 + z**2 < (2.5e-3) ** 2, 0.0)],
    )  # fmt: skip
    field = StaggeredElectricField3D(mesh, np.zeros(mesh.shape))
    tracer = BeamTracer3D(mesh, field)
    offsets = np.linspace(-4e-3, 4e-3, 17)
    position = np.stack([np.zeros(17), offsets, np.full(17, 0.3e-3)])
    velocity = np.stack([np.full(17, 1e5), np.zeros(17), np.zeros(17)])
    result = tracer.trace(_protons(position, velocity))
    hits = np.hypot(offsets, 0.3e-3) < 2.5e-3
    np.testing.assert_array_equal(result.status == 3, hits)
    np.testing.assert_array_equal(result.status[~hits], 0)
    np.testing.assert_array_equal(result.electrode[hits], 0)


def test_a_quarter_with_symmetry_planes_is_the_mirrored_whole() -> None:
    """A focusing field symmetric in y and z: the quarter domain traces the whole."""
    h = 0.5e-3
    whole = IonOpticsMesh3D((41, 21, 21), (0.0, -5e-3, -5e-3), h)
    quarter = IonOpticsMesh3D(
        (41, 11, 11), (0.0, 0.0, 0.0), h, symmetry_planes=("y_min", "z_min")
    )

    def potential(mesh):
        x, y, z = mesh.nodes()
        return -1e5 * x + 4e7 * (y**2 + 0.5 * z**2) * np.sin(np.pi * x / 0.02) ** 2

    n = 12
    rng = np.random.default_rng(0)
    position = np.stack(
        [np.zeros(n), rng.uniform(0.5e-3, 3e-3, n), rng.uniform(0.5e-3, 3e-3, n)]
    )
    velocity = np.stack([np.full(n, 2e5), np.zeros(n), np.zeros(n)])
    ends = {}
    for name, mesh in (("whole", whole), ("quarter", quarter)):
        tracer = BeamTracer3D(
            mesh, StaggeredElectricField3D(mesh, potential(mesh)), planes=[0.0195]
        )
        result = tracer.trace(_protons(position, velocity))
        plane = result.planes[0]
        assert np.all(result.status == 0)
        ends[name] = (plane.point, plane.velocity)
    crossed = ends["quarter"][0][1] < 0.0
    assert crossed.any()  # the field focuses them across the planes
    np.testing.assert_allclose(ends["quarter"][0], ends["whole"][0], atol=1e-12)
    np.testing.assert_allclose(
        ends["quarter"][1], ends["whole"][1], rtol=1e-9, atol=1e-6
    )


def test_the_tracer_checks_its_settings() -> None:
    mesh = IonOpticsMesh3D((5, 5, 5), (0, 0, 0), 1.0)
    with pytest.raises(ValueError, match="step_fraction"):
        BeamTracer3D(mesh, step_fraction=0.0)
    with pytest.raises(ValueError, match="integrator"):
        BeamTracer3D(mesh, integrator="euler")
    with pytest.raises(ValueError, match="magnetic"):
        BeamTracer3D(mesh, magnetic=np.zeros((3, 2, 2, 2)))
    with pytest.raises(RuntimeError, match="no electric field"):
        BeamTracer3D(mesh).trace(_protons(np.zeros((3, 1)), np.ones((3, 1))))
