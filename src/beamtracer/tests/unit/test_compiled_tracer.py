"""The compiled tracer (the ``trace_trajectories_2d`` kernel) against the NumPy one.

Both take the same Runge--Kutta steps, so they agree to rounding: the same statuses, the
same crossings in the same order and the same charge densities. The kernel's own versions
(Pyccel, uncompiled Python, CUDA on the CPU emulator) agree with each other.
"""

from __future__ import annotations

import numpy as np
import pytest
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
)
from beamtracer.kernels.trace_trajectories_2d import kernel
from beamtracer.sections import ObjectParams
from beamtracer.solver import (
    AxisymmetricMagneticField,
    BeamParticles,
    BeamTracer,
    FaceCondition,
    IonOpticsMesh,
    StaggeredElectricField,
)
from beamtracer.tests.unit.test_beam_optics import _extraction

pytestmark = pytest.mark.skipif(
    not BeamTracer.compiled_available(), reason="Pyccel is not available"
)


def _planar(**settings) -> BeamOpticsConfig:
    """IBSimu's vlasov2d: protons from a slit through three electrodes, coarse."""
    neumann = FieldBoundaryCondition.neumann(0.0)
    return BeamOpticsConfig(
        grid=GridParams(num_gridpoints=(121, 51), upper_bound=(0.12, 0.05)),
        objects={
            "plasma": ObjectParams(
                type="rectangle",
                midpoint=(0.01, 0.034),
                side_lengths=(0.02, 0.032),
                bc_type="dirichlet",
                bc_value=-3000.0,
            ),
            "puller": ObjectParams(
                type="rectangle",
                midpoint=(0.035, 0.035),
                side_lengths=(0.01, 0.03),
                bc_type="dirichlet",
                bc_value=-14000.0,
            ),
            "ground": ObjectParams(
                type="polygon",
                vertices=(
                    (0.06, 0.04),
                    (0.08, 0.03),
                    (0.12, 0.03),
                    (0.12, 0.05),
                    (0.06, 0.05),
                ),
                bc_type="dirichlet",
                bc_value=-1000.0,
            ),
        },  # fmt: skip
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(-3000.0),
            x_upper=FieldBoundaryCondition.dirichlet(-1000.0),
            y_lower=neumann,
            y_upper=neumann,
        ),
        beams=[
            BeamSpecies(
                "H+",
                50.0,
                (0.0, 0.0),
                (0.0, 0.012),
                mass=1.0,
                energy=3000.0,
                transverse_temperature=0.0,
                num_particles=300,
            )
        ],  # fmt: skip
        iterations=3,
        tolerance=0.0,
        diagnostic_planes=[0.05, 0.115],
        **settings,
    )


def _run(config: BeamOpticsConfig, compiled: bool) -> BeamOptics:
    model = BeamOptics(config)
    model.tracer.compiled = compiled
    return model.run()


def _assert_same(a: BeamOptics, b: BeamOptics) -> None:
    ta, tb = a.state.trace, b.state.trace
    np.testing.assert_array_equal(ta.status, tb.status)
    np.testing.assert_array_equal(ta.electrode, tb.electrode)
    assert ta.steps == tb.steps
    scale = np.abs(a.state.charge_density).max()
    np.testing.assert_allclose(
        a.state.charge_density, b.state.charge_density, rtol=0, atol=1e-10 * scale
    )
    np.testing.assert_allclose(a.potential, b.potential, rtol=0, atol=1e-8)
    for pa, pb in zip(ta.planes, tb.planes, strict=True):
        np.testing.assert_allclose(pa.point, pb.point, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(pa.velocity, pb.velocity, rtol=1e-10, atol=1e-6)
        np.testing.assert_array_equal(pa.current, pb.current)
        np.testing.assert_array_equal(pa.species, pb.species)
        np.testing.assert_array_equal(pa.direction, pb.direction)


def test_planar_extraction_agrees_with_the_numpy_tracer() -> None:
    compiled, numpy = _run(_planar(), True), _run(_planar(), False)
    _assert_same(compiled, numpy)
    stats = compiled.stats
    assert stats["electrode"] > 0 and stats["transmitted"] > 0


def _cylindrical() -> BeamOpticsConfig:
    config = _extraction(1.0)
    config.iterations = 4
    return config


def test_cylindrical_extraction_agrees_with_the_numpy_tracer() -> None:
    _assert_same(_run(_cylindrical(), True), _run(_cylindrical(), False))


def test_kept_trajectories_take_the_numpy_path() -> None:
    compiled = _run(_planar(save_trajectories=7), True)
    numpy = _run(_planar(save_trajectories=7), False)
    np.testing.assert_array_equal(compiled.state.trace.status, numpy.state.trace.status)
    np.testing.assert_allclose(
        compiled.state.charge_density, numpy.state.charge_density, rtol=1e-9, atol=1e-12
    )
    assert len(compiled.trajectories) == len(numpy.trajectories) > 0
    for a, b in zip(compiled.trajectories, numpy.trajectories, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)
    for pa, pb in zip(
        compiled.state.trace.planes, numpy.state.trace.planes, strict=True
    ):
        assert pa.current.size == pb.current.size
        assert pa.current.sum() == pytest.approx(pb.current.sum(), rel=1e-12)


def _well(threads: int, monkeypatch) -> tuple:
    """Protons oscillating in a harmonic well across the plane at its centre."""
    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    mesh = IonOpticsMesh(
        shape=(41, 11), origin=(0.0, 0.0), h=1e-3, geometry="planar",
        faces={"x_min": FaceCondition("dirichlet", 0.0)},
    )  # fmt: skip
    xn, _ = mesh.nodes()
    potential = 2e5 * (xn - 0.02) ** 2
    tracer = BeamTracer(
        mesh, StaggeredElectricField(mesh, potential), planes=[0.02], max_steps=3000
    )
    rng = np.random.default_rng(1)
    n = 50
    position = np.stack(
        [rng.uniform(0.012, 0.028, n), rng.uniform(0.001, 0.009, n), np.zeros(n)]
    )
    particles = BeamParticles(
        position, np.zeros((3, n)), np.full(n, 1.602e-19), np.full(n, 1.67e-27),
        np.full(n, 1e-3), np.zeros(n, dtype=np.int32),
    )  # fmt: skip
    return tracer, particles


def test_a_full_plane_buffer_is_retraced_larger(monkeypatch) -> None:
    tracer, particles = _well(2, monkeypatch)
    compiled = tracer.trace(particles)
    tracer.compiled = False
    numpy = tracer.trace(particles)
    assert (compiled.status == 4).all()  # oscillating until max_steps
    hits = compiled.planes[0]
    assert hits.current.size > 10 * len(particles)  # many crossings per trajectory
    np.testing.assert_allclose(
        hits.point, numpy.planes[0].point, rtol=1e-10, atol=1e-14
    )
    np.testing.assert_array_equal(hits.direction, numpy.planes[0].direction)
    np.testing.assert_allclose(
        compiled.charge_density, numpy.charge_density, rtol=1e-9, atol=1e-6
    )


@pytest.mark.parametrize("threads", [1, 3, 8])
def test_threads_agree(threads, monkeypatch) -> None:
    tracer, particles = _well(1, monkeypatch)
    expected = tracer.trace(particles)
    tracer, particles = _well(threads, monkeypatch)
    result = tracer.trace(particles)
    np.testing.assert_allclose(
        result.charge_density, expected.charge_density, rtol=1e-12, atol=1e-6
    )
    np.testing.assert_array_equal(result.planes[0].point, expected.planes[0].point)


def test_compiled_needs_the_solved_field_alone() -> None:
    mesh = IonOpticsMesh(shape=(11, 6), origin=(0.0, 0.0), h=1e-3)
    field = StaggeredElectricField(mesh, np.zeros(mesh.shape))
    magnet = AxisymmetricMagneticField(0.0, 1e-3, np.full(11, 0.1))
    tracer = BeamTracer(mesh, field, magnet, compiled=True)
    particles = BeamParticles(
        np.zeros((3, 1)), np.array([[1e5], [0.0], [0.0]]), np.ones(1), np.ones(1),
        np.ones(1), np.zeros(1, dtype=np.int32),
    )  # fmt: skip
    with pytest.raises(ValueError, match="magnetic"):
        tracer.trace(particles)
    tracer.compiled = None  # falls back to NumPy
    assert tracer.trace(particles).status[0] == 0


def _kernel_case(cylindrical: int) -> list:
    rng = np.random.default_rng(7)
    n, nx, ny, h = 40, 30, 12, 1e-3
    fx = rng.normal(size=(nx + 1, ny)) * 1e4 + 3e5
    fy = rng.normal(size=(nx, ny + 1)) * 1e4
    x = np.stack([rng.uniform(0, 3e-3, n), rng.uniform(0, 8e-3, n), np.zeros(n)])
    v = np.stack(
        [
            np.full(n, 3e5),
            rng.normal(size=n) * 1e4,
            rng.normal(size=n) * 1e4 * cylindrical,
        ]
    )
    cells = np.full((nx - 1, ny - 1), -1, dtype=np.int64)
    cells[20, :] = np.arange(ny - 1)  # a column of cells near an electrode
    subcells = rng.choice(np.array([-2, -1, 0], dtype=np.int64), size=(ny - 1) * 64)
    return [
        np.arange(n, dtype=np.int64), x, v, np.full(n, 9.6e7), np.ones(n),
        np.zeros(n, dtype=np.int64), np.full(n, -1, dtype=np.int64),
        np.full(n, -1, dtype=np.int64), fx, fy, 0.0, 0.0, h, (nx - 1) * h,
        (ny - 1) * h, cylindrical, 0.5, 400, cells, subcells, 8,
        np.array([5e-3, 15e-3]), np.zeros((n, 2, 3, 8)), np.zeros((n, 2), dtype=np.int64),
        2, np.zeros((2, nx, ny)),
    ]  # fmt: skip


def _call(function, args) -> list:
    args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    function(*args)
    return args


OUTPUTS = (1, 2, 5, 6, 7, 22, 23)


@pytest.mark.parametrize("cylindrical", [0, 1])
def test_kernel_versions_agree(cylindrical) -> None:
    args = _kernel_case(cylindrical)
    expected = _call(kernel.host_kernel.kernel.python, args)
    result = _call(kernel, args)
    assert set(np.unique(expected[6])) >= {-2, 0}  # stopped at surfaces and transmitted
    for i in OUTPUTS:
        np.testing.assert_allclose(result[i], expected[i], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result[25].sum(0), expected[25].sum(0), rtol=1e-12)


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
@pytest.mark.parametrize("cylindrical", [0, 1])
def test_cuda_version_agrees(cylindrical) -> None:
    args = _kernel_case(cylindrical)
    expected = _call(kernel.host_kernel.kernel.python, args)
    result = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    result[25] = np.zeros_like(result[25][:1])  # the GPU deposits into one slab
    emulate_cuda_kernel(kernel.cuda_kernel, *result, n_threads=args[0].shape[-1])
    for i in OUTPUTS:
        np.testing.assert_allclose(result[i], expected[i], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result[25][0], expected[25].sum(0), rtol=1e-10, atol=1e-18
    )
