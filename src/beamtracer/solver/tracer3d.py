"""The 3D beam tracer: trajectories through the fields of an `IonOpticsMesh3D`.

The 3D counterpart of `BeamTracer`, built on the ``trace_trajectories_3d`` kernel (compiled
with Pyccel, with a CUDA version; without Pyccel its source runs as Python). Every
trajectory is traced from start to end: Runge--Kutta or Boris steps through the staggered
electric field and a magnetic field on the nodes, the space charge deposited at each step's
Hermite midpoint, the crossings of the diagnostic planes ``x = const`` recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scope_profiler
from methodcard import method
from scipy import constants

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.kernels.trace_trajectories_3d import kernel as trace_trajectories_3d
from beamtracer.solver.fields3d import StaggeredElectricField3D
from beamtracer.solver.mesh3d import IonOpticsMesh3D
from beamtracer.solver.tracer import (
    _ALIVE,
    _PAUSED,
    ELECTRODE,
    INTEGRATORS,
    BeamParticles,
    PlaneHits,
    TraceResult,
    _gaussian_samples,
    _num_threads,
    share_of,
)

#: Sub-cells per cell edge of the 3D electrode lookup (``SUBCELLS**3`` per cell).
SUBCELLS_3D = 4


@method(
    equations={
        "rectangle": r"(y, z) = \mathbf c + \left(\left(i + \tfrac12\right) \frac{a}{n_y} - \frac a2,\ \left(j + \tfrac12\right) \frac{b}{n_z} - \frac b2\right), \qquad n_y \approx \sqrt{N a / b},\ n_z \approx N / n_y",
        "disc": r"r_k = R \sqrt{\frac{k + 1/2}{N}}, \qquad \theta_k = k\, \pi (3 - \sqrt 5)",
        "velocity": r"\mathbf v = (v_0 + \sigma_\parallel g_2,\ \sigma_\perp g_1,\ \sigma_\perp g_3), \qquad v_0 = \sqrt{\frac{2 E e}{m}}, \quad \sigma = \sqrt{\frac{T e}{m}}",
        "current": r"I_k = \frac{J A}{N}",
    },
    assumptions=(
        r"the emitting surface is perpendicular to $x$; every trajectory stands for an equal share of its area",
        r"$g_{1,2,3}$ an unscrambled 3D Sobol sequence through the inverse normal: the same beam on every call",
    ),
)
def beam_on_surface(
    num_particles: int,
    current_density: float,
    charge: float,
    mass: float,
    energy: float,
    parallel_temperature: float,
    transverse_temperature: float,
    centre: Sequence[float],
    size: Sequence[float],
    shape: str = "rectangle",
    *,
    species: int = 0,
) -> tuple[BeamParticles, float]:
    """Seed a beam emitted from a surface perpendicular to ``x``; return it and ``J / v``.

    ``shape`` ``"rectangle"`` of side lengths ``size`` ``(along y, along z)`` [m], the
    trajectories at the centres of a grid of about ``num_particles`` equal cells, or
    ``"disc"`` of radius ``size[0]``, the trajectories on a sunflower pattern of equal
    areas, centred at ``centre`` ``(x, y, z)`` [m]. They move along ``+x`` at ``energy``
    [eV] with Gaussian spreads [eV]; ``current_density`` [A/m^2], ``charge`` [e], ``mass``
    [u]. Every trajectory carries ``J A / N``.
    """
    q = charge * constants.e
    m = mass * constants.atomic_mass
    v = np.sqrt(2.0 * energy * constants.e / m)
    dvp = np.sqrt(parallel_temperature * constants.e / m)
    dvt = np.sqrt(transverse_temperature * constants.e / m)
    cx, cy, cz = (float(c) for c in centre)
    if shape == "rectangle":
        a, b = (float(s) for s in size)
        ny = max(1, round(np.sqrt(num_particles * a / b)))
        nz = max(1, round(num_particles / ny))
        j, k = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
        py = cy - 0.5 * a + (j.ravel() + 0.5) * a / ny
        pz = cz - 0.5 * b + (k.ravel() + 0.5) * b / nz
        area = a * b
    elif shape == "disc":
        radius = float(size[0])
        k = np.arange(num_particles)
        r = radius * np.sqrt((k + 0.5) / num_particles)
        theta = k * np.pi * (3.0 - np.sqrt(5.0))
        py, pz = cy + r * np.cos(theta), cz + r * np.sin(theta)
        area = np.pi * radius**2
    else:
        raise ValueError(f"shape must be 'rectangle' or 'disc', got {shape!r}")
    n = py.size
    g = _gaussian_samples(n, "sobol")
    particles = BeamParticles(
        position=np.stack([np.full(n, cx), py, pz]),
        velocity=np.stack([v + dvp * g[:, 1], dvt * g[:, 0], dvt * g[:, 2]]),
        charge=np.full(n, q),
        mass=np.full(n, m),
        current=np.full(n, current_density * area / n),
        species=np.full(n, species, dtype=np.int32),
    )
    return particles, current_density / v


@method(
    equations={
        "motion": r"\frac{d\mathbf x}{dt} = \mathbf v, \qquad \frac{d\mathbf v}{dt} = \frac{q}{m}\left(\mathbf E + \mathbf v \times \mathbf B\right)",
        "step": r"\Delta t = f \min\left(\frac{h}{|\mathbf v|}, \sqrt{\frac{2h}{|\mathbf a|}}\right)",
        "boris": r"\mathbf v^{\pm} = \mathbf v \pm \tfrac{q}{m} \mathbf E \tfrac{\tau}{2}, \quad \mathbf t = \tfrac{q}{m} \mathbf B \tfrac{\tau}{2}, \quad \mathbf v' = \mathbf v^- + \mathbf v^- \times \mathbf t, \quad \mathbf v^+ = \mathbf v^- + \frac{2\, \mathbf v' \times \mathbf t}{1 + t^2}, \qquad \tau = \Delta t / 2",
        "midpoint": r"\mathbf x_{1/2} = \tfrac12 (\mathbf x + \mathbf x') + \frac{\Delta t}{8} (\mathbf v - \mathbf v'), \qquad \Delta Q = \operatorname{sign}(q)\, I\, \Delta t",
        "density": r"\rho_{ijk} = \frac{Q_{ijk}}{h^3} \times 2 \text{ per face the node lies on (half cells)}",
        "mirror": r"\tilde y = y_0 + |y - y_0|: \quad E_y \to s E_y, \ (B_x, B_z) \to s (B_x, B_z), \qquad s = \operatorname{sign}(y - y_0) \text{ (and alike for } z)",
    },
    assumptions=(
        r"non-relativistic; $\mathbf E$ the staggered field interpolated trilinearly, $\mathbf B$ given on the nodes, interpolated trilinearly",
        r"Boris: two half steps of $\tau = \Delta t/2$, each a kick, a rotation and a kick, around the drift $\mathbf x' = \mathbf x + \Delta t\, \mathbf v_{1/2}$ (one field evaluation a step)",
        r"a trajectory ends transmitted ($x \ge x_\mathrm{max}$), backward, on the wall, on an electrode (looked up in sub-cells of $h/4$ sampled at their corners and centre, the electrode's own test where a surface cuts one) or after `max_steps`",
        r"a symmetry plane mirrors the field and the deposit; trajectories crossing it continue",
    ),
)
class BeamTracer3D:
    """Trace beam particles through the fields of a 3D mesh.

    ``electric`` is a `StaggeredElectricField3D` (set by the Vlasov iteration),
    ``magnetic`` an optional ``(3, nx, ny, nz)`` array of ``B`` [T] on the mesh's nodes.
    ``planes`` are the diagnostic planes ``x = const`` [m]; ``save_trajectories`` keeps every
    n-th trajectory (0: none); ``integrator`` is ``"rk4"`` or ``"boris"``.
    """

    def __init__(
        self,
        mesh: IonOpticsMesh3D,
        electric: StaggeredElectricField3D | None = None,
        magnetic: np.ndarray | None = None,
        *,
        step_fraction: float = 0.5,
        max_steps: int = 100_000,
        planes: Sequence[float] = (),
        save_trajectories: int = 0,
        integrator: str = "rk4",
    ) -> None:
        """Set up the tracer."""
        if not 0.0 < step_fraction <= 1.0:
            raise ValueError(f"step_fraction must be in (0, 1], got {step_fraction}")
        if integrator not in INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {INTEGRATORS}, got {integrator!r}"
            )
        if magnetic is not None and np.shape(magnetic) != (3, *mesh.shape):
            raise ValueError(
                f"magnetic must be (3, *mesh.shape) = {(3, *mesh.shape)}, "
                f"got {np.shape(magnetic)}"
            )
        self.mesh = mesh
        self.electric = electric
        self.magnetic = None if magnetic is None else np.ascontiguousarray(magnetic)
        self.step_fraction = float(step_fraction)
        self.max_steps = int(max_steps)
        self.planes = tuple(float(p) for p in planes)
        self.save_trajectories = int(save_trajectories)
        self.integrator = integrator
        self._lookup: tuple[np.ndarray, np.ndarray] | None = None
        #: Buffer sizes: crossings per plane and trajectory, points per kept trajectory;
        #: grown when a trace needs more.
        self._capacity = (4, 1024)
        #: Sums the node charges over MPI ranks that each trace a share of the beam.
        self.reduce: Callable[[np.ndarray], np.ndarray] | None = None
        #: With MPI, this rank and the number of ranks, and a function returning the
        #: arrays of all ranks in rank order: the ranks share the electrode lookup.
        self.ranks: tuple[int, int] = (0, 1)
        self.allgather: Callable[[np.ndarray], list[np.ndarray]] | None = None
        touch = (mesh.solid >= 0) | mesh.near_solid
        cells = np.zeros(tuple(n - 1 for n in mesh.shape), dtype=bool)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    cells |= touch[
                        di : di + cells.shape[0],
                        dj : dj + cells.shape[1],
                        dk : dk + cells.shape[2],
                    ]
        # cells with an electrode node or a node next to one: the electrodes are tested there
        self._electrode_cells = cells

    # -- geometry ---------------------------------------------------------------------------

    def _mesh_points(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The mesh coordinates of Cartesian points: mirrored onto a symmetry plane's side."""
        mesh = self.mesh
        px, py, pz = (np.asarray(x[a], dtype=float) for a in range(3))
        if "y_min" in mesh.symmetry_planes:
            py = mesh.origin[1] + np.abs(py - mesh.origin[1])
        if "z_min" in mesh.symmetry_planes:
            pz = mesh.origin[2] + np.abs(pz - mesh.origin[2])
        return px, py, pz

    def _subcells(self, ci: np.ndarray, cj: np.ndarray, ck: np.ndarray) -> np.ndarray:
        """The electrode of every sub-cell of cells ``(ci, cj, ck)``: ``(len(ci), S, S, S)``.

        The electrode containing the sub-cell's corners and centre, -1 for none, or -2
        when they disagree (a surface cuts it).
        """
        mesh, n = self.mesh, SUBCELLS_3D

        def sample(fraction: np.ndarray) -> np.ndarray:
            m = fraction.size
            px = mesh.origin[0] + mesh.h * (
                ci[:, None, None, None] + fraction[None, :, None, None]
            )
            py = mesh.origin[1] + mesh.h * (
                cj[:, None, None, None] + fraction[None, None, :, None]
            )
            pz = mesh.origin[2] + mesh.h * (
                ck[:, None, None, None] + fraction[None, None, None, :]
            )
            px, py, pz = np.broadcast_arrays(px, py, pz)
            return mesh.inside_solid(px.ravel(), py.ravel(), pz.ravel()).reshape(
                (ci.size, m, m, m)
            )  # fmt: skip

        corner = sample(np.arange(n + 1) / n)
        centre = sample((np.arange(n) + 0.5) / n)
        same = np.ones(centre.shape, dtype=bool)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    same &= corner[:, di : di + n, dj : dj + n, dk : dk + n] == centre
        return np.where(same, centre, -2).astype(np.int64)

    def _electrode_lookup(self) -> tuple[np.ndarray, np.ndarray]:
        """The electrode at every sub-cell of the cells that touch an electrode.

        ``cells`` numbers the touching cells (-1 for the others); ``subcells`` holds, for
        each, its ``SUBCELLS_3D**3`` sub-cells. Built once; with `allgather` every rank
        samples an equal share of the cells and the shares are gathered.
        """
        if self._lookup is None:
            cells = np.full(self._electrode_cells.shape, -1, dtype=np.int64)
            ci, cj, ck = np.nonzero(self._electrode_cells)
            if not self.mesh.electrodes or not ci.size:
                self._lookup = (cells, np.full(1, -1, dtype=np.int64))
                return self._lookup
            cells[ci, cj, ck] = np.arange(ci.size, dtype=np.int64)
            rank, size = self.ranks
            if size > 1 and self.allgather is not None:
                share = share_of(ci.size, rank, size)
                parts = self.allgather(self._subcells(ci[share], cj[share], ck[share]))
                subcells = np.concatenate(parts, axis=0)
            else:
                subcells = self._subcells(ci, cj, ck)
            self._lookup = (cells, subcells.ravel())
        return self._lookup

    def _finalize(self, charge: np.ndarray) -> np.ndarray:
        """Divide the node charges by the node volumes: ``h^3``, halved on each face."""
        density = charge / self.mesh.h**3
        for axis in range(3):
            index: list[Any] = [slice(None)] * 3
            for end in (0, -1):
                index[axis] = end
                density[tuple(index)] *= 2.0
        return density

    # -- tracing ----------------------------------------------------------------------------

    @scope_profiler.profile("beamtracer.trace", tags=PROFILE_TAGS)
    def trace(self, particles: BeamParticles) -> TraceResult:
        """Trace ``particles`` until they leave the mesh or hit an electrode.

        Those stopped in a sub-cell cut by an electrode surface are tested with the
        electrodes' own test, and the ones outside resume. A trace that records more
        crossings of a plane, or more points of a kept trajectory, than its buffers hold
        is repeated with larger ones.
        """
        field = self.electric
        if field is None:
            raise RuntimeError("the tracer has no electric field yet")
        mesh = self.mesh
        n = len(particles)
        x = np.array(particles.position, dtype=float, order="C")
        v = np.array(particles.velocity, dtype=float, order="C")
        qm = np.ascontiguousarray(particles.charge / particles.mass, dtype=float)
        current = particles.current.astype(float)
        dq = np.ascontiguousarray(np.sign(particles.charge) * np.abs(current))
        species = particles.species
        status = np.full(n, _ALIVE, dtype=np.int64)
        electrode = np.full(n, -1, dtype=np.int64)
        steps = np.zeros(n, dtype=np.int64)
        if mesh.electrodes and n:
            hit = mesh.inside_solid(*self._mesh_points(x))
            dead = hit >= 0
            status[dead] = ELECTRODE
            electrode[dead] = hit[dead]
        start = np.flatnonzero(status == _ALIVE).astype(np.int64)
        save = self.save_trajectories
        saved_ids = np.arange(0, n, save) if save else np.zeros(0, int)
        path_slot = np.full(n, -1, dtype=np.int64)
        path_slot[saved_ids] = np.arange(saved_ids.size)
        first = np.stack(self._mesh_points(x[:, saved_ids]))
        cells, subcells = self._electrode_lookup()
        magnetic = self.magnetic
        has_b = int(magnetic is not None)
        if magnetic is None:
            magnetic = np.zeros((3, 2, 2, 2))
        planes = np.array(self.planes or (np.nan,), dtype=float)
        threads = max(1, min(_num_threads(), start.size))
        mirror_y = int("y_min" in mesh.symmetry_planes)
        mirror_z = int("z_min" in mesh.symmetry_planes)
        initial = (x.copy(), v.copy(), status.copy(), electrode.copy())
        while True:
            capacity, path_capacity = self._capacity
            charge = np.zeros((threads, int(np.prod(mesh.shape))))
            hits = np.zeros((n, planes.size, capacity, 8))
            counts = np.zeros((n, planes.size), dtype=np.int64)
            paths = np.zeros((max(saved_ids.size, 1), path_capacity, 3))
            path_count = np.ones(max(saved_ids.size, 1), dtype=np.int64)
            paths[: saved_ids.size, 0] = first.T  # the start
            active = start
            while active.size:
                trace_trajectories_3d(
                    active, x, v, qm, dq, steps, status, electrode,
                    field.fx, field.fy, field.fz, magnetic, has_b,
                    *map(float, mesh.origin), mesh.h, *map(float, mesh.upper),
                    mirror_y, mirror_z, INTEGRATORS.index(self.integrator),
                    self.step_fraction, self.max_steps,
                    cells, subcells, SUBCELLS_3D, planes, hits, counts,
                    path_slot, paths, path_count, threads, charge,
                )  # fmt: skip
                paused = active[status[active] == _PAUSED]
                if not paused.size:
                    break
                hit = mesh.inside_solid(*self._mesh_points(x[:, paused]))
                inside = hit >= 0
                status[paused[inside]] = ELECTRODE
                electrode[paused[inside]] = hit[inside]
                active = paused[~inside]
                status[active] = _ALIVE
            most = int(counts.max()) if counts.size else 0
            longest = int(path_count.max())
            if most <= capacity and longest <= path_capacity:
                break
            self._capacity = (max(capacity, most), max(path_capacity, longest))
            x, v, status, electrode = (a.copy() for a in initial)
            steps[:] = 0
        node_charge: Any = charge.sum(axis=0).reshape(mesh.shape)
        if self.reduce is not None:
            node_charge = self.reduce(node_charge)
        hit_planes = []
        slots = np.arange(capacity)
        for p, position in enumerate(self.planes):
            k, c = np.nonzero(slots[None, :] < counts[:, p, None])
            record = hits[k, p, c]
            order = np.lexsort((k, record[:, 7]))  # in the order of the steps
            k, record = k[order], record[order]
            plane = PlaneHits(position)
            plane.append(
                record[:, 0:3].T.copy(),
                record[:, 3:6].T.copy(),
                current[k],
                species[k],
                record[:, 6].astype(np.int8),
            )
            hit_planes.append(plane)
        return TraceResult(
            charge_density=self._finalize(node_charge),
            status=status.astype(np.int8),
            electrode=electrode.astype(np.int32),
            planes=hit_planes,
            trajectories=[
                paths[i, : path_count[i]].copy() for i in range(saved_ids.size)
            ],
            trajectory_species=species[saved_ids],
            steps=int(steps.max()) if n else 0,
        )
