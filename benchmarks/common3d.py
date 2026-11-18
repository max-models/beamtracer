"""The 3D IBSimu benchmarks: cases on 3D grids, run in both codes on the same geometry.

As `common` in 2D: a case is described once (boxes and tubes as electrodes, faces,
particles, plasma, magnetic field) and run by `run_ibsimu` (IBSimu's MODE_3D through its
Python bindings, trajectory saving off) and `run_beamtracer` (`BeamOptics` on a 3D grid).
Both codes start from the same particles, given trajectory by trajectory, so the comparison
carries no sampling noise. The electrodes are shrunk by `SHRINK` cells, so no surface
passes through a node.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from common import SHRINK, Region, kinetic_energy, potential_difference

#: The faces of a 3D grid, in IBSimu's boundary numbering 1-6.
FACES_3D = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


@dataclass(frozen=True)
class Box:
    """A box electrode ``[x0, x1] x [y0, y1] x [z0, z1]`` at ``potential`` [V]."""

    name: str
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    potential: float

    def shrunk(self, d: float) -> Box:
        """The box with every face moved in by ``d``."""
        return Box(
            self.name,
            (self.x[0] + d, self.x[1] - d),
            (self.y[0] + d, self.y[1] - d),
            (self.z[0] + d, self.z[1] - d),
            self.potential,
        )

    def inside(self, x: float, y: float, z: float) -> bool:
        """Whether a point is inside."""
        return (
            self.x[0] < x < self.x[1]
            and self.y[0] < y < self.y[1]
            and self.z[0] < z < self.z[1]
        )

    def inside_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Whether points are inside, vectorised."""
        return (
            (x > self.x[0]) & (x < self.x[1])
            & (y > self.y[0]) & (y < self.y[1])
            & (z > self.z[0]) & (z < self.z[1])
        )  # fmt: skip

    def object_params(self) -> Any:
        """The electrode as a beamtracer `cuboid`."""
        from beamtracer.sections import ObjectParams

        return ObjectParams(
            type="cuboid",
            midpoint=tuple(0.5 * (a + b) for a, b in (self.x, self.y, self.z)),
            side_lengths=tuple(b - a for a, b in (self.x, self.y, self.z)),
            bc_type="dirichlet",
            bc_value=self.potential,
        )


@dataclass(frozen=True)
class Tube:
    """A tube electrode on the ``x`` axis: ``x0 < x < x1``, ``inner < r < outer``."""

    name: str
    x: tuple[float, float]
    inner: float
    outer: float
    potential: float

    def shrunk(self, d: float) -> Tube:
        """The tube with every surface moved in by ``d``."""
        return Tube(
            self.name,
            (self.x[0] + d, self.x[1] - d),
            self.inner + d,
            self.outer - d,
            self.potential,
        )

    def inside(self, x: float, y: float, z: float) -> bool:
        """Whether a point is inside."""
        r2 = y * y + z * z
        return self.x[0] < x < self.x[1] and self.inner**2 < r2 < self.outer**2

    def inside_array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Whether points are inside, vectorised."""
        r2 = y * y + z * z
        return (
            (x > self.x[0]) & (x < self.x[1])
            & (r2 > self.inner**2) & (r2 < self.outer**2)
        )  # fmt: skip

    def object_params(self) -> Any:
        """The electrode as a beamtracer `tube`."""
        from beamtracer.sections import ObjectParams

        return ObjectParams(
            type="tube",
            start=(self.x[0], 0.0, 0.0),
            end=(self.x[1], 0.0, 0.0),
            inner_radius=self.inner,
            outer_radius=self.outer,
            bc_type="dirichlet",
            bc_value=self.potential,
        )


@dataclass
class Exit3D:
    """The beam at the exit face: positions ``y``, ``z``, angles, currents, energies."""

    y: np.ndarray
    yp: np.ndarray
    z: np.ndarray
    zp: np.ndarray
    current: np.ndarray
    energy: np.ndarray


@dataclass
class Outcome3D:
    """What a code gives for a 3D case: the potential at the nodes and the exit beam."""

    potential: np.ndarray
    exit: Exit3D
    away: np.ndarray | None = None


@dataclass(frozen=True)
class Case3D:
    """One 3D benchmark case.

    Attributes:
        name, title, source: As `common.Case`.
        nodes: Mesh nodes ``(nx, ny, nz)`` at scale 1.
        origin: The position of node ``(0, 0, 0)`` [m].
        h: Mesh spacing at scale 1 [m].
        particles: Trajectories.
        iterations: Vlasov iterations.
        mass: Particle mass [u].
        symmetry_planes: Faces at which the geometry is mirrored (``"y_min"``,
            ``"z_min"``): IBSimu mirrors the particles there, beamtracer continues them.
        electrodes: Boxes and tubes, before `SHRINK`; where they overlap the last
            listed holds the nodes (IBSimu's rule; beamtracer takes them reversed).
        faces: Conditions on the six faces (`FACES_3D`): a potential [V] or None for a
            zero normal derivative.
        beam: ``make(particles) -> (position (3, N), velocity (3, N), current (N,))``
            [m, m/s, A], the same trajectories in both codes; ``charge`` [e].
        tolerance: Allowed differences (`compare`).
        solver: IBSimu's solver, ``"gs"`` or ``"bicgstab"``.
        plasma: ``initial_x``, ``potential``, ``electron_temperature`` and ``rho`` (the
            electrons' charge density in the plasma) [C/m^3], or None.
        solve: False keeps IBSimu's potential at zero.
        bx: ``bx(x)`` [T], an axial field uniform across the beam, or None.
        notes: Where the case departs from the example.
    """

    name: str
    title: str
    source: str
    nodes: tuple[int, int, int]
    origin: tuple[float, float, float]
    h: float
    particles: int
    iterations: int
    mass: float
    symmetry_planes: tuple[str, ...]
    electrodes: tuple[Box | Tube, ...]
    faces: tuple[float | None, ...]
    beam: dict[str, Any]
    tolerance: dict[str, float]
    solver: str = "gs"
    plasma: dict[str, float] | None = None
    solve: bool = True
    bx: Callable[[np.ndarray], np.ndarray] | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    #: A 3D case.
    three = True
    #: The 2D cases' flags, for the shared runner.
    cylindrical = False

    def mesh(self, scale: int) -> tuple[tuple[int, int, int], float]:
        """Return the nodes and the spacing with the cells divided by ``scale``."""
        return tuple((n - 1) * scale + 1 for n in self.nodes), self.h / scale  # type: ignore[return-value]

    def shrunk(self) -> list[Box | Tube]:
        """The electrodes as both codes see them."""
        return [e.shrunk(SHRINK * self.h) for e in self.electrodes]

    def node_coordinates(
        self, nodes: tuple[int, int, int], h: float
    ) -> tuple[np.ndarray, ...]:
        """The node coordinates ``(x, y, z)`` as ``(nx, ny, nz)`` arrays."""
        axes = [self.origin[a] + h * np.arange(nodes[a]) for a in range(3)]
        return tuple(np.meshgrid(*axes, indexing="ij"))


def _fold(u: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A beam traced across a symmetry plane folded back onto ``u >= 0``."""
    sign = np.where(u < 0.0, -1.0, 1.0)
    return u * sign, up * sign


def moments(beam: Exit3D, planes: tuple[str, ...]) -> dict[str, float]:
    """Current-weighted moments of the exit beam in both transverse planes.

    In a plane with a symmetry plane the beam is folded onto ``u >= 0`` and the moments
    are taken about 0; else about the mean.
    """
    w = beam.current / beam.current.sum()
    out: dict[str, float] = {"current": float(beam.current.sum())}
    for axis, (u, up), plane in (
        ("y", (beam.y, beam.yp), "y_min"),
        ("z", (beam.z, beam.zp), "z_min"),
    ):
        mirrored = plane in planes
        if mirrored:
            u, up = _fold(u, up)
        mean, mean_angle = np.sum(w * u), np.sum(w * up)
        cu, cp = (0.0, 0.0) if mirrored else (mean, mean_angle)
        uu = np.sum(w * (u - cu) ** 2)
        pp = np.sum(w * (up - cp) ** 2)
        xp = np.sum(w * (u - cu) * (up - cp))
        out[f"mean_{axis}"] = float(mean)
        out[f"rms_{axis}"] = float(np.sqrt(uu))
        out[f"rms_angle_{axis}"] = float(np.sqrt(pp))
        out[f"emittance_{axis}"] = float(np.sqrt(max(uu * pp - xp * xp, 0.0)))
    out["energy"] = float(np.sum(w * beam.energy))
    return out


def away_from_electrodes(
    case: Case3D, nodes: tuple[int, int, int], h: float
) -> np.ndarray:
    """The nodes outside every electrode whose six neighbours are outside too."""
    x, y, z = case.node_coordinates(nodes, h)
    solid = np.zeros(nodes, dtype=bool)
    for electrode in case.shrunk():
        solid |= electrode.inside_array(x, y, z)
    near = solid.copy()
    for axis in range(3):
        for step in (1, -1):
            near |= np.roll(
                solid, step, axis=axis
            )  # the ends wrap: a cell's margin less
    return ~near


def compare(
    case: Case3D, ibsimu: Outcome3D, beamtracer: Outcome3D
) -> list[dict[str, Any]]:
    """Compare the two codes on a 3D case (as `common.compare`)."""
    a = moments(ibsimu.exit, case.symmetry_planes)
    b = moments(beamtracer.exit, case.symmetry_planes)
    rows = []
    for quantity in a:
        scale = abs(a[quantity])
        if quantity.startswith("mean_") and quantity != "mean_energy":
            axis = quantity[-1]
            scale = max(scale, a[f"rms_{axis}"])  # a mean may be near zero
        rows.append(
            {
                "quantity": quantity,
                "ibsimu": a[quantity],
                "beamtracer": b[quantity],
                "difference": abs(b[quantity] - a[quantity]) / (scale or 1.0),
            }
        )
    for quantity, value in potential_difference(
        ibsimu.potential, beamtracer.potential, ibsimu.away
    ).items():
        rows.append(
            {
                "quantity": quantity,
                "ibsimu": None,
                "beamtracer": None,
                "difference": value,
            }
        )
    for row in rows:
        row["tolerance"] = case.tolerance.get(row["quantity"])
        row["ok"] = row["tolerance"] is None or row["difference"] <= row["tolerance"]
    return rows


# IBSimu ----------------------------------------------------------------------------


def _bound(value: float | None) -> Any:
    import ibsimu

    if value is None:
        return ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0)
    return ibsimu.Bound(ibsimu.BOUND_DIRICHLET, value)


def _exit(pdb: Any, case: Case3D, x_max: float) -> Exit3D:
    """The particles of an IBSimu run that ended on the exit face."""
    import ibsimu

    rows = []
    for i in range(pdb.size()):
        particle = pdb.particle(i)
        p = particle.x()
        if particle.get_status() != ibsimu.PARTICLE_OUT or p.x() < x_max * (1 - 1e-9):
            continue
        vx = p.vx()
        rows.append(
            (p.y(), p.vy() / vx, p.z(), p.vz() / vx, particle.IQ(),
             vx * vx + p.vy() ** 2 + p.vz() ** 2)
        )  # fmt: skip
    y, yp, z, zp, current, speed2 = (np.array(c) for c in zip(*rows, strict=True))
    return Exit3D(y, yp, z, zp, current, kinetic_energy(case.mass, speed2))


def run_ibsimu(
    case: Case3D,
    nodes: tuple[int, int, int],
    h: float,
    particles: int,
    threads: int,
    region: Region,
) -> Outcome3D:
    """Run a 3D case in IBSimu (MODE_3D), the particles added one by one."""
    import ibsimu

    ibsimu.ibsimu.set_thread_count(threads)
    position, velocity, current = case.beam["make"](particles)
    charge = case.beam.get("charge", 1.0)
    with region("setup"):
        geom = ibsimu.Geometry(
            ibsimu.MODE_3D, ibsimu.Int3D(*nodes), ibsimu.Vec3D(*case.origin), h
        )
        for k, value in enumerate(case.faces):
            geom.set_boundary(1 + k, _bound(value))
        for k, electrode in enumerate(case.shrunk()):
            geom.set_solid(7 + k, ibsimu.FuncSolid(electrode.inside))
            geom.set_boundary(7 + k, _bound(electrode.potential))
        geom.build_mesh()
        if case.solver == "gs":
            solver = ibsimu.EpotGSSolver(geom)
        else:
            solver = ibsimu.EpotBiCGSTABSolver(geom)
        plasma = case.plasma
        if plasma:
            solver.set_initial_plasma(
                plasma["potential"],
                ibsimu.InitialPlasma(ibsimu.AXIS_X, plasma["initial_x"]),
            )
        epot = ibsimu.EpotField(geom)
        scharge = ibsimu.MeshScalarField(geom)
        efield = ibsimu.EpotEfield(epot)
        if case.bx is not None:
            bfield = ibsimu.MeshVectorField(
                ibsimu.MODE_3D, [True, False, False], ibsimu.Int3D(*nodes),
                ibsimu.Vec3D(*case.origin), h,
            )  # fmt: skip
            values = np.zeros((*nodes, 3))
            xs = case.origin[0] + h * np.arange(nodes[0])
            values[..., 0] = case.bx(xs)[:, None, None]
            bfield.set_numpy(values)
        else:
            bfield = ibsimu.MeshVectorField()
        pdb = ibsimu.ParticleDataBase3D(geom)
        pdb.set_save_trajectories(False)
        mirror = [False] * 6
        extrapolation = [ibsimu.FIELD_EXTRAPOLATE] * 6
        for plane in case.symmetry_planes:
            index = FACES_3D.index(plane)
            mirror[index] = True
            extrapolation[index] = ibsimu.FIELD_SYMMETRIC_POTENTIAL
        pdb.set_mirror(mirror)
        efield.set_extrapolation(extrapolation)
        if plasma:
            pdb.set_polyint(True)
    for iteration in range(case.iterations):
        if plasma and iteration == 1:
            solver.set_pexp_plasma(
                -plasma["rho"], plasma["electron_temperature"], plasma["potential"]
            )
        if case.solve:
            with region("field_solve"):
                solver.solve(epot, scharge)
                efield.recalculate()
        with region("trace"):
            pdb.clear()
            for k in range(position.shape[1]):
                pdb.add_particle(
                    float(current[k]), charge, case.mass,
                    ibsimu.ParticleP3D(
                        0.0, position[0, k], velocity[0, k], position[1, k],
                        velocity[1, k], position[2, k], velocity[2, k],
                    ),
                )  # fmt: skip
            pdb.iterate_trajectories(scharge, efield, bfield)
    return Outcome3D(
        epot.numpy(), _exit(pdb, case, case.origin[0] + (nodes[0] - 1) * h)
    )


# beamtracer ------------------------------------------------------------------------


def run_beamtracer(
    case: Case3D,
    nodes: tuple[int, int, int],
    h: float,
    particles: int,
    integrator: str,
    region: Region,
) -> Outcome3D:
    """Run a 3D case as a beamtracer `BeamOptics` model on a 3D grid."""
    import tempfile
    from pathlib import Path

    from scipy.constants import atomic_mass, e

    from beamtracer import (
        BeamOptics,
        BeamOpticsConfig,
        ExtractionPlasma,
        FieldBCs,
        FieldBoundaryCondition,
        FieldMapParams,
        GridParams,
        ParticleBeam,
    )
    from beamtracer.field_map import FieldMap

    def bc(value: float | None) -> Any:
        if value is None:
            return FieldBoundaryCondition.neumann(0.0)
        return FieldBoundaryCondition.dirichlet(value)

    position, velocity, current = case.beam["make"](particles)
    upper = tuple(case.origin[a] + h * (nodes[a] - 1) for a in range(3))
    # IBSimu lets the last listed electrode hold shared nodes, beamtracer the first
    objects = {e.name: e.object_params() for e in reversed(case.shrunk())}
    args: dict[str, Any] = {
        "grid": GridParams(
            ndim=3, num_gridpoints=nodes, lower_bound=case.origin, upper_bound=upper
        ),
        "objects": objects,
        "bcs_field": FieldBCs(*(bc(v) for v in case.faces)),
        "beams": [
            ParticleBeam(
                name="ion",
                position=position.T.tolist(),
                velocity=velocity.T.tolist(),
                current=current.tolist(),
                charge=case.beam.get("charge", 1.0),
                mass=case.mass,
            )
        ],
        "integrator": integrator,
        "iterations": case.iterations,
        "tolerance": 0.0,
        "symmetry_planes": list(case.symmetry_planes),
    }
    if case.plasma:
        args["plasma"] = ExtractionPlasma(
            electron_temperature=case.plasma["electron_temperature"],
            plasma_potential=case.plasma["potential"],
            initial_plasma_x=case.plasma["initial_x"],
            rho_e=case.plasma["rho"],
        )
    folder = None
    if case.bx is not None:
        folder = tempfile.mkdtemp()
        path = Path(folder) / "bx.nc"
        axes = {
            name: case.origin[a] + h * np.arange(nodes[a])
            for a, name in enumerate("xyz")
        }
        bx = np.broadcast_to(case.bx(axes["x"])[:, None, None], nodes).copy()
        FieldMap.from_arrays(
            axes, {"b_field_x": bx, "b_field_y": 0 * bx, "b_field_z": 0 * bx}
        ).write(path)
        args["field_map"] = FieldMapParams(str(path))
    plane = upper[0] - 1e-9 * (upper[0] - case.origin[0])
    with region("setup"):
        model = BeamOptics(BeamOpticsConfig(diagnostic_planes=[plane], **args))
    with region("run"):
        model.run()
    hits = model.plane(plane)
    forward = hits.direction > 0
    p, v = hits.point[:, forward], hits.velocity[:, forward]
    mass = case.mass * atomic_mass
    energy = 0.5 * mass * np.sum(v * v, axis=0) / e
    exit_beam = Exit3D(
        p[1], v[1] / v[0], p[2], v[2] / v[0], hits.current[forward], energy
    )
    return Outcome3D(np.asarray(model.potential), exit_beam)
