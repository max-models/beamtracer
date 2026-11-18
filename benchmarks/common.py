"""Shared pieces of the IBSimu benchmarks.

A case is one of IBSimu's 2D or cylindrical Python examples (``examples_py`` of
max-models/ibsimu), described once (electrodes as polygons, faces, beam, plasma, magnetic
field) and run in both codes on the same mesh by `run_ibsimu` and `run_beamtracer`. Both
run with trajectory saving off and are compared on the potential at the nodes and on the
beam at the exit face ``x = x_max``, where IBSimu keeps the final state of every particle.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.constants import atomic_mass, e

Region = Callable[[str], contextlib.AbstractContextManager[Any]]
Point = tuple[float, float]

#: How much every electrode is shrunk, in cells of the case's mesh at scale 1: no surface
#: then passes through a node (at scales below 100), where the codes would classify the
#: node by different conventions (or by rounding) rather than by the geometry.
SHRINK = 0.37


@dataclass(frozen=True)
class Electrode:
    """An electrode: a polygon (its vertices, in ``(x, y)`` or ``(x, r)`` [m]) at a potential [V]."""

    name: str
    vertices: tuple[Point, ...]
    potential: float


@dataclass
class Exit:
    """The beam at the exit face, one entry per particle that reached it.

    Attributes:
        u: Transverse position ``y`` (planar) or ``r`` (cylindrical) [m].
        up: Transverse angle ``u' = v_u / v_x`` [rad].
        current: Current of the particle [A/m planar, A cylindrical].
        energy: Kinetic energy [eV].
    """

    u: np.ndarray
    up: np.ndarray
    current: np.ndarray
    energy: np.ndarray


@dataclass
class Outcome:
    """What a code gives for a case: the potential at the nodes and the exit beam.

    ``away`` marks the nodes at least a cell from every electrode, where the potentials are
    compared (the runner adds it); None compares all nodes.
    """

    potential: np.ndarray
    exit: Exit
    away: np.ndarray | None = None


def offset(vertices: Sequence[Point], grow: float) -> np.ndarray:
    """Return a polygon with every edge moved outwards by ``grow`` [m] (inwards if negative).

    Each vertex goes to the intersection of its two moved edges (a miter join).

    Args:
        vertices: The polygon, in either orientation.
        grow: The distance [m].

    Returns:
        The moved vertices, ``(n, 2)``.
    """
    p = np.asarray(vertices, dtype=float)
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    edges = np.roll(p, -1, axis=0) - p
    # outward normals: (dy, -dx) for a counterclockwise polygon
    normals = np.sign(area) * np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    bisector = np.roll(normals, 1, axis=0) + normals
    return p + bisector * (grow / np.sum(bisector * normals, axis=1))[:, None]


@dataclass(frozen=True)
class Case:
    """One benchmark case.

    Attributes:
        name: Short name, used on the command line.
        title: One line on what the case is.
        source: The IBSimu example it reproduces.
        cylindrical: ``(x, r)`` geometry, else planar ``(x, y)``.
        nodes: Mesh nodes ``(nx, ny)`` at scale 1.
        h: Mesh spacing at scale 1 [m].
        particles: Trajectories per iteration.
        iterations: Vlasov iterations (field solve, then trace).
        mass: Particle mass [u].
        symmetric: ``y = 0`` is a symmetry plane (planar, mirrored) or the axis; the
            moments are then taken about 0 with the beam folded onto ``u >= 0``, else
            about the mean.
        electrodes: The electrodes, before `SHRINK`.
        faces: Conditions on ``x_min, x_max, y_min, y_max``: a potential [V], or None
            for a zero normal derivative.
        beam: ``current_density`` [A/m^2], ``energy``, ``parallel_temperature`` and
            ``transverse_temperature`` [eV], ``charge`` [e], and the ``start`` and
            ``end`` of the emitting line [m]; it leaves perpendicular to the line,
            90 degrees clockwise from start to end, in both codes.
        tolerance: Allowed differences of the compared quantities (`compare`); a
            quantity without one is shown but not checked.
        solver: IBSimu's field solver as in the example, ``"gs"`` or ``"bicgstab"``.
        plasma: Positive-ion extraction: ``initial_x`` [m] and ``potential`` [V] of the
            first iteration's plasma, then Boltzmann electrons of
            ``electron_temperature`` [eV] balancing the beam's charge density at its
            start (IBSimu's ``pexp`` plasma, beamtracer's `ExtractionPlasma`).
        solve: False keeps IBSimu's potential at zero, as the examples without
            electrodes do.
        ibsimu_magnetic: ``f(nodes, h)`` returning IBSimu's magnetic field, if any.
        beamtracer_fields: ``f(nodes, h)`` returning further `BeamOpticsConfig`
            arguments (the same magnetic field), if any.
        notes: Where the case departs from the example.
    """

    name: str
    title: str
    source: str
    cylindrical: bool
    nodes: tuple[int, int]
    h: float
    particles: int
    iterations: int
    mass: float
    symmetric: bool
    electrodes: tuple[Electrode, ...]
    faces: tuple[float | None, float | None, float | None, float | None]
    beam: dict[str, Any]
    tolerance: dict[str, float]
    solver: str = "gs"
    plasma: dict[str, float] | None = None
    solve: bool = True
    ibsimu_magnetic: Callable[[tuple[int, int], float], Any] | None = None
    beamtracer_fields: Callable[[tuple[int, int], float], dict[str, Any]] | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def mesh(self, scale: int) -> tuple[tuple[int, int], float]:
        """Return the nodes and the spacing with the cells divided by ``scale``."""
        nx, ny = self.nodes
        return ((nx - 1) * scale + 1, (ny - 1) * scale + 1), self.h / scale

    def polygons(self) -> list[tuple[np.ndarray, float]]:
        """Return the electrodes as both codes see them: shrunk vertices and potentials."""
        return [
            (offset(electrode.vertices, -SHRINK * self.h), electrode.potential)
            for electrode in self.electrodes
        ]


def box(x0: float, x1: float, y0: float, y1: float) -> tuple[Point, ...]:
    """Return the vertices of the rectangle ``[x0, x1] x [y0, y1]``."""
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def contains(vertices: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return whether the points ``(x, y)`` lie inside a polygon (even-odd ray casting)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xc)
    return inside


def kinetic_energy(mass: float, speed_squared: np.ndarray) -> np.ndarray:
    """Return the kinetic energy [eV] of particles of ``mass`` [u]."""
    return 0.5 * mass * atomic_mass * speed_squared / e


def moments(beam: Exit, symmetric: bool) -> dict[str, float]:
    """Return the current, the transverse moments and the mean energy of an exit beam.

    The moments are current-weighted. For a symmetric beam they are taken about
    ``u = 0`` after folding the beam onto ``u >= 0`` (a planar beam traced across the
    symmetry plane continues at negative ``y``); else about the mean.

    Args:
        beam: The beam at the exit face.
        symmetric: Whether the beam is symmetric about ``u = 0``.

    Returns:
        ``current``, ``mean``, ``mean_angle``, ``rms``, ``rms_angle``, ``emittance``
        (rms, about the same centre) and ``energy`` (mean kinetic energy).
    """
    w = beam.current / beam.current.sum()
    u, up = beam.u, beam.up
    if symmetric:
        sign = np.where(u < 0.0, -1.0, 1.0)
        u, up = u * sign, up * sign
    mean, mean_angle = np.sum(w * u), np.sum(w * up)
    centre, centre_angle = (0.0, 0.0) if symmetric else (mean, mean_angle)
    du, dup = u - centre, up - centre_angle
    uu, pp, up_ = np.sum(w * du**2), np.sum(w * dup**2), np.sum(w * du * dup)
    return {
        "current": float(beam.current.sum()),
        "mean": float(mean),
        "mean_angle": float(mean_angle),
        "rms": float(np.sqrt(uu)),
        "rms_angle": float(np.sqrt(pp)),
        "emittance": float(np.sqrt(max(uu * pp - up_**2, 0.0))),
        "energy": float(np.sum(w * beam.energy)),
    }


def away_from_electrodes(case: Case, nodes: tuple[int, int], h: float) -> np.ndarray:
    """Return the nodes outside every electrode whose four neighbours are outside too.

    Next to an electrode the two codes discretise the surface each in its own way; the
    potentials are compared away from it.
    """
    x, y = np.meshgrid(np.arange(nodes[0]) * h, np.arange(nodes[1]) * h, indexing="ij")
    solid = np.zeros(nodes, dtype=bool)
    for vertices, _ in case.polygons():
        solid |= contains(vertices, x, y)
    near = solid.copy()
    near[1:] |= solid[:-1]
    near[:-1] |= solid[1:]
    near[:, 1:] |= solid[:, :-1]
    near[:, :-1] |= solid[:, 1:]
    return ~near


def potential_difference(
    ibsimu: np.ndarray, beamtracer: np.ndarray, away: np.ndarray | None = None
) -> dict[str, float]:
    """Return the rms and the largest node difference of two potentials, over the span.

    Only the nodes of ``away`` (all if None) are compared; the span is IBSimu's over all
    nodes.
    """
    span = float(np.ptp(ibsimu)) or 1.0
    diff = (beamtracer - ibsimu)[away if away is not None else slice(None)]
    return {
        "potential_rms": float(np.sqrt(np.mean(diff**2)) / span),
        "potential_max": float(np.max(np.abs(diff)) / span),
    }


def compare(case: Case, ibsimu: Outcome, beamtracer: Outcome) -> list[dict[str, Any]]:
    """Compare the two codes on a case.

    The beam quantities are compared relative to the IBSimu value (the mean angle
    relative to the larger of it and the rms angle, since it may be near zero), the
    potential relative to its span.

    Args:
        case: The case, whose ``tolerance`` sets which differences are checked.
        ibsimu: What IBSimu gave.
        beamtracer: What beamtracer gave.

    Returns:
        One row per quantity: ``quantity``, ``ibsimu``, ``beamtracer``, ``difference``,
        ``tolerance`` (or None) and ``ok``.
    """
    a = moments(ibsimu.exit, case.symmetric)
    b = moments(beamtracer.exit, case.symmetric)
    rows = []
    for quantity in a:
        scale = abs(a[quantity])
        if quantity == "mean_angle":
            scale = max(scale, a["rms_angle"])
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


def _ibsimu_bound(value: float | None) -> Any:
    """Return a Dirichlet bound at ``value``, or a zero Neumann one for None."""
    import ibsimu

    if value is None:
        return ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0)
    return ibsimu.Bound(ibsimu.BOUND_DIRICHLET, value)


def _ibsimu_inside(vertices: np.ndarray) -> Callable[[float, float, float], bool]:
    """Return an IBSimu inside test ``f(x, y, z)`` of a polygon."""
    edges = [
        (float(x1), float(y1), float(x2), float(y2))
        for (x1, y1), (x2, y2) in zip(
            vertices, np.roll(vertices, -1, axis=0), strict=True
        )
        if y1 != y2
    ]

    def inside(x: float, y: float, z: float) -> bool:
        result = False
        for x1, y1, x2, y2 in edges:
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                result = not result
        return result

    return inside


def _ibsimu_exit(pdb: Any, case: Case, x_max: float) -> Exit:
    """Return the particles of an IBSimu run that ended on the exit face."""
    import ibsimu

    u, up, current, speed2 = [], [], [], []
    for i in range(pdb.size()):
        particle = pdb.particle(i)
        p = particle.x()
        if particle.get_status() != ibsimu.PARTICLE_OUT or p.x() < x_max * (1 - 1e-9):
            continue
        if case.cylindrical:
            u.append(p.r())
            up.append(p.vr() / p.vx())
            speed2.append(p.vx() ** 2 + p.vr() ** 2 + (p.r() * p.w()) ** 2)
        else:
            u.append(p.y())
            up.append(p.vy() / p.vx())
            speed2.append(p.vx() ** 2 + p.vy() ** 2)
        current.append(particle.IQ())
    return Exit(
        np.array(u),
        np.array(up),
        np.array(current),
        kinetic_energy(case.mass, np.array(speed2)),
    )


def run_ibsimu(
    case: Case,
    nodes: tuple[int, int],
    h: float,
    particles: int,
    threads: int,
    region: Region,
) -> Outcome:
    """Run a case in IBSimu the way its example does.

    Each iteration solves the field with the space charge of the previous trace, then
    traces the beam. When the case is symmetric the y_min face is a symmetry plane or
    the axis: mirrored particles, a field extrapolated with an even potential.

    Args:
        case: The case.
        nodes: The mesh nodes.
        h: The mesh spacing [m].
        particles: Trajectories per iteration.
        threads: IBSimu's threads.
        region: Profiling regions: ``setup``, ``field_solve`` and ``trace``.

    Returns:
        The potential and the exit beam of the last iteration.
    """
    import ibsimu

    ibsimu.ibsimu.set_thread_count(threads)
    with region("setup"):
        mode = ibsimu.MODE_CYL if case.cylindrical else ibsimu.MODE_2D
        geom = ibsimu.Geometry(
            mode, ibsimu.Int3D(nodes[0], nodes[1], 1), ibsimu.Vec3D(0, 0, 0), h
        )
        for k, value in enumerate(case.faces):
            geom.set_boundary(1 + k, _ibsimu_bound(value))
        for k, (vertices, potential) in enumerate(case.polygons()):
            geom.set_solid(7 + k, ibsimu.FuncSolid(_ibsimu_inside(vertices)))
            geom.set_boundary(7 + k, _ibsimu_bound(potential))
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
        if case.ibsimu_magnetic:
            bfield = case.ibsimu_magnetic(nodes, h)
        else:
            bfield = ibsimu.MeshVectorField()
        database = (
            ibsimu.ParticleDataBaseCyl
            if case.cylindrical
            else ibsimu.ParticleDataBase2D
        )
        pdb = database(geom)
        pdb.set_save_trajectories(False)
        if case.symmetric:
            extrapolation = [ibsimu.FIELD_EXTRAPOLATE] * 6
            extrapolation[2] = ibsimu.FIELD_SYMMETRIC_POTENTIAL
            efield.set_extrapolation(extrapolation)
            pdb.set_mirror([False, False, True, False, False, False])
        if plasma:
            pdb.set_polyint(True)
    beam = case.beam
    (x1, y1), (x2, y2) = beam["start"], beam["end"]
    for iteration in range(case.iterations):
        if plasma and iteration == 1:
            solver.set_pexp_plasma(
                -pdb.get_rhosum(), plasma["electron_temperature"], plasma["potential"]
            )
        if case.solve:
            with region("field_solve"):
                solver.solve(epot, scharge)
                efield.recalculate()
        with region("trace"):
            pdb.clear()
            pdb.add_2d_beam_with_energy(
                particles, beam["current_density"], beam.get("charge", 1.0), case.mass,
                beam["energy"], beam.get("parallel_temperature", 0.0),
                beam.get("transverse_temperature", 0.0), x1, y1, x2, y2,
            )  # fmt: skip
            pdb.iterate_trajectories(scharge, efield, bfield)
    return Outcome(epot.numpy(), _ibsimu_exit(pdb, case, (nodes[0] - 1) * h))


# beamtracer ------------------------------------------------------------------------


def run_beamtracer(
    case: Case,
    nodes: tuple[int, int],
    h: float,
    particles: int,
    integrator: str,
    region: Region,
    **config_args: Any,
) -> Outcome:
    """Build and run a beamtracer `BeamOptics` model of a case.

    Args:
        case: The case.
        nodes: The mesh nodes.
        h: The mesh spacing [m].
        particles: Trajectories per iteration.
        integrator: ``"rk4"`` or ``"boris"``.
        region: Profiling regions: ``setup`` and ``run``.
        **config_args: Further `BeamOpticsConfig` arguments, overriding the case's.

    Returns:
        The potential and the beam crossing the exit face.
    """
    from beamtracer import (
        BeamOptics,
        BeamOpticsConfig,
        BeamSpecies,
        ExtractionPlasma,
        FieldBCs,
        FieldBoundaryCondition,
        GridParams,
    )
    from beamtracer.sections import ObjectParams

    def bc(value: float | None) -> Any:
        if value is None:
            return FieldBoundaryCondition.neumann(0.0)
        return FieldBoundaryCondition.dirichlet(value)

    x_lower, x_upper, y_lower, y_upper = (bc(v) for v in case.faces)
    beam = case.beam
    args: dict[str, Any] = {
        "grid": GridParams(
            num_gridpoints=nodes,
            upper_bound=((nodes[0] - 1) * h, (nodes[1] - 1) * h),
            coordinates="cylindrical" if case.cylindrical else "cartesian",
        ),
        "objects": {
            electrode.name: ObjectParams(
                type="polygon",
                vertices=tuple(map(tuple, vertices)),
                bc_type="dirichlet",
                bc_value=potential,
            )
            for electrode, (vertices, potential) in zip(
                case.electrodes, case.polygons(), strict=True
            )
        },
        "bcs_field": FieldBCs(
            x_lower=x_lower, x_upper=x_upper, y_lower=y_lower, y_upper=y_upper
        ),
        "beams": [
            BeamSpecies(
                "ion",
                beam["current_density"],
                beam["start"],
                beam["end"],
                charge=beam.get("charge", 1.0),
                mass=case.mass,
                energy=beam["energy"],
                parallel_temperature=beam.get("parallel_temperature", 0.0),
                transverse_temperature=beam.get("transverse_temperature", 0.0),
                num_particles=particles,
            )
        ],  # fmt: skip
        "integrator": integrator,
        "iterations": case.iterations,
        "tolerance": 0.0,
        "field_backend": os.environ.get("BENCHMARK_FIELD_BACKEND", "lu"),
        # IBSimu mirrors particles at y_min: the planar symmetric cases are half domains
        "symmetry_plane": case.symmetric and not case.cylindrical,
    }
    if case.plasma:
        args["plasma"] = ExtractionPlasma(
            electron_temperature=case.plasma["electron_temperature"],
            plasma_potential=case.plasma["potential"],
            initial_plasma_x=case.plasma["initial_x"],
        )
    if case.beamtracer_fields:
        args.update(case.beamtracer_fields(nodes, h))
    args.update(config_args)
    plane = (nodes[0] - 1) * h * (1.0 - 1e-9)
    with region("setup"):
        model = BeamOptics(BeamOpticsConfig(diagnostic_planes=[plane], **args))
    with region("run"):
        model.run()
    hits = model.plane(plane)
    forward = hits.direction > 0
    point, velocity = hits.point[:, forward], hits.velocity[:, forward]
    if case.cylindrical:
        r = np.hypot(point[1], point[2])
        radial = point[1] * velocity[1] + point[2] * velocity[2]
        vr = np.where(r > 0.0, radial / np.where(r > 0.0, r, 1.0), velocity[1])
        u, up = r, vr / velocity[0]
    else:
        u, up = point[1], velocity[1] / velocity[0]
    energy = kinetic_energy(case.mass, np.sum(velocity**2, axis=0))
    return Outcome(
        np.asarray(model.potential), Exit(u, up, hits.current[forward], energy)
    )


@contextlib.contextmanager
def no_region(_: str) -> Iterator[None]:
    """Time nothing: a stand-in for the runner's profiling regions."""
    yield
