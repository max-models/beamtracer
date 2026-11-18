"""Shared pieces of the IBSimu benchmarks.

A case is one of IBSimu's 2D or cylindrical Python examples (``examples_py`` of
max-models/ibsimu) set up in both codes on the same mesh. Both run with trajectory saving
off and are compared on the potential at the nodes and on the beam at the exit face
``x = x_max``, where IBSimu keeps the final state of every particle.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.constants import atomic_mass, e

Region = Callable[[str], contextlib.AbstractContextManager[Any]]


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
    compared (the runner adds it to IBSimu's outcome); None compares all nodes.
    """

    potential: np.ndarray
    exit: Exit
    away: np.ndarray | None = None


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
        symmetric: The beam is symmetric about ``y = 0`` (a mirrored half domain or
            the axis), so the moments are taken about 0 with the beam folded onto
            ``u >= 0``; else they are taken about the mean.
        tolerance: Allowed relative differences of the compared quantities
            (`compare`); a quantity without one is shown but not checked.
        ibsimu: ``ibsimu(nodes, h, particles, threads, region) -> Outcome``.
        beamtracer: ``beamtracer(nodes, h, particles, integrator, region) -> Outcome``.
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
    tolerance: dict[str, float]
    ibsimu: Callable[..., Outcome]
    beamtracer: Callable[..., Outcome]
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def mesh(self, scale: int) -> tuple[tuple[int, int], float]:
        """Return the nodes and the spacing with the cells divided by ``scale``."""
        nx, ny = self.nodes
        return ((nx - 1) * scale + 1, (ny - 1) * scale + 1), self.h / scale

    @property
    def x_max(self) -> float:
        """The exit face [m]."""
        return (self.nodes[0] - 1) * self.h


def kinetic_energy(mass: float, speed_squared: np.ndarray) -> np.ndarray:
    """Return the kinetic energy [eV] of particles of ``mass`` [u]."""
    return 0.5 * mass * atomic_mass * speed_squared / e


def moments(beam: Exit, symmetric: bool) -> dict[str, float]:
    """Return the current, the transverse moments and the mean energy of an exit beam.

    The moments are current-weighted. For a symmetric beam they are taken about
    ``u = 0`` after folding the beam onto ``u >= 0`` (a planar beam traced across the
    symmetry plane continues at negative ``y``); else about the mean.

    Parameters
    ----------
    beam : Exit
        The beam at the exit face.
    symmetric : bool
        Whether the beam is symmetric about ``u = 0``.

    Returns:
    -------
    dict
        ``current``, ``mean``, ``rms``, ``rms_angle``, ``emittance`` (rms, about the
        same centre) and ``energy`` (mean kinetic energy).
    """
    w = beam.current / beam.current.sum()
    u, up = beam.u, beam.up
    if symmetric:
        sign = np.where(u < 0.0, -1.0, 1.0)
        u, up = u * sign, up * sign
        mean, centre, centre_angle = np.sum(w * u), 0.0, 0.0
    else:
        mean = centre = np.sum(w * u)
        centre_angle = np.sum(w * up)
    du, dup = u - centre, up - centre_angle
    uu, pp, up_ = np.sum(w * du**2), np.sum(w * dup**2), np.sum(w * du * dup)
    return {
        "current": float(beam.current.sum()),
        "mean": float(mean),
        "rms": float(np.sqrt(uu)),
        "rms_angle": float(np.sqrt(pp)),
        "emittance": float(np.sqrt(max(uu * pp - up_**2, 0.0))),
        "energy": float(np.sum(w * beam.energy)),
    }


def away_from_electrodes(
    solids: list[tuple[Callable[[float, float, float], bool], float]],
    nodes: tuple[int, int],
    h: float,
) -> np.ndarray:
    """Return the nodes outside every electrode whose four neighbours are outside too.

    At a node inside or on an electrode the two codes may differ by convention (which of
    two overlapping electrodes holds it, whether a node on a surface is free), not by
    their solutions; such nodes and their neighbours are left out of the comparison.
    """
    x, y = np.arange(nodes[0]) * h, np.arange(nodes[1]) * h
    solid = np.array([[any(f(a, b, 0.0) for f, _ in solids) for b in y] for a in x])
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

    Parameters
    ----------
    case : Case
        The case, whose ``tolerance`` sets which differences are checked.
    ibsimu, beamtracer : Outcome
        What the two codes gave.

    Returns:
    -------
    list of dict
        One row per quantity: ``quantity``, ``ibsimu``, ``beamtracer``,
        ``difference`` (relative to the IBSimu value, or the potential differences
        relative to the potential's span), ``tolerance`` (or None) and ``ok``.
    """
    a, b = (
        moments(ibsimu.exit, case.symmetric),
        moments(beamtracer.exit, case.symmetric),
    )
    rows = []
    for quantity in a:
        scale = abs(a[quantity]) or 1.0
        rows.append(
            {
                "quantity": quantity,
                "ibsimu": a[quantity],
                "beamtracer": b[quantity],
                "difference": abs(b[quantity] - a[quantity]) / scale,
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


def _ibsimu_bound(value: float | None) -> Any:
    """Return a Dirichlet bound at ``value``, or a zero Neumann one for None."""
    import ibsimu

    if value is None:
        return ibsimu.Bound(ibsimu.BOUND_NEUMANN, 0.0)
    return ibsimu.Bound(ibsimu.BOUND_DIRICHLET, value)


# IBSimu ----------------------------------------------------------------------------


def ibsimu_geometry(
    case: Case,
    nodes: tuple[int, int],
    h: float,
    solids: list[tuple[Callable[[float, float, float], bool], float]],
    faces: list[float | None],
) -> Any:
    """Build an IBSimu geometry with solid electrodes and box boundary conditions.

    Parameters
    ----------
    case : Case
        The case (its geometry mode).
    nodes, h : tuple of int, float
        The mesh.
    solids : list of (callable, float)
        Each electrode's inside test ``f(x, y, z)`` and potential [V].
    faces : list of float or None
        Conditions on ``x_min, x_max, y_min, y_max``: a potential, or None for a
        zero normal derivative.

    Returns:
    -------
    ibsimu.Geometry
        The geometry with its mesh built.
    """
    import ibsimu

    mode = ibsimu.MODE_CYL if case.cylindrical else ibsimu.MODE_2D
    geom = ibsimu.Geometry(
        mode, ibsimu.Int3D(nodes[0], nodes[1], 1), ibsimu.Vec3D(0, 0, 0), h
    )
    for k, (inside, _) in enumerate(solids):
        geom.set_solid(7 + k, ibsimu.FuncSolid(inside))
    for k, value in enumerate(faces):
        geom.set_boundary(1 + k, _ibsimu_bound(value))
    for k, (_, value) in enumerate(solids):
        geom.set_boundary(7 + k, _ibsimu_bound(value))
    geom.build_mesh()
    return geom


def ibsimu_exit(pdb: Any, case: Case, x_max: float) -> Exit:
    """Return the particles of an IBSimu run that ended on the exit face."""
    import ibsimu

    u, up, current, speed2 = [], [], [], []
    tolerance = 1e-9 * x_max
    for i in range(pdb.size()):
        particle = pdb.particle(i)
        if particle.get_status() != ibsimu.PARTICLE_OUT:
            continue
        p = particle.x()
        if p.x() < x_max - tolerance:
            continue
        if case.cylindrical:
            vt = p.r() * p.w()
            u.append(p.r())
            up.append(p.vr() / p.vx())
            speed2.append(p.vx() ** 2 + p.vr() ** 2 + vt**2)
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


def ibsimu_beam(pdb: Any, case: Case, particles: int, beam: dict[str, Any]) -> None:
    """Add the case's beam to an IBSimu particle database.

    ``beam`` holds ``current_density`` [A/m^2], ``charge`` [e], ``energy``,
    ``parallel_temperature`` and ``transverse_temperature`` [eV], and the ``start`` and
    ``end`` of the emitting line [m], as `beamtracer_beam` reads them.
    """
    (x1, y1), (x2, y2) = beam["start"], beam["end"]
    pdb.add_2d_beam_with_energy(
        particles,
        beam["current_density"],
        beam.get("charge", 1.0),
        case.mass,
        beam["energy"],
        beam.get("parallel_temperature", 0.0),
        beam.get("transverse_temperature", 0.0),
        x1,
        y1,
        x2,
        y2,
    )


def ibsimu_fields(geom: Any, symmetric_axis: bool) -> tuple[Any, Any, Any]:
    """Return IBSimu's potential, space charge and electric field on ``geom``.

    With ``symmetric_axis`` the field is extrapolated symmetrically across ``y_min``
    (the potential is even there), as the mirrored and cylindrical examples do.
    """
    import ibsimu

    epot = ibsimu.EpotField(geom)
    scharge = ibsimu.MeshScalarField(geom)
    efield = ibsimu.EpotEfield(epot)
    if symmetric_axis:
        extrapolation = [ibsimu.FIELD_EXTRAPOLATE] * 6
        extrapolation[2] = ibsimu.FIELD_SYMMETRIC_POTENTIAL
        efield.set_extrapolation(extrapolation)
    return epot, scharge, efield


def ibsimu_database(geom: Any, case: Case, mirror_y_min: bool = False) -> Any:
    """Return an IBSimu particle database for the case, saving no trajectories."""
    import ibsimu

    pdb = (
        ibsimu.ParticleDataBaseCyl(geom)
        if case.cylindrical
        else ibsimu.ParticleDataBase2D(geom)
    )
    pdb.set_save_trajectories(False)
    if mirror_y_min:
        mirror = [False] * 6
        mirror[2] = True
        pdb.set_mirror(mirror)
    return pdb


def ibsimu_run(
    case: Case,
    nodes: tuple[int, int],
    h: float,
    particles: int,
    threads: int,
    region: Region,
    *,
    solids: list[tuple[Callable[[float, float, float], bool], float]],
    faces: list[float | None],
    beam: dict[str, Any],
    solver: str = "gs",
    plasma: dict[str, float] | None = None,
    magnetic: Callable[[Any, tuple[int, int], float], Any] | None = None,
    solve: bool = True,
) -> Outcome:
    """Run a case in IBSimu the way its examples do.

    Each iteration solves the field with the space charge of the previous trace, then
    traces the beam. The y_min face is a symmetry plane or the axis (mirrored particles,
    an even potential) when the case is symmetric.

    Parameters
    ----------
    case : Case
        The case.
    nodes, h : tuple of int, float
        The mesh.
    particles, threads : int
        Trajectories per iteration; IBSimu's threads.
    region : callable
        Profiling regions: ``setup``, ``field_solve`` and ``trace``.
    solids, faces : list
        The electrodes and the box conditions, as `ibsimu_geometry` takes them.
    beam : dict
        The beam, as `ibsimu_beam` takes it.
    solver : str
        ``"gs"`` (IBSimu's Gauss-Seidel) or ``"bicgstab"``.
    plasma : dict, optional
        Positive-ion extraction: ``initial_x`` [m] and ``potential`` [V] of the first
        iteration's plasma, then Boltzmann electrons of ``electron_temperature`` [eV]
        balancing the beam's charge density at its start (IBSimu's ``pexp`` plasma).
    magnetic : callable, optional
        ``magnetic(geom, nodes, h)`` returning IBSimu's magnetic field.
    solve : bool
        False keeps the potential at zero, as the examples without electrodes do.

    Returns:
    -------
    Outcome
        The potential and the exit beam of the last iteration.
    """
    import ibsimu

    ibsimu.ibsimu.set_thread_count(threads)
    with region("setup"):
        geom = ibsimu_geometry(case, nodes, h, solids, faces)
        if solver == "gs":
            field_solver = ibsimu.EpotGSSolver(geom)
        else:
            field_solver = ibsimu.EpotBiCGSTABSolver(geom)
        if plasma:
            field_solver.set_initial_plasma(
                plasma["potential"],
                ibsimu.InitialPlasma(ibsimu.AXIS_X, plasma["initial_x"]),
            )
        epot, scharge, efield = ibsimu_fields(geom, symmetric_axis=case.symmetric)
        bfield = magnetic(geom, nodes, h) if magnetic else ibsimu.MeshVectorField()
        pdb = ibsimu_database(geom, case, mirror_y_min=case.symmetric)
        if plasma:
            pdb.set_polyint(True)
    for iteration in range(case.iterations):
        if plasma and iteration == 1:
            field_solver.set_pexp_plasma(
                -pdb.get_rhosum(), plasma["electron_temperature"], plasma["potential"]
            )
        if solve:
            with region("field_solve"):
                field_solver.solve(epot, scharge)
                efield.recalculate()
        with region("trace"):
            pdb.clear()
            ibsimu_beam(pdb, case, particles, beam)
            pdb.iterate_trajectories(scharge, efield, bfield)
    return Outcome(epot.numpy(), ibsimu_exit(pdb, case, (nodes[0] - 1) * h))


# beamtracer ------------------------------------------------------------------------


def beamtracer_grid(case: Case, nodes: tuple[int, int], h: float) -> Any:
    """Return the beamtracer grid of the case's mesh."""
    from beamtracer import GridParams

    return GridParams(
        num_gridpoints=nodes,
        upper_bound=((nodes[0] - 1) * h, (nodes[1] - 1) * h),
        coordinates="cylindrical" if case.cylindrical else "cartesian",
    )


def beamtracer_faces(faces: list[float | None]) -> Any:
    """Return beamtracer's box conditions from ``x_min, x_max, y_min, y_max`` (see `ibsimu_geometry`)."""
    from beamtracer import FieldBCs, FieldBoundaryCondition

    def bc(value: float | None) -> Any:
        if value is None:
            return FieldBoundaryCondition.neumann(0.0)
        return FieldBoundaryCondition.dirichlet(value)

    x_lower, x_upper, y_lower, y_upper = (bc(v) for v in faces)
    return FieldBCs(x_lower=x_lower, x_upper=x_upper, y_lower=y_lower, y_upper=y_upper)


#: How far the beamtracer shapes are grown (non-strict IBSimu inside tests, ``>=``) or
#: shrunk (strict ones, ``>``) so that a node exactly on a surface is classified as IBSimu
#: classifies it [m]; far below the 1/255 of a cell to which the surface is located.
EPS = 1.0e-9


def offset(
    vertices: list[tuple[float, float]], grow: float
) -> list[tuple[float, float]]:
    """Return a polygon with every edge moved outwards by ``grow`` [m] (inwards if negative).

    Each vertex goes to the intersection of its two moved edges (a miter join), which for
    the tiny offsets used here keeps the shape otherwise unchanged.
    """
    p = np.asarray(vertices, dtype=float)
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    orientation = (
        1.0 if area > 0.0 else -1.0
    )  # counterclockwise: outward normal (dy, -dx)
    edges = np.roll(p, -1, axis=0) - p
    normals = orientation * np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    previous = np.roll(normals, 1, axis=0)  # the edge ending at each vertex
    bisector = previous + normals
    scale = grow / np.sum(bisector * normals, axis=1)
    moved = p + bisector * scale[:, None]
    return [(float(a), float(b)) for a, b in moved]


def polygon(
    vertices: list[tuple[float, float]], potential: float, strict: bool | None = False
) -> Any:
    """Return a polygonal electrode at ``potential``.

    ``strict`` mirrors an IBSimu inside test with strict inequalities: the shape is shrunk
    by `EPS`, so nodes on its surface are free; ``False`` grows it by `EPS`, so they are
    inside; ``None`` takes the vertices as they are (offset by the caller).
    """
    from beamtracer.sections import ObjectParams

    return ObjectParams(
        type="polygon",
        vertices=tuple(
            vertices if strict is None else offset(vertices, -EPS if strict else EPS)
        ),
        bc_type="dirichlet",
        bc_value=potential,
    )


def rectangle(
    x0: float, x1: float, y0: float, y1: float, potential: float, strict: bool = False
) -> Any:
    """Return a rectangular electrode ``[x0, x1] x [y0, y1]`` at ``potential`` (see `polygon`)."""
    return polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], potential, strict)


def beamtracer_beam(case: Case, particles: int, beam: dict[str, Any]) -> Any:
    """Return the case's beam (see `ibsimu_beam`) as a beamtracer species."""
    from beamtracer import BeamSpecies

    return BeamSpecies(
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


def beamtracer_run(
    case: Case,
    nodes: tuple[int, int],
    h: float,
    particles: int,
    integrator: str,
    region: Region,
    *,
    objects: dict[str, Any],
    faces: list[float | None],
    beam: dict[str, Any],
    **config_args: Any,
) -> Outcome:
    """Build and run a beamtracer `BeamOptics` model of a case and return its outcome.

    Parameters
    ----------
    case : Case
        The case.
    nodes, h : tuple of int, float
        The mesh.
    particles : int
        Trajectories per iteration.
    integrator : str
        ``"rk4"`` or ``"boris"``.
    region : callable
        Profiling regions: ``setup`` and ``run``.
    objects : dict
        The electrodes.
    faces : list
        The box conditions, as `beamtracer_faces` takes them.
    beam : dict
        The beam, as `beamtracer_beam` takes it.
    **config_args
        Further `BeamOpticsConfig` arguments (a plasma, magnetic fields).

    Returns:
    -------
    Outcome
        The potential and the beam crossing the exit face.
    """
    from beamtracer import BeamOptics, BeamOpticsConfig

    x_max = (nodes[0] - 1) * h
    plane = x_max * (1.0 - 1e-9)
    with region("setup"):
        config = BeamOpticsConfig(
            grid=beamtracer_grid(case, nodes, h),
            objects=objects,
            bcs_field=beamtracer_faces(faces),
            beams=[beamtracer_beam(case, particles, beam)],
            integrator=integrator,
            iterations=case.iterations,
            tolerance=0.0,
            diagnostic_planes=[plane],
            **config_args,
        )
        model = BeamOptics(config)
    with region("run"):
        model.run()
    hits = model.plane(plane)
    forward = hits.direction > 0
    point, velocity = hits.point[:, forward], hits.velocity[:, forward]
    if case.cylindrical:
        r = np.hypot(point[1], point[2])
        safe = np.where(r > 0.0, r, 1.0)
        vr = np.where(
            r > 0.0,
            (point[1] * velocity[1] + point[2] * velocity[2]) / safe,
            velocity[1],
        )
        u, up = r, vr / velocity[0]
    else:
        u, up = point[1], velocity[1] / velocity[0]
    exit_beam = Exit(
        u,
        up,
        hits.current[forward],
        kinetic_energy(case.mass, np.sum(velocity**2, axis=0)),
    )
    return Outcome(np.asarray(model.potential), exit_beam)


@contextlib.contextmanager
def no_region(_: str) -> Iterator[None]:
    """Time nothing: a stand-in for the runner's profiling regions."""
    yield
