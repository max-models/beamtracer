"""Steady ion beams in a self-consistent field.

The model solves the beam once: a Vlasov iteration of field solve (with Boltzmann plasma
electrons for positive-ion extraction, Boltzmann plasma ions for negative-ion
extraction), trajectory tracing through the electric and solenoid fields, and
space-charge deposition, until the potential stops changing. See
`beamtracer.solver` for the numerics.

Run it with `BeamOptics.run`::

    model = BeamOptics(config).run()
    model.beam_moments(x_exit)
"""

from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from maybempi import MPI
from methodcard import method
from mpifield import Field
from scipy import constants

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.config import (
    BeamOpticsConfig,
    BeamSpecies,
    BeamSpecies3D,
    NegativeExtractionPlasma,
)
from beamtracer.field_map import FieldMap
from beamtracer.sections import ObjectParams
from beamtracer.solver import (
    STATUS_NAMES,
    AxisymmetricMagneticField,
    BeamMoments,
    BeamParticles,
    BeamTracer,
    Compensation,
    Electrode,
    FaceCondition,
    IonOpticsMesh,
    NegativeIonPlasmaModel,
    PlaneHits,
    PlasmaModel,
    VlasovState,
    beam_with_energy,
    plane_moments,
    solenoid_on_axis,
    surface_field,
    vlasov_iterate,
)
from beamtracer.solver.adaptive import AdaptiveRefinement
from beamtracer.solver.mesh3d import FACES_3D, Electrode3D, IonOpticsMesh3D
from beamtracer.solver.refinement import Refinement
from beamtracer.solver.tracer3d import BeamTracer3D, beam_on_surface

_FACES = {
    "x_min": ("x", "min"),
    "x_max": ("x", "max"),
    "y_min": ("y", "min"),
    "y_max": ("y", "max"),
}


def _electrode(name: str, params: ObjectParams) -> Electrode:
    """Return the `Electrode` of an object with a Dirichlet potential."""
    if params.bc_type != "dirichlet" or params.bc_value is None:
        raise ValueError("BeamOptics electrodes need a Dirichlet potential")
    obj = params.build()

    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # the mesh is classified on the host, whatever the active backend
        with xp.use_backend("numpy"):
            return np.asarray(obj.contains_points(np.stack([x, y], axis=0)), dtype=bool)

    return Electrode(inside, float(params.bc_value), name)


def _electrode_3d(name: str, params: ObjectParams) -> Electrode3D:
    """Return the `Electrode3D` of a 3D object with a Dirichlet potential."""
    if params.bc_type != "dirichlet" or params.bc_value is None:
        raise ValueError("BeamOptics electrodes need a Dirichlet potential")
    obj = params.build()

    def inside(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        with xp.use_backend("numpy"):
            points = np.stack([x, y, z], axis=0)
            return np.asarray(obj.contains_points(points), dtype=bool)

    return Electrode3D(inside, float(params.bc_value), name)


@method(
    equations={
        "poisson": r"\nabla^2 \phi = -\frac{\rho_b + \rho_p}{\varepsilon_0}",
        "plasma": r"\rho_p = s\, \rho_0\, e^{-s (\phi - U_p) / T}, \qquad \rho_0 = \sum_\mathrm{beams} \frac{J}{v_0}",
        "motion": r"\frac{d\mathbf x}{dt} = \mathbf v, \qquad \frac{d\mathbf v}{dt} = \frac{q}{m}\left(\mathbf E + \mathbf v \times \mathbf B\right), \qquad \mathbf E = -\nabla\phi",
        "space_charge": r"\rho_b = \frac{1}{V_\mathrm{node}} \sum_\text{steps} \operatorname{sign}(q)\, I\, \Delta t \times (1 - f)",
    },
    assumptions=(
        r"steady: the beam is the fixed point of field solve, trace and deposit (`time` is always 0)",
        r"planar $(x, y)$ per unit depth or cylindrically symmetric $(x, r)$; the solenoids and a measured axial profile only in cylindrical geometry",
        r"an imposed static $\mathbf E$ and/or $\mathbf B$ from a field map (`beamtracer.field_map`) in the frame of the grid, added to the solved field and the solenoids",
        r"a Boltzmann plasma species: electrons ($s = -1$) for positive-ion extraction, positive ions ($s = +1$) for negative-ion extraction; the extracted species are traced beams",
        r"configuration in e, u and eV, converted to C, kg and J; with MPI the trajectories are shared between the ranks and every rank solves the field",
    ),
    components=(
        "beamtracer.solver.mesh.IonOpticsMesh",
        "beamtracer.solver.vlasov.vlasov_iterate",
        "beamtracer.solver.tracer.beam_with_energy",
        "beamtracer.solver.fields.AxisymmetricMagneticField",
        "beamtracer.solver.fields.solenoid_on_axis",
        "beamtracer.field_map.FieldMap",
        "beamtracer.solver.diagnostics.plane_moments",
    ),
)
class BeamOptics:
    """Steady ion beams in planar or cylindrical geometry.

    Extraction of positive or negative ions from a plasma, transport through electrodes
    and solenoids.

    `run` is the self-consistent beam solve. With MPI the trajectories are shared
    between the ranks, and every rank solves the field.
    """

    name = "BeamOptics"
    Config = BeamOpticsConfig

    # -- setup ------------------------------------------------------------------------------

    def _faces(self) -> dict[str, FaceCondition]:
        """Return the mesh face conditions from the field walls."""
        faces = {}
        cylindrical = self.config.grid.coordinates == "cylindrical"
        walls = self.config.bcs_field
        assert walls is not None  # filled in by the configuration
        for face, (direction, side) in _FACES.items():
            if cylindrical and face == "y_min":
                continue  # the symmetry axis
            bc = walls.get_bc(dir=direction, side=side)
            if bc.bc_type == "periodic":
                raise ValueError("BeamOptics does not support periodic walls")
            if self.config.symmetry_plane and face == "y_min":
                samples = np.linspace(0.0, 1.0, 5)
                derivative = bc.evaluate(
                    x=samples, y=0.0 * samples, z=0.0 * samples, t=0.0
                )
                if bc.bc_type != "neumann" or np.any(np.asarray(derivative) != 0.0):
                    raise ValueError(
                        "with symmetry_plane the y lower face needs a zero normal derivative"
                    )
                continue  # the symmetry plane: the mesh's default face

            def value(x, y, bc=bc):
                return np.asarray(bc.evaluate(x=x, y=y, z=0.0 * x, t=0.0), dtype=float)

            faces[face] = FaceCondition(bc.bc_type, value)
        return faces

    def _magnetic_field(self) -> AxisymmetricMagneticField | None:
        """Return the axisymmetric magnetic field of the solenoids and the axial field."""
        config = self.config
        if not config.solenoids and config.axial_field is None:
            return None
        if config.axial_field is not None:
            z0, dz = config.axial_field.z0, config.axial_field.dz
            bz = np.asarray(config.axial_field.bz, dtype=float)
            z = z0 + dz * np.arange(bz.size)
        else:
            dz = 0.5 * self.mesh.h
            lo, hi = self.mesh.origin[0], self.mesh.upper[0]
            z = np.arange(lo - 8 * dz, hi + 8.5 * dz, dz)
            z0 = float(z[0])
            bz = np.zeros(z.size)
        for solenoid in config.solenoids:
            bz = bz + solenoid_on_axis(
                z, solenoid.centre, solenoid.length, solenoid.radius, solenoid.peak
            )
        return AxisymmetricMagneticField(z0, dz, bz, order=config.field_order)

    def _field_map(self) -> FieldMap | None:
        """Read the field map of the configuration and check it fits the grid."""
        params = self.config.field_map
        if params is None:
            return None
        field_map = params.read()
        frame = "cylindrical" if self.mesh.cylindrical else "cartesian"
        if field_map.frame != frame:
            raise ValueError(
                f"the field map {params.file} is in the {field_map.frame!r} frame; "
                f"this grid needs {frame!r}"
            )
        three = self.config.grid.ndim == 3
        if ("z" in field_map.dims) != three and not (three and len(field_map.dims) < 3):
            raise ValueError(
                f"the field map {params.file} is on {field_map.dims}; "
                f"this grid needs one on {'(x, y, z)' if three else '(x[, y])'}"
            )
        if three and field_map.components("electric"):
            raise ValueError("an imposed electric field is not available in 3D yet")
        return field_map

    def _imposed_fields(self) -> tuple[Callable | None, Callable | None]:
        """The imposed E of the field map and the B of the solenoids and the map.

        As the tracer takes them: functions of ``(x, y)`` or ``(x, r)``.
        """
        field_map = self.field_map
        axial = self.axial_magnetic_field
        if field_map is None:
            return None, axial
        electric = (
            field_map.plane_function("electric")
            if field_map.components("electric")
            else None
        )
        if not field_map.components("magnetic"):
            return electric, axial
        mapped = field_map.plane_function("magnetic")
        if axial is None:
            return electric, mapped

        def magnetic(x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, ...]:
            bx, br, btheta = mapped(x, r)
            sx, sr = axial(x, r)
            return bx + sx, br + sr, btheta

        return electric, magnetic

    def _faces_3d(self) -> dict[str, FaceCondition]:
        """Return the 3D mesh's face conditions from the field walls.

        The symmetry planes keep the mesh's default (a zero normal derivative), after
        checking the walls give that.
        """
        faces = {}
        walls = self.config.bcs_field
        assert walls is not None  # filled in by the configuration
        for face in FACES_3D:
            direction, side = face.split("_")
            bc = walls.get_bc(dir=direction, side=side)
            if bc.bc_type == "periodic":
                raise ValueError("BeamOptics does not support periodic walls")

            def value(x, y, z, bc=bc):
                return np.asarray(bc.evaluate(x=x, y=y, z=z, t=0.0 * x), dtype=float)

            if face in self.config.symmetry_planes:
                samples = np.linspace(0.0, 1.0, 5)
                if bc.bc_type != "neumann" or np.any(value(samples, samples, samples)):
                    raise ValueError(
                        f"the symmetry plane {face} needs a zero normal derivative"
                    )
                continue
            faces[face] = FaceCondition(bc.bc_type, value)
        return faces

    def _magnetic_nodes(self) -> np.ndarray | None:
        """B [T] on the 3D mesh's nodes ``(3, nx, ny, nz)``: solenoids and a field map.

        The solenoids and the axial profile are axisymmetric about the ``x`` axis.
        """
        mesh = self.mesh
        axial = self.axial_magnetic_field
        field_map = self.field_map
        if axial is None and (
            field_map is None or not field_map.components("magnetic")
        ):
            return None
        assert isinstance(mesh, IonOpticsMesh3D)
        x, y, z = mesh.nodes()
        b = np.zeros((3, *mesh.shape))
        if axial is not None:
            r = np.hypot(y, z)
            bx, br = axial(x.ravel(), r.ravel())
            safe = np.where(r > 0.0, r, 1.0)
            b[0] += bx.reshape(mesh.shape)
            b[1] += (br.reshape(mesh.shape) * y / safe) * (r > 0.0)
            b[2] += (br.reshape(mesh.shape) * z / safe) * (r > 0.0)
        if field_map is not None and field_map.components("magnetic"):
            b += np.moveaxis(field_map.evaluate((x, y, z), "magnetic"), -1, 0)
        return b

    def __init__(
        self,
        config: BeamOpticsConfig | None = None,
        /,
        *,
        comm: MPI.Comm | None = None,
        verbose: bool = False,
        **settings: object,
    ) -> None:
        """Build the mesh, the imposed fields (solenoids, a field map) and the tracer.

        ``settings`` are fields of `BeamOpticsConfig`: without ``config`` they
        build it, with it they replace its fields. ``comm`` defaults to
        ``MPI.COMM_WORLD``.
        """
        if config is None:
            config = BeamOpticsConfig(**settings)  # type: ignore[arg-type]
        else:
            if type(config) is not BeamOpticsConfig:
                raise TypeError(
                    f"BeamOptics needs a BeamOpticsConfig, got {type(config).__name__}"
                )
            if settings:
                config = config.replace(**settings)
        self.config = config
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = 0 if self.comm is None else self.comm.Get_rank()
        self.verbose = verbose
        grid = config.grid
        self.species_names = [beam.name for beam in config.beams]
        self.species_mass = np.array(
            [beam.mass * constants.atomic_mass for beam in config.beams]
        )
        self._state: VlasovState | None = None
        self._is_finished = False
        self._fields: dict[tuple[str, bool], Field] = {}
        if grid.ndim == 3:
            self.mesh: IonOpticsMesh | IonOpticsMesh3D = IonOpticsMesh3D(
                shape=grid.num_gridpoints,
                origin=grid.lower_bound,
                h=grid.deltas[0],
                electrodes=[
                    _electrode_3d(name, obj) for name, obj in config.objects.items()
                ],
                faces=self._faces_3d(),
                symmetry_planes=tuple(config.symmetry_planes),
            )
            self.axial_magnetic_field = self._magnetic_field()
            self.field_map = self._field_map()
            self.tracer: BeamTracer | BeamTracer3D = BeamTracer3D(
                self.mesh,
                None,
                self._magnetic_nodes(),
                step_fraction=config.step_fraction,
                integrator=config.integrator,
                relativistic=config.relativistic,
                max_steps=config.max_steps,
                planes=config.diagnostic_planes,
                save_trajectories=config.save_trajectories,
            )
            return
        self.mesh = IonOpticsMesh(
            shape=grid.num_gridpoints,
            origin=grid.lower_bound,
            # along a graded x the spacing of y (the nodes along x are given)
            h=grid.deltas[1] if grid.graded else grid.deltas[0],
            x_nodes=grid.x_nodes,
            geometry="cylindrical" if grid.coordinates == "cylindrical" else "planar",
            electrodes=[_electrode(name, obj) for name, obj in config.objects.items()],
            faces=self._faces(),
            symmetry_plane=config.symmetry_plane,
        )
        self.axial_magnetic_field = self._magnetic_field()
        self.field_map = self._field_map()
        imposed_electric, magnetic = self._imposed_fields()
        self.tracer = BeamTracer(
            self.mesh,
            None,
            magnetic,
            imposed_electric=imposed_electric,
            step_fraction=config.step_fraction,
            integrator=config.integrator,
            relativistic=config.relativistic,
            max_steps=config.max_steps,
            planes=config.diagnostic_planes,
            save_trajectories=config.save_trajectories,
            charge_changing=(
                None
                if config.charge_changing is None
                else config.charge_changing.build()
            ),
        )
        self.tracer.stream = self.rank  # every rank its own analog charge changes

    def _particles(self) -> tuple[BeamParticles, float]:
        """Return this rank's share of the beams and the total ``sum J / v``."""
        parts = []
        j_over_v = 0.0
        cylindrical = self.mesh.cylindrical
        for number, beam in enumerate(self.config.beams):
            if isinstance(beam, BeamSpecies3D):
                particles, jv = beam_on_surface(
                    beam.num_particles,
                    beam.current_density,
                    beam.charge,
                    beam.mass,
                    beam.energy,
                    beam.parallel_temperature,
                    beam.transverse_temperature,
                    beam.centre,
                    beam.size,
                    beam.shape,
                    species=number,
                    relativistic=self.config.relativistic,
                )
                j_over_v += jv
            elif isinstance(beam, BeamSpecies):
                particles, jv = beam_with_energy(
                    beam.num_particles,
                    beam.current_density,
                    beam.charge,
                    beam.mass,
                    beam.energy,
                    beam.parallel_temperature,
                    beam.transverse_temperature,
                    beam.start,
                    beam.end,
                    cylindrical=cylindrical,
                    species=number,
                    relativistic=self.config.relativistic,
                )
                j_over_v += jv
            else:
                n = len(beam)
                # The configuration lists one trajectory per row; the tracer
                # wants the components first.
                particles = BeamParticles(
                    position=np.ascontiguousarray(
                        np.asarray(beam.position, dtype=float).reshape(n, 3).T
                    ),
                    velocity=np.ascontiguousarray(
                        np.asarray(beam.velocity, dtype=float).reshape(n, 3).T
                    ),
                    charge=np.full(n, beam.charge * constants.e),
                    mass=np.full(n, beam.mass * constants.atomic_mass),
                    current=np.asarray(beam.current, dtype=float),
                    species=np.full(n, number, dtype=np.int32),
                )
            parts.append(particles)
        if not parts:
            raise ValueError("BeamOptics needs at least one beam")
        particles = BeamParticles.concatenate(parts)
        size = 1 if self.comm is None else self.comm.Get_size()
        if size > 1:
            particles = particles.subset(slice(self.rank, None, size))
        return particles, j_over_v

    @property
    def _parallel(self) -> bool:
        return self.comm is not None and self.comm.Get_size() > 1

    def _allreduce(self, array: np.ndarray) -> np.ndarray:
        """Sum a float array over the ranks (a buffer ``Allreduce``, not pickled)."""
        array = np.ascontiguousarray(array, dtype=float)
        total = np.empty_like(array)
        self.comm.Allreduce(array, total, op=MPI.SUM)
        return total

    def _gather_results(self) -> None:
        """Give every rank the plane crossings, statuses and saved trajectories."""
        trace = self.state.trace
        for k, hits in enumerate(trace.planes):
            parts = self.comm.allgather(
                (
                    hits.point,
                    hits.velocity,
                    hits.current,
                    hits.species,
                    hits.direction,
                    hits.fractions,
                )
            )
            merged = PlaneHits(hits.position_x)
            for part in parts:
                merged.append(*part)
            trace.planes[k] = merged
        if trace.tallies is not None:
            tallies = self.comm.allgather(trace.tallies)
            merged_tallies = tallies[0]
            for other in tallies[1:]:
                merged_tallies.merge(other)
            trace.tallies = merged_tallies
            trace.fractions = np.concatenate(
                self.comm.allgather(trace.fractions), axis=1
            )
        trace.status = np.concatenate(self.comm.allgather(trace.status))
        trace.electrode = np.concatenate(self.comm.allgather(trace.electrode))
        trace.trajectories = [
            t for part in self.comm.allgather(trace.trajectories) for t in part
        ]
        trace.trajectory_species = np.concatenate(
            self.comm.allgather(trace.trajectory_species)
        )

    # -- running ----------------------------------------------------------------------------

    @sp.profile("beamtracer.run", tags=PROFILE_TAGS)
    def run(
        self, callback: Callable[[int, VlasovState], None] | None = None
    ) -> BeamOptics:
        """Run the Vlasov iteration to convergence (once); return the model.

        ``callback(iteration, state)`` is called after every iteration with its
        `beamtracer.solver.VlasovState` (with MPI, this rank's share of the
        trajectories), e.g. to follow a beam moment from iteration to iteration.
        """
        if self._is_finished:
            return self
        config = self.config
        plasma = None
        if isinstance(config.plasma, NegativeExtractionPlasma):
            plasma = NegativeIonPlasmaModel(
                config.plasma.ion_temperature,
                config.plasma.plasma_potential,
                config.plasma.initial_plasma_x,
                config.plasma.rho_p,
            )
        elif config.plasma is not None:
            plasma = PlasmaModel(
                config.plasma.electron_temperature,
                config.plasma.plasma_potential,
                config.plasma.initial_plasma_x,
                config.plasma.rho_e,
            )
        compensation = [
            Compensation(c.x_min, c.x_max, c.fraction) for c in config.compensation
        ]
        self.tracer.reduce = self._allreduce if self._parallel else None
        if self._parallel:
            self.tracer.ranks = (self.rank, self.comm.Get_size())
            self.tracer.allgather = self.comm.allgather
        self._state = vlasov_iterate(
            self.mesh,
            self._particles,
            self.tracer,
            iterations=config.iterations,
            plasma=plasma,
            compensation=compensation,
            tolerance=config.tolerance,
            relaxation=config.relaxation,
            mixing=config.mixing,
            anderson_depth=config.anderson_depth,
            coarse_particles=config.coarse_particles,
            coarse_tolerance=config.coarse_tolerance,
            field_backend=config.field_backend,
            refinement=[
                Refinement(r.level, r.lower, r.upper) for r in config.refinement
            ],
            adaptive=(
                None
                if config.adaptive_refinement is None
                else AdaptiveRefinement(
                    **dataclasses.asdict(config.adaptive_refinement)
                )
            ),
            comm=self.comm if self._parallel else None,
            self_magnetic=config.self_magnetic_field,
            callback=callback,
        )
        if self._parallel:
            self._gather_results()
        self._fields = {}
        self._is_finished = True
        return self

    # -- results ----------------------------------------------------------------------------

    @property
    def state(self) -> VlasovState:
        """Return the converged state (after the run)."""
        if self._state is None:
            raise RuntimeError("BeamOptics has not run yet")
        return self._state

    @property
    def potential(self) -> np.ndarray:
        """Return the potential [V] at the nodes, shape ``(nx, ny)``."""
        return self.state.potential

    def electric_field_at(
        self, x: np.ndarray, y: np.ndarray, *z: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Return ``(E_x, E_y)`` [V/m] at the points ``(x, y)``, ``(E_x, E_y, E_z)`` in 3D.

        ``y`` is ``r`` in cylindrical geometry; a 3D grid takes ``(x, y, z)``.
        """
        return self.state.electric(x, y, *z)  # pyright: ignore[reportCallIssue]

    def surface_fields(self) -> dict[str, float]:
        """Return the largest ``|E|`` [V/m] on the surface of every electrode, by name (2D).

        From the solved potential at the free nodes next to each electrode (see
        `beamtracer.solver.surface_field`); for voltage holding, e.g. a limit of
        some kV/mm. The field at a sharp corner is singular and grows as the mesh is
        refined: round the edges whose field matters.
        """
        if isinstance(self.mesh, IonOpticsMesh3D):
            raise NotImplementedError("surface_fields is not available in 3D yet")
        magnitude, electrode = surface_field(self.mesh, self.state.electric)
        fields = {}
        for number, name in enumerate(self.config.objects):
            mine = electrode == number
            fields[name] = float(np.max(magnitude[mine])) if mine.any() else 0.0
        return fields

    def _field(self, name: str, data: np.ndarray) -> Field:
        key = (name, self._state is not None)
        if key not in self._fields:
            grid = self.config.grid
            self._fields[key] = Field.from_array(
                data, grid.lower_bound, grid.upper_bound
            )
        return self._fields[key]

    @property
    def potential_field(self) -> Field:
        """Return the potential [V] as a `mpifield.Field` on the nodes.

        It is zero before the run.
        """
        data = np.zeros(self.mesh.shape) if self._state is None else self.potential
        return self._field("potential", data)

    @property
    def charge_density(self) -> Field:
        """Return the beam's (compensated) charge density [C/m^3].

        A `mpifield.Field` on the nodes, zero before the run.
        """
        data = (
            np.zeros(self.mesh.shape)
            if self._state is None
            else self.state.charge_density
        )
        return self._field("charge_density", data)

    @property
    def trajectories(self) -> list[np.ndarray]:
        """Return the saved trajectories, each ``(n, 2)`` points ``(x, y)`` or ``(x, r)``."""
        return self.state.trace.trajectories

    @property
    def trajectory_species(self) -> np.ndarray:
        """Return the beam index of every saved trajectory."""
        return self.state.trace.trajectory_species

    @property
    def convergence_history(self) -> list[dict[str, float]]:
        """Return, per iteration, the largest potential change [V] and solver counts."""
        return self.state.history

    @property
    def stats(self) -> dict[str, int]:
        """Return the number of trajectories per final status, and the iteration count."""
        status = self.state.trace.status
        stats = {name: int(np.sum(status == k)) for k, name in enumerate(STATUS_NAMES)}
        stats["launched"] = int(status.size)
        stats["iterations"] = len(self.state.history)
        stats["converged"] = int(
            self.state.history[-1]["max_potential_change"] < self.config.tolerance
        )
        return stats

    @property
    def charge_tallies(self):
        """Return what the beam did to its targets with charge changes, else ``None``.

        See `beamtracer.solver.charge_changing.ChargeTallies`.
        """
        return self.state.trace.tallies

    @property
    def final_fractions(self) -> np.ndarray | None:
        """With charge changes, every trajectory's charge-state fractions where it ended."""
        return self.state.trace.fractions

    def state_moments(self, x: float, state: str) -> BeamMoments:
        """Return the moments at the plane ``x`` of the particles in one charge state.

        The trajectories are weighted by their fraction in it.
        """
        changing = getattr(self.tracer, "charge_changing", None)
        if changing is None:
            raise ValueError("the beam has no charge changes")
        hits = self.plane(x)
        assert hits.fractions is not None  # recorded with charge changes
        weighted = PlaneHits(hits.position_x)
        share = hits.fractions[changing.states.index(state)]
        weighted.append(
            hits.point,
            hits.velocity,
            hits.current * share,
            hits.species,
            hits.direction,
        )
        return plane_moments(
            weighted, cylindrical=self.mesh.cylindrical, mass=self.species_mass
        )

    def plane(self, x: float) -> PlaneHits:
        """Return the crossings of the diagnostic plane at ``x``."""
        for hits in self.state.trace.planes:
            if np.isclose(hits.position_x, x):
                return hits
        raise KeyError(f"no diagnostic plane at x = {x}")

    def beam_moments(
        self, x: float, species: str | None = None, axis: str = "y"
    ) -> BeamMoments:
        """Return the current, emittance and Twiss parameters at the plane ``x``.

        Of one beam (by name) or of all; in the ``y``-``y'`` plane, or with ``axis="z"``
        (3D) in ``z``-``z'``.
        """
        index = None if species is None else self.species_names.index(species)
        return plane_moments(
            self.plane(x),
            cylindrical=self.mesh.cylindrical,
            mass=self.species_mass,
            species=index,
            axis=axis,
        )

    def dataset(self) -> Any:
        """Return the potential and the charge density as an `xarray.Dataset` on the nodes.

        Coordinates ``x``, ``y`` (``r`` in cylindrical geometry) and, in 3D, ``z`` [m];
        the variables carry ``units`` and a ``long_name`` for the plots.
        """
        import xarray as xr

        mesh = self.mesh
        if isinstance(mesh, IonOpticsMesh3D):
            axes = {"x": mesh.x, "y": mesh.y, "z": mesh.z}
        else:
            axes = {"x": mesh.x, "r" if mesh.cylindrical else "y": mesh.y}
        dims = tuple(axes)
        coords = {
            name: (name, values, {"units": "m", "long_name": f"${name}$"})
            for name, values in axes.items()
        }
        data = {
            "potential": (dims, self.potential, {"units": "V", "long_name": r"$\phi$"}),
            "charge_density": (
                dims,
                self.state.charge_density,
                {"units": "C/m^3", "long_name": r"$\rho_b$"},
            ),
        }
        return xr.Dataset(data, coords=coords)

    def write_files(self, path: Path) -> None:
        """Write the fields, the trajectories and the plane crossings as a pickle."""
        with open(Path(path) / "beam_optics.pkl", "wb") as file:
            pickle.dump(
                {
                    "x": self.mesh.x,
                    "y": self.mesh.y,
                    **(
                        {"z": self.mesh.z}
                        if isinstance(self.mesh, IonOpticsMesh3D)
                        else {}
                    ),
                    "potential": self.potential,
                    "charge_density": self.state.charge_density,
                    "trajectories": self.trajectories,
                    "trajectory_species": self.trajectory_species,
                    "planes": self.state.trace.planes,
                    "species": self.species_names,
                    "stats": self.stats,
                    "convergence_history": self.convergence_history,
                },
                file,
            )

    @property
    def is_finished(self) -> bool:
        """Return whether the beam has been solved."""
        return self._is_finished

    @property
    def time(self) -> float:
        """Return the simulation time (always 0: the beam is steady)."""
        return 0.0

    @property
    def iteration(self) -> int:
        """Return 1 once the beam is solved, else 0."""
        return int(self._is_finished)
