"""Steady ion beams in a self-consistent field.

The model solves the beam once: a Vlasov iteration of field solve (with Boltzmann plasma
electrons for positive-ion extraction, Boltzmann plasma ions for negative-ion
extraction), trajectory tracing through the electric and solenoid fields, and
space-charge deposition, until the potential stops changing. See
:mod:`beamtracer.solver` for the numerics.

Run it with :meth:`BeamOptics.run`::

    model = BeamOptics(config).run()
    model.beam_moments(x_exit)
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import cunumpy as xp
import numpy as np
import scope_profiler as sp
from maybempi import MPI
from scipy import constants

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.config import BeamOpticsConfig, BeamSpecies, NegativeExtractionPlasma
from beamtracer.output import NodeField
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

_FACES = {
    "x_min": ("x", "min"),
    "x_max": ("x", "max"),
    "y_min": ("y", "min"),
    "y_max": ("y", "max"),
}


def _electrode(name: str, params: ObjectParams) -> Electrode:
    """Return the :class:`Electrode` of an object with a Dirichlet potential."""
    if params.bc_type != "dirichlet" or params.bc_value is None:
        raise ValueError("BeamOptics electrodes need a Dirichlet potential")
    obj = params.build()

    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # the mesh is classified on the host, whatever the active backend
        with xp.use_backend("numpy"):
            return np.asarray(obj.contains_points(np.stack([x, y], axis=0)), dtype=bool)

    return Electrode(inside, float(params.bc_value), name)


class BeamOptics:
    """Steady ion beams in planar or cylindrical geometry.

    Extraction of positive or negative ions from a plasma, transport through electrodes
    and solenoids.

    :meth:`run` is the self-consistent beam solve. With MPI the trajectories are shared
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

    def __init__(
        self,
        config: BeamOpticsConfig | None = None,
        /,
        *,
        comm: MPI.Comm | None = None,
        verbose: bool = False,
        **settings: object,
    ) -> None:
        """Build the mesh, the magnetic field and the tracer.

        ``settings`` are fields of :class:`BeamOpticsConfig`: without ``config`` they
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
        self.mesh = IonOpticsMesh(
            shape=grid.num_gridpoints,
            origin=grid.lower_bound,
            h=grid.deltas[0],
            geometry="cylindrical" if grid.coordinates == "cylindrical" else "planar",
            electrodes=[_electrode(name, obj) for name, obj in config.objects.items()],
            faces=self._faces(),
        )
        self.axial_magnetic_field = self._magnetic_field()
        self.tracer = BeamTracer(
            self.mesh,
            None,
            self.axial_magnetic_field,
            step_fraction=config.step_fraction,
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
        self.species_names = [beam.name for beam in config.beams]
        self.species_mass = np.array(
            [beam.mass * constants.atomic_mass for beam in config.beams]
        )
        self._state: VlasovState | None = None
        self._is_finished = False
        self._fields: dict[tuple[str, bool], NodeField] = {}

    def _particles(self) -> tuple[BeamParticles, float]:
        """Return this rank's share of the beams and the total ``sum J / v``."""
        parts = []
        j_over_v = 0.0
        cylindrical = self.mesh.cylindrical
        for number, beam in enumerate(self.config.beams):
            if isinstance(beam, BeamSpecies):
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
        """Sum an array over the ranks."""
        return self.comm.allreduce(array, op=MPI.SUM)

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
        :class:`~beamtracer.solver.VlasovState` (with MPI, this rank's share of the
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
        self._state = vlasov_iterate(
            self.mesh,
            self._particles,
            self.tracer,
            iterations=config.iterations,
            plasma=plasma,
            compensation=compensation,
            tolerance=config.tolerance,
            relaxation=config.relaxation,
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
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(E_x, E_y)`` [V/m] at the points ``(x, y)``.

        ``y`` is ``r`` in cylindrical geometry.
        """
        return self.state.electric(x, y)

    def surface_fields(self) -> dict[str, float]:
        """Return the largest ``|E|`` [V/m] on the surface of every electrode, by name.

        From the solved potential at the free nodes next to each electrode (see
        :func:`~beamtracer.solver.surface_field`); for voltage holding, e.g. a limit of
        some kV/mm. The field at a sharp corner is singular and grows as the mesh is
        refined: round the edges whose field matters.
        """
        magnitude, electrode = surface_field(self.mesh, self.state.electric)
        fields = {}
        for number, name in enumerate(self.config.objects):
            mine = electrode == number
            fields[name] = float(np.max(magnitude[mine])) if mine.any() else 0.0
        return fields

    def _field(self, name: str, data: np.ndarray) -> NodeField:
        key = (name, self._state is not None)
        if key not in self._fields:
            grid = self.config.grid
            self._fields[key] = NodeField(data, grid.lower_bound, grid.upper_bound)
        return self._fields[key]

    @property
    def potential_field(self) -> NodeField:
        """Return the potential [V] as a :class:`~beamtracer.output.NodeField`.

        It is zero before the run.
        """
        data = np.zeros(self.mesh.shape) if self._state is None else self.potential
        return self._field("potential", data)

    @property
    def charge_density(self) -> NodeField:
        """Return the beam's (compensated) charge density [C/m^3].

        A :class:`~beamtracer.output.NodeField`, zero before the run.
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

        See :class:`~beamtracer.solver.charge_changing.ChargeTallies`.
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
        changing = self.tracer.charge_changing
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

    def beam_moments(self, x: float, species: str | None = None) -> BeamMoments:
        """Return the current, emittance and Twiss parameters at the plane ``x``.

        Of one beam (by name) or of all.
        """
        index = None if species is None else self.species_names.index(species)
        return plane_moments(
            self.plane(x),
            cylindrical=self.mesh.cylindrical,
            mass=self.species_mass,
            species=index,
        )

    def write_files(self, path: Path) -> None:
        """Write the fields, the trajectories and the plane crossings as a pickle."""
        with open(Path(path) / "beam_optics.pkl", "wb") as file:
            pickle.dump(
                {
                    "x": self.mesh.x,
                    "y": self.mesh.y,
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
