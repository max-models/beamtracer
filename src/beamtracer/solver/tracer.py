"""Steady-state trajectories through ion-optics fields.

Beams, tracing, space charge and diagnostic planes.

Particles move in 3D Cartesian coordinates ``(x, y, z)``. In cylindrical geometry the fields
depend on ``(x, r)`` with ``r = sqrt(y^2 + z^2)``: a trajectory stands for a ring of charge,
and the Cartesian form needs no special treatment of the axis. Each trajectory carries a
current ``I`` [A] (per unit depth [A/m] in planar geometry), a magnitude, and deposits the
charge ``sign(q) I dt`` at the midpoint of every step, with an area-weighted linear shape,
then the charge is divided by the node volumes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import scope_profiler
from scipy import constants

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.kernels.deposit_trajectories_2d import kernel as deposit_trajectories_2d
from beamtracer.solver.charge_changing import ChargeChanging, ChargeTallies
from beamtracer.solver.mesh import IonOpticsMesh

#: Status of a finished trajectory.
TRANSMITTED, BACKWARD, WALL, ELECTRODE, MAX_STEPS = 0, 1, 2, 3, 4
STATUS_NAMES = ("transmitted", "backward", "wall", "electrode", "max_steps")


@dataclass
class BeamParticles:
    """Particle starting states.

    Positions and velocities ``(3, N)``, charge [C], mass [kg], current [A] and species
    index per particle.
    """

    position: np.ndarray
    velocity: np.ndarray
    charge: np.ndarray
    mass: np.ndarray
    current: np.ndarray
    species: np.ndarray

    @staticmethod
    def concatenate(parts: Sequence[BeamParticles]) -> BeamParticles:
        """Join several particle sets."""
        return BeamParticles(
            np.concatenate([p.position for p in parts], axis=1),
            np.concatenate([p.velocity for p in parts], axis=1),
            *(
                np.concatenate([getattr(p, name) for p in parts])
                for name in ("charge", "mass", "current", "species")
            ),
        )

    def subset(self, index: np.ndarray | slice) -> BeamParticles:
        """Return the particles at ``index``."""
        return BeamParticles(
            self.position[:, index],
            self.velocity[:, index],
            self.charge[index],
            self.mass[index],
            self.current[index],
            self.species[index],
        )

    def __len__(self) -> int:
        """Return the number of particles."""
        return int(self.position.shape[1])


def _gaussian_samples(num: int, sampling: str | np.random.Generator) -> np.ndarray:
    """Return ``(num, 3)`` standard normal samples, quasi-random or random."""
    if isinstance(sampling, np.random.Generator):
        return sampling.standard_normal((num, 3))
    if sampling != "sobol":
        raise ValueError(
            f"sampling must be 'sobol' or a numpy Generator, got {sampling!r}"
        )
    from scipy.stats import norm, qmc

    sobol = qmc.Sobol(d=3, scramble=False)
    sobol.fast_forward(1)  # skip the origin, whose inverse Gaussian is infinite
    return norm.ppf(sobol.random(num))


def beam_with_energy(
    num_particles: int,
    current_density: float,
    charge: float,
    mass: float,
    energy: float,
    parallel_temperature: float,
    transverse_temperature: float,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    cylindrical: bool,
    species: int = 0,
    sampling: str | np.random.Generator = "sobol",
) -> tuple[BeamParticles, float]:
    """Seed a beam emitted from a line with a given energy; return it and ``J / v``.

    ``num_particles`` trajectories start evenly spaced on the line from ``start`` to ``end``
    (``(x, y)`` or ``(x, r)`` [m]), moving at ``energy`` [eV] perpendicular to it (to the
    right of the line direction), with Gaussian velocity spreads of ``parallel_temperature``
    along the motion and ``transverse_temperature`` across it (and, in cylindrical geometry,
    around the axis) [eV]. ``current_density`` [A/m^2], ``charge`` [e] and ``mass`` [u]. In
    cylindrical geometry the trajectory at radius ``r`` carries ``2 pi J s r / N`` (a ring),
    in planar geometry ``J s / N`` per unit depth, with ``s`` the line length.

    ``sampling`` ``"sobol"`` draws the velocity spreads from a 3D Sobol sequence through the
    inverse Gaussian distribution: the same beam on every call, so the
    Vlasov iteration reaches a fixed point. A :class:`numpy.random.Generator` draws them at
    random instead.
    """
    q = charge * constants.e
    m = mass * constants.atomic_mass
    v = np.sqrt(2.0 * energy * constants.e / m)
    dvp = np.sqrt(parallel_temperature * constants.e / m)
    dvt = np.sqrt(transverse_temperature * constants.e / m)
    (x1, y1), (x2, y2) = start, end
    s = float(np.hypot(x2 - x1, y2 - y1))
    if s == 0.0:
        raise ValueError("the beam start line has zero length")
    tangent = np.array([x2 - x1, y2 - y1]) / s
    normal = np.array([tangent[1], -tangent[0]])
    frac = (np.arange(num_particles) + 0.5) / num_particles
    px = x1 + (x2 - x1) * frac
    py = y1 + (y2 - y1) * frac
    g = _gaussian_samples(num_particles, sampling)
    vx = normal[0] * v + tangent[0] * dvt * g[:, 0] + normal[0] * dvp * g[:, 1]
    vy = normal[1] * v + tangent[1] * dvt * g[:, 0] + normal[1] * dvp * g[:, 1]
    vz = dvt * g[:, 2] if cylindrical else np.zeros(num_particles)
    if cylindrical:
        current = 2.0 * np.pi * current_density * s * py / num_particles
    else:
        current = np.full(num_particles, current_density * s / num_particles)
    particles = BeamParticles(
        position=np.stack([px, py, np.zeros(num_particles)], axis=0),
        velocity=np.stack([vx, vy, vz], axis=0),
        charge=np.full(num_particles, q),
        mass=np.full(num_particles, m),
        current=current,
        species=np.full(num_particles, species, dtype=np.int32),
    )
    return particles, current_density / v


@dataclass
class PlaneHits:
    """Crossings of a plane ``x = position``.

    The crossing point and velocity ``(3, N)``, current, species index and the direction of
    the crossing (+1 or -1); with charge changes also the charge-state fractions
    ``(states, N)``.
    """

    position_x: float
    point: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    current: np.ndarray = field(default_factory=lambda: np.zeros(0))
    species: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    direction: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    fractions: np.ndarray | None = None

    def append(
        self, point, velocity, current, species, direction, fractions=None
    ) -> None:
        """Add crossings."""
        self.point = np.concatenate([self.point, point], axis=1)
        self.velocity = np.concatenate([self.velocity, velocity], axis=1)
        self.current = np.concatenate([self.current, current])
        self.species = np.concatenate([self.species, species.astype(np.int32)])
        self.direction = np.concatenate([self.direction, direction.astype(np.int8)])
        if fractions is not None:
            self.fractions = (
                fractions
                if self.fractions is None
                else np.concatenate([self.fractions, fractions], axis=1)
            )


@dataclass
class TraceResult:
    """The outcome of tracing a set of particles."""

    charge_density: np.ndarray
    status: np.ndarray
    electrode: np.ndarray
    planes: list[PlaneHits]
    trajectories: list[np.ndarray]
    trajectory_species: np.ndarray
    steps: int
    #: With charge changes: every trajectory's final fractions ``(states, N)``.
    fractions: np.ndarray | None = None
    #: With charge changes: what the beam did to its targets (this rank's share).
    tallies: ChargeTallies | None = None


class BeamTracer:
    """Trace particles through a static electric (and magnetic) field.

    ``electric`` returns ``(E_x, E_y)`` at ``(x, y)`` (``y`` is the radius in cylindrical
    geometry), ``magnetic`` (cylindrical only) ``(B_z, B_r)`` at ``(x, r)``. The step of each
    particle is ``step_fraction`` times the estimate ``min(h / |v|, sqrt(2 h / |a|))``,
    integrated with the classical Runge--Kutta method.
    """

    def __init__(
        self,
        mesh: IonOpticsMesh,
        electric: (
            Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None
        ),
        magnetic: (
            Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None
        ) = None,
        *,
        step_fraction: float = 0.5,
        max_steps: int = 100_000,
        planes: Sequence[float] = (),
        save_trajectories: int = 0,
        charge_changing: ChargeChanging | None = None,
    ) -> None:
        """Set up the tracer; ``save_trajectories`` keeps every n-th trajectory (0: none).

        ``charge_changing`` lets every trajectory carry charge-state fractions that
        change on the targets it crosses, or one state that changes at sampled
        events (see :mod:`beamtracer.solver.charge_changing`).
        """
        if magnetic is not None and not mesh.cylindrical:
            raise ValueError("a magnetic field needs a cylindrical mesh")
        if not 0.0 < step_fraction <= 1.0:
            raise ValueError(f"step_fraction must be in (0, 1], got {step_fraction}")
        self.mesh = mesh
        self.electric = electric
        self.magnetic = magnetic
        self.step_fraction = float(step_fraction)
        self.max_steps = int(max_steps)
        self.planes = tuple(float(p) for p in planes)
        self.save_trajectories = int(save_trajectories)
        self.charge_changing = charge_changing
        #: Sums the node charges over MPI ranks that each trace a share of the beam.
        self.reduce: Callable[[np.ndarray], np.ndarray] | None = None
        #: The random stream of analog charge changes, e.g. the MPI rank.
        self.stream = 0
        # Cells with an electrode node or a node next to one: only particles in these cells
        # are tested against the electrodes.
        touch = (mesh.solid >= 0) | mesh.near_solid
        self._electrode_cells = (
            touch[:-1, :-1] | touch[1:, :-1] | touch[:-1, 1:] | touch[1:, 1:]
        )

    # -- fields -----------------------------------------------------------------------------

    def acceleration(self, x: np.ndarray, v: np.ndarray, qm: np.ndarray) -> np.ndarray:
        """Return the acceleration ``(3, N)`` at positions ``x`` and velocities ``v``."""
        electric = self.electric
        if electric is None:
            raise RuntimeError("the tracer has no electric field yet")
        if not self.mesh.cylindrical:
            ex, ey = electric(x[0], x[1])
            return qm * np.stack([ex, ey, np.zeros_like(ex)], axis=0)
        r = np.hypot(x[1], x[2])
        with np.errstate(invalid="ignore", divide="ignore"):
            cy = np.where(r > 0.0, x[1] / r, 0.0)
            cz = np.where(r > 0.0, x[2] / r, 0.0)
        ex, er = electric(x[0], r)
        force = np.stack([ex, er * cy, er * cz], axis=0)
        if self.magnetic is not None:
            bx, br = self.magnetic(x[0], r)
            b = np.stack([bx, br * cy, br * cz], axis=0)
            force += np.cross(v, b, axis=0)
        return qm * force

    def _rk4(self, x, v, a, dt, qm):
        """One classical Runge--Kutta step of ``dx/dt = v, dv/dt = a(x, v)``."""
        d = dt
        k1x, k1v = v, a
        k2x = v + 0.5 * d * k1v
        k2v = self.acceleration(x + 0.5 * d * k1x, k2x, qm)
        k3x = v + 0.5 * d * k2v
        k3v = self.acceleration(x + 0.5 * d * k2x, k3x, qm)
        k4x = v + d * k3v
        k4v = self.acceleration(x + d * k3x, k4x, qm)
        x_new = x + d / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v_new = v + d / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return x_new, v_new

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Return the mesh coordinates ``(x, y)`` or ``(x, r)`` of Cartesian points."""
        if self.mesh.cylindrical:
            return np.stack([x[0], np.hypot(x[1], x[2])], axis=0)
        return x[:2]

    def _near_electrode(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Return whether points lie in a cell that touches an electrode."""
        mesh = self.mesh
        i = np.clip(
            ((px - mesh.origin[0]) / mesh.h).astype(np.int64), 0, mesh.shape[0] - 2
        )
        j = np.clip(
            ((py - mesh.origin[1]) / mesh.h).astype(np.int64), 0, mesh.shape[1] - 2
        )
        return self._electrode_cells[i, j]

    def _finalize(self, charge: np.ndarray) -> np.ndarray:
        """Divide the node charges by the node volumes."""
        mesh = self.mesh
        h = mesh.h
        density = charge.copy()
        if mesh.cylindrical:
            j = np.arange(mesh.shape[1])
            r1 = (j - 0.5) * h + mesh.origin[1]
            r2 = (j + 0.5) * h + mesh.origin[1]
            volume = np.pi * h * (r2 * r2 - r1 * r1)
            volume[0] = np.pi * h * (h + mesh.origin[1]) ** 2
            density /= volume[None, :]
        else:
            density /= h * h
        density[:, 0] *= 2.0
        density[:, -1] *= 2.0
        density[0, :] *= 2.0
        density[-1, :] *= 2.0
        return density

    def _cross(
        self,
        hits: PlaneHits,
        x,
        v,
        x_new,
        v_new,
        current,
        species,
        f=None,
        f_new=None,
        switch=None,
    ) -> None:
        """Record the particles crossing the plane ``x = hits.position_x`` in this step.

        The fractions are interpolated along the step, or with ``switch`` (the
        share of the step where each trajectory changed state) taken before or
        after it.
        """
        xp = hits.position_x
        before = x[0] - xp
        after = x_new[0] - xp
        crossed = (before < 0.0) != (after < 0.0)
        if not crossed.any():
            return
        s = before[crossed] / (before[crossed] - after[crossed])
        point = x[:, crossed] + s * (x_new[:, crossed] - x[:, crossed])
        point[0] = xp
        velocity = v[:, crossed] + s * (v_new[:, crossed] - v[:, crossed])
        direction = np.where(after[crossed] >= 0.0, 1, -1)
        fractions = None
        if f is not None:
            assert f_new is not None
            if switch is None:
                fractions = f[:, crossed] + s * (f_new[:, crossed] - f[:, crossed])
            else:
                fractions = np.where(
                    s < switch[crossed], f[:, crossed], f_new[:, crossed]
                )
        hits.append(
            point, velocity, current[crossed], species[crossed], direction, fractions
        )

    def _inside(self, x: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """The share of every step from ``x`` to ``x_new`` before it leaves the mesh."""
        mesh = self.mesh
        start, end = self._project(x), self._project(x_new)
        share = np.ones(start.shape[1])
        lower = (mesh.origin[0], None if mesh.cylindrical else mesh.origin[1])
        for axis in range(2):
            span = end[axis] - start[axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                if lower[axis] is not None:
                    out = end[axis] < lower[axis]
                    share[out] = np.minimum(
                        share[out], (lower[axis] - start[axis][out]) / span[out]
                    )
                out = end[axis] > mesh.upper[axis]
                share[out] = np.minimum(
                    share[out], (mesh.upper[axis] - start[axis][out]) / span[out]
                )
        return np.clip(share, 0.0, 1.0)

    @scope_profiler.profile("beamtracer.deposit", tags=PROFILE_TAGS)
    def _deposit(self, charge: np.ndarray, points: np.ndarray, q: np.ndarray) -> None:
        """Add charges ``q`` at ``points`` to the node charges ``(nx, ny)``.

        With an area-weighted linear shape.
        """
        if not len(q):
            return
        mesh = self.mesh
        px, py = self._project(points)
        deposit_trajectories_2d(
            np.ascontiguousarray(px),
            np.ascontiguousarray(py),
            np.ascontiguousarray(q, dtype=float),
            *map(float, mesh.origin),
            mesh.h,
            int(mesh.cylindrical),
            charge,
        )

    # -- tracing ----------------------------------------------------------------------------

    @scope_profiler.profile("beamtracer.trace", tags=PROFILE_TAGS)
    def trace(self, particles: BeamParticles) -> TraceResult:
        """Trace ``particles`` until they leave the mesh or hit an electrode."""
        mesh = self.mesh
        nx, ny = mesh.shape
        h = mesh.h
        x0, y0 = mesh.origin
        x1, y1 = mesh.upper
        n = len(particles)
        charge = np.zeros((nx, ny))
        status = np.full(n, MAX_STEPS, dtype=np.int8)
        electrode = np.full(n, -1, dtype=np.int32)
        planes = [PlaneHits(p) for p in self.planes]

        index = np.arange(n)
        x = particles.position.astype(float).copy()
        v = particles.velocity.astype(float).copy()
        qm = particles.charge / particles.mass
        current = particles.current.astype(float)
        # the space charge carries the particles' sign: currents are magnitudes
        signed_current = np.sign(particles.charge) * np.abs(current)
        changing = self.charge_changing
        analog = changing is not None and changing.analog
        fractions = final_fractions = tallies = None
        # used only with charge changes
        state_charges = particle_rate = mass = np.zeros(0)
        # used only with analog charge changes: every trajectory's state, the
        # optical depth left to its next event and its number of events
        state = depth = generation = np.zeros(0, dtype=np.int64)
        rng = np.random.default_rng()
        if changing is not None:
            # every trajectory carries fractions; its current is e times its particle rate
            fractions = changing.initial_fractions(particles.charge)
            final_fractions = fractions.copy()
            state_charges = changing.state_charges
            particle_rate = np.abs(current) / constants.e
            mass = particles.mass.astype(float)
            tallies = ChargeTallies(changing.states)
            if analog:
                rng = np.random.default_rng(
                    np.random.SeedSequence(changing.seed, spawn_key=(int(self.stream),))
                )
                state = changing.initial_states(particles.charge)
                depth = rng.exponential(size=n)
                generation = np.zeros(n, dtype=np.int64)
        species = particles.species
        saved_ids = (
            np.arange(0, n, self.save_trajectories)
            if self.save_trajectories
            else np.zeros(0, int)
        )
        saved = {int(k): [self._project(x[:, k : k + 1])[:, 0]] for k in saved_ids}

        # Particles starting inside an electrode never move (and deposit nothing).
        if mesh.electrodes and n:
            px, py = self._project(x)
            hit = mesh.inside_solid(px, py)
            dead = hit >= 0
            status[dead] = ELECTRODE
            electrode[dead] = hit[dead]
            index, x, v = index[~dead], x[:, ~dead], v[:, ~dead]
            if fractions is not None:
                fractions = fractions[:, ~dead]
            if analog:
                state, depth = state[~dead], depth[~dead]
                generation = generation[~dead]

        step = 0
        while index.size and step < self.max_steps:
            step += 1
            if fractions is not None:
                qm[index] = (state_charges @ fractions) / mass[index]
            a = self.acceleration(x, v, qm[index])
            speed = np.linalg.norm(v, axis=0)
            accel = np.linalg.norm(a, axis=0)
            with np.errstate(divide="ignore"):
                dt = np.minimum(h / speed, np.sqrt(2.0 * h / accel))
            dt = self.step_fraction * np.where(np.isfinite(dt), dt, h)
            x_new, v_new = self._rk4(x, v, a, dt, qm[index])

            # Space charge: I dt at the (Hermite) midpoint of the step.
            mid = 0.5 * (x + x_new) + dt * (v - v_new) / 8.0
            fractions_new = None
            events = None
            if fractions is not None:
                assert changing is not None and tallies is not None
                # the path inside the mesh: a step that leaves it counts up to the face
                path = (
                    0.5
                    * dt
                    * (np.linalg.norm(v, axis=0) + np.linalg.norm(v_new, axis=0))
                    * self._inside(x, x_new)
                )
                lines = changing.line_densities(mid[0], path)
                if analog:
                    events = changing.sample(rng, state, depth, generation, lines)
                    tallies.add_events(events, particle_rate[index])
                    # the states after the step, split trajectories by their share
                    fractions_new = np.zeros_like(fractions)
                    np.add.at(
                        fractions_new, (events.state, events.parent), events.share
                    )
                else:
                    fractions_new = changing.advance(fractions, lines)
                middle = 0.5 * (fractions + fractions_new)
                tallies.add(changing, middle, particle_rate[index], path, lines)
                deposited = particle_rate[index] * (state_charges @ middle) * dt
            else:
                deposited = signed_current[index] * dt
            self._deposit(charge, mid, deposited)

            for hits in planes:
                self._cross(
                    hits, x, v, x_new, v_new, current[index], species[index],
                    fractions, fractions_new, None if events is None else events.at,
                )  # fmt: skip

            if analog:
                assert changing is not None and events is not None
                assert final_fractions is not None
                parent = events.parent
                extra = index[parent[index.size :]]  # split off these trajectories
                if extra.size:
                    new = np.arange(status.size, status.size + extra.size)
                    status = np.concatenate(
                        [status, np.full(extra.size, MAX_STEPS, np.int8)]
                    )
                    electrode = np.concatenate(
                        [electrode, np.full(extra.size, -1, np.int32)]
                    )
                    species = np.concatenate([species, species[extra]])
                    qm = np.concatenate([qm, qm[extra]])
                    mass = np.concatenate([mass, mass[extra]])
                    current = np.concatenate([current, current[extra]])
                    particle_rate = np.concatenate(
                        [particle_rate, particle_rate[extra]]
                    )
                    final_fractions = np.concatenate(
                        [final_fractions, final_fractions[:, extra]], axis=1
                    )
                    index = np.concatenate([index, new])
                    x_new, v_new = x_new[:, parent], v_new[:, parent]
                current[index] *= events.share
                particle_rate[index] *= events.share
                state, depth, generation = events.state, events.depth, events.generation
                fractions_new = changing.one_hot(state)

            # Finished trajectories.
            px, py = self._project(x_new)
            done = np.full(index.size, -1, dtype=np.int8)
            done[px >= x1] = TRANSMITTED
            done[(done < 0) & (px < x0)] = BACKWARD
            out_y = (py > y1) if mesh.cylindrical else ((py > y1) | (py < y0))
            done[(done < 0) & out_y] = WALL
            check = (done < 0) & self._near_electrode(px, py)
            hit = np.full(index.size, -1, dtype=np.int32)
            if mesh.electrodes and check.any():
                hit[check] = mesh.inside_solid(px[check], py[check])
                done[check & (hit >= 0)] = ELECTRODE

            if saved:
                for k in np.intersect1d(index, saved_ids, assume_unique=True):
                    row = np.searchsorted(index, k)
                    saved[int(k)].append(np.array([px[row], py[row]]))

            finished = done >= 0
            if fractions_new is not None:
                assert final_fractions is not None
                final_fractions[:, index] = fractions_new
            if finished.any():
                status[index[finished]] = done[finished]
                electrode[index[finished]] = hit[finished]
                keep = ~finished
                index, x_new, v_new = index[keep], x_new[:, keep], v_new[:, keep]
                if fractions_new is not None:
                    fractions_new = fractions_new[:, keep]
                if analog:
                    state, depth = state[keep], depth[keep]
                    generation = generation[keep]
            x, v = x_new, v_new
            if fractions is not None:
                fractions = fractions_new

        if self.reduce is not None:
            charge = self.reduce(charge)
        density = self._finalize(charge)
        return TraceResult(
            charge_density=density,
            status=status,
            electrode=electrode,
            planes=planes,
            trajectories=[np.array(saved[int(k)]) for k in saved_ids],
            trajectory_species=species[saved_ids],
            steps=step,
            fractions=final_fractions,
            tallies=tallies,
        )
