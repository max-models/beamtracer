"""The Vlasov iteration: solve the field, trace the beam, deposit its space charge, repeat.

Iteration 0 solves with the initial plasma region fixed at the plasma potential (if there is a
plasma), later iterations with a Boltzmann plasma species whose density is the beams' charge
density at their start: electrons for positive-ion extraction, positive ions for negative-ion
extraction. Space-charge compensation scales the beam's charge density in regions
of ``x`` before the next solve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scope_profiler
from methodcard import method

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.solver.fields import StaggeredElectricField
from beamtracer.solver.fields3d import StaggeredElectricField3D
from beamtracer.solver.mesh import IonOpticsMesh
from beamtracer.solver.mesh3d import IonOpticsMesh3D
from beamtracer.solver.poisson import (
    BoltzmannElectrons,
    BoltzmannIons,
    IonOpticsPoisson,
    Layout3D,
)
from beamtracer.solver.refinement import Layout, PatchedElectricField, Refinement
from beamtracer.solver.tracer import BeamParticles, BeamTracer, TraceResult
from beamtracer.solver.tracer3d import BeamTracer3D


@dataclass(frozen=True)
class PlasmaModel:
    """Plasma electrons for positive-ion extraction, Boltzmann distributed.

    ``initial_plasma_x`` [m]: in iteration 0 the free nodes with ``x`` below it are held at
    ``plasma_potential`` [V]. ``rho_e`` [C/m^3] of ``None`` is the beam's ``sum J / v``.
    """

    electron_temperature: float
    plasma_potential: float
    initial_plasma_x: float
    rho_e: float | None = None

    @property
    def density(self) -> float | None:
        """The plasma electrons' charge density [C/m^3], if given."""
        return self.rho_e

    def boltzmann(self, density: float) -> BoltzmannElectrons:
        """The Boltzmann electrons of charge density ``density`` [C/m^3]."""
        return BoltzmannElectrons(
            density, self.electron_temperature, self.plasma_potential
        )


@dataclass(frozen=True)
class NegativeIonPlasmaModel:
    """Plasma positive ions for negative-ion extraction.

    The negative ions and the electrons extracted with them are the beams, traced; the
    positive ions, repelled by the extraction field, follow a Boltzmann distribution of
    ``ion_temperature`` [eV] about ``plasma_potential`` [V]. ``initial_plasma_x`` [m]: in
    iteration 0 the free nodes with ``x`` below it are held at ``plasma_potential``.
    ``rho_p`` [C/m^3] of ``None`` is the beams' ``sum J / v``, which neutralises the plasma
    at the start of the beams.
    """

    ion_temperature: float
    plasma_potential: float
    initial_plasma_x: float
    rho_p: float | None = None

    @property
    def density(self) -> float | None:
        """The plasma ions' charge density [C/m^3], if given."""
        return self.rho_p

    def boltzmann(self, density: float) -> BoltzmannIons:
        """The Boltzmann ions of charge density ``density`` [C/m^3]."""
        return BoltzmannIons(density, self.ion_temperature, self.plasma_potential)


@dataclass(frozen=True)
class Compensation:
    """Space-charge compensation.

    The beam's charge density is multiplied by ``1 - fraction`` between ``x_min`` and
    ``x_max`` [m].
    """

    x_min: float
    x_max: float
    fraction: float


@dataclass
class VlasovState:
    """The converged (or last) iteration: fields, charge, trace and history."""

    potential: np.ndarray
    charge_density: np.ndarray
    electric: StaggeredElectricField | PatchedElectricField | StaggeredElectricField3D
    trace: TraceResult
    history: list[dict[str, float]] = field(default_factory=list)
    #: The Boltzmann plasma species of the last solve (``None`` without a plasma, and in
    #: iteration 0).
    plasma: BoltzmannElectrons | BoltzmannIons | None = None
    #: The composite nodes (base mesh and refined boxes); `potential` and
    #: `charge_density` are their values on the base mesh.
    layout: Layout | Layout3D | None = None
    #: The potential and the charge density on the composite nodes.
    composite_potential: np.ndarray | None = None
    composite_charge_density: np.ndarray | None = None


@method(
    equations={
        "anderson": r"\rho^{n+1} = \rho^n + \omega f^n - (\Delta X + \omega\, \Delta F)\, \gamma, \qquad \gamma = \arg\min_\gamma \| f^n - \Delta F \gamma \|_2, \qquad f^n = G(\rho^n) - \rho^n",
        "differences": r"\Delta X = [\rho^{k+1} - \rho^k]_{k = n-m}^{n-1}, \qquad \Delta F = [f^{k+1} - f^k]_{k = n-m}^{n-1}",
    },
    assumptions=(
        r"$G$ the map from a charge density to the deposit of the beam traced in its potential, $\omega$ = `relaxation`, $m$ = `depth` previous iterates",
        r"Anderson's method of type II (Walker and Ni, SIAM J. Numer. Anal. 49, 2011); with no history it is the linear mixing $\rho + \omega f$",
        r"the least-squares problem solved with a cut-off of $10^{-10}$ on the singular values, so nearly dependent differences do not blow it up",
    ),
)
class AndersonMixer:
    """Anderson acceleration of the fixed point ``rho = G(rho)`` of the Vlasov iteration.

    `update` takes the density ``rho`` an iteration started from and its new deposit
    ``G(rho)``, and returns the density of the next iteration, from the last ``depth``
    iterates. `reset` forgets them, e.g. when ``G`` changes.
    """

    def __init__(self, depth: int = 5, relaxation: float = 1.0) -> None:
        """Keep ``depth`` previous iterates; mix with ``relaxation``."""
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.depth = int(depth)
        self.relaxation = float(relaxation)
        self._x: list[np.ndarray] = []
        self._f: list[np.ndarray] = []

    def reset(self) -> None:
        """Forget the previous iterates."""
        self._x, self._f = [], []

    def update(self, rho: np.ndarray, deposit: np.ndarray) -> np.ndarray:
        """Return the next density from ``rho`` and its deposit ``G(rho)``."""
        x = np.asarray(rho, dtype=float).ravel()
        f = np.asarray(deposit, dtype=float).ravel() - x
        self._x.append(x)
        self._f.append(f)
        del self._x[: -self.depth - 1], self._f[: -self.depth - 1]
        omega = self.relaxation
        new = x + omega * f
        if len(self._x) > 1:
            dx = np.diff(np.array(self._x), axis=0).T
            df = np.diff(np.array(self._f), axis=0).T
            gamma = np.linalg.lstsq(df, f, rcond=1e-10)[0]
            new = new - (dx + omega * df) @ gamma
        return new.reshape(np.shape(rho))


def compensate(
    x: np.ndarray, rho: np.ndarray, compensation: Sequence[Compensation]
) -> np.ndarray:
    """Return ``rho`` at nodes of axial positions ``x`` with the compensation applied."""
    rho = np.array(rho, dtype=float)
    for region in compensation:
        inside = (x >= region.x_min) & (x <= region.x_max)
        rho[inside] *= 1.0 - region.fraction
    return rho


@method(
    equations={
        "compensation": r"\rho_b \leftarrow (1 - f)\, \rho_b \quad \text{on the node columns } x_\mathrm{min} \le x_i \le x_\mathrm{max}",
    },
    assumptions=(
        r"$f \in [0, 1]$ the compensated share, e.g. by the electrons of the background gas; regions applied in turn, so overlapping ones multiply",
    ),
)
def apply_compensation(
    mesh: IonOpticsMesh, rho: np.ndarray, compensation: Sequence[Compensation]
) -> np.ndarray:
    """Return ``rho`` ``(nx, ny)`` with the compensation of every region applied."""
    return compensate(np.broadcast_to(mesh.x[:, None], mesh.shape), rho, compensation)


def thin_particles(particles: BeamParticles, stride: int) -> BeamParticles:
    """Every ``stride``-th trajectory of each species, its current scaled to keep the species'."""
    keep = np.zeros(len(particles), dtype=bool)
    scale = np.ones(len(particles))
    for species in np.unique(particles.species):
        index = np.flatnonzero(particles.species == species)
        kept = index[::stride]
        keep[kept] = True
        total = np.abs(particles.current[index]).sum()
        part = np.abs(particles.current[kept]).sum()
        scale[kept] = total / part if part > 0.0 else 1.0
    thinned = particles.subset(keep)
    thinned.current = thinned.current * scale[keep]
    return thinned


@method(
    equations={
        "iteration_0": r"\phi^0 = \text{the vacuum solution, with the free nodes at } x < x_\mathrm{plasma} \text{ held at } U_p \text{ (with a plasma)}",
        "solve": r"\nabla^2 \phi^{n} = -\frac{\rho_b^{n} + \rho_p(\phi^{n})}{\varepsilon_0}, \qquad \text{Newton from } \phi^{n-1}",
        "relaxation": r"\rho_b^{n+1} = (1 - \omega)\, \rho_b^{n} + \omega\, \rho_\mathrm{new}(\phi^{n}), \qquad \rho_b^1 = \rho_\mathrm{new}(\phi^0)",
        "convergence": r"\max |\phi^{n} - \phi^{n-1}| < \text{tolerance}",
        "coarse": r"\text{while } \max |\phi^{n} - \phi^{n-1}| \ge \text{coarse\_tolerance}: \text{ every } k\text{-th trajectory},\ I \to I\, N_s / N_{s,\mathrm{kept}}",
    },
    assumptions=(
        r"$\rho_\mathrm{new}$ the compensated deposit of the beam traced in $\phi^n$, re-seeded identically every iteration; $\omega$ = `relaxation` in $(0, 1]$ acts on the charge density, not on the potential",
        r"the Boltzmann species built once, at $n = 1$, with $\rho_0$ given or the beams' $\sum J / v_0$",
        r"at most `iterations`; an unconverged iteration returns its last state without a warning (`stats['converged']` is 0); the returned trace is that of the newest potential",
        r"with MPI every rank solves the whole field and the deposits are summed",
        r"`mixing='anderson'` replaces the relaxation by `AndersonMixer` (from $n = 1$); the coarse phase traces every $k = \lceil 1 / \text{coarse\_particles} \rceil$-th trajectory with the current of each beam kept, and its iterations do not count as converged; the history of the mixer is dropped when it ends",
    ),
    components=(
        "beamtracer.solver.vlasov.AndersonMixer",
        "beamtracer.solver.poisson.IonOpticsPoisson.solve",
        "beamtracer.solver.fields.StaggeredElectricField",
        "beamtracer.solver.tracer.BeamTracer.trace",
        "beamtracer.solver.vlasov.apply_compensation",
    ),
)
@scope_profiler.profile("beamtracer.vlasov_iterate", tags=PROFILE_TAGS)
def vlasov_iterate(
    mesh: IonOpticsMesh | IonOpticsMesh3D,
    make_particles: Callable[[], tuple[BeamParticles, float]],
    tracer: BeamTracer | BeamTracer3D,
    *,
    iterations: int,
    plasma: PlasmaModel | NegativeIonPlasmaModel | None = None,
    compensation: Sequence[Compensation] = (),
    tolerance: float = 0.0,
    relaxation: float = 1.0,
    mixing: str = "linear",
    anderson_depth: int = 5,
    coarse_particles: float = 1.0,
    coarse_tolerance: float = 0.0,
    field_backend: str = "lu",
    comm: Any = None,
    refinement: Sequence[Refinement] = (),
    callback: Callable[[int, VlasovState], None] | None = None,
) -> VlasovState:
    """Run ``iterations`` Vlasov iterations.

    Stop early when the largest potential change is below ``tolerance`` [V].
    ``mixing`` is ``"linear"`` (``relaxation``) or ``"anderson"`` (`AndersonMixer` with
    ``anderson_depth`` previous iterates). With ``coarse_particles < 1`` the iterations
    trace that share of the trajectories (every ``k``-th, the current of each beam kept)
    until the potential changes by less than ``coarse_tolerance`` [V], then all of them.
    ``field_backend`` is patchwork's backend of the field solve, distributed over
    ``comm`` with ``"petsc"`` (`IonOpticsPoisson`). ``refinement`` are boxes of finer
    cells: the field is solved on the composite mesh and the trajectories step and deposit
    on the finest box they are in; the state's potential and charge density are then those
    of the base mesh, the composite ones beside them. On an `IonOpticsMesh3D` (with a
    `BeamTracer3D`) the same iteration runs in 3D, without refinement.

    ``make_particles`` returns the beam and its ``sum J / v`` (the beams' charge density
    magnitude at their start, the density of the Boltzmann plasma species); it is called
    every iteration, so every iteration starts from the same beam. ``tracer`` must already hold the field functions'
    owner: its ``electric`` attribute is replaced each iteration.
    """
    if mixing not in ("linear", "anderson"):
        raise ValueError(f"mixing must be 'linear' or 'anderson', got {mixing!r}")
    if not 0.0 < coarse_particles <= 1.0:
        raise ValueError(f"coarse_particles must be in (0, 1], got {coarse_particles}")
    mixer = AndersonMixer(anderson_depth, relaxation) if mixing == "anderson" else None
    stride = int(np.ceil(1.0 / coarse_particles - 1e-9))
    coarse = stride > 1
    if coarse and not coarse_tolerance > 0.0:
        raise ValueError("coarse_particles < 1 needs a positive coarse_tolerance")
    poisson = IonOpticsPoisson(
        mesh, backend=field_backend, comm=comm, refinement=refinement
    )
    layout = poisson.layout
    rho = np.zeros(layout.size)

    def initial_plasma(x: np.ndarray, *rest: np.ndarray) -> np.ndarray:
        assert plasma is not None
        return x < plasma.initial_plasma_x

    potential: np.ndarray | None = None
    state: VlasovState | None = None
    history: list[dict[str, float]] = []
    boltzmann: BoltzmannElectrons | BoltzmannIons | None = None
    j_over_v: float | None = None
    for iteration in range(iterations):
        if plasma is not None and iteration == 0:
            new = poisson.solve(
                rho,
                fixed_region=initial_plasma,
                fixed_potential=plasma.plasma_potential,
            )
        else:
            if plasma is not None and boltzmann is None:
                density = plasma.density if plasma.density is not None else j_over_v
                assert density is not None  # the beams of iteration 0 are known
                boltzmann = plasma.boltzmann(density)
            new = poisson.solve(rho, plasma=boltzmann, initial=potential)
        change = (
            float("inf")
            if potential is None
            else float(np.max(np.abs(new - potential)))
        )
        potential = new
        efield = layout.field(potential)
        tracer.electric = efield  # pyright: ignore[reportAttributeAccessIssue]
        if coarse and change < coarse_tolerance:
            coarse = False
            if mixer is not None:
                mixer.reset()
        particles, j_over_v = make_particles()
        if coarse:
            particles = thin_particles(particles, stride)
        trace = tracer.trace(particles)
        charge = trace.charge_density
        if isinstance(layout, Layout3D):  # the 3D tracer deposits on the mesh
            charge = layout.from_base(charge)
        deposited = compensate(layout.x, charge, compensation)
        if iteration == 0:
            rho = deposited
        elif mixer is not None:
            rho = mixer.update(rho, deposited)
        else:
            rho = (1.0 - relaxation) * rho + relaxation * deposited
        history.append(
            {
                "iteration": iteration,
                "max_potential_change": change,
                "newton_iterations": poisson.newton_iterations,
                "steps": trace.steps,
                "trajectories": len(particles),
            }
        )
        state = VlasovState(
            layout.on_base(potential),
            layout.on_base(rho),
            efield,
            trace,
            history,
            boltzmann,
            layout,
            potential,
            rho,
        )
        if callback is not None:
            callback(iteration, state)
        if change < tolerance and not coarse:
            break
    assert state is not None
    return state
