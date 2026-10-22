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

import numpy as np
import scope_profiler

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.solver.fields import StaggeredElectricField
from beamtracer.solver.mesh import IonOpticsMesh
from beamtracer.solver.poisson import (
    BoltzmannElectrons,
    BoltzmannIons,
    IonOpticsPoisson,
)
from beamtracer.solver.tracer import BeamParticles, BeamTracer, TraceResult


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
    electric: StaggeredElectricField
    trace: TraceResult
    history: list[dict[str, float]] = field(default_factory=list)
    #: The Boltzmann plasma species of the last solve (``None`` without a plasma, and in
    #: iteration 0).
    plasma: BoltzmannElectrons | BoltzmannIons | None = None


def apply_compensation(
    mesh: IonOpticsMesh, rho: np.ndarray, compensation: Sequence[Compensation]
) -> np.ndarray:
    """Return ``rho`` with the compensation of every region applied."""
    rho = rho.copy()
    x = mesh.x
    for region in compensation:
        inside = (x >= region.x_min) & (x <= region.x_max)
        rho[inside, :] *= 1.0 - region.fraction
    return rho


@scope_profiler.profile("beamtracer.vlasov_iterate", tags=PROFILE_TAGS)
def vlasov_iterate(
    mesh: IonOpticsMesh,
    make_particles: Callable[[], tuple[BeamParticles, float]],
    tracer: BeamTracer,
    *,
    iterations: int,
    plasma: PlasmaModel | NegativeIonPlasmaModel | None = None,
    compensation: Sequence[Compensation] = (),
    tolerance: float = 0.0,
    relaxation: float = 1.0,
    callback: Callable[[int, VlasovState], None] | None = None,
) -> VlasovState:
    """Run ``iterations`` Vlasov iterations.

    Stop early when the largest potential change is below ``tolerance`` [V].

    ``make_particles`` returns the beam and its ``sum J / v`` (the beams' charge density
    magnitude at their start, the density of the Boltzmann plasma species); it is called
    every iteration, so every iteration starts from the same beam. ``tracer`` must already hold the field functions'
    owner: its ``electric`` attribute is replaced each iteration.
    """
    poisson = IonOpticsPoisson(mesh)
    rho = np.zeros(mesh.shape)
    potential: np.ndarray | None = None
    state: VlasovState | None = None
    history: list[dict[str, float]] = []
    boltzmann: BoltzmannElectrons | BoltzmannIons | None = None
    j_over_v: float | None = None
    for iteration in range(iterations):
        if plasma is not None and iteration == 0:
            xn, _ = mesh.nodes()
            new = poisson.solve(
                rho,
                fixed_region=xn < plasma.initial_plasma_x,
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
        efield = StaggeredElectricField(mesh, potential)
        tracer.electric = efield
        particles, j_over_v = make_particles()
        trace = tracer.trace(particles)
        deposited = apply_compensation(mesh, trace.charge_density, compensation)
        rho = (
            deposited
            if iteration == 0
            else (1.0 - relaxation) * rho + relaxation * deposited
        )
        history.append(
            {
                "iteration": iteration,
                "max_potential_change": change,
                "newton_iterations": poisson.newton_iterations,
                "steps": trace.steps,
            }
        )
        state = VlasovState(potential, rho, efield, trace, history, boltzmann)
        if callback is not None:
            callback(iteration, state)
        if change < tolerance:
            break
    assert state is not None
    return state
