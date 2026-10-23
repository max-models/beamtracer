"""A neutralizer: beam trajectories through the channel, coupled to the plasma they make.

The plasma of a beam-driven plasma neutralizer is a global model of the channel
(:class:`wellbalanced.neutralizer.Neutralizer`, which also holds the channel, the
beam and its cross sections). Alone it treats the beam as a line integral along
the axis. :class:`NeutralizerChannel` traces the beam instead: a
:class:`~beamtracer.model.BeamOptics` run on the channel's section in
``(x, y)`` (planar, per unit height), grounded walls at ``y = +-w/2``, every
trajectory carrying its charge-state fractions on the gas and the plasma of the
global model (:mod:`beamtracer.solver.charge_changing`), its space charge
reduced by the plasma's ``compensation``. From the trace the global model takes
its beam sources: the path-integrated fractions ``sum_k N_k int F_s ds / (N L)``
in place of the fractions averaged along the axis, so beam particles lost to
the walls early feed the plasma less. The two are iterated to a fixed point
(:meth:`NeutralizerChannel.solve`), starting from the global model alone.

The trace gives what the global model cannot: the transmission of each charge
state to the exit, the beam lost on the walls, and the divergence of the
neutrals. It needs ``wellbalanced``::

    pip install "beamtracer[neutralizer]"

The module was ``picnic.neutralizer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import constants

from beamtracer.config import (
    BeamOpticsConfig,
    BeamSpecies,
    ChargeChangingParams,
    SpaceChargeCompensation,
)
from beamtracer.model import BeamOptics
from beamtracer.sections import FieldBCs, FieldBoundaryCondition, GridParams

if TYPE_CHECKING:
    from wellbalanced import GlobalState
    from wellbalanced.neutralizer import Neutralizer, NeutralizerResult

__all__ = ["NeutralizerChannel", "TracedNeutralizer"]


def _wellbalanced():
    try:
        import wellbalanced.neutralizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "beamtracer.neutralizer needs wellbalanced: pip install "
            "'beamtracer[neutralizer]'"
        ) from error
    return wellbalanced.neutralizer


@dataclass
class TracedNeutralizer:
    """The fixed point of the traced beam and the plasma.

    Attributes:
        exit_fractions: Particles reaching the exit in each state, per incoming particle.
        transmission: Share of the incoming particles reaching the exit.
        path_fractions: ``sum_k N_k int F_s ds / (N L)``, the fractions the plasma sees.
        neutral_divergence: rms angle (rad) of the neutrals at the exit.
        plasma: The plasma's steady state.
        model: The last :class:`~beamtracer.model.BeamOptics` run.
        axis: The global model alone (the beam as a line integral along the axis).
        iterations: Fixed-point iterations.
        converged: Whether the fixed point converged.
    """

    exit_fractions: dict[str, float]
    transmission: float
    path_fractions: np.ndarray
    neutral_divergence: float
    plasma: GlobalState
    model: BeamOptics
    axis: NeutralizerResult
    iterations: int
    converged: bool

    @property
    def neutral_fraction(self) -> float:
        """Neutrals reaching the exit per incoming particle."""
        return self.exit_fractions["0"]


@dataclass
class NeutralizerChannel:
    """One neutralizer channel traced in ``(x, y)``; see the module docstring.

    Attributes:
        neutralizer: The channel, the beam, its cross sections and the plasma model.
        beam_width: Width (m) of the beam at the entrance, centred in the channel.
        divergence: rms angle (rad) of the beam at the entrance.
        cell: Mesh spacing (m).
        num_particles: Trajectories.
        compensation: Share of the beam's space charge the plasma compensates.
        relaxation: Under-relaxation of the path fractions between iterations.
        tolerance: Convergence (largest change of a path fraction).
        max_iterations: Fixed-point iterations at most.
        step_fraction: Step of the tracer, a share of its estimate.
        mode: ``"fractions"`` (every trajectory carries the fractions of its
            states) or ``"analog"`` (one state per trajectory, changed at
            sampled events); see :mod:`beamtracer.solver.charge_changing`.
        splitting: In analog mode, the trajectories an event splits one into.
        seed: In analog mode, the seed of the random numbers.
        settings: Entries of the ``BeamOptics`` configuration to override.
    """

    neutralizer: Neutralizer
    beam_width: float
    divergence: float = 0.0
    cell: float = 5.0e-3
    num_particles: int = 200
    compensation: float = 1.0
    relaxation: float = 0.7
    tolerance: float = 1.0e-5
    max_iterations: int = 50
    step_fraction: float = 0.5
    mode: str = "fractions"
    splitting: int = 1
    seed: int | None = 0
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the beam against the channel."""
        if not 0.0 < self.beam_width <= self.neutralizer.width:
            raise ValueError(
                "the beam must fit in the channel: 0 < beam_width <= width"
            )
        if not 0.0 <= self.compensation <= 1.0:
            raise ValueError("compensation must be in [0, 1]")
        if set(self.neutralizer.beam.initial) != {"-"}:
            raise NotImplementedError("the traced beam enters as negative ions")

    @property
    def exit_plane(self) -> float:
        """The plane (m) where the exit is measured, half a cell before the end."""
        return self.neutralizer.length - 0.5 * self.cell

    def beam_config(
        self, gas_density: float, plasma_density: float
    ) -> BeamOpticsConfig:
        """The ``BeamOptics`` case of the channel at the given target densities (m^-3)."""
        n = self.neutralizer
        beam = n.beam
        cells_x = round(n.length / self.cell)
        cells_y = round(n.width / self.cell)
        if not np.isclose(cells_x * self.cell, n.length) or not np.isclose(
            cells_y * self.cell, n.width
        ):
            raise ValueError("cell must divide the channel's length and width")
        grounded = FieldBoundaryCondition.dirichlet(0.0)
        targets = n.data.transitions.cross_sections
        cross_sections: dict[str, dict[str, dict[str, float]]] = {}
        for target, table in targets.items():
            rows: dict[str, dict[str, float]] = {}
            for (start, end), sigma in table.items():
                rows.setdefault(start, {})[end] = sigma
            cross_sections[target] = rows
        densities = {"gas": gas_density}
        if "plasma" in targets:
            densities["plasma"] = plasma_density
        settings = {
            "grid": GridParams(
                ndim=2,
                num_gridpoints=(cells_x + 1, cells_y + 1),
                lower_bound=(0.0, -0.5 * n.width),
                upper_bound=(n.length, 0.5 * n.width),
                coordinates="cartesian",
            ),
            "bcs_field": FieldBCs(
                x_lower=grounded, x_upper=grounded, y_lower=grounded, y_upper=grounded
            ),
            "beams": [
                BeamSpecies(
                    "beam",
                    # the beam spans the channel's height: A/m^2 over its width
                    beam.current / (self.beam_width * n.height),
                    (0.0, -0.5 * self.beam_width),
                    (0.0, 0.5 * self.beam_width),
                    charge=-1.0,
                    mass=beam.mass / constants.atomic_mass,
                    energy=beam.energy,
                    parallel_temperature=0.0,
                    # rms angle theta: transverse temperature 2 E theta^2
                    transverse_temperature=2.0 * beam.energy * self.divergence**2,
                    num_particles=self.num_particles,
                )
            ],
            "charge_changing": ChargeChangingParams(
                cross_sections,
                densities,
                ionization={"gas": dict(n.data.gas_ionization)},
                mode=self.mode,
                splitting=self.splitting,
                seed=self.seed,
            ),
            "compensation": [SpaceChargeCompensation(0.0, n.length, self.compensation)],
            "iterations": 1 if self.compensation >= 1.0 else 10,
            "relaxation": 0.7,
            "step_fraction": self.step_fraction,
            "diagnostic_planes": [self.exit_plane],
            "save_trajectories": max(1, self.num_particles // 20),
        }
        return BeamOpticsConfig(**{**settings, **self.settings})

    def trace(self, gas_density: float, plasma_density: float) -> BeamOptics:
        """Run the beam through the channel at the given target densities."""
        return BeamOptics(self.beam_config(gas_density, plasma_density)).run()

    def path_fractions(self, model: BeamOptics) -> np.ndarray:
        """``sum_k N_k int F_s ds / (N L)`` of a traced beam."""
        n = self.neutralizer
        tallies = model.charge_tallies
        assert tallies is not None  # the beam changes charge
        # the tallies are per metre of height; the beam spans the height
        per_metre = n.beam.particle_rate / n.height
        return np.asarray(tallies.path) / (per_metre * n.length)

    def solve(self) -> TracedNeutralizer:
        """Iterate the traced beam and the plasma to a fixed point."""
        _wellbalanced()
        n = self.neutralizer
        axis = n.solve()
        state = axis.plasma
        mean: np.ndarray | None = None
        model: BeamOptics | None = None
        converged = False
        iterations = 0
        for _ in range(self.max_iterations):
            iterations += 1
            model = self.trace(
                n.gas_target_density(state.densities), state.electron_density
            )
            traced = self.path_fractions(model)
            change = np.inf if mean is None else float(np.max(np.abs(traced - mean)))
            mean = (
                traced
                if mean is None
                else (1.0 - self.relaxation) * mean + self.relaxation * traced
            )
            plasma_model = n.plasma_model(mean, state.gas_temperature)
            state = plasma_model.steady_state(state)
            if change < self.tolerance and state.converged:
                converged = True
                break
        assert model is not None and mean is not None
        incoming = n.beam.particle_rate / n.height  # per metre of height
        exit_fractions = {}
        changing = model.tracer.charge_changing
        assert changing is not None
        for s in changing.states:
            moments = model.state_moments(self.exit_plane, s)
            exit_fractions[s] = moments.current / constants.e / incoming
        neutrals = model.state_moments(self.exit_plane, "0")
        return TracedNeutralizer(
            exit_fractions=exit_fractions,
            transmission=sum(exit_fractions.values()),
            path_fractions=mean,
            neutral_divergence=float(neutrals.rms_angle),
            plasma=state,
            model=model,
            axis=axis,
            iterations=iterations,
            converged=converged,
        )
