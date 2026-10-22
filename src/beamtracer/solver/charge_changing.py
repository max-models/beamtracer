"""Charge changes of fast beam particles along their trajectories.

A fast beam crossing a gas (or a plasma) changes charge state: a negative ion
loses its electron (``- -> 0``), a neutral is ionized (``0 -> +``), an ion
captures one (``+ -> 0``). With :class:`ChargeChanging` every trajectory of the
:class:`~beamtracer.solver.tracer.BeamTracer` carries the fractions of its
particles in each state instead of one charge: per step of length ``ds`` the
fractions change by the matrix exponential of ``sum_t n_t ds M_t`` (the
transition matrices of :class:`plasmacoll.ChargeStateTransitions`, one per
target of density ``n_t``), here to third order in the step.

The force on a trajectory is that of its mean charge ``e sum_s F_s z_s``: a
trajectory does not split where its states would part (fine where the field
is weak, as in a neutralizer; not for a residual-ion dump). Its space charge
is that of every state with its own sign. Its ``current`` is ``e`` times its
particle rate, whatever its state.

The tracer tallies, over all trajectories (in particles per second, per unit
depth in planar geometry): the charge changes per target and transition, the
ionizations of each target by each state (``ionization``), and the
path-integrated fraction of every state, ``sum_k N_k int F_s ds`` (particles
times metres per second), from which a volume model takes its beam sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from plasmacoll import ChargeStateTransitions
from scipy import constants

__all__ = ["ChargeChanging", "ChargeTallies"]

#: The charge (in e) of the states of a hydrogen-like beam.
DEFAULT_CHARGES = {"-": -1.0, "0": 0.0, "+": 1.0}


@dataclass(frozen=True)
class ChargeChanging:
    """The targets a beam changes charge on.

    Attributes:
        transitions: Charge-changing cross sections per target (m^2).
        densities: Density of every target (m^-3), uniform.
        charges: The charge of every state in e.
        ionization: ``{target: {state: sigma}}``, cross sections (m^2) for
            ionizing the target, tallied only (they do not change the beam).
    """

    transitions: ChargeStateTransitions
    densities: Mapping[str, float]
    charges: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CHARGES))
    ionization: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the targets and states."""
        missing = set(self.transitions.states) - set(self.charges)
        if missing:
            raise ValueError(f"no charge given for the state(s) {sorted(missing)}")
        unknown = set(self.densities) - set(self.transitions.targets)
        unknown |= set(self.ionization) - set(self.densities)
        if unknown:
            raise ValueError(f"unknown target(s) {sorted(unknown)}")
        if any(n < 0.0 for n in self.densities.values()):
            raise ValueError("target densities must be >= 0")
        object.__setattr__(self, "densities", dict(self.densities))
        object.__setattr__(self, "charges", dict(self.charges))
        object.__setattr__(
            self, "ionization", {t: dict(s) for t, s in self.ionization.items()}
        )

    @property
    def states(self) -> tuple[str, ...]:
        """The state names, in the order of the fractions."""
        return tuple(self.transitions.states)

    @property
    def state_charges(self) -> np.ndarray:
        """The charge of every state in C, in the order of :attr:`states`."""
        return np.array([self.charges[s] for s in self.states]) * constants.e

    def initial_fractions(self, charge: np.ndarray) -> np.ndarray:
        """Return the ``(states, N)`` fractions of particles of charge ``charge`` (C).

        All are in the state of that charge.
        """
        z = np.asarray(charge, dtype=float) / constants.e
        fractions = np.zeros((len(self.states), z.size))
        for k, state in enumerate(self.states):
            fractions[k, np.isclose(z, self.charges[state])] = 1.0
        unmatched = fractions.sum(axis=0) == 0.0
        if unmatched.any():
            raise ValueError(
                f"particles of charge {np.unique(z[unmatched])} e match no state"
            )
        return fractions

    def rate_matrix(self) -> np.ndarray:
        """``sum_t n_t M_t`` (1/m): the change of the fractions per metre of path."""
        size = len(self.states)
        total = np.zeros((size, size))
        for target, density in self.densities.items():
            total += density * self.transitions.matrix(target)
        return total

    def advance(self, fractions: np.ndarray, path: np.ndarray) -> np.ndarray:
        """The fractions ``(states, N)`` after paths ``path`` (m), third order."""
        a = self.rate_matrix()
        first = (a @ fractions) * path
        second = (a @ first) * path / 2.0
        third = (a @ second) * path / 3.0
        return fractions + first + second + third


@dataclass
class ChargeTallies:
    """What the beam did to its targets, summed over trajectories (see the module docstring).

    Attributes:
        states: The state names.
        changes: ``{(target, from, to): rate}`` charge changes (particles/s).
        ionization: ``{(target, state): rate}`` ionizations of the targets (particles/s).
        path: ``sum_k N_k int F_s ds`` per state (particles m/s).
    """

    states: tuple[str, ...]
    changes: dict[tuple[str, str, str], float] = field(default_factory=dict)
    ionization: dict[tuple[str, str], float] = field(default_factory=dict)
    path: np.ndarray | None = None

    def add(
        self,
        model: ChargeChanging,
        fractions: np.ndarray,
        rate: np.ndarray,
        path: np.ndarray,
    ) -> None:
        """Add one step of every trajectory: fractions at its start, particle rates, paths."""
        weighted = fractions @ (rate * path)  # sum_k N_k F_s,k ds_k per state
        self.path = weighted if self.path is None else self.path + weighted
        index = {s: k for k, s in enumerate(self.states)}
        for target, density in model.densities.items():
            for (start, end), sigma in model.transitions.cross_sections[target].items():
                key = (target, start, end)
                value = density * sigma * weighted[index[start]]
                self.changes[key] = self.changes.get(key, 0.0) + value
            for state, sigma in model.ionization.get(target, {}).items():
                key2 = (target, state)
                value = density * sigma * weighted[index[state]]
                self.ionization[key2] = self.ionization.get(key2, 0.0) + value

    def merge(self, other: ChargeTallies) -> None:
        """Add another rank's tallies."""
        for key, value in other.changes.items():
            self.changes[key] = self.changes.get(key, 0.0) + value
        for key, value in other.ionization.items():
            self.ionization[key] = self.ionization.get(key, 0.0) + value
        if other.path is not None:
            self.path = other.path if self.path is None else self.path + other.path
