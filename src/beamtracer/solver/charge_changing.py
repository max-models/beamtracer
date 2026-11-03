"""Charge changes of fast beam particles along their trajectories.

A fast beam crossing a gas (or a plasma) changes charge state: a negative ion
loses its electron (``- -> 0``), a neutral is ionized (``0 -> +``), an ion
captures one (``+ -> 0``). `ChargeChanging` follows these changes on
every trajectory of the `beamtracer.solver.tracer.BeamTracer` in one of
two modes, with the cross sections of a `plasmacoll.ChargeStateTransitions`
(``M_t``, one transition matrix per target of density ``n_t``):

``"fractions"``
    Every trajectory carries the fractions of its particles in each state
    instead of one charge. Per step of length ``ds`` the fractions change by
    the matrix exponential of ``sum_t n_t ds M_t``, here to third order in the
    step. The force on a trajectory is that of its mean charge
    ``e sum_s F_s z_s``: a trajectory does not split where its states would
    part (fine where the field is weak, as in a neutralizer; not for a
    residual-ion dump or inside an accelerator).
``"analog"``
    Every trajectory is in one state and changes it at events sampled along
    its path: it carries an optical depth drawn from an exponential
    distribution that the total charge-changing rate of its state,
    ``sum_t n_t sum_j sigma_t(s -> j)``, uses up along the path; where it is
    used up the trajectory changes to a state ``j`` (and on a target ``t``)
    drawn with probability ``n_t sigma_t(s -> j)`` and draws a new depth.
    Several events in one step are followed in turn. The force is that of
    the actual charge, so states part where the field separates them: an H-
    stripped between the grids of an accelerator leaves as an atom of
    intermediate energy, in the direction it had there. With ``splitting``
    ``k > 1`` a trajectory splits at each of its first ``split_events``
    events into ``k`` trajectories of a ``k``-th of its current that draw
    their new states independently, which keeps the statistics of the rarer
    states. The random numbers come from ``seed`` (and the MPI rank) afresh
    on every trace, so the Vlasov iteration still reaches a fixed point. The
    fractions are then one-hot, so everything below holds in both modes, and
    the mean over trajectories is the fractions mode.

The target densities are uniform or `AxialProfile` s along ``x``, e.g.
the gas between the grids of an accelerator from conductances
(`conductance_profile`) or sampled from a neutral-transport run
(picnic's ``NeutralParticleTransport``).

The space charge of a trajectory is that of every state with its own sign. Its
``current`` is ``e`` times its particle rate, whatever its state.

The tracer tallies, over all trajectories (in particles per second, per unit
depth in planar geometry): the charge changes per target and transition, the
ionizations of each target by each state (``ionization``), and the
path-integrated fraction of every state, ``sum_k N_k int F_s ds`` (particles
times metres per second), from which a volume model takes its beam sources;
in analog mode also the sampled events (``events``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from methodcard import ConvergenceProblem, Rule, method
from plasmacoll import ChargeStateTransitions
from plasmacoll.units import convert_units, to_unit, unit_field
from scipy import constants

__all__ = [
    "AxialProfile",
    "ChargeChanging",
    "ChargeEvents",
    "ChargeTallies",
    "conductance_profile",
]

#: The charge (in e) of the states of a hydrogen-like beam.
DEFAULT_CHARGES = {"-": -1.0, "0": 0.0, "+": 1.0}
#: The modes of `ChargeChanging`.
MODES = ("fractions", "analog")


@dataclass(frozen=True)
@method(
    equations={
        "profile": r"n(x) = n_i + \frac{x - x_i}{x_{i+1} - x_i} (n_{i+1} - n_i), \quad x_i \le x \le x_{i+1}; \qquad n(x) = n_0,\ n_\mathrm{last} \text{ beyond}",
        "line_density": r"\int_a^b n\, dx = \text{the trapezoidal rule on } a, \text{ the points inside } (a, b), b \quad \text{(exact)}",
    },
    assumptions=(
        r"at least two points, increasing $x$, densities $\ge 0$",
        r"in the tracer the density is taken at the midpoint of each step, as if uniform over it",
    ),
)
class AxialProfile:
    """A target density (m^-3) along ``x`` (m), linear between the points.

    Held at the end values beyond the points. Both take Quantities or strings too
    (``x=points * u.mm``, ``density=["1e19 m-3", "1e18 m-3"]``).
    """

    x: Sequence[float] = unit_field("m")
    density: Sequence[float] = unit_field("m-3")

    def __post_init__(self) -> None:
        """Convert the units and check the table."""
        convert_units(self)
        x = np.asarray(self.x, dtype=float)
        density = np.asarray(self.density, dtype=float)
        if x.ndim != 1 or x.shape != density.shape or x.size < 2:
            raise ValueError("a profile needs x and density of equal length >= 2")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("the x of a profile must increase")
        if np.any(density < 0.0):
            raise ValueError("target densities must be >= 0")
        object.__setattr__(self, "x", tuple(x.tolist()))
        object.__setattr__(self, "density", tuple(density.tolist()))

    def line_density(self, start: float, end: float) -> float:
        """Return ``int n dx`` (m^-2) from ``start`` to ``end``, exact for the linear profile."""
        nodes = np.array(self.x)
        inside = nodes[(nodes > start) & (nodes < end)]
        points = np.concatenate([[start], inside, [end]])
        return float(np.trapezoid(self(points), points))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the density at ``x``."""
        return np.interp(x, self.x, self.density)


Density = float | AxialProfile


def _conductance_rules() -> dict:
    """The SymPy rule of `conductance_profile` (for the tests)."""
    import sympy

    a1, l1, a2, l2, up = sympy.symbols("a1 l1 a2 l2 upstream", positive=True)

    def resistance(a, length):
        return (1 + 3 * length / (8 * a)) / a**2

    def after_first(a1, l1, a2, l2, upstream):
        apertures = [(0.0, l1, a1), (l1 + 1.0, l1 + 1.0 + l2, a2)]
        return conductance_profile(apertures, upstream).density[1]

    drop = up * resistance(a1, l1) / (resistance(a1, l1) + resistance(a2, l2))
    return {"drops": Rule(up - drop, implementation=after_first)}


@method(
    equations={
        "drops": r"\Delta n_i = (n_\mathrm{up} - n_\mathrm{down}) \frac{1 / (a_i^2 K_i)}{\sum_j 1 / (a_j^2 K_j)}, \qquad K_i = \frac{1}{1 + 3 l_i / (8 a_i)}",
        "flow": r"Q = C_i\, \Delta n_i, \qquad C_i = \frac{\bar v}{4} \pi a_i^2 K_i",
    },
    assumptions=(
        r"free-molecular flow through round apertures of radius $a_i$ and length $l_i$; one flow $Q$ through all of them, so the mean speed $\bar v$ cancels",
        r"uniform density in the gaps between the grids (no pumping across them), linear through each aperture",
        r"$K_i$ Dushman's approximation of Clausing's transmission probability",
    ),
    references=("S. Dushman, Scientific Foundations of Vacuum Technique (Wiley)",),
    symbolic=_conductance_rules,
)
def conductance_profile(
    apertures: Sequence[tuple[float, float, float]],
    upstream: float,
    downstream: float = 0.0,
) -> AxialProfile:
    """Return the gas density through a stack of grid apertures in free-molecular flow.

    One gas flow ``Q`` crosses every aperture in turn, from the source at
    density ``upstream`` to the tank at ``downstream`` (m^-3), and each takes
    the density drop ``Q / C_i`` with ``C_i = (v / 4) pi a^2 K_i`` the
    conductance of a round aperture of radius ``a`` and length ``l`` and
    ``K_i = 1 / (1 + 3 l / (8 a))`` its transmission probability (Dushman's
    approximation of Clausing's factor). The mean speed ``v`` drops out of the
    densities. The density is uniform in the gaps between the grids (no
    pumping across them) and linear through each aperture.

    Args:
        apertures: ``(x_start, x_end, radius)`` (m) of every aperture, along
            the beam, not overlapping.
        upstream: The density before the first aperture (m^-3).
        downstream: The density after the last one (m^-3).

    Each takes Quantities or strings too (``"1 mm"``, ``"1e19 m-3"``).
    """
    apertures = to_unit(apertures, "m", "apertures")
    upstream = to_unit(upstream, "m-3", "upstream")
    downstream = to_unit(downstream, "m-3", "downstream")
    if not apertures:
        raise ValueError("at least one aperture is needed")
    starts = np.array([a[0] for a in apertures], dtype=float)
    ends = np.array([a[1] for a in apertures], dtype=float)
    radii = np.array([a[2] for a in apertures], dtype=float)
    if np.any(ends <= starts) or np.any(starts[1:] < ends[:-1]) or np.any(radii <= 0):
        raise ValueError("apertures must have a positive length and radius, in order")
    length = ends - starts
    resistance = 1.0 / (radii**2 * (1.0 / (1.0 + 3.0 * length / (8.0 * radii))))
    drops = (upstream - downstream) * resistance / resistance.sum()
    levels = upstream - np.concatenate([[0.0], np.cumsum(drops)])
    levels[-1] = downstream  # exactly: the sum of the drops can round below it
    x = np.ravel(np.column_stack([starts, ends]))
    density = np.ravel(np.column_stack([levels[:-1], levels[1:]]))
    return AxialProfile(x.tolist(), density.tolist())


@dataclass
class ChargeEvents:
    """The outcome of sampling the analog events of one step (see `ChargeChanging.sample`).

    Rows ``N`` and up are new trajectories split off the row ``parent``.

    Attributes:
        parent: The row every row comes from (itself for the first ``N``).
        state: Every row's state index after the step.
        depth: Every row's optical depth left.
        generation: Every row's number of events so far.
        share: Every row's share of its parent's current before the step.
        at: Where in the step (a share of it) the first event of each of the
            first ``N`` rows happened, 1 if none did.
        events: ``{(target, from, to): share-weighted count}`` per row of the
            first ``N``, as ``(rows, values)`` of the sampled events.
    """

    parent: np.ndarray
    state: np.ndarray
    depth: np.ndarray
    generation: np.ndarray
    share: np.ndarray
    at: np.ndarray
    events: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]


def _advance_problem() -> ConvergenceProblem:
    """Fractions on a uniform gas, against the matrix exponential."""
    from scipy.linalg import expm

    cross_sections = {
        ("-", "0"): 1.2,
        ("-", "+"): 0.1,
        ("0", "+"): 0.4,
        ("+", "0"): 0.2,
    }
    transitions = ChargeStateTransitions(
        {"gas": cross_sections}, states=("-", "0", "+")
    )
    model = ChargeChanging(transitions, {"gas": 1.0})
    start = np.array([1.0, 0.0, 0.0])
    exact = expm(2.0 * model.rate_matrix()) @ start

    def error(step: float) -> float:
        fractions = start[:, None]
        for _ in range(round(2.0 / step)):
            fractions = model.advance(fractions, {"gas": np.array([step])})
        return float(np.max(np.abs(fractions[:, 0] - exact)))

    return ConvergenceProblem(error, steps=[0.2, 0.1, 0.05, 0.025])


@dataclass(frozen=True)
@method(
    equations={
        "master": r"\frac{d\mathbf F}{ds} = \sum_t n_t(x)\, M_t\, \mathbf F, \qquad (M_t)_{js} = \sigma_t(s \to j), \quad (M_t)_{ss} = -\sum_{j \ne s} \sigma_t(s \to j)",
        "force": r"\bar q = e \sum_s z_s F_s \ \text{(fractions)}, \qquad q = e\, z_s \ \text{(analog)}",
        "current": r"N_k = \frac{|I_k|}{e} \ \text{in every state}, \qquad \Delta Q = N_k\, e \sum_s z_s \bar F_s\, \Delta t",
    },
    assumptions=(
        r"$\mathbf F$ the fractions of a trajectory over the states $s$ of charge $z_s$ (default `-`, `0`, `+`), $s$ the path length, $M_t$ from `plasmacoll.ChargeStateTransitions`",
        r"fractions mode: a trajectory feels the force of its mean charge, so states do not part where the field would separate them (fine in a weak field, as in a neutralizer; not in a residual-ion dump or inside an accelerator)",
        r"analog mode: one state per trajectory, changed at sampled events; the mean over trajectories is the fractions mode",
        r"a trajectory starting with $|z| \ne 1$ counts $N_k = |I_k| / e$ as if $|z| = 1$",
    ),
    references=(
        "[plasmacoll](https://github.com/max-models/plasmacoll) `ChargeStateTransitions`, `charge_state_fractions`",
    ),
    components=(
        "beamtracer.solver.charge_changing.ChargeChanging.advance",
        "beamtracer.solver.charge_changing.ChargeChanging.sample",
        "beamtracer.solver.charge_changing.ChargeTallies",
    ),
)
class ChargeChanging:
    """The targets a beam changes charge on, and how its trajectories follow the changes.

    Attributes:
        transitions: Charge-changing cross sections per target (m^2).
        densities: Density of every target (m^-3), uniform or an
            `AxialProfile`.
        charges: The charge of every state in e.
        ionization: ``{target: {state: sigma}}``, cross sections (m^2) for
            ionizing the target, tallied only (they do not change the beam).
        mode: ``"fractions"`` or ``"analog"`` (see the module docstring).
        splitting: In analog mode, the trajectories an event splits one into.
        split_events: In analog mode, the events of a trajectory (counted from
            its start) that split it.
        seed: In analog mode, the seed of the random numbers.
    """

    transitions: ChargeStateTransitions
    densities: Mapping[str, Density]
    charges: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CHARGES))
    ionization: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    mode: str = "fractions"
    splitting: int = 1
    split_events: int = 1
    seed: int | None = 0

    def __post_init__(self) -> None:
        """Validate the targets and states."""
        missing = set(self.transitions.states) - set(self.charges)
        if missing:
            raise ValueError(f"no charge given for the state(s) {sorted(missing)}")
        unknown = set(self.densities) - set(self.transitions.targets)
        unknown |= set(self.ionization) - set(self.densities)
        if unknown:
            raise ValueError(f"unknown target(s) {sorted(unknown)}")
        if any(
            not isinstance(n, AxialProfile) and n < 0.0 for n in self.densities.values()
        ):
            raise ValueError("target densities must be >= 0")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.splitting < 1 or self.split_events < 0:
            raise ValueError("splitting must be >= 1 and split_events >= 0")
        object.__setattr__(
            self,
            "densities",
            {
                t: n if isinstance(n, AxialProfile) else float(n)
                for t, n in self.densities.items()
            },
        )
        object.__setattr__(self, "charges", dict(self.charges))
        object.__setattr__(
            self, "ionization", {t: dict(s) for t, s in self.ionization.items()}
        )

    @property
    def states(self) -> tuple[str, ...]:
        """The state names, in the order of the fractions."""
        return tuple(self.transitions.states)

    @property
    def analog(self) -> bool:
        """Whether every trajectory is in one state."""
        return self.mode == "analog"

    @property
    def state_charges(self) -> np.ndarray:
        """The charge of every state in C, in the order of `states`."""
        return np.array([self.charges[s] for s in self.states]) * constants.e

    def initial_states(self, charge: np.ndarray) -> np.ndarray:
        """Return the state index of particles of charge ``charge`` (C)."""
        z = np.asarray(charge, dtype=float) / constants.e
        index = np.full(z.size, -1)
        for k, state in enumerate(self.states):
            index[np.isclose(z, self.charges[state])] = k
        if np.any(index < 0):
            raise ValueError(
                f"particles of charge {np.unique(z[index < 0])} e match no state"
            )
        return index

    def initial_fractions(self, charge: np.ndarray) -> np.ndarray:
        """Return the ``(states, N)`` fractions of particles of charge ``charge`` (C).

        All are in the state of that charge.
        """
        return self.one_hot(self.initial_states(charge))

    def one_hot(self, state: np.ndarray) -> np.ndarray:
        """Return the ``(states, N)`` fractions of trajectories in the states ``state``."""
        fractions = np.zeros((len(self.states), state.size))
        fractions[state, np.arange(state.size)] = 1.0
        return fractions

    def density(self, target: str, x: np.ndarray) -> np.ndarray:
        """Return the density (m^-3) of ``target`` at ``x`` (m)."""
        n = self.densities[target]
        if isinstance(n, AxialProfile):
            return n(np.asarray(x, dtype=float))
        return np.full(np.shape(x), n)

    def line_densities(self, x: np.ndarray, path: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``n_t ds`` (m^-2) of every target for steps of length ``path`` around ``x``."""
        return {t: self.density(t, x) * path for t in self.densities}

    def rate_matrix(self) -> np.ndarray:
        """``sum_t n_t M_t`` (1/m) of uniform targets: the change of the fractions per metre."""
        size = len(self.states)
        total = np.zeros((size, size))
        for target, density in self.densities.items():
            if isinstance(density, AxialProfile):
                raise ValueError(f"the density of {target!r} is not uniform")
            total += density * self.transitions.matrix(target)
        return total

    @method(
        equations={
            "taylor": r"\mathbf F \leftarrow \mathbf F + M \mathbf F + \tfrac12 M^2 \mathbf F + \tfrac16 M^3 \mathbf F, \qquad M = \sum_t n_t(x_{1/2})\, \Delta s\, M_t",
            "conservation": r"\sum_s (M_t)_{s j} = 0 \quad \Rightarrow \quad \sum_s F_s = \text{const}",
        },
        order=3,
        assumptions=(
            r"the third-order Taylor polynomial of $e^{M}$ per step; the densities at the step's midpoint, uniform over the step",
            r"the order is in the line density per step $n \Delta s$",
        ),
        problem=_advance_problem,
    )
    def advance(
        self, fractions: np.ndarray, lines: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """The fractions ``(states, N)`` after the steps of line densities ``lines``, third order."""
        matrices = {t: self.transitions.matrix(t) for t in lines}

        def apply(f: np.ndarray) -> np.ndarray:
            return sum((matrices[t] @ f) * lines[t] for t in lines) + 0.0 * f

        first = apply(fractions)
        second = apply(first) / 2.0
        third = apply(second) / 3.0
        return fractions + first + second + third

    def _channels(self) -> list[tuple[str, int, int, float]]:
        """``(target, from, to, sigma)`` of every transition."""
        index = {s: k for k, s in enumerate(self.states)}
        return [
            (t, index[a], index[b], sigma)
            for t in self.densities
            for (a, b), sigma in self.transitions.cross_sections[t].items()
            if sigma > 0.0
        ]

    @method(
        equations={
            "rate": r"\lambda_s(x) = \sum_t n_t(x) \sum_{j \ne s} \sigma_t(s \to j)",
            "depth": r"\tau \sim \mathrm{Exp}(1), \qquad \tau \leftarrow \tau - \lambda_s \Delta s, \qquad \text{an event where } \tau \le 0, \text{ at the share } 1 + \frac{\tau}{\lambda_s \Delta s} \text{ of the step}",
            "channel": r"P(t, j) = \frac{n_t\, \sigma_t(s \to j)}{\lambda_s}",
            "splitting": r"\text{at each of the first events: } k \text{ trajectories of } I / k, \text{ drawing their new states independently}",
        },
        assumptions=(
            r"the probability of no event over a path is $\exp(-\int \lambda_s\, ds)$, so the depth of the event is $\mathrm{Exp}(1)$ in $\int \lambda_s\, ds$",
            r"several events per step followed in turn, each new state using up the rest of the step; the force changes from the next step on",
            r"random numbers from `seed` and the MPI rank, afresh on every trace, so the Vlasov iteration has a fixed point",
        ),
    )
    def sample(
        self,
        rng: np.random.Generator,
        state: np.ndarray,
        depth: np.ndarray,
        generation: np.ndarray,
        lines: Mapping[str, np.ndarray],
    ) -> ChargeEvents:
        """Use up the optical depth of one step and sample the events where it runs out.

        ``lines`` are the steps' line densities ``n_t ds`` (assumed uniform
        along each step); see the module docstring for the rest.
        """
        channels = self._channels()
        destinations = np.array([b for _, _, b, _ in channels], dtype=state.dtype)
        rows = state.size
        parent = np.arange(rows)
        state = state.copy()
        generation = generation.copy()
        share = np.ones(rows)
        at = np.ones(rows)
        seen = np.zeros(rows, dtype=bool)
        lines = {t: np.asarray(v, dtype=float) for t, v in lines.items()}

        def loss(rows_state: np.ndarray, rows_parent: np.ndarray) -> np.ndarray:
            total = np.zeros(rows_state.size)
            for t, a, _, sigma in channels:
                total += np.where(rows_state == a, sigma * lines[t][rows_parent], 0.0)
            return total

        depth = depth - loss(state, parent)
        events: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
        while True:
            pending = np.flatnonzero(depth <= 0.0)
            if not pending.size:
                break
            # the share of the step left after the event
            old = loss(state[pending], parent[pending])
            left = np.where(old > 0.0, -depth[pending] / np.where(old > 0, old, 1), 0.0)
            first = pending < at.size
            first[first] = ~seen[pending[first]]
            at[pending[first]] = 1.0 - left[first]
            seen[pending[first]] = True
            split = pending[
                (generation[pending] < self.split_events) & (self.splitting > 1)
            ]
            if split.size:
                copies = np.repeat(split, self.splitting - 1)
                share[split] /= self.splitting
                left = np.concatenate(
                    [left, np.repeat(left[np.isin(pending, split)], self.splitting - 1)]
                )
                pending = np.concatenate([pending, np.arange(rows, rows + copies.size)])
                parent = np.concatenate([parent, parent[copies]])
                state = np.concatenate([state, state[copies]])
                generation = np.concatenate([generation, generation[copies]])
                share = np.concatenate([share, share[copies]])
                depth = np.concatenate([depth, depth[copies]])
                rows += copies.size
            # the channel: probability n_t sigma_t(s -> j) among those out of s
            rates = np.stack(
                [
                    np.where(
                        state[pending] == a, sigma * lines[t][parent[pending]], 0.0
                    )
                    for t, a, _, sigma in channels
                ]
            )
            cumulative = np.cumsum(rates, axis=0)
            draw = rng.random(pending.size) * cumulative[-1]
            chosen = np.minimum(
                np.sum(cumulative <= draw[None, :], axis=0), len(channels) - 1
            )
            for c, (t, a, b, _) in enumerate(channels):
                hit = chosen == c
                if hit.any():
                    key = (t, self.states[a], self.states[b])
                    events.setdefault(key, []).append(
                        (parent[pending[hit]], share[pending[hit]])
                    )
            new_state = destinations[chosen]
            state[pending] = new_state
            generation[pending] += 1
            depth[pending] = rng.exponential(size=pending.size) - left * loss(
                new_state, parent[pending]
            )
        return ChargeEvents(
            parent=parent,
            state=state,
            depth=depth,
            generation=generation,
            share=share,
            at=at,
            events={
                key: (
                    np.concatenate([r for r, _ in parts]),
                    np.concatenate([s for _, s in parts]),
                )
                for key, parts in events.items()
            },
        )


@dataclass
@method(
    equations={
        "path": r"\text{path}_s = \sum_k N_k \int F_{s,k}\, ds",
        "changes": r"\text{changes}_{t, s \to j} = \sigma_t(s \to j) \sum_k N_k \int F_{s,k}\, n_t\, ds",
        "ionization": r"\text{ionization}_{t, s} = \sigma^\mathrm{ion}_{t, s} \sum_k N_k \int F_{s,k}\, n_t\, ds",
    },
    assumptions=(
        r"in particles per second (per metre of depth in planar geometry); the integrals summed step by step with the mean fractions $\bar F$ of each step",
        r"the ionization cross sections are tallied only: they do not change the beam",
        r"analog mode also sums the sampled events, whose mean is `changes`",
    ),
)
class ChargeTallies:
    """What the beam did to its targets, summed over trajectories (see the module docstring).

    Attributes:
        states: The state names.
        changes: ``{(target, from, to): rate}`` charge changes (particles/s),
            from the path in each state.
        ionization: ``{(target, state): rate}`` ionizations of the targets (particles/s).
        path: ``sum_k N_k int F_s ds`` per state (particles m/s).
        events: ``{(target, from, to): rate}`` sampled charge changes
            (particles/s), in analog mode; the mean of ``changes``.
    """

    states: tuple[str, ...]
    changes: dict[tuple[str, str, str], float] = field(default_factory=dict)
    ionization: dict[tuple[str, str], float] = field(default_factory=dict)
    path: np.ndarray | None = None
    events: dict[tuple[str, str, str], float] = field(default_factory=dict)

    def add(
        self,
        model: ChargeChanging,
        fractions: np.ndarray,
        rate: np.ndarray,
        path: np.ndarray,
        lines: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Add one step of every trajectory: fractions, particle rates, paths.

        ``lines`` are the steps' line densities ``n_t ds`` of every target (the
        uniform densities times ``path`` if None).
        """
        if lines is None:
            if any(isinstance(n, AxialProfile) for n in model.densities.values()):
                raise ValueError("target profiles need the line densities")
            lines = {t: model.density(t, path) * path for t in model.densities}
        weighted = fractions @ (rate * path)  # sum_k N_k F_s,k ds_k per state
        self.path = weighted if self.path is None else self.path + weighted
        index = {s: k for k, s in enumerate(self.states)}
        for target in model.densities:
            on_target = fractions @ (rate * lines[target])  # sum_k N_k F_s,k n ds_k
            for (start, end), sigma in model.transitions.cross_sections[target].items():
                key = (target, start, end)
                value = sigma * on_target[index[start]]
                self.changes[key] = self.changes.get(key, 0.0) + value
            for state, sigma in model.ionization.get(target, {}).items():
                key2 = (target, state)
                value = sigma * on_target[index[state]]
                self.ionization[key2] = self.ionization.get(key2, 0.0) + value

    def add_events(self, events: ChargeEvents, rate: np.ndarray) -> None:
        """Add the sampled events of one step; ``rate`` are the particle rates before it."""
        for key, (rows, share) in events.events.items():
            self.events[key] = self.events.get(key, 0.0) + float(
                np.sum(rate[rows] * share)
            )

    def merge(self, other: ChargeTallies) -> None:
        """Add another rank's tallies."""
        for key, value in other.changes.items():
            self.changes[key] = self.changes.get(key, 0.0) + value
        for key, value in other.ionization.items():
            self.ionization[key] = self.ionization.get(key, 0.0) + value
        for key, value in other.events.items():
            self.events[key] = self.events.get(key, 0.0) + value
        if other.path is not None:
            self.path = other.path if self.path is None else self.path + other.path
