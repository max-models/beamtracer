"""The beamline after the accelerator: neutrals through the duct to the port.

The beam leaves the traced region (:class:`~beamtracer.model.BeamOptics`, the
neutralizer) as :class:`Markers`: points in 3D with a velocity, a power and a charge
state. Beyond it no field acts on the neutrals, so they move on straight lines and
no mesh is needed: :class:`Duct` intersects each line with the walls, given as
simgeometry shapes (:func:`simgeometry.crossings.crossings`), and stops it at the
first wall it enters or at the port, the plane ``x = port``. The beam moves along
``+x``.

On the way the duct gas reionizes the neutrals. A neutral leaves the neutral state
at the rate ``sum_t n_t(x) sigma_t`` per unit length, ``sigma_t`` the cross sections
out of the neutral state on target ``t`` (:class:`plasmacoll.ChargeStateTransitions`);
its free path is sampled from the exponential distribution of the line density along
its path, ``int n_t ds = int n_t dx |v| / v_x``. A reionized neutral is deposited
where it is born, as a bound on where its power goes; the stray field that bends it
is not applied. The transitions back into the neutral state are not followed, as a
reionized particle leaves the beam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from simgeometry.crossings import crossings

from beamtracer.solver.charge_changing import AxialProfile, Density

__all__ = ["Duct", "DuctResult", "Markers", "gaussian_beam"]

#: A crossing this close to the end of a line counts as reaching the port.
PORT_TOLERANCE = 1e-9


def _as_profile(density: Density) -> AxialProfile:
    if isinstance(density, AxialProfile):
        return density
    return AxialProfile((0.0, 1.0), (float(density), float(density)))


@dataclass
class Markers:
    """Beam markers in 3D.

    Attributes:
        position: ``(3, N)`` in m.
        velocity: ``(3, N)`` in m/s.
        power: ``(N,)`` in W.
        state: ``(N,)`` index into ``states``.
        states: The charge-state names.
    """

    position: np.ndarray
    velocity: np.ndarray
    power: np.ndarray
    state: np.ndarray
    states: tuple[str, ...] = ("-", "0", "+")

    def __post_init__(self) -> None:
        """Check the shapes."""
        self.position = np.asarray(self.position, dtype=float).reshape(3, -1)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3, -1)
        num = self.position.shape[1]
        self.power = np.broadcast_to(np.asarray(self.power, dtype=float), (num,)).copy()
        self.state = np.broadcast_to(np.asarray(self.state, dtype=int), (num,)).copy()
        self.states = tuple(self.states)
        if self.velocity.shape[1] != num:
            raise ValueError("position and velocity need the same number of markers")
        if num and (self.state.min() < 0 or self.state.max() >= len(self.states)):
            raise ValueError("state must index states")

    def subset(self, index: np.ndarray | slice) -> Markers:
        """The markers at ``index``."""
        return Markers(
            self.position[:, index],
            self.velocity[:, index],
            self.power[index],
            self.state[index],
            self.states,
        )

    @property
    def total_power(self) -> float:
        """The power of all markers, in W."""
        return float(self.power.sum())

    def footprint(
        self,
        bins: int = 40,
        extent: tuple[float, float, float, float] | None = None,
    ) -> xr.DataArray:
        """The power density (W/m^2) of the markers over ``(y, z)``.

        ``extent`` is ``(y_min, y_max, z_min, z_max)``, the markers' range by default.
        """
        y, z = self.position[1], self.position[2]
        if extent is None:
            extent = (y.min(), y.max(), z.min(), z.max())
        hist, ye, ze = np.histogram2d(
            y,
            z,
            bins=bins,
            range=((extent[0], extent[1]), (extent[2], extent[3])),
            weights=self.power,
        )
        area = np.outer(np.diff(ye), np.diff(ze))
        return xr.DataArray(
            hist / area,
            dims=("y", "z"),
            coords={
                "y": ("y", 0.5 * (ye[1:] + ye[:-1]), {"units": "m"}),
                "z": ("z", 0.5 * (ze[1:] + ze[:-1]), {"units": "m"}),
            },
            name="power_density",
            attrs={"units": "W/m^2", "long_name": "power density"},
        )

    def along_x(self, edges: Sequence[float]) -> xr.DataArray:
        """The power (W/m) of the markers per unit length in ``x``, between ``edges``."""
        e = np.asarray(edges, dtype=float)
        hist, _ = np.histogram(self.position[0], bins=e, weights=self.power)
        return xr.DataArray(
            hist / np.diff(e),
            dims=("x",),
            coords={"x": ("x", 0.5 * (e[1:] + e[:-1]), {"units": "m"})},
            name="power_per_length",
            attrs={"units": "W/m", "long_name": "power per length"},
        )

    def __len__(self) -> int:
        """The number of markers."""
        return self.position.shape[1]


def gaussian_beam(
    num: int,
    *,
    power: float,
    speed: float,
    width: float,
    divergence: float,
    x: float = 0.0,
    state: str = "0",
    states: Sequence[str] = ("-", "0", "+"),
    seed: int | np.random.Generator | None = 0,
) -> Markers:
    """A round Gaussian beam along ``+x``, starting in the plane ``x``.

    Each of ``y`` and ``z`` has the standard deviation ``width`` (m), and each of the
    slopes ``v_y / v_x`` and ``v_z / v_x`` has the standard deviation ``divergence``
    (rad); at a distance ``L`` the width is ``sqrt(width^2 + (L divergence)^2)``
    exactly. Every marker carries ``power / num``.
    """
    rng = np.random.default_rng(seed)
    position = np.zeros((3, num))
    position[0] = x
    position[1:] = rng.normal(0.0, width, (2, num))
    slope = rng.normal(0.0, divergence, (2, num))
    direction = np.vstack([np.ones(num), slope])
    velocity = speed * direction / np.linalg.norm(direction, axis=0)
    states = tuple(states)
    return Markers(
        position,
        velocity,
        np.full(num, power / num),
        np.full(num, states.index(state)),
        states,
    )


@dataclass
class DuctResult:
    """Where the neutrals entering the duct end.

    Attributes:
        port: The neutrals reaching the port, at the port.
        wall: The neutrals stopped by a wall, at the point they hit it.
        wall_index: ``(len(wall),)`` the index of the wall each one hit.
        wall_normal: ``(3, len(wall))`` the wall's outward normal there.
        reionized: The reionized neutrals, where they were reionized, in their new
            state.
        power: The power entering, in W.
    """

    port: Markers
    wall: Markers
    wall_index: np.ndarray
    wall_normal: np.ndarray
    reionized: Markers
    power: float

    @property
    def transmission(self) -> float:
        """The share of the power reaching the port."""
        return self.port.total_power / self.power if self.power else 0.0

    @property
    def reionized_fraction(self) -> float:
        """The share of the power reionized in the duct."""
        return self.reionized.total_power / self.power if self.power else 0.0

    def wall_power(self, walls: int) -> np.ndarray:
        """The power (W) on each of the ``walls`` walls."""
        return np.bincount(self.wall_index, weights=self.wall.power, minlength=walls)

    def summary(self) -> xr.Dataset:
        """The power balance as a dataset: in, port, walls, reionized (W)."""
        return xr.Dataset(
            {
                "power_in": self.power,
                "power_port": self.port.total_power,
                "power_wall": self.wall.total_power,
                "power_reionized": self.reionized.total_power,
                "transmission": self.transmission,
                "reionized_fraction": self.reionized_fraction,
            }
        )


@dataclass
class Duct:
    """The duct from the end of the traced region to the port.

    Attributes:
        walls: The solid walls, simgeometry 3D shapes; the beam runs in the space
            between them. Shapes with analytic crossings (boxes, spheres, slanted tubes)
            are exact and fast; others are sampled (see
            :func:`simgeometry.crossings.crossings`).
        port: The plane ``x = port`` (m) where the duct ends.
        transitions: The charge-changing cross sections
            (:class:`plasmacoll.ChargeStateTransitions`), for reionization; none
            without gas.
        gas: The target densities ``{target: n}`` (m^-3), uniform or an
            :class:`~beamtracer.solver.charge_changing.AxialProfile` along ``x``.
        samples: The samples per line for the walls whose crossings are sampled.
        seed: The random numbers of the free paths.
    """

    walls: Sequence[Any]
    port: float
    transitions: Any = None
    gas: Mapping[str, Density] | None = None
    samples: int = 64
    seed: int | np.random.Generator | None = 0

    def __post_init__(self) -> None:
        """Check the gas against the transitions."""
        self.walls = list(self.walls)
        self._gas: dict[str, Density] = dict(self.gas or {})
        if self._gas and self.transitions is None:
            raise ValueError("gas needs the transitions")
        if self.transitions is not None:
            unknown = set(self._gas) - set(self.transitions.targets)
            if unknown:
                raise ValueError(
                    f"no cross sections for the target(s) {sorted(unknown)}"
                )
        self._rng = np.random.default_rng(self.seed)

    def _loss(self, neutral: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        """The targets, their rates out of ``neutral`` (m^2) and branching ``(t, s)``."""
        targets = list(self._gas)
        rates = np.zeros(len(targets))
        branches = np.zeros((len(targets), len(self.transitions.states)))
        for k, target in enumerate(targets):
            column = self.transitions.matrix(target)[:, neutral].copy()
            rates[k] = -column[neutral]
            column[neutral] = 0.0
            branches[k] = column
        return targets, rates, branches

    def _optical_depth(self, start: float, rates: np.ndarray, targets: list[str]):
        """``tau(x) = sum_t sigma_t int_start^x n_t dx`` on nodes from ``start`` to the port."""
        profiles = [_as_profile(self._gas[t]) for t in targets]
        nodes = np.unique(
            np.concatenate(
                [np.linspace(start, self.port, 1025)]
                + [np.clip(np.asarray(p.x), start, self.port) for p in profiles]
            )
        )
        rate = np.zeros_like(nodes)
        for r, p in zip(rates, profiles, strict=True):
            rate += r * p(nodes)
        steps = 0.5 * (rate[1:] + rate[:-1]) * np.diff(nodes)
        return nodes, np.concatenate([[0.0], np.cumsum(steps)])

    def trace(self, markers: Markers) -> DuctResult:
        """Trace the neutral ``markers`` to the walls or the port.

        Every marker must be neutral (``"0"``) and in front of the port, moving to it;
        the ions are removed before the duct (the residual-ion dump).
        """
        if "0" not in markers.states:
            raise ValueError("the markers have no neutral state '0'")
        neutral = markers.states.index("0")
        if len(markers) and np.any(markers.state != neutral):
            raise ValueError("only neutrals enter the duct; dump the ions first")
        x0 = markers.position[0]
        vx = markers.velocity[0]
        if np.any(vx <= 0.0) or np.any(x0 >= self.port):
            raise ValueError(
                "every marker must move along +x and start before the port"
            )
        num = len(markers)
        start = markers.position
        end = start + markers.velocity * ((self.port - x0) / vx)

        # the first wall along each line, as a fraction of it (1: the port)
        first = np.ones(num)
        index = np.full(num, -1)
        normal = np.full((3, num), np.nan)
        for k, wall in enumerate(self.walls):
            inside = np.asarray(wall.contains_points(start), dtype=bool)
            hit = crossings(wall, start, end, samples=self.samples)
            fraction = np.where(np.asarray(hit.hit), np.asarray(hit.fraction), 2.0)
            fraction = np.where(inside, 0.0, fraction)
            closer = (fraction < first) & (fraction < 1.0 - PORT_TOLERANCE)
            first = np.where(closer, fraction, first)
            index = np.where(closer, k, index)
            normal[:, closer] = np.where(
                inside[closer], np.nan, np.asarray(hit.normal)[:, closer]
            )

        # the reionization point, as a fraction of the line
        born = np.full(num, np.inf)
        state = markers.state.copy()
        if self._gas and num:
            targets, rates, branches = self._loss(neutral)
            nodes, tau = self._optical_depth(float(x0.min()), rates, targets)
            stretch = np.linalg.norm(markers.velocity, axis=0) / vx
            depth = self._rng.exponential(1.0, num) / stretch + np.interp(
                x0, nodes, tau
            )
            reached = depth < tau[-1]
            x_born = np.interp(depth[reached], tau, nodes)
            born[reached] = (x_born - x0[reached]) / (self.port - x0[reached])
            # the product: by the targets' shares of the rate there and their branches
            shares = np.array(
                [
                    r * _as_profile(self._gas[t])(x_born)
                    for t, r in zip(targets, rates, strict=True)
                ]
            )
            weights = (shares.T @ branches) if len(targets) else np.zeros((0, 0))
            cumulative = np.cumsum(weights, axis=1)
            cumulative /= cumulative[:, -1:]
            draw = self._rng.random(reached.sum())
            state[reached] = (draw[:, None] > cumulative).sum(axis=1)

        reionized = born < first
        walled = ~reionized & (index >= 0)
        passed = ~reionized & (index < 0)
        stop = np.where(reionized, born, first)
        point = start + (end - start) * stop
        moved = Markers(point, markers.velocity, markers.power, state, markers.states)
        return DuctResult(
            port=moved.subset(passed),
            wall=moved.subset(walled),
            wall_index=index[walled],
            wall_normal=normal[:, walled],
            reionized=moved.subset(reionized),
            power=markers.total_power,
        )
