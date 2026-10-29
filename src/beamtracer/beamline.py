"""The beamline after the accelerator: neutrals through the duct to the port.

The beam leaves the traced region (:class:`~beamtracer.model.BeamOptics`, the
neutralizer) as :class:`Markers`: points in 3D with a velocity, a power and a charge
state. :func:`beamlet_markers` makes them from a beamlet traced in 2D, and
:class:`ApertureGrid` copies the beamlet to every aperture of the grid with its aim. Beyond it no field acts on the neutrals, so they move on straight lines and
no mesh is needed: :class:`Duct` intersects each line with the walls, given as
simgeometry shapes (:func:`simgeometry.crossings.crossings`), and stops it at the
first wall it enters or at the port, the plane ``x = port``. The beam moves along
``+x``.

On the way the duct gas reionizes the neutrals. A neutral leaves the neutral state
at the rate ``sum_t n_t(x) sigma_t`` per unit length, ``sigma_t`` the cross sections
out of the neutral state on target ``t`` (:class:`plasmacoll.ChargeStateTransitions`);
its free path is sampled from the exponential distribution of the line density along
its path, ``int n_t ds = int n_t dx |v| / v_x``. A reionized neutral stops where it is
born; the transitions back into the neutral state are not followed, as a reionized
particle leaves the beam.

:class:`FieldRegion` pushes charged markers through static electric and magnetic
fields to the walls: the residual-ion dump, whose plates deflect the ions left after the
neutralizer out of the beam, and the stray field of the tokamak, which bends the ions
reionized in the duct onto its walls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy import constants
from simgeometry.crossings import crossings

from beamtracer.solver.charge_changing import AxialProfile, Density
from beamtracer.solver.tracer import PlaneHits

__all__ = [
    "ApertureGrid",
    "Duct",
    "DuctResult",
    "FieldRegion",
    "Markers",
    "RegionResult",
    "beamlet_markers",
    "box_field",
    "gaussian_beam",
    "profile_field",
]

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

    def drift_to(self, x: float) -> Markers:
        """The markers moved on straight lines to the plane ``x``."""
        time = (x - self.position[0]) / self.velocity[0]
        return Markers(
            self.position + self.velocity * time,
            self.velocity,
            self.power,
            self.state,
            self.states,
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


def beamlet_markers(
    hits: PlaneHits,
    *,
    mass: float,
    cylindrical: bool,
    copies: int = 1,
    charge: float = 1.0,
    slit_length: float | None = None,
    states: Sequence[str] = ("-", "0", "+"),
    state: str = "-",
    species: int | None = None,
    seed: int | np.random.Generator | None = 0,
) -> Markers:
    """The 3D markers of a beamlet traced in 2D, from its crossings of a plane.

    ``hits`` are the crossings of a diagnostic plane
    (:meth:`~beamtracer.model.BeamOptics.plane`). In cylindrical geometry a trajectory
    stands for a ring: each becomes ``copies`` markers at random azimuths, its point
    and velocity rotated about the axis. In planar geometry it stands for a slit's
    section per unit depth: each becomes ``copies`` markers spread at random over
    ``z`` in ``(-slit_length / 2, slit_length / 2)``, carrying the current times the
    slit length. The power of a trajectory is its particle current ``I / (charge e)``
    (``charge`` the charge of the particles the current counts, in e) times their
    kinetic energy, ``mass`` (kg) each; with charge-state ``fractions`` it is split
    into one marker per state, else all is in ``state``. Only the forward crossings
    count, of the beam ``species`` (an index) if given.
    """
    if copies < 1:
        raise ValueError("copies must be at least 1")
    if not cylindrical and slit_length is None:
        raise ValueError("a planar beamlet needs the slit_length")
    states = tuple(states)
    rng = np.random.default_rng(seed)
    keep = hits.direction > 0
    if species is not None:
        keep &= hits.species == species
    point, speed = hits.point[:, keep], hits.velocity[:, keep]
    speed2 = np.sum(speed**2, axis=0)
    power = hits.current[keep] / (charge * constants.e) * 0.5 * mass * speed2
    if not cylindrical:
        power = power * float(slit_length)  # type: ignore[arg-type]
    num = power.size
    if hits.fractions is None:
        share = np.zeros((len(states), num))
        share[states.index(state)] = 1.0
    else:
        share = np.asarray(hits.fractions, dtype=float)[:, keep]
        if share.shape[0] != len(states):
            raise ValueError("the fractions need one row per state")
    # one marker per (trajectory, state) with a share, then the copies
    s_index, t_index = np.nonzero(share > 0.0)
    marker_power = power[t_index] * share[s_index, t_index] / copies
    t_index = np.repeat(t_index, copies)
    s_index = np.repeat(s_index, copies)
    marker_power = np.repeat(marker_power, copies)
    position = point[:, t_index].copy()
    velocity = speed[:, t_index].copy()
    if cylindrical:
        angle = rng.uniform(0.0, 2.0 * np.pi, t_index.size)
        c, s = np.cos(angle), np.sin(angle)
        for vector in (position, velocity):
            y, z = vector[1].copy(), vector[2].copy()
            vector[1] = c * y - s * z
            vector[2] = s * y + c * z
    else:
        position[2] = rng.uniform(-0.5, 0.5, t_index.size) * float(slit_length)  # type: ignore[arg-type]
    return Markers(position, velocity, marker_power, s_index, states)


def _rotation_from_x(direction: np.ndarray) -> np.ndarray:
    """The rotation taking the unit vector along ``x`` to the unit ``direction``."""
    d = direction / np.linalg.norm(direction)
    axis = np.cross([1.0, 0.0, 0.0], d)
    sine, cosine = np.linalg.norm(axis), d[0]
    if sine < 1e-15:
        return np.eye(3)
    k = axis / sine
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sine * cross + (1 - cosine) * cross @ cross


@dataclass
class ApertureGrid:
    """The apertures of an accelerator grid, each emitting a copy of one beamlet.

    The beamlet is traced about the ``x`` axis; :meth:`markers` copies it to every
    aperture centre, turned to the aperture's aim. The grid lies in the plane of the
    beamlet's markers.

    Attributes:
        centres: ``(2, M)`` the apertures' ``(y, z)`` (m).
        focal_length: ``(f_y, f_z)`` (m), the distance at which the grid's curvature
            aims the beamlets of each direction at the axis: a beamlet at ``y`` gets
            the slope ``-y / f_y``. ``inf`` for a flat grid.
        steering: ``(2, M)`` further slopes (rad) of each beamlet in ``y`` and ``z``,
            e.g. from aperture offsets.
    """

    centres: np.ndarray
    focal_length: tuple[float, float] = (np.inf, np.inf)
    steering: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Check the shapes."""
        self.centres = np.asarray(self.centres, dtype=float).reshape(2, -1)
        steering = (
            np.zeros_like(self.centres) if self.steering is None else self.steering
        )
        self.steering = np.broadcast_to(
            np.asarray(steering, dtype=float), self.centres.shape
        ).copy()
        self.focal_length = (float(self.focal_length[0]), float(self.focal_length[1]))

    @classmethod
    def regular(
        cls,
        rows: int,
        columns: int,
        pitch: tuple[float, float],
        **kwargs: Any,
    ) -> ApertureGrid:
        """``rows`` (along ``z``) by ``columns`` (along ``y``) apertures, centred on the axis.

        ``pitch`` is the spacing ``(in y, in z)`` (m); ``kwargs`` go to the grid.
        """
        y = (np.arange(columns) - 0.5 * (columns - 1)) * pitch[0]
        z = (np.arange(rows) - 0.5 * (rows - 1)) * pitch[1]
        yy, zz = np.meshgrid(y, z, indexing="xy")
        return cls(np.vstack([yy.ravel(), zz.ravel()]), **kwargs)

    @property
    def slopes(self) -> np.ndarray:
        """``(2, M)`` the slopes ``v_y / v_x`` and ``v_z / v_x`` of each beamlet's axis."""
        f = np.array(self.focal_length)[:, None]
        assert self.steering is not None
        return -self.centres / f + self.steering

    def markers(self, beamlet: Markers) -> Markers:
        """The whole beam: ``beamlet`` (about the ``x`` axis) copied to every aperture."""
        origin = np.array([beamlet.position[0].mean(), 0.0, 0.0])[:, None]
        offset = beamlet.position - origin
        positions, velocities = [], []
        for centre, slope in zip(self.centres.T, self.slopes.T, strict=True):
            rotation = _rotation_from_x(np.array([1.0, slope[0], slope[1]]))
            shift = origin + np.array([0.0, centre[0], centre[1]])[:, None]
            positions.append(rotation @ offset + shift)
            velocities.append(rotation @ beamlet.velocity)
        count = len(self)
        return Markers(
            np.hstack(positions),
            np.hstack(velocities),
            np.tile(beamlet.power, count),
            np.tile(beamlet.state, count),
            beamlet.states,
        )

    def __len__(self) -> int:
        """The number of apertures."""
        return self.centres.shape[1]


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


def box_field(
    vector: Sequence[float],
    lower: Sequence[float] = (-np.inf, -np.inf, -np.inf),
    upper: Sequence[float] = (np.inf, np.inf, np.inf),
):
    """A uniform field ``vector`` inside the box from ``lower`` to ``upper``, zero outside.

    E.g. the field between two parallel plates (V/m), or a uniform stray field (T).
    """
    value = np.asarray(vector, dtype=float)[:, None]
    lo = np.asarray(lower, dtype=float)[:, None]
    hi = np.asarray(upper, dtype=float)[:, None]

    def field(points: np.ndarray) -> np.ndarray:
        inside = np.all((points >= lo) & (points <= hi), axis=0)
        return value * inside

    return field


def profile_field(x: Sequence[float], values: Sequence[Sequence[float]]):
    """A field along ``x``: ``values`` is ``(3, len(x))``, linear between, held beyond.

    E.g. the stray field of the tokamak along the beamline (T).
    """
    nodes = np.asarray(x, dtype=float)
    table = np.asarray(values, dtype=float).reshape(3, nodes.size)
    if nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("a profile needs at least two increasing x")

    def field(points: np.ndarray) -> np.ndarray:
        return np.vstack([np.interp(points[0], nodes, row) for row in table])

    return field


@dataclass
class RegionResult:
    """Where the markers crossing a :class:`FieldRegion` end.

    Attributes:
        exit: The markers reaching the exit plane, there.
        wall: The markers stopped by a wall, at the point they hit it.
        wall_index: ``(len(wall),)`` the index of the wall each one hit.
        wall_normal: ``(3, len(wall))`` the wall's outward normal there.
        lost: The markers still in the region after ``max_steps``.
        power: The power entering, in W.
        trajectories: The paths ``(3, steps)`` of the first ``save`` markers.
    """

    exit: Markers
    wall: Markers
    wall_index: np.ndarray
    wall_normal: np.ndarray
    lost: Markers
    power: float
    trajectories: list[np.ndarray]

    def wall_power(self, walls: int) -> np.ndarray:
        """The power (W) on each of the ``walls`` walls."""
        return np.bincount(self.wall_index, weights=self.wall.power, minlength=walls)

    def summary(self) -> xr.Dataset:
        """The power balance as a dataset: in, exit, walls, lost (W)."""
        return xr.Dataset(
            {
                "power_in": self.power,
                "power_exit": self.exit.total_power,
                "power_wall": self.wall.total_power,
                "power_lost": self.lost.total_power,
            }
        )


@dataclass
class FieldRegion:
    """Markers through static electric and magnetic fields, to the walls or an exit plane.

    The residual-ion dump (plates biased to deflect the ions out of the beam) or a
    stretch of the beamline in the stray field. Every marker moves by the Boris scheme
    in steps of ``step`` (m) along its path, its charge set by its state; the neutrals
    move on straight lines. Each step is tested against the walls as a straight segment
    (:func:`simgeometry.crossings.crossings`), so ``step`` should be well below the
    radius of curvature.

    Attributes:
        walls: The solid walls, simgeometry 3D shapes.
        exit: The plane ``x = exit`` (m) where the region ends.
        mass: The mass of the beam particles (kg).
        electric: ``E(points)``, ``(3, N)`` in V/m at ``(3, N)`` points; none if absent.
        magnetic: ``B(points)`` in T, the same way; :func:`box_field` and
            :func:`profile_field` make simple ones.
        charges: The charge of each state, in e.
        step: The path length per step (m).
        max_steps: The steps after which a marker counts as lost.
        save: Keep the paths of the first ``save`` markers.
        samples: The samples per step for the walls whose crossings are sampled.
    """

    walls: Sequence[Any]
    exit: float
    mass: float
    electric: Any = None
    magnetic: Any = None
    charges: Mapping[str, float] | None = None
    step: float = 5e-3
    max_steps: int = 10_000
    save: int = 0
    samples: int = 16

    def __post_init__(self) -> None:
        """Fill in the charges."""
        self.walls = list(self.walls)
        self._charges = dict(self.charges or {"-": -1.0, "0": 0.0, "+": 1.0})
        if self.step <= 0.0:
            raise ValueError("step must be positive")

    def _field(self, which: Any, points: np.ndarray) -> np.ndarray:
        if which is None:
            return np.zeros_like(points)
        return np.asarray(which(points), dtype=float).reshape(points.shape)

    def _boris(
        self, x: np.ndarray, v: np.ndarray, qm: np.ndarray, dt: np.ndarray
    ) -> np.ndarray:
        """The velocity after ``dt`` in the fields at ``x`` (the Boris rotation)."""
        half = 0.5 * qm * dt
        e = self._field(self.electric, x)
        b = self._field(self.magnetic, x)
        minus = v + half * e
        t = half * b
        s = 2.0 * t / (1.0 + np.sum(t * t, axis=0))
        prime = minus + np.cross(minus, t, axis=0)
        return minus + np.cross(prime, s, axis=0) + half * e

    def trace(self, markers: Markers) -> RegionResult:
        """Push the ``markers`` until each hits a wall or crosses the exit plane."""
        missing = set(markers.states) - set(self._charges)
        if missing:
            raise ValueError(f"no charge for the state(s) {sorted(missing)}")
        num = len(markers)
        charge = np.array([self._charges[s] for s in markers.states])[markers.state]
        qm = charge * constants.e / self.mass
        x = markers.position.copy()
        v = markers.velocity.copy()
        outcome = np.full(num, -2)  # -2 lost, -1 exit, k wall k
        normal = np.full((3, num), np.nan)
        paths = [[x[:, k].copy()] for k in range(min(self.save, num))]

        active = np.arange(num)
        for k, wall in enumerate(self.walls):
            inside = np.asarray(wall.contains_points(x), dtype=bool)
            outcome[inside & (outcome == -2)] = k
        active = active[outcome[active] == -2]
        done = x[0] >= self.exit
        outcome[active[done[active]]] = -1
        active = active[~done[active]]

        # the leapfrog starts from the velocity half a step back
        if active.size:
            xa, va = x[:, active], v[:, active]
            dt = self.step / np.linalg.norm(va, axis=0)
            v[:, active] = self._boris(xa, va, qm[active], -0.5 * dt)

        for _ in range(self.max_steps):
            if active.size == 0:
                break
            xa, va, q = x[:, active], v[:, active], qm[active]
            dt = self.step / np.linalg.norm(va, axis=0)
            vn = self._boris(xa, va, q, dt)
            xn = xa + vn * dt

            # the first event along the step: a wall, or the exit plane
            first = np.full(active.size, 2.0)
            which = np.full(active.size, -2)
            step_normal = np.full((3, active.size), np.nan)
            for k, wall in enumerate(self.walls):
                hit = crossings(wall, xa, xn, samples=self.samples)
                fraction = np.where(np.asarray(hit.hit), np.asarray(hit.fraction), 2.0)
                closer = fraction < first
                first = np.where(closer, fraction, first)
                which = np.where(closer, k, which)
                step_normal[:, closer] = np.asarray(hit.normal)[:, closer]
            with np.errstate(divide="ignore", invalid="ignore"):
                to_exit = (self.exit - xa[0]) / (xn[0] - xa[0])
            leaves = (xn[0] >= self.exit) & (to_exit <= first)
            first = np.where(leaves, to_exit, first)
            which = np.where(leaves, -1, which)
            stops = first <= 1.0
            point = np.where(stops, xa + (xn - xa) * np.minimum(first, 1.0), xn)
            x[:, active] = point
            v[:, active] = vn
            for k in range(len(paths)):
                where = np.flatnonzero(active == k)
                if where.size:
                    paths[k].append(point[:, where[0]].copy())
            ended = active[stops]
            outcome[ended] = which[stops]
            normal[:, ended] = step_normal[:, stops]
            active = active[~stops]

        moved = Markers(x, v, markers.power, markers.state, markers.states)
        walled = outcome >= 0
        return RegionResult(
            exit=moved.subset(outcome == -1),
            wall=moved.subset(walled),
            wall_index=outcome[walled],
            wall_normal=normal[:, walled],
            lost=moved.subset(outcome == -2),
            power=markers.total_power,
            trajectories=[np.array(p).T for p in paths],
        )
