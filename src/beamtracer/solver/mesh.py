"""The ion-optics mesh: node types, electrode potentials and wall distances.

The mesh is a uniform node grid of spacing ``h`` in ``(x, y)``, or in ``(x, r)``
for a cylindrically symmetric (``"cylindrical"``) geometry with the axis at ``r = 0``. A node is

- *solid* when an electrode contains it (its potential is the electrode's),
- *Dirichlet* on a face with a Dirichlet condition,
- *near solid* when it is free and a neighbour along ``x`` or ``y`` is solid; the distance to
  the electrode surface along that direction is bracketed in 8 bisection steps, in units of
  ``h / 255``,
- *free* otherwise (including the free nodes of Neumann faces and of the symmetry axis).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Signature of an electrode: a vectorised ``inside(x, y) -> bool array`` test.
InsideFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: The four faces: x min, x max, y (or r) min, y (or r) max.
FACES = ("x_min", "x_max", "y_min", "y_max")

#: The directions of the neighbours: (axis, step) for -x, +x, -y, +y.
DIRECTIONS = ((0, -1), (0, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class Electrode:
    """A solid electrode: an ``inside(x, y)`` test and its potential [V]."""

    inside: InsideFunction
    potential: float
    name: str = ""


@dataclass(frozen=True)
class FaceCondition:
    """The condition on one face of the mesh.

    ``kind`` is ``"dirichlet"`` (``value`` is the potential [V], a number or a vectorised
    function of ``(x, y)``) or ``"neumann"`` (``value`` is the outward normal derivative
    [V/m]).
    """

    kind: str = "neumann"
    value: float | Callable[[np.ndarray, np.ndarray], np.ndarray] = 0.0

    def __post_init__(self) -> None:
        """Check the kind."""
        if self.kind not in ("dirichlet", "neumann"):
            raise ValueError(f"unknown face condition {self.kind!r}")

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the face value at the points ``(x, y)``."""
        if callable(self.value):
            return np.broadcast_to(np.asarray(self.value(x, y), dtype=float), x.shape)
        return np.full(x.shape, float(self.value))


@dataclass
class IonOpticsMesh:
    """A uniform 2D node mesh with electrodes, in planar or cylindrical geometry.

    ``shape`` is the number of nodes ``(nx, ny)``, ``origin`` the position of node ``(0, 0)``
    and ``h`` the node spacing. ``faces`` maps ``"x_min"``, ``"x_max"``, ``"y_min"`` and
    ``"y_max"`` to a :class:`FaceCondition` (default: Neumann). In cylindrical geometry the
    ``y_min`` face is the symmetry axis and must be at ``r = 0``.
    """

    shape: tuple[int, int]
    origin: tuple[float, float]
    h: float
    geometry: str = "cylindrical"
    electrodes: Sequence[Electrode] = ()
    faces: dict[str, FaceCondition] = field(default_factory=dict)

    def _neighbour_solid(self, axis: int, step: int) -> np.ndarray:
        """Return the electrode index of the neighbour of every node in one direction."""
        out = np.full(self.shape, -1, dtype=np.int32)
        if axis == 0:
            if step > 0:
                out[:-1, :] = self.solid[1:, :]
            else:
                out[1:, :] = self.solid[:-1, :]
        else:
            if step > 0:
                out[:, :-1] = self.solid[:, 1:]
            else:
                out[:, 1:] = self.solid[:, :-1]
        return out

    def _bracket(
        self, i: np.ndarray, j: np.ndarray, solid: np.ndarray, axis: int, step: int
    ) -> np.ndarray:
        """Bracket the surface between nodes and their solid neighbours in 8 steps.

        A bisection along the edge, vectorised: the result is in units of ``h / 255`` and
        never zero.
        """
        x0 = self.origin[0] + self.h * i
        y0 = self.origin[1] + self.h * j
        surf = np.full(i.shape, 0x80, dtype=np.int64)
        bit = 0x80
        for _ in range(8):
            offset = step * self.h * surf / 255.0
            xt = x0 + offset if axis == 0 else x0
            yt = y0 + offset if axis == 1 else y0
            inside = np.zeros(i.shape, dtype=bool)
            for number, electrode in enumerate(self.electrodes):
                mine = solid == number
                if mine.any():
                    inside[mine] = np.asarray(
                        electrode.inside(xt[mine], yt[mine]), dtype=bool
                    )
            surf = np.where(inside, surf - bit, surf)
            bit >>= 1
            surf = surf + bit
        surf = np.where(surf == 0, 1, surf)
        return surf / 255.0

    # -- classification ---------------------------------------------------------------------

    def _classify(self) -> None:
        """Set ``solid``, ``fixed``, ``fixed_value`` and the near-solid distances."""
        nx, ny = self.shape
        xn, yn = self.nodes()
        #: Electrode index of every node (-1 for no electrode).
        self.solid = self.inside_solid(xn, yn)
        is_solid = self.solid >= 0
        potentials = np.array([e.potential for e in self.electrodes] + [0.0])
        fixed = is_solid.copy()
        value = np.where(is_solid, potentials[self.solid], 0.0)
        face_nodes = {
            "x_min": (slice(0, 1), slice(None)),
            "x_max": (slice(nx - 1, nx), slice(None)),
            "y_min": (slice(None), slice(0, 1)),
            "y_max": (slice(None), slice(ny - 1, ny)),
        }
        for face in FACES:
            condition = self.faces[face]
            if condition.kind != "dirichlet":
                continue
            index = face_nodes[face]
            free = ~is_solid[index]
            face_value = condition.evaluate(xn[index], yn[index])
            value[index] = np.where(free, face_value, value[index])
            fixed[index] |= free
        #: Nodes with a prescribed potential (electrodes and Dirichlet faces).
        self.fixed = fixed
        #: The prescribed potential of the fixed nodes (0 elsewhere).
        self.fixed_value = value
        #: The fraction of ``h`` to the electrode surface along -x, +x, -y, +y: 1 without one.
        self.distance = np.ones((4, nx, ny))
        #: Free nodes with an electrode neighbour.
        self.near_solid = np.zeros((nx, ny), dtype=bool)
        for d, (axis, step) in enumerate(DIRECTIONS):
            neighbour = self._neighbour_solid(axis, step)
            near = (~is_solid) & (neighbour >= 0)
            if not near.any():
                continue
            self.near_solid |= near
            i, j = np.nonzero(near)
            self.distance[d, i, j] = self._bracket(i, j, neighbour[i, j], axis, step)

    def __post_init__(self) -> None:
        """Classify the nodes and bracket the wall distances."""
        if self.geometry not in ("planar", "cylindrical"):
            raise ValueError(
                f"geometry must be 'planar' or 'cylindrical', got {self.geometry!r}"
            )
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if min(self.shape) < 3:
            raise ValueError(
                f"the mesh needs at least 3 nodes per axis, got {self.shape}"
            )
        self.origin = (float(self.origin[0]), float(self.origin[1]))
        self.h = float(self.h)
        if not self.h > 0.0:
            raise ValueError(f"h must be positive, got {self.h}")
        self.electrodes = tuple(self.electrodes)
        self.faces = {face: self.faces.get(face, FaceCondition()) for face in FACES}
        unknown = set(self.faces) - set(FACES)
        if unknown:
            raise ValueError(f"unknown faces {sorted(unknown)}")
        if self.cylindrical:
            if abs(self.origin[1]) > 1e-12 * max(1.0, self.h):
                raise ValueError("a cylindrical mesh must start on the axis (r = 0)")
            if self.faces["y_min"] != FaceCondition("neumann", 0.0):
                raise ValueError(
                    "the y_min face of a cylindrical mesh is the symmetry axis"
                )
        self._classify()

    # -- geometry ---------------------------------------------------------------------------

    @property
    def cylindrical(self) -> bool:
        """Whether the mesh is cylindrically symmetric (``y`` is the radius)."""
        return self.geometry == "cylindrical"

    @property
    def x(self) -> np.ndarray:
        """The node coordinates along ``x``."""
        return self.origin[0] + self.h * np.arange(self.shape[0])

    @property
    def y(self) -> np.ndarray:
        """The node coordinates along ``y`` (the radius in cylindrical geometry)."""
        return self.origin[1] + self.h * np.arange(self.shape[1])

    @property
    def upper(self) -> tuple[float, float]:
        """The position of the last node ``(nx - 1, ny - 1)``."""
        return (
            self.origin[0] + self.h * (self.shape[0] - 1),
            self.origin[1] + self.h * (self.shape[1] - 1),
        )

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the node coordinates as two ``(nx, ny)`` arrays."""
        return np.meshgrid(self.x, self.y, indexing="ij")

    def inside_solid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the electrode index containing each point, or -1."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        index = np.full(np.broadcast(x, y).shape, -1, dtype=np.int32)
        for number, electrode in enumerate(self.electrodes):
            free = index < 0
            if not free.any():
                break
            inside = np.zeros(index.shape, dtype=bool)
            inside[free] = np.asarray(
                electrode.inside(
                    np.broadcast_to(x, index.shape)[free],
                    np.broadcast_to(y, index.shape)[free],
                ),
                dtype=bool,
            )
            index[inside] = number
        return index
