"""The 3D ion-optics mesh: node types, electrode potentials and wall distances.

The 3D counterpart of `beamtracer.solver.mesh`: a uniform node grid of spacing ``h`` in
``(x, y, z)``. A node is

- *solid* when an electrode contains it (its potential is the electrode's),
- *Dirichlet* on a face with a Dirichlet condition,
- *near solid* when it is free and a neighbour along ``x``, ``y`` or ``z`` is solid; the
  distance to the electrode surface along that direction is bracketed in 8 bisection steps,
  in units of ``h / 255``,
- *free* otherwise (including the free nodes of Neumann faces and of symmetry planes).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from methodcard import method

from beamtracer.solver.mesh import FaceCondition

#: Signature of a 3D electrode: a vectorised ``inside(x, y, z) -> bool array`` test.
InsideFunction3D = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

#: The six faces: x min, x max, y min, y max, z min, z max.
FACES_3D = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

#: The directions of the neighbours: (axis, step) for -x, +x, -y, +y, -z, +z.
DIRECTIONS_3D = ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))

#: The faces that can be symmetry planes.
SYMMETRY_PLANES = ("y_min", "z_min")


@dataclass(frozen=True)
class Electrode3D:
    """A solid electrode in 3D: an ``inside(x, y, z)`` test and its potential [V]."""

    inside: InsideFunction3D
    potential: float
    name: str = ""


def _shift(array: np.ndarray, axis: int, step: int, fill: float | int) -> np.ndarray:
    """``out[i] = array[i + step]`` along ``axis``, ``fill`` beyond the ends."""
    out = np.full_like(array, fill)
    n = array.shape[axis]
    target = [slice(None)] * array.ndim
    source = [slice(None)] * array.ndim
    if step > 0:
        target[axis], source[axis] = slice(0, n - step), slice(step, n)
    else:
        target[axis], source[axis] = slice(-step, n), slice(0, n + step)
    out[tuple(target)] = array[tuple(source)]
    return out


@dataclass
@method(
    equations={
        "nodes": r"x_i = x_0 + i h, \qquad y_j = y_0 + j h, \qquad z_k = z_0 + k h",
        "classes": r"\text{solid} \;(\phi = V_\mathrm{electrode}), \quad \text{Dirichlet face}, \quad \text{near solid} \;(\text{a solid neighbour along } \pm x, \pm y, \pm z), \quad \text{free}",
        "distance": r"d = \frac{s}{255}\, h, \quad s \in \{1, \dots, 255\}, \qquad s \text{ bracketed by 8 bisection steps of the electrode's inside test}",
    },
    assumptions=(
        r"equal spacing $h$ on all axes; at least 3 nodes per axis",
        r"where electrodes overlap the first in the list owns the node; a Dirichlet face fixes only its non-solid nodes",
        r"a symmetry plane (``y_min`` or ``z_min``) has a zero normal derivative: the geometry is mirrored there",
    ),
)
class IonOpticsMesh3D:
    """A uniform 3D node mesh with electrodes.

    ``shape`` is the number of nodes ``(nx, ny, nz)``, ``origin`` the position of node
    ``(0, 0, 0)`` and ``h`` the node spacing. ``faces`` maps the six `FACES_3D` to a
    `FaceCondition` (default: Neumann), whose functions take ``(x, y, z)``.
    ``symmetry_planes`` lists the lower ``y`` and ``z`` faces at which the geometry is
    mirrored: they need a zero normal derivative, and trajectories crossing them continue
    in the mirrored field.
    """

    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    h: float
    electrodes: Sequence[Electrode3D] = ()
    faces: dict[str, FaceCondition] = field(default_factory=dict)
    symmetry_planes: tuple[str, ...] = ()

    #: Planar coordinates, never cylindrical (the 2D mesh's flag, for shared code).
    cylindrical = False
    #: The number of dimensions.
    ndim = 3

    # -- classification ---------------------------------------------------------------------

    def _bracket(
        self, index: tuple[np.ndarray, ...], solid: np.ndarray, axis: int, step: int
    ) -> np.ndarray:
        """Bracket the surface between nodes and their solid neighbours in 8 steps.

        A bisection along the edge, vectorised: the result is in units of ``h / 255`` and
        never zero.
        """
        start = [self.origin[a] + self.h * index[a] for a in range(3)]
        surf = np.full(index[0].shape, 0x80, dtype=np.int64)
        bit = 0x80
        for _ in range(8):
            point = list(start)
            point[axis] = start[axis] + step * self.h * surf / 255.0
            inside = np.zeros(surf.shape, dtype=bool)
            for number, electrode in enumerate(self.electrodes):
                mine = solid == number
                if mine.any():
                    inside[mine] = np.asarray(
                        electrode.inside(*(p[mine] for p in point)), dtype=bool
                    )
            surf = np.where(inside, surf - bit, surf)
            bit >>= 1
            surf = surf + bit
        surf = np.where(surf == 0, 1, surf)
        return surf / 255.0

    def _classify(self) -> None:
        """Set ``solid``, ``fixed``, ``fixed_value`` and the near-solid distances."""
        points = self.nodes()
        #: Electrode index of every node (-1 for no electrode).
        self.solid = self.inside_solid(*points)
        is_solid = self.solid >= 0
        potentials = np.array([e.potential for e in self.electrodes] + [0.0])
        fixed = is_solid.copy()
        value = np.where(is_solid, potentials[self.solid], 0.0)
        for number, face in enumerate(FACES_3D):
            condition = self.faces[face]
            if condition.kind != "dirichlet":
                continue
            axis, upper = divmod(number, 2)
            index = [slice(None)] * 3
            index[axis] = slice(-1, None) if upper else slice(0, 1)
            where = tuple(index)
            free = ~is_solid[where]
            face_value = condition.evaluate(*(p[where] for p in points))
            value[where] = np.where(free, face_value, value[where])
            fixed[where] |= free
        #: Nodes with a prescribed potential (electrodes and Dirichlet faces).
        self.fixed = fixed
        #: The prescribed potential of the fixed nodes (0 elsewhere).
        self.fixed_value = value
        #: The fraction of ``h`` to the electrode surface along -x, +x, -y, +y, -z, +z:
        #: 1 without one.
        self.distance = np.ones((6, *self.shape))
        #: Free nodes with an electrode neighbour.
        self.near_solid = np.zeros(self.shape, dtype=bool)
        for d, (axis, step) in enumerate(DIRECTIONS_3D):
            neighbour = _shift(self.solid, axis, step, -1)
            near = ~is_solid & (neighbour >= 0)
            self.near_solid |= near
            index = np.nonzero(near)
            self.distance[(d, *index)] = self._bracket(
                index, neighbour[index], axis, step
            )

    def __post_init__(self) -> None:
        """Check the settings, classify the nodes and bracket the wall distances."""
        self.shape = tuple(int(n) for n in self.shape)  # type: ignore[assignment]
        if len(self.shape) != 3 or min(self.shape) < 3:
            raise ValueError(
                f"the mesh needs 3 axes of at least 3 nodes, got {self.shape}"
            )
        self.origin = tuple(float(o) for o in self.origin)  # type: ignore[assignment]
        self.h = float(self.h)
        if not self.h > 0.0:
            raise ValueError(f"h must be positive, got {self.h}")
        self.electrodes = tuple(self.electrodes)
        unknown = set(self.faces) - set(FACES_3D)
        if unknown:
            raise ValueError(f"unknown faces {sorted(unknown)}")
        self.faces = {face: self.faces.get(face, FaceCondition()) for face in FACES_3D}
        self.symmetry_planes = tuple(self.symmetry_planes)
        for plane in self.symmetry_planes:
            if plane not in SYMMETRY_PLANES:
                raise ValueError(
                    f"a symmetry plane is one of {SYMMETRY_PLANES}, got {plane!r}"
                )
            if self.faces[plane] != FaceCondition("neumann", 0.0):
                raise ValueError(
                    f"the symmetry plane {plane} needs a zero normal derivative"
                )
        self._classify()

    # -- geometry ---------------------------------------------------------------------------

    def _axis(self, axis: int) -> np.ndarray:
        """The node coordinates along ``axis``."""
        return self.origin[axis] + self.h * np.arange(self.shape[axis])

    @property
    def x(self) -> np.ndarray:
        """The node coordinates along ``x``."""
        return self._axis(0)

    @property
    def y(self) -> np.ndarray:
        """The node coordinates along ``y``."""
        return self._axis(1)

    @property
    def z(self) -> np.ndarray:
        """The node coordinates along ``z``."""
        return self._axis(2)

    @property
    def upper(self) -> tuple[float, float, float]:
        """The position of the last node."""
        return tuple(  # type: ignore[return-value]
            self.origin[a] + self.h * (self.shape[a] - 1) for a in range(3)
        )

    def nodes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the node coordinates as three ``(nx, ny, nz)`` arrays."""
        return tuple(np.meshgrid(self.x, self.y, self.z, indexing="ij"))  # type: ignore[return-value]

    def inside_solid(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the electrode index containing each point, or -1."""
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        index = np.full(x.shape, -1, dtype=np.int32)
        for number, electrode in enumerate(self.electrodes):
            free = index < 0
            if not free.any():
                break
            inside = np.zeros(index.shape, dtype=bool)
            inside[free] = np.asarray(
                electrode.inside(x[free], y[free], z[free]), dtype=bool
            )
            index[inside] = number
        return index
