"""Refined boxes of an ion-optics mesh: their patches, the composite nodes and the field.

The field is solved on patchwork's composite mesh: the base mesh with boxes of finer cells
(each level halves the cells of the one below). Every patch, the base mesh included, is a
uniform `IonOpticsMesh` of its own (its electrodes classified at its spacing), whose nodes
are composite nodes; a trajectory takes the field of, and deposits its charge on, the finest
patch it is in. In cylindrical geometry patchwork's axes are ``(r, x)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from beamtracer.solver.fields import StaggeredElectricField
from beamtracer.solver.mesh import IonOpticsMesh


@dataclass(frozen=True)
class Refinement:
    """A box ``lower``-``upper`` [m] (``(x, y)`` corners) refined to ``level``."""

    level: int
    lower: tuple[float, float]
    upper: tuple[float, float]


@dataclass
class Patch:
    """A uniform part of the composite mesh: its level, cells and mesh.

    ``box`` is ``((i0, i1), (j0, j1))`` in the level's cells along ``(x, y)``; ``index`` is
    the composite node of every node of ``mesh`` (set by the layout).
    """

    level: int
    box: tuple[tuple[int, int], tuple[int, int]]
    mesh: IonOpticsMesh
    index: np.ndarray | None = None
    #: The mesh of the patch's field: one cell of the level below wider on every side
    #: inside the domain, so the field at the box's edges sees the coarser potential
    #: beyond it instead of extrapolating; ``inner`` is the box's nodes in it.
    field_mesh: IonOpticsMesh | None = None
    inner: tuple[slice, slice] = (slice(None), slice(None))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The patch's node extent ``(x_lower, y_lower, x_upper, y_upper)`` [m]."""
        (x0, y0), (x1, y1) = self.mesh.origin, self.mesh.upper
        return x0, y0, x1, y1


def patches_of(mesh: IonOpticsMesh, refinement: Sequence[Refinement]) -> list[Patch]:
    """The base patch and one patch per refined box, coarse to fine.

    A box is widened to the cells of the level below, so it starts and ends on that level's
    nodes, and clipped to the mesh.
    """
    nx, ny = mesh.shape
    patches = [Patch(0, ((0, nx - 1), (0, ny - 1)), mesh)]
    for box in sorted(refinement, key=lambda b: b.level):
        level = int(box.level)
        parent_h = mesh.h / 2 ** (level - 1)
        cells = []
        for axis in range(2):
            lower = max(mesh.origin[axis], box.lower[axis])
            upper = min(mesh.upper[axis], box.upper[axis])
            n_parent = round((mesh.upper[axis] - mesh.origin[axis]) / parent_h)
            i0 = int(np.floor((lower - mesh.origin[axis]) / parent_h + 1e-9))
            i1 = int(np.ceil((upper - mesh.origin[axis]) / parent_h - 1e-9))
            i0, i1 = max(i0, 0), min(i1, n_parent)
            if i1 <= i0:
                raise ValueError(f"the refined box {box} is outside the mesh")
            cells.append((2 * i0, 2 * i1))
        h = mesh.h / 2**level
        origin = tuple(mesh.origin[a] + cells[a][0] * h for a in range(2))
        shape = tuple(cells[a][1] - cells[a][0] + 1 for a in range(2))
        patch_mesh = IonOpticsMesh(
            shape=shape,
            origin=origin,
            h=h,
            geometry=mesh.geometry,
            electrodes=mesh.electrodes,
            patch=True,
        )
        # the field's mesh: two cells (one of the level below) more on each side
        lower_margin = [
            min(2, round((origin[a] - mesh.origin[a]) / h)) for a in range(2)
        ]
        upper_margin = [
            min(2, round((mesh.upper[a] - (origin[a] + (shape[a] - 1) * h)) / h))
            for a in range(2)
        ]
        field_mesh = IonOpticsMesh(
            shape=tuple(shape[a] + lower_margin[a] + upper_margin[a] for a in range(2)),
            origin=tuple(origin[a] - lower_margin[a] * h for a in range(2)),
            h=h,
            geometry=mesh.geometry,
            electrodes=mesh.electrodes,
            patch=True,
        )
        inner = (
            slice(lower_margin[0], lower_margin[0] + shape[0]),
            slice(lower_margin[1], lower_margin[1] + shape[1]),
        )
        patches.append(
            Patch(level, (cells[0], cells[1]), patch_mesh, None, field_mesh, inner)
        )
    return patches


@dataclass
class Layout:
    """The composite nodes of a refined mesh: the patches' node maps and the volumes.

    ``volumes`` are the control volumes the deposit divides by: a quarter of every leaf
    cell (of its ring volume ``2 pi r_c |K|`` in cylindrical geometry) at each corner, the
    volumes the area-weighted (in r**2) deposit is consistent with. ``x`` and ``y`` are the
    nodes' coordinates.
    """

    patches: list[Patch]
    x: np.ndarray
    y: np.ndarray
    volumes: np.ndarray

    @property
    def size(self) -> int:
        """The number of composite nodes."""
        return int(self.x.size)

    @property
    def base(self) -> Patch:
        """The base patch: the whole mesh."""
        return self.patches[0]

    @property
    def refined(self) -> bool:
        """Whether there is a patch besides the base."""
        return len(self.patches) > 1

    def on_base(self, values: np.ndarray) -> np.ndarray:
        """The ``(nx, ny)`` array of composite values on the base mesh."""
        index = self.base.index
        assert index is not None
        return np.asarray(values)[index]

    def from_base(self, array: np.ndarray) -> np.ndarray:
        """Composite values of an ``(nx, ny)`` array, without refinement only."""
        if self.refined:
            raise ValueError(
                "with refinement give values on the composite nodes (`Layout.x`, `Layout.y`)"
            )
        index = self.base.index
        assert index is not None
        out = np.empty(self.size)
        out[index.ravel()] = np.asarray(array, dtype=float).ravel()
        return out

    def deposit_total(self, charges: Sequence[np.ndarray]) -> np.ndarray:
        """The composite node charges of per-patch node charges (shared nodes added)."""
        total = np.zeros(self.size)
        for patch, charge in zip(self.patches, charges, strict=True):
            assert patch.index is not None
            np.add.at(total, patch.index.ravel(), np.asarray(charge).ravel())
        return total


def layout_of(mesh: IonOpticsMesh, patches: list[Patch], nodes) -> Layout:
    """The `Layout` of ``patches`` on patchwork's `CompositeNodes` ``nodes``.

    patchwork's axes are ``(x, y)``, or ``(r, x)`` in cylindrical geometry.
    """
    cylindrical = mesh.cylindrical
    finest = max(p.level for p in patches)
    lattice = np.asarray(nodes.lattice, dtype=np.int64)
    # the mesh's axes (x, y) of the finest lattice
    lx, ly = (
        (lattice[:, 1], lattice[:, 0])
        if cylindrical
        else (lattice[:, 0], lattice[:, 1])
    )
    span = int(ly.max()) + 1
    keys = lx * span + ly
    order = np.argsort(keys)
    for patch in patches:
        scale = 2 ** (finest - patch.level)
        nxp, nyp = patch.mesh.shape
        i = (patch.box[0][0] + np.arange(nxp))[:, None] * scale
        j = (patch.box[1][0] + np.arange(nyp))[None, :] * scale
        wanted = (i * span + j).ravel()
        found = order[np.searchsorted(keys, wanted, sorter=order)]
        if not np.array_equal(keys[found], wanted):  # pragma: no cover
            raise RuntimeError("a patch node is not a composite node")
        patch.index = found.reshape(nxp, nyp)
    coordinates = np.asarray(nodes.coordinates)
    x, y = (
        (coordinates[:, 1], coordinates[:, 0])
        if cylindrical
        else (coordinates[:, 0], coordinates[:, 1])
    )
    cell = np.prod(nodes.cell_size, axis=1)
    if cylindrical:
        cell = 2.0 * np.pi * nodes.cell_centers[:, 0] * cell
    corners = nodes.cell_corners
    volumes = np.bincount(
        corners.reshape(-1),
        weights=np.repeat(cell / corners.shape[1], corners.shape[1]),
        minlength=x.size,
    )
    return Layout(patches, x.copy(), y.copy(), volumes)


def _extrapolation_of(mesh: IonOpticsMesh) -> list[str]:
    """The field's treatment of a mesh's faces: symmetric only on the axis."""
    faces = ["extrapolate"] * 4
    if mesh.cylindrical and abs(mesh.origin[1]) <= 1e-12 * mesh.h:
        faces[2] = "symmetric"
    return faces


def _bilinear(
    patches: list[Patch], arrays: list[np.ndarray], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """The potential at points from the patches done so far (``arrays``), finest wins."""
    out = np.zeros(x.size)
    for patch, values in zip(patches, arrays, strict=False):
        mesh = patch.field_mesh if patch.field_mesh is not None else patch.mesh
        x0, y0, x1, y1 = patch.bounds
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if not inside.any():
            continue
        nxp, nyp = mesh.shape
        s = (x[inside] - mesh.origin[0]) / mesh.h
        t = (y[inside] - mesh.origin[1]) / mesh.h
        i = np.clip(np.floor(s).astype(np.int64), 0, nxp - 2)
        j = np.clip(np.floor(t).astype(np.int64), 0, nyp - 2)
        u, w = s - i, t - j
        out[inside] = (
            (1 - u) * (1 - w) * values[i, j]
            + (1 - u) * w * values[i, j + 1]
            + u * (1 - w) * values[i + 1, j]
            + u * w * values[i + 1, j + 1]
        )
    return out


class PatchedElectricField:
    """The electric field of a composite potential: a `StaggeredElectricField` per patch.

    At a point the field is that of the finest patch containing it. ``fx`` and ``fy``
    are the base patch's edge fields (as `StaggeredElectricField`'s).
    """

    def __init__(self, layout: Layout, potential: np.ndarray) -> None:
        """Precompute the edge fields of every patch from the composite ``potential``.

        A refined patch's field covers its `Patch.field_mesh`: its nodes outside the box
        take the potential of the coarser patches (bilinear), those in an electrode its
        potential.
        """
        self.layout = layout
        self.patches = layout.patches
        potential = np.asarray(potential)
        arrays: list[np.ndarray] = []
        self.fields = []
        for patch in layout.patches:
            values = potential[patch.index]
            mesh = patch.field_mesh
            if mesh is None:
                mesh = patch.mesh
            else:
                xn, yn = mesh.nodes()
                full = _bilinear(layout.patches, arrays, xn.ravel(), yn.ravel())
                full = full.reshape(mesh.shape)
                solid = mesh.solid >= 0
                full[solid] = mesh.fixed_value[solid]
                full[patch.inner] = values
                values = full
            arrays.append(values)
            self.fields.append(
                StaggeredElectricField(mesh, values, _extrapolation_of(mesh))
            )
        self.mesh = layout.base.mesh
        self.fx, self.fy = self.fields[0].fx, self.fields[0].fy

    def patch_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The finest patch containing each point."""
        which = np.zeros(np.shape(x), dtype=np.int64)
        for number, patch in enumerate(self.patches[1:], start=1):
            x0, y0, x1, y1 = patch.bounds
            which[(x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)] = number
        return which

    def spacing(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The node spacing of the finest patch at each point."""
        h = np.array([patch.mesh.h for patch in self.patches])
        return h[self.patch_of(x, y)]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(E_x, E_y)`` [V/m] at the points ``(x, y)``."""
        x = np.ascontiguousarray(x, dtype=float).ravel()
        y = np.ascontiguousarray(y, dtype=float).ravel()
        ex, ey = self.fields[0](x, y)
        if len(self.patches) > 1:
            which = self.patch_of(x, y)
            for number in range(1, len(self.patches)):
                inside = which == number
                if inside.any():
                    ex[inside], ey[inside] = self.fields[number](x[inside], y[inside])
        return ex, ey
