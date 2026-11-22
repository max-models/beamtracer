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
from typing import Any

import numpy as np

from beamtracer.solver.fields import StaggeredElectricField
from beamtracer.solver.mesh import IonOpticsMesh


@dataclass(frozen=True)
class Refinement:
    """A box ``lower``-``upper`` [m] (``(x, y)`` corners, ``(x, y, z)`` in 3D) refined to ``level``."""

    level: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]


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


def _on_plane(mesh: IonOpticsMesh, y0: float) -> bool:
    """Whether a box of ``mesh`` starting at ``y0`` lies on its symmetry plane."""
    return mesh.symmetry_plane and abs(y0 - mesh.origin[1]) <= 1e-9 * mesh.h


def _part(
    mesh: IonOpticsMesh, first: int, count: int, first_y: int, level: int
) -> dict[str, Any]:
    """The settings of a box of ``mesh`` refined to ``level``: ``count`` nodes along x.

    From node ``first`` along x and ``first_y`` along y of the level. Along a graded x the
    base nodes, and those of the level evenly between them.
    """
    h = mesh.h / 2**level
    y0 = mesh.origin[1] + first_y * h
    if not mesh.graded:
        x0 = mesh.origin[0] + first * h
        return {"origin": (x0, y0), "symmetry_plane": _on_plane(mesh, y0)}
    x_nodes = mesh.position_x(np.arange(first, first + count) / 2**level)
    return {
        "origin": (float(x_nodes[0]), y0),
        "x_nodes": x_nodes,
        "logical_x0": mesh.logical_x0 + first * h,
        "symmetry_plane": _on_plane(mesh, y0),
    }


def patches_of(mesh: IonOpticsMesh, refinement: Sequence[Refinement]) -> list[Patch]:
    """The base patch and one patch per refined box, coarse to fine.

    A box is widened to the cells of the level below, so it starts and ends on that level's
    nodes, and clipped to the mesh.
    """
    nx, ny = mesh.shape
    patches = [Patch(0, ((0, nx - 1), (0, ny - 1)), mesh)]
    for box in sorted(refinement, key=lambda b: b.level):
        level = int(box.level)
        scale = 2 ** (level - 1)
        cells = []
        for axis in range(2):
            lower = max(mesh.origin[axis], box.lower[axis])
            upper = min(mesh.upper[axis], box.upper[axis])
            n_parent = (mesh.shape[axis] - 1) * scale
            # the box in the parent level's cells (along a graded x, by node index)
            if axis == 0:
                s0, s1 = mesh.index_x(lower) * scale, mesh.index_x(upper) * scale
            else:
                parent_h = mesh.h / scale
                s0 = (lower - mesh.origin[1]) / parent_h
                s1 = (upper - mesh.origin[1]) / parent_h
            i0 = int(np.floor(s0 + 1e-9))
            i1 = int(np.ceil(s1 - 1e-9))
            i0, i1 = max(i0, 0), min(i1, n_parent)
            if i1 <= i0:
                raise ValueError(f"the refined box {box} is outside the mesh")
            cells.append((2 * i0, 2 * i1))
        h = mesh.h / 2**level
        fine = 2**level
        shape = tuple(cells[a][1] - cells[a][0] + 1 for a in range(2))
        # the field's mesh: two cells (one of the level below) more on each side
        lower_margin = [min(2, cells[a][0]) for a in range(2)]
        upper_margin = [
            min(2, (mesh.shape[a] - 1) * fine - cells[a][1]) for a in range(2)
        ]

        patch_mesh = IonOpticsMesh(
            shape=shape,
            h=h,
            geometry=mesh.geometry,
            electrodes=mesh.electrodes,
            patch=True,
            **_part(mesh, cells[0][0], shape[0], cells[1][0], level),
        )
        settings = _part(
            mesh,
            cells[0][0] - lower_margin[0],
            shape[0] + lower_margin[0] + upper_margin[0],
            cells[1][0] - lower_margin[1],
            level,
        )
        field_mesh = IonOpticsMesh(
            shape=tuple(shape[a] + lower_margin[a] + upper_margin[a] for a in range(2)),
            h=h,
            geometry=mesh.geometry,
            electrodes=mesh.electrodes,
            patch=True,
            **settings,
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
    nodes' coordinates, ``hierarchy`` patchwork's `AMRHierarchy` of the patches (its axes
    ``(r, x)`` in cylindrical geometry).
    """

    patches: list[Patch]
    x: np.ndarray
    y: np.ndarray
    volumes: np.ndarray
    hierarchy: Any = None

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

    def plot_cells(self, ax: Any = None, *, scale: float = 1.0, **kwargs: Any) -> Any:
        """Draw the cells of every patch on ``ax`` with patchwork's `plot_patches`.

        The mesh's ``x`` is horizontal, ``y`` (or ``r``) vertical, the coordinates times
        ``scale`` (``1e3`` for mm). Every level's grid is drawn in full, the finer ones on
        top; ``kwargs`` go to `plot_patches` (e.g. ``colors``, ``labels``, ``grid_alpha``).
        Returns the axes (``None`` on the ranks other than patchwork's plotting rank).
        """
        from patchwork import plot_patches

        if self.hierarchy is None:
            raise ValueError("the layout has no patchwork hierarchy")
        options = {"mask_covered": False, "title": ""} | kwargs
        drawn = plot_patches(
            self.hierarchy,
            ax=ax,
            transpose=self.base.mesh.cylindrical,
            scale=scale,
            **options,
        )
        return None if drawn is None else drawn[1]

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

    def field(self, potential: np.ndarray) -> PatchedElectricField:
        """The electric field of a potential over the composite nodes."""
        return PatchedElectricField(self, potential)


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
    return Layout(patches, x.copy(), y.copy(), volumes, nodes.hierarchy)


def _extrapolation_of(mesh: IonOpticsMesh) -> list[str]:
    """The field's treatment of a mesh's faces: symmetric on the axis or a symmetry plane."""
    faces = ["extrapolate"] * 4
    on_axis = mesh.cylindrical and abs(mesh.origin[1]) <= 1e-12 * mesh.h
    if on_axis or mesh.symmetry_plane:
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
        s = mesh.index_x(x[inside])
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
        """The node spacing of the finest patch at each point (its smallest edge there)."""
        which = self.patch_of(x, y)
        out = np.empty(np.shape(x))
        for number, patch in enumerate(self.patches):
            mine = which == number
            if mine.any():
                out[mine] = patch.mesh.spacing_at(np.asarray(x)[mine])
        return out

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
