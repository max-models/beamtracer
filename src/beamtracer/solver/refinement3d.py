"""Refined boxes of a 3D ion-optics mesh: their patches, the composite nodes and the field.

The 3D counterpart of `beamtracer.solver.refinement`: the field is solved on patchwork's
composite mesh, the base mesh with boxes of finer cells (each level halves the cells of the
one below). Every patch is a uniform `IonOpticsMesh3D` of its own (its electrodes classified
at its spacing), whose nodes are composite nodes; a trajectory takes the field of, steps with
the spacing of and deposits its charge on the finest patch it is in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from beamtracer.solver.fields3d import StaggeredElectricField3D
from beamtracer.solver.mesh3d import IonOpticsMesh3D
from beamtracer.solver.refinement import Refinement

Box3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass
class Patch3D:
    """A uniform part of the composite 3D mesh: its level, cells and mesh.

    ``box`` is ``((i0, i1), (j0, j1), (k0, k1))`` in the level's cells; ``index`` is the
    composite node of every node of ``mesh`` (set by the layout). ``field_mesh`` is the mesh
    of the patch's field, two of its cells (one of the level below) wider on every side
    inside the domain, so the field at the box's faces sees the coarser potential beyond
    it; ``inner`` is the box's nodes in it.
    """

    level: int
    box: Box3D
    mesh: IonOpticsMesh3D
    index: np.ndarray | None = None
    field_mesh: IonOpticsMesh3D | None = None
    inner: tuple[slice, slice, slice] = (slice(None), slice(None), slice(None))

    @property
    def bounds(self) -> tuple[float, ...]:
        """The node extent ``(x0, y0, z0, x1, y1, z1)`` [m]."""
        return (*self.mesh.origin, *self.mesh.upper)


def _planes_of(mesh: IonOpticsMesh3D, origin: Sequence[float]) -> tuple[str, ...]:
    """The symmetry planes of ``mesh`` that a box starting at ``origin`` lies on."""
    planes = []
    for plane, axis in (("y_min", 1), ("z_min", 2)):
        if plane in mesh.symmetry_planes:
            if abs(origin[axis] - mesh.origin[axis]) <= 1e-9 * mesh.h:
                planes.append(plane)
    return tuple(planes)


def patches_of_3d(
    mesh: IonOpticsMesh3D, refinement: Sequence[Refinement]
) -> list[Patch3D]:
    """The base patch and one patch per refined box, coarse to fine.

    A box is widened to the cells of the level below, so it starts and ends on that level's
    nodes, and clipped to the mesh.
    """
    shape = mesh.shape
    patches = [Patch3D(0, tuple((0, n - 1) for n in shape), mesh)]  # type: ignore[arg-type]
    for box in sorted(refinement, key=lambda b: b.level):
        level = int(box.level)
        if len(box.lower) != 3 or len(box.upper) != 3:
            raise ValueError(f"a refined box of a 3D mesh needs 3D corners, got {box}")
        parent_h = mesh.h / 2 ** (level - 1)
        cells = []
        for axis in range(3):
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
        origin = tuple(mesh.origin[a] + cells[a][0] * h for a in range(3))
        size = tuple(cells[a][1] - cells[a][0] + 1 for a in range(3))
        patch_mesh = IonOpticsMesh3D(
            size, origin, h, mesh.electrodes, symmetry_planes=_planes_of(mesh, origin)
        )
        lower_margin = [
            min(2, round((origin[a] - mesh.origin[a]) / h)) for a in range(3)
        ]
        upper_margin = [
            min(2, round((mesh.upper[a] - (origin[a] + (size[a] - 1) * h)) / h))
            for a in range(3)
        ]
        field_origin = tuple(origin[a] - lower_margin[a] * h for a in range(3))
        field_mesh = IonOpticsMesh3D(
            tuple(size[a] + lower_margin[a] + upper_margin[a] for a in range(3)),
            field_origin,
            h,
            mesh.electrodes,
            symmetry_planes=_planes_of(mesh, field_origin),
        )
        inner = tuple(
            slice(lower_margin[a], lower_margin[a] + size[a]) for a in range(3)
        )
        patches.append(
            Patch3D(level, tuple(cells), patch_mesh, None, field_mesh, inner)  # type: ignore[arg-type]
        )
    return patches


@dataclass
class Layout3D:
    """The composite nodes of a 3D mesh and its refined boxes, as patchwork numbers them.

    ``volumes`` are the control volumes the deposit divides by: an eighth of every leaf
    cell at each corner (so a node on a face has half a cell, as the trilinear deposit
    implies). ``x`` is the nodes' ``x``.
    """

    patches: list[Patch3D]
    x: np.ndarray
    volumes: np.ndarray

    @property
    def mesh(self) -> IonOpticsMesh3D:
        """The base mesh."""
        return self.patches[0].mesh

    @property
    def size(self) -> int:
        """The number of composite nodes."""
        return int(self.x.size)

    @property
    def refined(self) -> bool:
        """Whether there is a patch besides the base."""
        return len(self.patches) > 1

    def on_base(self, values: np.ndarray) -> np.ndarray:
        """The ``(nx, ny, nz)`` array of composite values on the base mesh."""
        index = self.patches[0].index
        assert index is not None
        return np.asarray(values)[index]

    def from_base(self, values: np.ndarray) -> np.ndarray:
        """Composite values of an ``(nx, ny, nz)`` array, without refinement only."""
        if self.refined:
            raise ValueError("with refinement give values on the composite nodes")
        index = self.patches[0].index
        assert index is not None
        out = np.empty(self.size)
        out[index.ravel()] = np.asarray(values, dtype=float).ravel()
        return out

    def deposit_total(self, charges: Sequence[np.ndarray]) -> np.ndarray:
        """The composite node charges of per-patch node charges (shared nodes added)."""
        total = np.zeros(self.size)
        for patch, charge in zip(self.patches, charges, strict=True):
            assert patch.index is not None
            np.add.at(total, patch.index.ravel(), np.asarray(charge).ravel())
        return total

    def field(self, potential: np.ndarray) -> PatchedElectricField3D:
        """The electric field of a potential over the composite nodes."""
        return PatchedElectricField3D(self, potential)


def layout_of_3d(patches: list[Patch3D], nodes) -> Layout3D:
    """The `Layout3D` of ``patches`` on patchwork's `CompositeNodes` ``nodes``."""
    finest = max(p.level for p in patches)
    lattice = np.asarray(nodes.lattice, dtype=np.int64)
    span = lattice.max(axis=0) + 1
    keys = (lattice[:, 0] * span[1] + lattice[:, 1]) * span[2] + lattice[:, 2]
    order = np.argsort(keys)
    for patch in patches:
        scale = 2 ** (finest - patch.level)
        axes = [
            (patch.box[a][0] + np.arange(patch.mesh.shape[a])) * scale for a in range(3)
        ]
        i, j, k = np.meshgrid(*axes, indexing="ij")
        wanted = ((i * span[1] + j) * span[2] + k).ravel()
        found = order[np.searchsorted(keys, wanted, sorter=order)]
        if not np.array_equal(keys[found], wanted):  # pragma: no cover
            raise RuntimeError("a patch node is not a composite node")
        patch.index = found.reshape(patch.mesh.shape)
    coordinates = np.asarray(nodes.coordinates)
    cell = np.prod(nodes.cell_size, axis=1)
    corners = nodes.cell_corners
    volumes = np.bincount(
        corners.reshape(-1),
        weights=np.repeat(cell / corners.shape[1], corners.shape[1]),
        minlength=coordinates.shape[0],
    )
    return Layout3D(patches, coordinates[:, 0].copy(), volumes)


def _trilinear(
    patches: list[Patch3D], arrays: list[np.ndarray], points: np.ndarray
) -> np.ndarray:
    """The potential at points ``(3, N)`` from the patches done so far, finest wins."""
    out = np.zeros(points.shape[1])
    for patch, values in zip(patches, arrays, strict=False):
        mesh = patch.field_mesh if patch.field_mesh is not None else patch.mesh
        lo, hi = np.array(patch.bounds[:3]), np.array(patch.bounds[3:])
        inside = np.all((points >= lo[:, None]) & (points <= hi[:, None]), axis=0)
        if not inside.any():
            continue
        s = (points[:, inside] - np.array(mesh.origin)[:, None]) / mesh.h
        index = [
            np.clip(np.floor(s[a]).astype(np.int64), 0, mesh.shape[a] - 2)
            for a in range(3)
        ]
        f = [s[a] - index[a] for a in range(3)]
        total = np.zeros(s.shape[1])
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    weight = (
                        (f[0] if di else 1 - f[0])
                        * (f[1] if dj else 1 - f[1])
                        * (f[2] if dk else 1 - f[2])
                    )
                    total += (
                        weight * values[index[0] + di, index[1] + dj, index[2] + dk]
                    )
        out[inside] = total
    return out


class PatchedElectricField3D:
    """The electric field of a composite 3D potential: a `StaggeredElectricField3D` per patch.

    At a point the field is that of the finest patch containing it. A refined patch's field
    covers its `Patch3D.field_mesh`: its nodes outside the box take the potential of the
    coarser patches (trilinear), those in an electrode the electrode's.
    """

    def __init__(self, layout: Layout3D, potential: np.ndarray) -> None:
        """Precompute the edge fields of every patch."""
        self.layout = layout
        self.patches = layout.patches
        potential = np.asarray(potential)
        arrays: list[np.ndarray] = []
        self.fields: list[StaggeredElectricField3D] = []
        self.meshes: list[IonOpticsMesh3D] = []
        for patch in layout.patches:
            values = potential[patch.index]
            mesh = patch.field_mesh
            if mesh is None:
                mesh = patch.mesh
            else:
                points = np.stack([c.ravel() for c in mesh.nodes()])
                full = _trilinear(layout.patches, arrays, points).reshape(mesh.shape)
                solid = mesh.solid >= 0
                full[solid] = mesh.fixed_value[solid]
                full[patch.inner] = values
                values = full
            arrays.append(values)
            self.meshes.append(mesh)
            self.fields.append(StaggeredElectricField3D(mesh, values))
        self.mesh = layout.mesh
        self.fx, self.fy, self.fz = (
            self.fields[0].fx,
            self.fields[0].fy,
            self.fields[0].fz,
        )

    def patch_of(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The finest patch containing each point."""
        which = np.zeros(np.shape(x), dtype=np.int64)
        for number, patch in enumerate(self.patches[1:], start=1):
            x0, y0, z0, x1, y1, z1 = patch.bounds
            inside = (
                (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
            )
            which[inside] = number
        return which

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(E_x, E_y, E_z)`` [V/m] at the points ``(x, y, z)``."""
        x, y, z = (np.asarray(a, dtype=float).ravel() for a in (x, y, z))
        ex, ey, ez = (np.asarray(c) for c in self.fields[0](x, y, z))
        if len(self.patches) > 1:
            which = self.patch_of(x, y, z)
            for number in range(1, len(self.patches)):
                inside = which == number
                if inside.any():
                    fx, fy, fz = self.fields[number](x[inside], y[inside], z[inside])
                    ex[inside], ey[inside], ez[inside] = fx, fy, fz
        return ex, ey, ez
