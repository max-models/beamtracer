"""Adaptive refinement: boxes of finer cells where the electric field changes fastest.

Between Vlasov iterations `adapted_boxes` tags the cells of each level across which the
field strength changes by more than ``threshold`` times its largest value, grows the tags
by ``buffer`` cells and covers them with boxes (patchwork's Berger--Rigoutsos clustering,
`patchwork.mesh.regrid.cluster`), each inside the boxes of the level below. The field is
that of the last iteration, on the finest patch at every point, so a level's tags come
from the field that the level below resolves. `sample` moves a potential from one
composite mesh to the next.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from beamtracer.solver.refinement import Refinement


@dataclass(frozen=True)
class AdaptiveRefinement:
    """When and where `vlasov_iterate` refines on its own.

    The mesh is regridded every ``every`` iterations (from iteration ``every`` on), up to
    ``max_level`` levels of boxes. A cell is tagged where the field strength changes across
    it by more than ``threshold`` times the largest field strength on the mesh; the tags
    grow by ``buffer`` cells of their level, and a box is at least ``efficiency`` full of
    tagged cells.
    """

    max_level: int = 1
    threshold: float = 0.05
    every: int = 4
    #: The mesh is adapted at most this many times (at iterations ``every``, ``2 every``, ...).
    regrids: int = 2
    #: Boxes of fewer cells of their level along an axis are not made.
    min_size: int = 4
    buffer: int = 2
    efficiency: float = 0.5

    def __post_init__(self) -> None:
        """Check the settings."""
        if self.max_level < 1:
            raise ValueError("max_level must be at least 1")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.every < 1:
            raise ValueError("every must be at least 1")
        if self.regrids < 1:
            raise ValueError("regrids must be at least 1")
        if self.buffer < 0:
            raise ValueError("buffer must not be negative")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency must be in (0, 1], got {self.efficiency}")


def _grid(mesh: Any, level: int) -> tuple[list[np.ndarray], float]:
    """The node coordinates of the whole mesh at the cells of ``level``, and their size."""
    h = mesh.h / 2**level
    axes = [
        mesh.origin[a] + h * np.arange((mesh.shape[a] - 1) * 2**level + 1)
        for a in range(len(mesh.shape))
    ]
    return axes, h


def field_variation(
    field: Callable[..., tuple[np.ndarray, ...]], mesh: Any, level: int
) -> np.ndarray:
    """The change of the field strength across every cell of ``level`` (span over its corners)."""
    axes, _ = _grid(mesh, level)
    points = np.meshgrid(*axes, indexing="ij")
    components = field(*(p.ravel() for p in points))
    strength = np.sqrt(sum(np.asarray(c) ** 2 for c in components)).reshape(
        points[0].shape
    )
    ndim = strength.ndim
    corners = []
    for offset in np.ndindex(*(2,) * ndim):
        index = tuple(
            slice(o, o + n - 1) for o, n in zip(offset, strength.shape, strict=True)
        )
        corners.append(strength[index])
    stack = np.stack(corners)
    return stack.max(axis=0) - stack.min(axis=0)


def _grow(tags: np.ndarray, width: int) -> np.ndarray:
    """``tags`` widened by ``width`` cells along every axis (also diagonally)."""
    grown = tags.copy()
    for _ in range(width):
        current = grown.copy()
        for axis in range(tags.ndim):
            current[
                tuple(
                    slice(1, None) if a == axis else slice(None)
                    for a in range(tags.ndim)
                )
            ] |= grown[
                tuple(
                    slice(None, -1) if a == axis else slice(None)
                    for a in range(tags.ndim)
                )
            ]
            current[
                tuple(
                    slice(None, -1) if a == axis else slice(None)
                    for a in range(tags.ndim)
                )
            ] |= grown[
                tuple(
                    slice(1, None) if a == axis else slice(None)
                    for a in range(tags.ndim)
                )
            ]
        grown = current
    return grown


def _outer(masks: list[np.ndarray]) -> np.ndarray:
    """The tensor product of one boolean mask per axis."""
    result = masks[0]
    for mask in masks[1:]:
        result = result[..., None] & mask
    return result


def _inside(boxes: Sequence[Refinement], mesh: Any, level: int) -> np.ndarray:
    """The cells of ``level`` inside the boxes of ``level`` (all of them for level 0)."""
    axes, h = _grid(mesh, level)
    shape = tuple(a.size - 1 for a in axes)
    if level == 0:
        return np.ones(shape, dtype=bool)
    inside = np.zeros(shape, dtype=bool)
    centres = [a[:-1] + h / 2 for a in axes]
    for box in boxes:
        if box.level != level:
            continue
        masks = [
            (c > lo) & (c < hi)
            for c, lo, hi in zip(centres, box.lower, box.upper, strict=True)
        ]
        inside |= _outer(masks)
    return inside


def _density_variation(density: np.ndarray, level: int) -> np.ndarray:
    """The change of ``|density|`` (base nodes) across every base cell, on the cells of ``level``."""
    density = np.abs(np.asarray(density, dtype=float))
    shape = tuple(n - 1 for n in density.shape)
    corners = [
        density[tuple(slice(o, o + n) for o, n in zip(offset, shape, strict=True))]
        for offset in np.ndindex(*(2,) * density.ndim)
    ]
    stack = np.stack(corners)
    variation = stack.max(axis=0) - stack.min(axis=0)
    for axis in range(density.ndim):
        variation = np.repeat(variation, 2**level, axis=axis)
    return variation


def _beam_cells(density: np.ndarray, level: int, width: int) -> np.ndarray:
    """The cells of ``level`` within ``width`` of its cells of the beam's charge.

    ``density`` is on the base mesh's nodes; a base cell holds beam where a corner's
    density is above a thousandth of the largest.
    """
    density = np.abs(np.asarray(density, dtype=float))
    largest = float(density.max()) if density.size else 0.0
    ndim = density.ndim
    shape = tuple(n - 1 for n in density.shape)
    beam = np.zeros(shape, dtype=bool)
    if largest > 0.0:
        for offset in np.ndindex(*(2,) * ndim):
            index = tuple(slice(o, o + n) for o, n in zip(offset, shape, strict=True))
            beam |= density[index] > 1e-3 * largest
    for axis in range(ndim):
        beam = np.repeat(beam, 2**level, axis=axis)
    return _grow(beam, width)


def _largest_strength(field: Callable[..., tuple[np.ndarray, ...]], mesh: Any) -> float:
    """The largest field strength at the base mesh's nodes."""
    axes, _ = _grid(mesh, 0)
    points = np.meshgrid(*axes, indexing="ij")
    components = field(*(p.ravel() for p in points))
    return float(np.sqrt(sum(np.asarray(c) ** 2 for c in components)).max())


def adapted_boxes(
    field: Callable[..., tuple[np.ndarray, ...]],
    mesh: Any,
    settings: AdaptiveRefinement,
    density: np.ndarray | None = None,
) -> list[Refinement]:
    """The refined boxes for ``field`` on ``mesh`` (see the module).

    A level is tagged only where the level below is refined, in the new boxes. With the
    beam's charge ``density`` on the base mesh's nodes, cells across which it changes by
    more than ``threshold`` times its largest value are tagged too (the meniscus, where the
    beam forms), and only cells near the beam are: an electrode's face, where the field
    drops to zero, is refined only where the beam passes it.
    """
    from patchwork.mesh.regrid import cluster

    boxes: list[Refinement] = []
    for level in range(1, settings.max_level + 1):
        parent = level - 1
        variation = field_variation(field, mesh, parent)
        # relative to the largest field strength, not to the largest change
        strength = _largest_strength(field, mesh)
        if strength <= 0.0:
            break
        tags = variation > settings.threshold * strength
        if density is not None:
            # the meniscus: where the beam forms, its charge density changes fastest
            largest = float(np.abs(density).max())
            if largest > 0.0:
                tags |= (
                    _density_variation(density, parent) > settings.threshold * largest
                )
        tags = _grow(tags, settings.buffer)
        if density is not None:
            tags &= _beam_cells(density, parent, settings.buffer)
        allowed = _inside(boxes, mesh, parent)
        tags &= allowed
        if not tags.any():
            break
        axes, h = _grid(mesh, parent)
        for found in cluster(
            tags,
            efficiency=settings.efficiency,
            allowed=allowed,
            min_size=settings.min_size,
        ):
            lower = tuple(a[lo] for a, (lo, _) in zip(axes, found, strict=True))
            upper = tuple(a[hi] for a, (_, hi) in zip(axes, found, strict=True))
            boxes.append(Refinement(level, lower, upper))
    return boxes


def sample(layout: Any, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``values`` on the composite nodes of ``layout`` at ``points`` ``(ndim, n)``.

    Multilinear on the finest patch that contains each point.
    """
    points = np.asarray(points, dtype=float)
    out = np.zeros(points.shape[1])
    values = np.asarray(values)
    for patch in layout.patches:
        mesh = patch.mesh
        origin = np.asarray(mesh.origin, dtype=float)
        upper = origin + mesh.h * (np.asarray(mesh.shape) - 1)
        inside = np.all(
            (points >= origin[:, None] - 1e-12) & (points <= upper[:, None] + 1e-12),
            axis=0,
        )
        if not inside.any():
            continue
        on = values[patch.index]
        s = (points[:, inside] - origin[:, None]) / mesh.h
        index = [
            np.clip(np.floor(s[a]).astype(np.int64), 0, mesh.shape[a] - 2)
            for a in range(s.shape[0])
        ]
        fraction = [s[a] - index[a] for a in range(s.shape[0])]
        total = np.zeros(s.shape[1])
        for corner in np.ndindex(*(2,) * s.shape[0]):
            weight = np.ones(s.shape[1])
            for a, c in enumerate(corner):
                weight *= fraction[a] if c else 1.0 - fraction[a]
            total += weight * on[tuple(index[a] + c for a, c in enumerate(corner))]
        out[inside] = total
    return out
