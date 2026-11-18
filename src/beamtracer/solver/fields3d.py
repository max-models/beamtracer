"""The staggered electric field of a potential on a 3D ion-optics mesh."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from methodcard import ConvergenceProblem, method

from beamtracer.solver.fields import _edge_field
from beamtracer.solver.mesh3d import IonOpticsMesh3D


def _trilinear(
    values: np.ndarray, s: np.ndarray, t: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Interpolate ``values[i, j, k]`` at fractional indices, clamping the cell."""
    n = values.shape
    i = np.clip(np.floor(s).astype(np.int64), 0, n[0] - 2)
    j = np.clip(np.floor(t).astype(np.int64), 0, n[1] - 2)
    k = np.clip(np.floor(u).astype(np.int64), 0, n[2] - 2)
    a, b, c = s - i, t - j, u - k
    out = np.zeros(np.shape(s))
    for di, wi in ((0, 1.0 - a), (1, a)):
        for dj, wj in ((0, 1.0 - b), (1, b)):
            for dk, wk in ((0, 1.0 - c), (1, c)):
                out += wi * wj * wk * values[i + di, j + dj, k + dk]
    return out


def _staggered_problem() -> ConvergenceProblem:
    """The field of ``sin(x) cos(y) cos(z)`` inside the mesh, against its gradient."""

    def error(h: float) -> float:
        n = round(2.0 / h)
        mesh = IonOpticsMesh3D((n + 1, n + 1, n + 1), (0.0, 0.0, 0.0), h)
        x, y, z = mesh.nodes()
        field = StaggeredElectricField3D(mesh, np.sin(x) * np.cos(y) * np.cos(z))
        px, py, pz = np.random.default_rng(0).uniform(0.5, 1.5, (3, 200))
        ex, ey, ez = field(px, py, pz)
        return float(
            max(
                np.max(np.abs(ex + np.cos(px) * np.cos(py) * np.cos(pz))),
                np.max(np.abs(ey - np.sin(px) * np.sin(py) * np.cos(pz))),
                np.max(np.abs(ez - np.sin(px) * np.cos(py) * np.sin(pz))),
            )
        )

    return ConvergenceProblem(error, steps=[0.2, 0.1, 0.05])


@method(
    equations={
        "edges": r"E_{x,\, i-1/2,\, j,\, k} = \frac{\phi_{i-1,j,k} - \phi_{i,j,k}}{h} \quad \text{(and alike along } y, z)",
        "surface": r"E_\text{edge} = \pm\frac{\phi_\text{near} - \phi_\text{electrode}}{d\, h} \quad \text{between a near-solid node and its electrode node}",
        "interpolation": r"E(x, y, z) = \text{trilinear interpolation on each staggered grid}",
    },
    order=2,
    assumptions=(
        r"the 2D rule of `StaggeredElectricField` along each axis: $d$ the bracketed surface distance in units of $h$",
        r"a symmetry plane mirrors, so the normal field vanishes there; the other faces extrapolate linearly",
        r"points outside the mesh are clamped to its edges; the order is in $h$, away from electrodes and faces",
    ),
    problem=_staggered_problem,
)
class StaggeredElectricField3D:
    """The electric field of a potential on the edges between the nodes of a 3D mesh.

    ``E_x`` lives on the edges along ``x`` ``(nx + 1, ny, nz)``, ``E_y`` on those along
    ``y`` ``(nx, ny + 1, nz)`` and ``E_z`` on those along ``z`` ``(nx, ny, nz + 1)``, each
    with the ghost edges beyond the faces; all are interpolated trilinearly.
    ``extrapolation`` gives, per face (`FACES_3D` order), ``"symmetric"`` or
    ``"extrapolate"``.
    """

    def __init__(
        self,
        mesh: IonOpticsMesh3D,
        potential: np.ndarray,
        extrapolation: Sequence[str] | None = None,
    ) -> None:
        """Precompute the edge fields of ``potential`` ``(nx, ny, nz)``."""
        if extrapolation is None:
            extrapolation = ["extrapolate"] * 6
            for plane in mesh.symmetry_planes:
                extrapolation[("y_min", "z_min").index(plane) * 2 + 2] = "symmetric"
        self.mesh = mesh
        self.extrapolation = tuple(extrapolation)
        phi = np.asarray(potential, dtype=float)
        solid = mesh.solid >= 0
        near = mesh.near_solid
        edges = []
        for axis in range(3):

            def front(a: np.ndarray, axis: int = axis) -> np.ndarray:
                return np.moveaxis(a, axis, 0)

            edge = _edge_field(
                front(phi),
                front(solid),
                front(near),
                front(mesh.distance[2 * axis]),
                front(mesh.distance[2 * axis + 1]),
                mesh.h,
                self.extrapolation[2 * axis : 2 * axis + 2],
            )
            edges.append(np.ascontiguousarray(np.moveaxis(edge, 0, axis)))
        self.fx, self.fy, self.fz = edges

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(E_x, E_y, E_z)`` [V/m] at the points ``(x, y, z)``."""
        mesh = self.mesh
        s = (np.asarray(x, dtype=float) - mesh.origin[0]) / mesh.h
        t = (np.asarray(y, dtype=float) - mesh.origin[1]) / mesh.h
        u = (np.asarray(z, dtype=float) - mesh.origin[2]) / mesh.h
        return (
            _trilinear(self.fx, s + 0.5, t, u),
            _trilinear(self.fy, s, t + 0.5, u),
            _trilinear(self.fz, s, t, u + 0.5),
        )
