"""The beam's own magnetic field: Ampère's law of magnetostatics, ``curl B = mu_0 J``.

A steady beam has no displacement current, so of Maxwell's equations only Gauss's law (the
field solve of the potential) and Ampère's law of magnetostatics remain, uncoupled. The
current density ``J`` is deposited by the tracer along with the charge (the current
element of every step); its field pinches the beam and, for a beam along ``x``, scales its
space-charge force by about ``1 - f - beta^2`` with ``f`` the compensated share.

- Cylindrical ``(x, r)``: Ampère's law on the disc of radius ``r`` at ``x``,
  ``B_theta = mu_0 I(x, r) / (2 pi r)``, exact for a steady axisymmetric beam
  (`axisymmetric_field`).
- Planar ``(x, y)``: ``dB_z / dy = mu_0 J_x``, integrated across the beam from the
  symmetry plane (``B_z = 0`` there) or from the middle of the total current
  (`planar_field`), exact for a steady beam whose current is conserved.
- 3D: the vector potential, ``-laplace A = mu_0 J`` component by component (Coulomb gauge,
  ``div J = 0`` for a steady beam), ``B = curl A`` (`VectorPotential3D`).

The azimuthal current of a beam rotating in a solenoid and the transverse currents in 2D are
left out: their fields are of order ``beta^2`` times the solenoid's own change of the
rotation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from methodcard import method
from scipy.constants import mu_0

from beamtracer.solver.mesh import IonOpticsMesh
from beamtracer.solver.mesh3d import FACES_3D, IonOpticsMesh3D


@method(
    equations={
        "ampere": r"\oint \mathbf B \cdot d\mathbf l = \mu_0 \int \mathbf J \cdot d\mathbf A \quad\Rightarrow\quad B_\theta(x, r) = \frac{\mu_0 I(x, r)}{2 \pi r}, \qquad I(x, r) = \int_0^r J_x\, 2 \pi r'\, dr'",
        "trapezoid": r"I(x, r_j) = \pi \sum_{k < j} \tfrac12 (J_k + J_{k+1}) (r_{k+1}^2 - r_k^2), \qquad B_\theta(x, 0) = 0",
    },
    assumptions=(
        r"steady and axisymmetric: no displacement current, $B_\theta$ depends on the current through the disc only",
        r"$J_x$ linear in $r^2$ between nodes, the shape of the deposit, so the trapezoid in $r^2$ is exact for it",
        r"the axial and radial self fields of an azimuthal current (a beam rotating in a solenoid) left out",
    ),
)
def axisymmetric_field(mesh: IonOpticsMesh, jx: np.ndarray) -> np.ndarray:
    """``B_theta`` [T] ``(nx, ny)`` on the nodes of a cylindrical mesh from ``J_x`` [A/m^2]."""
    r = mesh.y
    s = r * r
    jx = np.asarray(jx, dtype=float)
    ring = 0.5 * np.pi * (jx[:, 1:] + jx[:, :-1]) * np.diff(s)[None, :]
    enclosed = np.concatenate(
        [np.zeros((jx.shape[0], 1)), np.cumsum(ring, axis=1)], axis=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(
            r[None, :] > 0.0, mu_0 * enclosed / (2.0 * np.pi * r[None, :]), 0.0
        )
    return b


@method(
    equations={
        "ampere": r"\nabla \times (B_z \hat{\mathbf z}) = \mu_0 \mathbf J \quad\Rightarrow\quad \frac{\partial B_z}{\partial y} = \mu_0 J_x",
        "symmetry_plane": r"B_z(x, y) = \mu_0 \int_{y_0}^{y} J_x\, dy'",
        "free": r"B_z(x, y) = \mu_0 \left(\int_{y_\mathrm{min}}^{y} J_x\, dy' - \tfrac12 \int_{y_\mathrm{min}}^{y_\mathrm{max}} J_x\, dy'\right)",
    },
    assumptions=(
        r"steady, planar (no dependence on $z$), currents in the plane: $\mathbf B = B_z \hat{\mathbf z}$; the $y$ integral is path independent when $\nabla \cdot \mathbf J = 0$",
        r"without a symmetry plane the field far above and below the beam is $\pm \mu_0 K / 2$, $K$ the current per unit depth (free space)",
        r"trapezoid rule over the nodes",
    ),
)
def planar_field(mesh: IonOpticsMesh, jx: np.ndarray) -> np.ndarray:
    """``B_z`` [T] ``(nx, ny)`` on the nodes of a planar mesh from ``J_x`` [A/m^2]."""
    jx = np.asarray(jx, dtype=float)
    strip = 0.5 * (jx[:, 1:] + jx[:, :-1]) * mesh.h
    integral = np.concatenate(
        [np.zeros((jx.shape[0], 1)), np.cumsum(strip, axis=1)], axis=1
    )
    if not mesh.symmetry_plane:
        integral -= 0.5 * integral[:, -1:]
    return mu_0 * integral


class NodeMagneticField:
    """A magnetic field component on the nodes of a 2D mesh, interpolated bilinearly.

    Called with ``(x, y)`` (``y`` the radius in cylindrical geometry) it returns
    ``(B_x, B_r, B_theta)`` (cylindrical) or ``(B_x, B_y, B_z)`` (planar), all but the
    last zero: the tracer's convention for ``magnetic``.
    """

    def __init__(self, mesh: IonOpticsMesh, values: np.ndarray) -> None:
        """Hold ``values`` ``(nx, ny)`` of the out-of-plane component on ``mesh``'s nodes."""
        self.mesh = mesh
        self.values = np.asarray(values, dtype=float)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        """The field at the points ``(x, y)``, constant beyond the mesh."""
        mesh = self.mesh
        nx, ny = mesh.shape
        sx = np.clip(mesh.index_x(x), 0.0, nx - 1.0)
        sy = np.clip((np.asarray(y) - mesh.origin[1]) / mesh.h, 0.0, ny - 1.0)
        i = np.minimum(sx.astype(np.int64), nx - 2)
        j = np.minimum(sy.astype(np.int64), ny - 2)
        a, b = sx - i, sy - j
        v = self.values
        out = (1.0 - a) * ((1.0 - b) * v[i, j] + b * v[i, j + 1]) + a * (
            (1.0 - b) * v[i + 1, j] + b * v[i + 1, j + 1]
        )
        zero = np.zeros_like(out)
        return zero, zero, out


def field_2d(mesh: IonOpticsMesh, jx: np.ndarray) -> NodeMagneticField:
    """The beam's own field on a 2D mesh from its axial current density ``J_x``."""
    if mesh.cylindrical:
        return NodeMagneticField(mesh, axisymmetric_field(mesh, jx))
    return NodeMagneticField(mesh, planar_field(mesh, jx))


@method(
    equations={
        "potential": r"-\nabla^2 A_c = \mu_0 J_c, \quad c = x, y, z, \qquad \mathbf B = \nabla \times \mathbf A",
        "faces": r"\partial_n A_c = 0 \text{ on } x = x_\mathrm{min}, x_\mathrm{max}; \quad A_c = 0 \text{ on the transverse faces}; \quad \text{at a symmetry plane } \partial_n A_c = 0, \text{ its normal component } 0",
    },
    assumptions=(
        r"Coulomb gauge, which the steady current ($\nabla \cdot \mathbf J = 0$) keeps; the electrodes are not magnetic and do not enter",
        r"the beam continues through the $x$ faces; $\mathbf A = 0$ on the transverse faces is exact for a round beam centred in a round pipe and close to it in a box much wider than the beam",
        r"patchwork's `NodalPoisson` on the base mesh, factorised once per set of face conditions; the curl by second-order central differences (one-sided on the faces)",
    ),
)
class VectorPotential3D:
    """The beam's own magnetic field on an `IonOpticsMesh3D`, from its vector potential."""

    def __init__(self, mesh: IonOpticsMesh3D, backend: str = "lu") -> None:
        """Build the solvers of the three components on ``mesh`` (patchwork ``backend``)."""
        self.mesh = mesh
        self.backend = backend
        self._solvers: dict[tuple[str, ...], Any] = {}

    def _kinds(self, component: int) -> tuple[str, ...]:
        """The face conditions of component ``component``, in the order of `FACES_3D`."""
        kinds = []
        for face in FACES_3D:
            axis = FACES_3D.index(face) // 2
            if axis == 0:
                kinds.append("neumann")
            elif face in self.mesh.symmetry_planes:
                kinds.append("dirichlet" if component == axis else "neumann")
            else:
                kinds.append("dirichlet")
        return tuple(kinds)

    def _solver(self, kinds: tuple[str, ...]) -> tuple[Any, Any]:
        """The (cached) patchwork solver and its layout for a set of face conditions."""
        if kinds not in self._solvers:
            from patchwork import AMRHierarchy, Dirichlet, Neumann, NodalPoisson

            from beamtracer.solver.poisson import _patchwork_backend
            from beamtracer.solver.refinement3d import layout_of_3d, patches_of_3d

            mesh = self.mesh
            origin, upper = mesh.origin, mesh.upper
            hierarchy = AMRHierarchy(
                base_shape=tuple(n - 1 for n in mesh.shape),
                bounds=tuple((origin[a], upper[a]) for a in range(3)),
            )
            faces = {
                face.replace("_min", "_lower").replace("_max", "_upper"): (
                    Dirichlet(0.0) if kind == "dirichlet" else Neumann(0.0)
                )
                for face, kind in zip(FACES_3D, kinds, strict=True)
            }
            solver = NodalPoisson(
                hierarchy, faces=faces, backend=_patchwork_backend(self.backend)
            )
            layout = layout_of_3d(patches_of_3d(mesh, ()), solver.nodes)
            self._solvers[kinds] = (solver, layout)
        return self._solvers[kinds]

    def potential(self, current_density: np.ndarray) -> np.ndarray:
        """``A`` [T m] ``(3, nx, ny, nz)`` from ``J`` [A/m^2] ``(3, nx, ny, nz)``."""
        out = np.zeros(np.shape(current_density))
        for c in range(3):
            solver, layout = self._solver(self._kinds(c))
            source = layout.from_base(
                mu_0 * np.asarray(current_density[c], dtype=float)
            )
            out[c] = layout.on_base(solver.solve(source))
        return out

    def __call__(self, current_density: np.ndarray) -> np.ndarray:
        """``B`` [T] ``(3, nx, ny, nz)`` on the nodes from ``J`` [A/m^2] ``(3, nx, ny, nz)``."""
        a = self.potential(current_density)
        h = self.mesh.h

        def d(component: int, axis: int) -> np.ndarray:
            return np.gradient(a[component], h, axis=axis, edge_order=2)

        return np.stack(
            [d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)], axis=0
        )
