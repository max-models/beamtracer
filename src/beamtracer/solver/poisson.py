r"""Poisson's equation on an `IonOpticsMesh`, with a Boltzmann plasma species.

The finite differences are those of patchwork's `NodalPoisson`, the solver beamtracer shares
with picnic: the 5-point Laplacian as finite volumes, in cylindrical geometry over the rings
(``(1 -/+ 1/2j)`` weights on the radial neighbours and ``4 (phi_1 - phi_0)`` on the axis),
Shortley--Weller rows next to electrodes with the distances the mesh brackets, and the flux of
Neumann faces.

For positive-ion extraction the plasma electrons follow a Boltzmann distribution,

$$
\rho_e = -\rho_{e0} \exp\left(\frac{\phi - U_p}{T_e}\right),
$$

and for negative-ion extraction the plasma's positive ions do, the beams of negative ions and
electrons being traced,

$$
\rho_p = \rho_{p0} \exp\left(-\frac{\phi - U_p}{T_p}\right),
$$

on every free node; the nonlinear equation is solved with Newton's method. Before the first
beam is known, the potential of an *initial plasma* region is fixed to ``U_p`` instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scope_profiler
from methodcard import ConvergenceProblem, method
from scipy.constants import epsilon_0

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.solver.mesh import IonOpticsMesh
from beamtracer.solver.mesh3d import FACES_3D, IonOpticsMesh3D
from beamtracer.solver.refinement import Layout, Refinement, layout_of, patches_of
from beamtracer.solver.refinement3d import Layout3D, layout_of_3d, patches_of_3d

#: A Newton--Krylov step whose GMRES stops short of `NEWTON_KRYLOV_RTOL` is still taken
#: when its true relative residual is below this (an inexact Newton step).
NEWTON_KRYLOV_ACCEPT = 0.1

#: The backends whose Newton steps are GMRES with a kept multigrid preconditioner.
KRYLOV_BACKENDS = ("amg", "multigrid")

#: The multigrid's coarsest level (``backend="multigrid"``): small enough to invert densely.
COARSEST_UNKNOWNS = 30

#: The backends that precondition every Newton step with its own Jacobian.
REFRESHED_BACKENDS = ("multigrid",)

#: The relative residual of a Newton step by GMRES with ``backend="amg"`` or ``"multigrid"`` (an inexact
#: Newton step: near the solution each step gains this factor).
NEWTON_KRYLOV_RTOL = 1e-3
#: GMRES iterations after which the kept AMG preconditioner is built again (a build of
#: the classical AMG hierarchy costs about as much as a dozen iterations).
NEWTON_KRYLOV_REBUILD = 12


@dataclass(frozen=True)
@method(
    equations={
        "density": r"\rho_e = -\rho_{e0} \exp\left(\frac{\phi - U_p}{T_e}\right)",
        "neutral": r"\rho_{e0} = \sum_\mathrm{beams} \frac{J}{v_0}, \qquad v_0 = \sqrt{\frac{2 E e}{m}}",
    },
    assumptions=(
        r"isothermal electrons in equilibrium with the potential: no inertia, flow or sheath beyond the Boltzmann factor",
        r"$\rho_{e0}$ given, or by default the charge density of the beams at their start, so the plasma is neutral there (`ParticleBeam`s add nothing)",
        r"$T_e$ and $U_p$ in eV and V; the species has the sign $s = -1$",
    ),
)
class BoltzmannElectrons:
    """Boltzmann plasma electrons for positive-ion extraction.

    ``rho_e`` [C/m^3] is the electron charge density magnitude in the plasma, usually the
    ion charge density of the beam at its start (``sum J / v``), ``electron_temperature`` [eV]
    and ``plasma_potential`` [V] are ``T_e`` and ``U_p``.
    """

    rho_e: float
    electron_temperature: float
    plasma_potential: float

    #: The sign of the species' charge.
    sign = -1

    @property
    def density(self) -> float:
        """The charge density magnitude [C/m^3] at the plasma potential."""
        return self.rho_e

    @property
    def temperature(self) -> float:
        """The temperature [eV]."""
        return self.electron_temperature


@dataclass(frozen=True)
@method(
    equations={
        "density": r"\rho_p = \rho_{p0} \exp\left(-\frac{\phi - U_p}{T_p}\right)",
        "neutral": r"\rho_{p0} = \sum_\mathrm{beams} \frac{J}{v_0}, \qquad v_0 = \sqrt{\frac{2 E e}{m}}",
        "both_signs": r"\rho = s\, \rho_0\, e^{-s (\phi - U_p) / T}, \qquad s = +1 \text{ (ions)}, \ -1 \text{ (electrons)}",
    },
    assumptions=(
        r"the plasma's positive ions, repelled by the extraction field; the negative ions and the co-extracted electrons are traced beams",
        r"isothermal ions in equilibrium with the potential, $T_p > 0$",
        r"$\rho_{p0}$ given, or by default the negative charge density of the beams at their start",
    ),
)
class BoltzmannIons:
    """Thermal positive plasma ions for negative-ion extraction.

    The counterpart of `BoltzmannElectrons`: the negative ions and the electrons are
    traced as beams, the positive ions repelled by the extraction field follow
    ``rho_p exp(-(phi - U_p) / T_p)``. ``rho_p`` [C/m^3] is their charge density in the
    plasma, usually the negative charge density of the beams at their start (``sum J / v``),
    ``ion_temperature`` [eV] and ``plasma_potential`` [V] are ``T_p`` and ``U_p``.
    """

    rho_p: float
    ion_temperature: float
    plasma_potential: float

    #: The sign of the species' charge.
    sign = 1

    @property
    def density(self) -> float:
        """The charge density magnitude [C/m^3] at the plasma potential."""
        return self.rho_p

    @property
    def temperature(self) -> float:
        """The temperature [eV]."""
        return self.ion_temperature


#: The faces of `IonOpticsMesh` as patchwork's, in planar and cylindrical geometry
#: (the radius is patchwork's first axis; the axis ``y_min`` takes no condition).
_FACES = {
    False: {
        "x_min": "x_lower",
        "x_max": "x_upper",
        "y_min": "y_lower",
        "y_max": "y_upper",
    },
    True: {"x_min": "y_lower", "x_max": "y_upper", "y_max": "x_upper"},
}


def _patchwork_backend(backend: str) -> Any:
    """Patchwork's backend for a name: ``"multigrid"`` smooths by Gauss--Seidel (compiled)."""
    if backend == "multigrid":
        from patchwork import MultigridBackend

        return MultigridBackend(
            smoother="gauss-seidel",
            smoothing_steps=1,
            coarsest_unknowns=COARSEST_UNKNOWNS,
        )
    return backend


class _Field:
    """The patchwork `NodalPoisson` of an `IonOpticsMesh` and its refined boxes.

    In cylindrical geometry patchwork's hierarchy is the ``(r, x)`` plane: its points are
    ``(r, x)`` where the mesh's are ``(x, r)``. ``region`` (the initial plasma) is a mask
    of the mesh's nodes (without refinement) or a function of ``(x, y)``.
    """

    def __init__(
        self,
        mesh: IonOpticsMesh,
        region: Any,
        region_potential: float,
        backend: str,
        comm: Any = None,
        refinement: Sequence[Refinement] = (),
    ) -> None:
        from patchwork import (
            AMRHierarchy,
            Dirichlet,
            EmbeddedDirichlet,
            Neumann,
            NodalPoisson,
        )

        nx, ny = mesh.shape
        (x0, y0), (x1, y1) = mesh.origin, mesh.upper
        cylindrical = mesh.cylindrical
        self.mesh = mesh
        # a graded beam axis: its nodes are the base nodes of patchwork's axis
        graded = {}
        if mesh.graded:
            nodes = (None, mesh.x_nodes) if cylindrical else (mesh.x_nodes, None)
            graded["axis_nodes"] = nodes
        if cylindrical:
            hierarchy = AMRHierarchy(
                base_shape=(ny - 1, nx - 1),
                bounds=((y0, y1), (x0, x1)),
                geometry="cylindrical",
                comm=comm,
                **graded,
            )
        else:
            hierarchy = AMRHierarchy(
                base_shape=(nx - 1, ny - 1),
                bounds=((x0, x1), (y0, y1)),
                comm=comm,
                **graded,
            )
        patches = patches_of(mesh, refinement)
        for patch in patches[1:]:
            box = (patch.box[1], patch.box[0]) if cylindrical else patch.box
            hierarchy.add_patch(level=patch.level, box=box)

        def mesh_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """The mesh's ``(x, y)`` of patchwork's points."""
            return (points[1], points[0]) if cylindrical else (points[0], points[1])

        def fraction(start: np.ndarray, end: np.ndarray) -> np.ndarray:
            """The bracketed distance from a node to the surface towards ``end``.

            From the mesh of the patch, of the edge's level, that holds the node: the
            distances the field takes too.
            """
            xs, ys = mesh_points(start)
            xe, ye = mesh_points(end)
            # the edge in base-node units: its level, and along x or y
            di = mesh.index_x(xe) - mesh.index_x(xs)
            dj = (ye - ys) / mesh.h
            length = np.maximum(np.abs(di), np.abs(dj))
            level = np.rint(np.log2(1.0 / length)).astype(np.int64)
            direction = np.where(
                np.abs(di) >= np.abs(dj),
                np.where(di < 0, 0, 1),
                np.where(dj < 0, 2, 3),
            )
            out = np.ones(xs.size)
            done = np.zeros(xs.size, dtype=bool)
            for patch in reversed(patches):
                px0, py0, px1, py1 = patch.bounds
                h = patch.mesh.h
                tiny = 1e-9 * h
                mine = (
                    ~done
                    & (level == patch.level)
                    & (xs >= px0 - tiny)
                    & (xs <= px1 + tiny)
                    & (ys >= py0 - tiny)
                    & (ys <= py1 + tiny)
                )
                if mine.any():
                    i = np.rint(patch.mesh.index_x(xs[mine])).astype(np.int64)
                    j = np.rint((ys[mine] - py0) / h).astype(np.int64)
                    out[mine] = patch.mesh.distance[direction[mine], i, j]
                    done |= mine
            return out

        faces = {}
        for face, name in _FACES[cylindrical].items():
            condition = mesh.faces[face]
            value = condition.value
            given = (
                (lambda *p, c=condition: c.evaluate(*mesh_points(np.array(p))))
                if callable(value)
                else float(value)
            )
            faces[name] = (
                Dirichlet(given) if condition.kind == "dirichlet" else Neumann(given)
            )
        bodies = []
        if region is not None:
            # the initial plasma: nodes held at its potential, where no electrode or
            # Dirichlet face fixes them, with the stencil of fixed nodes (fraction 1)
            dirichlet = [f for f, c in mesh.faces.items() if c.kind == "dirichlet"]

            def in_region(points: np.ndarray) -> np.ndarray:
                x, y = mesh_points(points)
                if not callable(region):
                    held = np.asarray(region, dtype=bool) & ~mesh.fixed
                    i = np.rint(mesh.index_x(x)).astype(np.int64)
                    j = np.rint((y - y0) / mesh.h).astype(np.int64)
                    return held[i, j]
                inside = np.asarray(region(x, y), dtype=bool).copy()
                tiny = 1e-9 * mesh.h
                on_face = {
                    "x_min": x <= x0 + tiny,
                    "x_max": x >= x1 - tiny,
                    "y_min": y <= y0 + tiny,
                    "y_max": y >= y1 - tiny,
                }
                for face in dirichlet:
                    inside &= ~on_face[face]
                return inside

            bodies.append(
                EmbeddedDirichlet(
                    in_region,
                    float(region_potential),
                    fraction=lambda start, end: np.ones(start.shape[1]),
                )
            )
        # patchwork lets the last body listed own a node, the mesh the first electrode
        for electrode in reversed(mesh.electrodes):
            bodies.append(
                EmbeddedDirichlet(
                    lambda points, e=electrode: np.asarray(
                        e.inside(*mesh_points(points)), dtype=bool
                    ),
                    float(electrode.potential),
                    fraction=fraction,
                )
            )
        self.solver = NodalPoisson(
            hierarchy,
            faces=faces,
            embedded=bodies,
            backend=_patchwork_backend(backend),
            metric="cylindrical" if cylindrical else "cartesian",
            min_fraction=1.0 / 255.0,
        )
        #: The patches and the composite nodes.
        self.layout = layout_of(mesh, patches, self.solver.nodes)
        nodes = self.solver.nodes
        #: The hanging nodes (on coarse–fine edges) and their interpolation.
        self.hanging = np.flatnonzero(nodes.hanging)
        self.hang = nodes.reduction[self.hanging].tocsr()
        self.jacobian = False

    def fill_hanging(self, phi: np.ndarray) -> None:
        """Interpolate the hanging nodes of ``phi`` from the nodes they hang from."""
        if self.hanging.size:
            phi[self.hanging] = self.hang @ phi


class _Field3D:
    """The patchwork `NodalPoisson` of an `IonOpticsMesh3D` and its refined boxes.

    As `_Field` in 2D. ``region`` (the initial plasma) is a mask of the mesh's nodes
    (without refinement) or a function of ``(x, y, z)``.
    """

    def __init__(
        self,
        mesh: IonOpticsMesh3D,
        region: Any,
        region_potential: float,
        backend: str,
        comm: Any = None,
        refinement: Sequence[Refinement] = (),
    ) -> None:
        from patchwork import (
            AMRHierarchy,
            Dirichlet,
            EmbeddedDirichlet,
            Neumann,
            NodalPoisson,
        )

        self.mesh = mesh
        shape, h = mesh.shape, mesh.h
        origin, upper = np.array(mesh.origin), np.array(mesh.upper)
        hierarchy = AMRHierarchy(
            base_shape=tuple(n - 1 for n in shape),
            bounds=tuple((origin[a], upper[a]) for a in range(3)),
            comm=comm,
        )
        patches = patches_of_3d(mesh, refinement)
        for patch in patches[1:]:
            hierarchy.add_patch(level=patch.level, box=patch.box)

        def node_index(points: np.ndarray) -> tuple[np.ndarray, ...]:
            return tuple(
                np.rint((points[a] - origin[a]) / h).astype(np.int64) for a in range(3)
            )

        def fraction(start: np.ndarray, end: np.ndarray) -> np.ndarray:
            """The bracketed distance from a node to the surface towards ``end``.

            From the mesh of the patch, of the edge's level, that holds the node: the
            distances the field takes too.
            """
            step = end - start
            axis = np.argmax(np.abs(step), axis=0)
            length = np.abs(step[axis, np.arange(axis.size)])
            direction = 2 * axis + (step[axis, np.arange(axis.size)] > 0)
            level = np.rint(np.log2(h / length)).astype(np.int64)
            out = np.ones(axis.size)
            done = np.zeros(axis.size, dtype=bool)
            for patch in reversed(patches):
                lo, hi = np.array(patch.bounds[:3]), np.array(patch.bounds[3:])
                tiny = 1e-9 * patch.mesh.h
                mine = (
                    ~done
                    & (level == patch.level)
                    & np.all(start >= lo[:, None] - tiny, axis=0)
                    & np.all(start <= hi[:, None] + tiny, axis=0)
                )
                if mine.any():
                    index = tuple(
                        np.rint(
                            (start[a, mine] - patch.mesh.origin[a]) / patch.mesh.h
                        ).astype(np.int64)
                        for a in range(3)
                    )
                    out[mine] = patch.mesh.distance[(direction[mine], *index)]
                    done |= mine
            return out

        faces = {}
        for face in FACES_3D:
            condition = mesh.faces[face]
            value = condition.value
            given = (
                (lambda *p, c=condition: c.evaluate(*p))
                if callable(value)
                else float(value)
            )
            name = face.replace("_min", "_lower").replace("_max", "_upper")
            faces[name] = (
                Dirichlet(given) if condition.kind == "dirichlet" else Neumann(given)
            )
        bodies = []
        if region is not None:
            dirichlet = [
                FACES_3D.index(f)
                for f, c in mesh.faces.items()
                if c.kind == "dirichlet"
            ]

            def in_region(points: np.ndarray) -> np.ndarray:
                if not callable(region):
                    held = np.asarray(region, dtype=bool) & ~mesh.fixed
                    return held[node_index(points)]
                inside = np.asarray(region(*points), dtype=bool).copy()
                tiny = 1e-9 * h
                for number in dirichlet:
                    axis, high = divmod(number, 2)
                    if high:
                        inside &= points[axis] < upper[axis] - tiny
                    else:
                        inside &= points[axis] > origin[axis] + tiny
                return inside

            bodies.append(
                EmbeddedDirichlet(
                    in_region,
                    float(region_potential),
                    fraction=lambda start, end: np.ones(start.shape[1]),
                )
            )
        # patchwork lets the last body listed own a node, the mesh the first electrode
        for electrode in reversed(mesh.electrodes):
            bodies.append(
                EmbeddedDirichlet(
                    lambda points, e=electrode: np.asarray(
                        e.inside(*points), dtype=bool
                    ),
                    float(electrode.potential),
                    fraction=fraction,
                )
            )
        self.solver = NodalPoisson(
            hierarchy,
            faces=faces,
            embedded=bodies,
            backend=_patchwork_backend(backend),
            min_fraction=1.0 / 255.0,
        )
        nodes = self.solver.nodes
        #: The patches and the composite nodes.
        self.layout = layout_of_3d(patches, nodes)
        #: The hanging nodes (on coarse-fine faces) and their interpolation.
        self.hanging = np.flatnonzero(nodes.hanging)
        self.hang = nodes.reduction[self.hanging].tocsr()
        self.jacobian = False

    def fill_hanging(self, phi: np.ndarray) -> None:
        """Interpolate the hanging nodes of ``phi`` from the nodes they hang from."""
        if self.hanging.size:
            phi[self.hanging] = self.hang @ phi


def _poisson_problem() -> ConvergenceProblem:
    """A manufactured potential in cylindrical geometry, against its exact values."""
    from beamtracer.solver.mesh import FaceCondition

    def error(h: float) -> float:
        n = round(1.0 / h)
        zero = FaceCondition("dirichlet", 0.0)
        mesh = IonOpticsMesh(
            (2 * n + 1, n + 1),
            (-1.0, 0.0),
            h,
            "cylindrical",
            faces={"x_min": zero, "x_max": zero, "y_max": zero},
        )
        x, r = mesh.nodes()
        cx, cr = np.cos(0.5 * np.pi * x), np.cos(0.5 * np.pi * r)
        # sin(pi r / 2) / r, pi / 2 on the axis
        sinc = 0.5 * np.pi * np.sinc(0.5 * r)
        laplacian = -0.5 * np.pi**2 * cx * cr - 0.5 * np.pi * cx * sinc
        phi = IonOpticsPoisson(mesh).solve(-epsilon_0 * laplacian)
        return float(np.max(np.abs(phi - cx * cr)))

    return ConvergenceProblem(error, steps=[1 / 8, 1 / 16, 1 / 32, 1 / 64])


@method(
    equations={
        "poisson": r"\nabla^2 \phi = -\frac{\rho_b + \rho_p}{\varepsilon_0}, \qquad \nabla^2 = \partial_x^2 + \frac{1}{r}\partial_r (r\, \partial_r) \text{ (cylindrical, } \partial_\theta = 0)",
        "shortley_weller": r"h^2 \partial_x^2 \phi \approx \frac{2\, \phi_-}{(\alpha + \beta)\alpha} + \frac{2\, \phi_+}{(\alpha + \beta)\beta} - \frac{2\, \phi_0}{\alpha \beta}",
        "radial": r"h^2 \frac{1}{r}\partial_r (r\, \partial_r \phi) \approx \frac{2/\alpha - 1/j}{\alpha + \beta}\, \phi_- + \frac{2/\beta + 1/j}{\alpha + \beta}\, \phi_+ - \frac{2}{\alpha \beta}\, \phi_0, \quad j \ge 1",
        "axis": r"h^2 \frac{1}{r}\partial_r (r\, \partial_r \phi)\Big|_{r = 0} \approx 4 (\phi_1 - \phi_0)",
        "newton": r"\big(K + \mathrm{diag}(V)\, \tfrac{\rho_0 e^{-s(\phi - U_p)/T}}{T \varepsilon_0}\big)\, \delta\phi = -R(\phi), \qquad R = K\phi - V\, \frac{\rho_b + \rho_p(\phi)}{\varepsilon_0} - b",
    },
    order=2,
    assumptions=(
        r"patchwork's `NodalPoisson` (lumped, finite volumes; `metric='cylindrical'` in $(x, r)$, patchwork's $(r, z)$); $K$ its operator, $V$ the control volumes, $b$ the share of the fixed values",
        r"$\alpha$, $\beta$ the distances to the lower and upper neighbour or electrode surface in units of $h$, those the mesh brackets (its `distance`, as the field uses them); Shortley--Weller rows at nodes with four cells, the cut edge's two-point flux at nodes on the domain's faces",
        r"an initial plasma region is a body held at $U_p$ on its nodes (a cut at the node); a Dirichlet face or an electrode keeps its own nodes",
        r"the Newton Jacobian is patchwork's operator with the reaction $\rho_0 e^{\ldots} / (T \varepsilon_0)$ (`NodalPoisson.set_reaction`), factorised by its backend",
        r"the order is in $h$, for a smooth potential away from electrode corners",
    ),
    references=("G. H. Shortley and R. Weller, J. Appl. Phys. 9, 334 (1938)",),
    problem=_poisson_problem,
)
class IonOpticsPoisson:
    """Solve ``laplace(phi) = -rho / epsilon_0`` (plus a Boltzmann plasma species) on a mesh.

    The finite differences are patchwork's `NodalPoisson`, the ecosystem's solver (the
    cylindrical metric in r–z), with the mesh's electrode distances, on the base mesh and
    the boxes of ``refinement`` (patchwork's composite mesh, `layout`). ``fixed_region``
    is an optional mask of extra nodes held at ``fixed_potential`` (the initial plasma), or
    a function of ``(x, y)``. The factorisation is reused between solves; ``backend`` is
    patchwork's: ``"lu"`` (every rank solves the whole mesh), ``"petsc"`` (with ``comm``,
    the ranks share the mesh's rows: a Krylov method with algebraic multigrid), ``"amg"``
    (every rank, algebraic multigrid preconditioning a Krylov method: no factorisation,
    for large and 3D meshes; with a plasma a Newton--Krylov method whose AMG
    preconditioner is kept, `_newton_krylov`), ``"multigrid"`` (every rank, patchwork's
    geometric multigrid with a compiled Gauss--Seidel V-cycle; with a plasma every
    Newton step preconditioned by its own Jacobian) or ``"auto"``.
    """

    def __init__(
        self,
        mesh: IonOpticsMesh | IonOpticsMesh3D,
        reuse_jacobian: bool = True,
        backend: str = "lu",
        comm: Any = None,
        refinement: Sequence[Refinement] = (),
    ) -> None:
        """Prepare the solver for ``mesh``.

        With ``reuse_jacobian`` Newton's method keeps its factorised Jacobian from step to
        step and from solve to solve (a chord method), refactorising it when an update
        does not halve; otherwise every step factorises its own.
        """
        self.mesh = mesh
        self.reuse_jacobian = bool(reuse_jacobian)
        self.backend = backend
        #: The communicator the solve is distributed over (``"petsc"``), or None.
        self.comm = comm
        self.refinement = tuple(refinement)
        self._fields: dict[object, _Field | _Field3D] = {}
        self.newton_iterations = 0
        #: The Jacobian factorisations of the last solve.
        self.factorisations = 0
        #: The Newton steps of the last solve taken by GMRES with the factorised Jacobian
        #: as preconditioner, where the chord step stopped contracting.
        self.krylov_steps = 0

    def _field(self, region: Any, potential: float) -> _Field | _Field3D:
        """The (cached) patchwork solver for a set of fixed nodes."""
        if region is None:
            key: object = b""
        elif callable(region):
            key = (id(region), potential)
        else:
            key = np.packbits(region).tobytes() + repr(potential).encode()
        if key not in self._fields:
            # the plain one and at most one with an initial plasma
            self._fields = {k: v for k, v in self._fields.items() if k == b""}
            if isinstance(self.mesh, IonOpticsMesh3D):
                self._fields[key] = _Field3D(
                    self.mesh,
                    region,
                    potential,
                    self.backend,
                    self.comm,
                    self.refinement,
                )
            else:
                self._fields[key] = _Field(
                    self.mesh,
                    region,
                    potential,
                    self.backend,
                    self.comm,
                    self.refinement,
                )
        return self._fields[key]

    @property
    def layout(self) -> Layout | Layout3D:
        """The patches and the composite nodes the potential lives on."""
        return self._field(None, 0.0).layout

    @property
    def coordinates(self) -> np.ndarray:
        """The positions of the composite nodes, ``(ndim, nodes)`` [m] (``(x, r)`` in r-z).

        In the mesh's order of axes: patchwork's r-z hierarchies put ``r`` first.
        """
        coordinates = np.asarray(self._field(None, 0.0).solver.nodes.coordinates).T
        if getattr(self.mesh, "cylindrical", False):
            coordinates = coordinates[::-1]
        return np.ascontiguousarray(coordinates)

    def _newton_krylov(
        self,
        field: _Field | _Field3D,
        phi: np.ndarray,
        u: np.ndarray,
        base: np.ndarray,
        amplitude: float,
        temperature: float,
        up: float,
        sign: int,
        stop: float,
        max_newton: int,
    ) -> np.ndarray:
        """Newton's method with every step by GMRES, preconditioned by a multigrid cycle.

        The step solves the current Jacobian ``K + L diag(c)`` (a sparse matrix without
        hanging nodes) to a relative residual of `NEWTON_KRYLOV_RTOL`. With ``"amg"`` its
        preconditioner is the V-cycle of the algebraic-multigrid hierarchy of an earlier
        Jacobian, kept across steps, solves and Vlasov iterations, and built again only
        when GMRES fails or takes more than `NEWTON_KRYLOV_REBUILD` iterations. With
        ``"multigrid"`` (`REFRESHED_BACKENDS`) every step is preconditioned by the
        geometric multigrid of its own Jacobian ``K + T^T diag(V c) T`` (``T`` the
        hanging nodes from the unknowns, ``V`` the control volumes), which patchwork
        takes from the weights ``V c`` without a Galerkin product on the finest levels
        (``CompositeMultigrid.update_reaction``), also with refined boxes. Returns the
        unknowns.
        """
        from scipy import sparse
        from scipy.sparse.linalg import LinearOperator, gmres

        solver = field.solver
        unknowns = solver.unknowns
        n = unknowns.size
        hanging = getattr(field, "hanging", np.zeros(0))
        load_unknowns = None if np.size(hanging) else solver.load[:, unknowns].tocsr()
        builds = iterations = krylov_iterations = 0
        # a multigrid takes a new diagonal cheaply: every step its own Jacobian
        refresh = self.backend in REFRESHED_BACKENDS
        lumped = None
        take = getattr(solver.solver, "update_reaction", None)
        if refresh and callable(take):
            # the lumped reaction is K + T^T diag(V c) T (T: the hanging nodes from the
            # unknowns): hand the multigrid only the weights V c
            transfer = solver.nodes.reduction[:, unknowns]
            volumes = np.asarray(solver.volumes, dtype=float)
            probe = np.random.default_rng(0).standard_normal(phi.size)
            loaded = solver.load @ probe
            if np.allclose(
                loaded, transfer.T @ (volumes * probe), rtol=0.0,
                atol=1e-12 * float(np.abs(loaded).max() or 1.0),
            ):  # fmt: skip
                lumped = (take, volumes)
        for iterations in range(1, max_newton + 1):  # noqa: B007 (counted after the loop)
            phi[unknowns] = u
            field.fill_hanging(phi)
            density = amplitude * np.exp(
                np.minimum(-sign * (phi - up) / temperature, 50.0)
            )
            residual = solver.stiffness @ u - base - solver.load @ (sign * density)
            if solver.solver is None:  # no unknowns
                break
            reaction = density / temperature
            if lumped is not None:
                take, volumes = lumped
                take(volumes * reaction)
                field.jacobian = True
                builds += 1
                # this Jacobian, as the multigrid holds it
                operator = solver.solver.matrix  # pyright: ignore[reportAttributeAccessIssue]
            elif load_unknowns is not None:
                # no hanging nodes: the Jacobian as a sparse matrix
                operator: Any = solver.stiffness + load_unknowns @ sparse.diags(
                    reaction[unknowns]
                )
            else:

                def jacobian(
                    v: np.ndarray, reaction: np.ndarray = reaction
                ) -> np.ndarray:
                    spread = np.zeros_like(phi)
                    spread[unknowns] = v
                    field.fill_hanging(spread)
                    return solver.stiffness @ v + solver.load @ (reaction * spread)

                operator = LinearOperator((n, n), matvec=jacobian, dtype=float)  # pyright: ignore[reportCallIssue]
            delta = None
            for attempt in range(2):
                if attempt == 1 or (lumped is None and (not field.jacobian or refresh)):
                    take = getattr(solver.solver, "update_operator", None)
                    if load_unknowns is not None and callable(take):
                        take(operator)  # the multigrid of this Jacobian, as assembled
                    else:
                        solver.set_reaction(reaction)  # the hierarchy of this Jacobian
                    field.jacobian = True
                    builds += 1
                count = [0]

                def counter(_: Any, count: list[int] = count) -> None:
                    count[0] += 1

                step, info = gmres(
                    operator, -residual, M=solver.solver.preconditioner,  # pyright: ignore[reportAttributeAccessIssue]
                    rtol=NEWTON_KRYLOV_RTOL, restart=40, maxiter=2,
                    callback=counter, callback_type="pr_norm",
                )  # fmt: skip
                krylov_iterations += count[0]
                if info != 0:
                    # SciPy's GMRES stops on the preconditioned residual: a step that
                    # misses `NEWTON_KRYLOV_RTOL` but is well inside an inexact Newton
                    # method's forcing range is still taken
                    left = np.linalg.norm(operator @ step + residual)
                    if left < NEWTON_KRYLOV_ACCEPT * np.linalg.norm(residual):
                        info = 0
                # a preconditioner of this very Jacobian is not built again
                if info == 0 and (
                    count[0] <= NEWTON_KRYLOV_REBUILD or lumped is not None
                ):
                    delta = step
                    break
                delta = step if info == 0 else None
            if delta is None:
                raise RuntimeError(
                    "the Newton-Krylov step did not converge, even with a new preconditioner"
                )
            u = u + delta
            if float(np.max(np.abs(delta))) < stop:
                break
        self.newton_iterations = iterations
        self.factorisations = builds
        self.krylov_steps = krylov_iterations
        return u

    @staticmethod
    def _krylov_step(
        field: _Field | _Field3D,
        phi: np.ndarray,
        reaction: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray | None:
        """A Newton step by GMRES, preconditioned by the field's factorised Jacobian.

        The Jacobian is patchwork's operator plus the reaction on the unknowns (and, through
        their interpolation, on the hanging nodes): ``K v + L (c * phi(v))``. Returns None
        when GMRES does not reach a relative residual of 1e-3 (an inexact Newton step) in
        20 iterations.
        """
        from scipy.sparse.linalg import LinearOperator, gmres

        solver = field.solver
        unknowns = solver.unknowns
        n = unknowns.size

        def jacobian(v: np.ndarray) -> np.ndarray:
            spread = np.zeros_like(phi)
            spread[unknowns] = v
            field.fill_hanging(spread)
            return solver.stiffness @ v + solver.load @ (reaction * spread)

        factorised = solver.solver
        assert factorised is not None
        operator = LinearOperator((n, n), matvec=jacobian, dtype=float)  # pyright: ignore[reportCallIssue]
        preconditioner = LinearOperator((n, n), matvec=factorised.solve, dtype=float)  # pyright: ignore[reportCallIssue]
        delta, info = gmres(
            operator, -residual, M=preconditioner, rtol=1e-3, restart=20, maxiter=1
        )
        return delta if info == 0 else None

    @method(
        equations={
            "residual": r"R(u) = A u - b + s\, a\, e^{-s (u - U_p) / T}, \qquad a = \frac{\rho_0 h^2}{\varepsilon_0}",
            "newton": r"\left(A - \operatorname{diag}\frac{a\, e^{-s (u - U_p) / T}}{T}\right) \delta u = -R(u), \qquad u \leftarrow u + \delta u",
            "stop": r"\max |\delta u| < 10^{-6} \max\left(\operatorname{ptp}(\phi_\text{fixed}),\ T,\ 1\ \mathrm{V}\right)",
        },
        assumptions=(
            r"$s = \pm 1$ the sign of the Boltzmann species; the exponent capped at 50 to keep a bad first guess finite",
            r"started from the vacuum solution or the previous potential; undamped, at most 100 steps, the last iterate returned without a warning",
            r"with `reuse_jacobian` (the default) a chord method: the factorised Jacobian is kept across steps and solves and refactorised when $\max|\delta u|$ does not halve from one step to the next, so a step it stops on contracts by at least 2 and the error is at most the last update; it stops at a tenth of the tolerance",
            r"without a plasma species one linear solve; an initial plasma region is held at $U_p$ as extra fixed nodes",
        ),
        components=("beamtracer.solver.poisson.IonOpticsPoisson",),
    )
    @scope_profiler.profile("beamtracer.field_solve", tags=PROFILE_TAGS)
    def solve(
        self,
        rho: np.ndarray | None = None,
        *,
        plasma: BoltzmannElectrons | BoltzmannIons | None = None,
        fixed_region: np.ndarray | Callable | None = None,
        fixed_potential: float = 0.0,
        initial: np.ndarray | None = None,
        tolerance: float = 1.0e-6,
        max_newton: int = 100,
    ) -> np.ndarray:
        """Return the potential for the charge density ``rho`` [C/m^3].

        ``rho``, ``initial`` and the result are ``(nx, ny)`` arrays of the mesh's nodes, or
        arrays over the composite nodes of `layout` (with refinement they must be). With
        ``plasma``, Newton's method starts from ``initial`` (default: the vacuum solution)
        and stops when the largest update is below ``tolerance`` times the largest
        potential difference in the mesh.
        """
        region: Any = (
            fixed_region
            if fixed_region is None or callable(fixed_region)
            else np.asarray(fixed_region, dtype=bool)
        )
        field = self._field(region, float(fixed_potential))
        solver = field.solver
        layout = field.layout
        on_mesh = rho is not None and np.shape(rho) == self.mesh.shape
        if initial is not None and np.shape(initial) == self.mesh.shape:
            on_mesh = True
            initial = layout.from_base(initial)
        if rho is None:
            source = np.zeros(solver.nodes.num_nodes)
        elif on_mesh:
            source = layout.from_base(rho) / epsilon_0
        else:
            source = np.asarray(rho, dtype=float) / epsilon_0
        if rho is None and not layout.refined:
            on_mesh = True
        if plasma is None or initial is None:
            if (
                field.jacobian
            ):  # the operator holds a Newton Jacobian: the Laplacian again
                solver.set_reaction(None)
                field.jacobian = False
            phi = solver.solve(source)
            if plasma is None:
                self.newton_iterations = 0
                self.factorisations = 0
                return layout.on_base(phi) if on_mesh else phi
        else:
            phi = np.array(initial, dtype=float)
            fixed = solver.fixed
            phi[fixed] = solver.values[fixed]

        # -laplace(phi) = (rho + s rho_0 exp(-s (phi - Up) / T)) / epsilon_0 by Newton's
        # method on patchwork's unknowns; the Jacobian adds the reaction
        # rho_0 exp(...) / (T epsilon_0), which patchwork factorises with the operator
        unknowns = solver.unknowns
        u = phi[unknowns].copy()
        base = solver.load @ source + solver.constant
        amplitude = plasma.density / epsilon_0
        temperature = float(plasma.temperature)
        up = float(plasma.plasma_potential)
        sign = plasma.sign
        prescribed = solver.values[solver.fixed]
        scale = max(
            float(np.ptp(prescribed)) if prescribed.size else 0.0, temperature, 1.0
        )
        if self.backend in KRYLOV_BACKENDS:
            u = self._newton_krylov(
                field, phi, u, base, amplitude, temperature, up, sign,
                tolerance * scale, max_newton,
            )  # fmt: skip
            phi[unknowns] = u
            field.fill_hanging(phi)
            return layout.on_base(phi) if on_mesh else phi
        iteration = 0
        factorisations = 0
        krylov = 0
        # chord steps contract linearly, not quadratically: they stop ten times tighter
        stop = tolerance * scale * (0.1 if self.reuse_jacobian else 1.0)
        refresh = not self.reuse_jacobian or not field.jacobian
        fresh = False
        previous = np.inf
        for iteration in range(1, max_newton + 1):  # noqa: B007 (counted after the loop)
            phi[unknowns] = u
            field.fill_hanging(phi)
            # the exponent is limited only to keep a bad first guess finite
            density = amplitude * np.exp(
                np.minimum(-sign * (phi - up) / temperature, 50.0)
            )
            residual = solver.stiffness @ u - base - solver.load @ (sign * density)
            if refresh:
                solver.set_reaction(density / temperature)
                field.jacobian = True
                factorisations += 1
                fresh = True
            if solver.solver is None:  # no unknowns
                break
            delta = solver.solver.solve(-residual)
            size = float(np.max(np.abs(delta)))
            if self.reuse_jacobian and not fresh and not size < 0.5 * previous:
                # a stale Jacobian that no longer contracts: a Newton step with the
                # current Jacobian by GMRES, preconditioned by the stale one; if that
                # does not converge, redo the step with a new factorisation
                delta = self._krylov_step(field, phi, density / temperature, residual)
                if delta is None:
                    refresh = True
                    previous = np.inf
                    continue
                krylov += 1
                size = float(np.max(np.abs(delta)))
            u += delta
            if size < stop:
                break
            refresh = not self.reuse_jacobian
            fresh = False
            previous = size
        phi[unknowns] = u
        field.fill_hanging(phi)
        self.newton_iterations = iteration
        self.factorisations = factorisations
        self.krylov_steps = krylov
        return layout.on_base(phi) if on_mesh else phi
