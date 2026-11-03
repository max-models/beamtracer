r"""Poisson's equation on an `IonOpticsMesh`, with a Boltzmann plasma species.

The finite differences are the 5-point Laplacian, in
cylindrical geometry ``(1 -/+ 1/2j)`` weights on the radial neighbours and ``4 (phi_1 - phi_0)``
on the axis, Shortley--Weller weights next to electrodes, and mirrored neighbours on Neumann
faces. Each row is scaled by ``h**2``.

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

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scope_profiler
from methodcard import ConvergenceProblem, method
from scipy.constants import epsilon_0

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.solver.mesh import IonOpticsMesh


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


class _System:
    """The matrix of the free nodes, the right-hand side of the fixed ones and their LU."""

    def __init__(self, matrix: sp.csr_matrix, rhs_fixed: np.ndarray, free: np.ndarray):
        self.matrix = matrix
        self.rhs_fixed = rhs_fixed
        self.free = free
        self._lu = None

    def lu(self):
        """Return the LU factorisation of the matrix (computed once)."""
        if self._lu is None:
            self._lu = spla.splu(self.matrix.tocsc())
        return self._lu


def _assemble(mesh: IonOpticsMesh, fixed: np.ndarray, value: np.ndarray) -> _System:
    """Assemble the finite differences for the free nodes."""
    nx, ny = mesh.shape
    h = mesh.h
    flat_fixed = fixed.ravel()
    flat_value = value.ravel()
    free = np.flatnonzero(~flat_fixed)
    number = np.full(nx * ny, -1, dtype=np.int64)
    number[free] = np.arange(free.size)
    i, j = np.unravel_index(free, (nx, ny))
    alpha_x, beta_x, alpha_y, beta_y = (
        mesh.distance[d].ravel()[free] for d in range(4)
    )

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs = np.zeros(free.size)
    centre = np.zeros(free.size)

    def link(mask: np.ndarray, di: int, dj: int, weight: np.ndarray | float) -> None:
        """Couple the free nodes in ``mask`` to their neighbour ``(i + di, j + dj)``."""
        sel = np.flatnonzero(mask)
        if sel.size == 0:
            return
        neighbour = (i[sel] + di) * ny + (j[sel] + dj)
        w = np.broadcast_to(weight, mask.shape)[sel]
        known = flat_fixed[neighbour]
        rhs[sel[known]] -= w[known] * flat_value[neighbour[known]]
        rows.append(sel[~known])
        cols.append(number[neighbour[~known]])
        vals.append(w[~known])

    def face_value(face: str, mask: np.ndarray) -> np.ndarray:
        """Return the outward normal derivative of a Neumann face at the masked nodes."""
        out = np.zeros(free.size)
        if mask.any():
            x = mesh.origin[0] + h * i[mask]
            y = mesh.origin[1] + h * j[mask]
            out[mask] = mesh.faces[face].evaluate(x, y)
        return out

    # x direction
    xmin = i == 0
    xmax = i == nx - 1
    inner = ~(xmin | xmax)
    beta2 = 2.0 / beta_x**2
    alpha2 = 2.0 / alpha_x**2
    centre += np.where(xmin, beta2, np.where(xmax, alpha2, 2.0 / (alpha_x * beta_x)))
    link(xmin, 1, 0, beta2)
    link(xmax, -1, 0, alpha2)
    link(inner, -1, 0, 2.0 / ((alpha_x + beta_x) * alpha_x))
    link(inner, 1, 0, 2.0 / ((alpha_x + beta_x) * beta_x))
    rhs -= 2.0 * h * face_value("x_min", xmin) / beta_x
    rhs -= 2.0 * h * face_value("x_max", xmax) / alpha_x

    # y (or r) direction
    ymin = j == 0
    ymax = j == ny - 1
    inner = ~(ymin | ymax)
    alpha2 = 2.0 / alpha_y**2
    if mesh.cylindrical:
        jr = np.maximum(j, 1).astype(float)
        centre += np.where(ymin, 4.0, np.where(ymax, alpha2, 2.0 / (alpha_y * beta_y)))
        link(ymin, 0, 1, 4.0)
        link(ymax, 0, -1, alpha2)
        link(inner, 0, -1, (2.0 / alpha_y - 1.0 / jr) / (alpha_y + beta_y))
        link(inner, 0, 1, (2.0 / beta_y + 1.0 / jr) / (alpha_y + beta_y))
        rhs -= (
            (2.0 / alpha_y + 1.0 / jr)
            / (2.0 * alpha_y)
            * 2.0
            * h
            * face_value("y_max", ymax)
            / alpha_y
        )
    else:
        beta2 = 2.0 / beta_y**2
        centre += np.where(
            ymin, beta2, np.where(ymax, alpha2, 2.0 / (alpha_y * beta_y))
        )
        link(ymin, 0, 1, beta2)
        link(ymax, 0, -1, alpha2)
        link(inner, 0, -1, 2.0 / ((alpha_y + beta_y) * alpha_y))
        link(inner, 0, 1, 2.0 / ((alpha_y + beta_y) * beta_y))
        rhs -= 2.0 * h * face_value("y_min", ymin) / beta_y
        rhs -= 2.0 * h * face_value("y_max", ymax) / alpha_y

    rows.append(np.arange(free.size))
    cols.append(np.arange(free.size))
    vals.append(-centre)
    matrix = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(free.size, free.size),
    )
    return _System(matrix, rhs, free)


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
        "neumann": r"\phi_{-1} = \phi_1 + 2 h g: \qquad h^2 \partial_x^2 \phi \approx \frac{2 (\phi_1 - \phi_0)}{\beta^2} + \frac{2 h g}{\beta}",
        "rhs": r"b = -\frac{h^2 \rho}{\varepsilon_0} - \sum_\text{fixed neighbours} w\, \phi_\text{fixed}",
    },
    order=2,
    assumptions=(
        r"$\alpha$, $\beta$ the distances to the lower and upper neighbour or electrode surface in units of $h$ (1 away from electrodes); for $\alpha = \beta = 1$ the 5-point stencil",
        r"$g$ the outward normal derivative of a Neumann face [V/m]",
        r"the linear system solved by SuperLU (`scipy.sparse.linalg.splu`), its factorisation cached per set of fixed nodes",
        r"the order is in $h$, for a smooth potential away from electrode corners",
    ),
    references=("G. H. Shortley and R. Weller, J. Appl. Phys. 9, 334 (1938)",),
    problem=_poisson_problem,
)
class IonOpticsPoisson:
    """Solve ``laplace(phi) = -rho / epsilon_0`` (plus a Boltzmann plasma species) on a mesh.

    ``fixed_region`` is an optional ``(nx, ny)`` mask of extra nodes held at ``fixed_potential``
    (the initial plasma). The LU factorisation of the matrix is reused between solves
    without plasma electrons.
    """

    def __init__(self, mesh: IonOpticsMesh) -> None:
        """Prepare the solver for ``mesh``."""
        self.mesh = mesh
        self._systems: dict[bytes, _System] = {}
        self.newton_iterations = 0

    def _system(self, fixed: np.ndarray, value: np.ndarray) -> _System:
        """Return the (cached) linear system for a set of fixed nodes."""
        key = np.packbits(fixed).tobytes() + value[fixed].tobytes()
        if key not in self._systems:
            self._systems = {key: _assemble(self.mesh, fixed, value)}
        return self._systems[key]

    @method(
        equations={
            "residual": r"R(u) = A u - b + s\, a\, e^{-s (u - U_p) / T}, \qquad a = \frac{\rho_0 h^2}{\varepsilon_0}",
            "newton": r"\left(A - \operatorname{diag}\frac{a\, e^{-s (u - U_p) / T}}{T}\right) \delta u = -R(u), \qquad u \leftarrow u + \delta u",
            "stop": r"\max |\delta u| < 10^{-6} \max\left(\operatorname{ptp}(\phi_\text{fixed}),\ T,\ 1\ \mathrm{V}\right)",
        },
        assumptions=(
            r"$s = \pm 1$ the sign of the Boltzmann species; the exponent capped at 50 to keep a bad first guess finite",
            r"started from the vacuum solution or the previous potential; undamped, at most 100 steps, the last iterate returned without a warning",
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
        fixed_region: np.ndarray | None = None,
        fixed_potential: float = 0.0,
        initial: np.ndarray | None = None,
        tolerance: float = 1.0e-6,
        max_newton: int = 100,
    ) -> np.ndarray:
        """Return the potential ``(nx, ny)`` for the charge density ``rho`` [C/m^3].

        With ``plasma``, Newton's method starts from ``initial`` (default: the vacuum
        solution) and stops when the largest update is below ``tolerance`` times the largest
        potential difference in the mesh.
        """
        mesh = self.mesh
        fixed = mesh.fixed.copy()
        value = mesh.fixed_value.copy()
        if fixed_region is not None:
            extra = np.asarray(fixed_region, dtype=bool) & ~fixed
            fixed |= extra
            value[extra] = fixed_potential
        system = self._system(fixed, value)
        rhs = system.rhs_fixed.copy()
        if rho is not None:
            rhs -= (
                np.asarray(rho, dtype=float).ravel()[system.free]
                * mesh.h**2
                / epsilon_0
            )
        phi = value.ravel().copy()
        if plasma is None:
            phi[system.free] = system.lu().solve(rhs)
            self.newton_iterations = 0
            return phi.reshape(mesh.shape)

        if initial is None:
            u = system.lu().solve(rhs)
        else:
            u = np.asarray(initial, dtype=float).ravel()[system.free].copy()
        amplitude = plasma.density * mesh.h**2 / epsilon_0
        temperature = float(plasma.temperature)
        up = float(plasma.plasma_potential)
        sign = plasma.sign
        scale = max(
            float(np.ptp(value[fixed])) if fixed.any() else 0.0, temperature, 1.0
        )
        diagonal = sp.diags(np.ones(system.free.size))
        iteration = 0
        for iteration in range(1, max_newton + 1):  # noqa: B007 (counted after the loop)
            # A u = rhs - s a exp(-s (u - Up) / T) for a species of charge sign s; the
            # exponent is limited only to keep a bad first guess finite.
            density = amplitude * np.exp(
                np.minimum(-sign * (u - up) / temperature, 50.0)
            )
            residual = system.matrix @ u - rhs + sign * density
            jacobian = system.matrix - diagonal.multiply(density / temperature)
            delta = spla.splu(jacobian.tocsc()).solve(-residual)
            u += delta
            if np.max(np.abs(delta)) < tolerance * scale:
                break
        self.newton_iterations = iteration
        phi[system.free] = u
        return phi.reshape(mesh.shape)
