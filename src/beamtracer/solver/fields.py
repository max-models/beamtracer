"""Fields for ion-optics particle tracing.

A staggered electric field and an axisymmetric magnetic field from its on-axis profile.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from methodcard import ConvergenceProblem, Rule, method

from beamtracer.kernels.staggered_field_2d import kernel as staggered_field_2d
from beamtracer.solver.mesh import IonOpticsMesh


def _edge_field(phi, solid, near, dist_lo, dist_hi, h, extrapolation):
    """Return the edge field ``(phi[i-1] - phi[i]) / h`` along axis 0.

    The edges next to electrode surfaces are taken from the bracketed surface distance;
    the result is an ``(n + 1, m)`` array.
    """
    n, m = phi.shape
    out = np.zeros((n + 1, m))
    a, b = phi[:-1], phi[1:]  # nodes i - 1 and i of the edges i = 1 .. n - 1
    sa, sb = solid[:-1], solid[1:]
    na, nb = near[:-1], near[1:]
    field = (a - b) / h
    # A free node next to the electrode node: the difference over the surface distance.
    case = na & sb
    field = np.where(case, (a - b) / (dist_hi[:-1] * h), field)
    case = nb & sa
    field = np.where(case, (a - b) / (dist_lo[1:] * h), field)
    # Both nodes in the electrode: use the surface on either side, if there is just one.
    both = sa & sb
    near0 = np.zeros_like(na)
    near0[1:] = near[:-2]  # node i - 2
    near3 = np.zeros_like(na)
    near3[:-1] = near[2:]  # node i + 1
    phi0 = np.zeros_like(a)
    phi0[1:] = phi[:-2]
    phi3 = np.zeros_like(a)
    phi3[:-1] = phi[2:]
    d0 = np.ones_like(a)
    d0[1:] = dist_hi[:-2]
    d3 = np.ones_like(a)
    d3[:-1] = dist_lo[2:]
    inside = np.zeros_like(field)
    inside = np.where(near0 & ~near3, (phi0 - a) / (d0 * h), inside)
    inside = np.where(near3 & ~near0, (b - phi3) / (d3 * h), inside)
    field = np.where(both, inside, field)
    out[1:-1] = field
    for end, (edge, first, second) in enumerate(((0, 1, 2), (n, n - 1, n - 2))):
        if extrapolation[end] == "symmetric":
            out[edge] = -out[first]
        else:
            out[edge] = 2.0 * out[first] - out[second]
    return out


def _staggered_problem() -> ConvergenceProblem:
    """The field of ``sin(x) cos(y)`` at points inside the mesh, against its gradient."""

    def error(h: float) -> float:
        n = round(2.0 / h)
        mesh = IonOpticsMesh((n + 1, n + 1), (0.0, 0.0), h, "planar")
        x, y = mesh.nodes()
        field = StaggeredElectricField(mesh, np.sin(x) * np.cos(y))
        px, py = np.random.default_rng(0).uniform(0.5, 1.5, (2, 200))
        ex, ey = field(px, py)
        return float(
            max(
                np.max(np.abs(ex + np.cos(px) * np.cos(py))),
                np.max(np.abs(ey - np.sin(px) * np.sin(py))),
            )
        )

    return ConvergenceProblem(error, steps=[0.1, 0.05, 0.025, 0.0125])


@method(
    equations={
        "edges": r"E_{x,\, i-1/2,\, j} = \frac{\phi_{i-1,j} - \phi_{i,j}}{h}, \qquad E_{y,\, i,\, j-1/2} = \frac{\phi_{i,j-1} - \phi_{i,j}}{h}",
        "surface": r"E_\text{edge} = \pm\frac{\phi_\text{near} - \phi_\text{electrode}}{d\, h} \quad \text{between a near-solid node and its electrode node}",
        "faces": r"E_\text{face} = -E_\text{first} \ \text{(mirror: the axis)} \quad \text{or} \quad E_\text{face} = 2 E_\text{first} - E_\text{second} \ \text{(linear)}",
        "interpolation": r"E(x, y) = \text{bilinear interpolation on each staggered grid}, \qquad E_x \text{ at the index } \frac{x - x_0}{h} + \tfrac12",
    },
    order=2,
    assumptions=(
        r"$d$ the bracketed distance to the surface in units of $h$; on an edge inside an electrode the field is extrapolated from the one adjacent surface, if there is exactly one",
        r"the axis of a cylindrical mesh mirrors, so $E_r = 0$ there; the other faces extrapolate linearly",
        r"points outside the mesh are clamped to its edges; the order is in $h$, away from electrodes and faces",
    ),
    problem=_staggered_problem,
)
class StaggeredElectricField:
    """The electric field of a potential on the edges between nodes.

    ``E_x`` lives on the edges between nodes along ``x`` and ``E_y`` on those along ``y``;
    both are interpolated bilinearly. On an edge between a free node next to an electrode
    and the electrode's first node, the difference is taken over the distance to the
    surface instead of ``h``, so the field at an electrode surface is that of the surface.
    ``extrapolation`` gives, per face (``x_min, x_max, y_min, y_max``), ``"symmetric"`` (the
    field vanishes on the face, e.g. the symmetry axis) or ``"extrapolate"`` (linear).
    """

    def __init__(
        self,
        mesh: IonOpticsMesh,
        potential: np.ndarray,
        extrapolation: Sequence[str] | None = None,
    ) -> None:
        """Precompute the edge fields of ``potential`` ``(nx, ny)``."""
        if extrapolation is None:
            extrapolation = ["extrapolate"] * 4
            if mesh.cylindrical:
                extrapolation[2] = "symmetric"
        self.mesh = mesh
        self.extrapolation = tuple(extrapolation)
        phi = np.asarray(potential, dtype=float)
        solid = mesh.solid >= 0
        near = mesh.near_solid
        # E_x on (nx + 1, ny) edges and E_y on (nx, ny + 1), from the same 1D rule.
        self.fx = _edge_field(
            phi,
            solid,
            near,
            mesh.distance[0],
            mesh.distance[1],
            mesh.h,
            self.extrapolation[0:2],
        )
        self.fy = np.ascontiguousarray(
            _edge_field(
                phi.T,
                solid.T,
                near.T,
                mesh.distance[2].T,
                mesh.distance[3].T,
                mesh.h,
                self.extrapolation[2:4],
            ).T
        )

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(E_x, E_y)`` [V/m] at the points ``(x, y)``."""
        mesh = self.mesh
        x = np.ascontiguousarray(x, dtype=float).ravel()
        y = np.ascontiguousarray(y, dtype=float).ravel()
        ex, ey = np.empty_like(x), np.empty_like(y)
        if x.size:
            staggered_field_2d(
                x, y, self.fx, self.fy, *map(float, mesh.origin), mesh.h, ex, ey
            )
        return ex, ey


def _magnetic_problem() -> ConvergenceProblem:
    """A Gaussian profile, off the axis, against the series with its exact derivatives."""
    z = np.linspace(-1.0, 1.0, 41) + 0.0123
    r = 0.2
    g = np.exp(-(z**2))
    bz = g - r**2 / 4.0 * (4.0 * z**2 - 2.0) * g
    br = -r / 2.0 * (-2.0 * z * g) + r**3 / 16.0 * (-8.0 * z**3 + 12.0 * z) * g

    def error(hz: float) -> float:
        samples = np.arange(-5.0, 5.0 + 0.5 * hz, hz)
        field = AxisymmetricMagneticField(-5.0, hz, np.exp(-(samples**2)), order=3)
        got_z, got_r = field(z, np.full_like(z, r))
        return float(max(np.max(np.abs(got_z - bz)), np.max(np.abs(got_r - br))))

    return ConvergenceProblem(error, steps=[0.2, 0.1, 0.05, 0.025])


@method(
    equations={
        "expansion": r"B_z = \sum_k (-1)^k \left(\frac{r}{2}\right)^{2k} \frac{B^{(2k)}}{(k!)^2}, \qquad B_r = \sum_k (-1)^{k+1} \left(\frac{r}{2}\right)^{2k+1} \frac{B^{(2k+1)}}{k!\,(k+1)!}",
        "truncated": r"B_z = B - \frac{r^2}{4} B'' + \frac{r^4}{64} B'''' - \frac{r^6}{2304} B^{(6)}, \qquad B_r = -\frac{r}{2} B' + \frac{r^3}{16} B''' - \frac{r^5}{384} B^{(5)}",
        "stencils": r"B^{(k)}_i = \frac{1}{\Delta z^k} \sum_{a=-3}^{3} c^{(k)}_a B_{i+a}, \quad \text{of orders } 6, 6, 4, 4, 2, 2 \text{ for } k = 1, \dots, 6",
        "interpolation": r"B^{(k)}(z) = (1 - t)\, B^{(k)}_i + t\, B^{(k)}_{i+1}, \qquad t = \frac{z - z_0}{\Delta z} - i",
    },
    order=2,
    assumptions=(
        r"current-free and divergence-free near the axis; $B_z$ kept to $r^{2\lfloor n/2\rfloor}$ and $B_r$ to $r^{2\lceil n/2\rceil - 1}$ for `order` $n \le 6$",
        r"at least 8 samples; the three end samples have no derivatives, and beyond the valid cells the derivatives are extrapolated linearly",
        r"$\nabla \cdot \mathbf B = 0$ to $O(\Delta z^2)$: $B_z$ is interpolated linearly, $B_r$ uses the differenced derivatives",
        r"the order is in the sample spacing $\Delta z$, between the samples (the linear interpolation)",
    ),
    references=(
        "[Reiser, Theory and Design of Charged Particle Beams, 2nd ed. (2008)](https://doi.org/10.1002/9783527622047)",
    ),
    problem=_magnetic_problem,
)
class AxisymmetricMagneticField:
    r"""A static axisymmetric magnetic field from its on-axis profile ``B_z(z)``.

    ``bz`` [T] is sampled at ``z0 + hz * k``. Off the axis, the field is the power series in
    ``r`` up to ``order`` (at most 6), with the derivatives of the profile from 7-point finite
    differences:

    $$
    B_z = B - \frac{r^2}{4} B'' + \frac{r^4}{64} B'''' - \dots, \qquad
    B_r = -\frac{r}{2} B' + \frac{r^3}{16} B''' - \dots
    $$
    """

    def __init__(
        self, z0: float, hz: float, bz: Sequence[float] | np.ndarray, order: int = 6
    ) -> None:
        """Precompute the derivatives of the profile at its samples."""
        samples = np.asarray(bz, dtype=float)
        if samples.ndim != 1 or samples.size < 8:
            raise ValueError("the on-axis profile needs at least 8 samples")
        if not 0 <= order <= 6:
            raise ValueError(f"order must be 0 to 6, got {order}")
        self.z0 = float(z0)
        self.hz = float(hz)
        self.order = int(order)
        self.bz = samples
        stencils = (
            (0, 0, 0, 1, 0, 0, 0),
            (-1 / 60, 9 / 60, -45 / 60, 0, 45 / 60, -9 / 60, 1 / 60),
            (2 / 180, -27 / 180, 270 / 180, -490 / 180, 270 / 180, -27 / 180, 2 / 180),
            (1 / 8, -8 / 8, 13 / 8, 0, -13 / 8, 8 / 8, -1 / 8),
            (-1 / 6, 12 / 6, -39 / 6, 56 / 6, -39 / 6, 12 / 6, -1 / 6),
            (-1 / 2, 4 / 2, -5 / 2, 0, 5 / 2, -4 / 2, 1 / 2),
            (1, -6, 15, -20, 15, -6, 1),
        )
        n = samples.size
        centre = np.arange(3, n - 3)
        # Only the derivatives the expansion uses.
        self._derivatives = np.zeros((self.order + 1, n))
        for k, stencil in enumerate(stencils[: self.order + 1]):
            d = sum(c * samples[centre + a - 3] for a, c in enumerate(stencil))
            self._derivatives[k, centre] = d / self.hz**k
        # Row per sample, for fast gathers.
        self._samples = np.ascontiguousarray(self._derivatives.T)

    def derivatives(self, z: np.ndarray) -> np.ndarray:
        """Return ``d^k B_z / dz^k`` for ``k = 0..order`` at ``z``.

        The shape is ``(order + 1, len(z))``.
        """
        n = self.bz.size
        s = (np.asarray(z, dtype=float) - self.z0) / self.hz
        i = np.clip(np.floor(s).astype(np.int64), 3, n - 5)
        t = (s - i)[:, None]
        low = self._samples[i]
        return (low + t * (self._samples[i + 1] - low)).T

    def __call__(self, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(B_z, B_r)`` [T] at axial positions ``z`` and radii ``r``."""
        d = self.derivatives(z)
        r = np.asarray(r, dtype=float)
        r2 = r * r
        bz_coefficients = (1.0, -1.0 / 4.0, 1.0 / 64.0, -1.0 / 2304.0)
        br_coefficients = (-1.0 / 2.0, 1.0 / 16.0, -1.0 / 384.0)
        bz = d[0].copy()
        br = np.zeros_like(bz)
        power = np.ones_like(r2)
        for k in range(1, self.order // 2 + 1):
            power = power * r2
            bz += bz_coefficients[k] * power * d[2 * k]
        power = r.copy()
        for k in range((self.order + 1) // 2):
            br += br_coefficients[k] * power * d[2 * k + 1]
            power = power * r2
        return bz, br


def _solenoid_rules() -> dict:
    """The SymPy rule of `solenoid_on_axis` (for the tests)."""
    import sympy

    z, c, length, a, peak = sympy.symbols("z centre length radius peak", real=True)

    def shape(at):
        low, high = at - c + length / 2, at - c - length / 2
        return low / sympy.sqrt(low**2 + a**2) - high / sympy.sqrt(high**2 + a**2)

    ranges = {"z": (-2.0, 2.0), "centre": (-0.5, 0.5)}
    return {"shape": Rule(peak * shape(z) / shape(c), ranges=ranges)}


@method(
    equations={
        "shape": r"B(z) = B_0 \frac{S(z)}{S(c)}, \qquad S(z) = \frac{z - c + L/2}{\sqrt{(z - c + L/2)^2 + a^2}} - \frac{z - c - L/2}{\sqrt{(z - c - L/2)^2 + a^2}}",
    },
    assumptions=(
        r"a thin-walled solenoid (a finite current sheet) of length $L$, winding radius $a$ and centre $c$; $S$ is its on-axis field in units of $\mu_0 n I / 2$",
        r"scaled so that $B_0$ is the field at the centre",
        r"`BeamOptics` samples the solenoids at $\Delta z = h/2$ from $x_\mathrm{min} - 8\Delta z$ to $x_\mathrm{max} + 8\Delta z$, or adds them to a measured axial field",
    ),
    symbolic=_solenoid_rules,
)
def solenoid_on_axis(
    z: np.ndarray, centre: float, length: float, radius: float, peak: float
) -> np.ndarray:
    """Return the on-axis ``B_z`` [T] of a thin-walled solenoid scaled to ``peak`` [T].

    The solenoid of ``length`` and winding ``radius`` [m] is centred at ``centre``.
    """
    z = np.asarray(z, dtype=float)

    def shape(zz):
        a = zz - centre + 0.5 * length
        b = zz - centre - 0.5 * length
        return a / np.hypot(a, radius) - b / np.hypot(b, radius)

    return peak * shape(z) / shape(np.asarray(centre))


def _node_component(
    low: np.ndarray, high: np.ndarray, solid_low: np.ndarray, solid_high: np.ndarray
) -> np.ndarray:
    """Return a field component at the nodes from its edges below and above them.

    The edge to a solid neighbour is the field at the surface; with solid neighbours on
    both sides the larger one counts, and without any the mean of the two edges.
    """
    mean = 0.5 * (low + high)
    larger = np.where(np.abs(low) >= np.abs(high), low, high)
    return np.where(
        solid_low & solid_high,
        larger,
        np.where(solid_low, low, np.where(solid_high, high, mean)),
    )


@method(
    equations={
        "component": r"E_{x,\text{node}} = \begin{cases} E \text{ on the edge to the electrode} & \text{a solid neighbour on one side} \\ \text{the larger in magnitude} & \text{solid neighbours on both sides} \\ \tfrac12 (E_{i-1/2} + E_{i+1/2}) & \text{none} \end{cases}",
        "magnitude": r"|E| = \sqrt{E_x^2 + E_y^2} \quad \text{at the near-solid nodes}",
    },
    assumptions=(
        r"the edge to an electrode is taken over the bracketed distance to its surface, so it is the field at the surface",
        r"at a sharp corner the field is singular and this value grows as the mesh is refined: round the electrode edges whose field matters",
    ),
    components=("beamtracer.solver.fields.StaggeredElectricField",),
)
def surface_field(
    mesh: IonOpticsMesh, field: StaggeredElectricField
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``|E|`` [V/m] at the free nodes next to an electrode, and that electrode.

    Both arrays have the mesh shape. At a free node with an electrode neighbour along an
    axis, that component of the field is the one on the edge to the neighbour, taken over
    the bracketed distance to the surface (the field at the surface); otherwise it is the
    mean of the node's two edges. ``|E|`` is ``NaN`` and the electrode index ``-1`` away
    from the electrodes. At a sharp corner of an electrode the field is singular and this
    value grows as the mesh is refined, so round the edges of electrodes whose surface
    field matters.
    """
    # the electrode index of the neighbour at -x, +x, -y, +y (-1: none)
    neighbours = [
        mesh._neighbour_solid(axis, step)
        for axis, step in ((0, -1), (0, 1), (1, -1), (1, 1))
    ]
    solid = [n >= 0 for n in neighbours]
    ex = _node_component(field.fx[:-1], field.fx[1:], solid[0], solid[1])
    ey = _node_component(field.fy[:, :-1], field.fy[:, 1:], solid[2], solid[3])
    near = mesh.near_solid
    magnitude = np.where(near, np.hypot(ex, ey), np.nan)
    electrode = np.full(mesh.shape, -1, dtype=np.int32)
    for neighbour in neighbours:
        electrode = np.where(electrode < 0, neighbour, electrode)
    return magnitude, np.where(near, electrode, -1).astype(np.int32)
