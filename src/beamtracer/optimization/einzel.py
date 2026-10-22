r"""An einzel lens: the first design problem of :mod:`beamtracer.optimization`.

Three coaxial tubes in cylindrical ``(x, r)`` geometry, the beam axis along ``x``: the
outer two at 0 V, the middle one at the design voltage ``V``, inside a grounded wall.
Each tube is ``electrode_length`` long and ``thickness`` thick, its bore at the radius
``aperture``, with half-round ends (made of a simgeometry rectangle and two circles: a
sharp corner has a singular surface field, which grows as the mesh is refined); ``gap``
separates neighbouring tubes. A parallel beam of ions (protons by default) is launched upstream with ``energy`` and
radius ``beam_radius``, and the objective is its rms size at the ``target`` plane
downstream. Trajectories that hit an electrode are lost, and the lost fraction of the
current is a constraint, as is the largest field on the electrode surfaces.

:class:`EinzelLens` holds the fixed settings and makes the
:class:`~beamtracer.optimization.Design` and
:class:`~beamtracer.optimization.Problem` of any of the parameters ``voltage``,
``gap`` and ``aperture``; :meth:`EinzelLens.thin_lens_voltage` and
:meth:`EinzelLens.paraxial_voltage` are the voltages that focus at the target in the
thin-lens and paraxial approximations.

The thin lens
-------------
For a paraxial ray in an axially symmetric electrostatic field, with :math:`\phi(z)` the
potential relative to the ions' zero of kinetic energy (the kinetic energy per charge,
:math:`\phi = T / q = T_0 / q - V(z)` on the axis for ions starting at :math:`V = 0`),
the ray equation (non-relativistic)

.. math:: r'' + \frac{\phi'}{2\phi} r' + \frac{\phi''}{4\phi} r = 0

becomes, with the Picht transformation :math:`R = r \phi^{1/4}`,

.. math:: R'' + \frac{3}{16} \left(\frac{\phi'}{\phi}\right)^2 R = 0 .

An einzel lens has the same potential on both sides, so :math:`R` and :math:`r` agree up
to a constant outside it; for a weak (thin) lens :math:`R` hardly changes inside it, and
a parallel ray of radius :math:`r` leaves with the slope :math:`-r/f`,

.. math:: \frac{1}{f} = \frac{3}{16} \int \left(\frac{\phi'}{\phi}\right)^2 dz .

See the paraxial ray equation and electrostatic lenses in M. Reiser, *Theory and Design
of Charged Particle Beams* (Wiley, 2nd ed. 2008), S. Humphries, *Principles of Charged
Particle Acceleration* (Wiley, 1986) and H. Wollnik, *Optics of Charged Particles*
(Academic Press, 1987). It holds for paraxial rays and
a lens short compared with ``f``; :math:`\phi` must stay positive (no ion is
reflected). Here the thin lens sits at the middle electrode, and its focal length is the
distance to the target. :meth:`EinzelLens.paraxial_voltage` integrates the Picht equation
through the whole lens instead (thick, still paraxial and without aberrations).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq

from beamtracer.config import BeamOpticsConfig, BeamSpecies
from beamtracer.model import BeamOptics
from beamtracer.optimization.design import Design, Evaluation, Parameter, Problem
from beamtracer.sections import (
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ObjectParams,
)

#: The design parameters of the lens and their units.
PARAMETERS = {"voltage": "V", "gap": "m", "aperture": "m"}

#: The names of the three tubes; a rounded tube's end circles are ``<name>.<side>``.
ELECTRODES = ("upstream", "middle", "downstream")

#: The current [A] of a beam without space charge (``current=0``): the moments at the
#: target need a current, and a nanoampere's space charge is negligible.
NEGLIGIBLE_CURRENT = 1e-9


@dataclass(frozen=True)
class EinzelLens:
    """The fixed settings of an einzel lens, and its problems.

    The beam: ions of ``energy`` [eV], ``charge`` [e] and ``mass`` [u] (protons by
    default), ``current`` [A] (0: no space charge), launched parallel at
    ``x = start`` [m] from the axis to ``beam_radius`` [m] as ``num_particles`` rings.
    The lens: three tubes of ``electrode_length`` [m] and ``thickness`` [m], the middle
    one centred on ``x = centre`` [m], with half-round ends if ``rounded``;
    ``voltage`` [V], ``gap`` [m] and ``aperture`` [m] are the values of the parameters
    not being optimized. The domain is ``length`` by ``radius`` [m] with nodes every
    ``cell`` [m], grounded at both ends and on the outer wall; the objective is the rms size (``sqrt(<r^2> / 2)``, the projected rms) at
    ``x = target`` [m]. The constraints are the largest surface field over
    ``field_limit`` [V/m], minus 1, and the lost fraction of the current minus
    ``max_loss``. ``iterations``, ``tolerance`` [V] and ``relaxation`` set the Vlasov
    iteration with space charge; without it, one field solve and trace is the solution.

    A frozen dataclass, so it pickles: its :meth:`config` and :meth:`evaluate` are the
    build and evaluation functions of its problems, also in a process pool.
    """

    energy: float = 30e3
    charge: float = 1.0
    mass: float = 1.007276
    current: float = 0.0
    beam_radius: float = 2e-3
    num_particles: int = 60
    start: float = 1e-3
    voltage: float = 15e3
    gap: float = 4e-3
    aperture: float = 5e-3
    electrode_length: float = 8e-3
    thickness: float = 2e-3
    rounded: bool = True
    centre: float = 40e-3
    target: float = 120e-3
    length: float = 130e-3
    radius: float = 20e-3
    cell: float = 0.5e-3
    field_limit: float = 5e6
    max_loss: float = 0.0
    iterations: int = 40
    tolerance: float = 1e-2
    relaxation: float = 0.8
    save_trajectories: int = 0

    # -- the configuration ----------------------------------------------------------------

    def values(self, values: Mapping[str, float] | None = None) -> dict[str, float]:
        """Return all three parameters: ``values``, the defaults for the others."""
        given: dict[str, float] = dict(values or {})
        unknown = set(given) - set(PARAMETERS)
        if unknown:
            raise KeyError(
                f"unknown parameters {sorted(unknown)}; known {list(PARAMETERS)}"
            )
        return {
            name: float(given[name] if name in given else getattr(self, name))
            for name in PARAMETERS
        }

    def electrode(
        self, name: str, x: float, aperture: float, voltage: float
    ) -> dict[str, ObjectParams]:
        """Return the objects of a tube centred on ``x`` with its bore at ``aperture``.

        A tube of ``electrode_length`` along the axis and ``thickness`` across it. With
        ``rounded`` its ends are half circles (a rectangle and two circles, named
        ``name``, ``name.upstream`` and ``name.downstream``); without, it is a rectangle
        with sharp corners, whose surface field is singular.
        """
        rho = 0.5 * self.thickness
        middle = aperture + rho
        if not self.rounded:
            return {
                name: ObjectParams(
                    type="rectangle",
                    bc_type="dirichlet",
                    bc_value=float(voltage),
                    midpoint=(float(x), float(middle)),
                    side_lengths=(float(self.electrode_length), float(self.thickness)),
                )
            }
        straight = self.electrode_length - self.thickness
        ends = (x - 0.5 * straight, x + 0.5 * straight)
        objects = {
            name: ObjectParams(
                type="rectangle",
                bc_type="dirichlet",
                bc_value=float(voltage),
                midpoint=(float(x), float(middle)),
                side_lengths=(float(straight), float(self.thickness)),
            )
        }
        for side, end in zip(("upstream", "downstream"), ends, strict=True):
            objects[f"{name}.{side}"] = ObjectParams(
                type="circle",
                bc_type="dirichlet",
                bc_value=float(voltage),
                midpoint=(float(end), float(middle)),
                radius=float(rho),
            )
        return objects

    def electrode_positions(self, gap: float) -> tuple[float, float, float]:
        """Return the centres [m] of the three tubes."""
        pitch = self.electrode_length + gap
        return (self.centre - pitch, self.centre, self.centre + pitch)

    def config(self, values: Mapping[str, float] | None = None) -> BeamOpticsConfig:
        """Return the :class:`~beamtracer.BeamOpticsConfig` of the lens.

        ``values`` are some of ``voltage``, ``gap`` and ``aperture``; the others are the
        lens defaults. Raises :class:`ValueError` for a lens that does not fit between
        the start of the beam and the target.
        """
        v = self.values(values)
        positions = self.electrode_positions(v["gap"])
        half = 0.5 * self.electrode_length
        if positions[0] - half <= self.start or positions[2] + half >= self.target:
            raise ValueError("the lens must lie between the beam start and the target")
        if not 0.0 < self.target < self.length:
            raise ValueError("the target must lie inside the domain")
        if v["aperture"] + self.thickness >= self.radius:
            raise ValueError("the electrodes must fit inside the outer wall")
        if self.rounded and self.electrode_length < self.thickness:
            raise ValueError("rounded tubes need electrode_length >= thickness")
        nx = round(self.length / self.cell) + 1
        ny = round(self.radius / self.cell) + 1
        grounded = FieldBoundaryCondition.dirichlet(0.0)
        neumann = FieldBoundaryCondition.neumann(0.0)
        current = self.current if self.current > 0.0 else NEGLIGIBLE_CURRENT
        return BeamOpticsConfig(
            grid=GridParams(
                num_gridpoints=(nx, ny),
                upper_bound=((nx - 1) * self.cell, (ny - 1) * self.cell),
                coordinates="cylindrical",
            ),
            bcs_field=FieldBCs(grounded, grounded, neumann, grounded),
            objects={
                key: obj
                for name, x, voltage in zip(
                    ELECTRODES, positions, (0.0, v["voltage"], 0.0), strict=True
                )
                for key, obj in self.electrode(name, x, v["aperture"], voltage).items()
            },
            beams=[
                BeamSpecies(
                    "ion",
                    current / (np.pi * self.beam_radius**2),
                    (self.start, 0.0),
                    (self.start, self.beam_radius),
                    charge=self.charge,
                    mass=self.mass,
                    energy=self.energy,
                    parallel_temperature=0.0,
                    transverse_temperature=0.0,
                    num_particles=self.num_particles,
                )
            ],
            # without space charge the field does not depend on the beam: one pass
            iterations=self.iterations if self.current > 0.0 else 1,
            tolerance=self.tolerance,
            relaxation=self.relaxation,
            diagnostic_planes=[self.target],
            save_trajectories=self.save_trajectories,
        )

    def solve(self, values: Mapping[str, float] | None = None) -> BeamOptics:
        """Return the solved :class:`~beamtracer.BeamOptics` of the lens."""
        return BeamOptics(self.config(values)).run()

    # -- the evaluation -------------------------------------------------------------------

    def measure(self, model: BeamOptics) -> Evaluation:
        """Return the :class:`~beamtracer.optimization.Evaluation` of a solved lens.

        The objective ``spot`` is the rms size [m] at the target; the constraints are
        ``surface_field`` (the largest surface field over ``field_limit``, minus 1) and
        ``loss`` (the lost fraction of the current minus ``max_loss``). The diagnostics
        hold the field [V/m], the transmission, the rms, 90 % and largest radii [m] at
        the target, and the Vlasov iterations.
        """
        hits = model.plane(self.target)
        forward = hits.direction > 0
        weight = np.abs(hits.current[forward])
        r = np.hypot(hits.point[1, forward], hits.point[2, forward])
        launched = sum(
            np.pi * beam.current_density * beam.end[1] ** 2
            for beam in model.config.beams
            if isinstance(beam, BeamSpecies)
        )
        transmission = float(weight.sum() / launched)
        lost = max(0.0, 1.0 - transmission)
        lost = 0.0 if lost < 1e-9 else lost  # the rounding of the sum
        moments = model.beam_moments(self.target)
        fields = self.surface_fields(model)
        field = max(fields.values())
        stats = model.stats
        if r.size:
            order = np.argsort(r)
            share = np.cumsum(weight[order]) / weight.sum()
            r90 = float(r[order][np.searchsorted(share, 0.9)])
            r_max = float(r.max())
        else:
            r90 = r_max = math.nan
        return Evaluation(
            objectives={"spot": moments.rms_size},
            constraints={
                "surface_field": field / self.field_limit - 1.0,
                "loss": lost - self.max_loss,
            },
            converged=bool(stats["converged"]) or self.current == 0.0,
            diagnostics={
                "field": field,
                **{f"field_{name}": value for name, value in fields.items()},
                "transmission": transmission,
                "rms_radius": math.sqrt(2.0) * moments.rms_size,
                "r90": r90,
                "r_max": r_max,
                "iterations": stats["iterations"],
                "potential_change": model.convergence_history[-1][
                    "max_potential_change"
                ],
            },
        )

    @staticmethod
    def surface_fields(model: BeamOptics) -> dict[str, float]:
        """Return the largest surface field [V/m] of each tube (of all its objects)."""
        fields: dict[str, float] = dict.fromkeys(ELECTRODES, 0.0)
        for key, value in model.surface_fields().items():
            name = key.split(".")[0]
            fields[name] = max(fields[name], value)
        return fields

    def evaluate(self, config: BeamOpticsConfig) -> Evaluation:
        """Solve the lens of a configuration and return its :meth:`measure`."""
        return self.measure(BeamOptics(config).run())

    def design(self, bounds: Mapping[str, tuple[float, float]]) -> Design:
        """Return the design of the parameters in ``bounds`` (``{name: (lower, upper)}``)."""
        return Design(
            tuple(
                Parameter(name, lower, upper, PARAMETERS[name])
                for name, (lower, upper) in bounds.items()
            ),
            build=self.config,
        )

    def problem(
        self,
        bounds: Mapping[str, tuple[float, float]],
        constraints: tuple[str, ...] = ("loss",),
    ) -> Problem:
        """Return the problem of minimizing the spot over the parameters in ``bounds``.

        ``constraints`` are some of ``"loss"`` and ``"surface_field"``.
        """
        return Problem(
            self.design(bounds),
            self.evaluate,
            objectives=("spot",),
            constraints=constraints,
        )

    # -- the thin lens --------------------------------------------------------------------

    def axis_potential(
        self, values: Mapping[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``x`` [m] and the potential on the axis per volt on the middle tube.

        From a field solve at the given geometry (without space charge, the potential is
        proportional to the voltage).
        """
        v = self.values(values)
        reference = 1e3
        lens = replace(self, current=0.0, num_particles=1, save_trajectories=0)
        model = lens.solve({**v, "voltage": reference})
        return model.mesh.x, model.potential[:, 0] / reference

    def kinetic_potential(self, potential: np.ndarray) -> np.ndarray:
        """Return the ions' kinetic energy per charge, ``phi`` [V], at a potential [V]."""
        return self.energy / self.charge - np.asarray(potential)

    def thin_lens_focal_length(self, x: np.ndarray, potential: np.ndarray) -> float:
        """Return the thin-lens focal length [m] of the on-axis ``potential`` [V].

        ``1 / f = (3 / 16) int (phi' / phi)^2 dx`` (see the module docstring), with
        ``phi`` the kinetic energy per charge; infinite without a lens, ``NaN`` if an ion
        is stopped.
        """
        phi = self.kinetic_potential(potential)
        if np.any(phi <= 0.0):
            return math.nan
        power = 3.0 / 16.0 * trapezoid((np.gradient(phi, x) / phi) ** 2, x)
        return math.inf if power == 0.0 else 1.0 / power

    def _branch(self, unit: np.ndarray) -> tuple[float, float]:
        """Return the sign of the voltages to search and the largest magnitude [V].

        The sign is that of the default ``voltage``. A lens that decelerates the ions
        (``charge * voltage > 0``) stops them on the axis where ``charge * V(x)``
        reaches the energy.
        """
        sign = 1.0 if self.voltage >= 0.0 else -1.0
        if sign * self.charge > 0.0:
            peak = float(np.max(np.abs(unit)))
            return sign, 0.999 * self.energy / (abs(self.charge) * peak)
        return sign, 1e3 * self.energy / abs(self.charge)

    def thin_lens_voltage(self, values: Mapping[str, float] | None = None) -> float:
        """Return the voltage whose thin lens at the middle tube focuses on the target.

        The voltage has the sign of the lens default ``voltage`` (for positive ions,
        positive decelerates them and negative accelerates them), at the geometry of
        ``values``. Raises :class:`ValueError` if no voltage of that sign focuses there.
        """
        x, unit = self.axis_potential(values)
        distance = self.target - self.centre
        sign, upper = self._branch(unit)

        def mismatch(magnitude: float) -> float:
            focal = self.thin_lens_focal_length(x, sign * magnitude * unit)
            return 1.0 / focal - 1.0 / distance

        if mismatch(upper) < 0.0:
            raise ValueError("no voltage of this sign focuses on the target")
        return sign * cast(
            float, brentq(mismatch, 1e-9 * upper, upper, xtol=1e-6 * upper)
        )

    def paraxial_focus(self, x: np.ndarray, potential: np.ndarray) -> float:
        """Return where a paraxial ray, parallel at the start, crosses the axis [m].

        The Picht equation integrated through the on-axis ``potential`` [V] (a thick
        lens, without aberrations); a ray still away from the axis at the end of the
        domain is continued straight (the field is zero there). ``inf`` if it diverges.
        """
        phi = self.kinetic_potential(potential)
        if np.any(phi <= 0.0):
            return math.nan
        strength = 3.0 / 16.0 * (np.gradient(phi, x) / phi) ** 2

        def rhs(z: float, y: np.ndarray) -> list[float]:
            return [y[1], -float(np.interp(z, x, strength)) * y[0]]

        def crossing(z: float, y: np.ndarray) -> float:
            return y[0]

        crossing.terminal = True  # type: ignore[attr-defined]
        crossing.direction = -1  # type: ignore[attr-defined]
        span = (float(x[0]), float(x[-1]))
        solution = solve_ivp(
            rhs,
            span,
            [1.0, 0.0],
            events=crossing,
            max_step=float(x[1] - x[0]),
            rtol=1e-9,
        )
        if solution.t_events[0].size:
            return float(solution.t_events[0][0])
        height, slope = solution.y[:, -1]
        return float(span[1] - height / slope) if slope < 0.0 else math.inf

    def paraxial_voltage(self, values: Mapping[str, float] | None = None) -> float:
        """Return the voltage whose paraxial (thick-lens) focus is on the target.

        Like :meth:`thin_lens_voltage`, but integrating the paraxial ray through the
        lens (:meth:`paraxial_focus`).
        """
        x, unit = self.axis_potential(values)
        sign, upper = self._branch(unit)

        def mismatch(magnitude: float) -> float:
            focus = self.paraxial_focus(x, sign * magnitude * unit)
            return (focus if math.isfinite(focus) else 1e3 * self.length) - self.target

        if mismatch(upper) > 0.0:
            raise ValueError("no voltage of this sign focuses on the target")
        return sign * cast(
            float, brentq(mismatch, 1e-3 * upper, upper, xtol=1e-6 * upper)
        )
