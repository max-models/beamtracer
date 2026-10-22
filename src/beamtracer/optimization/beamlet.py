r"""A negative-ion beamlet: the extraction of H$^-$ as a problem of :mod:`beamtracer.optimization`.

One round beamlet in cylindrical ``(x, r)`` geometry, solved with
:class:`~beamtracer.BeamOptics` and :class:`~beamtracer.NegativeExtractionPlasma`: the
H$^-$ ions and the electrons extracted with them are traced from the plasma, the plasma's
positive ions follow a Boltzmann distribution and form the meniscus, and a Vlasov
iteration converges the field and the trajectories together (see the example
``negative_ion_extraction_optics``, whose beamlet is this one's default).

Three grids, each a plate around the axis from its aperture radius outwards: the plasma
grid (PG, 0 V) with a chamfer opening towards the extraction gap, the extraction grid
(EG, at ``extraction_voltage``) and the grounded grid (GG, at ``grounded_grid_voltage``,
the beam's final energy). ``gap`` separates the PG from the EG and ``acceleration_gap``
the EG from the GG; the domain ends ``drift`` behind the GG, where the beam is measured.
With ``rounded`` the aperture edges of the EG and the GG are half circles (a plate and a
circle each, named ``eg`` and ``eg.edge``): a sharp corner has a singular surface field,
which grows as the mesh is refined. The PG's chamfer ends in a knife edge whose field is
singular too; it is reported, not constrained.

The objective is the rms divergence of the H$^-$ behind the GG (``rms_angle`` of
:meth:`~beamtracer.BeamOptics.beam_moments`). The constraints, ``<= 0`` when met:

- ``loss``: the lost fraction of the emitted H$^-$ current minus ``max_loss``;
- ``surface_field``: the largest field on the EG and GG surfaces over ``field_limit``,
  minus 1;
- ``gap_field``: the larger mean field of the two gaps (the voltage across a gap over
  its length) over ``gap_field_limit``, minus 1: voltage holding, with
  ``surface_field``;
- ``divergence_limit``: the divergence over ``divergence_limit`` (7 mrad), minus 1.

The H$^-$ current density is an input of the steady model (``current_density``, about
300 A/m²): the source sets it, and a design must extract it at that perveance; the
extracted current density (the transmitted current over the PG aperture) is a
diagnostic. Where the beam grazes the EG the Vlasov iteration can settle into a limit
cycle of a few volts instead of converging: such an evaluation has ``converged=False``,
and the searches rank it behind every converged one; its diagnostics keep the spread of
the potential changes over the last iterations.

:class:`NegativeIonBeamlet` holds the fixed settings and makes the
:class:`~beamtracer.optimization.Design` and :class:`~beamtracer.optimization.Problem`
of any of :data:`PARAMETERS`: a few of them, or the full design (:data:`FULL_DESIGN`,
with the constraints :data:`FULL_CONSTRAINTS`). A design that cannot be built (an
aperture that does not fit, a gap the mesh does not resolve) is a failed evaluation.
:data:`COARSE` is the mesh and the number of trajectories of a search, compared with the
example's in the tutorial "Optimising a negative-ion beamlet".

A search on a coarse mesh is followed by a check of its best designs with a better
model: :func:`~beamtracer.optimization.recheck` evaluates them again with another
function of the same beamlet, e.g. :meth:`NegativeIonBeamlet.evaluate` of a beamlet on
a finer mesh, or a PIC model of the source written in user code, which takes the
:class:`NegativeIonBeamlet` of a design (its geometry and voltages) and returns an
:class:`~beamtracer.optimization.Evaluation`; beamtracer does not import the PIC code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from scipy import constants

from beamtracer.config import (
    BeamOpticsConfig,
    BeamSpecies,
    NegativeExtractionPlasma,
    ParticleBeam,
)
from beamtracer.model import BeamOptics
from beamtracer.optimization.design import Design, Evaluation, Parameter, Problem
from beamtracer.sections import (
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ObjectParams,
)
from beamtracer.solver import STATUS_NAMES, BeamMoments, VlasovState, plane_moments

#: The design parameters of the beamlet and their units: the voltage split (the EG
#: voltage of the total on the GG), the aperture radii, the grid thicknesses and the gaps.
PARAMETERS = {
    "extraction_voltage": "V",
    "eg_radius": "m",
    "gap": "m",
    "pg_radius": "m",
    "gg_radius": "m",
    "pg_thickness": "m",
    "eg_thickness": "m",
    "gg_thickness": "m",
    "acceleration_gap": "m",
}

#: The bounds of the full design, all of :data:`PARAMETERS`; some corners do not fit in
#: the domain radius (the half pitch of the beamlets) and are failed evaluations.
FULL_DESIGN = {
    "extraction_voltage": (3e3, 10e3),
    "eg_radius": (2e-3, 3.5e-3),
    "gap": (3e-3, 6e-3),
    "pg_radius": (3e-3, 4.5e-3),
    "gg_radius": (2.5e-3, 4e-3),
    "pg_thickness": (0.5e-3, 1.5e-3),
    "eg_thickness": (1e-3, 2.5e-3),
    "gg_thickness": (1e-3, 2.5e-3),
    "acceleration_gap": (5e-3, 10e-3),
}

#: The constraints of the full design: transmission and voltage holding.
FULL_CONSTRAINTS = ("loss", "surface_field", "gap_field")

#: The grids, in order along the beam; a rounded grid's edge circle is ``<name>.edge``.
GRIDS = ("pg", "eg", "gg")

#: The mass of H- [u].
H_MINUS_MASS = 1.00849

#: The iterations whose potential changes make the spread of an unconverged solve.
SPREAD_ITERATIONS = 5


@dataclass(frozen=True)
class NegativeIonBeamlet:
    """The fixed settings of a round H- beamlet, and its problems.

    Lengths in m, voltages in V. The grids: aperture radii ``pg_radius``, ``eg_radius``
    and ``gg_radius``, thicknesses ``pg_thickness``, ``eg_thickness`` and
    ``gg_thickness``; the PG's upstream face at ``pg_front``, the gaps ``gap`` (PG to EG)
    and ``acceleration_gap`` (EG to GG), and ``drift`` behind the GG to the end of the
    domain, which reaches ``radius`` from the axis. ``chamfer`` opens the PG aperture
    towards the EG at 45 degrees over its thickness; ``rounded`` rounds the aperture
    edges of the EG and the GG.

    The beams: H- of ``current_density`` [A/m^2] and ``electron_ratio`` times as much
    electron current, emitted from the plasma side (``x = 0``) up to
    ``emission_margin`` inside the PG aperture, with starting energies and transverse
    temperatures [eV]; the plasma: ``plasma_potential`` [V] and Boltzmann ions of
    ``ion_temperature`` [eV]. Numerics: nodes every ``cell``, ``num_particles``
    trajectories per species, at most ``iterations`` Vlasov iterations to a potential
    change below ``tolerance`` [V], with ``relaxation`` of the charge density.

    The limits of the constraints: ``max_loss`` (fraction of the H- current),
    ``field_limit`` [V/m] (on the surfaces), ``gap_field_limit`` [V/m] (the mean
    field of a gap) and ``divergence_limit`` [rad]. A frozen dataclass, so it
    pickles: its :meth:`config` and :meth:`evaluate` are the build and evaluation
    functions of its problems, also in a process pool.
    """

    pg_radius: float = 4.0e-3
    eg_radius: float = 3.0e-3
    gg_radius: float = 3.5e-3
    pg_thickness: float = 1.0e-3
    eg_thickness: float = 2.0e-3
    gg_thickness: float = 2.0e-3
    pg_front: float = 1.0e-3
    gap: float = 4.0e-3
    acceleration_gap: float = 8.0e-3
    drift: float = 6.0e-3
    radius: float = 6.0e-3
    chamfer: bool = True
    rounded: bool = True
    extraction_voltage: float = 5.0e3
    grounded_grid_voltage: float = 20.0e3
    plasma_potential: float = -1.0
    current_density: float = 300.0
    electron_ratio: float = 1.0
    h_minus_energy: float = 1.0
    h_minus_temperature: float = 1.0
    electron_energy: float = 1.0
    electron_temperature: float = 2.0
    ion_temperature: float = 1.0
    emission_margin: float = 0.2e-3
    cell: float = 0.1e-3
    num_particles: int = 300
    iterations: int = 40
    tolerance: float = 1.0e-2
    relaxation: float = 0.5
    max_loss: float = 0.01
    field_limit: float = 5.0e6
    gap_field_limit: float = 3.0e6
    divergence_limit: float = 7.0e-3
    save_trajectories: int = 0

    # -- the geometry ---------------------------------------------------------------------

    @property
    def pg(self) -> tuple[float, float]:
        """Return the ``(front, back)`` of the PG along the axis [m]."""
        return (self.pg_front, self.pg_front + self.pg_thickness)

    @property
    def eg(self) -> tuple[float, float]:
        """Return the ``(front, back)`` of the EG [m]."""
        front = self.pg[1] + self.gap
        return (front, front + self.eg_thickness)

    @property
    def gg(self) -> tuple[float, float]:
        """Return the ``(front, back)`` of the GG [m]."""
        front = self.eg[1] + self.acceleration_gap
        return (front, front + self.gg_thickness)

    @property
    def length(self) -> float:
        """Return the length of the domain [m]."""
        return self.gg[1] + self.drift

    @property
    def exit_plane(self) -> float:
        """Return the plane behind the GG where the beam is measured [m]."""
        return self.length - 0.5 * self.drift

    @property
    def emission_radius(self) -> float:
        """Return the radius the beams are emitted to [m]."""
        return self.pg_radius - self.emission_margin

    @property
    def emitted_current(self) -> float:
        """Return the emitted H- current [A]."""
        return math.pi * self.emission_radius**2 * self.current_density

    @property
    def child_langmuir(self) -> float:
        """Return the Child-Langmuir current density [A/m^2] of H- over the PG-EG gap.

        The gap from the PG's upstream face to the EG's, as in the example.
        """
        gap = self.eg[0] - self.pg[0]
        return (
            4.0 * constants.epsilon_0 / 9.0
            * math.sqrt(2.0 * constants.e / (H_MINUS_MASS * constants.atomic_mass))
            * self.extraction_voltage**1.5 / gap**2
        )  # fmt: skip

    def values(self, values: Mapping[str, float] | None = None) -> dict[str, float]:
        """Return all :data:`PARAMETERS`: ``values``, the defaults for the others."""
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

    def at(self, values: Mapping[str, float] | None = None) -> NegativeIonBeamlet:
        """Return the beamlet with the parameter ``values`` (some of :data:`PARAMETERS`).

        The build function of its designs: the evaluation takes the beamlet, which
        knows its geometry, and solves its :meth:`config`. Raises :class:`ValueError`
        for a beamlet that cannot be built (see :meth:`check`).
        """
        case = replace(self, **self.values(values))
        case.check()
        return case

    def check(self) -> None:
        """Raise :class:`ValueError` for a beamlet that cannot be built."""
        names = {f.name for f in fields(self)}
        for name in sorted(names):
            value = getattr(self, name)
            if name.endswith(("_radius", "_thickness", "gap", "drift")) and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.emission_radius <= 0.0:
            raise ValueError("the emission must start inside the PG aperture")
        reach = {
            "pg": self.pg_radius + (self.pg_thickness if self.chamfer else 0.0),
            "eg": self.eg_radius + (self.eg_thickness if self.rounded else 0.0),
            "gg": self.gg_radius + (self.gg_thickness if self.rounded else 0.0),
        }
        for name, r in reach.items():
            if r >= self.radius:
                raise ValueError(
                    f"the {name.upper()} aperture must fit inside the domain radius"
                )
        if self.cell > 0.5 * min(self.gap, self.acceleration_gap):
            raise ValueError("the mesh must resolve the gaps by at least two cells")
        if self.cell > 0.5 * min(
            self.pg_thickness, self.eg_thickness, self.gg_thickness
        ):
            raise ValueError("the mesh must resolve the grids by at least two cells")
        if not 0.0 < self.extraction_voltage < self.grounded_grid_voltage:
            raise ValueError(
                "the extraction voltage must lie between 0 and the GG voltage"
            )

    @property
    def gap_fields(self) -> dict[str, float]:
        """Return the mean field [V/m] of the extraction and the acceleration gap."""
        return {
            "extraction": abs(self.extraction_voltage) / self.gap,
            "acceleration": abs(self.grounded_grid_voltage - self.extraction_voltage)
            / self.acceleration_gap,
        }

    def _plate(
        self, name: str, span: tuple[float, float], aperture: float, voltage: float
    ) -> dict[str, ObjectParams]:
        """Return the objects of a grid plate around the axis, from its aperture out."""
        x0, x1 = span
        outer = self.radius + self.cell  # the plates reach past the outer face

        def plate(vertices: tuple[tuple[float, float], ...]) -> ObjectParams:
            return ObjectParams(
                type="aperture2d",
                axis=0,  # type: ignore[arg-type]
                position=0.0,
                vertices=vertices,
                bc_type="dirichlet",
                bc_value=float(voltage),
            )

        if name == "pg":
            inner = aperture + (x1 - x0) if self.chamfer else aperture
            return {
                name: plate(((x0, aperture), (x0, outer), (x1, outer), (x1, inner)))
            }
        if not self.rounded:
            return {
                name: plate(((x0, aperture), (x0, outer), (x1, outer), (x1, aperture)))
            }
        rho = 0.5 * (x1 - x0)
        edge = aperture + rho
        return {
            name: plate(((x0, edge), (x0, outer), (x1, outer), (x1, edge))),
            f"{name}.edge": ObjectParams(
                type="circle",
                bc_type="dirichlet",
                bc_value=float(voltage),
                midpoint=(float(x0 + rho), float(edge)),
                radius=float(rho),
            ),
        }

    def config(self, values: Mapping[str, float] | None = None) -> BeamOpticsConfig:
        """Return the :class:`~beamtracer.BeamOpticsConfig` of the beamlet at ``values``.

        Raises :class:`ValueError` for a beamlet that cannot be built (see :meth:`check`).
        """
        case = self.at(values)
        neumann = FieldBoundaryCondition.neumann(0.0)
        nx = round(case.length / case.cell) + 1
        ny = round(case.radius / case.cell) + 1
        h_minus = BeamSpecies(
            "H-",
            case.current_density,
            (0.0, 0.0),
            (0.0, case.emission_radius),
            charge=-1.0,
            mass=H_MINUS_MASS,
            energy=case.h_minus_energy,
            transverse_temperature=case.h_minus_temperature,
            num_particles=case.num_particles,
        )
        beams: list[BeamSpecies | ParticleBeam] = [h_minus]
        if case.electron_ratio > 0.0:
            beams.append(
                replace(
                    h_minus,
                    name="e",
                    mass=constants.m_e / constants.atomic_mass,
                    current_density=case.electron_ratio * case.current_density,
                    energy=case.electron_energy,
                    transverse_temperature=case.electron_temperature,
                )
            )
        voltages = (0.0, case.extraction_voltage, case.grounded_grid_voltage)
        radii = (case.pg_radius, case.eg_radius, case.gg_radius)
        spans = (case.pg, case.eg, case.gg)
        objects: dict[str, ObjectParams] = {}
        for name, span, aperture, voltage in zip(
            GRIDS, spans, radii, voltages, strict=True
        ):
            objects.update(case._plate(name, span, aperture, voltage))
        return BeamOpticsConfig(
            grid=GridParams(
                num_gridpoints=(nx, ny),
                upper_bound=((nx - 1) * case.cell, (ny - 1) * case.cell),
                coordinates="cylindrical",
            ),
            bcs_field=FieldBCs(
                x_lower=FieldBoundaryCondition.dirichlet(case.plasma_potential),
                x_upper=FieldBoundaryCondition.dirichlet(case.grounded_grid_voltage),
                y_lower=neumann,
                y_upper=neumann,
            ),
            objects=objects,
            beams=beams,
            plasma=NegativeExtractionPlasma(
                case.ion_temperature, case.plasma_potential, case.pg[0]
            ),
            iterations=case.iterations,
            tolerance=case.tolerance,
            relaxation=case.relaxation,
            diagnostic_planes=[case.exit_plane],
            save_trajectories=case.save_trajectories,
        )

    def solve(
        self,
        values: Mapping[str, float] | None = None,
        iterates: list[BeamMoments] | None = None,
    ) -> BeamOptics:
        """Return the solved :class:`~beamtracer.BeamOptics` of the beamlet.

        With a list ``iterates``, the H- moments at the exit plane of every Vlasov
        iteration are appended to it.
        """
        model = BeamOptics(self.config(values))
        if iterates is None:
            return model.run()

        def track(iteration: int, state: VlasovState) -> None:
            iterates.append(
                plane_moments(
                    state.trace.planes[0],
                    cylindrical=True,
                    mass=model.species_mass,
                    species=0,
                )
            )

        return model.run(callback=track)

    # -- the evaluation -------------------------------------------------------------------

    @staticmethod
    def surface_fields(model: BeamOptics) -> dict[str, float]:
        """Return the largest surface field [V/m] of each grid (of all its objects)."""
        result: dict[str, float] = dict.fromkeys(GRIDS, 0.0)
        for key, value in model.surface_fields().items():
            name = key.split(".")[0]
            result[name] = max(result[name], value)
        return result

    @staticmethod
    def h_minus_losses(model: BeamOptics) -> dict[str, float]:
        """Return where the H- trajectories that are not transmitted end.

        ``lost_pg``, ``lost_eg`` and ``lost_gg``: the fraction of the H- trajectories
        (rings, not current) that end on each grid; ``returned``: the fraction turned
        back towards the plasma (at the edge of the meniscus).
        """
        trace = model.state.trace
        species = np.repeat(
            np.arange(len(model.config.beams)),
            [beam.num_particles for beam in model.config.beams],  # type: ignore[union-attr]
        )
        h_minus = species == 0
        count = max(int(h_minus.sum()), 1)
        names = [key.split(".")[0] for key in model.config.objects]
        ends = trace.electrode[h_minus]
        losses = {
            f"lost_{name}": sum(names[k] == name for k in ends if k >= 0) / count
            for name in GRIDS
        }
        backward = STATUS_NAMES.index("backward")
        losses["returned"] = float(np.sum(trace.status[h_minus] == backward)) / count
        return losses

    def measure(
        self, model: BeamOptics, iterates: Sequence[BeamMoments] = ()
    ) -> Evaluation:
        """Return the :class:`~beamtracer.optimization.Evaluation` of a solved beamlet.

        The model must be that of this beamlet's :meth:`config`; ``iterates`` are the H-
        moments of its iterations (see :meth:`solve`). The objective ``divergence`` is
        the rms angle [rad] of the H- at the exit plane; the constraints are ``loss``,
        ``surface_field``, ``gap_field`` and ``divergence_limit`` (see the module
        docstring). The diagnostics hold the transmission, the extracted current density
        [A/m^2], the emittance [m rad], the mean energy [eV], the perveance over its
        Child-Langmuir value, the surface field of each grid and the mean field of each
        gap [V/m], where the lost H- end (:meth:`h_minus_losses`), the electrons'
        transmission, the Vlasov iterations, the largest potential change [V] of the
        last iteration and over the last :data:`SPREAD_ITERATIONS`, and over those (with
        ``iterates``) the spread of the divergence [rad] and of the transmission.
        """
        h_minus = model.beam_moments(self.exit_plane, "H-")
        transmission = h_minus.current / self.emitted_current
        lost = max(0.0, 1.0 - transmission)
        lost = 0.0 if lost < 1e-9 else lost  # the rounding of the sum
        grids = self.surface_fields(model)
        field = max(grids["eg"], grids["gg"])
        stats = model.stats
        changes = [h["max_potential_change"] for h in model.convergence_history[1:]]
        diagnostics: dict[str, Any] = {
            "transmission": transmission,
            "current_density": h_minus.current / (math.pi * self.pg_radius**2),
            "emittance": h_minus.emittance,
            "normalized_emittance": h_minus.normalized_emittance,
            "energy": h_minus.mean_energy,
            "rms_size": h_minus.rms_size,
            "perveance_ratio": self.current_density / self.child_langmuir,
            "field": field,
            **{f"field_{name}": value for name, value in grids.items()},
            **{f"gap_field_{name}": v for name, v in self.gap_fields.items()},
            "iterations": stats["iterations"],
            "potential_change": changes[-1] if changes else math.nan,
            "potential_spread": max(changes[-SPREAD_ITERATIONS:])
            if changes
            else math.nan,
        }
        diagnostics.update(self.h_minus_losses(model))
        last = list(iterates)[-SPREAD_ITERATIONS:]
        if last:
            angles = [m.rms_angle for m in last]
            currents = [m.current / self.emitted_current for m in last]
            diagnostics["divergence_spread"] = max(angles) - min(angles)
            diagnostics["transmission_spread"] = max(currents) - min(currents)
        if "e" in model.species_names:
            electrons = model.beam_moments(self.exit_plane, "e")
            diagnostics["electron_transmission"] = electrons.current / (
                self.electron_ratio * self.emitted_current
            )
        divergence = h_minus.rms_angle if h_minus.count else math.nan
        return Evaluation(
            objectives={"divergence": divergence},
            constraints={
                "loss": lost - self.max_loss,
                "surface_field": field / self.field_limit - 1.0,
                "gap_field": max(self.gap_fields.values()) / self.gap_field_limit - 1.0,
                "divergence_limit": divergence / self.divergence_limit - 1.0,
            },
            converged=bool(stats["converged"]),
            diagnostics=diagnostics,
        )

    def evaluate(self, case: NegativeIonBeamlet) -> Evaluation:
        """Solve a beamlet (this one at some parameters, see :meth:`at`) and measure it."""
        iterates: list[BeamMoments] = []
        model = case.solve(iterates=iterates)
        return case.measure(model, iterates)

    def design(self, bounds: Mapping[str, tuple[float, float]]) -> Design:
        """Return the design of the parameters in ``bounds`` (``{name: (lower, upper)}``)."""
        return Design(
            tuple(
                Parameter(name, lower, upper, PARAMETERS[name])
                for name, (lower, upper) in bounds.items()
            ),
            build=self.at,
        )

    def problem(
        self,
        bounds: Mapping[str, tuple[float, float]],
        constraints: tuple[str, ...] = ("loss", "surface_field"),
    ) -> Problem:
        """Return the problem of minimizing the divergence over the parameters in ``bounds``.

        ``constraints`` are some of ``"loss"``, ``"surface_field"``, ``"gap_field"``
        and ``"divergence_limit"``; the others are kept as diagnostics. The full
        design is ``problem(FULL_DESIGN, FULL_CONSTRAINTS)``.
        """
        return Problem(
            self.design(bounds),
            self.evaluate,
            objectives=("divergence",),
            constraints=constraints,
        )


#: The settings of a search, against the example's 0.1 mm mesh and 300 trajectories per
#: species: two to three times faster, within 2 % of the fine divergence where both
#: converge with the whole beam transmitted, but 19 % below it at the optimum of the
#: tutorial's search, next to the designs that do not converge (see the tutorial
#: "Optimising a negative-ion beamlet").
COARSE = {"cell": 0.2e-3, "num_particles": 100}
