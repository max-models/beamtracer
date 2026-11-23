"""The configuration of a beam: `BeamOpticsConfig` and its sections.

The sections are the beams, the plasma, the solenoids, a field map, the space-charge
compensation and the charge changes; the grid, the walls and the electrodes are in
`beamtracer.sections`. All values are SI, except charges in elementary charges,
masses in atomic mass units (u) and energies and temperatures in eV. Every physical
setting also takes an astropy Quantity (``5 * u.keV``) or a string astropy parses
(``"5 keV"``, as in a configuration file), converted to these units when the setting
is made (`plasmacoll.units`); temperatures convert between K and eV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from plasmacoll.units import (
    convert_units,
    dataclass_to_dict,
    to_unit,
    unit_field,
    unit_metadata,
    with_unit,
)

from beamtracer.field_map import OUTSIDE, FieldMap
from beamtracer.sections import (
    OBJECT_TYPES_3D,
    ConfigDict,
    FieldBCs,
    GridParams,
    ObjectParams,
    plain_value,
)


@dataclass(frozen=True)
class BeamSpecies:
    """A beam emitted from a line with a given energy.

    ``num_particles`` trajectories start evenly spaced on the segment from ``start`` to
    ``end`` (``(x, y)`` or ``(x, r)`` [m]) with ``energy`` [eV] perpendicular to it and
    Gaussian spreads ``parallel_temperature`` / ``transverse_temperature`` [eV].
    ``current_density`` [A/m^2], ``charge`` [e], ``mass`` [u]. The physical settings
    take Quantities or strings too (``energy="5 keV"``, ``mass=2.014 * u.u``).
    """

    name: str
    current_density: float = field(metadata=unit_metadata("A/m2"))
    start: tuple[float, float] = field(metadata=unit_metadata("m"))
    end: tuple[float, float] = field(metadata=unit_metadata("m"))
    charge: float = 1.0
    mass: float = unit_field("u", 1.007276)
    energy: float = unit_field("eV", 5.0)
    parallel_temperature: float = unit_field("eV", 0.0)
    transverse_temperature: float = unit_field("eV", 0.5)
    num_particles: int = 1000

    def __post_init__(self) -> None:
        """Convert the units and check the values."""
        convert_units(self)
        if len(self.start) != 2 or len(self.end) != 2:
            raise ValueError("start and end are (x, y) points")
        if self.num_particles < 1:
            raise ValueError("num_particles must be at least 1")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.energy <= 0.0:
            raise ValueError("energy must be positive")


@dataclass(frozen=True)
class BeamSpecies3D:
    """A beam emitted from a surface of a 3D grid, perpendicular to ``x``.

    ``num_particles`` trajectories start on the ``shape`` centred at ``centre``
    ``(x, y, z)`` [m] in the plane ``x = centre[0]``: a ``"rectangle"`` of side
    lengths ``size`` ``(along y, along z)`` [m] on a grid of cells of equal area, or a
    ``"disc"`` of radius ``size[0]`` on a sunflower pattern of equal areas. They move
    along ``+x`` with ``energy`` [eV] and Gaussian spreads ``parallel_temperature`` /
    ``transverse_temperature`` [eV] (as IBSimu's ``add_rectangular_beam_with_energy`` and
    ``add_cylindrical_beam_with_energy``), each carrying ``current_density`` [A/m^2] times
    its share of the area. ``charge`` [e], ``mass`` [u].
    """

    name: str
    current_density: float = field(metadata=unit_metadata("A/m2"))
    centre: tuple[float, float, float] = field(metadata=unit_metadata("m"))
    size: tuple[float, ...] = field(metadata=unit_metadata("m"))
    shape: str = "rectangle"
    charge: float = 1.0
    mass: float = unit_field("u", 1.007276)
    energy: float = unit_field("eV", 5.0)
    parallel_temperature: float = unit_field("eV", 0.0)
    transverse_temperature: float = unit_field("eV", 0.5)
    num_particles: int = 1000

    def __post_init__(self) -> None:
        """Convert the units and check the values."""
        convert_units(self)
        object.__setattr__(self, "centre", tuple(float(c) for c in self.centre))
        size = np.atleast_1d(np.asarray(self.size, dtype=float))
        object.__setattr__(self, "size", tuple(size.tolist()))
        if len(self.centre) != 3:
            raise ValueError("centre is an (x, y, z) point")
        if self.shape not in ("rectangle", "disc"):
            raise ValueError(f"shape must be 'rectangle' or 'disc', got {self.shape!r}")
        if len(self.size) != (2 if self.shape == "rectangle" else 1):
            raise ValueError(
                "size is (along y, along z) for a rectangle, (radius,) for a disc"
            )
        if min(self.size) <= 0.0:
            raise ValueError("size must be positive")
        if self.num_particles < 1:
            raise ValueError("num_particles must be at least 1")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.energy <= 0.0:
            raise ValueError("energy must be positive")

    @property
    def area(self) -> float:
        """The emitting area [m^2]."""
        if self.shape == "disc":
            return float(np.pi * self.size[0] ** 2)
        return float(self.size[0] * self.size[1])


@dataclass(frozen=True)
class ParticleBeam:
    """A beam given trajectory by trajectory, e.g. handed over from another simulation.

    ``position`` and ``velocity`` are ``(N, 3)`` Cartesian [m, m/s] (in cylindrical geometry
    ``x`` is the axis), ``current`` [A] per trajectory, one ``charge`` [e] and ``mass`` [u].
    The arrays and the mass take Quantities too (``position=points * u.mm``).
    """

    name: str
    position: tuple = field(metadata=unit_metadata("m", array=True))
    velocity: tuple = field(metadata=unit_metadata("m/s", array=True))
    current: tuple = field(metadata=unit_metadata("A", array=True))
    charge: float = 1.0
    mass: float = unit_field("u", 1.007276)

    def __post_init__(self) -> None:
        """Convert the units and store the arrays as nested tuples."""
        import numpy as np

        convert_units(self)
        position = np.asarray(self.position, dtype=float).reshape(-1, 3)
        velocity = np.asarray(self.velocity, dtype=float).reshape(-1, 3)
        current = np.asarray(self.current, dtype=float).ravel()
        if not position.shape[0] == velocity.shape[0] == current.size:
            raise ValueError(
                "position, velocity and current need one row per trajectory"
            )
        object.__setattr__(self, "position", tuple(map(tuple, position)))
        object.__setattr__(self, "velocity", tuple(map(tuple, velocity)))
        object.__setattr__(self, "current", tuple(current))

    def __len__(self) -> int:
        """Return the number of trajectories."""
        return len(self.current)


@dataclass(frozen=True)
class ExtractionPlasma:
    """Boltzmann plasma electrons for positive-ion extraction.

    In the first iteration the free nodes with ``x < initial_plasma_x`` [m] are held at
    ``plasma_potential`` [V]; later the electron charge density is
    ``-rho_e exp((phi - plasma_potential) / electron_temperature)`` [eV], with ``rho_e``
    [C/m^3] of ``None`` the ion charge density of the beams at their start
    (``sum J / v``). Each takes a Quantity or a string too (``"2 eV"``, ``"20000 K"``).
    """

    electron_temperature: float = field(metadata=unit_metadata("eV"))
    plasma_potential: float = field(metadata=unit_metadata("V"))
    initial_plasma_x: float = field(metadata=unit_metadata("m"))
    rho_e: float | None = unit_field("C/m3", None)

    def __post_init__(self) -> None:
        """Convert the units."""
        convert_units(self)


@dataclass(frozen=True)
class NegativeExtractionPlasma:
    """Boltzmann plasma ions for negative-ion extraction.

    The negative ions and the electrons extracted with them are the beams, traced; the
    positive ions, repelled by the extraction field, have the charge density
    ``rho_p exp(-(phi - plasma_potential) / ion_temperature)`` [eV], with ``rho_p``
    [C/m^3] of ``None`` the negative charge density of the beams at their start
    (``sum J / v``). In the first iteration the free nodes with ``x < initial_plasma_x``
    [m] are held at ``plasma_potential`` [V]. Each takes a Quantity or a string too.
    """

    ion_temperature: float = field(metadata=unit_metadata("eV"))
    plasma_potential: float = field(metadata=unit_metadata("V"))
    initial_plasma_x: float = field(metadata=unit_metadata("m"))
    rho_p: float | None = unit_field("C/m3", None)

    def __post_init__(self) -> None:
        """Convert the units and check the temperature."""
        convert_units(self)
        if not self.ion_temperature > 0.0:
            raise ValueError(
                f"ion_temperature must be positive, got {self.ion_temperature}"
            )


def _entry(value: Any, name: str) -> Any:
    """Return ``value[name]`` of a mapping, else the attribute ``name``."""
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


@dataclass(frozen=True)
class ChargeChangingParams:
    """Charge changes of the beam on targets (gas, plasma), e.g. in a neutralizer.

    ``cross_sections`` are ``{target: {from_state: {to_state: sigma}}}`` (m^2),
    ``densities`` the target densities ``{target: n}`` (m^-3), uniform or a profile
    along ``x`` ``{"x": [...], "density": [...]}`` (e.g. the gas between the grids
    of an accelerator, `beamtracer.solver.charge_changing.conductance_profile`),
    ``charges`` the charge of every state in e, ``ionization``
    ``{target: {state: sigma}}`` cross sections for ionizing the targets, tallied
    only. In ``mode="fractions"`` every trajectory carries the fractions of its
    particles in each state; in ``mode="analog"`` it is in one state that changes
    at sampled events, and an event splits it into ``splitting`` trajectories
    for its first ``split_events`` events, with random numbers from ``seed``; see
    `beamtracer.solver.charge_changing`. The cross sections, densities and ``x`` take
    Quantities or strings too (``"1e-20 m2"``, ``"1e19 m-3"``).
    """

    cross_sections: dict
    densities: dict
    charges: dict = None  # type: ignore[assignment]
    ionization: dict = None  # type: ignore[assignment]
    mode: str = "fractions"
    splitting: int = 1
    split_events: int = 1
    seed: int | None = 0

    def __post_init__(self) -> None:
        """Fill in the defaults and copy the mappings."""
        charges = self.charges or {"-": -1.0, "0": 0.0, "+": 1.0}
        object.__setattr__(
            self, "charges", {str(k): float(v) for k, v in charges.items()}
        )
        object.__setattr__(
            self,
            "cross_sections",
            {
                str(t): {
                    str(a): {str(b): float(v) for b, v in row.items()}
                    for a, row in table.items()
                }
                for t, table in to_unit(
                    self.cross_sections, "m2", "cross_sections"
                ).items()
            },
        )
        object.__setattr__(
            self,
            "densities",
            {
                str(t): (
                    {
                        "x": list(to_unit(_entry(n, "x"), "m", f"densities[{t!r}].x")),
                        "density": list(
                            to_unit(
                                _entry(n, "density"), "m-3", f"densities[{t!r}].density"
                            )
                        ),
                    }
                    if isinstance(n, Mapping) or hasattr(n, "density")
                    else to_unit(n, "m-3", f"densities[{t!r}]")
                )
                for t, n in self.densities.items()
            },
        )
        object.__setattr__(
            self,
            "ionization",
            {
                str(t): {str(k): float(v) for k, v in row.items()}
                for t, row in to_unit(self.ionization or {}, "m2", "ionization").items()
            },
        )

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form (cross sections in m^2, densities in m^-3)."""
        return {
            "cross_sections": with_unit(self.cross_sections, "m2"),
            "densities": {
                t: (
                    {
                        "x": with_unit(n["x"], "m"),
                        "density": with_unit(n["density"], "m-3"),
                    }
                    if isinstance(n, dict)
                    else with_unit(n, "m-3")
                )
                for t, n in self.densities.items()
            },
            "charges": dict(self.charges),
            "ionization": with_unit(self.ionization, "m2"),
            "mode": self.mode,
            "splitting": self.splitting,
            "split_events": self.split_events,
            "seed": self.seed,
        }

    def build(self):
        """The `beamtracer.solver.charge_changing.ChargeChanging` of these settings."""
        from plasmacoll import ChargeStateTransitions

        from beamtracer.solver.charge_changing import AxialProfile, ChargeChanging

        transitions = ChargeStateTransitions(
            {
                target: {
                    (a, b): sigma
                    for a, row in table.items()
                    for b, sigma in row.items()
                }
                for target, table in self.cross_sections.items()
            },
            states=tuple(self.charges),
        )
        densities = {
            t: AxialProfile(n["x"], n["density"]) if isinstance(n, dict) else n
            for t, n in self.densities.items()
        }
        return ChargeChanging(
            transitions,
            densities,
            self.charges,
            self.ionization,
            mode=self.mode,
            splitting=self.splitting,
            split_events=self.split_events,
            seed=self.seed,
        )


@dataclass(frozen=True)
class Solenoid:
    """A thin-walled solenoid on the axis.

    ``centre``, ``length``, winding ``radius`` [m] and on-axis ``peak`` field [T]; each
    takes a Quantity or a string too (``"50 mm"``, ``"0.1 T"``).
    """

    centre: float = field(metadata=unit_metadata("m"))
    length: float = field(metadata=unit_metadata("m"))
    radius: float = field(metadata=unit_metadata("m"))
    peak: float = field(metadata=unit_metadata("T"))

    def __post_init__(self) -> None:
        """Convert the units."""
        convert_units(self)


@dataclass(frozen=True)
class AxialField:
    """A measured on-axis field ``bz`` [T] sampled at ``z0 + dz * k`` [m].

    Each takes a Quantity or a string too (``bz=samples * u.mT``).
    """

    z0: float = field(metadata=unit_metadata("m"))
    dz: float = field(metadata=unit_metadata("m"))
    bz: tuple = field(metadata=unit_metadata("T"))

    def __post_init__(self) -> None:
        """Convert the units; store the samples as a tuple."""
        convert_units(self)


@dataclass(frozen=True)
class FieldMapParams:
    """An imposed E and/or B from a field map file (`beamtracer.field_map`).

    ``file`` is a netCDF field map in the shared layout (as picnic writes them), its
    ``frame`` that of the grid (``"cartesian"`` planar, ``"cylindrical"`` for
    ``(x, r)``) and its dims in ``(x, y)``. Its ``e_field_*`` add to the solved
    field, its ``b_field_*`` to the solenoids; ``outside`` is what the field is beyond
    the map: ``"clip"`` (the boundary value), ``"zero"`` or ``"error"``.
    """

    file: str
    outside: str = "clip"

    def __post_init__(self) -> None:
        """Store the file as a string and check ``outside``."""
        object.__setattr__(self, "file", str(self.file))
        if self.outside not in OUTSIDE:
            raise ValueError(f"outside must be one of {OUTSIDE}, got {self.outside!r}")

    def read(self) -> FieldMap:
        """Read the map."""
        return FieldMap.read(self.file, outside=self.outside)


@dataclass(frozen=True)
class AdaptiveRefinementParams:
    """Refined boxes that the Vlasov iteration places itself where the field changes fastest.

    Every ``every`` iterations the cells near the beam across which the field strength
    changes by more than ``threshold`` times its largest value are tagged, grown by
    ``buffer`` cells and covered with boxes (at least ``efficiency`` full of tagged cells,
    at least ``min_size`` cells long), up to ``max_level`` levels; the field is then solved
    and the beam traced on the new boxes (`beamtracer.solver.adaptive`).
    """

    max_level: int = 1
    threshold: float = 0.02
    every: int = 4
    #: The mesh is adapted at most this many times (at iterations ``every``, ``2 every``, ...).
    regrids: int = 2
    buffer: int = 2
    efficiency: float = 0.5
    min_size: int = 4

    def __post_init__(self) -> None:
        """Check the settings (as `beamtracer.solver.adaptive.AdaptiveRefinement` does)."""
        from beamtracer.solver.adaptive import AdaptiveRefinement

        AdaptiveRefinement(**dataclass_to_dict(self))


@dataclass(frozen=True)
class RefinementParams:
    """A box refined to ``level`` (cells ``2**level`` times smaller than the grid's).

    ``lower`` and ``upper`` are corners ``(x, y)`` [m] (``(x, r)`` in cylindrical
    geometry, ``(x, y, z)`` on a 3D grid), widened to the cells of the level below; a
    level's boxes must lie in those of the level below. Each takes Quantities or strings too (``("1 mm", "0 mm")``).
    """

    level: int
    lower: tuple[float, ...] = field(metadata=unit_metadata("m"))
    upper: tuple[float, ...] = field(metadata=unit_metadata("m"))

    def __post_init__(self) -> None:
        """Convert the units and check the box."""
        convert_units(self)
        if int(self.level) != self.level or self.level < 1:
            raise ValueError(f"level must be a positive integer, got {self.level}")
        object.__setattr__(self, "level", int(self.level))
        lower, upper = tuple(map(float, self.lower)), tuple(map(float, self.upper))
        if (
            len(lower) not in (2, 3)
            or len(upper) != len(lower)
            or not all(a < b for a, b in zip(lower, upper, strict=True))
        ):
            raise ValueError(f"a refined box needs lower < upper, got {lower}, {upper}")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class SpaceChargeCompensation:
    """The beam's space charge is reduced by ``fraction`` between ``x_min`` and ``x_max``.

    ``x_min`` and ``x_max`` in m, or Quantities or strings such as ``"10 mm"``.
    """

    x_min: float = field(metadata=unit_metadata("m"))
    x_max: float = field(metadata=unit_metadata("m"))
    fraction: float

    def __post_init__(self) -> None:
        """Convert the units and check the fraction."""
        convert_units(self)
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")


def beam_from_value(value: Any) -> BeamSpecies | BeamSpecies3D | ParticleBeam:
    """Return a beam from a dataclass or its mapping (its ``type`` tells which).

    ``"line"`` (`BeamSpecies`, the default), ``"surface"`` (`BeamSpecies3D`) or
    ``"particles"`` (`ParticleBeam`).
    """
    if isinstance(value, (BeamSpecies, BeamSpecies3D, ParticleBeam)):
        return value
    params = dict(value)
    kind = params.pop("type", "line")
    classes = {"line": BeamSpecies, "surface": BeamSpecies3D, "particles": ParticleBeam}
    if kind not in classes:
        raise ValueError(f"beam type must be one of {sorted(classes)}, got {kind!r}")
    return classes[kind](**params)


def beam_to_dict(beam: BeamSpecies | BeamSpecies3D | ParticleBeam) -> ConfigDict:
    """Return the configuration-file form of a beam."""
    params = dataclass_to_dict(beam)
    if isinstance(beam, ParticleBeam):
        params["type"] = "particles"
    elif isinstance(beam, BeamSpecies3D):
        params["type"] = "surface"
    else:
        params["type"] = "line"
    return params


def plasma_from_value(
    value: Any,
) -> ExtractionPlasma | NegativeExtractionPlasma | None:
    """Return a plasma from a dataclass or its mapping.

    Its ``type``, ``"positive"`` or ``"negative"``, tells which ions are extracted.
    """
    if value is None or isinstance(value, (ExtractionPlasma, NegativeExtractionPlasma)):
        return value
    params = dict(value)
    kind = params.pop("type", "positive")
    if kind not in ("positive", "negative"):
        raise ValueError(f"plasma type must be 'positive' or 'negative', got {kind!r}")
    return (NegativeExtractionPlasma if kind == "negative" else ExtractionPlasma)(
        **params
    )


def plasma_to_dict(
    plasma: ExtractionPlasma | NegativeExtractionPlasma | None,
) -> ConfigDict | None:
    """Return the configuration-file form of a plasma."""
    if plasma is None:
        return None
    params = dataclass_to_dict(plasma)
    params["type"] = (
        "negative" if isinstance(plasma, NegativeExtractionPlasma) else "positive"
    )
    return params


def coerce(cls: type, value: Any) -> Any:
    """Return ``value`` as ``cls`` (from a mapping), or ``None``."""
    if value is None or isinstance(value, cls):
        return value
    return cls(**value)


def _section(cls: type, value: Any) -> Any:
    """Return ``value`` as ``cls``, reading it with ``cls.from_dict`` from a mapping."""
    if value is None or isinstance(value, cls):
        return value
    return cls.from_dict(value)


@dataclass(kw_only=True)
class BeamOpticsConfig:
    """Steady ion beams in a self-consistent field.

    The ``grid`` is 2D with equal node spacing on both axes, ``coordinates="cartesian"``
    (planar, per unit depth) or ``"cylindrical"`` (``(x, r)`` with the axis at ``r = 0``).
    ``objects`` are electrodes with Dirichlet potentials and ``bcs_field`` the faces
    (Dirichlet or Neumann; in cylindrical geometry the ``y`` lower face is the axis);
    ``None`` is grounded faces.

    ``beams`` are `BeamSpecies` (emitted from a line) or `ParticleBeam`
    (given trajectory by trajectory). ``plasma`` adds the Boltzmann plasma species: electrons
    for positive-ion extraction (`ExtractionPlasma`), positive ions for negative-ion
    extraction (`NegativeExtractionPlasma`, the negative ions and electrons being
    beams), ``solenoids`` and ``axial_field`` an axisymmetric magnetic field expanded
    off the axis to ``field_order``, ``field_map`` an imposed E and/or B from a field
    map file (`FieldMapParams`, a path or its mapping), ``compensation`` reduces the beam's space charge in
    regions of ``x``. ``charge_changing`` lets the beam change charge state on
    uniform targets (a gas, a plasma), every trajectory carrying the fractions of its
    particles in each state. The Vlasov iteration runs ``iterations`` times or until the
    largest potential change is below ``tolerance`` [V]; ``relaxation`` mixes the new
    charge density with the previous one. Each trajectory moves ``step_fraction`` of
    the step estimate ``min(h / |v|, sqrt(2 h / |a|))`` per Runge--Kutta step, for at most
    ``max_steps`` steps. ``integrator`` is ``"rk4"`` (the classical Runge--Kutta
    method, four field evaluations a step) or ``"boris"`` (Boris's kick--drift--kick,
    second order with one evaluation a step). ``relativistic`` traces by the relativistic
    equations of motion (the proper velocity, mpiparticles' relativistic Boris step) and seeds
    the beams with the relativistic speed of their energy. ``mixing="anderson"`` replaces the
    relaxation by Anderson's acceleration over ``anderson_depth`` previous iterations;
    ``coarse_particles < 1`` traces that share of the trajectories until the potential
    changes by less than ``coarse_tolerance`` [V], then all of them (see
    `beamtracer.solver.vlasov_iterate`). ``field_backend`` solves the field with
    patchwork's ``"lu"`` (every MPI rank the whole mesh), ``"petsc"`` (the ranks share
    it: Krylov with algebraic multigrid, needs petsc4py), ``"amg"`` (every rank, algebraic
    multigrid preconditioning a Krylov method, no factorisation; with a plasma a
    Newton--Krylov method keeping its preconditioner; needs pyamg), ``"multigrid"``
    (every rank, patchwork's geometric multigrid with a compiled Gauss--Seidel V-cycle;
    with a plasma every Newton step preconditioned by its own Jacobian, which the
    multigrid takes as a diagonal update; the fastest for 3D and large meshes) or
    ``"auto"``. ``refinement``
    lists boxes solved and traced on finer cells (`RefinementParams`): the field on
    patchwork's composite mesh, each trajectory stepping and depositing on the finest box
    it is in; ``adaptive_refinement`` (`AdaptiveRefinementParams`) places such boxes
    during the iteration, where the field changes fastest near the beam. ``symmetry_plane`` makes a planar grid half of a geometry mirrored at its
    ``y`` lower face, which needs a zero normal derivative there: a trajectory crossing
    it continues at negative ``y`` in the mirrored field (as IBSimu's mirrored
    boundaries), instead of leaving.
    ``self_magnetic_field`` adds the beam's own magnetic field, from its current by
    Ampère's law (`beamtracer.solver.self_field`; in 2D with the NumPy tracer).
    ``diagnostic_planes`` [m] record the trajectories crossing ``x = const``;
    ``save_trajectories`` keeps every n-th trajectory for plotting (0: none).
    ``tolerance`` and ``diagnostic_planes`` take Quantities or strings too, like the
    settings of the sections.
    """

    #: The ``type`` of the configuration-file form.
    model_name: ClassVar[str] = "BeamOptics"

    grid: GridParams = field(default_factory=GridParams)
    objects: dict[str, ObjectParams] = field(default_factory=dict)
    bcs_field: FieldBCs | None = None
    beams: Sequence[BeamSpecies | BeamSpecies3D | ParticleBeam] = field(
        default_factory=list
    )
    plasma: ExtractionPlasma | NegativeExtractionPlasma | None = None
    solenoids: list[Solenoid] = field(default_factory=list)
    axial_field: AxialField | None = None
    field_order: int = 3
    field_map: FieldMapParams | None = None
    compensation: list[SpaceChargeCompensation] = field(default_factory=list)
    charge_changing: ChargeChangingParams | None = None
    iterations: int = 20
    tolerance: float = unit_field("V", 1.0e-3)
    relaxation: float = 1.0
    step_fraction: float = 0.5
    max_steps: int = 100_000
    integrator: str = "rk4"
    relativistic: bool = False
    mixing: str = "linear"
    anderson_depth: int = 5
    coarse_particles: float = 1.0
    coarse_tolerance: float = unit_field("V", 0.0)
    field_backend: str = "lu"
    refinement: list[RefinementParams] = field(default_factory=list)
    adaptive_refinement: AdaptiveRefinementParams | None = None
    symmetry_plane: bool = False
    symmetry_planes: list[str] = field(default_factory=list)
    self_magnetic_field: bool = False
    diagnostic_planes: list[float] = unit_field("m", default_factory=list)
    save_trajectories: int = 0

    def _check_dimension(self) -> None:
        """Check that the beams, objects and settings fit the grid's dimension."""
        three = self.grid.ndim == 3
        for name, obj in self.objects.items():
            if any((s.type in OBJECT_TYPES_3D) != three for s in obj.shape_list):
                raise ValueError(
                    f"object {name!r} has a {'2D' if three else '3D'} shape on a "
                    f"{self.grid.ndim}D grid"
                )
        for beam in self.beams:
            if isinstance(beam, BeamSpecies) and three:
                raise ValueError("a 3D grid takes BeamSpecies3D, not line beams")
            if isinstance(beam, BeamSpecies3D) and not three:
                raise ValueError("BeamSpecies3D needs a 3D grid")
        if not three:
            if any(len(box.lower) != 2 for box in self.refinement):
                raise ValueError("a refined box of a 2D grid has (x, y) corners")
            if self.symmetry_planes:
                raise ValueError(
                    "symmetry_planes is for a 3D grid (2D: symmetry_plane)"
                )
            return
        unsupported = {
            "charge_changing": self.charge_changing,
            "symmetry_plane": self.symmetry_plane,
        }
        given = sorted(name for name, value in unsupported.items() if value)
        if given:
            raise ValueError(f"a 3D grid does not take {', '.join(given)} yet")
        for box in self.refinement:
            if len(box.lower) != 3:
                raise ValueError(
                    f"a refined box of a 3D grid needs (x, y, z) corners: {box}"
                )
        for plane in self.symmetry_planes:
            if plane not in ("y_min", "z_min"):
                raise ValueError(
                    f"a symmetry plane is 'y_min' or 'z_min', got {plane!r}"
                )

    def __post_init__(self) -> None:
        """Convert the units, check the grid and read the sections."""
        convert_units(self)
        self.grid = _section(GridParams, self.grid)
        self.objects = {
            name: _section(ObjectParams, obj)
            for name, obj in (self.objects or {}).items()
        }
        three = self.grid.ndim == 3
        if self.bcs_field is None:
            self.bcs_field = FieldBCs.grounded(self.grid.ndim)
        else:
            self.bcs_field = _section(FieldBCs, self.bcs_field)
        assert self.bcs_field is not None
        if self.bcs_field.has_z != three:
            raise ValueError("a 3D grid needs the z faces in bcs_field, a 2D one none")
        if self.grid.coordinates not in ("cartesian", "cylindrical"):
            raise ValueError(
                f"BeamOptics needs cartesian or cylindrical coordinates, got "
                f"{self.grid.coordinates!r}"
            )
        # a graded x axis has its own spacing; the others are equal
        deltas = self.grid.deltas[1:] if self.grid.graded else self.grid.deltas
        if max(deltas) - min(deltas) > 1e-9 * max(deltas):
            raise ValueError(f"BeamOptics needs equal node spacing, got {deltas}")
        self.beams = [beam_from_value(beam) for beam in self.beams]
        self.plasma = plasma_from_value(self.plasma)
        self.solenoids = [coerce(Solenoid, s) for s in self.solenoids]
        self.axial_field = coerce(AxialField, self.axial_field)
        if isinstance(self.field_map, (str, Path)):
            self.field_map = FieldMapParams(str(self.field_map))
        self.field_map = coerce(FieldMapParams, self.field_map)
        self.compensation = [
            coerce(SpaceChargeCompensation, c) for c in self.compensation
        ]
        self.refinement = [coerce(RefinementParams, r) for r in self.refinement]
        self.adaptive_refinement = coerce(
            AdaptiveRefinementParams, self.adaptive_refinement
        )
        if self.grid.graded and self.adaptive_refinement is not None:
            raise ValueError(
                "adaptive refinement needs a uniform grid: use refinement boxes with "
                "x_spacing"
            )
        self.charge_changing = coerce(ChargeChangingParams, self.charge_changing)
        self.diagnostic_planes = list(self.diagnostic_planes)
        if (
            (self.solenoids or self.axial_field)
            and self.grid.coordinates != "cylindrical"
            and self.grid.ndim != 3
        ):
            raise ValueError(
                "solenoids need cylindrical coordinates (or a 3D grid, on the x axis)"
            )
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError(f"relaxation must be in (0, 1], got {self.relaxation}")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.mixing not in ("linear", "anderson"):
            raise ValueError(
                f"mixing must be 'linear' or 'anderson', got {self.mixing!r}"
            )
        if self.anderson_depth < 1:
            raise ValueError("anderson_depth must be at least 1")
        if not 0.0 < self.coarse_particles <= 1.0:
            raise ValueError(
                f"coarse_particles must be in (0, 1], got {self.coarse_particles}"
            )
        if self.coarse_particles < 1.0 and not self.coarse_tolerance > 0.0:
            raise ValueError("coarse_particles < 1 needs a positive coarse_tolerance")
        if self.field_backend not in ("lu", "petsc", "amg", "multigrid", "auto"):
            raise ValueError(
                "field_backend must be 'lu', 'petsc', 'amg', 'multigrid' or 'auto', got "
                f"{self.field_backend!r}"
            )
        if self.integrator not in ("rk4", "boris"):
            raise ValueError(
                f"integrator must be 'rk4' or 'boris', got {self.integrator!r}"
            )
        if self.symmetry_plane and self.grid.coordinates == "cylindrical":
            raise ValueError("symmetry_plane needs a planar (cartesian) grid")
        self.symmetry_planes = list(self.symmetry_planes)
        self._check_dimension()

    def replace(self, **changes: Any) -> BeamOpticsConfig:
        """Return a copy with ``changes`` applied (and checked again)."""
        return replace(self, **changes)

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form, which `from_dict` reads."""
        params: ConfigDict = {"type": self.model_name}
        for entry in fields(self):
            value = getattr(self, entry.name)
            if value is None:
                continue
            unit = entry.metadata.get("unit")
            params[entry.name] = value if unit is None else with_unit(value, unit)
        params["grid"] = self.grid.to_dict()
        params["objects"] = {name: obj.to_dict() for name, obj in self.objects.items()}
        params["bcs_field"] = self.bcs_field.to_dict()  # type: ignore[union-attr]
        params["beams"] = [beam_to_dict(beam) for beam in self.beams]
        params["plasma"] = plasma_to_dict(self.plasma)
        params["solenoids"] = [dataclass_to_dict(s) for s in self.solenoids]
        params["axial_field"] = (
            None if self.axial_field is None else dataclass_to_dict(self.axial_field)
        )
        params["field_map"] = (
            None if self.field_map is None else dataclass_to_dict(self.field_map)
        )
        params["compensation"] = [dataclass_to_dict(c) for c in self.compensation]
        params["refinement"] = [dataclass_to_dict(r) for r in self.refinement]
        params["adaptive_refinement"] = (
            None
            if self.adaptive_refinement is None
            else dataclass_to_dict(self.adaptive_refinement)
        )
        params["charge_changing"] = (
            None if self.charge_changing is None else self.charge_changing.to_dict()
        )
        return plain_value(params)

    @classmethod
    def from_dict(cls, params: ConfigDict) -> BeamOpticsConfig:
        """Create the configuration from its configuration-file form.

        ``type``, when given, must be ``"BeamOptics"``. Unknown entries are an error, so a
        misspelt setting is not silently ignored; picnic's ``random`` and ``solver``
        entries, which the beam tracer does not use, are accepted and ignored.
        """
        params = dict(params)
        model_name = params.pop("type", cls.model_name)
        if model_name != cls.model_name:
            raise ValueError(
                f"{cls.__name__} configures {cls.model_name!r}, not {model_name!r}"
            )
        for ignored in ("random", "solver"):
            params.pop(ignored, None)
        known = {entry.name for entry in fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise KeyError(
                f"unknown {cls.model_name} settings {sorted(unknown)}; known: {sorted(known)}"
            )
        return cls(**params)
