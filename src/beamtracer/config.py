"""The configuration of a beam: :class:`BeamOpticsConfig` and its sections.

The sections are the beams, the plasma, the solenoids, the space-charge compensation and
the charge changes; the grid, the walls and the electrodes are in
:mod:`beamtracer.sections`. All values are SI, except charges in elementary charges,
masses in atomic mass units and energies and temperatures in eV.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, ClassVar

from beamtracer.sections import (
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
    ``current_density`` [A/m^2], ``charge`` [e], ``mass`` [u].
    """

    name: str
    current_density: float
    start: tuple[float, float]
    end: tuple[float, float]
    charge: float = 1.0
    mass: float = 1.007276
    energy: float = 5.0
    parallel_temperature: float = 0.0
    transverse_temperature: float = 0.5
    num_particles: int = 1000

    def __post_init__(self) -> None:
        """Check the values."""
        object.__setattr__(self, "start", tuple(float(v) for v in self.start))
        object.__setattr__(self, "end", tuple(float(v) for v in self.end))
        if len(self.start) != 2 or len(self.end) != 2:
            raise ValueError("start and end are (x, y) points")
        if self.num_particles < 1:
            raise ValueError("num_particles must be at least 1")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.energy <= 0.0:
            raise ValueError("energy must be positive")


@dataclass(frozen=True)
class ParticleBeam:
    """A beam given trajectory by trajectory, e.g. handed over from another simulation.

    ``position`` and ``velocity`` are ``(N, 3)`` Cartesian [m, m/s] (in cylindrical geometry
    ``x`` is the axis), ``current`` [A] per trajectory, one ``charge`` [e] and ``mass`` [u].
    """

    name: str
    position: tuple
    velocity: tuple
    current: tuple
    charge: float = 1.0
    mass: float = 1.007276

    def __post_init__(self) -> None:
        """Store the arrays as nested tuples."""
        import numpy as np

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
    ``-rho_e exp((phi - plasma_potential) / electron_temperature)``, with ``rho_e`` of
    ``None`` the ion charge density of the beams at their start (``sum J / v``).
    """

    electron_temperature: float
    plasma_potential: float
    initial_plasma_x: float
    rho_e: float | None = None


@dataclass(frozen=True)
class NegativeExtractionPlasma:
    """Boltzmann plasma ions for negative-ion extraction.

    The negative ions and the electrons extracted with them are the beams, traced; the
    positive ions, repelled by the extraction field, have the charge density
    ``rho_p exp(-(phi - plasma_potential) / ion_temperature)``, with ``rho_p`` of ``None``
    the negative charge density of the beams at their start (``sum J / v``). In the first
    iteration the free nodes with ``x < initial_plasma_x`` [m] are held at
    ``plasma_potential`` [V].
    """

    ion_temperature: float
    plasma_potential: float
    initial_plasma_x: float
    rho_p: float | None = None

    def __post_init__(self) -> None:
        """Check the temperature."""
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
    of an accelerator, :func:`~beamtracer.solver.charge_changing.conductance_profile`),
    ``charges`` the charge of every state in e, ``ionization``
    ``{target: {state: sigma}}`` cross sections for ionizing the targets, tallied
    only. In ``mode="fractions"`` every trajectory carries the fractions of its
    particles in each state; in ``mode="analog"`` it is in one state that changes
    at sampled events, and an event splits it into ``splitting`` trajectories
    for its first ``split_events`` events, with random numbers from ``seed``; see
    :mod:`beamtracer.solver.charge_changing`.
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
                for t, table in self.cross_sections.items()
            },
        )
        object.__setattr__(
            self,
            "densities",
            {
                str(t): (
                    {
                        "x": [float(v) for v in _entry(n, "x")],
                        "density": [float(v) for v in _entry(n, "density")],
                    }
                    if isinstance(n, Mapping) or hasattr(n, "density")
                    else float(n)
                )
                for t, n in self.densities.items()
            },
        )
        object.__setattr__(
            self,
            "ionization",
            {
                str(t): {str(k): float(v) for k, v in row.items()}
                for t, row in (self.ionization or {}).items()
            },
        )

    def build(self):
        """The :class:`~beamtracer.solver.charge_changing.ChargeChanging` of these settings."""
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

    ``centre``, ``length``, winding ``radius`` [m] and on-axis ``peak`` field [T].
    """

    centre: float
    length: float
    radius: float
    peak: float


@dataclass(frozen=True)
class AxialField:
    """A measured on-axis field ``bz`` [T] sampled at ``z0 + dz * k`` [m]."""

    z0: float
    dz: float
    bz: tuple

    def __post_init__(self) -> None:
        """Store the samples as a tuple."""
        object.__setattr__(self, "bz", tuple(float(b) for b in self.bz))


@dataclass(frozen=True)
class SpaceChargeCompensation:
    """The beam's space charge is reduced by ``fraction`` between ``x_min`` and ``x_max``."""

    x_min: float
    x_max: float
    fraction: float

    def __post_init__(self) -> None:
        """Check the fraction."""
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")


def beam_from_value(value: Any) -> BeamSpecies | ParticleBeam:
    """Return a beam from a dataclass or its mapping (its ``type`` tells which)."""
    if isinstance(value, (BeamSpecies, ParticleBeam)):
        return value
    params = dict(value)
    kind = params.pop("type", "line")
    return ParticleBeam(**params) if kind == "particles" else BeamSpecies(**params)


def beam_to_dict(beam: BeamSpecies | ParticleBeam) -> ConfigDict:
    """Return the configuration-file form of a beam."""
    params = asdict(beam)
    params["type"] = "particles" if isinstance(beam, ParticleBeam) else "line"
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
    params = asdict(plasma)
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

    ``beams`` are :class:`BeamSpecies` (emitted from a line) or :class:`ParticleBeam`
    (given trajectory by trajectory). ``plasma`` adds the Boltzmann plasma species: electrons
    for positive-ion extraction (:class:`ExtractionPlasma`), positive ions for negative-ion
    extraction (:class:`NegativeExtractionPlasma`, the negative ions and electrons being
    beams), ``solenoids`` and ``axial_field`` an axisymmetric magnetic field expanded
    off the axis to ``field_order``, ``compensation`` reduces the beam's space charge in
    regions of ``x``. ``charge_changing`` lets the beam change charge state on
    uniform targets (a gas, a plasma), every trajectory carrying the fractions of its
    particles in each state. The Vlasov iteration runs ``iterations`` times or until the
    largest potential change is below ``tolerance`` [V]; ``relaxation`` mixes the new
    charge density with the previous one. Each trajectory moves ``step_fraction`` of
    the step estimate ``min(h / |v|, sqrt(2 h / |a|))`` per Runge--Kutta step, for at most
    ``max_steps`` steps.
    ``diagnostic_planes`` [m] record the trajectories crossing ``x = const``;
    ``save_trajectories`` keeps every n-th trajectory for plotting (0: none).
    """

    #: The ``type`` of the configuration-file form.
    model_name: ClassVar[str] = "BeamOptics"

    grid: GridParams = field(default_factory=GridParams)
    objects: dict[str, ObjectParams] = field(default_factory=dict)
    bcs_field: FieldBCs | None = None
    beams: list[BeamSpecies | ParticleBeam] = field(default_factory=list)
    plasma: ExtractionPlasma | NegativeExtractionPlasma | None = None
    solenoids: list[Solenoid] = field(default_factory=list)
    axial_field: AxialField | None = None
    field_order: int = 3
    compensation: list[SpaceChargeCompensation] = field(default_factory=list)
    charge_changing: ChargeChangingParams | None = None
    iterations: int = 20
    tolerance: float = 1.0e-3
    relaxation: float = 1.0
    step_fraction: float = 0.5
    max_steps: int = 100_000
    diagnostic_planes: list[float] = field(default_factory=list)
    save_trajectories: int = 0

    def __post_init__(self) -> None:
        """Check the grid and read the sections."""
        self.grid = _section(GridParams, self.grid)
        self.objects = {
            name: _section(ObjectParams, obj)
            for name, obj in (self.objects or {}).items()
        }
        if self.bcs_field is None:
            self.bcs_field = FieldBCs.grounded()
        else:
            self.bcs_field = _section(FieldBCs, self.bcs_field)
        if self.grid.coordinates not in ("cartesian", "cylindrical"):
            raise ValueError(
                f"BeamOptics needs cartesian or cylindrical coordinates, got "
                f"{self.grid.coordinates!r}"
            )
        dx, dy = self.grid.deltas
        if abs(dx - dy) > 1e-9 * max(dx, dy):
            raise ValueError(f"BeamOptics needs equal node spacing, got {dx} and {dy}")
        self.beams = [beam_from_value(beam) for beam in self.beams]
        self.plasma = plasma_from_value(self.plasma)
        self.solenoids = [coerce(Solenoid, s) for s in self.solenoids]
        self.axial_field = coerce(AxialField, self.axial_field)
        self.compensation = [
            coerce(SpaceChargeCompensation, c) for c in self.compensation
        ]
        self.charge_changing = coerce(ChargeChangingParams, self.charge_changing)
        self.diagnostic_planes = [float(x) for x in self.diagnostic_planes]
        if (
            self.solenoids or self.axial_field
        ) and self.grid.coordinates != "cylindrical":
            raise ValueError("solenoids need cylindrical coordinates")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError(f"relaxation must be in (0, 1], got {self.relaxation}")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")

    def replace(self, **changes: Any) -> BeamOpticsConfig:
        """Return a copy with ``changes`` applied (and checked again)."""
        return replace(self, **changes)

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form, which :meth:`from_dict` reads."""
        params: ConfigDict = {"type": self.model_name}
        for entry in fields(self):
            value = getattr(self, entry.name)
            if value is None:
                continue
            params[entry.name] = value
        params["grid"] = self.grid.to_dict()
        params["objects"] = {name: obj.to_dict() for name, obj in self.objects.items()}
        params["bcs_field"] = self.bcs_field.to_dict()  # type: ignore[union-attr]
        params["beams"] = [beam_to_dict(beam) for beam in self.beams]
        params["plasma"] = plasma_to_dict(self.plasma)
        params["solenoids"] = [asdict(s) for s in self.solenoids]
        params["axial_field"] = (
            None if self.axial_field is None else asdict(self.axial_field)
        )
        params["compensation"] = [asdict(c) for c in self.compensation]
        params["charge_changing"] = (
            None if self.charge_changing is None else asdict(self.charge_changing)
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
