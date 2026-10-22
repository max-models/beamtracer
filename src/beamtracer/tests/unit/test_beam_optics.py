"""Tests for the BeamOptics model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import constants

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    ExtractionPlasma,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    NegativeExtractionPlasma,
    ObjectParams,
    ParticleBeam,
    Solenoid,
    SpaceChargeCompensation,
)


def _diode(
    *, current_density: float, num_gridpoints=(41, 21), **settings
) -> BeamOpticsConfig:
    """A planar diode: ions from x = 0 (at 0 V) to x = 10 mm (at -1 kV)."""
    neumann = FieldBoundaryCondition.neumann(0.0)
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=num_gridpoints,
            lower_bound=(0.0, 0.0),
            upper_bound=(10e-3, 5e-3),
            coordinates="cartesian",
        ),
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(0.0),
            x_upper=FieldBoundaryCondition.dirichlet(-1000.0),
            y_lower=neumann,
            y_upper=neumann,
        ),
        beams=[
            BeamSpecies(
                "H+",
                current_density,
                (0.0, 0.0),
                (0.0, 5e-3),
                energy=0.01,
                transverse_temperature=0.0,
                num_particles=40,
            )
        ],
        diagnostic_planes=[9.9e-3],
        **settings,
    )


def test_beam_optics_rejects_other_configurations() -> None:
    with pytest.raises(TypeError, match="BeamOpticsConfig"):
        BeamOptics(GridParams(ndim=2))  # type: ignore[arg-type]


def test_config_needs_equal_spacing_and_cylindrical_solenoids() -> None:
    with pytest.raises(ValueError, match="equal node spacing"):
        BeamOpticsConfig(
            grid=GridParams(ndim=2, num_gridpoints=(11, 21), upper_bound=(1.0, 1.0))
        )
    with pytest.raises(ValueError, match="cylindrical"):
        BeamOpticsConfig(
            grid=GridParams(ndim=2, num_gridpoints=(11, 11)),
            solenoids=[Solenoid(0.0, 0.1, 0.05, 0.1)],
        )


def test_config_sections_round_trip() -> None:
    beam = ParticleBeam("H+", [[0.0, 1e-3, 0.0]], [[1e5, 0.0, 0.0]], [1e-6])
    config = BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(11, 6),
            upper_bound=(1e-2, 5e-3),
            coordinates="cylindrical",
        ),
        beams=[BeamSpecies("H+", 100.0, (0, 0), (0, 1e-3)), beam],
        plasma=ExtractionPlasma(5.0, 100.0, 1e-3),
        solenoids=[Solenoid(0.0, 0.1, 0.05, 0.1)],
        compensation=[SpaceChargeCompensation(0.0, 1.0, 0.9)],
    )
    again = BeamOpticsConfig.from_dict(config.to_dict())
    for name in ("beams", "plasma", "solenoids", "compensation"):
        assert getattr(again, name) == getattr(config, name)

    negative = config.replace(plasma=NegativeExtractionPlasma(2.0, -5.0, 1e-3))
    assert config.to_dict()["plasma"]["type"] == "positive"
    assert negative.to_dict()["plasma"]["type"] == "negative"
    assert BeamOpticsConfig.from_dict(negative.to_dict()).plasma == negative.plasma
    with pytest.raises(ValueError, match="plasma type"):
        config.replace(plasma={"type": "neutral", "electron_temperature": 1.0})
    with pytest.raises(ValueError, match="ion_temperature"):
        NegativeExtractionPlasma(0.0, 0.0, 1e-3)


def test_child_langmuir_diode() -> None:
    """At the Child--Langmuir current density, phi = -V (x/d)^(4/3) between the plates."""
    d, voltage = 10e-3, 1000.0
    mass = 1.007276 * constants.atomic_mass
    j_cl = (
        4
        * constants.epsilon_0
        / 9
        * np.sqrt(2 * constants.e / mass)
        * voltage**1.5
        / d**2
    )
    config = _diode(
        current_density=0.95 * j_cl, iterations=60, relaxation=0.5, tolerance=1e-3
    )
    model = BeamOptics(config)
    model.run()
    assert model.stats["converged"]
    assert model.stats["transmitted"] == 40
    x = model.mesh.x
    phi = model.potential[:, 10]
    inside = (x > 1e-3) & (x < 9e-3)
    # space charge pushes the potential above the vacuum line towards (x/d)^(4/3)
    assert np.all(phi[inside] > -voltage * x[inside] / d)
    cl = -voltage * (x[inside] / d) ** (4 / 3)
    assert np.max(np.abs(phi[inside] - cl)) < 0.06 * voltage
    moments = model.beam_moments(9.9e-3)
    assert moments.current == pytest.approx(0.95 * j_cl * 5e-3, rel=1e-6)


def test_compensated_beam_feels_less_space_charge() -> None:
    def potential_bump(fraction):
        config = _diode(current_density=2.0, iterations=3).replace(
            compensation=[SpaceChargeCompensation(0.0, 1.0, fraction)]
        )
        model = BeamOptics(config)
        model.run()
        return np.max(model.potential[:, 10] + 1000.0 * model.mesh.x / 10e-3)

    assert potential_bump(0.9) == pytest.approx(0.1 * potential_bump(0.0), rel=0.05)


def test_fields_before_and_after_the_run() -> None:
    model = BeamOptics(_diode(current_density=1.0, iterations=1))
    assert np.all(model.potential_field.get_field_data(component=0) == 0.0)
    model.run()
    assert model.is_finished and model.iteration == 1
    phi = model.potential_field.get_field_data(component=0)
    assert phi.min() == pytest.approx(-1000.0)
    assert model.charge_density.get_field_data(component=0).max() > 0.0
    ex, _ = model.electric_field_at(np.array([5e-3]), np.array([2e-3]))
    assert ex[0] == pytest.approx(1e5, rel=0.05)


def test_electrodes_intercept_the_beam() -> None:
    plate = ObjectParams(
        type="rectangle",
        bc_type="dirichlet",
        bc_value=-500.0,
        midpoint=(5e-3, 4e-3),
        side_lengths=(1e-3, 2e-3),
    )
    config = _diode(current_density=1.0, iterations=1).replace(objects={"plate": plate})
    model = BeamOptics(config)
    model.run()
    stats = model.stats
    assert stats["electrode"] > 0
    assert stats["electrode"] + stats["transmitted"] == stats["launched"]


def test_a_negative_beam_deposits_negative_space_charge() -> None:
    """The diode with H- and the potentials reversed: the potential exactly reversed."""
    positive = BeamOptics(_diode(current_density=10.0, iterations=5))
    neumann = FieldBoundaryCondition.neumann(0.0)
    config = _diode(current_density=10.0, iterations=5)
    config = config.replace(
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(0.0),
            x_upper=FieldBoundaryCondition.dirichlet(1000.0),
            y_lower=neumann,
            y_upper=neumann,
        ),
        beams=[replace(config.beams[0], name="H-", charge=-1.0)],
    )
    negative = BeamOptics(config)
    for model in (positive, negative):
        model.run()

    assert negative.state.charge_density.max() <= 0.0
    assert negative.potential == pytest.approx(-positive.potential, abs=1e-9)


def _plate(x0: float, x1: float, r0: float, r1: float, voltage: float) -> ObjectParams:
    """A plate around the axis from radius ``r0`` outwards, between ``x0`` and ``x1``."""
    return ObjectParams(
        type="aperture2d",
        axis=0,
        position=0.0,
        vertices=((x0, r0), (x0, r1), (x1, r1), (x1, r0)),
        bc_type="dirichlet",
        bc_value=voltage,
    )


def _extraction(
    sign: float, beams: list[BeamSpecies] | None = None
) -> BeamOpticsConfig:
    """Ions of charge ``sign`` extracted through a round aperture from a plasma at
    ``5 sign`` V, the plasma electrode at 0 V and the extraction electrode at ``-2 sign`` kV.
    """
    plasma_potential, voltage = 5.0 * sign, -2000.0 * sign
    neumann = FieldBoundaryCondition.neumann(0.0)
    plasma = (
        ExtractionPlasma(5.0, plasma_potential, 0.6e-3)
        if sign > 0
        else NegativeExtractionPlasma(5.0, plasma_potential, 0.6e-3)
    )
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(81, 31),
            lower_bound=(0.0, 0.0),
            upper_bound=(8e-3, 3e-3),
            coordinates="cylindrical",
        ),
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(plasma_potential),
            x_upper=FieldBoundaryCondition.dirichlet(voltage),
            y_lower=neumann,
            y_upper=neumann,
        ),
        objects={
            "pg": _plate(0.5e-3, 1.0e-3, 1.0e-3, 3.1e-3, 0.0),
            "eg": _plate(3.0e-3, 4.0e-3, 1.2e-3, 3.1e-3, voltage),
        },
        beams=beams
        or [
            BeamSpecies(
                "ion",
                50.0,
                (0.0, 0.0),
                (0.0, 0.9e-3),
                charge=sign,
                mass=1.0,
                energy=5.0,
                transverse_temperature=0.5,
                num_particles=100,
            )
        ],
        plasma=plasma,
        iterations=30,
        tolerance=1e-4,
        relaxation=0.8,
        diagnostic_planes=[7.5e-3],
    )


def test_negative_ion_extraction_mirrors_positive_ion_extraction() -> None:
    """Boltzmann ions about -U_p for negative ions are Boltzmann electrons about U_p."""
    models = [BeamOptics(_extraction(sign)) for sign in (1.0, -1.0)]
    for model in models:
        model.run()
    positive, negative = models

    assert negative.stats == positive.stats
    assert negative.stats["converged"] and negative.stats["transmitted"] == 100
    assert negative.potential == pytest.approx(-positive.potential, abs=1e-9)
    assert type(negative.state.plasma).__name__ == "BoltzmannIons"
    assert negative.state.plasma.rho_p == pytest.approx(positive.state.plasma.rho_e)
    moments = [model.beam_moments(7.5e-3) for model in models]
    assert moments[1].current == pytest.approx(moments[0].current)
    assert moments[1].emittance == pytest.approx(moments[0].emittance)


def test_the_plasma_ions_neutralise_the_negative_ions_and_electrons() -> None:
    h_minus = BeamSpecies(
        "H-",
        50.0,
        (0.0, 0.0),
        (0.0, 0.9e-3),
        charge=-1.0,
        mass=1.00849,
        energy=5.0,
        transverse_temperature=0.5,
        num_particles=100,
    )
    electrons = replace(
        h_minus,
        name="e",
        mass=constants.m_e / constants.atomic_mass,
        energy=1.0,
        transverse_temperature=1.0,
    )
    model = BeamOptics(_extraction(-1.0, [h_minus, electrons]))
    model.run()

    def charge_at_start(beam: BeamSpecies) -> float:
        mass = beam.mass * constants.atomic_mass
        return beam.current_density / np.sqrt(2 * beam.energy * constants.e / mass)

    assert model.stats["converged"]
    assert model.state.plasma.rho_p == pytest.approx(
        charge_at_start(h_minus) + charge_at_start(electrons)
    )
    # both species are extracted, the H- current is the emitted one
    assert model.beam_moments(7.5e-3, "e").current > 0.9 * np.pi * 0.9e-3**2 * 50.0
    assert model.beam_moments(7.5e-3, "H-").current == pytest.approx(
        np.pi * 0.9e-3**2 * 50.0, rel=1e-3
    )
    assert model.potential.max() == pytest.approx(2000.0)
