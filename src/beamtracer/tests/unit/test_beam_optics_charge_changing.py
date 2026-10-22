"""BeamOptics with charge changes; the cross sections are synthetic."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import ChargeStateTransitions, charge_state_fractions
from scipy import constants

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    ChargeChangingParams,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
)
from beamtracer.solver.charge_changing import ChargeChanging

GAS = 1.0e20  # m^-3
LENGTH = 0.5
SIGMA = {"-": {"0": 1.2e-20, "+": 5e-22}, "0": {"+": 3.7e-21}}
IONIZATION = {"gas": {"-": 1e-20, "0": 8e-21, "+": 6e-21}}


def channel(current_density: float = 1.0e-6, **settings) -> BeamOpticsConfig:
    """A grounded planar channel: a 100 keV H- beam along x through a uniform gas."""
    grounded = FieldBoundaryCondition.dirichlet(0.0)
    defaults = {
        "grid": GridParams(
            ndim=2,
            num_gridpoints=(101, 11),
            lower_bound=(0.0, -0.025),
            upper_bound=(LENGTH, 0.025),
            coordinates="cartesian",
        ),
        "bcs_field": FieldBCs(
            x_lower=grounded, x_upper=grounded, y_lower=grounded, y_upper=grounded
        ),
        "beams": [
            BeamSpecies(
                "H-",
                current_density,
                (0.0, -0.01),
                (0.0, 0.01),
                charge=-1.0,
                mass=1.00849,
                energy=1.0e5,
                transverse_temperature=0.0,
                num_particles=20,
            )
        ],
        "charge_changing": ChargeChangingParams(
            {"gas": SIGMA}, {"gas": GAS}, ionization=IONIZATION
        ),
        "iterations": 2,
        "diagnostic_planes": [0.1, 0.3, LENGTH - 1e-3],
    }
    return BeamOpticsConfig(**{**defaults, **settings})


def expected(line_density: np.ndarray) -> np.ndarray:
    transitions = ChargeStateTransitions(
        {"gas": {(a, b): s for a, row in SIGMA.items() for b, s in row.items()}}
    )
    return charge_state_fractions(transitions, {"gas": line_density})


@pytest.fixture(scope="module")
def ballistic() -> BeamOptics:
    model = BeamOptics(channel())
    model.run()
    return model


def test_the_fractions_at_every_plane_are_plasmacolls_on_the_line_density(
    ballistic,
) -> None:
    for x in (0.1, 0.3, LENGTH - 1e-3):
        hits = ballistic.plane(x)
        assert hits.fractions.shape == (3, 20)
        reference = expected(np.array([0.0, GAS * x]))[-1]
        assert hits.fractions == pytest.approx(
            np.repeat(reference[:, None], 20, axis=1), abs=1e-6
        )
    assert ballistic.final_fractions.sum(axis=0) == pytest.approx(1.0)


def test_the_tallies_integrate_the_fractions_along_the_paths(ballistic) -> None:
    tallies = ballistic.charge_tallies
    rate = 1.0e-6 * 0.02 / constants.e  # particles per second per metre of depth
    x = np.linspace(0.0, LENGTH, 2001)
    integral = np.trapezoid(expected(GAS * x), x, axis=0)
    # the path is counted up to the exit face
    assert tallies.path / rate == pytest.approx(integral, rel=1e-4)
    stripped = tallies.changes[("gas", "-", "0")]
    assert stripped == pytest.approx(GAS * 1.2e-20 * tallies.path[0], rel=1e-12)
    ionized = tallies.ionization[("gas", "0")]
    assert ionized == pytest.approx(GAS * 8e-21 * tallies.path[1], rel=1e-12)


def test_each_state_carries_its_own_current(ballistic) -> None:
    x = LENGTH - 1e-3
    total = ballistic.beam_moments(x).current
    fraction = expected(np.array([0.0, GAS * x]))[-1]
    for k, state in enumerate(("-", "0", "+")):
        assert ballistic.state_moments(x, state).current == pytest.approx(
            total * fraction[k], rel=1e-6
        )


def test_the_space_charge_follows_the_charge_of_the_states() -> None:
    """A dense H- beam: its negative space charge fades as it is stripped."""
    model = BeamOptics(channel(current_density=1.0, iterations=1))
    model.run()
    rho = model.state.charge_density
    column = rho[:, 5]  # on the axis
    near, far = column[5], column[95]
    fraction = expected(np.array([0.0, GAS * 0.025, GAS * 0.475]))  # nodes 5, 95
    charge = fraction @ np.array([-1.0, 0.0, 1.0])
    assert near < 0.0
    assert far / near == pytest.approx(charge[2] / charge[1], rel=0.02)


def test_the_settings_round_trip_and_are_checked() -> None:
    config = channel()
    again = BeamOpticsConfig.from_dict(config.to_dict())
    assert again.charge_changing == config.charge_changing
    with pytest.raises(ValueError, match="unknown target"):
        ChargeChangingParams({"gas": SIGMA}, {"plasma": 1e17}).build()
    changing = config.charge_changing.build()
    assert isinstance(changing, ChargeChanging)
    with pytest.raises(ValueError, match="match no state"):
        changing.initial_fractions(np.array([2.0 * constants.e]))
