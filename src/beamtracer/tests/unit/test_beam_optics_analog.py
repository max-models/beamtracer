"""Trajectories that change state at sampled events, and target profiles; synthetic data."""

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
from beamtracer.solver.charge_changing import (
    AxialProfile,
    ChargeChanging,
    conductance_profile,
)

GAS = 1.0e20  # m^-3
LENGTH = 0.5
SIGMA = {"-": {"0": 1.2e-20, "+": 5e-22}, "0": {"+": 3.7e-21}, "+": {"0": 2e-21}}
PLANES = [0.1, 0.3, LENGTH - 1e-3]
TRANSITIONS = ChargeStateTransitions(
    {"gas": {(a, b): s for a, row in SIGMA.items() for b, s in row.items()}}
)


def channel(num: int = 4000, densities=None, **charge) -> BeamOpticsConfig:
    """A grounded planar channel: a 100 keV H- beam along x through a gas."""
    grounded = FieldBoundaryCondition.dirichlet(0.0)
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(101, 11),
            lower_bound=(0.0, -0.025),
            upper_bound=(LENGTH, 0.025),
            coordinates="cartesian",
        ),
        bcs_field=FieldBCs(
            x_lower=grounded, x_upper=grounded, y_lower=grounded, y_upper=grounded
        ),
        beams=[
            BeamSpecies(
                "H-",
                1.0e-6,
                (0.0, -0.01),
                (0.0, 0.01),
                charge=-1.0,
                mass=1.00849,
                energy=1.0e5,
                transverse_temperature=0.0,
                num_particles=num,
            )
        ],
        charge_changing=ChargeChangingParams(
            {"gas": SIGMA}, densities or {"gas": GAS}, **charge
        ),
        iterations=1,
        diagnostic_planes=PLANES,
    )


def shares(model: BeamOptics, x: float) -> np.ndarray:
    total = model.beam_moments(x).current
    return np.array(
        [model.state_moments(x, s).current / total for s in ("-", "0", "+")]
    )


@pytest.fixture(scope="module")
def fractions_mode() -> BeamOptics:
    return BeamOptics(channel(num=20)).run()


@pytest.fixture(scope="module")
def analog_mode() -> BeamOptics:
    return BeamOptics(channel(mode="analog", seed=7)).run()


def test_the_mean_over_trajectories_is_the_fractions_mode(
    fractions_mode, analog_mode
) -> None:
    num = 4000
    for x in PLANES:
        expected = shares(fractions_mode, x)
        found = shares(analog_mode, x)
        spread = np.sqrt(expected * (1.0 - expected) / num)
        assert np.all(np.abs(found - expected) < 4.0 * spread + 1e-12), x
        # every trajectory is in one state
        hits = analog_mode.plane(x).fractions
        assert set(np.unique(hits)) <= {0.0, 1.0}
    final = shares(fractions_mode, PLANES[-1])
    assert final[1] > 0.3 and final[2] > 0.01  # the states are all populated
    # the path in each state, and the sampled events against the expected ones
    reference = fractions_mode.charge_tallies
    tallies = analog_mode.charge_tallies
    rate = 1.0e-6 * 0.02 / constants.e
    # a path in one state is at most LENGTH long: its spread at most LENGTH / 2
    bound = 4.0 * 0.5 * LENGTH / np.sqrt(num)
    assert tallies.path / rate == pytest.approx(reference.path / rate, abs=bound)
    index = {"-": 0, "0": 1, "+": 2}
    for (target, start, end), value in tallies.changes.items():
        sigma = SIGMA[start][end]
        assert value == pytest.approx(GAS * sigma * tallies.path[index[start]])
        # the sampled events: a Poisson count of mean (n sigma path) per trajectory
        mean = reference.changes[(target, start, end)] / rate * num
        count = tallies.events.get((target, start, end), 0.0) / rate * num
        assert abs(count - mean) < 4.0 * np.sqrt(mean) + 1.0, (start, end)
    assert reference.events == {}


def test_splitting_keeps_the_mean_and_the_current() -> None:
    num = 1000
    model = BeamOptics(
        channel(num=num, mode="analog", splitting=4, split_events=1, seed=3)
    ).run()
    stats = model.stats
    assert stats["launched"] > num  # split trajectories
    assert stats["transmitted"] == stats["launched"]
    exit_plane = PLANES[-1]
    assert model.beam_moments(exit_plane).current == pytest.approx(
        1.0e-6 * 0.02, rel=1e-12
    )
    reference = charge_state_fractions(TRANSITIONS, {"gas": [0.0, GAS * exit_plane]})[
        -1
    ]
    # the stripped share is split in four: its spread is smaller than unsplit
    found = shares(model, exit_plane)
    spread = np.sqrt(reference * (1.0 - reference) / num)
    assert np.all(np.abs(found - reference) < 4.0 * spread)


def test_atoms_stripped_in_an_accelerating_gap_leave_with_intermediate_energy() -> None:
    """H- accelerated by a uniform field and stripped once: atoms of every energy between."""
    num, gap, voltage, energy = 4000, 0.1, 5.0e4, 1.0e4
    mu = 5.0  # n sigma (1/m): half the beam stripped across the gap
    neumann = FieldBoundaryCondition.neumann(0.0)
    config = BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(51, 11),
            lower_bound=(0.0, -0.01),
            upper_bound=(gap, 0.01),
            coordinates="cartesian",
        ),
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(0.0),
            x_upper=FieldBoundaryCondition.dirichlet(voltage),
            y_lower=neumann,
            y_upper=neumann,
        ),
        beams=[
            BeamSpecies(
                "H-",
                1.0e-6,
                (0.0, -0.005),
                (0.0, 0.005),
                charge=-1.0,
                mass=1.00849,
                energy=energy,
                transverse_temperature=0.0,
                num_particles=num,
            )
        ],
        charge_changing=ChargeChangingParams(
            {"gas": {"-": {"0": 1e-20}}},
            {"gas": mu / 1e-20},
            mode="analog",
            seed=11,
        ),
        iterations=1,
        diagnostic_planes=[gap - 1e-4],
    )
    model = BeamOptics(config).run()
    neutrals = model.state_moments(gap - 1e-4, "0")
    ions = model.state_moments(gap - 1e-4, "-")
    stripped = 1.0 - np.exp(-mu * gap)
    total = neutrals.current + ions.current
    spread = np.sqrt(stripped * (1.0 - stripped) / num)
    assert neutrals.current / total == pytest.approx(stripped, abs=4.0 * spread)
    # the H- gain the whole voltage; an atom what it gained before it was stripped
    assert ions.mean_energy == pytest.approx(energy + voltage, rel=2e-3)
    mean_x = 1.0 / mu - gap * np.exp(-mu * gap) / stripped
    mean_energy = energy + voltage * mean_x / gap
    count = stripped * num
    error = voltage * np.sqrt(1.0 / 12.0) / np.sqrt(count)
    assert neutrals.mean_energy == pytest.approx(mean_energy, abs=4.0 * error)
    hits = model.plane(gap - 1e-4)
    atom = hits.fractions[1] == 1.0
    kinetic = (
        0.5
        * 1.00849
        * constants.atomic_mass
        * np.sum(hits.velocity[:, atom] ** 2, axis=0)
        / constants.e
    )
    assert kinetic.min() > energy - 1.0 and kinetic.max() < energy + voltage + 1.0
    assert np.ptp(kinetic) > 0.8 * voltage


def test_fractions_follow_a_target_profile() -> None:
    profile = AxialProfile([0.0, 0.2, 0.4], [0.0, 2.0 * GAS, 0.5 * GAS])
    model = BeamOptics(
        channel(num=10, densities={"gas": {"x": profile.x, "density": profile.density}})
    ).run()
    for x in PLANES:
        line = profile.line_density(0.0, x)
        expected = charge_state_fractions(TRANSITIONS, {"gas": [0.0, line]})[-1]
        assert shares(model, x) == pytest.approx(expected, abs=1e-5), x
    assert profile.line_density(0.0, 0.4) == pytest.approx(0.2 * GAS + 0.25 * GAS)


def test_the_conductance_profile_drops_most_across_the_narrowest_aperture() -> None:
    profile = conductance_profile(
        [(0.0, 0.001, 0.007), (0.006, 0.011, 0.004), (0.02, 0.03, 0.007)],
        upstream=1.0e20,
        downstream=1.0e18,
    )
    density = np.array(profile.density)
    assert density[0] == 1.0e20 and density[-1] == pytest.approx(1.0e18)
    drops = density[0::2] - density[1::2]
    assert np.argmax(drops) == 1
    # free-molecular resistances add: 1 / (a^2 K) with K = 1 / (1 + 3 t / 8 a)
    resistance = [(1 + 3 * t / (8 * a)) / a**2 for t, a in ((1, 7), (5, 4), (10, 7))]
    assert drops / drops.sum() == pytest.approx(np.array(resistance) / sum(resistance))
    assert profile(np.array([0.003]))[0] == pytest.approx(density[1])  # the gap
    with pytest.raises(ValueError, match="in order"):
        conductance_profile([(0.0, 0.01, 0.005), (0.005, 0.02, 0.005)], 1e20)


def test_the_settings_round_trip_and_are_checked() -> None:
    config = channel(
        densities={"gas": AxialProfile([0.0, 0.5], [GAS, 0.0])},
        mode="analog",
        splitting=2,
    )
    again = BeamOpticsConfig.from_dict(config.to_dict())
    assert again.charge_changing == config.charge_changing
    changing = again.charge_changing.build()
    assert changing.analog and changing.splitting == 2
    assert isinstance(changing.densities["gas"], AxialProfile)
    with pytest.raises(ValueError, match="not uniform"):
        changing.rate_matrix()
    with pytest.raises(ValueError, match="mode"):
        ChargeChanging(TRANSITIONS, {"gas": GAS}, mode="exact")
    with pytest.raises(ValueError, match="splitting"):
        ChargeChanging(TRANSITIONS, {"gas": GAS}, splitting=0)
    with pytest.raises(ValueError, match="increase"):
        AxialProfile([0.0, 0.0], [1.0, 1.0])
