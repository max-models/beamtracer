"""The beamline: neutrals through the duct, against closed forms."""

import numpy as np
import pytest
from plasmacoll import ChargeStateTransitions, charge_state_fractions
from scipy.special import erf
from simgeometry import Cuboid, SlantedTube

from beamtracer.beamline import Duct, Markers, gaussian_beam
from beamtracer.solver.charge_changing import AxialProfile

LENGTH = 3.0
SPEED = 1e7
NUM = 200_000


def round_duct(bore: float) -> SlantedTube:
    """A circular duct along x from 0 to LENGTH (cut square to the axis)."""
    return SlantedTube((0, 0, 0), (LENGTH, 0, 0), bore, 2 * bore, (1, 0, 0))


def square_duct(half: float) -> list[Cuboid]:
    """Four plates around a square bore of half-width ``half``."""
    t = half
    return [
        Cuboid((LENGTH / 2, half + t / 2, 0), (LENGTH, t, 2 * half + 2 * t)),
        Cuboid((LENGTH / 2, -half - t / 2, 0), (LENGTH, t, 2 * half + 2 * t)),
        Cuboid((LENGTH / 2, 0, half + t / 2), (LENGTH, 2 * half, t)),
        Cuboid((LENGTH / 2, 0, -half - t / 2), (LENGTH, 2 * half, t)),
    ]


def binomial_tolerance(p: float, n: int = NUM) -> float:
    """Five standard deviations of a binomial share."""
    return 5 * np.sqrt(p * (1 - p) / n) + 1e-12


@pytest.mark.parametrize("divergence", [0.01, 0.03])
def test_round_duct_transmission(divergence):
    width, bore = 0.01, 0.06
    beam = gaussian_beam(
        NUM, power=1e6, speed=SPEED, width=width, divergence=divergence
    )
    result = Duct([round_duct(bore)], port=LENGTH).trace(beam)
    sigma2 = width**2 + (LENGTH * divergence) ** 2
    expected = 1 - np.exp(-(bore**2) / (2 * sigma2))
    assert result.transmission == pytest.approx(
        expected, abs=binomial_tolerance(expected)
    )
    # the hits lie on the bore, the normals point into the beam
    r = np.hypot(*result.wall.position[1:])
    assert np.allclose(r, bore)
    radial = result.wall.position[1:] / r
    assert np.allclose(np.sum(result.wall_normal[1:] * radial, axis=0), -1.0)
    assert np.allclose(result.port.position[0], LENGTH)


def test_square_duct_transmission():
    width, divergence, half = 0.01, 0.02, 0.05
    beam = gaussian_beam(
        NUM, power=1e6, speed=SPEED, width=width, divergence=divergence
    )
    walls = square_duct(half)
    result = Duct(walls, port=LENGTH).trace(beam)
    sigma = np.hypot(width, LENGTH * divergence)
    # freeofcharge.chain.duct_transmission
    expected = erf(half / (np.sqrt(2) * sigma)) ** 2
    assert result.transmission == pytest.approx(
        expected, abs=binomial_tolerance(expected)
    )
    assert result.wall_power(len(walls)).sum() == pytest.approx(result.wall.total_power)
    # the four walls take equal shares
    shares = result.wall_power(len(walls)) / result.wall.total_power
    assert np.allclose(shares, 0.25, atol=0.02)


TRANSITIONS = ChargeStateTransitions(
    {
        "D2": {("0", "+"): 2e-20, ("0", "-"): 3e-21, ("+", "0"): 1e-22},
        "e": {("0", "+"): 5e-20},
    }
)


@pytest.mark.parametrize(
    "gas",
    [
        {"D2": 2e18},
        {"D2": AxialProfile((0.0, 1.0, LENGTH), (4e18, 1e18, 0.0)), "e": 1e17},
    ],
)
def test_reionized_fraction(gas):
    beam = gaussian_beam(NUM, power=1e6, speed=SPEED, width=0.01, divergence=0.0)
    duct = Duct([round_duct(0.2)], port=LENGTH, transitions=TRANSITIONS, gas=gas)
    result = duct.trace(beam)
    assert len(result.wall) == 0
    # charge_state_fractions without the transitions back to the neutrals: a reionized
    # neutral leaves the beam
    first_loss = ChargeStateTransitions(
        {
            t: {k: v for k, v in table.items() if k[0] == "0"}
            for t, table in TRANSITIONS.cross_sections.items()
        }
    )
    # the uniform targets keep their ratio only on a uniform path; take it in steps
    xs = np.linspace(0.0, LENGTH, 401)
    line = {
        t: [
            (n.line_density(0.0, x) if isinstance(n, AxialProfile) else n * x)
            for x in xs
        ]
        for t, n in gas.items()
    }
    fractions = charge_state_fractions(first_loss, line, initial={"0": 1.0})[-1]
    states = first_loss.states
    plus = np.sum(result.reionized.state == states.index("+")) / NUM
    minus = np.sum(result.reionized.state == states.index("-")) / NUM
    assert plus == pytest.approx(
        fractions[states.index("+")], abs=binomial_tolerance(plus)
    )
    assert minus == pytest.approx(
        fractions[states.index("-")], abs=binomial_tolerance(minus)
    )
    assert result.transmission == pytest.approx(
        fractions[states.index("0")], abs=binomial_tolerance(result.transmission)
    )


def test_reionization_points_follow_the_exponential():
    n, sigma = 1e19, 2e-20  # free path 5 m
    beam = gaussian_beam(NUM, power=1.0, speed=SPEED, width=0.01, divergence=0.0)
    transitions = ChargeStateTransitions({"D2": {("0", "+"): sigma}})
    result = Duct(
        [round_duct(0.2)], port=LENGTH, transitions=transitions, gas={"D2": n}
    ).trace(beam)
    k = n * sigma
    # the mean of an exponential cut at LENGTH
    mean = 1 / k - LENGTH * np.exp(-k * LENGTH) / (1 - np.exp(-k * LENGTH))
    x = result.reionized.position[0]
    assert x.mean() == pytest.approx(mean, abs=5 * x.std() / np.sqrt(x.size))
    edges = np.linspace(0, LENGTH, 7)
    per_length = result.reionized.along_x(edges)
    expected = k * np.exp(-k * per_length.x.values)
    assert np.allclose(per_length.values, expected, rtol=0.03)


def test_power_balance_and_footprint():
    beam = gaussian_beam(NUM, power=2e6, speed=SPEED, width=0.02, divergence=0.02)
    transitions = ChargeStateTransitions({"D2": {("0", "+"): 2e-20}})
    result = Duct(
        square_duct(0.06), port=LENGTH, transitions=transitions, gas={"D2": 1e18}
    ).trace(beam)
    total = result.port.total_power + result.wall.total_power
    total += result.reionized.total_power
    assert total == pytest.approx(2e6)
    summary = result.summary()
    assert float(summary.power_in) == pytest.approx(2e6)
    footprint = result.port.footprint(bins=30, extent=(-0.06, 0.06, -0.06, 0.06))
    area = footprint.y.diff("y")[0] * footprint.z.diff("z")[0]
    assert float(footprint.sum() * area) == pytest.approx(result.port.total_power)
    assert footprint.attrs["units"] == "W/m^2"


def test_markers_starting_in_a_wall_stop_there():
    position = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 0.0]])
    velocity = np.array([[SPEED, SPEED], [0.0, 0.0], [0.0, 0.0]])
    markers = Markers(position, velocity, 1.0, 1)
    result = Duct([round_duct(0.06)], port=LENGTH).trace(markers)
    assert len(result.port) == 1
    assert len(result.wall) == 1
    assert np.allclose(result.wall.position[:, 0], [0.0, 0.1, 0.0])


def test_refused_inputs():
    ions = Markers(np.zeros((3, 1)), [[SPEED], [0], [0]], 1.0, 0)
    with pytest.raises(ValueError, match="dump the ions"):
        Duct([], port=1.0).trace(ions)
    backwards = Markers(np.zeros((3, 1)), [[-SPEED], [0], [0]], 1.0, 1)
    with pytest.raises(ValueError, match="move along"):
        Duct([], port=1.0).trace(backwards)
    with pytest.raises(ValueError, match="needs the transitions"):
        Duct([], port=1.0, gas={"D2": 1e18})
    with pytest.raises(ValueError, match="no cross sections"):
        Duct([], port=1.0, transitions=TRANSITIONS, gas={"H2": 1e18})
    with pytest.raises(ValueError, match="no neutral state"):
        Duct([], port=1.0).trace(
            Markers(np.zeros((3, 1)), [[1], [0], [0]], 1.0, 0, ("a", "b"))
        )
    with pytest.raises(ValueError, match="same number"):
        Markers(np.zeros((3, 2)), np.zeros((3, 1)), 1.0, 1)
    with pytest.raises(ValueError, match="index states"):
        Markers(np.zeros((3, 1)), np.zeros((3, 1)), 1.0, 5)
