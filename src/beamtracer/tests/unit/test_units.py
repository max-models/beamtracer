"""The settings take astropy Quantities, quantity strings and bare numbers."""

import json
import warnings

import astropy.units as u
import numpy as np
import pytest
from scipy import constants

from beamtracer import (
    AxialField,
    BeamOpticsConfig,
    BeamSpecies,
    ChargeChangingParams,
    ExtractionPlasma,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    NegativeExtractionPlasma,
    ObjectParams,
    ParticleBeam,
    ShapeParams,
    Solenoid,
    SpaceChargeCompensation,
    SymbolicSpaceFunction,
)
from beamtracer.beamline import (
    ApertureGrid,
    Calorimeter,
    Duct,
    FieldRegion,
    beamlet_markers,
    box_field,
    gaussian_beam,
    profile_field,
)
from beamtracer.optimization import Parameter
from beamtracer.optimization.beamlet import NegativeIonBeamlet
from beamtracer.optimization.einzel import EinzelLens
from beamtracer.solver.charge_changing import AxialProfile, conductance_profile
from beamtracer.solver.tracer import PlaneHits

EV_IN_K = constants.e / constants.k


def _round_trip(config: BeamOpticsConfig) -> BeamOpticsConfig:
    return BeamOpticsConfig.from_dict(json.loads(json.dumps(config.to_dict())))


# -- the sections ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bound",
    [(5 * u.mm, 2.5 * u.mm), ("5 mm", "2.5 mm"), [5, 2.5] * u.mm, (5e-3, 2.5e-3)],
)
def test_grid_bounds_take_quantities_strings_and_metres(bound) -> None:
    grid = GridParams(num_gridpoints=(11, 6), upper_bound=bound)
    assert grid.upper_bound == pytest.approx((5e-3, 2.5e-3))
    assert all(type(b) is float for b in grid.upper_bound)


def test_grid_bounds_round_trip_as_quantity_strings() -> None:
    grid = GridParams(lower_bound="-1 mm", upper_bound=("3 cm", "2.9 cm"))
    params = grid.to_dict()
    assert params["lower_bound"] == ["-0.001 m", "-0.001 m"]
    assert GridParams.from_dict(params) == grid


def test_grid_bounds_of_the_wrong_dimension_raise() -> None:
    with pytest.raises(ValueError, match="upper_bound must be in units of m"):
        GridParams(upper_bound=5 * u.V)


def test_objects_take_lengths_and_potentials_as_quantities() -> None:
    plate = ObjectParams(
        type="rectangle",
        bc_type="dirichlet",
        bc_value=5 * u.kV,
        midpoint=("5 mm", "2 mm"),
        side_lengths=[1, 4] * u.mm,
    )
    assert plate.bc_value == pytest.approx(5000.0)
    assert plate.midpoint == pytest.approx((5e-3, 2e-3))
    assert plate.side_lengths == pytest.approx((1e-3, 4e-3))
    params = json.loads(json.dumps(plate.to_dict()))
    assert params["bc"] == {"type": "dirichlet", "potential": "5000 V"}
    assert ObjectParams.from_dict(params) == plate


def test_a_neumann_object_is_in_volts_per_metre() -> None:
    wall = ObjectParams(
        type="circle",
        bc_type="neumann",
        bc_value="1 kV/mm",
        midpoint=(0.0, 0.0),
        radius=1e-3,
    )
    assert wall.bc_value == pytest.approx(1e6)
    assert wall.to_dict()["bc"] == {
        "type": "neumann",
        "normal_derivative": "1000000 V/m",
    }
    with pytest.raises(ValueError, match="bc_value must be in units of V/m"):
        ObjectParams(type="circle", bc_type="neumann", bc_value="1 kV", radius=1e-3)


def test_shapes_take_quantities_and_read_their_configuration_entries() -> None:
    shape = ShapeParams("circle", midpoint=(1 * u.mm, 2 * u.mm), radius="0.5 mm")
    assert shape.radius == pytest.approx(0.5e-3)
    assert ShapeParams.from_dict(json.loads(json.dumps(shape.to_dict()))) == shape
    with pytest.raises(ValueError, match="radius must be in units of m"):
        ShapeParams("circle", midpoint=(0, 0), radius=1 * u.s)


def test_an_old_unit_entry_is_rejected_with_a_hint() -> None:
    with pytest.raises(ValueError, match="'2.5 mm'"):
        ShapeParams("circle", midpoint=(0, 0), radius={"value": 0.5, "unit": "mm"})
    with pytest.raises(ValueError, match="5 kV"):
        FieldBCs.from_dict(
            {
                "x": {"type": "dirichlet", "min": {"value": 0, "unit": "V"}, "max": 0},
                "y": {"type": "neumann", "min": 0, "max": 0},
            }
        )


def test_field_faces_read_quantity_strings_and_numbers() -> None:
    faces = FieldBCs.from_dict(
        {
            "x": {"type": "dirichlet", "min": 0, "max": "-1 kV"},
            "y": {"type": "neumann", "min": "2 kV/m", "max": {"value": 5.0}},
        }
    )
    assert faces.get_bc("x", "max").evaluate(x=0.0, y=0.0) == pytest.approx(-1000.0)
    assert faces.get_bc("y", "min").evaluate(x=0.0, y=0.0) == pytest.approx(2000.0)
    assert faces.get_bc("y", "max").evaluate(x=0.0, y=0.0) == pytest.approx(5.0)
    assert FieldBCs.from_dict(faces.to_dict()).to_dict() == faces.to_dict()
    with pytest.raises(ValueError, match="in units of V/m"):
        FieldBCs.from_dict(
            {
                "x": {"type": "dirichlet", "min": 0, "max": 0},
                "y": {"type": "neumann", "min": "2 kV", "max": 0},
            }
        )


def test_a_face_condition_takes_a_quantity() -> None:
    face = FieldBoundaryCondition.dirichlet(-2 * u.kV)
    assert face.evaluate(x=0.0, y=0.0) == pytest.approx(-2000.0)
    kept = SymbolicSpaceFunction(3 * u.kV)
    assert kept.evaluate(x=0.0, y=0.0) == pytest.approx(3000.0)
    assert kept.to_dict() == {"value": "3 kV"}


# -- the configuration -------------------------------------------------------------------


def test_a_beam_takes_quantities_and_kelvin_for_its_temperatures() -> None:
    beam = BeamSpecies(
        "D-",
        "300 A/m2",
        ("0 mm", "0 mm"),
        (0, 4 * u.mm),
        mass=2.014 * u.u,
        energy="1 keV",
        parallel_temperature=EV_IN_K * u.K,
        transverse_temperature="0.5 eV",
    )
    assert beam.current_density == 300.0
    assert beam.end == pytest.approx((0.0, 4e-3))
    assert beam.mass == pytest.approx(2.014)
    assert beam.energy == pytest.approx(1000.0)
    assert beam.parallel_temperature == pytest.approx(1.0)
    assert beam.transverse_temperature == 0.5
    bare = BeamSpecies("D-", 300.0, (0.0, 0.0), (0.0, 4e-3), mass=2.014, energy=1e3)
    assert bare.mass == beam.mass and bare.end == beam.end
    with pytest.raises(ValueError, match="energy must be in units of eV"):
        BeamSpecies("D-", 300.0, (0.0, 0.0), (0.0, 4e-3), energy=1 * u.m)


def test_a_particle_beam_takes_quantity_arrays() -> None:
    position = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    beam = ParticleBeam(
        "H+",
        position * u.mm,
        [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]] * u.km / u.s,
        "[1, 2] mA",
        mass="1.007 u",
    )
    assert np.allclose(beam.position, position * 1e-3)
    assert beam.velocity[1] == pytest.approx((1e3, 1e2, 0.0))
    assert beam.current == pytest.approx((1e-3, 2e-3))
    assert beam.mass == pytest.approx(1.007)
    with pytest.raises(ValueError, match="current must be in units of A"):
        ParticleBeam("H+", position, position, [1, 2] * u.V)


def test_the_plasmas_take_quantities() -> None:
    plasma = ExtractionPlasma("2 eV", 20 * u.V, "1 mm", rho_e="1e-3 C/m3")
    assert (plasma.electron_temperature, plasma.plasma_potential) == (2.0, 20.0)
    assert plasma.initial_plasma_x == pytest.approx(1e-3)
    assert plasma.rho_e == pytest.approx(1e-3)
    negative = NegativeExtractionPlasma(EV_IN_K * u.K, "-1 V", 0.5 * u.mm)
    assert negative.ion_temperature == pytest.approx(1.0)
    assert negative.rho_p is None
    with pytest.raises(ValueError, match="ion_temperature must be in units of eV"):
        NegativeExtractionPlasma("1 V", 0.0, 0.0)


def test_magnets_and_compensation_take_quantities() -> None:
    solenoid = Solenoid("50 mm", 2 * u.cm, "30 mm", "100 mT")
    assert (solenoid.centre, solenoid.peak) == pytest.approx((0.05, 0.1))
    field = AxialField("-1 cm", 1 * u.mm, [1, 2, 3] * u.mT)
    assert field.bz == pytest.approx((1e-3, 2e-3, 3e-3))
    assert field.z0 == pytest.approx(-0.01)
    compensation = SpaceChargeCompensation("0 mm", 5 * u.mm, 0.5)
    assert compensation.x_max == pytest.approx(5e-3)
    with pytest.raises(ValueError, match="peak must be in units of T"):
        Solenoid(0.0, 0.1, 0.03, "1 A")


def test_charge_changing_takes_cross_sections_and_densities() -> None:
    params = ChargeChangingParams(
        {"gas": {"-": {"0": "1e-16 cm2"}}},
        {
            "gas": {"x": [0, 10] * u.mm, "density": ["1e13 cm-3", 0]},
            "plasma": 1e17 * u.m**-3,
        },
        ionization={"gas": {"-": 2e-20 * u.m**2}},
    )
    assert params.cross_sections["gas"]["-"]["0"] == pytest.approx(1e-20)
    assert params.densities["gas"]["x"] == pytest.approx([0.0, 0.01])
    assert params.densities["gas"]["density"] == pytest.approx([1e19, 0.0])
    assert params.densities["plasma"] == pytest.approx(1e17)
    assert params.ionization["gas"]["-"] == pytest.approx(2e-20)
    profile = ChargeChangingParams(
        {"gas": {"-": {"0": 1e-20}}},
        {"gas": AxialProfile([0.0, 0.01], [1e19, 0.0])},
    )
    assert profile.densities["gas"]["x"] == pytest.approx(params.densities["gas"]["x"])
    with pytest.raises(ValueError, match="must be in units of m2"):
        ChargeChangingParams({"gas": {"-": {"0": "1 m"}}}, {"gas": 1e19})


def test_the_configuration_round_trips_with_units() -> None:
    config = BeamOpticsConfig(
        grid=GridParams(num_gridpoints=(11, 6), upper_bound=("10 mm", "5 mm")),
        objects={
            "plate": ObjectParams(
                type="rectangle",
                bc_type="dirichlet",
                bc_value="-1 kV",
                midpoint=("9 mm", "4 mm"),
                side_lengths=("1 mm", "2 mm"),
            )
        },
        bcs_field=FieldBCs(
            FieldBoundaryCondition.dirichlet(0.0),
            FieldBoundaryCondition.dirichlet(-1 * u.kV),
            FieldBoundaryCondition.neumann(0.0),
            FieldBoundaryCondition.neumann(0.0),
        ),
        beams=[
            BeamSpecies("H+", "10 A/m2", (0, 0), ("0 mm", "4 mm"), energy="10 eV"),
            ParticleBeam("H+", [[0, 1e-3, 0]], [[1e5, 0, 0]], [1e-3]),
        ],
        plasma=ExtractionPlasma("5 eV", "10 V", "1 mm"),
        compensation=[SpaceChargeCompensation("0 mm", "5 mm", 0.5)],
        charge_changing=ChargeChangingParams(
            {"gas": {"-": {"0": "1e-20 m2"}}},
            {"gas": {"x": ["0 mm", "10 mm"], "density": ["1e19 m-3", 0]}},
        ),
        tolerance="1 mV",
        diagnostic_planes=["9.9 mm", 5e-3],
    )
    assert config.tolerance == pytest.approx(1e-3)
    assert config.diagnostic_planes == pytest.approx([9.9e-3, 5e-3])
    params = config.to_dict()
    assert params["tolerance"] == "0.001 V"
    assert params["beams"][0]["energy"] == "10 eV"
    assert params["plasma"]["electron_temperature"] == "5 eV"
    assert params["charge_changing"]["cross_sections"] == {
        "gas": {"-": {"0": "1e-20 m2"}}
    }
    again = _round_trip(config)
    assert again.to_dict() == params
    assert again.beams == config.beams
    assert again.charge_changing == config.charge_changing


def test_a_solenoid_configuration_round_trips() -> None:
    config = BeamOpticsConfig(
        grid=GridParams(
            num_gridpoints=(11, 6), upper_bound=(0.01, 0.005), coordinates="cylindrical"
        ),
        solenoids=[Solenoid("5 mm", "2 mm", "3 mm", "0.1 T")],
        axial_field=AxialField(0.0, "1 mm", [0.0, 0.01]),
    )
    params = config.to_dict()
    assert params["solenoids"][0]["peak"] == "0.1 T"
    assert _round_trip(config).to_dict() == params


def test_the_configuration_rejects_the_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="diagnostic_planes must be in units of m"):
        BeamOpticsConfig(diagnostic_planes=["1 V"])


# -- the beamline ----------------------------------------------------------------------


def test_a_gaussian_beam_takes_quantities() -> None:
    plain = gaussian_beam(20, power=1e3, speed=1e7, width=2e-3, divergence=5e-3, x=0.1)
    beam = gaussian_beam(
        20,
        power="1 kW",
        speed=1e4 * u.km / u.s,
        width=2 * u.mm,
        divergence="5 mrad",
        x="10 cm",
    )
    assert np.allclose(beam.position, plain.position)
    assert np.allclose(beam.velocity, plain.velocity)
    assert np.allclose(beam.power, plain.power)
    with pytest.raises(ValueError, match="divergence must be in units of rad"):
        gaussian_beam(2, power=1.0, speed=1.0, width=1.0, divergence="1 m")


def test_beamlet_markers_take_the_mass_and_slit_length_as_quantities() -> None:
    hits = PlaneHits(0.1)
    hits.append(
        np.array([[0.1], [1e-3], [0.0]]),
        np.array([[1e7], [0.0], [0.0]]),
        np.array([1e-3]),
        np.zeros(1, dtype=int),
        np.ones(1, dtype=int),
    )
    mass = 2.014 * constants.atomic_mass
    plain = beamlet_markers(hits, mass=mass, cylindrical=False, slit_length=0.05)
    markers = beamlet_markers(
        hits, mass=2.014 * u.u, cylindrical=False, slit_length="5 cm"
    )
    assert markers.power == pytest.approx(plain.power)
    with pytest.raises(ValueError, match="mass must be in units of kg"):
        beamlet_markers(hits, mass="1 m", cylindrical=True)


def test_an_aperture_grid_takes_quantities() -> None:
    grid = ApertureGrid(
        [[0, 20], [0, 0]] * u.mm,
        focal_length=("10 m", "inf m"),
        steering=[[1, 0], [0, 0]] * u.mrad,
    )
    assert np.allclose(grid.centres, [[0.0, 0.02], [0.0, 0.0]])
    assert grid.focal_length == (10.0, np.inf)
    assert grid.slopes[0] == pytest.approx([1e-3, -2e-3])
    regular = ApertureGrid.regular(1, 2, ("20 mm", 20 * u.mm))
    assert np.allclose(regular.centres, [[-0.01, 0.01], [0.0, 0.0]])


def test_the_duct_and_the_regions_take_quantities() -> None:
    from plasmacoll import ChargeStateTransitions

    transitions = ChargeStateTransitions({"D2": {("0", "+"): 2e-20}})
    duct = Duct(
        [],
        port="4 m",
        transitions=transitions,
        gas={"D2": "2e12 cm-3"},
    )
    assert duct.port == 4.0
    assert duct._gas["D2"] == pytest.approx(2e18)
    profile = AxialProfile([0, 4] * u.m, ["2e18 m-3", "1e18 m-3"])
    assert (
        Duct([], port=4.0, transitions=transitions, gas={"D2": profile})._gas["D2"]
        is profile
    )
    region = FieldRegion([], exit="150 cm", mass=2.014 * u.u, step="2 mm")
    assert region.exit == pytest.approx(1.5)
    assert region.mass == pytest.approx(2.014 * constants.atomic_mass)
    assert region.step == pytest.approx(2e-3)
    calorimeter = Calorimeter("5 m", ("-10 cm", "10 cm", "-10 cm", "10 cm"))
    assert calorimeter.extent == pytest.approx((-0.1, 0.1, -0.1, 0.1))
    with pytest.raises(ValueError, match="gas\\['D2'\\] must be in units of m-3"):
        Duct([], port=4.0, transitions=transitions, gas={"D2": "1 m"})


def test_the_simple_fields_take_their_positions_as_quantities() -> None:
    plates = box_field(
        (0, 1e5, 0), ("0 m", "-5 cm", "-1 m"), (1 * u.m, 5 * u.cm, 1 * u.m)
    )
    points = np.array([[0.5, 2.0], [0.0, 0.0], [0.0, 0.0]])
    assert plates(points)[1] == pytest.approx([1e5, 0.0])
    stray = profile_field([0, 100] * u.cm, [[0, 0], [0, 0], [0.0, 0.1]])
    assert stray(np.array([[0.5], [0.0], [0.0]]))[2] == pytest.approx([0.05])


def test_gas_profiles_take_quantities() -> None:
    profile = conductance_profile(
        [("0 mm", "1 mm", "2 mm"), (5e-3, 6e-3, 2e-3)], "1e13 cm-3", 0 * u.m**-3
    )
    plain = conductance_profile([(0.0, 1e-3, 2e-3), (5e-3, 6e-3, 2e-3)], 1e19)
    assert profile.x == pytest.approx(plain.x)
    assert profile.density == pytest.approx(plain.density)
    with pytest.raises(ValueError, match="upstream must be in units of m-3"):
        conductance_profile([(0.0, 1e-3, 2e-3)], "1 Pa")


# -- the neutralizer and the problems ----------------------------------------------------


def test_a_neutralizer_channel_takes_quantities() -> None:
    wellbalanced = pytest.importorskip("wellbalanced.neutralizer")
    from beamtracer.neutralizer import NeutralizerChannel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        neutralizer = wellbalanced.illustrative_deuterium_neutralizer(gas_inflow=6.0e21)
    channel = NeutralizerChannel(
        neutralizer, beam_width="6 cm", divergence=5 * u.mrad, cell="5 mm"
    )
    assert (channel.beam_width, channel.divergence, channel.cell) == pytest.approx(
        (0.06, 5e-3, 5e-3)
    )
    config = channel.beam_config("1e13 cm-3", 1e17 * u.m**-3)
    assert config.charge_changing is not None
    assert config.charge_changing.densities["gas"] == pytest.approx(1e19)
    with pytest.raises(ValueError, match="beam_width must be in units of m"):
        NeutralizerChannel(neutralizer, beam_width="6 V")


def test_the_problems_take_quantities() -> None:
    lens = EinzelLens(
        energy="30 keV", voltage=15 * u.kV, gap="4 mm", mass=1.007276 * u.u
    )
    assert lens == EinzelLens()
    beamlet = NegativeIonBeamlet(
        extraction_voltage="5 kV",
        ion_temperature=EV_IN_K * u.K,
        divergence_limit="7 mrad",
        current_density=30 * u.mA / u.cm**2,
    )
    assert beamlet == NegativeIonBeamlet()
    assert beamlet.at({"gap": 5e-3}).gap == 5e-3
    with pytest.raises(ValueError, match="gap must be in units of m"):
        EinzelLens(gap="4 V")


def test_parameter_bounds_take_quantities_in_their_unit() -> None:
    gap = Parameter("gap", "2 mm", 6 * u.mm, "m")
    assert (gap.lower, gap.upper) == pytest.approx((2e-3, 6e-3))
    assert Parameter("ratio", 0.1, 0.9).upper == 0.9
    with pytest.raises(ValueError, match="voltage.lower must be in units of V"):
        Parameter("voltage", "2 mm", 5e3, "V")
