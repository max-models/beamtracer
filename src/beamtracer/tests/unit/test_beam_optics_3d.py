"""Tests for BeamOptics on a 3D grid."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    ExtractionPlasma,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ObjectParams,
    Solenoid,
)
from beamtracer.config import BeamSpecies3D
from beamtracer.solver.tracer3d import beam_on_surface

NEUMANN = FieldBoundaryCondition.neumann(0.0)


def _plate(midpoint, sides, potential, three=True) -> ObjectParams:
    if three:
        return ObjectParams(
            type="cuboid", midpoint=midpoint, side_lengths=sides,
            bc_type="dirichlet", bc_value=potential,
        )  # fmt: skip
    return ObjectParams(
        type="rectangle", midpoint=midpoint, side_lengths=sides,
        bc_type="dirichlet", bc_value=potential,
    )  # fmt: skip


def _diode(three: bool, **settings) -> BeamOpticsConfig:
    """A planar extraction gap with a slit puller: 2D, or extruded along z in 3D."""
    h, depth = 0.25e-3, 1.0e-3
    nx, ny, nz = 41, 17, 5
    dirichlet = FieldBoundaryCondition.dirichlet
    faces = [dirichlet(0.0), dirichlet(-3000.0), NEUMANN, NEUMANN]
    objects = {
        "puller": _plate(
            (5.03e-3, 3.06e-3, 0.0) if three else (5.03e-3, 3.06e-3),
            (1.0e-3, 2.0e-3, 4 * depth) if three else (1.0e-3, 2.0e-3),
            -3000.0,
            three,
        )
    }
    if three:
        grid = GridParams(
            ndim=3, num_gridpoints=(nx, ny, nz),
            lower_bound=(0.0, 0.0, 0.0), upper_bound=(10e-3, 4e-3, depth),
        )  # fmt: skip
        beam = BeamSpecies3D(
            "H+", 30.0, (0.0, 0.75e-3, 0.5 * depth), (1.5e-3, depth),
            energy=5.0, transverse_temperature=0.0, num_particles=480,
        )  # fmt: skip
        faces += [NEUMANN, NEUMANN]
    else:
        grid = GridParams(
            num_gridpoints=(nx, ny), lower_bound=(0.0, 0.0), upper_bound=(10e-3, 4e-3)
        )
        beam = BeamSpecies(
            "H+", 30.0, (0.0, 0.0), (0.0, 1.5e-3),
            energy=5.0, transverse_temperature=0.0, num_particles=24,
        )  # fmt: skip
    assert np.isclose(grid.deltas[0], h)
    settings.setdefault("iterations", 4)
    return BeamOpticsConfig(
        grid=grid,
        objects=objects,
        bcs_field=FieldBCs(*faces),
        beams=[beam],
        tolerance=0.0,
        diagnostic_planes=[9.8e-3],
        **settings,
    )


def test_an_extruded_planar_model_runs_as_in_2d() -> None:
    flat = BeamOptics(_diode(three=False)).run()
    deep = BeamOptics(_diode(three=True)).run()
    a, b = flat.beam_moments(9.8e-3), deep.beam_moments(9.8e-3)
    depth = 1.0e-3  # the 2D current is per unit depth
    assert b.current == pytest.approx(a.current * depth, rel=1e-2)
    assert b.rms_size == pytest.approx(a.rms_size, rel=2e-2)
    assert b.rms_angle == pytest.approx(a.rms_angle, rel=2e-2)
    assert b.mean_energy == pytest.approx(a.mean_energy, rel=1e-4)
    # the extruded beam stays nearly flat: a small spread along z from the z faces' cut
    # cells (patchwork's two-point flux there)
    assert deep.beam_moments(9.8e-3, axis="z").rms_angle < 2e-2 * a.rms_angle
    flat_potential = flat.potential
    deep_potential = deep.potential
    assert deep_potential.shape == (41, 17, 5)
    assert np.max(np.abs(deep_potential[:, :, 2] - flat_potential)) < 5e-3 * 3000.0


def _quadrupole(quarter: bool) -> BeamOpticsConfig:
    """An electrostatic quadrupole, whole or its quarter with two symmetry planes."""
    objects = {}
    for name, axis, sign, potential in (
        ("y_plus", 1, 1, 2000.0), ("y_minus", 1, -1, 2000.0),
        ("z_plus", 2, 1, -2000.0), ("z_minus", 2, -1, -2000.0),
    ):  # fmt: skip
        if quarter and sign < 0:
            continue
        centre, sides = [0.0503, 0.0, 0.0], [0.02, 0.03, 0.03]
        centre[axis] = sign * 0.0116  # faces between nodes
        sides[axis] = 0.007
        objects[name] = _plate(tuple(centre), tuple(sides), potential)
    low = 0.0 if quarter else -0.01
    n = 11 if quarter else 21
    beam = BeamSpecies3D(
        "H+", 10.0,
        (0.0, 0.001, 0.001) if quarter else (0.0, 0.0, 0.0),
        (0.002, 0.002) if quarter else (0.004, 0.004),
        energy=10_000.0, transverse_temperature=0.0,
        num_particles=100 if quarter else 400,
    )  # fmt: skip
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=3,
            num_gridpoints=(101, n, n),
            lower_bound=(0.0, low, low),
            upper_bound=(0.1, 0.01, 0.01),
        ),  # fmt: skip
        objects=objects,
        bcs_field=FieldBCs(*([NEUMANN] * 6)),
        beams=[beam],
        iterations=2,
        tolerance=0.0,
        diagnostic_planes=[0.095],
        symmetry_planes=["y_min", "z_min"] if quarter else [],
    )


def test_a_quadrupole_focuses_in_one_plane_and_its_quarter_traces_the_whole() -> None:
    whole = BeamOptics(_quadrupole(quarter=False)).run()
    quarter = BeamOptics(_quadrupole(quarter=True)).run()
    y, z = whole.beam_moments(0.095), whole.beam_moments(0.095, axis="z")
    assert y.rms_size < z.rms_size  # focusing in y, defocusing in z
    for axis in ("y", "z"):
        a = whole.beam_moments(0.095, axis=axis)
        b = quarter.beam_moments(0.095, axis=axis)
        assert b.current == pytest.approx(a.current / 4, rel=1e-9)
    # The plates cross the other symmetry plane, and patchwork takes the cut edge's
    # two-point flux at nodes on a face (Shortley-Weller rows inside): next to the
    # plates the quarter's potential differs locally from the whole's (by up to 1% of
    # the span), its beam by 1e-3. The quarter's trajectories continue across the
    # planes: compare |y| and |z|.
    hits_whole, hits_quarter = whole.plane(0.095), quarter.plane(0.095)
    for a in (1, 2):
        rms = [
            np.sqrt(np.average(h.point[a] ** 2, weights=h.current))
            for h in (hits_whole, hits_quarter)
        ]
        assert rms[1] == pytest.approx(rms[0], rel=1e-3)
    difference = quarter.potential - whole.potential[:, 10:, 10:]
    assert np.sqrt(np.mean(difference**2)) < 1e-3 * 4000.0
    assert np.max(np.abs(difference)) < 1e-2 * 4000.0


def test_a_solenoid_on_the_x_axis_focuses_as_in_the_cylindrical_model() -> None:
    solenoid = Solenoid(centre=0.05, length=0.03, radius=0.02, peak=0.4)
    common = {
        "solenoids": [solenoid], "iterations": 1, "tolerance": 0.0,
        "diagnostic_planes": [0.095],
    }  # fmt: skip
    cylindrical = BeamOpticsConfig(
        grid=GridParams(
            num_gridpoints=(101, 11), lower_bound=(0.0, 0.0), upper_bound=(0.1, 0.01),
            coordinates="cylindrical",
        ),  # fmt: skip
        bcs_field=FieldBCs(NEUMANN, NEUMANN, NEUMANN, FieldBoundaryCondition.dirichlet(0.0)),
        beams=[BeamSpecies("H+", 1e-3, (0.0, 0.0), (0.0, 3e-3), energy=5000.0,
                           transverse_temperature=0.0, num_particles=200)],
        **common,
    )  # fmt: skip
    deep = BeamOpticsConfig(
        grid=GridParams(
            ndim=3, num_gridpoints=(101, 21, 21),
            lower_bound=(0.0, -0.01, -0.01), upper_bound=(0.1, 0.01, 0.01),
        ),  # fmt: skip
        bcs_field=FieldBCs(NEUMANN, NEUMANN, *([FieldBoundaryCondition.dirichlet(0.0)] * 4)),
        beams=[BeamSpecies3D("H+", 1e-3, (0.0, 0.0, 0.0), (3e-3,), shape="disc",
                             energy=5000.0, transverse_temperature=0.0, num_particles=600)],
        **common,
    )  # fmt: skip
    a = BeamOptics(cylindrical).run().beam_moments(0.095)
    b3 = BeamOptics(deep).run()
    b = b3.beam_moments(0.095)
    assert b.rms_size == pytest.approx(a.rms_size, rel=3e-2)
    assert b.rms_angle == pytest.approx(a.rms_angle, rel=3e-2)
    # the solenoid rotates the beam: the trajectories gain an azimuthal velocity
    hits = b3.plane(0.095)
    r = np.hypot(hits.point[1], hits.point[2])
    vt = (hits.point[1] * hits.velocity[2] - hits.point[2] * hits.velocity[1]) / r
    assert np.median(np.abs(vt)) > 1e3


def test_a_3d_extraction_with_a_plasma_converges() -> None:
    plasma = ExtractionPlasma(
        electron_temperature=5.0, plasma_potential=5.0, initial_plasma_x=0.5e-3
    )
    model = BeamOptics(_diode(three=True, plasma=plasma, iterations=6)).run()
    history = model.convergence_history
    assert history[-1]["max_potential_change"] < history[1]["max_potential_change"]
    assert history[-1]["newton_iterations"] >= 1
    assert model.stats["transmitted"] > 0


def test_beams_on_surfaces_share_the_area() -> None:
    particles, j_over_v = beam_on_surface(
        400, 20.0, 1.0, 1.0, 100.0, 0.0, 0.0, (0.0, 1e-3, 2e-3), (4e-3, 1e-3)
    )
    assert len(particles) == 400
    assert particles.current.sum() == pytest.approx(20.0 * 4e-6)
    y, z = particles.position[1], particles.position[2]
    assert np.all(np.abs(y - 1e-3) < 2e-3) and np.all(np.abs(z - 2e-3) < 0.5e-3)
    assert j_over_v == pytest.approx(
        20.0 / np.sqrt(2 * 100 * 1.602176634e-19 / 1.66053906892e-27)
    )
    disc, _ = beam_on_surface(
        1000, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0, (0, 0, 0), (2e-3,), "disc"
    )
    r = np.hypot(disc.position[1], disc.position[2])
    assert r.max() < 2e-3
    # equal areas: the share inside half the radius is a quarter
    assert np.mean(r < 1e-3) == pytest.approx(0.25, abs=2e-3)
    with pytest.raises(ValueError, match="shape"):
        beam_on_surface(10, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, (0, 0, 0), (1e-3,), "hexagon")


def test_the_configuration_checks_the_dimension() -> None:
    config = _diode(three=True)
    with pytest.raises(ValueError, match="line beams"):
        config.replace(beams=[_diode(three=False).beams[0]])
    with pytest.raises(ValueError, match="3D grid"):
        _diode(three=False).replace(beams=[config.beams[0]])
    with pytest.raises(ValueError, match="2D shape"):
        config.replace(objects=_diode(three=False).objects)
    with pytest.raises(ValueError, match="z faces"):
        config.replace(bcs_field=_diode(three=False).bcs_field)
    with pytest.raises(ValueError, match=r"needs \(x, y, z\) corners"):
        config.replace(
            refinement=[{"level": 1, "lower": (0.0, 0.0), "upper": (1e-3, 1e-3)}]
        )
    with pytest.raises(ValueError, match=r"has \(x, y\) corners"):
        _diode(three=False).replace(
            refinement=[{"level": 1, "lower": (0, 0, 0), "upper": (1e-3, 1e-3, 1e-3)}]
        )
    with pytest.raises(ValueError, match="symmetry plane is"):
        config.replace(symmetry_planes=["x_min"])
    with pytest.raises(ValueError, match="for a 3D grid"):
        _diode(three=False).replace(symmetry_planes=["y_min"])
    with pytest.raises(ValueError, match="cartesian"):
        GridParams(ndim=3, coordinates="cylindrical")
    round_trip = BeamOpticsConfig.from_dict(
        config.replace(symmetry_planes=["z_min"]).to_dict()
    )
    assert round_trip.symmetry_planes == ["z_min"]
    assert isinstance(round_trip.beams[0], BeamSpecies3D)
    assert round_trip.grid.ndim == 3
    with pytest.raises(NotImplementedError, match="3D"):
        BeamOptics(config).run().surface_fields()
