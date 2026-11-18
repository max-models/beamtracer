"""An electrostatic quadrupole for a 50 keV proton beam in 3D (IBSimu's ``sim_3d_quad``)."""

from typing import Any

from common3d import Box, Case3D


def make(particles: int) -> Any:
    """A parallel 50 keV proton beam from a 10 mm square at x = 0, 100 A/m^2."""
    from beamtracer.solver.tracer3d import beam_on_surface

    beam, _ = beam_on_surface(
        particles, 100.0, 1.0, 1.0, 50_000.0, 0.0, 0.0, (0.0, 0.0, 0.0), (0.01, 0.01)
    )
    return beam.position, beam.velocity, beam.current


CASE = Case3D(
    name="quad3d",
    title="Electrostatic quadrupole, 50 keV protons",
    source="sim_3d_quad.py",
    nodes=(101, 21, 21),
    origin=(0.0, -10e-3, -10e-3),
    h=1e-3,
    particles=1000,
    iterations=1,
    mass=1.0,
    symmetry_planes=(),
    # the plate pairs across y at +1 kV and across z at -1 kV; where they overlap in the
    # corners the z pair, listed last, holds the nodes (as in the example)
    electrodes=(
        Box("y_plus", (0.04, 0.06), (0.008, 0.011), (-0.015, 0.015), 1000.0),
        Box("y_minus", (0.04, 0.06), (-0.011, -0.008), (-0.015, 0.015), 1000.0),
        Box("z_plus", (0.04, 0.06), (-0.015, 0.015), (0.008, 0.011), -1000.0),
        Box("z_minus", (0.04, 0.06), (-0.015, 0.015), (-0.011, -0.008), -1000.0),
    ),
    faces=(None,) * 6,
    beam={"make": make},
    tolerance={
        "current": 0.01,
        "rms_y": 0.01,
        "rms_z": 0.01,
        "rms_angle_y": 0.02,
        "rms_angle_z": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    notes=(
        "The same 992 trajectories (a 32 x 31 grid on the square) in both codes; IBSimu "
        "samples its own. The plates are boxes reaching beyond the mesh."
    ),
)
