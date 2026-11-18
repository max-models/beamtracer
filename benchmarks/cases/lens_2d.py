"""A planar einzel lens focusing a 10 keV proton beam (IBSimu's ``sim_2d_lens``)."""

from common import Case, Electrode, box

CENTRE = 0.02


def pair(
    name: str, x0: float, x1: float, gap: float, potential: float
) -> tuple[Electrode, ...]:
    """Return an electrode as a plate above and one below an aperture ``|y - CENTRE| < gap``."""
    return (
        Electrode(
            f"{name}_top", box(x0, x1, CENTRE + gap, 2 * CENTRE + 0.01), potential
        ),
        Electrode(f"{name}_bottom", box(x0, x1, -0.01, CENTRE - gap), potential),
    )


CASE = Case(
    name="lens_2d",
    title="Planar einzel lens, 10 keV protons",
    source="sim_2d_lens.py",
    cylindrical=False,
    nodes=(101, 41),
    h=1.0e-3,
    particles=2000,
    iterations=2,
    mass=1.0,
    symmetric=False,
    electrodes=(
        *pair("left", -0.01, 0.04, 0.01, 0.0),
        *pair("middle", 0.045, 0.055, 0.005, -5000.0),
        *pair("right", 0.06, 0.11, 0.01, 0.0),
    ),
    faces=(None, None, None, None),
    beam={
        "current_density": 1.0e-3,
        "energy": 10_000.0,
        "start": (0.0, 0.015),
        "end": (0.0, 0.025),
    },  # fmt: skip
    tolerance={
        "current": 0.01,
        "mean": 0.001,
        "mean_angle": 0.02,
        "rms": 0.01,
        "rms_angle": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    notes="2000 trajectories instead of the example's 200.",
)
