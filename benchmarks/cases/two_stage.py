"""A two-gap planar accelerator, protons from 10 eV to 50 keV (IBSimu's ``sim_two_stage_accel``)."""

from common import Case, Electrode, box

CENTRE, GAP = 0.02, 0.005


def pair(name: str, x0: float, x1: float, potential: float) -> tuple[Electrode, ...]:
    """Return an electrode as a plate above and one below a 10 mm aperture."""
    return (
        Electrode(
            f"{name}_top", box(x0, x1, CENTRE + GAP, 2 * CENTRE + 0.01), potential
        ),
        Electrode(f"{name}_bottom", box(x0, x1, -0.01, CENTRE - GAP), potential),
    )


CASE = Case(
    name="two_stage",
    title="Two-gap planar accelerator, 10 eV to 50 keV",
    source="sim_two_stage_accel.py",
    cylindrical=False,
    nodes=(201, 41),
    h=1.0e-3,
    particles=2000,
    iterations=5,
    mass=1.0,
    symmetric=False,
    electrodes=(
        *pair("anode", -0.01, 0.01, 50_000.0),
        *pair("intermediate", 0.09, 0.11, 10_000.0),
        *pair("ground", 0.19, 0.21, 0.0),
    ),
    faces=(None, None, None, None),
    # the example's emitting line runs from (1 mm, 17 mm) to (0, 23 mm)
    beam={
        "current_density": 1.0e-3,
        "energy": 10.0,
        "start": (1.0e-3, 0.017),
        "end": (0.0, 0.023),
    },
    tolerance={
        "current": 0.01,
        "mean": 0.001,
        "mean_angle": 0.02,
        "rms": 0.02,
        "rms_angle": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    notes="2000 trajectories instead of the example's 200.",
)
