"""A cylindrical einzel lens in a beam pipe, 30 keV protons (IBSimu's ``sim_cyl_lens``)."""

from common import Case, Electrode, box

CASE = Case(
    name="lens_cyl",
    title="Cylindrical einzel lens, 30 keV protons, space charge",
    source="sim_cyl_lens.py",
    cylindrical=True,
    nodes=(201, 41),
    h=0.5e-3,
    particles=2000,
    iterations=2,
    mass=1.0,
    symmetric=True,
    electrodes=(
        Electrode("pipe", box(-0.01, 0.11, 0.015, 0.03), 0.0),
        Electrode("lens", box(0.04, 0.06, 0.01, 0.03), -15_000.0),
    ),
    faces=(None, None, None, None),
    beam={
        "current_density": 100.0,
        "energy": 30_000.0,
        "start": (0.0, 0.0),
        "end": (0.0, 8.0e-3),
    },
    tolerance={
        "current": 0.01,
        "rms": 0.01,
        "rms_angle": 0.02,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    notes="2000 trajectories instead of the example's 500.",
)
