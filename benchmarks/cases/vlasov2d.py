"""Planar proton extraction with three electrodes and space charge (IBSimu's ``vlasov2d``)."""

from common import Case, Electrode, box

CASE = Case(
    name="vlasov2d",
    title="Planar proton extraction, three electrodes, space charge",
    source="vlasov2d.py",
    cylindrical=False,
    nodes=(241, 101),
    h=0.5e-3,
    particles=10_000,
    iterations=5,
    mass=1.0,
    symmetric=True,
    electrodes=(
        Electrode("plasma", box(-0.01, 0.02, 0.018, 0.06), -3000.0),
        Electrode("puller", box(0.03, 0.04, 0.02, 0.06), -14_000.0),
        Electrode(
            "ground",
            ((0.06, 0.04), (0.08, 0.03), (0.13, 0.03), (0.13, 0.06), (0.06, 0.06)),
            -1000.0,
        ),
    ),
    faces=(-3000.0, -1000.0, None, None),
    beam={
        "current_density": 50.0,
        "energy": 3000.0,
        "start": (0.0, 0.0),
        "end": (0.0, 0.012),
    },
    tolerance={
        "current": 0.01,
        "rms": 0.01,
        "rms_angle": 0.01,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    solver="bicgstab",
)
