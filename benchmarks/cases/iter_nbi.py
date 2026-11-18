"""A seven-electrode cylindrical accelerator to 120 keV, chamfered apertures (IBSimu's ``sim_iter_nbi``)."""

import math

from common import Case, Electrode

CENTRES = (5e-3, 20e-3, 50e-3, 80e-3, 110e-3, 140e-3, 170e-3)
VOLTAGES = (120e3, 110e3, 85e3, 65e3, 45e3, 25e3, 0.0)
THICKNESS, APERTURE, SLOPE = 6e-3, 7e-3, math.tan(math.radians(30.0))


def electrode(k: int) -> Electrode:
    """Return electrode ``k``: its aperture radius grows from 7 mm upstream at 30 degrees."""
    x0, x1 = CENTRES[k] - THICKNESS / 2, CENTRES[k] + THICKNESS / 2
    vertices = (
        (x0, APERTURE),
        (x1, APERTURE + THICKNESS * SLOPE),
        (x1, 0.02),
        (x0, 0.02),
    )
    return Electrode(f"electrode{k + 1}", vertices, VOLTAGES[k])


CASE = Case(
    name="iter_nbi",
    title="Seven-electrode cylindrical accelerator to 120 keV",
    source="sim_iter_nbi.py",
    cylindrical=True,
    nodes=(401, 21),
    h=0.5e-3,
    particles=2000,
    iterations=1,
    mass=1.0,
    symmetric=True,
    electrodes=tuple(electrode(k) for k in range(len(CENTRES))),
    faces=(None, None, None, None),
    # the example's emitting line runs from (1 mm, 0) to (0, 6.5 mm)
    beam={
        "current_density": 100.0,
        "energy": 10.0,
        "start": (1.0e-3, 0.0),
        "end": (0.0, 6.5e-3),
    },
    tolerance={
        "current": 0.05,
        "rms": 0.03,
        "rms_angle": 0.03,
        "energy": 0.001,
        "potential_rms": 0.002,
    },  # fmt: skip
    solver="bicgstab",
    notes=(
        "2000 trajectories instead of the example's 500; one field solve and trace, no space "
        "charge. IBSimu's BiCGSTAB solver instead of the example's Gauss-Seidel, which stops "
        "about 35 V short on this long mesh. The outermost transmitted trajectories graze the "
        "upstream corner of the last electrode: they pass about 0.2 mm through it (on "
        "IBSimu's own saved trajectories), so beamtracer stops them, while IBSimu, testing "
        "only its step points, lets them through; that is the 4% less current."
    ),
)
