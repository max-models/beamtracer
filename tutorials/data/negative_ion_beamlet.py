"""Make the records of the tutorial "Optimising a negative-ion beamlet".

Run from the repository root (about 30 minutes with two workers)::

    python tutorials/data/negative_ion_beamlet.py

``resolution`` solves eight designs on the example's mesh and on the coarse one of the
search, ``search`` runs the search; each writes a JSON-lines record next to this script,
which a rerun resumes from (delete a record to evaluate it again).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from beamtracer.optimization import optimize, scan
from beamtracer.optimization.beamlet import COARSE, NegativeIonBeamlet

HERE = Path(__file__).parent
#: The search space: extraction voltage [V], EG aperture radius and PG-EG gap [m].
BOUNDS = {
    "extraction_voltage": (3e3, 10e3),
    "eg_radius": (2e-3, 4e-3),
    "gap": (3e-3, 6e-3),
}
#: The settings of the search.
SEARCH = {"seed": 1, "popsize": 5, "maxiter": 8}
WORKERS = 2


def resolution_points() -> np.ndarray:
    """Return eight designs spread over a part of the search space (a Latin hypercube)."""
    lower = np.array([3e3, 2e-3, 3e-3])
    upper = np.array([10e3, 4e-3, 6e-3])
    unit = qmc.LatinHypercube(d=3, seed=3).random(8)
    points = lower + unit * (upper - lower)
    return np.column_stack([points[:, 0].round(), points[:, 1:].round(6)])


def resolution() -> None:
    """Solve the designs of :func:`resolution_points` on the fine and the coarse mesh."""
    points = resolution_points()
    for name, beamlet in (
        ("fine", NegativeIonBeamlet()),
        ("coarse", NegativeIonBeamlet(**COARSE)),
    ):
        start = time.perf_counter()
        scan(
            beamlet.problem(BOUNDS),
            points,
            workers=WORKERS,
            record=HERE / f"beamlet_resolution_{name}.jsonl",
        )
        print(f"{name}: {time.perf_counter() - start:.0f} s")


def search() -> None:
    """Run the search on the coarse mesh."""
    start = time.perf_counter()
    result = optimize(
        NegativeIonBeamlet(**COARSE).problem(BOUNDS),
        workers=WORKERS,
        record=HERE / "beamlet_search.jsonl",
        **SEARCH,
    )
    print(
        f"search: {result.evaluated} evaluations in {time.perf_counter() - start:.0f} s, "
        f"best {result.x}, divergence {result.fun * 1e3:.1f} mrad"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", nargs="*", default=["resolution", "search"])
    for what in parser.parse_args().what:
        {"resolution": resolution, "search": search}[what]()
