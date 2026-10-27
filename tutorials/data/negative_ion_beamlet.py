"""Make the records of the tutorial "Optimising a negative-ion beamlet".

Run from the repository root (about an hour with two workers)::

    python tutorials/data/negative_ion_beamlet.py

``resolution`` solves eight designs on the example's mesh and on the coarse one of the
search, ``search`` runs the search over three parameters and ``full`` the one over the
full design, both on the coarse mesh; ``recheck`` solves the best designs of both again
on the example's mesh. Each writes a JSON-lines record next to this script, which a rerun
resumes from (delete a record to evaluate it again).

The full design at a larger budget, e.g. on a workstation::

    python tutorials/data/negative_ion_beamlet.py full --popsize 15 --maxiter 100 --workers -1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from beamtracer.optimization import Result, optimize, recheck, scan
from beamtracer.optimization.beamlet import (
    COARSE,
    FULL_CONSTRAINTS,
    FULL_DESIGN,
    NegativeIonBeamlet,
)

HERE = Path(__file__).parent
#: The search space: extraction voltage [V], EG aperture radius and PG-EG gap [m].
BOUNDS = {
    "extraction_voltage": (3e3, 10e3),
    "eg_radius": (2e-3, 4e-3),
    "gap": (3e-3, 6e-3),
}
#: The settings of the search, and of the search over the full design.
SEARCH = {"seed": 1, "popsize": 5, "maxiter": 8}
FULL = {"seed": 1, "popsize": 3, "maxiter": 5}
#: The best designs of each search solved again on the example's mesh.
TOP = 5
WORKERS = 2


def resolution_points() -> np.ndarray:
    """Return eight designs spread over a part of the search space (a Latin hypercube)."""
    lower = np.array([3e3, 2e-3, 3e-3])
    upper = np.array([10e3, 4e-3, 6e-3])
    unit = qmc.LatinHypercube(d=3, seed=3).random(8)
    points = lower + unit * (upper - lower)
    return np.column_stack([points[:, 0].round(), points[:, 1:].round(6)])


def resolution(workers: int = WORKERS) -> None:
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
            workers=workers,
            record=HERE / f"beamlet_resolution_{name}.jsonl",
        )
        print(f"{name}: {time.perf_counter() - start:.0f} s")


def report(name: str, result: Result, start: float) -> None:
    """Print what a search found."""
    best = result.best
    found = (
        f"best {result.x}, divergence {result.fun * 1e3:.1f} mrad"
        if best is not None and best.feasible
        else "nothing feasible"
    )
    print(
        f"{name}: {result.evaluated} new evaluations in "
        f"{time.perf_counter() - start:.0f} s, {found}"
    )


def search(workers: int = WORKERS, **settings: int) -> Result:
    """Run (or resume) the search over three parameters on the coarse mesh."""
    start = time.perf_counter()
    result = optimize(
        NegativeIonBeamlet(**COARSE).problem(BOUNDS),
        workers=workers,
        record=HERE / "beamlet_search.jsonl",
        **{**SEARCH, **settings},
    )
    report("search", result, start)
    return result


def full(workers: int = WORKERS, **settings: int) -> Result:
    """Run (or resume) the search over the full design on the coarse mesh."""
    start = time.perf_counter()
    result = optimize(
        NegativeIonBeamlet(**COARSE).problem(FULL_DESIGN, FULL_CONSTRAINTS),
        workers=workers,
        record=HERE / "beamlet_full_design.jsonl",
        **{**FULL, **settings},
    )
    report("full design", result, start)
    return result


def check(workers: int = WORKERS, **settings: int) -> None:
    """Solve the best designs of both searches again on the example's mesh.

    ``settings`` are those of the full design's search.
    """
    fine = NegativeIonBeamlet()
    for name, result, problem in (
        ("search", search(workers), fine.problem(BOUNDS)),
        (
            "full_design",
            full(workers, **settings),
            fine.problem(FULL_DESIGN, FULL_CONSTRAINTS),
        ),
    ):
        start = time.perf_counter()
        checked = recheck(
            result,
            problem,
            top=TOP,
            workers=workers,
            record=HERE / f"beamlet_{name}_fine.jsonl",
        )
        report(f"{name} on the fine mesh", checked, start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = {"resolution": resolution, "search": search, "full": full, "recheck": check}
    parser.add_argument(
        "steps", nargs="*", default=list(steps), help=f"some of {list(steps)}"
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--maxiter", type=int)
    args = parser.parse_args()
    settings = {
        k: v for k, v in (("popsize", args.popsize), ("maxiter", args.maxiter)) if v
    }
    for step in args.steps:
        if step == "resolution":
            resolution(args.workers)
        else:
            steps[step](args.workers, **settings)
