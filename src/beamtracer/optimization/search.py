"""Searches over a design: SciPy's differential evolution, and brute-force scans.

Both evaluate through the same machinery: the designs of a batch (a generation of the
evolution, or a scan) that are already in the record are taken from it, the others are
evaluated, serially or in a pool of ``workers`` processes, and recorded in the main
process. A process pool pickles the :class:`~beamtracer.optimization.Problem` for
every evaluation, so its ``design.build`` and ``function`` must be picklable and
importable by the workers (defined in a module, not in a notebook or a script's
``__main__`` when the pool spawns its processes, the default on macOS and Windows).
A crashed worker or an evaluation longer than ``timeout`` becomes a failed evaluation
(see :mod:`beamtracer.optimization.pool`).

Constraints follow SciPy's rules for differential evolution (Lampinen's): a feasible
design beats an infeasible one, two feasible ones compare by the objective, and two
infeasible ones by their constraint violations. An evaluation that failed or did not
converge counts as infeasible with an infinite violation, so it never wins.
"""

from __future__ import annotations

import inspect
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult, differential_evolution

from beamtracer.optimization.design import Evaluation, Problem
from beamtracer.optimization.pool import WorkerPool
from beamtracer.optimization.record import Record, to_dataset

#: The constraint value DE sees for a constraint of a failed or unconverged evaluation.
FAILED = math.inf


@dataclass
class Result:
    """The outcome of a search.

    ``best`` is the best evaluation of this search (see
    :meth:`~beamtracer.optimization.Evaluation.rank`): feasible if any was, ``None``
    without evaluations. ``evaluations`` are those the search asked for, in order,
    including those taken from the record; ``evaluated`` counts the new ones and
    ``reused`` those taken from the record (recorded by an earlier search, or earlier in
    this one). ``scipy`` is the
    :class:`~scipy.optimize.OptimizeResult` of differential evolution (``None`` for a
    scan).
    """

    problem: Problem
    best: Evaluation | None
    evaluations: list[Evaluation] = field(default_factory=list)
    evaluated: int = 0
    reused: int = 0
    record: Path | None = None
    scipy: OptimizeResult | None = None

    @property
    def x(self) -> dict[str, float]:
        """Return the parameters of the best evaluation."""
        if self.best is None:
            raise ValueError("the search has no evaluations")
        return dict(self.best.parameters)

    @property
    def fun(self) -> float:
        """Return the objective of the best evaluation."""
        if self.best is None:
            raise ValueError("the search has no evaluations")
        return self.best.objectives[self.problem.objective]

    def to_dataset(self) -> Any:
        """Return this search's evaluations as an :class:`xarray.Dataset`."""
        return to_dataset(self.evaluations)


class _Evaluator:
    """Evaluates batches of designs once each: from the record, serially or in a pool."""

    def __init__(self, problem: Problem, record: Record, pool: WorkerPool) -> None:
        self.problem = problem
        self.record = record
        self.pool = pool
        #: Every design asked for, in order (repeats included).
        self.evaluations: list[Evaluation] = []
        self.evaluated = 0
        self.reused = 0
        self._latest: dict[str, Evaluation] = {}

    def __call__(self, points: np.ndarray) -> list[Evaluation]:
        """Return the evaluations of the designs ``points`` ``(S, N)``."""
        design = self.problem.design
        points = np.atleast_2d(np.asarray(points, dtype=float))
        keys = [design.key(x) for x in points]
        results: dict[str, Evaluation] = {}
        todo: dict[str, np.ndarray] = {}
        for key, x in zip(keys, points, strict=True):
            known = self.record.get(key)
            if known is not None:
                results[key] = known
                self.reused += 1
            elif key in todo:  # a repeat within the batch: evaluated once
                self.reused += 1
            else:
                todo[key] = x
        if todo:
            new = self.pool.map([design.values(x) for x in todo.values()])
            self.record.append(new)
            self.evaluated += len(new)
            results.update(zip(todo, new, strict=True))
        batch = [results[key] for key in keys]
        self.evaluations.extend(batch)
        self._latest = dict(zip(keys, batch, strict=True))
        return batch

    def again(self, points: np.ndarray) -> list[Evaluation]:
        """Return the evaluations of designs of the last batch, without counting them.

        Differential evolution asks for the constraints of a generation first and then
        for the objective of its feasible members; other designs are evaluated.
        """
        design = self.problem.design
        points = np.atleast_2d(np.asarray(points, dtype=float))
        keys = [design.key(x) for x in points]
        if all(key in self._latest for key in keys):
            return [self._latest[key] for key in keys]
        return self(points)

    def constraints(self, evaluation: Evaluation) -> list[float]:
        """Return the values DE sees: the constraints, and 1 for a penalised point."""
        if not evaluation.ok:
            return [FAILED] * len(self.problem.constraints) + [1.0]
        values = [evaluation.constraints[name] for name in self.problem.constraints]
        return [FAILED if math.isnan(v) else v for v in values] + [0.0]


def _best(problem: Problem, evaluations: Sequence[Evaluation]) -> Evaluation | None:
    if not evaluations:
        return None
    return min(evaluations, key=lambda e: e.rank(problem.objective))


def optimize(
    problem: Problem,
    *,
    seed: int | None = None,
    workers: int = 1,
    maxiter: int = 100,
    popsize: int = 15,
    record: str | Path | None = None,
    tol: float = 0.01,
    atol: float = 0.0,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    init: str | np.ndarray = "latinhypercube",
    x0: Mapping[str, float] | Sequence[float] | None = None,
    callback: Callable[..., Any] | None = None,
    mp_context: Any = None,
    timeout: float | None = None,
    **options: Any,
) -> Result:
    """Minimize the problem's first objective with SciPy's differential evolution.

    The population of every generation is evaluated as one batch, in a pool of
    ``workers`` processes (``-1``: one per CPU) started with ``mp_context`` (see
    :mod:`multiprocessing`), each evaluation stopped after ``timeout`` seconds, and
    recorded. With a ``record`` path an existing record is
    resumed: with the same ``seed`` and settings the evolution proposes the same
    designs again and takes the recorded ones from the file, so an interrupted or
    extended search (a larger ``maxiter``) continues where it stopped. The problem's
    constraints are SciPy :class:`~scipy.optimize.NonlinearConstraint` (``<= 0``), with
    one more that is violated by failed and unconverged evaluations.

    ``seed``, ``maxiter``, ``popsize``, ``tol``, ``atol``, ``mutation``,
    ``recombination``, ``init``, ``x0`` and ``callback`` are those of
    :func:`scipy.optimize.differential_evolution`; further ``options`` are passed on,
    except ``polish``, ``updating``, ``vectorized`` and ``constraints``: the evolution
    runs a generation at a time and does not polish its result (a gradient search on a
    model with discretisation noise rarely helps).
    """
    fixed = {"polish", "updating", "vectorized", "constraints", "workers"}
    if fixed & set(options):
        raise TypeError(f"optimize sets {sorted(fixed & set(options))} itself")
    design = problem.design
    recorded = Record(design, record)  # before the pool: it may refuse the file
    pool = WorkerPool(problem, workers, mp_context, timeout)
    evaluator = _Evaluator(problem, recorded, pool)

    def objective(xt: np.ndarray) -> np.ndarray:
        evaluations = evaluator.again(np.asarray(xt).T.reshape(-1, len(design)))
        return np.array(
            [e.objectives[problem.objective] if e.ok else math.inf for e in evaluations]
        )

    def constraint(xt: np.ndarray) -> np.ndarray:
        evaluations = evaluator(np.asarray(xt).T.reshape(-1, len(design)))
        return np.array([evaluator.constraints(e) for e in evaluations]).T

    count = len(problem.constraints) + 1
    settings: dict[str, Any] = {
        "maxiter": maxiter,
        "popsize": popsize,
        "tol": tol,
        "atol": atol,
        "mutation": mutation,
        "recombination": recombination,
        "init": init,
        "callback": callback,
        "polish": False,
        "updating": "deferred",
        "vectorized": True,
        "constraints": NonlinearConstraint(
            constraint, -np.inf, np.zeros(count), keep_feasible=False
        ),
        **options,
    }
    if x0 is not None:
        settings["x0"] = design.vector(x0)
    # SciPy 1.15 renamed ``seed`` to ``rng``
    seed_name = (
        "rng"
        if "rng" in inspect.signature(differential_evolution).parameters
        else "seed"
    )
    settings[seed_name] = seed
    with pool:
        scipy_result = differential_evolution(objective, design.bounds, **settings)
    return Result(
        problem=problem,
        best=_best(problem, evaluator.evaluations),
        evaluations=evaluator.evaluations,
        evaluated=evaluator.evaluated,
        reused=evaluator.reused,
        record=evaluator.record.path,
        scipy=scipy_result,
    )


def scan(
    problem: Problem,
    grid: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray,
    *,
    workers: int = 1,
    record: str | Path | None = None,
    mp_context: Any = None,
    timeout: float | None = None,
) -> Result:
    """Evaluate a grid of designs, with the evaluation and record of :func:`optimize`.

    ``grid`` is ``{name: values}`` for every parameter (all combinations, the last
    parameter varying fastest) or an array of designs ``(S, N)``. The evaluations are
    in the order of the grid, designs already in the record taken from it.
    """
    design = problem.design
    if isinstance(grid, Mapping):
        missing = set(design.names) - set(grid)
        unknown = set(grid) - set(design.names)
        if missing or unknown:
            raise KeyError(
                f"the grid needs exactly the parameters {list(design.names)}; "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        points = np.array(
            list(itertools.product(*(grid[name] for name in design.names))),
            dtype=float,
        )
    else:
        points = np.atleast_2d(np.asarray(grid, dtype=float))
    recorded = Record(design, record)
    pool = WorkerPool(problem, workers, mp_context, timeout)
    evaluator = _Evaluator(problem, recorded, pool)
    with pool:
        evaluations = evaluator(points)
    return Result(
        problem=problem,
        best=_best(problem, evaluations),
        evaluations=evaluations,
        evaluated=evaluator.evaluated,
        reused=evaluator.reused,
        record=evaluator.record.path,
    )
