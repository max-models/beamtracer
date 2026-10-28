"""Searches over a design: SciPy's differential evolution, brute-force scans, and checks.

:func:`recheck` evaluates the best designs of a search again with another evaluation:
the second stage of a staged search, e.g. a finer mesh or a PIC model in user code.

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
from dataclasses import dataclass, field, replace
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
    without evaluations. ``evaluations`` are the designs the search asked for: for
    :func:`optimize` each design once, in the order first asked for, with its
    ``generation``; for :func:`scan` and :func:`recheck` in the order of the grid or the
    ranking. ``evaluated`` counts the new evaluations, ``reused`` those taken from a
    record of an earlier search, and ``repeats`` the asks for a design this search had
    already asked for (SciPy checks the best design between generations), which are
    answered from memory and not listed again. ``path`` is the best evaluation after each
    generation of :func:`optimize`, from the initial population (0) to the last (empty
    for a scan). ``scipy`` is the :class:`~scipy.optimize.OptimizeResult` of
    differential evolution (``None`` for a scan).
    """

    problem: Problem
    best: Evaluation | None
    evaluations: list[Evaluation] = field(default_factory=list)
    evaluated: int = 0
    reused: int = 0
    repeats: int = 0
    path: list[Evaluation] = field(default_factory=list)
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

    def path_to_dataset(self) -> Any:
        """Return :attr:`path` as an :class:`xarray.Dataset` along ``generation``.

        The variables are those of :func:`~beamtracer.optimization.to_dataset`, the
        generation in which each best design was asked for renamed to ``found``.
        """
        data = to_dataset(self.path)
        if "generation" in data:
            data = data.rename(generation="found")
        data = data.rename(evaluation="generation")
        return data.assign_coords(generation=np.arange(len(self.path)))


class _Evaluator:
    """Evaluates batches of designs once each: from memory, the record, or the pool.

    The designs are tagged with ``generation`` in the main process, whether they were
    evaluated here or in a worker, or taken from the record.
    """

    def __init__(self, problem: Problem, record: Record, pool: WorkerPool) -> None:
        self.problem = problem
        self.record = record
        self.pool = pool
        #: The generation of the designs asked for now (``None`` outside a search).
        self.generation: int | None = None
        #: Every design asked for, once, in the order first asked for.
        self.evaluations: list[Evaluation] = []
        self.evaluated = 0
        self.reused = 0
        self.repeats = 0
        self.seen: dict[str, Evaluation] = {}
        self._latest: list[str] = []

    def again(self, points: np.ndarray) -> list[Evaluation]:
        """Return the evaluations of designs of the last batch, without counting them.

        Differential evolution asks for the constraints of a generation first and then
        for the objective of its feasible members; other designs are evaluated.
        """
        design = self.problem.design
        points = np.atleast_2d(np.asarray(points, dtype=float))
        keys = [design.key(x) for x in points]
        if set(keys) <= set(self._latest):
            return [self.seen[key] for key in keys]
        return self(points)

    def constraints(self, evaluation: Evaluation) -> list[float]:
        """Return the values DE sees: the constraints, and 1 for a penalised point."""
        if not evaluation.ok:
            return [FAILED] * len(self.problem.constraints) + [1.0]
        values = [evaluation.constraints[name] for name in self.problem.constraints]
        return [FAILED if math.isnan(v) else v for v in values] + [0.0]

    def __call__(self, points: np.ndarray) -> list[Evaluation]:
        """Return the evaluations of the designs ``points`` ``(S, N)``."""
        design = self.problem.design
        points = np.atleast_2d(np.asarray(points, dtype=float))
        keys = [design.key(x) for x in points]
        first: list[str] = []  # the designs new to this search, in order
        found: dict[str, Evaluation] = {}
        todo: dict[str, np.ndarray] = {}
        for key, x in zip(keys, points, strict=True):
            if key in self.seen or key in found or key in todo:
                self.repeats += 1
                continue
            first.append(key)
            known = self.record.get(key)
            if known is not None:
                found[key] = replace(known, generation=self.generation)
                self.reused += 1
            else:
                todo[key] = x
        if todo:
            new = [
                replace(e, generation=self.generation)
                for e in self.pool.map([design.values(x) for x in todo.values()])
            ]
            self.record.append(new)
            self.evaluated += len(new)
            found.update(zip(todo, new, strict=True))
        for key in first:
            self.seen[key] = found[key]
            self.evaluations.append(found[key])
        self._latest = keys
        return [self.seen[key] for key in keys]


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
    :mod:`multiprocessing`; default forkserver, or spawn where there is none), each evaluation stopped after ``timeout`` seconds, and
    recorded. With a ``record`` path an existing record is
    resumed: with the same ``seed`` and settings the evolution proposes the same
    designs again and takes the recorded ones from the file, so an interrupted or
    extended search (a larger ``maxiter``) continues where it stopped. The problem's
    constraints are SciPy :class:`~scipy.optimize.NonlinearConstraint` (``<= 0``), with
    one more that is violated by failed and unconverged evaluations.

    Every evaluation carries the ``generation`` that asked for it (0 the initial
    population), and :attr:`Result.path` holds the best design after each generation;
    a design asked for again (SciPy checks the best one between generations) is answered
    from memory, counted in :attr:`Result.repeats` and not listed again.

    ``seed``, ``maxiter``, ``popsize``, ``tol``, ``atol``, ``mutation``,
    ``recombination``, ``init``, ``x0`` and ``callback`` are those of
    :func:`scipy.optimize.differential_evolution` (``callback`` with either of SciPy's
    signatures, ``callback(intermediate_result)`` or ``callback(x, convergence)``);
    further ``options`` are passed on,
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

    populations = itertools.count()

    def constraint(xt: np.ndarray) -> np.ndarray:
        points = np.asarray(xt).T.reshape(-1, len(design))
        # DE asks for the constraints of a whole population at once, the initial one
        # (generation 0) and then one per generation, of at least 5 members; it asks
        # for single designs to size the constraints and to check the best one
        if len(points) > 1:
            evaluator.generation = next(populations)
        evaluations = evaluator(points)
        return np.array([evaluator.constraints(e) for e in evaluations]).T

    path: list[Evaluation] = []

    def track(intermediate_result: OptimizeResult) -> Any:
        # SciPy passes the result only to a parameter of this name
        path.append(evaluator.seen[design.key(intermediate_result.x)])
        if callback is None:
            return None
        if "intermediate_result" in inspect.signature(callback).parameters:
            return callback(intermediate_result=intermediate_result)
        return callback(
            intermediate_result.x, convergence=intermediate_result.convergence
        )

    evaluator.generation = 0
    count = len(problem.constraints) + 1
    settings: dict[str, Any] = {
        "maxiter": maxiter,
        "popsize": popsize,
        "tol": tol,
        "atol": atol,
        "mutation": mutation,
        "recombination": recombination,
        "init": init,
        "callback": track,
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
    initial = _best(problem, [e for e in evaluator.evaluations if e.generation == 0])
    return Result(
        problem=problem,
        best=_best(problem, evaluator.evaluations),
        evaluations=evaluator.evaluations,
        evaluated=evaluator.evaluated,
        reused=evaluator.reused,
        repeats=evaluator.repeats,
        path=[] if initial is None else [initial, *path],
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
        repeats=evaluator.repeats,
        record=evaluator.record.path,
    )


def recheck(
    result: Result,
    check: Problem | Callable[[Any], Evaluation],
    *,
    top: int = 3,
    workers: int = 1,
    record: str | Path | None = None,
    mp_context: Any = None,
    timeout: float | None = None,
) -> Result:
    """Evaluate the ``top`` best designs of a search again, with another evaluation.

    The second stage of a staged search: a cheap model in the search, an expensive one
    (a finer mesh, a time-dependent PIC run of the source, a code beamtracer does not
    import) for the few best designs. The designs are the ``top`` best distinct ones of
    ``result`` by :meth:`~beamtracer.optimization.Evaluation.rank`: feasible ones by the
    objective, then infeasible ones by their constraint violation; failed and
    unconverged ones are never checked.

    ``check`` is a :class:`~beamtracer.optimization.Problem` over the same parameters,
    with its own build, evaluation, objectives and constraints; or a function, which
    then evaluates the configuration ``result.problem.design`` builds and must return
    the objectives and constraints of ``result.problem``. The evaluations run like a
    :func:`scan` (``workers``, ``mp_context``, ``timeout``), recorded in ``record``,
    which a rerun resumes from. The returned :class:`Result` has the evaluations in the
    order of the search's ranking, and its ``best`` is the best by the check.
    """
    if top < 1:
        raise ValueError(f"top must be positive, got {top}")
    if not isinstance(check, Problem):
        check = replace(result.problem, function=check)
    if set(check.design.names) != set(result.problem.design.names):
        raise ValueError(
            f"the check has the parameters {list(check.design.names)}, the search "
            f"{list(result.problem.design.names)}"
        )
    objective = result.problem.objective
    chosen: dict[str, np.ndarray] = {}
    for evaluation in sorted(
        (e for e in result.evaluations if e.ok), key=lambda e: e.rank(objective)
    ):
        x = check.design.vector(evaluation.parameters)
        chosen.setdefault(check.design.key(x), x)
        if len(chosen) == top:
            break
    recorded = Record(check.design, record)
    pool = WorkerPool(check, workers, mp_context, timeout)
    evaluator = _Evaluator(check, recorded, pool)
    with pool:
        evaluations = evaluator(np.array(list(chosen.values()))) if chosen else []
    return Result(
        problem=check,
        best=_best(check, evaluations),
        evaluations=evaluations,
        evaluated=evaluator.evaluated,
        reused=evaluator.reused,
        repeats=evaluator.repeats,
        record=evaluator.record.path,
    )
