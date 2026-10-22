"""Tests of the optimisation core on cheap analytic functions."""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pytest

from beamtracer.optimization import (
    Design,
    Evaluation,
    Parameter,
    Problem,
    Record,
    load_record,
    optimize,
    scan,
)

#: The designs the analytic functions were called with (in this process).
CALLS: list[dict[str, float]] = []


def bowl(values: dict[str, float]) -> Evaluation:
    """A paraboloid with its minimum 0 at (1, -2)."""
    CALLS.append(dict(values))
    x, y = values["x"], values["y"]
    return Evaluation(objectives={"objective": (x - 1.0) ** 2 + (y + 2.0) ** 2})


def disc(values: dict[str, float]) -> Evaluation:
    """The distance to (2, 2), constrained to x + y <= 2: the optimum is (1, 1)."""
    x, y = values["x"], values["y"]
    return Evaluation(
        objectives={"objective": (x - 2.0) ** 2 + (y - 2.0) ** 2},
        constraints={"line": x + y - 2.0},
    )


def treacherous(values: dict[str, float]) -> Evaluation:
    """x^2, but unconverged with a tempting -100 near 0 and raising for x > 1.5."""
    x = values["x"]
    if x > 1.5:
        raise RuntimeError("the model blew up")
    if abs(x) < 0.5:
        return Evaluation(objectives={"objective": -100.0}, converged=False)
    return Evaluation(objectives={"objective": x * x}, diagnostics={"note": "fine"})


def fragile(values: dict[str, float]) -> Evaluation:
    """x^2, but the process dies for x > 1 and hangs for x < -1."""
    x = values["x"]
    if x > 1.0:
        os._exit(3)
    if x < -1.0:
        time.sleep(60.0)
    return Evaluation(objectives={"objective": x * x})


def plane(values: dict[str, float]) -> dict[str, float]:
    """Not an Evaluation."""
    return dict(values)


def _square() -> Design:
    return Design((Parameter("x", -3.0, 3.0), Parameter("y", -3.0, 3.0)))


# -- the design ---------------------------------------------------------------------------


def test_parameters_need_ordered_finite_bounds() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        Parameter("x", 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        Parameter("x", 0.0, math.inf)
    with pytest.raises(ValueError, match="unique"):
        Design((Parameter("x", 0, 1), Parameter("x", 0, 2)))


def test_design_values_vectors_and_keys() -> None:
    design = _square()
    assert design.values([1.0, 2.0]) == {"x": 1.0, "y": 2.0}
    assert design.vector({"y": 2.0, "x": 1.0}).tolist() == [1.0, 2.0]
    assert design.normalized([-3.0, 3.0]).tolist() == [0.0, 1.0]
    assert design.key([1.0, 2.0]) == design.key({"x": 1.0, "y": 2.0 + 1e-13})
    assert design.key([1.0, 2.0]) != design.key([1.0, 2.0 + 1e-6])
    with pytest.raises(KeyError, match="missing"):
        design.vector({"x": 1.0})
    # without build, the evaluation function gets the values
    assert design.configuration([1.0, 2.0]) == {"x": 1.0, "y": 2.0}


def test_problem_evaluate_never_raises() -> None:
    design = Design((Parameter("x", -2.0, 2.0),))
    problem = Problem(design, treacherous)
    failed = problem.evaluate({"x": 1.8})
    assert not failed.ok and not failed.feasible
    assert failed.failure == "RuntimeError: the model blew up"
    assert math.isnan(failed.objectives["objective"])
    assert failed.parameters == {"x": 1.8} and failed.wall_time >= 0.0
    unconverged = problem.evaluate({"x": 0.1})
    assert not unconverged.ok and unconverged.rank("objective")[0] == 2
    assert problem.evaluate({"x": 1.0}).rank("objective") == (0, 1.0)

    not_an_evaluation = Problem(design, plane).evaluate({"x": 0.0})
    assert "not an Evaluation" in str(not_an_evaluation.failure)
    missing = Problem(design, treacherous, constraints="c").evaluate({"x": 1.0})
    assert "missing ['c']" in str(missing.failure)


def test_constraints_the_problem_does_not_enforce_are_diagnostics() -> None:
    design = _square()
    free = Problem(design, disc).evaluate({"x": 3.0, "y": 3.0})
    assert free.feasible and free.constraints == {}
    assert free.diagnostics["line"] == 4.0
    bound = Problem(design, disc, constraints="line").evaluate({"x": 3.0, "y": 3.0})
    assert not bound.feasible and bound.violation == 4.0
    assert bound.rank("objective") == (1, 4.0)


# -- the searches -------------------------------------------------------------------------


def test_the_search_finds_a_known_minimum() -> None:
    result = optimize(Problem(_square(), bowl), seed=3, maxiter=200, tol=1e-8)
    assert result.x["x"] == pytest.approx(1.0, abs=1e-3)
    assert result.x["y"] == pytest.approx(-2.0, abs=1e-3)
    assert result.fun < 1e-6
    assert result.best is not None and result.best.feasible
    assert result.scipy is not None and result.scipy.x == pytest.approx(
        [1.0, -2.0], abs=1e-3
    )
    assert result.evaluated == len(result.evaluations) - result.reused


def test_constraints_are_respected() -> None:
    problem = Problem(_square(), disc, constraints="line")
    result = optimize(problem, seed=1, maxiter=300, tol=1e-8)
    best = result.best
    assert best is not None and best.feasible
    assert best.constraints["line"] <= 0.0
    assert result.x["x"] == pytest.approx(1.0, abs=2e-2)
    assert result.x["y"] == pytest.approx(1.0, abs=2e-2)
    # the unconstrained minimum (2, 2) was seen but is infeasible
    assert any(not e.feasible for e in result.evaluations)


def test_a_penalised_point_never_wins() -> None:
    problem = Problem(Design((Parameter("x", -2.0, 2.0),)), treacherous)
    result = optimize(problem, seed=0, popsize=10, maxiter=60)
    assert result.best is not None and result.best.feasible
    assert abs(result.x["x"]) >= 0.5
    assert result.fun == pytest.approx(0.25, rel=1e-2)
    failures = [e for e in result.evaluations if not e.ok]
    assert any(e.failure for e in failures) and any(not e.converged for e in failures)
    assert all(e.objectives["objective"] >= 0.25 for e in result.evaluations if e.ok)

    grid = scan(problem, {"x": np.linspace(-2.0, 2.0, 41)})
    assert grid.best is not None and grid.fun == pytest.approx(0.25)
    assert abs(grid.x["x"]) == pytest.approx(0.5)


def test_a_resumed_run_continues_from_its_record(tmp_path) -> None:
    path = tmp_path / "bowl.jsonl"
    problem = Problem(_square(), bowl)
    CALLS.clear()
    first = optimize(problem, seed=7, maxiter=4, popsize=5, tol=0.0, record=path)
    assert len(CALLS) == first.evaluated == len(Record(problem.design, path))
    recorded = {problem.design.key(e.parameters) for e in Record(problem.design, path)}

    CALLS.clear()
    again = optimize(problem, seed=7, maxiter=4, popsize=5, tol=0.0, record=path)
    assert again.evaluated == 0 and not CALLS
    assert again.x == first.x

    CALLS.clear()
    longer = optimize(problem, seed=7, maxiter=8, popsize=5, tol=0.0, record=path)
    assert longer.reused >= first.evaluated
    assert len(CALLS) == longer.evaluated > 0
    assert not recorded & {problem.design.key(c) for c in CALLS}
    assert longer.fun <= first.fun
    assert len(Record(problem.design, path)) == first.evaluated + longer.evaluated


def test_a_record_of_another_design_is_refused(tmp_path) -> None:
    path = tmp_path / "bowl.jsonl"
    scan(Problem(_square(), bowl), [[0.0, 0.0]], record=path)
    other = Design((Parameter("z", 0.0, 1.0),))
    with pytest.raises(ValueError, match="parameters"):
        Record(other, path)


def test_process_pool_workers(tmp_path) -> None:
    problem = Problem(_square(), disc, constraints="line")
    serial = optimize(problem, seed=5, maxiter=15, popsize=6)
    pooled = optimize(
        problem, seed=5, maxiter=15, popsize=6, workers=2, record=tmp_path / "r.jsonl"
    )
    assert pooled.x == serial.x
    assert pooled.evaluated == serial.evaluated
    grid = scan(problem, {"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]}, workers=2)
    assert [e.parameters for e in grid.evaluations][:2] == [
        {"x": 0.0, "y": 0.0},
        {"x": 0.0, "y": 1.0},
    ]
    assert grid.x == {"x": 1.0, "y": 1.0}


def test_the_record_loads_back(tmp_path) -> None:
    path = tmp_path / "treacherous.jsonl"
    problem = Problem(Design((Parameter("x", -2.0, 2.0),)), treacherous)
    result = scan(problem, {"x": [-1.0, 0.0, 1.0, 2.0]}, record=path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["parameters"]["x"] for line in lines] == [-1.0, 0.0, 1.0, 2.0]
    assert {"key", "objectives", "constraints", "converged", "wall_time"} <= set(
        lines[0]
    )

    data = load_record(path)
    assert data.sizes["evaluation"] == 4
    assert data.x.values.tolist() == [-1.0, 0.0, 1.0, 2.0]
    assert data.objective.attrs["kind"] == "objective"
    assert data.feasible.values.tolist() == [True, False, True, False]
    assert data.converged.values.tolist() == [True, False, True, False]
    assert data.failure.values[3] == "RuntimeError: the model blew up"
    assert data.note.values[0] == "fine"
    assert np.isnan(data.objective.values[3])
    frame = data.to_dataframe()
    assert list(frame.index) == [0, 1, 2, 3]
    assert result.to_dataset().x.values.tolist() == data.x.values.tolist()

    # a scan of the same points takes them all from the record
    again = scan(problem, {"x": [2.0, -1.0]}, record=path)
    assert again.evaluated == 0 and again.reused == 2
    assert again.evaluations[0].failure == "RuntimeError: the model blew up"


def test_a_crashed_worker_or_a_hang_is_a_failed_evaluation() -> None:
    problem = Problem(Design((Parameter("x", -2.0, 2.0),)), fragile)
    start = time.perf_counter()
    grid = scan(problem, {"x": [0.5, 1.5, -1.5, -0.5]}, workers=2, timeout=1.0)
    assert time.perf_counter() - start < 30.0
    crashed, hung = grid.evaluations[1], grid.evaluations[2]
    assert "worker process died" in str(crashed.failure)
    assert "EvaluationTimeout" in str(hung.failure)
    assert [e.ok for e in grid.evaluations] == [True, False, False, True]
    assert grid.x == {"x": 0.5} or grid.x == {"x": -0.5}
    serial = scan(problem, {"x": [-1.5]}, timeout=0.5)
    assert "EvaluationTimeout" in str(serial.evaluations[0].failure)
