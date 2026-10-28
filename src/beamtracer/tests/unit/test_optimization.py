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
    recheck,
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


def shifted(values: dict[str, float]) -> Evaluation:
    """The check of a bowl: the same paraboloid, 0.5 higher, with a constraint x <= 1."""
    CALLS.append(dict(values))
    x, y = values["x"], values["y"]
    return Evaluation(
        objectives={"objective": (x - 1.0) ** 2 + (y + 2.0) ** 2 + 0.5},
        constraints={"line": x - 1.0},
    )


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


def test_the_generations_and_the_path() -> None:
    problem = Problem(_square(), disc, constraints="line")
    seen = []
    result = optimize(
        problem,
        seed=1,
        maxiter=10,
        popsize=5,
        tol=0.0,
        callback=lambda x, convergence: seen.append(x),
    )
    assert result.scipy is not None and result.scipy.nit == 10
    generations = [e.generation for e in result.evaluations]
    assert generations == sorted(generations)
    assert generations.count(0) == 10  # the initial population, 5 per parameter
    assert set(generations) == set(range(11))
    assert all(generations.count(g) <= 10 for g in range(1, 11))
    # each design once; SciPy's checks of the best design are repeats
    keys = [problem.design.key(e.parameters) for e in result.evaluations]
    assert len(set(keys)) == len(keys) == result.evaluated
    assert result.repeats >= 10
    # the best after each generation, from the initial population on
    assert len(result.path) == 11 and len(seen) == 10
    assert result.path[-1] is result.best
    assert all(e.generation <= g for g, e in enumerate(result.path))
    ranks = [e.rank("objective") for e in result.path]
    assert ranks == sorted(ranks, reverse=True)
    assert [problem.design.values(x) for x in seen] == [
        e.parameters for e in result.path[1:]
    ]
    data = result.to_dataset()
    assert data.generation.values.tolist() == generations
    path = result.path_to_dataset()
    assert path.generation.values.tolist() == list(range(11))
    assert path.found.values.tolist() == [e.generation for e in result.path]
    assert path.objective.values.tolist() == [
        e.objectives["objective"] for e in result.path
    ]


def test_a_callback_with_the_intermediate_result_can_stop_the_search() -> None:
    def stop(intermediate_result):
        return intermediate_result.nit == 3

    result = optimize(Problem(_square(), bowl), seed=1, popsize=5, callback=stop)
    assert result.scipy is not None and result.scipy.nit == 3
    assert len(result.path) == 4
    assert max(e.generation for e in result.evaluations) == 3


def test_a_resumed_run_has_the_same_generations(tmp_path) -> None:
    path = tmp_path / "bowl.jsonl"
    problem = Problem(_square(), bowl)
    settings = dict(seed=7, maxiter=4, popsize=5, tol=0.0, record=path)
    first = optimize(problem, **settings)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["generation"] for line in lines] == [
        e.generation for e in first.evaluations
    ]
    assert load_record(path).generation.values.tolist() == [
        e.generation for e in first.evaluations
    ]
    # a record written before the generations were recorded
    for line in lines:
        del line["generation"]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert "generation" not in load_record(path)
    again = optimize(problem, **settings)
    assert again.evaluated == 0 and again.reused == len(first.evaluations)
    assert again.evaluations == first.evaluations
    assert again.path == first.path


def test_a_scan_has_no_generations_and_counts_repeats() -> None:
    problem = Problem(_square(), bowl)
    grid = scan(problem, [[0.0, 0.0], [1.0, -2.0], [0.0, 0.0]])
    assert grid.evaluated == 2 and grid.repeats == 1 and grid.reused == 0
    assert [e.parameters["x"] for e in grid.evaluations] == [0.0, 1.0, 0.0]
    assert grid.path == [] and "generation" not in grid.to_dataset()
    assert grid.path_to_dataset().sizes["generation"] == 0


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
    assert [e.generation for e in pooled.evaluations] == [
        e.generation for e in serial.evaluations
    ]
    assert [e.parameters for e in pooled.path] == [e.parameters for e in serial.path]
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


# -- checking the best designs again ------------------------------------------------------


def test_recheck_evaluates_the_best_designs_again(tmp_path) -> None:
    problem = Problem(_square(), bowl)
    result = scan(problem, {"x": [-1.0, 0.0, 1.0, 2.0], "y": [-2.0, 0.0]})
    CALLS.clear()
    checked = recheck(result, shifted, top=3, record=tmp_path / "check.jsonl")
    # the three best of the scan by their rank, in that order, each evaluated once
    ranked = sorted(result.evaluations, key=lambda e: e.rank("objective"))
    assert [e.parameters for e in checked.evaluations] == [
        e.parameters for e in ranked[:3]
    ]
    assert CALLS == [e.parameters for e in ranked[:3]]
    assert checked.evaluated == 3 and checked.reused == 0
    assert [e.objectives["objective"] for e in checked.evaluations] == pytest.approx(
        [e.objectives["objective"] + 0.5 for e in ranked[:3]]
    )
    # the check's problem is the search's with the new function
    assert checked.problem.function is shifted
    assert checked.problem.design is problem.design
    assert checked.best is not None and checked.x == {"x": 1.0, "y": -2.0}

    # a rerun takes them from the record
    CALLS.clear()
    again = recheck(result, shifted, top=3, record=tmp_path / "check.jsonl")
    assert again.evaluated == 0 and again.reused == 3 and not CALLS


def test_recheck_with_its_own_problem_and_constraints() -> None:
    problem = Problem(_square(), bowl)
    result = scan(problem, {"x": [0.0, 1.0, 2.0], "y": [-2.0]}, workers=2)
    check = Problem(_square(), shifted, constraints="line")
    checked = recheck(result, check, top=3, workers=2)
    assert len(checked.evaluations) == 3
    # in the order of the scan's ranking (x = 0 and 2 tie); x = 2 violates the check's
    # constraint
    assert [e.parameters["x"] for e in checked.evaluations] == [1.0, 0.0, 2.0]
    assert [e.feasible for e in checked.evaluations] == [True, True, False]
    assert checked.x == {"x": 1.0, "y": -2.0}
    with pytest.raises(ValueError, match="parameters"):
        recheck(result, Problem(Design((Parameter("z", 0.0, 1.0),)), bowl))
    with pytest.raises(ValueError, match="top"):
        recheck(result, shifted, top=0)


def test_recheck_skips_penalised_and_repeated_designs() -> None:
    problem = Problem(Design((Parameter("x", -2.0, 2.0),)), treacherous)
    # the unconverged x = 0 and x = 0.2 (objective -100) and the failed x = 2 rank last;
    # x = 1 appears twice
    result = scan(problem, {"x": [0.0, 1.0, 0.2, 2.0, 1.0, -1.5]})
    checked = recheck(result, lambda values: Evaluation({"objective": 0.0}), top=5)
    assert [e.parameters["x"] for e in checked.evaluations] == [1.0, -1.5]
