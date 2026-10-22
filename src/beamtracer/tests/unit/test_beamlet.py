"""Tests of the negative-ion beamlet problem, on a tiny mesh."""

from __future__ import annotations

import math

import pytest

from beamtracer.optimization import scan
from beamtracer.optimization.beamlet import (
    FULL_CONSTRAINTS,
    FULL_DESIGN,
    GRIDS,
    PARAMETERS,
    NegativeIonBeamlet,
)

#: A tiny beamlet: 0.5 mm cells, few trajectories (a solve takes about a second).
TINY = NegativeIonBeamlet(cell=0.5e-3, num_particles=20, iterations=30)

BOUNDS = {"extraction_voltage": (3e3, 10e3), "eg_radius": (2e-3, 4e-3)}


def test_the_beamlet_configuration() -> None:
    beamlet = TINY.at({"gap": 3e-3, "extraction_voltage": 6e3})
    assert beamlet.eg == (beamlet.pg[1] + 3e-3, beamlet.pg[1] + 3e-3 + 2e-3)
    config = TINY.config({"gap": 3e-3, "extraction_voltage": 6e3})
    # the PG, and a plate and an edge circle for the EG and the GG
    assert set(config.objects) == {"pg", "eg", "eg.edge", "gg", "gg.edge"}
    assert config.objects["eg"].bc_value == 6e3
    assert config.objects["eg.edge"].radius == pytest.approx(0.5 * TINY.eg_thickness)
    assert config.objects["gg"].bc_value == TINY.grounded_grid_voltage
    assert config.grid.coordinates == "cylindrical"
    assert config.grid.upper_bound[0] == pytest.approx(beamlet.length)
    assert [beam.name for beam in config.beams] == ["H-", "e"]
    sharp = NegativeIonBeamlet(rounded=False).config()
    assert set(sharp.objects) == set(GRIDS)
    with pytest.raises(KeyError, match="unknown"):
        TINY.config({"length": 1.0})
    with pytest.raises(ValueError, match="fit inside"):
        TINY.config({"eg_radius": 5e-3})


def test_the_problem_builds_and_an_impossible_beamlet_fails() -> None:
    problem = TINY.problem({**BOUNDS, "gap": (0.5e-3, 6e-3)})
    assert problem.design.names == ("extraction_voltage", "eg_radius", "gap")
    assert [p.unit for p in problem.design.parameters] == [
        PARAMETERS[name] for name in problem.design.names
    ]
    assert problem.objective == "divergence"
    assert problem.constraints == ("loss", "surface_field")
    # a gap of less than two cells
    evaluation = problem.evaluate(
        {"extraction_voltage": 6e3, "eg_radius": 3e-3, "gap": 0.6e-3}
    )
    assert not evaluation.ok and "two cells" in str(evaluation.failure)


def test_one_evaluation_has_finite_objectives() -> None:
    problem = TINY.problem(
        BOUNDS, constraints=("loss", "surface_field", "divergence_limit")
    )
    evaluation = problem.evaluate({"extraction_voltage": 8e3, "eg_radius": 3e-3})
    assert evaluation.ok, evaluation.diagnostics
    divergence = evaluation.objectives["divergence"]
    assert math.isfinite(divergence) and 0.0 < divergence < 0.2
    assert evaluation.constraints["divergence_limit"] == pytest.approx(
        divergence / TINY.divergence_limit - 1.0
    )
    d = evaluation.diagnostics
    assert 0.0 < d["transmission"] <= 1.0 + 1e-9
    assert d["field"] == max(d["field_eg"], d["field_gg"]) > 0.0
    assert evaluation.constraints["surface_field"] == pytest.approx(
        d["field"] / TINY.field_limit - 1.0
    )
    assert 0.0 < d["current_density"] <= TINY.current_density
    assert d["energy"] == pytest.approx(TINY.grounded_grid_voltage, rel=0.05)
    assert d["iterations"] < TINY.iterations and d["potential_change"] < TINY.tolerance
    assert d["divergence_spread"] >= 0.0
    for key in ("lost_pg", "lost_eg", "lost_gg", "returned", "electron_transmission"):
        assert key in d


def test_unconverged_points_are_penalised() -> None:
    """Two Vlasov iterations do not converge: the evaluation is penalised, not raised."""
    hasty = NegativeIonBeamlet(cell=0.5e-3, num_particles=20, iterations=2)
    problem = hasty.problem(BOUNDS)
    evaluation = problem.evaluate({"extraction_voltage": 8e3, "eg_radius": 3e-3})
    assert evaluation.failure is None
    assert not evaluation.converged and not evaluation.ok and not evaluation.feasible
    assert math.isfinite(
        evaluation.objectives["divergence"]
    )  # measured, still penalised
    assert evaluation.rank("divergence") == (2, 0.0)
    assert evaluation.diagnostics["potential_change"] > hasty.tolerance
    # next to a converged design, the unconverged one never wins
    converged = TINY.problem(BOUNDS).evaluate(
        {"extraction_voltage": 8e3, "eg_radius": 3e-3}
    )
    assert converged.ok
    assert converged.rank("divergence") < evaluation.rank("divergence")
    result = scan(problem, {"extraction_voltage": [8e3], "eg_radius": [3e-3]})
    assert result.best is not None and not result.best.ok


def test_the_full_design() -> None:
    problem = TINY.problem(FULL_DESIGN, FULL_CONSTRAINTS)
    assert set(problem.design.names) == set(PARAMETERS)
    assert problem.constraints == ("loss", "surface_field", "gap_field")
    values = {
        "extraction_voltage": 6e3,
        "eg_radius": 3e-3,
        "gap": 3e-3,
        "pg_radius": 3.5e-3,
        "gg_radius": 3e-3,
        "pg_thickness": 1e-3,
        "eg_thickness": 1.5e-3,
        "gg_thickness": 1e-3,
        "acceleration_gap": 7e-3,
    }
    beamlet = TINY.at(values)
    assert beamlet.eg == pytest.approx((5e-3, 6.5e-3))
    assert beamlet.gg == pytest.approx((13.5e-3, 14.5e-3))
    assert beamlet.gap_fields == pytest.approx(
        {"extraction": 6e3 / 3e-3, "acceleration": 14e3 / 7e-3}
    )
    config = TINY.config(values)
    assert config.objects["gg.edge"].radius == pytest.approx(0.5e-3)
    evaluation = problem.evaluate(values)
    assert evaluation.failure is None
    assert evaluation.constraints["gap_field"] == pytest.approx(
        2e6 / TINY.gap_field_limit - 1.0
    )
    assert evaluation.diagnostics["gap_field_acceleration"] == pytest.approx(2e6)
    # a corner of the bounds that does not fit in the domain radius fails, penalised
    too_wide = problem.evaluate({**values, "pg_radius": 4.5e-3, "pg_thickness": 1.5e-3})
    assert not too_wide.ok and "fit inside" in str(too_wide.failure)
    with pytest.raises(ValueError, match="between 0 and the GG voltage"):
        TINY.at({"extraction_voltage": 25e3})
