"""Tests of the einzel lens problem, on a coarse mesh."""

from __future__ import annotations

import numpy as np
import pytest

from beamtracer.optimization import optimize, scan
from beamtracer.optimization.einzel import ELECTRODES, EinzelLens

#: A coarse lens: 1 mm cells, few rings (a solve takes a fraction of a second).
COARSE = EinzelLens(cell=1e-3, num_particles=20)


def test_the_lens_configuration() -> None:
    config = COARSE.config({"voltage": 12e3, "gap": 3e-3})
    # three tubes of a rectangle and two end circles each
    assert len(config.objects) == 9
    assert config.objects["middle"].bc_value == 12e3
    assert config.objects["upstream.downstream"].bc_value == 0.0
    assert config.objects["middle.upstream"].midpoint[0] == pytest.approx(
        COARSE.centre - 0.5 * (COARSE.electrode_length - COARSE.thickness)
    )
    assert config.grid.coordinates == "cylindrical"
    assert config.iterations == 1  # no space charge
    sharp = EinzelLens(rounded=False).config()
    assert set(sharp.objects) == {"upstream", "middle", "downstream"}
    with pytest.raises(KeyError, match="unknown"):
        COARSE.config({"length": 1.0})
    with pytest.raises(ValueError, match="between"):
        COARSE.config({"gap": 30e-3})


def test_an_impossible_lens_is_a_failed_evaluation() -> None:
    problem = COARSE.problem({"gap": (1e-3, 40e-3)})
    evaluation = problem.evaluate({"gap": 35e-3})
    assert not evaluation.ok
    assert "between" in str(evaluation.failure)


def test_the_evaluation_measures_the_spot_field_and_loss() -> None:
    model = COARSE.solve({"voltage": 15e3})
    evaluation = COARSE.measure(model)
    assert evaluation.ok and evaluation.constraints["loss"] == 0.0
    assert evaluation.diagnostics["transmission"] == pytest.approx(1.0)
    spot = evaluation.objectives["spot"]
    assert spot == pytest.approx(model.beam_moments(COARSE.target).rms_size)
    assert 0.0 < spot < 0.1 * COARSE.beam_radius
    field = evaluation.diagnostics["field"]
    tubes = [evaluation.diagnostics[f"field_{name}"] for name in ELECTRODES]
    assert field == max(tubes) > 0.0
    assert evaluation.constraints["surface_field"] == pytest.approx(
        field / COARSE.field_limit - 1.0
    )
    # a bore narrower than the beam loses its outer rings
    narrow = COARSE.measure(COARSE.solve({"aperture": 1.5e-3, "voltage": 2e3}))
    assert narrow.constraints["loss"] > 0.0 and not narrow.feasible


def test_the_scan_minimum_and_the_optimiser_agree_in_1d() -> None:
    problem = COARSE.problem({"voltage": (13e3, 17e3)})
    voltages = np.linspace(13e3, 17e3, 17)
    grid = scan(problem, {"voltage": voltages}, workers=2)
    result = optimize(problem, seed=1, popsize=5, maxiter=10, tol=1e-6)
    step = voltages[1] - voltages[0]
    assert abs(result.x["voltage"] - grid.x["voltage"]) < step
    assert result.fun <= grid.fun
    assert result.best is not None and result.best.feasible


def test_the_thin_lens_estimate_of_a_weak_lens() -> None:
    """A thin beam focused far downstream: V / U0 = 0.35, f = 21 lens lengths."""
    lens = EinzelLens(
        cell=1e-3, num_particles=10, beam_radius=0.5e-3, target=250e-3, length=260e-3
    )
    thin = lens.thin_lens_voltage()
    assert 0.3 < thin / lens.energy < 0.4
    voltages = np.linspace(0.96, 1.04, 9) * thin
    grid = scan(
        lens.problem({"voltage": (0.9 * thin, 1.1 * thin)}), {"voltage": voltages}
    )
    spots = np.array([e.objectives["spot"] for e in grid.evaluations])
    k = int(np.argmin(spots))
    assert 0 < k < len(voltages) - 1  # a minimum inside the scan
    a, b, _ = np.polyfit(voltages[k - 1 : k + 2], spots[k - 1 : k + 2], 2)
    optimum = -b / (2 * a)
    assert thin == pytest.approx(optimum, rel=0.01)
    assert lens.paraxial_voltage() == pytest.approx(optimum, rel=0.01)


def test_the_surface_field_constraint_holds_at_the_3d_optimum() -> None:
    lens = EinzelLens(cell=1e-3, num_particles=12, field_limit=3e6)
    problem = lens.problem(
        {"voltage": (8e3, 22e3), "gap": (1e-3, 10e-3), "aperture": (3e-3, 7e-3)},
        constraints=("loss", "surface_field"),
    )
    result = optimize(problem, seed=2, popsize=4, maxiter=6, workers=2)
    best = result.best
    assert best is not None and best.feasible
    assert best.constraints["surface_field"] <= 0.0
    assert best.diagnostics["field"] <= lens.field_limit
    # the limit excluded designs the search tried
    assert any(
        e.ok and e.constraints["surface_field"] > 0.0 for e in result.evaluations
    )
