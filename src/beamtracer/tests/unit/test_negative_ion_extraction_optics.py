"""Tests for the steady negative-ion extraction example (negative_ion_extraction_optics)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from beamtracer import NegativeExtractionPlasma
from beamtracer.examples import negative_ion_extraction_optics as example


def test_the_beamlet_extracts_h_minus_and_electrons_from_a_negative_ion_plasma() -> (
    None
):
    case = example.BeamletCase()
    config = example.beamlet_config(case)
    assert isinstance(config.plasma, NegativeExtractionPlasma)
    assert [beam.name for beam in config.beams] == ["H-", "e"]
    assert all(beam.charge == -1.0 for beam in config.beams)
    assert config.beams[1].current_density == case.current_density
    assert config.grid.coordinates == "cylindrical"
    assert [config.objects[g].bc_value for g in ("pg", "eg", "gg")] == [
        0.0,
        case.extraction_voltage,
        case.grounded_grid_voltage,
    ]
    assert case.child_langmuir == pytest.approx(770.0, rel=0.01)
    assert not example.beamlet_config(replace(case, electron_ratio=0.0)).beams[1:]


def test_a_coarse_beamlet_solves() -> None:
    case = replace(example.BeamletCase(), cell=0.25e-3, num_particles=40, iterations=4)
    run = example.run_case(case)
    assert run["iterations"] == 4
    assert 0.5 < run["transmitted"] <= 1.0 + 1e-9
    assert np.isfinite(run["divergence"]) and run["divergence"] > 0.0
    assert run["energy"] == pytest.approx(case.grounded_grid_voltage, rel=0.02)
