"""Tests for the ion extraction beam optics example."""

from __future__ import annotations

from beamtracer.examples.beam_optics_extraction_2d import (
    create_config,
    create_model,
    extraction_grid_2d,
    plasma_grid_2d,
)


def test_config_describes_the_example() -> None:
    config = create_config()
    assert config.model_name == "BeamOptics"
    assert config.grid.ndim == 2
    assert config.grid.num_gridpoints == (81, 41)
    assert set(config.objects) == {"pg", "eg"}
    assert [beam.name for beam in config.beams] == ["H+"]
    assert config.beams[0].charge == 1.0


def test_typical_pg_and_eg_objects_are_available() -> None:
    pg = plasma_grid_2d()
    eg = extraction_grid_2d()

    assert pg.type == "aperture2d"
    assert pg.bc_value == 0.0
    assert eg.type == "aperture2d"
    assert eg.bc_value == -80.0


def test_a_tiny_instance_runs_and_writes_its_file(tmp_path) -> None:
    model = create_model(num_particles=3, iterations=1, save_trajectories=1)

    assert model.run() is model

    assert model.is_finished
    assert model.stats["iterations"] == 1
    assert model.stats["launched"] == 3
    assert len(model.trajectories) == 3
    model.write_files(tmp_path)
    assert (tmp_path / "beam_optics.pkl").is_file()
