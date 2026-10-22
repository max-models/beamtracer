"""The opt-in scope-profiler regions: off unless the application enables the tag."""

from __future__ import annotations

import pytest
import scope_profiler as sp

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
)

REGIONS = {
    "beamtracer.run",
    "beamtracer.vlasov_iterate",
    "beamtracer.field_solve",
    "beamtracer.trace",
    "beamtracer.deposit",
}


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _tiny_diode() -> BeamOptics:
    neumann = FieldBoundaryCondition.neumann(0.0)
    return BeamOptics(
        BeamOpticsConfig(
            grid=GridParams(
                num_gridpoints=(11, 6), lower_bound=(0.0, 0.0), upper_bound=(1e-2, 5e-3)
            ),
            bcs_field=FieldBCs(
                FieldBoundaryCondition.dirichlet(0.0),
                FieldBoundaryCondition.dirichlet(-1000.0),
                neumann,
                neumann,
            ),
            beams=[BeamSpecies("H+", 1.0, (0.0, 0.0), (0.0, 5e-3), num_particles=4)],
            iterations=2,
        )
    )


def _region_names(**setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # an application region, recorded either way
    with sp.region("user.run"):
        _tiny_diode().run()
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default() -> None:
    assert _region_names() == {"user.run"}


def test_enabling_the_tag_records_the_regions() -> None:
    assert _region_names(enable_tags=["beamtracer"]) == REGIONS | {"user.run"}


def test_enabling_every_tag_records_them_too() -> None:
    # "*" also turns on the regions of the libraries (cunumpy's kernel calls)
    assert REGIONS | {"user.run"} <= _region_names(enable_tags=["*"])
