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
from beamtracer.solver import BeamTracer

REGIONS = {
    "beamtracer.run",
    "beamtracer.vlasov_iterate",
    "beamtracer.field_solve",
    "beamtracer.trace",
}
#: The regions of the tracer: the compiled one where Pyccel is available, else NumPy's.
TRACER_REGIONS = (
    {"beamtracer.trace_compiled"}
    if BeamTracer.compiled_available()
    else {"beamtracer.deposit"}
)


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _tiny_diode(compiled: bool | None = None) -> BeamOptics:
    neumann = FieldBoundaryCondition.neumann(0.0)
    model = BeamOptics(
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
    model.tracer.compiled = compiled
    return model


def _region_names(compiled: bool | None = None, **setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # an application region, recorded either way
    with sp.region("user.run"):
        _tiny_diode(compiled).run()
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default() -> None:
    assert _region_names() == {"user.run"}


def test_enabling_the_tag_records_the_regions() -> None:
    expected = REGIONS | TRACER_REGIONS | {"user.run"}
    assert _region_names(enable_tags=["beamtracer"]) == expected


def test_the_numpy_tracer_records_its_deposits() -> None:
    expected = REGIONS | {"beamtracer.deposit", "user.run"}
    assert _region_names(False, enable_tags=["beamtracer"]) == expected


def test_enabling_every_tag_records_them_too() -> None:
    # "*" also turns on the regions of the libraries (cunumpy's kernel calls)
    assert REGIONS | TRACER_REGIONS | {"user.run"} <= _region_names(enable_tags=["*"])


def test_the_opt_in_tag_has_an_entry_point():
    """``scope-profiler tags`` finds the tag through the package's entry point."""
    from importlib.metadata import entry_points

    from beamtracer._profiling import PROFILE_TAGS

    found = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="scope_profiler.opt_in_tags")
    }
    assert found.get("beamtracer") == "beamtracer"
    assert "beamtracer" in PROFILE_TAGS
