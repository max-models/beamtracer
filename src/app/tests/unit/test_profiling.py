"""The opt-in scope-profiler regions: off unless the application enables the tag."""

from __future__ import annotations

import pytest
import scope_profiler as sp

from app.main import main


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _region_names(**setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # an application region, recorded either way
    with sp.region("user.run"):
        main()
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default() -> None:
    assert _region_names() == {"user.run"}


@pytest.mark.parametrize("tags", [["template_python"], ["*"]])
def test_enabling_the_tag_records_the_regions(tags: list[str]) -> None:
    assert _region_names(enable_tags=tags) == {"template_python.main", "user.run"}
