"""Main module of the python package."""

import scope_profiler as sp

from beamtracer._profiling import PROFILE_TAGS


@sp.profile("beamtracer.main", tags=PROFILE_TAGS)
def main() -> None:
    """Print a greeting; the entry point of the ``beamtracer`` command."""
    print("Hello, world")
