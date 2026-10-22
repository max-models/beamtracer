"""Main module of the python package."""

import scope_profiler as sp

from app._profiling import PROFILE_TAGS


@sp.profile("template_python.main", tags=PROFILE_TAGS)
def main() -> None:
    """Print a greeting; the entry point of the ``template-python`` command."""
    print("Hello, world")
