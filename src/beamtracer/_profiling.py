"""The scope-profiler tag of beamtracer's profiling regions.

The regions are opt-in: :mod:`beamtracer` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["beamtracer"])``.
"""

#: Tags of every beamtracer region; the first is the opt-in tag.
PROFILE_TAGS = ("beamtracer",)
