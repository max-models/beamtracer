"""The scope-profiler tag of template_python's profiling regions.

The regions are opt-in: :mod:`template_python` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["template_python"])``.
"""

#: Tags of every template_python region; the first is the opt-in tag.
PROFILE_TAGS = ("template_python",)
