# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Opt-in scope-profiler regions, as in plasmacoll, wellbalanced and patchwork: `_profiling.py`
  with the package's tag (`beamtracer`, renamed by `setup_project.sh`), registered on
  import, and an example region on `main`. Off unless the application enables the tag (e.g.
  `scope_profiler.setup(enable_tags=["beamtracer"])`). Adds the `scope-profiler>=0.7.5`
  dependency.
- `PyPI dependencies` workflow and `scripts/check_pypi_dependencies.py`: on every pull request to
  `main`, fail if a dependency is a direct reference (`name @ git+https://...`), is not on PyPI, or
  has a version range that no release satisfies, instead of failing the release after the merge.

### Changed

- `__version__` is read from the installed package metadata (`importlib.metadata`), not hardcoded,
  so it can no longer disagree with `pyproject.toml` (a hardcoded one went stale in releases).

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
