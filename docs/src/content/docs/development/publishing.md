---
title: Publishing
description: Publish the package to PyPI and the documentation to GitHub Pages.
---

## Documentation

The `Docs` workflow builds the site on every push and pull request to `main` and `devel`. It
executes the tutorials, generates the API reference and uploads the site. Pushes to `devel`
also deploy it to GitHub Pages at https://max-models.github.io/beamtracer/.

Enable GitHub Pages once in the repository settings, with "GitHub Actions" as the source.

## Releases and PyPI

Before merging a release to `main`, update the version in `pyproject.toml` and `CITATION.cff`
(including its release date), and add the release notes to `CHANGELOG.md`. The package's
`__version__` is read from the installed metadata, so it follows `pyproject.toml`. On every push to `main`, the `Release` workflow:

1. tags the release `vX.Y.Z` and creates the GitHub release,
2. builds the package with `uv build` and publishes it to PyPI with trusted publishing (OIDC),
   if that version is not already on PyPI.

### One-time configuration

1. Create an account on [PyPI](https://pypi.org/) and create or claim your project name.
2. In the PyPI project settings go to "Publishing" → "Add a new publisher" and fill in:
   - **PyPI project name**: `beamtracer` (or your project name)
   - **Owner**: your GitHub username or organization
   - **Repository name**: `beamtracer`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
3. Optionally, in the GitHub repository settings create an environment named `pypi` whose
   deployment branches are limited to `main`.

### Dependencies must come from PyPI

PyPI rejects an upload whose metadata has a direct reference, such as
`cunumpy @ git+https://github.com/org/cunumpy.git@devel` ("400 Can't have direct dependency"),
and the `Release` workflow would only fail after the merge to `main`. The `PyPI dependencies`
workflow checks this on every pull request to `main`, with `scripts/check_pypi_dependencies.py`.
It fails when a dependency, or an optional dependency, is

- a direct reference (`name @ git+https://...` or `name @ file:///...`),
- a project that PyPI does not know, or
- a version range that no release on PyPI satisfies (for example `cunumpy>=0.7` before 0.7.0 is
  published), so a dependency has to be released before the package that needs it.

The `Test Pytest` workflow also builds the wheel on pull requests to `main` and fails if its
metadata has a direct reference. Run the check locally with
`uv run --no-project --with packaging python scripts/check_pypi_dependencies.py`.

### Development against the devel branches

Pull requests to `devel` test against the devel branches of the max-models packages, pinned in
`uv.lock`; pull requests to `main` and the release use PyPI only (CI sets `UV_NO_SOURCES`). Each
max-models dependency is a plain `"cunumpy>=0.6.3"` requirement plus an entry in
`[tool.uv.sources]`, spelled the same in every repository: `cunumpy = { git =
"https://github.com/max-models/cunumpy", branch = "devel" }`. Refresh the pinned devel commit
with `uv lock --upgrade-package cunumpy`, and reproduce a run against PyPI with
`UV_NO_SOURCES=true uv sync --extra test` (it rewrites `uv.lock`, so `git restore uv.lock`
afterwards).

### Coverage

The tests upload coverage to [Codecov](https://codecov.io/). Public repositories need no
configuration; for a private one add a `CODECOV_TOKEN` repository secret.
