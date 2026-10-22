# beamtracer


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/beamtracer/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/beamtracer/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/beamtracer/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/beamtracer/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/beamtracer/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/beamtracer/)
[![codecov](https://codecov.io/gh/max-models/beamtracer/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/beamtracer)
[![PyPI](https://img.shields.io/pypi/v/beamtracer.png)](https://pypi.org/project/beamtracer/)
[![Python](https://img.shields.io/pypi/pyversions/beamtracer.png)](https://pypi.org/project/beamtracer/)

Steady ion beams in a self-consistent field: extraction of positive or
negative ions from a plasma, transport through electrodes and solenoids,
and charge changes of a fast beam in a gas or a plasma, in planar or
cylindrical `(x, r)` geometry. `BeamOptics` converges the beam by a
Vlasov iteration (solve the field, trace the beam, deposit its space
charge, repeat); `beamtracer.neutralizer` couples the traced beam to a
global model of a beam-driven plasma neutralizer
([wellbalanced](https://github.com/max-models/wellbalanced)). The field
gather and the charge deposit are Pyccel, NumPy, CUDA and Metal kernels
through [cunumpy](https://github.com/max-models/cunumpy).

Documentation: <https://max-models.github.io/beamtracer/>

## Install

With [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
```

Or create and activate a Python environment (3.10 or newer) and install
with pip:

``` bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

The `neutralizer` extra installs wellbalanced, `examples` the plotting
libraries of the examples, and `test`, `docs` and `dev` the test runner,
the documentation tooling and the linters:

``` bash
pip install -e ".[dev]"
```

## A planar diode

``` python
from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
)

neumann = FieldBoundaryCondition.neumann(0.0)
config = BeamOpticsConfig(
    grid=GridParams(num_gridpoints=(41, 21), upper_bound=(10e-3, 5e-3)),
    bcs_field=FieldBCs(
        x_lower=FieldBoundaryCondition.dirichlet(0.0),
        x_upper=FieldBoundaryCondition.dirichlet(-1000.0),
        y_lower=neumann,
        y_upper=neumann,
    ),
    beams=[
        BeamSpecies("H+", 10.0, (0.0, 0.0), (0.0, 5e-3), energy=0.01, num_particles=40)
    ],
    diagnostic_planes=[9.9e-3],
)
model = BeamOptics(config).run()
print(model.beam_moments(9.9e-3).current)
```

The examples (`python -m beamtracer.examples.<name>`) are a planar
proton extraction, a round H⁻ beamlet with its co-extracted electrons,
and a D⁻ beam through a neutralizer channel.

## Development

Formatting and linting use [ruff](https://docs.astral.sh/ruff/), type
checking [pyright](https://microsoft.github.io/pyright/), run by
[pre-commit](https://pre-commit.com/) and in CI:

``` bash
make lint     # ruff check, ruff format --check, pyright
make test     # pytest with coverage
```

Commit messages follow [Conventional
Commits](https://www.conventionalcommits.org/); see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
and the API reference generated from the docstrings with
[starlight-pydocs](https://ewels.github.io/starlight-pydocs/). It needs
Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-dev         # live preview at http://localhost:4321/beamtracer/
make docs-build       # the static site in docs/dist
```

## Build the README

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/):

``` bash
make readme
```

## Releases

Before merging a release to `main`, update the version in
`pyproject.toml`, `src/beamtracer/__init__.py` and `CITATION.cff`
(including its release date), and add the release notes to
`CHANGELOG.md`. The push to `main` creates a GitHub release with a
`vX.Y.Z` tag and publishes the package to PyPI with trusted publishing
(OIDC). The one-time PyPI and GitHub configuration is described in the
[publishing
tutorial](https://max-models.github.io/beamtracer/development/publishing/).
