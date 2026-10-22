---
title: Installation
description: Create a Python environment and install beamtracer.
---

Create and activate a Python environment:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the package and its requirements with pip:

```bash
pip install -e .
```

## Optional extras

| Extra         | Installs                                                          |
| ------------- | ----------------------------------------------------------------- |
| `neutralizer` | `wellbalanced`, for `beamtracer.neutralizer`                      |
| `examples`    | matplotlib, plotly and plasma-plots, for the examples' figures    |
| `test`        | `pytest`, coverage, `pyccel` and the `neutralizer` extra          |
| `docs`        | the notebook runner and griffe for the API reference              |
| `dev`         | formatters and linters, plus the `docs`, `examples` and `test` extras |

```bash
pip install -e ".[dev]"
```

The kernels are compiled with Pyccel on first use and cached in `~/.cache/beamtracer/kernels`
(or `BEAMTRACER_KERNEL_CACHE`); without Pyccel the NumPy versions run.
