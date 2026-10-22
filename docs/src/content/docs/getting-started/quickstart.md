---
title: Quickstart
description: Run the command line entry point of beamtracer.
---

First, ensure that `beamtracer` is [installed](/beamtracer/getting-started/installation/).

## Basic usage

After installation, you can run the application with:

```bash
beamtracer
```

which calls `beamtracer.main.main` and prints a greeting.

## Use the package from Python

```python
from beamtracer.main import main

main()
```

See the [API reference](/beamtracer/api/beamtracer/) for every module and function.

## Profiling

beamtracer's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`beamtracer.main` here), tagged with the opt-in tag `beamtracer`: they are
off by default and cost a fraction of a microsecond per call. A code that profiles itself with
scope-profiler turns them on with the tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["beamtracer"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=beamtracer`,
`scope-profiler run --enable-tag beamtracer` or `enable_tags = ["beamtracer"]` in the
TOML configuration. New regions use `@sp.profile("beamtracer.<name>", tags=PROFILE_TAGS)`
with `PROFILE_TAGS` from the package's `_profiling` module.
