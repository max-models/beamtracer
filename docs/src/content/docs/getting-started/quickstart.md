---
title: Quickstart
description: Run the command line entry point of template-python.
---

First, ensure that `template-python` is [installed](/template-python/getting-started/installation/).

## Basic usage

After installation, you can run the application with:

```bash
template-python
```

which calls `app.main.main` and prints a greeting.

## Use the package from Python

```python
from app.main import main

main()
```

See the [API reference](/template-python/api/app/) for every module and function.

## Profiling

template-python's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`template_python.main` here), tagged with the opt-in tag `template_python`: they are
off by default and cost a fraction of a microsecond per call. A code that profiles itself with
scope-profiler turns them on with the tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["template_python"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=template_python`,
`scope-profiler run --enable-tag template_python` or `enable_tags = ["template_python"]` in the
TOML configuration. New regions use `@sp.profile("template_python.<name>", tags=PROFILE_TAGS)`
with `PROFILE_TAGS` from the package's `_profiling` module.
