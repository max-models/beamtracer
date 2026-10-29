---
title: Quickstart
description: Solve a self-consistent beam with beamtracer.
---

First, ensure that `beamtracer` is [installed](/beamtracer/getting-started/installation/).

## A planar diode

A proton beam from `x = 0` (at 0 V) to an electrode at `x = 10 mm` (at -1 kV), with its
space charge, solved by the Vlasov iteration:

```python
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
print(model.stats)
print(model.beam_moments(9.9e-3))
```

See the [beam optics guide](/beamtracer/model/beam-optics/) for the model and the
[API reference](/beamtracer/api/beamtracer/) for every module and function.

## Profiling

beamtracer's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`beamtracer.run`, `beamtracer.vlasov_iterate`, `beamtracer.field_solve`,
`beamtracer.trace`, `beamtracer.deposit`), tagged with the opt-in tag `beamtracer`: they are
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
