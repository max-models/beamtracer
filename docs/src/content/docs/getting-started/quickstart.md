---
title: Quickstart
description: Solve a self-consistent beam with beamtracer.
---

First, ensure that `beamtracer` is [installed](/beamtracer/getting-started/installation/).

## A planar diode

A proton beam from `x = 0` (at 0 V) to an electrode at `x = 10 mm` (at -1 kV), with its
space charge, solved by the Vlasov iteration:

```python
import astropy.units as u

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
    grid=GridParams(num_gridpoints=(41, 21), upper_bound=(10 * u.mm, 5 * u.mm)),
    bcs_field=FieldBCs(
        x_lower=FieldBoundaryCondition.dirichlet(0.0),
        x_upper=FieldBoundaryCondition.dirichlet(-1 * u.kV),
        y_lower=neumann,
        y_upper=neumann,
    ),
    beams=[
        BeamSpecies(
            "H+",
            10 * u.A / u.m**2,
            (0.0, 0.0),
            (0.0, 5 * u.mm),
            energy="10 meV",
            num_particles=40,
        )
    ],
    diagnostic_planes=["9.9 mm"],
)
model = BeamOptics(config).run()
print(model.stats)
print(model.beam_moments(9.9e-3))
```

Every physical setting takes an astropy Quantity (`5 * u.mm`), a string astropy parses
(`"5 mm"`, as in a configuration file) or a bare number in its documented unit: SI, with
energies and temperatures in eV (or K), masses in u and charges in e. It is converted when
the setting is made, so the bare numbers `upper_bound=(10e-3, 5e-3)`, `-1000.0` and
`energy=0.01` are the same diode. A quantity of the wrong dimension raises.

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
