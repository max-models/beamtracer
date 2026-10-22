---
title: Beam optics
description: The BeamOptics model of steady ion beams in a self-consistent field.
---

`BeamOptics` solves steady ion beams: extraction of positive or negative ions from a plasma,
transport through electrodes and solenoids, and the space charge of several species,
converged by a Vlasov iteration (solve the field, trace the beam, deposit its space charge,
repeat). The geometry is planar (per unit depth) or cylindrically symmetric `(x, r)`.

- **Mesh and field**: a uniform node mesh with Shortley--Weller cut cells at the electrodes
  (simgeometry objects with Dirichlet potentials, `ObjectParams`), Neumann or Dirichlet faces
  (`FieldBCs`), and the electric field on the edges between nodes, interpolated bilinearly.
- **Plasma**: `plasma=ExtractionPlasma(electron_temperature, plasma_potential, initial_plasma_x)`
  adds the electrons `-rho_e exp((phi - U_p) / T_e)` for positive-ion extraction, solved with
  Newton's method. For negative ions,
  `plasma=NegativeExtractionPlasma(ion_temperature, plasma_potential, initial_plasma_x)` adds
  the positive ions `rho_p exp(-(phi - U_p) / T_p)` instead, and the negative ions and the
  electrons extracted with them are beams (`charge=-1`), traced with their own space charge.
  `rho_e` and `rho_p` default to the beams' charge density at their start, `sum J / v`: the
  plasma is neutral where the beams leave it. The current density of each beam, and so the
  ratio of co-extracted electrons to negative ions, is an input of the model, not a result.
- **Beams**: `BeamSpecies` emits trajectories from a line (current density, energy, parallel
  and transverse temperatures, quasi-random Sobol velocities); `ParticleBeam` takes
  trajectories from another simulation.
- **Magnetic field**: `solenoids` or a measured `axial_field`, expanded off the axis to
  `field_order`.
- **Charge changes**: `charge_changing=ChargeChangingParams(cross_sections, densities)` lets
  a fast beam change charge state on uniform targets (a gas, a plasma), as in a neutralizer:
  every trajectory carries the fractions of its particles in each state, advanced along its
  path with plasmacoll's transition matrices; the force is that of the mean charge (fine
  where the field is weak), the space charge that of every state with its sign. The tracer
  tallies the charge changes and target ionizations (`model.charge_tallies`), and
  `model.state_moments(x, "0")` gives the moments of one state at a plane.
- **Space charge**: deposited at the midpoint of every step with an area-weighted linear
  shape, and reduced by `compensation` regions.
- **Diagnostics**: `diagnostic_planes` give the current, rms emittance and Twiss parameters of
  each species (`model.beam_moments(x, "H+")`).

```python
from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    ExtractionPlasma,
    GridParams,
)

config = BeamOpticsConfig(
    grid=GridParams(
        num_gridpoints=(241, 141), upper_bound=(12e-3, 7e-3), coordinates="cylindrical"
    ),
    objects={...},  # electrodes
    beams=[
        BeamSpecies(
            "He+",
            600.0,
            (0.0, 0.0),
            (0.0, 1.5e-3),
            mass=4.0,
            energy=5.0,
            transverse_temperature=0.5,
            num_particles=15000,
        )
    ],
    plasma=ExtractionPlasma(
        electron_temperature=5.0, plasma_potential=5.0, initial_plasma_x=0.55e-3
    ),
    diagnostic_planes=[11.9e-3],
)
model = BeamOptics(config).run()
print(model.beam_moments(11.9e-3))
```

Negative-ion extraction is the exact mirror of positive-ion extraction: negating every
potential and every beam charge gives the negated potential and the same trajectories (a test
checks this to round-off).

A configuration is written to and read from its configuration-file form with
`config.to_dict()` and `BeamOpticsConfig.from_dict(...)`; quantities with units
(`{"value": 2, "unit": "mm"}`) are converted to SI.

## Electrodes of several shapes

An electrode is one entry of `objects`: one name and one potential. `ObjectParams.union`
makes one electrode of several shapes (`ShapeParams`, the simgeometry shapes of
`ObjectParams`) at one potential, built as simgeometry's Boolean union (`|`) of the shapes.
The field solve, the hits on electrodes and the per-electrode diagnostics such as
`model.surface_fields()` see it as one electrode. A tube with half-round ends:

```python
from beamtracer import ObjectParams, ShapeParams

tube = ObjectParams.union(
    [
        ShapeParams("rectangle", midpoint=(40e-3, 6e-3), side_lengths=(6e-3, 2e-3)),
        ShapeParams("circle", midpoint=(37e-3, 6e-3), radius=1e-3),
        ShapeParams("circle", midpoint=(43e-3, 6e-3), radius=1e-3),
    ],
    bc_value=15e3,
)
config = config.replace(objects={"middle": tube})
```

In the configuration-file form the shapes are listed under `shapes`, each without a `bc`:

```python
{
    "type": "union",
    "shapes": [
        {
            "type": "rectangle",
            "midpoint": {"values": [40, 6], "unit": "mm"},
            "side_lengths": {"values": [6, 2], "unit": "mm"},
        },
        {
            "type": "circle",
            "midpoint": {"values": [37, 6], "unit": "mm"},
            "radius": {"value": 1, "unit": "mm"},
        },
        {
            "type": "circle",
            "midpoint": {"values": [43, 6], "unit": "mm"},
            "radius": {"value": 1, "unit": "mm"},
        },
    ],
    "bc": {"type": "dirichlet", "potential": {"value": 15, "unit": "kV"}},
}
```

## A neutralizer

`beamtracer.neutralizer.NeutralizerChannel` traces the beam through a channel whose gas and
plasma come from a global model ([wellbalanced](https://github.com/max-models/wellbalanced),
the `neutralizer` extra); the plasma takes its beam sources from the traced paths, and the
two are iterated to a fixed point. It gives what the global model alone cannot: the
transmission of every charge state, the beam lost on the walls, the divergence of the
neutrals.

## Examples

The examples are scripts in `beamtracer.examples` (the `examples` extra), each run with
`python -m beamtracer.examples.<name>`:

- `beam_optics_extraction_2d`: a small planar proton extraction, the entry point;
- `negative_ion_extraction_optics`: a round H⁻ beamlet and its electrons, with a scan of the
  extraction voltage;
- `neutralizer_channel_2d`: a 1 MeV D⁻ beam through a beam-driven plasma neutralizer.
