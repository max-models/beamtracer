# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The steady beam optics of picnic, moved from picnic at a3ac7eb (their history is picnic's):
  `BeamOptics` and `BeamOpticsConfig` (was `picnic.models.beam_optics` and
  `picnic.parameters.BeamOpticsConfig`), the solver `beamtracer.solver` (was
  `picnic.solver.ion_optics`), `beamtracer.neutralizer` (was `picnic.neutralizer`), the kernels
  `staggered_field_2d` and `deposit_trajectories_2d` (renamed in the move) with their
  catalog and a `PyccelCache("beamtracer")`, and the examples `beam_optics_extraction_2d`,
  `negative_ion_extraction_optics` and `neutralizer_channel_2d`.
- `beamtracer.sections`: the grid, the walls and the electrodes of a configuration
  (`GridParams`, `FieldBCs`, `FieldBoundaryCondition`, `SymbolicSpaceFunction`, `ObjectParams`),
  which read and write picnic's configuration-file forms of the same sections.
- `BeamOptics.run()` solves the beam (it ran through picnic's `Simulation`), and the results
  on the nodes are `beamtracer.output.NodeField`s (were picnic `Field`s).
- Opt-in scope-profiler regions around the run, the Vlasov iteration, the field solve, the
  trace and the deposit, off unless the `beamtracer` tag is enabled (e.g.
  `scope_profiler.setup(enable_tags=["beamtracer"])`).
- The tests run on NumPy and on cunumpy's fake CuPy backend.
- `beamtracer.optimization`, design optimisation that knows nothing about the model: a
  `Design` of named `Parameter`s builds a configuration with a user function, a `Problem`
  evaluates it with another into an `Evaluation` (objectives, constraints `<= 0`, a
  converged flag, wall time, diagnostics); `optimize` (SciPy's differential evolution,
  constraints by Lampinen's feasibility rules) and `scan` evaluate in a process pool and
  record every evaluation in a JSON-lines file that a later search resumes from
  (`Record`, `load_record` as an xarray Dataset). Exceptions and unconverged runs are
  penalised evaluations that never win.
- `beamtracer.optimization.einzel.EinzelLens`, the first problem: three rounded tubes in
  `(x, r)` geometry, the rms spot of a proton beam at a target plane, with the lost
  current and the surface field as constraints, and the thin-lens and paraxial voltages
  that focus on the target; the tutorial "Optimising an einzel lens".
- `BeamOptics.surface_fields()` and `beamtracer.solver.surface_field`: the largest field
  on each electrode's surface, from the solved potential next to it.
- Electrodes of several shapes at one potential: `ObjectParams.union(shapes, bc_value)` of
  `ShapeParams` (configuration-file form `type: union` with a `shapes` list), simgeometry's
  Boolean union of the shapes. The field, the hits and `surface_fields()` treat it as one
  electrode.

### Changed

- The solver's `Particles` is `BeamParticles` (it clashed with picnic's marker containers).
- The electrodes are evaluated on the host whatever cunumpy's active backend, as the rest of
  the mesh.
