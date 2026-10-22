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

### Changed

- The solver's `Particles` is `BeamParticles` (it clashed with picnic's marker containers).
- The electrodes are evaluated on the host whatever cunumpy's active backend, as the rest of
  the mesh.
