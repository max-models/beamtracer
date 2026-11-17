# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `BeamTracer`'s Boris step (`integrator="boris"`) rotates with the compiled
  `mpiparticles.boris_velocity` instead of its own NumPy copy of the scheme, as `FieldRegion`
  already does (the same velocities to 1e-16).
- The field is solved with patchwork's `NodalPoisson`, the ecosystem's solver that picnic uses
  too, instead of beamtracer's own finite differences: the cylindrical metric in r–z, the
  electrodes as embedded bodies with the distances the mesh brackets, the initial plasma as a
  body held on its nodes, and the Newton Jacobian of the Boltzmann species as patchwork's
  reaction term (factorised by its backend, still reused as a chord method). `IonOpticsPoisson`
  keeps its interface and takes patchwork's `backend` (`"lu"`, or `"petsc"` for a distributed
  solve). Results are unchanged within the discretisation (the beam of IBSimu's `vlasov2d` case
  to 1e-8, the negative-ion beamlet to 3e-6); on a Neumann face next to an electrode the
  potential differs by up to 5e-5 of the span, where patchwork takes the cut edge's two-point
  flux instead of a mirrored Shortley–Weller row. beamtracer now depends on `patchwork-amr`.

- With MPI the ranks share the compiled tracer's electrode lookup: each samples an equal
  share of the cells next to the electrodes, and the shares are gathered (1.9 s on one rank,
  0.25-0.5 s per rank on four, for IBSimu's `vlasov2d` case at 961 x 401 nodes). The node
  charges are summed with a buffer `Allreduce` instead of the pickled `allreduce`.

- Newton's method for the Boltzmann plasma species reuses its factorised Jacobian from step
  to step and from solve to solve (a chord method), refactorising it when an update does
  not halve, and stops at a tenth of the tolerance since its steps converge linearly.
  It factorises about a third as often: the field solve of the negative-ion beamlet at
  8 kV takes 1.2 s instead of 2.8 s, the beam unchanged within the tolerance.
  `IonOpticsPoisson(mesh, reuse_jacobian=False)` factorises every step as before.

- The tracer runs as one compiled kernel, `trace_trajectories_2d`, that follows every
  trajectory from start to end: the Runge--Kutta steps, the space-charge deposit, the plane
  crossings and the exit and electrode tests, in parallel over OpenMP threads (each with its
  own charge grid; `OMP_NUM_THREADS` sets them), with a CUDA version. It takes the same steps
  as the NumPy tracer and agrees with it to rounding, about 2.5 times faster on one core and
  about 10 times faster on four on IBSimu's `vlasov2d` case. It applies to the solved
  electric field alone (no magnetic or imposed field, no charge changes) and where Pyccel is
  installed; the NumPy tracer runs otherwise. The kernel records the trajectories kept for
  plotting itself. `BeamTracer(compiled=...)` chooses. Electrodes are looked up in sub-cells of `h/8`, and
  tested with their own `inside` function only where a surface cuts the sub-cell.

- `neutralizer_channel_2d` traces a gas neutralizer as in ITER's injectors (no cusps); the docs
  say that the cusp-confined channel is the beam-driven plasma neutralizer studied for DEMO.

### Added

- `BeamOpticsConfig(mixing="anderson")`: Anderson acceleration of the Vlasov iteration's
  fixed point in place of the relaxation (`beamtracer.solver.AndersonMixer`, over
  `anderson_depth` previous iterations, 5 by default). It reaches the same beam in fewer
  iterations: 7 instead of 11 for the extraction of the tests, 19 instead of 30 for the
  negative-ion beamlet at 8 kV, and it brings the beamlet at 5 kV, where the relaxation
  keeps oscillating by volts, to changes of a few tenths of a volt.
- `coarse_particles` and `coarse_tolerance`: the first iterations trace that share of the
  trajectories (the current of each beam kept) until the potential changes by less than
  `coarse_tolerance`, then all of them. The history records the trajectories traced per
  iteration.
- `BeamOpticsConfig(integrator="boris")` (and `BeamTracer(integrator=...)`): Boris's
  kick--drift--kick step, second order with one field evaluation a step against the four
  of the default `"rk4"`, in the NumPy and the compiled tracer (the field at the end of a
  step is reused at the start of the next). With a magnetic field (NumPy tracer) the
  Boris rotation keeps the speed. On IBSimu's `vlasov2d` case it halves the time of the
  trace, and the beam at the exit differs from RK4's by about 1e-5.
- Field maps (`beamtracer.FieldMap`): gridded E and B as netCDF/xarray in the field-map
  layout shared with picnic (`max-models-field-map-1.0`), read, written, checked
  (`validate_field_map`) and interpolated multilinearly, with the field outside the map
  clipped, zero or an error. `BeamOpticsConfig(field_map=...)` adds a map's E and/or B to
  the solved field and the solenoids; `FieldRegion` takes a map as its `electric` or
  `magnetic` field. The tracer takes a magnetic field in planar geometry too, and an
  azimuthal component in cylindrical geometry.
- Every physical setting of the setup (`BeamOpticsConfig` and its sections, the beamline,
  `NeutralizerChannel`, `EinzelLens`, `NegativeIonBeamlet`, `Parameter`, `AxialProfile`,
  `conductance_profile`) takes an astropy Quantity or a quantity string (`"5 mm"`) as well as
  a bare number in its documented unit, converted once with `plasmacoll.units`; a wrong
  dimension raises. The configuration-file form writes quantity strings (`"0.005 m"`,
  `potential: "-1000 V"`, a face `max: "5 kV"`) in place of `{"value": ..., "unit": ...}`
  entries, which are no longer read.
- Method cards (methodcard) on the equations of the model: the mesh, Poisson's equation with
  the Boltzmann plasma species and Newton's method, the staggered electric field, the
  axisymmetric magnetic field and the solenoids, the surface field, the beam seeding, the
  equations of motion, RK4 and the space-charge deposit, the charge changes (fractions,
  analog events, tallies, target profiles and conductances), the Vlasov iteration and the
  compensation, the beam moments and the mismatch, `BeamOptics`, the neutralizer coupling, the
  beamline (markers, aperture grid, duct, Boris push, calorimeter and fits), the einzel lens
  (thin lens, paraxial focus), the H- beamlet (Child-Langmuir, constraints) and the
  differential evolution with its ranking, with their assumptions and references.
  `test_method_cards.py` checks them: SymPy rules for six closed forms, the observed order of
  seven discretisations, and the cards of the docs against those of the code. Adds the
  `methodcard` dependency.
- A Numerics section in the docs (header and sidebar): an overview of the model's equations
  (`docs/src/content/docs/numerics/index.mdx`, with the main cards) and one page per module
  in the order of the model, and a Numerics section on each method's API page
  (starlight-theme 0.7.4, `methodcard: { pages: true }`).
- The neutralizer example takes the plasma's rates from OpenCR (`wellbalanced.cr`); the `neutralizer`
  extra installs `wellbalanced[cr]`.
- Trajectories that change state: `ChargeChangingParams(..., mode="analog")` puts every
  trajectory in one state that changes at events sampled along its path (an optical depth used
  up by the charge-changing rate of its state), with the force of its actual charge, so an H⁻
  stripped inside an accelerator leaves as an atom of intermediate energy. Optional splitting
  at the first events (`splitting`, `split_events`), random numbers from `seed` and the MPI
  rank, sampled events in `ChargeTallies.events`. The mean over trajectories equals the
  fractions mode. `NeutralizerChannel` takes the same `mode`.
- Target densities along `x`: `densities` takes `{"x": [...], "density": [...]}`
  (`AxialProfile`) as well as numbers, e.g. the gas between accelerator grids from
  `conductance_profile` (free-molecular flow through the apertures) or a neutral-transport run.

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
- `beamtracer.optimization.beamlet.NegativeIonBeamlet`, the second problem, with the same
  `Problem`, `optimize` and `scan`: one round H- beamlet solved with `BeamOptics` and
  `NegativeExtractionPlasma` (the beamlet of `negative_ion_extraction_optics`), its
  parameters the extraction voltage, the EG aperture radius and the PG-EG gap; the rms
  divergence of the H- behind the GG is the objective, the H- loss, the EG and GG surface
  field and the 7 mrad divergence limit are constraints. A solve that does not converge
  (a limit cycle where the beam grazes a grid or the meniscus edge) is a penalised
  evaluation, with the spread of the potential, the divergence and the transmission over
  its last iterations; `COARSE` is a 0.2 mm mesh with 100 trajectories per species. The
  tutorial "Optimising a negative-ion beamlet".
- `beamtracer.optimization.recheck`: the best few designs of a search evaluated again with
  another evaluation (a finer mesh, or a PIC model of the source from user code, which
  beamtracer does not import), recorded and resumable.
- The full beamlet design: `NegativeIonBeamlet` takes the PG and GG aperture radii, the
  three grid thicknesses and the EG-GG gap as parameters too (`FULL_DESIGN`), with the
  mean field of each gap as a voltage-holding constraint (`gap_field`, with
  `FULL_CONSTRAINTS`) and validity checks (apertures inside the half pitch, gaps and grids
  resolved by the mesh, the EG voltage between the PG's and the GG's); the tutorial checks
  the best designs of both searches on the example's mesh.
- `BeamOptics.run(callback=...)`: a function called after every Vlasov iteration with its
  state.
- `BeamOptics.surface_fields()` and `beamtracer.solver.surface_field`: the largest field
  on each electrode's surface, from the solved potential next to it.
- Electrodes of several shapes at one potential: `ObjectParams.union(shapes, bc_value)` of
  `ShapeParams` (configuration-file form `type: union` with a `shapes` list), simgeometry's
  Boolean union of the shapes. The field, the hits and `surface_fields()` treat it as one
  electrode.
- The full example "Optimising an einzel lens: the map and the optimiser's path": a scan of
  the spot over the voltage and the gap with the 5 MV/m surface-field limit, the path of
  differential evolution over it, and the lens it finds; its evaluations come from a
  committed record, so the docs build solves only the final lens.
- The optimiser's path: every `Evaluation` of `optimize` carries the `generation` that
  asked for it (0 the initial population; recorded in the record file and a `generation`
  variable of `to_dataset`, `-1` for designs without one), and `Result.path` is the best
  design after each generation (`Result.path_to_dataset()` along `generation`). A design
  asked for again in the same search (SciPy checks the best one between generations) is
  answered from memory, listed once in `Result.evaluations` and counted in
  `Result.repeats`; `Result.reused` now counts only the designs taken from a record.
  `optimize`'s `callback` takes either of SciPy's signatures. Records without generations
  still resume.
- A live demo in the docs (Getting started, "Live demo"): the einzel lens of
  `beamtracer.optimization.einzel`, solved and traced in the reader's browser with Pyodide
  (`docs/src/components/beamtracer_demo.py`), with sliders for the voltage, the energy, the
  beam radius and the aperture, and the thin-lens and paraxial foci. The docs workflow
  builds the wheels it installs (`pyodide: true`), and a Pyodide workflow checks that every
  module imports and the demo runs in Pyodide (starlight-theme's reusable check).

### Changed

- `FieldRegion` pushes its markers with mpiparticles' compiled Boris velocity update
  (`mpiparticles.boris_velocity`, a charge and a step per marker) instead of its own NumPy
  copy of the scheme; mpiparticles is a dependency now.
- The figures of the docs are interactive Plotly figures in the site's colours, drawn with
  plasma-plots, one figure per plot and with their formulas typeset as LaTeX; the example
  scripts' figures are converted to web figures with posters (`docs/tools/examples.py`).

### Fixed

- `conductance_profile` ends exactly at `downstream`: the sum of the drops could round below
  zero, which `AxialProfile` refused.
- The math in the docstrings renders on the API pages: `$$` blocks instead of reST
  `.. math::` and indented formulas (shown as code), and code spans instead of `:func:`,
  `:class:`, `:mod:` and `:math:` roles.

### Changed

- `BeamOptics.potential_field` and `charge_density` are `mpifield.Field`s (new dependency
  `mpifield`): `NodeField` and `beamtracer.output` are gone, the calls (`evaluate`,
  `get_field_data`, `to_numpy`) stay. The Metal deposit no longer goes through
  `kernels/_metal_grid.py`, a copy of mpifield's, which is removed.
- The kernels are built with cunumpy's `KernelPackage`
  (`beamtracer.kernels.KERNEL_PACKAGE`): `kernels/_build.py`, `kernels/_metal.py` and
  `kernels/compile.py` are gone, and the build cache is still `BEAMTRACER_KERNEL_CACHE`.
- The solver's `Particles` is `BeamParticles` (it clashed with picnic's marker containers).
- The electrodes are evaluated on the host whatever cunumpy's active backend, as the rest of
  the mesh.
